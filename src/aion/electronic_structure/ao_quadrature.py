"""Reusable backend-resident AO value and first-derivative quadrature blocks."""

from __future__ import annotations

import importlib.metadata
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from aion.backends import ArrayBackend, make_backend
from aion.config import BackendConfig, BackendKind, canonical_sha256
from aion.electronic_structure.data import PreparedReference, immutable_array
from aion.electronic_structure.pyscf_rks import reconstruct_mean_field
from aion.errors import ConfigurationError, ReferencePreparationError


class AOGridKind(StrEnum):
    """Supported sources of AO quadrature coordinates and weights."""

    REFERENCE = "reference"
    QUALIFICATION = "qualification"


@dataclass(frozen=True, slots=True)
class AOGridPolicy:
    """Select the authenticated stored grid or an unpruned qualification grid."""

    kind: AOGridKind = AOGridKind.REFERENCE
    level: int | None = None

    def __post_init__(self) -> None:
        try:
            kind = AOGridKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"unsupported AO grid kind {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)
        if kind is AOGridKind.REFERENCE:
            if self.level is not None:
                raise ConfigurationError("reference AO grid policy does not accept a level")
            return
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level < 0:
            raise ConfigurationError("qualification AO grid level must be a nonnegative integer")

    @classmethod
    def reference(cls) -> AOGridPolicy:
        return cls(AOGridKind.REFERENCE)

    @classmethod
    def qualification(cls, level: int) -> AOGridPolicy:
        return cls(AOGridKind.QUALIFICATION, level)


@dataclass(frozen=True, slots=True)
class AOQuadratureGrid:
    """Immutable host-side coordinates and weights used by an AO block stream."""

    coordinates_au: np.ndarray
    weights_au: np.ndarray
    kind: AOGridKind
    level: int
    pruning: str
    fingerprint_sha256: str

    def __post_init__(self) -> None:
        coordinates = immutable_array(
            self.coordinates_au,
            dtype=np.float64,
            ndim=2,
            name="AO quadrature coordinates",
        )
        weights = immutable_array(
            self.weights_au, dtype=np.float64, ndim=1, name="AO quadrature weights"
        )
        if coordinates.shape != (weights.size, 3) or weights.size == 0:
            raise ReferencePreparationError(
                "AO quadrature coordinates and weights must describe a nonempty Cartesian grid"
            )
        if not isinstance(self.kind, AOGridKind):
            raise ReferencePreparationError("AO quadrature grid kind is invalid")
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level < 0:
            raise ReferencePreparationError("AO quadrature grid level is invalid")
        if not isinstance(self.pruning, str) or not self.pruning:
            raise ReferencePreparationError("AO quadrature pruning metadata is invalid")
        expected = _grid_fingerprint(coordinates, weights, self.kind, self.level, self.pruning)
        if self.kind is AOGridKind.QUALIFICATION and self.fingerprint_sha256 != expected:
            raise ReferencePreparationError("qualification AO grid fingerprint is inconsistent")
        if not _is_sha256(self.fingerprint_sha256):
            raise ReferencePreparationError("AO quadrature grid fingerprint is not SHA-256")
        object.__setattr__(self, "coordinates_au", coordinates)
        object.__setattr__(self, "weights_au", weights)

    @property
    def npoints(self) -> int:
        return int(self.weights_au.size)


@dataclass(frozen=True, slots=True)
class AOEvaluatorProvenance:
    """Stable execution description for one AO block stream."""

    evaluator: str
    backend: str
    device_index: int | None
    block_size: int
    memory_budget_bytes: int | None
    estimated_block_bytes: int
    dependencies: tuple[tuple[str, str], ...]
    thread_limits: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class AOBlock:
    """One backend-resident block with a common CPU/GPU derivative layout."""

    index: int
    start: int
    stop: int
    coordinates_au: Any
    weights_au: Any
    values: Any
    gradients: Any

    @property
    def npoints(self) -> int:
        return self.stop - self.start


@dataclass(slots=True)
class AOQuadrature:
    """Prepared, repeatable AO block stream reconstructed without rerunning SCF."""

    reference: PreparedReference
    backend_config: BackendConfig
    grid_policy: AOGridPolicy = field(default_factory=AOGridPolicy.reference)
    block_size: int = 2048
    memory_budget_bytes: int | None = None
    backend: ArrayBackend = field(init=False)
    grid: AOQuadratureGrid = field(init=False)
    provenance: AOEvaluatorProvenance = field(init=False)
    _molecule: Any = field(init=False, repr=False)
    _numint: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.reference, PreparedReference):
            raise TypeError("reference must be a PreparedReference")
        if not isinstance(self.backend_config, BackendConfig):
            raise TypeError("backend_config must be a BackendConfig")
        if not isinstance(self.grid_policy, AOGridPolicy):
            raise TypeError("grid_policy must be an AOGridPolicy")
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, int):
            raise ConfigurationError("AO quadrature block_size must be an integer")
        if self.block_size <= 0:
            raise ConfigurationError("AO quadrature block_size must be positive")
        if self.memory_budget_bytes is not None and (
                isinstance(self.memory_budget_bytes, bool)
                or not isinstance(self.memory_budget_bytes, int)
                or self.memory_budget_bytes <= 0
        ):
            raise ConfigurationError("AO memory_budget_bytes must be a positive integer")

        backend = make_backend(self.backend_config)
        model = reconstruct_mean_field(
            self.reference, self.backend_config, backend=backend
        )
        molecule = model.mol
        grid = _resolve_grid(self.reference, molecule, self.grid_policy)
        estimated_bytes = estimate_ao_block_bytes(
            self.block_size, self.reference.core_operators.nao
        )
        if self.memory_budget_bytes is not None and estimated_bytes > self.memory_budget_bytes:
            raise ConfigurationError(
                "AO block requires "
                f"{estimated_bytes} bytes, exceeding memory_budget_bytes={self.memory_budget_bytes}"
            )

        if self.backend_config.kind is BackendKind.CPU:
            from pyscf.dft import numint

            evaluator = numint
            evaluator_name = "pyscf.dft.numint.eval_ao"
        else:
            try:
                from gpu4pyscf.dft import numint
            except Exception as exc:
                raise ReferencePreparationError(
                    "physical-GPU AO evaluation requires gpu4pyscf.dft.numint"
                ) from exc
            evaluator = numint.NumInt()
            first_stop = min(grid.npoints, self.block_size)
            first_coordinates = backend.asarray(grid.coordinates_au[:first_stop])
            evaluator.build(molecule, first_coordinates)
            if evaluator.gdftopt is None:
                raise ReferencePreparationError("GPU4PySCF AO evaluator did not build gdftopt")
            evaluator_name = "gpu4pyscf.dft.numint.eval_ao"

        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "_molecule", molecule)
        object.__setattr__(self, "_numint", evaluator)
        object.__setattr__(
            self,
            "provenance",
            AOEvaluatorProvenance(
                evaluator=evaluator_name,
                backend=self.backend_config.kind.value,
                device_index=self.backend_config.device_index,
                block_size=self.block_size,
                memory_budget_bytes=self.memory_budget_bytes,
                estimated_block_bytes=estimated_bytes,
                dependencies=_dependencies(self.backend_config.kind),
                thread_limits=_thread_limits(),
            ),
        )

    def blocks(self) -> Iterator[AOBlock]:
        """Yield fresh AO blocks; no full-grid AO array is retained."""

        for block_index, start in enumerate(range(0, self.grid.npoints, self.block_size)):
            stop = min(start + self.block_size, self.grid.npoints)
            coordinates = self.backend.asarray(self.grid.coordinates_au[start:stop])
            weights = self.backend.asarray(self.grid.weights_au[start:stop])
            if self.backend_config.kind is BackendKind.CPU:
                ao = self._numint.eval_ao(
                    self._molecule,
                    coordinates,
                    deriv=1,
                )
            else:
                ao = self._numint.eval_ao(
                    self._molecule,
                    coordinates,
                    deriv=1,
                    gdftopt=self._numint.gdftopt,
                )
            self.backend.assert_resident(ao, name="AO values and gradients")
            values = ao[0]
            gradients = ao[1:4]
            expected = (stop - start, self.reference.core_operators.nao)
            if values.shape != expected or gradients.shape != (3, *expected):
                raise ReferencePreparationError(
                    "AO evaluator returned an incompatible value/gradient layout"
                )
            for name, value in (
                ("AO block coordinates", coordinates),
                ("AO block weights", weights),
                ("AO block values", values),
                ("AO block gradients", gradients),
            ):
                self.backend.assert_resident(value, name=name)
            yield AOBlock(
                index=block_index,
                start=start,
                stop=stop,
                coordinates_au=coordinates,
                weights_au=weights,
                values=values,
                gradients=gradients,
            )


def prepare_ao_quadrature(
    reference: PreparedReference,
    backend_config: BackendConfig,
    *,
    grid_policy: AOGridPolicy | None = None,
    block_size: int = 2048,
    memory_budget_bytes: int | None = None,
) -> AOQuadrature:
    """Construct a repeatable AO block stream without invoking SCF."""

    return AOQuadrature(
        reference=reference,
        backend_config=backend_config,
        grid_policy=AOGridPolicy.reference() if grid_policy is None else grid_policy,
        block_size=block_size,
        memory_budget_bytes=memory_budget_bytes,
    )


def estimate_ao_block_bytes(block_size: int, nao: int) -> int:
    """Conservative resident bytes for coordinates, weights, and deriv=1 AO data."""

    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ConfigurationError("AO block_size must be a positive integer")
    if isinstance(nao, bool) or not isinstance(nao, int) or nao <= 0:
        raise ConfigurationError("nao must be a positive integer")
    return 8 * block_size * (4 + 4 * nao)


def _resolve_grid(
    reference: PreparedReference, molecule: Any, policy: AOGridPolicy
) -> AOQuadratureGrid:
    if policy.kind is AOGridKind.REFERENCE:
        return AOQuadratureGrid(
            coordinates_au=reference.grid.coordinates_au,
            weights_au=reference.grid.weights_au,
            kind=AOGridKind.REFERENCE,
            level=reference.grid.level,
            pruning=reference.grid.pruning,
            fingerprint_sha256=reference.grid.fingerprint_sha256,
        )

    from pyscf.dft import gen_grid

    assert policy.level is not None
    generated = gen_grid.Grids(molecule)
    generated.level = policy.level
    generated.prune = None
    generated.build(with_non0tab=False)
    coordinates = np.asarray(generated.coords, dtype=np.float64)
    weights = np.asarray(generated.weights, dtype=np.float64)
    fingerprint = _grid_fingerprint(
        coordinates,
        weights,
        AOGridKind.QUALIFICATION,
        policy.level,
        "none",
    )
    return AOQuadratureGrid(
        coordinates_au=coordinates,
        weights_au=weights,
        kind=AOGridKind.QUALIFICATION,
        level=policy.level,
        pruning="none",
        fingerprint_sha256=fingerprint,
    )


def _grid_fingerprint(
    coordinates: np.ndarray,
    weights: np.ndarray,
    kind: AOGridKind,
    level: int,
    pruning: str,
) -> str:
    return canonical_sha256(
        {
            "schema": "aion.ao-quadrature-grid",
            "version": "1.0.0",
            "kind": kind.value,
            "level": level,
            "pruning": pruning,
            "coordinates_au": coordinates,
            "weights_au": weights,
        }
    )


def _dependencies(kind: BackendKind) -> tuple[tuple[str, str], ...]:
    packages = ["numpy", "pyscf"]
    if kind is BackendKind.GPU:
        packages.extend(("cupy-cuda12x", "gpu4pyscf-cuda12x"))
    values: list[tuple[str, str]] = []
    for package in packages:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "unavailable"
        values.append((package, version))
    return tuple(values)


def _thread_limits() -> tuple[tuple[str, str], ...]:
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    return tuple((name, os.environ.get(name, "unset")) for name in names)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
