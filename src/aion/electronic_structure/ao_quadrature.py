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
from aion.electronic_structure.data import (
    OneElectronAOReference,
    PreparedReference,
    immutable_array,
)
from aion.electronic_structure.local_potentials import (
    NuclearAttractionProvider,
    bind_local_potential,
)
from aion.electronic_structure.one_electron import reconstruct_one_electron_molecule
from aion.electronic_structure.pyscf_rks import reconstruct_mean_field
from aion.errors import ConfigurationError, ReferencePreparationError


class AOGridKind(StrEnum):
    """Supported sources of AO quadrature coordinates and weights."""

    REFERENCE = "reference"
    QUALIFICATION = "qualification"


class AOPruningKind(StrEnum):
    """Explicit PySCF atom-grid pruning choices used for qualification."""

    NONE = "none"
    NWCHEM = "nwchem"


@dataclass(frozen=True, slots=True)
class AOGridPolicy:
    """Select the authenticated stored grid or an unpruned qualification grid."""

    kind: AOGridKind = AOGridKind.REFERENCE
    level: int | None = None
    pruning: AOPruningKind = AOPruningKind.NONE

    def __post_init__(self) -> None:
        try:
            kind = AOGridKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"unsupported AO grid kind {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)
        try:
            pruning = AOPruningKind(self.pruning)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"unsupported AO pruning kind {self.pruning!r}") from exc
        object.__setattr__(self, "pruning", pruning)
        if kind is AOGridKind.REFERENCE:
            if self.level is not None:
                raise ConfigurationError("reference AO grid policy does not accept a level")
            if pruning is not AOPruningKind.NONE:
                raise ConfigurationError("reference AO grid policy does not select pruning")
            return
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level < 0:
            raise ConfigurationError("qualification AO grid level must be a nonnegative integer")

    @classmethod
    def reference(cls) -> AOGridPolicy:
        return cls(AOGridKind.REFERENCE)

    @classmethod
    def qualification(
        cls,
        level: int,
        *,
        pruning: AOPruningKind = AOPruningKind.NONE,
    ) -> AOGridPolicy:
        return cls(AOGridKind.QUALIFICATION, level, pruning)


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


@dataclass(frozen=True, slots=True)
class MatrixQuadratureComparison:
    """One raw grid matrix compared with its independent analytic value."""

    analytic: np.ndarray
    quadrature: np.ndarray
    absolute_frobenius_residual: float = field(init=False)
    relative_frobenius_residual: float = field(init=False)
    hermiticity_residual: float = field(init=False)
    max_absolute_element_residual: float = field(init=False)

    def __post_init__(self) -> None:
        analytic = immutable_array(
            self.analytic, dtype=np.complex128, ndim=2, name="analytic matrix"
        )
        quadrature = immutable_array(
            self.quadrature, dtype=np.complex128, ndim=2, name="quadrature matrix"
        )
        if analytic.shape[0] != analytic.shape[1] or quadrature.shape != analytic.shape:
            raise ReferencePreparationError(
                "analytic and quadrature matrices must have the same square shape"
            )
        difference = quadrature - analytic
        absolute = float(np.linalg.norm(difference))
        relative = absolute / max(1.0, float(np.linalg.norm(analytic)))
        hermiticity = float(
            np.linalg.norm(quadrature - quadrature.conj().T)
            / max(1.0, float(np.linalg.norm(quadrature)))
        )
        object.__setattr__(self, "analytic", analytic)
        object.__setattr__(self, "quadrature", quadrature)
        object.__setattr__(self, "absolute_frobenius_residual", absolute)
        object.__setattr__(self, "relative_frobenius_residual", relative)
        object.__setattr__(self, "hermiticity_residual", hermiticity)
        object.__setattr__(
            self,
            "max_absolute_element_residual",
            float(np.max(np.abs(difference))),
        )


@dataclass(frozen=True, slots=True)
class CartesianMatrixQuadratureComparison:
    """Three Cartesian AO matrices compared with analytic values."""

    analytic: np.ndarray
    quadrature: np.ndarray
    absolute_frobenius_residual: float = field(init=False)
    relative_frobenius_residual: float = field(init=False)
    component_relative_frobenius_residuals: np.ndarray = field(init=False)
    hermiticity_residual: float = field(init=False)
    max_absolute_element_residual: float = field(init=False)

    def __post_init__(self) -> None:
        analytic = immutable_array(
            self.analytic, dtype=np.complex128, ndim=3, name="analytic Cartesian matrices"
        )
        quadrature = immutable_array(
            self.quadrature,
            dtype=np.complex128,
            ndim=3,
            name="quadrature Cartesian matrices",
        )
        if (
            analytic.shape[0] != 3
            or analytic.shape[1] != analytic.shape[2]
            or quadrature.shape != analytic.shape
        ):
            raise ReferencePreparationError(
                "analytic and quadrature Cartesian matrices must have shape (3, nao, nao)"
            )
        difference = quadrature - analytic
        absolute = float(np.linalg.norm(difference))
        relative = absolute / max(1.0, float(np.linalg.norm(analytic)))
        components = np.asarray(
            [
                np.linalg.norm(difference[axis])
                / max(1.0, float(np.linalg.norm(analytic[axis])))
                for axis in range(3)
            ],
            dtype=np.float64,
        )
        components.setflags(write=False)
        adjoint = np.swapaxes(quadrature.conj(), -1, -2)
        hermiticity = float(
            np.linalg.norm(quadrature - adjoint)
            / max(1.0, float(np.linalg.norm(quadrature)))
        )
        object.__setattr__(self, "analytic", analytic)
        object.__setattr__(self, "quadrature", quadrature)
        object.__setattr__(self, "absolute_frobenius_residual", absolute)
        object.__setattr__(self, "relative_frobenius_residual", relative)
        object.__setattr__(self, "component_relative_frobenius_residuals", components)
        object.__setattr__(self, "hermiticity_residual", hermiticity)
        object.__setattr__(
            self,
            "max_absolute_element_residual",
            float(np.max(np.abs(difference))),
        )


@dataclass(frozen=True, slots=True)
class AtomPairBlockResiduals:
    """Ordered atom-pair Frobenius and maximum-element residuals."""

    pair_indices: np.ndarray
    absolute_frobenius: np.ndarray
    relative_frobenius: np.ndarray
    max_absolute_element: np.ndarray

    def __post_init__(self) -> None:
        pairs = immutable_array(
            self.pair_indices, dtype=np.int64, ndim=2, name="atom-pair indices"
        )
        if pairs.shape[1] != 2:
            raise ReferencePreparationError("atom-pair indices must have shape (npair, 2)")
        for name in ("absolute_frobenius", "relative_frobenius", "max_absolute_element"):
            value = immutable_array(
                getattr(self, name), dtype=np.float64, ndim=1, name=name.replace("_", " ")
            )
            if value.shape != (pairs.shape[0],):
                raise ReferencePreparationError(
                    "atom-pair residual arrays must match the pair-index count"
                )
            object.__setattr__(self, name, value)
        object.__setattr__(self, "pair_indices", pairs)


@dataclass(frozen=True, slots=True)
class ZeroFieldOverlapKineticResult:
    """WP1 overlap and kinetic calibration from one AO grid."""

    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str
    grid_kind: AOGridKind
    grid_level: int
    grid_pruning: str
    grid_npoints: int
    overlap: MatrixQuadratureComparison
    kinetic: MatrixQuadratureComparison


@dataclass(frozen=True, slots=True)
class ZeroFieldOneElectronResult:
    """Complete WP1 field-free lower-matrix and momentum calibration."""

    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str
    grid_kind: AOGridKind
    grid_level: int
    grid_pruning: str
    grid_npoints: int
    overlap: MatrixQuadratureComparison
    kinetic: MatrixQuadratureComparison
    nuclear_attraction: MatrixQuadratureComparison
    mechanical: MatrixQuadratureComparison
    canonical_momentum: CartesianMatrixQuadratureComparison
    opposite_momentum_sign_relative_residual: float


@dataclass(slots=True)
class AOQuadrature:
    """Prepared, repeatable AO block stream reconstructed without rerunning SCF."""

    reference: PreparedReference | OneElectronAOReference
    backend_config: BackendConfig
    grid_policy: AOGridPolicy = field(default_factory=AOGridPolicy.reference)
    block_size: int = 2048
    memory_budget_bytes: int | None = None
    backend: ArrayBackend = field(init=False)
    grid: AOQuadratureGrid = field(init=False)
    provenance: AOEvaluatorProvenance = field(init=False)
    _molecule: Any = field(init=False, repr=False)
    _numint: Any = field(init=False, repr=False)

    @property
    def pyscf_molecule(self) -> Any:
        """Live read-only PySCF molecule for analytic integral services.

        Consumers may request integrals from this object but must not mutate
        its atoms, basis, units, or AO ordering, which are authenticated by
        the prepared reference.
        """

        return self._molecule

    def __post_init__(self) -> None:
        if not isinstance(self.reference, PreparedReference | OneElectronAOReference):
            raise TypeError("reference must be PreparedReference or OneElectronAOReference")
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
        if isinstance(self.reference, PreparedReference):
            molecule = reconstruct_mean_field(
                self.reference, self.backend_config, backend=backend
            ).mol
        else:
            molecule = reconstruct_one_electron_molecule(self.reference)
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
    reference: PreparedReference | OneElectronAOReference,
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


def evaluate_zero_field_overlap_kinetic(
    quadrature: AOQuadrature,
    *,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> ZeroFieldOverlapKineticResult:
    """Reconstruct raw overlap and kinetic matrices from AO values and gradients."""

    if not isinstance(quadrature, AOQuadrature):
        raise TypeError("quadrature must be an AOQuadrature")
    checked_mass = _positive_finite(mass, "mass")
    checked_hbar = _positive_finite(hbar, "hbar")
    backend = quadrature.backend
    xp = backend.namespace
    nao = quadrature.reference.core_operators.nao
    overlap = backend.zeros((nao, nao), dtype=xp.complex128)
    kinetic = backend.zeros((nao, nao), dtype=xp.complex128)
    kinetic_scale = checked_hbar * checked_hbar / (2.0 * checked_mass)
    for block in quadrature.blocks():
        overlap += xp.einsum(
            "p,pm,pn->mn",
            block.weights_au,
            block.values.conj(),
            block.values,
            optimize=True,
        )
        gradients = xp.moveaxis(block.gradients, 0, -1)
        kinetic += kinetic_scale * xp.einsum(
            "p,pmx,pnx->mn",
            block.weights_au,
            gradients.conj(),
            gradients,
            optimize=True,
        )
    backend.synchronize()
    analytic_overlap = np.asarray(quadrature.reference.core_operators.overlap)
    analytic_kinetic = (
        checked_hbar
        * checked_hbar
        / checked_mass
        * np.asarray(quadrature.reference.core_operators.kinetic)
    )
    return ZeroFieldOverlapKineticResult(
        reference_fingerprint_sha256=quadrature.reference.fingerprint_sha256,
        grid_fingerprint_sha256=quadrature.grid.fingerprint_sha256,
        grid_kind=quadrature.grid.kind,
        grid_level=quadrature.grid.level,
        grid_pruning=quadrature.grid.pruning,
        grid_npoints=quadrature.grid.npoints,
        overlap=MatrixQuadratureComparison(
            analytic=analytic_overlap,
            quadrature=backend.to_host(overlap),
        ),
        kinetic=MatrixQuadratureComparison(
            analytic=analytic_kinetic,
            quadrature=backend.to_host(kinetic),
        ),
    )


def evaluate_zero_field_one_electron(
    quadrature: AOQuadrature,
    *,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> ZeroFieldOneElectronResult:
    """Reconstruct all WP1 lower matrices and canonical momentum in one stream."""

    if not isinstance(quadrature, AOQuadrature):
        raise TypeError("quadrature must be an AOQuadrature")
    checked_mass = _positive_finite(mass, "mass")
    checked_hbar = _positive_finite(hbar, "hbar")
    backend = quadrature.backend
    xp = backend.namespace
    reference = quadrature.reference
    nao = reference.core_operators.nao
    overlap = backend.zeros((nao, nao), dtype=xp.complex128)
    kinetic = backend.zeros((nao, nao), dtype=xp.complex128)
    nuclear = backend.zeros((nao, nao), dtype=xp.complex128)
    momentum = backend.zeros((3, nao, nao), dtype=xp.complex128)
    kinetic_scale = checked_hbar * checked_hbar / (2.0 * checked_mass)
    potential = bind_local_potential(NuclearAttractionProvider(), reference, backend)
    for block in quadrature.blocks():
        values_conjugate = block.values.conj()
        overlap += xp.einsum(
            "p,pm,pn->mn",
            block.weights_au,
            values_conjugate,
            block.values,
            optimize=True,
        )
        gradients = xp.moveaxis(block.gradients, 0, -1)
        kinetic += kinetic_scale * xp.einsum(
            "p,pmx,pnx->mn",
            block.weights_au,
            gradients.conj(),
            gradients,
            optimize=True,
        )
        nuclear += xp.einsum(
            "p,p,pm,pn->mn",
            block.weights_au,
            potential.values_au(block.coordinates_au),
            values_conjugate,
            block.values,
            optimize=True,
        )
        momentum += -1j * checked_hbar * xp.einsum(
            "p,pm,xpn->xmn",
            block.weights_au,
            values_conjugate,
            block.gradients,
            optimize=True,
        )
    backend.synchronize()
    analytic_overlap = np.asarray(reference.core_operators.overlap)
    analytic_kinetic = (
        checked_hbar
        * checked_hbar
        / checked_mass
        * np.asarray(reference.core_operators.kinetic)
    )
    analytic_nuclear = np.asarray(reference.core_operators.nuclear_attraction)
    analytic_momentum = checked_hbar * np.asarray(
        reference.core_operators.canonical_momentum
    )
    quadrature_overlap = backend.to_host(overlap)
    quadrature_kinetic = backend.to_host(kinetic)
    quadrature_nuclear = backend.to_host(nuclear)
    quadrature_momentum = backend.to_host(momentum)
    momentum_scale = max(1.0, float(np.linalg.norm(analytic_momentum)))
    opposite_sign = float(
        np.linalg.norm(-quadrature_momentum - analytic_momentum) / momentum_scale
    )
    return ZeroFieldOneElectronResult(
        reference_fingerprint_sha256=reference.fingerprint_sha256,
        grid_fingerprint_sha256=quadrature.grid.fingerprint_sha256,
        grid_kind=quadrature.grid.kind,
        grid_level=quadrature.grid.level,
        grid_pruning=quadrature.grid.pruning,
        grid_npoints=quadrature.grid.npoints,
        overlap=MatrixQuadratureComparison(
            analytic=analytic_overlap,
            quadrature=quadrature_overlap,
        ),
        kinetic=MatrixQuadratureComparison(
            analytic=analytic_kinetic,
            quadrature=quadrature_kinetic,
        ),
        nuclear_attraction=MatrixQuadratureComparison(
            analytic=analytic_nuclear,
            quadrature=quadrature_nuclear,
        ),
        mechanical=MatrixQuadratureComparison(
            analytic=analytic_kinetic + analytic_nuclear,
            quadrature=quadrature_kinetic + quadrature_nuclear,
        ),
        canonical_momentum=CartesianMatrixQuadratureComparison(
            analytic=analytic_momentum,
            quadrature=quadrature_momentum,
        ),
        opposite_momentum_sign_relative_residual=opposite_sign,
    )


def atom_pair_block_residuals(
    comparison: MatrixQuadratureComparison | CartesianMatrixQuadratureComparison,
    ao_to_atom: np.ndarray,
    *,
    natom: int | None = None,
) -> AtomPairBlockResiduals:
    """Measure every ordered atom-pair AO block without elementwise division."""

    if not isinstance(
        comparison, MatrixQuadratureComparison | CartesianMatrixQuadratureComparison
    ):
        raise TypeError("comparison must be a scalar or Cartesian matrix comparison")
    anchors = np.asarray(ao_to_atom)
    nao = comparison.analytic.shape[-1]
    if anchors.shape != (nao,) or anchors.dtype.kind not in "iu" or np.any(anchors < 0):
        raise ConfigurationError("ao_to_atom must contain one nonnegative integer per AO")
    inferred_natom = int(np.max(anchors)) + 1
    checked_natom = inferred_natom if natom is None else natom
    if (
        isinstance(checked_natom, bool)
        or not isinstance(checked_natom, int)
        or checked_natom < inferred_natom
    ):
        raise ConfigurationError("natom must cover every AO anchor")
    difference = comparison.quadrature - comparison.analytic
    pairs: list[tuple[int, int]] = []
    absolute: list[float] = []
    relative: list[float] = []
    maximum: list[float] = []
    for bra_atom in range(checked_natom):
        bra_indices = np.flatnonzero(anchors == bra_atom)
        for ket_atom in range(checked_natom):
            ket_indices = np.flatnonzero(anchors == ket_atom)
            analytic_block = np.take(
                np.take(comparison.analytic, bra_indices, axis=-2),
                ket_indices,
                axis=-1,
            )
            difference_block = np.take(
                np.take(difference, bra_indices, axis=-2),
                ket_indices,
                axis=-1,
            )
            absolute_value = float(np.linalg.norm(difference_block))
            pairs.append((bra_atom, ket_atom))
            absolute.append(absolute_value)
            relative.append(
                absolute_value / max(1.0, float(np.linalg.norm(analytic_block)))
            )
            maximum.append(
                0.0
                if difference_block.size == 0
                else float(np.max(np.abs(difference_block)))
            )
    return AtomPairBlockResiduals(
        pair_indices=np.asarray(pairs, dtype=np.int64),
        absolute_frobenius=np.asarray(absolute, dtype=np.float64),
        relative_frobenius=np.asarray(relative, dtype=np.float64),
        max_absolute_element=np.asarray(maximum, dtype=np.float64),
    )


def estimate_ao_block_bytes(block_size: int, nao: int) -> int:
    """Conservative resident bytes for coordinates, weights, and deriv=1 AO data."""

    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ConfigurationError("AO block_size must be a positive integer")
    if isinstance(nao, bool) or not isinstance(nao, int) or nao <= 0:
        raise ConfigurationError("nao must be a positive integer")
    return 8 * block_size * (4 + 4 * nao)


def _resolve_grid(
    reference: PreparedReference | OneElectronAOReference,
    molecule: Any,
    policy: AOGridPolicy,
) -> AOQuadratureGrid:
    if policy.kind is AOGridKind.REFERENCE:
        if isinstance(reference, OneElectronAOReference):
            raise ConfigurationError(
                "occupancy-independent AO references have no stored DFT grid; "
                "select an explicit qualification grid"
            )
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
    if policy.pruning is AOPruningKind.NONE:
        generated.prune = None
    elif policy.pruning is AOPruningKind.NWCHEM:
        generated.prune = gen_grid.nwchem_prune
    else:  # pragma: no cover - guarded by AOGridPolicy validation
        raise ConfigurationError(f"unsupported AO pruning kind {policy.pruning!r}")
    generated.build(with_non0tab=False)
    coordinates = np.asarray(generated.coords, dtype=np.float64)
    weights = np.asarray(generated.weights, dtype=np.float64)
    fingerprint = _grid_fingerprint(
        coordinates,
        weights,
        AOGridKind.QUALIFICATION,
        policy.level,
        policy.pruning.value,
    )
    return AOQuadratureGrid(
        coordinates_au=coordinates,
        weights_au=weights,
        kind=AOGridKind.QUALIFICATION,
        level=policy.level,
        pruning=policy.pruning.value,
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


def _positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a positive finite number")
    try:
        checked = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a positive finite number") from exc
    if not np.isfinite(checked) or checked <= 0.0:
        raise ConfigurationError(f"{name} must be a positive finite number")
    return checked
