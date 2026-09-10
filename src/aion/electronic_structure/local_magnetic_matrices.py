"""Generic exact/B1/B2 magnetic hierarchy for pointwise local potentials."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aion.electromagnetism.magnetic import (
    AffineMagneticGauge,
    UniformMagneticField,
    build_magnetic_pair_geometry,
    endpoint_links,
    triangle_phases,
)
from aion.electronic_structure.ao_quadrature import AOQuadrature
from aion.electronic_structure.local_potentials import (
    LocalPotentialIdentity,
    LocalPotentialProvider,
    bind_local_potential,
)
from aion.electronic_structure.magnetic_matrices import (
    ScalarMagneticHierarchy,
    estimate_magnetic_block_bytes,
)
from aion.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class LocalPotentialMagneticResult:
    """One provider/field scalar hierarchy and optional direct lower oracle."""

    identity: LocalPotentialIdentity
    field: UniformMagneticField
    endpoint_link: Any
    hierarchy: ScalarMagneticHierarchy
    direct_lower_grid: Any | None
    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str
    backend: str
    device_index: int | None
    charge: float
    hbar: float

    @property
    def lower_exact(self) -> Any:
        return self.endpoint_link * self.hierarchy.exact

    @property
    def lower_exact_grid(self) -> Any:
        return self.endpoint_link * self.hierarchy.exact_grid


@dataclass(slots=True)
class _ScalarAccumulator:
    quadrature_zero: Any
    first: Any
    second: Any
    exact_correction: Any
    direct: Any


def evaluate_local_potential_magnetic_matrices(
    quadrature: AOQuadrature,
    magnetic_fields: Sequence[UniformMagneticField],
    providers: Sequence[LocalPotentialProvider],
    *,
    direct_gauges: Sequence[AffineMagneticGauge] | None = None,
    charge: float = -1.0,
    hbar: float = 1.0,
    memory_budget_bytes: int | None = None,
    include_direct_oracle: bool = True,
    block_callback: Callable[[int, int], None] | None = None,
) -> tuple[LocalPotentialMagneticResult, ...]:
    """Evaluate every provider and field while traversing AO blocks once."""

    if not isinstance(quadrature, AOQuadrature):
        raise TypeError("quadrature must be an AOQuadrature")
    fields = tuple(magnetic_fields)
    provider_values = tuple(providers)
    if not fields or not all(isinstance(field, UniformMagneticField) for field in fields):
        raise ConfigurationError("magnetic_fields must contain UniformMagneticField values")
    if not provider_values:
        raise ConfigurationError("at least one local-potential provider is required")
    if not isinstance(include_direct_oracle, bool):
        raise ConfigurationError("include_direct_oracle must be boolean")
    if memory_budget_bytes is not None:
        if (
            isinstance(memory_budget_bytes, bool)
            or not isinstance(memory_budget_bytes, int)
            or memory_budget_bytes <= 0
        ):
            raise ConfigurationError("local magnetic memory budget must be a positive integer")
        required = estimate_magnetic_block_bytes(
            quadrature.block_size,
            quadrature.reference.core_operators.nao,
        )
        if required > memory_budget_bytes:
            raise ConfigurationError(
                f"local magnetic block requires {required} bytes, exceeding "
                f"memory_budget_bytes={memory_budget_bytes}"
            )
    checked_charge = _finite_parameter(charge, "charge")
    checked_hbar = _positive_parameter(hbar, "hbar")
    backend = quadrature.backend
    xp = backend.namespace
    bound = tuple(
        bind_local_potential(provider, quadrature.reference, backend)
        for provider in provider_values
    )
    fingerprints = tuple(provider.identity.fingerprint_sha256 for provider in bound)
    if len(set(fingerprints)) != len(fingerprints):
        raise ConfigurationError("local-potential provider identities must be unique")

    if direct_gauges is None:
        origin = quadrature.reference.config.molecule.electromagnetic_origin.position_au
        gauges = tuple(AffineMagneticGauge(field, origin_au=origin) for field in fields)
    else:
        gauges = tuple(direct_gauges)
        if len(gauges) != len(fields):
            raise ConfigurationError("direct_gauges must match magnetic_fields in length")
        for field, gauge in zip(fields, gauges, strict=True):
            if not isinstance(gauge, AffineMagneticGauge) or gauge.field != field:
                raise ConfigurationError("every direct gauge must represent its matching field")

    geometry = build_magnetic_pair_geometry(
        quadrature.reference.core_operators.nuclei.coordinates_au,
        quadrature.reference.anchor_topology.ao_to_atom,
        backend,
    )
    nao = quadrature.reference.core_operators.nao
    accumulators = tuple(
        tuple(_new_accumulator(nao, backend) for _ in bound) for _ in fields
    )
    total_blocks = (quadrature.grid.npoints + quadrature.block_size - 1) // quadrature.block_size
    for block in quadrature.blocks():
        potentials = tuple(provider.values_au(block.coordinates_au) for provider in bound)
        for values in potentials:
            backend.assert_resident(values, name="pointwise local-potential values")
            if values.shape != (block.npoints,):
                raise ConfigurationError(
                    "pointwise local-potential values must have shape (nblock,)"
                )
        for field, gauge, field_accumulators in zip(
            fields, gauges, accumulators, strict=True
        ):
            phase = triangle_phases(
                block.coordinates_au,
                geometry,
                field,
                backend,
                charge=checked_charge,
                hbar=checked_hbar,
            )
            exact_minus_one = xp.expm1(1j * phase)
            first_factor = 1j * phase
            second_factor = -0.5 * phase * phase
            if include_direct_oracle:
                line_integrals = gauge.anchor_to_point_line_integrals(
                    geometry.ao_anchor_coordinates_au, block.coordinates_au, backend
                )
                wilson = xp.exp((1j * checked_charge / checked_hbar) * line_integrals)
                dressed_values = wilson * block.values
            for potential, accumulator in zip(
                potentials, field_accumulators, strict=True
            ):
                weights = block.weights_au * potential
                accumulator.quadrature_zero += _ordinary_pair(
                    block.values, block.values, weights, xp
                )
                accumulator.first += _factorized_pair(
                    block.values, block.values, weights, first_factor, xp
                )
                accumulator.second += _factorized_pair(
                    block.values, block.values, weights, second_factor, xp
                )
                accumulator.exact_correction += _factorized_pair(
                    block.values, block.values, weights, exact_minus_one, xp
                )
                if include_direct_oracle:
                    accumulator.direct += _ordinary_pair(
                        dressed_values, dressed_values, weights, xp
                    )
        if block_callback is not None:
            block_callback(block.index + 1, total_blocks)

    results: list[LocalPotentialMagneticResult] = []
    for field, gauge, field_accumulators in zip(fields, gauges, accumulators, strict=True):
        endpoint = endpoint_links(
            gauge, geometry, backend, charge=checked_charge, hbar=checked_hbar
        )
        for provider, accumulator in zip(bound, field_accumulators, strict=True):
            hierarchy = ScalarMagneticHierarchy(
                zero=provider.zero_matrix_au,
                quadrature_zero=accumulator.quadrature_zero,
                first_F=accumulator.first,
                second_F2=accumulator.second,
                exact_grid=accumulator.quadrature_zero + accumulator.exact_correction,
                exact=provider.zero_matrix_au + accumulator.exact_correction,
            )
            results.append(
                LocalPotentialMagneticResult(
                    identity=provider.identity,
                    field=field,
                    endpoint_link=endpoint,
                    hierarchy=hierarchy,
                    direct_lower_grid=(
                        accumulator.direct if include_direct_oracle else None
                    ),
                    reference_fingerprint_sha256=quadrature.reference.fingerprint_sha256,
                    grid_fingerprint_sha256=quadrature.grid.fingerprint_sha256,
                    backend=quadrature.backend_config.kind.value,
                    device_index=quadrature.backend_config.device_index,
                    charge=checked_charge,
                    hbar=checked_hbar,
                )
            )
    backend.synchronize()
    return tuple(results)


def _new_accumulator(nao: int, backend: Any) -> _ScalarAccumulator:
    def zero() -> Any:
        return backend.zeros((nao, nao), dtype=backend.namespace.complex128)

    return _ScalarAccumulator(*(zero() for _ in range(5)))


def _ordinary_pair(left: Any, right: Any, weights: Any, xp: Any) -> Any:
    return xp.einsum("p,pm,pn->mn", weights, left.conj(), right, optimize=True)


def _factorized_pair(
    left: Any, right: Any, weights: Any, factor: Any, xp: Any
) -> Any:
    return xp.einsum(
        "p,pm,pn,pmn->mn", weights, left.conj(), right, factor, optimize=True
    )


def _finite_parameter(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be finite")
    try:
        checked = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be finite") from exc
    if not np.isfinite(checked):
        raise ConfigurationError(f"{name} must be finite")
    return checked


def _positive_parameter(value: float, name: str) -> float:
    checked = _finite_parameter(value, name)
    if checked <= 0.0:
        raise ConfigurationError(f"{name} must be positive")
    return checked
