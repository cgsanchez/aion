"""Projected spatial-connection exact and B1/B2 magnetic hierarchy."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aion.electromagnetism.magnetic import (
    AffineMagneticGauge,
    UniformMagneticField,
    anchored_vectors,
    build_magnetic_pair_geometry,
    endpoint_links,
    triangle_phases,
)
from aion.electronic_structure.ao_quadrature import AOQuadrature
from aion.electronic_structure.operators import build_e1_operators
from aion.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class SpatialConnectionHierarchy:
    """Cartesian exact and decomposed B1/B2 barred spatial connections."""

    zero: Any
    quadrature_zero: Any
    first_F: Any
    first_C: Any
    second_F2: Any
    second_FC: Any
    exact_grid: Any
    exact: Any

    @property
    def first(self) -> Any:
        return self.first_F + self.first_C

    @property
    def second(self) -> Any:
        return self.second_F2 + self.second_FC

    @property
    def b1(self) -> Any:
        return self.zero + self.first

    @property
    def b2(self) -> Any:
        return self.b1 + self.second


@dataclass(frozen=True, slots=True)
class SpatialConnectionDirectOracle:
    """Direct lower spatial connection built with the covariant derivative."""

    gauge: AffineMagneticGauge
    lower_grid: Any


@dataclass(frozen=True, slots=True)
class MagneticSpatialConnectionResult:
    """One field's complete projected spatial-connection benchmark result."""

    field: UniformMagneticField
    endpoint_link: Any
    spatial_connection: SpatialConnectionHierarchy
    e1_first_C_closure: Any
    direct_oracle: SpatialConnectionDirectOracle | None
    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str
    backend: str
    device_index: int | None
    charge: float
    hbar: float

    @property
    def lower_exact(self) -> Any:
        return self.endpoint_link[None, :, :] * self.spatial_connection.exact

    @property
    def lower_exact_grid(self) -> Any:
        return self.endpoint_link[None, :, :] * self.spatial_connection.exact_grid


@dataclass(slots=True)
class _SpatialAccumulators:
    zero: Any
    first_F: Any
    first_C: Any
    second_F2: Any
    second_FC: Any
    exact_correction: Any
    direct: Any


def evaluate_magnetic_spatial_connections(
    quadrature: AOQuadrature,
    magnetic_fields: Sequence[UniformMagneticField],
    *,
    direct_gauges: Sequence[AffineMagneticGauge] | None = None,
    charge: float = -1.0,
    hbar: float = 1.0,
    memory_budget_bytes: int | None = None,
    include_direct_oracle: bool = True,
    block_callback: Callable[[int, int], None] | None = None,
) -> tuple[MagneticSpatialConnectionResult, ...]:
    """Contract the spatial hierarchy independently of one-electron MB3 output."""

    if not isinstance(quadrature, AOQuadrature):
        raise TypeError("quadrature must be an AOQuadrature")
    fields = tuple(magnetic_fields)
    if not fields or not all(isinstance(field, UniformMagneticField) for field in fields):
        raise ConfigurationError("magnetic_fields must contain UniformMagneticField values")
    checked_charge = _finite_parameter(charge, "charge")
    checked_hbar = _positive_parameter(hbar, "hbar")
    if not isinstance(include_direct_oracle, bool):
        raise ConfigurationError("include_direct_oracle must be boolean")
    if memory_budget_bytes is not None:
        if (
            isinstance(memory_budget_bytes, bool)
            or not isinstance(memory_budget_bytes, int)
            or memory_budget_bytes <= 0
        ):
            raise ConfigurationError("spatial memory_budget_bytes must be a positive integer")
        required = estimate_spatial_connection_block_bytes(
            quadrature.block_size, quadrature.reference.core_operators.nao
        )
        if required > memory_budget_bytes:
            raise ConfigurationError(
                f"spatial-connection block requires {required} bytes, exceeding "
                f"memory_budget_bytes={memory_budget_bytes}"
            )

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

    backend = quadrature.backend
    xp = backend.namespace
    reference = quadrature.reference
    geometry = build_magnetic_pair_geometry(
        reference.core_operators.nuclei.coordinates_au,
        reference.anchor_topology.ao_to_atom,
        backend,
    )
    nao = reference.core_operators.nao
    # The stored operator is p=-i*grad in atomic units, hence i*p is the
    # analytic derivative-form spatial connection. For arbitrary bookkeeping
    # hbar, p_hbar=hbar*p_stored and i*p_hbar/hbar gives the same matrix.
    analytic_zero = backend.asarray(
        1j * reference.core_operators.canonical_momentum, dtype=xp.complex128
    )
    e1 = build_e1_operators(
        reference.core_operators,
        reference.anchor_topology,
        charge=checked_charge,
    )
    central_dipoles = backend.asarray(e1.central_dipoles, dtype=xp.complex128)
    accumulators = tuple(_new_accumulators(nao, backend) for _ in fields)
    total_blocks = (quadrature.grid.npoints + quadrature.block_size - 1) // quadrature.block_size

    for block in quadrature.blocks():
        gradients = xp.moveaxis(block.gradients, 0, -1)
        for field, gauge, accumulator in zip(fields, gauges, accumulators, strict=True):
            phase = triangle_phases(
                block.coordinates_au,
                geometry,
                field,
                backend,
                charge=checked_charge,
                hbar=checked_hbar,
            )
            exact_minus_one = xp.expm1(1j * phase)
            exact_factor = 1.0 + exact_minus_one
            first_factor = 1j * phase
            second_factor = -0.5 * phase * phase
            anchored = anchored_vectors(block.coordinates_au, geometry, field, backend)

            accumulator.zero += _ordinary_derivative_pair(
                block.values, gradients, block.weights_au, xp
            )
            accumulator.first_F += _factorized_derivative_pair(
                block.values, gradients, block.weights_au, first_factor, xp
            )
            accumulator.first_C += _anchored_connection_pair(
                block.values,
                anchored,
                block.weights_au,
                1.0,
                checked_charge,
                checked_hbar,
                xp,
            )
            accumulator.second_F2 += _factorized_derivative_pair(
                block.values, gradients, block.weights_au, second_factor, xp
            )
            accumulator.second_FC += _anchored_connection_pair(
                block.values,
                anchored,
                block.weights_au,
                first_factor,
                checked_charge,
                checked_hbar,
                xp,
            )
            accumulator.exact_correction += _factorized_derivative_pair(
                block.values, gradients, block.weights_au, exact_minus_one, xp
            ) + _anchored_connection_pair(
                block.values,
                anchored,
                block.weights_au,
                exact_factor,
                checked_charge,
                checked_hbar,
                xp,
            )

            if include_direct_oracle:
                line_integrals = gauge.anchor_to_point_line_integrals(
                    geometry.ao_anchor_coordinates_au, block.coordinates_au, backend
                )
                line_gradients = gauge.anchor_to_point_line_integral_gradients(
                    geometry.ao_anchor_coordinates_au, block.coordinates_au, backend
                )
                point_vector_potential = gauge.vector_potential(
                    block.coordinates_au, backend
                )
                residual = line_gradients - point_vector_potential[:, None, :]
                wilson = xp.exp((1j * checked_charge / checked_hbar) * line_integrals)
                dressed_values = wilson * block.values
                covariant_gradients = wilson[:, :, None] * (
                    gradients
                    + (1j * checked_charge / checked_hbar)
                    * residual
                    * block.values[:, :, None]
                )
                accumulator.direct += _ordinary_derivative_pair(
                    dressed_values,
                    covariant_gradients,
                    block.weights_au,
                    xp,
                )
        if block_callback is not None:
            block_callback(block.index + 1, total_blocks)

    results: list[MagneticSpatialConnectionResult] = []
    dipoles_last = xp.moveaxis(central_dipoles, 0, -1)
    for field, gauge, accumulator in zip(fields, gauges, accumulators, strict=True):
        endpoint = endpoint_links(
            gauge,
            geometry,
            backend,
            charge=checked_charge,
            hbar=checked_hbar,
        )
        exact_grid = accumulator.zero + accumulator.exact_correction
        exact = analytic_zero + accumulator.exact_correction
        field_array = backend.asarray(field.magnetic_field_au, dtype=xp.float64)
        e1_closure = (1j / (2.0 * checked_hbar)) * xp.moveaxis(
            xp.cross(dipoles_last, field_array), -1, 0
        )
        hierarchy = SpatialConnectionHierarchy(
            zero=analytic_zero,
            quadrature_zero=accumulator.zero,
            first_F=accumulator.first_F,
            first_C=accumulator.first_C,
            second_F2=accumulator.second_F2,
            second_FC=accumulator.second_FC,
            exact_grid=exact_grid,
            exact=exact,
        )
        results.append(
            MagneticSpatialConnectionResult(
                field=field,
                endpoint_link=endpoint,
                spatial_connection=hierarchy,
                e1_first_C_closure=e1_closure,
                direct_oracle=(
                    SpatialConnectionDirectOracle(gauge=gauge, lower_grid=accumulator.direct)
                    if include_direct_oracle
                    else None
                ),
                reference_fingerprint_sha256=reference.fingerprint_sha256,
                grid_fingerprint_sha256=quadrature.grid.fingerprint_sha256,
                backend=quadrature.backend_config.kind.value,
                device_index=quadrature.backend_config.device_index,
                charge=checked_charge,
                hbar=checked_hbar,
            )
        )
    backend.synchronize()
    return tuple(results)


def estimate_spatial_connection_block_bytes(block_size: int, nao: int) -> int:
    """Conservative peak temporary bytes for one field and AO block."""

    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ConfigurationError("spatial block_size must be a positive integer")
    if isinstance(nao, bool) or not isinstance(nao, int) or nao <= 0:
        raise ConfigurationError("nao must be a positive integer")
    return block_size * (64 * nao * nao + 256 * nao + 64)


def _new_accumulators(nao: int, backend: Any) -> _SpatialAccumulators:
    def zero() -> Any:
        return backend.zeros((3, nao, nao), dtype=backend.namespace.complex128)

    return _SpatialAccumulators(*(zero() for _ in range(7)))


def _ordinary_derivative_pair(
    values: Any, gradients: Any, weights: Any, xp: Any
) -> Any:
    right = xp.einsum(
        "p,pm,pnx->xmn", weights, values.conj(), gradients, optimize=True
    )
    left = xp.einsum(
        "p,pmx,pn->xmn", weights, gradients.conj(), values, optimize=True
    )
    return 0.5 * (right - left)


def _factorized_derivative_pair(
    values: Any, gradients: Any, weights: Any, factor: Any, xp: Any
) -> Any:
    right = xp.einsum(
        "p,pm,pnx,pmn->xmn",
        weights,
        values.conj(),
        gradients,
        factor,
        optimize=True,
    )
    left = xp.einsum(
        "p,pmx,pn,pmn->xmn",
        weights,
        gradients.conj(),
        values,
        factor,
        optimize=True,
    )
    return 0.5 * (right - left)


def _anchored_connection_pair(
    values: Any,
    anchored: Any,
    weights: Any,
    factor: Any,
    charge: float,
    hbar: float,
    xp: Any,
) -> Any:
    if isinstance(factor, float):
        left = factor * xp.einsum(
            "p,pm,pn,pmx->xmn",
            weights,
            values.conj(),
            values,
            anchored,
            optimize=True,
        )
        right = factor * xp.einsum(
            "p,pm,pn,pnx->xmn",
            weights,
            values.conj(),
            values,
            anchored,
            optimize=True,
        )
    else:
        left = xp.einsum(
            "p,pm,pn,pmx,pmn->xmn",
            weights,
            values.conj(),
            values,
            anchored,
            factor,
            optimize=True,
        )
        right = xp.einsum(
            "p,pm,pn,pnx,pmn->xmn",
            weights,
            values.conj(),
            values,
            anchored,
            factor,
            optimize=True,
        )
    return (1j * charge / (2.0 * hbar)) * (left + right)


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
