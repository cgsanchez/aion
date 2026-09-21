"""Exact straight-Wilson real-space density contractions.

The coefficient density consumed here is the contravariant matrix
``P = C f C^dagger``.  Pair data are assembled only for one AO quadrature
block at a time; no ``(npoint, nao, nao)`` trajectory array is retained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from aion.electromagnetism import (
    AffineMagneticGauge,
    build_magnetic_pair_geometry,
    endpoint_links,
    triangle_phases,
)
from aion.electronic_structure.ao_quadrature import AOQuadrature
from aion.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class ExactWilsonDensityResult:
    """Direct and endpoint-factorized evaluations of one Wilson density.

    ``density_direct`` and ``density_factorized`` retain their complex
    roundoff components.  Consumers must inspect the recorded imaginary
    residual before using the real part in a density functional.
    """

    density_direct: Any
    density_factorized: Any
    overlap_direct_grid: Any
    overlap_factorized_grid: Any
    overlap_stable: Any
    particle_number_direct_integral: Any
    particle_number_factorized_integral: Any
    particle_number_direct_metric: Any
    particle_number_factorized_metric: Any
    particle_number_stable_metric: Any
    density_direct_factorized_residual: float
    overlap_direct_factorized_residual: float
    density_direct_imaginary_max_abs: float
    density_factorized_imaginary_max_abs: float
    density_direct_real_minimum: float
    density_factorized_real_minimum: float
    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str
    backend: str
    device_index: int | None
    magnetic_field_au: tuple[float, float, float]
    gauge_kind: str
    gauge_origin_au: tuple[float, float, float]
    charge: float
    hbar: float
    estimated_block_bytes: int


@dataclass(frozen=True, slots=True)
class ExactWilsonDensityDirectionResult:
    """One analytic directional derivative of the exact Wilson density."""

    density_direction: Any
    overlap_direction_grid: Any
    particle_number_direction_integral: Any
    particle_number_direction_metric_grid: Any
    particle_number_direction_metric_stable: Any | None
    density_direction_imaginary_max_abs: float
    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str
    backend: str
    device_index: int | None
    direction_kind: str
    charge: float
    hbar: float


class StraightLineVectorPotentialDirection(Protocol):
    """Real vector-potential direction with straight-path line integrals."""

    def straight_line_integrals(
        self,
        starts_au: object,
        ends_au: object,
        backend: Any,
    ) -> Any: ...


def estimate_wilson_density_block_bytes(block_size: int, nao: int) -> int:
    """Conservative additional working memory for one density block.

    The estimate covers the direct Wilson AO values, line integrals,
    triangle phases, complete factorized pair array, and block densities.
    AO values and gradients owned by :class:`AOQuadrature` are excluded and
    retain their separately recorded memory estimate.
    """

    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ConfigurationError("block_size must be a positive integer")
    if isinstance(nao, bool) or not isinstance(nao, int) or nao <= 0:
        raise ConfigurationError("nao must be a positive integer")
    pair_bytes = block_size * nao * nao * (8 + 16)
    orbital_bytes = block_size * nao * (8 + 16)
    density_bytes = block_size * 2 * 16
    return pair_bytes + orbital_bytes + density_bytes


def evaluate_exact_uniform_magnetic_wilson_density(
    quadrature: AOQuadrature,
    coefficient_density: object,
    gauge: AffineMagneticGauge,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
    memory_budget_bytes: int | None = None,
) -> ExactWilsonDensityResult:
    r"""Evaluate the exact Wilson density by two algebraically equal routes.

    The direct route dresses each AO by its open Wilson factor.  The
    factorized route uses the endpoint link and triangular magnetic
    holonomy.  Both contract the complete complex contravariant density;
    neither replaces it by its real part.
    """

    if not isinstance(quadrature, AOQuadrature):
        raise TypeError("quadrature must be an AOQuadrature")
    if not isinstance(gauge, AffineMagneticGauge):
        raise TypeError("gauge must be an AffineMagneticGauge")
    checked_charge = _finite_scalar(charge, "charge")
    checked_hbar = _finite_scalar(hbar, "hbar")
    if checked_hbar <= 0.0:
        raise ConfigurationError("hbar must be positive")

    backend = quadrature.backend
    xp = backend.namespace
    reference = quadrature.reference
    nao = reference.core_operators.nao
    density = backend.asarray(coefficient_density, dtype=xp.complex128)
    backend.assert_resident(density, name="contravariant coefficient density")
    if density.shape != (nao, nao):
        raise ConfigurationError(
            f"coefficient_density has shape {density.shape}; expected {(nao, nao)}"
        )
    if not _control_bool(xp.all(xp.isfinite(density)), backend):
        raise ConfigurationError("coefficient_density contains non-finite values")
    hermiticity_scale = xp.maximum(xp.asarray(1.0), xp.linalg.norm(density))
    hermiticity = xp.linalg.norm(density - density.conj().T) / hermiticity_scale
    if backend.scalar_to_float(hermiticity) > 1.0e-11:
        raise ConfigurationError("coefficient_density must be Hermitian")

    estimated_bytes = estimate_wilson_density_block_bytes(quadrature.block_size, nao)
    if memory_budget_bytes is not None:
        if (
            isinstance(memory_budget_bytes, bool)
            or not isinstance(memory_budget_bytes, int)
            or memory_budget_bytes <= 0
        ):
            raise ConfigurationError("memory_budget_bytes must be a positive integer")
        if estimated_bytes > memory_budget_bytes:
            raise ConfigurationError(
                "Wilson-density block requires "
                f"{estimated_bytes} bytes, exceeding memory_budget_bytes={memory_budget_bytes}"
            )

    geometry = build_magnetic_pair_geometry(
        reference.core_operators.nuclei.coordinates_au,
        reference.anchor_topology.ao_to_atom,
        backend,
    )
    endpoint = endpoint_links(
        gauge,
        geometry,
        backend,
        charge=checked_charge,
        hbar=checked_hbar,
    )
    point_count = quadrature.grid.npoints
    direct_values = backend.zeros((point_count,), dtype=xp.complex128)
    factorized_values = backend.zeros((point_count,), dtype=xp.complex128)
    direct_overlap = backend.zeros((nao, nao), dtype=xp.complex128)
    factorized_overlap = backend.zeros((nao, nao), dtype=xp.complex128)
    bare_overlap_grid = backend.zeros((nao, nao), dtype=xp.complex128)
    direct_integral = backend.asarray(0.0j, dtype=xp.complex128)
    factorized_integral = backend.asarray(0.0j, dtype=xp.complex128)
    prefactor = 1j * checked_charge / checked_hbar

    for block in quadrature.blocks():
        values = block.values
        line_integrals = gauge.anchor_to_point_line_integrals(
            geometry.ao_anchor_coordinates_au,
            block.coordinates_au,
            backend,
        )
        wilson = xp.exp(prefactor * line_integrals)
        dressed = wilson * values
        direct_block = xp.einsum(
            "mn,pm,pn->p",
            density,
            dressed,
            dressed.conj(),
            optimize=True,
        )

        phase = triangle_phases(
            block.coordinates_au,
            geometry,
            gauge.field,
            backend,
            charge=checked_charge,
            hbar=checked_hbar,
        )
        bra_ket_factor = endpoint[None, :, :] * xp.exp(1j * phase)
        factorized_block = xp.einsum(
            "mn,pn,pm,pnm->p",
            density,
            values.conj(),
            values,
            bra_ket_factor,
            optimize=True,
        )

        direct_values[block.start : block.stop] = direct_block
        factorized_values[block.start : block.stop] = factorized_block
        direct_overlap += xp.einsum(
            "p,pi,pj->ij",
            block.weights_au,
            dressed.conj(),
            dressed,
            optimize=True,
        )
        factorized_overlap += xp.einsum(
            "p,pi,pj,pij->ij",
            block.weights_au,
            values.conj(),
            values,
            bra_ket_factor,
            optimize=True,
        )
        bare_overlap_grid += xp.einsum(
            "p,pi,pj->ij",
            block.weights_au,
            values.conj(),
            values,
            optimize=True,
        )
        direct_integral += xp.einsum(
            "p,p->", block.weights_au, direct_block, optimize=True
        )
        factorized_integral += xp.einsum(
            "p,p->", block.weights_au, factorized_block, optimize=True
        )

    direct_metric_number = xp.einsum(
        "mn,nm->", density, direct_overlap, optimize=True
    )
    factorized_metric_number = xp.einsum(
        "mn,nm->", density, factorized_overlap, optimize=True
    )
    analytic_bare_overlap = backend.asarray(
        reference.core_operators.overlap,
        dtype=xp.complex128,
    )
    stable_overlap = factorized_overlap + endpoint * (
        analytic_bare_overlap - bare_overlap_grid
    )
    stable_metric_number = xp.einsum(
        "mn,nm->", density, stable_overlap, optimize=True
    )
    density_residual = _relative_frobenius(
        direct_values - factorized_values,
        factorized_values,
        backend,
    )
    overlap_residual = _relative_frobenius(
        direct_overlap - factorized_overlap,
        factorized_overlap,
        backend,
    )
    direct_imaginary = backend.scalar_to_float(xp.max(xp.abs(xp.imag(direct_values))))
    factorized_imaginary = backend.scalar_to_float(
        xp.max(xp.abs(xp.imag(factorized_values)))
    )
    direct_minimum = backend.scalar_to_float(xp.min(xp.real(direct_values)))
    factorized_minimum = backend.scalar_to_float(xp.min(xp.real(factorized_values)))
    backend.synchronize()
    return ExactWilsonDensityResult(
        density_direct=direct_values,
        density_factorized=factorized_values,
        overlap_direct_grid=direct_overlap,
        overlap_factorized_grid=factorized_overlap,
        overlap_stable=stable_overlap,
        particle_number_direct_integral=direct_integral,
        particle_number_factorized_integral=factorized_integral,
        particle_number_direct_metric=direct_metric_number,
        particle_number_factorized_metric=factorized_metric_number,
        particle_number_stable_metric=stable_metric_number,
        density_direct_factorized_residual=density_residual,
        overlap_direct_factorized_residual=overlap_residual,
        density_direct_imaginary_max_abs=direct_imaginary,
        density_factorized_imaginary_max_abs=factorized_imaginary,
        density_direct_real_minimum=direct_minimum,
        density_factorized_real_minimum=factorized_minimum,
        reference_fingerprint_sha256=reference.fingerprint_sha256,
        grid_fingerprint_sha256=quadrature.grid.fingerprint_sha256,
        backend=quadrature.backend_config.kind.value,
        device_index=quadrature.backend_config.device_index,
        magnetic_field_au=gauge.field.magnetic_field_au,
        gauge_kind=gauge.kind.value,
        gauge_origin_au=gauge.origin_au,
        charge=checked_charge,
        hbar=checked_hbar,
        estimated_block_bytes=estimated_bytes,
    )


def contract_wilson_density_block(
    frame_values: object,
    coefficient_density: object,
    backend: Any,
) -> Any:
    r"""Contract ``chi P chi^dagger`` on one arbitrary coefficient frame.

    This low-level contraction is also the coefficient-frame-invariance
    boundary: under ``chi' = chi A`` and
    ``P' = A^-1 P A^-dagger`` it returns the same pointwise density.
    """

    xp = backend.namespace
    frame = backend.asarray(frame_values, dtype=xp.complex128)
    density = backend.asarray(coefficient_density, dtype=xp.complex128)
    backend.assert_resident(frame, name="coefficient frame values")
    backend.assert_resident(density, name="contravariant coefficient density")
    if frame.ndim != 2 or frame.shape[1] == 0:
        raise ConfigurationError("frame_values must have shape (npoint, ncoefficient)")
    dimension = frame.shape[1]
    if density.shape != (dimension, dimension):
        raise ConfigurationError(
            f"coefficient_density has shape {density.shape}; expected "
            f"{(dimension, dimension)}"
        )
    if not _control_bool(xp.all(xp.isfinite(frame)), backend) or not _control_bool(
        xp.all(xp.isfinite(density)), backend
    ):
        raise ConfigurationError("Wilson density contraction contains non-finite input")
    scale = xp.maximum(xp.asarray(1.0), xp.linalg.norm(density))
    if backend.scalar_to_float(xp.linalg.norm(density - density.conj().T) / scale) > 1.0e-11:
        raise ConfigurationError("coefficient_density must be Hermitian")
    return _contract_density_unchecked(frame, density, xp)


def evaluate_exact_uniform_magnetic_wilson_density_matter_direction(
    quadrature: AOQuadrature,
    coefficients: object,
    occupations: object,
    coefficient_direction: object,
    gauge: AffineMagneticGauge,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> ExactWilsonDensityDirectionResult:
    r"""Differentiate the Wilson density along an unrestricted matter direction.

    The coefficient variation is
    ``delta P = Z f C^dagger + C f Z^dagger`` at fixed Wilson frame, as in
    ``eq:wilson-hartree-ks-density-matter-rho-variation``.
    """

    if not isinstance(quadrature, AOQuadrature):
        raise TypeError("quadrature must be an AOQuadrature")
    if not isinstance(gauge, AffineMagneticGauge):
        raise TypeError("gauge must be an AffineMagneticGauge")
    checked_charge = _finite_scalar(charge, "charge")
    checked_hbar = _positive_scalar(hbar, "hbar")
    backend = quadrature.backend
    xp = backend.namespace
    reference = quadrature.reference
    nao = reference.core_operators.nao
    coefficient_array = backend.asarray(coefficients, dtype=xp.complex128)
    occupation_array = backend.asarray(occupations, dtype=xp.float64)
    direction_array = backend.asarray(coefficient_direction, dtype=xp.complex128)
    for value, name in (
        (coefficient_array, "coefficients"),
        (occupation_array, "occupations"),
        (direction_array, "coefficient direction"),
    ):
        backend.assert_resident(value, name=name)
        if not _control_bool(xp.all(xp.isfinite(value)), backend):
            raise ConfigurationError(f"{name} contains non-finite values")
    if coefficient_array.ndim != 2 or coefficient_array.shape[0] != nao:
        raise ConfigurationError(f"coefficients must have shape ({nao}, norbital)")
    if direction_array.shape != coefficient_array.shape:
        raise ConfigurationError("coefficient_direction must match coefficients")
    if occupation_array.shape != (coefficient_array.shape[1],):
        raise ConfigurationError("occupations must match the coefficient columns")
    if _control_bool(xp.any(occupation_array < 0.0), backend):
        raise ConfigurationError("occupations must be nonnegative")
    density_direction = xp.einsum(
        "mi,i,ni->mn",
        direction_array,
        occupation_array,
        coefficient_array.conj(),
        optimize=True,
    ) + xp.einsum(
        "mi,i,ni->mn",
        coefficient_array,
        occupation_array,
        direction_array.conj(),
        optimize=True,
    )

    geometry = build_magnetic_pair_geometry(
        reference.core_operators.nuclei.coordinates_au,
        reference.anchor_topology.ao_to_atom,
        backend,
    )
    endpoint = endpoint_links(
        gauge,
        geometry,
        backend,
        charge=checked_charge,
        hbar=checked_hbar,
    )
    values_direction = backend.zeros((quadrature.grid.npoints,), dtype=xp.complex128)
    overlap_grid = backend.zeros((nao, nao), dtype=xp.complex128)
    bare_overlap_grid = backend.zeros((nao, nao), dtype=xp.complex128)
    integral = backend.asarray(0.0j, dtype=xp.complex128)
    prefactor = 1j * checked_charge / checked_hbar
    for block in quadrature.blocks():
        values = block.values
        line_integrals = gauge.anchor_to_point_line_integrals(
            geometry.ao_anchor_coordinates_au,
            block.coordinates_au,
            backend,
        )
        dressed = xp.exp(prefactor * line_integrals) * values
        direction_block = _contract_density_unchecked(dressed, density_direction, xp)
        values_direction[block.start : block.stop] = direction_block
        integral += xp.einsum(
            "p,p->",
            block.weights_au,
            direction_block,
            optimize=True,
        )
        phase = triangle_phases(
            block.coordinates_au,
            geometry,
            gauge.field,
            backend,
            charge=checked_charge,
            hbar=checked_hbar,
        )
        factor = endpoint[None, :, :] * xp.exp(1j * phase)
        overlap_grid += xp.einsum(
            "p,pi,pj,pij->ij",
            block.weights_au,
            values.conj(),
            values,
            factor,
            optimize=True,
        )
        bare_overlap_grid += xp.einsum(
            "p,pi,pj->ij",
            block.weights_au,
            values.conj(),
            values,
            optimize=True,
        )
    analytic_bare_overlap = backend.asarray(
        reference.core_operators.overlap,
        dtype=xp.complex128,
    )
    stable_overlap = overlap_grid + endpoint * (
        analytic_bare_overlap - bare_overlap_grid
    )
    metric_grid = xp.einsum(
        "mn,nm->",
        density_direction,
        overlap_grid,
        optimize=True,
    )
    metric_stable = xp.einsum(
        "mn,nm->",
        density_direction,
        stable_overlap,
        optimize=True,
    )
    imaginary = backend.scalar_to_float(xp.max(xp.abs(xp.imag(values_direction))))
    backend.synchronize()
    return ExactWilsonDensityDirectionResult(
        density_direction=values_direction,
        overlap_direction_grid=backend.zeros((nao, nao), dtype=xp.complex128),
        particle_number_direction_integral=integral,
        particle_number_direction_metric_grid=metric_grid,
        particle_number_direction_metric_stable=metric_stable,
        density_direction_imaginary_max_abs=imaginary,
        reference_fingerprint_sha256=reference.fingerprint_sha256,
        grid_fingerprint_sha256=quadrature.grid.fingerprint_sha256,
        backend=quadrature.backend_config.kind.value,
        device_index=quadrature.backend_config.device_index,
        direction_kind="matter",
        charge=checked_charge,
        hbar=checked_hbar,
    )


def evaluate_exact_uniform_magnetic_wilson_density_source_direction(
    quadrature: AOQuadrature,
    coefficient_density: object,
    gauge: AffineMagneticGauge,
    vector_potential_direction: StraightLineVectorPotentialDirection,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> ExactWilsonDensityDirectionResult:
    r"""Differentiate the Wilson density at fixed coefficient history.

    Only the open Wilson frame is varied. The supplied real vector-potential
    direction provides the straight-path response ``b_i[alpha]`` in
    ``eq:wilson-hartree-ks-density-source-variation``.
    """

    if not isinstance(quadrature, AOQuadrature):
        raise TypeError("quadrature must be an AOQuadrature")
    if not isinstance(gauge, AffineMagneticGauge):
        raise TypeError("gauge must be an AffineMagneticGauge")
    line_integrals_method = getattr(vector_potential_direction, "straight_line_integrals", None)
    if not callable(line_integrals_method):
        raise TypeError("vector_potential_direction must provide straight_line_integrals")
    checked_charge = _finite_scalar(charge, "charge")
    checked_hbar = _positive_scalar(hbar, "hbar")
    backend = quadrature.backend
    xp = backend.namespace
    reference = quadrature.reference
    nao = reference.core_operators.nao
    density = _validated_coefficient_density(
        coefficient_density,
        nao,
        backend,
    )
    geometry = build_magnetic_pair_geometry(
        reference.core_operators.nuclei.coordinates_au,
        reference.anchor_topology.ao_to_atom,
        backend,
    )
    values_direction = backend.zeros((quadrature.grid.npoints,), dtype=xp.complex128)
    overlap_direction = backend.zeros((nao, nao), dtype=xp.complex128)
    integral = backend.asarray(0.0j, dtype=xp.complex128)
    prefactor = 1j * checked_charge / checked_hbar
    anchors = geometry.ao_anchor_coordinates_au
    for block in quadrature.blocks():
        base_line = gauge.anchor_to_point_line_integrals(
            anchors,
            block.coordinates_au,
            backend,
        )
        dressed = xp.exp(prefactor * base_line) * block.values
        direction_line = line_integrals_method(
            anchors[None, :, :],
            block.coordinates_au[:, None, :],
            backend,
        )
        frame_direction = prefactor * direction_line * dressed
        direction_block = _differentiate_density_frame_unchecked(
            dressed,
            frame_direction,
            density,
            xp,
        )
        values_direction[block.start : block.stop] = direction_block
        integral += xp.einsum(
            "p,p->",
            block.weights_au,
            direction_block,
            optimize=True,
        )
        overlap_direction += xp.einsum(
            "p,pi,pj->ij",
            block.weights_au,
            frame_direction.conj(),
            dressed,
            optimize=True,
        ) + xp.einsum(
            "p,pi,pj->ij",
            block.weights_au,
            dressed.conj(),
            frame_direction,
            optimize=True,
        )
    metric_grid = xp.einsum(
        "mn,nm->",
        density,
        overlap_direction,
        optimize=True,
    )
    imaginary = backend.scalar_to_float(xp.max(xp.abs(xp.imag(values_direction))))
    backend.synchronize()
    return ExactWilsonDensityDirectionResult(
        density_direction=values_direction,
        overlap_direction_grid=overlap_direction,
        particle_number_direction_integral=integral,
        particle_number_direction_metric_grid=metric_grid,
        particle_number_direction_metric_stable=None,
        density_direction_imaginary_max_abs=imaginary,
        reference_fingerprint_sha256=reference.fingerprint_sha256,
        grid_fingerprint_sha256=quadrature.grid.fingerprint_sha256,
        backend=quadrature.backend_config.kind.value,
        device_index=quadrature.backend_config.device_index,
        direction_kind="source_fixed_coefficients",
        charge=checked_charge,
        hbar=checked_hbar,
    )


def _contract_density_unchecked(frame: Any, density: Any, xp: Any) -> Any:
    return xp.einsum(
        "mn,pm,pn->p",
        density,
        frame,
        frame.conj(),
        optimize=True,
    )


def _differentiate_density_frame_unchecked(
    frame: Any,
    frame_direction: Any,
    density: Any,
    xp: Any,
) -> Any:
    return xp.einsum(
        "mn,pm,pn->p",
        density,
        frame_direction,
        frame.conj(),
        optimize=True,
    ) + xp.einsum(
        "mn,pm,pn->p",
        density,
        frame,
        frame_direction.conj(),
        optimize=True,
    )


def _validated_coefficient_density(
    coefficient_density: object,
    dimension: int,
    backend: Any,
) -> Any:
    xp = backend.namespace
    density = backend.asarray(coefficient_density, dtype=xp.complex128)
    backend.assert_resident(density, name="contravariant coefficient density")
    if density.shape != (dimension, dimension):
        raise ConfigurationError(
            f"coefficient_density has shape {density.shape}; expected "
            f"{(dimension, dimension)}"
        )
    if not _control_bool(xp.all(xp.isfinite(density)), backend):
        raise ConfigurationError("coefficient_density contains non-finite values")
    scale = xp.maximum(xp.asarray(1.0), xp.linalg.norm(density))
    if backend.scalar_to_float(xp.linalg.norm(density - density.conj().T) / scale) > 1.0e-11:
        raise ConfigurationError("coefficient_density must be Hermitian")
    return density


def _relative_frobenius(value: Any, reference: Any, backend: Any) -> float:
    xp = backend.namespace
    scale = xp.maximum(xp.asarray(1.0), xp.linalg.norm(reference))
    return float(backend.scalar_to_float(xp.linalg.norm(value) / scale))


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{name} must be a finite number")
    return result


def _positive_scalar(value: float, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ConfigurationError(f"{name} must be positive")
    return result


def _control_bool(value: object, backend: Any) -> bool:
    return bool(backend.scalar_to_float(value))
