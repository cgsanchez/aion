"""Exact temporal Wilson data for a Maxwell-consistent uniform magnetic field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aion.electromagnetism.magnetic import (
    UniformMagneticSourceSample,
    build_magnetic_pair_geometry,
    endpoint_line_integrals,
    triangle_phases,
)
from aion.electronic_structure.ao_quadrature import AOQuadrature
from aion.electronic_structure.magnetic_matrices import (
    ExactStaticMagneticOneElectronResult,
    OneElectronLowerMatrices,
    evaluate_exact_static_magnetic_one_electron_matrices,
)
from aion.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class ExactWilsonTimeConnection:
    """Exact and independently assembled temporal data for one source sample.

    ``connection`` and ``metric_dot`` use analytic bare AO moments plus a
    quadrature correction proportional to the magnetic triangle factor minus
    one.  The two ``*_grid`` members retain the uncorrected all-grid route for
    comparison with the direct dressed-AO oracle.
    """

    endpoint_link: Any
    endpoint_link_dot: Any
    radial_electric_integral: Any
    radial_electric_integral_grid: Any
    connection: Any
    factorized_connection_grid: Any
    direct_connection_grid: Any
    metric_dot: Any
    factorized_metric_dot_grid: Any
    direct_metric_dot_grid: Any
    metric_compatibility_residual: float
    direct_factorized_connection_residual: float
    direct_factorized_metric_dot_residual: float
    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str
    backend: str
    device_index: int | None
    charge: float
    hbar: float


@dataclass(frozen=True, slots=True)
class ExactWilsonOneElectronSample:
    """One source-consistent exact one-electron matrix sample."""

    source: UniformMagneticSourceSample
    lower_exact: OneElectronLowerMatrices
    lower_exact_grid: OneElectronLowerMatrices
    connection: ExactWilsonTimeConnection
    ordinary_derivative_matrix: Any
    static_result: ExactStaticMagneticOneElectronResult

    @property
    def metric(self) -> Any:
        return self.lower_exact.overlap

    @property
    def mechanical(self) -> Any:
        return self.lower_exact.mechanical


def evaluate_exact_uniform_magnetic_time_connection(
    quadrature: AOQuadrature,
    source: UniformMagneticSourceSample,
    static_result: ExactStaticMagneticOneElectronResult,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> ExactWilsonTimeConnection:
    r"""Evaluate ``omega_t=<chi|D_t^Phi chi>`` and ``Sdot`` by two routes.

    The endpoint-factorized production value implements

    ``omega_mn = (i q/hbar) Theta_mn [Phi(R_n) Sbar_mn - I^E_mn]``.

    The direct oracle differentiates the ket Wilson factor and adds the
    pointwise scalar potential before contracting dressed AOs.  ``metric_dot``
    is assembled independently by differentiating both endpoint and triangle
    factors; it is never manufactured from ``omega + omega^dagger``.
    """

    if not isinstance(quadrature, AOQuadrature):
        raise TypeError("quadrature must be an AOQuadrature")
    if not isinstance(source, UniformMagneticSourceSample):
        raise TypeError("source must be a UniformMagneticSourceSample")
    if not isinstance(static_result, ExactStaticMagneticOneElectronResult):
        raise TypeError("static_result must be an ExactStaticMagneticOneElectronResult")
    checked_charge = _finite_parameter(charge, "charge")
    checked_hbar = _positive_parameter(hbar, "hbar")
    if static_result.field != source.field:
        raise ConfigurationError("static result and time source use different magnetic fields")
    if static_result.reference_fingerprint_sha256 != quadrature.reference.fingerprint_sha256:
        raise ConfigurationError("static result belongs to a different AO reference")
    if static_result.grid_fingerprint_sha256 != quadrature.grid.fingerprint_sha256:
        raise ConfigurationError("static result belongs to a different AO quadrature grid")
    if static_result.charge != checked_charge or static_result.hbar != checked_hbar:
        raise ConfigurationError("static result uses different particle conventions")

    backend = quadrature.backend
    xp = backend.namespace
    reference = quadrature.reference
    geometry = build_magnetic_pair_geometry(
        reference.core_operators.nuclei.coordinates_au,
        reference.anchor_topology.ao_to_atom,
        backend,
    )
    anchors = geometry.ao_anchor_coordinates_au
    gauge = source.gauge
    gauge_rate = source.gauge_rate
    endpoint = static_result.endpoint_link
    endpoint_rate_integral = endpoint_line_integrals(gauge_rate, geometry, backend)
    prefactor = 1j * checked_charge / checked_hbar
    endpoint_dot = prefactor * endpoint_rate_integral * endpoint

    overlap0 = backend.asarray(reference.core_operators.overlap, dtype=xp.complex128)
    position0 = backend.asarray(reference.core_operators.position, dtype=xp.complex128)
    pair_midpoints = geometry.pair_midpoints_au
    electric_at_anchors = source.electric_field(anchors, backend)
    scalar_at_anchors = source.scalar_potential(anchors, backend)
    ket_displacements = xp.empty_like(position0)
    for axis in range(3):
        ket_displacements[axis] = (
            position0[axis] - overlap0 * anchors[None, :, axis]
        )
    bare_radial = xp.einsum(
        "nx,xmn->mn", electric_at_anchors, ket_displacements, optimize=True
    )

    magnetic_dot = backend.asarray(source.magnetic_field_dot_au, dtype=xp.float64)
    flux_rate_vectors = 0.5 * xp.cross(geometry.pair_displacements_au, magnetic_dot)
    central_positions = xp.empty_like(position0)
    for axis in range(3):
        central_positions[axis] = (
            position0[axis] - overlap0 * pair_midpoints[:, :, axis]
        )
    bare_barred_metric_dot = prefactor * xp.einsum(
        "xmn,mnx->mn", central_positions, flux_rate_vectors, optimize=True
    )

    nao = reference.core_operators.nao
    radial_grid = backend.zeros((nao, nao), dtype=xp.complex128)
    radial_correction = backend.zeros((nao, nao), dtype=xp.complex128)
    barred_metric_dot_grid = backend.zeros((nao, nao), dtype=xp.complex128)
    barred_metric_dot_correction = backend.zeros((nao, nao), dtype=xp.complex128)
    direct_connection = backend.zeros((nao, nao), dtype=xp.complex128)
    direct_metric_dot = backend.zeros((nao, nao), dtype=xp.complex128)

    for block in quadrature.blocks():
        values = block.values
        phase = triangle_phases(
            block.coordinates_au,
            geometry,
            source.field,
            backend,
            charge=checked_charge,
            hbar=checked_hbar,
        )
        phase_dot = triangle_phases(
            block.coordinates_au,
            geometry,
            source.gauge_rate.field,
            backend,
            charge=checked_charge,
            hbar=checked_hbar,
        )
        triangle = xp.exp(1j * phase)
        triangle_minus_one = xp.expm1(1j * phase)
        radial = source.radial_electric_line_integrals(
            anchors[None, :, :], block.coordinates_au[:, None, :], backend
        )
        radial_grid += _pair_with_factor(
            values,
            block.weights_au,
            triangle * radial[:, None, :],
            xp,
        )
        radial_correction += _pair_with_factor(
            values,
            block.weights_au,
            triangle_minus_one * radial[:, None, :],
            xp,
        )
        metric_rate_factor = 1j * phase_dot * triangle
        barred_metric_dot_grid += _pair_with_factor(
            values,
            block.weights_au,
            metric_rate_factor,
            xp,
        )
        barred_metric_dot_correction += _pair_with_factor(
            values,
            block.weights_au,
            1j * phase_dot * triangle_minus_one,
            xp,
        )

        line = gauge.anchor_to_point_line_integrals(
            anchors, block.coordinates_au, backend
        )
        line_dot = gauge_rate.anchor_to_point_line_integrals(
            anchors, block.coordinates_au, backend
        )
        wilson = xp.exp(prefactor * line)
        direct_pair_factor = wilson.conj()[:, :, None] * wilson[:, None, :]
        scalar = source.scalar_potential(block.coordinates_au, backend)
        direct_temporal_factor = prefactor * (
            line_dot[:, None, :] + scalar[:, None, None]
        )
        direct_connection += _pair_with_factor(
            values,
            block.weights_au,
            direct_pair_factor * direct_temporal_factor,
            xp,
        )
        direct_metric_factor = prefactor * (
            line_dot[:, None, :] - line_dot[:, :, None]
        )
        direct_metric_dot += _pair_with_factor(
            values,
            block.weights_au,
            direct_pair_factor * direct_metric_factor,
            xp,
        )

    radial_exact = bare_radial + radial_correction
    barred_overlap_exact = static_result.overlap.exact
    barred_overlap_grid = static_result.overlap.exact_grid
    connection = prefactor * endpoint * (
        barred_overlap_exact * scalar_at_anchors[None, :] - radial_exact
    )
    factorized_connection_grid = prefactor * endpoint * (
        barred_overlap_grid * scalar_at_anchors[None, :] - radial_grid
    )
    barred_metric_dot_exact = bare_barred_metric_dot + barred_metric_dot_correction
    metric_dot = endpoint_dot * barred_overlap_exact + endpoint * barred_metric_dot_exact
    factorized_metric_dot_grid = (
        endpoint_dot * barred_overlap_grid + endpoint * barred_metric_dot_grid
    )

    compatibility = _relative_frobenius(
        metric_dot - connection - connection.conj().T,
        metric_dot,
        backend,
    )
    connection_residual = _relative_frobenius(
        direct_connection - factorized_connection_grid,
        factorized_connection_grid,
        backend,
    )
    metric_dot_residual = _relative_frobenius(
        direct_metric_dot - factorized_metric_dot_grid,
        factorized_metric_dot_grid,
        backend,
    )
    backend.synchronize()
    return ExactWilsonTimeConnection(
        endpoint_link=endpoint,
        endpoint_link_dot=endpoint_dot,
        radial_electric_integral=radial_exact,
        radial_electric_integral_grid=radial_grid,
        connection=connection,
        factorized_connection_grid=factorized_connection_grid,
        direct_connection_grid=direct_connection,
        metric_dot=metric_dot,
        factorized_metric_dot_grid=factorized_metric_dot_grid,
        direct_metric_dot_grid=direct_metric_dot,
        metric_compatibility_residual=compatibility,
        direct_factorized_connection_residual=connection_residual,
        direct_factorized_metric_dot_residual=metric_dot_residual,
        reference_fingerprint_sha256=reference.fingerprint_sha256,
        grid_fingerprint_sha256=quadrature.grid.fingerprint_sha256,
        backend=quadrature.backend_config.kind.value,
        device_index=quadrature.backend_config.device_index,
        charge=checked_charge,
        hbar=checked_hbar,
    )


def evaluate_exact_wilson_one_electron_sample(
    quadrature: AOQuadrature,
    source: UniformMagneticSourceSample,
    *,
    charge: float = -1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    memory_budget_bytes: int | None = None,
    include_direct_spatial_oracle: bool = False,
) -> ExactWilsonOneElectronSample:
    """Evaluate ``S``, ``K``, ``omega_t``, and ``K-i*hbar*omega_t`` from one source."""

    static_result = evaluate_exact_static_magnetic_one_electron_matrices(
        quadrature,
        (source.field,),
        direct_gauges=(source.gauge,),
        charge=charge,
        mass=mass,
        hbar=hbar,
        memory_budget_bytes=memory_budget_bytes,
        include_direct_oracle=include_direct_spatial_oracle,
    )[0]
    connection = evaluate_exact_uniform_magnetic_time_connection(
        quadrature,
        source,
        static_result,
        charge=charge,
        hbar=hbar,
    )
    ordinary = static_result.lower_exact.mechanical - 1j * float(hbar) * (
        connection.connection
    )
    return ExactWilsonOneElectronSample(
        source=source,
        lower_exact=static_result.lower_exact,
        lower_exact_grid=static_result.lower_exact_grid,
        connection=connection,
        ordinary_derivative_matrix=ordinary,
        static_result=static_result,
    )


def _pair_with_factor(values: Any, weights: Any, factor: Any, xp: Any) -> Any:
    return xp.einsum(
        "p,pm,pn,pmn->mn",
        weights,
        values.conj(),
        values,
        factor,
        optimize=True,
    )


def _relative_frobenius(value: Any, reference: Any, backend: Any) -> float:
    xp = backend.namespace
    denominator = xp.maximum(xp.asarray(1.0), xp.linalg.norm(reference))
    return float(backend.scalar_to_float(xp.linalg.norm(value) / denominator))


def _finite_parameter(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be finite")
    checked = float(value)
    if not np.isfinite(checked):
        raise ConfigurationError(f"{name} must be finite")
    return checked


def _positive_parameter(value: float, name: str) -> float:
    checked = _finite_parameter(value, name)
    if checked <= 0.0:
        raise ConfigurationError(f"{name} must be positive")
    return checked
