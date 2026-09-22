"""Exact temporal Wilson data for a Maxwell-consistent uniform magnetic field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aion.electromagnetism.magnetic import (
    UniformMagneticField,
    UniformMagneticSourceSample,
    build_magnetic_pair_geometry,
    endpoint_line_integrals,
    triangle_phases,
)
from aion.electromagnetism.test_variations import GaussianScalarGaugeVariation
from aion.electronic_structure.ao_quadrature import AOQuadrature
from aion.electronic_structure.local_potentials import (
    NuclearAttractionProvider,
    bind_local_potential,
)
from aion.electronic_structure.magnetic_matrices import (
    ExactStaticMagneticOneElectronDirection,
    ExactStaticMagneticOneElectronResult,
    OneElectronLowerMatrices,
    evaluate_exact_static_magnetic_one_electron_direction,
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


@dataclass(frozen=True, slots=True)
class ExactTemporalSourceDirection:
    """Exact temporal-connection response to one physical source direction.

    ``electric_origin_direction_au`` varies the uniform electric field at the
    electromagnetic origin. ``magnetic_field_rate_direction_au`` varies
    ``Bdot`` at fixed instantaneous ``B``.  The response is split into the
    anchor scalar-potential and internal radial-electric contributions before
    recording their action-consistent sum.
    """

    electric_origin_direction_au: tuple[float, float, float]
    magnetic_field_rate_direction_au: tuple[float, float, float]
    site_scalar_connection: Any
    internal_electric_connection: Any
    connection: Any
    metric_rate: Any
    decomposition_residual: float


@dataclass(frozen=True, slots=True)
class ExactMagneticFieldSourceDirection:
    """Exact action-matrix response to an instantaneous uniform ``B`` direction.

    Endpoint, internal triangle, and anchored-vector responses remain
    separately inspectable.  ``Bdot`` and the electric field at the origin
    are held fixed while the instantaneous magnetic field is varied.
    """

    magnetic_field_direction_au: tuple[float, float, float]
    metric_endpoint: Any
    metric_internal: Any
    mechanical_endpoint: Any
    mechanical_internal_triangle: Any
    mechanical_internal_anchored: Any
    connection_endpoint: Any
    connection_internal: Any
    radial_electric_integral_direction: Any
    static_direction: ExactStaticMagneticOneElectronDirection

    @property
    def metric(self) -> Any:
        return self.metric_endpoint + self.metric_internal

    @property
    def mechanical_internal(self) -> Any:
        return self.mechanical_internal_triangle + self.mechanical_internal_anchored

    @property
    def mechanical(self) -> Any:
        return self.mechanical_endpoint + self.mechanical_internal

    @property
    def connection(self) -> Any:
        return self.connection_endpoint + self.connection_internal


@dataclass(frozen=True, slots=True)
class ExactWeakVectorPotentialSourceDirection:
    """Exact matrix response paired with one smooth static test variation.

    ``frame_overlap`` is ``F_alpha=<chi|delta_alpha chi>`` and
    ``frame_overlap_rate`` is its physical time derivative.  The latter is
    distinct from ``connection=delta_alpha omega_t`` on a dynamic source and
    is required by the tangential history variation
    ``delta Cdot=Gamma_dot C+Gamma Cdot``.
    """

    variation: Any
    frame_overlap: Any
    frame_overlap_rate: Any
    metric: Any
    kinetic_embedding: Any
    kinetic_explicit: Any
    kinetic: Any
    nuclear_attraction: Any
    connection: Any

    @property
    def mechanical(self) -> Any:
        return self.kinetic + self.nuclear_attraction


@dataclass(frozen=True, slots=True)
class ExactStaticWilsonGridOneElectronAction:
    """Static exact-Wilson one-electron action on one declared AO grid.

    This evaluator accepts a general nonuniform vector potential with
    straight-path Wilson integrals.  It is the finite-difference parent of
    :class:`ExactWeakVectorPotentialSourceDirection`; unlike the uniform-
    magnetic production matrices, every term here uses the same molecular
    quadrature and no analytic zero-field correction is inserted.
    """

    overlap: Any
    kinetic: Any
    nuclear_attraction: Any
    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str
    backend: str
    device_index: int | None
    charge: float
    mass: float
    hbar: float

    @property
    def mechanical(self) -> Any:
        return self.kinetic + self.nuclear_attraction


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
        ket_displacements[axis] = position0[axis] - overlap0 * anchors[None, :, axis]
    bare_radial = xp.einsum("nx,xmn->mn", electric_at_anchors, ket_displacements, optimize=True)

    magnetic_dot = backend.asarray(source.magnetic_field_dot_au, dtype=xp.float64)
    flux_rate_vectors = 0.5 * xp.cross(geometry.pair_displacements_au, magnetic_dot)
    central_positions = xp.empty_like(position0)
    for axis in range(3):
        central_positions[axis] = position0[axis] - overlap0 * pair_midpoints[:, :, axis]
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

        line = gauge.anchor_to_point_line_integrals(anchors, block.coordinates_au, backend)
        line_dot = gauge_rate.anchor_to_point_line_integrals(anchors, block.coordinates_au, backend)
        wilson = xp.exp(prefactor * line)
        direct_pair_factor = wilson.conj()[:, :, None] * wilson[:, None, :]
        scalar = source.scalar_potential(block.coordinates_au, backend)
        direct_temporal_factor = prefactor * (line_dot[:, None, :] + scalar[:, None, None])
        direct_connection += _pair_with_factor(
            values,
            block.weights_au,
            direct_pair_factor * direct_temporal_factor,
            xp,
        )
        direct_metric_factor = prefactor * (line_dot[:, None, :] - line_dot[:, :, None])
        direct_metric_dot += _pair_with_factor(
            values,
            block.weights_au,
            direct_pair_factor * direct_metric_factor,
            xp,
        )

    radial_exact = bare_radial + radial_correction
    barred_overlap_exact = static_result.overlap.exact
    barred_overlap_grid = static_result.overlap.exact_grid
    connection = (
        prefactor * endpoint * (barred_overlap_exact * scalar_at_anchors[None, :] - radial_exact)
    )
    factorized_connection_grid = (
        prefactor * endpoint * (barred_overlap_grid * scalar_at_anchors[None, :] - radial_grid)
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
    ordinary = static_result.lower_exact.mechanical - 1j * float(hbar) * (connection.connection)
    return ExactWilsonOneElectronSample(
        source=source,
        lower_exact=static_result.lower_exact,
        lower_exact_grid=static_result.lower_exact_grid,
        connection=connection,
        ordinary_derivative_matrix=ordinary,
        static_result=static_result,
    )


def evaluate_exact_temporal_source_direction(
    quadrature: AOQuadrature,
    sample: ExactWilsonOneElectronSample,
    *,
    electric_origin_direction_au: object = (0.0, 0.0, 0.0),
    magnetic_field_rate_direction_au: object = (0.0, 0.0, 0.0),
) -> ExactTemporalSourceDirection:
    r"""Evaluate an analytic source direction of the exact time connection.

    At fixed instantaneous magnetic field, the exact temporal connection is
    linear in the uniform electric field at the origin and in ``Bdot``.  A
    direction can therefore be evaluated directly with the same quadrature
    contraction, without a finite-difference step.  Finite differences of
    independently rebuilt samples remain the qualification oracle.
    """

    if not isinstance(quadrature, AOQuadrature):
        raise TypeError("quadrature must be an AOQuadrature")
    if not isinstance(sample, ExactWilsonOneElectronSample):
        raise TypeError("sample must be an ExactWilsonOneElectronSample")
    if sample.static_result.reference_fingerprint_sha256 != (
        quadrature.reference.fingerprint_sha256
    ):
        raise ConfigurationError("sample belongs to a different AO reference")
    if sample.static_result.grid_fingerprint_sha256 != quadrature.grid.fingerprint_sha256:
        raise ConfigurationError("sample belongs to a different AO quadrature grid")
    electric = _direction_vector(
        electric_origin_direction_au,
        "electric_origin_direction_au",
    )
    magnetic_rate = _direction_vector(
        magnetic_field_rate_direction_au,
        "magnetic_field_rate_direction_au",
    )
    source = UniformMagneticSourceSample(
        time_au=sample.source.time_au,
        field=sample.source.field,
        magnetic_field_dot_au=magnetic_rate,
        electric_field_origin_au=electric,
        origin_au=sample.source.origin_au,
        gauge_kind=sample.source.gauge_kind,
        landau_axis=sample.source.landau_axis,
    )
    value = evaluate_exact_uniform_magnetic_time_connection(
        quadrature,
        source,
        sample.static_result,
        charge=sample.static_result.charge,
        hbar=sample.static_result.hbar,
    )
    backend = quadrature.backend
    xp = backend.namespace
    geometry = build_magnetic_pair_geometry(
        quadrature.reference.core_operators.nuclei.coordinates_au,
        quadrature.reference.anchor_topology.ao_to_atom,
        backend,
    )
    scalar_at_anchors = source.scalar_potential(
        geometry.ao_anchor_coordinates_au,
        backend,
    )
    prefactor = 1j * sample.static_result.charge / sample.static_result.hbar
    site = (
        prefactor
        * sample.static_result.endpoint_link
        * (sample.static_result.overlap.exact * scalar_at_anchors[None, :])
    )
    internal = -prefactor * sample.static_result.endpoint_link * value.radial_electric_integral
    residual = _relative_frobenius(site + internal - value.connection, value.connection, backend)
    return ExactTemporalSourceDirection(
        electric_origin_direction_au=electric,
        magnetic_field_rate_direction_au=magnetic_rate,
        site_scalar_connection=xp.asarray(site, dtype=xp.complex128),
        internal_electric_connection=xp.asarray(internal, dtype=xp.complex128),
        connection=value.connection,
        metric_rate=value.metric_dot,
        decomposition_residual=residual,
    )


def evaluate_exact_magnetic_field_source_direction(
    quadrature: AOQuadrature,
    sample: ExactWilsonOneElectronSample,
    magnetic_field_direction_au: object,
    *,
    memory_budget_bytes: int | None = None,
) -> ExactMagneticFieldSourceDirection:
    r"""Differentiate ``(S,K,omega_t)`` along instantaneous uniform ``B``.

    The source direction is evaluated at arbitrary finite ``B``.  It uses
    ``delta F=i delta(phi)F`` for the internal triangle holonomy, analytic
    derivatives of both anchored kinetic vectors, and the exact derivative
    of the endpoint link.  The prescribed ``Bdot`` and electric field at the
    origin are not changed by this variation.
    """

    if not isinstance(quadrature, AOQuadrature):
        raise TypeError("quadrature must be an AOQuadrature")
    if not isinstance(sample, ExactWilsonOneElectronSample):
        raise TypeError("sample must be an ExactWilsonOneElectronSample")
    if sample.static_result.reference_fingerprint_sha256 != (
        quadrature.reference.fingerprint_sha256
    ):
        raise ConfigurationError("sample belongs to a different AO reference")
    if sample.static_result.grid_fingerprint_sha256 != quadrature.grid.fingerprint_sha256:
        raise ConfigurationError("sample belongs to a different AO quadrature grid")

    direction = _direction_vector(
        magnetic_field_direction_au,
        "magnetic_field_direction_au",
    )
    static_direction = evaluate_exact_static_magnetic_one_electron_direction(
        quadrature,
        sample.static_result,
        direction,
        gauge=sample.source.gauge,
        memory_budget_bytes=memory_budget_bytes,
    )
    backend = quadrature.backend
    xp = backend.namespace
    reference = quadrature.reference
    geometry = build_magnetic_pair_geometry(
        reference.core_operators.nuclei.coordinates_au,
        reference.anchor_topology.ao_to_atom,
        backend,
    )
    direction_field = UniformMagneticField(direction)
    anchors = geometry.ao_anchor_coordinates_au
    nao = reference.core_operators.nao
    radial_direction = backend.zeros((nao, nao), dtype=xp.complex128)
    for block in quadrature.blocks():
        phase = triangle_phases(
            block.coordinates_au,
            geometry,
            sample.source.field,
            backend,
            charge=sample.static_result.charge,
            hbar=sample.static_result.hbar,
        )
        phase_direction = triangle_phases(
            block.coordinates_au,
            geometry,
            direction_field,
            backend,
            charge=sample.static_result.charge,
            hbar=sample.static_result.hbar,
        )
        factor_direction = 1j * phase_direction * xp.exp(1j * phase)
        radial = sample.source.radial_electric_line_integrals(
            anchors[None, :, :],
            block.coordinates_au[:, None, :],
            backend,
        )
        radial_direction += _pair_with_factor(
            block.values,
            block.weights_au,
            factor_direction * radial[:, None, :],
            xp,
        )

    prefactor = 1j * sample.static_result.charge / sample.static_result.hbar
    scalar_at_anchors = sample.source.scalar_potential(anchors, backend)
    connection_endpoint = static_direction.endpoint_phase_direction * sample.connection.connection
    connection_internal = (
        prefactor
        * sample.static_result.endpoint_link
        * (static_direction.barred_overlap * scalar_at_anchors[None, :] - radial_direction)
    )
    mechanical_endpoint = static_direction.endpoint.mechanical
    mechanical_internal_triangle = sample.static_result.endpoint_link * (
        static_direction.barred_kinetic_triangle + static_direction.barred_nuclear_attraction
    )
    mechanical_internal_anchored = (
        sample.static_result.endpoint_link * static_direction.barred_kinetic_anchored
    )
    backend.synchronize()
    return ExactMagneticFieldSourceDirection(
        magnetic_field_direction_au=direction,
        metric_endpoint=static_direction.endpoint.overlap,
        metric_internal=static_direction.internal.overlap,
        mechanical_endpoint=mechanical_endpoint,
        mechanical_internal_triangle=mechanical_internal_triangle,
        mechanical_internal_anchored=mechanical_internal_anchored,
        connection_endpoint=connection_endpoint,
        connection_internal=connection_internal,
        radial_electric_integral_direction=radial_direction,
        static_direction=static_direction,
    )


def evaluate_exact_static_wilson_grid_one_electron_action(
    quadrature: AOQuadrature,
    vector_potential: Any,
    *,
    charge: float = -1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> ExactStaticWilsonGridOneElectronAction:
    r"""Evaluate the static exact-Wilson one-electron grid action.

    For each dressed AO ``chi_i=exp(i q a_i/hbar) phi_i``, the covariant
    momentum is evaluated as

    ``pi_A chi_i = exp(i q a_i/hbar)
       [-i hbar grad(phi_i) + q(grad(a_i)-A) phi_i]``.

    The same blocked quadrature is used for overlap, kinetic, and local
    electron--nuclear terms.  This makes source finite differences an
    independent rebuild of the action rather than a finite difference of an
    already differentiated matrix.
    """

    if not isinstance(quadrature, AOQuadrature):
        raise TypeError("quadrature must be an AOQuadrature")
    _validate_static_vector_potential(vector_potential)
    checked_charge = _finite_parameter(charge, "charge")
    checked_mass = _positive_parameter(mass, "mass")
    checked_hbar = _positive_parameter(hbar, "hbar")
    backend = quadrature.backend
    xp = backend.namespace
    reference = quadrature.reference
    geometry = build_magnetic_pair_geometry(
        reference.core_operators.nuclei.coordinates_au,
        reference.anchor_topology.ao_to_atom,
        backend,
    )
    anchors = geometry.ao_anchor_coordinates_au
    nuclear_provider = bind_local_potential(
        NuclearAttractionProvider(),
        reference,
        backend,
    )
    nao = reference.core_operators.nao
    overlap = backend.zeros((nao, nao), dtype=xp.complex128)
    kinetic = backend.zeros((nao, nao), dtype=xp.complex128)
    nuclear = backend.zeros((nao, nao), dtype=xp.complex128)
    prefactor = 1j * checked_charge / checked_hbar

    for block in quadrature.blocks():
        values = block.values
        gradients = xp.moveaxis(block.gradients, 0, -1)
        bare_momentum = -1j * checked_hbar * gradients
        line = vector_potential.straight_line_integrals(
            anchors[None, :, :],
            block.coordinates_au[:, None, :],
            backend,
        )
        line_gradient = vector_potential.straight_line_integral_gradients(
            anchors[None, :, :],
            block.coordinates_au[:, None, :],
            backend,
        )
        potential = vector_potential.vector_potential(block.coordinates_au, backend)
        residual = line_gradient - potential[:, None, :]
        wilson = xp.exp(prefactor * line)
        dressed_values = wilson * values
        dressed_momentum = wilson[:, :, None] * (
            bare_momentum + checked_charge * residual * values[:, :, None]
        )
        overlap += _ordinary_pair(
            dressed_values,
            dressed_values,
            block.weights_au,
            xp,
        )
        kinetic += (1.0 / (2.0 * checked_mass)) * _ordinary_vector_pair(
            dressed_momentum,
            dressed_momentum,
            block.weights_au,
            xp,
        )
        nuclear_weights = block.weights_au * nuclear_provider.values_au(
            block.coordinates_au
        )
        nuclear += _ordinary_pair(
            dressed_values,
            dressed_values,
            nuclear_weights,
            xp,
        )

    backend.synchronize()
    return ExactStaticWilsonGridOneElectronAction(
        overlap=overlap,
        kinetic=kinetic,
        nuclear_attraction=nuclear,
        reference_fingerprint_sha256=reference.fingerprint_sha256,
        grid_fingerprint_sha256=quadrature.grid.fingerprint_sha256,
        backend=quadrature.backend_config.kind.value,
        device_index=quadrature.backend_config.device_index,
        charge=checked_charge,
        mass=checked_mass,
        hbar=checked_hbar,
    )


def evaluate_exact_weak_vector_potential_source_direction(
    quadrature: AOQuadrature,
    sample: ExactWilsonOneElectronSample,
    variation: Any,
) -> ExactWeakVectorPotentialSourceDirection:
    r"""Differentiate exact dressed-AO data along a smooth static ``alpha(r)``.

    This is a weak continuum-current probe: it returns the response paired
    with the supplied spatial test variation and does not claim to reconstruct
    a pointwise current density.  The variation has zero time derivative.
    For a dynamic background, ``delta omega_t`` is nevertheless nonzero
    because both sides of the temporal matrix inherit the varied Wilson
    frame.  That embedding response is evaluated below from the same dressed
    AO grid action; it vanishes identically for a temporally static source.
    """

    if not isinstance(quadrature, AOQuadrature):
        raise TypeError("quadrature must be an AOQuadrature")
    if not isinstance(sample, ExactWilsonOneElectronSample):
        raise TypeError("sample must be an ExactWilsonOneElectronSample")
    _validate_static_vector_potential(variation)
    if sample.static_result.reference_fingerprint_sha256 != (
        quadrature.reference.fingerprint_sha256
    ):
        raise ConfigurationError("sample belongs to a different AO reference")
    if sample.static_result.grid_fingerprint_sha256 != quadrature.grid.fingerprint_sha256:
        raise ConfigurationError("sample belongs to a different AO quadrature grid")
    backend = quadrature.backend
    xp = backend.namespace
    reference = quadrature.reference
    geometry = build_magnetic_pair_geometry(
        reference.core_operators.nuclei.coordinates_au,
        reference.anchor_topology.ao_to_atom,
        backend,
    )
    anchors = geometry.ao_anchor_coordinates_au
    gauge = sample.source.gauge
    prefactor = 1j * sample.static_result.charge / sample.static_result.hbar
    nuclear_provider = bind_local_potential(
        NuclearAttractionProvider(),
        reference,
        backend,
    )
    nao = reference.core_operators.nao
    frame_overlap = backend.zeros((nao, nao), dtype=xp.complex128)
    frame_overlap_rate = backend.zeros((nao, nao), dtype=xp.complex128)
    metric = backend.zeros((nao, nao), dtype=xp.complex128)
    kinetic_embedding = backend.zeros((nao, nao), dtype=xp.complex128)
    kinetic_explicit = backend.zeros((nao, nao), dtype=xp.complex128)
    nuclear = backend.zeros((nao, nao), dtype=xp.complex128)
    connection = backend.zeros((nao, nao), dtype=xp.complex128)
    kinetic_scale = 1.0 / (2.0 * sample.static_result.mass)

    for block in quadrature.blocks():
        values = block.values
        gradients = xp.moveaxis(block.gradients, 0, -1)
        bare_momentum = -1j * sample.static_result.hbar * gradients
        base_line = gauge.anchor_to_point_line_integrals(
            anchors,
            block.coordinates_au,
            backend,
        )
        base_line_gradient = gauge.anchor_to_point_line_integral_gradients(
            anchors,
            block.coordinates_au,
            backend,
        )
        base_vector = gauge.vector_potential(block.coordinates_au, backend)
        base_residual = base_line_gradient - base_vector[:, None, :]
        wilson = xp.exp(prefactor * base_line)
        dressed_values = wilson * values
        reduced_momentum = (
            bare_momentum + sample.static_result.charge * base_residual * values[:, :, None]
        )
        dressed_momentum = wilson[:, :, None] * reduced_momentum

        direction_line = variation.straight_line_integrals(
            anchors[None, :, :],
            block.coordinates_au[:, None, :],
            backend,
        )
        direction_line_gradient = variation.straight_line_integral_gradients(
            anchors[None, :, :],
            block.coordinates_au[:, None, :],
            backend,
        )
        direction_vector = variation.vector_potential(block.coordinates_au, backend)
        dressed_values_direction = prefactor * direction_line * dressed_values
        dressed_momentum_embedding = wilson[:, :, None] * (
            prefactor * direction_line[:, :, None] * reduced_momentum
            + sample.static_result.charge
            * direction_line_gradient
            * values[:, :, None]
        )
        dressed_momentum_explicit = wilson[:, :, None] * (
            -sample.static_result.charge
            * direction_vector[:, None, :]
            * values[:, :, None]
        )
        base_line_rate = sample.source.gauge_rate.anchor_to_point_line_integrals(
            anchors,
            block.coordinates_au,
            backend,
        )
        scalar = sample.source.scalar_potential(block.coordinates_au, backend)
        temporal_factor = prefactor * (base_line_rate + scalar[:, None])
        dressed_temporal_values = temporal_factor * dressed_values
        dressed_temporal_values_direction = temporal_factor * dressed_values_direction

        frame_overlap += _ordinary_pair(
            dressed_values,
            dressed_values_direction,
            block.weights_au,
            xp,
        )
        frame_overlap_rate += _ordinary_pair(
            dressed_temporal_values,
            dressed_values_direction,
            block.weights_au,
            xp,
        ) + _ordinary_pair(
            dressed_values,
            dressed_temporal_values_direction,
            block.weights_au,
            xp,
        )
        metric += _ordinary_pair(
            dressed_values_direction,
            dressed_values,
            block.weights_au,
            xp,
        ) + _ordinary_pair(
            dressed_values,
            dressed_values_direction,
            block.weights_au,
            xp,
        )
        kinetic_embedding += kinetic_scale * (
            _ordinary_vector_pair(
                dressed_momentum_embedding,
                dressed_momentum,
                block.weights_au,
                xp,
            )
            + _ordinary_vector_pair(
                dressed_momentum,
                dressed_momentum_embedding,
                block.weights_au,
                xp,
            )
        )
        kinetic_explicit += kinetic_scale * (
            _ordinary_vector_pair(
                dressed_momentum_explicit,
                dressed_momentum,
                block.weights_au,
                xp,
            )
            + _ordinary_vector_pair(
                dressed_momentum,
                dressed_momentum_explicit,
                block.weights_au,
                xp,
            )
        )
        nuclear_weights = block.weights_au * nuclear_provider.values_au(block.coordinates_au)
        nuclear += _ordinary_pair(
            dressed_values_direction,
            dressed_values,
            nuclear_weights,
            xp,
        ) + _ordinary_pair(
            dressed_values,
            dressed_values_direction,
            nuclear_weights,
            xp,
        )
        connection += _ordinary_pair(
            dressed_values_direction,
            dressed_temporal_values,
            block.weights_au,
            xp,
        ) + _ordinary_pair(
            dressed_values,
            dressed_temporal_values_direction,
            block.weights_au,
            xp,
        )

    kinetic = kinetic_embedding + kinetic_explicit
    if isinstance(variation, GaussianScalarGaugeVariation):
        # A time-independent pure-gauge direction transforms the exact
        # temporal matrix homogeneously in the anchor frame.  Use the stable
        # production connection here rather than commuting the variation
        # with its less accurate all-grid oracle; this is an analytic source
        # specialization, not a state or metric correction.
        gauge_values = variation.scalar_field(anchors, backend)
        eta = prefactor * gauge_values
        connection = (
            eta[:, None] * sample.connection.connection
            - sample.connection.connection * eta[None, :]
        )
    backend.synchronize()
    return ExactWeakVectorPotentialSourceDirection(
        variation=variation,
        frame_overlap=frame_overlap,
        frame_overlap_rate=frame_overlap_rate,
        metric=metric,
        kinetic_embedding=kinetic_embedding,
        kinetic_explicit=kinetic_explicit,
        kinetic=kinetic,
        nuclear_attraction=nuclear,
        connection=connection,
    )


def _validate_static_vector_potential(value: Any) -> None:
    for method in (
        "vector_potential",
        "straight_line_integrals",
        "straight_line_integral_gradients",
    ):
        if not callable(getattr(value, method, None)):
            raise TypeError(f"vector potential must provide {method}")


def _pair_with_factor(values: Any, weights: Any, factor: Any, xp: Any) -> Any:
    return xp.einsum(
        "p,pm,pn,pmn->mn",
        weights,
        values.conj(),
        values,
        factor,
        optimize=True,
    )


def _ordinary_pair(left: Any, right: Any, weights: Any, xp: Any) -> Any:
    return xp.einsum(
        "p,pm,pn->mn",
        weights,
        left.conj(),
        right,
        optimize=True,
    )


def _ordinary_vector_pair(left: Any, right: Any, weights: Any, xp: Any) -> Any:
    return xp.einsum(
        "p,pmx,pnx->mn",
        weights,
        left.conj(),
        right,
        optimize=True,
    )


def _direction_vector(value: object, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ConfigurationError(f"{name} must be a finite Cartesian vector")
    return (float(array[0]), float(array[1]), float(array[2]))


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
