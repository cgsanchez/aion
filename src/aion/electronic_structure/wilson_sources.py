"""Action-derived sources and static observables for nonlinear Wilson models.

The objects in this module compose the accepted exact-Wilson one-electron,
RI--Wilson Hartree, and Wilson-LDA actions.  No ambient momentum observable is
borrowed from another formulation: every current value is a weak derivative
of the same declared discrete action used for its matter equation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aion.electromagnetism import GaussianScalarGaugeVariation
from aion.electronic_structure.adiabatic import expectation
from aion.electronic_structure.ri_wilson_hartree import RIWilsonHartreeResult
from aion.electronic_structure.time_connection import (
    ExactStaticWilsonGridOneElectronAction,
    ExactWeakVectorPotentialSourceDirection,
    ExactWilsonOneElectronSample,
    evaluate_exact_static_wilson_grid_one_electron_action,
)
from aion.electronic_structure.wilson_density import (
    ExactWilsonDensityResult,
    evaluate_exact_uniform_magnetic_wilson_density,
    evaluate_exact_uniform_magnetic_wilson_density_source_direction,
)
from aion.electronic_structure.wilson_gga import WilsonGGAResult
from aion.electronic_structure.wilson_lda import WilsonLDAResult
from aion.electronic_structure.wilson_stationary import ExactWilsonStationaryModel
from aion.errors import ConfigurationError, FormulationError
from aion.formulations.action import (
    OneElectronActionContraction,
    OneElectronActionHistoryDirection,
    OneElectronActionMatrixDirection,
    one_electron_velocity_density,
    restricted_one_electron_action_directional_derivative,
    restricted_one_electron_action_full_directional_derivative,
    restricted_one_electron_action_history_directional_derivative,
    restricted_one_electron_action_value,
)
from aion.formulations.exact_one_electron import (
    evaluate_exact_weak_current_pairing,
    exact_pure_gauge_action_direction,
    exact_wilson_one_electron_triple,
)
from aion.formulations.types import EOMTriple


@dataclass(frozen=True, slots=True)
class ExactWilsonChargeObservation:
    """Exact signed Wilson charge and its grid and stable-metric totals."""

    particle_density: Any
    signed_charge_density: Any
    integrated_charge_grid: Any
    integrated_charge_metric: Any
    metric_particle_number: Any
    density_evaluation: ExactWilsonDensityResult


@dataclass(frozen=True, slots=True)
class StaticNonlinearWilsonGridAction:
    """One static fixed-history nonlinear action value on a common grid."""

    one_electron_matrices: ExactStaticWilsonGridOneElectronAction
    one_electron_action: OneElectronActionContraction
    hartree: RIWilsonHartreeResult
    exchange_correlation: WilsonLDAResult | WilsonGGAResult | None
    closure_energy: Any
    electronic_action_value: Any
    triple: EOMTriple


@dataclass(frozen=True, slots=True)
class NonlinearWeakCurrentPairing:
    """Resolved weak current pairing of one nonlinear Wilson action.

    ``embedding_pairing`` is the complete residual-dependent frame-motion
    term off shell.  ``tangential_pairing`` is its coefficient-variation
    component.  ``normal_subspace_pairing`` is the remaining normal virtual
    work and becomes the intrinsic subspace-response current on shell.
    """

    response: ExactWeakVectorPotentialSourceDirection
    one_electron_action: OneElectronActionContraction
    hartree: RIWilsonHartreeResult
    exchange_correlation: WilsonLDAResult | WilsonGGAResult | None
    closure_pairing: Any
    total_pairing: Any
    ambient_minimal_pairing: Any
    one_electron_embedding_pairing: Any
    embedding_pairing: Any
    tangential_pairing: Any
    normal_subspace_pairing: Any
    on_shell_pairing: Any
    on_shell_decomposition_residual: Any
    frame_connection: Any
    frame_connection_rate: Any
    coefficient_density_direction: Any
    velocity_density_direction: Any


@dataclass(frozen=True, slots=True)
class NonlinearPureGaugeWardResult:
    """Complete off-shell pure-gauge action identity and matter defect."""

    source_pairing: Any
    matter_pairing: Any
    total_ward_residual: Any
    one_electron_source_pairing: Any
    one_electron_matter_pairing: Any
    closure_source_pairing: Any
    closure_matter_pairing: Any
    lower_coefficient_residual: Any
    lower_coefficient_residual_relative_norm: float
    coefficient_density: Any
    velocity_density: Any


@dataclass(frozen=True, slots=True)
class NonlinearDensityPureGaugeWardResult:
    """Pure-gauge Ward identity expressed entirely in density variables."""

    source_pairing: Any
    matter_pairing: Any
    total_ward_residual: Any
    one_electron_source_pairing: Any
    one_electron_matter_pairing: Any
    closure_source_pairing: Any
    closure_matter_pairing: Any
    lower_density_shell_residual: Any
    lower_density_shell_residual_relative_norm: float
    coefficient_density: Any
    velocity_density: Any


@dataclass(frozen=True, slots=True)
class NonlinearWeakContinuityResult:
    """Instantaneous weak finite-region and global charge balances."""

    charge: ExactWilsonChargeObservation
    velocity_density: Any
    coefficient_density_derivative: Any
    weighted_charge: Any
    weighted_charge_derivative: Any
    weak_current_pairing: NonlinearWeakCurrentPairing
    finite_region_residual: Any
    total_charge_rate: Any
    global_charge_residual: Any


def evaluate_exact_wilson_charge(
    model: ExactWilsonStationaryModel,
    coefficient_density: object,
) -> ExactWilsonChargeObservation:
    """Evaluate ``rho=q*n_W`` and its integrated total without an EOM."""

    _validate_model(model)
    density = model.hartree_evaluator._validated_density(coefficient_density)
    result = evaluate_exact_uniform_magnetic_wilson_density(
        model.quadrature,
        density,
        model.gauge,
        charge=model.hartree_action.charge,
        hbar=model.hartree_action.hbar,
    )
    xp = model.namespace
    particle_density = xp.real(result.density_direct)
    charge = model.hartree_action.charge
    metric_number = xp.real(
        xp.einsum("ij,ji->", density, model.overlap, optimize=True)
    )
    return ExactWilsonChargeObservation(
        particle_density=particle_density,
        signed_charge_density=charge * particle_density,
        integrated_charge_grid=charge * xp.real(result.particle_number_direct_integral),
        integrated_charge_metric=charge * metric_number,
        metric_particle_number=metric_number,
        density_evaluation=result,
    )


def evaluate_static_nonlinear_wilson_grid_action(
    model: ExactWilsonStationaryModel,
    coefficient_density: object,
    velocity_density: object,
    vector_potential: Any,
) -> StaticNonlinearWilsonGridAction:
    """Rebuild one static nonlinear grid action for a source finite difference."""

    _validate_model(model)
    backend = model.backend
    xp = model.namespace
    density = model.hartree_evaluator._validated_density(coefficient_density)
    velocity = backend.asarray(velocity_density, dtype=xp.complex128)
    backend.assert_resident(velocity, name="velocity density")
    if velocity.shape != density.shape:
        raise ConfigurationError("velocity_density has an incompatible shape")
    one_electron = evaluate_exact_static_wilson_grid_one_electron_action(
        model.quadrature,
        vector_potential,
        charge=model.hartree_action.charge,
        mass=model.one_electron.mass,
        hbar=model.hartree_action.hbar,
    )
    zero_connection = backend.zeros(density.shape, dtype=xp.complex128)
    triple = EOMTriple(
        metric=one_electron.overlap,
        hamiltonian_eom=one_electron.mechanical,
        connection=zero_connection,
    )
    one_action = restricted_one_electron_action_value(
        density,
        velocity,
        triple,
        backend,
        hbar=model.hartree_action.hbar,
    )
    hartree = model.hartree_evaluator.evaluate(
        density,
        vector_potential,
        charge=model.hartree_action.charge,
        hbar=model.hartree_action.hbar,
    )
    evaluator = model.xc_evaluator
    xc = (
        None
        if evaluator is None
        else evaluator.evaluate(
            density,
            vector_potential,
            charge=model.hartree_action.charge,
            hbar=model.hartree_action.hbar,
        )
    )
    closure = hartree.energy + (
        backend.asarray(0.0, dtype=xp.float64) if xc is None else xc.energy
    )
    return StaticNonlinearWilsonGridAction(
        one_electron_matrices=one_electron,
        one_electron_action=one_action,
        hartree=hartree,
        exchange_correlation=xc,
        closure_energy=closure,
        electronic_action_value=one_action.total - closure,
        triple=triple,
    )


def evaluate_nonlinear_weak_current_pairing(
    model: ExactWilsonStationaryModel,
    sample: ExactWilsonOneElectronSample,
    coefficient_density: object,
    velocity_density: object,
    variation: Any,
    *,
    one_electron_triple: EOMTriple | None = None,
) -> NonlinearWeakCurrentPairing:
    r"""Evaluate the complete off-shell weak vector-current pairing.

    The fixed coefficient history is represented by
    ``P=C f C^dagger`` and ``R=Cdot f C^dagger``.  The optional base triple
    selects the one-electron action used in the tangential matter variation;
    it is useful when qualifying the all-grid realization by finite
    differences.  The source response itself is always rebuilt from the
    exact dressed-AO grid action.
    """

    _validate_model_sample(model, sample)
    backend = model.backend
    xp = model.namespace
    density = model.hartree_evaluator._validated_density(coefficient_density)
    velocity = backend.asarray(velocity_density, dtype=xp.complex128)
    backend.assert_resident(velocity, name="velocity density")
    if velocity.shape != density.shape:
        raise ConfigurationError("velocity_density has an incompatible shape")

    one = evaluate_exact_weak_current_pairing(
        model.quadrature,
        sample,
        density,
        velocity,
        variation,
    )
    hartree, xc = _closure_source_evaluations(model, density, variation)
    assert hartree.source_energy_direction is not None
    closure_pairing = -hartree.source_energy_direction
    if xc is not None:
        assert xc.source_energy_direction is not None
        closure_pairing = closure_pairing - xc.source_energy_direction
    total = one.value + closure_pairing

    zero = backend.zeros(density.shape, dtype=xp.complex128)
    minimal_direction = OneElectronActionMatrixDirection(
        metric=zero,
        mechanical=one.response.kinetic_explicit,
        connection=zero,
    )
    minimal = restricted_one_electron_action_directional_derivative(
        density,
        velocity,
        minimal_direction,
        backend,
        hbar=model.hartree_action.hbar,
    ).total
    one_embedding = one.value - minimal
    embedding = one_embedding + closure_pairing

    base = (
        exact_wilson_one_electron_triple(sample)
        if one_electron_triple is None
        else one_electron_triple
    )
    try:
        frame_connection = xp.linalg.solve(base.metric, one.response.frame_overlap)
        frame_connection_rate = xp.linalg.solve(
            base.metric,
            one.response.frame_overlap_rate
            - sample.connection.metric_dot @ frame_connection,
        )
    except Exception as exc:
        raise FormulationError("weak frame-connection or rate solve failed") from exc
    density_direction = (
        frame_connection @ density + density @ frame_connection.conj().T
    )
    velocity_direction = (
        frame_connection_rate @ density
        + frame_connection @ velocity
        + velocity @ frame_connection.conj().T
    )
    tangential_one = restricted_one_electron_action_history_directional_derivative(
        OneElectronActionHistoryDirection(
            density=density_direction,
            velocity_density=velocity_direction,
        ),
        base,
        backend,
        hbar=model.hartree_action.hbar,
    ).total
    closure_lower = hartree.lower_coulomb_matrix
    if xc is not None:
        closure_lower = closure_lower + xc.lower_xc_matrix
    tangential_closure = -expectation(density_direction, closure_lower, xp)
    tangential = tangential_one + tangential_closure
    normal = embedding - tangential
    on_shell = minimal + normal
    return NonlinearWeakCurrentPairing(
        response=one.response,
        one_electron_action=one.action_contraction,
        hartree=hartree,
        exchange_correlation=xc,
        closure_pairing=closure_pairing,
        total_pairing=total,
        ambient_minimal_pairing=minimal,
        one_electron_embedding_pairing=one_embedding,
        embedding_pairing=embedding,
        tangential_pairing=tangential,
        normal_subspace_pairing=normal,
        on_shell_pairing=on_shell,
        on_shell_decomposition_residual=total - tangential - on_shell,
        frame_connection=frame_connection,
        frame_connection_rate=frame_connection_rate,
        coefficient_density_direction=density_direction,
        velocity_density_direction=velocity_direction,
    )


def evaluate_nonlinear_pure_gauge_ward(
    model: ExactWilsonStationaryModel,
    sample: ExactWilsonOneElectronSample,
    coefficients: object,
    coefficient_velocities: object,
    occupations: object,
    gauge_parameter: GaussianScalarGaugeVariation,
    *,
    gauge_parameter_rate: GaussianScalarGaugeVariation | None = None,
) -> NonlinearPureGaugeWardResult:
    """Evaluate the complete off-shell pure-gauge Ward action identity."""

    _validate_model_sample(model, sample)
    if not isinstance(gauge_parameter, GaussianScalarGaugeVariation):
        raise TypeError("gauge_parameter must be a GaussianScalarGaugeVariation")
    if gauge_parameter_rate is not None and not isinstance(
        gauge_parameter_rate, GaussianScalarGaugeVariation
    ):
        raise TypeError("gauge_parameter_rate must be a GaussianScalarGaugeVariation")
    backend = model.backend
    xp = model.namespace
    coefficient_array = backend.asarray(coefficients, dtype=xp.complex128)
    velocity_array = backend.asarray(coefficient_velocities, dtype=xp.complex128)
    occupation_array = backend.asarray(occupations, dtype=xp.float64)
    if (
        coefficient_array.ndim != 2
        or velocity_array.shape != coefficient_array.shape
        or occupation_array.shape != (coefficient_array.shape[1],)
        or coefficient_array.shape[0] != sample.metric.shape[0]
    ):
        raise ConfigurationError("coefficient history has incompatible shapes")
    density = xp.einsum(
        "mi,i,ni->mn",
        coefficient_array,
        occupation_array,
        coefficient_array.conj(),
        optimize=True,
    )
    velocity_density = xp.einsum(
        "mi,i,ni->mn",
        velocity_array,
        occupation_array,
        coefficient_array.conj(),
        optimize=True,
    )
    topology = model.quadrature.reference.anchor_topology
    site_coordinates = model.quadrature.reference.core_operators.nuclei.coordinates_au
    site_values = gauge_parameter.scalar_field(site_coordinates, backend)
    site_rates = (
        backend.zeros(site_values.shape, dtype=xp.float64)
        if gauge_parameter_rate is None
        else gauge_parameter_rate.scalar_field(site_coordinates, backend)
    )
    pure = exact_pure_gauge_action_direction(
        sample,
        density,
        velocity_density,
        site_values,
        site_rates,
        topology.ao_to_atom,
        backend,
    )
    one_full = restricted_one_electron_action_full_directional_derivative(
        density,
        velocity_density,
        exact_wilson_one_electron_triple(sample),
        pure.matrix,
        pure.history,
        backend,
        hbar=model.hartree_action.hbar,
    )
    hartree, xc = _closure_source_evaluations(model, density, gauge_parameter)
    assert hartree.source_energy_direction is not None
    closure_source = -hartree.source_energy_direction
    closure_lower = hartree.lower_coulomb_matrix
    if xc is not None:
        assert xc.source_energy_direction is not None
        closure_source = closure_source - xc.source_energy_direction
        closure_lower = closure_lower + xc.lower_xc_matrix
    closure_matter = -expectation(pure.history.density, closure_lower, xp)
    source = one_full.source.total + closure_source
    matter = one_full.history.total + closure_matter

    full_action = model.evaluate(density)
    lower_residual = (
        1j * model.hartree_action.hbar * sample.metric @ velocity_array
        - (
            full_action.lower_mechanical_matrix
            - 1j * model.hartree_action.hbar * sample.connection.connection
        )
        @ coefficient_array
    )
    residual_scale = xp.maximum(
        xp.asarray(1.0),
        xp.linalg.norm(full_action.lower_mechanical_matrix @ coefficient_array),
    )
    relative_residual = backend.scalar_to_float(
        xp.linalg.norm(lower_residual) / residual_scale
    )
    return NonlinearPureGaugeWardResult(
        source_pairing=source,
        matter_pairing=matter,
        total_ward_residual=source + matter,
        one_electron_source_pairing=one_full.source.total,
        one_electron_matter_pairing=one_full.history.total,
        closure_source_pairing=closure_source,
        closure_matter_pairing=closure_matter,
        lower_coefficient_residual=lower_residual,
        lower_coefficient_residual_relative_norm=relative_residual,
        coefficient_density=density,
        velocity_density=velocity_density,
    )


def evaluate_nonlinear_density_pure_gauge_ward(
    model: ExactWilsonStationaryModel,
    sample: ExactWilsonOneElectronSample,
    coefficient_density: object,
    velocity_density: object,
    gauge_parameter: GaussianScalarGaugeVariation,
    *,
    gauge_parameter_rate: GaussianScalarGaugeVariation | None = None,
) -> NonlinearDensityPureGaugeWardResult:
    """Evaluate the pure-gauge Ward identity without reconstructing orbitals.

    The matter-shell diagnostic is the density form of the coefficient
    equation, ``i*hbar*S*R-(K_beta-i*hbar*omega)*P``.  This is the natural
    audit for a mixed-index density trajectory and is not a projection of
    that trajectory onto an orbital representation.
    """

    _validate_model_sample(model, sample)
    if not isinstance(gauge_parameter, GaussianScalarGaugeVariation):
        raise TypeError("gauge_parameter must be a GaussianScalarGaugeVariation")
    if gauge_parameter_rate is not None and not isinstance(
        gauge_parameter_rate, GaussianScalarGaugeVariation
    ):
        raise TypeError("gauge_parameter_rate must be a GaussianScalarGaugeVariation")
    backend = model.backend
    xp = model.namespace
    density = model.hartree_evaluator._validated_density(coefficient_density)
    velocity = backend.asarray(velocity_density, dtype=xp.complex128)
    backend.assert_resident(velocity, name="velocity density")
    if velocity.shape != density.shape:
        raise ConfigurationError("velocity_density has an incompatible shape")

    topology = model.quadrature.reference.anchor_topology
    site_coordinates = model.quadrature.reference.core_operators.nuclei.coordinates_au
    site_values = gauge_parameter.scalar_field(site_coordinates, backend)
    site_rates = (
        backend.zeros(site_values.shape, dtype=xp.float64)
        if gauge_parameter_rate is None
        else gauge_parameter_rate.scalar_field(site_coordinates, backend)
    )
    pure = exact_pure_gauge_action_direction(
        sample,
        density,
        velocity,
        site_values,
        site_rates,
        topology.ao_to_atom,
        backend,
    )
    one_full = restricted_one_electron_action_full_directional_derivative(
        density,
        velocity,
        exact_wilson_one_electron_triple(sample),
        pure.matrix,
        pure.history,
        backend,
        hbar=model.hartree_action.hbar,
    )
    hartree, xc = _closure_source_evaluations(model, density, gauge_parameter)
    assert hartree.source_energy_direction is not None
    closure_source = -hartree.source_energy_direction
    closure_lower = hartree.lower_coulomb_matrix
    if xc is not None:
        assert xc.source_energy_direction is not None
        closure_source = closure_source - xc.source_energy_direction
        closure_lower = closure_lower + xc.lower_xc_matrix
    closure_matter = -expectation(pure.history.density, closure_lower, xp)
    source = one_full.source.total + closure_source
    matter = one_full.history.total + closure_matter

    full_action = model.evaluate(density)
    shell_residual = (
        1j * model.hartree_action.hbar * sample.metric @ velocity
        - (
            full_action.lower_mechanical_matrix
            - 1j * model.hartree_action.hbar * sample.connection.connection
        )
        @ density
    )
    residual_scale = xp.maximum(
        xp.asarray(1.0),
        xp.linalg.norm(full_action.lower_mechanical_matrix @ density),
    )
    relative_residual = backend.scalar_to_float(
        xp.linalg.norm(shell_residual) / residual_scale
    )
    return NonlinearDensityPureGaugeWardResult(
        source_pairing=source,
        matter_pairing=matter,
        total_ward_residual=source + matter,
        one_electron_source_pairing=one_full.source.total,
        one_electron_matter_pairing=one_full.history.total,
        closure_source_pairing=closure_source,
        closure_matter_pairing=closure_matter,
        lower_density_shell_residual=shell_residual,
        lower_density_shell_residual_relative_norm=relative_residual,
        coefficient_density=density,
        velocity_density=velocity,
    )


def evaluate_nonlinear_weak_continuity(
    model: ExactWilsonStationaryModel,
    sample: ExactWilsonOneElectronSample,
    coefficient_density: object,
    spatial_weight: GaussianScalarGaugeVariation,
    *,
    one_electron_triple: EOMTriple | None = None,
) -> NonlinearWeakContinuityResult:
    """Evaluate one instantaneous on-shell weak continuity balance."""

    _validate_model_sample(model, sample)
    if not isinstance(spatial_weight, GaussianScalarGaugeVariation):
        raise TypeError("spatial_weight must be a GaussianScalarGaugeVariation")
    backend = model.backend
    xp = model.namespace
    density = model.hartree_evaluator._validated_density(coefficient_density)
    action = model.evaluate(density)
    base = (
        exact_wilson_one_electron_triple(sample)
        if one_electron_triple is None
        else one_electron_triple
    )
    closure_lower = (
        action.lower_mechanical_matrix - action.one_electron_matrix
    )
    full_triple = EOMTriple(
        metric=base.metric,
        hamiltonian_eom=base.hamiltonian_eom + closure_lower,
        connection=base.connection,
    )
    velocity = one_electron_velocity_density(
        density,
        full_triple,
        backend,
        hbar=model.hartree_action.hbar,
    )
    density_dot = velocity + velocity.conj().T
    current = evaluate_nonlinear_weak_current_pairing(
        model,
        sample,
        density,
        velocity,
        spatial_weight,
        one_electron_triple=base,
    )
    charge = evaluate_exact_wilson_charge(model, density)
    density_dot_grid = evaluate_exact_uniform_magnetic_wilson_density(
        model.quadrature,
        density_dot,
        model.gauge,
        charge=model.hartree_action.charge,
        hbar=model.hartree_action.hbar,
    )
    density_source_dot_grid = (
        evaluate_exact_uniform_magnetic_wilson_density_source_direction(
            model.quadrature,
            density,
            model.gauge,
            sample.source.gauge_rate,
            charge=model.hartree_action.charge,
            hbar=model.hartree_action.hbar,
        )
    )
    weights = backend.asarray(model.quadrature.grid.weights_au, dtype=xp.float64)
    coordinates = backend.asarray(
        model.quadrature.grid.coordinates_au,
        dtype=xp.float64,
    )
    test_values = spatial_weight.scalar_field(coordinates, backend)
    weighted_charge = xp.einsum(
        "p,p,p->",
        weights,
        test_values,
        charge.signed_charge_density,
        optimize=True,
    )
    weighted_charge_dot = model.hartree_action.charge * xp.real(
        xp.einsum(
            "p,p,p->",
            weights,
            test_values,
            density_dot_grid.density_direct
            + density_source_dot_grid.density_direction,
            optimize=True,
        )
    )
    metric_dot = sample.connection.metric_dot
    total_charge_rate = model.hartree_action.charge * xp.real(
        xp.einsum("ij,ji->", density_dot, base.metric, optimize=True)
        + xp.einsum("ij,ji->", density, metric_dot, optimize=True)
    )
    return NonlinearWeakContinuityResult(
        charge=charge,
        velocity_density=velocity,
        coefficient_density_derivative=density_dot,
        weighted_charge=weighted_charge,
        weighted_charge_derivative=weighted_charge_dot,
        weak_current_pairing=current,
        finite_region_residual=weighted_charge_dot - current.on_shell_pairing,
        total_charge_rate=total_charge_rate,
        global_charge_residual=total_charge_rate,
    )


def _closure_source_evaluations(
    model: ExactWilsonStationaryModel,
    density: Any,
    variation: Any,
) -> tuple[RIWilsonHartreeResult, WilsonLDAResult | WilsonGGAResult | None]:
    hartree = model.hartree_evaluator.evaluate(
        density,
        model.gauge,
        source_direction=variation,
        charge=model.hartree_action.charge,
        hbar=model.hartree_action.hbar,
    )
    if hartree.source_energy_direction is None:
        raise FormulationError("RI Hartree source direction was not evaluated")
    evaluator = model.xc_evaluator
    xc = (
        None
        if evaluator is None
        else evaluator.evaluate(
            density,
            model.gauge,
            source_direction=variation,
            charge=model.hartree_action.charge,
            hbar=model.hartree_action.hbar,
        )
    )
    if xc is not None and xc.source_energy_direction is None:
        raise FormulationError("Wilson XC source direction was not evaluated")
    return hartree, xc


def _validate_model(model: ExactWilsonStationaryModel) -> None:
    if not isinstance(model, ExactWilsonStationaryModel):
        raise TypeError("model must be an ExactWilsonStationaryModel")


def _validate_model_sample(
    model: ExactWilsonStationaryModel,
    sample: ExactWilsonOneElectronSample,
) -> None:
    _validate_model(model)
    if not isinstance(sample, ExactWilsonOneElectronSample):
        raise TypeError("sample must be an ExactWilsonOneElectronSample")
    if sample.static_result.reference_fingerprint_sha256 != (
        model.quadrature.reference.fingerprint_sha256
    ):
        raise ConfigurationError("model and sample use different AO references")
    if sample.static_result.grid_fingerprint_sha256 != model.quadrature.grid.fingerprint_sha256:
        raise ConfigurationError("model and sample use different AO grids")
    if sample.source.gauge != model.gauge:
        raise ConfigurationError("model and sample use different gauge representatives")
