"""Nonlinear exact-Wilson dynamics and instantaneous power observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aion.electromagnetism import (
    AffineMagneticGauge,
    AffineVectorFieldVariation,
    UniformMagneticSourceSample,
)
from aion.electronic_structure.adiabatic import expectation
from aion.electronic_structure.magnetic_matrices import (
    ExactStaticMagneticOneElectronResult,
    evaluate_exact_static_magnetic_one_electron_matrices,
)
from aion.electronic_structure.ri_wilson_hartree import PreparedRIWilsonHartreeAction
from aion.electronic_structure.time_connection import (
    ExactWilsonOneElectronSample,
    evaluate_exact_magnetic_field_source_direction,
    evaluate_exact_uniform_magnetic_time_connection,
)
from aion.electronic_structure.wilson_sources import (
    NonlinearWeakCurrentPairing,
    evaluate_nonlinear_weak_current_pairing,
)
from aion.electronic_structure.wilson_stationary import (
    ExactWilsonActionEvaluation,
    ExactWilsonStationaryFactory,
    ExactWilsonStationaryModel,
    WilsonStationaryBranch,
)
from aion.errors import ConfigurationError, FormulationError
from aion.formulations import EOMTriple, one_electron_velocity_density


@dataclass(frozen=True, slots=True)
class ExactWilsonDynamicSample:
    """One source-fixed nonlinear model and its exact temporal connection."""

    source: UniformMagneticSourceSample
    one_electron: ExactWilsonOneElectronSample
    model: ExactWilsonStationaryModel

    @property
    def backend(self) -> Any:
        return self.model.backend

    def evaluate(self, coefficient_density: object) -> ExactWilsonDynamicEvaluation:
        """Evaluate the complete action and nonlinear coefficient triple."""

        action = self.model.evaluate(coefficient_density)
        triple = EOMTriple(
            metric=self.one_electron.metric,
            hamiltonian_eom=action.lower_mechanical_matrix,
            connection=self.one_electron.connection.connection,
        )
        return ExactWilsonDynamicEvaluation(sample=self, action=action, triple=triple)


@dataclass(frozen=True, slots=True)
class PreparedExactWilsonDynamicSpatialAction:
    """Density-independent exact spatial data reusable over temporal sources."""

    factory: ExactWilsonStationaryFactory
    gauge: AffineMagneticGauge
    one_electron: ExactStaticMagneticOneElectronResult
    hartree_action: PreparedRIWilsonHartreeAction

    def sample(
        self,
        source: UniformMagneticSourceSample,
        branch: WilsonStationaryBranch | str,
    ) -> ExactWilsonDynamicSample:
        """Attach one temporal source and nonlinear closure branch."""

        return self.sample_from_one_electron(
            self.temporal_sample(source),
            branch,
        )

    def temporal_sample(
        self,
        source: UniformMagneticSourceSample,
    ) -> ExactWilsonOneElectronSample:
        """Attach temporal data while retaining the prepared spatial action."""

        if not isinstance(source, UniformMagneticSourceSample):
            raise TypeError("source must be a UniformMagneticSourceSample")
        if source.gauge != self.gauge:
            raise ConfigurationError("temporal source does not match the prepared spatial gauge")
        connection = evaluate_exact_uniform_magnetic_time_connection(
            self.factory.quadrature,
            source,
            self.one_electron,
            charge=self.factory.charge,
            hbar=self.factory.hbar,
        )
        return ExactWilsonOneElectronSample(
            source=source,
            lower_exact=self.one_electron.lower_exact,
            lower_exact_grid=self.one_electron.lower_exact_grid,
            connection=connection,
            ordinary_derivative_matrix=(
                self.one_electron.lower_exact.mechanical
                - 1j * self.factory.hbar * connection.connection
            ),
            static_result=self.one_electron,
        )

    def sample_from_one_electron(
        self,
        one_electron: ExactWilsonOneElectronSample,
        branch: WilsonStationaryBranch | str,
    ) -> ExactWilsonDynamicSample:
        """Bind a cached temporal sample to one nonlinear closure branch."""

        if not isinstance(one_electron, ExactWilsonOneElectronSample):
            raise TypeError("one_electron must be an ExactWilsonOneElectronSample")
        if one_electron.static_result is not self.one_electron:
            raise ConfigurationError("temporal sample does not belong to this spatial action")
        try:
            selected = WilsonStationaryBranch(branch)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"unsupported dynamic branch {branch!r}") from exc
        model = ExactWilsonStationaryModel(
            quadrature=self.factory.quadrature,
            gauge=self.gauge,
            branch=selected,
            one_electron=self.one_electron,
            hartree_evaluator=self.factory.hartree_evaluator,
            hartree_action=self.hartree_action,
            lda_evaluator=(
                self.factory.lda_evaluator
                if selected is WilsonStationaryBranch.KOHN_SHAM_LDA
                else None
            ),
            gga_evaluator=(
                self.factory.gga_evaluator
                if selected is WilsonStationaryBranch.KOHN_SHAM_GGA
                else None
            ),
            nuclear_repulsion_au=self.factory.nuclear_repulsion_au,
        )
        return ExactWilsonDynamicSample(
            source=one_electron.source,
            one_electron=one_electron,
            model=model,
        )


@dataclass(frozen=True, slots=True)
class ExactWilsonDynamicEvaluation:
    """Complete instantaneous action evaluation used by the nonlinear EOM."""

    sample: ExactWilsonDynamicSample
    action: ExactWilsonActionEvaluation
    triple: EOMTriple


@dataclass(frozen=True, slots=True)
class ExactWilsonPowerObservation:
    """Two independent forms of the complete nonlinear mechanical power."""

    velocity_density: Any
    electric_field_variation: AffineVectorFieldVariation
    current: NonlinearWeakCurrentPairing
    source_power_au: Any
    one_electron_matrix_rate_au: Any
    hartree_fixed_density_rate_au: Any
    exchange_correlation_fixed_density_rate_au: Any
    closure_fixed_density_rate_au: Any
    matrix_mechanical_energy_rate_au: Any
    power_identity_residual_au: Any


def prepare_exact_wilson_dynamic_sample(
    factory: ExactWilsonStationaryFactory,
    source: UniformMagneticSourceSample,
    branch: WilsonStationaryBranch | str,
) -> ExactWilsonDynamicSample:
    """Prepare one reusable nonlinear action at a prescribed source sample."""

    if not isinstance(factory, ExactWilsonStationaryFactory):
        raise TypeError("factory must be an ExactWilsonStationaryFactory")
    if not isinstance(source, UniformMagneticSourceSample):
        raise TypeError("source must be a UniformMagneticSourceSample")
    spatial = prepare_exact_wilson_dynamic_spatial_action(factory, source.gauge)
    return spatial.sample(source, branch)


def prepare_exact_wilson_dynamic_spatial_action(
    factory: ExactWilsonStationaryFactory,
    gauge: AffineMagneticGauge,
) -> PreparedExactWilsonDynamicSpatialAction:
    """Prepare exact spatial matrices and RI data once for one gauge sample."""

    if not isinstance(factory, ExactWilsonStationaryFactory):
        raise TypeError("factory must be an ExactWilsonStationaryFactory")
    if not isinstance(gauge, AffineMagneticGauge):
        raise TypeError("gauge must be an AffineMagneticGauge")
    static = evaluate_exact_static_magnetic_one_electron_matrices(
        factory.quadrature,
        (gauge.field,),
        direct_gauges=(gauge,),
        charge=factory.charge,
        mass=factory.mass,
        hbar=factory.hbar,
        include_direct_oracle=False,
    )[0]
    hartree_action = factory.hartree_evaluator.prepare_action(
        gauge,
        charge=factory.charge,
        hbar=factory.hbar,
    )
    return PreparedExactWilsonDynamicSpatialAction(
        factory=factory,
        gauge=gauge,
        one_electron=static,
        hartree_action=hartree_action,
    )


def evaluate_exact_wilson_power(
    evaluation: ExactWilsonDynamicEvaluation,
    coefficient_density: object,
) -> ExactWilsonPowerObservation:
    r"""Audit ``dU/dt=<j,E>`` at one on-shell nonlinear state.

    The matrix route evaluates Chapter 13's finite-matrix identity.  The
    source route pairs the complete action-derived on-shell current with the
    physical uniform-plus-induction electric field.  Fixed nuclei and the
    time-independent Coulomb kernel contribute no gauge-neutral power.
    """

    if not isinstance(evaluation, ExactWilsonDynamicEvaluation):
        raise TypeError("evaluation must be an ExactWilsonDynamicEvaluation")
    sample = evaluation.sample
    model = sample.model
    backend = model.backend
    xp = model.namespace
    density = model.hartree_evaluator._validated_density(coefficient_density)
    velocity = one_electron_velocity_density(
        density,
        evaluation.triple,
        backend,
        hbar=model.hartree_action.hbar,
    )
    electric = AffineVectorFieldVariation.from_uniform_magnetic_electric_field(
        electric_field_origin_au=sample.source.electric_field_origin_au,
        magnetic_field_dot_au=sample.source.magnetic_field_dot_au,
        origin_au=sample.source.origin_au,
    )
    current = evaluate_nonlinear_weak_current_pairing(
        model,
        sample.one_electron,
        density,
        velocity,
        electric,
    )

    magnetic_rate = sample.source.magnetic_field_dot_au
    if all(value == 0.0 for value in magnetic_rate):
        one_electron_rate = backend.zeros(density.shape, dtype=xp.complex128)
    else:
        one_electron_rate = evaluate_exact_magnetic_field_source_direction(
            model.quadrature,
            sample.one_electron,
            magnetic_rate,
        ).mechanical

    hartree_direction = model.hartree_evaluator.evaluate(
        density,
        model.gauge,
        source_direction=sample.source.gauge_rate,
        charge=model.hartree_action.charge,
        hbar=model.hartree_action.hbar,
    ).source_energy_direction
    if hartree_direction is None:
        raise FormulationError("Hartree fixed-density time direction was not evaluated")
    xc_direction = backend.asarray(0.0, dtype=xp.float64)
    xc_evaluator = model.xc_evaluator
    if xc_evaluator is not None:
        xc_value = xc_evaluator.evaluate(
            density,
            model.gauge,
            source_direction=sample.source.gauge_rate,
            charge=model.hartree_action.charge,
            hbar=model.hartree_action.hbar,
        ).source_energy_direction
        if xc_value is None:
            raise FormulationError("XC fixed-density time direction was not evaluated")
        xc_direction = xc_value
    closure_direction = hartree_direction + xc_direction

    try:
        gamma = xp.linalg.solve(
            evaluation.triple.metric,
            evaluation.triple.connection,
        )
    except Exception as exc:
        raise FormulationError("power-audit connection solve failed") from exc
    lower = evaluation.action.lower_mechanical_matrix
    matrix_operator = one_electron_rate - gamma.conj().T @ lower - lower @ gamma
    matrix_rate = expectation(density, matrix_operator, xp) + closure_direction
    source_power = xp.real(current.on_shell_pairing)
    return ExactWilsonPowerObservation(
        velocity_density=velocity,
        electric_field_variation=electric,
        current=current,
        source_power_au=source_power,
        one_electron_matrix_rate_au=expectation(density, one_electron_rate, xp),
        hartree_fixed_density_rate_au=hartree_direction,
        exchange_correlation_fixed_density_rate_au=xc_direction,
        closure_fixed_density_rate_au=closure_direction,
        matrix_mechanical_energy_rate_au=matrix_rate,
        power_identity_residual_au=matrix_rate - source_power,
    )
