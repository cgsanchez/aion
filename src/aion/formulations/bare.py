"""Bare length- and velocity-gauge formulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aion.config import FormulationKind, GaugeRepresentation
from aion.electronic_structure import expectation, hermitian_part
from aion.formulations.base import FormulationContext
from aion.formulations.events import apply_electric_kick
from aion.formulations.kernels import density_derivative, uniform_mechanical_current
from aion.formulations.types import (
    AODensity,
    CurrentLedger,
    EnergyLedger,
    EOMTriple,
    FormulationSourceSample,
    InstantaneousEvaluation,
    PowerLedger,
)


def _baseline(value: Any | None, context: FormulationContext) -> Any | None:
    if value is None:
        return None
    return context.backend.asarray(value, dtype=context.namespace.float64)


@dataclass(frozen=True, slots=True)
class BareLengthGauge:
    context: FormulationContext
    kind: FormulationKind = field(default=FormulationKind.BARE_LENGTH_GAUGE, init=False)
    gauge: GaugeRepresentation = field(default=GaugeRepresentation.LENGTH, init=False)

    @property
    def observable_dependencies(self) -> frozenset[str]:
        return frozenset({"density", "source", "eom", "density_dot", "field_free_dft"})

    def evaluate(
        self, density: AODensity, source: FormulationSourceSample
    ) -> InstantaneousEvaluation:
        self.context.validate_source(source, self.gauge)
        rho = self.context.require_density(density)
        xp = self.context.namespace
        dft = self.context.electronic_model.build(rho)
        position = self.context.relative_position_operators()
        source_coupling = -self.context.charge * xp.einsum(
            "x,xij->ij", source.electric_field, position, optimize=True
        )
        hamiltonian = hermitian_part(dft.hamiltonian + source_coupling)
        metric = self.context.backend.asarray(
            self.context.workspace.require("operators.overlap"), dtype=xp.complex128
        )
        zero = xp.zeros_like(metric, dtype=xp.complex128)
        triple = EOMTriple(metric, hamiltonian, zero)
        rho_dot = density_derivative(
            rho,
            metric,
            hamiltonian,
            zero,
            hbar=self.context.hbar,
            backend=self.context.backend,
        )
        return InstantaneousEvaluation(
            triple=triple,
            density=density,
            density_dot=rho_dot,
            field_free_density=rho,
            field_free_density_dot=rho_dot,
            field_free_dft=dft,
            hamiltonian_dynamic=hamiltonian,
            source_hamiltonian=hamiltonian,
            metric_dot=zero,
            d_site_metric=zero,
        )

    def currents(
        self, evaluation: InstantaneousEvaluation, source: FormulationSourceSample
    ) -> CurrentLedger:
        self.context.validate_source(source, self.gauge)
        xp = self.context.namespace
        rho = evaluation.density.matrix
        electronic = self.context.bare_electronic_dipole(rho)
        nuclear = self.context.fixed_nuclear_dipole()
        dipole_dot = self.context.charge * xp.real(
            xp.einsum(
                "ij,xji->x",
                evaluation.density_dot,
                self.context.relative_position_operators(),
                optimize=True,
            )
        )
        mechanical = uniform_mechanical_current(
            rho,
            self.context.workspace.require("operators.canonical_momentum"),
            evaluation.triple.metric,
            source.vector_potential_reduced,
            charge=self.context.charge,
            mass=self.context.mass,
            backend=self.context.backend,
        )
        power = xp.dot(source.electric_field, dipole_dot)
        return CurrentLedger(
            electronic_dipole=electronic,
            fixed_nuclear_dipole=nuclear,
            total_dipole=electronic + nuclear,
            dipole_derivative_analytic=dipole_dot,
            primary_current=dipole_dot,
            source_current=dipole_dot,
            source_work_rate=power,
            mechanical_paramagnetic_current=mechanical.paramagnetic,
            mechanical_diamagnetic_current=mechanical.diamagnetic,
            mechanical_total_current=mechanical.total,
            source_current_dipole_derivative_residual=dipole_dot - dipole_dot,
        )

    def energy(
        self,
        evaluation: InstantaneousEvaluation,
        source: FormulationSourceSample,
        *,
        initial_matter_energy: Any | None = None,
        accumulated_source_work: Any | None = None,
    ) -> EnergyLedger:
        internal = self.context.electronic_model.energy(evaluation.field_free_dft)
        currents = self.currents(evaluation, source)
        xp = self.context.namespace
        zero = self.context.zero_scalar()
        electronic_scalar = -xp.dot(source.electric_field, currents.electronic_dipole)
        nuclear_scalar = -xp.dot(source.electric_field, currents.fixed_nuclear_dipole)
        scalar_total = electronic_scalar + nuclear_scalar
        matter = internal.energy_internal_total
        generator = matter + scalar_total
        rates = self._power_with_currents(evaluation, source, currents)
        initial = _baseline(initial_matter_energy, self.context)
        accumulated = _baseline(accumulated_source_work, self.context)
        return EnergyLedger(
            energy_kinetic_canonical=internal.energy_kinetic_canonical,
            energy_kinetic_vector_potential_linear=zero,
            energy_kinetic_diamagnetic=zero,
            energy_kinetic_mechanical=internal.energy_kinetic_canonical,
            energy_electron_nuclear=internal.energy_electron_nuclear,
            energy_hartree=internal.energy_hartree,
            energy_exchange_correlation=internal.energy_exchange_correlation,
            energy_nuclear_repulsion=internal.energy_nuclear_repulsion,
            energy_electromagnetic_electronic_scalar=electronic_scalar,
            energy_electromagnetic_fixed_nuclear_scalar=nuclear_scalar,
            energy_electromagnetic_scalar_total=scalar_total,
            energy_e1_coupling=zero,
            energy_matter_total=matter,
            energy_generator_total=generator,
            energy_absorbed=None if initial is None else matter - initial,
            source_work_accumulated=accumulated,
            energy_matter_rate_analytic=rates.energy_matter_rate_analytic,
            energy_generator_rate_analytic=rates.energy_generator_rate_analytic,
            source_work_rate=rates.source_work_rate,
            energy_ward_residual=rates.energy_ward_residual,
        )

    def _power_with_currents(
        self,
        evaluation: InstantaneousEvaluation,
        source: FormulationSourceSample,
        currents: CurrentLedger,
    ) -> PowerLedger:
        xp = self.context.namespace
        matter_rate = self.context.electronic_model.energy_rate(
            evaluation.field_free_dft, evaluation.field_free_density_dot
        )
        return PowerLedger(
            energy_matter_rate_analytic=matter_rate,
            energy_generator_rate_analytic=-xp.dot(
                source.electric_field_dot, currents.total_dipole
            ),
            source_work_rate=currents.source_work_rate,
            energy_ward_residual=matter_rate - currents.source_work_rate,
        )

    def power(
        self,
        evaluation: InstantaneousEvaluation,
        source: FormulationSourceSample,
    ) -> PowerLedger:
        return self._power_with_currents(evaluation, source, self.currents(evaluation, source))

    def apply_kick(
        self,
        coefficients: Any,
        impulse_au: tuple[float, float, float],
        evaluation_before: InstantaneousEvaluation,
        source_before: FormulationSourceSample,
        source_after: FormulationSourceSample,
    ) -> Any:
        return apply_electric_kick(
            kind=self.kind,
            gauge=self.gauge,
            context=self.context,
            coefficients=coefficients,
            impulse_au=impulse_au,
            evaluation_before=evaluation_before,
            source_before=source_before,
            source_after=source_after,
        )


@dataclass(frozen=True, slots=True)
class BareVelocityGauge:
    context: FormulationContext
    kind: FormulationKind = field(default=FormulationKind.BARE_VELOCITY_GAUGE, init=False)
    gauge: GaugeRepresentation = field(default=GaugeRepresentation.VELOCITY, init=False)

    @property
    def observable_dependencies(self) -> frozenset[str]:
        return frozenset({"density", "source", "eom", "density_dot", "field_free_dft"})

    def evaluate(
        self, density: AODensity, source: FormulationSourceSample
    ) -> InstantaneousEvaluation:
        self.context.validate_source(source, self.gauge)
        rho = self.context.require_density(density)
        xp = self.context.namespace
        dft = self.context.electronic_model.build(rho)
        overlap = self.context.workspace.require("operators.overlap")
        momentum = self.context.workspace.require("operators.canonical_momentum")
        linear = -(self.context.charge / self.context.mass) * xp.einsum(
            "x,xij->ij",
            source.vector_potential_reduced,
            momentum,
            optimize=True,
        )
        quadratic = (
            self.context.charge**2
            / (2.0 * self.context.mass)
            * xp.dot(source.vector_potential_reduced, source.vector_potential_reduced)
            * overlap
        )
        hamiltonian = hermitian_part(dft.hamiltonian + linear + quadratic)
        metric = self.context.backend.asarray(overlap, dtype=xp.complex128)
        zero = xp.zeros_like(metric, dtype=xp.complex128)
        triple = EOMTriple(metric, hamiltonian, zero)
        rho_dot = density_derivative(
            rho,
            metric,
            hamiltonian,
            zero,
            hbar=self.context.hbar,
            backend=self.context.backend,
        )
        return InstantaneousEvaluation(
            triple=triple,
            density=density,
            density_dot=rho_dot,
            field_free_density=rho,
            field_free_density_dot=rho_dot,
            field_free_dft=dft,
            hamiltonian_dynamic=hamiltonian,
            source_hamiltonian=hamiltonian,
            metric_dot=zero,
            d_site_metric=zero,
        )

    def currents(
        self, evaluation: InstantaneousEvaluation, source: FormulationSourceSample
    ) -> CurrentLedger:
        self.context.validate_source(source, self.gauge)
        xp = self.context.namespace
        rho = evaluation.density.matrix
        electronic = self.context.bare_electronic_dipole(rho)
        nuclear = self.context.fixed_nuclear_dipole()
        dipole_dot = self.context.charge * xp.real(
            xp.einsum(
                "ij,xji->x",
                evaluation.density_dot,
                self.context.relative_position_operators(),
                optimize=True,
            )
        )
        mechanical = uniform_mechanical_current(
            rho,
            self.context.workspace.require("operators.canonical_momentum"),
            evaluation.triple.metric,
            source.vector_potential_reduced,
            charge=self.context.charge,
            mass=self.context.mass,
            backend=self.context.backend,
        )
        power = xp.dot(source.electric_field, mechanical.total)
        return CurrentLedger(
            electronic_dipole=electronic,
            fixed_nuclear_dipole=nuclear,
            total_dipole=electronic + nuclear,
            dipole_derivative_analytic=dipole_dot,
            primary_current=mechanical.total,
            source_current=mechanical.total,
            source_work_rate=power,
            mechanical_paramagnetic_current=mechanical.paramagnetic,
            mechanical_diamagnetic_current=mechanical.diamagnetic,
            mechanical_total_current=mechanical.total,
            source_current_dipole_derivative_residual=mechanical.total - dipole_dot,
        )

    def energy(
        self,
        evaluation: InstantaneousEvaluation,
        source: FormulationSourceSample,
        *,
        initial_matter_energy: Any | None = None,
        accumulated_source_work: Any | None = None,
    ) -> EnergyLedger:
        internal = self.context.electronic_model.energy(evaluation.field_free_dft)
        currents = self.currents(evaluation, source)
        xp = self.context.namespace
        rho = evaluation.density.matrix
        momentum = self.context.workspace.require("operators.canonical_momentum")
        momentum_expectation = xp.real(xp.einsum("ij,xji->x", rho, momentum, optimize=True))
        electron_count = self.context.electron_count(rho, evaluation.triple.metric)
        vector = source.vector_potential_reduced
        linear = -(self.context.charge / self.context.mass) * xp.dot(vector, momentum_expectation)
        diamagnetic = (
            self.context.charge**2
            / (2.0 * self.context.mass)
            * electron_count
            * xp.dot(vector, vector)
        )
        mechanical = internal.energy_kinetic_canonical + linear + diamagnetic
        matter = internal.energy_internal_total + linear + diamagnetic
        zero = self.context.zero_scalar()
        rates = self._power_with_currents(evaluation, source, currents)
        initial = _baseline(initial_matter_energy, self.context)
        accumulated = _baseline(accumulated_source_work, self.context)
        return EnergyLedger(
            energy_kinetic_canonical=internal.energy_kinetic_canonical,
            energy_kinetic_vector_potential_linear=linear,
            energy_kinetic_diamagnetic=diamagnetic,
            energy_kinetic_mechanical=mechanical,
            energy_electron_nuclear=internal.energy_electron_nuclear,
            energy_hartree=internal.energy_hartree,
            energy_exchange_correlation=internal.energy_exchange_correlation,
            energy_nuclear_repulsion=internal.energy_nuclear_repulsion,
            energy_electromagnetic_electronic_scalar=zero,
            energy_electromagnetic_fixed_nuclear_scalar=zero,
            energy_electromagnetic_scalar_total=zero,
            energy_e1_coupling=zero,
            energy_matter_total=matter,
            energy_generator_total=matter,
            energy_absorbed=None if initial is None else matter - initial,
            source_work_accumulated=accumulated,
            energy_matter_rate_analytic=rates.energy_matter_rate_analytic,
            energy_generator_rate_analytic=rates.energy_generator_rate_analytic,
            source_work_rate=rates.source_work_rate,
            energy_ward_residual=rates.energy_ward_residual,
        )

    def _power_with_currents(
        self,
        evaluation: InstantaneousEvaluation,
        source: FormulationSourceSample,
        currents: CurrentLedger,
    ) -> PowerLedger:
        xp = self.context.namespace
        rho = evaluation.density.matrix
        momentum = self.context.workspace.require("operators.canonical_momentum")
        momentum_expectation = xp.real(xp.einsum("ij,xji->x", rho, momentum, optimize=True))
        electron_count = self.context.electron_count(rho, evaluation.triple.metric)
        vector = source.vector_potential_reduced
        # Functional derivative with respect to P is exactly H_VG.  The final
        # two terms are the explicit source-time derivative at fixed P.
        matter_rate = (
            expectation(evaluation.density_dot, evaluation.hamiltonian_dynamic, xp)
            - (self.context.charge / self.context.mass)
            * xp.dot(source.vector_potential_reduced_dot, momentum_expectation)
            + (self.context.charge**2 / self.context.mass)
            * electron_count
            * xp.dot(vector, source.vector_potential_reduced_dot)
        )
        return PowerLedger(
            energy_matter_rate_analytic=matter_rate,
            energy_generator_rate_analytic=matter_rate,
            source_work_rate=currents.source_work_rate,
            energy_ward_residual=matter_rate - currents.source_work_rate,
        )

    def power(
        self,
        evaluation: InstantaneousEvaluation,
        source: FormulationSourceSample,
    ) -> PowerLedger:
        return self._power_with_currents(evaluation, source, self.currents(evaluation, source))

    def apply_kick(
        self,
        coefficients: Any,
        impulse_au: tuple[float, float, float],
        evaluation_before: InstantaneousEvaluation,
        source_before: FormulationSourceSample,
        source_after: FormulationSourceSample,
    ) -> Any:
        return apply_electric_kick(
            kind=self.kind,
            gauge=self.gauge,
            context=self.context,
            coefficients=coefficients,
            impulse_au=impulse_au,
            evaluation_before=evaluation_before,
            source_before=source_before,
            source_after=source_after,
        )
