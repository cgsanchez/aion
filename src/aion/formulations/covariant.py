"""Wilson-covariant P0 and composable P0+E1 formulations."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any

from aion.config import FormulationKind, GaugeRepresentation
from aion.electronic_structure import expectation, hermitian_part
from aion.formulations.bare import _baseline
from aion.formulations.base import FormulationContext
from aion.formulations.events import apply_electric_kick
from aion.formulations.kernels import (
    covariant_ambient_mechanical_current,
    density_derivative,
    e1_tensors,
    inverse_dressed_density,
    inverse_dressed_density_dot,
    p0_continuity_residual,
    p0_geometry,
    p0_graph_current,
    p0_pair_currents,
    p0_source_power,
    site_charge_derivatives,
    site_charges,
)
from aion.formulations.types import (
    AODensity,
    CurrentLedger,
    EnergyLedger,
    EOMTriple,
    FormulationSourceSample,
    InstantaneousEvaluation,
    PowerLedger,
    resolve_velocity_fraction,
)


def _evaluate(
    context: FormulationContext,
    gauge: GaugeRepresentation,
    gauge_velocity_fraction: float,
    density: AODensity,
    source: FormulationSourceSample,
    *,
    include_e1: bool,
) -> InstantaneousEvaluation:
    context.validate_source(source, gauge, gauge_velocity_fraction)
    rho = context.require_density(density)
    geometry = p0_geometry(
        context.workspace.require("operators.overlap"),
        context.workspace.require("anchors.ao_to_atom"),
        context.pairs,
        source.node_scalar_potential,
        source.pair_link,
        source.pair_link_dot,
        natom=context.natom,
        charge=context.charge,
        hbar=context.hbar,
        backend=context.backend,
    )
    rho0 = inverse_dressed_density(rho, geometry.theta)
    dft = context.electronic_model.build(rho0)
    hamiltonian_p0 = hermitian_part(geometry.theta * dft.hamiltonian)
    e1 = None
    connection = geometry.connection
    dynamic = hamiltonian_p0
    if include_e1:
        e1 = e1_tensors(
            context.central_dipoles(),
            geometry.theta,
            geometry.theta_dot,
            context.workspace.require("anchors.ao_to_atom"),
            context.workspace.require("nuclei.coordinates_au"),
            source.electric_field,
            charge=context.charge,
            hbar=context.hbar,
            backend=context.backend,
        )
        # V_E1 appears exactly once: geometrically as (i/hbar)V_E1 in omega.
        connection = connection + e1.connection
        dynamic = hamiltonian_p0 + e1.potential
    triple = EOMTriple(geometry.metric, hamiltonian_p0, connection)
    rho_dot = density_derivative(
        rho,
        triple.metric,
        triple.hamiltonian_eom,
        triple.connection,
        hbar=context.hbar,
        backend=context.backend,
    )
    rho0_dot = inverse_dressed_density_dot(rho, rho_dot, geometry.theta, geometry.theta_dot)
    return InstantaneousEvaluation(
        triple=triple,
        density=density,
        density_dot=rho_dot,
        field_free_density=rho0,
        field_free_density_dot=rho0_dot,
        field_free_dft=dft,
        hamiltonian_dynamic=hermitian_part(dynamic),
        source_hamiltonian=hamiltonian_p0,
        metric_dot=geometry.metric_dot,
        d_site_metric=geometry.d_site_metric,
        theta=geometry.theta,
        theta_dot=geometry.theta_dot,
        e1_potential=None if e1 is None else e1.potential,
        dressed_central_dipoles=None if e1 is None else e1.dressed_dipoles,
        dressed_central_dipole_dots=(None if e1 is None else e1.dressed_dipole_dots),
        dressed_central_dipole_vector_derivatives=(None if e1 is None else e1.vector_derivatives),
    )


def _currents(
    context: FormulationContext,
    gauge: GaugeRepresentation,
    gauge_velocity_fraction: float,
    evaluation: InstantaneousEvaluation,
    source: FormulationSourceSample,
    *,
    include_e1: bool,
) -> CurrentLedger:
    context.validate_source(source, gauge, gauge_velocity_fraction)
    xp = context.namespace
    rho = evaluation.density.matrix
    assert evaluation.theta is not None
    charges = site_charges(
        rho,
        evaluation.triple.metric,
        context.workspace.require("anchors.site_projector_diagonals"),
        charge=context.charge,
        backend=context.backend,
    )
    charge_dots = site_charge_derivatives(
        rho,
        evaluation.density_dot,
        evaluation.triple.metric,
        evaluation.metric_dot,
        context.workspace.require("anchors.site_projector_diagonals"),
        charge=context.charge,
        backend=context.backend,
    )
    source_pairs = p0_pair_currents(
        rho,
        evaluation.triple.metric,
        evaluation.d_site_metric,
        evaluation.hamiltonian_dynamic,
        evaluation.source_hamiltonian,
        context.workspace.require("anchors.ao_to_atom"),
        context.pairs,
        charge=context.charge,
        hbar=context.hbar,
        backend=context.backend,
    )
    continuity_pairs = (
        source_pairs
        if not include_e1
        else p0_pair_currents(
            rho,
            evaluation.triple.metric,
            evaluation.d_site_metric,
            evaluation.hamiltonian_dynamic,
            evaluation.hamiltonian_dynamic,
            context.workspace.require("anchors.ao_to_atom"),
            context.pairs,
            charge=context.charge,
            hbar=context.hbar,
            backend=context.backend,
        )
    )
    graph = p0_graph_current(
        source_pairs, context.workspace.require("anchors.pair_displacements_au")
    )
    coordinates = context.workspace.require("nuclei.coordinates_au")
    origin = xp.asarray(
        context.reference.config.molecule.electromagnetic_origin.position_au,
        dtype=xp.float64,
    )
    site_dipole = xp.einsum("a,ax->x", charges, coordinates - origin[None, :], optimize=True)
    site_dipole_dot = xp.einsum(
        "a,ax->x", charge_dots, coordinates - origin[None, :], optimize=True
    )
    intrinsic_dipole = context.zero_vector()
    intrinsic_current = context.zero_vector()
    phase_current = context.zero_vector()
    residual_current = context.zero_vector()
    if include_e1:
        dressed = evaluation.dressed_central_dipoles
        dressed_dot = evaluation.dressed_central_dipole_dots
        derivatives = evaluation.dressed_central_dipole_vector_derivatives
        assert dressed is not None and dressed_dot is not None and derivatives is not None
        intrinsic_dipole = xp.real(xp.einsum("ij,xji->x", rho, dressed, optimize=True))
        intrinsic_current = xp.real(
            xp.einsum("ij,xji->x", evaluation.density_dot, dressed, optimize=True)
            + xp.einsum("ij,xji->x", rho, dressed_dot, optimize=True)
        )
        phase_current = xp.real(
            xp.einsum(
                "a,abij,ji->b",
                source.electric_field,
                derivatives,
                rho,
                optimize=True,
            )
        )
        residual_current = intrinsic_current + phase_current
    electronic = site_dipole + intrinsic_dipole
    dipole_dot = site_dipole_dot + intrinsic_current
    source_current = graph + residual_current
    nuclear = context.fixed_nuclear_dipole()
    mechanical = covariant_ambient_mechanical_current(
        rho,
        evaluation.theta,
        context.workspace.require("operators.overlap"),
        context.workspace.require("operators.canonical_momentum"),
        source.vector_potential_reduced,
        charge=context.charge,
        mass=context.mass,
        backend=context.backend,
    )
    source_power = (
        xp.dot(source.electric_field, source_current)
        if include_e1
        else p0_source_power(source_pairs, source.pair_electromotive_potential, xp)
    )
    return CurrentLedger(
        electronic_dipole=electronic,
        fixed_nuclear_dipole=nuclear,
        total_dipole=electronic + nuclear,
        dipole_derivative_analytic=dipole_dot,
        primary_current=source_current,
        source_current=source_current,
        source_work_rate=source_power,
        mechanical_paramagnetic_current=mechanical.paramagnetic,
        mechanical_diamagnetic_current=mechanical.diamagnetic,
        mechanical_total_current=mechanical.total,
        ambient_projected_mechanical_current=mechanical.total,
        site_charges=charges,
        site_charge_derivatives=charge_dots,
        pair_currents_continuity=continuity_pairs,
        pair_currents_p0_source=source_pairs,
        p0_graph_current=graph,
        e1_intrinsic_polarization_current=(intrinsic_current if include_e1 else None),
        e1_phase_response_current=phase_current if include_e1 else None,
        e1_residual_source_current=residual_current if include_e1 else None,
        continuity_residual=p0_continuity_residual(
            charge_dots,
            context.workspace.require("anchors.incidence"),
            continuity_pairs,
        ),
        source_current_dipole_derivative_residual=source_current - dipole_dot,
    )


def _energy(
    context: FormulationContext,
    gauge: GaugeRepresentation,
    gauge_velocity_fraction: float,
    evaluation: InstantaneousEvaluation,
    source: FormulationSourceSample,
    *,
    include_e1: bool,
    initial_matter_energy: Any | None,
    accumulated_source_work: Any | None,
) -> EnergyLedger:
    currents = _currents(
        context,
        gauge,
        gauge_velocity_fraction,
        evaluation,
        source,
        include_e1=include_e1,
    )
    internal = context.electronic_model.energy(evaluation.field_free_dft)
    xp = context.namespace
    zero = context.zero_scalar()
    assert currents.site_charges is not None
    electronic_scalar = xp.dot(source.node_scalar_potential, currents.site_charges)
    nuclear_scalar = xp.dot(
        source.node_scalar_potential, context.workspace.require("nuclei.charges")
    )
    scalar_total = electronic_scalar + nuclear_scalar
    e1_coupling = (
        zero
        if evaluation.e1_potential is None
        else expectation(evaluation.density.matrix, evaluation.e1_potential, xp)
    )
    matter = internal.energy_internal_total
    generator = matter + scalar_total + e1_coupling
    rates = _power_from_currents(context, evaluation, currents)
    initial = _baseline(initial_matter_energy, context)
    accumulated = _baseline(accumulated_source_work, context)
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
        energy_e1_coupling=e1_coupling,
        energy_matter_total=matter,
        energy_generator_total=generator,
        energy_absorbed=None if initial is None else matter - initial,
        source_work_accumulated=accumulated,
        energy_matter_rate_analytic=rates.energy_matter_rate_analytic,
        energy_generator_rate_analytic=rates.energy_generator_rate_analytic,
        source_work_rate=rates.source_work_rate,
        energy_ward_residual=rates.energy_ward_residual,
    )


def _power_from_currents(
    context: FormulationContext,
    evaluation: InstantaneousEvaluation,
    currents: CurrentLedger,
) -> PowerLedger:
    matter_rate = context.electronic_model.energy_rate(
        evaluation.field_free_dft, evaluation.field_free_density_dot
    )
    return PowerLedger(
        energy_matter_rate_analytic=matter_rate,
        energy_generator_rate_analytic=None,
        source_work_rate=currents.source_work_rate,
        energy_ward_residual=matter_rate - currents.source_work_rate,
    )


def _power(
    context: FormulationContext,
    gauge: GaugeRepresentation,
    gauge_velocity_fraction: float,
    evaluation: InstantaneousEvaluation,
    source: FormulationSourceSample,
    *,
    include_e1: bool,
) -> PowerLedger:
    currents = _currents(
        context,
        gauge,
        gauge_velocity_fraction,
        evaluation,
        source,
        include_e1=include_e1,
    )
    return _power_from_currents(context, evaluation, currents)


@dataclass(frozen=True, slots=True)
class P0:
    context: FormulationContext
    gauge: GaugeRepresentation = GaugeRepresentation.LENGTH
    velocity_fraction: InitVar[float | None] = None
    gauge_velocity_fraction: float = field(init=False)
    kind: FormulationKind = field(default=FormulationKind.P0, init=False)

    def __post_init__(self, velocity_fraction: float | None) -> None:
        object.__setattr__(
            self,
            "gauge_velocity_fraction",
            resolve_velocity_fraction(self.gauge, velocity_fraction),
        )

    @property
    def observable_dependencies(self) -> frozenset[str]:
        return frozenset({"density", "source.node_link", "eom", "density_dot", "field_free_dft"})

    def evaluate(
        self, density: AODensity, source: FormulationSourceSample
    ) -> InstantaneousEvaluation:
        return _evaluate(
            self.context,
            self.gauge,
            self.gauge_velocity_fraction,
            density,
            source,
            include_e1=False,
        )

    def currents(
        self, evaluation: InstantaneousEvaluation, source: FormulationSourceSample
    ) -> CurrentLedger:
        return _currents(
            self.context,
            self.gauge,
            self.gauge_velocity_fraction,
            evaluation,
            source,
            include_e1=False,
        )

    def energy(
        self,
        evaluation: InstantaneousEvaluation,
        source: FormulationSourceSample,
        *,
        initial_matter_energy: Any | None = None,
        accumulated_source_work: Any | None = None,
    ) -> EnergyLedger:
        return _energy(
            self.context,
            self.gauge,
            self.gauge_velocity_fraction,
            evaluation,
            source,
            include_e1=False,
            initial_matter_energy=initial_matter_energy,
            accumulated_source_work=accumulated_source_work,
        )

    def power(
        self,
        evaluation: InstantaneousEvaluation,
        source: FormulationSourceSample,
    ) -> PowerLedger:
        return _power(
            self.context,
            self.gauge,
            self.gauge_velocity_fraction,
            evaluation,
            source,
            include_e1=False,
        )

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
            gauge_velocity_fraction=self.gauge_velocity_fraction,
            context=self.context,
            coefficients=coefficients,
            impulse_au=impulse_au,
            evaluation_before=evaluation_before,
            source_before=source_before,
            source_after=source_after,
        )


@dataclass(frozen=True, slots=True)
class P0E1:
    context: FormulationContext
    gauge: GaugeRepresentation = GaugeRepresentation.LENGTH
    velocity_fraction: InitVar[float | None] = None
    gauge_velocity_fraction: float = field(init=False)
    kind: FormulationKind = field(default=FormulationKind.P0_E1, init=False)

    def __post_init__(self, velocity_fraction: float | None) -> None:
        object.__setattr__(
            self,
            "gauge_velocity_fraction",
            resolve_velocity_fraction(self.gauge, velocity_fraction),
        )

    @property
    def observable_dependencies(self) -> frozenset[str]:
        return frozenset(
            {
                "density",
                "source.node_link",
                "source.electric_field",
                "eom",
                "density_dot",
                "field_free_dft",
                "e1.central_dipoles",
            }
        )

    def evaluate(
        self, density: AODensity, source: FormulationSourceSample
    ) -> InstantaneousEvaluation:
        return _evaluate(
            self.context,
            self.gauge,
            self.gauge_velocity_fraction,
            density,
            source,
            include_e1=True,
        )

    def currents(
        self, evaluation: InstantaneousEvaluation, source: FormulationSourceSample
    ) -> CurrentLedger:
        return _currents(
            self.context,
            self.gauge,
            self.gauge_velocity_fraction,
            evaluation,
            source,
            include_e1=True,
        )

    def energy(
        self,
        evaluation: InstantaneousEvaluation,
        source: FormulationSourceSample,
        *,
        initial_matter_energy: Any | None = None,
        accumulated_source_work: Any | None = None,
    ) -> EnergyLedger:
        return _energy(
            self.context,
            self.gauge,
            self.gauge_velocity_fraction,
            evaluation,
            source,
            include_e1=True,
            initial_matter_energy=initial_matter_energy,
            accumulated_source_work=accumulated_source_work,
        )

    def power(
        self,
        evaluation: InstantaneousEvaluation,
        source: FormulationSourceSample,
    ) -> PowerLedger:
        return _power(
            self.context,
            self.gauge,
            self.gauge_velocity_fraction,
            evaluation,
            source,
            include_e1=True,
        )

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
            gauge_velocity_fraction=self.gauge_velocity_fraction,
            context=self.context,
            coefficients=coefficients,
            impulse_au=impulse_au,
            evaluation_before=evaluation_before,
            source_before=source_before,
            source_after=source_after,
        )
