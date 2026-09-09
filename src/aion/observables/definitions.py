"""Persistent names and mathematical definitions for WP3 observables."""

from __future__ import annotations

from aion.config import AtomicUnit, FormulationKind, PhysicalDimension
from aion.observables.records import ObservableDefinition, SamplingLocation


def _definition(
    field_name: str,
    formulation: FormulationKind,
    dimension: PhysicalDimension,
    unit: AtomicUnit,
    shape: tuple[int, ...],
    equation: str,
) -> ObservableDefinition:
    return ObservableDefinition(
        definition_id=f"{field_name}.{formulation.value}",
        originating_formulation=formulation.value,
        physical_dimension=dimension,
        unit=unit,
        shape=shape,
        sampling_location=SamplingLocation.ENDPOINT,
        mathematical_definition=equation,
    )


def observable_definition_catalog(
    formulation: FormulationKind,
    *,
    natom: int,
    npair: int,
) -> dict[str, ObservableDefinition]:
    """Return the exact storage identity for every active WP3 ledger field."""

    vector = (3,)
    scalar = (1,)
    definitions = {
        "electronic_dipole": _definition(
            "dipole.electronic",
            formulation,
            PhysicalDimension.DIPOLE,
            AtomicUnit.DIPOLE,
            vector,
            "formulation-owned electronic dipole about the configured EM origin",
        ),
        "fixed_nuclear_dipole": _definition(
            "dipole.fixed_nuclear",
            formulation,
            PhysicalDimension.DIPOLE,
            AtomicUnit.DIPOLE,
            vector,
            "sum_a Z_a (R_a-O)",
        ),
        "total_dipole": _definition(
            "dipole.combined",
            formulation,
            PhysicalDimension.DIPOLE,
            AtomicUnit.DIPOLE,
            vector,
            "electronic_dipole + fixed_nuclear_dipole",
        ),
        "dipole_derivative_analytic": _definition(
            "current.dipole_derivative_analytic",
            formulation,
            PhysicalDimension.CURRENT,
            AtomicUnit.CURRENT,
            vector,
            "analytic d(electronic_dipole)/dt from the formulation EOM",
        ),
        "source_current": _definition(
            "current.variational_source",
            formulation,
            PhysicalDimension.CURRENT,
            AtomicUnit.CURRENT,
            vector,
            "current conjugate to the formulation source",
        ),
        "mechanical_paramagnetic_current": _definition(
            "current.mechanical_paramagnetic",
            formulation,
            PhysicalDimension.CURRENT,
            AtomicUnit.CURRENT,
            vector,
            "(q/m) Re Tr[P p_can]",
        ),
        "mechanical_diamagnetic_current": _definition(
            "current.mechanical_diamagnetic",
            formulation,
            PhysicalDimension.CURRENT,
            AtomicUnit.CURRENT,
            vector,
            "-(q^2/m) Re Tr[P Aop]",
        ),
        "mechanical_total_current": _definition(
            "current.mechanical_total",
            formulation,
            PhysicalDimension.CURRENT,
            AtomicUnit.CURRENT,
            vector,
            "mechanical_paramagnetic_current + mechanical_diamagnetic_current",
        ),
        "source_work_rate": _definition(
            "power.source_work_rate",
            formulation,
            PhysicalDimension.POWER,
            AtomicUnit.POWER,
            scalar,
            "formulation-owned instantaneous source work rate",
        ),
    }
    primary_equation = {
        FormulationKind.BARE_LENGTH_GAUGE: "analytic d(electronic_dipole)/dt",
        FormulationKind.BARE_VELOCITY_GAUGE: "mechanical_total_current",
        FormulationKind.P0: "P0 variational graph source current",
        FormulationKind.P0_E1: "P0 graph + E1 residual variational source current",
    }[formulation]
    definitions["primary_current"] = _definition(
        "current.primary",
        formulation,
        PhysicalDimension.CURRENT,
        AtomicUnit.CURRENT,
        vector,
        primary_equation,
    )
    for field_name in (
        "energy_kinetic_canonical",
        "energy_kinetic_vector_potential_linear",
        "energy_kinetic_diamagnetic",
        "energy_kinetic_mechanical",
        "energy_electron_nuclear",
        "energy_hartree",
        "energy_exchange_correlation",
        "energy_nuclear_repulsion",
        "energy_electromagnetic_electronic_scalar",
        "energy_electromagnetic_fixed_nuclear_scalar",
        "energy_electromagnetic_scalar_total",
        "energy_e1_coupling",
        "energy_matter_total",
        "energy_generator_total",
        "energy_absorbed",
        "source_work_accumulated",
    ):
        definitions[field_name] = _definition(
            field_name.replace("_", "."),
            formulation,
            PhysicalDimension.ENERGY,
            AtomicUnit.ENERGY,
            scalar,
            field_name,
        )
    for field_name in (
        "energy_matter_rate_analytic",
        "energy_generator_rate_analytic",
        "energy_ward_residual",
    ):
        definitions[field_name] = _definition(
            field_name.replace("_", "."),
            formulation,
            PhysicalDimension.POWER,
            AtomicUnit.POWER,
            scalar,
            field_name,
        )
    if formulation in {FormulationKind.P0, FormulationKind.P0_E1}:
        definitions.update(
            {
                "ambient_projected_mechanical_current": _definition(
                    "current.ambient_projected_mechanical",
                    formulation,
                    PhysicalDimension.CURRENT,
                    AtomicUnit.CURRENT,
                    vector,
                    "ambient-projected mechanical diagnostic; not source current",
                ),
                "site_charges": _definition(
                    "charge.site",
                    formulation,
                    PhysicalDimension.CHARGE,
                    AtomicUnit.CHARGE,
                    (natom,),
                    "(q/2) Re Tr[P {M_a,S}]",
                ),
                "site_charge_derivatives": _definition(
                    "charge_flow.site_derivative",
                    formulation,
                    PhysicalDimension.CHARGE_FLOW_RATE,
                    AtomicUnit.CHARGE_FLOW_RATE,
                    (natom,),
                    "analytic d(site_charges)/dt",
                ),
                "pair_currents_continuity": _definition(
                    "charge_flow.pair_continuity",
                    formulation,
                    PhysicalDimension.CHARGE_FLOW_RATE,
                    AtomicUnit.CHARGE_FLOW_RATE,
                    (npair,),
                    "full-dynamical pair current satisfying site continuity",
                ),
                "pair_currents_p0_source": _definition(
                    "charge_flow.pair_p0_source",
                    formulation,
                    PhysicalDimension.CHARGE_FLOW_RATE,
                    AtomicUnit.CHARGE_FLOW_RATE,
                    (npair,),
                    "action-split P0 source pair current",
                ),
                "p0_graph_current": _definition(
                    "current.p0_graph_source",
                    formulation,
                    PhysicalDimension.CURRENT,
                    AtomicUnit.CURRENT,
                    vector,
                    "sum_(a<b) I_ab^(P0 source) (R_b-R_a)",
                ),
            }
        )
    if formulation is FormulationKind.P0_E1:
        for field_name, equation in (
            (
                "e1_intrinsic_polarization_current",
                "d Re Tr[P D_beta]/dt",
            ),
            (
                "e1_phase_response_current",
                "sum_alpha E_alpha Re Tr[P D_(alpha;beta)]",
            ),
            (
                "e1_residual_source_current",
                "intrinsic-polarization current + phase-response current",
            ),
        ):
            definitions[field_name] = _definition(
                field_name.replace("_", "."),
                formulation,
                PhysicalDimension.CURRENT,
                AtomicUnit.CURRENT,
                vector,
                equation,
            )
    return definitions
