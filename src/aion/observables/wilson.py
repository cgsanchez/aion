"""Formulation-owned observations for the exact straight-Wilson action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aion.electromagnetism import AffineVectorFieldVariation
from aion.electronic_structure import (
    ExactWilsonDynamicEvaluation,
    ExactWilsonPowerObservation,
    NonlinearWeakCurrentPairing,
    evaluate_exact_wilson_charge,
    evaluate_exact_wilson_power,
    evaluate_nonlinear_weak_current_pairing,
)


@dataclass(frozen=True, slots=True)
class WilsonEnergyObservation:
    kinetic_au: Any
    electron_nuclear_au: Any
    one_electron_au: Any
    hartree_au: Any
    exchange_correlation_au: Any
    nuclear_repulsion_au: Any
    electronic_au: Any
    molecular_total_au: Any


@dataclass(frozen=True, slots=True)
class ExactWilsonEndpointObservation:
    """One endpoint observation with action-native definitions."""

    time_au: float
    electronic_dipole_au: Any
    fixed_nuclear_dipole_au: Any
    molecular_total_dipole_au: Any
    uniform_source_current_au: Any
    uniform_source_current_pairings: tuple[NonlinearWeakCurrentPairing, ...]
    metric_particle_number: Any
    integrated_charge_grid: Any
    integrated_charge_metric: Any
    charge_grid_metric_residual: Any
    source_power_au: Any
    matrix_mechanical_energy_rate_au: Any
    power_identity_residual_au: Any
    power: ExactWilsonPowerObservation
    energy: WilsonEnergyObservation | None


def evaluate_exact_wilson_endpoint_observation(
    evaluation: ExactWilsonDynamicEvaluation,
    coefficient_density: object,
    *,
    include_energy: bool,
) -> ExactWilsonEndpointObservation:
    """Evaluate dipole, source current, power, charge, and scheduled energy."""

    if not isinstance(evaluation, ExactWilsonDynamicEvaluation):
        raise TypeError("evaluation must be an ExactWilsonDynamicEvaluation")
    sample = evaluation.sample
    model = sample.model
    backend = model.backend
    xp = model.namespace
    density = model.hartree_evaluator._validated_density(coefficient_density)
    power = evaluate_exact_wilson_power(evaluation, density)
    charge = evaluate_exact_wilson_charge(model, density)

    unit_pairings: list[NonlinearWeakCurrentPairing] = []
    components: list[Any] = []
    cartesian_offsets = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    for offset in cartesian_offsets:
        variation = AffineVectorFieldVariation(
            offset_au=offset,
            origin_au=sample.source.origin_au,
        )
        pairing = evaluate_nonlinear_weak_current_pairing(
            model,
            sample.one_electron,
            density,
            power.velocity_density,
            variation,
        )
        unit_pairings.append(pairing)
        components.append(xp.real(pairing.on_shell_pairing))
    current = xp.stack(components)

    grid = model.quadrature.grid
    coordinates = backend.asarray(grid.coordinates_au, dtype=xp.float64)
    weights = backend.asarray(grid.weights_au, dtype=xp.float64)
    origin = backend.asarray(sample.source.origin_au, dtype=xp.float64)
    relative = coordinates - origin
    electronic_dipole = xp.einsum(
        "p,px,p->x",
        weights,
        relative,
        charge.signed_charge_density,
        optimize=True,
    )
    nuclei = model.quadrature.reference.core_operators.nuclei
    nuclear_coordinates = backend.asarray(nuclei.coordinates_au, dtype=xp.float64)
    nuclear_charges = backend.asarray(nuclei.charges, dtype=xp.float64)
    fixed_nuclear_dipole = xp.einsum(
        "ax,a->x",
        nuclear_coordinates - origin,
        nuclear_charges,
        optimize=True,
    )
    action = evaluation.action
    energy = (
        WilsonEnergyObservation(
            kinetic_au=action.energy_kinetic_au,
            electron_nuclear_au=action.energy_electron_nuclear_au,
            one_electron_au=action.energy_one_electron_au,
            hartree_au=action.energy_hartree_au,
            exchange_correlation_au=action.energy_exchange_correlation_au,
            nuclear_repulsion_au=action.energy_nuclear_repulsion_au,
            electronic_au=action.energy_electronic_au,
            molecular_total_au=action.energy_molecular_total_au,
        )
        if include_energy
        else None
    )
    return ExactWilsonEndpointObservation(
        time_au=sample.source.time_au,
        electronic_dipole_au=electronic_dipole,
        fixed_nuclear_dipole_au=fixed_nuclear_dipole,
        molecular_total_dipole_au=electronic_dipole + fixed_nuclear_dipole,
        uniform_source_current_au=current,
        uniform_source_current_pairings=tuple(unit_pairings),
        metric_particle_number=charge.metric_particle_number,
        integrated_charge_grid=charge.integrated_charge_grid,
        integrated_charge_metric=charge.integrated_charge_metric,
        charge_grid_metric_residual=(
            charge.integrated_charge_grid - charge.integrated_charge_metric
        ),
        source_power_au=power.source_power_au,
        matrix_mechanical_energy_rate_au=power.matrix_mechanical_energy_rate_au,
        power_identity_residual_au=power.power_identity_residual_au,
        power=power,
        energy=energy,
    )
