from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from aion.config import (
    BackendConfig,
    FormulationConfig,
    FormulationKind,
    GaugeRepresentation,
    IntegratorKind,
    PropagationConfig,
    ReferenceLinkConfig,
    SimulationConfig,
    Sin2VectorPotentialPulseConfig,
)
from aion.electromagnetism import Sin2VectorPotentialPulse, pulse_aligned_time_grid
from aion.errors import MidpointConvergenceError
from aion.formulations import SourceSampling
from aion.propagation import ConnectionAwareTransport
from aion.workflows import build_simulation
from test_reference_integration import molecular_config
from test_wp3_formulation_integration import _simulation_config

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def propagation_references() -> dict[str, object]:
    from aion.electronic_structure import prepare_pyscf_reference

    return {name: prepare_pyscf_reference(molecular_config(name)) for name in ("h2", "lih")}


def _endpoint(simulation: object, state: object) -> tuple[object, object, object]:
    source = simulation.source_sample(SourceSampling.ENDPOINT, state.step_index)
    evaluation = simulation.formulation.evaluate(state.density(), source)
    current = simulation.formulation.currents(evaluation, source)
    energy = simulation.formulation.energy(evaluation, source)
    return evaluation, current, energy


@pytest.mark.parametrize(
    ("kind", "gauge"),
    (
        (FormulationKind.BARE_LENGTH_GAUGE, None),
        (FormulationKind.BARE_VELOCITY_GAUGE, None),
        (FormulationKind.P0, GaugeRepresentation.LENGTH),
        (FormulationKind.P0_E1, GaugeRepresentation.VELOCITY),
    ),
)
def test_all_formulations_use_one_scem_engine_and_preserve_their_metric(
    propagation_references: dict[str, object],
    kind: FormulationKind,
    gauge: GaugeRepresentation | None,
) -> None:
    reference = propagation_references["lih"]
    simulation = build_simulation(_simulation_config(reference, kind, gauge=gauge), reference)
    scratch_ids = {name: id(value) for name, value in simulation.workspace.scratch.items()}
    result = simulation.step()
    assert result.state.step_index == 1
    assert simulation.state is result.state
    assert result.diagnostics.converged
    assert result.diagnostics.density_residual <= simulation.config.propagation.density_tolerance
    assert np.isfinite(result.diagnostics.hamiltonian_residual)
    assert result.diagnostics.initial_metric_residual < 2.0e-12
    assert result.diagnostics.final_metric_residual < 2.0e-12
    assert result.diagnostics.maximum_hermitian_cleanup_norm < 2.0e-14
    if kind in {FormulationKind.P0, FormulationKind.P0_E1}:
        assert result.diagnostics.predictor_link is not None
        assert result.diagnostics.half_step_link.correction_applied
        assert result.diagnostics.full_step_link.correction_applied
        assert result.diagnostics.full_step_link.corrected_metric_residual < 2.0e-12
    else:
        assert result.diagnostics.predictor_link is None
        assert not result.diagnostics.half_step_link.correction_applied
        assert result.diagnostics.maximum_correction_norm == 0.0

    saved_midpoint = np.array(result.midpoint_evaluation.density.matrix, copy=True)
    simulation.step()
    assert np.array_equal(result.midpoint_evaluation.density.matrix, saved_midpoint)
    assert {name: id(value) for name, value in simulation.workspace.scratch.items()} == scratch_ids


@pytest.mark.parametrize(
    ("name", "kind"),
    (("h2", FormulationKind.P0), ("lih", FormulationKind.P0_E1)),
)
def test_connection_aware_trajectories_are_covariant_between_length_and_velocity(
    propagation_references: dict[str, object],
    name: str,
    kind: FormulationKind,
) -> None:
    reference = propagation_references[name]
    simulations = {
        gauge: build_simulation(_simulation_config(reference, kind, gauge=gauge), reference)
        for gauge in (GaugeRepresentation.LENGTH, GaugeRepresentation.VELOCITY)
    }
    for _ in range(4):
        results = {gauge: simulation.step() for gauge, simulation in simulations.items()}
        left = results[GaugeRepresentation.LENGTH]
        right = results[GaugeRepresentation.VELOCITY]
        eval_l, current_l, energy_l = _endpoint(simulations[GaugeRepresentation.LENGTH], left.state)
        eval_v, current_v, energy_v = _endpoint(
            simulations[GaugeRepresentation.VELOCITY], right.state
        )
        assert np.linalg.norm(eval_l.field_free_density - eval_v.field_free_density) < 8.0e-12
        assert np.linalg.norm(current_l.electronic_dipole - current_v.electronic_dipole) < 8.0e-11
        assert np.linalg.norm(current_l.source_current - current_v.source_current) < 8.0e-10
        assert float(energy_l.energy_matter_total) == pytest.approx(
            float(energy_v.energy_matter_total), abs=8.0e-11
        )
        assert left.diagnostics.iterations == right.diagnostics.iterations
        assert left.diagnostics.density_residual == pytest.approx(
            right.diagnostics.density_residual, rel=1.0e-5, abs=2.0e-14
        )


def test_nonconverged_midpoint_is_a_hard_failure_with_structured_diagnostics(
    propagation_references: dict[str, object],
) -> None:
    reference = propagation_references["lih"]
    base = _simulation_config(reference, FormulationKind.BARE_LENGTH_GAUGE)
    config = replace(
        base,
        propagation=PropagationConfig(
            base.propagation.time_grid,
            base.propagation.integrator,
            density_tolerance=1.0e-30,
            max_iterations=1,
        ),
    )
    simulation = build_simulation(config, reference)
    initial = np.array(simulation.state.coefficients, copy=True)
    with pytest.raises(MidpointConvergenceError) as captured:
        simulation.step()
    error = captured.value
    assert error.step_index == 0
    assert error.iterations == 1
    assert error.density_residual > config.propagation.density_tolerance
    assert np.isfinite(error.hamiltonian_residual)
    assert simulation.state.step_index == 0
    assert np.array_equal(simulation.state.coefficients, initial)


def test_metric_correction_can_only_be_disabled_as_an_explicit_diagnostic(
    propagation_references: dict[str, object],
) -> None:
    reference = propagation_references["lih"]
    simulation = build_simulation(
        _simulation_config(
            reference,
            FormulationKind.P0_E1,
            gauge=GaugeRepresentation.VELOCITY,
        ),
        reference,
    )
    simulation.propagator.transport = ConnectionAwareTransport(
        simulation.formulation,
        simulation.config.propagation.time_grid.step_au,
        apply_metric_correction=False,
    )
    result = simulation.step()
    assert result.diagnostics.metric_correction_diagnostic_mode
    assert not result.diagnostics.half_step_link.correction_applied
    assert not result.diagnostics.full_step_link.correction_applied
    assert result.diagnostics.full_step_link.raw_metric_residual > 1.0e-8
    assert result.diagnostics.final_metric_residual > 1.0e-8


@pytest.mark.parametrize("name", ("h2", "lih"))
def test_complete_connection_scem_has_second_order_molecular_convergence(
    propagation_references: dict[str, object],
    name: str,
) -> None:
    reference = propagation_references[name]
    pulse = Sin2VectorPotentialPulseConfig(
        peak_electric_field_au=0.02,
        angular_frequency_au=2.0 * math.pi,
        cycles=1,
        polarization=(0.2, -0.3, 1.0),
        carrier_phase_rad=0.37,
    )
    final_densities = []
    for maximum_step in (0.125, 0.0625, 0.03125):
        grid = pulse_aligned_time_grid(Sin2VectorPotentialPulse(pulse), maximum_step)
        config = SimulationConfig(
            reference=ReferenceLinkConfig(reference.fingerprint_sha256),
            formulation=FormulationConfig(
                FormulationKind.P0_E1,
                GaugeRepresentation.VELOCITY,
            ),
            source=pulse,
            propagation=PropagationConfig(
                grid,
                IntegratorKind.CONNECTION_AWARE_SCEM,
                density_tolerance=1.0e-13,
            ),
            backend=BackendConfig(),
        )
        simulation = build_simulation(config, reference)
        state = simulation.state
        for result in simulation.propagator.propagate(state):
            state = result.state
        source = simulation.source_sample(SourceSampling.ENDPOINT, state.step_index)
        evaluation = simulation.formulation.evaluate(state.density(), source)
        final_densities.append(np.array(evaluation.field_free_density, copy=True))
    coarse_difference = np.linalg.norm(final_densities[0] - final_densities[1])
    fine_difference = np.linalg.norm(final_densities[1] - final_densities[2])
    measured_order = math.log2(coarse_difference / fine_difference)
    assert 1.9 < measured_order < 2.4
