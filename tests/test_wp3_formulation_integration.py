from __future__ import annotations

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
    ZeroSourceConfig,
)
from aion.electromagnetism import Sin2VectorPotentialPulse, pulse_aligned_time_grid
from aion.formulations import P0, P0E1, AODensity, SourceSampling
from aion.workflows import build_simulation
from test_reference_integration import molecular_config

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def references() -> dict[str, object]:
    from aion.electronic_structure import prepare_pyscf_reference

    return {name: prepare_pyscf_reference(molecular_config(name)) for name in ("h2", "lih")}


def _pulse() -> Sin2VectorPotentialPulseConfig:
    return Sin2VectorPotentialPulseConfig(
        peak_electric_field_au=0.007,
        angular_frequency_au=0.61,
        cycles=1,
        polarization=(0.2, -0.3, 1.0),
        carrier_phase_rad=0.37,
    )


def _simulation_config(
    reference: object,
    kind: FormulationKind,
    *,
    gauge: GaugeRepresentation | None = None,
    velocity_fraction: float | None = None,
    source: object | None = None,
    backend: BackendConfig | None = None,
) -> SimulationConfig:
    pulse = _pulse()
    selected_source = pulse if source is None else source
    grid = pulse_aligned_time_grid(Sin2VectorPotentialPulse(pulse), 0.7)
    integrator = (
        IntegratorKind.FIXED_METRIC_SCEM
        if kind
        in {
            FormulationKind.BARE_LENGTH_GAUGE,
            FormulationKind.BARE_VELOCITY_GAUGE,
        }
        else IntegratorKind.CONNECTION_AWARE_SCEM
    )
    return SimulationConfig(
        reference=ReferenceLinkConfig(reference.fingerprint_sha256),
        formulation=FormulationConfig(kind, gauge, velocity_fraction),
        source=selected_source,
        propagation=PropagationConfig(grid, integrator),
        backend=BackendConfig() if backend is None else backend,
    )


def _host(simulation: object, value: object) -> np.ndarray:
    return simulation.workspace.backend.to_host(value)


@pytest.mark.parametrize("name", ("h2", "lih"))
def test_all_formulations_share_zero_field_reference_and_energy_components(
    references: dict[str, object], name: str
) -> None:
    reference = references[name]
    ledgers = {}
    for kind, gauge in (
        (FormulationKind.BARE_LENGTH_GAUGE, None),
        (FormulationKind.BARE_VELOCITY_GAUGE, None),
        (FormulationKind.P0, GaugeRepresentation.LENGTH),
        (FormulationKind.P0_E1, GaugeRepresentation.VELOCITY),
    ):
        simulation = build_simulation(
            _simulation_config(
                reference,
                kind,
                gauge=gauge,
                source=ZeroSourceConfig(),
            ),
            reference,
        )
        sample = simulation.source_sample(SourceSampling.ENDPOINT, 0)
        evaluation = simulation.formulation.evaluate(simulation.density, sample)
        ledger = simulation.formulation.energy(evaluation, sample)
        ledgers[kind] = float(_host(simulation, ledger.energy_matter_total))
        components = sum(
            float(_host(simulation, value))
            for value in (
                ledger.energy_kinetic_mechanical,
                ledger.energy_electron_nuclear,
                ledger.energy_hartree,
                ledger.energy_exchange_correlation,
                ledger.energy_nuclear_repulsion,
            )
        )
        assert components == pytest.approx(ledgers[kind], abs=2.0e-10)
        assert float(_host(simulation, ledger.energy_ward_residual)) == pytest.approx(
            0.0, abs=2.0e-9
        )
    for value in ledgers.values():
        assert value == pytest.approx(reference.ground_state.energy_total_au, abs=2.0e-9)


def test_p0_e1_lih_is_gauge_covariant_and_places_e1_once(references: dict[str, object]) -> None:
    reference = references["lih"]
    length = build_simulation(
        _simulation_config(reference, FormulationKind.P0_E1, gauge=GaugeRepresentation.LENGTH),
        reference,
    )
    velocity = build_simulation(
        _simulation_config(reference, FormulationKind.P0_E1, gauge=GaugeRepresentation.VELOCITY),
        reference,
    )
    velocity_p0 = build_simulation(
        _simulation_config(reference, FormulationKind.P0, gauge=GaugeRepresentation.VELOCITY),
        reference,
    )
    index = 2
    source_l = length.source_sample(SourceSampling.MIDPOINT, index)
    source_v = velocity.source_sample(SourceSampling.MIDPOINT, index)
    excited_matrix = np.array(length.density.matrix, copy=True)
    excited_matrix[0, -1] += 0.025j
    excited_matrix[-1, 0] -= 0.025j
    excited_length = AODensity.from_matrix(excited_matrix, length.workspace.backend)
    eval_l = length.formulation.evaluate(excited_length, source_l)
    trial_v = velocity.formulation.evaluate(velocity.density, source_v)
    assert trial_v.theta is not None
    transformed = AODensity.from_matrix(
        trial_v.theta * excited_length.matrix, velocity.workspace.backend
    )
    eval_v = velocity.formulation.evaluate(transformed, source_v)
    current_l = length.formulation.currents(eval_l, source_l)
    current_v = velocity.formulation.currents(eval_v, source_v)
    energy_l = length.formulation.energy(eval_l, source_l)
    energy_v = velocity.formulation.energy(eval_v, source_v)

    assert (
        np.linalg.norm(
            _host(length, eval_l.field_free_density) - _host(velocity, eval_v.field_free_density)
        )
        < 2.0e-11
    )
    assert (
        np.linalg.norm(
            _host(length, current_l.electronic_dipole)
            - _host(velocity, current_v.electronic_dipole)
        )
        < 2.0e-10
    )
    assert (
        np.linalg.norm(
            _host(length, current_l.source_current) - _host(velocity, current_v.source_current)
        )
        < 2.0e-9
    )
    assert float(_host(length, energy_l.energy_matter_total)) == pytest.approx(
        float(_host(velocity, energy_v.energy_matter_total)), abs=2.0e-10
    )
    for simulation, current, energy in (
        (length, current_l, energy_l),
        (velocity, current_v, energy_v),
    ):
        assert current.e1_residual_source_current is not None
        assert np.linalg.norm(_host(simulation, current.e1_residual_source_current)) > 1.0e-7
        assert np.linalg.norm(_host(simulation, current.continuity_residual)) < 3.0e-9
        assert (
            np.linalg.norm(_host(simulation, current.source_current_dipole_derivative_residual))
            < 3.0e-9
        )
        assert abs(float(_host(simulation, energy.energy_ward_residual))) < 3.0e-8
        assert current.ambient_projected_mechanical_current is not None
        assert eval_l.theta is not None
        current_evaluation = eval_l if simulation is length else eval_v
        theta = current_evaluation.theta
        assert theta is not None
        expected_mechanical = (
            simulation.formulation.context.charge
            / simulation.formulation.context.mass
            * simulation.workspace.backend.namespace.real(
                simulation.workspace.backend.namespace.einsum(
                    "ij,xji->x",
                    current_evaluation.density.matrix,
                    theta[None, :, :]
                    * simulation.workspace.require("operators.canonical_momentum"),
                    optimize=True,
                )
            )
        )
        assert (
            np.linalg.norm(
                _host(simulation, current.ambient_projected_mechanical_current)
                - _host(simulation, expected_mechanical)
            )
            < 2.0e-11
        )

    p0_source = velocity_p0.source_sample(SourceSampling.MIDPOINT, index)
    p0_density = AODensity.from_matrix(
        trial_v.theta * excited_length.matrix, velocity_p0.workspace.backend
    )
    eval_p0 = velocity_p0.formulation.evaluate(p0_density, p0_source)
    assert eval_v.e1_potential is not None
    assert (
        np.linalg.norm(
            _host(velocity, eval_v.triple.hamiltonian_eom)
            - _host(velocity_p0, eval_p0.triple.hamiltonian_eom)
        )
        < 2.0e-10
    )
    assert (
        np.linalg.norm(
            _host(velocity, eval_v.hamiltonian_dynamic - eval_v.triple.hamiltonian_eom)
            - _host(velocity, eval_v.e1_potential)
        )
        < 2.0e-12
    )
    assert (
        np.linalg.norm(
            _host(velocity, eval_v.triple.connection)
            - _host(velocity_p0, eval_p0.triple.connection)
            - 1j * _host(velocity, eval_v.e1_potential)
        )
        < 2.0e-12
    )


@pytest.mark.parametrize("fraction", (0.25, 0.5, 0.75))
def test_p0_e1_mixed_gauge_interpolates_source_and_observables_are_covariant(
    references: dict[str, object],
    fraction: float,
) -> None:
    reference = references["lih"]
    length = build_simulation(
        _simulation_config(reference, FormulationKind.P0_E1, gauge=GaugeRepresentation.LENGTH),
        reference,
    )
    velocity = build_simulation(
        _simulation_config(reference, FormulationKind.P0_E1, gauge=GaugeRepresentation.VELOCITY),
        reference,
    )
    mixed = build_simulation(
        _simulation_config(
            reference,
            FormulationKind.P0_E1,
            velocity_fraction=fraction,
        ),
        reference,
    )
    index = 2
    source_l = length.source_sample(SourceSampling.MIDPOINT, index)
    source_v = velocity.source_sample(SourceSampling.MIDPOINT, index)
    source_m = mixed.source_sample(SourceSampling.MIDPOINT, index)
    assert source_m.gauge is GaugeRepresentation.MIXED
    assert source_m.velocity_fraction == fraction
    for name in (
        "vector_potential_reduced",
        "vector_potential_reduced_dot",
        "node_scalar_potential",
        "pair_link",
        "pair_link_dot",
        "pair_electromotive_potential",
    ):
        expected = (1.0 - fraction) * getattr(source_l, name) + fraction * getattr(source_v, name)
        assert np.allclose(getattr(source_m, name), expected, atol=2.0e-15)
    assert np.allclose(
        source_m.pair_electromotive_potential,
        source_l.pair_electromotive_potential,
        atol=2.0e-15,
    )

    excited_matrix = np.array(length.density.matrix, copy=True)
    excited_matrix[0, -1] += 0.025j
    excited_matrix[-1, 0] -= 0.025j
    excited_length = AODensity.from_matrix(excited_matrix, length.workspace.backend)
    evaluation_l = length.formulation.evaluate(excited_length, source_l)
    trial_m = mixed.formulation.evaluate(mixed.density, source_m)
    assert trial_m.theta is not None
    transformed = AODensity.from_matrix(
        trial_m.theta * excited_length.matrix,
        mixed.workspace.backend,
    )
    evaluation_m = mixed.formulation.evaluate(transformed, source_m)
    current_l = length.formulation.currents(evaluation_l, source_l)
    current_m = mixed.formulation.currents(evaluation_m, source_m)
    energy_l = length.formulation.energy(evaluation_l, source_l)
    energy_m = mixed.formulation.energy(evaluation_m, source_m)
    assert np.linalg.norm(evaluation_l.field_free_density - evaluation_m.field_free_density) < (
        2.0e-11
    )
    assert np.linalg.norm(current_l.electronic_dipole - current_m.electronic_dipole) < 2.0e-10
    assert np.linalg.norm(current_l.source_current - current_m.source_current) < 2.0e-9
    assert float(energy_l.energy_matter_total) == pytest.approx(
        float(energy_m.energy_matter_total), abs=2.0e-10
    )


def test_h2_is_e1_null_control_and_lih_is_positive_control(references: dict[str, object]) -> None:
    norms = {}
    for name in ("h2", "lih"):
        reference = references[name]
        simulation = build_simulation(
            _simulation_config(
                reference,
                FormulationKind.P0_E1,
                gauge=GaugeRepresentation.LENGTH,
                source=ZeroSourceConfig(),
            ),
            reference,
        )
        assert isinstance(simulation.formulation, P0E1)
        central = simulation.formulation.context.central_dipoles()
        norms[name] = float(np.linalg.norm(_host(simulation, central)))
    assert norms["h2"] < 1.0e-11
    assert norms["lih"] > 0.1


def test_e1_operator_bundle_is_reproducible_and_fingerprinted(
    references: dict[str, object],
) -> None:
    from aion.electronic_structure import build_e1_operators

    reference = references["h2"]
    first = build_e1_operators(reference.core_operators, reference.anchor_topology)
    second = build_e1_operators(reference.core_operators, reference.anchor_topology)
    assert first.fingerprint_sha256 == second.fingerprint_sha256
    assert np.array_equal(first.central_dipoles, second.central_dipoles)
    assert not first.central_dipoles.flags.writeable


def test_bare_velocity_energy_contains_linear_and_diamagnetic_terms(
    references: dict[str, object],
) -> None:
    reference = references["lih"]
    simulation = build_simulation(
        _simulation_config(reference, FormulationKind.BARE_VELOCITY_GAUGE), reference
    )
    sample = simulation.source_sample(SourceSampling.MIDPOINT, 2)
    excited_matrix = np.array(simulation.density.matrix, copy=True)
    excited_matrix[0, -1] += 0.025j
    excited_matrix[-1, 0] -= 0.025j
    excited = AODensity.from_matrix(excited_matrix, simulation.workspace.backend)
    evaluation = simulation.formulation.evaluate(excited, sample)
    ledger = simulation.formulation.energy(evaluation, sample)
    assert abs(float(_host(simulation, ledger.energy_kinetic_vector_potential_linear))) > 0.0
    assert float(_host(simulation, ledger.energy_kinetic_diamagnetic)) > 0.0
    expected = (
        float(_host(simulation, ledger.energy_kinetic_canonical))
        + float(_host(simulation, ledger.energy_kinetic_vector_potential_linear))
        + float(_host(simulation, ledger.energy_kinetic_diamagnetic))
    )
    assert float(_host(simulation, ledger.energy_kinetic_mechanical)) == pytest.approx(
        expected, abs=2.0e-12
    )
    assert abs(float(_host(simulation, ledger.energy_ward_residual))) < 3.0e-8


def test_length_gauge_generator_rate_uses_analytic_field_derivative(
    references: dict[str, object],
) -> None:
    reference = references["lih"]
    simulation = build_simulation(
        _simulation_config(reference, FormulationKind.BARE_LENGTH_GAUGE), reference
    )
    sample = simulation.source_sample(SourceSampling.MIDPOINT, 2)
    evaluation = simulation.formulation.evaluate(simulation.density, sample)
    current = simulation.formulation.currents(evaluation, sample)
    ledger = simulation.formulation.energy(evaluation, sample)
    assert ledger.energy_generator_rate_analytic is not None
    expected = -np.dot(
        _host(simulation, sample.electric_field_dot),
        _host(simulation, current.total_dipole),
    )
    assert float(_host(simulation, ledger.energy_generator_rate_analytic)) == pytest.approx(
        expected, abs=2.0e-12
    )


def test_pure_dft_energy_functional_derivative_is_independent_of_source_power(
    references: dict[str, object],
) -> None:
    reference = references["lih"]
    simulation = build_simulation(
        _simulation_config(
            reference,
            FormulationKind.BARE_LENGTH_GAUGE,
            source=ZeroSourceConfig(),
        ),
        reference,
    )
    model = simulation.formulation.context.electronic_model
    density = np.asarray(simulation.density.matrix)
    rng = np.random.default_rng(419)
    raw = rng.normal(size=density.shape) + 1j * rng.normal(size=density.shape)
    direction_host = 0.5 * (raw + raw.conj().T)
    direction_host /= np.linalg.norm(direction_host)
    direction = simulation.workspace.backend.asarray(direction_host, dtype=np.complex128)
    build = model.build(simulation.density.matrix)
    analytic = float(_host(simulation, model.energy_rate(build, direction)))
    epsilon = 2.0e-5

    def internal_energy(matrix: np.ndarray) -> float:
        candidate = AODensity.from_matrix(matrix, simulation.workspace.backend)
        return float(
            _host(simulation, model.energy(model.build(candidate.matrix)).energy_internal_total)
        )

    finite_difference = (
        internal_energy(density + epsilon * direction_host)
        - internal_energy(density - epsilon * direction_host)
    ) / (2.0 * epsilon)
    assert analytic == pytest.approx(finite_difference, rel=2.0e-6, abs=2.0e-7)


def test_covariant_gauge_is_part_of_simulation_identity(references: dict[str, object]) -> None:
    reference = references["h2"]
    length = _simulation_config(reference, FormulationKind.P0, gauge=GaugeRepresentation.LENGTH)
    velocity = replace(
        length,
        formulation=FormulationConfig(FormulationKind.P0, GaugeRepresentation.VELOCITY),
    )
    assert length.scientific_id != velocity.scientific_id
    assert isinstance(
        build_simulation(length, reference).formulation,
        P0,
    )
