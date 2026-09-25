from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aion.config import (
    AffineElectromagneticSourceConfig,
    BackendConfig,
    ExactWilsonActionConfig,
    FixedTimeGrid,
    OutputConfig,
    RationalApproximation,
    ReferenceLinkConfig,
    WilsonGridKind,
    WilsonGridPruning,
    WilsonIntegratorKind,
    WilsonMagneticGaugeKind,
    WilsonNumericsConfig,
    WilsonPropagationConfig,
    WilsonSimulationConfig,
    WilsonStationaryBranch,
    WilsonStationaryConfig,
    WilsonStationaryOutputConfig,
    WilsonStationaryPolicyConfig,
    WilsonStationaryStateLinkConfig,
    XCFamily,
    ZeroSourceConfig,
)
from aion.electromagnetism import build_affine_electromagnetic_source
from aion.electronic_structure import (
    AOGridPolicy,
    ExactWilsonDynamicSample,
    PreparedReference,
    RIMetricRankPolicy,
    StationarySCFPolicy,
    WilsonStationaryStateData,
    auxiliary_space_fingerprint,
    capture_wilson_stationary_state,
    prepare_ao_quadrature,
    prepare_exact_wilson_dynamic_sample,
    prepare_exact_wilson_stationary_factory,
    prepare_pyscf_reference,
)
from aion.observables import evaluate_exact_wilson_endpoint_observation
from aion.propagation import (
    NonlinearGaussMagnusPolicy,
    propagate_nonlinear_contravariant_density,
)
from aion.workflows import (
    BuiltWilsonSimulation,
    ExactWilsonDynamicCache,
    build_simulation,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.integration


def prepare_exact_wilson_inputs() -> tuple[
    WilsonSimulationConfig,
    PreparedReference,
    WilsonStationaryStateData,
]:
    base = molecular_config("h2")
    reference_config = replace(
        base,
        electronic_structure=replace(
            base.electronic_structure,
            functional="lda,vwn",
            xc_family=XCFamily.LDA,
            grid_level=1,
            density_fitting=True,
            auxiliary_basis="weigend",
        ),
    )
    reference = prepare_pyscf_reference(reference_config)
    numerics = WilsonNumericsConfig(
        grid_kind=WilsonGridKind.REFERENCE,
        grid_level=None,
        grid_pruning=WilsonGridPruning.NONE,
        block_size=512,
        dynamic_cache_entries=2,
        auxiliary_basis="weigend",
        ri_relative_threshold=0.0,
        ri_absolute_threshold=1.0e-7,
        ri_maximum_rank=None,
    )
    source_config = AffineElectromagneticSourceConfig(
        electric=ZeroSourceConfig(),
        electric_field_origin_offset_au=(0.0, 0.0, 0.0),
        magnetic_field_reference_au=(0.0, 0.0, 0.01),
        magnetic_field_rate_au=(0.0, 0.0, 0.0),
        magnetic_reference_time_au=0.0,
        magnetic_gauge=WilsonMagneticGaugeKind.SYMMETRIC,
    )
    action = ExactWilsonActionConfig(WilsonStationaryBranch.KOHN_SHAM_LDA)
    stationary_config = WilsonStationaryConfig(
        reference=ReferenceLinkConfig(reference.fingerprint_sha256, Path("reference.h5")),
        action=action,
        numerics=numerics,
        source=source_config,
        source_time_au=0.0,
        stationary=WilsonStationaryPolicyConfig(
            maximum_iterations=80,
            density_tolerance=2.0e-9,
            orbital_tolerance=2.0e-9,
            energy_tolerance_au=2.0e-10,
            damping=0.5,
            diis_start_iteration=2,
            diis_space=8,
        ),
        backend=BackendConfig(),
        output=WilsonStationaryOutputConfig(Path("stationary.h5")),
    )
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.reference(),
        block_size=numerics.block_size,
    )
    rank_policy = RIMetricRankPolicy(
        relative_threshold=numerics.ri_relative_threshold,
        absolute_threshold=numerics.ri_absolute_threshold,
    )
    factory = prepare_exact_wilson_stationary_factory(
        quadrature,
        auxiliary_basis=numerics.auxiliary_basis,
        functional=reference.config.electronic_structure.functional,
        rank_policy=rank_policy,
    )
    source_provider = build_affine_electromagnetic_source(
        source_config,
        reference.electromagnetic_origin_au,
    )
    source = source_provider.sample(0.0)
    model = factory.model(source.gauge, action.branch)
    solved = model.solve(
        policy=StationarySCFPolicy(
            maximum_iterations=80,
            density_tolerance=2.0e-9,
            orbital_tolerance=2.0e-9,
            energy_tolerance_au=2.0e-10,
        )
    )
    state = capture_wilson_stationary_state(
        stationary_config,
        solved,
        source,
        grid_fingerprint_sha256=quadrature.grid.fingerprint_sha256,
        auxiliary_space_fingerprint_sha256=auxiliary_space_fingerprint(factory.hartree_evaluator),
        backend=quadrature.backend,
    )
    simulation = WilsonSimulationConfig(
        reference=stationary_config.reference,
        stationary_state=WilsonStationaryStateLinkConfig(
            state.fingerprint_sha256,
            Path("stationary.h5"),
        ),
        action=action,
        numerics=numerics,
        source=source_config,
        propagation=WilsonPropagationConfig(
            time_grid=FixedTimeGrid(0.0, 0.02, 2),
            integrator=WilsonIntegratorKind.NONLINEAR_GAUSS_MAGNUS,
            rational_approximation=RationalApproximation.PADE_22,
            nonlinear_tolerance=1.0e-10,
            maximum_iterations=40,
        ),
        backend=BackendConfig(),
        output=OutputConfig(Path("run")),
    )
    return simulation, reference, state


@pytest.fixture(scope="module")
def exact_wilson_inputs() -> tuple[
    WilsonSimulationConfig,
    PreparedReference,
    WilsonStationaryStateData,
]:
    return prepare_exact_wilson_inputs()


def test_exact_runtime_build_step_and_action_observables(
    exact_wilson_inputs: tuple[
        WilsonSimulationConfig, PreparedReference, WilsonStationaryStateData
    ],
) -> None:
    config, reference, state = exact_wilson_inputs
    runtime = build_simulation(config, reference, stationary_state=state)
    assert isinstance(runtime, BuiltWilsonSimulation)
    assert runtime.simulation_id == config.scientific_id
    runtime.quadrature.backend.assert_resident(runtime.density, name="runtime density")

    lightweight = runtime.observe_endpoint()
    assert lightweight.energy is None
    assert lightweight.uniform_source_current_au.shape == (3,)
    assert lightweight.electronic_dipole_au.shape == (3,)
    np.testing.assert_allclose(
        lightweight.molecular_total_dipole_au,
        lightweight.electronic_dipole_au + lightweight.fixed_nuclear_dipole_au,
    )
    assert float(lightweight.metric_particle_number) == pytest.approx(2.0, abs=2.0e-10)

    complete = runtime.observe_endpoint(include_energy=True)
    assert complete.energy is not None
    assert float(complete.energy.molecular_total_au) == pytest.approx(
        state.energies.molecular_total_au,
        abs=2.0e-9,
    )
    direct_samples: dict[float, ExactWilsonDynamicSample] = {}

    def direct_sample(time_au: float) -> ExactWilsonDynamicSample:
        if time_au not in direct_samples:
            direct_samples[time_au] = prepare_exact_wilson_dynamic_sample(
                runtime.factory,
                runtime.source_provider.sample(time_au),
                config.action.branch,
            )
        return direct_samples[time_au]

    direct = propagate_nonlinear_contravariant_density(
        state.contravariant_density,
        initial_time_au=0.0,
        interval_au=0.02,
        intervals=1,
        metric_provider=lambda time: direct_sample(time).one_electron.metric,
        eom_provider=lambda time, density: direct_sample(time).evaluate(density).triple,
        backend=runtime.quadrature.backend,
        policy=NonlinearGaussMagnusPolicy(tolerance=1.0e-10, maximum_iterations=40),
    )
    result = runtime.step()
    assert runtime.boundary_index == 1
    assert runtime.current_time_au == pytest.approx(0.02)
    assert result.metric_correction_applied is False
    assert result.density_update == "coefficient_congruence"
    np.testing.assert_allclose(
        result.contravariant_density,
        direct.contravariant_densities[-1],
        atol=2.0e-13,
        rtol=2.0e-13,
    )


def test_dynamic_cache_is_bounded_and_uniform_current_pairs_to_power(
    exact_wilson_inputs: tuple[
        WilsonSimulationConfig, PreparedReference, WilsonStationaryStateData
    ],
) -> None:
    config, reference, state = exact_wilson_inputs
    runtime = build_simulation(config, reference, stationary_state=state)
    changing_source = replace(
        config.source,
        electric_field_origin_offset_au=(3.0e-4, -2.0e-4, 1.0e-4),
        magnetic_field_rate_au=(0.0, 0.0, 1.0e-4),
    )
    changing_provider = build_affine_electromagnetic_source(
        changing_source,
        reference.electromagnetic_origin_au,
    )
    cache = ExactWilsonDynamicCache(
        factory=runtime.factory,
        source_provider=changing_provider,
        branch=config.action.branch,
        maximum_entries=2,
    )
    for time_au in (0.0, 0.1, 0.2, 0.3, 0.4):
        cache.sample(time_au)
        assert cache.statistics.spatial_entries <= 2
        assert cache.statistics.sample_entries <= 2
    assert cache.statistics.spatial_misses == 5

    uniform_source = replace(changing_source, magnetic_field_rate_au=(0.0, 0.0, 0.0))
    uniform_cache = ExactWilsonDynamicCache(
        factory=runtime.factory,
        source_provider=build_affine_electromagnetic_source(
            uniform_source,
            reference.electromagnetic_origin_au,
        ),
        branch=config.action.branch,
        maximum_entries=2,
    )
    evaluation = uniform_cache.evaluate(0.37, runtime.density)
    observation = evaluate_exact_wilson_endpoint_observation(
        evaluation,
        runtime.density,
        include_energy=False,
    )
    expected_power = np.dot(
        np.asarray(observation.uniform_source_current_au),
        np.asarray(evaluation.sample.source.electric_field_origin_au),
    )
    np.testing.assert_allclose(
        observation.source_power_au,
        expected_power,
        atol=2.0e-10,
        rtol=2.0e-10,
    )
