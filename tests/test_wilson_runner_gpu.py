from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind, StepSchedule, WilsonStationaryStateLinkConfig
from aion.electronic_structure import save_wilson_stationary_state
from aion.io import WilsonTrajectory, load_wilson_checkpoint
from aion.workflows import BuiltWilsonSimulation, build_simulation, run
from test_wilson_workflow import prepare_exact_wilson_inputs

pytestmark = pytest.mark.gpu


def test_wilson_streaming_and_checkpoint_boundaries_work_on_physical_gpu(
    tmp_path: Path,
) -> None:
    config, reference, state = prepare_exact_wilson_inputs()
    reference_path = tmp_path / "reference.h5"
    stationary_path = tmp_path / "stationary.h5"
    reference.save(reference_path)
    save_wilson_stationary_state(state, stationary_path)
    gpu_config = replace(
        config,
        reference=replace(config.reference, path=reference_path),
        stationary_state=WilsonStationaryStateLinkConfig(
            state.fingerprint_sha256,
            stationary_path,
        ),
        backend=BackendConfig(kind=BackendKind.GPU, device_index=0),
        output=replace(
            config.output,
            directory=tmp_path / "run",
            schedules=replace(config.output.schedules, energy=StepSchedule(every=1)),
        ),
    )
    simulation = build_simulation(gpu_config, reference, stationary_state=state)
    assert isinstance(simulation, BuiltWilsonSimulation)
    trajectory = run(simulation)
    assert isinstance(trajectory, WilsonTrajectory)
    assert trajectory.final_step == gpu_config.propagation.time_grid.intervals
    simulation.quadrature.backend.assert_resident(
        simulation.density,
        name="final streamed GPU density",
    )
    checkpoint = load_wilson_checkpoint(
        tmp_path / "run" / f"checkpoint_{gpu_config.propagation.time_grid.intervals:08d}.h5"
    )
    np.testing.assert_array_equal(
        checkpoint.contravariant_density,
        simulation.quadrature.backend.to_host(simulation.density),
    )

    off = StepSchedule(every=0, include_initial=False, include_final=False)
    energy_config = replace(
        gpu_config,
        output=replace(
            gpu_config.output,
            directory=tmp_path / "energy_only",
            schedules=replace(
                gpu_config.output.schedules,
                dipole_current=off,
                diagnostics=off,
                energy=StepSchedule(every=1),
            ),
        ),
    )
    energy_only = build_simulation(energy_config, reference, stationary_state=state)
    energy_trajectory = run(energy_only)
    energy_only.quadrature.backend.assert_resident(
        energy_only.density, name="energy-only GPU Wilson density"
    )
    np.testing.assert_array_equal(
        energy_only.quadrature.backend.to_host(energy_only.density),
        simulation.quadrature.backend.to_host(simulation.density),
    )
    assert energy_trajectory.accumulated_source_work_au == trajectory.accumulated_source_work_au
    assert energy_trajectory.read_series("current/uniform_source").steps.tolist() == [0]
    assert energy_trajectory.read_series("energy/molecular_total").steps.tolist() == list(
        range(energy_config.propagation.time_grid.intervals + 1)
    )
    np.testing.assert_array_equal(
        energy_trajectory.read_series("energy/molecular_total").values,
        trajectory.read_series("energy/molecular_total").values,
    )
