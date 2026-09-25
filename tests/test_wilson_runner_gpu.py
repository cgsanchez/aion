from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind, WilsonStationaryStateLinkConfig
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
        output=replace(config.output, directory=tmp_path / "run"),
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
