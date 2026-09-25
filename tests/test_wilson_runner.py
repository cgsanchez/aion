from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from aion.config import FixedTimeGrid, WilsonStationaryStateLinkConfig
from aion.electromagnetism import build_affine_electromagnetic_source
from aion.electronic_structure import save_wilson_stationary_state
from aion.errors import RunCancelledError
from aion.io import (
    WilsonTrajectory,
    load_wilson_checkpoint,
    load_wilson_trajectory,
)
from aion.workflows import (
    BuiltWilsonSimulation,
    RunControl,
    build_simulation,
    load_trajectory,
    resume,
    run,
)
from test_wilson_workflow import prepare_exact_wilson_inputs

pytestmark = pytest.mark.integration


def _production_inputs(root: Path, *, intervals: int = 4) -> tuple[object, object, object]:
    base_config, reference, state = prepare_exact_wilson_inputs()
    reference_path = root / "reference.h5"
    stationary_path = root / "stationary.h5"
    reference.save(reference_path)
    save_wilson_stationary_state(state, stationary_path)
    propagation = replace(
        base_config.propagation,
        time_grid=FixedTimeGrid(
            base_config.propagation.time_grid.start_au,
            base_config.propagation.time_grid.step_au,
            intervals,
        ),
    )
    config = replace(
        base_config,
        reference=replace(base_config.reference, path=reference_path),
        stationary_state=WilsonStationaryStateLinkConfig(
            state.fingerprint_sha256,
            stationary_path,
        ),
        propagation=propagation,
    )
    return config, reference, state


def test_wilson_streaming_run_checkpoint_dispatch_and_bounded_cache(tmp_path: Path) -> None:
    config, reference, state = _production_inputs(tmp_path, intervals=8)
    config = replace(config, output=replace(config.output, directory=tmp_path / "run"))
    simulation = build_simulation(config, reference, stationary_state=state)
    assert isinstance(simulation, BuiltWilsonSimulation)
    trajectory = run(simulation)
    assert isinstance(trajectory, WilsonTrajectory)
    assert trajectory.final_step == 8
    assert trajectory.series_names
    assert simulation.dynamic_cache.statistics.spatial_entries <= (
        config.numerics.dynamic_cache_entries
    )
    assert simulation.dynamic_cache.statistics.sample_entries <= (
        config.numerics.dynamic_cache_entries
    )

    dipole = trajectory.read_series("dipole/molecular_total")
    assert dipole.steps.tolist() == list(range(9))
    assert dipole.values.shape == (9, 3)
    loaded_direct = load_wilson_trajectory(trajectory.path)
    loaded_dispatched = load_trajectory(trajectory.path)
    assert loaded_direct.sha256 == trajectory.sha256
    assert isinstance(loaded_dispatched, WilsonTrajectory)
    assert loaded_dispatched.sha256 == trajectory.sha256
    final_checkpoint = load_wilson_checkpoint(tmp_path / "run/checkpoint_00000008.h5")
    assert final_checkpoint.global_step == 8
    np.testing.assert_array_equal(
        final_checkpoint.contravariant_density,
        simulation.quadrature.backend.to_host(simulation.density),
    )


def test_wilson_cancel_resume_matches_uninterrupted_boundary_exactly(tmp_path: Path) -> None:
    config, reference, state = _production_inputs(tmp_path, intervals=4)
    full_config = replace(
        config,
        output=replace(config.output, directory=tmp_path / "full"),
    )
    full = build_simulation(full_config, reference, stationary_state=state)
    full_trajectory = run(full)
    full_checkpoint = load_wilson_checkpoint(tmp_path / "full/checkpoint_00000004.h5")

    interrupted_config = replace(
        config,
        output=replace(config.output, directory=tmp_path / "interrupted"),
    )
    interrupted = build_simulation(interrupted_config, reference, stationary_state=state)
    control = RunControl()

    def cancel_after_first(step: int) -> None:
        if step == 1:
            control.request_cancel()

    control.after_accepted_step = cancel_after_first
    with pytest.raises(RunCancelledError):
        run(interrupted, control=control)
    assert (tmp_path / "interrupted/trajectory.failed.h5").exists()
    restart_path = tmp_path / "interrupted/checkpoint_00000001.h5"
    restart_checkpoint = load_wilson_checkpoint(restart_path)
    assert restart_checkpoint.global_step == 1

    resumed_trajectory = resume(restart_path, output=tmp_path / "resumed")
    assert isinstance(resumed_trajectory, WilsonTrajectory)
    assert resumed_trajectory.parent_run_id == restart_checkpoint.run_id
    assert resumed_trajectory.parent_checkpoint_sha256 is not None
    assert resumed_trajectory.global_step_offset == 1
    resumed_checkpoint = load_wilson_checkpoint(tmp_path / "resumed/checkpoint_00000004.h5")
    np.testing.assert_array_equal(
        resumed_checkpoint.contravariant_density,
        full_checkpoint.contravariant_density,
    )
    assert resumed_checkpoint.accumulated_source_work_au == pytest.approx(
        full_checkpoint.accumulated_source_work_au,
        abs=2.0e-14,
        rel=0.0,
    )
    assert resumed_trajectory.final_step == full_trajectory.final_step
    with (
        h5py.File(full_trajectory.path, "r") as full_handle,
        h5py.File(resumed_trajectory.path, "r") as resumed_handle,
    ):
        full_diagnostics = full_handle["diagnostics/propagation"]
        resumed_diagnostics = resumed_handle["diagnostics/propagation"]
        full_steps = np.asarray(full_diagnostics["step"][...])
        resumed_steps = np.asarray(resumed_diagnostics["step"][...])
        np.testing.assert_array_equal(resumed_steps, np.asarray((2, 3, 4)))
        mask = np.isin(full_steps, resumed_steps)
        for name in resumed_diagnostics:
            np.testing.assert_array_equal(
                resumed_diagnostics[name][...],
                full_diagnostics[name][...][mask],
            )


def test_wilson_interval_work_uses_the_two_converged_gauss_nodes(tmp_path: Path) -> None:
    base_config, reference, state = prepare_exact_wilson_inputs()
    source_config = replace(
        base_config.source,
        electric_field_origin_offset_au=(2.0e-4, -1.0e-4, 3.0e-4),
        magnetic_field_rate_au=(0.0, 0.0, 2.0e-4),
    )
    stationary_config = replace(state.config, source=source_config)
    source = build_affine_electromagnetic_source(
        source_config,
        reference.electromagnetic_origin_au,
    ).sample(0.0)
    driven_state = replace(
        state,
        config=stationary_config,
        source_sample=source,
    )
    reference_path = tmp_path / "reference.h5"
    stationary_path = tmp_path / "stationary.h5"
    reference.save(reference_path)
    save_wilson_stationary_state(driven_state, stationary_path)
    config = replace(
        base_config,
        reference=replace(base_config.reference, path=reference_path),
        stationary_state=WilsonStationaryStateLinkConfig(
            driven_state.fingerprint_sha256,
            stationary_path,
        ),
        source=source_config,
        output=replace(base_config.output, directory=tmp_path / "run"),
    )
    simulation = build_simulation(config, reference, stationary_state=driven_state)
    trajectory = run(simulation)
    with h5py.File(trajectory.path, "r") as handle:
        group = handle["diagnostics/gauss_interval_work"]
        powers = np.asarray(group["gauss_power_au"][...])
        increments = np.asarray(group["increment_au"][...])
        accumulated = np.asarray(group["accumulated_au"][...])
    expected = 0.5 * config.propagation.time_grid.step_au * np.sum(powers, axis=1)
    np.testing.assert_allclose(increments, expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(accumulated, np.cumsum(increments), atol=2.0e-16, rtol=0.0)
    assert np.max(np.abs(powers)) > 1.0e-9
