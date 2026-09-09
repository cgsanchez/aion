from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind, FormulationKind, GaugeRepresentation
from aion.errors import RunCancelledError
from aion.io.checkpoint import load_checkpoint
from aion.io.trajectory import load_trajectory
from aion.workflows import RunControl, build_simulation, resume, run
from test_reference_integration import molecular_config
from test_wp5_runner_integration import _config

pytestmark = pytest.mark.gpu


def _density(checkpoint: object) -> np.ndarray:
    return checkpoint.density


def test_wp5_cpu_gpu_output_event_and_restart_parity(tmp_path: Path) -> None:
    from aion.electronic_structure import prepare_pyscf_reference

    references = {}
    for name in ("h2", "lih"):
        path = tmp_path / f"{name}.reference.h5"
        reference = prepare_pyscf_reference(molecular_config(name, path))
        reference.save()
        references[name] = reference

    gpu_config = BackendConfig(BackendKind.GPU, device_index=0)
    gpu_full = None
    gpu_full_checkpoint = None
    for name, kind in (
        ("h2", FormulationKind.BARE_VELOCITY_GAUGE),
        ("lih", FormulationKind.P0_E1),
    ):
        reference = references[name]
        gauge = (
            None if kind is FormulationKind.BARE_VELOCITY_GAUGE else GaugeRepresentation.VELOCITY
        )
        cpu_simulation = build_simulation(
            _config(reference, tmp_path / f"cpu-{kind.value}", kind, gauge),
            reference,
        )
        gpu_simulation = build_simulation(
            _config(
                reference,
                tmp_path / f"gpu-{kind.value}",
                kind,
                gauge,
                backend=gpu_config,
            ),
            reference,
        )
        cpu = run(cpu_simulation)
        gpu = run(gpu_simulation)
        gpu_simulation.workspace.assert_all_resident()

        cpu_final = load_checkpoint(cpu.path.parent / "checkpoint_00000004.h5")
        gpu_final = load_checkpoint(gpu.path.parent / "checkpoint_00000004.h5")
        assert np.allclose(_density(gpu_final), _density(cpu_final), rtol=5.0e-7, atol=5.0e-8)
        assert gpu.accumulated_source_work_au == pytest.approx(
            cpu.accumulated_source_work_au,
            rel=5.0e-7,
            abs=5.0e-8,
        )
        assert gpu.event_steps == cpu.event_steps == (2,)
        for field in ("electronic_dipole", "primary_current", "energy_matter_total"):
            cpu_id = cpu_simulation.calculators.definitions[field].definition_id
            gpu_id = gpu_simulation.calculators.definitions[field].definition_id
            cpu_series = cpu.read_observable(cpu_id)
            gpu_series = gpu.read_observable(gpu_id)
            assert np.array_equal(gpu_series.steps, cpu_series.steps)
            assert np.allclose(gpu_series.values, cpu_series.values, rtol=5.0e-7, atol=5.0e-8)
        with h5py.File(gpu.path, "r") as handle:
            provenance = json.loads(handle["meta/provenance_json"][()].decode())
            assert provenance["backend"]["kind"] == "gpu"
            assert provenance["backend"]["device_index"] == 0
            assert provenance["backend"]["gpu_name"]

        if kind is FormulationKind.P0_E1:
            gpu_full = gpu
            gpu_full_checkpoint = gpu_final

    assert gpu_full is not None and gpu_full_checkpoint is not None
    reference = references["lih"]
    interrupted_config = _config(
        reference,
        tmp_path / "gpu-interrupted",
        FormulationKind.P0_E1,
        GaugeRepresentation.VELOCITY,
        backend=gpu_config,
    )
    interrupted_simulation = build_simulation(interrupted_config, reference)
    control = RunControl()

    def cancel_at_boundary(step: int) -> None:
        if step == 2:
            control.request_cancel()

    control.after_accepted_step = cancel_at_boundary
    with pytest.raises(RunCancelledError):
        run(interrupted_simulation, control=control)
    parent = load_trajectory(
        interrupted_config.output.directory / "trajectory.failed.h5",
        allow_incomplete=True,
    )
    child = resume(
        interrupted_config.output.directory / "checkpoint_00000002.h5",
        output=tmp_path / "gpu-resumed",
    )
    resumed_final = load_checkpoint(tmp_path / "gpu-resumed/checkpoint_00000004.h5")
    assert child.parent_run_id == parent.run_id
    assert child.parent_checkpoint_sha256 in parent.checkpoint_sha256
    assert child.event_steps == (2,)
    assert np.allclose(
        _density(resumed_final),
        _density(gpu_full_checkpoint),
        rtol=5.0e-7,
        atol=5.0e-8,
    )
    assert child.accumulated_source_work_au == pytest.approx(
        gpu_full.accumulated_source_work_au,
        rel=5.0e-7,
        abs=5.0e-8,
    )
