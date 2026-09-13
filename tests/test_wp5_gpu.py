from __future__ import annotations

import json
from collections.abc import Callable
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


def _cancel_at(control: RunControl, target_step: int) -> Callable[[int], None]:
    def callback(step: int) -> None:
        if step == target_step:
            control.request_cancel()

    return callback


def test_wp5_cpu_gpu_output_event_and_restart_parity(tmp_path: Path) -> None:
    from aion.electronic_structure import prepare_pyscf_reference

    references = {}
    for name in ("h2", "lih"):
        path = tmp_path / f"{name}.reference.h5"
        reference = prepare_pyscf_reference(molecular_config(name, path))
        reference.save()
        references[name] = reference

    gpu_config = BackendConfig(BackendKind.GPU, device_index=0)
    for name, kind, gauge, fraction in (
        ("h2", FormulationKind.BARE_LENGTH_GAUGE, None, None),
        ("h2", FormulationKind.BARE_VELOCITY_GAUGE, None, None),
        ("lih", FormulationKind.P0_E1, GaugeRepresentation.LENGTH, None),
        ("lih", FormulationKind.P0_E1, GaugeRepresentation.MIXED, 0.375),
        ("lih", FormulationKind.P0_E1, GaugeRepresentation.VELOCITY, None),
    ):
        reference = references[name]
        suffix = kind.value if gauge is None else f"{kind.value}-{gauge.value}"
        cpu_simulation = build_simulation(
            _config(
                reference,
                tmp_path / f"cpu-{suffix}",
                kind,
                gauge,
                velocity_fraction=fraction,
            ),
            reference,
        )
        gpu_simulation = build_simulation(
            _config(
                reference,
                tmp_path / f"gpu-{suffix}",
                kind,
                gauge,
                backend=gpu_config,
                velocity_fraction=fraction,
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

        interrupted_config = _config(
            reference,
            tmp_path / f"gpu-interrupted-{suffix}",
            kind,
            gauge,
            backend=gpu_config,
            velocity_fraction=fraction,
        )
        interrupted_simulation = build_simulation(interrupted_config, reference)
        control = RunControl()
        control.after_accepted_step = _cancel_at(control, 2)
        with pytest.raises(RunCancelledError):
            run(interrupted_simulation, control=control)
        parent = load_trajectory(
            interrupted_config.output.directory / "trajectory.failed.h5",
            allow_incomplete=True,
        )
        resumed_directory = tmp_path / f"gpu-resumed-{suffix}"
        child = resume(
            interrupted_config.output.directory / "checkpoint_00000002.h5",
            output=resumed_directory,
        )
        resumed_final = load_checkpoint(resumed_directory / "checkpoint_00000004.h5")
        assert child.parent_run_id == parent.run_id
        assert child.parent_checkpoint_sha256 in parent.checkpoint_sha256
        assert child.event_steps == (2,)
        assert np.allclose(
            _density(resumed_final),
            _density(gpu_final),
            rtol=5.0e-7,
            atol=5.0e-8,
        )
        assert child.accumulated_source_work_au == pytest.approx(
            gpu.accumulated_source_work_au,
            rel=5.0e-7,
            abs=5.0e-8,
        )
