from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind, FixedTimeGrid, OutputConfig
from aion.electronic_structure import prepare_pyscf_reference
from aion.spectroscopy import KickSpectrumConfig, TransformConfig, kick_spectrum_from_trajectory
from aion.workflows import build_simulation, run
from test_reference_integration import molecular_config
from test_wp6_spectroscopy_integration import _kick_config

pytestmark = pytest.mark.gpu


def test_h2_kick_trajectory_and_spectrum_cpu_gpu_parity(tmp_path: Path) -> None:
    reference_path = tmp_path / "h2.reference.h5"
    reference = prepare_pyscf_reference(molecular_config("h2", reference_path))
    reference.save()
    base = _kick_config(reference, tmp_path / "cpu")
    propagation = replace(
        base.propagation,
        time_grid=FixedTimeGrid(start_au=0.0, step_au=0.1, intervals=128),
    )
    cpu_config = replace(base, propagation=propagation)
    gpu_config = replace(
        base,
        propagation=propagation,
        backend=BackendConfig(BackendKind.GPU, device_index=0),
        output=OutputConfig(tmp_path / "gpu", base.output.schedules),
    )
    cpu_simulation = build_simulation(cpu_config, reference)
    gpu_simulation = build_simulation(gpu_config, reference)
    cpu = run(cpu_simulation)
    gpu = run(gpu_simulation)
    gpu_simulation.workspace.assert_all_resident()

    for field in ("electronic_dipole", "primary_current"):
        cpu_id = cpu_simulation.calculators.definitions[field].definition_id
        gpu_id = gpu_simulation.calculators.definitions[field].definition_id
        cpu_series = cpu.read_observable(cpu_id)
        gpu_series = gpu.read_observable(gpu_id)
        assert np.array_equal(gpu_series.steps, cpu_series.steps)
        assert np.array_equal(gpu_series.times_au, cpu_series.times_au)
        assert np.allclose(gpu_series.values, cpu_series.values, rtol=5.0e-7, atol=5.0e-8)

    analysis = KickSpectrumConfig(
        TransformConfig(
            damping_energy_au=0.03,
            zero_padding_factor=4,
            maximum_energy_au=2.0,
        )
    )
    cpu_spectrum = kick_spectrum_from_trajectory(cpu, analysis)
    gpu_spectrum = kick_spectrum_from_trajectory(gpu, analysis)
    assert np.array_equal(gpu_spectrum.omega_au, cpu_spectrum.omega_au)
    assert np.allclose(
        gpu_spectrum.polarizability_dipole_au,
        cpu_spectrum.polarizability_dipole_au,
        rtol=5.0e-6,
        atol=5.0e-5,
    )
    assert np.allclose(
        gpu_spectrum.polarizability_current_au,
        cpu_spectrum.polarizability_current_au,
        rtol=5.0e-6,
        atol=5.0e-5,
    )
