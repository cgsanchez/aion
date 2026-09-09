from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from aion.config import (
    BackendConfig,
    BackendKind,
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
from aion.formulations import AODensity, SourceSampling
from aion.workflows import build_simulation
from test_reference_integration import molecular_config

pytestmark = pytest.mark.gpu


def _config(reference: object, backend: BackendConfig, kind: FormulationKind) -> SimulationConfig:
    pulse = Sin2VectorPotentialPulseConfig(
        peak_electric_field_au=0.006,
        angular_frequency_au=0.57,
        cycles=1,
        polarization=(0.1, -0.2, 1.0),
        carrier_phase_rad=0.23,
    )
    grid = pulse_aligned_time_grid(Sin2VectorPotentialPulse(pulse), 0.8)
    integrator = (
        IntegratorKind.FIXED_METRIC_SCEM
        if kind is FormulationKind.BARE_VELOCITY_GAUGE
        else IntegratorKind.CONNECTION_AWARE_SCEM
    )
    gauge = None if kind is FormulationKind.BARE_VELOCITY_GAUGE else GaugeRepresentation.VELOCITY
    return SimulationConfig(
        reference=ReferenceLinkConfig(reference.fingerprint_sha256),
        formulation=FormulationConfig(kind, gauge),
        source=pulse,
        propagation=PropagationConfig(grid, integrator),
        backend=backend,
    )


def _assert_ledger_parity(cpu: object, gpu: object, left: object, right: object) -> None:
    for item in fields(left):
        cpu_value = getattr(left, item.name)
        gpu_value = getattr(right, item.name)
        if cpu_value is None:
            assert gpu_value is None
            continue
        gpu.workspace.backend.assert_resident(gpu_value, name=item.name)
        actual = gpu.workspace.backend.to_host(gpu_value)
        expected = cpu.workspace.backend.to_host(cpu_value)
        assert np.allclose(actual, expected, rtol=3.0e-7, atol=3.0e-8), item.name


def test_wp3_lih_cpu_gpu_formulation_observable_and_invariant_parity() -> None:
    from aion.electronic_structure import prepare_pyscf_reference

    reference = prepare_pyscf_reference(molecular_config("lih"))
    cpu_backend = BackendConfig()
    gpu_backend = BackendConfig(BackendKind.GPU, device_index=0)
    for kind in (FormulationKind.BARE_VELOCITY_GAUGE, FormulationKind.P0_E1):
        cpu = build_simulation(_config(reference, cpu_backend, kind), reference)
        gpu = build_simulation(_config(reference, gpu_backend, kind), reference)
        source_cpu = cpu.source_sample(SourceSampling.MIDPOINT, 2)
        source_gpu = gpu.source_sample(SourceSampling.MIDPOINT, 2)
        density_host = np.asarray(cpu.density.matrix, dtype=np.complex128).copy()
        density_host[0, -1] += 0.018j
        density_host[-1, 0] -= 0.018j
        density_cpu = AODensity.from_matrix(density_host, cpu.workspace.backend)
        density_gpu = AODensity.from_matrix(density_host, gpu.workspace.backend)
        if kind is FormulationKind.P0_E1:
            trial_cpu = cpu.formulation.evaluate(density_cpu, source_cpu)
            trial_gpu = gpu.formulation.evaluate(density_gpu, source_gpu)
            assert trial_cpu.theta is not None and trial_gpu.theta is not None
            density_cpu = AODensity.from_matrix(
                trial_cpu.theta * density_cpu.matrix, cpu.workspace.backend
            )
            density_gpu = AODensity.from_matrix(
                trial_gpu.theta * density_gpu.matrix, gpu.workspace.backend
            )
        evaluation_cpu = cpu.formulation.evaluate(density_cpu, source_cpu)
        evaluation_gpu = gpu.formulation.evaluate(density_gpu, source_gpu)
        current_cpu = cpu.formulation.currents(evaluation_cpu, source_cpu)
        current_gpu = gpu.formulation.currents(evaluation_gpu, source_gpu)
        energy_cpu = cpu.formulation.energy(evaluation_cpu, source_cpu)
        energy_gpu = gpu.formulation.energy(evaluation_gpu, source_gpu)

        for name in (
            "metric",
            "hamiltonian_eom",
            "connection",
        ):
            gpu_value = getattr(evaluation_gpu.triple, name)
            gpu.workspace.backend.assert_resident(gpu_value, name=name)
            assert np.allclose(
                gpu.workspace.backend.to_host(gpu_value),
                cpu.workspace.backend.to_host(getattr(evaluation_cpu.triple, name)),
                rtol=3.0e-7,
                atol=3.0e-8,
            )
        assert np.allclose(
            gpu.workspace.backend.to_host(evaluation_gpu.density_dot),
            cpu.workspace.backend.to_host(evaluation_cpu.density_dot),
            rtol=3.0e-7,
            atol=3.0e-8,
        )
        _assert_ledger_parity(cpu, gpu, current_cpu, current_gpu)
        _assert_ledger_parity(cpu, gpu, energy_cpu, energy_gpu)
        assert abs(float(gpu.workspace.backend.to_host(energy_gpu.energy_ward_residual))) < 2.0e-7
        if current_gpu.continuity_residual is not None:
            assert (
                np.linalg.norm(gpu.workspace.backend.to_host(current_gpu.continuity_residual))
                < 2.0e-7
            )
