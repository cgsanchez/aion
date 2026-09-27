from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind, WilsonInitialSourcePolicy
from aion.workflows import BuiltWilsonSimulation, build_simulation
from test_wilson_workflow import prepare_exact_wilson_inputs

pytestmark = pytest.mark.gpu


def test_exact_wilson_runtime_observables_and_step_remain_on_physical_gpu() -> None:
    cpu_config, reference, state = prepare_exact_wilson_inputs()
    cpu = build_simulation(cpu_config, reference, stationary_state=state)
    gpu_config = replace(
        cpu_config,
        backend=BackendConfig(kind=BackendKind.GPU, device_index=0),
    )
    gpu = build_simulation(gpu_config, reference, stationary_state=state)
    assert isinstance(gpu, BuiltWilsonSimulation)
    gpu.quadrature.backend.assert_resident(gpu.density, name="GPU Wilson density")

    cpu_observation = cpu.observe_endpoint(include_energy=True)
    gpu_observation = gpu.observe_endpoint(include_energy=True)
    assert cpu_observation.energy is not None
    assert gpu_observation.energy is not None
    cpu_energy_only = cpu.observe_energy()
    gpu_energy_only = gpu.observe_energy()
    gpu.quadrature.backend.assert_resident(
        gpu_energy_only.molecular_total_au,
        name="GPU Wilson energy-only observation",
    )
    np.testing.assert_array_equal(
        gpu.quadrature.backend.to_host(gpu_energy_only.molecular_total_au),
        gpu.quadrature.backend.to_host(gpu_observation.energy.molecular_total_au),
    )
    np.testing.assert_array_equal(
        cpu_energy_only.molecular_total_au,
        cpu_observation.energy.molecular_total_au,
    )
    gpu.quadrature.backend.assert_resident(
        gpu_observation.uniform_source_current_au,
        name="GPU Wilson source current",
    )
    gpu.quadrature.backend.assert_resident(
        gpu_observation.electronic_dipole_au,
        name="GPU Wilson dipole",
    )
    np.testing.assert_allclose(
        gpu.quadrature.backend.to_host(gpu_observation.uniform_source_current_au),
        cpu_observation.uniform_source_current_au,
        atol=2.0e-9,
        rtol=2.0e-9,
    )
    np.testing.assert_allclose(
        gpu.quadrature.backend.to_host(gpu_observation.electronic_dipole_au),
        cpu_observation.electronic_dipole_au,
        atol=2.0e-9,
        rtol=2.0e-9,
    )
    gpu.step()
    gpu.quadrature.backend.assert_resident(gpu.density, name="stepped GPU Wilson density")
    np.testing.assert_allclose(
        gpu.quadrature.backend.to_host(gpu.density),
        cpu.step().contravariant_density,
        atol=2.0e-9,
        rtol=2.0e-9,
    )


def test_continuous_density_source_quench_has_cpu_gpu_step_parity() -> None:
    baseline, reference, state = prepare_exact_wilson_inputs()
    quenched = replace(
        baseline,
        source=replace(
            baseline.source,
            electric_field_origin_offset_au=(2.0e-4, -1.0e-4, 3.0e-4),
            magnetic_field_rate_au=(0.0, 0.0, 1.0e-5),
        ),
        initial_source_policy=WilsonInitialSourcePolicy.CONTINUOUS_DENSITY_QUENCH,
    )
    cpu = build_simulation(quenched, reference, stationary_state=state)
    gpu = build_simulation(
        replace(
            quenched,
            backend=BackendConfig(kind=BackendKind.GPU, device_index=0),
        ),
        reference,
        stationary_state=state,
    )
    assert isinstance(cpu, BuiltWilsonSimulation)
    assert isinstance(gpu, BuiltWilsonSimulation)
    gpu.quadrature.backend.assert_resident(gpu.density, name="quenched GPU Wilson density")
    np.testing.assert_allclose(
        gpu.quadrature.backend.to_host(gpu.step().contravariant_density),
        cpu.step().contravariant_density,
        atol=2.0e-9,
        rtol=2.0e-9,
    )
