from __future__ import annotations

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind, FormulationKind, GaugeRepresentation
from aion.formulations import SourceSampling
from aion.workflows import build_simulation
from test_reference_integration import molecular_config
from test_wp3_formulation_integration import _simulation_config

pytestmark = pytest.mark.gpu


def _host(simulation: object, value: object) -> np.ndarray:
    return simulation.workspace.backend.to_host(value)


def test_wp4_h2_lih_cpu_gpu_trajectory_parity_residency_and_no_bulk_transfer() -> None:
    from aion.electronic_structure import prepare_pyscf_reference

    for name in ("h2", "lih"):
        reference = prepare_pyscf_reference(molecular_config(name))
        for kind, gauge, fraction in (
            (FormulationKind.BARE_VELOCITY_GAUGE, None, None),
            (FormulationKind.P0_E1, GaugeRepresentation.VELOCITY, None),
            (FormulationKind.P0_E1, GaugeRepresentation.MIXED, 0.375),
        ):
            cpu = build_simulation(
                _simulation_config(
                    reference,
                    kind,
                    gauge=gauge,
                    velocity_fraction=fraction,
                    backend=BackendConfig(),
                ),
                reference,
            )
            gpu = build_simulation(
                _simulation_config(
                    reference,
                    kind,
                    gauge=gauge,
                    velocity_fraction=fraction,
                    backend=BackendConfig(BackendKind.GPU, device_index=0),
                ),
                reference,
            )
            scratch_ids = {key: id(value) for key, value in gpu.workspace.scratch.items()}
            for _ in range(3):
                cpu_result = cpu.step()
                backend = gpu.workspace.backend
                original_to_host = backend.to_host

                def forbidden_bulk_transfer(value: object) -> np.ndarray:
                    del value
                    raise AssertionError("SCEM hot loop attempted a bulk GPU-to-host transfer")

                backend.to_host = forbidden_bulk_transfer  # type: ignore[method-assign]
                try:
                    gpu_result = gpu.step()
                finally:
                    backend.to_host = original_to_host  # type: ignore[method-assign]

                backend.assert_resident(gpu_result.state.coefficients, name="GPU coefficients")
                backend.assert_resident(
                    gpu_result.midpoint_evaluation.density.matrix,
                    name="GPU midpoint density",
                )
                for key, value in gpu.workspace.scratch.items():
                    backend.assert_resident(value, name=key)
                assert {key: id(value) for key, value in gpu.workspace.scratch.items()} == (
                    scratch_ids
                )
                assert np.allclose(
                    _host(gpu, gpu_result.state.density().matrix),
                    _host(cpu, cpu_result.state.density().matrix),
                    rtol=5.0e-7,
                    atol=5.0e-8,
                )
                assert np.allclose(
                    _host(gpu, gpu_result.midpoint_evaluation.density.matrix),
                    _host(cpu, cpu_result.midpoint_evaluation.density.matrix),
                    rtol=5.0e-7,
                    atol=5.0e-8,
                )
                assert gpu_result.diagnostics.density_residual == pytest.approx(
                    cpu_result.diagnostics.density_residual,
                    rel=2.0e-3,
                    abs=2.0e-11,
                )
                assert gpu_result.diagnostics.final_metric_residual < 2.0e-11

            source_cpu = cpu.source_sample(SourceSampling.ENDPOINT, cpu.state.step_index)
            source_gpu = gpu.source_sample(SourceSampling.ENDPOINT, gpu.state.step_index)
            evaluation_cpu = cpu.formulation.evaluate(cpu.state.density(), source_cpu)
            evaluation_gpu = gpu.formulation.evaluate(gpu.state.density(), source_gpu)
            current_cpu = cpu.formulation.currents(evaluation_cpu, source_cpu)
            current_gpu = gpu.formulation.currents(evaluation_gpu, source_gpu)
            energy_cpu = cpu.formulation.energy(evaluation_cpu, source_cpu)
            energy_gpu = gpu.formulation.energy(evaluation_gpu, source_gpu)
            assert np.allclose(
                _host(gpu, current_gpu.primary_current),
                _host(cpu, current_cpu.primary_current),
                rtol=5.0e-7,
                atol=5.0e-8,
            )
            assert float(_host(gpu, energy_gpu.energy_matter_total)) == pytest.approx(
                float(_host(cpu, energy_cpu.energy_matter_total)),
                rel=5.0e-8,
                abs=5.0e-8,
            )
