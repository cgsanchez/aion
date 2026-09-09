from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind, Sin2VectorPotentialPulseConfig
from aion.electromagnetism import (
    Sin2VectorPotentialPulse,
    compile_uniform_source,
    pulse_aligned_time_grid,
)
from aion.errors import BackendError, DeviceResidencyError
from test_reference_integration import molecular_config

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("name", ("h2", "lih"))
def test_h2_lih_reference_cpu_gpu_workspace_and_rks_parity(name: str) -> None:
    import cupy as cp

    from aion.electronic_structure import prepare_pyscf_reference

    reference = prepare_pyscf_reference(molecular_config(name))
    gpu_config = BackendConfig(BackendKind.GPU, device_index=0)
    gpu_reference = prepare_pyscf_reference(replace(reference.config, backend=gpu_config))
    assert gpu_reference.preparation_backend == "gpu:0"
    assert gpu_reference.ground_state.energy_total_au == pytest.approx(
        reference.ground_state.energy_total_au, rel=2.0e-10, abs=2.0e-10
    )
    assert np.allclose(
        gpu_reference.ground_state.density,
        reference.ground_state.density,
        rtol=2.0e-7,
        atol=2.0e-7,
    )
    assert np.array_equal(
        gpu_reference.core_operators.canonical_momentum,
        reference.core_operators.canonical_momentum,
    )
    assert np.array_equal(gpu_reference.grid.coordinates_au, reference.grid.coordinates_au)
    cpu = reference.create_workspace(BackendConfig())
    gpu = reference.create_workspace(gpu_config)
    gpu.assert_all_resident()
    for key, host in cpu.arrays.items():
        device = gpu.require(key)
        assert isinstance(device, cp.ndarray)
        assert int(device.device.id) == 0
        assert np.array_equal(cp.asnumpy(device), host)
    dm_cpu = cpu.require("ground_state.density")
    dm_gpu = gpu.require("ground_state.density")
    veff_cpu = np.asarray(cpu.electronic_model.get_veff(cpu.electronic_model.mol, dm_cpu))
    veff_gpu = gpu.electronic_model.get_veff(gpu.electronic_model.mol, dm_gpu)
    gpu.backend.assert_resident(veff_gpu, name="ground-state effective potential")
    assert np.allclose(cp.asnumpy(veff_gpu), veff_cpu, rtol=2.0e-9, atol=2.0e-9)


def test_density_fitted_reference_reconstructs_on_the_physical_gpu() -> None:
    import cupy as cp

    from aion.electronic_structure import prepare_pyscf_reference

    base = molecular_config("h2")
    reference = prepare_pyscf_reference(
        replace(
            base,
            electronic_structure=replace(
                base.electronic_structure,
                density_fitting=True,
                auxiliary_basis="weigend",
            ),
        )
    )
    cpu = reference.create_workspace(BackendConfig())
    gpu = reference.create_workspace(BackendConfig(BackendKind.GPU, device_index=0))
    density_cpu = cpu.require("ground_state.density")
    density_gpu = gpu.require("ground_state.density")
    veff_cpu = np.asarray(cpu.electronic_model.get_veff(cpu.electronic_model.mol, density_cpu))
    veff_gpu = gpu.electronic_model.get_veff(gpu.electronic_model.mol, density_gpu)
    gpu.backend.assert_resident(veff_gpu, name="density-fitted effective potential")
    assert gpu.electronic_model.with_df.auxbasis == "weigend"
    assert np.allclose(cp.asnumpy(veff_gpu), veff_cpu, rtol=2.0e-8, atol=2.0e-8)


def test_compiled_source_gpu_residency_and_no_fallback() -> None:
    import cupy as cp

    from aion.backends import CuPyBackend, Workspace

    reference = __import__(
        "aion.electronic_structure", fromlist=["prepare_pyscf_reference"]
    ).prepare_pyscf_reference(molecular_config("h2"))
    pulse = Sin2VectorPotentialPulse(Sin2VectorPotentialPulseConfig(0.02, 0.5, 1, (0.2, -0.1, 1.0)))
    source = compile_uniform_source(
        pulse,
        pulse_aligned_time_grid(pulse, 0.1),
        reference.core_operators.nuclei.coordinates_au,
        reference.config.molecule.electromagnetic_origin.position_au,
        reference.anchor_topology.pair_indices,
    )
    workspace = Workspace(CuPyBackend(0))
    source.install(workspace)
    workspace.assert_all_resident()
    assert all(isinstance(value, cp.ndarray) for value in workspace.arrays.values())
    assert np.array_equal(
        cp.asnumpy(workspace.require("source.midpoint.electric_field")),
        source.midpoint.electric_field,
    )
    with pytest.raises(DeviceResidencyError, match="not a CuPy"):
        workspace.backend.assert_resident(np.zeros(1), name="host leak")
    with pytest.raises(BackendError, match="CUDA reports"):
        CuPyBackend(cp.cuda.runtime.getDeviceCount())
