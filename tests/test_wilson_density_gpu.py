from __future__ import annotations

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind
from aion.electromagnetism import AffineMagneticGauge, UniformMagneticField
from aion.electronic_structure import (
    AOGridPolicy,
    evaluate_exact_uniform_magnetic_wilson_density,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.gpu


def test_exact_wilson_density_cpu_gpu_parity_without_host_fallback() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    cpu = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(2),
        block_size=257,
    )
    gpu = prepare_ao_quadrature(
        reference,
        BackendConfig(BackendKind.GPU, device_index=0),
        grid_policy=AOGridPolicy.qualification(2),
        block_size=257,
    )
    gauge = AffineMagneticGauge(
        UniformMagneticField((0.013, -0.009, 0.017)),
        origin_au=(0.17, -0.31, 0.23),
    )
    density = reference.ground_state.density.astype(np.complex128)
    cpu_result = evaluate_exact_uniform_magnetic_wilson_density(cpu, density, gauge)
    gpu_result = evaluate_exact_uniform_magnetic_wilson_density(gpu, density, gauge)

    for value in (
        gpu_result.density_direct,
        gpu_result.density_factorized,
        gpu_result.overlap_direct_grid,
        gpu_result.overlap_factorized_grid,
    ):
        gpu.backend.assert_resident(value)
    np.testing.assert_allclose(
        gpu.backend.to_host(gpu_result.density_direct),
        cpu_result.density_direct,
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    np.testing.assert_allclose(
        gpu.backend.to_host(gpu_result.overlap_factorized_grid),
        cpu_result.overlap_factorized_grid,
        atol=2.0e-11,
        rtol=2.0e-11,
    )
