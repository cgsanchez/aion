from __future__ import annotations

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind
from aion.electromagnetism import AffineMagneticGauge, UniformMagneticField
from aion.electronic_structure import (
    AOGridPolicy,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
    prepare_wilson_lda,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.gpu


def test_wilson_lda_cpu_gpu_action_and_source_parity() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    policy = AOGridPolicy.qualification(1)
    cpu_quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=policy,
        block_size=257,
    )
    gpu_quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(BackendKind.GPU, device_index=0),
        grid_policy=policy,
        block_size=257,
    )
    cpu = prepare_wilson_lda(cpu_quadrature, "lda,vwn")
    gpu = prepare_wilson_lda(gpu_quadrature, "lda,vwn")
    density = reference.ground_state.density.astype(np.complex128)
    gauge = AffineMagneticGauge(
        UniformMagneticField((0.013, -0.009, 0.017)),
        origin_au=(0.17, -0.31, 0.23),
    )
    direction = AffineMagneticGauge(
        UniformMagneticField((-0.4, 0.7, 0.2)),
        origin_au=gauge.origin_au,
    )
    cpu_result = cpu.evaluate(density, gauge, source_direction=direction)
    gpu_result = gpu.evaluate(density, gauge, source_direction=direction)

    for value in (
        gpu_result.energy,
        gpu_result.lower_xc_matrix,
        gpu_result.density,
        gpu_result.energy_per_particle,
        gpu_result.density_derivative,
        gpu_result.source_energy_direction,
        gpu_result.source_density_direction,
        gpu_result.electron_count_grid,
    ):
        gpu.backend.assert_resident(value)
    np.testing.assert_allclose(
        gpu.backend.to_host(gpu_result.energy),
        cpu_result.energy,
        atol=3.0e-11,
        rtol=3.0e-11,
    )
    np.testing.assert_allclose(
        gpu.backend.to_host(gpu_result.lower_xc_matrix),
        cpu_result.lower_xc_matrix,
        atol=3.0e-11,
        rtol=3.0e-11,
    )
    np.testing.assert_allclose(
        gpu.backend.to_host(gpu_result.source_energy_direction),
        cpu_result.source_energy_direction,
        atol=3.0e-11,
        rtol=3.0e-11,
    )
    np.testing.assert_allclose(
        gpu.backend.to_host(gpu_result.source_density_direction),
        cpu_result.source_density_direction,
        atol=3.0e-11,
        rtol=3.0e-11,
    )
