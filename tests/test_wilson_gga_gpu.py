from __future__ import annotations

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind
from aion.electromagnetism import AffineMagneticGauge, UniformMagneticField
from aion.electronic_structure import (
    AOGridPolicy,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
    prepare_wilson_gga,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.gpu


def test_wilson_gga_cpu_gpu_action_and_source_parity() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    policy = AOGridPolicy.qualification(1)
    cpu_quadrature = prepare_ao_quadrature(
        reference, BackendConfig(), grid_policy=policy, block_size=257
    )
    gpu_quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(BackendKind.GPU, device_index=0),
        grid_policy=policy,
        block_size=257,
    )
    cpu = prepare_wilson_gga(cpu_quadrature, "pbe")
    gpu = prepare_wilson_gga(gpu_quadrature, "pbe")
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
        gpu_result.density_gradient,
        gpu_result.density_derivative,
        gpu_result.density_gradient_derivative,
        gpu_result.source_energy_direction,
        gpu_result.source_density_direction,
        gpu_result.source_density_gradient_direction,
        gpu_result.electron_count_grid,
    ):
        gpu.backend.assert_resident(value)
    for name in (
        "energy",
        "lower_xc_matrix",
        "density",
        "density_gradient",
        "density_derivative",
        "density_gradient_derivative",
        "source_energy_direction",
        "source_density_direction",
        "source_density_gradient_direction",
        "electron_count_grid",
    ):
        np.testing.assert_allclose(
            gpu.backend.to_host(getattr(gpu_result, name)),
            getattr(cpu_result, name),
            atol=8.0e-10,
            rtol=8.0e-10,
        )
