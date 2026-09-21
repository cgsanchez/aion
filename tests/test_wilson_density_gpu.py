from __future__ import annotations

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind
from aion.electromagnetism import AffineMagneticGauge, UniformMagneticField
from aion.electronic_structure import (
    AOGridPolicy,
    evaluate_exact_uniform_magnetic_wilson_density,
    evaluate_exact_uniform_magnetic_wilson_density_matter_direction,
    evaluate_exact_uniform_magnetic_wilson_density_source_direction,
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
        gpu_result.overlap_stable,
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

    coefficients = reference.ground_state.coefficients.astype(np.complex128)
    occupations = reference.ground_state.occupations
    matter_direction = np.asarray(((0.17j, -0.08j), (0.11j, 0.19j)))
    matter_direction /= np.linalg.norm(matter_direction)
    cpu_matter = evaluate_exact_uniform_magnetic_wilson_density_matter_direction(
        cpu,
        coefficients,
        occupations,
        matter_direction,
        gauge,
    )
    gpu_matter = evaluate_exact_uniform_magnetic_wilson_density_matter_direction(
        gpu,
        coefficients,
        occupations,
        matter_direction,
        gauge,
    )
    source_direction = AffineMagneticGauge(
        UniformMagneticField((-0.4, 0.7, 0.2)),
        origin_au=gauge.origin_au,
    )
    cpu_source = evaluate_exact_uniform_magnetic_wilson_density_source_direction(
        cpu,
        density,
        gauge,
        source_direction,
    )
    gpu_source = evaluate_exact_uniform_magnetic_wilson_density_source_direction(
        gpu,
        density,
        gauge,
        source_direction,
    )
    for value in (gpu_matter.density_direction, gpu_source.density_direction):
        gpu.backend.assert_resident(value)
    np.testing.assert_allclose(
        gpu.backend.to_host(gpu_matter.density_direction),
        cpu_matter.density_direction,
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    np.testing.assert_allclose(
        gpu.backend.to_host(gpu_source.density_direction),
        cpu_source.density_direction,
        atol=2.0e-11,
        rtol=2.0e-11,
    )
