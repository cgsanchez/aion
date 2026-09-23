from __future__ import annotations

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind
from aion.electromagnetism import UniformMagneticField, UniformMagneticSourceSample
from aion.electronic_structure import (
    AOGridPolicy,
    ReducedWilsonLevel,
    WilsonStationaryBranch,
    prepare_ao_quadrature,
    prepare_exact_wilson_stationary_factory,
    prepare_pyscf_reference,
    prepare_reduced_wilson_factory,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.gpu


def test_reduced_actions_source_responses_and_power_cpu_gpu_parity() -> None:
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
    cpu_factory = prepare_reduced_wilson_factory(
        prepare_exact_wilson_stationary_factory(
            cpu_quadrature,
            auxiliary_basis="weigend",
        )
    )
    gpu_factory = prepare_reduced_wilson_factory(
        prepare_exact_wilson_stationary_factory(
            gpu_quadrature,
            auxiliary_basis="weigend",
        )
    )
    source = UniformMagneticSourceSample(
        time_au=0.31,
        field=UniformMagneticField((0.006, -0.004, 0.003)),
        magnetic_field_dot_au=(0.002, 0.001, -0.003),
        electric_field_origin_au=(0.001, -0.002, 0.004),
        origin_au=(0.13, -0.19, 0.07),
    )
    nao = reference.core_operators.nao
    density = np.asarray(reference.ground_state.density) + 0.2 * np.eye(nao)

    for level in ReducedWilsonLevel:
        cpu = cpu_factory.spatial_action(source.gauge, level).sample(
            source,
            WilsonStationaryBranch.KOHN_SHAM_LDA,
        )
        gpu = gpu_factory.spatial_action(source.gauge, level).sample(
            source,
            WilsonStationaryBranch.KOHN_SHAM_LDA,
        )
        cpu_action = cpu.evaluate(density)
        gpu_action = gpu.evaluate(density)
        cpu_response = cpu.source_response(density)
        gpu_response = gpu.source_response(density)
        cpu_power = cpu.power(density)
        gpu_power = gpu.power(density)

        for value in (
            gpu_action.overlap,
            gpu_action.lower_mechanical_matrix,
            gpu_action.closure.density.density_zero,
            gpu_action.closure.density.density_first,
            gpu_action.closure.hartree.lower_matrix,
            gpu_action.closure.exchange_correlation.lower_matrix,
            gpu_response.one_electron_matrix_rate,
            gpu_response.density_zero_rate,
            gpu_response.density_first_rate,
            gpu_response.mechanical_energy_rate_au,
            gpu_power.velocity_density,
            gpu_power.source_power_au,
        ):
            gpu.backend.assert_resident(value)

        for gpu_value, cpu_value in (
            (gpu_action.overlap, cpu_action.overlap),
            (gpu_action.lower_mechanical_matrix, cpu_action.lower_mechanical_matrix),
            (gpu_action.energy_electronic_au, cpu_action.energy_electronic_au),
            (gpu_response.one_electron_matrix_rate, cpu_response.one_electron_matrix_rate),
            (gpu_response.density_zero_rate, cpu_response.density_zero_rate),
            (gpu_response.density_first_rate, cpu_response.density_first_rate),
            (gpu_response.hartree_energy_rate_au, cpu_response.hartree_energy_rate_au),
            (
                gpu_response.exchange_correlation_energy_rate_au,
                cpu_response.exchange_correlation_energy_rate_au,
            ),
            (gpu_power.velocity_density, cpu_power.velocity_density),
            (gpu_power.source_power_au, cpu_power.source_power_au),
        ):
            np.testing.assert_allclose(
                gpu.backend.to_host(gpu_value),
                cpu_value,
                atol=8.0e-10,
                rtol=8.0e-10,
            )
