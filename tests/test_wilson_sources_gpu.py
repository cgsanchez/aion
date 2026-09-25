from __future__ import annotations

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind
from aion.electromagnetism import (
    AffineMagneticGauge,
    GaussianVectorPotentialVariation,
    UniformMagneticField,
    UniformMagneticSourceSample,
)
from aion.electronic_structure import (
    AOGridPolicy,
    WilsonStationaryBranch,
    evaluate_exact_wilson_charge,
    evaluate_exact_wilson_one_electron_sample,
    evaluate_nonlinear_weak_current_pairing,
    prepare_ao_quadrature,
    prepare_exact_wilson_stationary_factory,
    prepare_pyscf_reference,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    ("branch", "functional"),
    (
        (WilsonStationaryBranch.KOHN_SHAM_LDA, "lda,vwn"),
        (WilsonStationaryBranch.KOHN_SHAM_GGA, "pbe"),
    ),
)
def test_nonlinear_weak_sources_cpu_gpu_parity_and_residency(
    branch: WilsonStationaryBranch,
    functional: str,
) -> None:
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
    gauge = AffineMagneticGauge(
        UniformMagneticField((0.0, 0.0, 0.017)),
        origin_au=(0.13, -0.19, 0.07),
    )
    source = UniformMagneticSourceSample(
        0.0,
        gauge.field,
        origin_au=gauge.origin_au,
    )
    cpu_model = prepare_exact_wilson_stationary_factory(
        cpu_quadrature,
        auxiliary_basis="weigend",
        functional=functional,
    ).model(gauge, branch)
    gpu_model = prepare_exact_wilson_stationary_factory(
        gpu_quadrature,
        auxiliary_basis="weigend",
        functional=functional,
    ).model(gauge, branch)
    cpu_sample = evaluate_exact_wilson_one_electron_sample(cpu_quadrature, source)
    gpu_sample = evaluate_exact_wilson_one_electron_sample(gpu_quadrature, source)
    coefficient = np.asarray(((0.71 + 0.19j,), (-0.23 + 0.41j,)))
    coefficient /= np.sqrt(
        (coefficient.conj().T @ np.asarray(cpu_sample.metric) @ coefficient).real.item()
    )
    density = 2.0 * coefficient @ coefficient.conj().T
    velocity = np.asarray(((0.13 + 0.29j, -0.17 + 0.07j), (0.11 - 0.19j, -0.23 + 0.31j)))
    variation = GaussianVectorPotentialVariation(
        amplitude_au=(0.19, -0.13, 0.07),
        center_au=(0.23, -0.17, 0.11),
        exponent_au_inverse2=0.41,
        path_quadrature_order=24,
    )
    cpu = evaluate_nonlinear_weak_current_pairing(
        cpu_model,
        cpu_sample,
        density,
        velocity,
        variation,
    )
    gpu = evaluate_nonlinear_weak_current_pairing(
        gpu_model,
        gpu_sample,
        density,
        velocity,
        variation,
    )
    cpu_charge = evaluate_exact_wilson_charge(cpu_model, density)
    gpu_charge = evaluate_exact_wilson_charge(gpu_model, density)

    for value in (
        gpu.response.metric,
        gpu.response.kinetic_embedding,
        gpu.response.kinetic_explicit,
        gpu.response.nuclear_attraction,
        gpu.hartree.source_energy_direction,
        gpu.exchange_correlation.source_energy_direction,
        gpu.total_pairing,
        gpu.ambient_minimal_pairing,
        gpu.embedding_pairing,
        gpu.tangential_pairing,
        gpu.normal_subspace_pairing,
        gpu_charge.signed_charge_density,
        gpu_charge.integrated_charge_grid,
    ):
        gpu_model.backend.assert_resident(value)

    for name in (
        "closure_pairing",
        "total_pairing",
        "ambient_minimal_pairing",
        "one_electron_embedding_pairing",
        "embedding_pairing",
        "tangential_pairing",
        "normal_subspace_pairing",
        "on_shell_pairing",
    ):
        np.testing.assert_allclose(
            gpu_model.backend.to_host(getattr(gpu, name)),
            getattr(cpu, name),
            atol=5.0e-10,
            rtol=5.0e-10,
        )
    np.testing.assert_allclose(
        gpu_model.backend.to_host(gpu_charge.signed_charge_density),
        cpu_charge.signed_charge_density,
        atol=5.0e-11,
        rtol=5.0e-11,
    )
