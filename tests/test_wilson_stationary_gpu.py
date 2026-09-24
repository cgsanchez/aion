from __future__ import annotations

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind
from aion.electromagnetism import AffineMagneticGauge, UniformMagneticField
from aion.electronic_structure import (
    AOGridPolicy,
    WilsonStationaryBranch,
    prepare_ao_quadrature,
    prepare_exact_wilson_stationary_model,
    prepare_pyscf_reference,
)
from aion.errors import UnsupportedConfigurationError
from test_reference_integration import molecular_config

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    ("branch", "functional"),
    (
        (WilsonStationaryBranch.KOHN_SHAM_LDA, "lda,vwn"),
        (WilsonStationaryBranch.KOHN_SHAM_GGA, "pbe"),
    ),
)
def test_complete_stationary_action_cpu_gpu_parity_and_residency(
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
        UniformMagneticField((0.013, -0.009, 0.017)),
        origin_au=(0.17, -0.31, 0.23),
    )
    cpu = prepare_exact_wilson_stationary_model(
        cpu_quadrature,
        gauge,
        branch,
        auxiliary_basis="weigend",
        functional=functional,
    )
    gpu = prepare_exact_wilson_stationary_model(
        gpu_quadrature,
        gauge,
        branch,
        auxiliary_basis="weigend",
        functional=functional,
    )
    density = reference.ground_state.density.astype(np.complex128)
    cpu_result = cpu.evaluate(density)
    gpu_result = gpu.evaluate(density)

    for value in (
        gpu_result.coefficient_density,
        gpu_result.overlap,
        gpu_result.kinetic_matrix,
        gpu_result.nuclear_attraction_matrix,
        gpu_result.one_electron_matrix,
        gpu_result.lower_mechanical_matrix,
        gpu_result.hartree.energy,
        gpu_result.hartree.lower_coulomb_matrix,
        gpu_result.exchange_correlation.energy,
        gpu_result.exchange_correlation.lower_xc_matrix,
        gpu_result.energy_electronic_au,
        gpu_result.energy_molecular_total_au,
    ):
        gpu.backend.assert_resident(value)

    for name in (
        "overlap",
        "one_electron_matrix",
        "lower_mechanical_matrix",
        "energy_kinetic_au",
        "energy_electron_nuclear_au",
        "energy_hartree_au",
        "energy_exchange_correlation_au",
        "energy_electronic_au",
        "energy_molecular_total_au",
    ):
        np.testing.assert_allclose(
            gpu.backend.to_host(getattr(gpu_result, name)),
            getattr(cpu_result, name),
            atol=5.0e-11,
            rtol=5.0e-11,
        )

    with pytest.raises(UnsupportedConfigurationError, match="CPU-only"):
        gpu.solve()
