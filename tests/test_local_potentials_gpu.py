from __future__ import annotations

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind
from aion.electromagnetism import UniformMagneticField
from aion.electronic_structure import (
    NuclearAttractionProvider,
    ScaledLocalPotentialProvider,
    evaluate_local_potential_magnetic_matrices,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.gpu


def test_local_provider_hierarchies_are_gpu_resident_and_match_cpu() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    field = UniformMagneticField((0.011, -0.007, 0.005))
    provider = ScaledLocalPotentialProvider(NuclearAttractionProvider(), 0.625)
    cpu_quadrature = prepare_ao_quadrature(reference, BackendConfig(), block_size=4096)
    gpu_quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(BackendKind.GPU, device_index=0),
        block_size=4096,
    )
    cpu_result = evaluate_local_potential_magnetic_matrices(cpu_quadrature, (field,), (provider,))[
        0
    ]
    gpu_result = evaluate_local_potential_magnetic_matrices(gpu_quadrature, (field,), (provider,))[
        0
    ]
    for name in (
        "zero",
        "quadrature_zero",
        "first_F",
        "second_F2",
        "exact_grid",
        "exact",
    ):
        actual = getattr(gpu_result.hierarchy, name)
        gpu_quadrature.backend.assert_resident(actual)
        np.testing.assert_allclose(
            gpu_quadrature.backend.to_host(actual),
            getattr(cpu_result.hierarchy, name),
            atol=3.0e-11,
            rtol=3.0e-11,
        )
    assert gpu_result.direct_lower_grid is not None
    gpu_quadrature.backend.assert_resident(gpu_result.direct_lower_grid)
    np.testing.assert_allclose(
        gpu_quadrature.backend.to_host(gpu_result.direct_lower_grid),
        gpu_quadrature.backend.to_host(gpu_result.lower_exact_grid),
        atol=3.0e-11,
        rtol=3.0e-11,
    )
