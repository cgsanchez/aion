from __future__ import annotations

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind
from aion.electromagnetism import UniformMagneticField
from aion.electronic_structure import (
    evaluate_magnetic_spatial_connections,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.gpu


def test_spatial_connection_is_gpu_resident_and_matches_cpu() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    field = UniformMagneticField((0.011, -0.007, 0.005))
    cpu_quadrature = prepare_ao_quadrature(reference, BackendConfig(), block_size=4096)
    gpu_quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(BackendKind.GPU, device_index=0),
        block_size=4096,
    )
    cpu_result = evaluate_magnetic_spatial_connections(cpu_quadrature, (field,))[0]
    gpu_result = evaluate_magnetic_spatial_connections(gpu_quadrature, (field,))[0]
    cpu = cpu_result.spatial_connection
    gpu = gpu_result.spatial_connection
    for name in (
        "zero",
        "quadrature_zero",
        "first_F",
        "first_C",
        "second_F2",
        "second_FC",
        "exact_grid",
        "exact",
    ):
        actual = getattr(gpu, name)
        gpu_quadrature.backend.assert_resident(actual)
        np.testing.assert_allclose(
            gpu_quadrature.backend.to_host(actual),
            getattr(cpu, name),
            atol=3.0e-11,
            rtol=3.0e-11,
        )
    gpu_quadrature.backend.assert_resident(gpu_result.e1_first_C_closure)
    assert gpu_result.direct_oracle is not None
    gpu_quadrature.backend.assert_resident(gpu_result.direct_oracle.lower_grid)
    np.testing.assert_allclose(
        gpu_quadrature.backend.to_host(gpu_result.direct_oracle.lower_grid),
        gpu_quadrature.backend.to_host(gpu_result.lower_exact_grid),
        atol=3.0e-11,
        rtol=3.0e-11,
    )
