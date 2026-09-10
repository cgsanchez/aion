from __future__ import annotations

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind
from aion.electromagnetism import UniformMagneticField
from aion.electronic_structure import (
    evaluate_magnetic_one_electron_matrices,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.gpu


def _arrays(result: object) -> tuple[object, ...]:
    arrays: list[object] = [result.endpoint_link]  # type: ignore[attr-defined]
    for hierarchy_name in ("overlap", "nuclear_attraction"):
        hierarchy = getattr(result, hierarchy_name)
        arrays.extend(
            (
                hierarchy.zero,
                hierarchy.quadrature_zero,
                hierarchy.first_F,
                hierarchy.second_F2,
                hierarchy.exact_grid,
                hierarchy.exact,
            )
        )
    kinetic = result.kinetic  # type: ignore[attr-defined]
    arrays.extend(
        getattr(kinetic, name)
        for name in (
            "zero",
            "quadrature_zero",
            "first_F",
            "first_pC",
            "first_Cp",
            "second_F2",
            "second_F_pC",
            "second_F_Cp",
            "second_C2",
            "exact_grid",
            "exact",
        )
    )
    assert result.direct_oracle is not None  # type: ignore[attr-defined]
    arrays.extend(
        (
            result.direct_oracle.lower_grid.overlap,  # type: ignore[attr-defined]
            result.direct_oracle.lower_grid.kinetic,  # type: ignore[attr-defined]
            result.direct_oracle.lower_grid.nuclear_attraction,  # type: ignore[attr-defined]
        )
    )
    return tuple(arrays)


def test_magnetic_matrices_are_gpu_resident_and_match_cpu() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    field = UniformMagneticField((0.011, -0.007, 0.005))
    cpu_quadrature = prepare_ao_quadrature(reference, BackendConfig(), block_size=4096)
    gpu_quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(BackendKind.GPU, device_index=0),
        block_size=4096,
    )
    cpu_result = evaluate_magnetic_one_electron_matrices(cpu_quadrature, (field,))[0]
    gpu_result = evaluate_magnetic_one_electron_matrices(gpu_quadrature, (field,))[0]
    cpu_arrays = _arrays(cpu_result)
    gpu_arrays = _arrays(gpu_result)
    for expected, actual in zip(cpu_arrays, gpu_arrays, strict=True):
        gpu_quadrature.backend.assert_resident(actual)
        np.testing.assert_allclose(
            gpu_quadrature.backend.to_host(actual), expected, atol=3.0e-11, rtol=3.0e-11
        )
    assert gpu_result.direct_oracle is not None
    np.testing.assert_allclose(
        gpu_quadrature.backend.to_host(gpu_result.direct_oracle.lower_grid.kinetic),
        gpu_quadrature.backend.to_host(gpu_result.lower_exact_grid.kinetic),
        atol=3.0e-11,
        rtol=3.0e-11,
    )
