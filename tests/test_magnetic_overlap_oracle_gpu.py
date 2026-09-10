from __future__ import annotations

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind
from aion.electromagnetism import AffineMagneticGauge, UniformMagneticField
from aion.electronic_structure import (
    AOGridPolicy,
    analytic_uniform_magnetic_overlap,
    evaluate_magnetic_one_electron_matrices,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.gpu


def test_gpu_finite_field_overlap_matches_independent_gaussian_fourier_oracle() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    field = UniformMagneticField((0.09, -0.07, 0.11))
    gauge = AffineMagneticGauge(field, origin_au=(0.17, -0.31, 0.23))
    analytic = analytic_uniform_magnetic_overlap(reference, gauge)
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(BackendKind.GPU, device_index=0),
        grid_policy=AOGridPolicy.qualification(4),
        block_size=4096,
    )
    result = evaluate_magnetic_one_electron_matrices(
        quadrature,
        (field,),
        direct_gauges=(gauge,),
    )[0]
    quadrature.backend.assert_resident(result.overlap.exact)
    np.testing.assert_allclose(
        quadrature.backend.to_host(result.overlap.exact),
        analytic.barred,
        atol=2.0e-12,
    )
