from __future__ import annotations

import numpy as np
import pytest

from aion.backends import CuPyBackend, NumPyBackend
from aion.config import (
    AffineElectromagneticSourceConfig,
    WilsonMagneticGaugeKind,
    ZeroSourceConfig,
)
from aion.electromagnetism import build_affine_electromagnetic_source

pytestmark = pytest.mark.gpu


def test_affine_source_sample_cpu_gpu_field_and_potential_parity_and_residency() -> None:
    config = AffineElectromagneticSourceConfig(
        electric=ZeroSourceConfig(),
        electric_field_origin_offset_au=(0.01, -0.02, 0.03),
        magnetic_field_reference_au=(0.0, 0.0, 0.4),
        magnetic_field_rate_au=(0.0, 0.0, -0.02),
        magnetic_reference_time_au=2.0,
        magnetic_gauge=WilsonMagneticGaugeKind.LANDAU,
        landau_axis=(1.0, 0.0, 0.0),
    )
    source = build_affine_electromagnetic_source(config, (0.5, -0.25, 0.75))
    sample = source.sample(5.0)
    points = np.asarray(
        (
            (1.5, -0.25, 0.75),
            (0.5, 0.75, 0.75),
            (-0.5, -0.75, 1.25),
        )
    )
    cpu = NumPyBackend()
    gpu = CuPyBackend(0)
    device_points = gpu.asarray(points)

    device_electric = sample.electric_field(device_points, gpu)
    device_scalar = sample.scalar_potential(device_points, gpu)
    gpu.assert_resident(device_electric, name="electric field")
    gpu.assert_resident(device_scalar, name="scalar potential")
    np.testing.assert_allclose(
        gpu.to_host(device_electric),
        sample.electric_field(points, cpu),
        atol=1.0e-14,
        rtol=1.0e-14,
    )
    np.testing.assert_allclose(
        gpu.to_host(device_scalar),
        sample.scalar_potential(points, cpu),
        atol=1.0e-14,
        rtol=1.0e-14,
    )
