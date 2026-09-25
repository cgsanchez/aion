from __future__ import annotations

import numpy as np
import pytest

from aion.backends import NumPyBackend
from aion.config import (
    AffineElectromagneticSourceConfig,
    Sin2VectorPotentialPulseConfig,
    WilsonMagneticGaugeKind,
    ZeroSourceConfig,
)
from aion.electromagnetism import build_affine_electromagnetic_source
from aion.errors import SourceCompilationError

pytestmark = pytest.mark.fast


def source_config(
    *,
    gauge: WilsonMagneticGaugeKind = WilsonMagneticGaugeKind.SYMMETRIC,
) -> AffineElectromagneticSourceConfig:
    return AffineElectromagneticSourceConfig(
        electric=ZeroSourceConfig(),
        electric_field_origin_offset_au=(0.1, -0.2, 0.3),
        magnetic_field_reference_au=(0.0, 0.0, 0.4),
        magnetic_field_rate_au=(0.0, 0.0, -0.02),
        magnetic_reference_time_au=2.0,
        magnetic_gauge=gauge,
        landau_axis=(1.0, 0.0, 0.0) if gauge is WilsonMagneticGaugeKind.LANDAU else None,
    )


def test_affine_source_samples_exact_linear_magnetic_law_and_faraday_partner() -> None:
    origin = (0.5, -0.25, 0.75)
    provider = build_affine_electromagnetic_source(source_config(), origin)
    sample = provider.sample(5.0)
    np.testing.assert_allclose(sample.field.magnetic_field_au, (0.0, 0.0, 0.34))
    np.testing.assert_allclose(sample.magnetic_field_dot_au, (0.0, 0.0, -0.02))
    np.testing.assert_allclose(sample.electric_field_origin_au, (0.1, -0.2, 0.3))

    points = np.array([[1.5, -0.25, 0.75], [0.5, 0.75, 0.75]])
    relative = points - np.asarray(origin)
    expected = np.asarray((0.1, -0.2, 0.3)) + 0.5 * np.cross(
        relative,
        np.asarray((0.0, 0.0, -0.02)),
    )
    np.testing.assert_allclose(sample.electric_field(points, NumPyBackend()), expected)

    h = 1.0e-5
    plus = np.asarray(provider.sample(5.0 + h).field.magnetic_field_au)
    minus = np.asarray(provider.sample(5.0 - h).field.magnetic_field_au)
    np.testing.assert_allclose((plus - minus) / (2.0 * h), (0.0, 0.0, -0.02))


def test_affine_source_composes_pulse_field_with_constant_origin_offset() -> None:
    electric = Sin2VectorPotentialPulseConfig(
        peak_electric_field_au=2.0e-3,
        angular_frequency_au=0.4,
        cycles=2,
        polarization=(0.0, 1.0, 0.0),
    )
    config = AffineElectromagneticSourceConfig(
        electric=electric,
        electric_field_origin_offset_au=(0.01, 0.02, 0.03),
        magnetic_field_reference_au=(0.0, 0.0, 0.0),
        magnetic_field_rate_au=(0.0, 0.0, 0.0),
        magnetic_reference_time_au=0.0,
        magnetic_gauge=WilsonMagneticGaugeKind.SYMMETRIC,
    )
    provider = build_affine_electromagnetic_source(config, (0.0, 0.0, 0.0))
    time = np.pi / 0.4
    pulse_field = provider.electric_provider.sample(time).electric_field
    np.testing.assert_allclose(
        provider.sample(time).electric_field_origin_au,
        np.asarray((0.01, 0.02, 0.03)) + pulse_field,
    )


def test_affine_source_identity_depends_on_origin_and_is_deterministic() -> None:
    first = build_affine_electromagnetic_source(source_config(), (0.0, 0.0, 0.0))
    same = build_affine_electromagnetic_source(source_config(), (0.0, 0.0, 0.0))
    moved = build_affine_electromagnetic_source(source_config(), (0.0, 0.0, 0.1))
    assert first.fingerprint_sha256 == same.fingerprint_sha256
    assert first.fingerprint_sha256 != moved.fingerprint_sha256


def test_landau_axis_must_work_for_field_and_field_rate() -> None:
    bad = AffineElectromagneticSourceConfig(
        electric=ZeroSourceConfig(),
        electric_field_origin_offset_au=(0.0, 0.0, 0.0),
        magnetic_field_reference_au=(0.0, 0.0, 0.4),
        magnetic_field_rate_au=(0.02, 0.0, 0.0),
        magnetic_reference_time_au=0.0,
        magnetic_gauge=WilsonMagneticGaugeKind.LANDAU,
        landau_axis=(1.0, 0.0, 0.0),
    )
    with pytest.raises(SourceCompilationError, match="magnetic-field rate"):
        build_affine_electromagnetic_source(bad, (0.0, 0.0, 0.0))
