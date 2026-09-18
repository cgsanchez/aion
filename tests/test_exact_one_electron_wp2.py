from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigvalsh

from aion.backends import NumPyBackend
from aion.config import (
    AtomConfig,
    BackendConfig,
    ElectromagneticOrigin,
    OneElectronReferenceConfig,
    canonical_sha256,
)
from aion.electromagnetism import (
    AffineMagneticGauge,
    MagneticGaugeKind,
    UniformMagneticField,
    affine_gauge_difference_potential,
)
from aion.electronic_structure import (
    AOGridPolicy,
    analytic_uniform_magnetic_overlap,
    evaluate_magnetic_one_electron_matrices,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
)

pytestmark = pytest.mark.integration

_FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "exact_one_electron"
_MATRIX_NAMES = ("overlap", "kinetic", "nuclear_attraction")


def _config(name: str) -> OneElectronReferenceConfig:
    fixture = json.loads(
        (_FIXTURE_DIRECTORY / f"{name}_sto3g.fixture.json").read_text(
            encoding="utf-8"
        )
    )
    values = fixture["config"]
    return OneElectronReferenceConfig(
        atoms=tuple(
            AtomConfig(atom["symbol"], tuple(atom["position_au"]))
            for atom in values["atoms"]
        ),
        basis=values["basis"],
        electromagnetic_origin=ElectromagneticOrigin(
            tuple(values["electromagnetic_origin_au"])
        ),
    )


def _quadrature(
    name: str, level: int = 3, block_size: int = 1024
) -> tuple[object, object]:
    reference = prepare_one_electron_ao_reference(_config(name))
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(level),
        block_size=block_size,
    )
    return reference, quadrature


def _relative_residual(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(1.0, np.linalg.norm(right)))


def _mechanical_spectrum(result: object) -> np.ndarray:
    lower = result.lower_exact  # type: ignore[attr-defined]
    return eigvalsh(lower.mechanical, lower.overlap)


def test_wp2_zero_field_reduces_exactly_to_wp1_analytic_matrices() -> None:
    reference, quadrature = _quadrature("hh")
    result = evaluate_magnetic_one_electron_matrices(
        quadrature, (UniformMagneticField((0.0, 0.0, 0.0)),)
    )[0]
    core = reference.core_operators  # type: ignore[attr-defined]

    np.testing.assert_array_equal(result.endpoint_link, np.ones((2, 2)))
    np.testing.assert_array_equal(result.lower_exact.overlap, core.overlap)
    np.testing.assert_array_equal(result.lower_exact.kinetic, core.kinetic)
    np.testing.assert_array_equal(
        result.lower_exact.nuclear_attraction, core.nuclear_attraction
    )
    np.testing.assert_array_equal(
        result.lower_exact.mechanical, core.kinetic + core.nuclear_attraction
    )
    assert result.direct_oracle is not None
    for name in _MATRIX_NAMES:
        np.testing.assert_allclose(
            getattr(result.direct_oracle.lower_grid, name),
            getattr(result.lower_exact_grid, name),
            atol=8.0e-16,
            rtol=0.0,
        )


def test_wp2_exact_routes_reversal_sectors_and_metric_positivity() -> None:
    _, quadrature = _quadrature("hh")
    magnitude = 0.08
    directions = (
        np.array((0.0, 0.0, 1.0)),
        np.array((1.0, 0.0, 0.0)),
        np.array((1.0, 2.0, 3.0)) / np.sqrt(14.0),
    )
    positive = tuple(
        UniformMagneticField(tuple(magnitude * direction)) for direction in directions
    )
    negative = tuple(
        UniformMagneticField(tuple(-magnitude * direction)) for direction in directions
    )
    results = evaluate_magnetic_one_electron_matrices(
        quadrature, positive + negative
    )

    for plus, minus in zip(results[:3], results[3:], strict=True):
        assert plus.direct_oracle is not None
        np.testing.assert_allclose(
            plus.kinetic.quadrature_exact_sectors.total,
            plus.kinetic.exact_grid,
            atol=2.0e-15,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            plus.kinetic.exact_sectors.total,
            plus.kinetic.exact,
            atol=2.0e-15,
            rtol=0.0,
        )
        for name in _MATRIX_NAMES:
            direct = np.asarray(getattr(plus.direct_oracle.lower_grid, name))
            factorized = np.asarray(getattr(plus.lower_exact_grid, name))
            assert _relative_residual(direct, factorized) < 3.0e-12

            plus_barred = np.asarray(getattr(plus, name).exact)
            minus_barred = np.asarray(getattr(minus, name).exact)
            np.testing.assert_allclose(
                minus_barred, plus_barred.conj(), atol=3.0e-13, rtol=3.0e-13
            )
            np.testing.assert_allclose(
                plus_barred,
                plus_barred.conj().T,
                atol=3.0e-12,
                rtol=3.0e-12,
            )

        metric_eigenvalues = np.linalg.eigvalsh(plus.lower_exact.overlap)
        assert metric_eigenvalues[0] > 0.0
        assert np.linalg.cond(plus.lower_exact.overlap) < 10.0


def test_wp2_gauge_origin_covariance_and_generalized_spectra() -> None:
    reference, quadrature = _quadrature("hh")
    field = UniformMagneticField(tuple(0.08 * np.array((1.0, 2.0, 3.0)) / np.sqrt(14.0)))
    origin_a = (0.0, 0.0, 0.0)
    origin_b = (0.21, -0.17, 0.13)
    field_vector = np.asarray(field.magnetic_field_au)
    axis = np.cross(field_vector, np.array((0.31, -0.47, 0.79)))
    gauges = (
        AffineMagneticGauge(field, origin_au=origin_a),
        AffineMagneticGauge(field, origin_au=origin_b),
        AffineMagneticGauge(
            field,
            kind=MagneticGaugeKind.LANDAU,
            origin_au=origin_a,
            landau_axis=tuple(axis),
        ),
        AffineMagneticGauge(
            field,
            kind=MagneticGaugeKind.LANDAU,
            origin_au=origin_b,
            landau_axis=tuple(axis),
        ),
    )
    results = evaluate_magnetic_one_electron_matrices(
        quadrature,
        (field,) * len(gauges),
        direct_gauges=gauges,
    )
    anchors = reference.core_operators.nuclei.coordinates_au[
        reference.anchor_topology.ao_to_atom
    ]
    backend = NumPyBackend()
    baseline = results[0]
    baseline_spectrum = _mechanical_spectrum(baseline)

    for gauge, result in zip(gauges[1:], results[1:], strict=True):
        chi = affine_gauge_difference_potential(gauge, gauges[0], anchors, backend)
        coefficient_phase = np.exp(-1j * np.asarray(chi))
        for name in _MATRIX_NAMES:
            baseline_barred = np.asarray(getattr(baseline, name).exact)
            candidate_barred = np.asarray(getattr(result, name).exact)
            np.testing.assert_array_equal(candidate_barred, baseline_barred)

            baseline_lower = np.asarray(getattr(baseline.lower_exact, name))
            expected_lower = (
                coefficient_phase[:, None]
                * baseline_lower
                * coefficient_phase[None, :].conj()
            )
            np.testing.assert_allclose(
                getattr(result.lower_exact, name),
                expected_lower,
                atol=2.0e-14,
                rtol=2.0e-14,
            )
        np.testing.assert_allclose(
            _mechanical_spectrum(result),
            baseline_spectrum,
            atol=3.0e-13,
            rtol=3.0e-13,
        )

    analytic = analytic_uniform_magnetic_overlap(reference, gauges[0])
    assert _relative_residual(baseline.overlap.exact, analytic.barred) < 4.0e-9
    assert _relative_residual(baseline.lower_exact.overlap, analytic.lower) < 4.0e-9


def test_wp2_authenticated_oblique_reference_output_is_reproduced() -> None:
    fixture = json.loads(
        (_FIXTURE_DIRECTORY / "wp2_hh_oblique_reference.json").read_text(
            encoding="utf-8"
        )
    )
    reference, quadrature = _quadrature(
        "hh",
        level=fixture["quadrature"]["level"],
        block_size=fixture["quadrature"]["block_size"],
    )
    direction = np.asarray(fixture["field"]["direction"])
    field = UniformMagneticField(
        tuple(fixture["field"]["magnitude_au"] * direction)
    )
    result = evaluate_magnetic_one_electron_matrices(quadrature, (field,))[0]

    assert fixture["status"] == "executed_unreviewed_reference_output"
    assert reference.fingerprint_sha256 == fixture["reference_fingerprint_sha256"]
    assert (
        quadrature.grid.fingerprint_sha256
        == fixture["quadrature"]["fingerprint_sha256"]
    )
    sectors = result.kinetic.exact_sectors
    arrays = {
        "lower_exact_overlap": result.lower_exact.overlap,
        "lower_exact_kinetic": result.lower_exact.kinetic,
        "lower_exact_nuclear_attraction": result.lower_exact.nuclear_attraction,
        "lower_exact_mechanical": result.lower_exact.mechanical,
        "kinetic_T_pp_F": sectors.pp,
        "kinetic_T_pC_F": sectors.pC,
        "kinetic_T_Cp_F": sectors.Cp,
        "kinetic_T_CC_F": sectors.C2,
    }
    assert {
        name: canonical_sha256(np.asarray(value)) for name, value in arrays.items()
    } == fixture["semantic_array_sha256"]


def _rotation_matrix() -> np.ndarray:
    axis = np.array((1.0, -2.0, 0.7))
    axis /= np.linalg.norm(axis)
    angle = 0.61
    cross = np.array(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        )
    )
    return (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def test_wp2_rigid_rotation_covariance_with_oxygen_p_mixing() -> None:
    config = _config("oh")
    rotation = _rotation_matrix()
    rotated_config = OneElectronReferenceConfig(
        atoms=tuple(
            AtomConfig(
                atom.symbol,
                tuple(rotation @ np.asarray(atom.position_au, dtype=np.float64)),
            )
            for atom in config.atoms
        ),
        basis=config.basis,
        electromagnetic_origin=ElectromagneticOrigin(
            tuple(rotation @ np.asarray(config.electromagnetic_origin.position_au))
        ),
    )
    reference = prepare_one_electron_ao_reference(config)
    rotated_reference = prepare_one_electron_ao_reference(rotated_config)
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(4),
        block_size=2048,
    )
    rotated_quadrature = prepare_ao_quadrature(
        rotated_reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(4),
        block_size=2048,
    )
    field_vector = 0.06 * np.array((1.0, 2.0, 3.0)) / np.sqrt(14.0)
    result = evaluate_magnetic_one_electron_matrices(
        quadrature, (UniformMagneticField(tuple(field_vector)),)
    )[0]
    rotated_result = evaluate_magnetic_one_electron_matrices(
        rotated_quadrature,
        (UniformMagneticField(tuple(rotation @ field_vector)),),
    )[0]

    ao_rotation = np.eye(reference.core_operators.nao)
    ao_rotation[2:5, 2:5] = rotation
    for name in _MATRIX_NAMES:
        expected = ao_rotation @ np.asarray(getattr(result, name).exact) @ ao_rotation.T
        actual = np.asarray(getattr(rotated_result, name).exact)
        assert _relative_residual(actual, expected) < 8.0e-8
    np.testing.assert_allclose(
        _mechanical_spectrum(rotated_result),
        _mechanical_spectrum(result),
        atol=8.0e-8,
        rtol=8.0e-8,
    )
