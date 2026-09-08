from __future__ import annotations

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")

from aion.gauge import UniformMagneticGauge
from aion.reference import evaluate_uniform_magnetic_matrices
from aion.reference.wilson_campaign import (
    MATRIX_NAMES,
    CampaignCase,
    PairConfiguration,
    accepted_pair_configurations,
    accepted_scan_cases,
    adaptive_log_field,
    ao_angular_momenta,
    build_numerical_floor_record,
    build_pair_molecule,
    campaign_case_from_dict,
    field_directions,
    first_threshold_bracket,
    gauge_for_case,
    paired_field_parity,
    result_arrays,
    signed_decade_fields,
    summarize_case,
    threshold_metric_values,
)


def test_accepted_manifest_covers_declared_pair_campaign_axes():
    configurations = accepted_pair_configurations()
    cases = accepted_scan_cases(grid_level=2)

    assert len(configurations) == 12
    assert len(cases) == 510
    assert {configuration.system for configuration in configurations} == {"H-H", "O-H"}
    assert {configuration.basis for configuration in configurations} == {
        "sto-3g",
        "cc-pvdz",
        "aug-cc-pvdz",
    }
    assert {configuration.contracted for configuration in configurations} == {
        False,
        True,
    }
    assert {case.orientation for case in cases} == {
        "parallel",
        "perpendicular_1",
        "perpendicular_2",
        "oblique",
    }
    fields = signed_decade_fields()
    np.testing.assert_allclose(fields, -np.asarray(fields)[::-1], atol=0.0, rtol=0.0)
    assert min(abs(field) for field in fields if field != 0.0) == pytest.approx(1.0e-7)
    assert max(fields) == pytest.approx(1.0)

    case = cases[123]
    assert campaign_case_from_dict(case.to_dict()) == case


def test_pair_geometry_directions_and_uncontracted_basis_metadata():
    configuration = PairConfiguration("O-H", 1.8, "cc-pvdz")
    mol = build_pair_molecule(configuration)
    directions = field_directions(mol)

    for direction in directions.values():
        np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=2.0e-15)
    np.testing.assert_allclose(
        np.dot(directions["parallel"], directions["perpendicular_1"]),
        0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        np.dot(directions["parallel"], directions["perpendicular_2"]),
        0.0,
        atol=2.0e-15,
    )
    assert abs(np.dot(directions["parallel"], directions["oblique"])) > 0.2
    assert abs(np.dot(directions["perpendicular_1"], directions["oblique"])) > 0.2

    angular = ao_angular_momenta(mol)
    assert angular.shape == (mol.nao_nr(),)
    assert {0, 1, 2}.issubset(set(angular))
    uncontracted = build_pair_molecule(
        PairConfiguration("O-H", 1.8, "cc-pvdz", contracted=False)
    )
    assert uncontracted.nao_nr() > mol.nao_nr()


def test_floor_record_and_case_summary_use_refinement_derived_policies():
    configuration = PairConfiguration("H-H", 1.4, "sto-3g")
    mol = build_pair_molecule(configuration)
    zero_gauge = UniformMagneticGauge(np.zeros(3))
    coarse = evaluate_uniform_magnetic_matrices(
        mol,
        zero_gauge,
        grid_level=0,
        block_size=257,
    )
    fine = evaluate_uniform_magnetic_matrices(
        mol,
        zero_gauge,
        grid_level=1,
        block_size=257,
    )
    floors = build_numerical_floor_record(coarse, fine, mol)
    for name in MATRIX_NAMES:
        assert floors["matrices"][name]["element_absolute"] > 0.0
        assert floors["matrices"][name]["frobenius_absolute"] > 0.0
        assert floors["matrices"][name]["blocks"]["0-1"]["absolute_frobenius"] > 0.0

    case = CampaignCase(configuration, "oblique", 0.1, 1)
    result = evaluate_uniform_magnetic_matrices(
        mol,
        gauge_for_case(case, mol),
        grid_level=1,
        block_size=257,
    )
    summary = summarize_case(result, mol, floors)
    metrics = threshold_metric_values(summary)
    assert "overlap.element_max_diagonal_scaled" in metrics
    assert "mechanical.intersite_relative_frobenius" in metrics
    assert "spectrum.maximum_absolute_shift_hartree" in metrics
    assert summary["phase_spread"]["defined_count"] > 0
    assert summary["matrices"]["mechanical"]["channels"]["intersite_s-s"]["count"] > 0
    arrays = result_arrays(result)
    assert "form_factor_barred_mechanical" in arrays
    assert "anchored_vector_barred_mechanical" in arrays
    assert "exact_sector_c_c" in arrays


def test_threshold_bracketing_adds_an_interior_logarithmic_field():
    bracket = first_threshold_bracket(
        [
            (0.0, 0.0, 1.0e-8),
            (1.0e-3, 2.0e-5, 1.0e-8),
            (1.0e-2, 2.0e-3, 1.0e-8),
            (1.0e-1, 0.2, 1.0e-8),
        ],
        threshold=1.0e-3,
    )
    assert bracket is not None
    assert bracket.lower_field_au == 1.0e-3
    assert bracket.upper_field_au == 1.0e-2
    assert bracket.lower_field_au < bracket.adaptive_field_au < bracket.upper_field_au
    assert adaptive_log_field(0.0, 1.0e-4, 0.0, 2.0e-3, 1.0e-3) == pytest.approx(
        1.0e-4 / np.sqrt(10.0)
    )


def test_signed_field_parity_uses_matrices_instead_of_scalar_norms():
    zero_matrix = np.array([[1.0, 0.2], [0.2, 0.8]], dtype=np.complex128)
    odd = 1j * np.array([[0.0, 0.03], [-0.03, 0.0]])
    even = np.array([[0.01, -0.02], [-0.02, 0.04]])
    positive = {}
    negative = {}
    zero = {}
    for name in MATRIX_NAMES:
        key = f"exact_barred_{name}"
        zero[key] = zero_matrix
        positive[key] = zero_matrix + odd + even
        negative[key] = zero_matrix - odd + even
    report = paired_field_parity(positive, negative, zero)
    for name in MATRIX_NAMES:
        np.testing.assert_allclose(
            report[name]["odd_frobenius"],
            np.linalg.norm(odd, ord="fro"),
        )
        np.testing.assert_allclose(
            report[name]["even_frobenius"],
            np.linalg.norm(even, ord="fro"),
        )
        assert report[name]["time_reversal_residual"] == 0.0
