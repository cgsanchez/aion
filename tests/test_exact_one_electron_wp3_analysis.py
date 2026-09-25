from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aion.config import (
    AtomConfig,
    BackendConfig,
    ElectromagneticOrigin,
    OneElectronReferenceConfig,
)
from aion.electromagnetism import UniformMagneticField
from aion.electronic_structure import (
    AOGridPolicy,
    angular_channel_changes,
    ao_angular_momenta,
    compare_generalized_spectra,
    diagonal_scaled_element_change,
    evaluate_exact_static_magnetic_one_electron_matrices,
    evaluate_magnetic_one_electron_matrices,
    matrix_partition_changes,
    metric_spectrum,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
    static_magnetic_diagnostic_models,
    static_magnetic_first_order_models,
)


def test_metric_spectrum_reports_indefiniteness_without_regularization() -> None:
    positive = metric_spectrum(np.diag([0.25, 2.0]))
    assert positive.positive
    assert positive.minimum_eigenvalue == pytest.approx(0.25)
    assert positive.condition_number == pytest.approx(8.0)

    indefinite = metric_spectrum(np.diag([-0.125, 1.0]))
    assert not indefinite.positive
    assert indefinite.minimum_eigenvalue == pytest.approx(-0.125)
    np.testing.assert_array_equal(indefinite.eigenvalues, [-0.125, 1.0])


pytestmark = pytest.mark.integration

_FIXTURE = Path(__file__).parent / "fixtures/exact_one_electron/hh_sto3g.fixture.json"
_MATRIX_NAMES = ("overlap", "kinetic", "nuclear_attraction", "mechanical")


def _quadrature() -> object:
    values = json.loads(_FIXTURE.read_text(encoding="utf-8"))["config"]
    config = OneElectronReferenceConfig(
        atoms=tuple(
            AtomConfig(atom["symbol"], tuple(atom["position_au"])) for atom in values["atoms"]
        ),
        basis=values["basis"],
        electromagnetic_origin=ElectromagneticOrigin(tuple(values["electromagnetic_origin_au"])),
    )
    reference = prepare_one_electron_ao_reference(config)
    return prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(3),
        block_size=1024,
    )


def test_wp3_phase_spread_is_stable_and_isolates_parallel_geometry() -> None:
    quadrature = _quadrature()
    fields = (
        UniformMagneticField((0.0, 0.0, 0.0)),
        UniformMagneticField((0.0, 0.0, 0.08)),
        UniformMagneticField((0.08, 0.0, 0.0)),
        UniformMagneticField((-0.08, 0.0, 0.0)),
    )
    zero, parallel, perpendicular, reversed_field = evaluate_magnetic_one_electron_matrices(
        quadrature,
        fields,
        include_direct_oracle=False,
    )

    assert np.all(zero.phase_spread.absolute_product_integral > 0.0)
    np.testing.assert_array_equal(zero.phase_spread.rms, np.zeros((2, 2)))
    np.testing.assert_array_equal(parallel.phase_spread.rms, np.zeros((2, 2)))
    np.testing.assert_array_equal(perpendicular.phase_spread.rms, reversed_field.phase_spread.rms)
    np.testing.assert_array_equal(np.diag(perpendicular.phase_spread.rms), np.zeros(2))
    assert perpendicular.phase_spread.rms[0, 1] > 1.0e-3
    np.testing.assert_allclose(
        perpendicular.phase_spread.rms,
        perpendicular.phase_spread.rms.T,
        atol=2.0e-15,
        rtol=0.0,
    )


def test_wp3_static_diagnostic_models_have_declared_sector_content() -> None:
    quadrature = _quadrature()
    zero, parallel, oblique = evaluate_magnetic_one_electron_matrices(
        quadrature,
        (
            UniformMagneticField((0.0, 0.0, 0.0)),
            UniformMagneticField((0.0, 0.0, 0.08)),
            UniformMagneticField((0.02, -0.03, 0.05)),
        ),
        include_direct_oracle=False,
    )

    zero_models = static_magnetic_diagnostic_models(zero)
    for name in _MATRIX_NAMES:
        baseline = getattr(zero_models.lower.p0, name)
        for model in (
            zero_models.lower.form_factor_only,
            zero_models.lower.anchored_vector_only,
            zero_models.lower.exact,
        ):
            np.testing.assert_array_equal(getattr(model, name), baseline)

    parallel_models = static_magnetic_diagnostic_models(parallel)
    for name in _MATRIX_NAMES:
        np.testing.assert_array_equal(
            getattr(parallel_models.barred.form_factor_only, name),
            getattr(parallel_models.barred.p0, name),
        )
        np.testing.assert_allclose(
            getattr(parallel_models.barred.anchored_vector_only, name),
            getattr(parallel_models.barred.exact, name),
            atol=2.0e-15,
            rtol=0.0,
        )

    oblique_models = static_magnetic_diagnostic_models(oblique)
    np.testing.assert_array_equal(
        oblique_models.barred.form_factor_only.kinetic,
        oblique.kinetic.exact_sectors.pp,
    )
    np.testing.assert_array_equal(
        oblique_models.barred.anchored_vector_only.kinetic,
        oblique.kinetic.zero
        + oblique.kinetic.first_pC
        + oblique.kinetic.first_Cp
        + oblique.kinetic.second_C2,
    )
    for name in _MATRIX_NAMES:
        np.testing.assert_array_equal(
            getattr(oblique_models.lower.exact, name),
            getattr(oblique.lower_exact, name),
        )


def test_wp3_block_and_generalized_spectral_diagnostics_are_complete() -> None:
    quadrature = _quadrature()
    result = evaluate_magnetic_one_electron_matrices(
        quadrature,
        (UniformMagneticField((0.02, -0.03, 0.05)),),
        include_direct_oracle=False,
    )[0]
    models = static_magnetic_diagnostic_models(result).lower
    mapping = quadrature.reference.anchor_topology.ao_to_atom
    angular = ao_angular_momenta(quadrature.reference)

    partitions = matrix_partition_changes(
        models.exact.mechanical,
        models.p0.mechanical,
        mapping,
        floor=1.0e-12,
    )
    assert partitions.labels == (
        "onsite_diagonal",
        "same_anchor_offdiagonal",
        "intersite",
    )
    assert np.all(np.isfinite(partitions.relative_frobenius))
    np.testing.assert_array_equal(angular, np.zeros(2, dtype=np.int64))

    channels = angular_channel_changes(
        models.exact.mechanical,
        models.p0.mechanical,
        mapping,
        angular,
        floor=1.0e-12,
    )
    assert len(channels) == 4
    assert sum(value.classification == "onsite" for value in channels) == 2
    assert sum(value.classification == "intersite" for value in channels) == 2
    assert all(value.singular_values.shape == (1,) for value in channels)

    scaled = diagonal_scaled_element_change(
        models.exact.mechanical,
        models.p0.mechanical,
        floor=1.0e-12,
    )
    assert scaled.shape == (2, 2)
    assert np.all(np.isfinite(scaled))

    spectral = compare_generalized_spectra(
        models.exact.mechanical,
        models.exact.overlap,
        models.p0.mechanical,
        models.p0.overlap,
    )
    assert spectral.eigenvalue_shifts.shape == (2,)
    assert spectral.cumulative_subspace_angles_rad.shape == (1,)
    assert np.all(np.isfinite(spectral.eigenvalue_shifts))
    assert np.all(spectral.cumulative_subspace_angles_rad >= 0.0)


def test_wp3_exact_only_scan_path_matches_complete_hierarchy() -> None:
    quadrature = _quadrature()
    fields = (
        UniformMagneticField((0.0, 0.0, 0.08)),
        UniformMagneticField((0.02, -0.03, 0.05)),
    )
    complete = evaluate_magnetic_one_electron_matrices(
        quadrature, fields, include_direct_oracle=False
    )
    reduced = evaluate_exact_static_magnetic_one_electron_matrices(quadrature, fields)
    for complete_result, reduced_result in zip(complete, reduced, strict=True):
        for name in _MATRIX_NAMES:
            np.testing.assert_array_equal(
                getattr(reduced_result.lower_exact, name),
                getattr(complete_result.lower_exact, name),
            )
        for name in ("pp", "pC", "Cp", "C2"):
            np.testing.assert_array_equal(
                getattr(reduced_result.kinetic.exact_sectors, name),
                getattr(complete_result.kinetic.exact_sectors, name),
            )
        np.testing.assert_array_equal(
            reduced_result.kinetic.anchored_pC,
            complete_result.kinetic.first_pC,
        )
        np.testing.assert_array_equal(
            reduced_result.kinetic.anchored_Cp,
            complete_result.kinetic.first_Cp,
        )
        np.testing.assert_array_equal(
            reduced_result.kinetic.anchored_C2,
            complete_result.kinetic.second_C2,
        )
        np.testing.assert_array_equal(
            reduced_result.phase_spread.rms,
            complete_result.phase_spread.rms,
        )
        reduced_models = static_magnetic_diagnostic_models(reduced_result)
        complete_models = static_magnetic_diagnostic_models(complete_result)
        for name in _MATRIX_NAMES:
            np.testing.assert_array_equal(
                getattr(reduced_models.lower.anchored_vector_only, name),
                getattr(complete_models.lower.anchored_vector_only, name),
            )


def test_wp4_named_static_models_have_exact_declared_sector_content() -> None:
    quadrature = _quadrature()
    result = evaluate_magnetic_one_electron_matrices(
        quadrature,
        (UniformMagneticField((0.02, -0.03, 0.05)),),
        include_direct_oracle=False,
    )[0]
    models = static_magnetic_first_order_models(result)

    np.testing.assert_array_equal(
        models.barred.geometric_b1.overlap - models.barred.p0.overlap,
        result.overlap.first_F,
    )
    np.testing.assert_array_equal(
        models.barred.geometric_b1.mechanical,
        models.barred.p0.mechanical,
    )
    np.testing.assert_array_equal(
        models.barred.full_b1.overlap,
        models.barred.geometric_b1.overlap,
    )
    np.testing.assert_array_equal(
        models.barred.full_b1.mechanical - models.barred.geometric_b1.mechanical,
        result.kinetic.first + result.nuclear_attraction.first_F,
    )
    for name in _MATRIX_NAMES:
        np.testing.assert_array_equal(
            getattr(models.barred.complete_first, name),
            getattr(models.barred.full_b1, name),
        )
        np.testing.assert_array_equal(
            getattr(models.lower.complete_first, name),
            getattr(models.lower.full_b1, name),
        )
        np.testing.assert_array_equal(
            getattr(models.lower.exact, name),
            getattr(result.lower_exact, name),
        )


def test_wp4_named_static_models_reject_exact_only_scan_results() -> None:
    quadrature = _quadrature()
    result = evaluate_exact_static_magnetic_one_electron_matrices(
        quadrature,
        (UniformMagneticField((0.02, -0.03, 0.05)),),
    )[0]
    with pytest.raises(TypeError, match="MagneticOneElectronResult"):
        static_magnetic_first_order_models(result)  # type: ignore[arg-type]
