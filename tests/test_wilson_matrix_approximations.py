from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

pyscf = pytest.importorskip("pyscf")
from pyscf import gto

from aion.gauge import AOAnchors, PeierlsGeometry, UniformMagneticGauge
from aion.reference import (
    MetricNotPositiveDefiniteError,
    PhaseSpreadMoments,
    atom_pair_block_error,
    build_unpruned_grid,
    compare_spectral_subspaces,
    eigenvalue_clusters,
    elementwise_matrix_error,
    estimate_frobenius_floor,
    evaluate_uniform_magnetic_matrices,
    generalized_hermitian_spectrum,
    phase_spread,
)


_FIELD = np.array([0.09, -0.07, 0.11])
_ORIGIN = np.array([0.17, -0.31, 0.23])
_ATOL = 2.0e-11


@pytest.fixture(scope="module")
def oh_molecule():
    return gto.M(
        atom=[
            ("O", (0.30, -0.25, 0.40)),
            ("H", (1.60, 0.43, -0.21)),
        ],
        basis="sto-3g",
        unit="Bohr",
        spin=1,
        verbose=0,
    )


@pytest.fixture(scope="module")
def qualification_grid(oh_molecule):
    return build_unpruned_grid(oh_molecule, level=1)


@pytest.fixture(scope="module")
def symmetric_gauge():
    return UniformMagneticGauge(
        _FIELD,
        gauge="symmetric",
        origin=_ORIGIN,
    )


@pytest.fixture(scope="module")
def exact_result(oh_molecule, qualification_grid, symmetric_gauge):
    return evaluate_uniform_magnetic_matrices(
        oh_molecule,
        symmetric_gauge,
        grid=qualification_grid,
        block_size=257,
    )


def _matrix_names():
    return ("overlap", "kinetic", "potential", "mechanical")


def test_four_diagnostic_calculations_have_the_declared_sector_content(
    exact_result,
):
    diagnostics = exact_result.diagnostics

    for name in _matrix_names():
        np.testing.assert_allclose(
            getattr(diagnostics.p0.barred, name),
            getattr(exact_result.bare, name),
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            getattr(diagnostics.full_exact.barred, name),
            getattr(exact_result.barred, name),
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            getattr(diagnostics.full_exact.lower, name),
            getattr(exact_result.factorized, name),
            atol=0.0,
            rtol=0.0,
        )

    p0_sectors = diagnostics.p0.kinetic_sectors
    np.testing.assert_allclose(p0_sectors.pp, exact_result.T0, atol=0.0, rtol=0.0)
    for deleted in (p0_sectors.p_c, p0_sectors.c_p, p0_sectors.c_c):
        np.testing.assert_allclose(deleted, 0.0, atol=0.0, rtol=0.0)

    form_factor = diagnostics.form_factor_only
    np.testing.assert_allclose(
        form_factor.barred.overlap,
        exact_result.barred.overlap,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        form_factor.barred.potential,
        exact_result.barred.potential,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        form_factor.barred.kinetic,
        exact_result.kinetic_sectors.pp,
        atol=0.0,
        rtol=0.0,
    )
    for deleted in (
        form_factor.kinetic_sectors.p_c,
        form_factor.kinetic_sectors.c_p,
        form_factor.kinetic_sectors.c_c,
    ):
        np.testing.assert_allclose(deleted, 0.0, atol=0.0, rtol=0.0)

    anchored = diagnostics.anchored_vector_only
    np.testing.assert_allclose(anchored.barred.overlap, exact_result.S0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(anchored.barred.potential, exact_result.V0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        anchored.kinetic_sectors.pp,
        exact_result.T0,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        anchored.barred.kinetic,
        anchored.kinetic_sectors.total,
        atol=_ATOL,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        anchored.kinetic_sectors.p_c,
        anchored.kinetic_sectors.c_p.conj().T,
        atol=_ATOL,
        rtol=0.0,
    )
    assert np.linalg.norm(anchored.barred.kinetic - exact_result.T0) > 1.0e-3


def test_p0_matches_peierls_geometry_and_exact_minus_p0_factorization(
    oh_molecule,
    symmetric_gauge,
    exact_result,
):
    anchors = AOAnchors.from_mol(oh_molecule)
    p0 = exact_result.diagnostics.p0
    geometry = PeierlsGeometry(
        anchors,
        exact_result.S0,
        magnetic=symmetric_gauge,
        charge=exact_result.charge,
        hbar=exact_result.hbar,
    )

    np.testing.assert_allclose(exact_result.theta, geometry.theta(0.0), atol=_ATOL, rtol=0.0)
    np.testing.assert_allclose(p0.lower.overlap, geometry.metric(0.0), atol=_ATOL, rtol=0.0)
    for name in ("kinetic", "potential", "mechanical"):
        np.testing.assert_allclose(
            getattr(p0.lower, name),
            geometry.dress_matrix(getattr(exact_result.bare, name), 0.0),
            atol=_ATOL,
            rtol=0.0,
        )

    for name in _matrix_names():
        exact_minus_p0 = (
            getattr(exact_result.diagnostics.full_exact.lower, name)
            - getattr(p0.lower, name)
        )
        barred_difference = (
            getattr(exact_result.barred, name)
            - getattr(exact_result.bare, name)
        )
        np.testing.assert_allclose(
            exact_minus_p0,
            exact_result.theta * barred_difference,
            atol=_ATOL,
            rtol=0.0,
        )


def test_phase_spread_uses_absolute_overlap_and_safe_undefined_policy(
    exact_result,
):
    measured = phase_spread(
        exact_result.phase_spread_moments,
        denominator_floor=1.0e-12,
    )
    assert np.all(measured.defined)
    assert np.all(np.isfinite(measured.values))
    assert np.all(measured.values >= 0.0)
    np.testing.assert_allclose(measured.values, measured.values.T, atol=5.0e-15, rtol=0.0)

    same_anchor = np.linalg.norm(
        exact_result.ao_anchors[:, None, :] - exact_result.ao_anchors[None, :, :],
        axis=2,
    ) < 1.0e-14
    np.testing.assert_allclose(measured.values[same_anchor], 0.0, atol=0.0, rtol=0.0)
    assert np.max(measured.values[~same_anchor]) > 1.0e-3

    synthetic = phase_spread(
        PhaseSpreadMoments(
            numerator=np.array([[0.0, 1.0], [4.0, 1.0]]),
            denominator=np.array([[0.0, 4.0], [1.0e-15, 1.0]]),
        ),
        denominator_floor=1.0e-12,
    )
    np.testing.assert_array_equal(
        synthetic.defined,
        np.array([[False, True], [False, True]]),
    )
    assert np.isnan(synthetic.values[0, 0])
    assert np.isnan(synthetic.values[1, 0])
    assert synthetic.values[0, 1] == 0.5
    assert synthetic.values[1, 1] == 1.0


def test_elementwise_and_atom_pair_measures_declare_floors_and_are_rotation_stable():
    reference = np.array(
        [
            [2.0, 0.3, 0.0, 0.4j],
            [0.3, 1.5, -0.2j, 0.1],
            [0.0, 0.2j, 1.2, -0.1],
            [-0.4j, 0.1, -0.1, 0.9],
        ],
        dtype=np.complex128,
    )
    perturbation = np.array(
        [
            [0.01, 0.02j, 0.03, -0.01j],
            [-0.02j, -0.01, 0.04j, 0.02],
            [0.03, -0.04j, 0.02, 0.01j],
            [0.01j, 0.02, -0.01j, -0.02],
        ],
        dtype=np.complex128,
    )
    exact = reference + perturbation
    mapping = np.array([0, 0, 1, 1])
    floor = 1.0e-8

    elementwise = elementwise_matrix_error(exact, reference, floor=floor)
    np.testing.assert_allclose(
        elementwise.absolute,
        np.abs(perturbation),
        atol=5.0e-16,
        rtol=0.0,
    )
    assert not elementwise.relative_defined[0, 2]
    assert np.isnan(elementwise.relative[0, 2])
    np.testing.assert_allclose(
        elementwise.diagonal_scaled[0, 2],
        abs(perturbation[0, 2]) / np.sqrt(abs(reference[0, 0] * reference[2, 2])),
        atol=1.0e-15,
        rtol=0.0,
    )

    original = atom_pair_block_error(
        exact,
        reference,
        mapping,
        0,
        1,
        floor=floor,
    )
    unitary_a = scipy.linalg.expm(
        1j * np.array([[0.2, 0.4 - 0.1j], [0.4 + 0.1j, -0.3]])
    )
    unitary_b = scipy.linalg.expm(
        1j * np.array([[-0.1, 0.2j], [-0.2j, 0.35]])
    )
    local_unitary = scipy.linalg.block_diag(unitary_a, unitary_b)
    rotated = atom_pair_block_error(
        local_unitary.conj().T @ exact @ local_unitary,
        local_unitary.conj().T @ reference @ local_unitary,
        mapping,
        0,
        1,
        floor=floor,
    )
    np.testing.assert_allclose(
        rotated.absolute_frobenius,
        original.absolute_frobenius,
        atol=1.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        rotated.relative_frobenius,
        original.relative_frobenius,
        atol=1.0e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        rotated.difference_singular_values,
        original.difference_singular_values,
        atol=1.0e-14,
        rtol=0.0,
    )

    numerical_floor = estimate_frobenius_floor(
        reference + 2.0 * perturbation,
        reference + perturbation,
        analytic_reference=reference,
    )
    expected_floor = np.linalg.norm(perturbation, ord="fro")
    np.testing.assert_allclose(numerical_floor.refinement_change, expected_floor)
    np.testing.assert_allclose(numerical_floor.analytic_error, expected_floor)
    np.testing.assert_allclose(numerical_floor.value, expected_floor)


def test_generalized_spectra_and_degenerate_subspace_comparison():
    operator = np.diag([0.0, 1.0, 1.0, 3.0])
    metric = np.eye(4)
    spectrum = generalized_hermitian_spectrum(
        operator,
        metric,
        metric_floor=1.0e-12,
    )
    np.testing.assert_allclose(spectrum.eigenvalues, np.diag(operator), atol=0.0, rtol=0.0)
    clusters = eigenvalue_clusters(
        spectrum.eigenvalues,
        absolute_gap=1.0e-10,
    )
    assert [cluster.tolist() for cluster in clusters] == [[0], [1, 2], [3]]

    degenerate_vectors = spectrum.eigenvectors[:, clusters[1]]
    internal_rotation = np.array(
        [
            [1.0, 1.0j],
            [1.0j, 1.0],
        ],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    rotated_vectors = degenerate_vectors @ internal_rotation
    assert np.linalg.norm(degenerate_vectors - rotated_vectors) > 0.5
    comparison = compare_spectral_subspaces(
        degenerate_vectors,
        rotated_vectors,
        metric,
        metric_floor=1.0e-12,
    )
    assert comparison.largest_principal_angle < 5.0e-8
    assert comparison.chordal_distance < 5.0e-8

    with pytest.raises(MetricNotPositiveDefiniteError):
        generalized_hermitian_spectrum(
            np.eye(2),
            np.diag([1.0, 1.0e-14]),
            metric_floor=1.0e-12,
        )


def test_barred_and_spectral_p0_comparisons_are_gauge_independent(
    oh_molecule,
    qualification_grid,
    exact_result,
):
    landau_seed = np.array([0.31, 0.27, -0.19])
    landau_u = np.cross(_FIELD, landau_seed)
    landau_u /= np.linalg.norm(landau_u)
    landau_result = evaluate_uniform_magnetic_matrices(
        oh_molecule,
        UniformMagneticGauge(
            _FIELD,
            gauge="landau",
            origin=np.array([-0.21, 0.37, -0.16]),
            landau_u=landau_u,
        ),
        grid=qualification_grid,
        block_size=257,
    )

    floor = 1.0e-10
    symmetric_error = elementwise_matrix_error(
        exact_result.barred.mechanical,
        exact_result.K0,
        floor=floor,
    )
    landau_error = elementwise_matrix_error(
        landau_result.barred.mechanical,
        landau_result.K0,
        floor=floor,
    )
    np.testing.assert_allclose(
        landau_error.diagonal_scaled,
        symmetric_error.diagonal_scaled,
        atol=_ATOL,
        rtol=0.0,
    )

    mapping = AOAnchors.from_mol(oh_molecule).ao_to_atom
    symmetric_block = atom_pair_block_error(
        exact_result.barred.mechanical,
        exact_result.K0,
        mapping,
        0,
        1,
        floor=floor,
    )
    landau_block = atom_pair_block_error(
        landau_result.barred.mechanical,
        landau_result.K0,
        mapping,
        0,
        1,
        floor=floor,
    )
    np.testing.assert_allclose(
        landau_block.absolute_frobenius,
        symmetric_block.absolute_frobenius,
        atol=_ATOL,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        landau_block.difference_singular_values,
        symmetric_block.difference_singular_values,
        atol=_ATOL,
        rtol=0.0,
    )

    def eigenvalue_shift(result):
        exact_spectrum = generalized_hermitian_spectrum(
            result.diagnostics.full_exact.lower.mechanical,
            result.diagnostics.full_exact.lower.overlap,
            metric_floor=floor,
        )
        p0_spectrum = generalized_hermitian_spectrum(
            result.diagnostics.p0.lower.mechanical,
            result.diagnostics.p0.lower.overlap,
            metric_floor=floor,
        )
        return exact_spectrum.eigenvalues - p0_spectrum.eigenvalues

    np.testing.assert_allclose(
        eigenvalue_shift(landau_result),
        eigenvalue_shift(exact_result),
        atol=_ATOL,
        rtol=0.0,
    )
