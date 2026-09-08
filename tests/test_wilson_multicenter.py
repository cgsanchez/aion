from __future__ import annotations

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")

from aion.gauge import AOAnchors, UniformMagneticGauge
from aion.reference import (
    MetricNotPositiveDefiniteError,
    affine_gauge_endpoint_congruence,
    analytic_b1_one_electron_matrices,
    analytic_uniform_wilson_overlap,
    build_h3_framework,
    build_second_row_diatomic,
    build_unpruned_grid,
    compare_spectral_subspaces,
    diatomic_axial_ao_channels,
    distorted_h3_geometry,
    equilateral_h3_geometry,
    evaluate_uniform_magnetic_matrices,
    evaluate_uniform_magnetic_taylor_matrices,
    first_positivity_loss_bracket,
    generalized_hermitian_spectrum,
    metric_diagnostics,
    oriented_polygon_area_vector,
    p0_one_electron_matrices,
    track_generalized_eigenbranches,
    track_hermitian_eigenbranches,
    uniform_loop_flux,
    uniform_loop_link_product,
)


def _normal_and_area(geometry):
    area_vector = oriented_polygon_area_vector(geometry)
    area = np.linalg.norm(area_vector)
    return area_vector / area, area


def _axis_angle_rotation(axis, angle):
    direction = np.asarray(axis, dtype=float)
    direction /= np.linalg.norm(direction)
    cross = np.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )
    return (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(direction, direction)
        + np.sin(angle) * cross
    )


def test_loop_area_flux_and_holonomy_reverse_by_complex_conjugation():
    geometry = distorted_h3_geometry()
    field = np.array([0.0, 0.0, 0.73])
    forward = (0, 1, 2)
    reverse = (0, 2, 1)
    forward_area = oriented_polygon_area_vector(geometry, forward)
    reverse_area = oriented_polygon_area_vector(geometry, reverse)
    np.testing.assert_allclose(reverse_area, -forward_area, atol=2.0e-16)

    forward_flux = uniform_loop_flux(geometry, field, forward)
    reverse_flux = uniform_loop_flux(geometry, field, reverse)
    assert reverse_flux == pytest.approx(-forward_flux, abs=2.0e-16)

    gauges = (
        UniformMagneticGauge(
            field,
            gauge="symmetric",
            origin=np.array([0.21, -0.17, 0.33]),
        ),
        UniformMagneticGauge(
            field,
            gauge="landau",
            origin=np.array([-0.13, 0.29, -0.24]),
            landau_u=np.array([1.0, 0.0, 0.0]),
        ),
    )
    expected = np.exp(-1j * forward_flux)
    for gauge in gauges:
        forward_link = uniform_loop_link_product(gauge, geometry, forward)
        reverse_link = uniform_loop_link_product(gauge, geometry, reverse)
        assert forward_link == pytest.approx(expected, abs=5.0e-16)
        assert reverse_link == pytest.approx(forward_link.conjugate(), abs=5.0e-16)


def test_n2_pi_degeneracy_and_co_inversion_breaking_are_resolved():
    field = np.array([0.0, 0.0, 0.12])
    onsite_norms = {}
    for system, bond_length in (("N2", 2.074), ("CO", 2.132)):
        mol = build_second_row_diatomic(
            system,
            bond_length_bohr=bond_length,
            basis="sto-3g",
        )
        channels = diatomic_axial_ao_channels(mol)
        assert channels["sigma"].size == 6
        assert channels["pi"].size == 4
        result = evaluate_uniform_magnetic_matrices(
            mol,
            UniformMagneticGauge(field),
            grid_level=1,
            block_size=1024,
        )
        correction = result.barred.mechanical - result.bare.mechanical
        mapping = AOAnchors.from_mol(mol).ao_to_atom
        onsite_norms[system] = [
            np.linalg.norm(correction[np.ix_(mapping == atom, mapping == atom)])
            for atom in range(2)
        ]
        sigma_norm = np.linalg.norm(
            correction[np.ix_(channels["sigma"], channels["sigma"])]
        )
        pi_norm = np.linalg.norm(
            correction[np.ix_(channels["pi"], channels["pi"])]
        )
        assert pi_norm > 20.0 * sigma_norm

        x_indices = np.asarray(
            [
                index
                for index, label in enumerate(mol.ao_labels(fmt=False))
                if label[3] == "x"
            ]
        )
        y_indices = np.asarray(
            [
                index
                for index, label in enumerate(mol.ao_labels(fmt=False))
                if label[3] == "y"
            ]
        )
        np.testing.assert_allclose(
            result.barred.mechanical[np.ix_(x_indices, x_indices)],
            result.barred.mechanical[np.ix_(y_indices, y_indices)],
            atol=5.0e-12,
        )

    np.testing.assert_allclose(
        onsite_norms["N2"][0],
        onsite_norms["N2"][1],
        atol=5.0e-13,
    )
    assert abs(onsite_norms["CO"][0] - onsite_norms["CO"][1]) > 1.0e-5


def test_analytic_finite_field_overlap_is_independent_of_grid_and_gauge():
    geometry = equilateral_h3_geometry()
    normal, area = _normal_and_area(geometry)
    field = (0.8 / area) * normal
    mol = build_h3_framework(geometry, basis="cc-pvdz")
    symmetric = UniformMagneticGauge(field)
    landau = UniformMagneticGauge(
        field,
        gauge="landau",
        origin=np.array([-0.17, 0.29, -0.13]),
        landau_u=np.array([1.0, 0.0, 0.0]),
    )

    analytic = analytic_uniform_wilson_overlap(mol, symmetric)
    analytic_landau = analytic_uniform_wilson_overlap(mol, landau)
    np.testing.assert_allclose(
        analytic.barred,
        analytic_landau.barred,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        analytic.atom_pair_wavevectors,
        -np.swapaxes(analytic.atom_pair_wavevectors, 0, 1),
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        analytic.lower,
        analytic.lower.conj().T,
        atol=2.0e-14,
    )

    grid_exact = evaluate_uniform_magnetic_matrices(
        mol,
        symmetric,
        grid_level=2,
        block_size=1024,
    )
    np.testing.assert_allclose(
        grid_exact.barred.overlap,
        analytic.barred,
        atol=5.0e-5,
    )
    np.testing.assert_allclose(
        grid_exact.direct.overlap,
        analytic.lower,
        atol=5.0e-5,
    )

    zero = analytic_uniform_wilson_overlap(
        mol,
        UniformMagneticGauge(np.zeros(3)),
    )
    np.testing.assert_allclose(
        zero.lower,
        mol.intor_symmetric("int1e_ovlp"),
        atol=2.0e-14,
    )

    analytic_b1 = analytic_b1_one_electron_matrices(mol, symmetric)
    grid_b1 = evaluate_uniform_magnetic_taylor_matrices(
        mol,
        symmetric,
        max_order=1,
        grid_level=2,
        block_size=1024,
    ).b1_one_electron_lower
    for name in ("overlap", "kinetic", "potential", "mechanical"):
        np.testing.assert_allclose(
            getattr(grid_b1, name),
            getattr(analytic_b1, name),
            atol=7.0e-5,
        )


def test_eigenbranch_tracking_follows_states_through_sorted_order_change():
    parameters = (-1.0, -0.4, 0.4, 1.0)
    matrices = [np.diag([parameter, -parameter]) for parameter in parameters]
    tracking = track_hermitian_eigenbranches(matrices)
    np.testing.assert_allclose(
        tracking.eigenvalues[:, 0],
        parameters,
        atol=0.0,
    )
    np.testing.assert_allclose(
        tracking.eigenvalues[:, 1],
        -np.asarray(parameters),
        atol=0.0,
    )
    np.testing.assert_allclose(tracking.adjacent_overlaps, 1.0, atol=0.0)
    assert tracking.ambiguous_steps == ()


def test_generalized_eigenbranch_tracking_uses_a_common_metric():
    comparison_metric = np.diag([2.0, 3.0])
    spectra = []
    for parameter in (-1.0, -0.4, 0.4, 1.0):
        matrix = np.diag([parameter, -parameter])
        spectra.append(
            generalized_hermitian_spectrum(
                matrix,
                np.eye(2),
                metric_floor=0.0,
            )
        )
    tracking = track_generalized_eigenbranches(
        spectra,
        comparison_metric,
    )
    np.testing.assert_allclose(
        tracking.eigenvalues[:, 0],
        (-1.0, -0.4, 0.4, 1.0),
        atol=0.0,
    )
    np.testing.assert_allclose(
        tracking.eigenvalues[:, 1],
        (1.0, 0.4, -0.4, -1.0),
        atol=0.0,
    )
    np.testing.assert_allclose(tracking.adjacent_overlaps, 1.0, atol=0.0)


def test_p0_and_b1_lose_loop_metric_positivity_while_exact_gram_remains_positive():
    geometry = equilateral_h3_geometry()
    normal, area = _normal_and_area(geometry)
    mol = build_h3_framework(geometry)
    phases = np.array([0.0, 1.4, 1.6, 2.0])
    fields = phases / area
    p0_minima = []
    for field in fields:
        p0 = p0_one_electron_matrices(
            mol,
            UniformMagneticGauge(field * normal),
        )
        p0_minima.append(metric_diagnostics(p0.overlap).minimum_eigenvalue)
    bracket = first_positivity_loss_bracket(fields, p0_minima)
    assert bracket is not None
    assert bracket.lower_field == pytest.approx(1.6 / area)
    assert bracket.upper_field == pytest.approx(2.0 / area)

    field = 2.0 / area
    grid = build_unpruned_grid(mol, level=1)
    exact = evaluate_uniform_magnetic_matrices(
        mol,
        UniformMagneticGauge(field * normal),
        grid=grid,
        block_size=1024,
    )
    b1 = evaluate_uniform_magnetic_taylor_matrices(
        mol,
        UniformMagneticGauge(field * normal),
        max_order=1,
        grid=grid,
        block_size=1024,
    )
    assert metric_diagnostics(exact.direct.overlap).minimum_eigenvalue > 0.4
    assert (
        metric_diagnostics(
            exact.diagnostics.p0.lower.overlap
        ).minimum_eigenvalue
        < 0.0
    )
    assert metric_diagnostics(b1.b1_one_electron_lower.overlap).minimum_eigenvalue < 0.0
    with pytest.raises(MetricNotPositiveDefiniteError):
        generalized_hermitian_spectrum(
            exact.diagnostics.p0.lower.mechanical,
            exact.diagnostics.p0.lower.overlap,
            metric_floor=1.0e-10,
        )


def test_three_center_gauge_congruence_and_low_energy_subspaces():
    geometry = distorted_h3_geometry()
    normal, area = _normal_and_area(geometry)
    field = (0.8 / area) * normal
    mol = build_h3_framework(geometry)
    grid = build_unpruned_grid(mol, level=1)
    source_gauge = UniformMagneticGauge(
        field,
        gauge="symmetric",
        origin=np.array([0.11, -0.23, 0.31]),
    )
    target_gauge = UniformMagneticGauge(
        field,
        gauge="landau",
        origin=np.array([-0.17, 0.29, -0.13]),
        landau_u=np.array([1.0, 0.0, 0.0]),
    )
    source = evaluate_uniform_magnetic_matrices(
        mol,
        source_gauge,
        grid=grid,
        block_size=1024,
    )
    target = evaluate_uniform_magnetic_matrices(
        mol,
        target_gauge,
        grid=grid,
        block_size=1024,
    )
    congruence = affine_gauge_endpoint_congruence(
        source_gauge,
        target_gauge,
        source.ao_anchors,
    )
    for name in ("overlap", "mechanical"):
        source_matrix = getattr(source.direct, name)
        target_matrix = getattr(target.direct, name)
        np.testing.assert_allclose(
            target_matrix,
            congruence @ source_matrix @ congruence.conj().T,
            atol=2.0e-12,
        )

    b1 = evaluate_uniform_magnetic_taylor_matrices(
        mol,
        source_gauge,
        max_order=1,
        grid=grid,
        block_size=1024,
    )
    exact_spectrum = generalized_hermitian_spectrum(
        source.direct.mechanical,
        source.direct.overlap,
        metric_floor=1.0e-10,
    )
    for model in (source.diagnostics.p0.lower, b1.b1_one_electron_lower):
        model_spectrum = generalized_hermitian_spectrum(
            model.mechanical,
            model.overlap,
            metric_floor=1.0e-10,
        )
        comparison = compare_spectral_subspaces(
            exact_spectrum.eigenvectors[:, :1],
            model_spectrum.eigenvectors[:, :1],
            source.S0,
            metric_floor=1.0e-12,
        )
        assert np.isfinite(comparison.largest_principal_angle)
        assert comparison.largest_principal_angle < 0.5


def test_rigid_rotation_preserves_multicenter_spectra_and_block_singular_values():
    geometry = distorted_h3_geometry()
    normal, area = _normal_and_area(geometry)
    field = (0.7 / area) * normal
    rotation = _axis_angle_rotation(np.array([0.37, -0.52, 0.77]), 0.61)
    reference_mol = build_h3_framework(geometry, basis="cc-pvdz")
    rotated_mol = build_h3_framework(geometry @ rotation.T, basis="cc-pvdz")
    reference = evaluate_uniform_magnetic_matrices(
        reference_mol,
        UniformMagneticGauge(field),
        grid_level=2,
        block_size=1024,
    )
    rotated = evaluate_uniform_magnetic_matrices(
        rotated_mol,
        UniformMagneticGauge(rotation @ field),
        grid_level=2,
        block_size=1024,
    )
    reference_spectrum = generalized_hermitian_spectrum(
        reference.direct.mechanical,
        reference.direct.overlap,
        metric_floor=1.0e-10,
    )
    rotated_spectrum = generalized_hermitian_spectrum(
        rotated.direct.mechanical,
        rotated.direct.overlap,
        metric_floor=1.0e-10,
    )
    np.testing.assert_allclose(
        rotated_spectrum.eigenvalues,
        reference_spectrum.eigenvalues,
        atol=7.0e-5,
    )

    reference_mapping = AOAnchors.from_mol(reference_mol).ao_to_atom
    rotated_mapping = AOAnchors.from_mol(rotated_mol).ao_to_atom
    for atom_a, atom_b in ((0, 0), (0, 1), (1, 2)):
        reference_block = (
            reference.barred.mechanical - reference.bare.mechanical
        )[np.ix_(reference_mapping == atom_a, reference_mapping == atom_b)]
        rotated_block = (
            rotated.barred.mechanical - rotated.bare.mechanical
        )[np.ix_(rotated_mapping == atom_a, rotated_mapping == atom_b)]
        np.testing.assert_allclose(
            np.linalg.svd(rotated_block, compute_uv=False),
            np.linalg.svd(reference_block, compute_uv=False),
            atol=7.0e-5,
        )
