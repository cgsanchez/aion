from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

pyscf = pytest.importorskip("pyscf")
from pyscf import gto

from aion.gauge import UniformMagneticGauge
from aion.reference import (
    build_unpruned_grid,
    evaluate_uniform_magnetic_matrices,
    uniform_triangle_flux,
)


_FIELD = np.array([0.09, -0.07, 0.11])
_ORIGIN_1 = np.array([0.17, -0.31, 0.23])
_ORIGIN_2 = np.array([-0.21, 0.37, -0.16])
_ALGEBRAIC_ATOL = 2.0e-11


def _h_h_molecule():
    return gto.M(
        atom=[
            ("H", (0.25, -0.40, 0.15)),
            ("H", (1.32, 0.51, -0.27)),
        ],
        basis="sto-3g",
        unit="Bohr",
        spin=0,
        verbose=0,
    )


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
def oblique_grid(oh_molecule):
    return build_unpruned_grid(oh_molecule, level=1)


@pytest.fixture(scope="module")
def oblique_result(oh_molecule, oblique_grid):
    gauge = UniformMagneticGauge(
        _FIELD,
        gauge="symmetric",
        origin=_ORIGIN_1,
    )
    return evaluate_uniform_magnetic_matrices(
        oh_molecule,
        gauge,
        grid=oblique_grid,
        block_size=257,
    )


def _matrix_names():
    return ("overlap", "kinetic", "potential", "mechanical")


def _endpoint_gauge_congruence(
    source: UniformMagneticGauge,
    target: UniformMagneticGauge,
    anchors: np.ndarray,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> np.ndarray:
    """Return G for ``A_target = A_source + grad Lambda``.

    The irrelevant additive constant in the affine-gauge transition function
    is fixed to zero.
    """

    np.testing.assert_allclose(
        target.magnetic_field,
        source.magnetic_field,
        atol=0.0,
        rtol=0.0,
    )
    quadratic = target.affine_matrix - source.affine_matrix
    np.testing.assert_allclose(quadratic, quadratic.T, atol=5.0e-16, rtol=0.0)
    linear = (
        -target.affine_matrix @ target.origin
        + source.affine_matrix @ source.origin
    )
    gauge_function = (
        0.5 * np.einsum("ix,xy,iy->i", anchors, quadratic, anchors)
        + anchors @ linear
    )
    return np.diag(np.exp((1j * charge / hbar) * gauge_function))


def test_zero_field_grid_converges_to_pyscf_integrals():
    mol = _h_h_molecule()
    zero_gauge = UniformMagneticGauge(np.zeros(3))
    errors = {name: [] for name in _matrix_names()}

    # Levels 1 -> 2 -> 3 are two genuine refinements of the same unpruned
    # atom-centered grid construction.
    for level in (1, 2, 3):
        result = evaluate_uniform_magnetic_matrices(
            mol,
            zero_gauge,
            grid_level=level,
            block_size=1024,
        )
        np.testing.assert_allclose(result.theta, 1.0, atol=0.0, rtol=0.0)
        for name in _matrix_names():
            direct = getattr(result.direct, name)
            barred = getattr(result.barred, name)
            factorized = getattr(result.factorized, name)
            analytic = getattr(result.bare, name)
            assert np.linalg.norm(direct - barred) < _ALGEBRAIC_ATOL
            assert np.linalg.norm(direct - factorized) < _ALGEBRAIC_ATOL
            errors[name].append(np.linalg.norm(direct - analytic))

    for name, sequence in errors.items():
        assert sequence[2] < sequence[1] < sequence[0], (name, sequence)
        assert sequence[2] < 5.0e-8, (name, sequence)


def test_direct_factorized_pair_reversal_and_kinetic_sector_identities(
    oblique_result,
):
    result = oblique_result

    np.testing.assert_allclose(
        result.theta,
        result.theta.conj().T,
        atol=5.0e-15,
        rtol=0.0,
    )
    np.testing.assert_allclose(np.abs(result.theta), 1.0, atol=5.0e-15, rtol=0.0)

    for matrices in (result.bare, result.direct, result.barred, result.factorized):
        for name in _matrix_names():
            matrix = getattr(matrices, name)
            assert np.linalg.norm(matrix - matrix.conj().T) < _ALGEBRAIC_ATOL

    for name in _matrix_names():
        direct = getattr(result.direct, name)
        factorized = getattr(result.factorized, name)
        assert np.linalg.norm(direct - factorized) < _ALGEBRAIC_ATOL

    sectors = result.kinetic_sectors
    assert np.linalg.norm(sectors.pp - sectors.pp.conj().T) < _ALGEBRAIC_ATOL
    assert np.linalg.norm(sectors.c_c - sectors.c_c.conj().T) < _ALGEBRAIC_ATOL
    assert np.linalg.norm(sectors.p_c - sectors.c_p.conj().T) < _ALGEBRAIC_ATOL
    assert np.linalg.norm(sectors.total - result.barred.kinetic) < _ALGEBRAIC_ATOL

    sample_points = np.array(
        [
            (0.21, 0.37, -0.19),
            (-0.83, 0.42, 0.71),
            (1.44, -0.32, 0.08),
        ]
    )
    flux = uniform_triangle_flux(
        sample_points,
        result.ao_anchors,
        result.gauge.magnetic_field,
    )
    triangle_factor = np.exp((1j * result.charge / result.hbar) * flux)
    np.testing.assert_allclose(flux, -flux.swapaxes(1, 2), atol=5.0e-16, rtol=0.0)
    np.testing.assert_allclose(
        triangle_factor,
        triangle_factor.swapaxes(1, 2).conj(),
        atol=5.0e-15,
        rtol=0.0,
    )


def test_exact_gram_matrix_is_positive_with_resolved_eigenvalue_margin(
    oblique_result,
):
    overlap = oblique_result.direct.overlap
    eigenvalues = np.linalg.eigvalsh(overlap)
    condition_number = float(eigenvalues[-1] / eigenvalues[0])
    eigenvalue_floor = (
        np.finfo(float).eps * overlap.shape[0] * np.linalg.norm(overlap, ord=2)
    )

    assert eigenvalues[0] > 1.0e10 * eigenvalue_floor
    assert np.isfinite(condition_number)
    assert condition_number < 10.0


def test_same_anchor_pairs_have_unit_links_but_nonzero_kinetic_correction(
    oh_molecule,
    oblique_result,
):
    oxygen_position = oh_molecule.atom_coords(unit="Bohr")[0]
    same_anchor = np.flatnonzero(
        np.linalg.norm(oblique_result.ao_anchors - oxygen_position, axis=1) < 1.0e-14
    )
    assert same_anchor.size > 1
    block = np.ix_(same_anchor, same_anchor)

    np.testing.assert_allclose(
        oblique_result.theta[block],
        1.0,
        atol=0.0,
        rtol=0.0,
    )
    sample_points = np.array(
        [
            (0.91, -0.14, 0.32),
            (-0.28, 0.63, -0.47),
        ]
    )
    same_anchor_flux = uniform_triangle_flux(
        sample_points,
        oblique_result.ao_anchors[same_anchor],
        _FIELD,
    )
    np.testing.assert_allclose(same_anchor_flux, 0.0, atol=0.0, rtol=0.0)

    sectors = oblique_result.kinetic_sectors
    anchored_correction = sectors.p_c + sectors.c_p + sectors.c_c
    assert np.linalg.norm(anchored_correction[block]) > 1.0e-3
    np.testing.assert_allclose(
        (oblique_result.barred.kinetic - sectors.pp)[block],
        anchored_correction[block],
        atol=_ALGEBRAIC_ATOL,
        rtol=0.0,
    )


def test_field_parallel_to_bond_has_unit_triangle_factor_and_kinetic_signal():
    mol = _h_h_molecule()
    atom_coords = mol.atom_coords(unit="Bohr")
    bond = atom_coords[1] - atom_coords[0]
    field = 0.16 * bond / np.linalg.norm(bond)
    result = evaluate_uniform_magnetic_matrices(
        mol,
        UniformMagneticGauge(field, origin=_ORIGIN_1),
        grid_level=1,
        block_size=257,
    )

    sample_points = np.array(
        [
            (0.20, 0.30, 0.40),
            (-1.10, 0.70, 0.90),
            (0.73, -0.28, 0.51),
        ]
    )
    flux = uniform_triangle_flux(sample_points, result.ao_anchors, field)
    triangle_factor = np.exp((1j * result.charge / result.hbar) * flux)
    np.testing.assert_allclose(triangle_factor[:, 0, 1], 1.0, atol=5.0e-15, rtol=0.0)

    sectors = result.kinetic_sectors
    anchored_correction = sectors.p_c + sectors.c_p + sectors.c_c
    assert abs(anchored_correction[0, 1]) > 1.0e-4
    np.testing.assert_allclose(
        result.barred.kinetic[0, 1] - sectors.pp[0, 1],
        anchored_correction[0, 1],
        atol=_ALGEBRAIC_ATOL,
        rtol=0.0,
    )


def test_affine_gauge_changes_are_endpoint_congruences_with_equal_spectra(
    oh_molecule,
    oblique_grid,
    oblique_result,
):
    reference_gauge = UniformMagneticGauge(
        _FIELD,
        gauge="symmetric",
        origin=_ORIGIN_1,
    )
    landau_seed = np.array([0.31, 0.27, -0.19])
    landau_u = np.cross(_FIELD, landau_seed)
    landau_u /= np.linalg.norm(landau_u)
    target_gauges = (
        UniformMagneticGauge(
            _FIELD,
            gauge="symmetric",
            origin=_ORIGIN_2,
        ),
        UniformMagneticGauge(
            _FIELD,
            gauge="landau",
            origin=_ORIGIN_2,
            landau_u=landau_u,
        ),
    )

    reference_spectrum = scipy.linalg.eigvalsh(
        oblique_result.direct.mechanical,
        oblique_result.direct.overlap,
        check_finite=True,
    )
    for target_gauge in target_gauges:
        target = evaluate_uniform_magnetic_matrices(
            oh_molecule,
            target_gauge,
            grid=oblique_grid,
            block_size=257,
        )
        congruence = _endpoint_gauge_congruence(
            reference_gauge,
            target_gauge,
            oblique_result.ao_anchors,
            charge=oblique_result.charge,
            hbar=oblique_result.hbar,
        )

        np.testing.assert_allclose(
            target.theta,
            congruence @ oblique_result.theta @ congruence.conj().T,
            atol=_ALGEBRAIC_ATOL,
            rtol=0.0,
        )
        for name in _matrix_names():
            reference_matrix = getattr(oblique_result.direct, name)
            target_matrix = getattr(target.direct, name)
            np.testing.assert_allclose(
                target_matrix,
                congruence @ reference_matrix @ congruence.conj().T,
                atol=_ALGEBRAIC_ATOL,
                rtol=0.0,
            )
            np.testing.assert_allclose(
                getattr(target.barred, name),
                getattr(oblique_result.barred, name),
                atol=_ALGEBRAIC_ATOL,
                rtol=0.0,
            )

        target_spectrum = scipy.linalg.eigvalsh(
            target.direct.mechanical,
            target.direct.overlap,
            check_finite=True,
        )
        np.testing.assert_allclose(
            target_spectrum,
            reference_spectrum,
            atol=_ALGEBRAIC_ATOL,
            rtol=0.0,
        )
