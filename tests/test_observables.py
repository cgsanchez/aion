from __future__ import annotations

import numpy as np
import scipy.linalg

from aion.gauge import AOAnchors, PeierlsGeometry
from aion.matrix_models import density_from_coefficients
from aion.observables import (
    coefficient_orthonormality_error,
    electron_count,
    electronic_energy,
    matrix_expectation,
    p0_dipole_moment,
    p0_site_populations,
)


def _orthonormal_coefficients(metric: np.ndarray, nocc: int) -> np.ndarray:
    eig, vec = scipy.linalg.eigh(metric, check_finite=False)
    invsqrt = (vec * eig**-0.5) @ vec.conj().T
    trial = np.array(
        [
            [1.0, 0.2],
            [0.1, 1.1],
            [0.4, -0.3],
        ],
        dtype=np.complex128,
    )
    q, _ = np.linalg.qr(trial)
    return invsqrt @ q[:, :nocc]


def test_p0_populations_sum_to_metric_trace_and_dipole_uses_charge_sign():
    anchors = AOAnchors(
        atom_coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.2, 0.1, 0.0],
            ]
        ),
        ao_to_atom=np.array([0, 0, 1]),
    )
    s0 = np.array(
        [
            [1.0, 0.12, 0.05],
            [0.12, 1.0, -0.03],
            [0.05, -0.03, 1.0],
        ],
        dtype=np.complex128,
    )
    geometry = PeierlsGeometry(anchors, s0)
    occupations = np.array([2.0, 1.0])
    coeff = _orthonormal_coefficients(s0, occupations.size)
    rho = density_from_coefficients(coeff, occupations)

    populations = p0_site_populations(rho, geometry, 0.0)
    assert np.isclose(np.sum(populations), electron_count(rho, s0), atol=1.0e-12)
    assert np.isclose(np.sum(populations), np.sum(occupations), atol=1.0e-12)

    dipole = p0_dipole_moment(rho, geometry, 0.0)
    expected = geometry.charge * np.einsum("a,ax->x", populations, anchors.atom_coords)
    assert np.linalg.norm(dipole - expected) < 1.0e-14


def test_matrix_expectation_and_energy_match_explicit_orbital_sum():
    anchors = AOAnchors(
        atom_coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ao_to_atom=np.array([0, 1, 1]),
    )
    s0 = np.array(
        [
            [1.0, 0.06, 0.02],
            [0.06, 1.0, 0.04],
            [0.02, 0.04, 1.0],
        ],
        dtype=np.complex128,
    )
    h = np.array(
        [
            [-0.7, -0.1, 0.03],
            [-0.1, -0.2, -0.08],
            [0.03, -0.08, 0.1],
        ],
        dtype=np.complex128,
    )
    occupations = np.array([2.0, 0.5])
    coeff = _orthonormal_coefficients(s0, occupations.size)
    rho = density_from_coefficients(coeff, occupations)

    explicit = 0.0
    for orbital, occupation in enumerate(occupations):
        explicit += occupation * (coeff[:, orbital].conj().T @ h @ coeff[:, orbital]).real

    assert np.isclose(matrix_expectation(rho, h), explicit, atol=1.0e-13)
    assert np.isclose(electronic_energy(rho, h), explicit, atol=1.0e-13)
    assert coefficient_orthonormality_error(coeff, s0) < 1.0e-13
