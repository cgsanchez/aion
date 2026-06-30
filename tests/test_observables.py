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
    p0_continuity_residual,
    p0_dipole_moment,
    p0_graph_currents,
    p0_site_charge_derivative,
    p0_site_charges,
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


def _constant_metric_step(
    coeff: np.ndarray,
    metric: np.ndarray,
    hamiltonian: np.ndarray,
    dt: float,
) -> np.ndarray:
    eig_s, vec_s = scipy.linalg.eigh(metric, check_finite=False)
    s_sqrt = (vec_s * eig_s**0.5) @ vec_s.conj().T
    s_invsqrt = (vec_s * eig_s**-0.5) @ vec_s.conj().T
    h_orth = s_invsqrt @ hamiltonian @ s_invsqrt
    eig_h, vec_h = scipy.linalg.eigh(
        0.5 * (h_orth + h_orth.conj().T),
        check_finite=False,
    )
    coeff_orth = s_sqrt @ coeff
    propagated = vec_h @ (
        np.exp(-1j * dt * eig_h)[:, None] * (vec_h.conj().T @ coeff_orth)
    )
    return s_invsqrt @ propagated


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


def test_p0_graph_currents_reduce_to_orthogonal_bond_current():
    anchors = AOAnchors(
        atom_coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        ao_to_atom=np.array([0, 1, 2]),
    )
    geometry = PeierlsGeometry(anchors, np.eye(3))
    h = np.array(
        [
            [-0.5, -0.2 + 0.03j, 0.04 - 0.01j],
            [-0.2 - 0.03j, -0.1, -0.08 + 0.05j],
            [0.04 + 0.01j, -0.08 - 0.05j, 0.2],
        ],
        dtype=np.complex128,
    )
    coeff = _orthonormal_coefficients(np.eye(3), nocc=2)
    occupations = np.array([1.0, 0.7])
    rho = density_from_coefficients(coeff, occupations)

    currents = p0_graph_currents(rho, h, geometry, 0.0)
    expected = np.zeros_like(currents)
    for a in range(3):
        for b in range(3):
            if a != b:
                expected[a, b] = 2.0 * geometry.charge * np.imag(rho[a, b] * h[b, a])

    assert np.linalg.norm(currents + currents.T) < 1.0e-14
    assert np.linalg.norm(currents - expected) < 1.0e-14


def test_p0_graph_currents_satisfy_nonorthogonal_continuity():
    anchors = AOAnchors(
        atom_coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.1, 0.2, 0.0],
                [-0.2, 0.9, 0.1],
            ]
        ),
        ao_to_atom=np.array([0, 1, 2]),
    )
    s0 = np.array(
        [
            [1.0, 0.08, -0.04 + 0.02j],
            [0.08, 1.0, 0.05],
            [-0.04 - 0.02j, 0.05, 1.0],
        ],
        dtype=np.complex128,
    )
    h = np.array(
        [
            [-0.4, -0.15 + 0.02j, 0.03],
            [-0.15 - 0.02j, -0.05, -0.09 - 0.03j],
            [0.03, -0.09 + 0.03j, 0.12],
        ],
        dtype=np.complex128,
    )
    geometry = PeierlsGeometry(anchors, s0)
    occupations = np.array([1.0, 0.8])
    coeff = _orthonormal_coefficients(s0, occupations.size)
    rho = density_from_coefficients(coeff, occupations)
    currents = p0_graph_currents(rho, h, geometry, 0.0)

    coeff_dot = -1j * np.linalg.solve(s0, h @ coeff)
    rho_dot = (
        (coeff_dot * occupations[None, :]) @ coeff.conj().T
        + (coeff * occupations[None, :]) @ coeff_dot.conj().T
    )
    exact_charge_derivative = p0_site_charge_derivative(
        rho,
        rho_dot,
        geometry,
        0.0,
    )
    exact_residual = p0_continuity_residual(exact_charge_derivative, currents)

    dt = 1.0e-5
    coeff_plus = _constant_metric_step(coeff, s0, h, dt)
    coeff_minus = _constant_metric_step(coeff, s0, h, -dt)
    rho_plus = density_from_coefficients(coeff_plus, occupations)
    rho_minus = density_from_coefficients(coeff_minus, occupations)
    charge_derivative = (
        p0_site_charges(rho_plus, geometry, 0.0)
        - p0_site_charges(rho_minus, geometry, 0.0)
    ) / (2.0 * dt)
    residual = p0_continuity_residual(charge_derivative, currents)

    assert np.linalg.norm(currents + currents.T) < 1.0e-14
    assert np.linalg.norm(exact_residual) < 1.0e-14
    assert np.linalg.norm(charge_derivative - exact_charge_derivative) < 1.0e-9
    assert np.linalg.norm(residual) < 1.0e-9
