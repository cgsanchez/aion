"""Observable helpers for gauge-covariant finite-basis dynamics."""

from __future__ import annotations

import numpy as np

from .gauge import PeierlsGeometry
from .matrix_models import site_populations


def _square_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return array


def matrix_expectation(density: np.ndarray, matrix: np.ndarray) -> float:
    """Return ``Re Tr[rho matrix]`` for the current AO representation."""

    rho = _square_matrix(density, name="density")
    op = _square_matrix(matrix, name="matrix")
    if rho.shape != op.shape:
        raise ValueError("density and matrix must be square arrays with the same shape")
    return float(np.trace(rho @ op).real)


def electronic_energy(density: np.ndarray, hamiltonian: np.ndarray) -> float:
    """Return the one-body electronic energy associated with ``hamiltonian``."""

    return matrix_expectation(density, hamiltonian)


def electron_count(density: np.ndarray, metric: np.ndarray) -> float:
    """Return ``Tr[rho S]`` for a density matrix and metric/overlap matrix."""

    return matrix_expectation(density, metric)


def coefficient_orthonormality_error(
    coeff: np.ndarray,
    metric: np.ndarray,
    *,
    target: np.ndarray | None = None,
) -> float:
    """Return the Frobenius norm of ``C^dagger S C - target``."""

    c = np.asarray(coeff, dtype=np.complex128)
    s = np.asarray(metric, dtype=np.complex128)
    if c.ndim != 2:
        raise ValueError("coeff must be a two-dimensional array")
    if s.shape != (c.shape[0], c.shape[0]):
        raise ValueError("metric must match the AO dimension of coeff")
    if target is None:
        target = np.eye(c.shape[1], dtype=np.complex128)
    target = np.asarray(target, dtype=np.complex128)
    if target.shape != (c.shape[1], c.shape[1]):
        raise ValueError("target must match the occupied-orbital dimension")
    return float(np.linalg.norm(c.conj().T @ s @ c - target))


def p0_site_populations(
    density: np.ndarray,
    geometry: PeierlsGeometry,
    t: float,
) -> np.ndarray:
    """Return P0 Mulliken/source populations without the charge factor."""

    return site_populations(density, geometry, t)


def p0_site_charges(
    density: np.ndarray,
    geometry: PeierlsGeometry,
    t: float,
    *,
    charge: float | None = None,
) -> np.ndarray:
    """Return source charges ``q N_a`` conjugate to P0 site scalar sources."""

    q = geometry.charge if charge is None else float(charge)
    return q * p0_site_populations(density, geometry, t)


def p0_dipole_moment(
    density: np.ndarray,
    geometry: PeierlsGeometry,
    t: float,
    *,
    charge: float | None = None,
) -> np.ndarray:
    """Return the P0 charge-weighted source dipole ``q sum_a N_a R_a``.

    The default charge is ``geometry.charge``.  For electronic dynamics this is
    normally ``-1`` in atomic units, so this function returns the electronic
    contribution to the dipole.  Nuclear terms, if desired, should be added by
    the caller because their charges are model-specific.
    """

    q = geometry.charge if charge is None else float(charge)
    populations = p0_site_populations(density, geometry, t)
    return q * np.einsum("a,ax->x", populations, geometry.atom_coords)


def p0_directed_block(
    matrix: np.ndarray,
    geometry: PeierlsGeometry,
    row_atom: int,
    col_atom: int,
) -> np.ndarray:
    """Return the AO matrix containing only one directed atom-pair block."""

    if row_atom < 0 or row_atom >= geometry.anchors.natom:
        raise ValueError("row_atom out of range")
    if col_atom < 0 or col_atom >= geometry.anchors.natom:
        raise ValueError("col_atom out of range")
    source = _square_matrix(matrix, name="matrix")
    if source.shape != geometry.overlap0.shape:
        raise ValueError(f"matrix must have shape {geometry.overlap0.shape}")
    block = np.zeros_like(source)
    rows = geometry.anchors.ao_to_atom == row_atom
    cols = geometry.anchors.ao_to_atom == col_atom
    block[np.ix_(rows, cols)] = source[np.ix_(rows, cols)]
    return block


def p0_graph_currents(
    density: np.ndarray,
    hamiltonian: np.ndarray,
    geometry: PeierlsGeometry,
    t: float,
    *,
    covariant_metric_dot: np.ndarray | None = None,
) -> np.ndarray:
    """Return the antisymmetric P0 source graph current matrix.

    The current is the source derivative with respect to the oriented bond
    source ``Acal_ab`` and satisfies the Ward identity
    ``dQ_a/dt + sum_b I_ab = 0`` in the continuous P0 equations of motion.

    For a pair of sites ``a != b`` the implemented density-matrix form is

    ``I_ab = q/hbar Im Tr[rho(S_ab S^-1 K + K^dag S^-1 S_ab)]``
    ``      + 2q/hbar Im Tr[rho H_ba]``,

    with ``K = H - i hbar/2 D_t S``.  In an orthogonal basis the overlap-block
    term vanishes and this reduces to the usual bond current
    ``2q/hbar Im Tr[rho H_ba]``.
    """

    rho = _square_matrix(density, name="density")
    h = _square_matrix(hamiltonian, name="hamiltonian")
    if rho.shape != geometry.overlap0.shape or h.shape != geometry.overlap0.shape:
        raise ValueError(f"density and hamiltonian must have shape {geometry.overlap0.shape}")

    s = geometry.metric(t)
    s_inv = np.linalg.inv(s)
    ds_cov = (
        geometry.covariant_metric_dot(t)
        if covariant_metric_dot is None
        else _square_matrix(covariant_metric_dot, name="covariant_metric_dot")
    )
    if ds_cov.shape != geometry.overlap0.shape:
        raise ValueError(f"covariant_metric_dot must have shape {geometry.overlap0.shape}")
    k = h - 0.5j * geometry.hbar * ds_cov

    natom = geometry.anchors.natom
    currents = np.zeros((natom, natom), dtype=float)
    prefactor = geometry.charge / geometry.hbar
    for a in range(natom):
        for b in range(natom):
            if a == b:
                continue
            s_ab = p0_directed_block(s, geometry, a, b)
            h_ba = p0_directed_block(h, geometry, b, a)
            overlap_term = np.trace(rho @ (s_ab @ s_inv @ k + k.conj().T @ s_inv @ s_ab))
            hamiltonian_term = np.trace(rho @ h_ba)
            currents[a, b] = (
                prefactor * overlap_term.imag
                + 2.0 * prefactor * hamiltonian_term.imag
            )
    return 0.5 * (currents - currents.T)


def p0_continuity_residual(
    charge_derivative: np.ndarray,
    graph_currents: np.ndarray,
) -> np.ndarray:
    """Return ``dQ_a/dt + sum_b I_ab`` for P0 source charges/currents."""

    dqdt = np.asarray(charge_derivative, dtype=float)
    currents = np.asarray(graph_currents, dtype=float)
    if currents.ndim != 2 or currents.shape[0] != currents.shape[1]:
        raise ValueError("graph_currents must be a square matrix")
    if dqdt.shape != (currents.shape[0],):
        raise ValueError("charge_derivative must match graph_currents dimension")
    return dqdt + np.sum(currents, axis=1)
