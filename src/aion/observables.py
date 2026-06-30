"""Observable helpers for gauge-covariant finite-basis dynamics."""

from __future__ import annotations

import numpy as np

from .gauge import PeierlsGeometry
from .matrix_models import site_populations


def matrix_expectation(density: np.ndarray, matrix: np.ndarray) -> float:
    """Return ``Re Tr[rho matrix]`` for the current AO representation."""

    rho = np.asarray(density, dtype=np.complex128)
    op = np.asarray(matrix, dtype=np.complex128)
    if rho.shape != op.shape or rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
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
