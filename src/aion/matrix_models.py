"""Small Hermitian matrix functionals for gauge-geometry tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gauge import PeierlsGeometry


def _array_module(value):
    try:
        import cupy
    except Exception:
        return np
    return cupy.get_array_module(value)


def density_from_coefficients(coeff: np.ndarray, occupations: np.ndarray) -> np.ndarray:
    xp = _array_module(coeff)
    coeff = xp.asarray(coeff, dtype=xp.complex128)
    occ = np.asarray(occupations, dtype=float)
    if coeff.ndim != 2:
        raise ValueError("coeff must be a two-dimensional array")
    if occ.shape != (coeff.shape[1],):
        raise ValueError("occupations must have one entry per occupied column")
    occ_backend = xp.asarray(occ, dtype=float)
    return (coeff * occ_backend[None, :]) @ coeff.conj().T


def hermitian_part(matrix: np.ndarray) -> np.ndarray:
    xp = _array_module(matrix)
    matrix = xp.asarray(matrix, dtype=xp.complex128)
    return 0.5 * (matrix + matrix.conj().T)


def _matrix_expectation(density: np.ndarray, matrix: np.ndarray) -> float:
    return float(np.trace(np.asarray(density, dtype=np.complex128) @ matrix).real)


def site_population_operators(geometry: PeierlsGeometry, t: float) -> list[np.ndarray]:
    """Return ``0.5 * {M_a, S(t)}`` for every atom site."""

    metric = geometry.metric(t)
    operators = []
    for atom in range(geometry.anchors.natom):
        diag = geometry.anchors.site_projector_diagonal(atom)
        left = diag[:, None] * metric
        right = metric * diag[None, :]
        operators.append(0.5 * (left + right))
    return operators


def site_populations(
    density: np.ndarray,
    geometry: PeierlsGeometry,
    t: float,
) -> np.ndarray:
    """Return P0 Mulliken/source site populations without the charge factor."""

    rho = np.asarray(density, dtype=np.complex128)
    populations = [
        float(np.trace(rho @ op).real)
        for op in site_population_operators(geometry, t)
    ]
    return np.asarray(populations)


@dataclass(frozen=True)
class LinearOneBodyModel:
    """Peierls-dressed one-body matrix model."""

    hcore0: np.ndarray

    def __post_init__(self) -> None:
        hcore = np.asarray(self.hcore0, dtype=np.complex128)
        if hcore.ndim != 2 or hcore.shape[0] != hcore.shape[1]:
            raise ValueError("hcore0 must be a square matrix")
        object.__setattr__(self, "hcore0", hermitian_part(hcore))

    def hamiltonian(
        self,
        _density: np.ndarray,
        t: float,
        geometry: PeierlsGeometry,
    ) -> np.ndarray:
        return hermitian_part(geometry.dress_matrix(self.hcore0, t))

    def energy(
        self,
        density: np.ndarray,
        t: float,
        geometry: PeierlsGeometry,
    ) -> float:
        return _matrix_expectation(density, self.hamiltonian(density, t, geometry))


@dataclass(frozen=True)
class SiteHubbardModel:
    """SCC/Hubbard-like site-charge model for nonlinear SCEM tests."""

    hcore0: np.ndarray
    hubbard_u: np.ndarray | float
    reference_populations: np.ndarray

    def __post_init__(self) -> None:
        hcore = np.asarray(self.hcore0, dtype=np.complex128)
        if hcore.ndim != 2 or hcore.shape[0] != hcore.shape[1]:
            raise ValueError("hcore0 must be a square matrix")
        u = np.asarray(self.hubbard_u, dtype=float)
        ref = np.asarray(self.reference_populations, dtype=float)
        if ref.ndim != 1:
            raise ValueError("reference_populations must be one-dimensional")
        if u.ndim == 0:
            u = np.full(ref.shape, float(u))
        if u.shape != ref.shape:
            raise ValueError("hubbard_u must be scalar or match reference_populations")
        object.__setattr__(self, "hcore0", hermitian_part(hcore))
        object.__setattr__(self, "hubbard_u", u)
        object.__setattr__(self, "reference_populations", ref)

    def hamiltonian(
        self,
        density: np.ndarray,
        t: float,
        geometry: PeierlsGeometry,
    ) -> np.ndarray:
        if self.reference_populations.shape != (geometry.anchors.natom,):
            raise ValueError("reference_populations must match geometry atom count")
        h = geometry.dress_matrix(self.hcore0, t)
        populations = site_populations(density, geometry, t)
        operators = site_population_operators(geometry, t)
        for coeff, op in zip(
            self.hubbard_u * (populations - self.reference_populations),
            operators,
        ):
            h = h + coeff * op
        return hermitian_part(h)

    def energy(
        self,
        density: np.ndarray,
        t: float,
        geometry: PeierlsGeometry,
    ) -> float:
        if self.reference_populations.shape != (geometry.anchors.natom,):
            raise ValueError("reference_populations must match geometry atom count")
        hcore = geometry.dress_matrix(self.hcore0, t)
        populations = site_populations(density, geometry, t)
        delta = populations - self.reference_populations
        return _matrix_expectation(density, hcore) + 0.5 * float(
            np.sum(self.hubbard_u * delta**2)
        )
