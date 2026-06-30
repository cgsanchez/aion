"""Pure Peierls P0 gauge geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anchors import AOAnchors
from .sources import UniformElectricGauge, UniformMagneticGauge


@dataclass(frozen=True)
class PeierlsGeometry:
    """Pure Peierls geometry for an atom-anchored AO basis."""

    anchors: AOAnchors
    overlap0: np.ndarray
    electric: UniformElectricGauge | None = None
    magnetic: UniformMagneticGauge | None = None
    charge: float = -1.0
    hbar: float = 1.0

    def __post_init__(self) -> None:
        s0 = np.asarray(self.overlap0, dtype=np.complex128)
        if s0.shape != (self.anchors.nao, self.anchors.nao):
            raise ValueError(f"overlap0 must have shape {(self.anchors.nao, self.anchors.nao)}")
        if self.hbar <= 0.0:
            raise ValueError("hbar must be positive")
        object.__setattr__(self, "overlap0", s0)

    @property
    def atom_coords(self) -> np.ndarray:
        return self.anchors.atom_coords

    def site_scalar_potential(self, t: float) -> np.ndarray:
        phi = np.zeros(self.anchors.natom, dtype=float)
        if self.electric is not None:
            phi = phi + self.electric.site_scalar_potential(self.atom_coords, t)
        if self.magnetic is not None:
            phi = phi + self.magnetic.site_scalar_potential(self.atom_coords, t)
        return phi

    def site_scalar_integral(self, t0: float, t1: float) -> np.ndarray:
        integral = np.zeros(self.anchors.natom, dtype=float)
        if self.electric is not None:
            integral = integral + self.electric.site_scalar_integral(
                self.atom_coords,
                t0,
                t1,
            )
        if self.magnetic is not None:
            integral = integral + self.magnetic.site_scalar_integral(
                self.atom_coords,
                t0,
                t1,
            )
        return integral

    def site_sigma(self, t: float) -> np.ndarray:
        return (1j * self.charge / self.hbar) * self.site_scalar_potential(t)

    def ao_sigma(self, t: float) -> np.ndarray:
        return self.anchors.lift_site_vector(self.site_sigma(t))

    def site_bond_line_integrals(self, t: float) -> np.ndarray:
        acal = np.zeros((self.anchors.natom, self.anchors.natom), dtype=float)
        if self.electric is not None:
            acal = acal + self.electric.bond_line_integrals(self.atom_coords, t)
        if self.magnetic is not None:
            acal = acal + self.magnetic.bond_line_integrals(self.atom_coords, t)
        return acal

    def site_bond_line_integral_dots(self, t: float) -> np.ndarray:
        acal_dot = np.zeros((self.anchors.natom, self.anchors.natom), dtype=float)
        if self.electric is not None:
            acal_dot = acal_dot + self.electric.bond_line_integral_dots(
                self.atom_coords, t
            )
        if self.magnetic is not None:
            acal_dot = acal_dot + self.magnetic.bond_line_integral_dots(
                self.atom_coords, t
            )
        return acal_dot

    def site_bond_electromotive_forces(self, t: float) -> np.ndarray:
        emf = np.zeros((self.anchors.natom, self.anchors.natom), dtype=float)
        if self.electric is not None:
            emf = emf + self.electric.bond_electromotive_forces(self.atom_coords, t)
        if self.magnetic is not None:
            emf = emf + self.magnetic.bond_electromotive_forces(self.atom_coords, t)
        return emf

    def site_theta(self, t: float) -> np.ndarray:
        phase = (1j * self.charge / self.hbar) * self.site_bond_line_integrals(t)
        return np.exp(phase)

    def theta(self, t: float) -> np.ndarray:
        return self.anchors.lift_site_matrix(self.site_theta(t))

    def metric(self, t: float) -> np.ndarray:
        return self.theta(t) * self.overlap0

    def dress_matrix(self, matrix0: np.ndarray, t: float) -> np.ndarray:
        matrix = np.asarray(matrix0, dtype=np.complex128)
        if matrix.shape != self.overlap0.shape:
            raise ValueError(f"matrix0 must have shape {self.overlap0.shape}")
        return self.theta(t) * matrix

    def covariant_metric_dot(self, t: float) -> np.ndarray:
        emf = self.anchors.lift_site_matrix(self.site_bond_electromotive_forces(t))
        return (-1j * self.charge / self.hbar) * emf * self.metric(t)

    def ordinary_metric_dot(self, t: float) -> np.ndarray:
        acal_dot = self.anchors.lift_site_matrix(self.site_bond_line_integral_dots(t))
        return (1j * self.charge / self.hbar) * acal_dot * self.metric(t)

    def time_connection(self, t: float) -> np.ndarray:
        metric = self.metric(t)
        sigma = self.ao_sigma(t)
        return metric * sigma[None, :] + 0.5 * self.covariant_metric_dot(t)

    def site_transport(self, t0: float, t1: float) -> np.ndarray:
        """Exact site-parallel transport for the analytic source family."""

        if t1 == t0:
            return np.ones(self.anchors.natom, dtype=np.complex128)
        sigma_integral = (
            1j * self.charge / self.hbar
        ) * self.site_scalar_integral(t0, t1)
        return np.exp(-sigma_integral)

    def ao_transport(self, t0: float, t1: float) -> np.ndarray:
        return self.anchors.lift_site_vector(self.site_transport(t0, t1))

    def transport_matrix(self, matrix: np.ndarray, transport: np.ndarray) -> np.ndarray:
        """Return ``U^dagger matrix U`` for diagonal AO transport ``U``."""

        u = np.asarray(transport, dtype=np.complex128)
        if u.shape != (self.anchors.nao,):
            raise ValueError(f"transport must have shape {(self.anchors.nao,)}")
        return u.conj()[:, None] * np.asarray(matrix) * u[None, :]

    def transport_coefficients(self, coeff: np.ndarray, transport: np.ndarray) -> np.ndarray:
        u = np.asarray(transport, dtype=np.complex128)
        return u[:, None] * np.asarray(coeff)
