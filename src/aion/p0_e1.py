"""First electric finite-spread layer on top of P0 Peierls geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .gauge import AOAnchors, PeierlsGeometry
from .matrix_models import hermitian_part
from .observables import (
    electronic_energy,
    matrix_expectation,
    p0_dipole_derivative,
    p0_dipole_moment,
)


def _array_module(value):
    try:
        import cupy
    except Exception:
        return np
    return cupy.get_array_module(value)


def _central_dipole_array(central_dipoles0: np.ndarray) -> np.ndarray:
    xp = _array_module(central_dipoles0)
    array = xp.asarray(central_dipoles0, dtype=xp.complex128)
    if array.ndim != 3 or array.shape[0] != 3 or array.shape[1] != array.shape[2]:
        raise ValueError("central_dipoles0 must have shape (3, nao, nao)")
    return array


def ao_pair_anchor_centers(anchors: AOAnchors) -> np.ndarray:
    """Return ``R_mu_nu = 0.5 * (R_anchor(mu) + R_anchor(nu))``."""

    ao_coords = anchors.atom_coords[anchors.ao_to_atom]
    return 0.5 * (ao_coords[:, None, :] + ao_coords[None, :, :])


def central_dipole_matrices(
    *,
    anchors: AOAnchors,
    overlap0: np.ndarray,
    position_matrices: np.ndarray,
    charge: float = -1.0,
) -> np.ndarray:
    """Return AO-pair central finite-spread dipoles.

    The returned matrices are

    ``d^alpha_mu_nu = q (r^alpha_mu_nu - R^alpha_mu_nu S_mu_nu)``.

    They are the E1 objects that can be dressed by the Peierls/Wilson phase.
    """

    s0 = np.asarray(overlap0, dtype=np.complex128)
    if s0.shape != (anchors.nao, anchors.nao):
        raise ValueError(f"overlap0 must have shape {(anchors.nao, anchors.nao)}")
    r = np.asarray(position_matrices, dtype=np.complex128)
    if r.shape != (3, anchors.nao, anchors.nao):
        raise ValueError(
            f"position_matrices must have shape {(3, anchors.nao, anchors.nao)}"
        )

    centers = ao_pair_anchor_centers(anchors)
    dipoles = np.empty_like(r, dtype=np.complex128)
    for axis in range(3):
        dipoles[axis] = float(charge) * (r[axis] - centers[:, :, axis] * s0)
        dipoles[axis] = hermitian_part(dipoles[axis])
    return dipoles


def pyscf_central_dipole_matrices(
    reference: Any,
    *,
    charge: float | None = None,
) -> np.ndarray:
    """Build central finite-spread dipoles from a `PyscfP0Reference`-like object."""

    q = -1.0 if charge is None else float(charge)
    position_matrices = np.asarray(
        reference.mol.intor("int1e_r", comp=3),
        dtype=np.complex128,
    )
    return central_dipole_matrices(
        anchors=reference.anchors,
        overlap0=reference.overlap0,
        position_matrices=position_matrices,
        charge=q,
    )


def dressed_central_dipole_matrices(
    central_dipoles0: np.ndarray,
    geometry: PeierlsGeometry,
    t: float,
) -> np.ndarray:
    """Return ``d_P^alpha(t) = Theta(t) * d^alpha``."""

    dipoles0 = _central_dipole_array(central_dipoles0)
    xp = _array_module(dipoles0)
    if dipoles0.shape[1:] != geometry.overlap0.shape:
        raise ValueError(
            f"central_dipoles0 must have AO shape {geometry.overlap0.shape}"
        )
    theta = xp.asarray(geometry.theta(t), dtype=xp.complex128)
    dressed = xp.empty_like(dipoles0)
    for axis in range(3):
        dressed[axis] = hermitian_part(theta * dipoles0[axis])
    return dressed


def dressed_central_dipole_matrix_dots(
    central_dipoles0: np.ndarray,
    geometry: PeierlsGeometry,
    t: float,
) -> np.ndarray:
    """Return the ordinary time derivative of ``d_P^alpha(t)``."""

    dipoles0 = _central_dipole_array(central_dipoles0)
    xp = _array_module(dipoles0)
    if dipoles0.shape[1:] != geometry.overlap0.shape:
        raise ValueError(
            f"central_dipoles0 must have AO shape {geometry.overlap0.shape}"
        )
    acal_dot = xp.asarray(
        geometry.anchors.lift_site_matrix(geometry.site_bond_line_integral_dots(t)),
        dtype=float,
    )
    theta = xp.asarray(geometry.theta(t), dtype=xp.complex128)
    theta_dot = (1j * geometry.charge / geometry.hbar) * acal_dot * theta
    dots = xp.empty_like(dipoles0)
    for axis in range(3):
        dots[axis] = hermitian_part(theta_dot * dipoles0[axis])
    return dots


def p0_e1_uniform_electric_potential(
    central_dipoles0: np.ndarray,
    geometry: PeierlsGeometry,
    t: float,
    *,
    field: np.ndarray | None = None,
) -> np.ndarray:
    """Return the P0+E1 residual ``-E_alpha(t) d_P^alpha(t)`` matrix."""

    if field is None:
        electric_field = (
            np.zeros(3)
            if geometry.electric is None
            else geometry.electric.electric_field(t)
        )
    else:
        electric_field = np.asarray(field, dtype=float)
    if electric_field.shape != (3,):
        raise ValueError("field must have shape (3,)")
    dressed = dressed_central_dipole_matrices(central_dipoles0, geometry, t)
    xp = _array_module(dressed)
    field_backend = xp.asarray(electric_field, dtype=float)
    return hermitian_part(-xp.einsum("x,xij->ij", field_backend, dressed))


def p0_pair_scalar_potential_matrix(
    geometry: PeierlsGeometry,
    t: float,
) -> np.ndarray:
    """Return the Hermitian pair-center scalar matrix ``q Phi_mu_nu S_mu_nu``.

    This helper is mainly a reconstruction diagnostic.  The propagation code
    treats the P0 scalar potential as part of the time connection.
    """

    site_phi = geometry.site_scalar_potential(t)
    ao_phi = geometry.anchors.lift_site_vector(site_phi)
    pair_phi = 0.5 * (ao_phi[:, None] + ao_phi[None, :])
    return hermitian_part(geometry.charge * pair_phi * geometry.metric(t))


def p0_e1_dipole_moment(
    density: np.ndarray,
    central_dipoles0: np.ndarray,
    geometry: PeierlsGeometry,
    t: float,
    *,
    include_site: bool = True,
) -> np.ndarray:
    """Return ``q sum_a N_a R_a + Tr[rho d_P]`` at P0+E1."""

    rho = np.asarray(density, dtype=np.complex128)
    dressed = dressed_central_dipole_matrices(central_dipoles0, geometry, t)
    spread = np.asarray([matrix_expectation(rho, dressed[axis]) for axis in range(3)])
    if not include_site:
        return spread
    return p0_dipole_moment(rho, geometry, t) + spread


def p0_e1_dipole_derivative(
    *,
    charge_derivative: np.ndarray,
    density: np.ndarray,
    density_dot: np.ndarray,
    central_dipoles0: np.ndarray,
    geometry: PeierlsGeometry,
    t: float,
    include_site: bool = True,
) -> np.ndarray:
    """Return ``d/dt [q sum_a N_a R_a + Tr(rho d_P)]``."""

    rho = np.asarray(density, dtype=np.complex128)
    rho_dot = np.asarray(density_dot, dtype=np.complex128)
    dressed = dressed_central_dipole_matrices(central_dipoles0, geometry, t)
    dressed_dot = dressed_central_dipole_matrix_dots(central_dipoles0, geometry, t)
    spread_dot = np.asarray(
        [
            (
                np.trace(rho_dot @ dressed[axis])
                + np.trace(rho @ dressed_dot[axis])
            ).real
            for axis in range(3)
        ]
    )
    if not include_site:
        return spread_dot
    return p0_dipole_derivative(charge_derivative, geometry) + spread_dot


def p0_e1_dipole_power(
    *,
    charge_derivative: np.ndarray,
    density: np.ndarray,
    density_dot: np.ndarray,
    central_dipoles0: np.ndarray,
    geometry: PeierlsGeometry,
    t: float,
    electric_field: np.ndarray,
) -> float:
    """Return ``E . dmu_E1/dt`` for a spatially uniform electric field."""

    field = np.asarray(electric_field, dtype=float)
    if field.shape != (3,):
        raise ValueError("electric_field must have shape (3,)")
    dipole_dot = p0_e1_dipole_derivative(
        charge_derivative=charge_derivative,
        density=density,
        density_dot=density_dot,
        central_dipoles0=central_dipoles0,
        geometry=geometry,
        t=t,
    )
    return float(np.dot(field, dipole_dot))


@dataclass(frozen=True)
class P0E1Model:
    """Add the first electric finite-spread residual to a P0 model."""

    base_model: Any
    central_dipoles0: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "central_dipoles0",
            _central_dipole_array(self.central_dipoles0),
        )

    def e1_hamiltonian(
        self,
        t: float,
        geometry: PeierlsGeometry,
        *,
        field: np.ndarray | None = None,
    ) -> np.ndarray:
        return p0_e1_uniform_electric_potential(
            self.central_dipoles0,
            geometry,
            t,
            field=field,
        )

    def hamiltonian(
        self,
        density: np.ndarray,
        t: float,
        geometry: PeierlsGeometry,
    ) -> np.ndarray:
        base_h = self.base_model.hamiltonian(density, t, geometry)
        return hermitian_part(base_h + self.e1_hamiltonian(t, geometry))

    def energy(
        self,
        density: np.ndarray,
        t: float,
        geometry: PeierlsGeometry,
    ) -> float:
        """Return the internal/base material energy, excluding source coupling."""

        if hasattr(self.base_model, "energy"):
            return float(self.base_model.energy(density, t, geometry))
        base_h = self.base_model.hamiltonian(density, t, geometry)
        return electronic_energy(density, base_h)

    def coupling_energy(
        self,
        density: np.ndarray,
        t: float,
        geometry: PeierlsGeometry,
    ) -> float:
        """Return the instantaneous external E1 source coupling expectation."""

        return matrix_expectation(
            density,
            self.e1_hamiltonian(t, geometry),
        )
