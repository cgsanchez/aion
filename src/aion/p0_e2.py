"""Electric E2 pair-centered second-moment helpers for P0 geometry."""

from __future__ import annotations

from typing import Any

import numpy as np

from .gauge import AOAnchors, PeierlsGeometry
from .matrix_models import hermitian_part
from .p0_e1 import (
    ao_pair_anchor_centers,
    dressed_central_dipole_matrices,
    pyscf_central_dipole_matrices,
)


def _array_module(value):
    try:
        import cupy
    except Exception:
        return np
    return cupy.get_array_module(value)


def _as_second_moment_tensor(second_moment_matrices: np.ndarray):
    xp = _array_module(second_moment_matrices)
    second = xp.asarray(second_moment_matrices, dtype=xp.complex128)
    if second.ndim == 3 and second.shape[0] == 9:
        second = second.reshape(3, 3, second.shape[1], second.shape[2])
    if (
        second.ndim != 4
        or second.shape[0] != 3
        or second.shape[1] != 3
        or second.shape[2] != second.shape[3]
    ):
        raise ValueError(
            "second_moment_matrices must have shape (9, nao, nao) "
            "or (3, 3, nao, nao)"
        )
    return second


def central_second_moment_matrices(
    *,
    anchors: AOAnchors,
    overlap0: np.ndarray,
    position_matrices: np.ndarray,
    second_moment_matrices: np.ndarray,
    charge: float = -1.0,
) -> np.ndarray:
    """Return pair-centered raw second moments.

    The returned object is

    ``Q_ab[mu,nu] = q <mu|(r_a-R_a)(r_b-R_b)|nu>``

    where ``R`` is the AO-pair anchor center
    ``0.5 * (R_anchor(mu) + R_anchor(nu))``.  This is the E2 analogue of the
    central dipoles: it is the local finite-spread object that can be Peierls
    dressed directly.  No traceless or spectroscopy-specific convention is
    applied here.
    """

    s0 = np.asarray(overlap0, dtype=np.complex128)
    if s0.shape != (anchors.nao, anchors.nao):
        raise ValueError(f"overlap0 must have shape {(anchors.nao, anchors.nao)}")
    r = np.asarray(position_matrices, dtype=np.complex128)
    if r.shape != (3, anchors.nao, anchors.nao):
        raise ValueError(
            f"position_matrices must have shape {(3, anchors.nao, anchors.nao)}"
        )
    rr = np.asarray(_as_second_moment_tensor(second_moment_matrices), dtype=np.complex128)
    if rr.shape != (3, 3, anchors.nao, anchors.nao):
        raise ValueError(
            "second_moment_matrices must have AO shape "
            f"{(anchors.nao, anchors.nao)}"
        )

    centers = ao_pair_anchor_centers(anchors)
    moments = np.empty_like(rr, dtype=np.complex128)
    q = float(charge)
    for a in range(3):
        for b in range(3):
            moments[a, b] = q * (
                rr[a, b]
                - centers[:, :, a] * r[b]
                - centers[:, :, b] * r[a]
                + centers[:, :, a] * centers[:, :, b] * s0
            )
    raw = moments.copy()
    for a in range(3):
        for b in range(3):
            symmetric_tensor = 0.5 * (raw[a, b] + raw[b, a])
            moments[a, b] = hermitian_part(symmetric_tensor)
    return moments


def pyscf_central_second_moment_matrices(
    reference: Any,
    *,
    charge: float | None = None,
) -> np.ndarray:
    """Build pair-centered second moments from PySCF AO integrals."""

    q = -1.0 if charge is None else float(charge)
    position_matrices = np.asarray(
        reference.mol.intor("int1e_r", comp=3),
        dtype=np.complex128,
    )
    second_moment_matrices = np.asarray(
        reference.mol.intor("int1e_rr", comp=9),
        dtype=np.complex128,
    )
    return central_second_moment_matrices(
        anchors=reference.anchors,
        overlap0=reference.overlap0,
        position_matrices=position_matrices,
        second_moment_matrices=second_moment_matrices,
        charge=q,
    )


def dressed_central_second_moment_matrices(
    central_second_moments0: np.ndarray,
    geometry: PeierlsGeometry,
    t: float,
) -> np.ndarray:
    """Return Peierls-dressed pair-centered second moments."""

    moments0 = _as_second_moment_tensor(central_second_moments0)
    xp = _array_module(moments0)
    if moments0.shape[2:] != geometry.overlap0.shape:
        raise ValueError(
            "central_second_moments0 must have AO shape "
            f"{geometry.overlap0.shape}"
        )
    theta = xp.asarray(geometry.theta(t), dtype=xp.complex128)
    dressed = xp.empty_like(moments0)
    for a in range(3):
        for b in range(3):
            dressed[a, b] = hermitian_part(theta * moments0[a, b])
    return dressed


def pair_center_second_moment_matrices(
    *,
    anchors: AOAnchors,
    overlap0: np.ndarray,
    charge: float = -1.0,
) -> np.ndarray:
    """Return ``q R_a R_b S`` on AO-pair anchor centers."""

    s0 = np.asarray(overlap0, dtype=np.complex128)
    centers = ao_pair_anchor_centers(anchors)
    matrices = np.empty((3, 3, anchors.nao, anchors.nao), dtype=np.complex128)
    q = float(charge)
    for a in range(3):
        for b in range(3):
            matrices[a, b] = q * centers[:, :, a] * centers[:, :, b] * s0
            matrices[a, b] = hermitian_part(matrices[a, b])
    return matrices


def p0_e2_full_second_moment_matrices(
    *,
    central_dipoles0: np.ndarray,
    central_second_moments0: np.ndarray,
    geometry: PeierlsGeometry,
    t: float,
    charge: float | None = None,
) -> np.ndarray:
    """Return dressed raw second-moment operators reconstructed through E2.

    The AO-pair identity is

    ``q r_a r_b = q R_a R_b S + R_a d_b + R_b d_a + Q_ab``

    with ``d`` the central dipole and ``Q`` the central second moment.
    This helper reconstructs the full dressed operator, which is useful for
    validating the E2 primitives against ordinary AO ``int1e_rr`` integrals.
    """

    q = geometry.charge if charge is None else float(charge)
    dressed_dipoles = dressed_central_dipole_matrices(
        central_dipoles0,
        geometry,
        t,
    )
    dressed_second = dressed_central_second_moment_matrices(
        central_second_moments0,
        geometry,
        t,
    )
    xp = _array_module(dressed_second)
    centers = xp.asarray(ao_pair_anchor_centers(geometry.anchors), dtype=float)
    theta = xp.asarray(geometry.theta(t), dtype=xp.complex128)
    overlap0 = xp.asarray(geometry.overlap0, dtype=xp.complex128)
    full = xp.empty_like(dressed_second)
    for a in range(3):
        for b in range(3):
            site = q * centers[:, :, a] * centers[:, :, b] * overlap0
            full[a, b] = hermitian_part(
                theta * site
                + centers[:, :, a] * dressed_dipoles[b]
                + centers[:, :, b] * dressed_dipoles[a]
                + dressed_second[a, b]
            )
    return full


def p0_e2_second_moment(
    density: np.ndarray,
    full_second_moment_matrices: np.ndarray,
) -> np.ndarray:
    """Return ``Tr[rho M_ab]`` for a dressed raw second-moment operator."""

    xp = _array_module(full_second_moment_matrices)
    rho = xp.asarray(density, dtype=xp.complex128)
    moments = _as_second_moment_tensor(full_second_moment_matrices)
    values = xp.empty((3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            values[a, b] = xp.trace(rho @ moments[a, b]).real
    if xp is np:
        return values
    return xp.asnumpy(values)


def traceless_quadrupole_from_second_moment(
    second_moment: np.ndarray,
    *,
    prefactor: float = 0.5,
) -> np.ndarray:
    """Return ``prefactor * (3 M_ab - delta_ab Tr M)``.

    This is a conventional traceless Cartesian quadrupole map.  The raw
    second moment is kept separate so spectroscopy-specific prefactors can be
    chosen explicitly at the call site.
    """

    moment = np.asarray(second_moment, dtype=float)
    if moment.shape != (3, 3):
        raise ValueError("second_moment must have shape (3, 3)")
    trace = float(np.trace(moment))
    return float(prefactor) * (3.0 * moment - np.eye(3) * trace)


def pyscf_e2_primitives(reference: Any, *, charge: float | None = None):
    """Return central dipoles and central second moments for one reference."""

    return (
        pyscf_central_dipole_matrices(reference, charge=charge),
        pyscf_central_second_moment_matrices(reference, charge=charge),
    )
