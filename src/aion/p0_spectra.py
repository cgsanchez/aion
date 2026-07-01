"""P0/P0+E1 kick and observable helpers for spectra."""

from __future__ import annotations

from typing import Any

import numpy as np

from .backends import CPUBackend, make_backend
from .gauge import PeierlsGeometry
from .matrix_models import hermitian_part
from .p0_e1 import dressed_central_dipole_matrices
from .p0_e2 import p0_e2_full_second_moment_matrices


def _backend_or_default(backend: Any | None):
    return CPUBackend() if backend is None else make_backend(backend)


def metric_unitary_transform(
    coeff: np.ndarray,
    *,
    metric: np.ndarray,
    integrated_hamiltonian: np.ndarray,
    backend: Any | None = None,
    hbar: float = 1.0,
) -> np.ndarray:
    """Apply ``exp[-i S^-1 H_int / hbar]`` in orthonormal representation."""

    if hbar <= 0.0:
        raise ValueError("hbar must be positive")
    xp = _backend_or_default(backend)
    c = xp.asarray(coeff, dtype=np.complex128)
    s = hermitian_part(xp.asarray(metric, dtype=np.complex128))
    h_int = hermitian_part(xp.asarray(integrated_hamiltonian, dtype=np.complex128))
    if s.shape[0] != s.shape[1] or h_int.shape != s.shape or c.shape[0] != s.shape[0]:
        raise ValueError("coeff, metric, and integrated_hamiltonian dimensions mismatch")

    eig_s, vec_s = xp.eigh(s)
    if xp.min_float(eig_s) <= 0.0:
        raise np.linalg.LinAlgError("metric is not positive definite")
    s_sqrt = (vec_s * eig_s**0.5) @ vec_s.conj().T
    s_invsqrt = (vec_s * eig_s**-0.5) @ vec_s.conj().T
    h_orth = hermitian_part(s_invsqrt @ h_int @ s_invsqrt)
    eig_h, vec_h = xp.eigh(h_orth)
    c_orth = s_sqrt @ c
    propagated = vec_h @ (
        xp.exp((-1j / hbar) * eig_h)[:, None] * (vec_h.conj().T @ c_orth)
    )
    return s_invsqrt @ propagated


def e1_central_impulse_hamiltonian(
    impulse: np.ndarray,
    central_dipoles0: np.ndarray,
    geometry: PeierlsGeometry,
    t: float,
    *,
    backend: Any | None = None,
) -> np.ndarray:
    """Return ``integral -E(t).d_P(t) dt`` for an impulsive E1 coupling."""

    xp = _backend_or_default(backend)
    kick = xp.asarray(np.asarray(impulse, dtype=float), dtype=float)
    if kick.shape != (3,):
        raise ValueError("impulse must have shape (3,)")
    dressed = dressed_central_dipole_matrices(central_dipoles0, geometry, t)
    dressed = xp.asarray(dressed, dtype=np.complex128)
    return hermitian_part(-xp.einsum("x,xij->ij", kick, dressed))


def apply_e1_central_delta_kick(
    coeff: np.ndarray,
    *,
    impulse: np.ndarray,
    central_dipoles0: np.ndarray,
    geometry: PeierlsGeometry,
    time: float = 0.0,
    backend: Any | None = None,
    hbar: float = 1.0,
) -> np.ndarray:
    """Apply the finite-spread E1 part of a delta electric kick."""

    xp = _backend_or_default(backend)
    h_int = e1_central_impulse_hamiltonian(
        impulse,
        central_dipoles0,
        geometry,
        time,
        backend=xp,
    )
    return metric_unitary_transform(
        coeff,
        metric=geometry.metric(time),
        integrated_hamiltonian=h_int,
        backend=xp,
        hbar=hbar,
    )


def p0_dipole_operator_matrices(
    geometry: PeierlsGeometry,
    t: float,
    *,
    backend: Any | None = None,
) -> np.ndarray:
    """Return operators whose expectations are the P0 source dipole."""

    xp = _backend_or_default(backend)
    metric = xp.asarray(geometry.metric(t), dtype=np.complex128)
    coords = xp.asarray(geometry.atom_coords, dtype=float)
    operators = xp.asarray(
        np.zeros((3, geometry.anchors.nao, geometry.anchors.nao), dtype=np.complex128),
        dtype=np.complex128,
    )
    for atom in range(geometry.anchors.natom):
        diag = xp.asarray(geometry.anchors.site_projector_diagonal(atom), dtype=float)
        population_op = 0.5 * (diag[:, None] * metric + metric * diag[None, :])
        for axis in range(3):
            operators[axis] = operators[axis] + (
                geometry.charge * coords[atom, axis] * population_op
            )
    for axis in range(3):
        operators[axis] = hermitian_part(operators[axis])
    return operators


def p0_e1_dipole_operator_matrices(
    central_dipoles0: np.ndarray,
    geometry: PeierlsGeometry,
    t: float,
    *,
    backend: Any | None = None,
) -> np.ndarray:
    """Return operators whose expectations are the P0+E1 dipole."""

    xp = _backend_or_default(backend)
    p0_ops = p0_dipole_operator_matrices(geometry, t, backend=xp)
    spread = xp.asarray(
        dressed_central_dipole_matrices(central_dipoles0, geometry, t),
        dtype=np.complex128,
    )
    return p0_ops + spread


def p0_e2_second_moment_operator_matrices(
    *,
    central_dipoles0: np.ndarray,
    central_second_moments0: np.ndarray,
    geometry: PeierlsGeometry,
    t: float,
    backend: Any | None = None,
) -> np.ndarray:
    """Return dressed full raw second-moment operators reconstructed through E2."""

    xp = _backend_or_default(backend)
    return xp.asarray(
        p0_e2_full_second_moment_matrices(
            central_dipoles0=central_dipoles0,
            central_second_moments0=central_second_moments0,
            geometry=geometry,
            t=t,
        ),
        dtype=np.complex128,
    )


def operator_expectations(
    density: np.ndarray,
    operators: np.ndarray,
    *,
    backend: Any | None = None,
    asnumpy: bool = True,
) -> np.ndarray:
    """Return ``Re Tr[rho O]`` for an operator stack."""

    xp = _backend_or_default(backend)
    rho = xp.asarray(density, dtype=np.complex128)
    ops = xp.asarray(operators, dtype=np.complex128)
    if ops.shape[-2:] != rho.shape:
        raise ValueError("operator AO dimensions must match density")
    flat = ops.reshape((-1, ops.shape[-2], ops.shape[-1]))
    values = xp.einsum("ij,nji->n", rho, flat).real.reshape(ops.shape[:-2])
    if asnumpy and getattr(xp, "is_gpu", False):
        return xp.asnumpy(values)
    return values
