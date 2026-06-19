"""Length-gauge real-time TDDFT over a PySCF AO density matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import scipy.linalg


ElectricField = Callable[[float], np.ndarray]


@dataclass
class PropagationRecord:
    """Diagnostics for one real-time propagation step."""

    step: int
    time: float
    electron_number: float
    idempotency_error: float
    field_free_energy: float
    field_coupling_energy: float
    total_energy: float
    dipole: np.ndarray
    field: np.ndarray


class LengthGaugeRTTDDFT:
    """Pure length-gauge real-time TDDFT using PySCF matrices.

    The propagated object is the AO density matrix convention used by PySCF:

        rho = sum_n f_n C_n C_n^dagger

    In a nonorthogonal AO basis the conserved electron number is Tr(rho S).
    """

    def __init__(
        self,
        mf,
        field: ElectricField,
        origin: np.ndarray | None = None,
        *,
        charge: float = -1.0,
        hbar: float = 1.0,
        idempotency_factor: float | None = None,
        symmetrize_density: bool = True,
    ) -> None:
        self.mf = mf
        self.mol = mf.mol
        self.field = field
        self.charge = charge
        self.hbar = hbar
        self.idempotency_factor = (
            self._infer_idempotency_factor()
            if idempotency_factor is None
            else float(idempotency_factor)
        )
        self.symmetrize_density = symmetrize_density

        self.origin = (
            np.zeros(3, dtype=float)
            if origin is None
            else np.asarray(origin, dtype=float)
        )
        if self.origin.shape != (3,):
            raise ValueError("origin must have shape (3,)")

        self.s = np.asarray(self.mol.intor("int1e_ovlp"), dtype=np.complex128)
        self.hcore = np.asarray(self.mf.get_hcore(), dtype=np.complex128)
        self.rint = np.asarray(self.mol.intor("int1e_r", comp=3), dtype=np.complex128)
        self.dipole_position = self.rint - self.origin[:, None, None] * self.s[None, :, :]

        # Cholesky is faster and more stable than explicitly forming S^{-1}.
        self._s_cho = scipy.linalg.cho_factor(self.s, lower=True, check_finite=False)

    @classmethod
    def from_ground_state(
        cls,
        mf,
        field: ElectricField,
        origin: np.ndarray | None = None,
        **kwargs,
    ) -> "LengthGaugeRTTDDFT":
        if not getattr(mf, "converged", False):
            raise ValueError("mean-field object is not converged")
        return cls(mf, field, origin, **kwargs)

    def initial_density(self) -> np.ndarray:
        """Return the PySCF ground-state AO density matrix."""

        return np.asarray(self.mf.make_rdm1(), dtype=np.complex128)

    def solve_s_left(self, a: np.ndarray) -> np.ndarray:
        """Compute S^{-1} A by solving S X = A."""

        return scipy.linalg.cho_solve(self._s_cho, a, check_finite=False)

    def solve_s_right(self, a: np.ndarray) -> np.ndarray:
        """Compute A S^{-1} without explicitly forming S^{-1}."""

        return self.solve_s_left(a.conj().T).conj().T

    def external_potential(self, t: float) -> np.ndarray:
        """Length-gauge uniform-field potential matrix.

        V_ext = -q E(t) . d, with d_mu_nu = <mu|r-origin|nu>.
        For electrons q=-1, so V_ext = E.d.
        """

        e_t = np.asarray(self.field(t), dtype=float)
        if e_t.shape != (3,):
            raise ValueError("field(t) must return shape (3,)")
        return -self.charge * np.einsum("x,xij->ij", e_t, self.dipole_position)

    def fock_matrix(self, rho: np.ndarray, t: float) -> np.ndarray:
        """Build the instantaneous adiabatic KS matrix plus external field."""

        rho_h = self.hermitian_part(rho)
        veff = np.asarray(self.mf.get_veff(self.mol, rho_h), dtype=np.complex128)
        return self.hcore + veff + self.external_potential(t)

    def rhs(self, rho: np.ndarray, t: float) -> np.ndarray:
        """Evaluate dot rho = -i/hbar (S^{-1} F rho - rho F S^{-1})."""

        fock = self.fock_matrix(rho, t)
        left = self.solve_s_left(fock) @ rho
        right = rho @ self.solve_s_right(fock)
        return (-1j / self.hbar) * (left - right)

    def backward_euler_previous(self, rho0: np.ndarray, t0: float, dt: float) -> np.ndarray:
        """Cheap leapfrog starter: rho_{-1} = rho_0 - dt R[rho_0,t_0]."""

        return rho0 - dt * self.rhs(rho0, t0)

    def leapfrog(
        self,
        rho0: np.ndarray,
        *,
        dt: float,
        nsteps: int,
        t0: float = 0.0,
    ) -> Iterator[tuple[np.ndarray, PropagationRecord]]:
        """Propagate with rho_{n+1}=rho_{n-1}+2 dt R[rho_n,t_n]."""

        rho = np.asarray(rho0, dtype=np.complex128)
        if self.symmetrize_density:
            rho = self.hermitian_part(rho)
        rho_prev = self.backward_euler_previous(rho, t0, dt)

        yield rho.copy(), self.record(0, t0, rho)

        t = t0
        for step in range(1, nsteps + 1):
            rho_next = rho_prev + 2.0 * dt * self.rhs(rho, t)
            if self.symmetrize_density:
                rho_next = self.hermitian_part(rho_next)

            t_next = t0 + step * dt
            yield rho_next.copy(), self.record(step, t_next, rho_next)

            rho_prev, rho = rho, rho_next
            t = t_next

    def electron_number(self, rho: np.ndarray) -> float:
        return float(np.trace(rho @ self.s).real)

    def idempotency_error(self, rho: np.ndarray) -> float:
        target = self.idempotency_factor * rho
        return float(np.linalg.norm(rho @ self.s @ rho - target))

    def electronic_dipole(self, rho: np.ndarray) -> np.ndarray:
        """Return q Tr(rho d) as the electronic dipole vector."""

        return self.charge * np.einsum("ij,xji->x", rho, self.dipole_position).real

    def field_free_energy(self, rho: np.ndarray) -> float:
        """Return the PySCF DFT/HF total energy without the laser coupling."""

        rho_h = self.hermitian_part(rho)
        return float(self.mf.energy_tot(dm=rho_h, h1e=self.hcore).real)

    def field_coupling_energy(self, rho: np.ndarray, time: float) -> float:
        """Return Tr(rho V_ext(t)), the instantaneous laser coupling energy."""

        return float(np.einsum("ij,ji->", rho, self.external_potential(time)).real)

    def total_energy(self, rho: np.ndarray, time: float) -> float:
        """Return field-free molecular energy plus external coupling energy."""

        return self.field_free_energy(rho) + self.field_coupling_energy(rho, time)

    def record(self, step: int, time: float, rho: np.ndarray) -> PropagationRecord:
        field_free_energy = self.field_free_energy(rho)
        field_coupling_energy = self.field_coupling_energy(rho, time)
        return PropagationRecord(
            step=step,
            time=time,
            electron_number=self.electron_number(rho),
            idempotency_error=self.idempotency_error(rho),
            field_free_energy=field_free_energy,
            field_coupling_energy=field_coupling_energy,
            total_energy=field_free_energy + field_coupling_energy,
            dipole=self.electronic_dipole(rho),
            field=np.asarray(self.field(time), dtype=float),
        )

    @staticmethod
    def hermitian_part(a: np.ndarray) -> np.ndarray:
        return 0.5 * (a + a.conj().T)

    def _infer_idempotency_factor(self) -> float:
        mo_occ = getattr(self.mf, "mo_occ", None)
        if mo_occ is None:
            return 1.0
        if isinstance(mo_occ, (tuple, list)):
            return float(max(np.max(np.asarray(x)) for x in mo_occ))
        return float(np.max(np.asarray(mo_occ)))
