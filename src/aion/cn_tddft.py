"""Generalized Crank-Nicolson length-gauge RT-TDDFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import scipy.linalg


ElectricField = Callable[[float], np.ndarray]


@dataclass
class CNPropagationRecord:
    """Diagnostics for one Crank-Nicolson propagation step."""

    step: int
    time: float
    electron_number: float
    orthonormality_error: float
    idempotency_error: float
    field_free_energy: float | None
    field_coupling_energy: float
    total_energy: float | None
    dipole: np.ndarray
    field: np.ndarray


class LengthGaugeCNRTTDDFT:
    """Fixed-basis length-gauge RT-TDDFT with generalized CN propagation.

    This class propagates the occupied AO coefficient block ``C`` rather than
    the AO density matrix directly. It currently targets closed-shell PySCF RKS
    objects, for which ``P = C f C^dagger`` with ``f = 2 I_occ``.
    """

    def __init__(
        self,
        mf,
        field: ElectricField,
        origin: np.ndarray | None = None,
        *,
        charge: float = -1.0,
        hbar: float = 1.0,
        real_density_for_veff: bool = True,
        reorthonormalize_every: int | None = None,
        reorthonormalize_tolerance: float | None = None,
    ) -> None:
        self.mf = mf
        self.mol = mf.mol
        self.field = field
        self.charge = charge
        self.hbar = hbar
        self.real_density_for_veff = real_density_for_veff
        self.reorthonormalize_every = reorthonormalize_every
        self.reorthonormalize_tolerance = reorthonormalize_tolerance

        self._validate_reference()

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
        self.dipole_position = (
            self.rint - self.origin[:, None, None] * self.s[None, :, :]
        )
        self._s_cho = scipy.linalg.cho_factor(self.s, lower=True, check_finite=False)

        mo_occ = np.asarray(self.mf.mo_occ, dtype=float)
        occ_mask = mo_occ > 0
        self.occ = mo_occ[occ_mask]
        self.nocc = int(self.occ.size)
        self.idempotency_factor = float(np.max(self.occ))

    @classmethod
    def from_ground_state(
        cls,
        mf,
        field: ElectricField,
        origin: np.ndarray | None = None,
        **kwargs,
    ) -> "LengthGaugeCNRTTDDFT":
        if not getattr(mf, "converged", False):
            raise ValueError("mean-field object is not converged")
        return cls(mf, field, origin, **kwargs)

    def initial_coefficients(self) -> np.ndarray:
        """Return occupied ground-state MO coefficients."""

        mo_occ = np.asarray(self.mf.mo_occ, dtype=float)
        coeff = np.asarray(self.mf.mo_coeff[:, mo_occ > 0], dtype=np.complex128)
        return coeff.copy()

    def initial_density(self) -> np.ndarray:
        """Return the ground-state AO density from occupied coefficients."""

        return self.density_from_coefficients(self.initial_coefficients())

    def solve_s_left(self, a: np.ndarray) -> np.ndarray:
        """Compute ``S^{-1} A`` by solving ``S X = A``."""

        return scipy.linalg.cho_solve(self._s_cho, a, check_finite=False)

    def apply_delta_kick(
        self,
        coeff: np.ndarray,
        electric_field_impulse: np.ndarray,
    ) -> np.ndarray:
        """Apply an impulsive uniform electric field to occupied orbitals.

        The impulse is ``K = integral E(t) dt`` in atomic units. With the
        length-gauge convention used by :meth:`external_potential`, the
        integrated potential is ``-q K.r`` and the occupied coefficients jump as

            C(0+) = exp[-i S^{-1} (-q K.r)] C(0-).

        The matrix ``S^{-1} (-q K.r)`` is self-adjoint in the AO metric, so the
        transformation is ``S``-unitary up to numerical roundoff.
        """

        coeff = np.asarray(coeff, dtype=np.complex128)
        if coeff.shape != (self.s.shape[0], self.nocc):
            raise ValueError(f"coeff must have shape {(self.s.shape[0], self.nocc)}")

        impulse = np.asarray(electric_field_impulse, dtype=float)
        if impulse.shape != (3,):
            raise ValueError("electric_field_impulse must have shape (3,)")

        integrated_potential = -self.charge * np.einsum(
            "x,xij->ij", impulse, self.dipole_position
        )
        generator = self.solve_s_left(integrated_potential)
        kick = scipy.linalg.expm((-1j / self.hbar) * generator)
        return kick @ coeff

    def density_from_coefficients(self, coeff: np.ndarray) -> np.ndarray:
        coeff = np.asarray(coeff, dtype=np.complex128)
        if coeff.shape != (self.s.shape[0], self.nocc):
            raise ValueError(f"coeff must have shape {(self.s.shape[0], self.nocc)}")
        return (coeff * self.occ[None, :]) @ coeff.conj().T

    def external_potential(self, t: float) -> np.ndarray:
        """Length-gauge uniform-field potential matrix."""

        e_t = np.asarray(self.field(t), dtype=float)
        if e_t.shape != (3,):
            raise ValueError("field(t) must return shape (3,)")
        return -self.charge * np.einsum("x,xij->ij", e_t, self.dipole_position)

    def density_for_veff(self, rho: np.ndarray) -> np.ndarray:
        rho_h = self.hermitian_part(rho)
        if self.real_density_for_veff:
            return np.asarray(rho_h.real, dtype=float)
        return rho_h

    def hamiltonian_from_density(self, rho: np.ndarray, t: float):
        """Build H[rho,t] and return ``(H, veff)``.

        ``veff`` is returned with PySCF metadata intact so energy diagnostics can
        reuse it when it corresponds to the same density.
        """

        veff = self.mf.get_veff(self.mol, self.density_for_veff(rho))
        h = self.hcore + np.asarray(veff, dtype=np.complex128) + self.external_potential(t)
        return self.hermitian_part(h), veff

    def cn_step(self, coeff: np.ndarray, h_mid: np.ndarray, dt: float) -> np.ndarray:
        """Apply one generalized CN step with a frozen midpoint Hamiltonian."""

        alpha = 0.5j * dt / self.hbar
        a = self.s + alpha * h_mid
        b = self.s - alpha * h_mid
        rhs = b @ coeff
        lu, piv = scipy.linalg.lu_factor(a, check_finite=False)
        return scipy.linalg.lu_solve((lu, piv), rhs, check_finite=False)

    def midpoint_density(
        self,
        rho: np.ndarray,
        rho_prev: np.ndarray | None,
    ) -> np.ndarray:
        """Second-order explicit midpoint density predictor."""

        if rho_prev is None:
            return self.hermitian_part(rho)
        return self.hermitian_part(1.5 * rho - 0.5 * rho_prev)

    def propagate(
        self,
        coeff0: np.ndarray,
        *,
        dt: float,
        nsteps: int,
        t0: float = 0.0,
        corrector_iterations: int = 0,
        record_energy: bool = False,
        energy_stride: int | None = None,
    ) -> Iterator[tuple[np.ndarray, CNPropagationRecord]]:
        """Propagate occupied coefficients with generalized CN.

        ``corrector_iterations=0`` uses the explicit midpoint density predictor.
        Each corrector iteration adds one extra Hamiltonian build.
        """

        if dt <= 0:
            raise ValueError("dt must be positive")
        if nsteps < 0:
            raise ValueError("nsteps must be nonnegative")
        if corrector_iterations < 0:
            raise ValueError("corrector_iterations must be nonnegative")

        coeff = np.asarray(coeff0, dtype=np.complex128)
        if coeff.shape != (self.s.shape[0], self.nocc):
            raise ValueError(f"coeff0 must have shape {(self.s.shape[0], self.nocc)}")

        coeff = self.maybe_reorthonormalize(coeff, step=0)
        rho = self.density_from_coefficients(coeff)
        rho_prev = None

        yield coeff.copy(), self.record(
            0,
            t0,
            coeff,
            rho,
            record_energy=self._should_record_energy(0, record_energy, energy_stride),
        )

        for step in range(1, nsteps + 1):
            t = t0 + (step - 1) * dt
            t_next = t0 + step * dt
            t_mid = t + 0.5 * dt

            rho_mid = self.midpoint_density(rho, rho_prev)
            h_mid, _ = self.hamiltonian_from_density(rho_mid, t_mid)
            coeff_next = self.cn_step(coeff, h_mid, dt)

            for _ in range(corrector_iterations):
                rho_next_trial = self.density_from_coefficients(coeff_next)
                rho_mid = self.hermitian_part(0.5 * (rho + rho_next_trial))
                h_mid, _ = self.hamiltonian_from_density(rho_mid, t_mid)
                coeff_next = self.cn_step(coeff, h_mid, dt)

            coeff_next = self.maybe_reorthonormalize(coeff_next, step=step)
            rho_next = self.density_from_coefficients(coeff_next)

            yield coeff_next.copy(), self.record(
                step,
                t_next,
                coeff_next,
                rho_next,
                record_energy=self._should_record_energy(
                    step, record_energy, energy_stride
                ),
            )

            rho_prev, rho = rho, rho_next
            coeff = coeff_next

    def maybe_reorthonormalize(self, coeff: np.ndarray, *, step: int) -> np.ndarray:
        do_reorth = False
        if self.reorthonormalize_every is not None:
            if self.reorthonormalize_every <= 0:
                raise ValueError("reorthonormalize_every must be positive or None")
            do_reorth = step > 0 and step % self.reorthonormalize_every == 0
        if self.reorthonormalize_tolerance is not None:
            do_reorth = (
                do_reorth
                or self.orthonormality_error(coeff) > self.reorthonormalize_tolerance
            )
        if do_reorth:
            return self.orthonormalize(coeff)
        return coeff

    def orthonormalize(self, coeff: np.ndarray) -> np.ndarray:
        metric = self.hermitian_part(coeff.conj().T @ self.s @ coeff)
        eig, vec = scipy.linalg.eigh(metric, check_finite=False)
        if np.min(eig) <= 0:
            raise np.linalg.LinAlgError("occupied metric is not positive definite")
        metric_mhalf = (vec * eig**-0.5) @ vec.conj().T
        return coeff @ metric_mhalf

    def orthonormality_error(self, coeff: np.ndarray) -> float:
        metric = coeff.conj().T @ self.s @ coeff
        return float(np.linalg.norm(metric - np.eye(self.nocc)))

    def electron_number(self, rho: np.ndarray) -> float:
        return float(np.trace(rho @ self.s).real)

    def idempotency_error(self, rho: np.ndarray) -> float:
        target = self.idempotency_factor * rho
        return float(np.linalg.norm(rho @ self.s @ rho - target))

    def electronic_dipole(self, rho: np.ndarray) -> np.ndarray:
        """Return q Tr(rho d) as the electronic dipole vector."""

        return self.charge * np.einsum("ij,xji->x", rho, self.dipole_position).real

    def field_free_energy(self, rho: np.ndarray) -> float:
        rho_for_veff = self.density_for_veff(rho)
        return float(self.mf.energy_tot(dm=rho_for_veff, h1e=self.hcore).real)

    def field_coupling_energy(self, rho: np.ndarray, time: float) -> float:
        return float(np.einsum("ij,ji->", rho, self.external_potential(time)).real)

    def total_energy(self, rho: np.ndarray, time: float) -> float:
        return self.field_free_energy(rho) + self.field_coupling_energy(rho, time)

    def record(
        self,
        step: int,
        time: float,
        coeff: np.ndarray,
        rho: np.ndarray,
        *,
        record_energy: bool,
    ) -> CNPropagationRecord:
        field_coupling_energy = self.field_coupling_energy(rho, time)
        field_free_energy = None
        total_energy = None
        if record_energy:
            field_free_energy = self.field_free_energy(rho)
            total_energy = field_free_energy + field_coupling_energy

        return CNPropagationRecord(
            step=step,
            time=time,
            electron_number=self.electron_number(rho),
            orthonormality_error=self.orthonormality_error(coeff),
            idempotency_error=self.idempotency_error(rho),
            field_free_energy=field_free_energy,
            field_coupling_energy=field_coupling_energy,
            total_energy=total_energy,
            dipole=self.electronic_dipole(rho),
            field=np.asarray(self.field(time), dtype=float),
        )

    @staticmethod
    def hermitian_part(a: np.ndarray) -> np.ndarray:
        return 0.5 * (a + a.conj().T)

    @staticmethod
    def _should_record_energy(
        step: int,
        record_energy: bool,
        energy_stride: int | None,
    ) -> bool:
        if not record_energy:
            return False
        if energy_stride is None:
            return True
        if energy_stride <= 0:
            raise ValueError("energy_stride must be positive or None")
        return step % energy_stride == 0

    def _validate_reference(self) -> None:
        mo_occ = getattr(self.mf, "mo_occ", None)
        mo_coeff = getattr(self.mf, "mo_coeff", None)
        if mo_occ is None or mo_coeff is None:
            raise ValueError("mean-field object must have mo_occ and mo_coeff")
        if isinstance(mo_occ, (tuple, list)) or isinstance(mo_coeff, (tuple, list)):
            raise NotImplementedError("CN propagation currently supports RKS only")
        occ = np.asarray(mo_occ, dtype=float)
        occupied = occ[occ > 0]
        if occupied.size == 0:
            raise ValueError("no occupied orbitals found")
        if not np.allclose(occupied, occupied[0]):
            raise NotImplementedError("fractional/nonuniform occupations are not supported")
        if not np.isclose(occupied[0], 2.0):
            raise NotImplementedError("CN propagation currently supports closed-shell RKS")
