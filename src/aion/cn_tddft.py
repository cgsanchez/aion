"""Generalized Crank-Nicolson length-gauge RT-TDDFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import scipy.linalg

from .backends import CPUBackend, make_backend


ElectricField = Callable[[float], np.ndarray]


def _asarray_host(a, dtype=None) -> np.ndarray:
    """Return a NumPy array, explicitly copying CuPy-like arrays to host."""

    if hasattr(a, "get"):
        a = a.get()
    return np.asarray(a, dtype=dtype)


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
    midpoint_iterations: int = 0
    midpoint_converged: bool = True
    hamiltonian_residual: float | None = None
    density_residual: float | None = None
    fock_builds: int = 0


@dataclass
class SCEMStepResult:
    """Result of one strict self-consistent exponential midpoint step."""

    coeff_next: np.ndarray
    rho_next: np.ndarray
    h_mid: np.ndarray
    rho_mid: np.ndarray
    iterations: int
    fock_builds: int
    hamiltonian_residual: float
    density_residual: float | None
    converged: bool


@dataclass
class SCEMRunSummary:
    """Summary of a non-yielding SCEM propagation run."""

    coeff_final: np.ndarray
    rho_final: np.ndarray
    record_final: CNPropagationRecord
    steps: int
    fock_builds: int
    midpoint_iterations: int


class MidpointConvergenceError(RuntimeError):
    """Raised when a strict midpoint fixed-point solve fails."""


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
        backend: str = "cpu",
    ) -> None:
        self.mf = mf
        self.mol = mf.mol
        self.field = field
        self.charge = charge
        self.hbar = hbar
        self.backend = make_backend(backend)
        self.real_density_for_veff = real_density_for_veff
        self.reorthonormalize_every = reorthonormalize_every
        self.reorthonormalize_tolerance = reorthonormalize_tolerance
        self._veff_on_gpu = hasattr(getattr(self.mf, "mo_coeff", None), "get")
        self._hybrid_coeff = self._reference_hybrid_coeff()

        self._validate_reference()
        self._validate_density_backend()
        if self.backend.is_gpu and not self._veff_on_gpu:
            raise ValueError(
                "GPU propagation backend requires a GPU4PySCF mean-field object; "
                "call density_fit().to_gpu() or to_gpu() before constructing the "
                "propagator."
            )

        self.origin = (
            np.zeros(3, dtype=float)
            if origin is None
            else np.asarray(origin, dtype=float)
        )
        if self.origin.shape != (3,):
            raise ValueError("origin must have shape (3,)")

        self.s = self.backend.asarray(self.mol.intor("int1e_ovlp"), dtype=np.complex128)
        self.hcore = self.backend.asarray(
            _asarray_host(self.mf.get_hcore(), dtype=np.complex128),
            dtype=np.complex128,
        )
        self.rint = self.backend.asarray(
            self.mol.intor("int1e_r", comp=3),
            dtype=np.complex128,
        )
        origin_backend = self.backend.asarray(self.origin, dtype=float)
        self.dipole_position = (
            self.rint - origin_backend[:, None, None] * self.s[None, :, :]
        )
        if self.backend.is_gpu:
            self._s_cho = None
            self._s_eval, self._s_evec = self.backend.eigh(self.s)
        else:
            self._s_cho = scipy.linalg.cho_factor(self.s, lower=True, check_finite=False)
            self._s_eval, self._s_evec = scipy.linalg.eigh(self.s, check_finite=False)
        if self.backend.min_float(self._s_eval) <= 0:
            raise np.linalg.LinAlgError("AO overlap matrix is not positive definite")
        self.s_sqrt = (self._s_evec * self._s_eval**0.5) @ self._s_evec.conj().T
        self.s_invsqrt = (self._s_evec * self._s_eval**-0.5) @ self._s_evec.conj().T

        mo_occ = _asarray_host(self.mf.mo_occ, dtype=float)
        occ_mask = mo_occ > 0
        self.occ = mo_occ[occ_mask]
        self.occ_backend = self.backend.asarray(self.occ, dtype=float)
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

    def _array_backend(self):
        if not hasattr(self, "backend"):
            self.backend = CPUBackend()
        return self.backend

    def initial_coefficients(self) -> np.ndarray:
        """Return occupied ground-state MO coefficients."""

        mo_occ = _asarray_host(self.mf.mo_occ, dtype=float)
        mo_coeff = _asarray_host(self.mf.mo_coeff, dtype=np.complex128)
        coeff = mo_coeff[:, mo_occ > 0]
        return self._array_backend().asarray(coeff.copy(), dtype=np.complex128)

    def initial_density(self) -> np.ndarray:
        """Return the ground-state AO density from occupied coefficients."""

        return self.density_from_coefficients(self.initial_coefficients())

    def solve_s_left(self, a: np.ndarray) -> np.ndarray:
        """Compute ``S^{-1} A`` by solving ``S X = A``."""

        if self._array_backend().is_gpu:
            return self._array_backend().solve(self.s, a)
        return scipy.linalg.cho_solve(self._s_cho, a, check_finite=False)

    def to_orthonormal_hamiltonian(self, h: np.ndarray) -> np.ndarray:
        """Return ``S^{-1/2} H S^{-1/2}``."""

        return self.s_invsqrt @ h @ self.s_invsqrt

    def to_orthonormal_density(self, rho: np.ndarray) -> np.ndarray:
        """Return ``S^{1/2} rho S^{1/2}``."""

        return self.s_sqrt @ rho @ self.s_sqrt

    def orthonormal_hamiltonian_residual(
        self,
        h_new: np.ndarray,
        h_old: np.ndarray,
    ) -> float:
        """Relative midpoint-Hamiltonian residual in an orthonormal metric."""

        h_new_tilde = self.to_orthonormal_hamiltonian(h_new)
        h_old_tilde = self.to_orthonormal_hamiltonian(h_old)
        backend = self._array_backend()
        numerator = backend.norm_float(h_new_tilde - h_old_tilde)
        denominator = max(1.0, backend.norm_float(h_new_tilde))
        return float(numerator / denominator)

    def orthonormal_density_residual(
        self,
        rho_new: np.ndarray,
        rho_old: np.ndarray,
    ) -> float:
        """Relative endpoint-density residual in an orthonormal metric."""

        rho_new_tilde = self.to_orthonormal_density(rho_new)
        rho_old_tilde = self.to_orthonormal_density(rho_old)
        backend = self._array_backend()
        numerator = backend.norm_float(rho_new_tilde - rho_old_tilde)
        denominator = max(1.0, backend.norm_float(rho_new_tilde))
        return float(numerator / denominator)

    def apply_delta_kick(
        self,
        coeff: np.ndarray,
        electric_field_impulse: np.ndarray,
    ) -> np.ndarray:
        """Apply an impulsive uniform electric field to occupied orbitals.

        The impulse is ``K = integral E(t) dt`` in atomic units. With the
        length-gauge convention used by :meth:`external_potential`, the
        integrated potential is ``V_K = -q K.r``. The kick is applied in the
        orthonormal representation:

            C(0+) = S^{-1/2} exp[-i S^{-1/2} V_K S^{-1/2}] S^{1/2} C(0-).

        This is mathematically equivalent to ``exp[-i S^{-1} V_K] C`` but uses
        an ordinary Hermitian eigensolver for the exponent.
        """

        backend = self._array_backend()
        coeff = backend.asarray(coeff, dtype=np.complex128)
        if coeff.shape != (self.s.shape[0], self.nocc):
            raise ValueError(f"coeff must have shape {(self.s.shape[0], self.nocc)}")

        impulse_host = np.asarray(electric_field_impulse, dtype=float)
        if impulse_host.shape != (3,):
            raise ValueError("electric_field_impulse must have shape (3,)")
        impulse = backend.asarray(impulse_host, dtype=float)

        integrated_potential = -self.charge * backend.einsum(
            "x,xij->ij", impulse, self.dipole_position
        )
        v_tilde = self.hermitian_part(
            self.to_orthonormal_hamiltonian(integrated_potential)
        )
        eig, vec = backend.eigh(v_tilde)
        phase = (vec * backend.exp((-1j / self.hbar) * eig)) @ vec.conj().T
        coeff_tilde = self.s_sqrt @ coeff
        return self.s_invsqrt @ (phase @ coeff_tilde)

    def exponential_step(
        self,
        coeff: np.ndarray,
        h_frozen: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Apply the exact frozen-Hamiltonian exponential in orthonormal form."""

        backend = self._array_backend()
        coeff = backend.asarray(coeff, dtype=np.complex128)
        if coeff.shape != (self.s.shape[0], self.nocc):
            raise ValueError(f"coeff must have shape {(self.s.shape[0], self.nocc)}")
        h_tilde = self.hermitian_part(self.to_orthonormal_hamiltonian(h_frozen))
        eig, vec = backend.eigh(h_tilde)
        phase = (vec * backend.exp((-1j * dt / self.hbar) * eig)) @ vec.conj().T
        return self.s_invsqrt @ (phase @ (self.s_sqrt @ coeff))

    def exponential_action_from_eigh(
        self,
        coeff_orth: np.ndarray,
        eig: np.ndarray,
        vec: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Apply a frozen orthonormal Hamiltonian eigendecomposition."""

        backend = self._array_backend()
        propagated_orth = vec @ (
            backend.exp((-1j * dt / self.hbar) * eig)[:, None]
            * (vec.conj().T @ coeff_orth)
        )
        return self.s_invsqrt @ propagated_orth

    def exponent_hamiltonian_residual(
        self,
        h_new: np.ndarray,
        h_old: np.ndarray,
        dt: float,
    ) -> float:
        """Dimensionless residual for the frozen one-step exponent."""

        h_new_tilde = self.to_orthonormal_hamiltonian(h_new)
        h_old_tilde = self.to_orthonormal_hamiltonian(h_old)
        backend = self._array_backend()
        numerator = abs(dt) * backend.norm_float(h_new_tilde - h_old_tilde)
        return float(numerator / np.sqrt(h_new.shape[0]))

    def density_from_coefficients(self, coeff: np.ndarray) -> np.ndarray:
        coeff = self._array_backend().asarray(coeff, dtype=np.complex128)
        if coeff.shape != (self.s.shape[0], self.nocc):
            raise ValueError(f"coeff must have shape {(self.s.shape[0], self.nocc)}")
        occ = getattr(self, "occ_backend", None)
        if occ is None:
            occ = self._array_backend().asarray(self.occ, dtype=float)
        return (coeff * occ[None, :]) @ coeff.conj().T

    def external_potential(self, t: float) -> np.ndarray:
        """Length-gauge uniform-field potential matrix."""

        backend = self._array_backend()
        e_t_host = np.asarray(self.field(t), dtype=float)
        e_t = backend.asarray(e_t_host, dtype=float)
        if e_t.shape != (3,):
            raise ValueError("field(t) must return shape (3,)")
        return -self.charge * backend.einsum("x,xij->ij", e_t, self.dipole_position)

    def density_for_veff(self, rho: np.ndarray) -> np.ndarray:
        backend = self._array_backend()
        rho_h = self.hermitian_part(rho)
        if self.real_density_for_veff:
            density = backend.asarray(rho_h.real, dtype=float)
        else:
            density = backend.asarray(rho_h, dtype=np.complex128)
        if self._veff_on_gpu:
            import cupy

            return cupy.asarray(density)
        return backend.asnumpy(density)

    def hamiltonian_from_density(self, rho: np.ndarray, t: float):
        """Build H[rho,t] and return ``(H, veff)``.

        ``veff`` is returned with PySCF metadata intact so energy diagnostics can
        reuse it when it corresponds to the same density.
        """

        backend = self._array_backend()
        veff = self.mf.get_veff(self.mol, self.density_for_veff(rho))
        h = (
            self.hcore
            + backend.asarray(veff, dtype=np.complex128)
            + self.external_potential(t)
        )
        return self.hermitian_part(h), veff

    def cn_step(self, coeff: np.ndarray, h_mid: np.ndarray, dt: float) -> np.ndarray:
        """Apply one generalized CN step with a frozen midpoint Hamiltonian."""

        backend = self._array_backend()
        coeff = backend.asarray(coeff, dtype=np.complex128)
        h_mid = backend.asarray(h_mid, dtype=np.complex128)
        alpha = 0.5j * dt / self.hbar
        a = self.s + alpha * h_mid
        b = self.s - alpha * h_mid
        rhs = b @ coeff
        lu, piv = backend.lu_factor(a)
        return backend.lu_solve((lu, piv), rhs)

    def midpoint_density(
        self,
        rho: np.ndarray,
        rho_prev: np.ndarray | None,
    ) -> np.ndarray:
        """Second-order explicit midpoint density predictor."""

        if rho_prev is None:
            return self.hermitian_part(rho)
        return self.hermitian_part(1.5 * rho - 0.5 * rho_prev)

    def scem_step(
        self,
        coeff: np.ndarray,
        *,
        time: float,
        dt: float,
        h_guess: np.ndarray,
        midpoint_tolerance: float = 1.0e-10,
        density_tolerance: float | None = 1.0e-10,
        max_iterations: int = 50,
        initial_mixing: float = 1.0,
        minimum_mixing: float = 0.1,
        predictor_fock_builds: int = 0,
    ) -> SCEMStepResult:
        """Take one strict self-consistent exponential midpoint step.

        The nonlinear fixed point is solved in Hamiltonian space.  Every trial
        midpoint density is generated by an S-unitary exponential half-step
        from ``coeff``; no chord-averaged density is used.
        """

        if dt <= 0:
            raise ValueError("dt must be positive")
        if midpoint_tolerance <= 0:
            raise ValueError("midpoint_tolerance must be positive")
        if density_tolerance is not None and density_tolerance <= 0:
            raise ValueError("density_tolerance must be positive or None")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not (0.0 < initial_mixing <= 1.0):
            raise ValueError("initial_mixing must be in (0, 1]")
        if not (0.0 < minimum_mixing <= initial_mixing):
            raise ValueError("minimum_mixing must be in (0, initial_mixing]")
        if predictor_fock_builds < 0:
            raise ValueError("predictor_fock_builds must be nonnegative")

        backend = self._array_backend()
        coeff = backend.asarray(coeff, dtype=np.complex128)
        if coeff.shape != (self.s.shape[0], self.nocc):
            raise ValueError(f"coeff must have shape {(self.s.shape[0], self.nocc)}")

        t_mid = time + 0.5 * dt
        coeff_orth = self.s_sqrt @ coeff
        h_iter = self.hermitian_part(backend.asarray(h_guess, dtype=np.complex128))
        alpha = float(initial_mixing)
        previous_rh = None
        growth_count = 0
        previous_pmid_orth = None
        fock_builds = int(predictor_fock_builds)
        h_residual = float("nan")
        d_residual = None

        for iteration in range(1, max_iterations + 1):
            h_iter_orth = self.hermitian_part(
                self.to_orthonormal_hamiltonian(h_iter)
            )
            eig, vec = backend.eigh(h_iter_orth)

            c_mid = self.exponential_action_from_eigh(
                coeff_orth, eig, vec, 0.5 * dt
            )
            rho_mid = self.hermitian_part(self.density_from_coefficients(c_mid))
            rho_mid_orth = self.to_orthonormal_density(rho_mid)

            h_built, _ = self.hamiltonian_from_density(rho_mid, t_mid)
            h_built = self.hermitian_part(h_built)
            fock_builds += 1
            h_residual = self.exponent_hamiltonian_residual(
                h_built, h_iter, dt
            )

            if previous_pmid_orth is not None:
                numerator = backend.norm_float(rho_mid_orth - previous_pmid_orth)
                denominator = max(1.0, backend.norm_float(rho_mid_orth))
                d_residual = float(numerator / denominator)

            converged = h_residual <= midpoint_tolerance
            if density_tolerance is not None and d_residual is not None:
                converged = converged and d_residual <= density_tolerance

            if converged:
                coeff_next = self.exponential_action_from_eigh(
                    coeff_orth, eig, vec, dt
                )
                rho_next = self.hermitian_part(
                    self.density_from_coefficients(coeff_next)
                )
                return SCEMStepResult(
                    coeff_next=coeff_next,
                    rho_next=rho_next,
                    h_mid=h_iter,
                    rho_mid=rho_mid,
                    iterations=iteration,
                    fock_builds=fock_builds,
                    hamiltonian_residual=h_residual,
                    density_residual=d_residual,
                    converged=True,
                )

            if previous_rh is not None and h_residual > previous_rh:
                growth_count += 1
            else:
                growth_count = 0
            if growth_count >= 2:
                alpha = max(0.5 * alpha, minimum_mixing)
                growth_count = 0

            h_iter = self.hermitian_part((1.0 - alpha) * h_iter + alpha * h_built)
            previous_pmid_orth = rho_mid_orth
            previous_rh = h_residual

        raise MidpointConvergenceError(
            "SCEM midpoint did not converge: "
            f"time={time} dt={dt} iterations={max_iterations} "
            f"h_residual={h_residual} density_residual={d_residual}"
        )

    def propagate(
        self,
        coeff0: np.ndarray,
        *,
        dt: float,
        nsteps: int,
        t0: float = 0.0,
        corrector_iterations: int = 0,
        max_corrector_iterations: int | None = None,
        hamiltonian_tolerance: float | None = None,
        density_tolerance: float | None = None,
        final_midpoint_solve: bool = False,
        record_energy: bool = False,
        energy_stride: int | None = None,
    ) -> Iterator[tuple[np.ndarray, CNPropagationRecord]]:
        """Propagate occupied coefficients with generalized CN.

        ``corrector_iterations=0`` uses the explicit midpoint density predictor.
        Each fixed corrector iteration adds one extra Hamiltonian build. If
        either tolerance is supplied, the corrector becomes a convergence loop
        capped by ``max_corrector_iterations``.
        """

        if dt <= 0:
            raise ValueError("dt must be positive")
        if nsteps < 0:
            raise ValueError("nsteps must be nonnegative")
        if corrector_iterations < 0:
            raise ValueError("corrector_iterations must be nonnegative")
        if max_corrector_iterations is not None and max_corrector_iterations < 0:
            raise ValueError("max_corrector_iterations must be nonnegative or None")
        if hamiltonian_tolerance is not None and hamiltonian_tolerance <= 0:
            raise ValueError("hamiltonian_tolerance must be positive or None")
        if density_tolerance is not None and density_tolerance <= 0:
            raise ValueError("density_tolerance must be positive or None")

        use_residual_convergence = (
            hamiltonian_tolerance is not None or density_tolerance is not None
        )
        if use_residual_convergence:
            corrector_limit = (
                corrector_iterations
                if max_corrector_iterations is None
                else max_corrector_iterations
            )
            if corrector_limit <= 0:
                raise ValueError(
                    "residual convergence requires at least one corrector iteration"
                )
        else:
            corrector_limit = corrector_iterations

        backend = self._array_backend()
        coeff = backend.asarray(coeff0, dtype=np.complex128)
        if coeff.shape != (self.s.shape[0], self.nocc):
            raise ValueError(f"coeff0 must have shape {(self.s.shape[0], self.nocc)}")

        coeff = self.maybe_reorthonormalize(coeff, step=0)
        rho = self.density_from_coefficients(coeff)
        rho_prev = None

        yield backend.copy(coeff), self.record(
            0,
            t0,
            coeff,
            rho,
            record_energy=self._should_record_energy(0, record_energy, energy_stride),
            fock_builds=0,
        )

        for step in range(1, nsteps + 1):
            t = t0 + (step - 1) * dt
            t_next = t0 + step * dt
            t_mid = t + 0.5 * dt

            rho_mid = self.midpoint_density(rho, rho_prev)
            h_mid, _ = self.hamiltonian_from_density(rho_mid, t_mid)
            coeff_next = self.cn_step(coeff, h_mid, dt)
            midpoint_iterations = 0
            midpoint_converged = not use_residual_convergence
            h_residual = None
            d_residual = None
            rho_next_previous = None

            for _ in range(corrector_limit):
                rho_next_trial = self.density_from_coefficients(coeff_next)
                rho_mid = self.hermitian_part(0.5 * (rho + rho_next_trial))
                h_new, _ = self.hamiltonian_from_density(rho_mid, t_mid)
                midpoint_iterations += 1

                if use_residual_convergence:
                    h_residual = self.orthonormal_hamiltonian_residual(h_new, h_mid)
                    if rho_next_previous is not None:
                        d_residual = self.orthonormal_density_residual(
                            rho_next_trial, rho_next_previous
                        )
                    h_ok = (
                        hamiltonian_tolerance is None
                        or h_residual < hamiltonian_tolerance
                    )
                    d_ok = (
                        density_tolerance is None
                        or (
                            d_residual is not None
                            and d_residual < density_tolerance
                        )
                    )
                    h_mid = h_new
                    if h_ok and d_ok:
                        midpoint_converged = True
                        if final_midpoint_solve:
                            coeff_next = self.cn_step(coeff, h_mid, dt)
                        break
                    rho_next_previous = rho_next_trial
                    coeff_next = self.cn_step(coeff, h_mid, dt)
                else:
                    h_mid = h_new
                    coeff_next = self.cn_step(coeff, h_mid, dt)

            if use_residual_convergence and not midpoint_converged:
                raise RuntimeError(
                    "CN midpoint did not converge: "
                    f"step={step} h_residual={h_residual} "
                    f"density_residual={d_residual}"
                )

            coeff_next = self.maybe_reorthonormalize(coeff_next, step=step)
            rho_next = self.density_from_coefficients(coeff_next)

            yield backend.copy(coeff_next), self.record(
                step,
                t_next,
                coeff_next,
                rho_next,
                record_energy=self._should_record_energy(
                    step, record_energy, energy_stride
                ),
                midpoint_iterations=midpoint_iterations,
                midpoint_converged=midpoint_converged,
                hamiltonian_residual=h_residual,
                density_residual=d_residual,
                fock_builds=1 + midpoint_iterations,
            )

            rho_prev, rho = rho, rho_next
            coeff = coeff_next

    def propagate_scem(
        self,
        coeff0: np.ndarray,
        *,
        dt: float,
        nsteps: int,
        t0: float = 0.0,
        midpoint_tolerance: float = 1.0e-10,
        density_tolerance: float | None = 1.0e-10,
        max_iterations: int = 50,
        initial_mixing: float = 1.0,
        minimum_mixing: float = 0.1,
        record_energy: bool = False,
        energy_stride: int | None = None,
    ) -> Iterator[tuple[np.ndarray, CNPropagationRecord]]:
        """Propagate with strict self-consistent exponential midpoint."""

        if dt <= 0:
            raise ValueError("dt must be positive")
        if nsteps < 0:
            raise ValueError("nsteps must be nonnegative")

        backend = self._array_backend()
        coeff = backend.asarray(coeff0, dtype=np.complex128)
        if coeff.shape != (self.s.shape[0], self.nocc):
            raise ValueError(f"coeff0 must have shape {(self.s.shape[0], self.nocc)}")

        coeff = self.maybe_reorthonormalize(coeff, step=0)
        rho = self.density_from_coefficients(coeff)
        h_current, _ = self.hamiltonian_from_density(rho, t0)
        midpoint_history: list[np.ndarray] = []

        yield backend.copy(coeff), self.record(
            0,
            t0,
            coeff,
            rho,
            record_energy=self._should_record_energy(0, record_energy, energy_stride),
            fock_builds=1,
        )

        for step in range(1, nsteps + 1):
            t = t0 + (step - 1) * dt
            t_next = t0 + step * dt
            if len(midpoint_history) >= 2:
                h_guess = self.hermitian_part(
                    2.0 * midpoint_history[-1] - midpoint_history[-2]
                )
                predictor_builds = 0
            elif len(midpoint_history) == 1:
                h_guess = midpoint_history[-1]
                predictor_builds = 0
            else:
                h_guess = h_current
                predictor_builds = 0

            result = self.scem_step(
                coeff,
                time=t,
                dt=dt,
                h_guess=h_guess,
                midpoint_tolerance=midpoint_tolerance,
                density_tolerance=density_tolerance,
                max_iterations=max_iterations,
                initial_mixing=initial_mixing,
                minimum_mixing=minimum_mixing,
                predictor_fock_builds=predictor_builds,
            )

            coeff_next = self.maybe_reorthonormalize(result.coeff_next, step=step)
            rho_next = self.density_from_coefficients(coeff_next)
            midpoint_history.append(result.h_mid)
            if len(midpoint_history) > 2:
                midpoint_history.pop(0)

            yield backend.copy(coeff_next), self.record(
                step,
                t_next,
                coeff_next,
                rho_next,
                record_energy=self._should_record_energy(
                    step, record_energy, energy_stride
                ),
                midpoint_iterations=result.iterations,
                midpoint_converged=result.converged,
                hamiltonian_residual=result.hamiltonian_residual,
                density_residual=result.density_residual,
                fock_builds=result.fock_builds,
            )

            coeff = coeff_next
            rho = rho_next

    def run_scem(
        self,
        coeff0: np.ndarray,
        *,
        dt: float,
        nsteps: int,
        t0: float = 0.0,
        midpoint_tolerance: float = 1.0e-10,
        density_tolerance: float | None = 1.0e-10,
        max_iterations: int = 50,
        initial_mixing: float = 1.0,
        minimum_mixing: float = 0.1,
        record_energy: bool = False,
        energy_stride: int | None = None,
    ) -> SCEMRunSummary:
        """Run SCEM without per-step coefficient snapshots or records.

        This is the preferred timing path for GPU benchmarking.  The nonlinear
        midpoint solve is unchanged from :meth:`propagate_scem`, but diagnostics
        are computed only at the final time, avoiding avoidable host transfers.
        """

        if dt <= 0:
            raise ValueError("dt must be positive")
        if nsteps < 0:
            raise ValueError("nsteps must be nonnegative")

        backend = self._array_backend()
        coeff = backend.asarray(coeff0, dtype=np.complex128)
        if coeff.shape != (self.s.shape[0], self.nocc):
            raise ValueError(f"coeff0 must have shape {(self.s.shape[0], self.nocc)}")

        coeff = self.maybe_reorthonormalize(coeff, step=0)
        rho = self.density_from_coefficients(coeff)
        h_current, _ = self.hamiltonian_from_density(rho, t0)
        midpoint_history: list[np.ndarray] = []
        total_fock_builds = 1
        total_midpoint_iterations = 0

        for step in range(1, nsteps + 1):
            t = t0 + (step - 1) * dt
            if len(midpoint_history) >= 2:
                h_guess = self.hermitian_part(
                    2.0 * midpoint_history[-1] - midpoint_history[-2]
                )
                predictor_builds = 0
            elif len(midpoint_history) == 1:
                h_guess = midpoint_history[-1]
                predictor_builds = 0
            else:
                h_guess = h_current
                predictor_builds = 0

            result = self.scem_step(
                coeff,
                time=t,
                dt=dt,
                h_guess=h_guess,
                midpoint_tolerance=midpoint_tolerance,
                density_tolerance=density_tolerance,
                max_iterations=max_iterations,
                initial_mixing=initial_mixing,
                minimum_mixing=minimum_mixing,
                predictor_fock_builds=predictor_builds,
            )

            coeff_next = self.maybe_reorthonormalize(result.coeff_next, step=step)
            rho_next = self.density_from_coefficients(coeff_next)
            midpoint_history.append(result.h_mid)
            if len(midpoint_history) > 2:
                midpoint_history.pop(0)

            total_fock_builds += result.fock_builds
            total_midpoint_iterations += result.iterations
            coeff = coeff_next
            rho = rho_next

        final_step = nsteps
        final_time = t0 + nsteps * dt
        final_record = self.record(
            final_step,
            final_time,
            coeff,
            rho,
            record_energy=self._should_record_energy(
                final_step, record_energy, energy_stride
            ),
            fock_builds=total_fock_builds,
        )
        return SCEMRunSummary(
            coeff_final=coeff,
            rho_final=rho,
            record_final=final_record,
            steps=nsteps,
            fock_builds=total_fock_builds,
            midpoint_iterations=total_midpoint_iterations,
        )

    def propagate_ep_pc1(
        self,
        coeff0: np.ndarray,
        *,
        dt: float,
        nsteps: int,
        t0: float = 0.0,
        tolerance: float = 1.0e-7,
        max_corrector_iterations: int = 6,
        strict_endpoint_hamiltonian: bool = True,
        record_energy: bool = False,
        energy_stride: int | None = None,
    ) -> Iterator[tuple[np.ndarray, CNPropagationRecord]]:
        """Propagate occupied coefficients with orthonormal EP-PC1.

        EP-PC1 predicts an endpoint density with ``exp(-i dt H_n)`` and then
        corrects with the exponential of the trapezoidal Hamiltonian
        ``0.5 * (H_n + H_{n+1}^{pred})``. The residual is the relative change
        between predicted and corrected endpoint densities in the orthonormal
        representation.
        """

        if dt <= 0:
            raise ValueError("dt must be positive")
        if nsteps < 0:
            raise ValueError("nsteps must be nonnegative")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if max_corrector_iterations <= 0:
            raise ValueError("max_corrector_iterations must be positive")

        backend = self._array_backend()
        coeff = backend.asarray(coeff0, dtype=np.complex128)
        if coeff.shape != (self.s.shape[0], self.nocc):
            raise ValueError(f"coeff0 must have shape {(self.s.shape[0], self.nocc)}")

        coeff = self.maybe_reorthonormalize(coeff, step=0)
        rho = self.density_from_coefficients(coeff)
        h_current, _ = self.hamiltonian_from_density(rho, t0)

        yield backend.copy(coeff), self.record(
            0,
            t0,
            coeff,
            rho,
            record_energy=self._should_record_energy(0, record_energy, energy_stride),
            fock_builds=1,
        )

        for step in range(1, nsteps + 1):
            t = t0 + (step - 1) * dt
            t_next = t0 + step * dt

            coeff_pred = self.exponential_step(coeff, h_current, dt)
            rho_pred = self.density_from_coefficients(coeff_pred)
            coeff_next = coeff_pred
            rho_next = rho_pred
            residual = None
            h_pred = None
            converged = False
            iterations = 0

            for _ in range(max_corrector_iterations):
                h_pred, _ = self.hamiltonian_from_density(rho_pred, t_next)
                h_trap = self.hermitian_part(0.5 * (h_current + h_pred))
                coeff_corr = self.exponential_step(coeff, h_trap, dt)
                rho_corr = self.density_from_coefficients(coeff_corr)
                iterations += 1
                residual = self.orthonormal_density_residual(rho_corr, rho_pred)
                coeff_next = coeff_corr
                rho_next = rho_corr
                if residual < tolerance:
                    converged = True
                    break
                rho_pred = rho_corr

            if not converged:
                raise RuntimeError(
                    "EP-PC1 endpoint did not converge: "
                    f"step={step} density_residual={residual}"
                )

            coeff_next = self.maybe_reorthonormalize(coeff_next, step=step)
            rho_next = self.density_from_coefficients(coeff_next)
            if strict_endpoint_hamiltonian or h_pred is None:
                h_current, _ = self.hamiltonian_from_density(rho_next, t_next)
                fock_builds = iterations + 1
            else:
                h_current = h_pred
                fock_builds = iterations

            yield backend.copy(coeff_next), self.record(
                step,
                t_next,
                coeff_next,
                rho_next,
                record_energy=self._should_record_energy(
                    step, record_energy, energy_stride
                ),
                midpoint_iterations=iterations,
                midpoint_converged=converged,
                density_residual=residual,
                fock_builds=fock_builds,
            )

            coeff = coeff_next
            rho = rho_next

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
        backend = self._array_backend()
        metric = self.hermitian_part(coeff.conj().T @ self.s @ coeff)
        eig, vec = backend.eigh(metric)
        if backend.min_float(eig) <= 0:
            raise np.linalg.LinAlgError("occupied metric is not positive definite")
        metric_mhalf = (vec * eig**-0.5) @ vec.conj().T
        return coeff @ metric_mhalf

    def orthonormality_error(self, coeff: np.ndarray) -> float:
        backend = self._array_backend()
        metric = coeff.conj().T @ self.s @ coeff
        return backend.norm_float(metric - backend.eye(self.nocc, dtype=np.complex128))

    def electron_number(self, rho: np.ndarray) -> float:
        backend = self._array_backend()
        return backend.real_float(backend.trace(rho @ self.s))

    def idempotency_error(self, rho: np.ndarray) -> float:
        backend = self._array_backend()
        target = self.idempotency_factor * rho
        return backend.norm_float(rho @ self.s @ rho - target)

    def electronic_dipole(self, rho: np.ndarray) -> np.ndarray:
        """Return q Tr(rho d) as the electronic dipole vector."""

        backend = self._array_backend()
        dipole = self.charge * backend.einsum("ij,xji->x", rho, self.dipole_position).real
        return backend.asnumpy(dipole)

    def field_free_energy(self, rho: np.ndarray) -> float:
        rho_for_veff = self.density_for_veff(rho)
        h1e = self._array_backend().asnumpy(self.hcore)
        return float(self.mf.energy_tot(dm=rho_for_veff, h1e=h1e).real)

    def field_coupling_energy(self, rho: np.ndarray, time: float) -> float:
        backend = self._array_backend()
        coupling = backend.einsum("ij,ji->", rho, self.external_potential(time))
        return backend.real_float(coupling)

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
        midpoint_iterations: int = 0,
        midpoint_converged: bool = True,
        hamiltonian_residual: float | None = None,
        density_residual: float | None = None,
        fock_builds: int = 0,
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
            midpoint_iterations=midpoint_iterations,
            midpoint_converged=midpoint_converged,
            hamiltonian_residual=hamiltonian_residual,
            density_residual=density_residual,
            fock_builds=fock_builds,
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
        occ = _asarray_host(mo_occ, dtype=float)
        occupied = occ[occ > 0]
        if occupied.size == 0:
            raise ValueError("no occupied orbitals found")
        if not np.allclose(occupied, occupied[0]):
            raise NotImplementedError("fractional/nonuniform occupations are not supported")
        if not np.isclose(occupied[0], 2.0):
            raise NotImplementedError("CN propagation currently supports closed-shell RKS")

    def _reference_hybrid_coeff(self) -> float:
        xc = getattr(self.mf, "xc", None)
        numint = getattr(self.mf, "_numint", None)
        if xc is None or numint is None:
            return 0.0
        rsh_and_hybrid_coeff = getattr(numint, "rsh_and_hybrid_coeff", None)
        if rsh_and_hybrid_coeff is None:
            return 0.0
        try:
            _, alpha, hyb = rsh_and_hybrid_coeff(xc, spin=self.mol.spin)
        except Exception:
            return 0.0
        return float(max(abs(alpha), abs(hyb)))

    def _validate_density_backend(self) -> None:
        if self._hybrid_coeff > 1.0e-14 and self.real_density_for_veff:
            raise NotImplementedError(
                "Hybrid/HF RT propagation requires the full complex density "
                "matrix in get_veff. Set real_density_for_veff=False and use a "
                "backend validated for complex density matrices."
            )
        if self._veff_on_gpu and not self.real_density_for_veff:
            raise NotImplementedError(
                "GPU4PySCF complex-density get_veff is not supported by this "
                "propagator path yet. The current GPU path is limited to pure "
                "local/semi-local DFT with real_density_for_veff=True."
            )
