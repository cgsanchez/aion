"""Variable-metric exponential midpoint propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import scipy.linalg

from .backends import CPUBackend, make_backend
from .gauge import PeierlsGeometry
from .matrix_models import hermitian_part


@dataclass(frozen=True)
class VariableMetricSCEMStepResult:
    """Result of one variable-metric SCEM step."""

    coeff_next: np.ndarray
    rho_next: np.ndarray
    h_mid: np.ndarray
    rho_mid: np.ndarray
    iterations: int
    hamiltonian_residual: float
    density_residual: float | None
    converged: bool


class VariableMetricConvergenceError(RuntimeError):
    """Raised when the variable-metric midpoint fixed point fails."""


def _backend_hermitian_part(matrix, backend):
    matrix = backend.asarray(matrix, dtype=np.complex128)
    return 0.5 * (matrix + matrix.conj().T)


def _transport_matrix_backend(matrix, transport, backend):
    u = backend.asarray(transport, dtype=np.complex128)
    m = backend.asarray(matrix, dtype=np.complex128)
    return u.conj()[:, None] * m * u[None, :]


def _transport_coefficients_backend(coeff, transport, backend):
    u = backend.asarray(transport, dtype=np.complex128)
    c = backend.asarray(coeff, dtype=np.complex128)
    return u[:, None] * c


def _metric_factors(metric: np.ndarray, backend=None) -> tuple[np.ndarray, np.ndarray]:
    backend = CPUBackend() if backend is None else backend
    metric = _backend_hermitian_part(metric, backend)
    eig, vec = backend.eigh(metric)
    if backend.min_float(eig) <= 0.0:
        raise np.linalg.LinAlgError("metric is not positive definite")
    sqrt = (vec * eig**0.5) @ vec.conj().T
    invsqrt = (vec * eig**-0.5) @ vec.conj().T
    return sqrt, invsqrt


def _orthonormal_metric_connection(
    metric: np.ndarray,
    metric_dot: np.ndarray,
    *,
    metric_invsqrt: np.ndarray,
    backend=None,
) -> np.ndarray:
    r"""Return the anti-Hermitian metric connection in the Löwdin frame.

    For ``X = S^{1/2} C`` and

    ``Cdot = ... - 1/2 S^{-1} Sdot C``,

    the orthonormal-frame equation contains

    ``K = (d S^{1/2}/dt) S^{-1/2}
           - 1/2 S^{-1/2} Sdot S^{-1/2}``.

    The Fréchet derivative of the positive square root is evaluated in the
    eigenbasis of ``S``.  ``K`` is anti-Hermitian; explicit projection removes
    only roundoff before it is folded into the effective Hermitian generator.
    """

    backend = CPUBackend() if backend is None else backend
    metric = _backend_hermitian_part(metric, backend)
    metric_dot = _backend_hermitian_part(metric_dot, backend)
    eig, vec = backend.eigh(metric)
    if backend.min_float(eig) <= 0.0:
        raise np.linalg.LinAlgError("metric is not positive definite")
    sqrt_eig = eig**0.5
    dot_eigenbasis = vec.conj().T @ metric_dot @ vec
    sqrt_dot_eigenbasis = dot_eigenbasis / (
        sqrt_eig[:, None] + sqrt_eig[None, :]
    )
    sqrt_dot = vec @ sqrt_dot_eigenbasis @ vec.conj().T
    connection = (
        sqrt_dot @ metric_invsqrt
        - 0.5 * metric_invsqrt @ metric_dot @ metric_invsqrt
    )
    return 0.5 * (connection - connection.conj().T)


def _orthonormal_hamiltonian(
    hamiltonian: np.ndarray,
    metric_invsqrt: np.ndarray,
    backend=None,
) -> np.ndarray:
    backend = CPUBackend() if backend is None else backend
    return _backend_hermitian_part(
        metric_invsqrt @ hamiltonian @ metric_invsqrt,
        backend,
    )


def _exponential_action(
    coeff: np.ndarray,
    *,
    start_metric_sqrt: np.ndarray,
    target_metric_invsqrt: np.ndarray,
    h_orth: np.ndarray,
    dt: float,
    hbar: float,
    backend=None,
) -> np.ndarray:
    backend = CPUBackend() if backend is None else backend
    eig, vec = backend.eigh(_backend_hermitian_part(h_orth, backend))
    coeff_orth = start_metric_sqrt @ coeff
    propagated = vec @ (
        backend.exp((-1j * dt / hbar) * eig)[:, None]
        * (vec.conj().T @ coeff_orth)
    )
    return target_metric_invsqrt @ propagated


class VariableMetricSCEM:
    """Strict exponential midpoint step for time-dependent Peierls metrics."""

    def __init__(
        self,
        geometry: PeierlsGeometry,
        model,
        occupations: np.ndarray,
        *,
        hbar: float = 1.0,
        backend: str | CPUBackend = "cpu",
    ) -> None:
        self.geometry = geometry
        self.model = model
        self.backend = make_backend(backend)
        self.occupations = np.asarray(occupations, dtype=float)
        self.occupations_backend = self.backend.asarray(self.occupations, dtype=float)
        if self.occupations.ndim != 1 or self.occupations.size == 0:
            raise ValueError("occupations must be a nonempty one-dimensional array")
        if hbar <= 0.0:
            raise ValueError("hbar must be positive")
        self.hbar = float(hbar)

    @property
    def nocc(self) -> int:
        return int(self.occupations.size)

    @property
    def nao(self) -> int:
        return self.geometry.anchors.nao

    def density_from_coefficients(self, coeff: np.ndarray) -> np.ndarray:
        coeff = self.backend.asarray(coeff, dtype=np.complex128)
        if coeff.shape != (self.nao, self.nocc):
            raise ValueError(f"coeff must have shape {(self.nao, self.nocc)}")
        return (coeff * self.occupations_backend[None, :]) @ coeff.conj().T

    def orthonormality_error(self, coeff: np.ndarray, t: float) -> float:
        metric = self.backend.asarray(self.geometry.metric(t), dtype=np.complex128)
        coeff = self.backend.asarray(coeff, dtype=np.complex128)
        overlap = coeff.conj().T @ metric @ coeff
        eye = self.backend.eye(self.nocc, dtype=np.complex128)
        return self.backend.norm_float(overlap - eye)

    def step(
        self,
        coeff: np.ndarray,
        *,
        time: float,
        dt: float,
        midpoint_tolerance: float = 1.0e-10,
        density_tolerance: float | None = 1.0e-10,
        max_iterations: int = 50,
        mixing: float = 1.0,
    ) -> VariableMetricSCEMStepResult:
        """Advance one step with a midpoint fixed-point solve."""

        if dt == 0.0:
            raise ValueError("dt must be nonzero")
        if midpoint_tolerance <= 0.0:
            raise ValueError("midpoint_tolerance must be positive")
        if density_tolerance is not None and density_tolerance <= 0.0:
            raise ValueError("density_tolerance must be positive or None")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not (0.0 < mixing <= 1.0):
            raise ValueError("mixing must be in (0, 1]")

        backend = self.backend
        coeff = backend.asarray(coeff, dtype=np.complex128)
        if coeff.shape != (self.nao, self.nocc):
            raise ValueError(f"coeff must have shape {(self.nao, self.nocc)}")

        t_mid = time + 0.5 * dt
        t_next = time + dt
        u_mid = backend.asarray(
            self.geometry.ao_transport(time, t_mid),
            dtype=np.complex128,
        )
        u_next = backend.asarray(
            self.geometry.ao_transport(time, t_next),
            dtype=np.complex128,
        )

        s_start = backend.asarray(self.geometry.metric(time), dtype=np.complex128)
        s_mid = _transport_matrix_backend(self.geometry.metric(t_mid), u_mid, backend)
        s_next = _transport_matrix_backend(self.geometry.metric(t_next), u_next, backend)
        s_mid_dot = _transport_matrix_backend(
            self.geometry.covariant_metric_dot(t_mid),
            u_mid,
            backend,
        )

        s_start_sqrt, _ = _metric_factors(s_start, backend)
        _, s_mid_invsqrt = _metric_factors(s_mid, backend)
        _, s_next_invsqrt = _metric_factors(s_next, backend)
        orthonormal_connection = _orthonormal_metric_connection(
            s_mid,
            s_mid_dot,
            metric_invsqrt=s_mid_invsqrt,
            backend=backend,
        )

        rho_start = self.density_from_coefficients(coeff)
        h_iter_original = self.model.hamiltonian(rho_start, t_mid, self.geometry)
        h_iter = _transport_matrix_backend(h_iter_original, u_mid, backend)
        previous_rho_mid = None
        h_residual = float("nan")
        d_residual = None

        for iteration in range(1, max_iterations + 1):
            h_orth = _orthonormal_hamiltonian(h_iter, s_mid_invsqrt, backend)
            h_effective_orth = _backend_hermitian_part(
                h_orth + 1j * self.hbar * orthonormal_connection,
                backend,
            )
            coeff_mid_parallel = _exponential_action(
                coeff,
                start_metric_sqrt=s_start_sqrt,
                target_metric_invsqrt=s_mid_invsqrt,
                h_orth=h_effective_orth,
                dt=0.5 * dt,
                hbar=self.hbar,
                backend=backend,
            )
            coeff_mid = _transport_coefficients_backend(
                coeff_mid_parallel,
                u_mid,
                backend,
            )
            rho_mid = _backend_hermitian_part(
                self.density_from_coefficients(coeff_mid),
                backend,
            )
            h_built_original = self.model.hamiltonian(rho_mid, t_mid, self.geometry)
            h_built = _transport_matrix_backend(h_built_original, u_mid, backend)
            h_built_orth = _orthonormal_hamiltonian(
                h_built,
                s_mid_invsqrt,
                backend,
            )

            numerator = backend.norm_float(h_built_orth - h_orth)
            denominator = max(1.0, backend.norm_float(h_built_orth))
            h_residual = float(numerator / denominator)
            if previous_rho_mid is not None:
                dnum = backend.norm_float(rho_mid - previous_rho_mid)
                dden = max(1.0, backend.norm_float(rho_mid))
                d_residual = float(dnum / dden)

            converged = h_residual <= midpoint_tolerance
            if density_tolerance is not None and d_residual is not None:
                converged = converged and d_residual <= density_tolerance

            if converged:
                h_accept_orth = _backend_hermitian_part(
                    h_built_orth + 1j * self.hbar * orthonormal_connection,
                    backend,
                )
                coeff_next_parallel = _exponential_action(
                    coeff,
                    start_metric_sqrt=s_start_sqrt,
                    target_metric_invsqrt=s_next_invsqrt,
                    h_orth=h_accept_orth,
                    dt=dt,
                    hbar=self.hbar,
                    backend=backend,
                )
                coeff_next = _transport_coefficients_backend(
                    coeff_next_parallel,
                    u_next,
                    backend,
                )
                rho_next = _backend_hermitian_part(
                    self.density_from_coefficients(coeff_next),
                    backend,
                )
                return VariableMetricSCEMStepResult(
                    coeff_next=coeff_next,
                    rho_next=rho_next,
                    h_mid=hermitian_part(h_built_original),
                    rho_mid=rho_mid,
                    iterations=iteration,
                    hamiltonian_residual=h_residual,
                    density_residual=d_residual,
                    converged=True,
                )

            previous_rho_mid = rho_mid
            h_iter = _backend_hermitian_part(
                (1.0 - mixing) * h_iter + mixing * h_built,
                backend,
            )

        raise VariableMetricConvergenceError(
            "variable-metric SCEM midpoint did not converge: "
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
        **step_kwargs,
    ) -> Iterator[tuple[np.ndarray, VariableMetricSCEMStepResult | None]]:
        if nsteps < 0:
            raise ValueError("nsteps must be nonnegative")
        coeff = np.asarray(coeff0, dtype=np.complex128)
        yield coeff.copy(), None
        for step in range(1, nsteps + 1):
            result = self.step(
                coeff,
                time=t0 + (step - 1) * dt,
                dt=dt,
                **step_kwargs,
            )
            coeff = result.coeff_next
            yield coeff.copy(), result
