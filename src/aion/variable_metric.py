"""Variable-metric exponential midpoint propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import scipy.linalg

from .gauge import PeierlsGeometry
from .matrix_models import density_from_coefficients, hermitian_part


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


def _metric_factors(metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    metric = hermitian_part(metric)
    eig, vec = scipy.linalg.eigh(metric, check_finite=False)
    if np.min(eig) <= 0.0:
        raise np.linalg.LinAlgError("metric is not positive definite")
    sqrt = (vec * eig**0.5) @ vec.conj().T
    invsqrt = (vec * eig**-0.5) @ vec.conj().T
    return sqrt, invsqrt


def _orthonormal_hamiltonian(hamiltonian: np.ndarray, metric_invsqrt: np.ndarray) -> np.ndarray:
    return hermitian_part(metric_invsqrt @ hamiltonian @ metric_invsqrt)


def _exponential_action(
    coeff: np.ndarray,
    *,
    start_metric_sqrt: np.ndarray,
    target_metric_invsqrt: np.ndarray,
    h_orth: np.ndarray,
    dt: float,
    hbar: float,
) -> np.ndarray:
    eig, vec = scipy.linalg.eigh(hermitian_part(h_orth), check_finite=False)
    coeff_orth = start_metric_sqrt @ coeff
    propagated = vec @ (
        np.exp((-1j * dt / hbar) * eig)[:, None] * (vec.conj().T @ coeff_orth)
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
    ) -> None:
        self.geometry = geometry
        self.model = model
        self.occupations = np.asarray(occupations, dtype=float)
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
        coeff = np.asarray(coeff, dtype=np.complex128)
        if coeff.shape != (self.nao, self.nocc):
            raise ValueError(f"coeff must have shape {(self.nao, self.nocc)}")
        return density_from_coefficients(coeff, self.occupations)

    def orthonormality_error(self, coeff: np.ndarray, t: float) -> float:
        metric = self.geometry.metric(t)
        overlap = coeff.conj().T @ metric @ coeff
        return float(np.linalg.norm(overlap - np.eye(self.nocc)))

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

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if midpoint_tolerance <= 0.0:
            raise ValueError("midpoint_tolerance must be positive")
        if density_tolerance is not None and density_tolerance <= 0.0:
            raise ValueError("density_tolerance must be positive or None")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not (0.0 < mixing <= 1.0):
            raise ValueError("mixing must be in (0, 1]")

        coeff = np.asarray(coeff, dtype=np.complex128)
        if coeff.shape != (self.nao, self.nocc):
            raise ValueError(f"coeff must have shape {(self.nao, self.nocc)}")

        t_mid = time + 0.5 * dt
        t_next = time + dt
        u_mid = self.geometry.ao_transport(time, t_mid)
        u_next = self.geometry.ao_transport(time, t_next)

        s_start = self.geometry.metric(time)
        s_mid = self.geometry.transport_matrix(self.geometry.metric(t_mid), u_mid)
        s_next = self.geometry.transport_matrix(self.geometry.metric(t_next), u_next)

        s_start_sqrt, _ = _metric_factors(s_start)
        _, s_mid_invsqrt = _metric_factors(s_mid)
        _, s_next_invsqrt = _metric_factors(s_next)

        rho_start = self.density_from_coefficients(coeff)
        h_iter_original = self.model.hamiltonian(rho_start, t_mid, self.geometry)
        h_iter = self.geometry.transport_matrix(h_iter_original, u_mid)
        previous_rho_mid = None
        h_residual = float("nan")
        d_residual = None

        for iteration in range(1, max_iterations + 1):
            h_orth = _orthonormal_hamiltonian(h_iter, s_mid_invsqrt)
            coeff_mid_parallel = _exponential_action(
                coeff,
                start_metric_sqrt=s_start_sqrt,
                target_metric_invsqrt=s_mid_invsqrt,
                h_orth=h_orth,
                dt=0.5 * dt,
                hbar=self.hbar,
            )
            coeff_mid = self.geometry.transport_coefficients(coeff_mid_parallel, u_mid)
            rho_mid = hermitian_part(self.density_from_coefficients(coeff_mid))
            h_built_original = self.model.hamiltonian(rho_mid, t_mid, self.geometry)
            h_built = self.geometry.transport_matrix(h_built_original, u_mid)
            h_built_orth = _orthonormal_hamiltonian(h_built, s_mid_invsqrt)

            numerator = np.linalg.norm(h_built_orth - h_orth)
            denominator = max(1.0, float(np.linalg.norm(h_built_orth)))
            h_residual = float(numerator / denominator)
            if previous_rho_mid is not None:
                dnum = np.linalg.norm(rho_mid - previous_rho_mid)
                dden = max(1.0, float(np.linalg.norm(rho_mid)))
                d_residual = float(dnum / dden)

            converged = h_residual <= midpoint_tolerance
            if density_tolerance is not None and d_residual is not None:
                converged = converged and d_residual <= density_tolerance

            if converged:
                h_accept_orth = h_built_orth
                coeff_next_parallel = _exponential_action(
                    coeff,
                    start_metric_sqrt=s_start_sqrt,
                    target_metric_invsqrt=s_next_invsqrt,
                    h_orth=h_accept_orth,
                    dt=dt,
                    hbar=self.hbar,
                )
                coeff_next = self.geometry.transport_coefficients(
                    coeff_next_parallel,
                    u_next,
                )
                rho_next = hermitian_part(self.density_from_coefficients(coeff_next))
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
            h_iter = hermitian_part((1.0 - mixing) * h_iter + mixing * h_built)

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
