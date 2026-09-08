"""Connection-aware generalized-Cayley propagation in a nonorthogonal basis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .backends import CPUBackend, make_backend
from .gauge import PeierlsGeometry
from .variable_metric import VariableMetricConvergenceError


def _hermitian_part(matrix, backend):
    value = backend.asarray(matrix, dtype=np.complex128)
    return 0.5 * (value + value.conj().T)


def _transport_lower_matrix(matrix, transport, backend):
    """Return ``G^dagger matrix G`` for a diagonal AO transport ``G``."""

    value = backend.asarray(matrix, dtype=np.complex128)
    diagonal = backend.asarray(transport, dtype=np.complex128)
    return diagonal.conj()[:, None] * value * diagonal[None, :]


@dataclass(frozen=True)
class ConnectionLinkResult:
    """Finite connection transport between two time-dependent metric fibers."""

    matrix: np.ndarray
    raw_metric_error: float
    corrected_metric_error: float
    correction_norm: float


@dataclass(frozen=True)
class ConnectionCayleySCEMStepResult:
    """Result of one connection-aware self-consistent midpoint step."""

    coeff_next: np.ndarray
    rho_next: np.ndarray
    h_mid: np.ndarray
    rho_mid: np.ndarray
    iterations: int
    hamiltonian_residual: float
    density_residual: float | None
    converged: bool
    left_connection_metric_error: float
    right_connection_metric_error: float
    maximum_connection_correction_norm: float


class ConnectionCayleySCEM:
    r"""Analytic-site-transport, covariant-midpoint propagator.

    For the lower-index projected connection ``omega`` this class evaluates
    ``Gamma = S^{-1} omega`` directly from the analytic field source and AO
    matrices.  It removes the diagonal site connection with exact analytic
    transport and advances the remaining connection and intrinsic Hamiltonian
    in one site-parallel midpoint generator,

    ``A_m = S_m^{-1}[-omega_m - i H_int,m / hbar]``.

    The default rational map is diagonal Pade ``[2/2]``; a ``[1/1]`` Cayley
    map is available for diagnostics.  A right Cholesky correction removes
    only the finite-step cross-metric defect.  It tends to the identity and
    does not define the physical transport.  No time-dependent orthogonalizing
    frame is used.

    A model may expose ``connection_residual(t, geometry)`` and
    ``intrinsic_hamiltonian(rho, t, geometry)``.  ``P0E1Model`` uses these to
    place ``eta_E1 = i V_E1 / hbar`` in the projected connection while removing
    ``V_E1`` from the Hamiltonian sector.  Models without those methods reduce
    to pure P0 propagation.
    """

    def __init__(
        self,
        geometry: PeierlsGeometry,
        model,
        occupations: np.ndarray,
        *,
        hbar: float = 1.0,
        backend: str | CPUBackend = "cpu",
        hamiltonian_propagator: str = "pade22",
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
        if hamiltonian_propagator not in {"cayley", "pade22"}:
            raise ValueError("hamiltonian_propagator must be 'cayley' or 'pade22'")
        self.hbar = float(hbar)
        self.hamiltonian_propagator = hamiltonian_propagator

    @property
    def nocc(self) -> int:
        return int(self.occupations.size)

    @property
    def nao(self) -> int:
        return self.geometry.anchors.nao

    def density_from_coefficients(self, coeff: np.ndarray) -> np.ndarray:
        value = self.backend.asarray(coeff, dtype=np.complex128)
        if value.shape != (self.nao, self.nocc):
            raise ValueError(f"coeff must have shape {(self.nao, self.nocc)}")
        return (value * self.occupations_backend[None, :]) @ value.conj().T

    def orthonormality_error(self, coeff: np.ndarray, t: float) -> float:
        metric = self.backend.asarray(self.geometry.metric(t), dtype=np.complex128)
        value = self.backend.asarray(coeff, dtype=np.complex128)
        overlap = value.conj().T @ metric @ value
        identity = self.backend.eye(self.nocc, dtype=np.complex128)
        return self.backend.norm_float(overlap - identity)

    def lower_time_connection(self, t: float):
        """Return the complete analytic lower-index connection at ``t``."""

        backend = self.backend
        connection = backend.asarray(
            self.geometry.time_connection(t),
            dtype=np.complex128,
        )
        residual_function = getattr(self.model, "connection_residual", None)
        if residual_function is not None:
            connection = connection + backend.asarray(
                residual_function(t, self.geometry),
                dtype=np.complex128,
            )
        return connection

    def mixed_time_connection(self, t: float):
        """Return ``Gamma(t) = S(t)^{-1} omega(t)``."""

        metric = self.backend.asarray(self.geometry.metric(t), dtype=np.complex128)
        return self.backend.solve(metric, self.lower_time_connection(t))

    def _intrinsic_hamiltonian(self, density, t: float):
        function = getattr(self.model, "intrinsic_hamiltonian", None)
        if function is None:
            function = self.model.hamiltonian
        return _hermitian_part(function(density, t, self.geometry), self.backend)

    def _metric_corrected_link(self, raw, start_metric, target_metric):
        """Project a high-order link onto the exact cross-metric constraint.

        The right Cholesky correction is the identity when the raw link already
        satisfies ``R^dagger S_b R = S_a``.  It corrects only the finite-step
        integration defect; it is not used to define the physical connection.
        """

        backend = self.backend
        start = _hermitian_part(start_metric, backend)
        target = _hermitian_part(target_metric, backend)
        raw_metric = _hermitian_part(raw.conj().T @ target @ raw, backend)
        raw_error = backend.norm_float(raw_metric - start)

        raw_upper = backend.cholesky(raw_metric).conj().T
        start_upper = backend.cholesky(start).conj().T
        correction = backend.solve(raw_upper, start_upper)
        corrected = raw @ correction
        corrected_error = backend.norm_float(
            corrected.conj().T @ target @ corrected - start
        )
        identity = backend.eye(self.nao, dtype=np.complex128)
        correction_norm = backend.norm_float(correction - identity)
        return corrected, raw_error, corrected_error, correction_norm

    def connection_link(self, start_time: float, target_time: float) -> ConnectionLinkResult:
        """Return a second-order physical connection link between two times."""

        backend = self.backend
        if target_time == start_time:
            identity = backend.eye(self.nao, dtype=np.complex128)
            return ConnectionLinkResult(identity, 0.0, 0.0, 0.0)

        interval = float(target_time - start_time)
        midpoint = start_time + 0.5 * interval
        site_mid = backend.asarray(
            self.geometry.ao_transport(start_time, midpoint),
            dtype=np.complex128,
        )
        site_target = backend.asarray(
            self.geometry.ao_transport(start_time, target_time),
            dtype=np.complex128,
        )

        metric_start = backend.asarray(
            self.geometry.metric(start_time),
            dtype=np.complex128,
        )
        metric_mid_original = backend.asarray(
            self.geometry.metric(midpoint),
            dtype=np.complex128,
        )
        metric_target_original = backend.asarray(
            self.geometry.metric(target_time),
            dtype=np.complex128,
        )
        metric_mid = _transport_lower_matrix(
            metric_mid_original,
            site_mid,
            backend,
        )
        metric_target = _transport_lower_matrix(
            metric_target_original,
            site_target,
            backend,
        )

        sigma = backend.asarray(self.geometry.ao_sigma(midpoint), dtype=np.complex128)
        site_connection = metric_mid_original * sigma[None, :]
        residual_original = self.lower_time_connection(midpoint) - site_connection
        residual_mid = _transport_lower_matrix(
            residual_original,
            site_mid,
            backend,
        )
        gamma_mid = backend.solve(metric_mid, residual_mid)

        identity = backend.eye(self.nao, dtype=np.complex128)
        raw_parallel = backend.solve(
            identity + 0.5 * interval * gamma_mid,
            identity - 0.5 * interval * gamma_mid,
        )
        parallel, raw_error, corrected_error, correction_norm = (
            self._metric_corrected_link(
                raw_parallel,
                metric_start,
                metric_target,
            )
        )
        link = site_target[:, None] * parallel
        original_error = backend.norm_float(
            link.conj().T @ metric_target_original @ link - metric_start
        )
        return ConnectionLinkResult(
            matrix=link,
            raw_metric_error=raw_error,
            corrected_metric_error=max(corrected_error, original_error),
            correction_norm=correction_norm,
        )

    def _rational_map(self, generator, interval: float):
        """Return a diagonal-Pade approximation to ``exp(interval A)``."""

        backend = self.backend
        identity = backend.eye(self.nao, dtype=np.complex128)
        scaled = interval * generator
        if self.hamiltonian_propagator == "cayley":
            return backend.solve(identity - 0.5 * scaled, identity + 0.5 * scaled)

        squared = scaled @ scaled
        numerator = identity + 0.5 * scaled + squared / 12.0
        denominator = identity - 0.5 * scaled + squared / 12.0
        return backend.solve(denominator, numerator)

    def _covariant_parallel_map(
        self,
        *,
        start_metric,
        target_metric,
        midpoint_metric,
        midpoint_connection,
        midpoint_hamiltonian,
        interval: float,
    ):
        """Build one full covariant midpoint map in a site-parallel frame."""

        generator = self.backend.solve(
            midpoint_metric,
            -midpoint_connection
            - (1j / self.hbar) * midpoint_hamiltonian,
        )
        raw = self._rational_map(generator, interval)
        return self._metric_corrected_link(raw, start_metric, target_metric)

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
    ) -> ConnectionCayleySCEMStepResult:
        """Advance one symmetric connection--Cayley midpoint step."""

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
        value = backend.asarray(coeff, dtype=np.complex128)
        if value.shape != (self.nao, self.nocc):
            raise ValueError(f"coeff must have shape {(self.nao, self.nocc)}")

        midpoint = time + 0.5 * dt
        target = time + dt
        site_mid = backend.asarray(
            self.geometry.ao_transport(time, midpoint),
            dtype=np.complex128,
        )
        site_target = backend.asarray(
            self.geometry.ao_transport(time, target),
            dtype=np.complex128,
        )
        metric_start = backend.asarray(self.geometry.metric(time), dtype=np.complex128)
        metric_mid_original = backend.asarray(
            self.geometry.metric(midpoint),
            dtype=np.complex128,
        )
        metric_target_original = backend.asarray(
            self.geometry.metric(target),
            dtype=np.complex128,
        )
        metric_mid = _transport_lower_matrix(metric_mid_original, site_mid, backend)
        metric_target = _transport_lower_matrix(
            metric_target_original,
            site_target,
            backend,
        )
        sigma_mid = backend.asarray(
            self.geometry.ao_sigma(midpoint),
            dtype=np.complex128,
        )
        site_connection_mid = metric_mid_original * sigma_mid[None, :]
        connection_mid = _transport_lower_matrix(
            self.lower_time_connection(midpoint) - site_connection_mid,
            site_mid,
            backend,
        )

        connection_guess = self.connection_link(time, midpoint)
        connection_mid_coeff = connection_guess.matrix @ value
        connection_mid_density = _hermitian_part(
            self.density_from_coefficients(connection_mid_coeff),
            backend,
        )
        h_iter = self._intrinsic_hamiltonian(connection_mid_density, midpoint)
        previous_rho_mid = None
        h_residual = float("nan")
        density_residual = None

        for iteration in range(1, max_iterations + 1):
            h_iter_parallel = _transport_lower_matrix(h_iter, site_mid, backend)
            midpoint_map, _, midpoint_map_error, midpoint_correction = (
                self._covariant_parallel_map(
                    start_metric=metric_start,
                    target_metric=metric_mid,
                    midpoint_metric=metric_mid,
                    midpoint_connection=connection_mid,
                    midpoint_hamiltonian=h_iter_parallel,
                    interval=0.5 * dt,
                )
            )
            coeff_mid_parallel = midpoint_map @ value
            coeff_mid = site_mid[:, None] * coeff_mid_parallel
            rho_mid = _hermitian_part(
                self.density_from_coefficients(coeff_mid),
                backend,
            )
            h_built = self._intrinsic_hamiltonian(rho_mid, midpoint)
            numerator = backend.norm_float(h_built - h_iter)
            denominator = max(1.0, backend.norm_float(h_built))
            h_residual = float(numerator / denominator)
            if previous_rho_mid is not None:
                density_residual = float(
                    backend.norm_float(rho_mid - previous_rho_mid)
                    / max(1.0, backend.norm_float(rho_mid))
                )

            converged = h_residual <= midpoint_tolerance
            if density_tolerance is not None and density_residual is not None:
                converged = converged and density_residual <= density_tolerance

            if converged:
                h_built_parallel = _transport_lower_matrix(h_built, site_mid, backend)
                endpoint_map, _, endpoint_map_error, endpoint_correction = (
                    self._covariant_parallel_map(
                        start_metric=metric_start,
                        target_metric=metric_target,
                        midpoint_metric=metric_mid,
                        midpoint_connection=connection_mid,
                        midpoint_hamiltonian=h_built_parallel,
                        interval=dt,
                    )
                )
                coeff_next = site_target[:, None] * (endpoint_map @ value)
                rho_next = _hermitian_part(
                    self.density_from_coefficients(coeff_next),
                    backend,
                )
                return ConnectionCayleySCEMStepResult(
                    coeff_next=coeff_next,
                    rho_next=rho_next,
                    h_mid=h_built,
                    rho_mid=rho_mid,
                    iterations=iteration,
                    hamiltonian_residual=h_residual,
                    density_residual=density_residual,
                    converged=True,
                    left_connection_metric_error=midpoint_map_error,
                    right_connection_metric_error=endpoint_map_error,
                    maximum_connection_correction_norm=max(
                        midpoint_correction,
                        endpoint_correction,
                    ),
                )

            previous_rho_mid = rho_mid
            h_iter = _hermitian_part(
                (1.0 - mixing) * h_iter + mixing * h_built,
                backend,
            )

        raise VariableMetricConvergenceError(
            "connection-Cayley SCEM midpoint did not converge: "
            f"time={time} dt={dt} iterations={max_iterations} "
            f"h_residual={h_residual} density_residual={density_residual}"
        )

    def propagate(
        self,
        coeff0: np.ndarray,
        *,
        dt: float,
        nsteps: int,
        t0: float = 0.0,
        **step_kwargs,
    ) -> Iterator[tuple[np.ndarray, ConnectionCayleySCEMStepResult | None]]:
        if nsteps < 0:
            raise ValueError("nsteps must be nonnegative")
        coeff = self.backend.asarray(coeff0, dtype=np.complex128)
        yield self.backend.copy(coeff), None
        for step in range(1, nsteps + 1):
            result = self.step(
                coeff,
                time=t0 + (step - 1) * dt,
                dt=dt,
                **step_kwargs,
            )
            coeff = result.coeff_next
            yield self.backend.copy(coeff), result
