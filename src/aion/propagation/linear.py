"""Linear propagation for a prequalified history of Aion EOM triples."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from aion.backends import ArrayBackend
from aion.config import RationalApproximation
from aion.errors import PropagationError
from aion.formulations import EOMTriple
from aion.propagation.linalg import (
    hermitian_cleanup,
    metric_roundoff_limit,
    orbital_metric_residual,
    rational_map,
    right_cholesky_metric_link,
)
from aion.propagation.types import LinkDiagnostics


@dataclass(frozen=True, slots=True)
class LinearMatrixHistory:
    """Endpoint metrics and midpoint triples on one fixed time grid."""

    endpoint_metrics: tuple[Any, ...]
    midpoint_triples: tuple[EOMTriple, ...]
    interval_au: float
    hbar: float = 1.0

    def __post_init__(self) -> None:
        if len(self.endpoint_metrics) != len(self.midpoint_triples) + 1:
            raise PropagationError(
                "linear history requires one more endpoint metric than midpoint triple"
            )
        if not self.midpoint_triples:
            raise PropagationError("linear history must contain at least one interval")
        if not math.isfinite(self.interval_au) or self.interval_au <= 0.0:
            raise PropagationError("linear-history interval must be finite and positive")
        if not math.isfinite(self.hbar) or self.hbar <= 0.0:
            raise PropagationError("linear-history hbar must be finite and positive")


@dataclass(frozen=True, slots=True)
class LinearStepDiagnostics:
    """Norm and cross-metric evidence for one accepted linear step."""

    link: LinkDiagnostics
    input_metric_residual: float
    output_metric_residual: float


@dataclass(frozen=True, slots=True)
class LinearPropagationHistory:
    """Boundary coefficients and diagnostics from one linear trajectory."""

    coefficients: tuple[Any, ...]
    diagnostics: tuple[LinearStepDiagnostics, ...]
    metric_correction_applied: bool


def propagate_linear_matrix_history(
    initial_coefficients: Any,
    history: LinearMatrixHistory,
    backend: ArrayBackend,
    *,
    approximation: RationalApproximation = RationalApproximation.PADE_22,
    apply_metric_correction: bool = True,
) -> LinearPropagationHistory:
    r"""Propagate ``i*hbar*(S Cdot+omega C)=K C`` by midpoint rational maps.

    The right-Cholesky operation enforces the cross-metric constraint and is
    recorded explicitly. It is a roundoff/discretization correction, not a
    replacement for the supplied temporal connection; qualification must
    show its norm decreases under time-step refinement.
    """

    if not isinstance(history, LinearMatrixHistory):
        raise TypeError("history must be a LinearMatrixHistory")
    if not isinstance(approximation, RationalApproximation):
        raise PropagationError("unknown rational approximation")
    if not isinstance(apply_metric_correction, bool):
        raise PropagationError("apply_metric_correction must be boolean")
    backend.assert_resident(initial_coefficients, name="initial coefficients")
    if initial_coefficients.ndim != 2 or initial_coefficients.shape[1] == 0:
        raise PropagationError("initial coefficients must have shape (nao, norbital)")
    xp = backend.namespace
    current = xp.array(initial_coefficients, dtype=xp.complex128, copy=True)
    coefficients = [xp.array(current, dtype=xp.complex128, copy=True)]
    diagnostics: list[LinearStepDiagnostics] = []
    identity = xp.eye(current.shape[0], dtype=xp.complex128)
    for index, triple in enumerate(history.midpoint_triples):
        start_metric = history.endpoint_metrics[index]
        endpoint_metric = history.endpoint_metrics[index + 1]
        for name, value in (
            ("start metric", start_metric),
            ("midpoint metric", triple.metric),
            ("endpoint metric", endpoint_metric),
            ("midpoint mechanical matrix", triple.hamiltonian_eom),
            ("midpoint connection", triple.connection),
        ):
            backend.assert_resident(value, name=name)
            if value.shape != (current.shape[0], current.shape[0]):
                raise PropagationError(f"{name} has an incompatible shape")
        initial_residual = orbital_metric_residual(current, start_metric, backend)
        generator = xp.linalg.solve(
            triple.metric,
            -triple.connection
            - (1j / float(history.hbar)) * triple.hamiltonian_eom,
        )
        raw = rational_map(
            generator,
            history.interval_au,
            approximation,
            backend=backend,
            identity=identity,
        )
        link, link_diagnostics = right_cholesky_metric_link(
            raw,
            start_metric,
            endpoint_metric,
            backend=backend,
            apply_correction=apply_metric_correction,
        )
        current = link @ current
        output_residual = orbital_metric_residual(current, endpoint_metric, backend)
        if apply_metric_correction and output_residual > metric_roundoff_limit(current.shape[0]):
            raise PropagationError(
                "corrected linear step violates its endpoint metric: "
                f"step={index} residual={output_residual:.3e}"
            )
        coefficients.append(xp.array(current, dtype=xp.complex128, copy=True))
        diagnostics.append(
            LinearStepDiagnostics(
                link=link_diagnostics,
                input_metric_residual=initial_residual,
                output_metric_residual=output_residual,
            )
        )
    return LinearPropagationHistory(
        coefficients=tuple(coefficients),
        diagnostics=tuple(diagnostics),
        metric_correction_applied=apply_metric_correction,
    )


def generalized_spectral_trajectory(
    initial_coefficients: Any,
    metric: Any,
    mechanical: Any,
    times_au: tuple[float, ...],
    backend: ArrayBackend,
    *,
    hbar: float = 1.0,
) -> tuple[Any, ...]:
    """Return the exact fixed generalized-Hamiltonian trajectory."""

    if not times_au or not all(math.isfinite(value) and value >= 0.0 for value in times_au):
        raise PropagationError("spectral times must be finite, nonnegative, and nonempty")
    if not math.isfinite(hbar) or hbar <= 0.0:
        raise PropagationError("hbar must be finite and positive")
    for name, value in (
        ("initial coefficients", initial_coefficients),
        ("metric", metric),
        ("mechanical matrix", mechanical),
    ):
        backend.assert_resident(value, name=name)
    if metric.shape != mechanical.shape or metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise PropagationError("metric and mechanical matrix must be compatible and square")
    if initial_coefficients.shape[0] != metric.shape[0] or initial_coefficients.ndim != 2:
        raise PropagationError("initial coefficients have an incompatible shape")
    xp = backend.namespace
    metric_clean, _ = hermitian_cleanup(
        metric,
        threshold=metric_roundoff_limit(metric.shape[0]),
        backend=backend,
        name="spectral metric",
    )
    mechanical_clean, _ = hermitian_cleanup(
        mechanical,
        threshold=metric_roundoff_limit(mechanical.shape[0]),
        backend=backend,
        name="spectral mechanical matrix",
    )
    try:
        lower = xp.linalg.cholesky(metric_clean)
        identity = xp.eye(metric.shape[0], dtype=xp.complex128)
        inverse_lower = xp.linalg.solve(lower, identity)
        orthogonal_mechanical = inverse_lower @ mechanical_clean @ inverse_lower.conj().T
        eigenvalues, eigenvectors = xp.linalg.eigh(orthogonal_mechanical)
    except Exception as exc:
        raise PropagationError("generalized spectral decomposition failed") from exc
    orthogonal_initial = lower.conj().T @ initial_coefficients
    modal_initial = eigenvectors.conj().T @ orthogonal_initial
    output = []
    for time in times_au:
        phase = xp.exp((-1j / float(hbar)) * eigenvalues * float(time))
        orthogonal = eigenvectors @ (phase[:, None] * modal_initial)
        coefficients = xp.linalg.solve(lower.conj().T, orthogonal)
        output.append(xp.asarray(coefficients, dtype=xp.complex128))
    return tuple(output)
