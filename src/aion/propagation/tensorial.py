"""Experimental mixed-index density propagation on an evolving AO manifold.

This module deliberately does not participate in the production propagator
factory.  It implements the tensorial density ``D = P S`` and advances it by
similarity with an uncorrected fourth-order Gauss--Magnus transport link.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from aion.backends import ArrayBackend
from aion.config import RationalApproximation
from aion.errors import PropagationError
from aion.formulations import EOMTriple
from aion.propagation.linalg import cross_metric_residual, rational_map, relative_frobenius


@dataclass(frozen=True, slots=True)
class ExperimentalGaussMagnusHistory:
    """Exact endpoint metrics and EOM triples at both Gauss nodes per step."""

    endpoint_metrics: tuple[Any, ...]
    gauss_minus_triples: tuple[EOMTriple, ...]
    gauss_plus_triples: tuple[EOMTriple, ...]
    interval_au: float
    hbar: float = 1.0

    def __post_init__(self) -> None:
        intervals = len(self.gauss_minus_triples)
        if intervals == 0:
            raise PropagationError("Gauss--Magnus history must contain at least one interval")
        if len(self.gauss_plus_triples) != intervals:
            raise PropagationError("Gauss--Magnus node histories must have equal length")
        if len(self.endpoint_metrics) != intervals + 1:
            raise PropagationError(
                "Gauss--Magnus history requires one more endpoint metric than interval"
            )
        if not math.isfinite(self.interval_au) or self.interval_au <= 0.0:
            raise PropagationError("Gauss--Magnus interval must be finite and positive")
        if not math.isfinite(self.hbar) or self.hbar <= 0.0:
            raise PropagationError("Gauss--Magnus hbar must be finite and positive")


@dataclass(frozen=True, slots=True)
class MixedDensityStepDiagnostics:
    """Uncorrected geometric and algebraic residuals for one Magnus step."""

    cross_metric_residual: float
    trace_drift: float
    trace_imaginary_abs: float
    input_idempotency_residual: float
    output_idempotency_residual: float
    metric_hermiticity_residual: float
    contravariant_hermiticity_residual: float


@dataclass(frozen=True, slots=True)
class ExperimentalMixedDensityPropagation:
    """Boundary mixed densities, raw links, and diagnostics for a trajectory."""

    mixed_densities: tuple[Any, ...]
    contravariant_densities: tuple[Any, ...]
    links: tuple[Any, ...]
    diagnostics: tuple[MixedDensityStepDiagnostics, ...]
    metric_correction_applied: bool = False


def contravariant_to_mixed_density(
    contravariant_density: Any,
    metric: Any,
    backend: ArrayBackend,
) -> Any:
    """Raise one AO-density index: ``D^mu_nu = P^{mu lambda} S_lambda nu``."""

    _validate_square_pair(contravariant_density, metric, backend, "contravariant density")
    return contravariant_density @ metric


def mixed_to_contravariant_density(
    mixed_density: Any,
    metric: Any,
    backend: ArrayBackend,
) -> Any:
    """Recover ``P = D S^-1`` by a linear solve, without forming an inverse."""

    _validate_square_pair(mixed_density, metric, backend, "mixed density")
    xp = backend.namespace
    try:
        result = xp.linalg.solve(metric.T, mixed_density.T).T
    except Exception as exc:
        raise PropagationError("mixed-density metric solve failed") from exc
    _assert_finite(result, backend, "contravariant density")
    return result


def mixed_eom_generator(
    triple: EOMTriple,
    backend: ArrayBackend,
    *,
    hbar: float = 1.0,
) -> Any:
    r"""Return ``G=S^-1(-omega-i K/hbar)`` for ``Cdot=G C``."""

    if not isinstance(triple, EOMTriple):
        raise TypeError("triple must be an EOMTriple")
    if not math.isfinite(hbar) or hbar <= 0.0:
        raise PropagationError("hbar must be finite and positive")
    _validate_triple(triple, backend)
    xp = backend.namespace
    try:
        result = xp.linalg.solve(
            triple.metric,
            -triple.connection - (1j / float(hbar)) * triple.hamiltonian_eom,
        )
    except Exception as exc:
        raise PropagationError("mixed EOM generator solve failed") from exc
    _assert_finite(result, backend, "mixed EOM generator")
    return result


def fourth_order_gauss_magnus_link(
    gauss_minus: EOMTriple,
    gauss_plus: EOMTriple,
    interval_au: float,
    backend: ArrayBackend,
    *,
    hbar: float = 1.0,
) -> Any:
    r"""Build the uncorrected fourth-order two-node Gauss--Magnus link.

    With the earlier Gauss node denoted by ``-`` and the later by ``+``,

    ``Omega = h/2 (G_- + G_+) - sqrt(3) h^2/12 [G_-, G_+]``.

    The exponential is evaluated with the fourth-order-compatible diagonal
    ``[2/2]`` Pade map.  No endpoint-metric correction is applied.
    """

    interval = float(interval_au)
    if not math.isfinite(interval) or interval <= 0.0:
        raise PropagationError("Gauss--Magnus interval must be finite and positive")
    left = mixed_eom_generator(gauss_minus, backend, hbar=hbar)
    right = mixed_eom_generator(gauss_plus, backend, hbar=hbar)
    if left.shape != right.shape:
        raise PropagationError("Gauss-node generators have incompatible shapes")
    commutator = left @ right - right @ left
    omega = (
        0.5 * interval * (left + right) - math.sqrt(3.0) * interval * interval * commutator / 12.0
    )
    return rational_map(
        omega,
        1.0,
        RationalApproximation.PADE_22,
        backend=backend,
    )


def similarity_transport(mixed_density: Any, link: Any, backend: ArrayBackend) -> Any:
    """Return ``link D link^-1`` using a right-side linear solve."""

    _validate_square_pair(mixed_density, link, backend, "mixed density")
    xp = backend.namespace
    left = link @ mixed_density
    try:
        result = xp.linalg.solve(link.T, left.T).T
    except Exception as exc:
        raise PropagationError("mixed-density similarity solve failed") from exc
    _assert_finite(result, backend, "transported mixed density")
    return result


def mixed_idempotency_residual(mixed_density: Any, backend: ArrayBackend) -> float:
    """Return the relative projector defect ``||D^2-D||``."""

    _validate_square(mixed_density, backend, "mixed density")
    return relative_frobenius(
        mixed_density @ mixed_density - mixed_density,
        mixed_density,
        backend,
    )


def mixed_metric_hermiticity_residual(
    mixed_density: Any,
    metric: Any,
    backend: ArrayBackend,
) -> float:
    r"""Return the mixed-index Hermiticity defect ``||D^dag S-S D||``."""

    _validate_square_pair(mixed_density, metric, backend, "mixed density")
    reference = metric @ mixed_density
    return relative_frobenius(mixed_density.conj().T @ metric - reference, reference, backend)


def propagate_experimental_mixed_density(
    initial_mixed_density: Any,
    history: ExperimentalGaussMagnusHistory,
    backend: ArrayBackend,
) -> ExperimentalMixedDensityPropagation:
    """Propagate a mixed AO density by raw fourth-order Magnus similarities."""

    if not isinstance(history, ExperimentalGaussMagnusHistory):
        raise TypeError("history must be an ExperimentalGaussMagnusHistory")
    _validate_square(initial_mixed_density, backend, "initial mixed density")
    xp = backend.namespace
    dimension = initial_mixed_density.shape[0]
    current = xp.array(initial_mixed_density, dtype=xp.complex128, copy=True)
    mixed_densities = [xp.array(current, dtype=xp.complex128, copy=True)]
    links: list[Any] = []
    diagnostics: list[MixedDensityStepDiagnostics] = []
    initial_metric = history.endpoint_metrics[0]
    _validate_matrix_shape(initial_metric, dimension, backend, "initial endpoint metric")
    contravariant = mixed_to_contravariant_density(current, initial_metric, backend)
    contravariant_densities = [xp.array(contravariant, dtype=xp.complex128, copy=True)]

    for index, (gauss_minus, gauss_plus) in enumerate(
        zip(history.gauss_minus_triples, history.gauss_plus_triples, strict=True)
    ):
        start_metric = history.endpoint_metrics[index]
        target_metric = history.endpoint_metrics[index + 1]
        _validate_matrix_shape(start_metric, dimension, backend, "start endpoint metric")
        _validate_matrix_shape(target_metric, dimension, backend, "target endpoint metric")
        for triple in (gauss_minus, gauss_plus):
            _validate_triple(triple, backend, expected_dimension=dimension)

        input_trace = xp.trace(current)
        input_idempotency = mixed_idempotency_residual(current, backend)
        link = fourth_order_gauss_magnus_link(
            gauss_minus,
            gauss_plus,
            history.interval_au,
            backend,
            hbar=history.hbar,
        )
        current = similarity_transport(current, link, backend)
        contravariant = mixed_to_contravariant_density(current, target_metric, backend)
        output_trace = xp.trace(current)
        trace_scale = xp.maximum(xp.asarray(1.0), xp.abs(input_trace))
        trace_drift = backend.scalar_to_float(xp.abs(output_trace - input_trace) / trace_scale)
        trace_imaginary_abs = backend.scalar_to_float(xp.abs(xp.imag(output_trace)))
        contravariant_hermiticity = relative_frobenius(
            contravariant - contravariant.conj().T,
            contravariant,
            backend,
        )
        diagnostics.append(
            MixedDensityStepDiagnostics(
                cross_metric_residual=cross_metric_residual(
                    link,
                    start_metric,
                    target_metric,
                    backend,
                ),
                trace_drift=trace_drift,
                trace_imaginary_abs=trace_imaginary_abs,
                input_idempotency_residual=input_idempotency,
                output_idempotency_residual=mixed_idempotency_residual(current, backend),
                metric_hermiticity_residual=mixed_metric_hermiticity_residual(
                    current,
                    target_metric,
                    backend,
                ),
                contravariant_hermiticity_residual=contravariant_hermiticity,
            )
        )
        mixed_densities.append(xp.array(current, dtype=xp.complex128, copy=True))
        contravariant_densities.append(xp.array(contravariant, dtype=xp.complex128, copy=True))
        links.append(xp.array(link, dtype=xp.complex128, copy=True))

    return ExperimentalMixedDensityPropagation(
        mixed_densities=tuple(mixed_densities),
        contravariant_densities=tuple(contravariant_densities),
        links=tuple(links),
        diagnostics=tuple(diagnostics),
    )


def _validate_square(value: Any, backend: ArrayBackend, name: str) -> None:
    backend.assert_resident(value, name=name)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise PropagationError(f"{name} must be a square matrix")
    _assert_finite(value, backend, name)


def _validate_square_pair(
    value: Any,
    other: Any,
    backend: ArrayBackend,
    name: str,
) -> None:
    _validate_square(value, backend, name)
    _validate_matrix_shape(other, value.shape[0], backend, "metric or link")


def _validate_matrix_shape(
    value: Any,
    dimension: int,
    backend: ArrayBackend,
    name: str,
) -> None:
    _validate_square(value, backend, name)
    if value.shape != (dimension, dimension):
        raise PropagationError(f"{name} has an incompatible shape")


def _validate_triple(
    triple: EOMTriple,
    backend: ArrayBackend,
    *,
    expected_dimension: int | None = None,
) -> None:
    if not isinstance(triple, EOMTriple):
        raise TypeError("Gauss-node value must be an EOMTriple")
    dimension = triple.metric.shape[0] if getattr(triple.metric, "ndim", 0) == 2 else -1
    for name, value in (
        ("Gauss-node metric", triple.metric),
        ("Gauss-node mechanical matrix", triple.hamiltonian_eom),
        ("Gauss-node connection", triple.connection),
    ):
        _validate_matrix_shape(value, dimension, backend, name)
    if expected_dimension is not None and dimension != expected_dimension:
        raise PropagationError("Gauss-node triple has an incompatible shape")


def _assert_finite(value: Any, backend: ArrayBackend, name: str) -> None:
    xp = backend.namespace
    finite = backend.scalar_to_float(xp.asarray(xp.all(xp.isfinite(value)), dtype=xp.float64))
    if not bool(finite):
        raise PropagationError(f"{name} contains non-finite values")
