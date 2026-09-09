"""Backend-neutral dense algebra for SCEM transport and constraints."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from aion.backends import ArrayBackend
from aion.config import RationalApproximation
from aion.errors import MetricConstraintError, PropagationError, StateIntegrityError
from aion.propagation.types import LinkDiagnostics


def _float(value: Any, backend: ArrayBackend) -> float:
    return backend.scalar_to_float(value)


def _finite(value: Any, backend: ArrayBackend) -> bool:
    xp = backend.namespace
    return bool(_float(xp.all(xp.isfinite(value)), backend))


def relative_frobenius(value: Any, reference: Any, backend: ArrayBackend) -> float:
    """Return ``||value||_F / max(1, ||reference||_F)`` on one backend."""

    xp = backend.namespace
    denominator = xp.maximum(xp.asarray(1.0), xp.linalg.norm(reference))
    return _float(xp.linalg.norm(value) / denominator, backend)


def hermitian_cleanup(
    value: Any,
    *,
    threshold: float,
    backend: ArrayBackend,
    name: str,
    out: Any | None = None,
) -> tuple[Any, float]:
    """Hermitize only contamination already inside a declared roundoff bound."""

    backend.assert_resident(value, name=name)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise StateIntegrityError(f"{name} must be a square matrix")
    if not _finite(value, backend):
        raise StateIntegrityError(f"{name} contains non-finite values")
    anti = value - value.conj().T
    residual = relative_frobenius(anti, value, backend)
    if residual > threshold:
        raise StateIntegrityError(
            f"{name} anti-Hermitian residual {residual:.3e} exceeds "
            f"cleanup threshold {threshold:.3e}"
        )
    cleaned = 0.5 * (value + value.conj().T)
    correction = relative_frobenius(cleaned - value, value, backend)
    if out is not None:
        backend.assert_resident(out, name=f"{name} cleanup buffer")
        if out.shape != value.shape or out.dtype != value.dtype:
            raise StateIntegrityError(f"{name} cleanup buffer is incompatible")
        out[...] = cleaned
        cleaned = out
    return cleaned, correction


def density_from_orbitals(
    coefficients: Any,
    occupations: Any,
    *,
    backend: ArrayBackend,
    out: Any | None = None,
) -> Any:
    """Construct the lower-index AO density ``P=C f C^dagger``."""

    backend.assert_resident(coefficients, name="orbital coefficients")
    backend.assert_resident(occupations, name="orbital occupations")
    if coefficients.ndim != 2 or occupations.shape != (coefficients.shape[1],):
        raise StateIntegrityError("coefficient and occupation shapes disagree")
    value = (coefficients * occupations[None, :]) @ coefficients.conj().T
    if out is None:
        return value
    backend.assert_resident(out, name="density buffer")
    if out.shape != value.shape or out.dtype != value.dtype:
        raise StateIntegrityError("density buffer is incompatible")
    out[...] = value
    return out


def metric_density_residual(
    output_density: Any,
    input_density: Any,
    metric: Any,
    *,
    electron_count: float,
    backend: ArrayBackend,
) -> float:
    """Evaluate the invariant SCEM midpoint-density fixed-point residual."""

    xp = backend.namespace
    delta = output_density - input_density
    squared = xp.real(xp.trace(delta.conj().T @ metric @ delta @ metric))
    squared_value = max(0.0, _float(squared, backend))
    return math.sqrt(squared_value) / max(float(electron_count), 1.0)


def hamiltonian_residual(output: Any, input_value: Any, backend: ArrayBackend) -> float:
    """Independent normalized lower-index Hamiltonian diagnostic."""

    return relative_frobenius(output - input_value, output, backend)


def orbital_metric_residual(coefficients: Any, metric: Any, backend: ArrayBackend) -> float:
    """Return the normalized occupied-orbital metric-orthonormality defect."""

    xp = backend.namespace
    gram = coefficients.conj().T @ metric @ coefficients
    identity = xp.eye(gram.shape[0], dtype=xp.complex128)
    return relative_frobenius(gram - identity, identity, backend)


def cross_metric_residual(
    link: Any,
    start_metric: Any,
    target_metric: Any,
    backend: ArrayBackend,
) -> float:
    return relative_frobenius(
        link.conj().T @ target_metric @ link - start_metric,
        start_metric,
        backend,
    )


def metric_roundoff_limit(size: int) -> float:
    """Dimension-scaled FP64 envelope used only for algebraic constraints."""

    return max(1.0e-12, 8192.0 * np.finfo(np.float64).eps * max(1, int(size)))


def rational_map(
    generator: Any,
    interval_au: float,
    approximation: RationalApproximation,
    *,
    backend: ArrayBackend,
    identity: Any | None = None,
) -> Any:
    """Apply the configured diagonal rational approximation to ``exp(h A)``."""

    backend.assert_resident(generator, name="SCEM generator")
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        raise PropagationError("SCEM generator must be square")
    interval = float(interval_au)
    if not math.isfinite(interval):
        raise PropagationError("SCEM interval must be finite")
    if not isinstance(approximation, RationalApproximation):
        raise PropagationError("unknown rational approximation")
    xp = backend.namespace
    if identity is None:
        unit = xp.eye(generator.shape[0], dtype=xp.complex128)
    else:
        backend.assert_resident(identity, name="SCEM identity")
        if identity.shape != generator.shape:
            raise PropagationError("SCEM identity has the wrong shape")
        unit = identity
    scaled = interval * generator
    if approximation is RationalApproximation.CAYLEY_11:
        numerator = unit + 0.5 * scaled
        denominator = unit - 0.5 * scaled
    else:
        squared = scaled @ scaled
        numerator = unit + 0.5 * scaled + squared / 12.0
        denominator = unit - 0.5 * scaled + squared / 12.0
    try:
        result = xp.linalg.solve(denominator, numerator)
    except Exception as exc:
        raise PropagationError("rational-map linear solve failed") from exc
    if not _finite(result, backend):
        raise PropagationError("rational map produced non-finite values")
    return result


def right_cholesky_metric_link(
    raw_link: Any,
    start_metric: Any,
    target_metric: Any,
    *,
    backend: ArrayBackend,
    apply_correction: bool = True,
) -> tuple[Any, LinkDiagnostics]:
    """Enforce ``U^dagger S_target U=S_start`` by a right correction."""

    for name, value in (
        ("raw transport link", raw_link),
        ("start metric", start_metric),
        ("target metric", target_metric),
    ):
        backend.assert_resident(value, name=name)
    if raw_link.shape != start_metric.shape or target_metric.shape != start_metric.shape:
        raise MetricConstraintError("transport link and endpoint metrics have different shapes")
    raw_residual = cross_metric_residual(raw_link, start_metric, target_metric, backend)
    if not apply_correction:
        return raw_link, LinkDiagnostics(raw_residual, raw_residual, 0.0, False)
    raw_metric = raw_link.conj().T @ target_metric @ raw_link
    raw_metric, _ = hermitian_cleanup(
        raw_metric,
        threshold=metric_roundoff_limit(raw_metric.shape[0]),
        backend=backend,
        name="raw transported metric",
    )
    start_metric_clean, _ = hermitian_cleanup(
        start_metric,
        threshold=metric_roundoff_limit(start_metric.shape[0]),
        backend=backend,
        name="start metric",
    )
    xp = backend.namespace
    try:
        raw_upper = xp.linalg.cholesky(raw_metric).conj().T
        start_upper = xp.linalg.cholesky(start_metric_clean).conj().T
        correction = xp.linalg.solve(raw_upper, start_upper)
    except Exception as exc:
        raise MetricConstraintError("right-Cholesky metric correction failed") from exc
    corrected = raw_link @ correction
    corrected_residual = cross_metric_residual(
        corrected, start_metric_clean, target_metric, backend
    )
    identity = xp.eye(correction.shape[0], dtype=xp.complex128)
    correction_norm = _float(xp.linalg.norm(correction - identity), backend)
    if not all(math.isfinite(item) for item in (corrected_residual, correction_norm)):
        raise MetricConstraintError("metric correction produced non-finite diagnostics")
    return corrected, LinkDiagnostics(
        raw_metric_residual=raw_residual,
        corrected_metric_residual=corrected_residual,
        correction_norm=correction_norm,
        correction_applied=True,
    )


def uncorrected_link_diagnostics(
    link: Any,
    metric: Any,
    backend: ArrayBackend,
) -> LinkDiagnostics:
    residual = cross_metric_residual(link, metric, metric, backend)
    return LinkDiagnostics(residual, residual, 0.0, False)


def pull_lower_matrix(matrix: Any, diagonal_transport: Any) -> Any:
    """Return ``G^dagger matrix G`` for a diagonal transport ``G``."""

    return diagonal_transport.conj()[:, None] * matrix * diagonal_transport[None, :]
