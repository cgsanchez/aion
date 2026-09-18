from __future__ import annotations

import math

import numpy as np

from aion.backends import NumPyBackend
from aion.formulations import EOMTriple
from aion.propagation import (
    ExperimentalGaussMagnusHistory,
    contravariant_to_mixed_density,
    mixed_to_contravariant_density,
    propagate_experimental_mixed_density,
)

_S0 = np.asarray(((1.31, 0.17 - 0.08j), (0.17 + 0.08j, 0.94)))
_C0 = np.asarray(((0.71 + 0.13j,), (-0.22 + 0.39j,)))
_C0 /= np.sqrt((_C0.conj().T @ _S0 @ _C0).real.item())
_P0 = _C0 @ _C0.conj().T
_D0 = _P0 @ _S0


def _transport(time: float) -> tuple[np.ndarray, np.ndarray]:
    diagonal_0 = np.exp(0.19 * time)
    diagonal_1 = np.exp(-0.11 * time)
    link = np.asarray(
        (
            (diagonal_0, 0.27 * np.sin(0.83 * time)),
            (0.0, diagonal_1),
        ),
        dtype=np.complex128,
    )
    derivative = np.asarray(
        (
            (0.19 * diagonal_0, 0.27 * 0.83 * np.cos(0.83 * time)),
            (0.0, -0.11 * diagonal_1),
        ),
        dtype=np.complex128,
    )
    return link, derivative


def _frame(time: float) -> tuple[np.ndarray, np.ndarray]:
    phases = np.asarray((0.31 * np.sin(0.67 * time), -0.23 * time * time))
    phase_dots = np.asarray((0.31 * 0.67 * np.cos(0.67 * time), -0.46 * time))
    value = np.diag(np.exp(1j * phases))
    derivative = np.diag(1j * phase_dots * np.exp(1j * phases))
    return value, derivative


def _metric_generator(
    time: float,
    *,
    transformed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    link, link_dot = _transport(time)
    inverse = np.linalg.inv(link)
    generator = link_dot @ inverse
    metric = inverse.conj().T @ _S0 @ inverse
    if transformed:
        frame, frame_dot = _frame(time)
        inverse_frame = np.linalg.inv(frame)
        generator = inverse_frame @ generator @ frame - inverse_frame @ frame_dot
        metric = frame.conj().T @ metric @ frame
    return metric, generator


def _triple(time: float, *, transformed: bool) -> EOMTriple:
    metric, generator = _metric_generator(time, transformed=transformed)
    zero = np.zeros_like(metric)
    return EOMTriple(metric, zero, -metric @ generator)


def _history(intervals: int, *, transformed: bool = False) -> ExperimentalGaussMagnusHistory:
    final_time = 1.7
    step = final_time / intervals
    offset = math.sqrt(3.0) / 6.0
    endpoints = tuple(
        _metric_generator(index * step, transformed=transformed)[0]
        for index in range(intervals + 1)
    )
    minus = tuple(
        _triple((index + 0.5 - offset) * step, transformed=transformed)
        for index in range(intervals)
    )
    plus = tuple(
        _triple((index + 0.5 + offset) * step, transformed=transformed)
        for index in range(intervals)
    )
    return ExperimentalGaussMagnusHistory(endpoints, minus, plus, step)


def _exact_final_mixed_density() -> np.ndarray:
    final_link, _ = _transport(1.7)
    return np.asarray(final_link @ _D0 @ np.linalg.inv(final_link))


def _relative(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value) / max(1.0, np.linalg.norm(reference)))


def test_mixed_and_contravariant_density_round_trip() -> None:
    backend = NumPyBackend()
    mixed = contravariant_to_mixed_density(_P0, _S0, backend)
    recovered = mixed_to_contravariant_density(mixed, _S0, backend)
    np.testing.assert_allclose(mixed, _D0, atol=2.0e-16, rtol=2.0e-16)
    np.testing.assert_allclose(recovered, _P0, atol=2.0e-16, rtol=2.0e-16)


def test_mixed_gauss_magnus_has_fourth_order_global_convergence() -> None:
    backend = NumPyBackend()
    exact = _exact_final_mixed_density()
    errors = []
    metric_residuals = []
    hermiticity_residuals = []
    for intervals in (4, 8, 16, 32):
        trajectory = propagate_experimental_mixed_density(
            backend.asarray(_D0),
            _history(intervals),
            backend,
        )
        errors.append(_relative(trajectory.mixed_densities[-1] - exact, exact))
        metric_residuals.append(max(item.cross_metric_residual for item in trajectory.diagnostics))
        hermiticity_residuals.append(trajectory.diagnostics[-1].metric_hermiticity_residual)
        assert trajectory.metric_correction_applied is False
        assert max(item.trace_drift for item in trajectory.diagnostics) < 8.0e-16
        assert max(item.output_idempotency_residual for item in trajectory.diagnostics) < 2.0e-15

    assert min(errors[index] / errors[index + 1] for index in range(3)) > 13.0
    assert (
        min(hermiticity_residuals[index] / hermiticity_residuals[index + 1] for index in range(3))
        > 13.0
    )
    # This is a local link defect and therefore decreases as h^5.
    assert min(metric_residuals[index] / metric_residuals[index + 1] for index in range(3)) > 25.0


def test_mixed_gauss_magnus_frame_discrepancy_converges_at_fourth_order() -> None:
    backend = NumPyBackend()
    discrepancies = []
    for intervals in (4, 8, 16, 32):
        ordinary = propagate_experimental_mixed_density(
            backend.asarray(_D0),
            _history(intervals),
            backend,
        )
        transformed = propagate_experimental_mixed_density(
            backend.asarray(_D0),
            _history(intervals, transformed=True),
            backend,
        )
        final_frame, _ = _frame(1.7)
        transformed_back = (
            final_frame @ transformed.mixed_densities[-1] @ np.linalg.inv(final_frame)
        )
        discrepancies.append(
            _relative(transformed_back - ordinary.mixed_densities[-1], ordinary.mixed_densities[-1])
        )

    assert min(discrepancies[index] / discrepancies[index + 1] for index in range(3)) > 13.0
