from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from aion.backends import NumPyBackend
from aion.config import RationalApproximation
from aion.errors import StateIntegrityError
from aion.propagation import (
    DampedPicardController,
    hermitian_cleanup,
    metric_density_residual,
    rational_map,
    right_cholesky_metric_link,
)

pytestmark = pytest.mark.fast


def _hermitian(rng: np.random.Generator, size: int) -> np.ndarray:
    raw = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    return 0.5 * (raw + raw.conj().T)


def _positive_metric(rng: np.random.Generator, size: int) -> np.ndarray:
    raw = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    return raw.conj().T @ raw + np.eye(size)


def test_damped_picard_starts_full_and_reduces_only_when_residual_worsens() -> None:
    controller = DampedPicardController(0.1)
    assert [controller.observe(value) for value in (1.0, 0.8, 0.9, 0.7, 0.75, 0.8)] == [
        1.0,
        1.0,
        0.5,
        0.5,
        0.25,
        0.125,
    ]
    assert controller.observe(0.9) == 0.1
    assert controller.observe(1.0) == 0.1


@pytest.mark.parametrize("approximation", tuple(RationalApproximation))
def test_fixed_metric_rational_map_preserves_the_nonorthogonal_norm(
    approximation: RationalApproximation,
) -> None:
    backend = NumPyBackend()
    rng = np.random.default_rng(904)
    metric = _positive_metric(rng, 5)
    hamiltonian = _hermitian(rng, 5)
    generator = np.linalg.solve(metric, -1j * hamiltonian)
    link = rational_map(generator, 0.37, approximation, backend=backend)
    assert np.linalg.norm(link.conj().T @ metric @ link - metric) < 3.0e-14


def test_rational_maps_have_the_expected_frozen_generator_local_orders() -> None:
    backend = NumPyBackend()
    rng = np.random.default_rng(1204)
    generator = -1j * _hermitian(rng, 4)
    for approximation, minimum_ratio in (
        (RationalApproximation.CAYLEY_11, 7.5),
        (RationalApproximation.PADE_22, 28.0),
    ):
        errors = []
        for interval in (0.2, 0.1):
            actual = rational_map(generator, interval, approximation, backend=backend)
            exact = scipy.linalg.expm(interval * generator)
            errors.append(np.linalg.norm(actual - exact))
        assert errors[0] / errors[1] > minimum_ratio


def test_right_cholesky_correction_enforces_the_cross_metric_constraint() -> None:
    backend = NumPyBackend()
    rng = np.random.default_rng(711)
    start = _positive_metric(rng, 4)
    target = _positive_metric(rng, 4)
    raw = np.eye(4, dtype=np.complex128) + 0.03 * (
        rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    )
    corrected, diagnostics = right_cholesky_metric_link(raw, start, target, backend=backend)
    assert diagnostics.raw_metric_residual > 0.1
    assert diagnostics.corrected_metric_residual < 2.0e-15
    assert diagnostics.correction_norm > 0.0
    assert diagnostics.correction_applied
    assert np.linalg.norm(corrected.conj().T @ target @ corrected - start) < 2.0e-13

    uncorrected, diagnostic = right_cholesky_metric_link(
        raw,
        start,
        target,
        backend=backend,
        apply_correction=False,
    )
    assert uncorrected is raw
    assert not diagnostic.correction_applied
    assert diagnostic.corrected_metric_residual == diagnostic.raw_metric_residual


def test_metric_correction_size_tends_to_zero_with_the_raw_defect() -> None:
    backend = NumPyBackend()
    start = np.asarray([[1.4, 0.2], [0.2, 1.1]], dtype=np.complex128)
    direction = np.asarray([[0.3, -0.1j], [0.1j, -0.2]], dtype=np.complex128)
    norms = []
    for step in (0.2, 0.1, 0.05):
        target = start + step**2 * direction
        _, diagnostics = right_cholesky_metric_link(
            np.eye(2, dtype=np.complex128), start, target, backend=backend
        )
        norms.append(diagnostics.correction_norm)
    assert norms[0] / norms[1] == pytest.approx(4.0, rel=0.08)
    assert norms[1] / norms[2] == pytest.approx(4.0, rel=0.04)


def test_hermitian_cleanup_is_roundoff_only_and_density_residual_is_invariant() -> None:
    backend = NumPyBackend()
    base = np.asarray([[1.0, 0.2j], [-0.2j, 0.7]], dtype=np.complex128)
    contaminated = base.copy()
    contaminated[0, 1] += 2.0e-14
    cleaned, correction = hermitian_cleanup(
        contaminated,
        threshold=1.0e-12,
        backend=backend,
        name="test density",
    )
    assert correction > 0.0
    assert np.array_equal(cleaned, cleaned.conj().T)
    badly_contaminated = base.copy()
    badly_contaminated[0, 1] += 2.0e-6
    with pytest.raises(StateIntegrityError, match="exceeds cleanup threshold"):
        hermitian_cleanup(
            badly_contaminated,
            threshold=1.0e-12,
            backend=backend,
            name="test density",
        )

    rng = np.random.default_rng(144)
    metric = _positive_metric(rng, 3)
    left = _hermitian(rng, 3)
    right = _hermitian(rng, 3)
    residual = metric_density_residual(left, right, metric, electron_count=2.0, backend=backend)
    transform = rng.normal(size=(3, 3))
    while abs(np.linalg.det(transform)) < 0.2:
        transform = rng.normal(size=(3, 3))
    inverse = np.linalg.inv(transform)
    metric_transformed = transform.conj().T @ metric @ transform
    left_transformed = inverse @ left @ inverse.conj().T
    right_transformed = inverse @ right @ inverse.conj().T
    transformed_residual = metric_density_residual(
        left_transformed,
        right_transformed,
        metric_transformed,
        electron_count=2.0,
        backend=backend,
    )
    assert transformed_residual == pytest.approx(residual, rel=2.0e-14)


def test_midpoint_product_is_globally_second_order_for_time_dependent_generator() -> None:
    backend = NumPyBackend()
    hamiltonian = np.asarray([[0.3, 0.17], [0.17, -0.2]], dtype=np.complex128)
    initial = np.asarray([1.0, 0.0], dtype=np.complex128)
    exact = scipy.linalg.expm((-1j * 4.0 / 3.0) * hamiltonian) @ initial
    errors = []
    for intervals in (8, 16, 32):
        step = 1.0 / intervals
        state = initial.copy()
        for index in range(intervals):
            midpoint = (index + 0.5) * step
            generator = -1j * (1.0 + midpoint**2) * hamiltonian
            state = (
                rational_map(
                    generator,
                    step,
                    RationalApproximation.PADE_22,
                    backend=backend,
                )
                @ state
            )
        errors.append(np.linalg.norm(state - exact))
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.03)
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.02)
