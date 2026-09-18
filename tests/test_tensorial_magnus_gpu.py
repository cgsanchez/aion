from __future__ import annotations

import math

import numpy as np
import pytest

from aion.backends import CuPyBackend, NumPyBackend
from aion.formulations import EOMTriple
from aion.propagation import (
    ExperimentalGaussMagnusHistory,
    propagate_experimental_mixed_density,
)

pytestmark = pytest.mark.gpu


def _host_history(intervals: int) -> tuple[np.ndarray, ExperimentalGaussMagnusHistory]:
    initial_metric = np.asarray(((1.23, 0.13 - 0.04j), (0.13 + 0.04j, 0.91)))
    coefficient = np.asarray(((0.61 + 0.17j,), (-0.29 + 0.31j,)))
    coefficient /= np.sqrt((coefficient.conj().T @ initial_metric @ coefficient).real.item())
    initial = coefficient @ coefficient.conj().T @ initial_metric
    final_time = 0.9
    step = final_time / intervals
    offset = math.sqrt(3.0) / 6.0

    def metric_and_triple(time: float) -> tuple[np.ndarray, EOMTriple]:
        link = np.asarray(
            (
                (np.exp(0.13 * time), 0.21 * np.sin(0.71 * time)),
                (0.0, np.exp(-0.09 * time)),
            ),
            dtype=np.complex128,
        )
        link_dot = np.asarray(
            (
                (
                    0.13 * np.exp(0.13 * time),
                    0.21 * 0.71 * np.cos(0.71 * time),
                ),
                (0.0, -0.09 * np.exp(-0.09 * time)),
            ),
            dtype=np.complex128,
        )
        inverse = np.linalg.inv(link)
        metric = inverse.conj().T @ initial_metric @ inverse
        generator = link_dot @ inverse
        zero = np.zeros_like(metric)
        return metric, EOMTriple(metric, zero, -metric @ generator)

    endpoints = tuple(metric_and_triple(index * step)[0] for index in range(intervals + 1))
    minus = tuple(metric_and_triple((index + 0.5 - offset) * step)[1] for index in range(intervals))
    plus = tuple(metric_and_triple((index + 0.5 + offset) * step)[1] for index in range(intervals))
    return initial, ExperimentalGaussMagnusHistory(endpoints, minus, plus, step)


def _on_backend(
    history: ExperimentalGaussMagnusHistory,
    backend: NumPyBackend | CuPyBackend,
) -> ExperimentalGaussMagnusHistory:
    def triple(value: EOMTriple) -> EOMTriple:
        return EOMTriple(
            backend.asarray(value.metric),
            backend.asarray(value.hamiltonian_eom),
            backend.asarray(value.connection),
        )

    return ExperimentalGaussMagnusHistory(
        endpoint_metrics=tuple(backend.asarray(value) for value in history.endpoint_metrics),
        gauss_minus_triples=tuple(triple(value) for value in history.gauss_minus_triples),
        gauss_plus_triples=tuple(triple(value) for value in history.gauss_plus_triples),
        interval_au=history.interval_au,
        hbar=history.hbar,
    )


def test_experimental_mixed_gauss_magnus_cpu_gpu_parity() -> None:
    host_initial, host_history = _host_history(12)
    cpu = NumPyBackend()
    gpu = CuPyBackend(0)
    cpu_result = propagate_experimental_mixed_density(
        cpu.asarray(host_initial),
        _on_backend(host_history, cpu),
        cpu,
    )
    gpu_result = propagate_experimental_mixed_density(
        gpu.asarray(host_initial),
        _on_backend(host_history, gpu),
        gpu,
    )
    for value in (
        *gpu_result.mixed_densities,
        *gpu_result.contravariant_densities,
        *gpu_result.links,
    ):
        gpu.assert_resident(value, name="experimental mixed-density GPU result")
    np.testing.assert_allclose(
        gpu.to_host(gpu_result.mixed_densities[-1]),
        cpu_result.mixed_densities[-1],
        atol=3.0e-13,
        rtol=3.0e-13,
    )
    np.testing.assert_allclose(
        gpu.to_host(gpu_result.contravariant_densities[-1]),
        cpu_result.contravariant_densities[-1],
        atol=3.0e-13,
        rtol=3.0e-13,
    )
    np.testing.assert_allclose(
        [item.cross_metric_residual for item in gpu_result.diagnostics],
        [item.cross_metric_residual for item in cpu_result.diagnostics],
        atol=3.0e-13,
        rtol=3.0e-13,
    )
