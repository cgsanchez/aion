#!/usr/bin/env python3
"""Run the authenticated WP6 exact-time-connection and linear-dynamics campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg

from aion.config import (
    AtomConfig,
    BackendConfig,
    ElectromagneticOrigin,
    OneElectronReferenceConfig,
    canonical_sha256,
)
from aion.electromagnetism import (
    MagneticGaugeKind,
    UniformMagneticField,
    UniformMagneticSourceSample,
    affine_gauge_difference_potential,
)
from aion.electronic_structure import (
    AOGridPolicy,
    evaluate_exact_static_magnetic_one_electron_matrices,
    evaluate_exact_wilson_one_electron_sample,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
)
from aion.formulations import (
    ExactOneElectronModelContext,
    ExactOneElectronModelTriples,
    exact_one_electron_model_triples,
    exact_wilson_one_electron_triple,
    prepare_exact_one_electron_model_context,
)
from aion.propagation import (
    LinearMatrixHistory,
    generalized_spectral_trajectory,
    propagate_linear_matrix_history,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_FORMAL_PLAN = Path(
    "/home/cgs/00_WORK/Projection_Full_Formalism/REVIEW/implementation/"
    "exact_one_electron_numerical_qualification_plan.md"
)
_FIXTURE = (
    _REPOSITORY
    / "tests/fixtures/exact_one_electron/wp6_linear_dynamics.fixture.json"
)
_WP5_FIXTURE = (
    _REPOSITORY
    / "tests/fixtures/exact_one_electron/wp5_multicentre.fixture.json"
)
_G5_REVIEW = _REPOSITORY / "docs/reviews/exact_one_electron_g5_review_20260918.json"
_LOCK = _REPOSITORY / "conda-linux-64.lock"
_DEFAULT_WP5_ANALYSIS = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "exact_one_electron_qualification/"
    "wp5_multicentre_20260918T200808Z_3fd2a83b77d1/analysis"
)
_DEFAULT_OUTPUT_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "exact_one_electron_qualification"
)
_MODEL_NAMES = (
    "exact",
    "p0",
    "e1",
    "geometric_b1",
    "full_b1",
    "complete_first_order",
)


def _vector3(values: Any) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("expected a finite Cartesian vector")
    return (float(array[0]), float(array[1]), float(array[2]))


@dataclass(frozen=True, slots=True)
class EvaluatedHistory:
    endpoint_sources: tuple[UniformMagneticSourceSample, ...]
    midpoint_sources: tuple[UniformMagneticSourceSample, ...]
    endpoint_models: tuple[ExactOneElectronModelTriples, ...]
    midpoint_models: tuple[ExactOneElectronModelTriples, ...]
    maximum_direct_connection_residual: float
    maximum_direct_metric_dot_residual: float
    maximum_metric_compatibility_residual: float


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wp5-analysis", type=Path, default=_DEFAULT_WP5_ANALYSIS)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execution-directory", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return dict(value)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)  # type: ignore[arg-type]
    os.replace(temporary, path)


def _config(
    atoms: list[dict[str, Any]],
    basis: str,
    origin: list[float],
) -> OneElectronReferenceConfig:
    return OneElectronReferenceConfig(
        atoms=tuple(
            AtomConfig(str(atom["symbol"]), _vector3(atom["position_au"]))
            for atom in atoms
        ),
        basis=basis,
        electromagnetic_origin=ElectromagneticOrigin(_vector3(origin)),
    )


def _h3_config(
    coordinates: list[list[float]],
    basis: str,
    origin: list[float],
) -> OneElectronReferenceConfig:
    return OneElectronReferenceConfig(
        atoms=tuple(AtomConfig("H", _vector3(row)) for row in coordinates),
        basis=basis,
        electromagnetic_origin=ElectromagneticOrigin(_vector3(origin)),
    )


def _source_factory(
    *,
    duration: float,
    peak_magnetic: np.ndarray,
    peak_electric: np.ndarray,
    origin: tuple[float, float, float],
    gauge_kind: MagneticGaugeKind,
    landau_axis: tuple[float, float, float] | None,
) -> Callable[[float], UniformMagneticSourceSample]:
    def source(time_au: float) -> UniformMagneticSourceSample:
        if time_au <= 0.0 or time_au >= duration:
            envelope = 0.0
            derivative = 0.0
        else:
            phase = math.pi * time_au / duration
            envelope = math.sin(phase) ** 2
            derivative = (math.pi / duration) * math.sin(2.0 * phase)
        return UniformMagneticSourceSample(
            time_au,
            UniformMagneticField(tuple(envelope * peak_magnetic)),
            magnetic_field_dot_au=tuple(derivative * peak_magnetic),
            electric_field_origin_au=tuple(envelope * peak_electric),
            origin_au=origin,
            gauge_kind=gauge_kind,
            landau_axis=landau_axis,
        )

    return source


def _model(value: ExactOneElectronModelTriples, name: str) -> Any:
    return getattr(value, name)


def _evaluate_history(
    quadrature: Any,
    context: ExactOneElectronModelContext,
    source: Callable[[float], UniformMagneticSourceSample],
    *,
    duration: float,
    fine_intervals: int,
    progress_prefix: str,
) -> EvaluatedHistory:
    endpoint_sources = tuple(
        source(duration * index / fine_intervals)
        for index in range(fine_intervals + 1)
    )
    midpoint_sources = tuple(
        source(duration * (index + 0.5) / fine_intervals)
        for index in range(fine_intervals)
    )
    endpoint_models: list[ExactOneElectronModelTriples] = []
    midpoint_models: list[ExactOneElectronModelTriples] = []
    connection_residuals: list[float] = []
    metric_dot_residuals: list[float] = []
    compatibility_residuals: list[float] = []
    work = [
        *(("endpoint", value) for value in endpoint_sources),
        *(("midpoint", value) for value in midpoint_sources),
    ]
    total = len(work)
    for index, (location, source_value) in enumerate(work, start=1):
        evaluated = evaluate_exact_wilson_one_electron_sample(quadrature, source_value)
        models = exact_one_electron_model_triples(evaluated, context)
        if location == "endpoint":
            endpoint_models.append(models)
        else:
            midpoint_models.append(models)
        connection_residuals.append(
            evaluated.connection.direct_factorized_connection_residual
        )
        metric_dot_residuals.append(
            evaluated.connection.direct_factorized_metric_dot_residual
        )
        compatibility_residuals.append(
            evaluated.connection.metric_compatibility_residual
        )
        print(
            f"{progress_prefix} sample={index}/{total} location={location} "
            f"time_au={source_value.time_au:.8f}",
            flush=True,
        )
    return EvaluatedHistory(
        endpoint_sources=endpoint_sources,
        midpoint_sources=midpoint_sources,
        endpoint_models=tuple(endpoint_models),
        midpoint_models=tuple(midpoint_models),
        maximum_direct_connection_residual=max(connection_residuals),
        maximum_direct_metric_dot_residual=max(metric_dot_residuals),
        maximum_metric_compatibility_residual=max(compatibility_residuals),
    )


def _history_for_resolution(
    evaluated: EvaluatedHistory,
    model_name: str,
    *,
    duration: float,
    fine_intervals: int,
    intervals: int,
) -> LinearMatrixHistory:
    if fine_intervals % intervals:
        raise ValueError("comparison interval count must divide the fine history")
    stride = fine_intervals // intervals
    endpoint = tuple(
        _model(evaluated.endpoint_models[index * stride], model_name).triple.metric
        for index in range(intervals + 1)
    )
    if stride == 1:
        midpoint = tuple(
            _model(value, model_name).triple for value in evaluated.midpoint_models
        )
    elif stride % 2 == 0:
        midpoint = tuple(
            _model(
                evaluated.endpoint_models[index * stride + stride // 2],
                model_name,
            ).triple
            for index in range(intervals)
        )
    else:
        raise ValueError("coarse midpoints are not present in the fine history")
    return LinearMatrixHistory(
        endpoint_metrics=endpoint,
        midpoint_triples=midpoint,
        interval_au=duration / intervals,
    )


def _initial_ground(models: ExactOneElectronModelTriples) -> np.ndarray:
    triple = models.exact.triple
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        np.asarray(triple.hamiltonian_eom),
        np.asarray(triple.metric),
    )
    del eigenvalues
    return np.asarray(eigenvectors[:, :1], dtype=np.complex128)


def _norm(coefficients: Any, metric: Any) -> float:
    value = coefficients.conj().T @ metric @ coefficients
    return float(np.real(np.asarray(value).item()))


def _propagate_models(
    evaluated: EvaluatedHistory,
    initial: np.ndarray,
    backend: Any,
    *,
    duration: float,
    fine_intervals: int,
    comparison_intervals: tuple[int, ...],
    prefix: str,
    arrays: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], Any]]:
    rows: list[dict[str, Any]] = []
    trajectories: dict[tuple[str, int], Any] = {}
    for model_name in _MODEL_NAMES:
        for intervals in comparison_intervals:
            history = _history_for_resolution(
                evaluated,
                model_name,
                duration=duration,
                fine_intervals=fine_intervals,
                intervals=intervals,
            )
            trajectory = propagate_linear_matrix_history(initial, history, backend)
            raw = propagate_linear_matrix_history(
                initial,
                history,
                backend,
                apply_metric_correction=False,
            )
            trajectories[(model_name, intervals)] = trajectory
            norms = np.asarray(
                [
                    _norm(coefficients, metric)
                    for coefficients, metric in zip(
                        trajectory.coefficients,
                        history.endpoint_metrics,
                        strict=True,
                    )
                ]
            )
            raw_norms = np.asarray(
                [
                    _norm(coefficients, metric)
                    for coefficients, metric in zip(
                        raw.coefficients,
                        history.endpoint_metrics,
                        strict=True,
                    )
                ]
            )
            arrays[f"{prefix}__{model_name}__n{intervals}__coefficients"] = np.asarray(
                trajectory.coefficients
            )
            arrays[f"{prefix}__{model_name}__n{intervals}__norms"] = norms
            arrays[f"{prefix}__{model_name}__n{intervals}__raw_norms"] = raw_norms
            rows.append(
                {
                    "prefix": prefix,
                    "model": model_name,
                    "intervals": intervals,
                    "step_au": duration / intervals,
                    "maximum_corrected_norm_drift": float(np.max(np.abs(norms - norms[0]))),
                    "maximum_raw_norm_drift": float(
                        np.max(np.abs(raw_norms - raw_norms[0]))
                    ),
                    "maximum_metric_correction_norm": max(
                        value.link.correction_norm for value in trajectory.diagnostics
                    ),
                    "maximum_raw_cross_metric_residual": max(
                        value.link.raw_metric_residual for value in trajectory.diagnostics
                    ),
                    "minimum_endpoint_metric_eigenvalue": min(
                        float(np.linalg.eigvalsh(np.asarray(metric))[0])
                        for metric in history.endpoint_metrics
                    ),
                }
            )
    exact_fine = trajectories[("exact", max(comparison_intervals))].coefficients[-1]
    exact_density = exact_fine @ exact_fine.conj().T
    for row in rows:
        key = (str(row["model"]), int(row["intervals"]))
        final = trajectories[key].coefficients[-1]
        density = final @ final.conj().T
        row["final_density_distance_from_fine_exact"] = float(
            np.linalg.norm(np.asarray(density - exact_density))
        )
    return rows, trajectories


def _matrix_checks(quadrature: Any, arrays: dict[str, np.ndarray]) -> dict[str, float]:
    source = UniformMagneticSourceSample(
        0.73,
        UniformMagneticField((0.0, 0.0, 0.031)),
        magnetic_field_dot_au=(0.0, 0.0, -0.007),
        electric_field_origin_au=(0.013, -0.009, 0.017),
        origin_au=(0.11, -0.07, 0.05),
    )
    exact = evaluate_exact_wilson_one_electron_sample(quadrature, source)
    step = 2.0e-5
    current = np.asarray(source.field.magnetic_field_au)
    direction = np.asarray(source.magnetic_field_dot_au)
    plus, minus = evaluate_exact_static_magnetic_one_electron_matrices(
        quadrature,
        (
            UniformMagneticField(tuple(current + step * direction)),
            UniformMagneticField(tuple(current - step * direction)),
        ),
        include_direct_oracle=False,
    )
    finite_difference = (plus.lower_exact.overlap - minus.lower_exact.overlap) / (2.0 * step)
    metric_rate_residual = float(
        np.linalg.norm(finite_difference - exact.connection.metric_dot)
        / max(1.0, float(np.linalg.norm(exact.connection.metric_dot)))
    )
    landau_source = UniformMagneticSourceSample(
        source.time_au,
        source.field,
        magnetic_field_dot_au=source.magnetic_field_dot_au,
        electric_field_origin_au=source.electric_field_origin_au,
        origin_au=source.origin_au,
        gauge_kind=MagneticGaugeKind.LANDAU,
        landau_axis=(1.0, 0.0, 0.0),
    )
    landau = evaluate_exact_wilson_one_electron_sample(quadrature, landau_source)
    anchors = quadrature.reference.core_operators.nuclei.coordinates_au[
        quadrature.reference.anchor_topology.ao_to_atom
    ]
    gauge_lambda = affine_gauge_difference_potential(
        landau_source.gauge,
        source.gauge,
        anchors,
        quadrature.backend,
    )
    gauge_lambda_dot = affine_gauge_difference_potential(
        landau_source.gauge_rate,
        source.gauge_rate,
        anchors,
        quadrature.backend,
    )
    diagonal = np.exp(-1j * np.asarray(gauge_lambda))
    diagonal_dot = -1j * np.asarray(gauge_lambda_dot) * diagonal

    def transform(matrix: Any) -> np.ndarray:
        return diagonal[:, None] * np.asarray(matrix) * diagonal.conj()[None, :]

    expected_connection = transform(exact.connection.connection) + (
        diagonal[:, None]
        * np.asarray(exact.metric)
        * diagonal_dot.conj()[None, :]
    )
    gauge_residual = max(
        float(
            np.linalg.norm(np.asarray(landau.metric) - transform(exact.metric))
            / max(1.0, float(np.linalg.norm(exact.metric)))
        ),
        float(
            np.linalg.norm(np.asarray(landau.mechanical) - transform(exact.mechanical))
            / max(1.0, float(np.linalg.norm(exact.mechanical)))
        ),
        float(
            np.linalg.norm(np.asarray(landau.connection.connection) - expected_connection)
            / max(1.0, float(np.linalg.norm(expected_connection)))
        ),
    )
    arrays["matrix_check__metric"] = np.asarray(exact.metric)
    arrays["matrix_check__mechanical"] = np.asarray(exact.mechanical)
    arrays["matrix_check__connection"] = np.asarray(exact.connection.connection)
    arrays["matrix_check__metric_dot"] = np.asarray(exact.connection.metric_dot)
    arrays["matrix_check__metric_dot_finite_difference"] = np.asarray(finite_difference)
    return {
        "direct_factorized_connection_residual": (
            exact.connection.direct_factorized_connection_residual
        ),
        "direct_factorized_metric_dot_residual": (
            exact.connection.direct_factorized_metric_dot_residual
        ),
        "metric_compatibility_residual": exact.connection.metric_compatibility_residual,
        "metric_rate_finite_difference_residual": metric_rate_residual,
        "maximum_matrix_gauge_covariance_residual": gauge_residual,
    }


def _static_spectral(
    quadrature: Any,
    fixture: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    config = fixture["h2"]["static_spectral"]
    source = UniformMagneticSourceSample(
        0.0,
        UniformMagneticField(tuple(config["magnetic_field_au"])),
        origin_au=tuple(fixture["h2"]["electromagnetic_origin_au"]),
    )
    sample = evaluate_exact_wilson_one_electron_sample(quadrature, source)
    triple = exact_wilson_one_electron_triple(sample)
    initial = np.ones((sample.metric.shape[0], 1), dtype=np.complex128)
    initial[:, 0] += 0.17j * np.arange(sample.metric.shape[0])
    initial /= math.sqrt(_norm(initial, sample.metric))
    duration = float(config["duration_au"])
    spectral = generalized_spectral_trajectory(
        initial,
        sample.metric,
        sample.mechanical,
        (duration,),
        quadrature.backend,
    )[0]
    rows: list[dict[str, Any]] = []
    for intervals_value in config["intervals"]:
        intervals = int(intervals_value)
        history = LinearMatrixHistory(
            endpoint_metrics=(sample.metric,) * (intervals + 1),
            midpoint_triples=(triple,) * intervals,
            interval_au=duration / intervals,
        )
        propagated = propagate_linear_matrix_history(initial, history, quadrature.backend)
        error = float(np.linalg.norm(propagated.coefficients[-1] - spectral))
        arrays[f"static__n{intervals}__final_coefficients"] = np.asarray(
            propagated.coefficients[-1]
        )
        rows.append(
            {
                "intervals": intervals,
                "step_au": duration / intervals,
                "coefficient_error": error,
                "maximum_norm_drift": max(
                    value.output_metric_residual for value in propagated.diagnostics
                ),
            }
        )
    for coarse, fine in pairwise(rows):
        coarse["measured_order_to_next"] = math.log(
            float(coarse["coefficient_error"]) / float(fine["coefficient_error"])
        ) / math.log(float(fine["intervals"]) / float(coarse["intervals"]))
    rows[-1]["measured_order_to_next"] = None
    return rows


def _store_exact_matrix_history(
    evaluated: EvaluatedHistory,
    arrays: dict[str, np.ndarray],
) -> None:
    for location, sources, models in (
        ("endpoint", evaluated.endpoint_sources, evaluated.endpoint_models),
        ("midpoint", evaluated.midpoint_sources, evaluated.midpoint_models),
    ):
        arrays[f"h2_matrix_history__{location}__times_au"] = np.asarray(
            [source.time_au for source in sources]
        )
        arrays[f"h2_matrix_history__{location}__magnetic_field_au"] = np.asarray(
            [source.field.magnetic_field_au for source in sources]
        )
        arrays[f"h2_matrix_history__{location}__magnetic_field_dot_au"] = np.asarray(
            [source.magnetic_field_dot_au for source in sources]
        )
        arrays[f"h2_matrix_history__{location}__electric_field_origin_au"] = np.asarray(
            [source.electric_field_origin_au for source in sources]
        )
        arrays[f"h2_matrix_history__{location}__metric"] = np.asarray(
            [value.exact.triple.metric for value in models]
        )
        arrays[f"h2_matrix_history__{location}__mechanical"] = np.asarray(
            [value.exact.triple.hamiltonian_eom for value in models]
        )
        arrays[f"h2_matrix_history__{location}__connection"] = np.asarray(
            [value.exact.triple.connection for value in models]
        )
        arrays[f"h2_matrix_history__{location}__ordinary_derivative_matrix"] = np.asarray(
            [
                value.exact.triple.hamiltonian_eom
                - 1j * value.exact.triple.connection
                for value in models
            ]
        )
        arrays[f"h2_matrix_history__{location}__metric_dot"] = np.asarray(
            [value.exact.metric_dot for value in models]
        )


def _gauge_trajectory_rows(
    symmetric: EvaluatedHistory,
    landau: EvaluatedHistory,
    initial: np.ndarray,
    backend: Any,
    *,
    duration: float,
    fine_intervals: int,
    comparison_intervals: tuple[int, ...],
    ao_anchors: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for intervals in comparison_intervals:
        symmetric_history = _history_for_resolution(
            symmetric,
            "exact",
            duration=duration,
            fine_intervals=fine_intervals,
            intervals=intervals,
        )
        landau_history = _history_for_resolution(
            landau,
            "exact",
            duration=duration,
            fine_intervals=fine_intervals,
            intervals=intervals,
        )
        left = propagate_linear_matrix_history(initial, symmetric_history, backend)
        right = propagate_linear_matrix_history(initial, landau_history, backend)
        stride = fine_intervals // intervals
        residuals = []
        for index, (left_coefficients, right_coefficients) in enumerate(
            zip(left.coefficients, right.coefficients, strict=True)
        ):
            source_left = symmetric.endpoint_sources[index * stride]
            source_right = landau.endpoint_sources[index * stride]
            gauge_lambda = affine_gauge_difference_potential(
                source_right.gauge,
                source_left.gauge,
                ao_anchors,
                backend,
            )
            diagonal = np.exp(-1j * np.asarray(gauge_lambda))
            difference = right_coefficients - diagonal[:, None] * left_coefficients
            metric = landau_history.endpoint_metrics[index]
            residuals.append(math.sqrt(max(0.0, _norm(difference, metric))))
        arrays[f"h2_gauge__n{intervals}__trajectory_residuals"] = np.asarray(residuals)
        rows.append(
            {
                "intervals": intervals,
                "step_au": duration / intervals,
                "maximum_coefficient_metric_residual": max(residuals),
                "final_coefficient_metric_residual": residuals[-1],
            }
        )
    for coarse, fine in pairwise(rows):
        coarse_value = float(coarse["maximum_coefficient_metric_residual"])
        fine_value = float(fine["maximum_coefficient_metric_residual"])
        coarse["measured_order_to_next"] = (
            None
            if fine_value == 0.0 or coarse_value == 0.0
            else math.log(coarse_value / fine_value)
            / math.log(float(fine["intervals"]) / float(coarse["intervals"]))
        )
    rows[-1]["measured_order_to_next"] = None
    return rows


def _trajectory_convergence_rows(
    rows: list[dict[str, Any]],
    trajectories: dict[tuple[str, int], Any],
    comparison_intervals: tuple[int, ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    row_index = {(str(row["model"]), int(row["intervals"])): row for row in rows}
    for model_name in _MODEL_NAMES:
        values = []
        for coarse, fine in pairwise(comparison_intervals):
            coarse_final = trajectories[(model_name, coarse)].coefficients[-1]
            fine_final = trajectories[(model_name, fine)].coefficients[-1]
            difference = float(np.linalg.norm(np.asarray(coarse_final - fine_final)))
            values.append((coarse, fine, difference))
        for index, (coarse, fine, difference) in enumerate(values):
            order = None
            if index + 1 < len(values):
                next_difference = values[index + 1][2]
                if difference > 0.0 and next_difference > 0.0:
                    order = math.log(difference / next_difference) / math.log(
                        float(values[index + 1][1]) / float(fine)
                    )
            output.append(
                {
                    "model": model_name,
                    "coarse_intervals": coarse,
                    "fine_intervals": fine,
                    "final_coefficient_difference": difference,
                    "measured_order_to_next": order,
                    "coarse_maximum_correction_norm": row_index[
                        (model_name, coarse)
                    ]["maximum_metric_correction_norm"],
                    "fine_maximum_correction_norm": row_index[
                        (model_name, fine)
                    ]["maximum_metric_correction_norm"],
                }
            )
    return output


def _in_memory_restart_check(
    evaluated: EvaluatedHistory,
    initial: np.ndarray,
    backend: Any,
    *,
    duration: float,
    fine_intervals: int,
) -> dict[str, Any]:
    history = _history_for_resolution(
        evaluated,
        "exact",
        duration=duration,
        fine_intervals=fine_intervals,
        intervals=fine_intervals,
    )
    split = fine_intervals // 2
    first = LinearMatrixHistory(
        endpoint_metrics=history.endpoint_metrics[: split + 1],
        midpoint_triples=history.midpoint_triples[:split],
        interval_au=history.interval_au,
        hbar=history.hbar,
    )
    second = LinearMatrixHistory(
        endpoint_metrics=history.endpoint_metrics[split:],
        midpoint_triples=history.midpoint_triples[split:],
        interval_au=history.interval_au,
        hbar=history.hbar,
    )
    uninterrupted = propagate_linear_matrix_history(initial, history, backend)
    first_result = propagate_linear_matrix_history(initial, first, backend)
    restarted = propagate_linear_matrix_history(
        first_result.coefficients[-1], second, backend
    )
    return {
        "mode": "in_memory_boundary_state_restart",
        "persistence_status": "deferred_to_stage_c_after_g6",
        "split_interval": split,
        "coefficient_residual": float(
            np.linalg.norm(
                np.asarray(restarted.coefficients[-1])
                - np.asarray(uninterrupted.coefficients[-1])
            )
        ),
        "bitwise_equal": bool(
            np.array_equal(
                np.asarray(restarted.coefficients[-1]),
                np.asarray(uninterrupted.coefficients[-1]),
            )
        ),
    }


def _selected_cases(path: Path) -> dict[str, dict[str, float]]:
    import csv

    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["basis"] != "cc-pvdz" or row["selection_role"] != "wp6_selected":
                continue
            key = f"{row['geometry']}:{row['label']}"
            rows[key] = {
                "field_z_au": float(row["field_z_au"]),
                "loop_flux_au": float(row["loop_flux_au"]),
                "minimum_model_metric_eigenvalue": float(
                    row["minimum_model_metric_eigenvalue"]
                ),
            }
    if len(rows) != 6:
        raise ValueError("WP6 requires six accepted cc-pVDZ three-centre cases")
    return rows


def _run_three_centre_level(
    *,
    geometry_name: str,
    coordinates: list[list[float]],
    level: int,
    fixture: dict[str, Any],
    selected: dict[str, dict[str, float]],
    smoke: bool,
    arrays: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    settings = fixture["three_centre"]
    reference = prepare_one_electron_ao_reference(
        _h3_config(
            coordinates,
            str(settings["basis"]),
            list(settings["electromagnetic_origin_au"]),
        )
    )
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(level),
        block_size=int(settings["block_size"]),
    )
    context = prepare_exact_one_electron_model_context(quadrature)
    duration = float(settings["pulse_duration_au"])
    fine_intervals = 4 if smoke else int(settings["fine_intervals"])
    comparison_intervals = (2, 4) if smoke else tuple(
        int(value) for value in settings["comparison_intervals"]
    )
    rows: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    final_densities: dict[str, np.ndarray] = {}
    for label in settings["selection_roles"]:
        case = selected[f"{geometry_name}:{label}"]
        peak = np.asarray((0.0, 0.0, case["field_z_au"]), dtype=np.float64)
        source = _source_factory(
            duration=duration,
            peak_magnetic=peak,
            peak_electric=np.zeros(3),
            origin=tuple(settings["electromagnetic_origin_au"]),
            gauge_kind=MagneticGaugeKind.SYMMETRIC,
            landau_axis=None,
        )
        evaluated = _evaluate_history(
            quadrature,
            context,
            source,
            duration=duration,
            fine_intervals=fine_intervals,
            progress_prefix=f"h3/{geometry_name}/level{level}/{label}",
        )
        initial = _initial_ground(evaluated.endpoint_models[0])
        prefix = f"h3_{geometry_name}_level{level}_{label}"
        case_rows, trajectories = _propagate_models(
            evaluated,
            initial,
            quadrature.backend,
            duration=duration,
            fine_intervals=fine_intervals,
            comparison_intervals=comparison_intervals,
            prefix=prefix,
            arrays=arrays,
        )
        for row in case_rows:
            row.update(
                {
                    "geometry": geometry_name,
                    "grid_level": level,
                    "case": label,
                    "peak_field_z_au": case["field_z_au"],
                    "loop_flux_au": case["loop_flux_au"],
                    "grid_npoints": quadrature.grid.npoints,
                    "maximum_direct_connection_residual": (
                        evaluated.maximum_direct_connection_residual
                    ),
                    "maximum_direct_metric_dot_residual": (
                        evaluated.maximum_direct_metric_dot_residual
                    ),
                    "maximum_metric_compatibility_residual": (
                        evaluated.maximum_metric_compatibility_residual
                    ),
                }
            )
        rows.extend(case_rows)
        case_convergence = _trajectory_convergence_rows(
            case_rows,
            trajectories,
            comparison_intervals,
        )
        for row in case_convergence:
            row.update(
                {
                    "geometry": geometry_name,
                    "grid_level": level,
                    "case": label,
                    "peak_field_z_au": case["field_z_au"],
                }
            )
        convergence.extend(case_convergence)
        fine = max(comparison_intervals)
        for model_name in _MODEL_NAMES:
            coefficients = trajectories[(model_name, fine)].coefficients[-1]
            final_densities[f"{label}:{model_name}"] = np.asarray(
                coefficients @ coefficients.conj().T
            )
    return rows, convergence, final_densities


def _plan(args: argparse.Namespace, fixture: dict[str, Any], wp5_analysis: Path) -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    selected_path = wp5_analysis / "tables/selected_dynamical_cases.csv"
    plan: dict[str, Any] = {
        "schema": "aion.exact-one-electron-wp6-execution-plan",
        "version": "1.0.0",
        "status": "smoke_not_qualification" if args.smoke else "qualification_requested",
        "fixture": fixture,
        "wp5_analysis_directory": str(wp5_analysis),
        "provenance": {
            "code": {
                "repository": str(_REPOSITORY),
                "branch": _git("branch", "--show-current"),
                "commit": commit,
                "dirty": dirty,
            },
            "environment": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "thread_limits": {
                    name: os.environ.get(name, "unreported")
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
            },
            "input_hashes": {
                "fixture": _sha256(_FIXTURE),
                "wp5_fixture": _sha256(_WP5_FIXTURE),
                "formal_plan": _sha256(_FORMAL_PLAN),
                "g5_review": _sha256(_G5_REVIEW),
                "conda_lock": _sha256(_LOCK),
                "wp5_selected_cases": _sha256(selected_path),
                "time_connection": _sha256(
                    _REPOSITORY / "src/aion/electronic_structure/time_connection.py"
                ),
                "model_adapter": _sha256(
                    _REPOSITORY / "src/aion/formulations/exact_one_electron.py"
                ),
                "linear_propagator": _sha256(
                    _REPOSITORY / "src/aion/propagation/linear.py"
                ),
            },
        },
    }
    plan["plan_id"] = canonical_sha256(plan)
    return plan


def main() -> None:
    args = _arguments()
    fixture = _json(_FIXTURE)
    wp5_analysis = args.wp5_analysis.resolve()
    selected_path = wp5_analysis / "tables/selected_dynamical_cases.csv"
    selected = _selected_cases(selected_path)
    plan = _plan(args, fixture, wp5_analysis)
    if args.execution_directory is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        suffix = str(plan["provenance"]["code"]["commit"])[:12]
        execution = args.output_root.resolve() / f"wp6_linear_{stamp}_{suffix}"
    else:
        execution = args.execution_directory.resolve()
    execution.mkdir(parents=True, exist_ok=True)
    lock_path = execution / ".campaign.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = execution / "execution_plan.json"
        if plan_path.exists():
            previous = _json(plan_path)
            if previous["plan_id"] != plan["plan_id"]:
                raise ValueError("existing WP6 directory belongs to a different plan")
        else:
            _write_json(plan_path, plan)

        start = time.time()
        arrays: dict[str, np.ndarray] = {}
        h2_settings = fixture["h2"]
        h2_reference = prepare_one_electron_ao_reference(
            _config(
                list(h2_settings["atoms"]),
                str(h2_settings["basis"]),
                list(h2_settings["electromagnetic_origin_au"]),
            )
        )
        h2_level = 2 if args.smoke else int(h2_settings["qualification_grid_level"])
        h2_quadrature = prepare_ao_quadrature(
            h2_reference,
            BackendConfig(),
            grid_policy=AOGridPolicy.qualification(h2_level),
            block_size=int(h2_settings["block_size"]),
        )
        h2_context = prepare_exact_one_electron_model_context(h2_quadrature)
        matrix_checks = _matrix_checks(h2_quadrature, arrays)
        static_rows = _static_spectral(h2_quadrature, fixture, arrays)

        pulse = h2_settings["pulse"]
        duration = float(pulse["duration_au"])
        fine_intervals = 8 if args.smoke else int(pulse["fine_intervals"])
        comparison_intervals = (2, 4, 8) if args.smoke else tuple(
            int(value) for value in pulse["comparison_intervals"]
        )
        symmetric_source = _source_factory(
            duration=duration,
            peak_magnetic=np.asarray(pulse["peak_magnetic_field_au"]),
            peak_electric=np.asarray(pulse["peak_uniform_electric_field_au"]),
            origin=_vector3(h2_settings["electromagnetic_origin_au"]),
            gauge_kind=MagneticGaugeKind.SYMMETRIC,
            landau_axis=None,
        )
        landau_source = _source_factory(
            duration=duration,
            peak_magnetic=np.asarray(pulse["peak_magnetic_field_au"]),
            peak_electric=np.asarray(pulse["peak_uniform_electric_field_au"]),
            origin=_vector3(h2_settings["electromagnetic_origin_au"]),
            gauge_kind=MagneticGaugeKind.LANDAU,
            landau_axis=(1.0, 0.0, 0.0),
        )
        h2_symmetric = _evaluate_history(
            h2_quadrature,
            h2_context,
            symmetric_source,
            duration=duration,
            fine_intervals=fine_intervals,
            progress_prefix="h2/symmetric",
        )
        h2_landau = _evaluate_history(
            h2_quadrature,
            h2_context,
            landau_source,
            duration=duration,
            fine_intervals=fine_intervals,
            progress_prefix="h2/landau",
        )
        initial = _initial_ground(h2_symmetric.endpoint_models[0])
        h2_rows, h2_trajectories = _propagate_models(
            h2_symmetric,
            initial,
            h2_quadrature.backend,
            duration=duration,
            fine_intervals=fine_intervals,
            comparison_intervals=comparison_intervals,
            prefix=f"h2_level{h2_level}",
            arrays=arrays,
        )
        h2_convergence = _trajectory_convergence_rows(
            h2_rows,
            h2_trajectories,
            comparison_intervals,
        )
        restart_check = _in_memory_restart_check(
            h2_symmetric,
            initial,
            h2_quadrature.backend,
            duration=duration,
            fine_intervals=fine_intervals,
        )
        gauge_rows = _gauge_trajectory_rows(
            h2_symmetric,
            h2_landau,
            initial,
            h2_quadrature.backend,
            duration=duration,
            fine_intervals=fine_intervals,
            comparison_intervals=comparison_intervals,
            ao_anchors=h2_reference.core_operators.nuclei.coordinates_au[
                h2_reference.anchor_topology.ao_to_atom
            ],
            arrays=arrays,
        )
        _store_exact_matrix_history(h2_symmetric, arrays)

        geometries = _json(_WP5_FIXTURE)["geometries"]
        h3_rows: list[dict[str, Any]] = []
        h3_convergence: list[dict[str, Any]] = []
        h3_final: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        settings = fixture["three_centre"]
        qualification_level = 3 if args.smoke else int(
            settings["qualification_grid_level"]
        )
        refinement_level = 4 if args.smoke else int(settings["refinement_grid_level"])
        qualification_geometries = (
            ("equilateral",)
            if args.smoke
            else tuple(settings["qualification_geometries"])
        )
        refinement_geometries = () if args.smoke else tuple(
            settings["refinement_geometries"]
        )
        for geometry_name in qualification_geometries:
            rows, convergence, final = _run_three_centre_level(
                geometry_name=geometry_name,
                coordinates=list(geometries[geometry_name]["coordinates_au"]),
                level=qualification_level,
                fixture=fixture,
                selected=selected,
                smoke=args.smoke,
                arrays=arrays,
            )
            h3_rows.extend(rows)
            h3_convergence.extend(convergence)
            h3_final[(geometry_name, qualification_level)] = final
        for geometry_name in refinement_geometries:
            rows, convergence, final = _run_three_centre_level(
                geometry_name=geometry_name,
                coordinates=list(geometries[geometry_name]["coordinates_au"]),
                level=refinement_level,
                fixture=fixture,
                selected=selected,
                smoke=args.smoke,
                arrays=arrays,
            )
            h3_rows.extend(rows)
            h3_convergence.extend(convergence)
            h3_final[(geometry_name, refinement_level)] = final

        quadrature_rows: list[dict[str, Any]] = []
        for geometry_name in refinement_geometries:
            coarse = h3_final[(geometry_name, qualification_level)]
            refined = h3_final[(geometry_name, refinement_level)]
            for key in sorted(coarse):
                difference = float(np.linalg.norm(refined[key] - coarse[key]))
                exact_key = f"{key.split(':', 1)[0]}:exact"
                model_difference = float(
                    np.linalg.norm(refined[key] - refined[exact_key])
                )
                quadrature_rows.append(
                    {
                        "geometry": geometry_name,
                        "case": key.split(":", 1)[0],
                        "model": key.split(":", 1)[1],
                        "qualification_level": qualification_level,
                        "refinement_level": refinement_level,
                        "final_density_refinement_difference": difference,
                        "refined_model_difference_from_exact": model_difference,
                        "refinement_fraction_of_model_difference": (
                            difference / model_difference
                            if model_difference > np.finfo(np.float64).tiny
                            else None
                        ),
                    }
                )

        array_semantic = {name: canonical_sha256(value) for name, value in arrays.items()}
        result: dict[str, Any] = {
            "schema": "aion.exact-one-electron-wp6-result",
            "version": "1.0.0",
            "status": "smoke_not_qualification" if args.smoke else "executed_unreviewed",
            "plan_id": plan["plan_id"],
            "matrix_checks": matrix_checks,
            "static_spectral_convergence": static_rows,
            "h2_trajectory_rows": h2_rows,
            "h2_timestep_convergence": h2_convergence,
            "h2_gauge_trajectory_rows": gauge_rows,
            "h2_restart_check": restart_check,
            "h3_trajectory_rows": h3_rows,
            "h3_timestep_convergence": h3_convergence,
            "h3_quadrature_stability": quadrature_rows,
            "exact_matrix_history_fixture": {
                "system": "H2",
                "basis": h2_settings["basis"],
                "grid_level": h2_level,
                "fine_intervals": fine_intervals,
                "array_prefix": "h2_matrix_history__",
            },
            "array_semantic_sha256": array_semantic,
            "wall_time_seconds": time.time() - start,
        }
        arrays_path = execution / "arrays.npz"
        result_path = execution / "result.json"
        _write_npz(arrays_path, arrays)
        _write_json(result_path, result)
        completed = {
            "schema": "aion.exact-one-electron-wp6-completed",
            "version": "1.0.0",
            "status": result["status"],
            "plan_id": plan["plan_id"],
            "execution_plan_sha256": _sha256(plan_path),
            "result_sha256": _sha256(result_path),
            "arrays_sha256": _sha256(arrays_path),
            "completed_utc": datetime.now(UTC).isoformat(),
        }
        _write_json(execution / "completed.json", completed)
        print(f"execution_directory={execution}")
        print(f"status={result['status']}")
        print(f"wall_time_seconds={result['wall_time_seconds']:.3f}")


if __name__ == "__main__":
    main()
