#!/usr/bin/env python3
"""Run the targeted 128-step WP6 mixed-Magnus diagnostic continuation."""

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import run_exact_one_electron_wp6_linear as linear
import run_exact_one_electron_wp6_mixed_magnus as mixed

from aion.config import BackendConfig, canonical_sha256
from aion.electromagnetism import MagneticGaugeKind
from aion.electronic_structure import (
    AOGridPolicy,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
)
from aion.formulations import prepare_exact_one_electron_model_context
from aion.propagation import mixed_eom_generator, propagate_experimental_mixed_density

_REPOSITORY = Path(__file__).resolve().parents[1]
_FIXTURE = (
    _REPOSITORY
    / "tests/fixtures/exact_one_electron/wp6_mixed_magnus_n128_diagnostics.fixture.json"
)
_LOCK = _REPOSITORY / "conda-linux-64.lock"
_DEFAULT_OUTPUT_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "exact_one_electron_qualification"
)
_DIAGNOSTIC_COLUMNS = (
    "step_index",
    "time_midpoint_au",
    "field_z_midpoint_au",
    "field_dot_z_midpoint_au",
    "start_metric_minimum_eigenvalue",
    "target_metric_minimum_eigenvalue",
    "target_metric_condition_number",
    "generator_minus_frobenius",
    "generator_plus_frobenius",
    "generator_commutator_frobenius",
    "cross_metric_residual",
    "metric_hermiticity_residual",
    "contravariant_hermiticity_residual",
    "idempotency_residual",
    "trace_drift",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wp5-analysis", type=Path, default=linear._DEFAULT_WP5_ANALYSIS)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execution-directory", type=Path)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)  # type: ignore[arg-type]
    os.replace(temporary, path)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _authenticate_execution(
    root: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result = _json(root / "result.json")
    completed = _json(root / "completed.json")
    for filename, key in (
        ("execution_plan.json", "execution_plan_sha256"),
        ("result.json", "result_sha256"),
        ("arrays.npz", "arrays_sha256"),
    ):
        if _sha256(root / filename) != completed[key]:
            raise ValueError(f"source execution hash mismatch: {filename}")
    if result["plan_id"] != completed["plan_id"]:
        raise ValueError("source execution plan identity mismatch")
    arrays: dict[str, np.ndarray] = {}
    semantic = result["array_semantic_sha256"]
    with np.load(root / "arrays.npz") as archive:
        if set(archive.files) != set(semantic):
            raise ValueError("source semantic array inventory mismatch")
        for name in archive.files:
            value = np.asarray(archive[name])
            if canonical_sha256(value) != semantic[name]:
                raise ValueError(f"source semantic array hash mismatch: {name}")
            arrays[name] = value
    return result, arrays


def _continuation_rows(
    fixture: dict[str, Any],
    coarse_arrays: dict[str, np.ndarray],
    medium_arrays: dict[str, np.ndarray],
    fine_arrays: dict[str, np.ndarray],
    prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coarse = int(fixture["coarse_intervals"])
    medium = int(fixture["medium_intervals"])
    fine = int(fixture["new_intervals"])
    convergence: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []
    exact_medium = medium_arrays[f"{prefix}__exact__n{medium}__P"][-1]
    exact_fine = fine_arrays[f"{prefix}__exact__n{fine}__P"][-1]
    for model in fixture["models"]:
        coarse_density = coarse_arrays[f"{prefix}__{model}__n{coarse}__P"][-1]
        medium_density = medium_arrays[f"{prefix}__{model}__n{medium}__P"][-1]
        fine_density = fine_arrays[f"{prefix}__{model}__n{fine}__P"][-1]
        coarse_change = float(np.linalg.norm(medium_density - coarse_density))
        fine_change = float(np.linalg.norm(fine_density - medium_density))
        order = (
            math.log(coarse_change / fine_change) / math.log(2.0)
            if coarse_change > 0.0 and fine_change > 0.0
            else None
        )
        convergence.append(
            {
                "prefix": prefix,
                "model": model,
                "coarse_intervals": coarse,
                "medium_intervals": medium,
                "fine_intervals": fine,
                "coarse_density_change": coarse_change,
                "fine_density_change": fine_change,
                "measured_order": order,
            }
        )
        if model == "exact":
            continue
        medium_difference = medium_density - exact_medium
        fine_difference = fine_density - exact_fine
        refinement_change = float(np.linalg.norm(fine_difference - medium_difference))
        fine_distance = float(np.linalg.norm(fine_difference))
        stability.append(
            {
                "prefix": prefix,
                "model": model,
                "coarse_intervals": medium,
                "fine_intervals": fine,
                "coarse_model_difference": float(np.linalg.norm(medium_difference)),
                "fine_model_difference": fine_distance,
                "model_difference_refinement_change": refinement_change,
                "refinement_fraction_of_fine_model_difference": (
                    refinement_change / fine_distance
                    if fine_distance > np.finfo(np.float64).tiny
                    else None
                ),
            }
        )
    return convergence, stability


def _diagnostic_array(
    history: Any,
    trajectory: Any,
    source: Any,
    backend: Any,
) -> np.ndarray:
    interval = float(history.interval_au)
    rows: list[tuple[float, ...]] = []
    for index, (gauss_minus, gauss_plus, diagnostics) in enumerate(
        zip(
            history.gauss_minus_triples,
            history.gauss_plus_triples,
            trajectory.diagnostics,
            strict=True,
        )
    ):
        start_metric = np.asarray(history.endpoint_metrics[index])
        target_metric = np.asarray(history.endpoint_metrics[index + 1])
        start_eigenvalues = np.linalg.eigvalsh(start_metric)
        target_eigenvalues = np.linalg.eigvalsh(target_metric)
        generator_minus = np.asarray(mixed_eom_generator(gauss_minus, backend))
        generator_plus = np.asarray(mixed_eom_generator(gauss_plus, backend))
        commutator = generator_minus @ generator_plus - generator_plus @ generator_minus
        midpoint = (index + 0.5) * interval
        source_midpoint = source(midpoint)
        rows.append(
            (
                float(index + 1),
                midpoint,
                float(source_midpoint.field.magnetic_field_au[2]),
                float(source_midpoint.magnetic_field_dot_au[2]),
                float(start_eigenvalues[0]),
                float(target_eigenvalues[0]),
                float(np.linalg.cond(target_metric)),
                float(np.linalg.norm(generator_minus)),
                float(np.linalg.norm(generator_plus)),
                float(np.linalg.norm(commutator)),
                float(diagnostics.cross_metric_residual),
                float(diagnostics.metric_hermiticity_residual),
                float(diagnostics.contravariant_hermiticity_residual),
                float(diagnostics.output_idempotency_residual),
                float(diagnostics.trace_drift),
            )
        )
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (len(trajectory.diagnostics), len(_DIAGNOSTIC_COLUMNS)):
        raise ValueError("unexpected time-resolved diagnostic shape")
    return result


def _run_case(
    quadrature: Any,
    context: Any,
    source: Any,
    fixture: dict[str, Any],
    *,
    prefix: str,
    peak_field: float,
    arrays: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    duration = float(fixture["pulse_duration_au"])
    intervals = int(fixture["new_intervals"])
    cache: dict[float, Any] = {}
    progress_prefix = "mixed-magnus-n128/distorted/above_threshold"
    initial_models = mixed._evaluate(
        quadrature,
        context,
        source,
        0.0,
        cache,
        progress_prefix=progress_prefix,
    ).models
    initial_coefficients = linear._initial_ground(initial_models)
    initial_contravariant = initial_coefficients @ initial_coefficients.conj().T
    rows: list[dict[str, Any]] = []
    final_exact: np.ndarray | None = None
    diagnostic_arrays: dict[str, np.ndarray] = {}
    for model_name in fixture["models"]:
        history = mixed._history(
            quadrature,
            context,
            source,
            cache,
            str(model_name),
            duration=duration,
            intervals=intervals,
            progress_prefix=progress_prefix,
        )
        initial_mixed = initial_contravariant @ history.endpoint_metrics[0]
        trajectory = propagate_experimental_mixed_density(
            initial_mixed,
            history,
            quadrature.backend,
        )
        mixed_density = np.asarray(trajectory.mixed_densities)
        contravariant = np.asarray(trajectory.contravariant_densities)
        diagnostic = _diagnostic_array(history, trajectory, source, quadrature.backend)
        arrays[f"{prefix}__{model_name}__n{intervals}__D"] = mixed_density
        arrays[f"{prefix}__{model_name}__n{intervals}__P"] = contravariant
        diagnostic_name = f"{prefix}__{model_name}__n{intervals}__diagnostics"
        arrays[diagnostic_name] = diagnostic
        diagnostic_arrays[str(model_name)] = diagnostic
        rows.append(
            {
                "prefix": prefix,
                "geometry": fixture["geometry"],
                "grid_level": int(fixture["grid_level"]),
                "case": fixture["selection_role"],
                "peak_field_z_au": peak_field,
                "model": model_name,
                "intervals": intervals,
                "step_au": history.interval_au,
                "diagnostic_array": diagnostic_name,
                "maximum_cross_metric_residual": float(diagnostic[:, 10].max()),
                "maximum_trace_drift": float(diagnostic[:, 14].max()),
                "maximum_idempotency_residual": float(diagnostic[:, 13].max()),
                "maximum_metric_hermiticity_residual": float(diagnostic[:, 11].max()),
                "maximum_contravariant_hermiticity_residual": float(
                    diagnostic[:, 12].max()
                ),
                "minimum_endpoint_metric_eigenvalue": float(
                    min(diagnostic[:, 4].min(), diagnostic[:, 5].min())
                ),
                "maximum_endpoint_metric_condition_number": float(
                    diagnostic[:, 6].max()
                ),
                "maximum_generator_frobenius": float(
                    max(diagnostic[:, 7].max(), diagnostic[:, 8].max())
                ),
                "maximum_generator_commutator_frobenius": float(
                    diagnostic[:, 9].max()
                ),
            }
        )
        if model_name == "exact":
            final_exact = contravariant[-1]
    if final_exact is None:
        raise RuntimeError("fine exact trajectory was not produced")
    direct_connection = max(value.direct_connection_residual for value in cache.values())
    direct_metric_dot = max(value.direct_metric_dot_residual for value in cache.values())
    metric_compatibility = max(value.metric_compatibility_residual for value in cache.values())
    for row in rows:
        model_name = str(row["model"])
        final = arrays[f"{prefix}__{model_name}__n{intervals}__P"][-1]
        row.update(
            {
                "final_density_distance_from_fine_exact": float(
                    np.linalg.norm(final - final_exact)
                ),
                "unique_matrix_samples": len(cache),
                "maximum_direct_connection_residual": direct_connection,
                "maximum_direct_metric_dot_residual": direct_metric_dot,
                "maximum_metric_compatibility_residual": metric_compatibility,
            }
        )
    return rows, {name: str(array.shape) for name, array in diagnostic_arrays.items()}


def _plan(fixture: dict[str, Any], wp5_analysis: Path) -> dict[str, Any]:
    source_mixed = Path(str(fixture["source_mixed_magnus_execution"]))
    source_n64 = Path(str(fixture["source_n64_execution"]))
    selected_path = wp5_analysis / "tables/selected_dynamical_cases.csv"
    plan: dict[str, Any] = {
        "schema": "aion.exact-one-electron-wp6-mixed-magnus-n128-diagnostics-plan",
        "version": "1.0.0",
        "status": "targeted_diagnostic_continuation_requested",
        "fixture": fixture,
        "wp5_analysis_directory": str(wp5_analysis),
        "provenance": {
            "code": {
                "repository": str(_REPOSITORY),
                "branch": _git("branch", "--show-current"),
                "commit": _git("rev-parse", "HEAD"),
                "dirty": bool(_git("status", "--porcelain")),
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
                "conda_lock": _sha256(_LOCK),
                "wp5_selected_cases": _sha256(selected_path),
                "source_mixed_magnus_completed": _sha256(source_mixed / "completed.json"),
                "source_n64_completed": _sha256(source_n64 / "completed.json"),
                "runner": _sha256(Path(__file__).resolve()),
                "tensorial_propagator": _sha256(
                    _REPOSITORY / "src/aion/propagation/tensorial.py"
                ),
            },
        },
    }
    plan["plan_id"] = canonical_sha256(plan)
    return plan


def main() -> None:
    args = _arguments()
    fixture = _json(_FIXTURE)
    if tuple(fixture["models"]) != linear._MODEL_NAMES:
        raise ValueError("continuation model inventory disagrees with WP6")
    wp5_analysis = args.wp5_analysis.resolve()
    selected = linear._selected_cases(wp5_analysis / "tables/selected_dynamical_cases.csv")
    _source_result, source_arrays = _authenticate_execution(
        Path(str(fixture["source_mixed_magnus_execution"]))
    )
    _n64_result, n64_arrays = _authenticate_execution(
        Path(str(fixture["source_n64_execution"]))
    )
    plan = _plan(fixture, wp5_analysis)
    if args.execution_directory is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        suffix = str(plan["provenance"]["code"]["commit"])[:12]
        execution = args.output_root.resolve() / (
            f"wp6_mixed_magnus_n128_diagnostics_{stamp}_{suffix}"
        )
    else:
        execution = args.execution_directory.resolve()
    execution.mkdir(parents=True, exist_ok=True)
    with (execution / ".campaign.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = execution / "execution_plan.json"
        if plan_path.exists():
            if _json(plan_path)["plan_id"] != plan["plan_id"]:
                raise ValueError("existing diagnostic directory has a different plan")
        else:
            _write_json(plan_path, plan)

        start = time.time()
        geometry_name = str(fixture["geometry"])
        case = str(fixture["selection_role"])
        geometries = _json(linear._WP5_FIXTURE)["geometries"]
        reference = prepare_one_electron_ao_reference(
            linear._h3_config(
                list(geometries[geometry_name]["coordinates_au"]),
                str(fixture["basis"]),
                [0.0, 0.0, 0.0],
            )
        )
        quadrature = prepare_ao_quadrature(
            reference,
            BackendConfig(),
            grid_policy=AOGridPolicy.qualification(int(fixture["grid_level"])),
            block_size=int(fixture["block_size"]),
        )
        context = prepare_exact_one_electron_model_context(quadrature)
        selected_case = selected[f"{geometry_name}:{case}"]
        peak_field = float(selected_case["field_z_au"])
        source = linear._source_factory(
            duration=float(fixture["pulse_duration_au"]),
            peak_magnetic=np.asarray((0.0, 0.0, peak_field)),
            peak_electric=np.zeros(3),
            origin=(0.0, 0.0, 0.0),
            gauge_kind=MagneticGaugeKind.SYMMETRIC,
            landau_axis=None,
        )
        prefix = f"h3_{geometry_name}_level{int(fixture['grid_level'])}_{case}"
        arrays: dict[str, np.ndarray] = {}
        trajectory_rows, diagnostic_shapes = _run_case(
            quadrature,
            context,
            source,
            fixture,
            prefix=prefix,
            peak_field=peak_field,
            arrays=arrays,
        )
        convergence, stability = _continuation_rows(
            fixture,
            source_arrays,
            n64_arrays,
            arrays,
            prefix,
        )
        result = {
            "schema": "aion.exact-one-electron-wp6-mixed-magnus-n128-diagnostics-result",
            "version": "1.0.0",
            "status": "targeted_diagnostic_continuation_executed_unreviewed",
            "plan_id": plan["plan_id"],
            "diagnostic_columns": _DIAGNOSTIC_COLUMNS,
            "diagnostic_shapes": diagnostic_shapes,
            "trajectory_rows": trajectory_rows,
            "temporal_convergence": convergence,
            "model_difference_stability": stability,
            "array_semantic_sha256": {
                name: canonical_sha256(value) for name, value in arrays.items()
            },
            "wall_time_seconds": time.time() - start,
        }
        arrays_path = execution / "arrays.npz"
        result_path = execution / "result.json"
        _write_npz(arrays_path, arrays)
        _write_json(result_path, result)
        _write_json(
            execution / "completed.json",
            {
                "schema": (
                    "aion.exact-one-electron-wp6-mixed-magnus-"
                    "n128-diagnostics-completed"
                ),
                "version": "1.0.0",
                "status": result["status"],
                "plan_id": plan["plan_id"],
                "execution_plan_sha256": _sha256(plan_path),
                "result_sha256": _sha256(result_path),
                "arrays_sha256": _sha256(arrays_path),
                "completed_utc": datetime.now(UTC).isoformat(),
            },
        )
        print(f"execution_directory={execution}")
        print(f"status={result['status']}")
        print(f"wall_time_seconds={result['wall_time_seconds']:.3f}")


if __name__ == "__main__":
    main()
