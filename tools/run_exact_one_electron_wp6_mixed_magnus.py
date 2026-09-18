#!/usr/bin/env python3
"""Run the authenticated WP6 mixed-density Gauss--Magnus experiment."""

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
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import run_exact_one_electron_wp6_linear as base

from aion.config import BackendConfig, canonical_sha256
from aion.electromagnetism import MagneticGaugeKind
from aion.electronic_structure import (
    AOGridPolicy,
    evaluate_exact_wilson_one_electron_sample,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
)
from aion.formulations import (
    ExactOneElectronModelContext,
    ExactOneElectronModelTriples,
    exact_one_electron_model_triples,
    prepare_exact_one_electron_model_context,
)
from aion.propagation import (
    ExperimentalGaussMagnusHistory,
    propagate_experimental_mixed_density,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_FIXTURE = (
    _REPOSITORY / "tests/fixtures/exact_one_electron/wp6_mixed_magnus_experiment.fixture.json"
)
_LOCK = _REPOSITORY / "conda-linux-64.lock"
_DEFAULT_OUTPUT_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/exact_one_electron_qualification"
)


@dataclass(frozen=True, slots=True)
class EvaluatedSample:
    """All model triples and exact-connection residuals at one time."""

    models: ExactOneElectronModelTriples
    direct_connection_residual: float
    direct_metric_dot_residual: float
    metric_compatibility_residual: float


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wp5-analysis", type=Path, default=base._DEFAULT_WP5_ANALYSIS)
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


def _model(value: ExactOneElectronModelTriples, name: str) -> Any:
    return getattr(value, name)


def _evaluate(
    quadrature: Any,
    context: ExactOneElectronModelContext,
    source: Any,
    time_au: float,
    cache: dict[float, EvaluatedSample],
    *,
    progress_prefix: str,
) -> EvaluatedSample:
    key = float(time_au)
    if key in cache:
        return cache[key]
    evaluated = evaluate_exact_wilson_one_electron_sample(quadrature, source(key))
    sample = EvaluatedSample(
        models=exact_one_electron_model_triples(evaluated, context),
        direct_connection_residual=(evaluated.connection.direct_factorized_connection_residual),
        direct_metric_dot_residual=(evaluated.connection.direct_factorized_metric_dot_residual),
        metric_compatibility_residual=(evaluated.connection.metric_compatibility_residual),
    )
    cache[key] = sample
    print(
        f"{progress_prefix} unique_sample={len(cache)} time_au={key:.12f}",
        flush=True,
    )
    return sample


def _history(
    quadrature: Any,
    context: ExactOneElectronModelContext,
    source: Any,
    cache: dict[float, EvaluatedSample],
    model_name: str,
    *,
    duration: float,
    intervals: int,
    progress_prefix: str,
) -> ExperimentalGaussMagnusHistory:
    step = duration / intervals
    offset = math.sqrt(3.0) / 6.0
    endpoints = tuple(
        _model(
            _evaluate(
                quadrature,
                context,
                source,
                index * step,
                cache,
                progress_prefix=progress_prefix,
            ).models,
            model_name,
        ).triple.metric
        for index in range(intervals + 1)
    )
    gauss_minus = tuple(
        _model(
            _evaluate(
                quadrature,
                context,
                source,
                (index + 0.5 - offset) * step,
                cache,
                progress_prefix=progress_prefix,
            ).models,
            model_name,
        ).triple
        for index in range(intervals)
    )
    gauss_plus = tuple(
        _model(
            _evaluate(
                quadrature,
                context,
                source,
                (index + 0.5 + offset) * step,
                cache,
                progress_prefix=progress_prefix,
            ).models,
            model_name,
        ).triple
        for index in range(intervals)
    )
    return ExperimentalGaussMagnusHistory(
        endpoint_metrics=endpoints,
        gauss_minus_triples=gauss_minus,
        gauss_plus_triples=gauss_plus,
        interval_au=step,
    )


def _trajectory_convergence(
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    intervals: tuple[int, ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    prefixes = sorted({str(row["prefix"]) for row in rows})
    for prefix in prefixes:
        for model in base._MODEL_NAMES:
            differences: list[float] = []
            pairs = tuple(pairwise(intervals))
            for coarse, fine in pairs:
                coarse_density = arrays[f"{prefix}__{model}__n{coarse}__P"][-1]
                fine_density = arrays[f"{prefix}__{model}__n{fine}__P"][-1]
                differences.append(float(np.linalg.norm(fine_density - coarse_density)))
            for index, ((coarse, fine), difference) in enumerate(
                zip(pairs, differences, strict=True)
            ):
                order = None
                if index + 1 < len(differences) and difference > 0.0:
                    following = differences[index + 1]
                    if following > 0.0:
                        order = math.log(difference / following) / math.log(2.0)
                output.append(
                    {
                        "prefix": prefix,
                        "model": model,
                        "coarse_intervals": coarse,
                        "fine_intervals": fine,
                        "final_contravariant_density_change": difference,
                        "measured_order_to_next": order,
                    }
                )
    return output


def _model_difference_stability(
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    intervals: tuple[int, ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metadata = {str(row["prefix"]): row for row in rows}
    for prefix in sorted(metadata):
        differences: dict[tuple[str, int], np.ndarray] = {}
        distances: dict[tuple[str, int], float] = {}
        for interval in intervals:
            exact = arrays[f"{prefix}__exact__n{interval}__P"][-1]
            for model in base._MODEL_NAMES[1:]:
                value = arrays[f"{prefix}__{model}__n{interval}__P"][-1]
                difference = np.asarray(value - exact)
                differences[(model, interval)] = difference
                distances[(model, interval)] = float(np.linalg.norm(difference))
        for model in base._MODEL_NAMES[1:]:
            model_rows: list[dict[str, Any]] = []
            for coarse, fine in pairwise(intervals):
                change = float(
                    np.linalg.norm(differences[(model, fine)] - differences[(model, coarse)])
                )
                fine_distance = distances[(model, fine)]
                source = metadata[prefix]
                model_rows.append(
                    {
                        "prefix": prefix,
                        "geometry": source["geometry"],
                        "case": source["case"],
                        "peak_field_z_au": source["peak_field_z_au"],
                        "model": model,
                        "coarse_intervals": coarse,
                        "fine_intervals": fine,
                        "coarse_model_difference": distances[(model, coarse)],
                        "fine_model_difference": fine_distance,
                        "model_difference_refinement_change": change,
                        "refinement_fraction_of_fine_model_difference": (
                            change / fine_distance
                            if fine_distance > np.finfo(np.float64).tiny
                            else None
                        ),
                    }
                )
            for index, row in enumerate(model_rows):
                order = None
                if index + 1 < len(model_rows):
                    current = float(row["model_difference_refinement_change"])
                    following = float(model_rows[index + 1]["model_difference_refinement_change"])
                    if current > 0.0 and following > 0.0:
                        order = math.log(current / following) / math.log(2.0)
                row["measured_refinement_order_to_next"] = order
            output.extend(model_rows)
    return output


def _run_case(
    quadrature: Any,
    context: ExactOneElectronModelContext,
    source: Any,
    *,
    duration: float,
    intervals: tuple[int, ...],
    prefix: str,
    geometry: str,
    grid_level: int,
    case: str,
    peak_field: float,
    arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    cache: dict[float, EvaluatedSample] = {}
    progress_prefix = f"mixed-magnus/{geometry}/{case}"
    initial_models = _evaluate(
        quadrature,
        context,
        source,
        0.0,
        cache,
        progress_prefix=progress_prefix,
    ).models
    initial_coefficients = base._initial_ground(initial_models)
    initial_contravariant = initial_coefficients @ initial_coefficients.conj().T
    rows: list[dict[str, Any]] = []
    final_exact: np.ndarray | None = None
    for count in intervals:
        for model_name in base._MODEL_NAMES:
            history = _history(
                quadrature,
                context,
                source,
                cache,
                model_name,
                duration=duration,
                intervals=count,
                progress_prefix=progress_prefix,
            )
            initial_mixed = initial_contravariant @ history.endpoint_metrics[0]
            trajectory = propagate_experimental_mixed_density(
                initial_mixed,
                history,
                quadrature.backend,
            )
            mixed = np.asarray(trajectory.mixed_densities)
            contravariant = np.asarray(trajectory.contravariant_densities)
            arrays[f"{prefix}__{model_name}__n{count}__D"] = mixed
            arrays[f"{prefix}__{model_name}__n{count}__P"] = contravariant
            rows.append(
                {
                    "prefix": prefix,
                    "geometry": geometry,
                    "grid_level": grid_level,
                    "case": case,
                    "peak_field_z_au": peak_field,
                    "model": model_name,
                    "intervals": count,
                    "step_au": history.interval_au,
                    "maximum_cross_metric_residual": max(
                        item.cross_metric_residual for item in trajectory.diagnostics
                    ),
                    "maximum_trace_drift": max(item.trace_drift for item in trajectory.diagnostics),
                    "maximum_trace_imaginary_abs": max(
                        item.trace_imaginary_abs for item in trajectory.diagnostics
                    ),
                    "maximum_idempotency_residual": max(
                        item.output_idempotency_residual for item in trajectory.diagnostics
                    ),
                    "maximum_metric_hermiticity_residual": max(
                        item.metric_hermiticity_residual for item in trajectory.diagnostics
                    ),
                    "maximum_contravariant_hermiticity_residual": max(
                        item.contravariant_hermiticity_residual for item in trajectory.diagnostics
                    ),
                    "minimum_endpoint_metric_eigenvalue": min(
                        float(np.linalg.eigvalsh(np.asarray(metric))[0])
                        for metric in history.endpoint_metrics
                    ),
                }
            )
            if model_name == "exact" and count == max(intervals):
                final_exact = contravariant[-1]
    if final_exact is None:
        raise RuntimeError("fine exact trajectory was not produced")
    for row in rows:
        value = arrays[f"{prefix}__{row['model']}__n{row['intervals']}__P"][-1]
        row["final_density_distance_from_fine_exact"] = float(np.linalg.norm(value - final_exact))
    for row in rows:
        row.update(
            {
                "unique_matrix_samples": len(cache),
                "maximum_direct_connection_residual": max(
                    value.direct_connection_residual for value in cache.values()
                ),
                "maximum_direct_metric_dot_residual": max(
                    value.direct_metric_dot_residual for value in cache.values()
                ),
                "maximum_metric_compatibility_residual": max(
                    value.metric_compatibility_residual for value in cache.values()
                ),
            }
        )
    return rows


def _plan(fixture: dict[str, Any], wp5_analysis: Path) -> dict[str, Any]:
    selected_path = wp5_analysis / "tables/selected_dynamical_cases.csv"
    source_supplement = Path(str(fixture["source_midpoint_supplement"]))
    plan: dict[str, Any] = {
        "schema": "aion.exact-one-electron-wp6-mixed-magnus-execution-plan",
        "version": "1.0.0",
        "status": "experimental_qualification_requested",
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
                "source_supplement_completed": _sha256(source_supplement / "completed.json"),
                "runner": _sha256(Path(__file__).resolve()),
                "time_connection": _sha256(
                    _REPOSITORY / "src/aion/electronic_structure/time_connection.py"
                ),
                "model_adapter": _sha256(
                    _REPOSITORY / "src/aion/formulations/exact_one_electron.py"
                ),
                "tensorial_propagator": _sha256(_REPOSITORY / "src/aion/propagation/tensorial.py"),
            },
        },
    }
    plan["plan_id"] = canonical_sha256(plan)
    return plan


def main() -> None:
    args = _arguments()
    fixture = _json(_FIXTURE)
    wp5_analysis = args.wp5_analysis.resolve()
    selected = base._selected_cases(wp5_analysis / "tables/selected_dynamical_cases.csv")
    if tuple(fixture["models"]) != base._MODEL_NAMES:
        raise ValueError("experiment model inventory disagrees with the WP6 adapter")
    plan = _plan(fixture, wp5_analysis)
    if args.execution_directory is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        suffix = str(plan["provenance"]["code"]["commit"])[:12]
        execution = args.output_root.resolve() / f"wp6_mixed_magnus_{stamp}_{suffix}"
    else:
        execution = args.execution_directory.resolve()
    execution.mkdir(parents=True, exist_ok=True)
    with (execution / ".campaign.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = execution / "execution_plan.json"
        if plan_path.exists():
            if _json(plan_path)["plan_id"] != plan["plan_id"]:
                raise ValueError("existing experiment directory has a different plan")
        else:
            _write_json(plan_path, plan)

        start = time.time()
        arrays: dict[str, np.ndarray] = {}
        rows: list[dict[str, Any]] = []
        intervals = tuple(int(value) for value in fixture["comparison_intervals"])
        geometries = _json(base._WP5_FIXTURE)["geometries"]
        for geometry_name in fixture["geometries"]:
            reference = prepare_one_electron_ao_reference(
                base._h3_config(
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
            for case in fixture["selection_roles"]:
                selected_case = selected[f"{geometry_name}:{case}"]
                peak_field = float(selected_case["field_z_au"])
                source = base._source_factory(
                    duration=float(fixture["pulse_duration_au"]),
                    peak_magnetic=np.asarray((0.0, 0.0, peak_field)),
                    peak_electric=np.zeros(3),
                    origin=(0.0, 0.0, 0.0),
                    gauge_kind=MagneticGaugeKind.SYMMETRIC,
                    landau_axis=None,
                )
                prefix = f"h3_{geometry_name}_level{int(fixture['grid_level'])}_{case}"
                rows.extend(
                    _run_case(
                        quadrature,
                        context,
                        source,
                        duration=float(fixture["pulse_duration_au"]),
                        intervals=intervals,
                        prefix=prefix,
                        geometry=str(geometry_name),
                        grid_level=int(fixture["grid_level"]),
                        case=str(case),
                        peak_field=peak_field,
                        arrays=arrays,
                    )
                )
        convergence = _trajectory_convergence(rows, arrays, intervals)
        stability = _model_difference_stability(rows, arrays, intervals)
        result = {
            "schema": "aion.exact-one-electron-wp6-mixed-magnus-result",
            "version": "1.0.0",
            "status": "experimental_executed_unreviewed",
            "plan_id": plan["plan_id"],
            "trajectory_rows": rows,
            "timestep_convergence": convergence,
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
                "schema": "aion.exact-one-electron-wp6-mixed-magnus-completed",
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
