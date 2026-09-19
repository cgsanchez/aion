#!/usr/bin/env python3
"""Run the targeted 64-step WP6 mixed-density Magnus continuation."""

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

_REPOSITORY = Path(__file__).resolve().parents[1]
_FIXTURE = _REPOSITORY / "tests/fixtures/exact_one_electron/wp6_mixed_magnus_n64.fixture.json"
_LOCK = _REPOSITORY / "conda-linux-64.lock"
_DEFAULT_OUTPUT_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/exact_one_electron_qualification"
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


def _authenticate_source(
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
            raise ValueError(f"source mixed-Magnus hash mismatch: {filename}")
    if result["plan_id"] != completed["plan_id"]:
        raise ValueError("source mixed-Magnus plan identity mismatch")
    arrays: dict[str, np.ndarray] = {}
    with np.load(root / "arrays.npz") as archive:
        semantic = result["array_semantic_sha256"]
        for name in archive.files:
            value = np.asarray(archive[name])
            if canonical_sha256(value) != semantic[name]:
                raise ValueError(f"source array semantic hash mismatch: {name}")
            arrays[name] = value
    return result, arrays


def _continuation_rows(
    fixture: dict[str, Any],
    source_arrays: dict[str, np.ndarray],
    new_arrays: dict[str, np.ndarray],
    prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coarse, medium = (int(value) for value in fixture["source_intervals"])
    fine = int(fixture["new_intervals"])
    convergence: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []
    exact_medium = source_arrays[f"{prefix}__exact__n{medium}__P"][-1]
    exact_fine = new_arrays[f"{prefix}__exact__n{fine}__P"][-1]
    for model in fixture["models"]:
        coarse_density = source_arrays[f"{prefix}__{model}__n{coarse}__P"][-1]
        medium_density = source_arrays[f"{prefix}__{model}__n{medium}__P"][-1]
        fine_density = new_arrays[f"{prefix}__{model}__n{fine}__P"][-1]
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


def _plan(fixture: dict[str, Any], wp5_analysis: Path) -> dict[str, Any]:
    source = Path(str(fixture["source_mixed_magnus_execution"]))
    selected_path = wp5_analysis / "tables/selected_dynamical_cases.csv"
    plan: dict[str, Any] = {
        "schema": "aion.exact-one-electron-wp6-mixed-magnus-n64-plan",
        "version": "1.0.0",
        "status": "targeted_experimental_continuation_requested",
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
                "source_mixed_magnus_completed": _sha256(source / "completed.json"),
                "runner": _sha256(Path(__file__).resolve()),
                "tensorial_propagator": _sha256(_REPOSITORY / "src/aion/propagation/tensorial.py"),
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
    source_root = Path(str(fixture["source_mixed_magnus_execution"]))
    source_result, source_arrays = _authenticate_source(source_root)
    del source_result
    plan = _plan(fixture, wp5_analysis)
    if args.execution_directory is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        suffix = str(plan["provenance"]["code"]["commit"])[:12]
        execution = args.output_root.resolve() / f"wp6_mixed_magnus_n64_{stamp}_{suffix}"
    else:
        execution = args.execution_directory.resolve()
    execution.mkdir(parents=True, exist_ok=True)
    with (execution / ".campaign.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = execution / "execution_plan.json"
        if plan_path.exists():
            if _json(plan_path)["plan_id"] != plan["plan_id"]:
                raise ValueError("existing continuation directory has a different plan")
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
        trajectory_rows = mixed._run_case(
            quadrature,
            context,
            source,
            duration=float(fixture["pulse_duration_au"]),
            intervals=(int(fixture["new_intervals"]),),
            prefix=prefix,
            geometry=geometry_name,
            grid_level=int(fixture["grid_level"]),
            case=case,
            peak_field=peak_field,
            arrays=arrays,
        )
        convergence, stability = _continuation_rows(
            fixture,
            source_arrays,
            arrays,
            prefix,
        )
        result = {
            "schema": "aion.exact-one-electron-wp6-mixed-magnus-n64-result",
            "version": "1.0.0",
            "status": "targeted_experimental_continuation_executed_unreviewed",
            "plan_id": plan["plan_id"],
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
                "schema": "aion.exact-one-electron-wp6-mixed-magnus-n64-completed",
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
