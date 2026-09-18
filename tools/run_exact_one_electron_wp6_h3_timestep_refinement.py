#!/usr/bin/env python3
"""Run the authenticated WP6 three-centre timestep-refinement supplement."""

from __future__ import annotations

import argparse
import copy
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
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import run_exact_one_electron_wp6_linear as base

from aion.config import canonical_sha256

_REPOSITORY = Path(__file__).resolve().parents[1]
_FIXTURE = (
    _REPOSITORY
    / "tests/fixtures/exact_one_electron/wp6_h3_timestep_refinement.fixture.json"
)
_LOCK = _REPOSITORY / "conda-linux-64.lock"
_DEFAULT_OUTPUT_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "exact_one_electron_qualification"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wp5-analysis", type=Path, default=base._DEFAULT_WP5_ANALYSIS)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execution-directory", type=Path)
    return parser.parse_args()


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


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _model_difference_stability(
    rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    intervals: tuple[int, ...],
) -> list[dict[str, Any]]:
    prefixes = sorted({str(row["prefix"]) for row in rows})
    metadata = {str(row["prefix"]): row for row in rows}
    output: list[dict[str, Any]] = []
    for prefix in prefixes:
        source = metadata[prefix]
        differences: dict[tuple[str, int], np.ndarray] = {}
        distances: dict[tuple[str, int], float] = {}
        for interval in intervals:
            exact = arrays[f"{prefix}__exact__n{interval}__coefficients"][-1]
            exact_density = exact @ exact.conj().T
            for model in base._MODEL_NAMES[1:]:
                value = arrays[f"{prefix}__{model}__n{interval}__coefficients"][-1]
                model_density = value @ value.conj().T
                difference = np.asarray(model_density - exact_density)
                differences[(model, interval)] = difference
                distances[(model, interval)] = float(np.linalg.norm(difference))
        for model in base._MODEL_NAMES[1:]:
            model_rows: list[dict[str, Any]] = []
            for coarse, fine in pairwise(intervals):
                change = float(
                    np.linalg.norm(
                        differences[(model, fine)] - differences[(model, coarse)]
                    )
                )
                fine_distance = distances[(model, fine)]
                model_rows.append(
                    {
                        "prefix": prefix,
                        "geometry": source["geometry"],
                        "grid_level": source["grid_level"],
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
                    following = float(
                        model_rows[index + 1][
                            "model_difference_refinement_change"
                        ]
                    )
                    if current > 0.0 and following > 0.0:
                        order = math.log(current / following) / math.log(2.0)
                row["measured_refinement_order_to_next"] = order
            output.extend(model_rows)
    return output


def _plan(
    fixture: dict[str, Any],
    wp5_analysis: Path,
) -> dict[str, Any]:
    primary = Path(str(fixture["source_primary_execution"]))
    selected_path = wp5_analysis / "tables/selected_dynamical_cases.csv"
    plan: dict[str, Any] = {
        "schema": "aion.exact-one-electron-wp6-h3-refinement-execution-plan",
        "version": "1.0.0",
        "status": "qualification_supplement_requested",
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
                "wp6_primary_fixture": _sha256(base._FIXTURE),
                "wp5_fixture": _sha256(base._WP5_FIXTURE),
                "g5_review": _sha256(base._G5_REVIEW),
                "conda_lock": _sha256(_LOCK),
                "wp5_selected_cases": _sha256(selected_path),
                "primary_execution_plan": _sha256(primary / "execution_plan.json"),
                "primary_result": _sha256(primary / "result.json"),
                "primary_arrays": _sha256(primary / "arrays.npz"),
                "runner": _sha256(Path(__file__).resolve()),
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
    selected = base._selected_cases(wp5_analysis / "tables/selected_dynamical_cases.csv")
    plan = _plan(fixture, wp5_analysis)
    if args.execution_directory is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        suffix = str(plan["provenance"]["code"]["commit"])[:12]
        execution = args.output_root.resolve() / f"wp6_h3_refinement_{stamp}_{suffix}"
    else:
        execution = args.execution_directory.resolve()
    execution.mkdir(parents=True, exist_ok=True)
    with (execution / ".campaign.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = execution / "execution_plan.json"
        if plan_path.exists():
            if _json(plan_path)["plan_id"] != plan["plan_id"]:
                raise ValueError("existing supplement directory has a different plan")
        else:
            _write_json(plan_path, plan)

        start = time.time()
        original = _json(base._FIXTURE)
        run_fixture = copy.deepcopy(original)
        settings = run_fixture["three_centre"]
        settings["fine_intervals"] = max(
            int(value) for value in fixture["comparison_intervals"]
        )
        settings["comparison_intervals"] = list(fixture["comparison_intervals"])
        settings["qualification_grid_level"] = int(fixture["grid_level"])
        settings["basis"] = str(fixture["basis"])
        settings["block_size"] = int(fixture["block_size"])
        settings["pulse_duration_au"] = float(fixture["pulse_duration_au"])
        settings["selection_roles"] = list(fixture["selection_roles"])
        geometries = _json(base._WP5_FIXTURE)["geometries"]
        arrays: dict[str, np.ndarray] = {}
        rows: list[dict[str, Any]] = []
        convergence: list[dict[str, Any]] = []
        for geometry_name in fixture["geometries"]:
            geometry_rows, geometry_convergence, _final = base._run_three_centre_level(
                geometry_name=str(geometry_name),
                coordinates=list(geometries[str(geometry_name)]["coordinates_au"]),
                level=int(fixture["grid_level"]),
                fixture=run_fixture,
                selected=selected,
                smoke=False,
                arrays=arrays,
            )
            rows.extend(geometry_rows)
            convergence.extend(geometry_convergence)
        intervals = tuple(int(value) for value in fixture["comparison_intervals"])
        stability = _model_difference_stability(rows, arrays, intervals)
        semantic = {name: canonical_sha256(value) for name, value in arrays.items()}
        result = {
            "schema": "aion.exact-one-electron-wp6-h3-refinement-result",
            "version": "1.0.0",
            "status": "executed_unreviewed",
            "plan_id": plan["plan_id"],
            "trajectory_rows": rows,
            "timestep_convergence": convergence,
            "model_difference_stability": stability,
            "array_semantic_sha256": semantic,
            "wall_time_seconds": time.time() - start,
        }
        arrays_path = execution / "arrays.npz"
        result_path = execution / "result.json"
        _write_npz(arrays_path, arrays)
        _write_json(result_path, result)
        _write_json(
            execution / "completed.json",
            {
                "schema": "aion.exact-one-electron-wp6-h3-refinement-completed",
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
