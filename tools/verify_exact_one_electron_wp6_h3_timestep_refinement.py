#!/usr/bin/env python3
"""Authenticate and validate the WP6 three-centre timestep supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from aion.config import canonical_sha256


def _arguments() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution_directory", type=Path)
    value: Path = parser.parse_args().execution_directory
    return value.resolve()


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


def main() -> None:
    execution = _arguments()
    plan_path = execution / "execution_plan.json"
    result_path = execution / "result.json"
    arrays_path = execution / "arrays.npz"
    completed_path = execution / "completed.json"
    plan = _json(plan_path)
    core = dict(plan)
    plan_id = str(core.pop("plan_id"))
    if canonical_sha256(core) != plan_id:
        raise ValueError("supplement plan identity mismatch")
    result = _json(result_path)
    completed = _json(completed_path)
    if result["plan_id"] != plan_id or completed["plan_id"] != plan_id:
        raise ValueError("supplement identity was not propagated")
    for path, field in (
        (plan_path, "execution_plan_sha256"),
        (result_path, "result_sha256"),
        (arrays_path, "arrays_sha256"),
    ):
        if _sha256(path) != completed[field]:
            raise ValueError(f"supplement artifact hash mismatch: {path.name}")
    if result["status"] != "executed_unreviewed":
        raise ValueError("supplement execution is incomplete")

    semantic = result["array_semantic_sha256"]
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(semantic):
            raise ValueError("supplement array inventory mismatch")
        for name in arrays.files:
            value = np.asarray(arrays[name])
            if not np.all(np.isfinite(value)):
                raise ValueError(f"non-finite supplement array: {name}")
            if canonical_sha256(value) != semantic[name]:
                raise ValueError(f"supplement semantic hash mismatch: {name}")

    trajectories = result["trajectory_rows"]
    convergence = result["timestep_convergence"]
    stability = result["model_difference_stability"]
    if len(trajectories) != 108:
        raise ValueError("supplement trajectory inventory is incomplete")
    if len(convergence) != 72:
        raise ValueError("supplement convergence inventory is incomplete")
    if len(stability) != 60:
        raise ValueError("supplement model-difference inventory is incomplete")
    if {int(row["intervals"]) for row in trajectories} != {8, 16, 32}:
        raise ValueError("supplement timestep sequence is incomplete")
    if {str(row["geometry"]) for row in trajectories} != {
        "equilateral",
        "distorted",
    }:
        raise ValueError("supplement geometry inventory is incomplete")

    tolerances = plan["fixture"]["acceptance_tolerances"]
    maximum_norm = max(
        float(row["maximum_corrected_norm_drift"]) for row in trajectories
    )
    minimum_metric = min(
        float(row["minimum_endpoint_metric_eigenvalue"]) for row in trajectories
    )
    maximum_compatibility = max(
        float(row["maximum_metric_compatibility_residual"])
        for row in trajectories
    )
    orders = [
        float(row["measured_order_to_next"])
        for row in convergence
        if row["measured_order_to_next"] is not None
    ]
    finest_fractions = [
        float(row["refinement_fraction_of_fine_model_difference"])
        for row in stability
        if int(row["fine_intervals"]) == 32
        and row["refinement_fraction_of_fine_model_difference"] is not None
    ]
    numerical_criteria_pass = bool(
        maximum_norm <= float(tolerances["corrected_norm_drift"])
        and minimum_metric > float(tolerances["minimum_metric_eigenvalue"])
        and maximum_compatibility
        <= float(tolerances["metric_compatibility_relative"])
        and len(orders) == 36
        and min(orders) >= float(tolerances["minimum_time_convergence_order"])
        and len(finest_fractions) == 30
        and max(finest_fractions)
        <= float(tolerances["fine_model_difference_refinement_fraction"])
    )

    print(f"execution={execution}")
    print("status=authenticated_executed_unreviewed")
    print(f"arrays_verified={len(semantic)}")
    print(f"maximum_norm_drift={maximum_norm:.6e}")
    print(f"minimum_metric_eigenvalue={minimum_metric:.6e}")
    print(f"maximum_metric_compatibility_residual={maximum_compatibility:.6e}")
    print(f"minimum_time_convergence_order={min(orders):.6f}")
    print(f"maximum_fine_model_refinement_fraction={max(finest_fractions):.6e}")
    print(f"numerical_criteria_pass={numerical_criteria_pass}")


if __name__ == "__main__":
    main()
