#!/usr/bin/env python3
"""Authenticate and validate a WP5 multicentre execution."""

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
    return parser.parse_args().execution_directory.resolve()


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


def _identity(path: Path) -> tuple[dict[str, Any], str]:
    value = _json(path)
    core = dict(value)
    identity = core.pop("manifest_id", core.pop("plan_id", None))
    if identity is None or canonical_sha256(core) != identity:
        raise ValueError(f"canonical identity mismatch: {path}")
    return value, str(identity)


def main() -> None:
    execution = _arguments()
    plan, plan_id = _identity(execution / "execution_plan.json")
    index = _json(execution / "execution_index.json")
    if index["plan_id"] != plan_id:
        raise ValueError("WP5 execution index does not reference its plan")
    qualification = plan["status"] == "executed_unreviewed"
    expected_status = (
        "authenticated_executed_unreviewed"
        if qualification
        else "authenticated_smoke_not_qualification"
    )
    if index["status"] != expected_status:
        raise ValueError("WP5 execution status is incomplete or inconsistent")
    if index["completed_member_count"] != index["requested_member_count"]:
        raise ValueError("WP5 execution member count is incomplete")

    fixture = plan["fixture"]
    tolerances = fixture["acceptance_tolerances"]
    selection_margin = float(
        fixture["selected_case_targets"]["selection_minimum_metric_eigenvalue"]
    )
    maximum_loop = 0.0
    maximum_hermiticity = 0.0
    maximum_gauge = 0.0
    maximum_reversal = 0.0
    maximum_pair_ratio = 0.0
    minimum_exact_metric = np.inf
    approximate_losses: dict[str, int] = {
        "p0": 0,
        "geometric_b1": 0,
        "full_b1": 0,
    }
    array_count = 0
    for record in index["completed_members"]:
        directory = Path(record["member_directory"])
        manifest, manifest_id = _identity(directory / "manifest.json")
        result = _json(directory / "result.json")
        completed = _json(directory / "completed.json")
        if completed["manifest_id"] != manifest_id or result["manifest_id"] != manifest_id:
            raise ValueError(f"WP5 member identity was not propagated: {directory}")
        for filename, field in (
            ("manifest.json", "manifest_sha256"),
            ("result.json", "result_sha256"),
            ("arrays.npz", "arrays_sha256"),
        ):
            value = _sha256(directory / filename)
            if value != completed[field] or value != record[field]:
                raise ValueError(f"WP5 member hash mismatch: {directory / filename}")
        for source in manifest["accepted_pair_sources"]:
            source_directory = Path(source["directory"])
            for filename, field in (
                ("manifest.json", "manifest_sha256"),
                ("result.json", "result_sha256"),
                ("arrays.npz", "arrays_sha256"),
            ):
                if _sha256(source_directory / filename) != source[field]:
                    raise ValueError(f"accepted pair source changed: {source_directory}")

        semantic = result["array_semantic_sha256"]
        with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != set(semantic):
                raise ValueError(f"WP5 array inventory mismatch: {directory}")
            for name in arrays.files:
                value = np.asarray(arrays[name])
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"non-finite WP5 array: {directory}:{name}")
                if canonical_sha256(value) != semantic[name]:
                    raise ValueError(f"WP5 semantic array hash mismatch: {directory}:{name}")
                array_count += 1
            if not np.array_equal(
                arrays["metric_scan__geometric_b1__matrices"],
                arrays["metric_scan__full_b1__matrices"],
            ):
                raise ValueError("gB1 and B1 metrics differ in the static magnetic sector")
            reversal_names = [
                name for name in arrays.files if name.startswith("reversal_residual__")
            ]
            maximum_reversal = max(
                maximum_reversal,
                *(float(np.asarray(arrays[name])) for name in reversal_names),
            )

        for row in result["metric_scan"]:
            maximum_loop = max(
                maximum_loop,
                float(row["maximum_flux_identity_residual"]),
                float(row["maximum_gauge_residual"]),
                float(row["maximum_orientation_reversal_residual"]),
            )
            minimum_exact_metric = min(
                minimum_exact_metric, float(row["exact_minimum_eigenvalue"])
            )
            maximum_hermiticity = max(
                maximum_hermiticity,
                *(
                    float(row[f"{model}_hermiticity_residual"])
                    for model in ("exact", "p0", "geometric_b1", "full_b1")
                ),
            )
            for model in approximate_losses:
                if float(row[f"{model}_minimum_eigenvalue"]) <= 0.0:
                    approximate_losses[model] += 1

        selected = result["selected_dynamical_cases"]
        if {row["label"] for row in selected} != {
            "sub_threshold",
            "near_threshold",
            "above_threshold",
        }:
            raise ValueError("WP5 did not select the three declared dynamical cases")
        if any(
            float(row["minimum_model_metric_eigenvalue"]) <= selection_margin
            for row in selected
        ):
            raise ValueError("a selected WP5 dynamical case lacks metric safety margin")

        for row in result["generalized_spectral_comparisons"]:
            maximum_gauge = max(
                maximum_gauge,
                float(row["exact_gauge_barred_residual"]),
                float(row["exact_gauge_congruence_residual"]),
                float(row["exact_gauge_spectrum_residual"]),
            )
        pair_rows = result["pair_restrictions"]
        if {int(row["edge_index"]) for row in pair_rows} != {0, 1, 2}:
            raise ValueError("WP5 pair restrictions do not cover all three edges")
        if len(pair_rows) != 3 * 4 * 2:
            raise ValueError("WP5 pair restriction model/family inventory is incomplete")
        maximum_pair_ratio = max(
            maximum_pair_ratio,
            *(float(row["level5_residual_to_floor"]) for row in pair_rows),
        )

    if maximum_loop > float(tolerances["loop_holonomy_absolute"]):
        raise ValueError("WP5 loop holonomy identity failed")
    if maximum_hermiticity > float(tolerances["metric_hermiticity_relative"]):
        raise ValueError("WP5 metric Hermiticity failed")
    if maximum_gauge > float(tolerances["gauge_congruence_relative"]):
        raise ValueError("WP5 exact gauge identity failed")
    if maximum_reversal > float(tolerances["field_reversal_relative"]):
        raise ValueError("WP5 field reversal identity failed")
    if maximum_pair_ratio > float(tolerances["pair_restriction_floor_multiplier"]):
        raise ValueError("WP5 pair restriction exceeds its refined numerical floor")
    if minimum_exact_metric <= 0.0:
        raise ValueError("WP5 exact analytic Gram metric lost positivity")

    print(f"execution={execution}")
    print(f"members_verified={index['completed_member_count']}")
    print(f"arrays_verified={array_count}")
    print(f"maximum_loop_identity_residual={maximum_loop:.6e}")
    print(f"maximum_exact_gauge_residual={maximum_gauge:.6e}")
    print(f"maximum_field_reversal_residual={maximum_reversal:.6e}")
    print(f"maximum_pair_residual_to_floor={maximum_pair_ratio:.6e}")
    print(f"minimum_exact_metric_eigenvalue={minimum_exact_metric:.6e}")
    print(f"approximate_nonpositive_counts={approximate_losses}")
    print(f"status={expected_status}")


if __name__ == "__main__":
    main()
