#!/usr/bin/env python3
"""Authenticate the combined primary-plus-supplement G6 analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

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


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    execution = _arguments()
    analysis = execution / "analysis"
    completed = _json(analysis / "completed.json")
    manifest = _json(analysis / "manifest.json")
    core = dict(manifest)
    manifest_id = str(core.pop("manifest_id"))
    if canonical_sha256(core) != manifest_id:
        raise ValueError("combined G6 analysis manifest identity mismatch")
    summary = _json(analysis / "summary.json")
    if completed["manifest_id"] != manifest_id or summary["manifest_id"] != manifest_id:
        raise ValueError("combined G6 analysis identity was not propagated")
    for filename, field in (
        ("manifest.json", "manifest_sha256"),
        ("summary.json", "summary_sha256"),
        ("report.md", "report_sha256"),
    ):
        if _sha256(analysis / filename) != completed[field]:
            raise ValueError(f"combined G6 analysis hash mismatch: {filename}")
    for filename, field in (
        ("execution_plan.json", "supplement_execution_plan_sha256"),
        ("result.json", "supplement_result_sha256"),
        ("arrays.npz", "supplement_arrays_sha256"),
    ):
        if _sha256(execution / filename) != manifest[field]:
            raise ValueError(f"supplement source changed after analysis: {filename}")
    plan = _json(execution / "execution_plan.json")
    primary = Path(str(plan["fixture"]["source_primary_execution"]))
    if _sha256(primary / "analysis/summary.json") != manifest[
        "primary_analysis_summary_sha256"
    ]:
        raise ValueError("primary analysis summary changed after combined analysis")
    if _sha256(primary / "analysis/manifest.json") != manifest[
        "primary_analysis_manifest_sha256"
    ]:
        raise ValueError("primary analysis manifest changed after combined analysis")
    if summary["status"] != "derived_executed_unreviewed":
        raise ValueError("combined G6 analysis status is incorrect")
    if summary["review"] != {
        "reviewed": False,
        "reviewer": None,
        "decision": "pending",
    }:
        raise ValueError("combined G6 analysis review state was mutated")
    if summary["checks"]["metric_regularization_used"]:
        raise ValueError("combined G6 analysis claims metric regularization")

    artifact_count = 0
    for category in ("tables", "figures"):
        for name, expected in summary["artifacts"][category].items():
            path = analysis / category / name
            if _sha256(path) != expected:
                raise ValueError(f"combined G6 artifact hash mismatch: {path}")
            artifact_count += 1
    counts: dict[str, int] = {}
    for name in summary["artifacts"]["tables"]:
        counts[Path(name).stem] = len(_csv(analysis / "tables" / name))
    if counts != summary["row_counts"]:
        raise ValueError("combined G6 table counts disagree with summary")
    if counts != {
        "trajectory_diagnostics": 108,
        "timestep_convergence": 72,
        "model_difference_stability": 60,
        "focused_resolution_case": 5,
    }:
        raise ValueError("combined G6 evidence inventory is incomplete")

    print(f"analysis={analysis}")
    print(f"manifest_id={manifest_id}")
    print(f"artifacts_verified={artifact_count}")
    print(f"g6_gate_ready_for_review={summary['checks']['g6_gate_ready_for_review']}")
    print("status=authenticated_derived_executed_unreviewed")


if __name__ == "__main__":
    main()
