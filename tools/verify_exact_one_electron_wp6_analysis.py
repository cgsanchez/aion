#!/usr/bin/env python3
"""Authenticate a derived WP6 linear-dynamics analysis package."""

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
        raise ValueError("WP6 analysis manifest identity mismatch")
    summary = _json(analysis / "summary.json")
    if completed["manifest_id"] != manifest_id or summary["manifest_id"] != manifest_id:
        raise ValueError("WP6 analysis identity was not propagated")
    for filename, field in (
        ("manifest.json", "manifest_sha256"),
        ("summary.json", "summary_sha256"),
        ("report.md", "report_sha256"),
    ):
        if _sha256(analysis / filename) != completed[field]:
            raise ValueError(f"WP6 analysis hash mismatch: {filename}")
    for filename, field in (
        ("execution_plan.json", "source_execution_plan_sha256"),
        ("result.json", "source_result_sha256"),
        ("arrays.npz", "source_arrays_sha256"),
    ):
        if _sha256(execution / filename) != manifest[field]:
            raise ValueError(f"WP6 source changed after analysis: {filename}")
    if summary["status"] != "derived_executed_unreviewed":
        raise ValueError("WP6 analysis status is not derived_executed_unreviewed")
    if summary["review"] != {
        "reviewed": False,
        "reviewer": None,
        "decision": "pending",
    }:
        raise ValueError("WP6 analysis review state was mutated")
    if summary["checks"]["metric_regularization_used"]:
        raise ValueError("WP6 analysis claims metric regularization")
    if not summary["checks"]["declared_scalar_criteria_pass"]:
        raise ValueError("WP6 primary scalar criteria did not pass")
    if summary["checks"]["g6_gate_ready_for_review"]:
        raise ValueError("WP6 primary analysis hides its timestep-evidence gap")
    if summary["checks"]["h3_model_timestep_stability_status"] != (
        "not_demonstrated_by_primary_4_8_evidence"
    ):
        raise ValueError("WP6 primary timestep-evidence status is incorrect")

    artifact_count = 0
    for category in ("tables", "figures"):
        for name, expected in summary["artifacts"][category].items():
            path = analysis / category / name
            if _sha256(path) != expected:
                raise ValueError(f"WP6 analysis artifact hash mismatch: {path}")
            artifact_count += 1

    counts: dict[str, int] = {}
    for name in summary["artifacts"]["tables"]:
        counts[Path(name).stem] = len(_csv(analysis / "tables" / name))
    if counts != summary["row_counts"]:
        raise ValueError("WP6 analysis table row counts disagree with summary")
    if counts["matrix_checks"] != 5:
        raise ValueError("WP6 analysis matrix-check inventory is incomplete")
    if counts["h2_model_comparisons"] != 6:
        raise ValueError("WP6 analysis H2 model inventory is incomplete")
    if counts["h3_model_comparisons"] != 54:
        raise ValueError("WP6 analysis H3 model/case/grid inventory is incomplete")
    if counts["h3_model_timestep_stability_primary"] != 30:
        raise ValueError("WP6 primary model-difference stability inventory is incomplete")
    if counts["h3_quadrature_stability"] != 18:
        raise ValueError("WP6 analysis quadrature inventory is incomplete")

    print(f"analysis={analysis}")
    print(f"manifest_id={manifest_id}")
    print(f"artifacts_verified={artifact_count}")
    print(f"tables_verified={len(counts)}")
    print("status=authenticated_derived_executed_unreviewed")


if __name__ == "__main__":
    main()
