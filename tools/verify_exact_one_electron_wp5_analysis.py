#!/usr/bin/env python3
"""Authenticate a derived WP5 multicentre analysis package."""

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


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    execution = _arguments()
    analysis = execution / "analysis"
    completed = _json(analysis / "completed.json")
    manifest = _json(analysis / "manifest.json")
    core = dict(manifest)
    manifest_id = core.pop("manifest_id")
    if canonical_sha256(core) != manifest_id:
        raise ValueError("WP5 analysis manifest identity mismatch")
    summary = _json(analysis / "summary.json")
    if completed["manifest_id"] != manifest_id or summary["manifest_id"] != manifest_id:
        raise ValueError("WP5 analysis identity was not propagated")
    for filename, field in (
        ("manifest.json", "manifest_sha256"),
        ("summary.json", "summary_sha256"),
        ("report.md", "report_sha256"),
    ):
        if _sha256(analysis / filename) != completed[field]:
            raise ValueError(f"WP5 analysis hash mismatch: {filename}")
    if manifest["execution_index_sha256"] != _sha256(
        execution / "execution_index.json"
    ):
        raise ValueError("WP5 source execution changed after analysis")
    if summary["status"] != "derived_executed_unreviewed":
        raise ValueError("WP5 analysis status is not derived_executed_unreviewed")
    if summary["review"] != {
        "reviewed": False,
        "reviewer": None,
        "decision": "pending",
    }:
        raise ValueError("WP5 analysis review state was mutated")
    if not summary["checks"]["g5_numerical_criteria_pass"]:
        raise ValueError("WP5 numerical gate criteria did not pass")
    if summary["checks"]["metric_regularization_used"]:
        raise ValueError("WP5 analysis claims metric regularization")

    artifact_count = 0
    for category in ("tables", "figures"):
        for name, expected in summary["artifacts"][category].items():
            path = analysis / category / name
            if _sha256(path) != expected:
                raise ValueError(f"WP5 analysis artifact hash mismatch: {path}")
            artifact_count += 1

    metric = _csv(analysis / "tables/metric_domain.csv")
    selected = _csv(analysis / "tables/selected_dynamical_cases.csv")
    spectral = _csv(analysis / "tables/generalized_spectral_comparisons.csv")
    pair = _csv(analysis / "tables/pair_restrictions.csv")
    structural = _csv(analysis / "tables/structural_checks.csv")
    expected_counts = summary["row_counts"]
    actual_counts = {
        "metric_domain": len(metric),
        "selected_dynamical_cases": len(selected),
        "generalized_spectral_comparisons": len(spectral),
        "pair_restrictions": len(pair),
        "structural_checks": len(structural),
    }
    if actual_counts != expected_counts:
        raise ValueError("WP5 analysis table row counts disagree with summary")
    if len(selected) != 12 or len(structural) != 4:
        raise ValueError("WP5 analysis does not cover four members and three fields")
    if {row["label"] for row in selected} != {
        "sub_threshold",
        "near_threshold",
        "above_threshold",
    }:
        raise ValueError("WP5 selected-field labels are incomplete")
    if sum(row["selection_role"] == "wp6_selected" for row in selected) != 6:
        raise ValueError("WP5 did not select three cc-pVDZ cases per geometry for WP6")

    print(f"analysis={analysis}")
    print(f"manifest_id={manifest_id}")
    print(f"artifacts_verified={artifact_count}")
    print(f"metric_rows_verified={len(metric)}")
    print(f"selected_rows_verified={len(selected)}")
    print(f"spectral_rows_verified={len(spectral)}")
    print(f"pair_rows_verified={len(pair)}")
    print("status=authenticated_derived_executed_unreviewed")


if __name__ == "__main__":
    main()
