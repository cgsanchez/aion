#!/usr/bin/env python3
"""Authenticate the combined WP3 gate-G3 analysis package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from aion.config import canonical_sha256


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distance_execution_directory", type=Path)
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


def main() -> None:
    distance_execution = _arguments().distance_execution_directory.resolve()
    analysis = distance_execution / "g3_analysis"
    completed = _json(analysis / "completed.json")
    manifest = _json(analysis / "manifest.json")
    manifest_core = dict(manifest)
    manifest_id = manifest_core.pop("manifest_id")
    if canonical_sha256(manifest_core) != manifest_id:
        raise ValueError("G3 manifest identity mismatch")
    summary = _json(analysis / "summary.json")
    if completed["manifest_id"] != manifest_id or summary["manifest_id"] != manifest_id:
        raise ValueError("G3 manifest identity was not propagated")
    if completed["manifest_sha256"] != _sha256(analysis / "manifest.json"):
        raise ValueError("G3 manifest hash mismatch")
    if completed["summary_sha256"] != _sha256(analysis / "summary.json"):
        raise ValueError("G3 summary hash mismatch")
    if completed["report_sha256"] != _sha256(analysis / "report.md"):
        raise ValueError("G3 report hash mismatch")

    reference_execution = Path(manifest["reference_execution_directory"])
    if manifest["reference_execution_index_sha256"] != _sha256(
        reference_execution / "execution_index.json"
    ):
        raise ValueError("reference execution index changed after G3 analysis")
    if manifest["distance_execution_index_sha256"] != _sha256(
        distance_execution / "execution_index.json"
    ):
        raise ValueError("distance execution index changed after G3 analysis")
    if manifest["reference_audit_result_sha256"] != _sha256(
        reference_execution / "reference_audit/result.json"
    ):
        raise ValueError("reference audit changed after G3 analysis")
    if manifest["distance_audit_result_sha256"] != _sha256(
        distance_execution / "distance_audit/result.json"
    ):
        raise ValueError("distance audit changed after G3 analysis")
    if summary["status"] != "derived_executed_unreviewed":
        raise ValueError("G3 analysis status is not derived_executed_unreviewed")
    if summary["review"] != {
        "reviewed": False,
        "reviewer": None,
        "decision": "pending",
    }:
        raise ValueError("G3 review state was mutated")
    if not summary["checks"]["g3_numerical_criteria_pass"]:
        raise ValueError("G3 numerical criteria did not pass")
    if summary["checks"]["failed_parity_fit_count"] != 0:
        raise ValueError("a resolved G3 parity fit failed")
    if summary["coverage"] != {
        "members": 36,
        "field_cases": summary["coverage"]["field_cases"],
        "systems": ["hh", "oh", "n2", "co"],
        "bases": ["sto-3g", "cc-pvdz", "aug-cc-pvdz"],
        "bond_scales": [0.8, 1.0, 1.25],
    }:
        raise ValueError("G3 coverage is not the declared 36-member campaign")

    artifact_count = 0
    for category in ("tables", "figures"):
        for name, expected in summary["artifacts"][category].items():
            path = analysis / category / name
            if _sha256(path) != expected:
                raise ValueError(f"G3 artifact hash mismatch: {path}")
            artifact_count += 1

    thresholds_path = analysis / "tables/thresholds.csv"
    with thresholds_path.open(encoding="utf-8", newline="") as stream:
        thresholds = list(csv.DictReader(stream))
    scales = {float(row["bond_scale"]) for row in thresholds}
    if scales != {0.8, 1.0, 1.25}:
        raise ValueError("G3 threshold table does not contain every bond scale")
    crossed = [row for row in thresholds if row["crossed"] == "True"]
    minimum_ratio = min(float(row["signal_to_refined_floor"]) for row in crossed)
    if minimum_ratio != summary["checks"]["minimum_crossing_signal_to_floor"]:
        raise ValueError("G3 threshold minimum does not reproduce from CSV")
    if minimum_ratio < manifest["rules"]["threshold_floor_multiplier"]:
        raise ValueError("a G3 threshold is below the declared floor multiplier")

    print(f"analysis={analysis}")
    print(f"manifest_id={manifest_id}")
    print(f"artifacts_verified={artifact_count}")
    print(f"threshold_rows_verified={len(thresholds)}")
    print(f"minimum_crossing_signal_to_floor={minimum_ratio:.6g}")
    print("status=authenticated_derived_executed_unreviewed")


if __name__ == "__main__":
    main()
