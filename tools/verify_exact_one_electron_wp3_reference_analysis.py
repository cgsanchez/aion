#!/usr/bin/env python3
"""Authenticate the derived WP3 reference-distance analysis package."""

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
    parser.add_argument("execution_directory", type=Path)
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


def _verify_identity(path: Path) -> tuple[dict[str, Any], str]:
    manifest = _json(path)
    core = dict(manifest)
    manifest_id = core.pop("manifest_id")
    if canonical_sha256(core) != manifest_id:
        raise ValueError(f"manifest identity mismatch: {path}")
    return manifest, manifest_id


def main() -> None:
    execution = _arguments().execution_directory.resolve()
    analysis = execution / "analysis"
    audit = execution / "reference_audit"
    completed = _json(analysis / "completed.json")
    manifest, manifest_id = _verify_identity(analysis / "manifest.json")
    summary = _json(analysis / "summary.json")
    if completed["manifest_id"] != manifest_id or summary["manifest_id"] != manifest_id:
        raise ValueError("analysis manifest identity was not propagated")
    if completed["manifest_sha256"] != _sha256(analysis / "manifest.json"):
        raise ValueError("analysis manifest hash mismatch")
    if completed["summary_sha256"] != _sha256(analysis / "summary.json"):
        raise ValueError("analysis summary hash mismatch")
    if completed["report_sha256"] != _sha256(analysis / "report.md"):
        raise ValueError("analysis report hash mismatch")
    if manifest["source_execution_index_sha256"] != _sha256(
        execution / "execution_index.json"
    ):
        raise ValueError("source execution index changed after analysis")
    if manifest["source_audit_result_sha256"] != _sha256(audit / "result.json"):
        raise ValueError("source reference audit changed after analysis")
    if summary["status"] != "derived_executed_unreviewed":
        raise ValueError("analysis status is not derived_executed_unreviewed")
    if summary["review"] != {
        "reviewed": False,
        "reviewer": None,
        "decision": "pending",
    }:
        raise ValueError("analysis review state was mutated")
    if not summary["checks"]["reference_checkpoint_pass"]:
        raise ValueError("reference checkpoint did not pass")
    if summary["checks"]["failed_parity_fit_count"] != 0:
        raise ValueError("a resolved parity fit failed")

    artifact_count = 0
    for category in ("tables", "figures"):
        directory = analysis / category
        for name, expected in summary["artifacts"][category].items():
            path = directory / name
            if _sha256(path) != expected:
                raise ValueError(f"analysis artifact hash mismatch: {path}")
            artifact_count += 1

    thresholds_path = analysis / "tables/thresholds.csv"
    with thresholds_path.open(encoding="utf-8", newline="") as stream:
        thresholds = list(csv.DictReader(stream))
    crossed = [row for row in thresholds if row["crossed"] == "True"]
    minimum_ratio = min(float(row["signal_to_refined_floor"]) for row in crossed)
    if minimum_ratio != summary["checks"]["minimum_crossing_signal_to_floor"]:
        raise ValueError("threshold minimum does not reproduce from CSV")
    if minimum_ratio < manifest["rules"]["threshold_floor_multiplier"]:
        raise ValueError("a retained threshold is below the declared floor multiplier")

    print(f"analysis={analysis}")
    print(f"manifest_id={manifest_id}")
    print(f"artifacts_verified={artifact_count}")
    print(f"threshold_rows_verified={len(thresholds)}")
    print(f"minimum_crossing_signal_to_floor={minimum_ratio:.6g}")
    print("status=authenticated_derived_executed_unreviewed")


if __name__ == "__main__":
    main()
