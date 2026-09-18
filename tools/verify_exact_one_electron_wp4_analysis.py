#!/usr/bin/env python3
"""Authenticate a derived WP4 first-order analysis package."""

from __future__ import annotations

import csv
import hashlib
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from aion.config import canonical_sha256


def _arguments() -> Path:
    parser = ArgumentParser(description=__doc__)
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
        raise ValueError("WP4 analysis manifest identity mismatch")
    summary = _json(analysis / "summary.json")
    if completed["manifest_id"] != manifest_id or summary["manifest_id"] != manifest_id:
        raise ValueError("WP4 analysis identity was not propagated")
    if completed["manifest_sha256"] != _sha256(analysis / "manifest.json"):
        raise ValueError("WP4 analysis manifest hash mismatch")
    if completed["summary_sha256"] != _sha256(analysis / "summary.json"):
        raise ValueError("WP4 analysis summary hash mismatch")
    if completed["report_sha256"] != _sha256(analysis / "report.md"):
        raise ValueError("WP4 analysis report hash mismatch")
    if manifest["execution_index_sha256"] != _sha256(execution / "execution_index.json"):
        raise ValueError("WP4 source execution changed after analysis")
    if manifest["e1_refinement_audit_result_sha256"] != _sha256(
        execution / "e1_refinement_audit/result.json"
    ):
        raise ValueError("WP4 E1 refinement audit changed after analysis")
    if summary["status"] != "derived_executed_unreviewed":
        raise ValueError("WP4 analysis status is not derived_executed_unreviewed")
    if summary["review"] != {
        "reviewed": False,
        "reviewer": None,
        "decision": "pending",
    }:
        raise ValueError("WP4 analysis review state was mutated")
    if not summary["checks"]["g4_numerical_criteria_pass"]:
        raise ValueError("WP4 numerical gate criteria did not pass")
    if summary["checks"]["derivative_resolved_fail"] != 0:
        raise ValueError("a resolved magnetic derivative fit failed")
    if summary["checks"]["remainder_resolved_fail"] != 0:
        raise ValueError("a resolved model-remainder fit failed")

    artifact_count = 0
    for category in ("tables", "figures"):
        for name, expected in summary["artifacts"][category].items():
            path = analysis / category / name
            if _sha256(path) != expected:
                raise ValueError(f"WP4 analysis artifact hash mismatch: {path}")
            artifact_count += 1

    derivative_rows = _csv(analysis / "tables/magnetic_derivative_fits.csv")
    remainder_rows = _csv(analysis / "tables/remainder_fits.csv")
    validity_rows = _csv(analysis / "tables/model_validity.csv")
    channel_rows = _csv(analysis / "tables/mechanical_channel_classification.csv")
    if {float(row["bond_scale"]) for row in derivative_rows} != {0.8, 1.0, 1.25}:
        raise ValueError("WP4 derivative table omits a bond scale")
    if any(row["status"] == "resolved_fail" for row in derivative_rows):
        raise ValueError("WP4 derivative CSV contains a resolved failure")
    if any(row["status"] == "resolved_fail" for row in remainder_rows):
        raise ValueError("WP4 remainder CSV contains a resolved failure")

    print(f"analysis={analysis}")
    print(f"manifest_id={manifest_id}")
    print(f"artifacts_verified={artifact_count}")
    print(f"derivative_rows_verified={len(derivative_rows)}")
    print(f"remainder_rows_verified={len(remainder_rows)}")
    print(f"validity_rows_verified={len(validity_rows)}")
    print(f"channel_rows_verified={len(channel_rows)}")
    print("status=authenticated_derived_executed_unreviewed")


if __name__ == "__main__":
    main()
