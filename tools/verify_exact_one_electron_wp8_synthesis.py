#!/usr/bin/env python3
"""Verify an authenticated WP8 exact one-electron synthesis directory."""

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
    parser.add_argument("synthesis_directory", type=Path)
    return parser.parse_args().synthesis_directory.resolve()


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


def _row_count(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def main() -> None:
    root = _arguments()
    manifest = _json(root / "manifest.json")
    manifest_core = dict(manifest)
    manifest_id = str(manifest_core.pop("manifest_id"))
    if canonical_sha256(manifest_core) != manifest_id:
        raise ValueError("WP8 synthesis manifest identity mismatch")
    for source, expected in manifest["source_sha256"].items():
        if _sha256(Path(source)) != expected:
            raise ValueError(f"WP8 source changed after synthesis: {source}")
    summary = _json(root / "summary.json")
    completed = _json(root / "completed.json")
    if summary["manifest_id"] != manifest_id or completed["manifest_id"] != manifest_id:
        raise ValueError("WP8 manifest identity was not propagated")
    for name, field in (
        ("manifest.json", "manifest_sha256"),
        ("summary.json", "summary_sha256"),
        ("synthesis.md", "synthesis_sha256"),
    ):
        if _sha256(root / name) != completed[field]:
            raise ValueError(f"WP8 synthesis hash mismatch: {name}")
    for name, expected in summary["artifacts"]["tables"].items():
        path = root / "tables" / name
        if _sha256(path) != expected:
            raise ValueError(f"WP8 table hash mismatch: {name}")
        if _row_count(path) != summary["row_counts"][Path(name).stem]:
            raise ValueError(f"WP8 row count mismatch: {name}")
    if summary["accepted_gate_inputs"] != ["G3", "G4", "G5", "G6", "G7"]:
        raise ValueError("WP8 synthesis does not use the complete accepted evidence chain")
    if summary["branch_recommendation"]["decision_status"] != ("recommended_pending_user_review"):
        raise ValueError("WP8 branch recommendation has an invalid review state")
    if summary["review"] != {
        "reviewed": False,
        "reviewer": None,
        "decision": "pending",
    }:
        raise ValueError("WP8 synthesis review state was mutated")
    print(f"synthesis={root}")
    print(f"manifest_id={manifest_id}")
    print(f"tables_verified={len(summary['artifacts']['tables'])}")
    print("status=authenticated_derived_executed_unreviewed")


if __name__ == "__main__":
    main()
