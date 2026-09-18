#!/usr/bin/env python3
"""Authenticate a WP4 uniform-electric E1 refinement audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from aion.config import canonical_sha256


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_directory", type=Path)
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
    audit = _arguments().audit_directory.resolve()
    manifest = _json(audit / "manifest.json")
    result = _json(audit / "result.json")
    completed = _json(audit / "completed.json")

    manifest_core = dict(manifest)
    manifest_id = manifest_core.pop("manifest_id", None)
    if manifest_id is None or canonical_sha256(manifest_core) != manifest_id:
        raise ValueError("E1 audit manifest identity mismatch")
    if result["manifest_id"] != manifest_id or completed["manifest_id"] != manifest_id:
        raise ValueError("E1 audit manifest identity was not propagated")
    if completed["status"] != "authenticated_executed_unreviewed":
        raise ValueError("E1 audit does not have authenticated executed status")

    for filename, field in (
        ("manifest.json", "manifest_sha256"),
        ("result.json", "result_sha256"),
        ("arrays.npz", "arrays_sha256"),
    ):
        if _sha256(audit / filename) != completed[field]:
            raise ValueError(f"E1 audit file hash mismatch: {filename}")

    source_execution = Path(manifest["source_execution_directory"])
    if _sha256(source_execution / "execution_index.json") != manifest[
        "source_execution_index_sha256"
    ]:
        raise ValueError("WP4 source execution changed after the E1 audit")

    semantic_hashes = result["array_semantic_sha256"]
    with np.load(audit / "arrays.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != set(semantic_hashes):
            raise ValueError("E1 audit array inventory mismatch")
        for name in arrays.files:
            value = np.asarray(arrays[name])
            if not np.all(np.isfinite(value)):
                raise ValueError(f"non-finite E1 audit array: {name}")
            if canonical_sha256(value) != semantic_hashes[name]:
                raise ValueError(f"E1 audit semantic array hash mismatch: {name}")

    members = result["members"]
    if result["member_count"] != len(members):
        raise ValueError("E1 audit member count mismatch")
    expected_array_count = len(members)
    if len(semantic_hashes) != expected_array_count:
        raise ValueError("E1 audit does not contain one level-5 tensor per member")
    for row in members:
        required_floor = max(
            float(row["level4_absolute_residual"]),
            float(row["level5_absolute_residual"]),
            float(row["level5_vs_level4_absolute"]),
        )
        floor = float(row["refined_working_floor"])
        if floor < required_floor:
            raise ValueError("E1 audit working floor omits a measured residual")
        if float(row["level5_residual_to_refined_floor"]) > 1.0 + 1.0e-14:
            raise ValueError("E1 audit residual exceeds its declared working floor")

    print(f"audit={audit}")
    print(f"members_verified={len(members)}")
    print(f"arrays_verified={len(semantic_hashes)}")
    print(
        "maximum_level5_relative_residual="
        f"{float(result['maximum_level5_relative_residual']):.6e}"
    )
    print(
        "maximum_level5_residual_to_refined_floor="
        f"{float(result['maximum_level5_residual_to_refined_floor']):.6e}"
    )
    print("status=authenticated_executed_unreviewed")


if __name__ == "__main__":
    main()
