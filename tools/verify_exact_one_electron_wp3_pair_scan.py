#!/usr/bin/env python3
"""Authenticate completed members of a WP3 exact static pair scan."""

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
    parser.add_argument("execution_directory", type=Path)
    parser.add_argument("--expected-members", type=int)
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


def _verify_member(record: dict[str, Any]) -> tuple[int, int]:
    directory = Path(record["member_directory"])
    manifest_path = directory / "manifest.json"
    result_path = directory / "result.json"
    arrays_path = directory / "arrays.npz"
    for path in (manifest_path, result_path, arrays_path, directory / "completed.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = _json(manifest_path)
    result = _json(result_path)
    if manifest["status"] != "executed_unreviewed":
        raise ValueError(f"member manifest is not qualification evidence: {directory}")
    if result["status"] != "executed_unreviewed":
        raise ValueError(f"member result is not qualification evidence: {directory}")
    if result["review"] != {
        "reviewed": False,
        "reviewer": None,
        "decision": "pending",
    }:
        raise ValueError(f"member review state was mutated: {directory}")
    manifest_core = dict(manifest)
    manifest_id = manifest_core.pop("manifest_id")
    if canonical_sha256(manifest_core) != manifest_id:
        raise ValueError(f"manifest identity mismatch: {directory}")
    if result["manifest_id"] != manifest_id or record["manifest_id"] != manifest_id:
        raise ValueError(f"manifest identity propagation mismatch: {directory}")
    if _sha256(arrays_path) != result["artifact"]["sha256"]:
        raise ValueError(f"array file hash mismatch: {directory}")
    if _sha256(arrays_path) != record["arrays_sha256"]:
        raise ValueError(f"completed array hash mismatch: {directory}")
    if _sha256(result_path) != record["result_sha256"]:
        raise ValueError(f"completed result hash mismatch: {directory}")
    if _sha256(manifest_path) != record["manifest_sha256"]:
        raise ValueError(f"completed manifest hash mismatch: {directory}")

    cases = result["cases"]
    case_ids = result["case_ids_in_array_order"]
    if len(cases) != len(case_ids) or len(cases) != record["case_count"]:
        raise ValueError(f"case cardinality mismatch: {directory}")
    signs: dict[tuple[str, float], set[int]] = {}
    for case in cases:
        if case["sign"] == 0:
            continue
        key = (case["direction"], float(case["magnitude_au"]))
        signs.setdefault(key, set()).add(int(case["sign"]))
    if not signs or any(value != {-1, 1} for value in signs.values()):
        raise ValueError(f"both field signs were not executed: {directory}")
    if result["minimum_exact_metric_eigenvalue"] <= 0.0:
        raise ValueError(f"exact metric lost positivity: {directory}")
    if result["minimum_p0_metric_eigenvalue"] <= 0.0:
        raise ValueError(f"P0 pair metric lost positivity: {directory}")
    for crossing in result["threshold_crossings"]:
        if crossing["crossed"] and crossing["signal_to_numerical_floor"] < 4.0:
            raise ValueError(f"threshold below numerical floor: {directory}")

    semantic = result["array_semantic_sha256"]
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(semantic):
            raise ValueError(f"semantic array inventory mismatch: {directory}")
        for name in arrays.files:
            value = np.asarray(arrays[name])
            if not np.all(np.isfinite(value)):
                raise ValueError(f"non-finite array {name}: {directory}")
            if canonical_sha256(value) != semantic[name]:
                raise ValueError(f"semantic hash mismatch for {name}: {directory}")
        if arrays["field_vectors_au"].shape != (len(cases), 3):
            raise ValueError(f"field-vector shape mismatch: {directory}")
    return len(semantic), len(cases)


def main() -> None:
    arguments = _arguments()
    execution = arguments.execution_directory.resolve()
    index_path = execution / "execution_index.json"
    index = _json(index_path)
    records = index["completed_members"]
    if len(records) != index["completed_member_count"]:
        raise ValueError("execution index member count mismatch")
    if arguments.expected_members is not None and len(records) != arguments.expected_members:
        raise ValueError(
            f"expected {arguments.expected_members} members, found {len(records)}"
        )
    array_count = 0
    case_count = 0
    for record in records:
        member_arrays, member_cases = _verify_member(record)
        array_count += member_arrays
        case_count += member_cases
    print(f"execution={execution}")
    print(f"execution_index_sha256={_sha256(index_path)}")
    print(f"members_verified={len(records)}")
    print(f"arrays_verified={array_count}")
    print(f"field_cases_verified={case_count}")
    print("status=authenticated_executed_unreviewed")


if __name__ == "__main__":
    main()
