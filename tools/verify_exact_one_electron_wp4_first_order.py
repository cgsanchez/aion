#!/usr/bin/env python3
"""Authenticate a WP4 first-order hierarchy execution."""

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


def _identity(path: Path) -> tuple[dict[str, Any], str]:
    value = _json(path)
    core = dict(value)
    identity = core.pop("manifest_id", core.pop("plan_id", None))
    if identity is None or canonical_sha256(core) != identity:
        raise ValueError(f"canonical identity mismatch: {path}")
    return value, str(identity)


def main() -> None:
    execution = _arguments().execution_directory.resolve()
    _plan, plan_id = _identity(execution / "execution_plan.json")
    index = _json(execution / "execution_index.json")
    if index["plan_id"] != plan_id:
        raise ValueError("WP4 execution index does not reference its plan")
    if index["status"] != "authenticated_executed_unreviewed":
        raise ValueError("WP4 execution is not complete and authenticated")
    if index["completed_member_count"] != index["requested_member_count"]:
        raise ValueError("WP4 execution member count is incomplete")

    array_count = 0
    maximum_adjoint = 0.0
    maximum_sector = 0.0
    maximum_e1_adjoint = 0.0
    maximum_e1_parent_residual = 0.0
    for record in index["completed_members"]:
        directory = Path(record["member_directory"])
        completed = _json(directory / "completed.json")
        manifest, manifest_id = _identity(directory / "manifest.json")
        result = _json(directory / "result.json")
        if completed["manifest_id"] != manifest_id or result["manifest_id"] != manifest_id:
            raise ValueError(f"WP4 member identity was not propagated: {directory}")
        for filename, field in (
            ("manifest.json", "manifest_sha256"),
            ("result.json", "result_sha256"),
            ("arrays.npz", "arrays_sha256"),
        ):
            value = _sha256(directory / filename)
            if value != completed[field] or value != record[field]:
                raise ValueError(f"WP4 member hash mismatch: {directory / filename}")
        source = manifest["source_wp3"]
        source_directory = Path(source["directory"])
        for filename, field in (
            ("manifest.json", "manifest_sha256"),
            ("result.json", "result_sha256"),
            ("arrays.npz", "arrays_sha256"),
        ):
            if _sha256(source_directory / filename) != source[field]:
                raise ValueError(f"WP3 source changed after WP4: {source_directory / filename}")
        with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != set(result["array_semantic_sha256"]):
                raise ValueError(f"WP4 array inventory mismatch: {directory}")
            for name in arrays.files:
                value = np.asarray(arrays[name])
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"non-finite WP4 array: {directory}:{name}")
                if canonical_sha256(value) != result["array_semantic_sha256"][name]:
                    raise ValueError(f"WP4 semantic array hash mismatch: {directory}:{name}")
                array_count += 1
        maximum_adjoint = max(
            maximum_adjoint,
            *(float(row["relative_residual"]) for row in result["coefficient_adjoint_checks"]),
        )
        maximum_sector = max(
            maximum_sector,
            *(
                abs(float(value))
                for row in result["sector_increment_identities"]
                for name, value in row.items()
                if name.endswith("residual")
            ),
        )
        maximum_e1_adjoint = max(
            maximum_e1_adjoint,
            float(result["electric_e1"]["connection_adjoint_residual"]),
        )
        maximum_e1_parent_residual = max(
            maximum_e1_parent_residual,
            *(
                float(row["absolute_error_vs_quadrature"])
                for row in result["electric_e1"]["finite_source_differences"]
            ),
        )

    if maximum_adjoint > 1.0e-10:
        raise ValueError("WP4 magnetic coefficient adjoint check failed")
    if maximum_sector > 1.0e-12:
        raise ValueError("WP4 named-model sector identity failed")
    if maximum_e1_adjoint > 1.0e-10:
        raise ValueError("WP4 E1 connection adjoint check failed")
    if maximum_e1_parent_residual > 1.0e-10:
        raise ValueError("WP4 direct electric parent finite difference failed")

    print(f"execution={execution}")
    print(f"members_verified={index['completed_member_count']}")
    print(f"arrays_verified={array_count}")
    print(f"maximum_coefficient_adjoint_residual={maximum_adjoint:.6e}")
    print(f"maximum_sector_identity_residual={maximum_sector:.6e}")
    print(f"maximum_e1_parent_residual={maximum_e1_parent_residual:.6e}")
    print("status=authenticated_executed_unreviewed")


if __name__ == "__main__":
    main()
