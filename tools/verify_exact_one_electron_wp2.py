#!/usr/bin/env python3
"""Authenticate a completed exact one-electron WP2 execution directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from aion.config import canonical_sha256

_REPOSITORY = Path(__file__).resolve().parents[1]
_DEFAULT_EXECUTION = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "exact_one_electron_qualification/wp2_20260916T192437Z_3fd2a83b77d1"
)
_REGRESSION_FIXTURE = (
    _REPOSITORY
    / "tests/fixtures/exact_one_electron/wp2_hh_oblique_reference.json"
)
_FIXTURE_TO_RESULT = {
    "lower_exact_overlap": "hh/oblique_plus/lower_exact/overlap",
    "lower_exact_kinetic": "hh/oblique_plus/lower_exact/kinetic",
    "lower_exact_nuclear_attraction": (
        "hh/oblique_plus/lower_exact/nuclear_attraction"
    ),
    "lower_exact_mechanical": "hh/oblique_plus/lower_exact/mechanical",
    "kinetic_T_pp_F": "hh/oblique_plus/kinetic_sectors/T_pp_F",
    "kinetic_T_pC_F": "hh/oblique_plus/kinetic_sectors/T_pC_F",
    "kinetic_T_Cp_F": "hh/oblique_plus/kinetic_sectors/T_Cp_F",
    "kinetic_T_CC_F": "hh/oblique_plus/kinetic_sectors/T_CC_F",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution_directory", nargs="?", type=Path, default=_DEFAULT_EXECUTION)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _require_keys(value: dict[str, Any], names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if name not in value]
    if missing:
        raise ValueError(f"{label} is missing keys: {missing}")


def main() -> None:
    execution = _arguments().execution_directory.resolve()
    paths = {
        "manifest.json": execution / "manifest.json",
        "result.json": execution / "result.json",
        "arrays.npz": execution / "arrays.npz",
        "wp2_report.md": execution / "wp2_report.md",
        "execution_index.json": execution / "execution_index.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = _load_json(paths["manifest.json"])
    result = _load_json(paths["result.json"])
    index = _load_json(paths["execution_index.json"])
    fixture = _load_json(_REGRESSION_FIXTURE)
    _require_keys(
        manifest,
        (
            "schema",
            "version",
            "status",
            "manifest_id",
            "execution",
            "code",
            "environment",
            "fixture",
            "quadrature",
            "source",
            "conventions",
            "tolerances",
            "input_hashes",
        ),
        "manifest",
    )
    _require_keys(
        result,
        (
            "schema",
            "version",
            "status",
            "manifest_id",
            "matrices",
            "residuals",
            "artifact",
            "review",
        ),
        "result",
    )
    if manifest["schema"] != "aion.exact-one-electron-run-manifest":
        raise ValueError("unexpected manifest schema")
    if result["schema"] != "aion.exact-one-electron-matrix-result":
        raise ValueError("unexpected result schema")
    if manifest["status"] != "executed_unreviewed":
        raise ValueError("raw manifest status was mutated")
    if result["status"] != "executed_unreviewed":
        raise ValueError("raw result status was mutated")
    if result["review"] != {
        "reviewed": False,
        "reviewer": None,
        "decision": "pending",
    }:
        raise ValueError("raw result review state was mutated")

    manifest_core = dict(manifest)
    manifest_id = manifest_core.pop("manifest_id")
    if canonical_sha256(manifest_core) != manifest_id:
        raise ValueError("manifest identity does not reproduce")
    if result["manifest_id"] != manifest_id or index["manifest_id"] != manifest_id:
        raise ValueError("manifest identity is not propagated consistently")

    for name, expected in index["artifacts"].items():
        if _sha256(execution / name) != expected:
            raise ValueError(f"file SHA-256 mismatch: {name}")
    if result["artifact"]["sha256"] != _sha256(paths["arrays.npz"]):
        raise ValueError("result artifact hash does not match arrays.npz")

    matrix_records = {record["name"]: record for record in result["matrices"]}
    if len(matrix_records) != len(result["matrices"]):
        raise ValueError("matrix result contains duplicate semantic names")
    with np.load(paths["arrays.npz"], allow_pickle=False) as arrays:
        for name, record in matrix_records.items():
            key = name.replace("/", "__")
            if key not in arrays:
                raise ValueError(f"matrix array is missing: {name}")
            value = np.asarray(arrays[key])
            if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
                raise ValueError(f"matrix shape or dtype mismatch: {name}")
            if canonical_sha256(value) != record["data_sha256"]:
                raise ValueError(f"semantic array hash mismatch: {name}")

    failed_residuals = [
        record
        for record in result["residuals"]
        if record["tolerance"] is not None
        and float(record["value"]) > float(record["tolerance"])
    ]
    if failed_residuals:
        raise ValueError(f"{len(failed_residuals)} residuals exceed tolerance")
    if not index["gate_checks"] or not all(index["gate_checks"].values()):
        raise ValueError("execution index contains a failed gate check")

    source = fixture["source_execution"]
    if source["manifest_id"] != manifest_id:
        raise ValueError("regression fixture manifest identity mismatch")
    if source["execution_index_sha256"] != _sha256(paths["execution_index.json"]):
        raise ValueError("regression fixture execution-index hash mismatch")
    if source["arrays_sha256"] != _sha256(paths["arrays.npz"]):
        raise ValueError("regression fixture arrays hash mismatch")
    if source["result_sha256"] != _sha256(paths["result.json"]):
        raise ValueError("regression fixture result hash mismatch")
    for fixture_name, result_name in _FIXTURE_TO_RESULT.items():
        if (
            fixture["semantic_array_sha256"][fixture_name]
            != matrix_records[result_name]["data_sha256"]
        ):
            raise ValueError(f"regression fixture semantic hash mismatch: {fixture_name}")

    print(f"execution={execution}")
    print(f"manifest_id={manifest_id}")
    print(f"matrices_verified={len(matrix_records)}")
    print(f"residuals_verified={len(result['residuals'])}")
    print("status=authenticated_executed_unreviewed")


if __name__ == "__main__":
    main()
