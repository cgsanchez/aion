#!/usr/bin/env python3
"""Authenticate and close a completed Chapter 13 NQ7 raw campaign.

This tool is intentionally separate from the numerical campaign driver.  The
driver's original synthesis step predates the immutable-review convention and
therefore rejected accepted NQ4/NQ6 campaigns whose raw completion markers
remain ``executed_unreviewed``.  Acceptance lives in separate review records;
raw campaign markers are never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from aion.electronic_structure import DependencyVersions

_CAMPAIGN_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "chapter13_wilson_adiabatic_qualification"
)
_NQ4_ROOT = _CAMPAIGN_ROOT / "nq4_stationary_20260921T232232Z_6197d6828bb3"
_NQ6_ROOT = _CAMPAIGN_ROOT / "nq6_dynamics_20260922T205959Z_a197fbc55b62"
_FIELDS = (0.0, 0.001, 0.003, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
_DOMAIN_FIELDS = (0.0, 0.03, 0.06, 0.1, 0.2, 0.4, 0.8, 1.2)
_DOMAIN_BASES = ("sto-3g", "cc-pvdz", "aug-cc-pvdz")
_LEVELS = ("p0", "e1", "strict_c1", "density_resummed_c1")
_BRANCHES = ("hartree", "kohn_sham_lda")
_DYNAMIC_CASES = (
    "electric_field_free",
    "electric_static_b",
    "magnetic_induction",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _field_tag(value: float) -> str:
    return f"b{value:.3f}".replace(".", "p")


def _expected_ids() -> dict[str, set[str]]:
    stationary = {
        f"{_field_tag(field)}_{branch}_{level}"
        for field in _FIELDS
        for branch in _BRANCHES
        for level in _LEVELS
    }
    domain = {
        f"{basis}_{_field_tag(field)}_{level}"
        for basis in _DOMAIN_BASES
        for field in _DOMAIN_FIELDS
        for level in _LEVELS
    }
    dynamics: set[str] = set()
    for case in _DYNAMIC_CASES:
        for branch in _BRANCHES:
            for level in _LEVELS:
                intervals = (
                    (16, 32)
                    if level
                    in {
                        "strict_c1",
                        "density_resummed_c1",
                    }
                    else (32,)
                )
                dynamics.update(f"{case}_{branch}_{level}_n{count}" for count in intervals)
    return {"stationary": stationary, "domain": domain, "dynamics": dynamics}


def _authenticate_parent(
    repo: Path,
    gate: str,
    root: Path,
    review_name: str,
) -> dict[str, Any]:
    completed_path = root / "completed.json"
    provenance_path = root / "provenance.json"
    result_path = root / "result.json"
    completed = _load_json(completed_path)
    for label, path in (
        ("result", result_path),
        ("provenance", provenance_path),
    ):
        expected = str(completed[f"{label}_sha256"])
        if _sha256(path) != expected:
            raise RuntimeError(f"{gate} raw {label} hash mismatch: {path}")
    provenance = _load_json(provenance_path)
    for relative, expected in provenance["artifacts_sha256"].items():
        path = root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"{gate} raw artifact hash mismatch: {path}")

    review_path = repo / "docs" / "reviews" / review_name
    review = _load_json(review_path)
    if review.get("gate") != gate or review.get("decision") != "accepted":
        raise RuntimeError(f"{gate} review is not an accepted decision")
    evidence = dict(review["evidence"])
    if Path(evidence["execution_root"]).resolve() != root.resolve():
        raise RuntimeError(f"{gate} review points to another execution root")
    actual = {
        "result_sha256": _sha256(result_path),
        "provenance_sha256": _sha256(provenance_path),
        "completed_sha256": _sha256(completed_path),
    }
    for name, value in actual.items():
        if evidence[name] != value:
            raise RuntimeError(f"{gate} review {name} does not match raw evidence")
    analysis_path = root / str(evidence["accepted_analysis"])
    if _sha256(analysis_path) != evidence["analysis_summary_sha256"]:
        raise RuntimeError(f"{gate} accepted analysis hash mismatch")
    return {
        "root": str(root),
        "raw_status": completed.get("status"),
        **actual,
        "review_record": str(review_path.relative_to(repo)),
        "review_record_sha256": _sha256(review_path),
        "review_decision": "accepted",
        "accepted_analysis": str(analysis_path),
        "accepted_analysis_sha256": _sha256(analysis_path),
    }


def _validate_npz(path: Path, required: set[str]) -> None:
    with np.load(path) as arrays:
        if not required.issubset(arrays.files):
            missing = sorted(required.difference(arrays.files))
            raise RuntimeError(f"missing arrays in {path}: {missing}")
        for name in arrays.files:
            value = np.asarray(arrays[name])
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"non-finite array {name} in {path}")


def _validate_checkpoints(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = _expected_ids()
    required_arrays = {
        "stationary": {
            "coefficients",
            "coefficient_density",
            "mixed_density",
            "lower_matrix",
            "density_grid",
            "matched_lower_matrix",
            "matched_density_grid",
        },
        "domain": {"metric", "density_zero", "density_first"},
        "dynamics": {
            "times_au",
            "final_mixed_density",
            "energies_au",
            "source_powers_au",
            "density_zero_minima",
            "density_assembled_minima",
            "metric_minimum_eigenvalues",
            "metric_condition_numbers",
        },
    }
    records: dict[str, list[dict[str, Any]]] = {}
    validation: dict[str, Any] = {}
    for phase, expected_ids in expected.items():
        directory = output / "checkpoints" / phase
        json_paths = sorted(directory.glob("*.json"))
        npz_paths = sorted(directory.glob("*.npz"))
        found_ids = {path.stem for path in json_paths}
        if found_ids != expected_ids:
            missing = sorted(expected_ids.difference(found_ids))
            extra = sorted(found_ids.difference(expected_ids))
            raise RuntimeError(f"{phase} checkpoint IDs differ; missing={missing}, extra={extra}")
        phase_records: list[dict[str, Any]] = []
        complete_ids: set[str] = set()
        failed_ids: set[str] = set()
        for path in json_paths:
            record = _load_json(path)
            if record.get("run_id") != path.stem:
                raise RuntimeError(f"run_id/path mismatch: {path}")
            status = record.get("status")
            if status == "complete":
                complete_ids.add(path.stem)
                npz_path = path.with_suffix(".npz")
                if not npz_path.is_file():
                    raise RuntimeError(f"complete checkpoint lacks arrays: {path}")
                _validate_npz(npz_path, required_arrays[phase])
            elif status == "failed":
                failed_ids.add(path.stem)
                if path.with_suffix(".npz").exists():
                    raise RuntimeError(f"failed checkpoint has arrays: {path}")
                if not record.get("exception_type") or not record.get("traceback"):
                    raise RuntimeError(f"failed checkpoint lacks diagnostics: {path}")
            else:
                raise RuntimeError(f"unexpected checkpoint status in {path}")
            phase_records.append(record)
        if {path.stem for path in npz_paths} != complete_ids:
            raise RuntimeError(f"{phase} array/checkpoint pairing is inconsistent")
        records[phase] = phase_records
        validation[phase] = {
            "expected": len(expected_ids),
            "records": len(phase_records),
            "complete": len(complete_ids),
            "failed": len(failed_ids),
            "array_archives": len(npz_paths),
        }
    if validation["stationary"]["failed"] or validation["dynamics"]["failed"]:
        raise RuntimeError("stationary or dynamic NQ7 checkpoints contain failures")
    return records, validation


def _validate_identity(output: Path, repo: Path) -> dict[str, Any]:
    identity_path = output / "campaign_identity.json"
    identity = _load_json(identity_path)
    for relative, expected in identity["source_sha256"].items():
        path = repo / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"implementation source changed since execution: {path}")
    implementation_commit = str(identity["implementation_commit"])
    _git(repo, "cat-file", "-e", f"{implementation_commit}^{{commit}}")
    return {
        **identity,
        "identity_sha256": _sha256(identity_path),
    }


def _raw_artifacts(output: Path) -> list[Path]:
    excluded_names = {"result.json", "provenance.json", "completed.json"}
    return [
        path
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name not in excluded_names
        and not any(part.startswith("analysis_") for part in path.relative_to(output).parts)
    ]


def _verify_closed(output: Path) -> None:
    result_path = output / "result.json"
    provenance_path = output / "provenance.json"
    completed_path = output / "completed.json"
    completed = _load_json(completed_path)
    if completed.get("status") != "complete_with_visible_domain_failures":
        raise RuntimeError("unexpected NQ7 completion status")
    if completed["result_sha256"] != _sha256(result_path):
        raise RuntimeError("NQ7 result hash mismatch")
    if completed["provenance_sha256"] != _sha256(provenance_path):
        raise RuntimeError("NQ7 provenance hash mismatch")
    provenance = _load_json(provenance_path)
    for relative, expected in provenance["artifacts_sha256"].items():
        path = output / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"NQ7 raw artifact hash mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    repo = Path(__file__).resolve().parents[1]

    if args.verify:
        _verify_closed(output)
        print("NQ7 raw synthesis verified")
        return 0
    if any(
        (output / name).exists()
        for name in (
            "result.json",
            "provenance.json",
            "completed.json",
        )
    ):
        raise RuntimeError("refusing to overwrite an existing NQ7 synthesis")

    identity = _validate_identity(output, repo)
    parents = {
        "nq4": _authenticate_parent(
            repo,
            "NQ4",
            _NQ4_ROOT,
            "chapter13_nq4_review_20260921.json",
        ),
        "nq6": _authenticate_parent(
            repo,
            "NQ6",
            _NQ6_ROOT,
            "chapter13_nq6_review_20260923.json",
        ),
    }
    records, validation = _validate_checkpoints(output)
    result = {
        "schema": "aion.chapter13.nq7.result.v2",
        "status": "executed_unreviewed_with_visible_domain_failures",
        "implementation_commit": identity["implementation_commit"],
        "implementation_source_sha256": identity["source_sha256"],
        "accepted_inputs": parents,
        "profile": {
            "system": "equilateral H3+; charge +1; two electrons",
            "main_basis": "cc-pvdz",
            "domain_bases": list(_DOMAIN_BASES),
            "functional": "lda,vwn",
            "auxiliary_basis": "weigend",
            "main_grid_level": 4,
            "domain_grid_level": 3,
            "fields_au": list(_FIELDS),
            "domain_fields_au": list(_DOMAIN_FIELDS),
            "levels": list(_LEVELS),
            "branches": list(_BRANCHES),
            "dynamic_cases": list(_DYNAMIC_CASES),
            "final_time_au": 2.0,
            "dynamic_intervals": [16, 32],
        },
        "checkpoint_validation": validation,
        "records": records,
    }
    result_path = output / "result.json"
    _write_json(result_path, result)
    artifacts = {str(path.relative_to(output)): _sha256(path) for path in _raw_artifacts(output)}
    provenance = {
        "schema": "aion.chapter13.nq7.provenance.v2",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "repository": str(repo),
        "execution_implementation_commit": identity["implementation_commit"],
        "synthesis_commit": _git(repo, "rev-parse", "HEAD").strip(),
        "git_status_porcelain": _git(repo, "status", "--porcelain").splitlines(),
        "environment": DependencyVersions.current().as_mapping(),
        "thread_limits": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "command": " ".join(sys.argv),
        "artifacts_sha256": artifacts,
    }
    provenance_path = output / "provenance.json"
    _write_json(provenance_path, provenance)
    _write_json(
        output / "completed.json",
        {
            "schema": "aion.chapter13.nq7.completed.v2",
            "status": "complete_with_visible_domain_failures",
            "result_sha256": _sha256(result_path),
            "provenance_sha256": _sha256(provenance_path),
        },
    )
    _verify_closed(output)
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
