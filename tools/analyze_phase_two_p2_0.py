#!/usr/bin/env python3
"""Verify and synthesize the Phase Two P2-0 evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verify_subcampaign(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    completed_path = path / "completed.json"
    result_path = path / "result.json"
    provenance_path = path / "provenance.json"
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    if completed["result_sha256"] != _sha256(result_path):
        raise RuntimeError(f"result hash failed for {path}")
    if completed["provenance_sha256"] != _sha256(provenance_path):
        raise RuntimeError(f"provenance hash failed for {path}")
    return json.loads(result_path.read_text(encoding="utf-8")), {
        "root": str(path),
        "result_sha256": completed["result_sha256"],
        "provenance_sha256": completed["provenance_sha256"],
        "completed_sha256": _sha256(completed_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.root.expanduser().resolve()
    output = root / "analysis"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir()
    names = (
        "reconciliation",
        "quality",
        "benchmarks/h3plus",
        "benchmarks/co_completed",
    )
    results: dict[str, dict[str, Any]] = {}
    authentication: dict[str, dict[str, str]] = {}
    for name in names:
        result, hashes = _verify_subcampaign(root / name)
        results[name] = result
        authentication[name] = hashes

    quality_passed = bool(results["quality"]["all_passed"])
    nq9_verified = results["reconciliation"]["nq9_verification"]["status"] == ("verified_accepted")
    benchmarks_complete = all(
        results[name]["status"] == "executed_unreviewed"
        for name in ("benchmarks/h3plus", "benchmarks/co_completed")
    )
    passed = quality_passed and nq9_verified and benchmarks_complete
    result = {
        "schema": "aion.phase-two.p2-0.analysis",
        "schema_version": "1.0.0",
        "status": "analyzed_unreviewed",
        "proposed_gate_result": "pass" if passed else "fail",
        "checks": {
            "accepted_nq9_verified": nq9_verified,
            "all_software_quality_gates_passed": quality_passed,
            "h3plus_and_co_baselines_completed": benchmarks_complete,
            "ammonia_heritage_authenticated": (
                results["reconciliation"]["ammonia_heritage_manifest_sha256"]
                == _sha256(root / "reconciliation/ammonia_heritage_manifest.json")
            ),
        },
        "authentication": authentication,
        "quality_gates": results["quality"]["gates"],
        "benchmark_summary": {
            name.split("/")[-1]: {
                "total_elapsed_seconds": results[name]["total_elapsed_seconds"],
                "process_peak_rss_bytes": results[name]["process_peak_rss_bytes"],
                "stage_timings_seconds": results[name]["stage_timings_seconds"],
                "stationary": results[name]["record"]["stationary"],
                "dynamics": results[name]["record"]["dynamics"],
            }
            for name in ("benchmarks/h3plus", "benchmarks/co_completed")
        },
        "inherited_wp7": results["reconciliation"]["wp7_reconciliation"],
        "limitations": [
            "The locked Ruff formatter initially rejected 47 accepted-parent files; "
            "a dedicated mechanical formatting commit repaired the baseline before "
            "the authenticated quality run.",
            "The historical bounded WP7 6-31G analyzer summary is absent; retained "
            "aug-cc-pVTZ evidence remains explicit and Phase Two uses its own gates.",
            "Stationary baseline timings are warm-start timings from accepted states, "
            "not from-scratch convergence costs.",
            "The first CO timing attempt is retained as failed-visible evidence. Its "
            "bytewise reference-fingerprint equality check was invalid for a freshly "
            "repeated SCF calculation; the completed retry uses explicit scientific "
            "configuration, energy, electron-count, AO-dimension, and stationary "
            "residual checks.",
            "P2-0 does not execute an NH3 exact-Wilson trajectory.",
        ],
    }
    result_path = output / "result.json"
    _write_json(result_path, result)
    repo = Path(__file__).resolve().parents[1]
    provenance = {
        "schema": "aion.phase-two.provenance",
        "schema_version": "1.0.0",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "repository": str(repo),
        "git_head": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "input_authentication": authentication,
    }
    provenance_path = output / "provenance.json"
    _write_json(provenance_path, provenance)
    completed = {
        "schema": "aion.phase-two.completed",
        "schema_version": "1.0.0",
        "status": "analyzed_unreviewed",
        "result_sha256": _sha256(result_path),
        "provenance_sha256": _sha256(provenance_path),
    }
    _write_json(output / "completed.json", completed)
    print(json.dumps({"output": str(output), **completed}, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
