#!/usr/bin/env python3
"""Authenticate NQ9, record lineage, and hash the selected NH3 heritage set."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ACCEPTED_PARENT = "dc6d4e2e0267bf2542074528ca1305e5fdff592b"
NH3_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "nh3_gauge_basis_validation/nh3_gauge_comparison"
)
WP7_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/aion_0p2_release_validation"
)
HERITAGE: tuple[tuple[str, str, str, str], ...] = (
    (
        "geometry",
        "nh3_pbe_ccpvdz_optimized.xyz",
        "geometry and orientation for the fixed G_DZ bridge",
        "PBE/cc-pVDZ optimized geometry; not an exact-Wilson result",
    ),
    (
        "geometry_provenance",
        "nh3_pbe_ccpvdz_optimized.optimization.json",
        "optimization settings and residuals for G_DZ",
        "historical optimization record",
    ),
    (
        "casida",
        "casida/nh3_pbe_cc-pvdz_df_casida.json",
        "prior axial bright-state energy and direction",
        "ordinary field-free PySCF Casida with RI-J",
    ),
    (
        "resonant_summary",
        "results/cc-pvdz_df/nh3_resonant_method_summary.json",
        "historical source, runtime, and formulation comparison",
        "bare/P0/P0+E1 campaign; not exact Wilson",
    ),
    (
        "resonant_bare_length",
        "results/cc-pvdz_df/nh3_bare_length_trajectory.csv",
        "zero-field ordinary length-gauge pulse comparison",
        "historical CSV without current Phase Two artifact schema",
    ),
    (
        "resonant_p0e1_length",
        "results/cc-pvdz_df/nh3_p0e1_length_connection_trajectory.csv",
        "historical P0+E1 pulse comparison",
        "reduced action and earlier propagator; not exact Wilson",
    ),
    (
        "kick_summary",
        "results/bare_kick_spectra/cc-pvdz_dt0p05_t1700/nh3_bare_kick_campaign_summary.json",
        "long-kick protocol and runtime",
        "one kick amplitude, timestep, and duration",
    ),
    (
        "kick_analysis",
        "results/bare_kick_spectra/cc-pvdz_dt0p05_t1700/nh3_bare_kick_analysis.json",
        "prior dipole/current spectral diagnostics",
        "ordinary bare formulations only",
    ),
    (
        "kick_peaks",
        "results/bare_kick_spectra/cc-pvdz_dt0p05_t1700/nh3_bare_kick_peaks.csv",
        "prior peak positions for comparison planning",
        "finite-record historical peak extraction",
    ),
    (
        "kick_spectra",
        "results/bare_kick_spectra/cc-pvdz_dt0p05_t1700/nh3_bare_kick_spectra.csv",
        "prior axial and transverse spectral curves",
        "ordinary bare formulations only",
    ),
    (
        "kick_axial_bare_length",
        "results/bare_kick_spectra/cc-pvdz_dt0p05_t1700/nh3_axial_bare_length_trajectory.csv",
        "ordinary length-gauge time-domain bridge",
        "historical CSV; no exact action observables",
    ),
    (
        "kick_axial_bare_velocity",
        "results/bare_kick_spectra/cc-pvdz_dt0p05_t1700/nh3_axial_bare_velocity_trajectory.csv",
        "finite-basis bare-gauge comparison",
        "historical CSV; residual current is diagnostic",
    ),
    (
        "p0e1_kick_summary",
        "results/aug-cc-pvdz_fixed_geometry/p0e1_length_kick_dt0p05_t1700/"
        "nh3_p0e1_length_kick_summary.json",
        "prior reduced-action long-kick protocol",
        "aug-cc-pVDZ and P0+E1 only; current was not recorded",
    ),
    (
        "p0e1_kick_trajectory",
        "results/aug-cc-pvdz_fixed_geometry/p0e1_length_kick_dt0p05_t1700/"
        "nh3_axial_p0e1_length_kick_trajectory.csv",
        "prior reduced-action dipole spectrum comparison",
        "not exact Wilson and has no stored action current",
    ),
    (
        "kick_cpu_timing",
        "results/kick_benchmark/nh3_bare_kick_benchmark.json",
        "historical CPU cost lower bound",
        "ordinary bare propagator cost, not exact-Wilson cost",
    ),
    (
        "kick_gpu_timing",
        "results/kick_benchmark/nh3_bare_kick_benchmark_gpu.json",
        "historical GPU cost lower bound",
        "ordinary bare propagator cost, not exact-Wilson cost",
    ),
)


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


def _git(repo: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
    )


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _wp7_status() -> dict[str, Any]:
    retained = WP7_ROOT / "retained_augccpvtz_evidence_summary.json"
    compact = WP7_ROOT / "results/qualification/compact_631g/qualification_summary.json"
    statuses = sorted((WP7_ROOT / "results/trajectories").glob("**/status.json"))
    counts: dict[str, int] = {}
    for path in statuses:
        phase = str(json.loads(path.read_text(encoding="utf-8"))["phase"])
        counts[phase] = counts.get(phase, 0) + 1
    return {
        "root": str(WP7_ROOT),
        "retained_augccpvtz_summary": str(retained),
        "retained_augccpvtz_summary_exists": retained.is_file(),
        "retained_augccpvtz_summary_sha256": _sha256(retained) if retained.is_file() else None,
        "trajectory_status_counts": counts,
        "bounded_compact_qualification_summary": str(compact),
        "bounded_compact_qualification_completed": compact.is_file(),
        "interpretation": (
            "The retained aug-cc-pVTZ evidence exists, but the later bounded 6-31G "
            "qualification did not produce its final analyzer summary. It is inherited "
            "evidence, not the Phase Two exact-Wilson release gate."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]

    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    parent_check = _git(repo, "merge-base", "--is-ancestor", ACCEPTED_PARENT, head, check=False)
    if parent_check.returncode != 0:
        raise RuntimeError("accepted Chapter 13 parent is not an ancestor of Phase Two")
    nq9 = subprocess.run(
        (sys.executable, str(repo / "tools/verify_chapter13_nq9_report.py")),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    nq9_result = json.loads(nq9.stdout)

    heritage_records: list[dict[str, Any]] = []
    for role, relative, use, limitation in HERITAGE:
        path = NH3_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        heritage_records.append(
            {
                "role": role,
                "path": str(path),
                "path_relative_to_nh3_root": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "admissible_use": use,
                "limitation": limitation,
            }
        )
    heritage_manifest = {
        "schema": "aion.phase-two.p2-0.ammonia-heritage",
        "schema_version": "1.0.0",
        "status": "authenticated_heritage_not_exact_wilson_evidence",
        "root": str(NH3_ROOT),
        "geometry_policy": (
            "Use fixed G_DZ for the first exact-Wilson bridge. Do not combine the "
            "G_DZ cc-pVXZ ladder with G_augTZ as one geometry-fixed basis series."
        ),
        "artifacts": heritage_records,
    }
    heritage_path = output / "ammonia_heritage_manifest.json"
    _write_json(heritage_path, heritage_manifest)

    result = {
        "schema": "aion.phase-two.p2-0.reconciliation-result",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "accepted_parent": ACCEPTED_PARENT,
        "accepted_parent_is_ancestor": True,
        "phase_two_head": head,
        "package_version": _package_version("aion"),
        "nq9_verification": nq9_result,
        "environment": {
            distribution: _package_version(distribution)
            for distribution in (
                "numpy",
                "scipy",
                "h5py",
                "pyscf",
                "cupy-cuda12x",
                "gpu4pyscf-cuda12x",
                "ruff",
                "mypy",
                "pytest",
                "pytest-xdist",
            )
        },
        "wp7_reconciliation": _wp7_status(),
        "ammonia_heritage_manifest": heritage_path.name,
        "ammonia_heritage_manifest_sha256": _sha256(heritage_path),
    }
    result_path = output / "result.json"
    _write_json(result_path, result)
    provenance = {
        "schema": "aion.phase-two.provenance",
        "schema_version": "1.0.0",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "repository": str(repo),
        "git_head": head,
        "git_status_porcelain": _git(
            repo,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).stdout.splitlines(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "host": platform.node(),
        "artifacts_sha256": {
            heritage_path.name: _sha256(heritage_path),
        },
    }
    provenance_path = output / "provenance.json"
    _write_json(provenance_path, provenance)
    completed = {
        "schema": "aion.phase-two.completed",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "result_sha256": _sha256(result_path),
        "provenance_sha256": _sha256(provenance_path),
    }
    _write_json(output / "completed.json", completed)
    print(json.dumps({"output": str(output), **completed}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
