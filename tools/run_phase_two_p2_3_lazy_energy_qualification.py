#!/usr/bin/env python3
"""Qualify lazy exact-Wilson endpoint energy against the frozen P2-3B runs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import resource
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import h5py
import numpy as np
from run_phase_two_p2_3_profile import (
    _authenticate_accepted_review,
    _authenticate_campaign,
    _git,
    _sha256,
    _write_json,
)
from run_phase_two_p2_3_schedule_audit import _VARIANTS, _run_variant

from aion.config import WilsonSimulationConfig, loads_config
from aion.electronic_structure import DependencyVersions, load_wilson_stationary_state
from aion.io import load_wilson_checkpoint
from aion.workflows import load_reference


def _numeric_datasets(path: Path) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:

        def collect(name: str, item: h5py.Group | h5py.Dataset) -> None:
            if isinstance(item, h5py.Dataset) and np.issubdtype(item.dtype, np.number):
                result[name] = np.asarray(item[...])

        handle.visititems(collect)
    return result


def _compare_to_frozen(
    output: Path, baseline: Path, name: str, record: dict[str, object], density: np.ndarray
) -> dict[str, object]:
    frozen_dir = baseline / name
    candidate_dir = output / name
    frozen_trajectory = _numeric_datasets(frozen_dir / "trajectory.h5")
    candidate_trajectory = _numeric_datasets(candidate_dir / "trajectory.h5")
    if set(candidate_trajectory) != set(frozen_trajectory):
        raise RuntimeError(f"trajectory numeric dataset inventory changed: {name}")
    for path, values in frozen_trajectory.items():
        if not np.array_equal(candidate_trajectory[path], values):
            raise RuntimeError(f"trajectory numeric dataset differs: {name}/{path}")

    frozen_checkpoints = sorted(frozen_dir.glob("checkpoint_*.h5"))
    candidate_checkpoints = sorted(candidate_dir.glob("checkpoint_*.h5"))
    if [path.name for path in frozen_checkpoints] != [path.name for path in candidate_checkpoints]:
        raise RuntimeError(f"checkpoint schedule changed: {name}")
    for frozen_path, candidate_path in zip(frozen_checkpoints, candidate_checkpoints, strict=True):
        frozen = load_wilson_checkpoint(frozen_path)
        candidate = load_wilson_checkpoint(candidate_path)
        if not np.array_equal(candidate.contravariant_density, frozen.contravariant_density):
            raise RuntimeError(f"checkpoint density changed: {name}/{candidate_path.name}")
        for field in (
            "global_step",
            "accumulated_source_work_au",
            "initial_molecular_energy_au",
            "observer_schedule_state",
            "source_fingerprint_sha256",
        ):
            if getattr(candidate, field) != getattr(frozen, field):
                raise RuntimeError(f"checkpoint {field} changed: {name}/{candidate_path.name}")
    if not np.array_equal(
        density, load_wilson_checkpoint(frozen_checkpoints[-1]).contravariant_density
    ):
        raise RuntimeError(f"final in-memory density changed: {name}")

    frozen_record = json.loads((baseline / f"{name}.json").read_text(encoding="utf-8"))
    frozen_calls = cast(dict[str, int], frozen_record["call_counts"])
    candidate_calls = cast(dict[str, int], record["call_counts"])
    for key in ("propagation.steps", "propagation.full_action", "runner.pbe_evaluate.directional"):
        if candidate_calls.get(key) != frozen_calls.get(key):
            raise RuntimeError(f"mandatory {key} call count changed: {name}")
    if name == "full":
        if candidate_calls != frozen_calls:
            raise RuntimeError("unchanged full-output control altered call counts")
    else:
        if candidate_calls.get("endpoint.observe_endpoint") != 1:
            raise RuntimeError("energy-only schedule evaluated a full post-initial endpoint")
        if candidate_calls.get("endpoint.full_action") != 5:
            raise RuntimeError("energy-only schedule changed endpoint action count")
        if candidate_calls.get("endpoint.pbe_evaluate.directional") != 7:
            raise RuntimeError("energy-only schedule recomputed unrequested directions")
        if candidate_calls.get("endpoint.pbe_evaluate", 0) >= frozen_calls.get(
            "endpoint.pbe_evaluate", 0
        ):
            raise RuntimeError("energy-only endpoint did not reduce PBE work")
    return {
        "frozen_record_sha256": _sha256(baseline / f"{name}.json"),
        "numeric_trajectory_datasets_bitwise_equal": len(frozen_trajectory),
        "checkpoints_bitwise_equal": len(frozen_checkpoints),
        "frozen_run_seconds": frozen_record["run_seconds"],
        "candidate_run_seconds": record["run_seconds"],
        "frozen_endpoint_pbe_calls": frozen_calls.get("endpoint.pbe_evaluate", 0),
        "candidate_endpoint_pbe_calls": candidate_calls.get("endpoint.pbe_evaluate", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    baseline = arguments.schedule_root.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "campaign.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another lazy-energy qualification owns this output root") from None
    if (output / "completed.json").exists():
        raise FileExistsError("refusing to overwrite a completed qualification")
    if os.environ.get("AION_GPU_LAUNCHER") != "1":
        raise RuntimeError("lazy-energy qualification requires tools/gpu-python")
    repo = Path(__file__).resolve().parents[1]
    try:
        _write_json(
            output / "campaign_status.json",
            {"state": "running", "stage": "authenticate", "pid": os.getpid()},
        )
        frozen = _authenticate_campaign(baseline)
        baseline_identity = json.loads(
            (baseline / "campaign_identity.json").read_text(encoding="utf-8")
        )
        recorded_inherited = baseline_identity["inherited_evidence"]
        inherited = {
            name: _authenticate_campaign(Path(recorded_inherited[name]["root"]))
            for name in ("base", "refinement", "profile")
        }
        for name, record in inherited.items():
            if record != recorded_inherited[name]:
                raise RuntimeError(f"frozen inherited evidence changed: {name}")
        accepted_review = _authenticate_accepted_review(repo, inherited)
        base = Path(cast(str, inherited["base"]["root"]))
        resolved = loads_config((base / "nh3/exact_gpu.toml").read_text(encoding="utf-8"))
        base_config = resolved.config
        if not isinstance(base_config, WilsonSimulationConfig):
            raise RuntimeError("accepted NH3 input is not a Wilson configuration")
        reference = load_reference(base / "nh3/reference.h5")
        stationary = load_wilson_stationary_state(base / "nh3/stationary.h5")
        identity = {
            "schema": "aion.phase-two.p2-3c1.lazy-energy-identity",
            "schema_version": "1.0.0",
            "git_head": _git(repo, "rev-parse", "HEAD"),
            "git_branch": _git(repo, "branch", "--show-current"),
            "driver_sha256": _sha256(Path(__file__).resolve()),
            "frozen_schedule_audit": frozen,
            "inherited_evidence": inherited,
            "accepted_p2_2_review": accepted_review,
        }
        _write_json(output / "campaign_identity.json", identity)
        schedules = dict(_VARIANTS)
        results: list[dict[str, object]] = []
        for name in ("full", "energy_only"):
            _write_json(
                output / "campaign_status.json",
                {
                    "state": "running",
                    "stage": name,
                    "completed_variants": len(results),
                    "total_variants": 2,
                    "pid": os.getpid(),
                },
            )
            record, density, _ = _run_variant(
                name=name,
                schedules=schedules[name],
                root=output,
                base_config=base_config,
                reference=reference,
                stationary=stationary,
            )
            record["frozen_comparison"] = _compare_to_frozen(
                output, baseline, name, record, density
            )
            _write_json(output / f"{name}.json", record)
            results.append(record)
        result = {
            "schema": "aion.phase-two.p2-3c1.lazy-energy-result",
            "schema_version": "1.0.0",
            "status": "executed_unreviewed",
            "system": "NH3/cc-pVDZ/PBE accepted P2-2 physical-GPU bridge",
            "qualification_boundary": (
                "Only endpoint energy selection changes. Full-output control and every "
                "stored numeric trajectory dataset/checkpoint state are compared bitwise "
                "with frozen P2-3B. Single-run wall times are directional."
            ),
            "variants": results,
        }
        result_path = output / "result.json"
        _write_json(result_path, result)
        artifacts = {
            relative: sha
            for record in results
            for relative, sha in cast(dict[str, str], record["artifact_sha256"]).items()
        }
        artifacts.update(
            {
                f"{record['variant']}.json": _sha256(output / f"{record['variant']}.json")
                for record in results
            }
        )
        provenance = {
            "schema": "aion.phase-two.p2-3c1.lazy-energy-provenance",
            "schema_version": "1.0.0",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git_head": _git(repo, "rev-parse", "HEAD"),
            "git_branch": _git(repo, "branch", "--show-current"),
            "git_status_porcelain": _git(
                repo, "status", "--porcelain=v1", "--untracked-files=all"
            ).splitlines(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "host": platform.node(),
            "dependencies": DependencyVersions.current().as_mapping(),
            "thread_limits": {
                key: os.environ.get(key)
                for key in (
                    "AION_HOST_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * 1024,
            "campaign_identity_sha256": _sha256(output / "campaign_identity.json"),
            "artifacts_sha256": artifacts,
        }
        provenance_path = output / "provenance.json"
        _write_json(provenance_path, provenance)
        _write_json(
            output / "completed.json",
            {
                "schema": "aion.phase-two.p2-3c1.lazy-energy-completed",
                "schema_version": "1.0.0",
                "status": "executed_unreviewed",
                "result_sha256": _sha256(result_path),
                "provenance_sha256": _sha256(provenance_path),
            },
        )
        _write_json(
            output / "campaign_status.json",
            {
                "state": "completed",
                "stage": "all",
                "completed_variants": len(results),
                "total_variants": len(results),
                "pid": os.getpid(),
            },
        )
    except Exception as exc:
        _write_json(
            output / "failure.json",
            {
                "schema": "aion.phase-two.p2-3c1.lazy-energy-failure",
                "schema_version": "1.0.0",
                "status": "failed_visible",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        _write_json(
            output / "campaign_status.json",
            {"state": "failed_visible", "stage": "qualification", "pid": os.getpid()},
        )
        raise
    finally:
        lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
