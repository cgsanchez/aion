#!/usr/bin/env python3
"""Run the bounded GPU timestep refinement for the P2-2 NH3 current check."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import run_phase_two_p2_2_bridge as bridge

from aion.config import (
    BackendConfig,
    BackendKind,
    FixedTimeGrid,
    MetadataConfig,
    OutputConfig,
    dumps_config,
)
from aion.electronic_structure import (
    AOGridPolicy,
    AOPruningKind,
    DependencyVersions,
    load_wilson_stationary_state,
    prepare_ao_quadrature,
)
from aion.io import load_wilson_checkpoint, load_wilson_trajectory
from aion.io.checkpoint import load_checkpoint
from aion.io.trajectory import load_trajectory
from aion.workflows import (
    BuiltSimulation,
    BuiltWilsonSimulation,
    build_simulation,
    load_reference,
    run,
)

_STEP_AU = 0.0125
_INTERVALS = 16
_CURRENT_THRESHOLD_AU = 5.0e-5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _maximum_absolute(candidate: object, reference: object) -> float:
    return float(np.max(np.abs(np.asarray(candidate) - np.asarray(reference))))


def _relative(candidate: object, reference: object) -> float:
    left = np.asarray(candidate)
    right = np.asarray(reference)
    return float(np.linalg.norm(left - right) / max(1.0, float(np.linalg.norm(right))))


def _authenticate_base(root: Path) -> dict[str, Any]:
    completed = json.loads((root / "completed.json").read_text(encoding="utf-8"))
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    if completed["result_sha256"] != _sha256(root / "result.json"):
        raise RuntimeError("base P2-2 result hash mismatch")
    if completed["provenance_sha256"] != _sha256(root / "provenance.json"):
        raise RuntimeError("base P2-2 provenance hash mismatch")
    for stage, expected in provenance["stage_records_sha256"].items():
        stage_path = root / "stages" / f"{stage}.json"
        if expected != _sha256(stage_path):
            raise RuntimeError(f"base P2-2 stage hash mismatch: {stage}")
        stage_record = json.loads(stage_path.read_text(encoding="utf-8"))
        for relative_path, artifact_sha256 in stage_record.get("artifacts_sha256", {}).items():
            artifact_path = root / relative_path
            if artifact_sha256 != _sha256(artifact_path):
                raise RuntimeError(f"base P2-2 artifact hash mismatch: {relative_path}")
    return cast(
        dict[str, Any],
        json.loads((root / "result.json").read_text(encoding="utf-8")),
    )


def _write_config(path: Path, config: object) -> None:
    path.write_text(dumps_config(config), encoding="utf-8")  # type: ignore[arg-type]


def _run_exact(output: Path, base: Path) -> tuple[dict[str, object], tuple[Path, ...]]:
    reference = load_reference(base / "nh3/reference.h5")
    stationary = load_wilson_stationary_state(base / "nh3/stationary.h5")
    backend = BackendConfig(kind=BackendKind.GPU, device_index=0)
    config = bridge._wilson_simulation_config(
        base,
        "nh3",
        reference.fingerprint_sha256,
        stationary.fingerprint_sha256,
        backend,
    )
    config = replace(
        config,
        propagation=replace(
            config.propagation,
            time_grid=FixedTimeGrid(0.0, _STEP_AU, _INTERVALS),
        ),
        output=OutputConfig(output / "exact_gpu", schedules=bridge._schedules()),
        metadata=MetadataConfig(label="phase-two-p2-2-nh3-current-refinement-exact-gpu"),
    )
    config_path = output / "exact_gpu.toml"
    _write_config(config_path, config)
    simulation = build_simulation(config, reference, stationary_state=stationary)
    if not isinstance(simulation, BuiltWilsonSimulation):
        raise RuntimeError("refinement exact configuration built the wrong runtime")
    simulation.quadrature.backend.assert_resident(
        simulation.density,
        name="NH3 refinement exact initial density",
    )
    counter = bridge._instrument_wilson(simulation)
    started = perf_counter()
    trajectory = run(simulation)
    elapsed = perf_counter() - started
    simulation.quadrature.backend.assert_resident(
        simulation.density,
        name="NH3 refinement exact final density",
    )
    run_directory = config.output.directory
    return (
        {
            "trajectory_sha256": trajectory.sha256,
            "elapsed_seconds": elapsed,
            "action_evaluation_calls": int(counter["calls"]),
            "action_evaluation_seconds": float(counter["seconds"]),
            "gpu_residency_asserted": True,
        },
        (
            config_path,
            run_directory / "trajectory.h5",
            run_directory / f"checkpoint_{_INTERVALS:08d}.h5",
            run_directory / "status.json",
        ),
    )


def _run_bare(output: Path, base: Path) -> tuple[dict[str, object], tuple[Path, ...]]:
    reference = load_reference(base / "nh3/reference.h5")
    backend = BackendConfig(kind=BackendKind.GPU, device_index=0)
    config = bridge._bare_nh3_config(
        base,
        reference.fingerprint_sha256,
        backend,
    )
    config = replace(
        config,
        propagation=replace(
            config.propagation,
            time_grid=FixedTimeGrid(0.0, _STEP_AU, _INTERVALS),
        ),
        output=OutputConfig(
            output / "bare_length_gpu",
            schedules=bridge._schedules(),
        ),
        metadata=MetadataConfig(label="phase-two-p2-2-nh3-current-refinement-bare-gpu"),
    )
    config_path = output / "bare_length_gpu.toml"
    _write_config(config_path, config)
    simulation = build_simulation(config, reference)
    if not isinstance(simulation, BuiltSimulation):
        raise RuntimeError("refinement bare configuration built the wrong runtime")
    simulation.workspace.backend.assert_resident(
        simulation.density.matrix,
        name="NH3 refinement bare initial density",
    )
    started = perf_counter()
    trajectory = run(simulation)
    elapsed = perf_counter() - started
    simulation.workspace.backend.assert_resident(
        simulation.density.matrix,
        name="NH3 refinement bare final density",
    )
    run_directory = config.output.directory
    return (
        {
            "trajectory_sha256": trajectory.sha256,
            "elapsed_seconds": elapsed,
            "gpu_residency_asserted": True,
        },
        (
            config_path,
            run_directory / "trajectory.h5",
            run_directory / f"checkpoint_{_INTERVALS:08d}.h5",
            run_directory / "status.json",
        ),
    )


def _analyze(
    output: Path,
    base: Path,
    base_result: dict[str, Any],
    exact_stage: dict[str, object],
    bare_stage: dict[str, object],
) -> dict[str, object]:
    exact_path = output / "exact_gpu/trajectory.h5"
    bare_path = output / "bare_length_gpu/trajectory.h5"
    exact = load_wilson_trajectory(exact_path)
    bare = load_trajectory(bare_path)
    exact_current = exact.read_series("current/uniform_source").values
    bare_current = bare.read_observable("current.variational_source.bare_length_gauge").values
    exact_dipole = exact.read_series("dipole/molecular_total").values
    bare_dipole = bare.read_observable("dipole.combined.bare_length_gauge").values
    exact_energy = exact.read_series("energy/molecular_total").values
    bare_energy = bare.read_observable("energy.matter.total.bare_length_gauge").values.reshape(-1)
    exact_checkpoint = load_wilson_checkpoint(output / f"exact_gpu/checkpoint_{_INTERVALS:08d}.h5")
    bare_checkpoint = load_checkpoint(output / f"bare_length_gpu/checkpoint_{_INTERVALS:08d}.h5")

    coarse_exact = load_wilson_trajectory(base / "nh3/exact_gpu/trajectory.h5")
    coarse_bare = load_trajectory(base / "nh3/bare_length_gpu/trajectory.h5")
    coarse_exact_current = coarse_exact.read_series("current/uniform_source").values
    coarse_bare_current = coarse_bare.read_observable(
        "current.variational_source.bare_length_gauge"
    ).values

    reference = load_reference(base / "nh3/reference.h5")
    exact_grid = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(4, pruning=AOPruningKind.NONE),
        block_size=1024,
    ).grid
    grid_arrays_equal = bool(
        np.array_equal(reference.grid.coordinates_au, exact_grid.coordinates_au)
        and np.array_equal(reference.grid.weights_au, exact_grid.weights_au)
    )

    coarse_current_residual = _maximum_absolute(
        coarse_exact_current,
        coarse_bare_current,
    )
    fine_current_residual = _maximum_absolute(exact_current, bare_current)
    measurements = {
        "grid_arrays_bitwise_equal": grid_arrays_equal,
        "reference_grid_fingerprint_sha256": reference.grid.fingerprint_sha256,
        "qualification_grid_fingerprint_sha256": exact_grid.fingerprint_sha256,
        "grid_points": exact_grid.npoints,
        "coarse_exact_bare_current_maximum_absolute_residual_au": coarse_current_residual,
        "fine_exact_bare_current_maximum_absolute_residual_au": fine_current_residual,
        "current_residual_refinement_ratio": fine_current_residual / coarse_current_residual,
        "exact_current_coarse_fine_maximum_absolute_residual_au": _maximum_absolute(
            exact_current[::2],
            coarse_exact_current,
        ),
        "bare_current_coarse_fine_maximum_absolute_residual_au": _maximum_absolute(
            bare_current[::2],
            coarse_bare_current,
        ),
        "fine_exact_bare_dipole_maximum_absolute_residual_au": _maximum_absolute(
            exact_dipole,
            bare_dipole,
        ),
        "fine_exact_bare_energy_maximum_absolute_residual_au": _maximum_absolute(
            exact_energy,
            bare_energy,
        ),
        "fine_exact_bare_final_density_relative_residual": _relative(
            exact_checkpoint.contravariant_density,
            bare_checkpoint.density,
        ),
        **bridge._wilson_diagnostics(exact_path),
    }
    excluded = {"nh3_grids_match", "nh3_exact_bare_current_matches"}
    inherited_checks = {
        name: bool(value) for name, value in base_result["checks"].items() if name not in excluded
    }
    checks = {
        **inherited_checks,
        "nh3_grid_arrays_match": grid_arrays_equal,
        "nh3_current_residual_decreases_under_timestep_halving": (
            fine_current_residual < coarse_current_residual
        ),
        "nh3_refined_exact_bare_current_matches": (fine_current_residual <= _CURRENT_THRESHOLD_AU),
        "nh3_refined_exact_bare_dipole_matches": (
            measurements["fine_exact_bare_dipole_maximum_absolute_residual_au"]
            <= base_result["thresholds"]["nh3_exact_bare_dipole_absolute_au"]
        ),
        "nh3_refined_exact_bare_energy_matches": (
            measurements["fine_exact_bare_energy_maximum_absolute_residual_au"]
            <= base_result["thresholds"]["nh3_exact_bare_energy_absolute_au"]
        ),
        "nh3_refined_exact_bare_final_density_matches": (
            measurements["fine_exact_bare_final_density_relative_residual"]
            <= base_result["thresholds"]["nh3_exact_bare_final_density_relative"]
        ),
    }
    return {
        "schema": "aion.phase-two.p2-2.current-refinement-result",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "proposed_gate_result": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "measurements": measurements,
        "thresholds": {
            **base_result["thresholds"],
            "refined_current_absolute_au": _CURRENT_THRESHOLD_AU,
        },
        "stage_measurements": {"exact_gpu": exact_stage, "bare_gpu": bare_stage},
        "interpretation_boundary": (
            "This supplement corrects the semantic-grid-fingerprint comparison and tests "
            "the sole remaining current discrepancy by one GPU timestep halving. It inherits "
            "the base campaign's CPU/GPU parity and does not replace its raw evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    base = arguments.base_root.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "sequence.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another P2-2 refinement owns this output root") from None
    if (output / "completed.json").exists():
        raise FileExistsError("refusing to overwrite completed P2-2 refinement")
    if os.environ.get("AION_GPU_LAUNCHER") != "1":
        raise RuntimeError("the refinement must be launched through tools/gpu-python")

    repo = Path(__file__).resolve().parents[1]
    try:
        base_result = _authenticate_base(base)
        identity = {
            "schema": "aion.phase-two.p2-2.current-refinement-identity",
            "schema_version": "1.0.0",
            "git_head": _git(repo, "rev-parse", "HEAD"),
            "git_branch": _git(repo, "branch", "--show-current"),
            "driver_sha256": _sha256(Path(__file__).resolve()),
            "base_root": str(base),
            "base_completed_sha256": _sha256(base / "completed.json"),
            "base_result_sha256": _sha256(base / "result.json"),
            "base_provenance_sha256": _sha256(base / "provenance.json"),
            "time_step_au": _STEP_AU,
            "intervals": _INTERVALS,
        }
        _write_json(output / "campaign_identity.json", identity)
        _write_json(
            output / "campaign_status.json",
            {"state": "running", "stage": "exact_gpu", "pid": os.getpid()},
        )
        exact_stage, exact_artifacts = _run_exact(output, base)
        _write_json(
            output / "campaign_status.json",
            {"state": "running", "stage": "bare_gpu", "pid": os.getpid()},
        )
        bare_stage, bare_artifacts = _run_bare(output, base)
        result = _analyze(output, base, base_result, exact_stage, bare_stage)
        result_path = output / "result.json"
        _write_json(result_path, result)
        artifacts = (*exact_artifacts, *bare_artifacts)
        provenance = {
            "schema": "aion.phase-two.p2-2.current-refinement-provenance",
            "schema_version": "1.0.0",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "repository": str(repo),
            "git_head": _git(repo, "rev-parse", "HEAD"),
            "git_branch": _git(repo, "branch", "--show-current"),
            "git_status_porcelain": _git(
                repo,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ).splitlines(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "host": platform.node(),
            "dependencies": DependencyVersions.current().as_mapping(),
            "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * 1024,
            "artifacts_sha256": {
                str(path.relative_to(output)): _sha256(path) for path in artifacts
            },
            "campaign_identity_sha256": _sha256(output / "campaign_identity.json"),
        }
        provenance_path = output / "provenance.json"
        _write_json(provenance_path, provenance)
        _write_json(
            output / "completed.json",
            {
                "schema": "aion.phase-two.p2-2.current-refinement-completed",
                "schema_version": "1.0.0",
                "status": "executed_unreviewed",
                "result_sha256": _sha256(result_path),
                "provenance_sha256": _sha256(provenance_path),
            },
        )
        _write_json(
            output / "campaign_status.json",
            {"state": "completed", "stage": "analyze", "pid": os.getpid()},
        )
    except Exception as exc:
        _write_json(
            output / "failure.json",
            {
                "schema": "aion.phase-two.p2-2.current-refinement-failure",
                "schema_version": "1.0.0",
                "status": "failed_visible",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        _write_json(
            output / "campaign_status.json",
            {"state": "failed_visible", "stage": "unknown", "pid": os.getpid()},
        )
        raise
    finally:
        lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
