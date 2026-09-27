#!/usr/bin/env python3
"""Authenticated four-step GPU schedule ablation of the accepted NH3 bridge."""

from __future__ import annotations

import argparse
import fcntl
import os
import platform
import resource
import sys
import traceback
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from unittest.mock import patch

import numpy as np
from run_phase_two_p2_3_profile import (
    _authenticate_accepted_review,
    _authenticate_campaign,
    _git,
    _gpu_memory,
    _sha256,
    _TransferCounter,
    _write_json,
)

from aion.config import (
    BackendKind,
    FixedTimeGrid,
    MetadataConfig,
    ObservableSchedules,
    OutputConfig,
    StepSchedule,
    WilsonSimulationConfig,
    dumps_config,
    loads_config,
)
from aion.electronic_structure import (
    DependencyVersions,
    ExactWilsonDynamicSample,
    PreparedRIWilsonHartreeAction,
    WilsonGGAEvaluator,
    load_wilson_stationary_state,
)
from aion.io import WilsonTrajectoryWriter, load_wilson_trajectory
from aion.workflows import BuiltWilsonSimulation, build_simulation, load_reference, run
from aion.workflows.wilson_runner import _WilsonRunExecutor

_OFF = StepSchedule(every=0, include_initial=False, include_final=False)
_EACH = StepSchedule(every=1, include_initial=True, include_final=True)
_ENDS = StepSchedule(every=0, include_initial=True, include_final=True)


def _schedules(
    *, current: bool, energy: bool, diagnostics: bool, sparse: bool
) -> ObservableSchedules:
    return ObservableSchedules(
        dipole_current=_EACH if current else _OFF,
        energy=_EACH if energy else _OFF,
        diagnostics=_EACH if diagnostics else _OFF,
        source=_EACH,
        checkpoints=_ENDS if sparse else _EACH,
        matrix_snapshots=StepSchedule(every=0, include_initial=False, include_final=True),
    )


_VARIANTS = (
    ("full", _schedules(current=True, energy=True, diagnostics=True, sparse=False)),
    ("no_diagnostics", _schedules(current=True, energy=True, diagnostics=False, sparse=False)),
    ("energy_only", _schedules(current=False, energy=True, diagnostics=False, sparse=False)),
    ("minimal", _schedules(current=False, energy=False, diagnostics=False, sparse=True)),
    (
        "full_sparse_checkpoint",
        _schedules(current=True, energy=True, diagnostics=True, sparse=True),
    ),
    ("full_repeat", _schedules(current=True, energy=True, diagnostics=True, sparse=False)),
)


class _CallAudit:
    """Count nested calls without introducing per-call GPU synchronization."""

    def __init__(self) -> None:
        self.phase = "runner"
        self.counts: dict[str, int] = defaultdict(int)
        self.stack = ExitStack()

    def __enter__(self) -> _CallAudit:
        for target, name, label in (
            (ExactWilsonDynamicSample, "evaluate", "full_action"),
            (WilsonGGAEvaluator, "evaluate", "pbe_evaluate"),
            (PreparedRIWilsonHartreeAction, "evaluate", "ri_contract"),
            (BuiltWilsonSimulation, "observe_endpoint", "observe_endpoint"),
            (WilsonTrajectoryWriter, "append_series", "series_append"),
            (WilsonTrajectoryWriter, "flush", "trajectory_flush"),
        ):
            original = getattr(target, name)

            def wrapped(
                instance: Any,
                *args: Any,
                _original: Any = original,
                _label: str = label,
                **kwargs: Any,
            ) -> Any:
                self.counts[f"{self.phase}.{_label}"] += 1
                if _label in ("pbe_evaluate", "observe_endpoint"):
                    if _label == "pbe_evaluate":
                        suffix = (
                            "directional" if kwargs.get("source_direction") is not None else "base"
                        )
                    else:
                        suffix = (
                            f"energy={bool(kwargs.get('include_energy'))},"
                            f"identities={bool(kwargs.get('include_identities'))}"
                        )
                    self.counts[f"{self.phase}.{_label}.{suffix}"] += 1
                return _original(instance, *args, **kwargs)

            self.stack.enter_context(patch.object(target, name, wrapped))
        for name, phase in (
            ("_record_boundary", "endpoint"),
            ("_checkpoint", "checkpoint"),
        ):
            original = getattr(_WilsonRunExecutor, name)

            def wrapped_runner(
                instance: Any,
                *args: Any,
                _original: Any = original,
                _phase: str = phase,
                **kwargs: Any,
            ) -> Any:
                previous = self.phase
                self.phase = _phase
                self.counts[f"{_phase}.calls"] += 1
                try:
                    return _original(instance, *args, **kwargs)
                finally:
                    self.phase = previous

            self.stack.enter_context(patch.object(_WilsonRunExecutor, name, wrapped_runner))
        original_step = BuiltWilsonSimulation.step

        def wrapped_step(instance: BuiltWilsonSimulation) -> Any:
            previous = self.phase
            self.phase = "propagation"
            self.counts["propagation.steps"] += 1
            try:
                return original_step(instance)
            finally:
                self.phase = previous

        self.stack.enter_context(patch.object(BuiltWilsonSimulation, "step", wrapped_step))
        return self

    def __exit__(self, *exc: object) -> None:
        self.stack.close()


def _run_variant(
    *,
    name: str,
    schedules: ObservableSchedules,
    root: Path,
    base_config: WilsonSimulationConfig,
    reference: Any,
    stationary: Any,
) -> tuple[dict[str, object], np.ndarray, tuple[str, ...]]:
    config = replace(
        base_config,
        propagation=replace(
            base_config.propagation,
            time_grid=FixedTimeGrid(0.0, 0.025, 4),
        ),
        output=OutputConfig(root / name, schedules),
        metadata=MetadataConfig(label=f"phase-two-p2-3b-{name}"),
    )
    config_path = root / f"{name}.toml"
    config_text = dumps_config(config)
    config_path.write_text(config_text, encoding="utf-8")
    build_started = perf_counter()
    simulation = build_simulation(
        config, reference, stationary_state=stationary, original_toml=config_text
    )
    if not isinstance(simulation, BuiltWilsonSimulation):
        raise RuntimeError("schedule audit constructed the wrong runtime")
    backend = simulation.quadrature.backend
    if backend.kind is not BackendKind.GPU:
        raise RuntimeError("schedule audit requires physical GPU")
    backend.assert_resident(simulation.density, name="schedule-audit density")
    backend.synchronize()
    build_seconds = perf_counter() - build_started
    memory_before = _gpu_memory(backend)
    with _CallAudit() as calls, _TransferCounter(backend) as transfers:
        backend.synchronize()
        started = perf_counter()
        trajectory = run(simulation)
        backend.synchronize()
        run_seconds = perf_counter() - started
    density = backend.to_host(simulation.density)
    memory_after = _gpu_memory(backend)
    cache = simulation.dynamic_cache.statistics
    if trajectory.final_step != 4 or not trajectory.complete:
        raise RuntimeError(f"incomplete schedule-audit variant: {name}")
    checkpoint_paths = sorted((root / name).glob("checkpoint_*.h5"))
    series_names = tuple(trajectory.series_names)
    record: dict[str, object] = {
        "variant": name,
        "config": str(config_path.relative_to(root)),
        "config_sha256": _sha256(config_path),
        "schedules": schedules.as_mapping(),
        "build_seconds": build_seconds,
        "run_seconds": run_seconds,
        "final_step": trajectory.final_step,
        "accumulated_source_work_au": trajectory.accumulated_source_work_au,
        "density_resident_at_end": backend.is_resident(simulation.density),
        "call_counts": dict(sorted(calls.counts.items())),
        "explicit_transfers": transfers.counts,
        "cache_statistics": {
            "spatial_entries": cache.spatial_entries,
            "spatial_hits": cache.spatial_hits,
            "spatial_misses": cache.spatial_misses,
            "sample_entries": cache.sample_entries,
            "sample_hits": cache.sample_hits,
            "sample_misses": cache.sample_misses,
        },
        "trajectory_sha256": trajectory.sha256,
        "trajectory_bytes": trajectory.path.stat().st_size,
        "checkpoint_count": len(checkpoint_paths),
        "checkpoint_bytes": sum(path.stat().st_size for path in checkpoint_paths),
        "series_names": series_names,
        "gpu_memory_before": memory_before,
        "gpu_memory_after": memory_after,
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "artifact_sha256": {
            str(path.relative_to(root)): _sha256(path)
            for path in (config_path, trajectory.path, *checkpoint_paths)
        },
    }
    _write_json(root / f"{name}.json", record)
    print(f"completed {name}: build={build_seconds:.3f}s run={run_seconds:.3f}s", flush=True)
    return record, density, series_names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--refinement-root", required=True, type=Path)
    parser.add_argument("--profile-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    base = arguments.base_root.expanduser().resolve()
    refinement = arguments.refinement_root.expanduser().resolve()
    profile = arguments.profile_root.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "campaign.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another schedule audit owns this output root") from None
    if (output / "completed.json").exists():
        raise FileExistsError("refusing to overwrite a completed schedule audit")
    if os.environ.get("AION_GPU_LAUNCHER") != "1":
        raise RuntimeError("schedule audit must run through tools/gpu-python")
    repo = Path(__file__).resolve().parents[1]
    try:
        _write_json(
            output / "campaign_status.json",
            {"state": "running", "stage": "authenticate", "pid": os.getpid()},
        )
        inherited = {
            "base": _authenticate_campaign(base),
            "refinement": _authenticate_campaign(refinement),
            "profile": _authenticate_campaign(profile),
        }
        accepted_review = _authenticate_accepted_review(repo, inherited)
        identity = {
            "schema": "aion.phase-two.p2-3b.schedule-identity",
            "schema_version": "1.0.0",
            "git_head": _git(repo, "rev-parse", "HEAD"),
            "git_branch": _git(repo, "branch", "--show-current"),
            "driver_sha256": _sha256(Path(__file__).resolve()),
            "inherited_evidence": inherited,
            "accepted_review": accepted_review,
        }
        _write_json(output / "campaign_identity.json", identity)
        config_path = base / "nh3/exact_gpu.toml"
        resolved = loads_config(config_path.read_text(encoding="utf-8"))
        base_config = resolved.config
        if not isinstance(base_config, WilsonSimulationConfig):
            raise RuntimeError("accepted NH3 input is not a Wilson configuration")
        reference = load_reference(base / "nh3/reference.h5")
        stationary = load_wilson_stationary_state(base / "nh3/stationary.h5")
        records: list[dict[str, object]] = []
        baseline_density: np.ndarray | None = None
        baseline_work: float | None = None
        baseline_series: tuple[str, ...] = ()
        for name, schedules in _VARIANTS:
            _write_json(
                output / "campaign_status.json",
                {
                    "state": "running",
                    "stage": name,
                    "completed_variants": len(records),
                    "total_variants": len(_VARIANTS),
                    "pid": os.getpid(),
                },
            )
            record, density, series = _run_variant(
                name=name,
                schedules=schedules,
                root=output,
                base_config=base_config,
                reference=reference,
                stationary=stationary,
            )
            if baseline_density is None:
                baseline_density = density
                baseline_work = cast(float, record["accumulated_source_work_au"])
                baseline_series = series
            else:
                record["final_density_bitwise_equal_to_full"] = bool(
                    np.array_equal(density, baseline_density)
                )
                record["source_work_bitwise_equal_to_full"] = bool(
                    cast(float, record["accumulated_source_work_au"]) == baseline_work
                )
                if (
                    not record["final_density_bitwise_equal_to_full"]
                    or not record["source_work_bitwise_equal_to_full"]
                ):
                    raise RuntimeError(
                        f"schedule change altered accepted state or source work: {name}"
                    )
                baseline_trajectory = load_wilson_trajectory(output / "full/trajectory.h5")
                variant_trajectory = load_wilson_trajectory(output / name / "trajectory.h5")
                common = sorted(set(baseline_series) & set(series))
                for series_name in common:
                    left = baseline_trajectory.read_series(series_name)
                    right = variant_trajectory.read_series(series_name)
                    left_by_step = {
                        int(step): value
                        for step, value in zip(left.steps, left.values, strict=True)
                    }
                    for step, value in zip(right.steps, right.values, strict=True):
                        if int(step) not in left_by_step or not np.array_equal(
                            value, left_by_step[int(step)]
                        ):
                            raise RuntimeError(
                                f"common series changed: {name}/{series_name}/{step}"
                            )
                record["common_series_bitwise_equal_to_full"] = len(common)
            _write_json(output / f"{name}.json", record)
            records.append(record)
        result = {
            "schema": "aion.phase-two.p2-3b.schedule-result",
            "schema_version": "1.0.0",
            "status": "executed_unreviewed",
            "system": "accepted P2-2 NH3/cc-pVDZ/PBE physical-GPU bridge",
            "grid": {"start_au": 0.0, "step_au": 0.025, "intervals": 4},
            "qualification_boundary": (
                "Schedule ablation only; mandatory Gauss-node source work and the "
                "runner-forced full initial observation remain in every run. Timings "
                "include normal production I/O and use only start/end synchronization. "
                "The final full repeat exposes cold-start/order drift; single-run timing "
                "contrasts are directional, while call counts are exact for this execution."
            ),
            "variants": records,
        }
        result_path = output / "result.json"
        _write_json(result_path, result)
        artifacts = {
            relative: sha
            for record in records
            for relative, sha in cast(dict[str, str], record["artifact_sha256"]).items()
        }
        artifacts.update(
            {
                f"{record['variant']}.json": _sha256(output / f"{record['variant']}.json")
                for record in records
            }
        )
        provenance = {
            "schema": "aion.phase-two.p2-3b.schedule-provenance",
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
                "schema": "aion.phase-two.p2-3b.schedule-completed",
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
                "completed_variants": len(records),
                "total_variants": len(records),
                "pid": os.getpid(),
            },
        )
    except Exception as exc:
        _write_json(
            output / "failure.json",
            {
                "schema": "aion.phase-two.p2-3b.schedule-failure",
                "schema_version": "1.0.0",
                "status": "failed_visible",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        _write_json(
            output / "campaign_status.json",
            {"state": "failed_visible", "stage": "audit", "pid": os.getpid()},
        )
        raise
    finally:
        lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
