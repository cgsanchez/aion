#!/usr/bin/env python3
"""Measure the accepted P2-2 NH3 exact-Wilson runtime without optimizing it."""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import traceback
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np

from aion.config import (
    BackendKind,
    MetadataConfig,
    OutputConfig,
    WilsonSimulationConfig,
    dumps_config,
    loads_config,
)
from aion.electromagnetism import (
    AffineVectorFieldVariation,
    GaussianScalarGaugeVariation,
)
from aion.electronic_structure import (
    AOGridPolicy,
    AOPruningKind,
    DependencyVersions,
    ExactWilsonDynamicSample,
    RIMetricRankPolicy,
    evaluate_exact_wilson_charge,
    evaluate_exact_wilson_power,
    evaluate_nonlinear_density_pure_gauge_ward,
    evaluate_nonlinear_weak_continuity,
    evaluate_nonlinear_weak_current_pairing,
    load_wilson_stationary_state,
    prepare_ao_quadrature,
    prepare_exact_wilson_dynamic_spatial_action,
    prepare_exact_wilson_stationary_factory,
)
from aion.io import (
    WilsonCheckpointData,
    WilsonTrajectoryWriter,
    save_wilson_checkpoint,
)
from aion.workflows import BuiltWilsonSimulation, build_simulation, load_reference

_PROFILE_TIME_AU = 0.075


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


def _authenticate_campaign(root: Path) -> dict[str, object]:
    completed_path = root / "completed.json"
    result_path = root / "result.json"
    provenance_path = root / "provenance.json"
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if completed["result_sha256"] != _sha256(result_path):
        raise RuntimeError(f"inherited result hash mismatch: {root}")
    if completed["provenance_sha256"] != _sha256(provenance_path):
        raise RuntimeError(f"inherited provenance hash mismatch: {root}")
    verified = 2
    identity_sha256 = provenance.get("campaign_identity_sha256")
    if identity_sha256 is not None:
        if identity_sha256 != _sha256(root / "campaign_identity.json"):
            raise RuntimeError(f"inherited campaign identity hash mismatch: {root}")
        verified += 1
    for relative, expected in provenance.get("artifacts_sha256", {}).items():
        if expected != _sha256(root / relative):
            raise RuntimeError(f"inherited artifact hash mismatch: {relative}")
        verified += 1
    for stage, expected in provenance.get("stage_records_sha256", {}).items():
        stage_path = root / "stages" / f"{stage}.json"
        if expected != _sha256(stage_path):
            raise RuntimeError(f"inherited stage hash mismatch: {stage}")
        verified += 1
        stage_record = json.loads(stage_path.read_text(encoding="utf-8"))
        for relative, artifact_sha256 in stage_record.get("artifacts_sha256", {}).items():
            if artifact_sha256 != _sha256(root / relative):
                raise RuntimeError(f"inherited stage artifact hash mismatch: {relative}")
            verified += 1
    return {
        "root": str(root),
        "completed_sha256": _sha256(completed_path),
        "result_sha256": _sha256(result_path),
        "provenance_sha256": _sha256(provenance_path),
        "verified_hashes": verified,
    }


def _authenticate_accepted_review(
    repo: Path,
    inherited: dict[str, dict[str, object]],
) -> dict[str, object]:
    path = repo / "docs/reviews/phase_two_p2_2_review_20260927.json"
    review = json.loads(path.read_text(encoding="utf-8"))
    if review.get("gate") != "P2-2" or review.get("decision") != "accepted":
        raise RuntimeError("P2-2 accepted review record is unavailable")
    evidence = review["evidence"]
    mappings = (
        ("base", "base"),
        ("refinement", "refinement"),
    )
    for inherited_name, evidence_prefix in mappings:
        record = inherited[inherited_name]
        for suffix in ("result_sha256", "provenance_sha256", "completed_sha256"):
            key = f"{evidence_prefix}_{suffix}"
            if evidence[key] != record[suffix]:
                raise RuntimeError(
                    f"P2-2 review disagrees with inherited {inherited_name} {suffix}"
                )
    for path_key, hash_key in (
        ("candidate_review", "candidate_review_sha256"),
        ("review_entry_point", "review_entry_point_sha256"),
        ("execution_matrix", "execution_matrix_sha256"),
        ("controlling_plan", "controlling_plan_sha256"),
    ):
        reviewed_path = repo / evidence[path_key]
        if evidence[hash_key] != _sha256(reviewed_path):
            raise RuntimeError(f"P2-2 review dependency hash mismatch: {reviewed_path}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "accepted_commit": review["accepted_commit"],
    }


def _gpu_memory(backend: Any) -> dict[str, int]:
    if backend.kind is not BackendKind.GPU:
        return {}
    xp = backend.namespace
    free_bytes, total_bytes = xp.cuda.runtime.memGetInfo()
    pool = xp.get_default_memory_pool()
    pinned = xp.get_default_pinned_memory_pool()
    return {
        "device_free_bytes": int(free_bytes),
        "device_total_bytes": int(total_bytes),
        "device_pool_used_bytes": int(pool.used_bytes()),
        "device_pool_total_bytes": int(pool.total_bytes()),
        "pinned_pool_free_blocks": int(pinned.n_free_blocks()),
    }


class _Measurements:
    def __init__(self, backend: Any | None = None) -> None:
        self.backend = backend
        self.records: list[dict[str, object]] = []

    def measure[T](
        self,
        name: str,
        operation: Callable[[], T],
        *,
        details: dict[str, object] | None = None,
    ) -> T:
        if self.backend is not None:
            self.backend.synchronize()
        memory_before = {} if self.backend is None else _gpu_memory(self.backend)
        started = perf_counter()
        value = operation()
        if self.backend is not None:
            self.backend.synchronize()
        elapsed = perf_counter() - started
        memory_after = {} if self.backend is None else _gpu_memory(self.backend)
        record: dict[str, object] = {
            "name": name,
            "elapsed_seconds": elapsed,
            "process_peak_rss_bytes": (
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
            ),
            "gpu_memory_before": memory_before,
            "gpu_memory_after": memory_after,
        }
        if details:
            record["details"] = details
        self.records.append(record)
        print(f"completed {name}: {elapsed:.6f} s", flush=True)
        return value

    def annotate_last(self, **values: object) -> None:
        details = cast(dict[str, object], self.records[-1].setdefault("details", {}))
        details.update(values)


class _TransferCounter:
    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.counts = {
            "to_host_calls": 0,
            "to_host_bytes": 0,
            "to_device_calls": 0,
            "to_device_bytes": 0,
            "scalar_to_host_calls": 0,
            "scalar_to_host_bytes": 0,
        }
        self._original_to_host: Callable[..., Any] | None = None
        self._original_asarray: Callable[..., Any] | None = None
        self._original_scalar: Callable[..., Any] | None = None

    def __enter__(self) -> _TransferCounter:
        self._original_to_host = self.backend.to_host
        self._original_asarray = self.backend.asarray
        self._original_scalar = self.backend.scalar_to_float

        def to_host(value: object) -> np.ndarray:
            assert self._original_to_host is not None
            result = cast(np.ndarray, self._original_to_host(value))
            self.counts["to_host_calls"] += 1
            self.counts["to_host_bytes"] += int(result.nbytes)
            return result

        def asarray(value: object, *, dtype: Any | None = None) -> Any:
            assert self._original_asarray is not None
            result = self._original_asarray(value, dtype=dtype)
            self.counts["to_device_calls"] += 1
            self.counts["to_device_bytes"] += int(getattr(result, "nbytes", 0))
            return result

        def scalar_to_float(value: object) -> float:
            assert self._original_scalar is not None
            result = float(self._original_scalar(value))
            self.counts["scalar_to_host_calls"] += 1
            self.counts["scalar_to_host_bytes"] += int(getattr(value, "nbytes", 8))
            return result

        self.backend.to_host = to_host
        self.backend.asarray = asarray
        self.backend.scalar_to_float = scalar_to_float
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._original_to_host is not None
        assert self._original_asarray is not None
        assert self._original_scalar is not None
        self.backend.to_host = self._original_to_host
        self.backend.asarray = self._original_asarray
        self.backend.scalar_to_float = self._original_scalar


@contextmanager
def _instrument_action_evaluations(
    backend: Any,
) -> Iterator[dict[str, float | int]]:
    counter: dict[str, float | int] = {"calls": 0, "seconds": 0.0}
    original = ExactWilsonDynamicSample.evaluate

    def wrapped(self: ExactWilsonDynamicSample, density: object) -> Any:
        backend.synchronize()
        started = perf_counter()
        result = original(self, density)
        backend.synchronize()
        counter["calls"] = int(counter["calls"]) + 1
        counter["seconds"] = float(counter["seconds"]) + perf_counter() - started
        return result

    ExactWilsonDynamicSample.evaluate = wrapped  # type: ignore[assignment]
    try:
        yield counter
    finally:
        ExactWilsonDynamicSample.evaluate = original  # type: ignore[method-assign]


def _scalar(backend: Any, value: object) -> float:
    return float(np.asarray(backend.to_host(value)).real)


def _profile(
    output: Path,
    base: Path,
    measurements: _Measurements,
) -> tuple[dict[str, object], tuple[Path, ...]]:
    config_path = base / "nh3/exact_gpu.toml"
    resolved = loads_config(config_path.read_text(encoding="utf-8"))
    if not isinstance(resolved.config, WilsonSimulationConfig):
        raise RuntimeError("P2-2 NH3 GPU input is not a Wilson simulation")
    config = replace(
        resolved.config,
        output=OutputConfig(output / "unused_production_output", resolved.config.output.schedules),
        metadata=MetadataConfig(label="phase-two-p2-3-unoptimized-profile"),
    )
    if config.backend.kind is not BackendKind.GPU:
        raise RuntimeError("P2-3A requires the accepted physical-GPU realization")
    profile_config = output / "profile_input.toml"
    profile_config.write_text(dumps_config(config), encoding="utf-8")
    reference_path = base / "nh3/reference.h5"
    stationary_path = base / "nh3/stationary.h5"
    reference = load_reference(reference_path)
    stationary = load_wilson_stationary_state(stationary_path)

    build_started = perf_counter()
    simulation = build_simulation(
        config,
        reference,
        stationary_state=stationary,
        original_toml=dumps_config(config),
    )
    if not isinstance(simulation, BuiltWilsonSimulation):
        raise RuntimeError("P2-3 profile reconstructed the wrong runtime")
    simulation.quadrature.backend.synchronize()
    build_elapsed = perf_counter() - build_started
    measurements.backend = simulation.quadrature.backend
    measurements.records.append(
        {
            "name": "runtime.build_complete",
            "elapsed_seconds": build_elapsed,
            "process_peak_rss_bytes": (
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
            ),
            "gpu_memory_after": _gpu_memory(simulation.quadrature.backend),
        }
    )
    print(f"completed runtime.build_complete: {build_elapsed:.6f} s", flush=True)

    backend = simulation.quadrature.backend
    backend.assert_resident(simulation.density, name="P2-3 profile density")
    if config.numerics.grid_kind.value == "reference":
        grid_policy = AOGridPolicy.reference()
    else:
        if config.numerics.grid_level is None:
            raise RuntimeError("qualification grid requires an explicit level")
        grid_policy = AOGridPolicy.qualification(
            config.numerics.grid_level,
            pruning=AOPruningKind(config.numerics.grid_pruning.value),
        )
    profile_quadrature = measurements.measure(
        "construction.ao_quadrature",
        partial(
            prepare_ao_quadrature,
            reference,
            config.backend,
            grid_policy=grid_policy,
            block_size=config.numerics.block_size,
            memory_budget_bytes=config.numerics.memory_budget_bytes,
        ),
    )
    rank_policy = RIMetricRankPolicy(
        relative_threshold=config.numerics.ri_relative_threshold,
        absolute_threshold=config.numerics.ri_absolute_threshold,
        maximum_rank=config.numerics.ri_maximum_rank,
    )
    profile_factory = measurements.measure(
        "construction.ri_hartree_metric_cache_and_pbe_factory",
        partial(
            prepare_exact_wilson_stationary_factory,
            profile_quadrature,
            auxiliary_basis=config.numerics.auxiliary_basis,
            functional=reference.config.electronic_structure.functional,
            rank_policy=rank_policy,
        ),
    )
    source = simulation.source_provider.sample(_PROFILE_TIME_AU)
    spatial = measurements.measure(
        "construction.exact_spatial_one_electron_and_ri_action",
        partial(
            prepare_exact_wilson_dynamic_spatial_action,
            profile_factory,
            source.gauge,
        ),
    )
    measurements.measure(
        "construction.temporal_connection_from_prepared_spatial",
        partial(spatial.temporal_sample, source),
    )
    del spatial, profile_factory, profile_quadrature
    gc.collect()
    backend.namespace.get_default_memory_pool().free_all_blocks()
    sample = measurements.measure(
        "cache.dynamic_sample_first_request",
        lambda: simulation.dynamic_cache.sample(_PROFILE_TIME_AU),
    )
    cached = measurements.measure(
        "cache.dynamic_sample_cached_request",
        lambda: simulation.dynamic_cache.sample(_PROFILE_TIME_AU),
    )
    if cached is not sample:
        raise RuntimeError("dynamic sample cache did not return the retained sample")

    density = simulation.density
    evaluation = measurements.measure(
        "action.complete_nonlinear_evaluation",
        lambda: sample.evaluate(density),
    )
    measurements.annotate_last(
        molecular_energy_au=_scalar(backend, evaluation.action.energy_molecular_total_au)
    )

    hartree = measurements.measure(
        "action.ri_hartree_contraction",
        lambda: sample.model.hartree_action.evaluate(density),
    )
    measurements.annotate_last(energy_au=_scalar(backend, hartree.energy))
    del hartree

    gga_evaluator = sample.model.gga_evaluator
    if gga_evaluator is None:
        raise RuntimeError("accepted NH3 profile is not a GGA calculation")
    with _TransferCounter(backend) as gga_transfers:
        gga = measurements.measure(
            "action.pbe_grid_evaluation",
            lambda: gga_evaluator.evaluate(density, sample.model.gauge),
        )
    measurements.annotate_last(
        energy_au=_scalar(backend, gga.energy),
        explicit_transfers=gga_transfers.counts,
    )
    del gga

    power = measurements.measure(
        "observable.source_power_complete",
        lambda: evaluate_exact_wilson_power(evaluation, density),
    )
    measurements.annotate_last(
        source_power_au=_scalar(backend, power.source_power_au),
        identity_residual_au=_scalar(backend, power.power_identity_residual_au),
    )

    current_components: list[float] = []
    cartesian_offsets = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    for axis, offset in zip("xyz", cartesian_offsets, strict=True):
        variation = AffineVectorFieldVariation(
            offset_au=offset,
            origin_au=sample.source.origin_au,
        )

        def current_operation(
            selected: AffineVectorFieldVariation = variation,
        ) -> Any:
            return evaluate_nonlinear_weak_current_pairing(
                sample.model,
                sample.one_electron,
                density,
                power.velocity_density,
                selected,
            )

        with _TransferCounter(backend) as current_transfers:
            pairing = measurements.measure(
                f"observable.uniform_current_{axis}",
                current_operation,
            )
        component = _scalar(backend, pairing.on_shell_pairing)
        current_components.append(component)
        measurements.annotate_last(
            current_component_au=component,
            explicit_transfers=current_transfers.counts,
        )
        del pairing

    charge = measurements.measure(
        "observable.charge",
        lambda: evaluate_exact_wilson_charge(sample.model, density),
    )
    measurements.annotate_last(
        metric_particle_number=_scalar(backend, charge.metric_particle_number)
    )
    del charge

    identity_variation = GaussianScalarGaugeVariation(
        amplitude=0.37,
        center_au=(0.11, -0.17, 0.23),
        exponent_au_inverse2=0.41,
    )
    ward = measurements.measure(
        "observable.ward_identity",
        lambda: evaluate_nonlinear_density_pure_gauge_ward(
            sample.model,
            sample.one_electron,
            density,
            power.velocity_density,
            identity_variation,
        ),
    )
    measurements.annotate_last(
        residual_abs=_scalar(backend, backend.namespace.abs(ward.total_ward_residual))
    )
    del ward
    continuity = measurements.measure(
        "observable.weak_continuity",
        lambda: evaluate_nonlinear_weak_continuity(
            sample.model,
            sample.one_electron,
            density,
            identity_variation,
        ),
    )
    measurements.annotate_last(
        finite_region_residual_abs=_scalar(
            backend,
            backend.namespace.abs(continuity.finite_region_residual),
        )
    )
    del continuity

    with _instrument_action_evaluations(backend) as action_counter:
        step = measurements.measure("propagation.one_gauss_magnus_step", simulation.step)
    step_elapsed = cast(float, measurements.records[-1]["elapsed_seconds"])
    action_seconds = float(action_counter["seconds"])
    measurements.annotate_last(
        nonlinear_iterations=step.diagnostics.nonlinear_iterations,
        nonlinear_residual=step.diagnostics.nonlinear_residual,
        action_evaluation_calls=int(action_counter["calls"]),
        action_evaluation_seconds=action_seconds,
        dense_link_and_iteration_upper_bound_seconds=max(0.0, step_elapsed - action_seconds),
    )

    gauss_minus_power = measurements.measure(
        "propagation.gauss_minus_power",
        lambda: evaluate_exact_wilson_power(
            step.gauss_minus_evaluation,
            step.gauss_minus_contravariant_density,
        ),
    )
    gauss_plus_power = measurements.measure(
        "propagation.gauss_plus_power",
        lambda: evaluate_exact_wilson_power(
            step.gauss_plus_evaluation,
            step.gauss_plus_contravariant_density,
        ),
    )
    gauss_powers = (
        _scalar(backend, gauss_minus_power.source_power_au),
        _scalar(backend, gauss_plus_power.source_power_au),
    )
    work_increment = 0.5 * config.propagation.time_grid.step_au * sum(gauss_powers)

    with _TransferCounter(backend) as endpoint_transfers:
        endpoint = measurements.measure(
            "observable.monolithic_endpoint_with_energy_and_identities",
            lambda: simulation.observe_endpoint(
                include_energy=True,
                include_identities=True,
            ),
        )
    measurements.annotate_last(explicit_transfers=endpoint_transfers.counts)
    endpoint_current_host = backend.to_host(endpoint.uniform_source_current_au)
    endpoint_source = simulation.dynamic_cache.sample(simulation.current_time_au).source

    host_density = measurements.measure(
        "transfer.final_density_to_host",
        lambda: backend.to_host(simulation.density),
    )
    measurements.annotate_last(bytes=int(host_density.nbytes))

    io_root = output / "io_microbenchmark"
    checkpoint_path = io_root / "checkpoint.h5"
    checkpoint = WilsonCheckpointData(
        run_id="phase-two-p2-3-profile",
        simulation_config=config,
        original_toml=dumps_config(config),
        reference_artifact_path=reference_path,
        stationary_state_artifact_path=stationary_path,
        source_fingerprint_sha256=simulation.source_provider.fingerprint_sha256,
        global_step=simulation.boundary_index,
        contravariant_density=host_density,
        accumulated_source_work_au=work_increment,
        initial_molecular_energy_au=stationary.energies.molecular_total_au,
        observer_schedule_state=(),
    )
    checkpoint_sha256 = measurements.measure(
        "io.transactional_checkpoint",
        lambda: save_wilson_checkpoint(checkpoint, checkpoint_path),
    )
    measurements.annotate_last(
        artifact_bytes=checkpoint_path.stat().st_size,
        artifact_sha256=checkpoint_sha256,
    )

    writer_root = io_root / "trajectory"
    writer = measurements.measure(
        "io.trajectory_initialize",
        lambda: WilsonTrajectoryWriter(
            writer_root,
            run_id=uuid.uuid4().hex,
            config=config,
            original_toml=dumps_config(config),
            reference_artifact_path=reference_path,
            stationary_state_artifact_path=stationary_path,
            source_fingerprint_sha256=simulation.source_provider.fingerprint_sha256,
            provenance_json="{}",
        ),
    )

    def append_representative_records() -> None:
        writer.append_source(step=1, sample=endpoint_source)
        writer.append_series(
            "current/uniform_source",
            step=1,
            time_au=simulation.current_time_au,
            value=endpoint_current_host,
            unit="electron_per_atomic_unit_of_time_times_bohr",
            physical_dimension="electric_current",
        )
        writer.append_interval_work(
            step=0,
            gauss_times_au=(step.gauss_minus_time_au, step.gauss_plus_time_au),
            gauss_power_au=gauss_powers,
            increment_au=work_increment,
            accumulated_au=work_increment,
        )
        writer.append_diagnostics(
            step=1,
            time_au=simulation.current_time_au,
            values={
                "nonlinear_residual": step.diagnostics.nonlinear_residual,
                "cross_metric_residual": step.diagnostics.cross_metric_residual,
            },
        )
        writer.append_snapshot(
            step=1,
            time_au=simulation.current_time_au,
            density=host_density,
        )

    measurements.measure(
        "io.trajectory_representative_append",
        append_representative_records,
    )
    measurements.measure("io.trajectory_flush", writer.flush)
    writer.set_summary(final_step=1, accumulated_source_work_au=work_increment)
    trajectory = measurements.measure("io.trajectory_finalize", writer.finalize)
    measurements.annotate_last(
        artifact_bytes=trajectory.path.stat().st_size,
        artifact_sha256=trajectory.sha256,
    )

    cache = simulation.dynamic_cache.statistics
    summary = {
        "schema": "aion.phase-two.p2-3.profile-result",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "profile_kind": "unoptimized_synchronized_component_baseline",
        "system": "NH3 G_DZ/cc-pVDZ/PBE accepted P2-2 bridge",
        "backend": "physical_gpu",
        "profile_time_au": _PROFILE_TIME_AU,
        "measurements": measurements.records,
        "cache_statistics": {
            "spatial_entries": cache.spatial_entries,
            "sample_entries": cache.sample_entries,
            "maximum_entries": cache.maximum_entries,
            "spatial_hits": cache.spatial_hits,
            "spatial_misses": cache.spatial_misses,
            "sample_hits": cache.sample_hits,
            "sample_misses": cache.sample_misses,
        },
        "scientific_checks": {
            "density_remained_device_resident": backend.is_resident(simulation.density),
            "step_nonlinear_residual": step.diagnostics.nonlinear_residual,
            "endpoint_power_identity_residual_au": _scalar(
                backend,
                endpoint.power_identity_residual_au,
            ),
            "endpoint_current_au": [
                _scalar(backend, value) for value in endpoint.uniform_source_current_au
            ],
        },
        "interpretation_boundary": (
            "Synchronized component timings diagnose cost but are not additive production "
            "wall-time predictions. No optimization or physics change is made by this run."
        ),
    }
    result_path = output / "result.json"
    _write_json(result_path, summary)
    return summary, (
        profile_config,
        checkpoint_path,
        trajectory.path,
        result_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--refinement-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    base = arguments.base_root.expanduser().resolve()
    refinement = arguments.refinement_root.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "campaign.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another P2-3 profile owns this output root") from None
    if (output / "completed.json").exists():
        raise FileExistsError("refusing to overwrite a completed P2-3 profile")
    if os.environ.get("AION_GPU_LAUNCHER") != "1":
        raise RuntimeError("P2-3A must run through tools/gpu-python")

    repo = Path(__file__).resolve().parents[1]
    try:
        _write_json(
            output / "campaign_status.json",
            {"state": "running", "stage": "authenticate", "pid": os.getpid()},
        )
        inherited = {
            "base": _authenticate_campaign(base),
            "refinement": _authenticate_campaign(refinement),
        }
        accepted_review = _authenticate_accepted_review(repo, inherited)
        identity = {
            "schema": "aion.phase-two.p2-3.profile-identity",
            "schema_version": "1.0.0",
            "git_head": _git(repo, "rev-parse", "HEAD"),
            "git_branch": _git(repo, "branch", "--show-current"),
            "driver_sha256": _sha256(Path(__file__).resolve()),
            "inherited_evidence": inherited,
            "accepted_review": accepted_review,
        }
        _write_json(output / "campaign_identity.json", identity)
        _write_json(
            output / "campaign_status.json",
            {"state": "running", "stage": "profile", "pid": os.getpid()},
        )
        measurements = _Measurements()
        result, artifacts = _profile(output, base, measurements)
        result_path = output / "result.json"
        provenance = {
            "schema": "aion.phase-two.p2-3.profile-provenance",
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
            "thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "AION_HOST_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "process_peak_rss_bytes": (
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
            ),
            "campaign_identity_sha256": _sha256(output / "campaign_identity.json"),
            "artifacts_sha256": {
                str(path.relative_to(output)): _sha256(path)
                for path in artifacts
                if path != result_path
            },
        }
        provenance_path = output / "provenance.json"
        _write_json(provenance_path, provenance)
        _write_json(
            output / "completed.json",
            {
                "schema": "aion.phase-two.p2-3.profile-completed",
                "schema_version": "1.0.0",
                "status": result["status"],
                "result_sha256": _sha256(result_path),
                "provenance_sha256": _sha256(provenance_path),
            },
        )
        _write_json(
            output / "campaign_status.json",
            {"state": "completed", "stage": "profile", "pid": os.getpid()},
        )
    except Exception as exc:
        _write_json(
            output / "failure.json",
            {
                "schema": "aion.phase-two.p2-3.profile-failure",
                "schema_version": "1.0.0",
                "status": "failed_visible",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        _write_json(
            output / "campaign_status.json",
            {"state": "failed_visible", "stage": "profile", "pid": os.getpid()},
        )
        raise
    finally:
        lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
