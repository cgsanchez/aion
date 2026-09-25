#!/usr/bin/env python3
"""Qualify the reusable P2-1 exact-Wilson workflow on H3+.

The accepted NQ4/NQ6 campaign remains an independent oracle.  This program
uses only public reusable workflow entry points for the integrated path and
the retained batch propagator for a separately constructed driven oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import traceback
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aion.config import (
    AffineElectromagneticSourceConfig,
    BackendConfig,
    BackendKind,
    ElectromagneticOrigin,
    ExactWilsonActionConfig,
    FixedTimeGrid,
    ObservableSchedules,
    OutputConfig,
    RationalApproximation,
    ReferenceConfig,
    ReferenceLinkConfig,
    Sin2VectorPotentialPulseConfig,
    StepSchedule,
    WilsonGridKind,
    WilsonGridPruning,
    WilsonIntegratorKind,
    WilsonMagneticGaugeKind,
    WilsonNumericsConfig,
    WilsonPropagationConfig,
    WilsonSimulationConfig,
    WilsonStationaryBranch,
    WilsonStationaryConfig,
    WilsonStationaryOutputConfig,
    WilsonStationaryPolicyConfig,
    WilsonStationaryStateLinkConfig,
    ZeroSourceConfig,
    dumps_config,
    load_config,
    loads_config,
)
from aion.electromagnetism import build_affine_electromagnetic_source
from aion.electronic_structure import (
    DependencyVersions,
    load_wilson_stationary_state,
    prepare_exact_wilson_dynamic_sample,
)
from aion.errors import RunCancelledError
from aion.io import load_wilson_checkpoint, load_wilson_trajectory
from aion.propagation import (
    NonlinearGaussMagnusPolicy,
    propagate_nonlinear_contravariant_density,
)
from aion.workflows import (
    BuiltWilsonSimulation,
    RunControl,
    build_simulation,
    load_reference,
    prepare_reference,
    prepare_wilson_stationary_state,
    resume,
    run,
)

_CAMPAIGN_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "chapter13_wilson_adiabatic_qualification"
)
_NQ4_ROOT = _CAMPAIGN_ROOT / "nq4_stationary_20260921T232232Z_6197d6828bb3"
_NQ6_ROOT = _CAMPAIGN_ROOT / "nq6_dynamics_20260922T205959Z_a197fbc55b62"
_GAUGE_ORIGIN = (0.17, -0.31, 0.23)
_STATIC_FIELD = (0.0, 0.0, 0.03)
_INTERVALS = 8
_STEP_AU = 0.25
_FINAL_TIME_AU = _INTERVALS * _STEP_AU
_NQ6_RUN_ID = "stationary_static_b_kohn_sham_lda_n8_tol1e-12_timestep"


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


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _relative(candidate: object, reference: object) -> float:
    candidate_array = np.asarray(candidate)
    reference_array = np.asarray(reference)
    return float(
        np.linalg.norm(candidate_array - reference_array)
        / max(1.0, float(np.linalg.norm(reference_array)))
    )


def _timed[T](stages: dict[str, float], name: str, operation: Callable[[], T]) -> T:
    started = perf_counter()
    result = operation()
    stages[name] = perf_counter() - started
    print(f"completed {name}: {stages[name]:.3f} s", flush=True)
    return result


def _authenticate_accepted(root: Path) -> dict[str, str]:
    completed_path = root / "completed.json"
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    result_path = root / "result.json"
    provenance_path = root / "provenance.json"
    if completed["result_sha256"] != _sha256(result_path):
        raise RuntimeError(f"accepted result hash changed: {root}")
    if completed["provenance_sha256"] != _sha256(provenance_path):
        raise RuntimeError(f"accepted provenance hash changed: {root}")
    return {
        "root": str(root),
        "result_sha256": completed["result_sha256"],
        "provenance_sha256": completed["provenance_sha256"],
        "completed_sha256": _sha256(completed_path),
    }


def _round_trip[T](config: T, path: Path) -> T:
    text = dumps_config(config)  # type: ignore[arg-type]
    path.write_text(text, encoding="utf-8")
    resolved = loads_config(text).config
    if resolved != config:
        raise RuntimeError(f"strict TOML round trip changed {path.name}")
    return resolved  # type: ignore[return-value]


def _reference_config(output: Path) -> ReferenceConfig:
    resolved = load_config(_NQ4_ROOT / "h3plus.reference.toml").config
    if not isinstance(resolved, ReferenceConfig):
        raise RuntimeError("accepted H3+ reference input has the wrong schema")
    return replace(
        resolved,
        molecule=replace(
            resolved.molecule,
            electromagnetic_origin=ElectromagneticOrigin(_GAUGE_ORIGIN),
        ),
        output=replace(resolved.output, artifact_path=output / "reference.h5"),
        metadata=replace(
            resolved.metadata,
            label="phase-two-p2-1-h3plus",
            timestamp_utc=None,
            host=None,
        ),
    )


def _numerics() -> WilsonNumericsConfig:
    return WilsonNumericsConfig(
        grid_kind=WilsonGridKind.QUALIFICATION,
        grid_level=4,
        grid_pruning=WilsonGridPruning.NONE,
        block_size=2048,
        dynamic_cache_entries=4,
        memory_budget_bytes=None,
        auxiliary_basis="weigend",
        ri_relative_threshold=0.0,
        ri_absolute_threshold=1.0e-7,
        ri_maximum_rank=None,
    )


def _static_source() -> AffineElectromagneticSourceConfig:
    return AffineElectromagneticSourceConfig(
        electric=ZeroSourceConfig(),
        electric_field_origin_offset_au=(0.0, 0.0, 0.0),
        magnetic_field_reference_au=_STATIC_FIELD,
        magnetic_field_rate_au=(0.0, 0.0, 0.0),
        magnetic_reference_time_au=0.0,
        magnetic_gauge=WilsonMagneticGaugeKind.SYMMETRIC,
    )


def _pulse_source() -> AffineElectromagneticSourceConfig:
    return replace(
        _static_source(),
        electric=Sin2VectorPotentialPulseConfig(
            peak_electric_field_au=0.005,
            angular_frequency_au=np.pi,
            cycles=1,
            polarization=(1.0, 0.0, 0.0),
            start_time_au=0.0,
            carrier_phase_rad=0.0,
            require_zero_impulse=True,
        ),
    )


def _stationary_config(
    output: Path,
    reference_fingerprint: str,
) -> WilsonStationaryConfig:
    return WilsonStationaryConfig(
        reference=ReferenceLinkConfig(reference_fingerprint, output / "reference.h5"),
        action=ExactWilsonActionConfig(WilsonStationaryBranch.KOHN_SHAM_LDA),
        numerics=_numerics(),
        source=_static_source(),
        source_time_au=0.0,
        stationary=WilsonStationaryPolicyConfig(
            maximum_iterations=160,
            density_tolerance=2.0e-12,
            orbital_tolerance=2.0e-12,
            energy_tolerance_au=2.0e-13,
            damping=0.5,
            diis_start_iteration=2,
            diis_space=8,
        ),
        backend=BackendConfig(),
        output=WilsonStationaryOutputConfig(output / "stationary.h5"),
    )


def _schedules() -> ObservableSchedules:
    endpoints = StepSchedule(every=0, include_initial=True, include_final=True)
    return ObservableSchedules(
        dipole_current=endpoints,
        energy=endpoints,
        diagnostics=endpoints,
        source=endpoints,
        checkpoints=endpoints,
        matrix_snapshots=StepSchedule(
            every=0,
            include_initial=False,
            include_final=False,
        ),
    )


def _simulation_config(
    output: Path,
    *,
    reference_fingerprint: str,
    state_fingerprint: str,
    state_path: Path,
    source: AffineElectromagneticSourceConfig,
    backend: BackendConfig,
    run_directory: str,
) -> WilsonSimulationConfig:
    return WilsonSimulationConfig(
        reference=ReferenceLinkConfig(reference_fingerprint, output / "reference.h5"),
        stationary_state=WilsonStationaryStateLinkConfig(state_fingerprint, state_path),
        action=ExactWilsonActionConfig(WilsonStationaryBranch.KOHN_SHAM_LDA),
        numerics=_numerics(),
        source=source,
        propagation=WilsonPropagationConfig(
            time_grid=FixedTimeGrid(0.0, _STEP_AU, _INTERVALS),
            integrator=WilsonIntegratorKind.NONLINEAR_GAUSS_MAGNUS,
            rational_approximation=RationalApproximation.PADE_22,
            nonlinear_tolerance=1.0e-12,
            maximum_iterations=80,
        ),
        backend=backend,
        output=OutputConfig(output / run_directory, schedules=_schedules()),
    )


def _accepted_stationary_row() -> dict[str, Any]:
    result = json.loads((_NQ4_ROOT / "result.json").read_text(encoding="utf-8"))
    rows = result["h3plus_main_scan"]["rows"]
    matches = [row for row in rows if row["branch"] == "kohn_sham_lda" and row["field_au"] == 0.03]
    if len(matches) != 1:
        raise RuntimeError("accepted NQ4 B=0.03 LDA row is not unique")
    return matches[0]


def _direct_pulse(
    simulation: BuiltWilsonSimulation,
    initial_density: np.ndarray,
) -> Any:
    samples: dict[float, Any] = {}

    def sample(time_au: float) -> Any:
        time = float(time_au)
        if time not in samples:
            samples[time] = prepare_exact_wilson_dynamic_sample(
                simulation.factory,
                simulation.source_provider.sample(time),
                WilsonStationaryBranch.KOHN_SHAM_LDA,
            )
        return samples[time]

    return propagate_nonlinear_contravariant_density(
        initial_density,
        initial_time_au=0.0,
        interval_au=_STEP_AU,
        intervals=_INTERVALS,
        metric_provider=lambda time: sample(time).one_electron.metric,
        eom_provider=lambda time, density: sample(time).evaluate(density).triple,
        backend=simulation.quadrature.backend,
        policy=NonlinearGaussMagnusPolicy(tolerance=1.0e-12, maximum_iterations=80),
    )


def _step_all(simulation: BuiltWilsonSimulation, *, require_residency: bool) -> None:
    backend = simulation.quadrature.backend
    if require_residency:
        backend.assert_resident(simulation.density, name="initial H3+ Wilson density")
    while simulation.boundary_index < _INTERVALS:
        step = simulation.step()
        if require_residency:
            backend.assert_resident(step.contravariant_density, name="H3+ Wilson density")
            backend.assert_resident(step.link, name="H3+ Wilson link")
    if require_residency:
        observation = simulation.observe_endpoint(include_energy=True)
        backend.assert_resident(
            observation.uniform_source_current_au,
            name="H3+ Wilson source current",
        )
        backend.assert_resident(
            observation.electronic_dipole_au,
            name="H3+ Wilson dipole",
        )


def _static_comparisons(
    stationary_state: Any,
    static_final_mixed: np.ndarray,
    static_final_energy: float,
    full_checkpoint: Any,
    resumed_checkpoint: Any,
) -> dict[str, object]:
    accepted_states = np.load(_NQ4_ROOT / "stationary_states.npz", allow_pickle=False)
    accepted_initial_density = np.asarray(accepted_states["h3plus_b0p030_kohn_sham_lda_density"])
    accepted_stationary = _accepted_stationary_row()
    nq6_arrays = np.load(_NQ6_ROOT / "trajectories.npz", allow_pickle=False)
    accepted_final_mixed = np.asarray(nq6_arrays[f"{_NQ6_RUN_ID}_final_mixed_density"])
    accepted_final_energy = float(nq6_arrays[f"{_NQ6_RUN_ID}_energies_au"][-1])
    return {
        "stationary_density_residual": _relative(
            stationary_state.contravariant_density,
            accepted_initial_density,
        ),
        "stationary_energy_residual": abs(
            stationary_state.energies.molecular_total_au
            - float(accepted_stationary["energy_molecular_total_au"])
        ),
        "nq6_final_density_residual": _relative(
            static_final_mixed,
            accepted_final_mixed,
        ),
        "nq6_final_energy_residual": abs(static_final_energy - accepted_final_energy),
        "restart_density_exact": np.array_equal(
            full_checkpoint.contravariant_density,
            resumed_checkpoint.contravariant_density,
        ),
        "restart_work_residual": abs(
            full_checkpoint.accumulated_source_work_au
            - resumed_checkpoint.accumulated_source_work_au
        ),
    }


def _fresh_static_evidence(
    output: Path,
    stages: dict[str, float],
) -> dict[str, object]:
    reference_config = _round_trip(
        _reference_config(output),
        output / "reference.toml",
    )
    reference = _timed(
        stages,
        "prepare_reference",
        lambda: prepare_reference(reference_config),
    )
    reference.save(output / "reference.h5")

    stationary_config = _round_trip(
        _stationary_config(output, reference.fingerprint_sha256),
        output / "stationary.toml",
    )
    stationary_state = _timed(
        stages,
        "prepare_stationary_state",
        lambda: prepare_wilson_stationary_state(stationary_config, reference),
    )
    stationary_state.save()
    reloaded_state = load_wilson_stationary_state(output / "stationary.h5")
    state_round_trip_exact = (
        reloaded_state.fingerprint_sha256 == stationary_state.fingerprint_sha256
        and np.array_equal(
            reloaded_state.contravariant_density,
            stationary_state.contravariant_density,
        )
    )

    static_config = _round_trip(
        _simulation_config(
            output,
            reference_fingerprint=reference.fingerprint_sha256,
            state_fingerprint=stationary_state.fingerprint_sha256,
            state_path=output / "stationary.h5",
            source=_static_source(),
            backend=BackendConfig(),
            run_directory="static_full",
        ),
        output / "static_simulation.toml",
    )
    static_simulation = build_simulation(
        static_config,
        reference,
        stationary_state=stationary_state,
    )
    _timed(stages, "static_full_run", lambda: run(static_simulation))
    static_final_density = np.asarray(static_simulation.density)
    static_final_metric = np.asarray(
        static_simulation.dynamic_cache.sample(_FINAL_TIME_AU).one_electron.metric
    )
    static_final_mixed = static_final_density @ static_final_metric
    static_final_energy = float(
        static_simulation.observe_endpoint(include_energy=True).energy.molecular_total_au  # type: ignore[union-attr]
    )

    interrupted_config = replace(
        static_config,
        output=replace(static_config.output, directory=output / "static_interrupted"),
    )
    interrupted = build_simulation(
        interrupted_config,
        reference,
        stationary_state=stationary_state,
    )
    control = RunControl()

    def cancel_at_three(step: int) -> None:
        if step == 3:
            control.request_cancel()

    control.after_accepted_step = cancel_at_three

    def execute_interrupted() -> None:
        try:
            run(interrupted, control=control)
        except RunCancelledError:
            return
        raise RuntimeError("H3+ restart qualification did not cancel at step 3")

    _timed(stages, "static_interrupted_prefix", execute_interrupted)
    interrupted_checkpoint = output / "static_interrupted/checkpoint_00000003.h5"
    _timed(
        stages,
        "static_resumed_suffix",
        lambda: resume(interrupted_checkpoint, output=output / "static_resumed"),
    )
    full_checkpoint = load_wilson_checkpoint(output / "static_full/checkpoint_00000008.h5")
    resumed_checkpoint = load_wilson_checkpoint(output / "static_resumed/checkpoint_00000008.h5")
    comparisons = _static_comparisons(
        stationary_state,
        static_final_mixed,
        static_final_energy,
        full_checkpoint,
        resumed_checkpoint,
    )
    return {
        "reference": reference,
        "stationary_state": stationary_state,
        "state_round_trip_exact": state_round_trip_exact,
        "reused_artifacts": None,
        **comparisons,
    }


def _reused_static_evidence(
    output: Path,
    root: Path,
) -> dict[str, object]:
    root = root.expanduser().resolve()
    required = (
        "failure.json",
        "reference.h5",
        "stationary.h5",
        "static_full/trajectory.h5",
        "static_full/checkpoint_00000008.h5",
        "static_interrupted/checkpoint_00000003.h5",
        "static_resumed/checkpoint_00000008.h5",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"reusable static evidence is incomplete: {missing}")
    reused_hashes = {name: _sha256(root / name) for name in required}
    for name in ("reference.h5", "stationary.h5"):
        shutil.copy2(root / name, output / name)
    for name in ("reference.toml", "stationary.toml", "static_simulation.toml"):
        if (root / name).is_file():
            shutil.copy2(root / name, output / name)

    reference = load_reference(output / "reference.h5")
    source_state = load_wilson_stationary_state(root / "stationary.h5")
    stationary_state = load_wilson_stationary_state(output / "stationary.h5")
    state_round_trip_exact = (
        source_state.fingerprint_sha256 == stationary_state.fingerprint_sha256
        and np.array_equal(
            source_state.contravariant_density,
            stationary_state.contravariant_density,
        )
    )
    full_checkpoint = load_wilson_checkpoint(root / "static_full/checkpoint_00000008.h5")
    resumed_checkpoint = load_wilson_checkpoint(root / "static_resumed/checkpoint_00000008.h5")
    trajectory = load_wilson_trajectory(root / "static_full/trajectory.h5")
    energy_series = trajectory.read_series("energy/molecular_total")
    if int(energy_series.steps[-1]) != _INTERVALS:
        raise RuntimeError("reused static trajectory lacks its final energy endpoint")
    static_final_mixed = np.asarray(full_checkpoint.contravariant_density) @ stationary_state.metric
    comparisons = _static_comparisons(
        stationary_state,
        static_final_mixed,
        float(energy_series.values[-1]),
        full_checkpoint,
        resumed_checkpoint,
    )
    return {
        "reference": reference,
        "stationary_state": stationary_state,
        "state_round_trip_exact": state_round_trip_exact,
        "reused_artifacts": {
            "root": str(root),
            "sha256": reused_hashes,
            "interpretation": (
                "Completed static and restart evidence was authenticated and reused "
                "after the retained attempt failed only on entering the driven oracle."
            ),
        },
        **comparisons,
    }


def _run_qualification(
    output: Path,
    include_gpu: bool,
    reuse_static_root: Path | None,
) -> dict[str, Any]:
    stages: dict[str, float] = {}
    accepted = {
        "NQ4": _authenticate_accepted(_NQ4_ROOT),
        "NQ6": _authenticate_accepted(_NQ6_ROOT),
    }
    static = (
        _fresh_static_evidence(output, stages)
        if reuse_static_root is None
        else _reused_static_evidence(output, reuse_static_root)
    )
    reference: Any = static["reference"]
    stationary_state: Any = static["stationary_state"]
    state_round_trip_exact = bool(static["state_round_trip_exact"])
    stationary_density_residual = float(static["stationary_density_residual"])
    stationary_energy_residual = float(static["stationary_energy_residual"])
    nq6_final_density_residual = float(static["nq6_final_density_residual"])
    nq6_final_energy_residual = float(static["nq6_final_energy_residual"])
    restart_density_exact = bool(static["restart_density_exact"])
    restart_work_residual = float(static["restart_work_residual"])

    pulse_config = replace(stationary_state.config, source=_pulse_source())
    pulse_provider = build_affine_electromagnetic_source(
        pulse_config.source,
        reference.electromagnetic_origin_au,
    )
    pulse_state = replace(
        stationary_state,
        config=pulse_config,
        source_sample=pulse_provider.sample(0.0),
    )
    pulse_state_path = output / "pulse_stationary.h5"
    pulse_state.save(pulse_state_path)
    pulse_simulation_config = _round_trip(
        _simulation_config(
            output,
            reference_fingerprint=reference.fingerprint_sha256,
            state_fingerprint=pulse_state.fingerprint_sha256,
            state_path=pulse_state_path,
            source=_pulse_source(),
            backend=BackendConfig(),
            run_directory="pulse_cpu",
        ),
        output / "pulse_simulation.toml",
    )
    direct_runtime = build_simulation(
        pulse_simulation_config,
        reference,
        stationary_state=pulse_state,
    )
    direct = _timed(
        stages,
        "pulse_direct_batch_oracle",
        lambda: _direct_pulse(direct_runtime, pulse_state.contravariant_density),
    )
    integrated_runtime = build_simulation(
        pulse_simulation_config,
        reference,
        stationary_state=pulse_state,
    )
    _timed(
        stages,
        "pulse_integrated_cpu",
        lambda: _step_all(integrated_runtime, require_residency=False),
    )
    pulse_direct_residual = _relative(
        np.asarray(integrated_runtime.density),
        np.asarray(direct.contravariant_densities[-1]),
    )

    gpu_residual: float | None = None
    gpu_residency_checked = False
    if include_gpu:
        if os.environ.get("AION_GPU_LAUNCHER") != "1":
            raise RuntimeError("--include-gpu must be launched through tools/gpu-python")
        gpu_config = replace(
            pulse_simulation_config,
            backend=BackendConfig(kind=BackendKind.GPU, device_index=0),
            output=replace(pulse_simulation_config.output, directory=output / "pulse_gpu"),
        )
        gpu_runtime = build_simulation(
            gpu_config,
            reference,
            stationary_state=pulse_state,
        )
        _timed(
            stages,
            "pulse_integrated_gpu",
            lambda: _step_all(gpu_runtime, require_residency=True),
        )
        gpu_final = gpu_runtime.quadrature.backend.to_host(gpu_runtime.density)
        gpu_residual = _relative(gpu_final, np.asarray(integrated_runtime.density))
        gpu_residency_checked = True

    thresholds = {
        "stationary_density_relative": 2.0e-10,
        "stationary_energy_absolute_au": 2.0e-10,
        "nq6_final_mixed_density_relative": 3.0e-10,
        "nq6_final_energy_absolute_au": 3.0e-10,
        "pulse_direct_density_relative": 2.0e-12,
        "restart_work_absolute_au": 2.0e-14,
        "cpu_gpu_density_relative": 2.0e-9,
    }
    checks = {
        "stationary_state_round_trip_exact": state_round_trip_exact,
        "stationary_density_matches_nq4": (
            stationary_density_residual <= thresholds["stationary_density_relative"]
        ),
        "stationary_energy_matches_nq4": (
            stationary_energy_residual <= thresholds["stationary_energy_absolute_au"]
        ),
        "static_nq6_final_density_matches": (
            nq6_final_density_residual <= thresholds["nq6_final_mixed_density_relative"]
        ),
        "static_nq6_final_energy_matches": (
            nq6_final_energy_residual <= thresholds["nq6_final_energy_absolute_au"]
        ),
        "uninterrupted_resumed_density_bitwise_equal": restart_density_exact,
        "uninterrupted_resumed_work_matches": (
            restart_work_residual <= thresholds["restart_work_absolute_au"]
        ),
        "driven_integrated_matches_direct_batch": (
            pulse_direct_residual <= thresholds["pulse_direct_density_relative"]
        ),
        "physical_gpu_executed": include_gpu,
        "physical_gpu_residency_checked": gpu_residency_checked,
        "cpu_gpu_driven_density_matches": (
            gpu_residual is not None and gpu_residual <= thresholds["cpu_gpu_density_relative"]
        ),
    }
    return {
        "schema": "aion.phase-two.p2-1.qualification-result",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "proposed_gate_result": "pass" if all(checks.values()) else "fail",
        "system": {
            "name": "equilateral H3+",
            "charge": 1,
            "electrons": 2,
            "basis": "cc-pvdz",
            "functional": "lda,vwn",
            "branch": "kohn_sham_lda",
            "grid": "unpruned PySCF level 4",
            "auxiliary_basis": "weigend",
            "gauge_origin_au": list(_GAUGE_ORIGIN),
            "static_magnetic_field_au": list(_STATIC_FIELD),
        },
        "trajectory": {
            "intervals": _INTERVALS,
            "time_step_au": _STEP_AU,
            "final_time_au": _FINAL_TIME_AU,
            "nonlinear_tolerance": 1.0e-12,
            "direct_nq6_case": _NQ6_RUN_ID,
            "driven_source": (
                "one-cycle compact sin2 vector-potential pulse, peak electric field "
                "0.005 au, omega=pi au, x polarized"
            ),
        },
        "checks": checks,
        "thresholds": thresholds,
        "measurements": {
            "stationary_density_relative_residual_to_nq4": stationary_density_residual,
            "stationary_energy_absolute_residual_to_nq4_au": stationary_energy_residual,
            "nq6_final_mixed_density_relative_residual": nq6_final_density_residual,
            "nq6_final_energy_absolute_residual_au": nq6_final_energy_residual,
            "pulse_integrated_direct_density_relative_residual": pulse_direct_residual,
            "restart_work_absolute_residual_au": restart_work_residual,
            "cpu_gpu_driven_density_relative_residual": gpu_residual,
            "stationary_iterations": len(stationary_state.iterations),
            "stationary_orbital_residual": stationary_state.residuals.orbital,
            "stationary_density_fixed_point_residual": (
                stationary_state.residuals.density_fixed_point
            ),
        },
        "stage_timings_seconds": stages,
        "accepted_inputs": accepted,
        "reused_static_evidence": static["reused_artifacts"],
        "scope_note": (
            "The accepted NQ6 static-B trajectory is reproduced directly. The driven "
            "comparison uses the reusable potential-first pulse admitted by P2-1 and "
            "the retained batch propagator as an independent implementation oracle; "
            "it is not relabelled as the different direct-electric NQ6 pulse."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-gpu", action="store_true")
    parser.add_argument("--reuse-static-root", type=Path)
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    started = perf_counter()
    try:
        result = _run_qualification(
            output,
            arguments.include_gpu,
            arguments.reuse_static_root,
        )
    except Exception as exc:
        _write_json(
            output / "failure.json",
            {
                "schema": "aion.phase-two.failure",
                "schema_version": "1.0.0",
                "status": "failed_visible",
                "phase": "p2-1-qualification",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    elapsed = perf_counter() - started
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result["total_elapsed_seconds"] = elapsed
    result["process_peak_rss_bytes"] = peak_rss_kib * 1024
    result_path = output / "result.json"
    _write_json(result_path, result)
    tracked_diff = _git(repo, "diff", "--binary", "HEAD")
    provenance = {
        "schema": "aion.phase-two.provenance",
        "schema_version": "1.0.0",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "repository": str(repo),
        "git_head": _git(repo, "rev-parse", "HEAD"),
        "git_status_porcelain": _git(
            repo,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).splitlines(),
        "git_tracked_diff_sha256": hashlib.sha256(tracked_diff.encode()).hexdigest(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "host": platform.node(),
        "dependencies": DependencyVersions.current().as_mapping(),
        "thread_limits": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "physical_gpu_launcher": os.environ.get("AION_GPU_LAUNCHER") == "1",
        "source_sha256": {
            str(Path(__file__).resolve().relative_to(repo)): _sha256(Path(__file__).resolve())
        },
        "artifact_sha256": {
            str(path.relative_to(output)): _sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name not in {"result.json", "provenance.json"}
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
    return 0 if result["proposed_gate_result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
