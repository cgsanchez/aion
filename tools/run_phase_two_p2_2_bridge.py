#!/usr/bin/env python3
"""Execute the bounded Phase Two P2-2 CO/NH3 transfer bridge stagewise."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

import h5py
import numpy as np

from aion.config import (
    AffineElectromagneticSourceConfig,
    AtomConfig,
    BackendConfig,
    BackendKind,
    ElectromagneticOrigin,
    ElectronicStructureConfig,
    ExactWilsonActionConfig,
    FixedTimeGrid,
    FormulationConfig,
    FormulationKind,
    GridPruning,
    IntegratorKind,
    MetadataConfig,
    MoleculeConfig,
    ObservableSchedules,
    OutputConfig,
    PropagationConfig,
    RationalApproximation,
    ReferenceConfig,
    ReferenceLinkConfig,
    ReferenceOutputConfig,
    SimulationConfig,
    Sin2VectorPotentialPulseConfig,
    StepSchedule,
    WilsonGridKind,
    WilsonGridPruning,
    WilsonInitialSourcePolicy,
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
    XCFamily,
    ZeroSourceConfig,
    dumps_config,
)
from aion.electromagnetism import Sin2VectorPotentialPulse
from aion.electronic_structure import (
    DependencyVersions,
    ExactWilsonDynamicEvaluation,
    load_wilson_stationary_state,
)
from aion.io import load_wilson_checkpoint, load_wilson_trajectory
from aion.io.checkpoint import load_checkpoint
from aion.io.trajectory import load_trajectory as load_bare_trajectory
from aion.workflows import (
    BuiltSimulation,
    BuiltWilsonSimulation,
    build_simulation,
    load_reference,
    prepare_reference,
    prepare_wilson_stationary_state,
    run,
)

_NQ8_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "chapter13_wilson_adiabatic_qualification/"
    "nq8_gga_transfer_20260924T180834Z_57eb8a1b538d"
)
_NH3_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "nh3_gauge_basis_validation/nh3_gauge_comparison"
)
_NH3_GEOMETRY = _NH3_ROOT / "nh3_pbe_ccpvdz_optimized.xyz"
_NH3_CASIDA = _NH3_ROOT / "casida/nh3_pbe_cc-pvdz_df_casida.json"
_ORIGIN = (0.17, -0.31, 0.23)
_CO_FIELD = (0.0, 0.0, 0.03)
_CO_FIELD_RATE = (0.0, 0.0, 0.001)
_CO_ELECTRIC = (0.002, -0.001, 0.0005)
_STEP_AU = 0.025
_INTERVALS = 8
_FINAL_TIME_AU = _STEP_AU * _INTERVALS
_NH3_PULSE_OMEGA_AU = 10.0 * math.pi
_NH3_PULSE_PEAK_ELECTRIC_AU = 0.005
_ANGSTROM_PER_BOHR = 0.52917721092


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


def _relative(candidate: object, reference: object) -> float:
    left = np.asarray(candidate)
    right = np.asarray(reference)
    return float(np.linalg.norm(left - right) / max(1.0, float(np.linalg.norm(right))))


def _maximum_absolute(candidate: object, reference: object) -> float:
    return float(np.max(np.abs(np.asarray(candidate) - np.asarray(reference))))


def _float(value: object) -> float:
    return float(np.asarray(value).real)


def _stage_record_path(output: Path, stage: str) -> Path:
    return output / "stages" / f"{stage}.json"


def _require_fresh_stage(output: Path, stage: str) -> None:
    path = _stage_record_path(output, stage)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite completed stage record {path}")


def _update_status(output: Path, stage: str, state: str) -> None:
    _write_json(
        output / "campaign_status.json",
        {
            "schema": "aion.phase-two.p2-2.status",
            "schema_version": "1.0.0",
            "stage": stage,
            "state": state,
            "pid": os.getpid(),
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def _stage_complete(
    output: Path,
    stage: str,
    started: float,
    result: dict[str, object],
    artifacts: tuple[Path, ...],
) -> None:
    record = {
        "schema": "aion.phase-two.p2-2.stage",
        "schema_version": "1.0.0",
        "stage": stage,
        "status": "executed_unreviewed",
        "elapsed_seconds": perf_counter() - started,
        "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "result": result,
        "artifacts_sha256": {str(path.relative_to(output)): _sha256(path) for path in artifacts},
    }
    _write_json(_stage_record_path(output, stage), record)
    _update_status(output, stage, "completed")


def _campaign_identity(output: Path, repo: Path) -> dict[str, object]:
    path = output / "campaign_identity.json"
    source = Path(__file__).resolve()
    accepted_paths = (
        _NQ8_ROOT / "completed.json",
        _NQ8_ROOT / "result.json",
        _NQ8_ROOT / "checkpoints/co.json",
        _NQ8_ROOT / "checkpoints/co.npz",
        _NH3_GEOMETRY,
        _NH3_CASIDA,
    )
    current: dict[str, object] = {
        "schema": "aion.phase-two.p2-2.identity",
        "schema_version": "1.0.0",
        "git_head": _git(repo, "rev-parse", "HEAD"),
        "git_branch": _git(repo, "branch", "--show-current"),
        "driver_sha256": _sha256(source),
        "accepted_inputs_sha256": {str(item): _sha256(item) for item in accepted_paths},
        "fixed_matrix": {
            "time_step_au": _STEP_AU,
            "intervals": _INTERVALS,
            "final_time_au": _FINAL_TIME_AU,
            "basis": "cc-pvdz",
            "functional": "pbe",
            "auxiliary_basis": "weigend",
            "grid": "unpruned PySCF level 4 qualification grid",
        },
    }
    if path.exists():
        stored = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        if stored != current:
            raise RuntimeError("campaign identity changed; use a new output root")
        return stored
    output.mkdir(parents=True, exist_ok=True)
    _write_json(path, current)
    return current


def _co_atoms() -> tuple[AtomConfig, ...]:
    return (
        AtomConfig("C", (0.0, 0.0, -1.066)),
        AtomConfig("O", (0.0, 0.0, 1.066)),
    )


def _nh3_atoms() -> tuple[AtomConfig, ...]:
    lines = _NH3_GEOMETRY.read_text(encoding="utf-8").splitlines()
    count = int(lines[0].strip())
    atoms: list[AtomConfig] = []
    for line in lines[2 : 2 + count]:
        symbol, *components = line.split()
        position = tuple(float(value) / _ANGSTROM_PER_BOHR for value in components)
        atoms.append(AtomConfig(symbol, position))  # type: ignore[arg-type]
    if len(atoms) != 4 or atoms[0].symbol != "N":
        raise RuntimeError("authenticated G_DZ geometry is not the expected oriented NH3")
    return tuple(atoms)


def _reference_config(output: Path, system: str) -> ReferenceConfig:
    if system == "co":
        atoms = _co_atoms()
        origin = _ORIGIN
        pruning = GridPruning.PYSCF_DEFAULT
    elif system == "nh3":
        atoms = _nh3_atoms()
        origin = atoms[0].position_au
        pruning = GridPruning.NONE
    else:
        raise ValueError(system)
    return ReferenceConfig(
        molecule=MoleculeConfig(
            atoms=atoms,
            charge=0,
            spin=0,
            electromagnetic_origin=ElectromagneticOrigin(origin),
        ),
        electronic_structure=ElectronicStructureConfig(
            basis="cc-pvdz",
            functional="pbe",
            xc_family=XCFamily.GGA,
            grid_level=4,
            grid_pruning=pruning,
            density_fitting=True,
            auxiliary_basis="weigend",
            scf_energy_tolerance_au=1.0e-12,
            scf_max_iterations=160,
        ),
        backend=BackendConfig(),
        output=ReferenceOutputConfig(output / system / "reference.h5"),
        metadata=MetadataConfig(label=f"phase-two-p2-2-{system}-pbe"),
    )


def _numerics() -> WilsonNumericsConfig:
    return WilsonNumericsConfig(
        grid_kind=WilsonGridKind.QUALIFICATION,
        grid_level=4,
        grid_pruning=WilsonGridPruning.NONE,
        block_size=1024,
        dynamic_cache_entries=4,
        auxiliary_basis="weigend",
        ri_relative_threshold=0.0,
        ri_absolute_threshold=1.0e-7,
        ri_maximum_rank=None,
        memory_budget_bytes=None,
    )


def _co_stationary_source() -> AffineElectromagneticSourceConfig:
    return AffineElectromagneticSourceConfig(
        electric=ZeroSourceConfig(),
        electric_field_origin_offset_au=(0.0, 0.0, 0.0),
        magnetic_field_reference_au=_CO_FIELD,
        magnetic_field_rate_au=(0.0, 0.0, 0.0),
        magnetic_reference_time_au=0.0,
        magnetic_gauge=WilsonMagneticGaugeKind.SYMMETRIC,
    )


def _co_dynamic_source() -> AffineElectromagneticSourceConfig:
    return replace(
        _co_stationary_source(),
        electric_field_origin_offset_au=_CO_ELECTRIC,
        magnetic_field_rate_au=_CO_FIELD_RATE,
    )


def _nh3_pulse() -> Sin2VectorPotentialPulseConfig:
    return Sin2VectorPotentialPulseConfig(
        peak_electric_field_au=_NH3_PULSE_PEAK_ELECTRIC_AU,
        angular_frequency_au=_NH3_PULSE_OMEGA_AU,
        cycles=1,
        polarization=(0.0, 0.0, 1.0),
        start_time_au=0.0,
        carrier_phase_rad=0.0,
        require_zero_impulse=True,
    )


def _nh3_source() -> AffineElectromagneticSourceConfig:
    return AffineElectromagneticSourceConfig(
        electric=_nh3_pulse(),
        electric_field_origin_offset_au=(0.0, 0.0, 0.0),
        magnetic_field_reference_au=(0.0, 0.0, 0.0),
        magnetic_field_rate_au=(0.0, 0.0, 0.0),
        magnetic_reference_time_au=0.0,
        magnetic_gauge=WilsonMagneticGaugeKind.SYMMETRIC,
    )


def _stationary_policy() -> WilsonStationaryPolicyConfig:
    return WilsonStationaryPolicyConfig(
        maximum_iterations=160,
        density_tolerance=2.0e-10,
        orbital_tolerance=2.0e-10,
        energy_tolerance_au=2.0e-11,
        damping=0.5,
        diis_start_iteration=2,
        diis_space=8,
    )


def _stationary_config(
    output: Path,
    system: str,
    reference_fingerprint: str,
) -> WilsonStationaryConfig:
    source = _co_stationary_source() if system == "co" else _nh3_source()
    return WilsonStationaryConfig(
        reference=ReferenceLinkConfig(
            reference_fingerprint,
            output / system / "reference.h5",
        ),
        action=ExactWilsonActionConfig(WilsonStationaryBranch.KOHN_SHAM_GGA),
        numerics=_numerics(),
        source=source,
        source_time_au=0.0,
        stationary=_stationary_policy(),
        backend=BackendConfig(),
        output=WilsonStationaryOutputConfig(output / system / "stationary.h5"),
        metadata=MetadataConfig(label=f"phase-two-p2-2-{system}-stationary"),
    )


def _schedules() -> ObservableSchedules:
    every = StepSchedule(every=1, include_initial=True, include_final=True)
    return ObservableSchedules(
        dipole_current=every,
        energy=every,
        diagnostics=every,
        source=every,
        checkpoints=StepSchedule(every=1, include_initial=True, include_final=True),
        matrix_snapshots=StepSchedule(every=0, include_initial=False, include_final=True),
    )


def _wilson_simulation_config(
    output: Path,
    system: str,
    reference_fingerprint: str,
    state_fingerprint: str,
    backend: BackendConfig,
) -> WilsonSimulationConfig:
    is_co = system == "co"
    source = _co_dynamic_source() if is_co else _nh3_source()
    backend_label = backend.kind.value
    return WilsonSimulationConfig(
        reference=ReferenceLinkConfig(
            reference_fingerprint,
            output / system / "reference.h5",
        ),
        stationary_state=WilsonStationaryStateLinkConfig(
            state_fingerprint,
            output / system / "stationary.h5",
        ),
        action=ExactWilsonActionConfig(WilsonStationaryBranch.KOHN_SHAM_GGA),
        numerics=_numerics(),
        source=source,
        propagation=WilsonPropagationConfig(
            time_grid=FixedTimeGrid(0.0, _STEP_AU, _INTERVALS),
            integrator=WilsonIntegratorKind.NONLINEAR_GAUSS_MAGNUS,
            rational_approximation=RationalApproximation.PADE_22,
            nonlinear_tolerance=1.0e-10,
            maximum_iterations=60,
        ),
        backend=backend,
        output=OutputConfig(
            output / system / f"exact_{backend_label}",
            schedules=_schedules(),
        ),
        initial_source_policy=(
            WilsonInitialSourcePolicy.CONTINUOUS_DENSITY_QUENCH
            if is_co
            else WilsonInitialSourcePolicy.MATCHED
        ),
        metadata=MetadataConfig(label=f"phase-two-p2-2-{system}-exact-{backend_label}"),
    )


def _bare_nh3_config(
    output: Path,
    reference_fingerprint: str,
    backend: BackendConfig,
) -> SimulationConfig:
    backend_label = backend.kind.value
    return SimulationConfig(
        reference=ReferenceLinkConfig(reference_fingerprint, output / "nh3/reference.h5"),
        formulation=FormulationConfig(FormulationKind.BARE_LENGTH_GAUGE),
        source=_nh3_pulse(),
        propagation=PropagationConfig(
            time_grid=FixedTimeGrid(0.0, _STEP_AU, _INTERVALS),
            integrator=IntegratorKind.FIXED_METRIC_SCEM,
            rational_approximation=RationalApproximation.PADE_22,
            density_tolerance=1.0e-10,
            max_iterations=60,
            minimum_damping=0.1,
            hermitian_cleanup_threshold=1.0e-12,
        ),
        backend=backend,
        output=OutputConfig(
            output / "nh3" / f"bare_length_{backend_label}",
            schedules=_schedules(),
        ),
        metadata=MetadataConfig(label=f"phase-two-p2-2-nh3-bare-length-{backend_label}"),
    )


def _write_config(path: Path, config: object) -> None:
    path.write_text(dumps_config(config), encoding="utf-8")  # type: ignore[arg-type]


def _prepare_system(output: Path, system: str) -> None:
    stage = f"prepare-{system}"
    _require_fresh_stage(output, stage)
    _update_status(output, stage, "running")
    started = perf_counter()
    directory = output / system
    directory.mkdir(parents=True, exist_ok=True)
    reference_config = _reference_config(output, system)
    _write_config(directory / "reference.toml", reference_config)
    reference = prepare_reference(reference_config)
    reference.save()
    stationary_config = _stationary_config(
        output,
        system,
        reference.fingerprint_sha256,
    )
    _write_config(directory / "stationary.toml", stationary_config)
    stationary = prepare_wilson_stationary_state(stationary_config, reference)
    stationary.save()
    _stage_complete(
        output,
        stage,
        started,
        {
            "reference_fingerprint_sha256": reference.fingerprint_sha256,
            "grid_fingerprint_sha256": stationary.grid_fingerprint_sha256,
            "stationary_state_fingerprint_sha256": stationary.fingerprint_sha256,
            "nao": reference.core_operators.nao,
            "grid_points": int(reference.grid.coordinates_au.shape[0]),
            "stationary_iterations": len(stationary.iterations),
            "stationary_orbital_residual": stationary.residuals.orbital,
            "stationary_density_fixed_point_residual": (stationary.residuals.density_fixed_point),
            "stationary_particle_number": stationary.residuals.particle_number,
            "stationary_energy_molecular_total_au": (stationary.energies.molecular_total_au),
        },
        (
            directory / "reference.toml",
            directory / "reference.h5",
            directory / "stationary.toml",
            directory / "stationary.h5",
        ),
    )


def _instrument_wilson(simulation: BuiltWilsonSimulation) -> dict[str, float | int]:
    counter: dict[str, float | int] = {"calls": 0, "seconds": 0.0}
    original = simulation.propagator.evaluation_provider
    backend = simulation.quadrature.backend

    def timed(time_au: float, density: Any) -> ExactWilsonDynamicEvaluation:
        backend.synchronize()
        started = perf_counter()
        result = original(time_au, density)
        backend.synchronize()
        counter["calls"] = int(counter["calls"]) + 1
        counter["seconds"] = float(counter["seconds"]) + perf_counter() - started
        return result

    simulation.propagator.evaluation_provider = timed
    return counter


def _run_wilson_stage(output: Path, system: str, backend_kind: BackendKind) -> None:
    backend_label = backend_kind.value
    stage = f"run-{system}-{backend_label}"
    _require_fresh_stage(output, stage)
    if backend_kind is BackendKind.GPU and os.environ.get("AION_GPU_LAUNCHER") != "1":
        raise RuntimeError("GPU stages must be launched through tools/gpu-python")
    _update_status(output, stage, "running")
    started = perf_counter()
    reference = load_reference(output / system / "reference.h5")
    stationary = load_wilson_stationary_state(output / system / "stationary.h5")
    backend = BackendConfig(
        kind=backend_kind,
        device_index=0 if backend_kind is BackendKind.GPU else None,
    )
    config = _wilson_simulation_config(
        output,
        system,
        reference.fingerprint_sha256,
        stationary.fingerprint_sha256,
        backend,
    )
    config_path = output / system / f"exact_{backend_label}.toml"
    _write_config(config_path, config)
    simulation = build_simulation(config, reference, stationary_state=stationary)
    if not isinstance(simulation, BuiltWilsonSimulation):
        raise RuntimeError("Wilson configuration built the wrong runtime")
    counter = _instrument_wilson(simulation)
    if backend_kind is BackendKind.GPU:
        simulation.quadrature.backend.assert_resident(
            simulation.density,
            name=f"{system} initial exact density",
        )
    trajectory = run(simulation)
    if backend_kind is BackendKind.GPU:
        simulation.quadrature.backend.assert_resident(
            simulation.density,
            name=f"{system} final exact density",
        )
        observation = simulation.observe_endpoint()
        simulation.quadrature.backend.assert_resident(
            observation.uniform_source_current_au,
            name=f"{system} exact source current",
        )
        simulation.quadrature.backend.assert_resident(
            observation.electronic_dipole_au,
            name=f"{system} exact dipole",
        )
    final_state_path = output / system / f"exact_{backend_label}_final_state.npz"
    final_metric = simulation.dynamic_cache.sample(simulation.current_time_au).one_electron.metric
    np.savez_compressed(
        final_state_path,
        contravariant_density=simulation.quadrature.backend.to_host(simulation.density),
        metric=simulation.quadrature.backend.to_host(final_metric),
    )
    calls = int(counter["calls"])
    measured = float(counter["seconds"])
    run_directory = config.output.directory
    final_checkpoint = run_directory / f"checkpoint_{_INTERVALS:08d}.h5"
    _stage_complete(
        output,
        stage,
        started,
        {
            "backend": backend_label,
            "trajectory_sha256": trajectory.sha256,
            "action_evaluation_calls": calls,
            "action_evaluation_seconds": measured,
            "mean_seconds_per_action_evaluation": measured / calls,
            "gpu_residency_asserted": backend_kind is BackendKind.GPU,
        },
        (
            config_path,
            run_directory / "trajectory.h5",
            final_checkpoint,
            final_state_path,
            run_directory / "status.json",
        ),
    )


def _run_bare_nh3_stage(output: Path, backend_kind: BackendKind) -> None:
    backend_label = backend_kind.value
    stage = f"run-nh3-bare-{backend_label}"
    _require_fresh_stage(output, stage)
    if backend_kind is BackendKind.GPU and os.environ.get("AION_GPU_LAUNCHER") != "1":
        raise RuntimeError("GPU stages must be launched through tools/gpu-python")
    _update_status(output, stage, "running")
    started = perf_counter()
    reference = load_reference(output / "nh3/reference.h5")
    backend = BackendConfig(
        kind=backend_kind,
        device_index=0 if backend_kind is BackendKind.GPU else None,
    )
    config = _bare_nh3_config(output, reference.fingerprint_sha256, backend)
    config_path = output / "nh3" / f"bare_length_{backend_label}.toml"
    _write_config(config_path, config)
    simulation = build_simulation(config, reference)
    if not isinstance(simulation, BuiltSimulation):
        raise RuntimeError("bare length configuration built the wrong runtime")
    if backend_kind is BackendKind.GPU:
        simulation.workspace.backend.assert_resident(
            simulation.density.matrix,
            name="NH3 initial bare-length density",
        )
    trajectory = run(simulation)
    if backend_kind is BackendKind.GPU:
        simulation.workspace.backend.assert_resident(
            simulation.density.matrix,
            name="NH3 final bare-length density",
        )
    run_directory = config.output.directory
    final_checkpoint = run_directory / f"checkpoint_{_INTERVALS:08d}.h5"
    _stage_complete(
        output,
        stage,
        started,
        {
            "backend": backend_label,
            "trajectory_sha256": trajectory.sha256,
            "gpu_residency_asserted": backend_kind is BackendKind.GPU,
        },
        (
            config_path,
            run_directory / "trajectory.h5",
            final_checkpoint,
            run_directory / "status.json",
        ),
    )


def _h5_values(path: Path, dataset: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[dataset][...])


def _wilson_diagnostics(path: Path) -> dict[str, float]:
    names = (
        "nonlinear_iterations",
        "nonlinear_residual",
        "cross_metric_residual",
        "trace_drift",
        "trace_imaginary_abs",
        "occupation_spectrum_drift",
        "metric_hermiticity_residual",
        "contravariant_hermiticity_residual",
    )
    result: dict[str, float] = {}
    for name in names:
        values = _h5_values(path, f"diagnostics/propagation/{name}")
        result[f"maximum_{name}"] = float(np.max(np.abs(values)))
    for name in (
        "ward_residual_abs",
        "density_shell_residual_relative_norm",
        "finite_region_continuity_residual_abs",
        "global_charge_residual_abs",
    ):
        values = _h5_values(path, f"observables/identity/{name}/values")
        result[f"maximum_{name}"] = float(np.max(np.abs(values)))
    number = _h5_values(path, "observables/charge/metric_particle_number/values")
    result["maximum_particle_number_drift"] = float(np.max(np.abs(number - number[0])))
    power = _h5_values(path, "observables/power/identity_residual/values")
    result["maximum_power_identity_residual_au"] = float(np.max(np.abs(power)))
    work = _h5_values(path, "observables/work/energy_residual/values")
    result["maximum_work_energy_residual_au"] = float(np.max(np.abs(work)))
    return result


def _read_wilson(path: Path, name: str) -> np.ndarray:
    return load_wilson_trajectory(path).read_series(name).values


def _read_bare(path: Path, name: str) -> np.ndarray:
    return load_bare_trajectory(path).read_observable(name).values


def _analyze(output: Path, repo: Path) -> None:
    stage = "analyze"
    _require_fresh_stage(output, stage)
    _update_status(output, stage, "running")
    started = perf_counter()
    required_stages = (
        "prepare-co",
        "run-co-cpu",
        "run-co-gpu",
        "prepare-nh3",
        "run-nh3-cpu",
        "run-nh3-gpu",
        "run-nh3-bare-cpu",
        "run-nh3-bare-gpu",
    )
    missing = [name for name in required_stages if not _stage_record_path(output, name).is_file()]
    if missing:
        raise RuntimeError(f"P2-2 analysis is missing stages: {missing}")

    nq8_checkpoint = json.loads((_NQ8_ROOT / "checkpoints/co.json").read_text())
    nq8_arrays = np.load(_NQ8_ROOT / "checkpoints/co.npz", allow_pickle=False)
    co_state = load_wilson_stationary_state(output / "co/stationary.h5")
    co_cpu_checkpoint = load_wilson_checkpoint(
        output / f"co/exact_cpu/checkpoint_{_INTERVALS:08d}.h5"
    )
    co_gpu_checkpoint = load_wilson_checkpoint(
        output / f"co/exact_gpu/checkpoint_{_INTERVALS:08d}.h5"
    )
    co_cpu_trajectory = output / "co/exact_cpu/trajectory.h5"
    co_gpu_trajectory = output / "co/exact_gpu/trajectory.h5"
    with np.load(output / "co/exact_cpu_final_state.npz", allow_pickle=False) as final:
        co_cpu_mixed = np.asarray(final["contravariant_density"]) @ np.asarray(final["metric"])
    co_accepted_energy = float(np.asarray(nq8_arrays["n8_energies_au"])[-1])
    co_cpu_energy = float(_read_wilson(co_cpu_trajectory, "energy/molecular_total")[-1])
    co_measurements = {
        "stationary_density_relative_residual_to_nq8": _relative(
            co_state.contravariant_density,
            nq8_arrays["field_density"],
        ),
        "stationary_energy_absolute_residual_to_nq8_au": abs(
            co_state.energies.molecular_total_au
            - float(nq8_checkpoint["static_field"]["energy_molecular_total_au"])
        ),
        "final_contravariant_density_relative_residual_to_nq8": _relative(
            co_cpu_checkpoint.contravariant_density,
            nq8_arrays["n8_final_density"],
        ),
        "final_mixed_density_relative_residual_to_nq8": _relative(
            co_cpu_mixed,
            nq8_arrays["n8_final_mixed_density"],
        ),
        "final_energy_absolute_residual_to_nq8_au": abs(co_cpu_energy - co_accepted_energy),
        "cpu_gpu_final_density_relative_residual": _relative(
            co_gpu_checkpoint.contravariant_density,
            co_cpu_checkpoint.contravariant_density,
        ),
        "cpu_gpu_energy_maximum_absolute_residual_au": _maximum_absolute(
            _read_wilson(co_gpu_trajectory, "energy/molecular_total"),
            _read_wilson(co_cpu_trajectory, "energy/molecular_total"),
        ),
        **_wilson_diagnostics(co_cpu_trajectory),
    }

    nh3_reference = load_reference(output / "nh3/reference.h5")
    nh3_state = load_wilson_stationary_state(output / "nh3/stationary.h5")
    nh3_exact_cpu_checkpoint = load_wilson_checkpoint(
        output / f"nh3/exact_cpu/checkpoint_{_INTERVALS:08d}.h5"
    )
    nh3_exact_gpu_checkpoint = load_wilson_checkpoint(
        output / f"nh3/exact_gpu/checkpoint_{_INTERVALS:08d}.h5"
    )
    nh3_bare_cpu_checkpoint = load_checkpoint(
        output / f"nh3/bare_length_cpu/checkpoint_{_INTERVALS:08d}.h5"
    )
    nh3_bare_gpu_checkpoint = load_checkpoint(
        output / f"nh3/bare_length_gpu/checkpoint_{_INTERVALS:08d}.h5"
    )
    nh3_exact_cpu_trajectory = output / "nh3/exact_cpu/trajectory.h5"
    nh3_bare_cpu_trajectory = output / "nh3/bare_length_cpu/trajectory.h5"
    exact_dipole = _read_wilson(nh3_exact_cpu_trajectory, "dipole/molecular_total")
    bare_dipole = _read_bare(
        nh3_bare_cpu_trajectory,
        "dipole.combined.bare_length_gauge",
    )
    exact_current = _read_wilson(nh3_exact_cpu_trajectory, "current/uniform_source")
    bare_current = _read_bare(
        nh3_bare_cpu_trajectory,
        "current.variational_source.bare_length_gauge",
    )
    exact_energy = _read_wilson(nh3_exact_cpu_trajectory, "energy/molecular_total")
    bare_energy = _read_bare(
        nh3_bare_cpu_trajectory,
        "energy.matter.total.bare_length_gauge",
    ).reshape(-1)
    pulse = Sin2VectorPotentialPulse(_nh3_pulse())
    pulse_start = pulse.sample(0.0)
    pulse_end = pulse.sample(_FINAL_TIME_AU)
    nh3_measurements = {
        "reference_and_exact_grid_fingerprint_equal": (
            nh3_reference.grid.fingerprint_sha256 == nh3_state.grid_fingerprint_sha256
        ),
        "stationary_density_relative_residual_to_bare_reference": _relative(
            nh3_state.contravariant_density,
            nh3_reference.ground_state.density,
        ),
        "stationary_energy_absolute_residual_to_bare_reference_au": abs(
            nh3_state.energies.molecular_total_au - nh3_reference.ground_state.energy_total_au
        ),
        "exact_bare_final_density_relative_residual": _relative(
            nh3_exact_cpu_checkpoint.contravariant_density,
            nh3_bare_cpu_checkpoint.density,
        ),
        "exact_bare_dipole_maximum_absolute_residual_au": _maximum_absolute(
            exact_dipole,
            bare_dipole,
        ),
        "exact_bare_current_maximum_absolute_residual_au": _maximum_absolute(
            exact_current,
            bare_current,
        ),
        "exact_bare_energy_maximum_absolute_residual_au": _maximum_absolute(
            exact_energy,
            bare_energy,
        ),
        "exact_cpu_gpu_final_density_relative_residual": _relative(
            nh3_exact_gpu_checkpoint.contravariant_density,
            nh3_exact_cpu_checkpoint.contravariant_density,
        ),
        "bare_cpu_gpu_final_density_relative_residual": _relative(
            nh3_bare_gpu_checkpoint.density,
            nh3_bare_cpu_checkpoint.density,
        ),
        "pulse_start_vector_potential_norm": float(
            np.linalg.norm(pulse_start.vector_potential_reduced)
        ),
        "pulse_start_electric_field_norm": float(np.linalg.norm(pulse_start.electric_field)),
        "pulse_end_vector_potential_norm": float(
            np.linalg.norm(pulse_end.vector_potential_reduced)
        ),
        "pulse_end_electric_field_norm": float(np.linalg.norm(pulse_end.electric_field)),
        **_wilson_diagnostics(nh3_exact_cpu_trajectory),
    }

    thresholds = {
        "co_stationary_density_relative": 2.0e-8,
        "co_stationary_energy_absolute_au": 2.0e-8,
        "co_final_density_relative": 1.0e-7,
        "co_final_energy_absolute_au": 1.0e-7,
        "cpu_gpu_density_relative": 2.0e-8,
        "cpu_gpu_energy_absolute_au": 2.0e-8,
        "nh3_stationary_density_relative": 2.0e-6,
        "nh3_stationary_energy_absolute_au": 2.0e-6,
        "nh3_exact_bare_final_density_relative": 5.0e-6,
        "nh3_exact_bare_dipole_absolute_au": 5.0e-5,
        "nh3_exact_bare_current_absolute_au": 5.0e-5,
        "nh3_exact_bare_energy_absolute_au": 2.0e-6,
        "particle_number_drift": 1.0e-9,
        "identity_residual": 1.0e-8,
        "work_energy_absolute_au": 1.0e-7,
        "source_endpoint_norm": 1.0e-12,
    }
    checks = {
        "co_stationary_density_matches_nq8": (
            co_measurements["stationary_density_relative_residual_to_nq8"]
            <= thresholds["co_stationary_density_relative"]
        ),
        "co_stationary_energy_matches_nq8": (
            co_measurements["stationary_energy_absolute_residual_to_nq8_au"]
            <= thresholds["co_stationary_energy_absolute_au"]
        ),
        "co_final_density_matches_nq8": (
            co_measurements["final_mixed_density_relative_residual_to_nq8"]
            <= thresholds["co_final_density_relative"]
        ),
        "co_final_energy_matches_nq8": (
            co_measurements["final_energy_absolute_residual_to_nq8_au"]
            <= thresholds["co_final_energy_absolute_au"]
        ),
        "co_cpu_gpu_density_matches": (
            co_measurements["cpu_gpu_final_density_relative_residual"]
            <= thresholds["cpu_gpu_density_relative"]
        ),
        "co_cpu_gpu_energy_matches": (
            co_measurements["cpu_gpu_energy_maximum_absolute_residual_au"]
            <= thresholds["cpu_gpu_energy_absolute_au"]
        ),
        "nh3_grids_match": nh3_measurements["reference_and_exact_grid_fingerprint_equal"],
        "nh3_stationary_density_reduces_to_bare": (
            nh3_measurements["stationary_density_relative_residual_to_bare_reference"]
            <= thresholds["nh3_stationary_density_relative"]
        ),
        "nh3_stationary_energy_reduces_to_bare": (
            nh3_measurements["stationary_energy_absolute_residual_to_bare_reference_au"]
            <= thresholds["nh3_stationary_energy_absolute_au"]
        ),
        "nh3_exact_bare_final_density_matches": (
            nh3_measurements["exact_bare_final_density_relative_residual"]
            <= thresholds["nh3_exact_bare_final_density_relative"]
        ),
        "nh3_exact_bare_dipole_matches": (
            nh3_measurements["exact_bare_dipole_maximum_absolute_residual_au"]
            <= thresholds["nh3_exact_bare_dipole_absolute_au"]
        ),
        "nh3_exact_bare_current_matches": (
            nh3_measurements["exact_bare_current_maximum_absolute_residual_au"]
            <= thresholds["nh3_exact_bare_current_absolute_au"]
        ),
        "nh3_exact_bare_energy_matches": (
            nh3_measurements["exact_bare_energy_maximum_absolute_residual_au"]
            <= thresholds["nh3_exact_bare_energy_absolute_au"]
        ),
        "nh3_exact_cpu_gpu_density_matches": (
            nh3_measurements["exact_cpu_gpu_final_density_relative_residual"]
            <= thresholds["cpu_gpu_density_relative"]
        ),
        "nh3_bare_cpu_gpu_density_matches": (
            nh3_measurements["bare_cpu_gpu_final_density_relative_residual"]
            <= thresholds["cpu_gpu_density_relative"]
        ),
        "nh3_particle_number_is_stable": (
            nh3_measurements["maximum_particle_number_drift"] <= thresholds["particle_number_drift"]
        ),
        "nh3_ward_identity_holds": (
            nh3_measurements["maximum_ward_residual_abs"] <= thresholds["identity_residual"]
        ),
        "nh3_weak_continuity_holds": (
            nh3_measurements["maximum_finite_region_continuity_residual_abs"]
            <= thresholds["identity_residual"]
        ),
        "nh3_power_identity_holds": (
            nh3_measurements["maximum_power_identity_residual_au"]
            <= thresholds["identity_residual"]
        ),
        "nh3_work_energy_is_bounded": (
            nh3_measurements["maximum_work_energy_residual_au"]
            <= thresholds["work_energy_absolute_au"]
        ),
        "nh3_pulse_has_field_free_endpoints": max(
            nh3_measurements["pulse_start_vector_potential_norm"],
            nh3_measurements["pulse_start_electric_field_norm"],
            nh3_measurements["pulse_end_vector_potential_norm"],
            nh3_measurements["pulse_end_electric_field_norm"],
        )
        <= thresholds["source_endpoint_norm"],
    }
    stage_records = {
        name: json.loads(_stage_record_path(output, name).read_text(encoding="utf-8"))
        for name in required_stages
    }
    result = {
        "schema": "aion.phase-two.p2-2.qualification-result",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "proposed_gate_result": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "thresholds": thresholds,
        "co_measurements": co_measurements,
        "nh3_measurements": nh3_measurements,
        "stage_measurements": {
            name: {
                "elapsed_seconds": record["elapsed_seconds"],
                "process_peak_rss_bytes": record["process_peak_rss_bytes"],
                **record["result"],
            }
            for name, record in stage_records.items()
        },
        "scope": {
            "co": (
                "Fresh integrated reproduction of the accepted NQ8 CO/PBE 0.2-au "
                "trajectory using an explicit continuous-density temporal-source quench."
            ),
            "nh3": (
                "G_DZ PBE/cc-pVDZ/weigend, unpruned level-4 grid, B=0, and a "
                "complete 0.2-au one-cycle C3-axis pulse. This is a numerical bridge, "
                "not a spectroscopy or absorption calculation."
            ),
            "ordinary_comparison": (
                "The matching bare-length run uses the same reference, unpruned grid, "
                "functional, RI auxiliary basis, pulse, timestep, duration, and backend."
            ),
        },
    }
    result_path = output / "result.json"
    _write_json(result_path, result)
    provenance = {
        "schema": "aion.phase-two.provenance",
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
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "campaign_identity_sha256": _sha256(output / "campaign_identity.json"),
        "stage_records_sha256": {
            name: _sha256(_stage_record_path(output, name)) for name in required_stages
        },
    }
    provenance_path = output / "provenance.json"
    _write_json(provenance_path, provenance)
    completed_path = output / "completed.json"
    _write_json(
        completed_path,
        {
            "schema": "aion.phase-two.completed",
            "schema_version": "1.0.0",
            "status": "executed_unreviewed",
            "result_sha256": _sha256(result_path),
            "provenance_sha256": _sha256(provenance_path),
        },
    )
    _stage_complete(
        output,
        stage,
        started,
        {
            "proposed_gate_result": result["proposed_gate_result"],
            "check_count": len(checks),
            "passing_check_count": sum(bool(value) for value in checks.values()),
        },
        (result_path, provenance_path, completed_path),
    )


def _run_stage(output: Path, stage: str, repo: Path) -> None:
    if stage == "prepare-co":
        _prepare_system(output, "co")
    elif stage == "run-co-cpu":
        _run_wilson_stage(output, "co", BackendKind.CPU)
    elif stage == "run-co-gpu":
        _run_wilson_stage(output, "co", BackendKind.GPU)
    elif stage == "prepare-nh3":
        _prepare_system(output, "nh3")
    elif stage == "run-nh3-cpu":
        _run_wilson_stage(output, "nh3", BackendKind.CPU)
    elif stage == "run-nh3-gpu":
        _run_wilson_stage(output, "nh3", BackendKind.GPU)
    elif stage == "run-nh3-bare-cpu":
        _run_bare_nh3_stage(output, BackendKind.CPU)
    elif stage == "run-nh3-bare-gpu":
        _run_bare_nh3_stage(output, BackendKind.GPU)
    elif stage == "analyze":
        _analyze(output, repo)
    else:  # pragma: no cover - argparse closes this branch
        raise ValueError(stage)


def main() -> int:
    stages = (
        "prepare-co",
        "run-co-cpu",
        "run-co-gpu",
        "prepare-nh3",
        "run-nh3-cpu",
        "run-nh3-gpu",
        "run-nh3-bare-cpu",
        "run-nh3-bare-gpu",
        "analyze",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=stages, required=True)
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    try:
        _campaign_identity(output, repo)
        _run_stage(output, arguments.stage, repo)
    except Exception as exc:
        _write_json(
            output / "failures" / f"{arguments.stage}.json",
            {
                "schema": "aion.phase-two.failure",
                "schema_version": "1.0.0",
                "status": "failed_visible",
                "phase": arguments.stage,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        _update_status(output, arguments.stage, "failed_visible")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
