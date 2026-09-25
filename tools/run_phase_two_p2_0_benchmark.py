#!/usr/bin/env python3
"""Measure P2-0 exact-Wilson CPU timing and process peak memory."""

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
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aion.config import (
    AtomConfig,
    BackendConfig,
    ElectronicStructureConfig,
    MetadataConfig,
    MoleculeConfig,
    ReferenceConfig,
    ReferenceOutputConfig,
    XCFamily,
)
from aion.electromagnetism import UniformMagneticField, UniformMagneticSourceSample
from aion.electronic_structure import (
    AOGridPolicy,
    DependencyVersions,
    StationarySCFPolicy,
    WilsonStationaryBranch,
    evaluate_exact_wilson_power,
    load_reference_data,
    prepare_ao_quadrature,
    prepare_exact_wilson_dynamic_sample,
    prepare_exact_wilson_dynamic_spatial_action,
    prepare_exact_wilson_stationary_factory,
    prepare_pyscf_reference,
)
from aion.propagation import (
    NonlinearGaussMagnusPolicy,
    propagate_nonlinear_contravariant_density,
)

NQ4_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "chapter13_wilson_adiabatic_qualification/"
    "nq4_stationary_20260921T232232Z_6197d6828bb3"
)
NQ8_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "chapter13_wilson_adiabatic_qualification/"
    "nq8_gga_transfer_20260924T180834Z_57eb8a1b538d"
)
ORIGIN = (0.17, -0.31, 0.23)


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


def _float(value: object) -> float:
    return float(np.asarray(value).real)


def _timed(stages: dict[str, float], name: str, operation: Any) -> Any:
    print(f"starting {name}", flush=True)
    started = perf_counter()
    result = operation()
    stages[name] = perf_counter() - started
    print(f"completed {name} in {stages[name]:.3f} s", flush=True)
    return result


def _co_reference(output: Path, timestamp: str) -> Any:
    config = ReferenceConfig(
        molecule=MoleculeConfig(
            atoms=(
                AtomConfig("C", (0.0, 0.0, -1.066)),
                AtomConfig("O", (0.0, 0.0, 1.066)),
            ),
            charge=0,
            spin=0,
        ),
        electronic_structure=ElectronicStructureConfig(
            basis="cc-pvdz",
            functional="pbe",
            xc_family=XCFamily.GGA,
            grid_level=4,
            density_fitting=True,
            auxiliary_basis="weigend",
            scf_energy_tolerance_au=1.0e-12,
            scf_max_iterations=160,
        ),
        backend=BackendConfig(),
        output=ReferenceOutputConfig(output / "co.reference.h5"),
        metadata=MetadataConfig(
            label="phase-two-p2-0-co-pbe-baseline",
            timestamp_utc=timestamp,
            host=platform.node(),
        ),
    )
    return prepare_pyscf_reference(config)


def _h3plus_benchmark(output: Path, stages: dict[str, float]) -> dict[str, Any]:
    reference_path = NQ4_ROOT / "h3plus.reference.h5"
    states_path = NQ4_ROOT / "stationary_states.npz"
    reference = _timed(stages, "load_reference", lambda: load_reference_data(reference_path))
    states = np.load(states_path, allow_pickle=False)
    initial_coefficients = np.asarray(states["h3plus_b0p030_kohn_sham_lda_coefficients"])
    quadrature = _timed(
        stages,
        "prepare_quadrature",
        lambda: prepare_ao_quadrature(
            reference,
            BackendConfig(),
            grid_policy=AOGridPolicy.qualification(4),
            block_size=2048,
        ),
    )
    factory = _timed(
        stages,
        "prepare_factory",
        lambda: prepare_exact_wilson_stationary_factory(
            quadrature,
            auxiliary_basis="weigend",
            functional="lda,vwn",
        ),
    )
    static_source = UniformMagneticSourceSample(
        time_au=0.0,
        field=UniformMagneticField((0.0, 0.0, 0.03)),
        origin_au=ORIGIN,
    )
    spatial = _timed(
        stages,
        "prepare_static_spatial_action",
        lambda: prepare_exact_wilson_dynamic_spatial_action(factory, static_source.gauge),
    )
    static_sample = spatial.sample(static_source, WilsonStationaryBranch.KOHN_SHAM_LDA)
    stationary = _timed(
        stages,
        "warm_start_stationary_solve",
        lambda: static_sample.model.solve(
            policy=StationarySCFPolicy(
                maximum_iterations=160,
                density_tolerance=2.0e-10,
                orbital_tolerance=2.0e-10,
                energy_tolerance_au=2.0e-11,
            ),
            initial_coefficients=initial_coefficients,
        ),
    )
    interval = 0.0625
    cache: dict[float, Any] = {}

    def dynamic(time_au: float) -> Any:
        key = float(time_au)
        if key not in cache:
            envelope = math.sin(math.pi * key / interval) ** 2
            source = UniformMagneticSourceSample(
                time_au=key,
                field=UniformMagneticField((0.0, 0.0, 0.03)),
                electric_field_origin_au=(0.005 * envelope, 0.0, 0.0),
                origin_au=ORIGIN,
            )
            cache[key] = spatial.sample(source, WilsonStationaryBranch.KOHN_SHAM_LDA)
        return cache[key]

    trajectory = _timed(
        stages,
        "one_step_dynamics",
        lambda: propagate_nonlinear_contravariant_density(
            stationary.coefficient_density,
            initial_time_au=0.0,
            interval_au=interval,
            intervals=1,
            metric_provider=lambda time: dynamic(time).one_electron.metric,
            eom_provider=lambda time, density: dynamic(time).evaluate(density).triple,
            backend=quadrature.backend,
            policy=NonlinearGaussMagnusPolicy(tolerance=1.0e-12, maximum_iterations=80),
        ),
    )
    final_density = trajectory.contravariant_densities[-1]
    final_evaluation = dynamic(interval).evaluate(final_density)
    power = evaluate_exact_wilson_power(final_evaluation, final_density)
    diagnostic = trajectory.diagnostics[-1]
    return {
        "system": "H3+",
        "basis": "cc-pvdz",
        "functional": "lda,vwn",
        "branch": "kohn_sham_lda",
        "grid_level": 4,
        "grid_points": quadrature.grid.npoints,
        "auxiliary_basis": "weigend",
        "nao": reference.core_operators.nao,
        "accepted_inputs": {
            str(reference_path): _sha256(reference_path),
            str(states_path): _sha256(states_path),
        },
        "stationary": {
            "warm_started": True,
            "energy_molecular_total_au": _float(stationary.action.energy_molecular_total_au),
            "orbital_residual": stationary.orbital_residual,
            "density_fixed_point_residual": stationary.density_fixed_point_residual,
            "iterations": len(stationary.iterations),
        },
        "dynamics": {
            "interval_au": interval,
            "nonlinear_iterations": diagnostic.nonlinear_iterations,
            "nonlinear_residual": diagnostic.nonlinear_residual,
            "cross_metric_residual": diagnostic.cross_metric_residual,
            "occupation_spectrum_drift": diagnostic.occupation_spectrum_drift,
            "power_identity_residual_au": abs(_float(power.power_identity_residual_au)),
        },
    }


def _co_benchmark(output: Path, stages: dict[str, float], timestamp: str) -> dict[str, Any]:
    checkpoint_path = NQ8_ROOT / "checkpoints/co.json"
    arrays_path = NQ8_ROOT / "checkpoints/co.npz"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    arrays = np.load(arrays_path, allow_pickle=False)
    initial_coefficients = np.asarray(arrays["field_coefficients"])
    reference = _timed(stages, "prepare_reference", lambda: _co_reference(output, timestamp))
    accepted_energy = float(checkpoint["zero_field"]["pyscf_reference_energy_au"])
    energy_residual = abs(reference.ground_state.energy_total_au - accepted_energy)
    accepted_electron_count = float(checkpoint["electrons"])
    electron_count_residual = abs(reference.ground_state.electron_count - accepted_electron_count)
    expected_shape = (int(checkpoint["nao"]), int(checkpoint["nao"]))
    reconstruction_checks = {
        "scientific_configuration_id": reference.config.scientific_id,
        "accepted_reference_fingerprint_sha256": checkpoint["reference_fingerprint_sha256"],
        "reconstructed_reference_fingerprint_sha256": reference.fingerprint_sha256,
        "bytewise_fingerprint_equal": (
            reference.fingerprint_sha256 == checkpoint["reference_fingerprint_sha256"]
        ),
        "energy_absolute_residual_au": energy_residual,
        "energy_tolerance_au": 1.0e-10,
        "electron_count": reference.ground_state.electron_count,
        "expected_electron_count": accepted_electron_count,
        "electron_count_absolute_residual": electron_count_residual,
        "electron_count_tolerance": 1.0e-10,
        "density_shape": list(reference.ground_state.density.shape),
        "accepted_state_shape": list(initial_coefficients.shape),
        "expected_ao_shape": list(expected_shape),
        "interpretation": (
            "The prepared-reference fingerprint includes bytewise SCF orbital and "
            "density arrays. A repeated converged PySCF calculation need not reproduce "
            "that hash even when its scientific inputs and energy agree. Compatibility "
            "is therefore established by the authenticated accepted checkpoint, fixed "
            "scientific configuration, energy, electron count, AO dimensions, and the "
            "subsequent stationary residuals."
        ),
    }
    if energy_residual > 1.0e-10:
        raise RuntimeError("reconstructed CO energy does not match accepted NQ8 input")
    if electron_count_residual > 1.0e-10:
        raise RuntimeError("reconstructed CO electron count does not match accepted NQ8 input")
    if reference.ground_state.density.shape != expected_shape:
        raise RuntimeError("reconstructed CO AO dimension does not match accepted NQ8 input")
    if initial_coefficients.shape != expected_shape:
        raise RuntimeError("accepted CO warm-start state has an incompatible AO dimension")
    quadrature = _timed(
        stages,
        "prepare_quadrature",
        lambda: prepare_ao_quadrature(
            reference,
            BackendConfig(),
            grid_policy=AOGridPolicy.qualification(4),
            block_size=1024,
        ),
    )
    factory = _timed(
        stages,
        "prepare_factory",
        lambda: prepare_exact_wilson_stationary_factory(
            quadrature,
            auxiliary_basis="weigend",
            functional="pbe",
        ),
    )
    static_source = UniformMagneticSourceSample(
        time_au=0.0,
        field=UniformMagneticField((0.0, 0.0, 0.03)),
        origin_au=ORIGIN,
    )
    static_sample = _timed(
        stages,
        "prepare_static_dynamic_sample",
        lambda: prepare_exact_wilson_dynamic_sample(
            factory,
            static_source,
            WilsonStationaryBranch.KOHN_SHAM_GGA,
        ),
    )
    stationary = _timed(
        stages,
        "warm_start_stationary_solve",
        lambda: static_sample.model.solve(
            policy=StationarySCFPolicy(
                maximum_iterations=160,
                density_tolerance=2.0e-10,
                orbital_tolerance=2.0e-10,
                energy_tolerance_au=2.0e-11,
            ),
            initial_coefficients=initial_coefficients,
        ),
    )
    interval = 0.025
    cache: dict[float, Any] = {}

    def dynamic(time_au: float) -> Any:
        key = float(time_au)
        if key not in cache:
            source = UniformMagneticSourceSample(
                time_au=key,
                field=UniformMagneticField((0.0, 0.0, 0.03 + 0.001 * key)),
                magnetic_field_dot_au=(0.0, 0.0, 0.001),
                electric_field_origin_au=(0.002, -0.001, 0.0005),
                origin_au=ORIGIN,
            )
            cache[key] = prepare_exact_wilson_dynamic_sample(
                factory,
                source,
                WilsonStationaryBranch.KOHN_SHAM_GGA,
            )
        return cache[key]

    trajectory = _timed(
        stages,
        "one_step_dynamics",
        lambda: propagate_nonlinear_contravariant_density(
            stationary.coefficient_density,
            initial_time_au=0.0,
            interval_au=interval,
            intervals=1,
            metric_provider=lambda time: dynamic(time).one_electron.metric,
            eom_provider=lambda time, density: dynamic(time).evaluate(density).triple,
            backend=quadrature.backend,
            policy=NonlinearGaussMagnusPolicy(tolerance=1.0e-10, maximum_iterations=60),
        ),
    )
    final_density = trajectory.contravariant_densities[-1]
    final_evaluation = dynamic(interval).evaluate(final_density)
    power = evaluate_exact_wilson_power(final_evaluation, final_density)
    diagnostic = trajectory.diagnostics[-1]
    return {
        "system": "CO",
        "basis": "cc-pvdz",
        "functional": "pbe",
        "branch": "kohn_sham_gga",
        "grid_level": 4,
        "grid_points": quadrature.grid.npoints,
        "auxiliary_basis": "weigend",
        "nao": reference.core_operators.nao,
        "accepted_inputs": {
            str(checkpoint_path): _sha256(checkpoint_path),
            str(arrays_path): _sha256(arrays_path),
        },
        "reference_reconstruction": reconstruction_checks,
        "stationary": {
            "warm_started": True,
            "energy_molecular_total_au": _float(stationary.action.energy_molecular_total_au),
            "orbital_residual": stationary.orbital_residual,
            "density_fixed_point_residual": stationary.density_fixed_point_residual,
            "iterations": len(stationary.iterations),
        },
        "dynamics": {
            "interval_au": interval,
            "nonlinear_iterations": diagnostic.nonlinear_iterations,
            "nonlinear_residual": diagnostic.nonlinear_residual,
            "cross_metric_residual": diagnostic.cross_metric_residual,
            "occupation_spectrum_drift": diagnostic.occupation_spectrum_drift,
            "power_identity_residual_au": abs(_float(power.power_identity_residual_au)),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("h3plus", "co"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).isoformat()
    stages: dict[str, float] = {}
    started = perf_counter()
    try:
        record = (
            _h3plus_benchmark(output, stages)
            if arguments.system == "h3plus"
            else _co_benchmark(output, stages, timestamp)
        )
    except Exception as exc:
        _write_json(
            output / "failure.json",
            {
                "schema": "aion.phase-two.failure",
                "schema_version": "1.0.0",
                "status": "failed_visible",
                "phase": "p2-0-baseline-benchmark",
                "system": arguments.system,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "completed_stage_timings_seconds": stages,
            },
        )
        raise
    total = perf_counter() - started
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result = {
        "schema": "aion.phase-two.p2-0.benchmark-result",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "backend": "cpu_float64_complex128",
        "thread_limits": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "stage_timings_seconds": stages,
        "total_elapsed_seconds": total,
        "process_peak_rss_kib": peak_rss_kib,
        "process_peak_rss_bytes": peak_rss_kib * 1024,
        "record": record,
    }
    result_path = output / "result.json"
    _write_json(result_path, result)
    provenance = {
        "schema": "aion.phase-two.provenance",
        "schema_version": "1.0.0",
        "timestamp_utc": timestamp,
        "repository": str(repo),
        "git_head": _git(repo, "rev-parse", "HEAD"),
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
        "source_sha256": {
            str(Path(__file__).resolve().relative_to(repo)): _sha256(Path(__file__).resolve())
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
