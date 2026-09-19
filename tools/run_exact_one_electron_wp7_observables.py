#!/usr/bin/env python3
"""Run the authenticated WP7 charge, continuity, Ward, and power campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg

from aion.config import (
    AtomConfig,
    BackendConfig,
    BackendKind,
    ElectromagneticOrigin,
    OneElectronReferenceConfig,
    canonical_sha256,
)
from aion.electromagnetism import UniformMagneticField, UniformMagneticSourceSample
from aion.electronic_structure import (
    AOGridPolicy,
    evaluate_exact_wilson_one_electron_sample,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
)
from aion.formulations import (
    evaluate_exact_discrete_continuity,
    evaluate_exact_uniform_electric_power,
    exact_pure_gauge_action_direction,
    exact_wilson_one_electron_triple,
    one_electron_velocity_density,
    restricted_one_electron_action_full_directional_derivative,
)
from aion.propagation import (
    ExperimentalGaussMagnusHistory,
    propagate_experimental_mixed_density,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_FIXTURE = _REPOSITORY / "tests/fixtures/exact_one_electron/wp7_observables.fixture.json"
_LOCK = _REPOSITORY / "conda-linux-64.lock"
_DEFAULT_OUTPUT_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/exact_one_electron_qualification"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execution-directory", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)  # type: ignore[arg-type]
    os.replace(temporary, path)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source(
    case: str,
    time_au: float,
    fixture: dict[str, Any],
) -> UniformMagneticSourceSample:
    duration = float(fixture["duration_au"])
    if time_au <= 0.0 or time_au >= duration:
        envelope = 0.0
        derivative = 0.0
    else:
        phase = math.pi * time_au / duration
        envelope = math.sin(phase) ** 2
        derivative = (math.pi / duration) * math.sin(2.0 * phase)
    origin = _vector3(fixture["electromagnetic_origin_au"])
    if case == "electric":
        electric_peak = np.asarray(fixture["electric_peak_au"], dtype=np.float64)
        return UniformMagneticSourceSample(
            time_au,
            UniformMagneticField((0.0, 0.0, 0.0)),
            electric_field_origin_au=tuple(envelope * electric_peak),
            origin_au=origin,
        )
    if case == "magnetic":
        magnetic_peak = np.asarray(fixture["magnetic_peak_au"], dtype=np.float64)
        return UniformMagneticSourceSample(
            time_au,
            UniformMagneticField(tuple(envelope * magnetic_peak)),
            magnetic_field_dot_au=tuple(derivative * magnetic_peak),
            origin_au=origin,
        )
    raise ValueError(f"unknown case {case!r}")


def _reference(fixture: dict[str, Any]) -> Any:
    return prepare_one_electron_ao_reference(
        OneElectronReferenceConfig(
            atoms=tuple(
                AtomConfig(str(atom["symbol"]), _vector3(atom["position_au"]))
                for atom in fixture["atoms"]
            ),
            basis=str(fixture["basis"]),
            electromagnetic_origin=ElectromagneticOrigin(
                _vector3(fixture["electromagnetic_origin_au"])
            ),
        )
    )


def _sample(
    quadrature: Any,
    fixture: dict[str, Any],
    case: str,
    time_au: float,
    cache: dict[float, Any],
) -> Any:
    key = float(time_au)
    if key not in cache:
        cache[key] = evaluate_exact_wilson_one_electron_sample(
            quadrature,
            _source(case, key, fixture),
        )
        print(
            f"wp7/{case} unique_sample={len(cache)} time_au={key:.12f}",
            flush=True,
        )
    return cache[key]


def _history(
    quadrature: Any,
    fixture: dict[str, Any],
    case: str,
    intervals: int,
    cache: dict[float, Any],
) -> ExperimentalGaussMagnusHistory:
    duration = float(fixture["duration_au"])
    step = duration / intervals
    offset = math.sqrt(3.0) / 6.0
    endpoints = tuple(
        _sample(quadrature, fixture, case, index * step, cache).metric
        for index in range(intervals + 1)
    )
    minus = tuple(
        exact_wilson_one_electron_triple(
            _sample(
                quadrature,
                fixture,
                case,
                (index + 0.5 - offset) * step,
                cache,
            )
        )
        for index in range(intervals)
    )
    plus = tuple(
        exact_wilson_one_electron_triple(
            _sample(
                quadrature,
                fixture,
                case,
                (index + 0.5 + offset) * step,
                cache,
            )
        )
        for index in range(intervals)
    )
    return ExperimentalGaussMagnusHistory(
        endpoint_metrics=endpoints,
        gauss_minus_triples=minus,
        gauss_plus_triples=plus,
        interval_au=step,
    )


def _initial_density(quadrature: Any, sample: Any) -> Any:
    backend = quadrature.backend
    metric = backend.to_host(sample.metric)
    mechanical = backend.to_host(sample.mechanical)
    _, coefficients = scipy.linalg.eigh(mechanical, metric)
    coefficient = coefficients[:, :1]
    density = coefficient @ coefficient.conj().T
    return backend.asarray(density, dtype=backend.namespace.complex128)


def _host_scalar(backend: Any, value: Any) -> float:
    return float(backend.scalar_to_float(backend.namespace.asarray(value)))


def _vector3(value: object) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("expected a finite Cartesian vector")
    return (float(array[0]), float(array[1]), float(array[2]))


def _diagnostics(
    quadrature: Any,
    fixture: dict[str, Any],
    case: str,
    intervals: int,
    cache: dict[float, Any],
    contravariant_densities: tuple[Any, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    backend = quadrature.backend
    duration = float(fixture["duration_au"])
    times = np.linspace(0.0, duration, intervals + 1)
    rows = np.empty((intervals + 1, 11), dtype=np.float64)
    mapping = backend.asarray(
        quadrature.reference.anchor_topology.ao_to_atom,
        dtype=backend.namespace.int64,
    )
    ward_parameter = backend.asarray(
        fixture["ward_site_parameter"],
        dtype=backend.namespace.float64,
    )
    ward_rate = backend.asarray(
        fixture["ward_site_parameter_rate"],
        dtype=backend.namespace.float64,
    )
    for index, (time_au, density) in enumerate(zip(times, contravariant_densities, strict=True)):
        sample = _sample(quadrature, fixture, case, float(time_au), cache)
        triple = exact_wilson_one_electron_triple(sample)
        velocity = one_electron_velocity_density(density, triple, backend)
        continuity = evaluate_exact_discrete_continuity(
            quadrature,
            sample,
            density,
            velocity_density=velocity,
        )
        gauge = exact_pure_gauge_action_direction(
            sample,
            density,
            velocity,
            ward_parameter,
            ward_rate,
            mapping,
            backend,
        )
        ward = restricted_one_electron_action_full_directional_derivative(
            density,
            velocity,
            triple,
            gauge.matrix,
            gauge.history,
            backend,
        )
        energy = backend.namespace.real(
            backend.namespace.einsum(
                "ij,ji->",
                density,
                sample.mechanical,
                optimize=True,
            )
        )
        # Magnetic-source power is outside the present uniform-electric
        # specialization; finite zero placeholders keep the array hashable,
        # while the case label makes those two columns explicitly inapplicable.
        power = 0.0
        power_residual = 0.0
        if case == "electric":
            power_value = evaluate_exact_uniform_electric_power(
                quadrature,
                sample,
                density,
                velocity_density=velocity,
            )
            power = _host_scalar(backend, power_value.source_work_rate)
            power_residual = _host_scalar(backend, power_value.power_residual)
        rows[index] = (
            time_au,
            _host_scalar(backend, continuity.total_electronic_charge),
            _host_scalar(backend, continuity.metric_particle_number),
            float(np.linalg.norm(backend.to_host(continuity.continuity_residual))),
            abs(_host_scalar(backend, ward.total.total)),
            _host_scalar(backend, energy),
            power,
            abs(power_residual),
            max(
                sample.connection.metric_compatibility_residual,
                sample.connection.direct_factorized_connection_residual,
                sample.connection.direct_factorized_metric_dot_residual,
            ),
            float(np.linalg.norm(backend.to_host(density - density.conj().T))),
            float(np.linalg.norm(backend.to_host(density @ sample.metric @ density - density))),
        )

    summary: dict[str, Any] = {
        "case": case,
        "intervals": intervals,
        "step_au": duration / intervals,
        "maximum_total_charge_error": float(np.max(np.abs(rows[:, 1] - rows[0, 1]))),
        "maximum_particle_number_error": float(np.max(np.abs(rows[:, 2] - rows[0, 2]))),
        "maximum_continuity_residual": float(np.max(rows[:, 3])),
        "maximum_ward_residual": float(np.max(rows[:, 4])),
        "maximum_matrix_identity_residual": float(np.max(rows[:, 8])),
        "maximum_density_hermiticity_residual": float(np.max(rows[:, 9])),
        "maximum_density_idempotency_residual": float(np.max(rows[:, 10])),
    }
    if case == "electric":
        work = float(np.trapezoid(rows[:, 6], rows[:, 0]))
        energy_change = float(rows[-1, 5] - rows[0, 5])
        derivative = np.gradient(rows[:, 5], rows[:, 0], edge_order=2)
        summary.update(
            {
                "maximum_instantaneous_power_residual": float(np.max(rows[:, 7])),
                "integrated_source_work": work,
                "mechanical_energy_change": energy_change,
                "integrated_work_defect": energy_change - work,
                "maximum_interior_trajectory_power_defect": float(
                    np.max(np.abs(derivative[2:-2] - rows[2:-2, 6]))
                ),
            }
        )
    return rows, summary


def _plan(
    fixture: dict[str, Any],
    args: argparse.Namespace,
    intervals: tuple[int, ...],
) -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--short")
    return {
        "schema": "aion.exact-one-electron-wp7-observables-plan",
        "version": "1.0.0",
        "status": "planned",
        "fixture": fixture,
        "backend": args.backend,
        "device_index": args.device_index if args.backend == "gpu" else None,
        "intervals": intervals,
        "cases": ["electric", "magnetic"],
        "provenance": {
            "repository": str(_REPOSITORY),
            "commit": commit,
            "git_status_short": status,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "fixture_sha256": _sha256(_FIXTURE),
            "conda_lock_sha256": _sha256(_LOCK),
            "runner_sha256": _sha256(Path(__file__).resolve()),
        },
    }


def main() -> None:
    args = _arguments()
    fixture = _json(_FIXTURE)
    declared = fixture["gpu_intervals"] if args.backend == "gpu" else fixture["cpu_intervals"]
    intervals = tuple(int(value) for value in declared)
    if args.smoke:
        intervals = (4,)
    plan = _plan(fixture, args, intervals)
    plan["plan_id"] = canonical_sha256(plan)
    if args.execution_directory is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        suffix = str(plan["provenance"]["commit"])[:12]
        execution = args.output_root.resolve() / f"wp7_observables_{args.backend}_{stamp}_{suffix}"
    else:
        execution = args.execution_directory.resolve()
    execution.mkdir(parents=True, exist_ok=True)
    with (execution / ".campaign.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _write_json(execution / "execution_plan.json", plan)
        start = time.time()
        reference = _reference(fixture)
        backend_config = (
            BackendConfig(BackendKind.GPU, device_index=args.device_index)
            if args.backend == "gpu"
            else BackendConfig()
        )
        quadrature = prepare_ao_quadrature(
            reference,
            backend_config,
            grid_policy=AOGridPolicy.qualification(int(fixture["grid_level"])),
            block_size=int(fixture["block_size"]),
        )
        arrays: dict[str, np.ndarray] = {}
        summaries: list[dict[str, Any]] = []
        for case in ("electric", "magnetic"):
            for count in intervals:
                cache: dict[float, Any] = {}
                history = _history(quadrature, fixture, case, count, cache)
                initial_sample = _sample(quadrature, fixture, case, 0.0, cache)
                initial_density = _initial_density(quadrature, initial_sample)
                initial_mixed = initial_density @ history.endpoint_metrics[0]
                trajectory = propagate_experimental_mixed_density(
                    initial_mixed,
                    history,
                    quadrature.backend,
                )
                diagnostic, summary = _diagnostics(
                    quadrature,
                    fixture,
                    case,
                    count,
                    cache,
                    trajectory.contravariant_densities,
                )
                name = f"{case}__n{count}"
                arrays[f"{name}__diagnostics"] = diagnostic
                arrays[f"{name}__density"] = np.stack(
                    [
                        quadrature.backend.to_host(value)
                        for value in trajectory.contravariant_densities
                    ]
                )
                summary.update(
                    {
                        "unique_matrix_samples": len(cache),
                        "maximum_cross_metric_residual": max(
                            value.cross_metric_residual for value in trajectory.diagnostics
                        ),
                        "maximum_trace_drift": max(
                            value.trace_drift for value in trajectory.diagnostics
                        ),
                    }
                )
                summaries.append(summary)
        result = {
            "schema": "aion.exact-one-electron-wp7-observables-result",
            "version": "1.0.0",
            "status": "executed_unreviewed",
            "plan_id": plan["plan_id"],
            "diagnostic_columns": [
                "time_au",
                "total_electronic_charge",
                "metric_particle_number",
                "continuity_residual_norm",
                "ward_residual_abs",
                "mechanical_energy_au",
                "source_power_au",
                "instantaneous_power_residual_abs",
                "matrix_identity_residual_max",
                "density_hermiticity_residual",
                "density_idempotency_residual",
            ],
            "summaries": summaries,
            "array_semantic_sha256": {
                name: canonical_sha256(value) for name, value in arrays.items()
            },
            "wall_time_seconds": time.time() - start,
        }
        arrays_path = execution / "arrays.npz"
        result_path = execution / "result.json"
        _write_npz(arrays_path, arrays)
        _write_json(result_path, result)
        _write_json(
            execution / "completed.json",
            {
                "schema": "aion.exact-one-electron-wp7-observables-completed",
                "version": "1.0.0",
                "status": result["status"],
                "plan_id": plan["plan_id"],
                "execution_plan_sha256": _sha256(execution / "execution_plan.json"),
                "result_sha256": _sha256(result_path),
                "arrays_sha256": _sha256(arrays_path),
            },
        )
        print(str(execution), flush=True)


if __name__ == "__main__":
    main()
