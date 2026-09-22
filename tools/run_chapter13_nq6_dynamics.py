#!/usr/bin/env python3
"""Execute the Chapter 13 NQ6 nonlinear dynamics and power campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from aion.config import BackendConfig
from aion.electromagnetism import (
    GaussianScalarGaugeVariation,
    UniformMagneticField,
    UniformMagneticSourceSample,
)
from aion.electronic_structure import (
    AOGridPolicy,
    DependencyVersions,
    ExactWilsonDynamicSample,
    PreparedExactWilsonDynamicSpatialAction,
    WilsonStationaryBranch,
    evaluate_exact_wilson_charge,
    evaluate_exact_wilson_power,
    evaluate_nonlinear_density_pure_gauge_ward,
    evaluate_nonlinear_weak_continuity,
    load_reference_data,
    prepare_ao_quadrature,
    prepare_exact_wilson_dynamic_spatial_action,
    prepare_exact_wilson_stationary_factory,
)
from aion.propagation import (
    NonlinearGaussMagnusPolicy,
    propagate_nonlinear_contravariant_density,
)

_CAMPAIGN_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "chapter13_wilson_adiabatic_qualification"
)
_NQ4_ROOT = _CAMPAIGN_ROOT / "nq4_stationary_20260921T232232Z_6197d6828bb3"
_NQ5_ROOT = _CAMPAIGN_ROOT / "nq5_sources_20260922T001500Z_23549ab7740a"
_FUNCTIONAL = "lda,vwn"
_AUXILIARY_BASIS = "weigend"
_GRID_LEVEL = 4
_GAUGE_ORIGIN = (0.17, -0.31, 0.23)
_STATIC_FIELD = 0.03
_FINAL_TIME = 2.0
_ELECTRIC_AMPLITUDE = 0.005
_MAGNETIC_AMPLITUDE = 0.005
_OCCUPATIONS = np.asarray((2.0,))
_BRANCHES = (
    WilsonStationaryBranch.HARTREE,
    WilsonStationaryBranch.KOHN_SHAM_LDA,
)
_CASES = (
    "stationary_static_b",
    "electric_field_free",
    "electric_static_b",
    "magnetic_induction",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return cast(str | bytes, completed.stdout)


def _float(value: object) -> float:
    return float(np.asarray(value).real)


def _relative(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value) / max(1.0, np.linalg.norm(reference)))


def _write_csv(output: Path, name: str, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with (output / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pulse_shape(time_au: float) -> float:
    if time_au <= 0.0 or time_au >= _FINAL_TIME:
        return 0.0
    return math.sin(math.pi * time_au / _FINAL_TIME) ** 2


def _pulse_shape_rate(time_au: float) -> float:
    if time_au <= 0.0 or time_au >= _FINAL_TIME:
        return 0.0
    return (math.pi / _FINAL_TIME) * math.sin(
        2.0 * math.pi * time_au / _FINAL_TIME
    )


def _source(case: str, time_au: float) -> UniformMagneticSourceSample:
    shape = _pulse_shape(time_au)
    if case == "stationary_static_b":
        field = _STATIC_FIELD
        field_rate = 0.0
        electric = 0.0
    elif case == "electric_field_free":
        field = 0.0
        field_rate = 0.0
        electric = _ELECTRIC_AMPLITUDE * shape
    elif case == "electric_static_b":
        field = _STATIC_FIELD
        field_rate = 0.0
        electric = _ELECTRIC_AMPLITUDE * shape
    elif case == "magnetic_induction":
        field = _STATIC_FIELD + _MAGNETIC_AMPLITUDE * shape
        field_rate = _MAGNETIC_AMPLITUDE * _pulse_shape_rate(time_au)
        electric = 0.0
    else:
        raise ValueError(f"unknown NQ6 case {case!r}")
    return UniformMagneticSourceSample(
        time_au=time_au,
        field=UniformMagneticField((0.0, 0.0, field)),
        magnetic_field_dot_au=(0.0, 0.0, field_rate),
        electric_field_origin_au=(electric, 0.0, 0.0),
        origin_au=_GAUGE_ORIGIN,
    )


def _source_key(source: UniformMagneticSourceSample) -> tuple[Any, ...]:
    return (
        source.time_au,
        source.field.magnetic_field_au,
        source.magnetic_field_dot_au,
        source.electric_field_origin_au,
        source.origin_au,
        source.gauge_kind.value,
        source.landau_axis,
    )


@dataclass(slots=True)
class _DynamicCache:
    factory: Any
    case: str
    spatial: dict[tuple[Any, ...], PreparedExactWilsonDynamicSpatialAction]
    temporal: dict[tuple[Any, ...], Any]
    dynamic: dict[tuple[Any, ...], ExactWilsonDynamicSample]

    @classmethod
    def create(cls, factory: Any, case: str) -> _DynamicCache:
        return cls(factory=factory, case=case, spatial={}, temporal={}, dynamic={})

    def at(
        self,
        time_au: float,
        branch: WilsonStationaryBranch,
    ) -> ExactWilsonDynamicSample:
        source = _source(self.case, time_au)
        source_key = _source_key(source)
        dynamic_key = (*source_key, branch.value)
        if dynamic_key in self.dynamic:
            return self.dynamic[dynamic_key]
        gauge = source.gauge
        spatial_key = (
            gauge.field.magnetic_field_au,
            gauge.kind.value,
            gauge.origin_au,
            gauge.landau_axis,
        )
        if spatial_key not in self.spatial:
            self.spatial[spatial_key] = prepare_exact_wilson_dynamic_spatial_action(
                self.factory,
                gauge,
            )
        spatial = self.spatial[spatial_key]
        if source_key not in self.temporal:
            self.temporal[source_key] = spatial.temporal_sample(source)
        result = spatial.sample_from_one_electron(self.temporal[source_key], branch)
        self.dynamic[dynamic_key] = result
        return result


def _initial_key(case: str, branch: WilsonStationaryBranch) -> str:
    field = "b0p000" if case == "electric_field_free" else "b0p030"
    return f"h3plus_{field}_{branch.value}"


def _simpson(values: list[float], interval: float) -> float:
    intervals = len(values) - 1
    if intervals <= 0 or intervals % 2:
        raise ValueError("composite Simpson integration requires a positive even count")
    return (interval / 3.0) * (
        values[0]
        + values[-1]
        + 4.0 * sum(values[1:-1:2])
        + 2.0 * sum(values[2:-1:2])
    )


def _run_trajectory(
    *,
    cache: _DynamicCache,
    branch: WilsonStationaryBranch,
    case: str,
    intervals: int,
    tolerance: float,
    initial_density: np.ndarray,
    initial_coefficients: np.ndarray,
    diagnostic_observables: bool,
    trajectory_rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    run_id: str,
) -> dict[str, Any]:
    backend = cache.factory.quadrature.backend
    step = _FINAL_TIME / intervals

    def metric_provider(time_au: float) -> Any:
        return cache.at(time_au, branch).one_electron.metric

    def eom_provider(time_au: float, density: Any) -> Any:
        sample = cache.at(time_au, branch)
        return sample.evaluate(density).triple

    propagation = propagate_nonlinear_contravariant_density(
        backend.asarray(initial_density),
        initial_time_au=0.0,
        interval_au=step,
        intervals=intervals,
        metric_provider=metric_provider,
        eom_provider=eom_provider,
        backend=backend,
        policy=NonlinearGaussMagnusPolicy(
            tolerance=tolerance,
            maximum_iterations=80,
        ),
        hbar=cache.factory.hbar,
    )

    coefficients = backend.asarray(initial_coefficients)
    coefficient_history = [np.asarray(coefficients)]
    for link in propagation.links:
        coefficients = link @ coefficients
        coefficient_history.append(np.asarray(coefficients))

    energies: list[float] = []
    source_powers: list[float] = []
    matrix_rates: list[float] = []
    ward_residuals: list[float] = []
    continuity_residuals: list[float] = []
    global_charge_residuals: list[float] = []
    gauge_test = GaussianScalarGaugeVariation(
        amplitude=0.37,
        center_au=(0.11, -0.17, 0.23),
        exponent_au_inverse2=0.41,
    )
    diagnostic_stride = max(1, intervals // 8)

    for index, (time_au, mixed_density, density) in enumerate(
        zip(
            propagation.times_au,
            propagation.mixed_densities,
            propagation.contravariant_densities,
            strict=True,
        )
    ):
        sample = cache.at(time_au, branch)
        evaluation = sample.evaluate(density)
        power = evaluate_exact_wilson_power(evaluation, density)
        charge = evaluate_exact_wilson_charge(sample.model, density)
        energy = _float(evaluation.action.energy_molecular_total_au)
        source_power = _float(power.source_power_au)
        matrix_rate = _float(power.matrix_mechanical_energy_rate_au)
        energies.append(energy)
        source_powers.append(source_power)
        matrix_rates.append(matrix_rate)

        coefficient = coefficient_history[index]
        metric = np.asarray(sample.one_electron.metric)
        density_host = np.asarray(density)
        mixed_host = np.asarray(mixed_density)
        gram = coefficient.conj().T @ metric @ coefficient
        reconstructed = (coefficient * _OCCUPATIONS[None, :]) @ coefficient.conj().T
        polynomial = mixed_host @ mixed_host - 2.0 * mixed_host
        diagnostics = None if index == 0 else propagation.diagnostics[index - 1]
        row = {
            "run_id": run_id,
            "case": case,
            "branch": branch.value,
            "intervals": intervals,
            "nonlinear_tolerance": tolerance,
            "step_index": index,
            "time_au": time_au,
            "energy_molecular_total_au": energy,
            "source_power_au": source_power,
            "matrix_energy_rate_au": matrix_rate,
            "instantaneous_power_residual_au": abs(
                _float(power.power_identity_residual_au)
            ),
            "particle_number_mixed": _float(np.trace(mixed_host)),
            "particle_number_metric": _float(charge.metric_particle_number),
            "particle_number_grid": -_float(charge.integrated_charge_grid),
            "charge_grid_metric_residual": abs(
                _float(charge.integrated_charge_grid)
                - _float(charge.integrated_charge_metric)
            ),
            "metric_orthonormality_residual": _relative(
                gram - np.eye(gram.shape[0]),
                np.eye(gram.shape[0]),
            ),
            "density_reconstruction_residual": _relative(
                density_host - reconstructed,
                density_host,
            ),
            "mixed_density_polynomial_residual": _relative(polynomial, mixed_host),
            "contravariant_hermiticity_residual": _relative(
                density_host - density_host.conj().T,
                density_host,
            ),
            "metric_compatibility_residual": (
                sample.one_electron.connection.metric_compatibility_residual
            ),
            "nonlinear_iterations": 0 if diagnostics is None else diagnostics.nonlinear_iterations,
            "nonlinear_residual": 0.0 if diagnostics is None else diagnostics.nonlinear_residual,
            "cross_metric_residual": (
                0.0 if diagnostics is None else diagnostics.cross_metric_residual
            ),
            "occupation_spectrum_drift": (
                0.0 if diagnostics is None else diagnostics.occupation_spectrum_drift
            ),
            "metric_correction_applied": propagation.metric_correction_applied,
        }

        if diagnostic_observables and (
            index in (0, intervals) or index % diagnostic_stride == 0
        ):
            ward = evaluate_nonlinear_density_pure_gauge_ward(
                sample.model,
                sample.one_electron,
                density,
                power.velocity_density,
                gauge_test,
            )
            continuity = evaluate_nonlinear_weak_continuity(
                sample.model,
                sample.one_electron,
                density,
                gauge_test,
            )
            ward_value = abs(_float(ward.total_ward_residual))
            continuity_value = abs(_float(continuity.finite_region_residual))
            global_value = abs(_float(continuity.global_charge_residual))
            ward_residuals.append(ward_value)
            continuity_residuals.append(continuity_value)
            global_charge_residuals.append(global_value)
            row.update(
                {
                    "ward_residual": ward_value,
                    "density_shell_residual": (
                        ward.lower_density_shell_residual_relative_norm
                    ),
                    "finite_region_continuity_residual": continuity_value,
                    "global_charge_residual": global_value,
                }
            )
        trajectory_rows.append(row)

    for index in range(1, intervals):
        trajectory_rows[-(intervals + 1) + index]["centered_energy_rate_au"] = (
            energies[index + 1] - energies[index - 1]
        ) / (2.0 * step)
        trajectory_rows[-(intervals + 1) + index][
            "centered_energy_source_power_residual_au"
        ] = abs(
            (energies[index + 1] - energies[index - 1]) / (2.0 * step)
            - source_powers[index]
        )

    source_work = _simpson(source_powers, step)
    matrix_work = _simpson(matrix_rates, step)
    energy_change = energies[-1] - energies[0]
    arrays[f"{run_id}_times_au"] = np.asarray(propagation.times_au)
    arrays[f"{run_id}_final_mixed_density"] = np.asarray(
        propagation.mixed_densities[-1]
    )
    arrays[f"{run_id}_energies_au"] = np.asarray(energies)
    arrays[f"{run_id}_source_powers_au"] = np.asarray(source_powers)
    arrays[f"{run_id}_matrix_rates_au"] = np.asarray(matrix_rates)
    return {
        "run_id": run_id,
        "case": case,
        "branch": branch.value,
        "intervals": intervals,
        "interval_au": step,
        "nonlinear_tolerance": tolerance,
        "diagnostic_observables": diagnostic_observables,
        "energy_change_au": energy_change,
        "integrated_source_work_au": source_work,
        "integrated_matrix_rate_au": matrix_work,
        "source_work_endpoint_energy_residual_au": abs(source_work - energy_change),
        "matrix_work_endpoint_energy_residual_au": abs(matrix_work - energy_change),
        "instantaneous_power_residual_max_au": max(
            abs(left - right) for left, right in zip(matrix_rates, source_powers, strict=True)
        ),
        "nonlinear_residual_max": max(
            value.nonlinear_residual for value in propagation.diagnostics
        ),
        "nonlinear_iterations_max": max(
            value.nonlinear_iterations for value in propagation.diagnostics
        ),
        "cross_metric_residual_max": max(
            value.cross_metric_residual for value in propagation.diagnostics
        ),
        "trace_drift_max": max(value.trace_drift for value in propagation.diagnostics),
        "occupation_spectrum_drift_max": max(
            value.occupation_spectrum_drift for value in propagation.diagnostics
        ),
        "ward_residual_max": max(ward_residuals, default=None),
        "finite_region_continuity_residual_max": max(
            continuity_residuals,
            default=None,
        ),
        "global_charge_residual_max": max(global_charge_residuals, default=None),
        "metric_correction_applied": propagation.metric_correction_applied,
        "density_update": propagation.density_update,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("smoke", "qualification"),
        default="qualification",
    )
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact directory: {output}")
    output.mkdir(parents=True)
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    repo = Path(__file__).resolve().parents[1]

    accepted_hashes: dict[str, dict[str, str]] = {}
    for label, root in (("NQ4", _NQ4_ROOT), ("NQ5", _NQ5_ROOT)):
        completed = json.loads((root / "completed.json").read_text(encoding="utf-8"))
        if completed["result_sha256"] != _sha256(root / "result.json"):
            raise RuntimeError(f"accepted {label} result hash changed")
        if completed["provenance_sha256"] != _sha256(root / "provenance.json"):
            raise RuntimeError(f"accepted {label} provenance hash changed")
        accepted_hashes[label] = {
            "root": str(root),
            "result_sha256": completed["result_sha256"],
            "provenance_sha256": completed["provenance_sha256"],
            "completed_sha256": _sha256(root / "completed.json"),
        }

    reference = load_reference_data(_NQ4_ROOT / "h3plus.reference.h5")
    accepted_states = np.load(_NQ4_ROOT / "stationary_states.npz", allow_pickle=False)
    selected_grid_level = 2 if arguments.profile == "smoke" else _GRID_LEVEL
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(selected_grid_level),
        block_size=2048,
    )
    factory = prepare_exact_wilson_stationary_factory(
        quadrature,
        auxiliary_basis=_AUXILIARY_BASIS,
        functional=_FUNCTIONAL,
    )

    branches: tuple[WilsonStationaryBranch, ...]
    cases: tuple[str, ...]
    run_specs: tuple[tuple[int, float, bool, str], ...]
    if arguments.profile == "smoke":
        branches = (WilsonStationaryBranch.KOHN_SHAM_LDA,)
        cases = ("stationary_static_b", "magnetic_induction")
        run_specs = ((8, 1.0e-8, True, "smoke"),)
    else:
        branches = _BRANCHES
        cases = _CASES
        run_specs = tuple(
            (intervals, 1.0e-12, intervals == 32, "timestep")
            for intervals in (4, 8, 16, 32)
        ) + tuple(
            (16, tolerance, False, "nonlinear_tolerance")
            for tolerance in (1.0e-6, 1.0e-8, 1.0e-10)
        )

    trajectory_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    caches = {case: _DynamicCache.create(factory, case) for case in cases}
    for case in cases:
        for branch in branches:
            key = _initial_key(case, branch)
            initial_density = np.asarray(
                accepted_states[f"{key}_density"],
                dtype=np.complex128,
            )
            initial_coefficients = np.asarray(
                accepted_states[f"{key}_coefficients"],
                dtype=np.complex128,
            )
            for intervals, tolerance, diagnostic, refinement in run_specs:
                run_id = (
                    f"{case}_{branch.value}_n{intervals}_"
                    f"tol{tolerance:.0e}_{refinement}"
                ).replace("+", "")
                print(f"starting {run_id}", flush=True)
                try:
                    summary = _run_trajectory(
                        cache=caches[case],
                        branch=branch,
                        case=case,
                        intervals=intervals,
                        tolerance=tolerance,
                        initial_density=initial_density,
                        initial_coefficients=initial_coefficients,
                        diagnostic_observables=diagnostic,
                        trajectory_rows=trajectory_rows,
                        arrays=arrays,
                        run_id=run_id,
                    )
                except Exception as exc:
                    failure = {
                        "schema": "aion.chapter13-nq6-failed-run",
                        "schema_version": "1.0.0",
                        "status": "failed_visible",
                        "run_id": run_id,
                        "case": case,
                        "branch": branch.value,
                        "intervals": intervals,
                        "nonlinear_tolerance": tolerance,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    (output / "failure.json").write_text(
                        json.dumps(failure, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    raise
                summary["refinement"] = refinement
                summaries.append(summary)
                print(
                    f"completed {run_id}: work error "
                    f"{summary['source_work_endpoint_energy_residual_au']:.3e}",
                    flush=True,
                )

    _write_csv(output, "trajectory.csv", trajectory_rows)
    _write_csv(output, "summary.csv", summaries)
    np.savez_compressed(output / "trajectories.npz", **arrays)  # type: ignore[arg-type]
    result = {
        "schema": "aion.chapter13-nq6-dynamics",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "profile": arguments.profile,
        "equations": [
            "eq:wilson-hartree-ks-density-eom",
            "eq:wilson-hartree-ks-power-identity",
            "eq:wilson-hartree-ks-power-field-current",
            "eq:wilson-hartree-ks-energy-matrix-derivative",
        ],
        "realization": {
            "system": "H3+",
            "charge": 1,
            "electrons": 2,
            "basis": "cc-pvdz",
            "functional": _FUNCTIONAL,
            "auxiliary_basis": _AUXILIARY_BASIS,
            "grid_level": selected_grid_level,
            "gauge_origin_au": list(_GAUGE_ORIGIN),
            "final_time_au": _FINAL_TIME,
            "electric_amplitude_au": _ELECTRIC_AMPLITUDE,
            "static_magnetic_field_au": _STATIC_FIELD,
            "magnetic_pulse_amplitude_au": _MAGNETIC_AMPLITUDE,
            "branches": [branch.value for branch in branches],
            "cases": list(cases),
            "precision": "float64_complex128",
            "propagator": (
                "self-consistent two-node fourth-order Gauss-Magnus with [2/2] "
                "Pade coefficient links and contravariant-density congruence"
            ),
            "metric_projection_or_correction": False,
            "current_label": "auxiliary Hartree/adiabatic-KS action source current",
            "not_claimed": "interacting physical transverse current",
        },
        "accepted_inputs": accepted_hashes,
        "trajectory_summaries": summaries,
    }
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tracked_diff = _git(repo, "diff", "--binary", "HEAD", binary=True)
    assert isinstance(tracked_diff, bytes)
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    assert isinstance(status, str)
    head = _git(repo, "rev-parse", "HEAD")
    assert isinstance(head, str)
    source_paths = (
        repo / "tools/run_chapter13_nq6_dynamics.py",
        repo / "src/aion/propagation/__init__.py",
        repo / "src/aion/propagation/tensorial.py",
        repo / "src/aion/electromagnetism/__init__.py",
        repo / "src/aion/electromagnetism/test_variations.py",
        repo / "src/aion/electronic_structure/__init__.py",
        repo / "src/aion/electronic_structure/ri_wilson_hartree.py",
        repo / "src/aion/electronic_structure/time_connection.py",
        repo / "src/aion/electronic_structure/wilson_dynamics.py",
        repo / "src/aion/electronic_structure/wilson_sources.py",
        repo / "src/aion/electronic_structure/wilson_stationary.py",
        repo / "environment.yml",
        repo / "conda-linux-64.lock",
    )
    artifact_paths = tuple(
        sorted(
            path
            for path in output.iterdir()
            if path.is_file() and path.name not in {"provenance.json", "completed.json"}
        )
    )
    provenance = {
        "timestamp_utc": timestamp,
        "command": " ".join(shlex.quote(value) for value in (sys.executable, *sys.argv)),
        "repository": str(repo),
        "git_head": head.strip(),
        "git_status_porcelain": status.splitlines(),
        "git_tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "environment": {
            "dependencies": DependencyVersions.current().as_mapping(),
            "loaded_modules": os.environ.get("LOADEDMODULES", ""),
            "thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "python_executable": sys.executable,
        },
        "source_sha256": {str(path): _sha256(path) for path in source_paths},
        "artifacts_sha256": {path.name: _sha256(path) for path in artifact_paths},
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completed = {
        "schema": "aion.chapter13-nq6-completed",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "result_sha256": _sha256(result_path),
        "provenance_sha256": _sha256(provenance_path),
    }
    (output / "completed.json").write_text(
        json.dumps(completed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "status": "executed_unreviewed"}, indent=2))


if __name__ == "__main__":
    main()
