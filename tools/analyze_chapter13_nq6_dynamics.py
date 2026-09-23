#!/usr/bin/env python3
"""Authenticate and analyze one Chapter 13 NQ6 dynamics campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_CASES = (
    "stationary_static_b",
    "electric_field_free",
    "electric_static_b",
    "magnetic_induction",
)
_DYNAMIC_CASES = _CASES[1:]
_BRANCHES = ("hartree", "kohn_sham_lda")
_INTERVALS = (4, 8, 16, 32)
_TOLERANCES = (1.0e-6, 1.0e-8, 1.0e-10)
_REFERENCE_TOLERANCE = 1.0e-12

_CASE_LABELS = {
    "stationary_static_b": "stationary, static B",
    "electric_field_free": "electric, B=0",
    "electric_static_b": "electric, static B",
    "magnetic_induction": "time-dependent B",
}
_BRANCH_LABELS = {"hartree": "Hartree", "kohn_sham_lda": "LDA KS"}
_COLORS = {
    "electric_field_free": "#1f77b4",
    "electric_static_b": "#ff7f0e",
    "magnetic_induction": "#2ca02c",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def _optional_float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    return None if value == "" else float(value)


def _relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(1.0, np.linalg.norm(right)))


def _simpson(values: np.ndarray, interval: float) -> float:
    intervals = len(values) - 1
    if intervals <= 0 or intervals % 2:
        raise ValueError("composite Simpson integration requires a positive even count")
    return float(
        (interval / 3.0)
        * (
            values[0]
            + values[-1]
            + 4.0 * np.sum(values[1:-1:2])
            + 2.0 * np.sum(values[2:-1:2])
        )
    )


def _run_id(
    case: str,
    branch: str,
    intervals: int,
    tolerance: float,
    refinement: str,
) -> str:
    return (
        f"{case}_{branch}_n{intervals}_tol{tolerance:.0e}_{refinement}"
    ).replace("+", "")


def _authenticate(raw: Path, result: dict[str, Any]) -> dict[str, Any]:
    completed_path = raw / "completed.json"
    provenance_path = raw / "provenance.json"
    result_path = raw / "result.json"
    completed = _load_json(completed_path)
    provenance = _load_json(provenance_path)
    if completed["result_sha256"] != _sha256(result_path):
        raise RuntimeError("completed record does not authenticate result.json")
    if completed["provenance_sha256"] != _sha256(provenance_path):
        raise RuntimeError("completed record does not authenticate provenance.json")
    for name, expected in provenance["artifacts_sha256"].items():
        path = raw / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"provenance hash mismatch for {name}")

    accepted_input_hashes: dict[str, dict[str, str]] = {}
    for label, record in result["accepted_inputs"].items():
        root = Path(record["root"])
        actual = {
            "result_sha256": _sha256(root / "result.json"),
            "provenance_sha256": _sha256(root / "provenance.json"),
            "completed_sha256": _sha256(root / "completed.json"),
        }
        for name, value in actual.items():
            if value != record[name]:
                raise RuntimeError(f"accepted {label} {name} changed")
        accepted_input_hashes[str(label)] = actual
    return {
        "result_sha256": _sha256(result_path),
        "provenance_sha256": _sha256(provenance_path),
        "completed_sha256": _sha256(completed_path),
        "accepted_inputs": accepted_input_hashes,
    }


def _expected_run_ids() -> set[str]:
    ids: set[str] = set()
    for case in _CASES:
        for branch in _BRANCHES:
            for intervals in _INTERVALS:
                ids.add(
                    _run_id(
                        case,
                        branch,
                        intervals,
                        _REFERENCE_TOLERANCE,
                        "timestep",
                    )
                )
            for tolerance in _TOLERANCES:
                ids.add(
                    _run_id(
                        case,
                        branch,
                        16,
                        tolerance,
                        "nonlinear_tolerance",
                    )
                )
    return ids


def _validate_matrix(
    raw: Path,
    result: dict[str, Any],
    summary_rows: list[dict[str, str]],
    trajectory_rows: list[dict[str, str]],
    arrays: Any,
) -> dict[str, Any]:
    expected = _expected_run_ids()
    summary_ids = [row["run_id"] for row in summary_rows]
    result_ids = [str(row["run_id"]) for row in result["trajectory_summaries"]]
    if len(summary_ids) != len(set(summary_ids)):
        raise RuntimeError("summary.csv contains duplicate run IDs")
    if set(summary_ids) != expected or set(result_ids) != expected:
        raise RuntimeError("NQ6 run matrix is incomplete or contains unexpected runs")
    if len(result_ids) != len(set(result_ids)):
        raise RuntimeError("result.json contains duplicate trajectory summaries")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[row["run_id"]].append(row)
    if set(grouped) != expected:
        raise RuntimeError("trajectory.csv run matrix differs from summary matrix")

    expected_array_keys: set[str] = set()
    numeric_recomputation_max = 0.0
    for row in summary_rows:
        run_id = row["run_id"]
        intervals = int(row["intervals"])
        records = grouped[run_id]
        if len(records) != intervals + 1:
            raise RuntimeError(f"wrong trajectory row count for {run_id}")
        for suffix in (
            "times_au",
            "final_mixed_density",
            "energies_au",
            "source_powers_au",
            "matrix_rates_au",
        ):
            expected_array_keys.add(f"{run_id}_{suffix}")
        times = np.asarray(arrays[f"{run_id}_times_au"])
        energies = np.asarray(arrays[f"{run_id}_energies_au"])
        source_powers = np.asarray(arrays[f"{run_id}_source_powers_au"])
        matrix_rates = np.asarray(arrays[f"{run_id}_matrix_rates_au"])
        trajectory_arrays = (times, energies, source_powers, matrix_rates)
        if not all(len(values) == intervals + 1 for values in trajectory_arrays):
            raise RuntimeError(f"wrong array length for {run_id}")
        step = float(row["interval_au"])
        independently_recomputed = {
            "energy_change_au": float(energies[-1] - energies[0]),
            "integrated_source_work_au": _simpson(source_powers, step),
            "integrated_matrix_rate_au": _simpson(matrix_rates, step),
        }
        for field, value in independently_recomputed.items():
            delta = abs(value - float(row[field]))
            numeric_recomputation_max = max(numeric_recomputation_max, delta)
            if delta > 5.0e-15:
                raise RuntimeError(f"independent recomputation failed for {run_id} {field}")
    if set(arrays.files) != expected_array_keys:
        raise RuntimeError("trajectories.npz contains missing or unexpected arrays")
    if (raw / "failure.json").exists():
        raise RuntimeError("completed campaign contains a visible failed trajectory")
    return {
        "expected_run_count": len(expected),
        "trajectory_row_count": len(trajectory_rows),
        "array_count": len(arrays.files),
        "numeric_recomputation_max_abs": numeric_recomputation_max,
        "failure_record_present": False,
    }


def _index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["run_id"]: row for row in rows}


def _timestep_analysis(
    summary_rows: list[dict[str, str]], arrays: Any
) -> list[dict[str, Any]]:
    indexed = _index(summary_rows)
    output: list[dict[str, Any]] = []
    for case in _CASES:
        for branch in _BRANCHES:
            pair_differences: dict[int, float] = {}
            for coarse, fine in pairwise(_INTERVALS):
                coarse_id = _run_id(
                    case, branch, coarse, _REFERENCE_TOLERANCE, "timestep"
                )
                fine_id = _run_id(
                    case, branch, fine, _REFERENCE_TOLERANCE, "timestep"
                )
                pair_differences[coarse] = _relative_difference(
                    np.asarray(arrays[f"{coarse_id}_final_mixed_density"]),
                    np.asarray(arrays[f"{fine_id}_final_mixed_density"]),
                )
            for position, intervals in enumerate(_INTERVALS):
                run_id = _run_id(
                    case, branch, intervals, _REFERENCE_TOLERANCE, "timestep"
                )
                row = indexed[run_id]
                order = None
                if position < len(_INTERVALS) - 2:
                    order = float(
                        np.log2(
                            pair_differences[intervals]
                            / pair_differences[_INTERVALS[position + 1]]
                        )
                    )
                output.append(
                    {
                        "case": case,
                        "branch": branch,
                        "run_id": run_id,
                        "intervals": intervals,
                        "interval_au": _float(row, "interval_au"),
                        "final_density_difference_to_next": pair_differences.get(intervals),
                        "observed_successive_difference_order": order,
                        "source_work_endpoint_energy_residual_au": _float(
                            row, "source_work_endpoint_energy_residual_au"
                        ),
                        "instantaneous_power_residual_max_au": _float(
                            row, "instantaneous_power_residual_max_au"
                        ),
                        "cross_metric_residual_max": _float(
                            row, "cross_metric_residual_max"
                        ),
                        "trace_drift_max": _float(row, "trace_drift_max"),
                        "occupation_spectrum_drift_max": _float(
                            row, "occupation_spectrum_drift_max"
                        ),
                    }
                )
    return output


def _tolerance_analysis(
    summary_rows: list[dict[str, str]], arrays: Any
) -> list[dict[str, Any]]:
    indexed = _index(summary_rows)
    output: list[dict[str, Any]] = []
    for case in _CASES:
        for branch in _BRANCHES:
            reference_id = _run_id(
                case, branch, 16, _REFERENCE_TOLERANCE, "timestep"
            )
            reference_row = indexed[reference_id]
            reference_density = np.asarray(
                arrays[f"{reference_id}_final_mixed_density"]
            )
            for tolerance in (*_TOLERANCES, _REFERENCE_TOLERANCE):
                run_id = (
                    reference_id
                    if tolerance == _REFERENCE_TOLERANCE
                    else _run_id(
                        case,
                        branch,
                        16,
                        tolerance,
                        "nonlinear_tolerance",
                    )
                )
                row = indexed[run_id]
                output.append(
                    {
                        "case": case,
                        "branch": branch,
                        "run_id": run_id,
                        "nonlinear_tolerance": tolerance,
                        "final_density_difference_from_1e-12": _relative_difference(
                            np.asarray(arrays[f"{run_id}_final_mixed_density"]),
                            reference_density,
                        ),
                        "energy_change_difference_from_1e-12_au": abs(
                            _float(row, "energy_change_au")
                            - _float(reference_row, "energy_change_au")
                        ),
                        "source_work_difference_from_1e-12_au": abs(
                            _float(row, "integrated_source_work_au")
                            - _float(reference_row, "integrated_source_work_au")
                        ),
                        "nonlinear_residual_max": _float(
                            row, "nonlinear_residual_max"
                        ),
                        "nonlinear_iterations_max": int(
                            row["nonlinear_iterations_max"]
                        ),
                    }
                )
    return output


def _finest_diagnostics(
    trajectory_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[row["run_id"]].append(row)
    fields = (
        "metric_orthonormality_residual",
        "density_reconstruction_residual",
        "mixed_density_polynomial_residual",
        "contravariant_hermiticity_residual",
        "metric_compatibility_residual",
        "charge_grid_metric_residual",
        "instantaneous_power_residual_au",
        "ward_residual",
        "finite_region_continuity_residual",
        "global_charge_residual",
    )
    output: list[dict[str, Any]] = []
    for case in _CASES:
        for branch in _BRANCHES:
            run_id = _run_id(
                case, branch, 32, _REFERENCE_TOLERANCE, "timestep"
            )
            records = grouped[run_id]
            result: dict[str, Any] = {
                "case": case,
                "branch": branch,
                "run_id": run_id,
                "particle_number_metric_residual_max": max(
                    abs(_float(row, "particle_number_metric") - 2.0)
                    for row in records
                ),
                "particle_number_mixed_residual_max": max(
                    abs(_float(row, "particle_number_mixed") - 2.0)
                    for row in records
                ),
                "particle_number_grid_residual_max": max(
                    abs(_float(row, "particle_number_grid") - 2.0)
                    for row in records
                ),
            }
            for field in fields:
                values = [
                    value
                    for row in records
                    if (value := _optional_float(row, field)) is not None
                ]
                result[f"{field}_max"] = max(map(abs, values), default=None)
            output.append(result)
    return output


def _max_present(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return max(values)


def _analyze(
    summary_rows: list[dict[str, str]],
    timestep_rows: list[dict[str, Any]],
    tolerance_rows: list[dict[str, Any]],
    finest_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_index = _index(summary_rows)
    dynamic_orders = [
        float(row["observed_successive_difference_order"])
        for row in timestep_rows
        if row["case"] in _DYNAMIC_CASES
        and row["observed_successive_difference_order"] is not None
    ]
    finest_timestep = [row for row in timestep_rows if row["intervals"] == 32]
    finest_summary = [
        summary_index[str(row["run_id"])] for row in finest_timestep
    ]
    tolerance_nonreference = [
        row
        for row in tolerance_rows
        if row["nonlinear_tolerance"] != _REFERENCE_TOLERANCE
    ]
    stationary_finest = [
        row for row in finest_summary if row["case"] == "stationary_static_b"
    ]
    metrics = {
        "dynamic_final_density_order_min": min(dynamic_orders),
        "finest_final_density_successive_difference_max": max(
            float(row["final_density_difference_to_next"])
            for row in timestep_rows
            if row["intervals"] == 16 and row["case"] in _DYNAMIC_CASES
        ),
        "finest_source_work_endpoint_energy_residual_max_au": max(
            _float(row, "source_work_endpoint_energy_residual_au")
            for row in finest_summary
        ),
        "stationary_finest_energy_drift_max_au": max(
            abs(_float(row, "energy_change_au")) for row in stationary_finest
        ),
        "nonlinear_tolerance_final_density_sensitivity_max": max(
            float(row["final_density_difference_from_1e-12"])
            for row in tolerance_nonreference
        ),
        "nonlinear_tolerance_energy_sensitivity_max_au": max(
            float(row["energy_change_difference_from_1e-12_au"])
            for row in tolerance_nonreference
        ),
        "finest_cross_metric_residual_max": max(
            _float(row, "cross_metric_residual_max") for row in finest_summary
        ),
        "finest_trace_drift_max": max(
            _float(row, "trace_drift_max") for row in finest_summary
        ),
        "finest_occupation_spectrum_drift_max": max(
            _float(row, "occupation_spectrum_drift_max")
            for row in finest_summary
        ),
        "finest_metric_orthonormality_residual_max": _max_present(
            finest_rows, "metric_orthonormality_residual_max"
        ),
        "finest_density_reconstruction_residual_max": _max_present(
            finest_rows, "density_reconstruction_residual_max"
        ),
        "finest_mixed_density_polynomial_residual_max": _max_present(
            finest_rows, "mixed_density_polynomial_residual_max"
        ),
        "finest_contravariant_hermiticity_residual_max": _max_present(
            finest_rows, "contravariant_hermiticity_residual_max"
        ),
        "finest_particle_number_metric_residual_max": _max_present(
            finest_rows, "particle_number_metric_residual_max"
        ),
        "finest_particle_number_grid_residual_max": _max_present(
            finest_rows, "particle_number_grid_residual_max"
        ),
        "finest_ward_residual_max": _max_present(
            finest_rows, "ward_residual_max"
        ),
        "finest_continuity_residual_max": _max_present(
            finest_rows, "finite_region_continuity_residual_max"
        ),
        "finest_global_charge_residual_max": _max_present(
            finest_rows, "global_charge_residual_max"
        ),
        "finest_instantaneous_power_residual_max_au": _max_present(
            finest_rows, "instantaneous_power_residual_au_max"
        ),
        "metric_correction_count": sum(
            row["metric_correction_applied"].lower() == "true"
            for row in summary_rows
        ),
    }
    upper_bounds = {
        "finest_final_density_successive_difference_max": 1.0e-7,
        "finest_source_work_endpoint_energy_residual_max_au": 6.0e-11,
        "stationary_finest_energy_drift_max_au": 5.0e-13,
        "nonlinear_tolerance_final_density_sensitivity_max": 1.0e-10,
        "nonlinear_tolerance_energy_sensitivity_max_au": 1.0e-9,
        "finest_cross_metric_residual_max": 1.0e-9,
        "finest_trace_drift_max": 5.0e-12,
        "finest_occupation_spectrum_drift_max": 2.0e-11,
        "finest_metric_orthonormality_residual_max": 2.0e-11,
        "finest_density_reconstruction_residual_max": 1.0e-12,
        "finest_mixed_density_polynomial_residual_max": 4.0e-11,
        "finest_contravariant_hermiticity_residual_max": 1.0e-12,
        "finest_particle_number_metric_residual_max": 4.0e-11,
        "finest_particle_number_grid_residual_max": 2.0e-8,
        "finest_ward_residual_max": 1.0e-12,
        "finest_continuity_residual_max": 3.0e-12,
        "finest_global_charge_residual_max": 1.0e-12,
        "finest_instantaneous_power_residual_max_au": 5.0e-13,
        "metric_correction_count": 0.0,
    }
    lower_bounds = {"dynamic_final_density_order_min": 3.7}
    checks = {
        **{name: metrics[name] <= limit for name, limit in upper_bounds.items()},
        **{name: metrics[name] >= limit for name, limit in lower_bounds.items()},
    }
    checks["density_update_is_uncorrected_congruence"] = all(
        row["density_update"] == "coefficient_congruence" for row in summary_rows
    )
    checks["all_nonlinear_steps_converged"] = all(
        _float(row, "nonlinear_residual_max")
        <= max(1.0e-12, _float(row, "nonlinear_tolerance"))
        for row in summary_rows
    )
    return {
        "status": "analyzed_unreviewed",
        "metrics": metrics,
        "proposed_upper_bound_thresholds": upper_bounds,
        "proposed_lower_bound_thresholds": lower_bounds,
        "checks": checks,
        "passed_proposed_thresholds": all(checks.values()),
        "interpretation_boundaries": [
            (
                "The observed fourth-order convergence applies to the three "
                "driven cases.  The stationary case is already at the floating-point "
                "and nonlinear-solver floor and is not used to fit a time order."
            ),
            (
                "The Kohn--Sham power and current are those of the declared auxiliary "
                "adiabatic LDA action, not an exact interacting transverse current."
            ),
            (
                "The level-4 real-space particle-number residual is a spatial-grid "
                "quadrature floor; metric particle number is tested separately."
            ),
            (
                "The campaign covers one H3+ geometry, cc-pVDZ/weigend, one pulse "
                "amplitude, one static magnetic field, and one pure LDA functional."
            ),
        ],
    }


def _line_style(branch: str) -> str:
    return "-" if branch == "hartree" else "--"


def _plot_timestep_convergence(
    rows: list[dict[str, Any]], output: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
    for case in _DYNAMIC_CASES:
        for branch in _BRANCHES:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["case"] == case and row["branch"] == branch
                ),
                key=lambda row: int(row["intervals"]),
            )
            label = f"{_CASE_LABELS[case]}, {_BRANCH_LABELS[branch]}"
            color = _COLORS[case]
            linestyle = _line_style(branch)
            difference_rows = [
                row
                for row in selected
                if row["final_density_difference_to_next"] is not None
            ]
            axes[0, 0].loglog(
                [row["interval_au"] for row in difference_rows],
                [row["final_density_difference_to_next"] for row in difference_rows],
                marker="o",
                color=color,
                linestyle=linestyle,
                label=label,
            )
            axes[0, 1].loglog(
                [row["interval_au"] for row in selected],
                [row["source_work_endpoint_energy_residual_au"] for row in selected],
                marker="o",
                color=color,
                linestyle=linestyle,
                label=label,
            )
            axes[1, 0].loglog(
                [row["interval_au"] for row in selected],
                [row["cross_metric_residual_max"] for row in selected],
                marker="o",
                color=color,
                linestyle=linestyle,
                label=label,
            )
            axes[1, 1].loglog(
                [row["interval_au"] for row in selected],
                [row["occupation_spectrum_drift_max"] for row in selected],
                marker="o",
                color=color,
                linestyle=linestyle,
                label=label,
            )
    guide_h = np.asarray((0.5, 0.25, 0.125))
    axes[0, 0].loglog(
        guide_h,
        1.2e-5 * (guide_h / 0.5) ** 4,
        color="0.25",
        linestyle=":",
        linewidth=1.5,
        label=r"$O(\Delta t^4)$ guide",
    )
    titles = (
        "Final mixed-density successive difference",
        "Integrated source work vs endpoint energy",
        "Cross-metric link residual",
        "Occupation-spectrum drift",
    )
    ylabels = (
        "relative Frobenius difference",
        "absolute residual (Ha)",
        "maximum residual",
        "maximum drift",
    )
    for axis, title, ylabel in zip(axes.flat, titles, ylabels, strict=True):
        axis.set_title(title)
        axis.set_xlabel(r"time step $\Delta t$ (a.u.)")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
        axis.invert_xaxis()
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.suptitle("NQ6 independent timestep refinement", fontsize=15)
    figure.tight_layout()
    figure.savefig(output / "timestep_convergence.png", dpi=180)
    plt.close(figure)


def _plot_tolerance_convergence(
    rows: list[dict[str, Any]], output: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.1))
    floor = 1.0e-16
    for case in _DYNAMIC_CASES:
        for branch in _BRANCHES:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["case"] == case and row["branch"] == branch
                ),
                key=lambda row: float(row["nonlinear_tolerance"]),
                reverse=True,
            )
            label = f"{_CASE_LABELS[case]}, {_BRANCH_LABELS[branch]}"
            color = _COLORS[case]
            linestyle = _line_style(branch)
            axes[0].loglog(
                [row["nonlinear_tolerance"] for row in selected],
                [
                    max(floor, float(row["final_density_difference_from_1e-12"]))
                    for row in selected
                ],
                marker="o",
                color=color,
                linestyle=linestyle,
                label=label,
            )
            axes[1].loglog(
                [row["nonlinear_tolerance"] for row in selected],
                [max(floor, float(row["nonlinear_residual_max"])) for row in selected],
                marker="o",
                color=color,
                linestyle=linestyle,
                label=label,
            )
    axes[0].set_title(r"Final state relative to $10^{-12}$ solve")
    axes[0].set_ylabel("relative Frobenius difference")
    axes[1].set_title("Attained nonlinear residual")
    axes[1].set_ylabel("maximum residual")
    for axis in axes:
        axis.set_xlabel("requested nonlinear tolerance")
        axis.grid(True, which="both", alpha=0.25)
        axis.invert_xaxis()
    axes[0].legend(fontsize=7, ncol=2)
    figure.suptitle(r"NQ6 nonlinear-tolerance refinement at $\Delta t=0.125$ a.u.")
    figure.tight_layout()
    figure.savefig(output / "nonlinear_tolerance_convergence.png", dpi=180)
    plt.close(figure)


def _cumulative_simpson_at_even_nodes(values: np.ndarray, step: float) -> np.ndarray:
    output = [0.0]
    for end in range(2, len(values), 2):
        output.append(_simpson(values[: end + 1], step))
    return np.asarray(output)


def _plot_energy_work(arrays: Any, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.1), sharex=True)
    for axis, branch in zip(axes, _BRANCHES, strict=True):
        run_id = _run_id(
            "magnetic_induction",
            branch,
            32,
            _REFERENCE_TOLERANCE,
            "timestep",
        )
        times = np.asarray(arrays[f"{run_id}_times_au"])
        energies = np.asarray(arrays[f"{run_id}_energies_au"])
        powers = np.asarray(arrays[f"{run_id}_source_powers_au"])
        matrix_rates = np.asarray(arrays[f"{run_id}_matrix_rates_au"])
        work = _cumulative_simpson_at_even_nodes(powers, times[1] - times[0])
        axis.plot(times, energies - energies[0], label=r"$E_{\rm mech}(t)-E_{\rm mech}(0)$")
        axis.plot(
            times[::2],
            work,
            marker="o",
            markersize=3,
            linestyle="--",
            label=r"$\int_0^t P_{\rm source}\,dt$",
        )
        inset = axis.inset_axes([0.56, 0.13, 0.40, 0.34])
        inset.semilogy(
            times,
            np.maximum(np.abs(matrix_rates - powers), 1.0e-18),
            color="#9467bd",
        )
        inset.set_title(r"$|\dot E_{\rm matrix}-P|$", fontsize=8)
        inset.tick_params(labelsize=7)
        inset.grid(True, which="both", alpha=0.2)
        axis.set_title(f"Time-dependent B: {_BRANCH_LABELS[branch]}")
        axis.set_xlabel("time (a.u.)")
        axis.set_ylabel("energy/work (Ha)")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("NQ6 mechanical energy and action-derived source work")
    figure.tight_layout()
    figure.savefig(output / "induction_energy_work.png", dpi=180)
    plt.close(figure)


def _plot_finest_identities(rows: list[dict[str, Any]], output: Path) -> None:
    measures = (
        ("metric_orthonormality_residual_max", "metric orthonormality"),
        ("mixed_density_polynomial_residual_max", "occupation polynomial"),
        ("particle_number_metric_residual_max", "metric particle number"),
        ("ward_residual_max", "Ward"),
        ("finite_region_continuity_residual_max", "weak continuity"),
        ("global_charge_residual_max", "global charge"),
        ("instantaneous_power_residual_au_max", "power identity"),
    )
    matrix = np.asarray(
        [
            [
                max(float(row[field]), 1.0e-20)
                if row.get(field) is not None
                else np.nan
                for field, _label in measures
            ]
            for row in rows
        ]
    )
    log_matrix = np.log10(matrix)
    figure, axis = plt.subplots(figsize=(12.8, 5.5))
    image = axis.imshow(log_matrix, aspect="auto", cmap="viridis", vmin=-20, vmax=-8)
    axis.set_xticks(
        range(len(measures)),
        [label for _field, label in measures],
        rotation=28,
        ha="right",
    )
    axis.set_yticks(
        range(len(rows)),
        [
            f"{_CASE_LABELS[str(row['case'])]} / {_BRANCH_LABELS[str(row['branch'])]}"
            for row in rows
        ],
    )
    for row_index in range(log_matrix.shape[0]):
        for column_index in range(log_matrix.shape[1]):
            value = log_matrix[row_index, column_index]
            if np.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value < -14 else "black",
                )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(r"$\log_{10}$ absolute residual")
    axis.set_title(r"NQ6 finest trajectories ($\Delta t=0.0625$ a.u.)")
    figure.tight_layout()
    figure.savefig(output / "finest_trajectory_identities.png", dpi=180)
    plt.close(figure)


def _interrupted_record(root: Path | None) -> dict[str, Any] | None:
    if root is None:
        return None
    record_path = root / "interrupted.json"
    record = _load_json(record_path)
    log_path = Path(record["log_path"])
    if record["log_sha256"] != _sha256(log_path):
        raise RuntimeError("interrupted-run log hash changed")
    return {
        "root": str(root),
        "record_sha256": _sha256(record_path),
        "log_path": str(log_path),
        "log_sha256": _sha256(log_path),
        "status": record["status"],
        "completed_trajectory_count_from_log": record[
            "completed_trajectory_count_from_log"
        ],
        "planned_trajectory_count": record["planned_trajectory_count"],
        "used_as_qualification_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interrupted-root", type=Path)
    arguments = parser.parse_args()
    raw = arguments.raw.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis directory: {output}")
    output.mkdir(parents=True)

    result = _load_json(raw / "result.json")
    if result.get("schema") != "aion.chapter13-nq6-dynamics":
        raise RuntimeError("raw result has the wrong schema")
    if result.get("profile") != "qualification":
        raise RuntimeError("NQ6 review analysis requires the qualification profile")
    hashes = _authenticate(raw, result)
    summary_rows = _read_csv(raw / "summary.csv")
    trajectory_rows = _read_csv(raw / "trajectory.csv")
    with np.load(raw / "trajectories.npz", allow_pickle=False) as arrays:
        matrix_validation = _validate_matrix(
            raw, result, summary_rows, trajectory_rows, arrays
        )
        timestep_rows = _timestep_analysis(summary_rows, arrays)
        tolerance_rows = _tolerance_analysis(summary_rows, arrays)
        finest_rows = _finest_diagnostics(trajectory_rows)
        analysis = _analyze(
            summary_rows,
            timestep_rows,
            tolerance_rows,
            finest_rows,
        )
        _plot_timestep_convergence(timestep_rows, output)
        _plot_tolerance_convergence(tolerance_rows, output)
        _plot_energy_work(arrays, output)
        _plot_finest_identities(finest_rows, output)

    _write_csv(output / "timestep_convergence.csv", timestep_rows)
    _write_csv(output / "nonlinear_tolerance_convergence.csv", tolerance_rows)
    _write_csv(output / "finest_trajectory_diagnostics.csv", finest_rows)
    analysis["raw_root"] = str(raw)
    analysis["raw_hashes"] = hashes
    analysis["matrix_validation"] = matrix_validation
    analysis["interrupted_attempt"] = _interrupted_record(
        None
        if arguments.interrupted_root is None
        else arguments.interrupted_root.expanduser().resolve()
    )
    figure_names = (
        "timestep_convergence.png",
        "nonlinear_tolerance_convergence.png",
        "induction_energy_work.png",
        "finest_trajectory_identities.png",
    )
    table_names = (
        "timestep_convergence.csv",
        "nonlinear_tolerance_convergence.csv",
        "finest_trajectory_diagnostics.csv",
    )
    analysis["figures_sha256"] = {
        name: _sha256(output / name) for name in figure_names
    }
    analysis["tables_sha256"] = {
        name: _sha256(output / name) for name in table_names
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "passed_proposed_thresholds": analysis[
                    "passed_proposed_thresholds"
                ],
                "summary_sha256": _sha256(summary_path),
            },
            indent=2,
        )
    )
    if not analysis["passed_proposed_thresholds"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
