#!/usr/bin/env python3
"""Build the authenticated WP6 linear-dynamics evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from aion.config import canonical_sha256

_MODEL_ORDER = (
    "exact",
    "p0",
    "e1",
    "geometric_b1",
    "full_b1",
    "complete_first_order",
)
_MODEL_LABEL = {
    "exact": "EX",
    "p0": "P0",
    "e1": "E1",
    "geometric_b1": "gB1",
    "full_b1": "B1",
    "complete_first_order": "C1",
}
_MODEL_COLOR = {
    "exact": "#245b8a",
    "p0": "#666666",
    "e1": "#b48b19",
    "geometric_b1": "#d36b27",
    "full_b1": "#6f7f34",
    "complete_first_order": "#9b4f78",
}
_MODEL_MARKER = {
    "exact": "o",
    "p0": "s",
    "e1": "^",
    "geometric_b1": "D",
    "full_b1": "v",
    "complete_first_order": "P",
}


def _arguments() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution_directory", type=Path)
    value: Path = parser.parse_args().execution_directory
    return value.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return dict(value)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty evidence table: {path.name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _authenticate_source(
    execution: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan_path = execution / "execution_plan.json"
    result_path = execution / "result.json"
    arrays_path = execution / "arrays.npz"
    completed_path = execution / "completed.json"
    plan = _json(plan_path)
    result = _json(result_path)
    completed = _json(completed_path)
    core = dict(plan)
    plan_id = str(core.pop("plan_id"))
    if canonical_sha256(core) != plan_id:
        raise ValueError("WP6 source plan identity mismatch")
    if result["plan_id"] != plan_id or completed["plan_id"] != plan_id:
        raise ValueError("WP6 source identity was not propagated")
    if result["status"] != "executed_unreviewed":
        raise ValueError("only a completed WP6 qualification run may be analyzed")
    for path, field in (
        (plan_path, "execution_plan_sha256"),
        (result_path, "result_sha256"),
        (arrays_path, "arrays_sha256"),
    ):
        if _sha256(path) != completed[field]:
            raise ValueError(f"WP6 source changed before analysis: {path.name}")
    return plan, result, completed


def _with_model_label(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**row, "model_label": _MODEL_LABEL[str(row["model"])]}
        for row in rows
    ]


def _primary_h3_model_timestep_stability(
    result: dict[str, Any],
    arrays_path: Path,
) -> list[dict[str, Any]]:
    rows = result["h3_trajectory_rows"]
    prefixes = sorted(
        {
            str(row["prefix"])
            for row in rows
            if int(row["grid_level"]) == 4
        }
    )
    metadata = {str(row["prefix"]): row for row in rows}
    output: list[dict[str, Any]] = []
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for prefix in prefixes:
            source = metadata[prefix]
            for model in _MODEL_ORDER[1:]:
                differences: dict[int, np.ndarray] = {}
                distances: dict[int, float] = {}
                for intervals in (4, 8):
                    exact = arrays[
                        f"{prefix}__exact__n{intervals}__coefficients"
                    ][-1]
                    value = arrays[
                        f"{prefix}__{model}__n{intervals}__coefficients"
                    ][-1]
                    exact_density = exact @ exact.conj().T
                    model_density = value @ value.conj().T
                    difference = np.asarray(model_density - exact_density)
                    differences[intervals] = difference
                    distances[intervals] = float(np.linalg.norm(difference))
                change = float(np.linalg.norm(differences[8] - differences[4]))
                output.append(
                    {
                        "prefix": prefix,
                        "geometry": source["geometry"],
                        "case": source["case"],
                        "peak_field_z_au": source["peak_field_z_au"],
                        "model": model,
                        "model_label": _MODEL_LABEL[model],
                        "coarse_intervals": 4,
                        "fine_intervals": 8,
                        "coarse_model_difference": distances[4],
                        "fine_model_difference": distances[8],
                        "model_difference_refinement_change": change,
                        "refinement_fraction_of_fine_model_difference": (
                            change / distances[8]
                            if distances[8] > np.finfo(np.float64).tiny
                            else None
                        ),
                    }
                )
    return output


def _tables(
    result: dict[str, Any],
    arrays_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    matrix = [
        {"check": key, "residual": value}
        for key, value in result["matrix_checks"].items()
    ]
    h2_rows = _with_model_label(result["h2_trajectory_rows"])
    h3_rows = _with_model_label(result["h3_trajectory_rows"])
    h2_fine = max(int(row["intervals"]) for row in h2_rows)
    h3_fine: dict[tuple[str, int, str], int] = {}
    for row in h3_rows:
        key = (str(row["geometry"]), int(row["grid_level"]), str(row["case"]))
        h3_fine[key] = max(h3_fine.get(key, 0), int(row["intervals"]))
    h3_final = [
        row
        for row in h3_rows
        if int(row["intervals"])
        == h3_fine[(str(row["geometry"]), int(row["grid_level"]), str(row["case"]))]
    ]
    return {
        "matrix_checks.csv": matrix,
        "static_spectral_convergence.csv": list(
            result["static_spectral_convergence"]
        ),
        "h2_trajectory_diagnostics.csv": h2_rows,
        "h2_timestep_convergence.csv": _with_model_label(
            result["h2_timestep_convergence"]
        ),
        "h2_gauge_convergence.csv": list(result["h2_gauge_trajectory_rows"]),
        "h2_model_comparisons.csv": [
            row for row in h2_rows if int(row["intervals"]) == h2_fine
        ],
        "h3_trajectory_diagnostics.csv": h3_rows,
        "h3_timestep_convergence.csv": _with_model_label(
            result["h3_timestep_convergence"]
        ),
        "h3_model_comparisons.csv": h3_final,
        "h3_model_timestep_stability_primary.csv": (
            _primary_h3_model_timestep_stability(result, arrays_path)
        ),
        "h3_quadrature_stability.csv": _with_model_label(
            result["h3_quadrature_stability"]
        ),
    }


def _style(axis: Any) -> None:
    axis.grid(True, color="#d8dde3", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def _save(figure: Any, stem: Path) -> list[Path]:
    figure.tight_layout()
    outputs = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    figure.savefig(outputs[0], dpi=220, bbox_inches="tight")
    figure.savefig(outputs[1], bbox_inches="tight")
    plt.close(figure)
    return outputs


def _convergence_figure(
    static_rows: list[dict[str, Any]],
    h2_rows: list[dict[str, Any]],
    gauge_rows: list[dict[str, Any]],
    output: Path,
) -> list[Path]:
    figure, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
    axis = axes[0]
    static = sorted(static_rows, key=lambda row: float(row["step_au"]), reverse=True)
    axis.loglog(
        [row["step_au"] for row in static],
        [row["coefficient_error"] for row in static],
        color=_MODEL_COLOR["exact"],
        marker="o",
        linewidth=1.8,
        label="Padé [2/2]",
    )
    axis.set_title("Static spectral reference")
    axis.set_xlabel("time step (a.u.)")
    axis.set_ylabel("coefficient error")
    _style(axis)

    axis = axes[1]
    for model in _MODEL_ORDER:
        subset = sorted(
            (row for row in h2_rows if row["model"] == model),
            key=lambda row: float(row["coarse_intervals"]),
        )
        axis.loglog(
            [4.0 / float(row["coarse_intervals"]) for row in subset],
            [row["final_coefficient_difference"] for row in subset],
            color=_MODEL_COLOR[model],
            marker=_MODEL_MARKER[model],
            linewidth=1.4,
            label=_MODEL_LABEL[model],
        )
    axis.set_title("H₂ time-step refinement")
    axis.set_xlabel("coarse time step (a.u.)")
    axis.set_ylabel("final coefficient difference")
    _style(axis)

    axis = axes[2]
    gauge = sorted(gauge_rows, key=lambda row: float(row["step_au"]), reverse=True)
    axis.loglog(
        [row["step_au"] for row in gauge],
        [row["maximum_coefficient_metric_residual"] for row in gauge],
        color="#d36b27",
        marker="D",
        linewidth=1.8,
        label="symmetric ↔ Landau",
    )
    axis.set_title("Coefficient-equation gauge covariance")
    axis.set_xlabel("time step (a.u.)")
    axis.set_ylabel("maximum metric residual")
    _style(axis)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    figure.suptitle("WP6 temporal convergence", fontsize=15)
    return _save(figure, output / "temporal_convergence")


def _h3_model_figure(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    for axis, geometry in zip(axes, ("equilateral", "distorted"), strict=True):
        geometry_rows = [row for row in rows if row["geometry"] == geometry]
        level = min(int(row["grid_level"]) for row in geometry_rows)
        geometry_rows = [row for row in geometry_rows if int(row["grid_level"]) == level]
        for model in _MODEL_ORDER[1:]:
            subset = sorted(
                (row for row in geometry_rows if row["model"] == model),
                key=lambda row: float(row["peak_field_z_au"]),
            )
            axis.semilogy(
                [row["peak_field_z_au"] for row in subset],
                [row["final_density_distance_from_fine_exact"] for row in subset],
                color=_MODEL_COLOR[model],
                marker=_MODEL_MARKER[model],
                linewidth=1.5,
                label=_MODEL_LABEL[model],
            )
        axis.set_title(f"{geometry}; grid level {level}")
        axis.set_xlabel("peak $B_z$ (a.u.)")
        _style(axis)
    axes[0].set_ylabel("final density distance from EX")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    figure.suptitle("Three-centre model comparison at the finest time step", fontsize=15)
    return _save(figure, output / "h3_model_comparisons")


def _quadrature_figure(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    case_order = {name: index for index, name in enumerate(
        ("sub_threshold", "near_threshold", "above_threshold")
    )}
    model_order = {name: index for index, name in enumerate(_MODEL_ORDER)}
    usable = sorted(
        (
            row
            for row in rows
            if row["refinement_fraction_of_model_difference"] is not None
        ),
        key=lambda row: (
            case_order[str(row["case"])],
            model_order[str(row["model"])],
        ),
    )
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(usable), dtype=np.float64)
    values = np.asarray(
        [float(row["refinement_fraction_of_model_difference"]) for row in usable]
    )
    positive = values[values > 0.0]
    plotting_floor = float(np.min(positive) / 3.0)
    colors = [_MODEL_COLOR[str(row["model"])] for row in usable]
    markers = [_MODEL_MARKER[str(row["model"])] for row in usable]
    for position, value, color, marker in zip(x, values, colors, markers, strict=True):
        if value > 0.0:
            axis.scatter(position, value, color=color, marker=marker, s=52, zorder=3)
        else:
            axis.scatter(
                position,
                plotting_floor,
                facecolors="none",
                edgecolors=color,
                marker=marker,
                s=60,
                linewidths=1.4,
                zorder=3,
                label=(
                    "measured zero (shown at floor)"
                    if "measured zero (shown at floor)"
                    not in axis.get_legend_handles_labels()[1]
                    else None
                ),
            )
    axis.axhline(
        0.25,
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        label="declared 0.25 limit",
    )
    axis.set_yscale("log")
    axis.set_xticks(x)
    axis.set_xticklabels(
        [f"{str(row['case']).removesuffix('_threshold')}\n{row['model_label']}" for row in usable],
        rotation=50,
        ha="right",
    )
    axis.set_ylim(plotting_floor / 1.8, 0.5)
    axis.set_ylabel("grid-refinement change / model difference")
    axis.set_title("Equilateral three-centre quadrature stability")
    legend = axis.legend(frameon=True, loc="upper left")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.95)
    _style(axis)
    return _save(figure, output / "h3_quadrature_stability")


def _primary_timestep_stability_figure(
    rows: list[dict[str, Any]],
    output: Path,
) -> list[Path]:
    case_order = {name: index for index, name in enumerate(
        ("sub_threshold", "near_threshold", "above_threshold")
    )}
    model_order = {name: index for index, name in enumerate(_MODEL_ORDER)}
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    for axis_index, (axis, geometry) in enumerate(
        zip(axes, ("equilateral", "distorted"), strict=True)
    ):
        subset = sorted(
            (row for row in rows if row["geometry"] == geometry),
            key=lambda row: (
                case_order[str(row["case"])],
                model_order[str(row["model"])],
            ),
        )
        x = np.arange(len(subset), dtype=np.float64)
        for position, row in zip(x, subset, strict=True):
            model = str(row["model"])
            axis.scatter(
                position,
                float(row["refinement_fraction_of_fine_model_difference"]),
                color=_MODEL_COLOR[model],
                marker=_MODEL_MARKER[model],
                s=48,
                zorder=3,
            )
        axis.axhline(
            0.25,
            color="#222222",
            linestyle="--",
            linewidth=1.2,
            label=(
                "predeclared supplement limit: 0.25"
                if axis_index == 0
                else None
            ),
        )
        axis.set_yscale("log")
        axis.set_xticks(x)
        axis.set_xticklabels(
            [
                f"{str(row['case']).removesuffix('_threshold')}\n{row['model_label']}"
                for row in subset
            ],
            rotation=50,
            ha="right",
        )
        axis.set_title(geometry)
        _style(axis)
    axes[0].set_ylabel("4→8 change / 8-step model difference")
    figure.legend(loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.04))
    figure.suptitle(
        "Primary three-centre run does not establish timestep stability",
        fontsize=15,
    )
    return _save(figure, output / "h3_timestep_stability_primary")


def _report(summary: dict[str, Any]) -> str:
    checks = summary["checks"]
    return "\n".join(
        (
            "# WP6 exact time connection and linear propagation",
            "",
            "Status: **derived numerical evidence; executed and not yet reviewed**.",
            "",
            "## Scope",
            "",
            "This package authenticates and summarizes the fixed-centre, linear ",
            "one-electron WP6 campaign. It does not accept gate G6 and does not ",
            "claim nonlinear TDDFT closure, persistent workflow restart, or WP7 ",
            "action-derived observables.",
            "",
            "The propagated coefficient equation is ",
            "`i (S Cdot + omega_t C) = K C`, with the ordinary-derivative matrix ",
            "`H = K - i omega_t`. A midpoint Padé [2/2] step is followed by the ",
            "recorded right-Cholesky cross-metric constraint. No metric ",
            "regularization is used.",
            "",
            "## Numerical findings",
            "",
            "- Declared scalar numerical criteria pass: "
            f"`{checks['declared_scalar_criteria_pass']}`.",
            "- Full G6 gate ready for review: "
            f"`{checks['g6_gate_ready_for_review']}`.",
            "- Largest exact temporal identity residual: "
            f"`{checks['maximum_temporal_identity_residual']:.6e}`.",
            "- Largest corrected physical-norm drift: "
            f"`{checks['maximum_corrected_norm_drift']:.6e}`.",
            f"- Smallest endpoint metric eigenvalue: `{checks['minimum_metric_eigenvalue']:.6e}`.",
            f"- Minimum measured static Padé order: `{checks['minimum_static_pade_order']:.6f}`.",
            "- Minimum measured time-refinement order: "
            f"`{checks['minimum_time_convergence_order']:.6f}`.",
            "- Minimum measured gauge-residual order: "
            f"`{checks['minimum_gauge_convergence_order']:.6f}`.",
            "- Largest nontrivial quadrature/model-difference ratio: "
            f"`{checks['maximum_quadrature_fraction']:.6e}`.",
            f"- Restart boundary check is bitwise equal: `{checks['restart_bitwise_equal']}`.",
            "- Largest primary 4→8 model-difference refinement fraction: "
            f"`{checks['maximum_primary_h3_model_timestep_fraction']:.6e}`.",
            "",
            "The time-dependent magnetic source includes the Maxwell-required ",
            "affine induction field `E(r,t) = E_origin(t) + 1/2 (r-origin) x Bdot(t)`. ",
            "The symmetric- and Landau-representative coefficient trajectories ",
            "are compared after the analytic AO gauge transformation.",
            "",
            "The primary three-centre run used only 4 and 8 intervals. Its ",
            "same-step model-minus-EX density differences are not stable under ",
            "that refinement, so this package does not claim that the complete ",
            "G6 gate is ready for review. The separately predeclared 8/16/32 ",
            "supplement resolves this evidence gap.",
            "",
            "## Evidence inventory",
            "",
            "Tables contain raw matrix residuals, convergence rows, trajectory ",
            "diagnostics, model comparisons, and quadrature refinement. Figures ",
            "use common model encodings and log scales where residuals span orders ",
            "of magnitude. EX, P0, E1, gB1, B1, and C1 name action-consistent ",
            "model triples; model differences are not error bars.",
            "",
            "The exact matrix-history arrays remain in the authenticated source ",
            "execution. All derived table and figure hashes are recorded in ",
            "`summary.json`.",
            "",
            "Passing these numerical criteria does not itself accept gate G6. ",
            "Only an explicit user review decision may create that record.",
            "",
        )
    )


def main() -> None:
    execution = _arguments()
    plan, result, _completed = _authenticate_source(execution)
    tolerances = plan["fixture"]["acceptance_tolerances"]
    table_values = _tables(result, execution / "arrays.npz")
    output = execution / "analysis"
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for name, rows in table_values.items():
        _write_csv(tables / name, rows)

    figure_paths = [
        *_convergence_figure(
            table_values["static_spectral_convergence.csv"],
            table_values["h2_timestep_convergence.csv"],
            table_values["h2_gauge_convergence.csv"],
            figures,
        ),
        *_h3_model_figure(table_values["h3_model_comparisons.csv"], figures),
        *_primary_timestep_stability_figure(
            table_values["h3_model_timestep_stability_primary.csv"],
            figures,
        ),
        *_quadrature_figure(table_values["h3_quadrature_stability.csv"], figures),
    ]

    matrix_checks = result["matrix_checks"]
    trajectory_rows = [*result["h2_trajectory_rows"], *result["h3_trajectory_rows"]]
    temporal_identity = max(
        float(matrix_checks["direct_factorized_connection_residual"]),
        float(matrix_checks["direct_factorized_metric_dot_residual"]),
        float(matrix_checks["metric_compatibility_residual"]),
        float(matrix_checks["maximum_matrix_gauge_covariance_residual"]),
    )
    norm_drift = max(
        float(row["maximum_corrected_norm_drift"]) for row in trajectory_rows
    )
    metric_minimum = min(
        float(row["minimum_endpoint_metric_eigenvalue"]) for row in trajectory_rows
    )
    static_order = min(
        float(row["measured_order_to_next"])
        for row in result["static_spectral_convergence"]
        if row["measured_order_to_next"] is not None
    )
    time_order = min(
        float(row["measured_order_to_next"])
        for row in result["h2_timestep_convergence"]
        if row["measured_order_to_next"] is not None
    )
    gauge_order = min(
        float(row["measured_order_to_next"])
        for row in result["h2_gauge_trajectory_rows"]
        if row["measured_order_to_next"] is not None
    )
    quadrature_fractions = [
        float(row["refinement_fraction_of_model_difference"])
        for row in result["h3_quadrature_stability"]
        if row["refinement_fraction_of_model_difference"] is not None
    ]
    quadrature_maximum = max(quadrature_fractions)
    primary_h3_time_fractions = [
        float(row["refinement_fraction_of_fine_model_difference"])
        for row in table_values["h3_model_timestep_stability_primary.csv"]
        if row["refinement_fraction_of_fine_model_difference"] is not None
    ]
    primary_h3_time_maximum = max(primary_h3_time_fractions)
    restart = result["h2_restart_check"]
    scalar_criteria_pass = bool(
        temporal_identity <= float(tolerances["metric_compatibility_relative"])
        and float(matrix_checks["metric_rate_finite_difference_residual"])
        <= float(tolerances["metric_rate_finite_difference_relative"])
        and norm_drift <= float(tolerances["corrected_norm_drift"])
        and metric_minimum > float(tolerances["minimum_metric_eigenvalue"])
        and static_order >= float(tolerances["minimum_static_pade_order"])
        and time_order >= float(tolerances["minimum_time_convergence_order"])
        and gauge_order >= float(tolerances["minimum_time_convergence_order"])
        and quadrature_maximum
        <= float(tolerances["quadrature_stability_fraction"])
        and bool(restart["bitwise_equal"])
    )
    checks = {
        "maximum_temporal_identity_residual": temporal_identity,
        "metric_rate_finite_difference_residual": float(
            matrix_checks["metric_rate_finite_difference_residual"]
        ),
        "maximum_corrected_norm_drift": norm_drift,
        "minimum_metric_eigenvalue": metric_minimum,
        "minimum_static_pade_order": static_order,
        "minimum_time_convergence_order": time_order,
        "minimum_gauge_convergence_order": gauge_order,
        "maximum_quadrature_fraction": quadrature_maximum,
        "maximum_primary_h3_model_timestep_fraction": primary_h3_time_maximum,
        "restart_bitwise_equal": bool(restart["bitwise_equal"]),
        "metric_regularization_used": False,
        "declared_scalar_criteria_pass": scalar_criteria_pass,
        "h3_model_timestep_stability_status": (
            "not_demonstrated_by_primary_4_8_evidence"
        ),
        "g6_gate_ready_for_review": False,
    }
    manifest_core = {
        "schema": "aion.exact-one-electron-wp6-analysis-manifest",
        "version": "1.0.0",
        "status": "derived_executed_unreviewed",
        "source_execution_directory": str(execution),
        "source_plan_id": result["plan_id"],
        "source_execution_plan_sha256": _sha256(execution / "execution_plan.json"),
        "source_result_sha256": _sha256(execution / "result.json"),
        "source_arrays_sha256": _sha256(execution / "arrays.npz"),
        "analyzer_sha256": _sha256(Path(__file__).resolve()),
    }
    manifest_id = canonical_sha256(manifest_core)
    manifest_path = output / "manifest.json"
    summary_path = output / "summary.json"
    report_path = output / "report.md"
    _write_json(manifest_path, {**manifest_core, "manifest_id": manifest_id})
    artifacts = {
        "tables": {name: _sha256(tables / name) for name in table_values},
        "figures": {path.name: _sha256(path) for path in figure_paths},
    }
    summary = {
        "schema": "aion.exact-one-electron-wp6-analysis-summary",
        "version": "1.0.0",
        "status": "derived_executed_unreviewed",
        "manifest_id": manifest_id,
        "checks": checks,
        "row_counts": {
            Path(name).stem: len(rows) for name, rows in table_values.items()
        },
        "artifacts": artifacts,
        "review": {"reviewed": False, "reviewer": None, "decision": "pending"},
    }
    _write_json(summary_path, summary)
    report_path.write_text(_report(summary), encoding="utf-8")
    _write_json(
        output / "completed.json",
        {
            "status": "authenticated_derived_executed_unreviewed",
            "manifest_id": manifest_id,
            "manifest_sha256": _sha256(manifest_path),
            "summary_sha256": _sha256(summary_path),
            "report_sha256": _sha256(report_path),
        },
    )
    print(f"analysis={output}")
    print(f"declared_scalar_criteria_pass={scalar_criteria_pass}")
    print("g6_gate_ready_for_review=False")
    print(f"maximum_corrected_norm_drift={norm_drift:.6e}")
    print(f"minimum_metric_eigenvalue={metric_minimum:.6e}")
    print(f"minimum_time_convergence_order={time_order:.6f}")
    print(f"minimum_gauge_convergence_order={gauge_order:.6f}")
    print(f"maximum_quadrature_fraction={quadrature_maximum:.6e}")
    print(f"maximum_primary_h3_model_timestep_fraction={primary_h3_time_maximum:.6e}")


if __name__ == "__main__":
    main()
