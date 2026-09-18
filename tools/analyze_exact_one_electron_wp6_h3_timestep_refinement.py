#!/usr/bin/env python3
"""Build the authenticated combined G6 analysis from the H3 supplement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
    core = dict(plan)
    plan_id = str(core.pop("plan_id"))
    if canonical_sha256(core) != plan_id:
        raise ValueError("supplement plan identity mismatch")
    result = _json(result_path)
    completed = _json(completed_path)
    if result["plan_id"] != plan_id or completed["plan_id"] != plan_id:
        raise ValueError("supplement identity was not propagated")
    if result["status"] != "executed_unreviewed":
        raise ValueError("supplement execution is incomplete")
    for path, field in (
        (plan_path, "execution_plan_sha256"),
        (result_path, "result_sha256"),
        (arrays_path, "arrays_sha256"),
    ):
        if _sha256(path) != completed[field]:
            raise ValueError(f"supplement source changed before analysis: {path.name}")
    primary = Path(str(plan["fixture"]["source_primary_execution"]))
    primary_summary = _json(primary / "analysis/summary.json")
    if not primary_summary["checks"]["declared_scalar_criteria_pass"]:
        raise ValueError("primary WP6 scalar evidence does not pass")
    if primary_summary["checks"]["g6_gate_ready_for_review"]:
        raise ValueError("primary WP6 analysis did not retain its evidence gap")
    return plan, result, primary_summary


def _label_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**row, "model_label": _MODEL_LABEL[str(row["model"])]}
        for row in rows
    ]


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


def _convergence_figure(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=True)
    for axis, geometry in zip(axes, ("equilateral", "distorted"), strict=True):
        subset = [row for row in rows if row["geometry"] == geometry]
        for model in _MODEL_ORDER:
            values = [row for row in subset if row["model"] == model]
            by_step: dict[float, list[float]] = {}
            for row in values:
                step = 4.0 / float(row["coarse_intervals"])
                by_step.setdefault(step, []).append(
                    float(row["final_coefficient_difference"])
                )
            steps = sorted(by_step, reverse=True)
            maxima = [max(by_step[step]) for step in steps]
            axis.loglog(
                steps,
                maxima,
                color=_MODEL_COLOR[model],
                marker=_MODEL_MARKER[model],
                linewidth=1.4,
                label=_MODEL_LABEL[model],
            )
        axis.set_title(geometry)
        axis.set_xlabel("coarse time step (a.u.)")
        _style(axis)
    axes[0].set_ylabel("maximum final coefficient difference across fields")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )
    figure.suptitle(
        "Three-centre pairwise timestep convergence (8→16 and 16→32)",
        fontsize=15,
    )
    return _save(figure, output / "h3_timestep_convergence")


def _focused_resolution_rows(
    rows: list[dict[str, Any]],
    limit: float,
) -> list[dict[str, Any]]:
    """Return all models for the executed case with the worst fine-grid ratio."""
    finest = [row for row in rows if int(row["fine_intervals"]) == 32]
    worst = max(
        finest,
        key=lambda row: float(row["refinement_fraction_of_fine_model_difference"]),
    )
    geometry = str(worst["geometry"])
    case = str(worst["case"])
    focused: list[dict[str, Any]] = []
    for model in _MODEL_ORDER[1:]:
        coarse_row = next(
            row
            for row in rows
            if row["geometry"] == geometry
            and row["case"] == case
            and row["model"] == model
            and int(row["coarse_intervals"]) == 8
            and int(row["fine_intervals"]) == 16
        )
        fine_row = next(
            row
            for row in finest
            if row["geometry"] == geometry
            and row["case"] == case
            and row["model"] == model
        )
        distance_16 = float(coarse_row["fine_model_difference"])
        if not np.isclose(
            distance_16,
            float(fine_row["coarse_model_difference"]),
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError("inconsistent 16-step model distance in stability rows")
        ratio = float(fine_row["refinement_fraction_of_fine_model_difference"])
        focused.append(
            {
                "geometry": geometry,
                "case": case,
                "peak_field_z_au": float(fine_row["peak_field_z_au"]),
                "model": model,
                "model_label": _MODEL_LABEL[model],
                "model_difference_n8": float(
                    coarse_row["coarse_model_difference"]
                ),
                "model_difference_n16": distance_16,
                "model_difference_n32": float(fine_row["fine_model_difference"]),
                "refinement_change_n16_to_n32": float(
                    fine_row["model_difference_refinement_change"]
                ),
                "refinement_fraction_of_n32_difference": ratio,
                "resolution_limit": limit,
                "resolved_at_n32": ratio <= limit,
            }
        )
    return focused


def _resolution_figure(
    focused: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    limit: float,
    output: Path,
) -> list[Path]:
    """Show directly whether model-minus-EX signals exceed timestep changes."""
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.0, 5.2),
        gridspec_kw={"width_ratios": (1.45, 1.0, 0.9)},
    )
    intervals = np.asarray((8, 16, 32), dtype=np.int64)
    for row in focused:
        model = str(row["model"])
        axes[0].semilogy(
            intervals,
            [
                row["model_difference_n8"],
                row["model_difference_n16"],
                row["model_difference_n32"],
            ],
            color=_MODEL_COLOR[model],
            marker=_MODEL_MARKER[model],
            linewidth=1.8,
            markersize=6,
            label=str(row["model_label"]),
        )
    axes[0].set_xticks(intervals)
    axes[0].set_xlabel("time intervals across the propagation window")
    axes[0].set_ylabel(r"model-EX density distance $D_n=\|\Delta P_n\|_F$")
    axes[0].set_title("(a) The apparent signal changes with the grid")
    axes[0].legend(frameon=False, ncol=2, loc="lower left")
    _style(axes[0])

    positions = np.arange(len(focused), dtype=np.float64)
    ratios = np.asarray(
        [float(row["refinement_fraction_of_n32_difference"]) for row in focused]
    )
    lower = min(float(np.min(ratios)) * 0.55, limit * 0.55)
    upper = max(float(np.max(ratios)) * 1.75, limit * 2.0)
    axes[1].set_xscale("log")
    axes[1].set_xlim(lower, upper)
    axes[1].axvspan(lower, limit, color="#245b8a", alpha=0.10, zorder=0)
    axes[1].axvline(
        limit,
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        zorder=1,
    )
    for position, row, ratio in zip(positions, focused, ratios, strict=True):
        resolved = bool(row["resolved_at_n32"])
        axes[1].scatter(
            ratio,
            position,
            color="#245b8a" if resolved else "#d36b27",
            marker="o" if resolved else "X",
            s=70,
            zorder=3,
        )
        axes[1].annotate(
            f"${ratio:.2f}\\times$",
            (ratio, position),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    axes[1].set_yticks(positions, [str(row["model_label"]) for row in focused])
    axes[1].invert_yaxis()
    axes[1].set_xlabel(r"last refinement change $R_{16\to32}/D_{32}$")
    axes[1].set_title("(b) Is the 32-step signal resolved?")
    axes[1].text(
        0.03,
        0.03,
        "resolved region",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        color="#245b8a",
        fontsize=9,
    )
    _style(axes[1])

    finest = [
        row for row in stability_rows if int(row["fine_intervals"]) == 32
    ]
    failure_counts = {
        model: sum(
            float(row["refinement_fraction_of_fine_model_difference"]) > limit
            for row in finest
            if row["model"] == model
        )
        for model in _MODEL_ORDER[1:]
    }
    bars = axes[2].barh(
        positions,
        [failure_counts[str(row["model"])] for row in focused],
        color=[
            "#d36b27" if failure_counts[str(row["model"])] else "#245b8a"
            for row in focused
        ],
        alpha=0.88,
    )
    axes[2].set_yticks(positions, [str(row["model_label"]) for row in focused])
    axes[2].invert_yaxis()
    axes[2].set_xlim(0.0, 6.8)
    axes[2].set_xticks(range(7))
    axes[2].set_xlabel("unresolved executed conditions (out of 6)")
    axes[2].set_title("(c) The problem is not one isolated case")
    for bar, row in zip(bars, focused, strict=True):
        count = failure_counts[str(row["model"])]
        axes[2].text(
            count + 0.12,
            bar.get_y() + bar.get_height() / 2.0,
            f"{count}/6",
            va="center",
            fontsize=9,
        )
    _style(axes[2])

    first = focused[0]
    case_label = str(first["case"]).replace("_", " ")
    figure.suptitle(
        "Can the model-EX difference be distinguished from timestep error?",
        fontsize=15,
        y=1.02,
    )
    figure.text(
        0.5,
        0.965,
        (
            f"Worst executed case selected by the predeclared metric: "
            f"{first['geometry']} H3, {case_label}, "
            f"peak $B_z={float(first['peak_field_z_au']):.5f}$ a.u."
        ),
        ha="center",
        fontsize=10,
    )
    figure.text(
        0.5,
        -0.01,
        (
            r"Resolution criterion: $R_{16\to32}=\|\Delta P_{16}-\Delta "
            r"P_{32}\|_F \leq 0.25\,D_{32}$. "
            "Crosses mark failures; the count panel covers both geometries and "
            "all three field strengths."
        ),
        ha="center",
        fontsize=9,
    )
    return _save(figure, output / "h3_model_resolution_diagnostic")


def _stability_figure(
    rows: list[dict[str, Any]],
    limit: float,
    output: Path,
) -> list[Path]:
    finest = [row for row in rows if int(row["fine_intervals"]) == 32]
    case_order = {name: index for index, name in enumerate(
        ("sub_threshold", "near_threshold", "above_threshold")
    )}
    model_order = {name: index for index, name in enumerate(_MODEL_ORDER)}
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    for axis_index, (axis, geometry) in enumerate(
        zip(axes, ("equilateral", "distorted"), strict=True)
    ):
        subset = sorted(
            (row for row in finest if row["geometry"] == geometry),
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
                s=50,
                zorder=3,
            )
        axis.axhline(
            limit,
            color="#222222",
            linestyle="--",
            linewidth=1.2,
            label=f"predeclared limit: {limit:g}" if axis_index == 0 else None,
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
    axes[0].set_ylabel("16→32 change / 32-step model difference")
    figure.legend(loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.04))
    figure.suptitle("Three-centre model-difference timestep stability", fontsize=15)
    return _save(figure, output / "h3_model_timestep_stability")


def _model_figure(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    finest = [row for row in rows if int(row["fine_intervals"]) == 32]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    for axis, geometry in zip(axes, ("equilateral", "distorted"), strict=True):
        geometry_rows = [row for row in finest if row["geometry"] == geometry]
        for model in _MODEL_ORDER[1:]:
            subset = sorted(
                (row for row in geometry_rows if row["model"] == model),
                key=lambda row: float(row["peak_field_z_au"]),
            )
            axis.semilogy(
                [row["peak_field_z_au"] for row in subset],
                [row["fine_model_difference"] for row in subset],
                color=_MODEL_COLOR[model],
                marker=_MODEL_MARKER[model],
                linewidth=1.5,
                label=_MODEL_LABEL[model],
            )
        axis.set_title(geometry)
        axis.set_xlabel("peak $B_z$ (a.u.)")
        _style(axis)
    axes[0].set_ylabel("32-step final density distance from EX")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )
    figure.suptitle("Timestep-refined three-centre model comparison", fontsize=15)
    return _save(figure, output / "h3_model_comparisons_refined")


def _report(summary: dict[str, Any]) -> str:
    checks = summary["checks"]
    status = (
        "ready for explicit user review"
        if checks["g6_gate_ready_for_review"]
        else "not ready for user review"
    )
    return "\n".join(
        (
            "# Combined G6 qualification evidence",
            "",
            f"Status: **executed and {status}; not accepted**.",
            "",
            "This package combines the authenticated primary WP6 execution with ",
            "the separately predeclared 8/16/32 three-centre timestep-refinement ",
            "supplement. The supplement was required because the primary 4/8 ",
            "histories did not establish stability of model-minus-EX density ",
            "differences.",
            "",
            "## Gate checks",
            "",
            f"- Primary scalar criteria pass: `{checks['primary_scalar_criteria_pass']}`.",
            "- Minimum supplement convergence order: "
            f"`{checks['minimum_supplement_time_order']:.6f}`.",
            "- Largest finest-pair model-difference refinement fraction: "
            f"`{checks['maximum_fine_model_refinement_fraction']:.6e}`.",
            f"- Largest corrected norm drift: `{checks['maximum_supplement_norm_drift']:.6e}`.",
            "- Smallest metric eigenvalue: "
            f"`{checks['minimum_supplement_metric_eigenvalue']:.6e}`.",
            f"- Full G6 gate ready for review: `{checks['g6_gate_ready_for_review']}`.",
            "",
            "The stability observable is the Frobenius norm of the refinement ",
            "change in the full model-minus-EX density matrix, divided by the ",
            "fine-grid model-minus-EX density norm. It therefore detects changes ",
            "of direction as well as changes of scalar magnitude.",
            "",
            "Passing these checks does not accept G6. Acceptance requires an ",
            "explicit user review decision and a separate immutable review record.",
            "",
        )
    )


def main() -> None:
    execution = _arguments()
    plan, result, primary_summary = _authenticate_source(execution)
    tolerances = plan["fixture"]["acceptance_tolerances"]
    tables = execution / "analysis/tables"
    figures = execution / "analysis/figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    table_values = {
        "trajectory_diagnostics.csv": _label_rows(result["trajectory_rows"]),
        "timestep_convergence.csv": _label_rows(result["timestep_convergence"]),
        "model_difference_stability.csv": _label_rows(
            result["model_difference_stability"]
        ),
    }
    limit = float(tolerances["fine_model_difference_refinement_fraction"])
    focused_rows = _focused_resolution_rows(
        table_values["model_difference_stability.csv"], limit
    )
    table_values["focused_resolution_case.csv"] = focused_rows
    for name, rows in table_values.items():
        _write_csv(tables / name, rows)
    figure_paths = [
        *_convergence_figure(table_values["timestep_convergence.csv"], figures),
        *_stability_figure(
            table_values["model_difference_stability.csv"], limit, figures
        ),
        *_model_figure(table_values["model_difference_stability.csv"], figures),
        *_resolution_figure(
            focused_rows,
            table_values["model_difference_stability.csv"],
            limit,
            figures,
        ),
    ]

    trajectories = result["trajectory_rows"]
    convergence = result["timestep_convergence"]
    stability = result["model_difference_stability"]
    norm_maximum = max(
        float(row["maximum_corrected_norm_drift"]) for row in trajectories
    )
    metric_minimum = min(
        float(row["minimum_endpoint_metric_eigenvalue"]) for row in trajectories
    )
    compatibility_maximum = max(
        float(row["maximum_metric_compatibility_residual"])
        for row in trajectories
    )
    orders = [
        float(row["measured_order_to_next"])
        for row in convergence
        if row["measured_order_to_next"] is not None
    ]
    finest_fractions = [
        float(row["refinement_fraction_of_fine_model_difference"])
        for row in stability
        if int(row["fine_intervals"]) == 32
        and row["refinement_fraction_of_fine_model_difference"] is not None
    ]
    minimum_order = min(orders)
    maximum_fraction = max(finest_fractions)
    supplement_pass = bool(
        norm_maximum <= float(tolerances["corrected_norm_drift"])
        and metric_minimum > float(tolerances["minimum_metric_eigenvalue"])
        and compatibility_maximum
        <= float(tolerances["metric_compatibility_relative"])
        and minimum_order >= float(tolerances["minimum_time_convergence_order"])
        and maximum_fraction <= limit
    )
    primary_pass = bool(primary_summary["checks"]["declared_scalar_criteria_pass"])
    checks = {
        "primary_scalar_criteria_pass": primary_pass,
        "maximum_supplement_norm_drift": norm_maximum,
        "minimum_supplement_metric_eigenvalue": metric_minimum,
        "maximum_supplement_metric_compatibility_residual": compatibility_maximum,
        "minimum_supplement_time_order": minimum_order,
        "maximum_fine_model_refinement_fraction": maximum_fraction,
        "supplement_criteria_pass": supplement_pass,
        "metric_regularization_used": False,
        "g6_gate_ready_for_review": primary_pass and supplement_pass,
    }
    primary = Path(str(plan["fixture"]["source_primary_execution"]))
    manifest_core = {
        "schema": "aion.exact-one-electron-wp6-combined-analysis-manifest",
        "version": "1.0.0",
        "status": "derived_executed_unreviewed",
        "supplement_execution_directory": str(execution),
        "supplement_plan_id": result["plan_id"],
        "supplement_execution_plan_sha256": _sha256(execution / "execution_plan.json"),
        "supplement_result_sha256": _sha256(execution / "result.json"),
        "supplement_arrays_sha256": _sha256(execution / "arrays.npz"),
        "primary_analysis_summary_sha256": _sha256(primary / "analysis/summary.json"),
        "primary_analysis_manifest_sha256": _sha256(primary / "analysis/manifest.json"),
        "analyzer_sha256": _sha256(Path(__file__).resolve()),
    }
    manifest_id = canonical_sha256(manifest_core)
    analysis = execution / "analysis"
    manifest_path = analysis / "manifest.json"
    summary_path = analysis / "summary.json"
    report_path = analysis / "report.md"
    _write_json(manifest_path, {**manifest_core, "manifest_id": manifest_id})
    artifacts = {
        "tables": {name: _sha256(tables / name) for name in table_values},
        "figures": {path.name: _sha256(path) for path in figure_paths},
    }
    summary = {
        "schema": "aion.exact-one-electron-wp6-combined-analysis-summary",
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
        analysis / "completed.json",
        {
            "status": "authenticated_derived_executed_unreviewed",
            "manifest_id": manifest_id,
            "manifest_sha256": _sha256(manifest_path),
            "summary_sha256": _sha256(summary_path),
            "report_sha256": _sha256(report_path),
        },
    )
    print(f"analysis={analysis}")
    print(f"supplement_criteria_pass={supplement_pass}")
    print(f"g6_gate_ready_for_review={checks['g6_gate_ready_for_review']}")
    print(f"minimum_supplement_time_order={minimum_order:.6f}")
    print(f"maximum_fine_model_refinement_fraction={maximum_fraction:.6e}")


if __name__ == "__main__":
    main()
