#!/usr/bin/env python3
"""Authenticate and visualize the WP6 mixed-density Magnus experiment."""

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
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from aion.config import canonical_sha256

_DEFAULT_EXECUTION = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "exact_one_electron_qualification/"
    "wp6_mixed_magnus_20260918T231057Z_38ce83eb2fe9"
)

_CASE_ORDER = (
    "h3_equilateral_level4_sub_threshold",
    "h3_equilateral_level4_near_threshold",
    "h3_equilateral_level4_above_threshold",
    "h3_distorted_level4_sub_threshold",
    "h3_distorted_level4_near_threshold",
    "h3_distorted_level4_above_threshold",
)
_CASE_LABELS = ("Eq sub", "Eq near", "Eq above", "Dist sub", "Dist near", "Dist above")
_MODEL_ORDER = ("p0", "e1", "geometric_b1", "full_b1", "complete_first_order")
_ORDER_MODELS = ("exact", *_MODEL_ORDER)
_MODEL_LABELS = {
    "exact": "EX",
    "p0": "P0",
    "e1": "E1",
    "geometric_b1": "gB1",
    "full_b1": "B1",
    "complete_first_order": "C1",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution", type=Path, nargs="?", default=_DEFAULT_EXECUTION)
    parser.add_argument("--output", type=Path)
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _authenticate(execution: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _json(execution / "execution_plan.json")
    result = _json(execution / "result.json")
    completed = _json(execution / "completed.json")
    for filename, key in (
        ("execution_plan.json", "execution_plan_sha256"),
        ("result.json", "result_sha256"),
        ("arrays.npz", "arrays_sha256"),
    ):
        if _sha256(execution / filename) != completed[key]:
            raise ValueError(f"source hash mismatch: {filename}")
    if not plan["plan_id"] == result["plan_id"] == completed["plan_id"]:
        raise ValueError("experiment plan identity was not propagated")
    if plan["provenance"]["code"]["dirty"]:
        raise ValueError("experiment did not originate from a clean commit")
    with np.load(execution / "arrays.npz") as arrays:
        semantic = result["array_semantic_sha256"]
        if set(arrays.files) != set(semantic):
            raise ValueError("array inventory differs from its semantic-hash inventory")
        for name in arrays.files:
            if canonical_sha256(arrays[name]) != semantic[name]:
                raise ValueError(f"array semantic hash mismatch: {name}")
    return plan, result


def _source_midpoint_result(plan: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = Path(str(plan["fixture"]["source_midpoint_supplement"]))
    completed = _json(root / "completed.json")
    if _sha256(root / "result.json") != completed["result_sha256"]:
        raise ValueError("source midpoint result hash mismatch")
    return root, _json(root / "result.json")


def _stability_table(
    mixed: dict[str, Any],
    midpoint: dict[str, Any],
) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[str, str]:
        return str(row["prefix"]), str(row["model"])

    new = {
        key(row): row
        for row in mixed["model_difference_stability"]
        if row["coarse_intervals"] == 16 and row["fine_intervals"] == 32
    }
    old = {
        key(row): row
        for row in midpoint["model_difference_stability"]
        if row["coarse_intervals"] == 16 and row["fine_intervals"] == 32
    }
    if set(new) != set(old) or len(new) != 30:
        raise ValueError("midpoint and Magnus stability inventories differ")
    output = []
    for prefix in _CASE_ORDER:
        for model in _MODEL_ORDER:
            current = new[(prefix, model)]
            previous = old[(prefix, model)]
            old_fraction = float(previous["refinement_fraction_of_fine_model_difference"])
            new_fraction = float(current["refinement_fraction_of_fine_model_difference"])
            output.append(
                {
                    "prefix": prefix,
                    "geometry": current["geometry"],
                    "case": current["case"],
                    "model": model,
                    "midpoint_cholesky_fraction": old_fraction,
                    "mixed_magnus_fraction": new_fraction,
                    "improvement_factor": old_fraction / new_fraction,
                    "mixed_magnus_passes_0p25": new_fraction <= 0.25,
                    "fine_model_difference": current["fine_model_difference"],
                    "mixed_magnus_refinement_change": current["model_difference_refinement_change"],
                }
            )
    return output


def _convergence_table(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in result["timestep_convergence"]
        if row["coarse_intervals"] == 8 and row["fine_intervals"] == 16
    ]
    if len(rows) != 36:
        raise ValueError("unexpected fourth-order convergence inventory")
    for row in rows:
        row["passes_order_3p5"] = float(row["measured_order_to_next"]) >= 3.5
    return sorted(
        rows,
        key=lambda row: (
            _CASE_ORDER.index(str(row["prefix"])),
            _ORDER_MODELS.index(str(row["model"])),
        ),
    )


def _diagnostic_envelopes(result: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for intervals in (8, 16, 32):
        all_rows = [row for row in result["trajectory_rows"] if row["intervals"] == intervals]
        exact_rows = [row for row in all_rows if row["model"] == "exact"]
        output.append(
            {
                "intervals": intervals,
                "step_au": 4.0 / intervals,
                "all_max_cross_metric_residual": max(
                    float(row["maximum_cross_metric_residual"]) for row in all_rows
                ),
                "exact_max_cross_metric_residual": max(
                    float(row["maximum_cross_metric_residual"]) for row in exact_rows
                ),
                "all_max_metric_hermiticity_residual": max(
                    float(row["maximum_metric_hermiticity_residual"]) for row in all_rows
                ),
                "exact_max_metric_hermiticity_residual": max(
                    float(row["maximum_metric_hermiticity_residual"]) for row in exact_rows
                ),
                "all_max_contravariant_hermiticity_residual": max(
                    float(row["maximum_contravariant_hermiticity_residual"]) for row in all_rows
                ),
                "exact_max_contravariant_hermiticity_residual": max(
                    float(row["maximum_contravariant_hermiticity_residual"]) for row in exact_rows
                ),
            }
        )
    return output


def _matrix(
    rows: list[dict[str, Any]],
    models: tuple[str, ...],
    value: str,
) -> np.ndarray:
    lookup = {(str(row["prefix"]), str(row["model"])): float(row[value]) for row in rows}
    return np.asarray([[lookup[(case, model)] for case in _CASE_ORDER] for model in models])


def _annotated_heatmap(
    axis: Any,
    values: np.ndarray,
    *,
    title: str,
    model_order: tuple[str, ...],
    norm: Any,
    cmap: str,
    failure: np.ndarray,
    format_value: Any,
) -> Any:
    image = axis.imshow(values, aspect="auto", norm=norm, cmap=cmap)
    color_map = matplotlib.colormaps[cmap]
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xticks(range(len(_CASE_LABELS)), _CASE_LABELS, rotation=35, ha="right")
    axis.set_yticks(range(len(model_order)), [_MODEL_LABELS[name] for name in model_order])
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            red, green, blue, _alpha = color_map(norm(values[row, column]))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "#1f2933" if luminance > 0.58 else "white"
            axis.text(
                column,
                row,
                format_value(values[row, column]),
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
            if failure[row, column]:
                axis.add_patch(
                    Rectangle(
                        (column - 0.48, row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#b42318",
                        linewidth=2.2,
                    )
                )
    return image


def _stability_figure(rows: list[dict[str, Any]], output: Path) -> None:
    old = np.asarray([float(row["midpoint_cholesky_fraction"]) for row in rows])
    new = np.asarray([float(row["mixed_magnus_fraction"]) for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.3), constrained_layout=True)
    geometry_colors = {"equilateral": "#3569a8", "distorted": "#d97724"}
    markers = {
        "p0": "o",
        "e1": "s",
        "geometric_b1": "^",
        "full_b1": "D",
        "complete_first_order": "P",
    }
    for row in rows:
        axes[0].scatter(
            row["midpoint_cholesky_fraction"],
            row["mixed_magnus_fraction"],
            color=geometry_colors[str(row["geometry"])],
            marker=markers[str(row["model"])],
            s=56,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
    lower = min(old.min(), new.min()) / 1.8
    upper = max(old.max(), new.max()) * 1.8
    axes[0].plot((lower, upper), (lower, upper), color="#4b5563", linestyle="--", linewidth=1.2)
    axes[0].axhline(0.25, color="#b42318", linestyle=":", linewidth=1.5)
    axes[0].axvline(0.25, color="#b42318", linestyle=":", linewidth=1.5)
    axes[0].set(xscale="log", yscale="log", xlim=(lower, upper), ylim=(lower, upper))
    axes[0].set_xlabel("midpoint + Cholesky fraction")
    axes[0].set_ylabel("mixed-density Magnus fraction")
    axes[0].set_title("A  Paired 16→32-step stability", loc="left", fontweight="bold")
    axes[0].grid(True, which="both", color="#d7dde5", linewidth=0.6, alpha=0.8)
    geometry_legend = axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=geometry_colors["equilateral"],
                label="equilateral",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                color=geometry_colors["distorted"],
                label="distorted",
            ),
            Line2D([0], [0], linestyle=":", color="#b42318", label="criterion = 0.25"),
        ],
        frameon=False,
        fontsize=9,
        loc="upper left",
    )
    axes[0].add_artist(geometry_legend)
    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=markers[model],
                linestyle="",
                color="#4b5563",
                label=_MODEL_LABELS[model],
            )
            for model in _MODEL_ORDER
        ],
        frameon=False,
        fontsize=8,
        loc="lower right",
        ncols=2,
    )

    matrix = _matrix(rows, _MODEL_ORDER, "mixed_magnus_fraction")
    image = _annotated_heatmap(
        axes[1],
        matrix,
        title="B  Mixed-density Magnus stability",
        model_order=_MODEL_ORDER,
        norm=LogNorm(vmin=4.0e-4, vmax=5.0e-1),
        cmap="viridis",
        failure=matrix > 0.25,
        format_value=lambda value: f"{value:.3g}",
    )
    colorbar = figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.03)
    colorbar.set_label("refinement change / fine model difference")
    figure.suptitle(
        "WP6 model-difference timestep stability",
        fontsize=15,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"model_difference_stability.{suffix}", dpi=220)
    plt.close(figure)


def _order_figure(rows: list[dict[str, Any]], output: Path) -> None:
    values = _matrix(rows, _ORDER_MODELS, "measured_order_to_next")
    figure, axis = plt.subplots(figsize=(9.3, 5.8), constrained_layout=True)
    image = _annotated_heatmap(
        axis,
        values,
        title="Observed global order from (8→16)/(16→32) density changes",
        model_order=_ORDER_MODELS,
        norm=Normalize(vmin=1.5, vmax=4.1),
        cmap="cividis",
        failure=values < 3.5,
        format_value=lambda value: f"{value:.2f}",
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("measured convergence order")
    figure.suptitle(
        "WP6 mixed-density Gauss-Magnus temporal convergence", fontsize=15, fontweight="bold"
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"temporal_convergence_order.{suffix}", dpi=220)
    plt.close(figure)


def _diagnostic_figure(rows: list[dict[str, Any]], output: Path) -> None:
    intervals = np.asarray([int(row["intervals"]) for row in rows])
    panels = (
        ("cross_metric_residual", "raw cross-metric defect", 5),
        ("metric_hermiticity_residual", r"mixed $D^\dagger S-SD$ defect", 4),
        ("contravariant_hermiticity_residual", r"recovered $P-P^\dagger$ defect", 4),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.7), constrained_layout=True)
    for axis, (key, title, order) in zip(axes, panels, strict=True):
        all_values = np.asarray([float(row[f"all_max_{key}"]) for row in rows])
        exact_values = np.asarray([float(row[f"exact_max_{key}"]) for row in rows])
        reference = exact_values[0] * (intervals[0] / intervals) ** order
        axis.semilogy(intervals, all_values, "o-", color="#d97724", label="all-model envelope")
        axis.semilogy(intervals, exact_values, "s-", color="#3569a8", label="EX envelope")
        axis.semilogy(
            intervals, reference, "--", color="#4b5563", label=rf"$N^{{-{order}}}$ reference"
        )
        axis.set_xticks(intervals, tuple(str(value) for value in intervals))
        axis.set_xlim(6.5, 33.5)
        axis.set_xlabel("time intervals over 4 a.u.")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(True, which="both", color="#d7dde5", linewidth=0.6)
    axes[0].set_ylabel("maximum residual")
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Uncorrected geometric diagnostics", fontsize=15, fontweight="bold")
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"geometric_residual_convergence.{suffix}", dpi=220)
    plt.close(figure)


def main() -> None:
    args = _arguments()
    execution = args.execution.resolve()
    output = (args.output or execution / "analysis").resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    plan, mixed = _authenticate(execution)
    midpoint_root, midpoint = _source_midpoint_result(plan)
    stability = _stability_table(mixed, midpoint)
    convergence = _convergence_table(mixed)
    diagnostics = _diagnostic_envelopes(mixed)
    _write_csv(tables / "model_difference_stability_comparison.csv", stability)
    _write_csv(tables / "temporal_convergence.csv", convergence)
    _write_csv(tables / "diagnostic_envelopes.csv", diagnostics)
    _stability_figure(stability, figures)
    _order_figure(convergence, figures)
    _diagnostic_figure(diagnostics, figures)

    fractions = np.asarray([float(row["mixed_magnus_fraction"]) for row in stability])
    improvements = np.asarray([float(row["improvement_factor"]) for row in stability])
    orders = np.asarray([float(row["measured_order_to_next"]) for row in convergence])
    trajectory = mixed["trajectory_rows"]
    criteria = plan["fixture"]["experimental_criteria"]
    checks = {
        "authenticated_source": True,
        "stable_model_differences": bool(
            np.all(fractions <= float(criteria["fine_model_difference_refinement_fraction"]))
        ),
        "expected_global_order": bool(
            np.all(orders >= float(criteria["minimum_time_convergence_order"]))
        ),
        "trace_preserved": max(float(row["maximum_trace_drift"]) for row in trajectory)
        <= float(criteria["maximum_trace_drift"]),
        "idempotency_preserved": max(
            float(row["maximum_idempotency_residual"]) for row in trajectory
        )
        <= float(criteria["maximum_idempotency_residual"]),
        "metrics_positive": min(
            float(row["minimum_endpoint_metric_eigenvalue"]) for row in trajectory
        )
        >= float(criteria["minimum_metric_eigenvalue"]),
    }
    summary: dict[str, Any] = {
        "schema": "aion.exact-one-electron-wp6-mixed-magnus-analysis",
        "version": "1.0.0",
        "status": "derived_executed_unreviewed",
        "source_execution": str(execution),
        "source_midpoint_execution": str(midpoint_root),
        "source_hashes": {
            "mixed_completed": _sha256(execution / "completed.json"),
            "midpoint_completed": _sha256(midpoint_root / "completed.json"),
        },
        "checks": checks,
        "measurements": {
            "wall_time_seconds": mixed["wall_time_seconds"],
            "stable_comparisons": int(np.sum(fractions <= 0.25)),
            "comparison_count": int(fractions.size),
            "maximum_mixed_magnus_stability_fraction": float(fractions.max()),
            "median_mixed_magnus_stability_fraction": float(np.median(fractions)),
            "minimum_improvement_factor": float(improvements.min()),
            "median_improvement_factor": float(np.median(improvements)),
            "maximum_improvement_factor": float(improvements.max()),
            "order_pass_count": int(np.sum(orders >= 3.5)),
            "order_count": int(orders.size),
            "minimum_measured_order": float(orders.min()),
            "median_measured_order": float(np.median(orders)),
            "maximum_trace_drift": max(float(row["maximum_trace_drift"]) for row in trajectory),
            "maximum_idempotency_residual": max(
                float(row["maximum_idempotency_residual"]) for row in trajectory
            ),
            "minimum_metric_eigenvalue": min(
                float(row["minimum_endpoint_metric_eigenvalue"]) for row in trajectory
            ),
        },
        "conclusion": (
            "The mixed-density Magnus method materially improves every tested "
            "model-difference stability fraction, but the predeclared experiment "
            "does not pass because distorted above-threshold P0/E1 remain above "
            "0.25 and nine trajectories remain below measured order 3.5."
        ),
    }
    _write_json(output / "summary.json", summary)
    print(f"analysis_directory={output}")
    print(f"stable_comparisons={summary['measurements']['stable_comparisons']}/30")
    print(
        "maximum_stability_fraction="
        f"{summary['measurements']['maximum_mixed_magnus_stability_fraction']:.6e}"
    )
    print(f"minimum_measured_order={summary['measurements']['minimum_measured_order']:.6f}")
    print(f"checks={json.dumps(checks, sort_keys=True)}")


if __name__ == "__main__":
    main()
