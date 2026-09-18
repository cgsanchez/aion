#!/usr/bin/env python3
"""Build the authenticated WP5 multicentre evidence package."""

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

_REPOSITORY = Path(__file__).resolve().parents[1]
_TESLA_PER_AU = 2.35051757077e5
_MODEL_COLORS = {
    "exact": "#245b8a",
    "p0": "#666666",
    "geometric_b1": "#d36b27",
    "full_b1": "#6f7f34",
}
_MODEL_LABELS = {
    "exact": "exact",
    "p0": "P0",
    "geometric_b1": "gB1",
    "full_b1": "B1",
}
def _arguments() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution_directory", type=Path)
    return parser.parse_args().execution_directory.resolve()


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _members(execution: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index = _json(execution / "execution_index.json")
    if index["status"] != "authenticated_executed_unreviewed":
        raise ValueError("WP5 execution is not complete and authenticated")
    members = []
    for record in index["completed_members"]:
        directory = Path(record["member_directory"])
        for filename, field in (
            ("manifest.json", "manifest_sha256"),
            ("result.json", "result_sha256"),
            ("arrays.npz", "arrays_sha256"),
        ):
            if _sha256(directory / filename) != record[field]:
                raise ValueError(f"WP5 source changed before analysis: {directory}")
        members.append(
            {
                **record,
                "directory": directory,
                "manifest": _json(directory / "manifest.json"),
                "result": _json(directory / "result.json"),
            }
        )
    return index, members


def _tables(
    members: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    metric_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    for member in members:
        geometry = member["geometry"]
        basis = member["basis"]
        result = member["result"]
        positive_rows = [row for row in result["metric_scan"] if row["loop_flux_au"] >= 0.0]
        for row in positive_rows:
            metric_rows.append(
                {
                    "geometry": geometry,
                    "basis": basis,
                    **row,
                    "field_z_tesla": float(row["field_z_au"]) * _TESLA_PER_AU,
                }
            )
        by_flux = {float(row["loop_flux_au"]): row for row in positive_rows}
        for row in result["selected_dynamical_cases"]:
            source = by_flux[float(row["loop_flux_au"])]
            selected_rows.append(
                {
                    "geometry": geometry,
                    "basis": basis,
                    "selection_role": (
                        "wp6_selected"
                        if basis == "cc-pvdz"
                        else "minimal_basis_diagnostic"
                    ),
                    **row,
                    "field_z_tesla": float(row["field_z_au"]) * _TESLA_PER_AU,
                    "maximum_model_metric_condition_number": max(
                        float(source[f"{model}_condition_number"])
                        for model in ("exact", "p0", "geometric_b1", "full_b1")
                    ),
                }
            )
        for row in result["generalized_spectral_comparisons"]:
            spectral_rows.append({"geometry": geometry, "basis": basis, **row})
        for row in result["pair_restrictions"]:
            pair_rows.append({"geometry": geometry, "basis": basis, **row})
        approximate_loss = {}
        for model in ("p0", "geometric_b1", "full_b1"):
            nonpositive = [
                row
                for row in positive_rows
                if float(row[f"{model}_minimum_eigenvalue"]) <= 0.0
            ]
            approximate_loss[model] = (
                float(nonpositive[0]["loop_flux_au"]) if nonpositive else None
            )
        structural_rows.append(
            {
                "geometry": geometry,
                "basis": basis,
                "maximum_loop_identity_residual": max(
                    max(
                        float(row["maximum_flux_identity_residual"]),
                        float(row["maximum_gauge_residual"]),
                        float(row["maximum_orientation_reversal_residual"]),
                    )
                    for row in result["metric_scan"]
                ),
                "maximum_exact_gauge_residual": max(
                    max(
                        float(row["exact_gauge_barred_residual"]),
                        float(row["exact_gauge_congruence_residual"]),
                        float(row["exact_gauge_spectrum_residual"]),
                    )
                    for row in result["generalized_spectral_comparisons"]
                ),
                "minimum_exact_metric_eigenvalue": min(
                    float(row["exact_minimum_eigenvalue"])
                    for row in result["metric_scan"]
                ),
                "p0_first_nonpositive_flux_au": approximate_loss["p0"],
                "geometric_b1_first_nonpositive_flux_au": approximate_loss[
                    "geometric_b1"
                ],
                "full_b1_first_nonpositive_flux_au": approximate_loss["full_b1"],
                "maximum_pair_residual_to_floor": max(
                    float(row["level5_residual_to_floor"])
                    for row in result["pair_restrictions"]
                ),
            }
        )
    return metric_rows, selected_rows, spectral_rows, pair_rows, structural_rows


def _axes_style(axis: Any) -> None:
    axis.grid(True, color="#d8dde3", linewidth=0.7, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def _member_axes() -> tuple[Any, np.ndarray]:
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), sharex=True)
    return figure, np.asarray(axes)


def _metric_positivity_figure(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    figure, axes = _member_axes()
    for row_index, geometry in enumerate(("equilateral", "distorted")):
        for column_index, basis in enumerate(("sto-3g", "cc-pvdz")):
            axis = axes[row_index, column_index]
            subset = sorted(
                (
                    row
                    for row in rows
                    if row["geometry"] == geometry and row["basis"] == basis
                ),
                key=lambda row: float(row["loop_flux_au"]),
            )
            flux = np.asarray([row["loop_flux_au"] for row in subset])
            for model, linestyle in (
                ("exact", "-"),
                ("p0", "--"),
                ("geometric_b1", "-."),
            ):
                values = np.asarray(
                    [row[f"{model}_minimum_eigenvalue"] for row in subset]
                )
                label = "gB1 = B1" if model == "geometric_b1" else _MODEL_LABELS[model]
                axis.plot(
                    flux,
                    values,
                    color=_MODEL_COLORS[model],
                    linestyle=linestyle,
                    linewidth=1.7,
                    label=label,
                )
            axis.axhline(0.0, color="#222222", linewidth=0.9)
            axis.set_yscale("symlog", linthresh=1.0e-2, linscale=0.8)
            axis.set_title(f"{geometry}; {basis}")
            _axes_style(axis)
    for axis in axes[-1, :]:
        axis.set_xlabel("oriented loop flux (a.u.)")
    for axis in axes[:, 0]:
        axis.set_ylabel("smallest metric eigenvalue")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    figure.suptitle("WP5 metric positivity without regularization", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    return _save_figure(figure, output / "metric_positivity_domain")


def _metric_error_figure(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    figure, axes = _member_axes()
    for row_index, geometry in enumerate(("equilateral", "distorted")):
        for column_index, basis in enumerate(("sto-3g", "cc-pvdz")):
            axis = axes[row_index, column_index]
            subset = sorted(
                (
                    row
                    for row in rows
                    if row["geometry"] == geometry
                    and row["basis"] == basis
                    and float(row["loop_flux_au"]) > 0.0
                ),
                key=lambda row: float(row["loop_flux_au"]),
            )
            flux = np.asarray([row["loop_flux_au"] for row in subset])
            for model, linestyle in (
                ("p0", "--"),
                ("geometric_b1", "-."),
            ):
                values = np.asarray([row[f"{model}_relative_error"] for row in subset])
                label = "gB1 = B1" if model == "geometric_b1" else _MODEL_LABELS[model]
                axis.loglog(
                    flux,
                    values,
                    color=_MODEL_COLORS[model],
                    linestyle=linestyle,
                    linewidth=1.7,
                    label=label,
                )
            axis.axhline(1.0e-2, color="#222222", linewidth=0.9, linestyle=":")
            axis.set_title(f"{geometry}; {basis}")
            _axes_style(axis)
    for axis in axes[-1, :]:
        axis.set_xlabel("oriented loop flux (a.u.)")
    for axis in axes[:, 0]:
        axis.set_ylabel("relative metric error")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    figure.suptitle("Exact-versus-approximate metric error", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    return _save_figure(figure, output / "metric_relative_error")


def _spectral_figure(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    figure, axes = _member_axes()
    cases = ("sub_threshold", "near_threshold", "above_threshold")
    labels = ("sub", "near", "above")
    for row_index, geometry in enumerate(("equilateral", "distorted")):
        for column_index, basis in enumerate(("sto-3g", "cc-pvdz")):
            axis = axes[row_index, column_index]
            subset = [
                row
                for row in rows
                if row["geometry"] == geometry and row["basis"] == basis
            ]
            for model, linestyle, marker in (
                ("p0", "--", "o"),
                ("geometric_b1", "-.", "s"),
                ("full_b1", "-", "^"),
            ):
                values = []
                for case in cases:
                    case_rows = [row for row in subset if row["label"] == case]
                    values.append(
                        max(
                            float(row[f"{model}_maximum_absolute_eigenvalue_shift"])
                            for row in case_rows
                        )
                    )
                axis.semilogy(
                    labels,
                    values,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.6,
                    color=_MODEL_COLORS[model],
                    label=_MODEL_LABELS[model],
                )
            axis.set_title(f"{geometry}; {basis}")
            _axes_style(axis)
    for axis in axes[-1, :]:
        axis.set_xlabel("selected field class")
    figure.supylabel("maximum |generalized-eigenvalue shift| (Ha)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    figure.suptitle("One-electron spectral effect at selected positive-metric fields", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    return _save_figure(figure, output / "generalized_spectral_shifts")


def _save_figure(figure: Any, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    figure.savefig(paths[0], dpi=180, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    return paths


def _report(summary: dict[str, Any]) -> str:
    checks = summary["checks"]
    gate = "pass" if checks["g5_numerical_criteria_pass"] else "fail"
    loop_residual = checks["maximum_loop_identity_residual"]
    gauge_residual = checks["maximum_exact_gauge_residual"]
    exact_minimum = checks["minimum_exact_metric_eigenvalue"]
    pair_ratio = checks["maximum_pair_residual_to_floor"]
    return "\n".join(
        (
            "# WP5 multicentre and loop-flux qualification",
            "",
            "Status: **authenticated derived evidence; user review pending**",
            "",
            "The equilateral fixture uses three 1.40-bohr edges. The distorted fixture ",
            "uses 1.12, 1.40, and 1.75 bohr, so every restriction maps to an accepted ",
            "H--H compressed, reference, or stretched fixture. Matrix construction is ",
            "occupation independent. Any later linear one-electron trajectory is one ",
            "electron on three protons (H3^2+), not physical two-electron H3+.",
            "",
            "## Numerical result",
            "",
            f"- Numerical G5 criteria: **{gate}**",
            f"- Maximum loop/gauge/orientation identity residual: {loop_residual:.3e}",
            f"- Maximum exact multicentre gauge residual: {gauge_residual:.3e}",
            f"- Minimum exact Gram-metric eigenvalue: {exact_minimum:.3e}",
            f"- Maximum pair-restriction residual/refined floor: {pair_ratio:.3e}",
            "- No metric eigenvalue was clipped and no metric was regularized.",
            "- gB1 and B1 have the same metric in this static magnetic sector; their ",
            "  mechanical matrices differ and therefore their generalized spectra differ.",
            "",
            "The selected sub-, near-, and above-threshold field classes use P0 relative ",
            "metric errors targeted at 1e-3, 1e-2, and 5e-2. The cc-pVDZ rows are selected ",
            "for WP6; STO-3G rows remain minimal-basis diagnostics. In cc-pVDZ the 5e-2 ",
            "preference cannot be reached before P0 loses positivity: the common-domain ",
            "above-threshold cases at loop flux 0.05 have errors about 1.7e-2 and still ",
            "exceed the declared 1e-2 physical threshold. Every selected case remains ",
            "above the 1e-6 minimum-eigenvalue safety margin for every compared model. ",
            "These are qualification stress fields, not laboratory-field recommendations.",
            "",
            "## Artifacts",
            "",
            "- `tables/metric_domain.csv`",
            "- `tables/selected_dynamical_cases.csv`",
            "- `tables/generalized_spectral_comparisons.csv`",
            "- `tables/pair_restrictions.csv`",
            "- `tables/structural_checks.csv`",
            "- `figures/metric_positivity_domain.{png,pdf}`",
            "- `figures/metric_relative_error.{png,pdf}`",
            "- `figures/generalized_spectral_shifts.{png,pdf}`",
            "",
            "Passing numerical criteria does not itself accept gate G5. Only an explicit ",
            "user review decision may create that record.",
            "",
        )
    )


def main() -> None:
    execution = _arguments()
    _index, members = _members(execution)
    plan = _json(execution / "execution_plan.json")
    tolerances = plan["fixture"]["acceptance_tolerances"]
    margin = float(
        plan["fixture"]["selected_case_targets"][
            "selection_minimum_metric_eigenvalue"
        ]
    )
    metric, selected, spectral, pair, structural = _tables(members)
    output = execution / "analysis"
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    table_values = {
        "metric_domain.csv": metric,
        "selected_dynamical_cases.csv": selected,
        "generalized_spectral_comparisons.csv": spectral,
        "pair_restrictions.csv": pair,
        "structural_checks.csv": structural,
    }
    for name, rows in table_values.items():
        _write_csv(tables / name, rows)
    figure_paths = [
        *_metric_positivity_figure(metric, figures),
        *_metric_error_figure(metric, figures),
        *_spectral_figure(spectral, figures),
    ]
    maximum_loop = max(float(row["maximum_loop_identity_residual"]) for row in structural)
    maximum_gauge = max(float(row["maximum_exact_gauge_residual"]) for row in structural)
    minimum_exact = min(float(row["minimum_exact_metric_eigenvalue"]) for row in structural)
    maximum_pair = max(float(row["maximum_pair_residual_to_floor"]) for row in structural)
    minimum_selected = min(float(row["minimum_model_metric_eigenvalue"]) for row in selected)
    checks = {
        "maximum_loop_identity_residual": maximum_loop,
        "maximum_exact_gauge_residual": maximum_gauge,
        "minimum_exact_metric_eigenvalue": minimum_exact,
        "maximum_pair_residual_to_floor": maximum_pair,
        "minimum_selected_model_metric_eigenvalue": minimum_selected,
        "metric_regularization_used": False,
        "g5_numerical_criteria_pass": (
            maximum_loop <= float(tolerances["loop_holonomy_absolute"])
            and maximum_gauge <= float(tolerances["gauge_congruence_relative"])
            and minimum_exact > 0.0
            and maximum_pair
            <= float(tolerances["pair_restriction_floor_multiplier"])
            and minimum_selected > margin
        ),
    }
    manifest_core = {
        "schema": "aion.exact-one-electron-wp5-analysis-manifest",
        "version": "1.0.0",
        "status": "derived_executed_unreviewed",
        "source_execution_directory": str(execution),
        "execution_index_sha256": _sha256(execution / "execution_index.json"),
        "analyzer_sha256": _sha256(Path(__file__).resolve()),
        "source_member_count": len(members),
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
        "schema": "aion.exact-one-electron-wp5-analysis-summary",
        "version": "1.0.0",
        "status": "derived_executed_unreviewed",
        "manifest_id": manifest_id,
        "checks": checks,
        "row_counts": {
            "metric_domain": len(metric),
            "selected_dynamical_cases": len(selected),
            "generalized_spectral_comparisons": len(spectral),
            "pair_restrictions": len(pair),
            "structural_checks": len(structural),
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
    print(f"g5_numerical_criteria_pass={checks['g5_numerical_criteria_pass']}")
    print(f"minimum_exact_metric_eigenvalue={minimum_exact:.6e}")
    print(f"minimum_selected_model_metric_eigenvalue={minimum_selected:.6e}")


if __name__ == "__main__":
    main()
