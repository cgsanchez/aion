#!/usr/bin/env python3
"""Synthesize authenticated reference and distance WP3 evidence for gate G3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import analyze_exact_one_electron_wp3_reference as reference
from matplotlib import pyplot as plt

from aion.config import canonical_sha256

_SCALES = (0.8, 1.0, 1.25)
_SYSTEMS = ("hh", "oh", "n2", "co")
_BASES = ("sto-3g", "cc-pvdz", "aug-cc-pvdz")
_DIRECTIONS = ("parallel", "perpendicular_x", "oblique")
_SYSTEM_LABELS = {"hh": "H-H", "oh": "O-H", "n2": "N2", "co": "CO"}
_BASIS_LABELS = {
    "sto-3g": "STO-3G",
    "cc-pvdz": "cc-pVDZ",
    "aug-cc-pvdz": "aug-cc-pVDZ",
}
_DIRECTION_LABELS = {
    "parallel": "parallel",
    "perpendicular_x": "perpendicular",
    "oblique": "oblique",
}
_DIRECTION_STYLES = {
    "parallel": ("o", "-"),
    "perpendicular_x": ("s", "--"),
    "oblique": ("^", ":"),
}
_BASIS_COLORS = {
    "sto-3g": "#245b8a",
    "cc-pvdz": "#d36b27",
    "aug-cc-pvdz": "#6f7f34",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_execution_directory", type=Path)
    parser.add_argument("distance_execution_directory", type=Path)
    return parser.parse_args()


def _distance_members(
    execution: Path, audit_result: dict[str, Any]
) -> dict[float, list[reference.Member]]:
    index = reference._json(execution / "execution_index.json")
    audit_by_key = {
        (row["system"], row["basis"], float(row["bond_scale"])): row
        for row in audit_result["members"]
    }
    groups: dict[float, list[reference.Member]] = {0.8: [], 1.25: []}
    for record in index["completed_members"]:
        scale = float(record["bond_scale"])
        if scale not in groups:
            raise ValueError(f"unexpected distance scale {scale}")
        directory = Path(record["member_directory"])
        for filename, field in (
            ("manifest.json", "manifest_sha256"),
            ("result.json", "result_sha256"),
            ("arrays.npz", "arrays_sha256"),
        ):
            if reference._sha256(directory / filename) != record[field]:
                raise ValueError(f"{filename} hash mismatch: {directory}")
        audit_row = audit_by_key[(record["system"], record["basis"], scale)]
        floors = {
            family: float(
                audit_row["floors_absolute_frobenius"][family]["refined_working_floor"]
            )
            for family in reference._FAMILIES
        }
        groups[scale].append(
            reference.Member(
                system=record["system"],
                basis=record["basis"],
                directory=directory,
                manifest=reference._json(directory / "manifest.json"),
                result=reference._json(directory / "result.json"),
                refined_floors=floors,
            )
        )
    for scale, members in groups.items():
        if len(members) != 12:
            raise ValueError(f"expected 12 members at bond scale {scale}, found {len(members)}")
        members.sort(
            key=lambda value: (
                reference._SYSTEM_ORDER.index(value.system),
                reference._BASIS_ORDER.index(value.basis),
            )
        )
    return groups


def _tag(rows: list[dict[str, Any]], scale: float) -> list[dict[str, Any]]:
    for row in rows:
        row["bond_scale"] = scale
    return rows


def _derive_rows(
    groups: dict[float, list[reference.Member]], fixture: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    combined = {
        "thresholds": [],
        "parity": [],
        "sign": [],
        "orientation": [],
        "sectors": [],
        "channels": [],
        "spectra": [],
    }
    moved = 0
    for scale in _SCALES:
        members = groups[scale]
        thresholds, moved_at_scale = reference._threshold_rows(members, fixture)
        sign, orientation = reference._identity_rows(members)
        sectors, channels, spectra = reference._selected_rows(members)
        combined["thresholds"].extend(_tag(thresholds, scale))
        combined["parity"].extend(_tag(reference._parity_rows(members), scale))
        combined["sign"].extend(_tag(sign, scale))
        combined["orientation"].extend(_tag(orientation, scale))
        combined["sectors"].extend(_tag(sectors, scale))
        combined["channels"].extend(_tag(channels, scale))
        combined["spectra"].extend(_tag(spectra, scale))
        moved += moved_at_scale
    return combined, moved


def _checks(
    groups: dict[float, list[reference.Member]],
    rows: dict[str, list[dict[str, Any]]],
    audits: tuple[dict[str, Any], dict[str, Any]],
    moved: int,
) -> dict[str, Any]:
    crossed = [row for row in rows["thresholds"] if row["crossed"]]
    resolved = [row for row in rows["parity"] if row["status"].startswith("resolved_")]
    failed = [row for row in resolved if row["status"] == "resolved_fail"]
    orientation = [
        row for row in rows["orientation"] if row["both_above_refined_floor"]
    ]
    values: dict[str, Any] = {
        "minimum_crossing_signal_to_floor": min(
            float(row["signal_to_refined_floor"]) for row in crossed
        ),
        "threshold_crossings_moved_by_refinement": moved,
        "crossings_below_ten_times_floor": sum(
            float(row["signal_to_refined_floor"]) < 10.0 for row in crossed
        ),
        "resolved_parity_fit_count": len(resolved),
        "failed_parity_fit_count": len(failed),
        "maximum_parity_power_error": max(
            abs(float(row["fitted_power"]) - float(row["expected_power"]))
            for row in resolved
        ),
        "symmetry_forbidden_fit_count": sum(
            row["status"] == "symmetry_forbidden_geometry" for row in rows["parity"]
        ),
        "symmetry_suppressed_fit_count": sum(
            row["status"] == "symmetry_suppressed_observed" for row in rows["parity"]
        ),
        "below_floor_fit_count": sum(
            row["status"] == "below_numerical_floor" for row in rows["parity"]
        ),
        "maximum_sign_reversal_relative": max(
            float(row["relative_residual"]) for row in rows["sign"]
        ),
        "maximum_perpendicular_relative_difference": max(
            float(row["relative_difference"]) for row in orientation
        ),
        "maximum_gauge_congruence_relative": max(
            float(audit["maximum_lower_congruence_relative"]) for audit in audits
        ),
        "maximum_gauge_barred_invariance_relative": max(
            float(audit["maximum_barred_invariance_relative"]) for audit in audits
        ),
        "maximum_gauge_spectrum_residual_hartree": max(
            float(audit["maximum_generalized_spectrum_residual_hartree"])
            for audit in audits
        ),
        "minimum_exact_metric_eigenvalue": min(
            float(member.result["minimum_exact_metric_eigenvalue"])
            for members in groups.values()
            for member in members
        ),
        "minimum_p0_metric_eigenvalue": min(
            float(member.result["minimum_p0_metric_eigenvalue"])
            for members in groups.values()
            for member in members
        ),
    }
    values["g3_numerical_criteria_pass"] = bool(
        values["minimum_crossing_signal_to_floor"]
        >= reference._THRESHOLD_FLOOR_MULTIPLIER
        and values["failed_parity_fit_count"] == 0
        and values["maximum_sign_reversal_relative"] <= reference._SIGN_REVERSAL_TOLERANCE
        and values["maximum_perpendicular_relative_difference"]
        <= reference._ORIENTATION_TOLERANCE
        and all(audit["all_gauge_checks_pass"] for audit in audits)
        and values["minimum_exact_metric_eigenvalue"] > 0.0
        and values["minimum_p0_metric_eigenvalue"] > 0.0
    )
    return values


def _write_tables(output: Path, rows: dict[str, list[dict[str, Any]]]) -> list[Path]:
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    names = {
        "thresholds": "thresholds.csv",
        "parity": "parity_fits.csv",
        "sign": "field_reversal.csv",
        "orientation": "perpendicular_orientation.csv",
        "sectors": "kinetic_sectors_at_1e-2.csv",
        "channels": "angular_channels_at_1e-2.csv",
        "spectra": "spectra_at_1e-2.csv",
    }
    paths = []
    for key, filename in names.items():
        path = tables / filename
        reference._write_csv(path, tuple(rows[key][0]), rows[key])
        paths.append(path)
    return paths


def _plot_thresholds(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    figures: list[Path] = []
    specifications = (
        ("mechanical_global", 1.0e-4, "Mechanical matrix threshold field"),
        ("spectral_max_abs_hartree", 1.0e-4, "Generalized-spectrum threshold field"),
    )
    for metric, threshold, title in specifications:
        fig, axes = plt.subplots(4, 3, figsize=(13.5, 12.0), sharex=True, sharey=True)
        for system_index, system in enumerate(_SYSTEMS):
            for basis_index, basis in enumerate(_BASES):
                axis = axes[system_index, basis_index]
                for direction in _DIRECTIONS:
                    selected = sorted(
                        (
                            row
                            for row in rows
                            if row["system"] == system
                            and row["basis"] == basis
                            and row["direction"] == direction
                            and row["metric"] == metric
                            and np.isclose(float(row["threshold"]), threshold)
                        ),
                        key=lambda row: float(row["bond_scale"]),
                    )
                    fields = np.asarray(
                        [
                            np.nan
                            if row["upper_bracket_au"] is None
                            else float(row["upper_bracket_au"])
                            for row in selected
                        ]
                    )
                    marker, linestyle = _DIRECTION_STYLES[direction]
                    axis.plot(
                        _SCALES,
                        fields,
                        marker=marker,
                        linestyle=linestyle,
                        color=_BASIS_COLORS[basis],
                        linewidth=1.4,
                        markersize=4.0,
                        label=_DIRECTION_LABELS[direction],
                    )
                axis.set_yscale("log")
                axis.grid(True, which="both", color="#d9dee5", linewidth=0.65)
                axis.set_title(
                    f"{_SYSTEM_LABELS[system]} / {_BASIS_LABELS[basis]}", fontsize=9.5
                )
                if basis_index == 0:
                    axis.set_ylabel("first sampled |B| (a.u.)")
                if system_index == len(_SYSTEMS) - 1:
                    axis.set_xlabel("bond-length scale")
        axes[0, 0].legend(frameon=False, fontsize=8, ncol=1)
        fig.suptitle(f"{title} at {threshold:g}", fontsize=14)
        fig.tight_layout()
        figures.extend(reference._save_figure(fig, output / f"{metric}_distance"))
    return figures


def _plot_metric(
    groups: dict[float, list[reference.Member]], output: Path
) -> list[Path]:
    fig, axes = plt.subplots(4, 3, figsize=(13.5, 12.0), sharex=True, sharey=True)
    for system_index, system in enumerate(_SYSTEMS):
        for basis_index, basis in enumerate(_BASES):
            axis = axes[system_index, basis_index]
            exact = []
            p0 = []
            for scale in _SCALES:
                member = next(
                    item
                    for item in groups[scale]
                    if item.system == system and item.basis == basis
                )
                exact.append(float(member.result["minimum_exact_metric_eigenvalue"]))
                p0.append(float(member.result["minimum_p0_metric_eigenvalue"]))
            axis.semilogy(_SCALES, exact, "o-", color="#245b8a", label="exact")
            axis.semilogy(_SCALES, p0, "s--", color="#d36b27", label="P0")
            axis.grid(True, which="both", color="#d9dee5", linewidth=0.65)
            axis.set_title(f"{_SYSTEM_LABELS[system]} / {_BASIS_LABELS[basis]}", fontsize=9.5)
            if basis_index == 0:
                axis.set_ylabel("minimum metric eigenvalue")
            if system_index == len(_SYSTEMS) - 1:
                axis.set_xlabel("bond-length scale")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Metric positivity across bond-length scan", fontsize=14)
    fig.tight_layout()
    return reference._save_figure(fig, output / "metric_positivity_distance")


def _report(summary: dict[str, Any]) -> str:
    checks = summary["checks"]
    return "\n".join(
        [
            "# WP3 gate-G3 numerical synthesis",
            "",
            "Status: **authenticated derived evidence; user review pending**",
            "",
            "This synthesis combines the reference, compressed (0.8), and stretched ",
            "(1.25) two-centre campaigns without changing the basis, grid, field range, ",
            "endpoint convention, or threshold rules. Passing the numerical criteria is ",
            "not itself a gate-acceptance record.",
            "",
            "## Coverage",
            "",
            f"- Members: {summary['coverage']['members']} (4 systems x 3 bases x 3 distances)",
            f"- Field cases: {summary['coverage']['field_cases']}",
            "- Both signs; parallel, two perpendicular, and oblique directions",
            "- Field range: 1e-7 through 1 a.u., including adaptive refinement",
            "",
            "## Gate checks",
            "",
            "- Numerical G3 criteria: "
            f"**{'pass' if checks['g3_numerical_criteria_pass'] else 'fail'}**",
            f"- Resolved parity fits: {checks['resolved_parity_fit_count']} passed, "
            f"{checks['failed_parity_fit_count']} failed",
            f"- Maximum exponent error: {checks['maximum_parity_power_error']:.3e}",
            f"- Minimum threshold signal/refined-floor ratio: "
            f"{checks['minimum_crossing_signal_to_floor']:.3f}",
            f"- Maximum field-reversal residual: "
            f"{checks['maximum_sign_reversal_relative']:.3e}",
            f"- Maximum resolved perpendicular x/y mismatch: "
            f"{checks['maximum_perpendicular_relative_difference']:.3e}",
            f"- Maximum finite-field gauge-congruence residual: "
            f"{checks['maximum_gauge_congruence_relative']:.3e}",
            f"- Maximum gauge spectral residual: "
            f"{checks['maximum_gauge_spectrum_residual_hartree']:.3e} Ha",
            f"- Minimum exact metric eigenvalue: "
            f"{checks['minimum_exact_metric_eigenvalue']:.3e}",
            "",
            "## Interpretation boundary",
            "",
            "The evidence establishes numerical visibility, odd/even small-field scaling, ",
            "block and generalized-spectral support, metric positivity, and gauge/orientation ",
            "checks over the executed pair campaign. It does not establish G4 first-order ",
            "validity; that requires direct derivative and model-remainder qualification.",
            "",
            "## Artifacts",
            "",
            "- `tables/thresholds.csv`: all refined threshold brackets",
            "- `tables/parity_fits.csv`: all fitted and unresolved parity channels",
            "- `tables/kinetic_sectors_at_1e-2.csv`: triangle/anchored decomposition",
            "- `tables/angular_channels_at_1e-2.csv`: block support",
            "- `tables/spectra_at_1e-2.csv`: generalized spectral support",
            "- `figures/*_distance.{png,pdf}`: distance dependence and metric positivity",
            "",
        ]
    )


def main() -> None:
    arguments = _arguments()
    reference_execution = arguments.reference_execution_directory.resolve()
    distance_execution = arguments.distance_execution_directory.resolve()
    reference_audit_path = reference_execution / "reference_audit"
    distance_audit_path = distance_execution / "distance_audit"
    reference_audit = reference._authenticate_audit(reference_audit_path)
    distance_audit = reference._authenticate_audit(distance_audit_path)
    reference_members = reference._load_members(reference_execution, reference_audit)
    groups = {1.0: reference_members, **_distance_members(distance_execution, distance_audit)}
    fixture = reference._json(reference_execution / "execution_plan.json")["fixture"]
    distance_fixture = reference._json(distance_execution / "execution_plan.json")["fixture"]
    if canonical_sha256(fixture) != canonical_sha256(distance_fixture):
        reference_scales = dict(fixture)
        distance_scales = dict(distance_fixture)
        reference_scales.pop("bond_scales", None)
        distance_scales.pop("bond_scales", None)
        if canonical_sha256(reference_scales) != canonical_sha256(distance_scales):
            raise ValueError("reference and distance fixtures differ beyond bond scales")

    rows, moved = _derive_rows(groups, fixture)
    checks = _checks(groups, rows, (reference_audit, distance_audit), moved)
    output = distance_execution / "g3_analysis"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    table_paths = _write_tables(output, rows)
    figure_paths = [
        *_plot_thresholds(rows["thresholds"], figures),
        *_plot_metric(groups, figures),
    ]

    manifest_core = {
        "schema": "aion.exact-one-electron-wp3-g3-analysis-manifest",
        "version": "1.0.0",
        "status": "derived_executed_unreviewed",
        "reference_execution_directory": str(reference_execution),
        "distance_execution_directory": str(distance_execution),
        "reference_execution_index_sha256": reference._sha256(
            reference_execution / "execution_index.json"
        ),
        "distance_execution_index_sha256": reference._sha256(
            distance_execution / "execution_index.json"
        ),
        "reference_audit_result_sha256": reference._sha256(
            reference_audit_path / "result.json"
        ),
        "distance_audit_result_sha256": reference._sha256(
            distance_audit_path / "result.json"
        ),
        "analysis_runner_sha256": reference._sha256(Path(__file__).resolve()),
        "rules": {
            "threshold_floor_multiplier": reference._THRESHOLD_FLOOR_MULTIPLIER,
            "parity_floor_multiplier": reference._PARITY_FLOOR_MULTIPLIER,
            "parity_fit_max_field_au": reference._PARITY_MAX_FIELD_AU,
            "parity_fit_max_signal": reference._PARITY_SIGNAL_MAX,
            "parity_power_tolerance": reference._PARITY_POWER_TOLERANCE,
            "sign_reversal_relative_tolerance": reference._SIGN_REVERSAL_TOLERANCE,
            "orientation_relative_tolerance": reference._ORIENTATION_TOLERANCE,
        },
    }
    manifest_id = canonical_sha256(manifest_core)
    manifest_path = output / "manifest.json"
    reference._write_json(manifest_path, {**manifest_core, "manifest_id": manifest_id})
    summary = {
        "schema": "aion.exact-one-electron-wp3-g3-analysis",
        "version": "1.0.0",
        "status": "derived_executed_unreviewed",
        "manifest_id": manifest_id,
        "coverage": {
            "members": 36,
            "field_cases": sum(
                len(member.result["cases"])
                for members in groups.values()
                for member in members
            ),
            "systems": list(_SYSTEMS),
            "bases": list(_BASES),
            "bond_scales": list(_SCALES),
        },
        "checks": checks,
        "artifacts": {
            "tables": {path.name: reference._sha256(path) for path in table_paths},
            "figures": {path.name: reference._sha256(path) for path in figure_paths},
        },
        "review": {"reviewed": False, "reviewer": None, "decision": "pending"},
    }
    summary_path = output / "summary.json"
    reference._write_json(summary_path, summary)
    report_path = output / "report.md"
    report_path.write_text(_report(summary), encoding="utf-8")
    reference._write_json(
        output / "completed.json",
        {
            "status": "authenticated_derived_executed_unreviewed",
            "manifest_id": manifest_id,
            "manifest_sha256": reference._sha256(manifest_path),
            "summary_sha256": reference._sha256(summary_path),
            "report_sha256": reference._sha256(report_path),
            "g3_numerical_criteria_pass": checks["g3_numerical_criteria_pass"],
        },
    )
    print(f"analysis={output}")
    print(f"g3_numerical_criteria_pass={checks['g3_numerical_criteria_pass']}")
    print(f"resolved_parity_fits={checks['resolved_parity_fit_count']}")
    print(f"failed_parity_fits={checks['failed_parity_fit_count']}")
    print(
        "minimum_crossing_signal_to_floor="
        f"{checks['minimum_crossing_signal_to_floor']:.6g}"
    )


if __name__ == "__main__":
    main()
