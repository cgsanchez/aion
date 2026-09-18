#!/usr/bin/env python3
"""Analyze an authenticated WP4 first-order hierarchy campaign."""

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

_FAMILIES = ("overlap", "kinetic", "nuclear_attraction", "mechanical")
_MODELS = ("p0", "geometric_b1", "full_b1", "complete_first")
_DIRECTIONS = ("parallel", "perpendicular_x", "perpendicular_y", "oblique")
_FIT_FIELDS = (1.0e-4, 1.0e-3, 1.0e-2)
_VALIDITY_THRESHOLDS = (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2)
_FLOOR_MULTIPLIER = 4.0
_POWER_TOLERANCE = 8.0e-2
_ADJOINT_TOLERANCE = 1.0e-10
_SECTOR_TOLERANCE = 1.0e-12
_E1_PARENT_TOLERANCE = 1.0e-10
_MODEL_COLORS = {
    "p0": "#7a7a7a",
    "geometric_b1": "#d36b27",
    "full_b1": "#245b8a",
    "complete_first": "#6f7f34",
}
_FAMILY_LABELS = {
    "overlap": "overlap",
    "kinetic": "kinetic",
    "nuclear_attraction": "nuclear attraction",
    "mechanical": "mechanical",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution_directory", type=Path)
    return parser.parse_args()


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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _members(execution: Path) -> list[dict[str, Any]]:
    index = _json(execution / "execution_index.json")
    if index["status"] != "authenticated_executed_unreviewed":
        raise ValueError("WP4 execution is not complete")
    members = []
    for record in index["completed_members"]:
        directory = Path(record["member_directory"])
        for filename, field in (
            ("manifest.json", "manifest_sha256"),
            ("result.json", "result_sha256"),
            ("arrays.npz", "arrays_sha256"),
        ):
            if _sha256(directory / filename) != record[field]:
                raise ValueError(f"WP4 member hash mismatch: {directory / filename}")
        members.append(
            {
                **record,
                "directory": directory,
                "manifest": _json(directory / "manifest.json"),
                "result": _json(directory / "result.json"),
            }
        )
    if len(members) != 36:
        raise ValueError(f"G4 analysis requires 36 members, found {len(members)}")
    return members


def _e1_refinement_audit(execution: Path) -> dict[str, Any]:
    audit = execution / "e1_refinement_audit"
    completed = _json(audit / "completed.json")
    for filename, field in (
        ("manifest.json", "manifest_sha256"),
        ("result.json", "result_sha256"),
        ("arrays.npz", "arrays_sha256"),
    ):
        if _sha256(audit / filename) != completed[field]:
            raise ValueError(f"E1 refinement-audit hash mismatch: {filename}")
    result = _json(audit / "result.json")
    if result["member_count"] != 36:
        raise ValueError("E1 refinement audit does not cover 36 members")
    return result


def _metadata(member: dict[str, Any]) -> dict[str, Any]:
    return {
        "system": member["system"],
        "basis": member["basis"],
        "bond_scale": float(member["bond_scale"]),
    }


def _derivative_rows(members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for member in members:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in member["result"]["magnetic_derivative_finite_differences"]:
            grouped.setdefault((row["direction"], row["family"]), []).append(row)
        for (direction, family), values in grouped.items():
            selected = sorted(
                (
                    row
                    for row in values
                    if any(np.isclose(float(row["step_au"]), field) for field in _FIT_FIELDS)
                    and float(row["analytic_signal_to_cancellation_floor"])
                    >= _FLOOR_MULTIPLIER
                ),
                key=lambda row: float(row["step_au"]),
            )
            maximum_error = max(float(row["absolute_error"]) for row in values)
            analytic_norm = max(float(row["analytic_norm"]) for row in values)
            roundoff_exact = maximum_error <= 1.0e-12 * max(1.0, analytic_norm)
            fitted: float | None = None
            status: str
            if roundoff_exact:
                status = "exact_linear_to_roundoff"
            elif len(selected) >= 2 and all(float(row["absolute_error"]) > 0.0 for row in selected):
                fitted = float(
                    np.polyfit(
                        np.log([float(row["step_au"]) for row in selected]),
                        np.log([float(row["absolute_error"]) for row in selected]),
                        1,
                    )[0]
                )
                status = (
                    "resolved_pass"
                    if abs(fitted - 2.0) <= _POWER_TOLERANCE
                    else "resolved_fail"
                )
            else:
                status = "below_cancellation_floor"
            output.append(
                {
                    **_metadata(member),
                    "direction": direction,
                    "family": family,
                    "status": status,
                    "expected_power": 2.0,
                    "fitted_power": fitted,
                    "fit_point_count": len(selected),
                    "fit_step_min_au": (
                        None if not selected else min(float(row["step_au"]) for row in selected)
                    ),
                    "fit_step_max_au": (
                        None if not selected else max(float(row["step_au"]) for row in selected)
                    ),
                    "minimum_relative_error": min(
                        float(row["relative_error"]) for row in values
                    ),
                    "maximum_analytic_signal_to_cancellation_floor": max(
                        float(row["analytic_signal_to_cancellation_floor"])
                        for row in values
                    ),
                }
            )
    return output


def _case_maps(source_result: dict[str, Any]) -> tuple[dict[str, int], dict[tuple[Any, ...], Any]]:
    indices = {
        case_id: index for index, case_id in enumerate(source_result["case_ids_in_array_order"])
    }
    cases = {
        (case["direction"], int(case["sign"]), float(case["magnitude_au"])): case
        for case in source_result["cases"]
        if int(case["sign"]) != 0
    }
    return indices, cases


def _predicted_remainder_power(
    model: str,
    family: str,
    coefficient_signal_at_max_field: float,
    floor: float,
) -> float | None:
    if model in ("full_b1", "complete_first"):
        return 2.0
    if model == "geometric_b1" and family == "overlap":
        return 2.0
    if coefficient_signal_at_max_field >= _FLOOR_MULTIPLIER * floor:
        return 1.0
    return None


def _remainder_and_validity_rows(
    members: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fits: list[dict[str, Any]] = []
    validity: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    for member in members:
        source = Path(member["manifest"]["source_wp3"]["directory"])
        source_result = _json(source / "result.json")
        source_manifest = _json(source / "manifest.json")
        indices, cases = _case_maps(source_result)
        directions = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in source_manifest["field_scan"]["directions"].items()
        }
        floors = member["result"]["refined_floors_absolute_frobenius"]
        with np.load(member["directory"] / "arrays.npz", allow_pickle=False) as arrays:
            coefficients = {
                family: np.asarray(arrays[f"derivative__{family}"])
                for family in _FAMILIES
            }
            for direction_name, direction in directions.items():
                for family in _FAMILIES:
                    coefficient_norm = float(
                        np.linalg.norm(np.einsum("x,xmn->mn", direction, coefficients[family]))
                    )
                    for model in _MODELS:
                        values = []
                        for field in _FIT_FIELDS:
                            case = cases[(direction_name, 1, field)]
                            value = float(
                                arrays[f"model_error_absolute__{model}__{family}"][
                                    indices[case["id"]]
                                ]
                            )
                            if value >= _FLOOR_MULTIPLIER * float(floors[family]):
                                values.append((field, value))
                        predicted = _predicted_remainder_power(
                            model,
                            family,
                            coefficient_norm * max(_FIT_FIELDS),
                            float(floors[family]),
                        )
                        fitted: float | None = None
                        if len(values) >= 2:
                            fitted = float(
                                np.polyfit(
                                    np.log([row[0] for row in values]),
                                    np.log([row[1] for row in values]),
                                    1,
                                )[0]
                            )
                        if predicted is None:
                            status = "linear_coefficient_unresolved"
                        elif fitted is None:
                            status = "remainder_below_floor"
                        else:
                            status = (
                                "resolved_pass"
                                if abs(fitted - predicted) <= _POWER_TOLERANCE
                                else "resolved_fail"
                            )
                        fits.append(
                            {
                                **_metadata(member),
                                "direction": direction_name,
                                "model": model,
                                "family": family,
                                "status": status,
                                "expected_power": predicted,
                                "fitted_power": fitted,
                                "fit_point_count": len(values),
                                "fit_field_min_au": (
                                    None if not values else min(row[0] for row in values)
                                ),
                                "fit_field_max_au": (
                                    None if not values else max(row[0] for row in values)
                                ),
                                "coefficient_norm": coefficient_norm,
                                "coefficient_signal_at_fit_max": coefficient_norm
                                * max(_FIT_FIELDS),
                                "refined_floor": float(floors[family]),
                            }
                        )

                        positive_cases = sorted(
                            (
                                (key[2], indices[case["id"]])
                                for key, case in cases.items()
                                if key[0] == direction_name and key[1] == 1
                            ),
                            key=lambda row: row[0],
                        )
                        absolute = np.asarray(
                            arrays[f"model_error_absolute__{model}__{family}"]
                        )
                        relative = np.asarray(
                            arrays[f"model_error_relative__{model}__{family}"]
                        )
                        for threshold in _VALIDITY_THRESHOLDS:
                            crossing = None
                            previous = 0.0
                            for field, case_index in positive_cases:
                                if (
                                    relative[case_index] >= threshold
                                    and absolute[case_index]
                                    >= _FLOOR_MULTIPLIER * float(floors[family])
                                ):
                                    crossing = (field, case_index)
                                    break
                                previous = field
                            validity.append(
                                {
                                    **_metadata(member),
                                    "direction": direction_name,
                                    "model": model,
                                    "family": family,
                                    "relative_threshold": threshold,
                                    "crossed": crossing is not None,
                                    "lower_bracket_au": previous,
                                    "upper_bracket_au": (
                                        None if crossing is None else crossing[0]
                                    ),
                                    "relative_error_at_upper": (
                                        None
                                        if crossing is None
                                        else float(relative[crossing[1]])
                                    ),
                                }
                            )

                field = 1.0e-2
                case = cases[(direction_name, 1, field)]
                case_index = indices[case["id"]]
                exact = np.load(source / "arrays.npz", allow_pickle=False)
                try:
                    exact_mechanical = np.asarray(exact["barred__exact__mechanical"])[case_index]
                    p0_mechanical = np.asarray(exact["barred__p0__mechanical"])[case_index]
                finally:
                    exact.close()
                increment = field * np.einsum(
                    "x,xmn->mn", direction, coefficients["mechanical"]
                )
                mapping = np.asarray(source_manifest["system"]["ao_to_atom"], dtype=np.int64)
                angular = np.asarray(
                    source_manifest["system"]["ao_angular_momenta"], dtype=np.int64
                )
                for bra_atom in np.unique(mapping):
                    for ket_atom in np.unique(mapping):
                        for bra_l in np.unique(angular[mapping == bra_atom]):
                            for ket_l in np.unique(angular[mapping == ket_atom]):
                                bra = np.flatnonzero((mapping == bra_atom) & (angular == bra_l))
                                ket = np.flatnonzero((mapping == ket_atom) & (angular == ket_l))
                                block = np.ix_(bra, ket)
                                increment_norm = float(np.linalg.norm(increment[block]))
                                gb1_error = float(
                                    np.linalg.norm((exact_mechanical - p0_mechanical)[block])
                                )
                                b1_error = float(
                                    np.linalg.norm(
                                        (exact_mechanical - p0_mechanical - increment)[block]
                                    )
                                )
                                resolved = increment_norm >= _FLOOR_MULTIPLIER * float(
                                    floors["mechanical"]
                                )
                                improvement = gb1_error / max(
                                    b1_error, np.finfo(np.float64).tiny
                                )
                                channels.append(
                                    {
                                        **_metadata(member),
                                        "direction": direction_name,
                                        "field_au": field,
                                        "bra_atom": int(bra_atom),
                                        "ket_atom": int(ket_atom),
                                        "bra_l": int(bra_l),
                                        "ket_l": int(ket_l),
                                        "mechanical_increment_norm": increment_norm,
                                        "geometric_b1_error": gb1_error,
                                        "full_b1_error": b1_error,
                                        "improvement_factor": improvement,
                                        "status": (
                                            "mechanical_b1_required"
                                            if resolved and improvement >= 10.0
                                            else (
                                                "resolved_no_decade_improvement"
                                                if resolved
                                                else "mechanical_increment_unresolved"
                                            )
                                        ),
                                    }
                                )
    return fits, validity, channels


def _structural_rows(
    members: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    structural = []
    electric = []
    for member in members:
        result = member["result"]
        for row in result["coefficient_adjoint_checks"]:
            structural.append(
                {
                    **_metadata(member),
                    "check": "coefficient_adjoint",
                    "family": row["family"],
                    "component": row["cartesian_axis"],
                    "residual": row["relative_residual"],
                }
            )
        for row in result["sector_increment_identities"]:
            for name, value in row.items():
                if name.endswith("residual"):
                    structural.append(
                        {
                            **_metadata(member),
                            "check": name,
                            "family": row["family"],
                            "component": "all_cases",
                            "residual": value,
                        }
                    )
        e1 = result["electric_e1"]
        for row in e1["finite_source_differences"]:
            electric.append(
                {
                    **_metadata(member),
                    "direction": row["direction"],
                    "step_au": row["step_au"],
                    "central_dipole_relative_residual": e1[
                        "central_dipole_relative_residual"
                    ],
                    "connection_adjoint_residual": e1["connection_adjoint_residual"],
                    "finite_difference_error_vs_analytic": row[
                        "absolute_error_vs_analytic"
                    ],
                    "finite_difference_error_vs_quadrature": row[
                        "absolute_error_vs_quadrature"
                    ],
                }
            )
    return structural, electric


def _save_figure(fig: Any, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=180, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _plot_derivatives(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(9.5, 5.0))
    x_positions = np.arange(len(_FAMILIES))
    for family_index, family in enumerate(_FAMILIES):
        values = [
            float(row["fitted_power"])
            for row in rows
            if row["family"] == family and row["status"].startswith("resolved_")
        ]
        jitter = np.linspace(-0.18, 0.18, len(values)) if values else np.asarray([])
        axis.scatter(
            family_index + jitter,
            values,
            s=15,
            color="#245b8a",
            alpha=0.55,
        )
    axis.axhline(2.0, color="#222222", linestyle="--", linewidth=1.2, label="expected")
    axis.set_ylim(1.95, 2.05)
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.set_xticks(x_positions, [_FAMILY_LABELS[value] for value in _FAMILIES])
    axis.set_ylabel("centered-difference error exponent")
    axis.set_title("Analytic magnetic derivatives versus exact-parent finite differences")
    axis.grid(True, axis="y", color="#d9dee5", linewidth=0.7)
    axis.legend(frameon=False)
    fig.tight_layout()
    return _save_figure(fig, output / "magnetic_derivative_convergence")


def _plot_remainders(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), sharey=True)
    for axis, family in zip(axes.flat, _FAMILIES, strict=True):
        for model_index, model in enumerate(_MODELS):
            selected = [
                row
                for row in rows
                if row["family"] == family and row["model"] == model
                and row["status"].startswith("resolved_")
            ]
            values = [float(row["fitted_power"]) for row in selected]
            jitter = np.linspace(-0.16, 0.16, len(values)) if values else np.asarray([])
            axis.scatter(
                model_index + jitter,
                values,
                s=14,
                alpha=0.5,
                color=_MODEL_COLORS[model],
            )
        axis.axhline(1.0, color="#777777", linestyle=":", linewidth=1.0)
        axis.axhline(2.0, color="#222222", linestyle="--", linewidth=1.0)
        axis.set_xticks(
            np.arange(len(_MODELS)),
            ("P0", "gB1", "B1", "C1"),
        )
        axis.set_title(_FAMILY_LABELS[family], loc="left", fontsize=11)
        axis.grid(True, axis="y", color="#d9dee5", linewidth=0.7)
    axes[0, 0].set_ylabel("remainder exponent")
    axes[1, 0].set_ylabel("remainder exponent")
    fig.suptitle("Observed finite-field order on each reported fit interval", fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, output / "model_remainder_orders")


def _plot_e1(
    audit: dict[str, Any], output: Path
) -> list[Path]:
    members = sorted(
        audit["members"],
        key=lambda row: (row["system"], row["basis"], float(row["bond_scale"])),
    )
    level4 = [float(row["level4_relative_residual"]) for row in members]
    level5 = [float(row["level5_relative_residual"]) for row in members]
    fig, axis = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(members))
    axis.semilogy(x, level4, "o", color="#d36b27", markersize=4, label="grid level 4")
    axis.semilogy(
        x,
        level5,
        "s",
        markerfacecolor="white",
        markeredgecolor="#245b8a",
        markersize=4,
        label="grid level 5",
    )
    axis.set_xlabel("authenticated system / basis / distance member")
    axis.set_ylabel("relative central-dipole quadrature residual")
    axis.set_title("Independent real-space qualification of the uniform-electric E1 tensor")
    axis.grid(True, which="both", color="#d9dee5", linewidth=0.7)
    axis.legend(frameon=False)
    fig.tight_layout()
    return _save_figure(fig, output / "electric_e1_quadrature")


def _report(summary: dict[str, Any]) -> str:
    checks = summary["checks"]
    return "\n".join(
        [
            "# WP4 first-order hierarchy qualification",
            "",
            "Status: **authenticated derived evidence; user review pending**",
            "",
            "The analytic magnetic coefficients and electric E1 tensor are evaluated from ",
            "the same exact Wilson parent and level-4 grids used by the authenticated WP3 source ",
            "campaign. Exact endpoint transport is common to P0, gB1, B1, and C1; only the ",
            "endpoint-removed internal amplitudes are compared here.",
            "",
            "## Numerical result",
            "",
            "- Numerical G4 criteria: "
            f"**{'pass' if checks['g4_numerical_criteria_pass'] else 'fail'}**",
            f"- Magnetic derivative fits: {checks['derivative_resolved_pass']} passed, "
            f"{checks['derivative_resolved_fail']} failed, "
            f"{checks['derivative_exact_linear']} exact-linear to roundoff",
            f"- Remainder fits: {checks['remainder_resolved_pass']} passed, "
            f"{checks['remainder_resolved_fail']} failed",
            f"- Maximum coefficient-adjoint residual: {checks['maximum_adjoint_residual']:.3e}",
            f"- Maximum named-sector identity residual: {checks['maximum_sector_residual']:.3e}",
            f"- Maximum direct E1-parent residual: {checks['maximum_e1_parent_residual']:.3e}",
            f"- Maximum E1 position-quadrature relative residual: "
            f"level 4 = {checks['maximum_e1_level4_relative_residual']:.3e}, "
            f"level 5 = {checks['maximum_e1_level5_relative_residual']:.3e}",
            f"- Maximum level-5 residual/refined-floor ratio: "
            f"{checks['maximum_e1_level5_residual_to_refined_floor']:.3f}",
            "",
            "Every claim of first- or second-order behavior is accompanied by the actual ",
            "field interval and number of above-floor points in `remainder_fits.csv`. Channels ",
            "below the refined floor are explicitly unresolved, not counted as passes. In the ",
            "static electric-free sector C1 equals full B1 exactly; the independent electric ",
            "campaign verifies E1 against position quadrature and direct source differences.",
            "",
            "## Artifacts",
            "",
            "- `tables/magnetic_derivative_fits.csv`",
            "- `tables/remainder_fits.csv`",
            "- `tables/model_validity.csv`",
            "- `tables/mechanical_channel_classification.csv`",
            "- `tables/structural_checks.csv`",
            "- `tables/electric_e1.csv`",
            "- `figures/magnetic_derivative_convergence.{png,pdf}`",
            "- `figures/model_remainder_orders.{png,pdf}`",
            "- `figures/electric_e1_quadrature.{png,pdf}`",
            "",
            "Passing numerical criteria does not itself accept gate G4. Only an explicit user ",
            "review decision may create that record.",
            "",
        ]
    )


def main() -> None:
    execution = _arguments().execution_directory.resolve()
    members = _members(execution)
    e1_audit = _e1_refinement_audit(execution)
    derivative_rows = _derivative_rows(members)
    remainder_rows, validity_rows, channel_rows = _remainder_and_validity_rows(members)
    structural_rows, electric_rows = _structural_rows(members)
    output = execution / "analysis"
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    table_rows = {
        "magnetic_derivative_fits.csv": derivative_rows,
        "remainder_fits.csv": remainder_rows,
        "model_validity.csv": validity_rows,
        "mechanical_channel_classification.csv": channel_rows,
        "structural_checks.csv": structural_rows,
        "electric_e1.csv": electric_rows,
    }
    table_paths = []
    for name, rows in table_rows.items():
        path = tables / name
        _write_csv(path, rows)
        table_paths.append(path)
    figure_paths = [
        *_plot_derivatives(derivative_rows, figures),
        *_plot_remainders(remainder_rows, figures),
        *_plot_e1(e1_audit, figures),
    ]

    derivative_pass = sum(row["status"] == "resolved_pass" for row in derivative_rows)
    derivative_fail = sum(row["status"] == "resolved_fail" for row in derivative_rows)
    derivative_exact = sum(
        row["status"] == "exact_linear_to_roundoff" for row in derivative_rows
    )
    remainder_pass = sum(row["status"] == "resolved_pass" for row in remainder_rows)
    remainder_fail = sum(row["status"] == "resolved_fail" for row in remainder_rows)
    checks = {
        "derivative_resolved_pass": derivative_pass,
        "derivative_resolved_fail": derivative_fail,
        "derivative_exact_linear": derivative_exact,
        "derivative_below_cancellation_floor": sum(
            row["status"] == "below_cancellation_floor" for row in derivative_rows
        ),
        "remainder_resolved_pass": remainder_pass,
        "remainder_resolved_fail": remainder_fail,
        "remainder_below_floor": sum(
            row["status"] == "remainder_below_floor" for row in remainder_rows
        ),
        "linear_coefficient_unresolved": sum(
            row["status"] == "linear_coefficient_unresolved" for row in remainder_rows
        ),
        "maximum_adjoint_residual": max(
            float(row["residual"])
            for row in structural_rows
            if row["check"] == "coefficient_adjoint"
        ),
        "maximum_sector_residual": max(
            float(row["residual"])
            for row in structural_rows
            if row["check"] != "coefficient_adjoint"
        ),
        "maximum_e1_parent_residual": max(
            float(row["finite_difference_error_vs_quadrature"]) for row in electric_rows
        ),
        "maximum_e1_level4_relative_residual": float(
            e1_audit["maximum_level4_relative_residual"]
        ),
        "maximum_e1_level5_relative_residual": float(
            e1_audit["maximum_level5_relative_residual"]
        ),
        "maximum_e1_level5_to_level4_residual_ratio": float(
            e1_audit["maximum_level5_to_level4_residual_ratio"]
        ),
        "maximum_e1_level5_residual_to_refined_floor": float(
            e1_audit["maximum_level5_residual_to_refined_floor"]
        ),
        "mechanical_b1_required_channel_count": sum(
            row["status"] == "mechanical_b1_required" for row in channel_rows
        ),
    }
    checks["g4_numerical_criteria_pass"] = bool(
        derivative_pass > 0
        and derivative_fail == 0
        and remainder_pass > 0
        and remainder_fail == 0
        and checks["maximum_adjoint_residual"] <= _ADJOINT_TOLERANCE
        and checks["maximum_sector_residual"] <= _SECTOR_TOLERANCE
        and checks["maximum_e1_parent_residual"] <= _E1_PARENT_TOLERANCE
        and checks["maximum_e1_level5_residual_to_refined_floor"] <= 1.0 + 1.0e-12
    )

    manifest_core = {
        "schema": "aion.exact-one-electron-wp4-first-order-analysis-manifest",
        "version": "1.0.0",
        "status": "derived_executed_unreviewed",
        "execution_directory": str(execution),
        "execution_index_sha256": _sha256(execution / "execution_index.json"),
        "e1_refinement_audit_result_sha256": _sha256(
            execution / "e1_refinement_audit/result.json"
        ),
        "analysis_runner_sha256": _sha256(Path(__file__).resolve()),
        "rules": {
            "fit_fields_au": list(_FIT_FIELDS),
            "floor_multiplier": _FLOOR_MULTIPLIER,
            "power_tolerance": _POWER_TOLERANCE,
            "adjoint_tolerance": _ADJOINT_TOLERANCE,
            "sector_tolerance": _SECTOR_TOLERANCE,
            "e1_parent_tolerance": _E1_PARENT_TOLERANCE,
            "e1_refined_floor_definition": (
                "max(level4_vs_analytic,level5_vs_analytic,level5_vs_level4,roundoff)"
            ),
            "validity_thresholds": list(_VALIDITY_THRESHOLDS),
        },
    }
    manifest_id = canonical_sha256(manifest_core)
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, {**manifest_core, "manifest_id": manifest_id})
    summary = {
        "schema": "aion.exact-one-electron-wp4-first-order-analysis",
        "version": "1.0.0",
        "status": "derived_executed_unreviewed",
        "manifest_id": manifest_id,
        "coverage": {
            "members": len(members),
            "systems": sorted({member["system"] for member in members}),
            "bases": sorted({member["basis"] for member in members}),
            "bond_scales": sorted({float(member["bond_scale"]) for member in members}),
        },
        "checks": checks,
        "artifacts": {
            "tables": {path.name: _sha256(path) for path in table_paths},
            "figures": {path.name: _sha256(path) for path in figure_paths},
        },
        "review": {"reviewed": False, "reviewer": None, "decision": "pending"},
    }
    summary_path = output / "summary.json"
    _write_json(summary_path, summary)
    report_path = output / "report.md"
    report_path.write_text(_report(summary), encoding="utf-8")
    _write_json(
        output / "completed.json",
        {
            "status": "authenticated_derived_executed_unreviewed",
            "manifest_id": manifest_id,
            "manifest_sha256": _sha256(manifest_path),
            "summary_sha256": _sha256(summary_path),
            "report_sha256": _sha256(report_path),
            "g4_numerical_criteria_pass": checks["g4_numerical_criteria_pass"],
        },
    )
    print(f"analysis={output}")
    print(f"g4_numerical_criteria_pass={checks['g4_numerical_criteria_pass']}")
    print(f"derivative_resolved={derivative_pass} failed={derivative_fail}")
    print(f"remainder_resolved={remainder_pass} failed={remainder_fail}")


if __name__ == "__main__":
    main()
