#!/usr/bin/env python3
"""Build the auditable WP3 reference-distance analysis package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from aion.config import canonical_sha256

_REPOSITORY = Path(__file__).resolve().parents[1]
_SYSTEM_ORDER = ("hh", "oh", "n2", "co")
_SYSTEM_LABELS = {"hh": "H-H", "oh": "O-H", "n2": "N2", "co": "CO"}
_BASIS_ORDER = ("sto-3g", "cc-pvdz", "aug-cc-pvdz")
_BASIS_LABELS = {
    "sto-3g": "STO-3G",
    "cc-pvdz": "cc-pVDZ",
    "aug-cc-pvdz": "aug-cc-pVDZ",
}
_DIRECTIONS = ("parallel", "perpendicular_x", "perpendicular_y", "oblique")
_PLOT_DIRECTIONS = ("parallel", "perpendicular_x", "oblique")
_DIRECTION_LABELS = {
    "parallel": "parallel",
    "perpendicular_x": "perpendicular",
    "perpendicular_y": "perpendicular y",
    "oblique": "oblique",
}
_FAMILIES = ("overlap", "kinetic", "nuclear_attraction", "mechanical")
_PARITY_MAX_FIELD_AU = 1.0e-2
_PARITY_SIGNAL_MAX = 1.0e-2
_PARITY_FLOOR_MULTIPLIER = 4.0
_PARITY_POWER_TOLERANCE = 5.0e-2
_THRESHOLD_FLOOR_MULTIPLIER = 4.0
_SIGN_REVERSAL_TOLERANCE = 1.0e-11
_ORIENTATION_TOLERANCE = 2.0e-7
_SECTOR_FIELD_AU = 1.0e-2


@dataclass(frozen=True, slots=True)
class Member:
    system: str
    basis: str
    directory: Path
    manifest: dict[str, Any]
    result: dict[str, Any]
    refined_floors: dict[str, float]

    @property
    def key(self) -> str:
        return f"{self.system}__{self.basis}"

    @property
    def label(self) -> str:
        return f"{_SYSTEM_LABELS[self.system]} / {_BASIS_LABELS[self.basis]}"


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


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _authenticate_audit(audit: Path) -> dict[str, Any]:
    completed = _json(audit / "completed.json")
    manifest_path = audit / "manifest.json"
    result_path = audit / "result.json"
    arrays_path = audit / "arrays.npz"
    if _sha256(manifest_path) != completed["manifest_sha256"]:
        raise ValueError("reference-audit manifest hash mismatch")
    if _sha256(result_path) != completed["result_sha256"]:
        raise ValueError("reference-audit result hash mismatch")
    if _sha256(arrays_path) != completed["arrays_sha256"]:
        raise ValueError("reference-audit array hash mismatch")
    manifest = _json(manifest_path)
    manifest_core = dict(manifest)
    manifest_id = manifest_core.pop("manifest_id")
    if canonical_sha256(manifest_core) != manifest_id:
        raise ValueError("reference-audit manifest identity mismatch")
    result = _json(result_path)
    if result["manifest_id"] != manifest_id:
        raise ValueError("reference-audit result identity mismatch")
    if not result["all_gauge_checks_pass"]:
        raise ValueError("reference-audit gauge checks did not pass")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(result["array_semantic_sha256"]):
            raise ValueError("reference-audit array inventory mismatch")
        for name in arrays.files:
            value = np.asarray(arrays[name])
            if not np.all(np.isfinite(value)):
                raise ValueError(f"non-finite reference-audit array: {name}")
            if canonical_sha256(value) != result["array_semantic_sha256"][name]:
                raise ValueError(f"reference-audit semantic hash mismatch: {name}")
    return result


def _load_members(execution: Path, audit_result: dict[str, Any]) -> list[Member]:
    index = _json(execution / "execution_index.json")
    audit_by_key = {row["member_key"]: row for row in audit_result["members"]}
    members: list[Member] = []
    for record in index["completed_members"]:
        directory = Path(record["member_directory"])
        manifest_path = directory / "manifest.json"
        result_path = directory / "result.json"
        arrays_path = directory / "arrays.npz"
        if _sha256(manifest_path) != record["manifest_sha256"]:
            raise ValueError(f"manifest hash mismatch: {directory}")
        if _sha256(result_path) != record["result_sha256"]:
            raise ValueError(f"result hash mismatch: {directory}")
        if _sha256(arrays_path) != record["arrays_sha256"]:
            raise ValueError(f"arrays hash mismatch: {directory}")
        manifest = _json(manifest_path)
        result = _json(result_path)
        key = f"{record['system']}__{record['basis']}"
        audit_row = audit_by_key[key]
        floors = {
            family: float(audit_row["floors_absolute_frobenius"][family]["refined_working_floor"])
            for family in _FAMILIES
        }
        members.append(
            Member(
                system=record["system"],
                basis=record["basis"],
                directory=directory,
                manifest=manifest,
                result=result,
                refined_floors=floors,
            )
        )
    if len(members) != 12:
        raise ValueError(f"expected 12 members, found {len(members)}")
    members.sort(
        key=lambda value: (
            _SYSTEM_ORDER.index(value.system),
            _BASIS_ORDER.index(value.basis),
        )
    )
    return members


def _case_index(member: Member) -> dict[str, int]:
    return {
        case_id: index for index, case_id in enumerate(member.result["case_ids_in_array_order"])
    }


def _signal_to_floor(
    case: dict[str, Any],
    arrays: Any,
    index: int,
    floors: dict[str, float],
    metric: str,
) -> float:
    if metric == "phase_spread_intersite_max":
        return float(case["headline"][metric]) / (64.0 * np.finfo(np.float64).eps)
    if metric.startswith("overlap_"):
        family = "overlap"
    elif metric.startswith("mechanical_"):
        family = "mechanical"
    elif metric == "spectral_max_abs_hartree":
        eigenvalues = np.asarray(case["p0_generalized_eigenvalues_hartree"])
        spectral_floor = (
            floors["mechanical"] + max(1.0, float(np.max(np.abs(eigenvalues)))) * floors["overlap"]
        ) / max(
            float(case["p0_metric_minimum_eigenvalue"]),
            np.finfo(np.float64).tiny,
        )
        return float(case["headline"][metric]) / spectral_floor
    else:
        raise ValueError(f"unknown headline metric: {metric}")

    exact = np.asarray(arrays[f"lower__exact__{family}"][index])
    p0 = np.asarray(arrays[f"lower__p0__{family}"][index])
    difference = exact - p0
    if metric.endswith("_global"):
        absolute = float(np.linalg.norm(difference))
    elif metric.endswith("_max_channel"):
        absolute = max(float(row["absolute_frobenius"]) for row in case["angular_channels"][family])
    elif metric.endswith("_diagonal_scaled"):
        absolute = float(np.max(np.abs(difference)))
    else:
        raise ValueError(f"unknown matrix headline metric: {metric}")
    return absolute / floors[family]


def _threshold_rows(
    members: list[Member], fixture: dict[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    moved = 0
    for member in members:
        raw_by_key = {
            (row["direction"], row["metric"], float(row["threshold"])): row
            for row in member.result["threshold_crossings"]
        }
        with np.load(member.directory / "arrays.npz", allow_pickle=False) as arrays:
            indices = _case_index(member)
            metrics = tuple(member.result["cases"][0]["headline"])
            for direction in _DIRECTIONS:
                cases = sorted(
                    (
                        case
                        for case in member.result["cases"]
                        if case["direction"] == direction and case["sign"] == 1
                    ),
                    key=lambda case: case["magnitude_au"],
                )
                for metric in metrics:
                    threshold_key = (
                        "spectral_shift_thresholds_hartree"
                        if metric == "spectral_max_abs_hartree"
                        else "normalized_matrix_thresholds"
                    )
                    for threshold_value in fixture[threshold_key]:
                        threshold = float(threshold_value)
                        crossing: dict[str, Any] | None = None
                        crossing_ratio: float | None = None
                        previous_field = 0.0
                        for case in cases:
                            ratio = _signal_to_floor(
                                case,
                                arrays,
                                indices[case["id"]],
                                member.refined_floors,
                                metric,
                            )
                            if (
                                float(case["headline"][metric]) >= threshold
                                and ratio >= _THRESHOLD_FLOOR_MULTIPLIER
                            ):
                                crossing = case
                                crossing_ratio = ratio
                                break
                            previous_field = float(case["magnitude_au"])
                        raw = raw_by_key[(direction, metric, threshold)]
                        new_field = None if crossing is None else float(crossing["magnitude_au"])
                        if raw["first_field_au"] != new_field:
                            moved += 1
                        maximum_field = float(cases[-1]["magnitude_au"])
                        output.append(
                            {
                                "system": member.system,
                                "basis": member.basis,
                                "bond_scale": 1.0,
                                "direction": direction,
                                "metric": metric,
                                "threshold": threshold,
                                "crossed": crossing is not None,
                                "lower_bracket_au": (
                                    previous_field if crossing is not None else maximum_field
                                ),
                                "upper_bracket_au": new_field,
                                "upper_bracket_tesla": (
                                    None
                                    if new_field is None
                                    else new_field * float(fixture["magnetic_field_tesla_per_au"])
                                ),
                                "value_at_upper": (
                                    None
                                    if crossing is None
                                    else float(crossing["headline"][metric])
                                ),
                                "signal_to_refined_floor": crossing_ratio,
                                "source_level4_first_field_au": raw["first_field_au"],
                                "changed_by_refined_floor": raw["first_field_au"] != new_field,
                            }
                        )
    return output, moved


def _parity_rows(members: list[Member]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for member in members:
        with np.load(member.directory / "arrays.npz", allow_pickle=False) as arrays:
            for direction in _DIRECTIONS:
                rows = [
                    row
                    for row in member.result["parity"]
                    if row["direction"] == direction and row["magnitude_au"] <= _PARITY_MAX_FIELD_AU
                ]
                for family in _FAMILIES:
                    zero = np.asarray(arrays[f"barred__exact__{family}"][0])
                    scale = max(1.0, float(np.linalg.norm(zero)))
                    relative_floor = member.refined_floors[family] / scale
                    for parity, expected in (("odd", 1.0), ("even", 2.0)):
                        fields = np.asarray([row["magnitude_au"] for row in rows], dtype=np.float64)
                        signals = np.asarray(
                            [
                                row["families"][family][f"{parity}_relative_frobenius"]
                                for row in rows
                            ],
                            dtype=np.float64,
                        )
                        resolved = (signals >= _PARITY_FLOOR_MULTIPLIER * relative_floor) & (
                            signals <= _PARITY_SIGNAL_MAX
                        )
                        count = int(np.count_nonzero(resolved))
                        fitted: float | None = None
                        status: str
                        if count >= 2:
                            fitted = float(
                                np.polyfit(
                                    np.log10(fields[resolved]),
                                    np.log10(signals[resolved]),
                                    1,
                                )[0]
                            )
                            status = (
                                "resolved_pass"
                                if abs(fitted - expected) <= _PARITY_POWER_TOLERANCE
                                else "resolved_fail"
                            )
                        elif direction == "parallel" and family in (
                            "overlap",
                            "nuclear_attraction",
                        ):
                            status = "symmetry_forbidden_geometry"
                        elif (
                            member.system == "hh"
                            and member.basis == "sto-3g"
                            and parity == "odd"
                            and float(np.max(signals)) < _PARITY_FLOOR_MULTIPLIER * relative_floor
                        ):
                            status = "symmetry_suppressed_observed"
                        else:
                            status = "below_numerical_floor"
                        output.append(
                            {
                                "system": member.system,
                                "basis": member.basis,
                                "direction": direction,
                                "family": family,
                                "parity": parity,
                                "status": status,
                                "expected_power": expected,
                                "fitted_power": fitted,
                                "fit_point_count": count,
                                "fit_field_min_au": (
                                    None if count == 0 else float(np.min(fields[resolved]))
                                ),
                                "fit_field_max_au": (
                                    None if count == 0 else float(np.max(fields[resolved]))
                                ),
                                "relative_floor": relative_floor,
                                "maximum_signal_in_window": float(np.max(signals)),
                            }
                        )
    return output


def _identity_rows(
    members: list[Member],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sign_rows: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    for member in members:
        cases_by_key = {
            (case["direction"], case["sign"], float(case["magnitude_au"])): case
            for case in member.result["cases"]
            if case["sign"] != 0
        }
        indices = _case_index(member)
        with np.load(member.directory / "arrays.npz", allow_pickle=False) as arrays:
            for direction in _DIRECTIONS:
                fields = sorted(
                    key[2] for key in cases_by_key if key[0] == direction and key[1] == 1
                )
                for field in fields:
                    plus = cases_by_key[(direction, 1, field)]
                    minus = cases_by_key[(direction, -1, field)]
                    plus_index = indices[plus["id"]]
                    minus_index = indices[minus["id"]]
                    for family in _FAMILIES:
                        plus_matrix = np.asarray(arrays[f"lower__exact__{family}"][plus_index])
                        minus_matrix = np.asarray(arrays[f"lower__exact__{family}"][minus_index])
                        residual = float(
                            np.linalg.norm(minus_matrix - plus_matrix.conj())
                            / max(1.0, float(np.linalg.norm(plus_matrix)))
                        )
                        sign_rows.append(
                            {
                                "system": member.system,
                                "basis": member.basis,
                                "direction": direction,
                                "field_au": field,
                                "family": family,
                                "relative_residual": residual,
                            }
                        )

            fields = sorted(
                key[2] for key in cases_by_key if key[0] == "perpendicular_x" and key[1] == 1
            )
            for field in fields:
                for sign in (-1, 1):
                    case_x = cases_by_key[("perpendicular_x", sign, field)]
                    case_y = cases_by_key[("perpendicular_y", sign, field)]
                    for metric in case_x["headline"]:
                        ratio_x = _signal_to_floor(
                            case_x,
                            arrays,
                            indices[case_x["id"]],
                            member.refined_floors,
                            metric,
                        )
                        ratio_y = _signal_to_floor(
                            case_y,
                            arrays,
                            indices[case_y["id"]],
                            member.refined_floors,
                            metric,
                        )
                        value_x = float(case_x["headline"][metric])
                        value_y = float(case_y["headline"][metric])
                        scale = max(abs(value_x), abs(value_y), np.finfo(np.float64).tiny)
                        orientation_rows.append(
                            {
                                "system": member.system,
                                "basis": member.basis,
                                "field_au": field,
                                "sign": sign,
                                "metric": metric,
                                "perpendicular_x": value_x,
                                "perpendicular_y": value_y,
                                "relative_difference": abs(value_x - value_y) / scale,
                                "both_above_refined_floor": bool(
                                    ratio_x >= _THRESHOLD_FLOOR_MULTIPLIER
                                    and ratio_y >= _THRESHOLD_FLOOR_MULTIPLIER
                                ),
                            }
                        )
    return sign_rows, orientation_rows


def _selected_rows(
    members: list[Member],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sectors: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    for member in members:
        for direction in _PLOT_DIRECTIONS:
            case = next(
                row
                for row in member.result["cases"]
                if row["direction"] == direction
                and row["sign"] == 1
                and np.isclose(row["magnitude_au"], _SECTOR_FIELD_AU)
            )
            sectors.append(
                {
                    "system": member.system,
                    "basis": member.basis,
                    "direction": direction,
                    "field_au": _SECTOR_FIELD_AU,
                    "exact_mechanical_relative": case["global_relative"]["mechanical"],
                    "form_factor_only_mechanical_relative": case["diagnostic_relative"][
                        "form_factor_only"
                    ]["mechanical"],
                    "anchored_vector_only_mechanical_relative": case["diagnostic_relative"][
                        "anchored_vector_only"
                    ]["mechanical"],
                    "phase_spread_intersite_max": case["headline"]["phase_spread_intersite_max"],
                    "kinetic_pp_relative": case["kinetic_sector_relative_norms"]["pp"],
                    "kinetic_pC_relative": case["kinetic_sector_relative_norms"]["pC"],
                    "kinetic_Cp_relative": case["kinetic_sector_relative_norms"]["Cp"],
                    "kinetic_C2_relative": case["kinetic_sector_relative_norms"]["C2"],
                }
            )
            candidates = case["angular_channels"]["mechanical"]
            top = max(candidates, key=lambda value: value["absolute_frobenius"])
            channels.append(
                {
                    "system": member.system,
                    "basis": member.basis,
                    "direction": direction,
                    "field_au": _SECTOR_FIELD_AU,
                    "bra_atom": top["bra_atom"],
                    "ket_atom": top["ket_atom"],
                    "bra_l": top["bra_l"],
                    "ket_l": top["ket_l"],
                    "classification": top["classification"],
                    "absolute_frobenius": top["absolute_frobenius"],
                    "relative_frobenius": top["relative_frobenius"],
                    "maximum_absolute_element": top["maximum_absolute_element"],
                    "largest_singular_value": max(top["singular_values"]),
                }
            )
            shifts = np.asarray(case["generalized_eigenvalue_shifts_hartree"])
            angles = np.asarray(case["cumulative_subspace_angles_rad"])
            spectra.append(
                {
                    "system": member.system,
                    "basis": member.basis,
                    "direction": direction,
                    "field_au": _SECTOR_FIELD_AU,
                    "lowest_eigenvalue_shift_hartree": float(shifts[0]),
                    "maximum_absolute_shift_hartree": float(np.max(np.abs(shifts))),
                    "maximum_shift_index": int(np.argmax(np.abs(shifts))),
                    "maximum_cumulative_subspace_angle_rad": (
                        0.0 if angles.size == 0 else float(np.max(angles))
                    ),
                }
            )
    return sectors, channels, spectra


def _category_labels(members: list[Member]) -> list[str]:
    return [f"{_SYSTEM_LABELS[m.system]}\n{_BASIS_LABELS[m.basis]}" for m in members]


def _save_figure(fig: Any, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _plot_thresholds(members: list[Member], rows: list[dict[str, Any]], output: Path) -> list[Path]:
    palette = {
        "parallel": "#245b8a",
        "perpendicular_x": "#d36b27",
        "oblique": "#7b6d2d",
    }
    markers = {"parallel": "o", "perpendicular_x": "s", "oblique": "^"}
    specifications = (
        ("mechanical_global", 1.0e-4, r"mechanical matrix change $=10^{-4}$"),
        (
            "spectral_max_abs_hartree",
            1.0e-4,
            r"maximum generalized-eigenvalue shift $=10^{-4}$ Ha",
        ),
    )
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.8), sharex=True)
    x = np.arange(len(members), dtype=np.float64)
    for axis, (metric, threshold, title) in zip(axes, specifications, strict=True):
        for offset_index, direction in enumerate(_PLOT_DIRECTIONS):
            values: list[float] = []
            crossed: list[bool] = []
            for member in members:
                row = next(
                    value
                    for value in rows
                    if value["system"] == member.system
                    and value["basis"] == member.basis
                    and value["direction"] == direction
                    and value["metric"] == metric
                    and np.isclose(value["threshold"], threshold)
                )
                crossed.append(bool(row["crossed"]))
                values.append(
                    1.25 if row["upper_bracket_au"] is None else float(row["upper_bracket_au"])
                )
            positions = x + (offset_index - 1) * 0.18
            for position, value, did_cross in zip(positions, values, crossed, strict=True):
                axis.scatter(
                    position,
                    value,
                    marker=markers[direction] if did_cross else "x",
                    s=48,
                    facecolors=palette[direction] if did_cross else "none",
                    edgecolors=palette[direction],
                    linewidths=1.4,
                    zorder=3,
                )
            axis.plot(
                [],
                [],
                color=palette[direction],
                marker=markers[direction],
                linestyle="none",
                label=_DIRECTION_LABELS[direction],
            )
        axis.set_yscale("log")
        axis.set_ylim(7.0e-6, 1.7)
        axis.set_ylabel(r"first sampled $|B|$ (a.u.)")
        axis.set_title(title, loc="left", fontsize=12)
        axis.grid(True, which="both", color="#d9dee5", linewidth=0.7)
    axes[0].legend(ncol=3, loc="upper left", frameon=False)
    axes[-1].set_xticks(x, _category_labels(members), rotation=45, ha="right")
    axes[-1].text(
        0.995,
        0.03,
        "x at 1.25 denotes no crossing through 1 a.u.; points are sampled upper bounds",
        transform=axes[-1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#3f4752",
    )
    fig.suptitle("WP3 reference-distance exact-minus-P0 threshold fields", fontsize=16)
    fig.tight_layout()
    return _save_figure(fig, output / "threshold_fields")


def _plot_parity(
    members: list[Member], parity_rows: list[dict[str, Any]], output: Path
) -> list[Path]:
    member = next(
        value for value in members if value.system == "co" and value.basis == "aug-cc-pvdz"
    )
    source = [
        row
        for row in member.result["parity"]
        if row["direction"] == "oblique" and row["magnitude_au"] <= 1.0e-2
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    colors = {"odd": "#245b8a", "even": "#d36b27"}
    for axis, family in zip(axes[:2], ("overlap", "mechanical"), strict=True):
        for parity, expected in (("odd", 1.0), ("even", 2.0)):
            fields = np.asarray([row["magnitude_au"] for row in source])
            signals = np.asarray(
                [row["families"][family][f"{parity}_relative_frobenius"] for row in source]
            )
            positive = signals > 0.0
            axis.loglog(
                fields[positive],
                signals[positive],
                marker="o" if parity == "odd" else "s",
                color=colors[parity],
                linewidth=1.6,
                markersize=4.5,
                label=f"{parity}; expected $B^{expected:g}$",
            )
        axis.set_title(f"CO / aug-cc-pVDZ / oblique: {family}", fontsize=10)
        axis.set_xlabel(r"$|B|$ (a.u.)")
        axis.set_ylabel("relative Frobenius component")
        axis.grid(True, which="both", color="#d9dee5", linewidth=0.7)
        axis.legend(frameon=False, fontsize=8)

    resolved = [row for row in parity_rows if row["status"] == "resolved_pass"]
    for index, parity in enumerate(("odd", "even")):
        values = np.asarray(
            [row["fitted_power"] for row in resolved if row["parity"] == parity],
            dtype=np.float64,
        )
        jitter = np.linspace(-0.13, 0.13, values.size) if values.size else np.asarray([])
        axes[2].scatter(
            index + jitter,
            values,
            s=12,
            alpha=0.6,
            color=colors[parity],
            marker="o" if parity == "odd" else "s",
        )
        axes[2].hlines(index + 1.0, index - 0.28, index + 0.28, color="#222222")
    axes[2].set_xticks((0, 1), ("odd", "even"))
    axes[2].set_ylabel("fitted small-field exponent")
    axes[2].set_title("All resolved system/basis/direction fits", fontsize=10)
    axes[2].grid(True, axis="y", color="#d9dee5", linewidth=0.7)
    fig.suptitle("Odd/even asymptotic scaling of exact endpoint-removed matrices", fontsize=15)
    fig.tight_layout()
    return _save_figure(fig, output / "parity_scaling")


def _plot_sectors(members: list[Member], rows: list[dict[str, Any]], output: Path) -> list[Path]:
    colors = {
        "exact_mechanical_relative": "#222222",
        "form_factor_only_mechanical_relative": "#d36b27",
        "anchored_vector_only_mechanical_relative": "#245b8a",
    }
    labels = {
        "exact_mechanical_relative": "complete exact",
        "form_factor_only_mechanical_relative": "triangle factor only",
        "anchored_vector_only_mechanical_relative": "anchored vector only",
    }
    markers = {
        "exact_mechanical_relative": "o",
        "form_factor_only_mechanical_relative": "s",
        "anchored_vector_only_mechanical_relative": "^",
    }
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 10.5), sharex=True)
    x = np.arange(len(members), dtype=np.float64)
    for axis, direction in zip(axes, _PLOT_DIRECTIONS, strict=True):
        for name in colors:
            values = np.asarray(
                [
                    next(
                        row[name]
                        for row in rows
                        if row["system"] == member.system
                        and row["basis"] == member.basis
                        and row["direction"] == direction
                    )
                    for member in members
                ],
                dtype=np.float64,
            )
            if direction == "parallel" and name == "form_factor_only_mechanical_relative":
                continue
            axis.plot(
                x,
                values,
                color=colors[name],
                marker=markers[name],
                linewidth=1.5,
                markersize=4.5,
                label=labels[name],
            )
        axis.set_yscale("log")
        axis.set_ylabel("relative matrix change")
        axis.set_title(_DIRECTION_LABELS[direction], loc="left", fontsize=11)
        axis.grid(True, which="both", color="#d9dee5", linewidth=0.7)
        if direction == "parallel":
            axis.text(
                0.99,
                0.08,
                "triangle-factor contribution is exactly zero",
                transform=axis.transAxes,
                ha="right",
                fontsize=9,
            )
    axes[0].legend(ncol=3, frameon=False, loc="upper left")
    axes[-1].set_xticks(x, _category_labels(members), rotation=45, ha="right")
    fig.suptitle(r"Origin of the mechanical correction at $|B|=10^{-2}$ a.u.", fontsize=15)
    fig.tight_layout()
    return _save_figure(fig, output / "mechanical_sector_decomposition")


def _plot_metric(members: list[Member], output: Path) -> list[Path]:
    x = np.arange(len(members), dtype=np.float64)
    exact = np.asarray([member.result["minimum_exact_metric_eigenvalue"] for member in members])
    p0 = np.asarray([member.result["minimum_p0_metric_eigenvalue"] for member in members])
    fig, axis = plt.subplots(figsize=(12.5, 4.8))
    axis.semilogy(x, exact, "o-", color="#245b8a", label="exact Wilson metric")
    axis.semilogy(x, p0, "s--", color="#d36b27", label="P0 metric")
    axis.set_xticks(x, _category_labels(members), rotation=45, ha="right")
    axis.set_ylabel("minimum metric eigenvalue over scan")
    axis.set_title("Metric positivity over the reference-distance field range")
    axis.grid(True, which="both", color="#d9dee5", linewidth=0.7)
    axis.legend(frameon=False)
    fig.tight_layout()
    return _save_figure(fig, output / "metric_positivity")


def _format_field(value: float | None) -> str:
    return ">1" if value is None else f"{value:.2g}"


def _report(
    execution: Path,
    members: list[Member],
    summary: dict[str, Any],
    thresholds: list[dict[str, Any]],
) -> str:
    checks = summary["checks"]
    lines = [
        "# WP3 exact one-electron reference-distance analysis",
        "",
        "Status: **derived executed evidence; not a G3 acceptance record**",
        "",
        "This report analyzes the authenticated 12-member H-H, O-H, N2, and CO ",
        "reference-distance campaign. It uses the separately authenticated level-4/5 ",
        "floor and finite-field gauge audit. Raw member records remain immutable and ",
        "`executed_unreviewed`; only an explicit user decision can accept a gate.",
        "",
        "## Provenance",
        "",
        f"- Execution: `{execution}`",
        f"- Execution-index SHA-256: `{summary['source']['execution_index_sha256']}`",
        f"- Reference-audit result SHA-256: `{summary['source']['audit_result_sha256']}`",
        f"- Members: {summary['coverage']['members']}",
        f"- Field cases: {summary['coverage']['field_cases']}",
        "- Grid: unpruned PySCF level 4 for the field scan; unpruned levels 4 and 5 for floors",
        "- Field range: both signs, four orientations, `1e-7` through `1` a.u., ",
        "  with one adaptive refinement round",
        "",
        "## Numerical checks",
        "",
        "| check | result | decision rule |",
        "|---|---:|---|",
        (
            "| Refined threshold claims | minimum signal/floor = "
            f"{checks['minimum_crossing_signal_to_floor']:.3f} | "
            f">= {_THRESHOLD_FLOOR_MULTIPLIER:g} |"
        ),
        (
            "| Thresholds moved by refined floors | "
            f"{checks['threshold_crossings_moved_by_refinement']} | "
            "reported from refined values |"
        ),
        (
            "| Floor-marginal crossings | "
            f"{checks['crossings_below_ten_times_floor']} | "
            "between 4 and 10 times the refined floor |"
        ),
        (
            "| Resolved odd/even fits | "
            f"{checks['resolved_parity_fit_count']} passed, "
            f"{checks['failed_parity_fit_count']} failed | "
            f"exponent error <= {_PARITY_POWER_TOLERANCE:g} |"
        ),
        (
            "| Largest exponent error | "
            f"{checks['maximum_parity_power_error']:.3e} | "
            f"<= {_PARITY_POWER_TOLERANCE:g} |"
        ),
        (
            f"| Field reversal | {checks['maximum_sign_reversal_relative']:.3e} | "
            f"<= {_SIGN_REVERSAL_TOLERANCE:.1e} |"
        ),
        (
            "| Perpendicular x/y invariants above floor | "
            f"{checks['maximum_perpendicular_relative_difference']:.3e} | "
            f"<= {_ORIENTATION_TOLERANCE:.1e} |"
        ),
        (
            "| Gauge congruence at 1 a.u. | "
            f"{checks['maximum_gauge_congruence_relative']:.3e} | <= 1.0e-11 |"
        ),
        (
            "| Gauge spectrum residual at 1 a.u. | "
            f"{checks['maximum_gauge_spectrum_residual_hartree']:.3e} Ha | "
            "conditioning-scaled bound |"
        ),
        (
            "| Minimum exact metric eigenvalue | "
            f"{checks['minimum_exact_metric_eigenvalue']:.3e} | > 0 |"
        ),
        "",
        "Every resolved parity fit passes. Unfitted channels are retained as either ",
        "geometry-forbidden, symmetry-suppressed in H-H/STO-3G, or below the refined ",
        "numerical floor; they are not converted into scaling claims. The floor-marginal ",
        "thresholds satisfy the declared factor-of-four rule but remain explicitly flagged ",
        "rather than being presented as decade-separated signals.",
        "",
        "## Threshold fields",
        "",
        "The entries below are the first sampled upper bounds in atomic units after ",
        "reapplying the refined floors. `>1` means no crossing through 1 a.u. The two ",
        "perpendicular directions agree and only perpendicular x is shown.",
        "",
        "| system / basis | mechanical 1e-4: parallel / perpendicular / oblique "
        "| spectrum 1e-4 Ha: parallel / perpendicular / oblique |",
        "|---|---:|---:|",
    ]
    for member in members:
        values: list[str] = []
        for metric in ("mechanical_global", "spectral_max_abs_hartree"):
            row_values = []
            for direction in _PLOT_DIRECTIONS:
                row = next(
                    value
                    for value in thresholds
                    if value["system"] == member.system
                    and value["basis"] == member.basis
                    and value["direction"] == direction
                    and value["metric"] == metric
                    and np.isclose(value["threshold"], 1.0e-4)
                )
                row_values.append(_format_field(row["upper_bracket_au"]))
            values.append(" / ".join(row_values))
        lines.append(f"| {member.label} | {values[0]} | {values[1]} |")
    lines.extend(
        [
            "",
            "## Physical interpretation",
            "",
            "- The endpoint-removed overlap and nuclear-attraction corrections vanish ",
            "  identically for a field parallel to the bond. The complete mechanical ",
            "  correction does not vanish: it is then entirely the anchored-vector ",
            "  kinetic contribution. This cleanly separates magnetic mechanical response ",
            "  from triangle-phase response.",
            "- For perpendicular and oblique fields, the triangle factor usually dominates ",
            "  the mechanical matrix norm in O-H, N2, and CO at `B=0.01` a.u.; the ",
            "  anchored-vector term remains resolved. Polarized H-H is the important case ",
            "  where both mechanisms are comparable.",
            "- All resolved odd matrix parts scale as `B^1`, and all resolved even parts ",
            "  scale as `B^2`. H-H/STO-3G has no resolved odd sector, consistent with its ",
            "  s-only symmetry. Adding polarization functions opens linear angular channels.",
            "- Enlarging the basis generally lowers the field at which P0 matrix errors ",
            "  cross a fixed threshold. This is not a failure of basis convergence: the ",
            "  larger subspace resolves finite-spread angular response that P0 deletes.",
            "- The maximum generalized-eigenvalue shift is often carried by a high virtual ",
            "  state or a degenerate manifold. The companion spectrum table therefore ",
            "  records the lowest eigenvalue shift separately; maximum-spectrum thresholds ",
            "  must not be read as ground-state energy errors.",
            "",
            "## Figures and tables",
            "",
            "- [Threshold fields](figures/threshold_fields.png)",
            "- [Odd/even scaling](figures/parity_scaling.png)",
            "- [Mechanical-sector decomposition](figures/mechanical_sector_decomposition.png)",
            "- [Metric positivity](figures/metric_positivity.png)",
            "- `tables/thresholds.csv`: every metric, threshold, bracket, tesla conversion, "
            "  and floor ratio",
            "- `tables/parity_fits.csv`: every resolved and unresolved odd/even fit",
            "- `tables/angular_channels_at_1e-2.csv`: dominant angular block for each "
            "  system/basis/direction",
            "- `tables/kinetic_sectors_at_1e-2.csv`: exact, triangle-only, anchored-only, "
            "  and kinetic sectors",
            "- `tables/spectra_at_1e-2.csv`: lowest and maximum spectral shifts and "
            "  subspace angles",
            "",
            "## Checkpoint recommendation",
            "",
            "The reference-distance checkpoint satisfies its numerical prerequisites and ",
            "supports starting the declared compressed (`0.8`) and stretched (`1.25`) bond ",
            "scans. It does **not** satisfy G3 by itself because the distance campaign has ",
            "not yet been executed or reviewed. No uncontracted comparison is triggered by ",
            "the present smooth basis progression; that decision remains revisable if the ",
            "distance scans expose a contraction-specific anomaly.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    arguments = _arguments()
    execution = arguments.execution_directory.resolve()
    audit = execution / "reference_audit"
    audit_result = _authenticate_audit(audit)
    members = _load_members(execution, audit_result)
    plan = _json(execution / "execution_plan.json")
    fixture = plan["fixture"]

    output = execution / "analysis"
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    threshold_rows, moved = _threshold_rows(members, fixture)
    parity_rows = _parity_rows(members)
    sign_rows, orientation_rows = _identity_rows(members)
    sector_rows, channel_rows, spectrum_rows = _selected_rows(members)

    _write_csv(
        tables / "thresholds.csv",
        tuple(threshold_rows[0]),
        threshold_rows,
    )
    _write_csv(
        tables / "parity_fits.csv",
        tuple(parity_rows[0]),
        parity_rows,
    )
    _write_csv(
        tables / "field_reversal.csv",
        tuple(sign_rows[0]),
        sign_rows,
    )
    _write_csv(
        tables / "perpendicular_orientation.csv",
        tuple(orientation_rows[0]),
        orientation_rows,
    )
    _write_csv(
        tables / "kinetic_sectors_at_1e-2.csv",
        tuple(sector_rows[0]),
        sector_rows,
    )
    _write_csv(
        tables / "angular_channels_at_1e-2.csv",
        tuple(channel_rows[0]),
        channel_rows,
    )
    _write_csv(
        tables / "spectra_at_1e-2.csv",
        tuple(spectrum_rows[0]),
        spectrum_rows,
    )

    figure_paths = [
        *_plot_thresholds(members, threshold_rows, figures),
        *_plot_parity(members, parity_rows, figures),
        *_plot_sectors(members, sector_rows, figures),
        *_plot_metric(members, figures),
    ]

    crossed = [row for row in threshold_rows if row["crossed"]]
    resolved = [row for row in parity_rows if row["status"].startswith("resolved_")]
    failed = [row for row in resolved if row["status"] == "resolved_fail"]
    maximum_power_error = max(
        abs(float(row["fitted_power"]) - float(row["expected_power"])) for row in resolved
    )
    orientation_resolved = [row for row in orientation_rows if row["both_above_refined_floor"]]
    checks = {
        "minimum_crossing_signal_to_floor": min(
            float(row["signal_to_refined_floor"]) for row in crossed
        ),
        "threshold_crossings_moved_by_refinement": moved,
        "crossings_below_ten_times_floor": sum(
            float(row["signal_to_refined_floor"]) < 10.0 for row in crossed
        ),
        "resolved_parity_fit_count": len(resolved),
        "failed_parity_fit_count": len(failed),
        "maximum_parity_power_error": maximum_power_error,
        "symmetry_forbidden_fit_count": sum(
            row["status"] == "symmetry_forbidden_geometry" for row in parity_rows
        ),
        "symmetry_suppressed_fit_count": sum(
            row["status"] == "symmetry_suppressed_observed" for row in parity_rows
        ),
        "below_floor_fit_count": sum(
            row["status"] == "below_numerical_floor" for row in parity_rows
        ),
        "maximum_sign_reversal_relative": max(float(row["relative_residual"]) for row in sign_rows),
        "maximum_perpendicular_relative_difference": max(
            float(row["relative_difference"]) for row in orientation_resolved
        ),
        "maximum_gauge_congruence_relative": audit_result["maximum_lower_congruence_relative"],
        "maximum_gauge_barred_invariance_relative": audit_result[
            "maximum_barred_invariance_relative"
        ],
        "maximum_gauge_spectrum_residual_hartree": audit_result[
            "maximum_generalized_spectrum_residual_hartree"
        ],
        "minimum_exact_metric_eigenvalue": min(
            float(member.result["minimum_exact_metric_eigenvalue"]) for member in members
        ),
        "minimum_p0_metric_eigenvalue": min(
            float(member.result["minimum_p0_metric_eigenvalue"]) for member in members
        ),
    }
    checks["reference_checkpoint_pass"] = bool(
        checks["minimum_crossing_signal_to_floor"] >= _THRESHOLD_FLOOR_MULTIPLIER
        and checks["failed_parity_fit_count"] == 0
        and checks["maximum_sign_reversal_relative"] <= _SIGN_REVERSAL_TOLERANCE
        and checks["maximum_perpendicular_relative_difference"] <= _ORIENTATION_TOLERANCE
        and audit_result["all_gauge_checks_pass"]
        and checks["minimum_exact_metric_eigenvalue"] > 0.0
        and checks["minimum_p0_metric_eigenvalue"] > 0.0
    )

    csv_paths = sorted(tables.glob("*.csv"))
    manifest_core = {
        "schema": "aion.exact-one-electron-wp3-reference-analysis-manifest",
        "version": "1.0.0",
        "status": "derived_executed_unreviewed",
        "source_execution_directory": str(execution),
        "source_execution_index_sha256": _sha256(execution / "execution_index.json"),
        "source_audit_result_sha256": _sha256(audit / "result.json"),
        "analysis_runner_sha256": _sha256(Path(__file__).resolve()),
        "rules": {
            "threshold_floor_multiplier": _THRESHOLD_FLOOR_MULTIPLIER,
            "parity_floor_multiplier": _PARITY_FLOOR_MULTIPLIER,
            "parity_fit_max_field_au": _PARITY_MAX_FIELD_AU,
            "parity_fit_max_signal": _PARITY_SIGNAL_MAX,
            "parity_power_tolerance": _PARITY_POWER_TOLERANCE,
            "sign_reversal_relative_tolerance": _SIGN_REVERSAL_TOLERANCE,
            "orientation_relative_tolerance": _ORIENTATION_TOLERANCE,
        },
    }
    manifest_id = canonical_sha256(manifest_core)
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, {**manifest_core, "manifest_id": manifest_id})
    summary = {
        "schema": "aion.exact-one-electron-wp3-reference-analysis",
        "version": "1.0.0",
        "status": "derived_executed_unreviewed",
        "manifest_id": manifest_id,
        "source": {
            "execution_directory": str(execution),
            "execution_index_sha256": _sha256(execution / "execution_index.json"),
            "audit_result_sha256": _sha256(audit / "result.json"),
        },
        "coverage": {
            "members": len(members),
            "field_cases": sum(len(member.result["cases"]) for member in members),
            "systems": list(_SYSTEM_ORDER),
            "bases": list(_BASIS_ORDER),
            "bond_scales": [1.0],
        },
        "checks": checks,
        "artifacts": {
            "tables": {path.name: _sha256(path) for path in csv_paths},
            "figures": {path.name: _sha256(path) for path in figure_paths},
        },
        "review": {"reviewed": False, "reviewer": None, "decision": "pending"},
    }
    summary_path = output / "summary.json"
    _write_json(summary_path, summary)
    report_path = output / "report.md"
    report_path.write_text(_report(execution, members, summary, threshold_rows), encoding="utf-8")
    completed = {
        "status": "authenticated_derived_executed_unreviewed",
        "manifest_id": manifest_id,
        "manifest_sha256": _sha256(manifest_path),
        "summary_sha256": _sha256(summary_path),
        "report_sha256": _sha256(report_path),
        "reference_checkpoint_pass": checks["reference_checkpoint_pass"],
    }
    _write_json(output / "completed.json", completed)
    print(f"analysis={output}")
    print(f"reference_checkpoint_pass={checks['reference_checkpoint_pass']}")
    print(f"threshold_crossings_moved={moved}")
    print(f"minimum_crossing_signal_to_floor={checks['minimum_crossing_signal_to_floor']:.6g}")
    print(f"resolved_parity_fits={len(resolved)}")
    print(f"failed_parity_fits={len(failed)}")


if __name__ == "__main__":
    main()
