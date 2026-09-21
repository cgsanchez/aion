#!/usr/bin/env python3
"""Authenticate and analyze one Chapter 13 NQ3 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _authenticate(raw: Path) -> dict[str, str]:
    provenance_path = raw / "provenance.json"
    completed_path = raw / "completed.json"
    result_path = raw / "result.json"
    provenance = _load_json(provenance_path)
    completed = _load_json(completed_path)
    if completed["result_sha256"] != _sha256(result_path):
        raise RuntimeError("completed record does not authenticate result.json")
    if completed["provenance_sha256"] != _sha256(provenance_path):
        raise RuntimeError("completed record does not authenticate provenance.json")
    for name, expected in provenance["artifacts_sha256"].items():
        path = raw / name
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"provenance hash mismatch for {name}")
    return {
        "result_sha256": _sha256(result_path),
        "provenance_sha256": _sha256(provenance_path),
        "completed_sha256": _sha256(completed_path),
    }


def _coarse_order(rows: list[dict[str, Any]]) -> float:
    selected = sorted(rows, key=lambda row: float(row["step"]), reverse=True)[:3]
    errors = np.asarray([float(row["central_absolute_error_au"]) for row in selected])
    steps = np.asarray([float(row["step"]) for row in selected])
    if np.any(errors <= 1.0e-15):
        return float("nan")
    return float(np.polyfit(np.log(steps), np.log(errors), 1)[0])


def _initial_pair_order(rows: list[dict[str, Any]]) -> float:
    selected = sorted(rows, key=lambda row: float(row["step"]), reverse=True)[:2]
    return float(
        np.log(
            float(selected[0]["central_absolute_error_au"])
            / float(selected[1]["central_absolute_error_au"])
        )
        / np.log(float(selected[0]["step"]) / float(selected[1]["step"]))
    )


def _analyze(result: dict[str, Any]) -> dict[str, Any]:
    systems = result["systems"]
    grid_rows = [row for system in systems for row in system["grid_refinement"]]
    penultimate = [system["grid_refinement"][-2] for system in systems]
    bases = [system["finite_field"]["base"] for system in systems]
    invariances = [system["finite_field"]["invariance"] for system in systems]
    derivatives: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for system in systems:
        for family in ("matter_derivatives", "source_derivatives"):
            for row in system["finite_field"][family]:
                derivatives[(system["system"], family, row["direction"])].append(row)

    matter_groups = [
        rows for (_, family, _), rows in derivatives.items() if family == "matter_derivatives"
    ]
    physical_source_groups = [
        rows
        for (_, family, direction), rows in derivatives.items()
        if family == "source_derivatives" and direction == "physical_magnetic"
    ]
    pure_gauge_groups = [
        rows
        for (_, family, direction), rows in derivatives.items()
        if family == "source_derivatives" and direction == "pure_gauge"
    ]
    metrics = {
        "same_grid_energy_absolute_residual_max_au": max(
            float(row["same_grid_energy_absolute_residual_au"]) for row in grid_rows
        ),
        "same_grid_lower_relative_residual_max": max(
            float(row["same_grid_lower_relative_residual"]) for row in grid_rows
        ),
        "same_grid_electron_count_absolute_residual_max": max(
            float(row["same_grid_electron_count_absolute_residual"]) for row in grid_rows
        ),
        "level4_to_level5_energy_absolute_residual_max_au": max(
            float(row["energy_to_finest_absolute_residual_au"]) for row in penultimate
        ),
        "level4_to_level5_lower_relative_residual_max": max(
            float(row["lower_to_finest_relative_residual"]) for row in penultimate
        ),
        "lower_hermiticity_residual_max": max(
            float(row["lower_hermiticity_residual"]) for row in grid_rows
        ),
        "density_imaginary_max_abs": max(
            [float(row["density_imaginary_max_abs"]) for row in grid_rows]
            + [float(row["density_imaginary_max_abs"]) for row in bases]
        ),
        "density_real_minimum": min(
            [float(row["density_real_minimum"]) for row in grid_rows]
            + [float(row["density_real_minimum"]) for row in bases]
        ),
        "finite_field_lower_hermiticity_residual_max": max(
            float(row["lower_hermiticity_residual"]) for row in bases
        ),
        "gauge_energy_absolute_residual_max_au": max(
            float(row["electromagnetic_gauge"]["energy_absolute_residual_au"])
            for row in invariances
        ),
        "gauge_density_relative_residual_max": max(
            float(row["electromagnetic_gauge"]["density_relative_residual"])
            for row in invariances
        ),
        "gauge_lower_covariance_relative_residual_max": max(
            float(row["electromagnetic_gauge"]["lower_covariance_relative_residual"])
            for row in invariances
        ),
        "frame_energy_absolute_residual_max_au": max(
            float(row["coefficient_frame"]["energy_absolute_residual_au"])
            for row in invariances
        ),
        "frame_density_relative_residual_max": max(
            float(row["coefficient_frame"]["density_relative_residual"])
            for row in invariances
        ),
        "frame_lower_congruence_relative_residual_max": max(
            float(row["coefficient_frame"]["lower_congruence_relative_residual"])
            for row in invariances
        ),
        "matter_best_central_absolute_error_max_au": max(
            min(float(row["central_absolute_error_au"]) for row in rows)
            for rows in matter_groups
        ),
        "matter_minimum_coarse_order": min(_coarse_order(rows) for rows in matter_groups),
        "physical_source_best_central_absolute_error_max_au": max(
            min(float(row["central_absolute_error_au"]) for row in rows)
            for rows in physical_source_groups
        ),
        "physical_source_minimum_coarse_order": min(
            _coarse_order(rows) for rows in physical_source_groups
        ),
        "pure_gauge_source_best_central_absolute_error_max_au": max(
            min(float(row["central_absolute_error_au"]) for row in rows)
            for rows in pure_gauge_groups
        ),
        "pure_gauge_source_minimum_initial_pair_order": min(
            _initial_pair_order(rows) for rows in pure_gauge_groups
        ),
        "source_density_imaginary_max_abs": max(
            abs(float(row["source_density_imaginary_max_abs"]))
            for rows in physical_source_groups + pure_gauge_groups
            for row in rows
        ),
    }
    thresholds = {
        "same_grid_energy_absolute_residual_max_au": 2.0e-12,
        "same_grid_lower_relative_residual_max": 2.0e-12,
        "same_grid_electron_count_absolute_residual_max": 2.0e-12,
        "level4_to_level5_energy_absolute_residual_max_au": 5.0e-9,
        "level4_to_level5_lower_relative_residual_max": 2.0e-8,
        "lower_hermiticity_residual_max": 2.0e-12,
        "density_imaginary_max_abs": 1.0e-14,
        "density_real_minimum": -1.0e-14,
        "finite_field_lower_hermiticity_residual_max": 2.0e-12,
        "gauge_energy_absolute_residual_max_au": 1.0e-11,
        "gauge_density_relative_residual_max": 1.0e-12,
        "gauge_lower_covariance_relative_residual_max": 1.0e-12,
        "frame_energy_absolute_residual_max_au": 1.0e-11,
        "frame_density_relative_residual_max": 1.0e-12,
        "frame_lower_congruence_relative_residual_max": 1.0e-12,
        "matter_best_central_absolute_error_max_au": 2.0e-10,
        "matter_minimum_coarse_order": 1.9,
        "physical_source_best_central_absolute_error_max_au": 2.0e-10,
        "physical_source_minimum_coarse_order": 1.9,
        "pure_gauge_source_best_central_absolute_error_max_au": 2.0e-10,
        "pure_gauge_source_minimum_initial_pair_order": 1.9,
        "source_density_imaginary_max_abs": 2.0e-12,
    }
    checks = {
        name: (
            metrics[name] >= threshold
            if name == "density_real_minimum" or name.endswith("order")
            else metrics[name] <= threshold
        )
        for name, threshold in thresholds.items()
    }
    functional_ids = sorted(
        {
            str(row["functional"])
            for system in systems
            for row in system["grid_refinement"]
        }
    )
    checks["single_fixed_functional"] = functional_ids == ["lda,vwn"]
    return {
        "status": "analyzed_unreviewed",
        "metrics": metrics,
        "proposed_thresholds": thresholds,
        "checks": checks,
        "fixed_functional_identifiers": functional_ids,
        "passed_proposed_thresholds": all(checks.values()),
    }


def _plot_grid(result: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for system in result["systems"]:
        rows = system["grid_refinement"]
        label = system["system"].upper()
        points = [row["points"] for row in rows]
        axes[0].loglog(
            points,
            [max(float(row["energy_to_finest_absolute_residual_au"]), 1.0e-16) for row in rows],
            marker="o",
            label=f"{label} grid sequence",
        )
        axes[0].loglog(
            points,
            [max(float(row["same_grid_energy_absolute_residual_au"]), 1.0e-16) for row in rows],
            marker="x",
            linestyle="--",
            label=f"{label} PySCF same-grid",
        )
        axes[1].loglog(
            points,
            [max(float(row["lower_to_finest_relative_residual"]), 1.0e-16) for row in rows],
            marker="o",
            label=f"{label} grid sequence",
        )
        axes[1].loglog(
            points,
            [max(float(row["same_grid_lower_relative_residual"]), 1.0e-16) for row in rows],
            marker="x",
            linestyle="--",
            label=f"{label} PySCF same-grid",
        )
    axes[0].set_title("Fixed-functional XC energy")
    axes[0].set_ylabel("absolute residual (Ha)")
    axes[1].set_title("Fixed-functional lower XC matrix")
    axes[1].set_ylabel("relative Frobenius residual")
    for axis in axes:
        axis.set_xlabel("unpruned grid points")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "grid_refinement_and_pyscf_oracle.png", dpi=180)
    plt.close(figure)


def _plot_derivatives(result: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for row_index, system in enumerate(result["systems"]):
        for column_index, family in enumerate(("matter_derivatives", "source_derivatives")):
            axis = axes[row_index, column_index]
            rows = system["finite_field"][family]
            for direction in sorted({row["direction"] for row in rows}):
                selected = [row for row in rows if row["direction"] == direction]
                axis.loglog(
                    [row["step"] for row in selected],
                    [max(float(row["central_absolute_error_au"]), 1.0e-16) for row in selected],
                    marker="o",
                    label=direction.replace("_", " "),
                )
            axis.set_title(f"{system['system'].upper()} {family.replace('_', ' ')}")
            axis.set_ylabel("central-difference absolute error (Ha)")
            axis.grid(True, which="both", alpha=0.3)
            axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("finite-difference step")
    figure.tight_layout()
    figure.savefig(output / "matter_and_source_derivatives.png", dpi=180)
    plt.close(figure)


def _plot_covariance(result: dict[str, Any], output: Path) -> None:
    labels: list[str] = []
    energy: list[float] = []
    density: list[float] = []
    lower: list[float] = []
    for system in result["systems"]:
        for kind, values in system["finite_field"]["invariance"].items():
            labels.append(f"{system['system'].upper()}\n{kind.replace('_', ' ')}")
            energy.append(max(float(values["energy_absolute_residual_au"]), 1.0e-18))
            density.append(max(float(values["density_relative_residual"]), 1.0e-18))
            lower_name = (
                "lower_covariance_relative_residual"
                if kind == "electromagnetic_gauge"
                else "lower_congruence_relative_residual"
            )
            lower.append(max(float(values[lower_name]), 1.0e-18))
    positions = np.arange(len(labels))
    width = 0.24
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.bar(positions - width, energy, width, label="energy absolute (Ha)")
    axis.bar(positions, density, width, label="density relative")
    axis.bar(positions + width, lower, width, label="lower-matrix relative")
    axis.set_yscale("log")
    axis.set_ylabel("covariance residual")
    axis.set_xticks(positions, labels)
    axis.grid(True, axis="y", which="both", alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "gauge_and_frame_covariance.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    raw = arguments.raw.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis directory: {output}")
    output.mkdir(parents=True)
    hashes = _authenticate(raw)
    result = _load_json(raw / "result.json")
    if result.get("schema") != "aion.chapter13-nq3-wilson-lda":
        raise RuntimeError("raw result has the wrong schema")
    summary = _analyze(result)
    summary["raw_root"] = str(raw)
    summary["raw_hashes"] = hashes
    _plot_grid(result, output)
    _plot_derivatives(result, output)
    _plot_covariance(result, output)
    summary["figures"] = {
        name: _sha256(output / name)
        for name in (
            "grid_refinement_and_pyscf_oracle.png",
            "matter_and_source_derivatives.png",
            "gauge_and_frame_covariance.png",
        )
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "passed_proposed_thresholds": summary["passed_proposed_thresholds"],
                "summary_sha256": _sha256(summary_path),
            },
            indent=2,
        )
    )
    if not summary["passed_proposed_thresholds"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
