#!/usr/bin/env python3
"""Authenticate and analyze one Chapter 13 NQ2 campaign."""

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
    if np.any(errors <= 1.0e-13):
        return float("nan")
    return float(np.polyfit(np.log(steps), np.log(errors), 1)[0])


def _analyze(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    systems = result["systems"]
    finest = [system["grid_refinement"][-1] for system in systems]
    selected_auxiliary = [system["auxiliary_convergence"][0] for system in systems]
    auxiliary_improvement = [
        min(row["energy_relative_error"] for row in system["auxiliary_convergence"])
        / system["auxiliary_convergence"][0]["energy_relative_error"]
        for system in systems
    ]
    full_rank = [system["rank_convergence"][-1] for system in systems]
    final_solve = [system["solve_convergence"][-1] for system in systems]
    bases = [system["finite_field"]["base"] for system in systems]
    invariances = [system["finite_field"]["invariance"] for system in systems]

    derivative_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for system in systems:
        for family in ("matter_derivatives", "source_derivatives"):
            for row in system["finite_field"][family]:
                derivative_groups[(system["system"], family, row["direction"])].append(row)
    best_derivative = {
        "|".join(key): min(float(row["central_absolute_error_au"]) for row in rows)
        for key, rows in derivative_groups.items()
    }
    matter_orders = [
        _coarse_order(rows)
        for (system, family, direction), rows in derivative_groups.items()
        if family == "matter_derivatives"
    ]
    physical_source_orders = [
        _coarse_order(rows)
        for (system, family, direction), rows in derivative_groups.items()
        if family == "source_derivatives" and direction == "physical_magnetic"
    ]
    source_imaginary = [
        abs(float(row["source_moment_imaginary_max_abs"]))
        for system in systems
        for row in system["finite_field"]["source_derivatives"]
    ]

    metrics = {
        "finest_grid_energy_relative_max": max(
            float(row["energy_relative_error"]) for row in finest
        ),
        "finest_grid_lower_relative_max": max(float(row["lower_relative_error"]) for row in finest),
        "finest_grid_three_index_relative_max": max(
            float(row["three_index_relative_error"]) for row in finest
        ),
        "selected_auxiliary_exact_energy_relative_max": max(
            float(row["energy_relative_error"]) for row in selected_auxiliary
        ),
        "best_auxiliary_to_selected_error_ratio_max": max(auxiliary_improvement),
        "full_rank_energy_absolute_error_max_au": max(
            float(row["full_rank_energy_absolute_error_au"]) for row in full_rank
        ),
        "full_rank_lower_relative_error_max": max(
            float(row["full_rank_lower_relative_error"]) for row in full_rank
        ),
        "final_iterative_solve_relative_residual_max": max(
            float(row["achieved_relative_residual"]) for row in final_solve
        ),
        "final_iterative_solve_energy_error_max_au": max(
            float(row["energy_absolute_error_au"]) for row in final_solve
        ),
        "production_pair_counting_residual_max": max(
            float(row["pair_counting_residual"]) for row in bases
        ),
        "production_lower_hermiticity_residual_max": max(
            float(row["lower_hermiticity_residual"]) for row in bases
        ),
        "production_solve_relative_residual_max": max(
            float(row["retained_solve_relative_residual"]) for row in bases
        ),
        "production_stationary_energy_residual_max_au": max(
            float(row["stationary_energy_residual"]) for row in bases
        ),
        "gauge_energy_absolute_residual_max_au": max(
            float(row["electromagnetic_gauge"]["energy_absolute_residual_au"])
            for row in invariances
        ),
        "gauge_moment_relative_residual_max": max(
            float(row["electromagnetic_gauge"]["moment_relative_residual"]) for row in invariances
        ),
        "frame_energy_absolute_residual_max_au": max(
            float(row["coefficient_frame"]["energy_absolute_residual_au"]) for row in invariances
        ),
        "frame_moment_relative_residual_max": max(
            float(row["coefficient_frame"]["moment_relative_residual"]) for row in invariances
        ),
        "frame_lower_congruence_relative_residual_max": max(
            float(row["coefficient_frame"]["lower_congruence_relative_residual"])
            for row in invariances
        ),
        "matter_best_central_absolute_error_max_au": max(
            value for key, value in best_derivative.items() if "|matter_derivatives|" in key
        ),
        "source_best_central_absolute_error_max_au": max(
            value for key, value in best_derivative.items() if "|source_derivatives|" in key
        ),
        "matter_minimum_coarse_order": min(matter_orders),
        "physical_source_minimum_coarse_order": min(physical_source_orders),
        "source_moment_imaginary_max_abs": max(source_imaginary),
    }
    thresholds = {
        "finest_grid_energy_relative_max": 5.0e-9,
        "finest_grid_lower_relative_max": 5.0e-9,
        "finest_grid_three_index_relative_max": 5.0e-9,
        "selected_auxiliary_exact_energy_relative_max": 1.0e-4,
        "best_auxiliary_to_selected_error_ratio_max": 0.8,
        "full_rank_energy_absolute_error_max_au": 2.0e-10,
        "full_rank_lower_relative_error_max": 2.0e-10,
        "final_iterative_solve_relative_residual_max": 2.0e-10,
        "final_iterative_solve_energy_error_max_au": 2.0e-10,
        "production_pair_counting_residual_max": 2.0e-12,
        "production_lower_hermiticity_residual_max": 2.0e-12,
        "production_solve_relative_residual_max": 2.0e-11,
        "production_stationary_energy_residual_max_au": 2.0e-11,
        "gauge_energy_absolute_residual_max_au": 1.0e-11,
        "gauge_moment_relative_residual_max": 2.0e-12,
        "frame_energy_absolute_residual_max_au": 1.0e-11,
        "frame_moment_relative_residual_max": 2.0e-12,
        "frame_lower_congruence_relative_residual_max": 2.0e-12,
        "matter_best_central_absolute_error_max_au": 2.0e-8,
        "source_best_central_absolute_error_max_au": 2.0e-8,
        "source_moment_imaginary_max_abs": 2.0e-12,
    }
    checks = {name: metrics[name] <= threshold for name, threshold in thresholds.items()}
    checks["matter_minimum_coarse_order"] = metrics["matter_minimum_coarse_order"] >= 1.9
    checks["physical_source_minimum_coarse_order"] = (
        metrics["physical_source_minimum_coarse_order"] >= 1.9
    )
    checks["iterative_solve_reports_success"] = all(int(row["info"]) == 0 for row in final_solve)
    checks["hartree_self_interaction_declared"] = all(
        bool(row["self_interaction_included"]) for row in bases
    )
    summary = {
        "schema": "aion.chapter13-nq2-analysis",
        "schema_version": "1.0.0",
        "status": "analyzed_unreviewed",
        "passed_proposed_thresholds": all(checks.values()),
        "metrics": metrics,
        "best_derivative_errors": best_derivative,
        "proposed_thresholds": thresholds,
        "checks": checks,
        "interpretation_boundary": {
            "qualified": (
                "one discrete Coulomb-metric RI-Wilson Hartree action and its "
                "matter/source descendants"
            ),
            "not_qualified": [
                "Wilson LDA/GGA exchange-correlation closure",
                "nonlinear stationary states",
                "nonlinear real-time propagation",
                "continuum-exact Hartree interaction",
            ],
            "self_interaction": (
                "the common-field Hartree action includes self-interaction by definition"
            ),
        },
    }
    return summary, metrics


def _plot_grid_auxiliary(result: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for system in result["systems"]:
        label = system["system"].upper()
        grid = system["grid_refinement"]
        points = [row["points"] for row in grid]
        for key, marker, suffix in (
            ("energy_relative_error", "o", "energy"),
            ("lower_relative_error", "s", "lower matrix"),
            ("three_index_relative_error", "^", "three-index"),
        ):
            axes[0].loglog(
                points,
                [max(float(row[key]), 1.0e-16) for row in grid],
                marker=marker,
                label=f"{label} {suffix}",
            )
        auxiliary = system["auxiliary_convergence"]
        axes[1].semilogy(
            [row["naux"] for row in auxiliary],
            [row["energy_relative_error"] for row in auxiliary],
            marker="o",
            label=label,
        )
    axes[0].set_xlabel("unpruned molecular-grid points")
    axes[0].set_ylabel("relative error against analytic PySCF RI data")
    axes[0].set_title("Grid convergence at fixed auxiliary action")
    axes[1].set_xlabel("auxiliary functions")
    axes[1].set_ylabel("RI energy relative error vs exact four-center Coulomb")
    axes[1].set_title("Auxiliary-space convergence")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "grid_and_auxiliary_convergence.png", dpi=180)
    plt.close(figure)


def _plot_rank_solve(result: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for system in result["systems"]:
        label = system["system"].upper()
        rank = system["rank_convergence"]
        axes[0].semilogy(
            [row["rank"] for row in rank],
            [max(float(row["full_rank_energy_absolute_error_au"]), 1.0e-16) for row in rank],
            marker="o",
            label=f"{label} energy",
        )
        axes[0].semilogy(
            [row["rank"] for row in rank],
            [max(float(row["full_rank_lower_relative_error"]), 1.0e-16) for row in rank],
            marker="s",
            label=f"{label} lower matrix",
        )
        solve = system["solve_convergence"]
        axes[1].loglog(
            [row["requested_relative_tolerance"] for row in solve],
            [max(float(row["achieved_relative_residual"]), 1.0e-16) for row in solve],
            marker="o",
            label=f"{label} residual",
        )
        axes[1].loglog(
            [row["requested_relative_tolerance"] for row in solve],
            [max(float(row["energy_absolute_error_au"]), 1.0e-16) for row in solve],
            marker="s",
            label=f"{label} objective error",
        )
    axes[0].set_xlabel("retained Coulomb-metric rank")
    axes[0].set_ylabel("error against full retained rank")
    axes[0].set_title("Rank-policy convergence")
    axes[1].set_xlabel("requested CG relative tolerance")
    axes[1].set_ylabel("achieved error")
    axes[1].set_title("Auxiliary solve convergence")
    axes[1].invert_xaxis()
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "rank_and_solve_convergence.png", dpi=180)
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
    figure.savefig(output / "derivative_sequences.png", dpi=180)
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
    if result.get("schema") != "aion.chapter13-nq2-ri-wilson-hartree":
        raise RuntimeError("raw result has the wrong schema")
    summary, _ = _analyze(result)
    summary["raw_root"] = str(raw)
    summary["raw_hashes"] = hashes
    _plot_grid_auxiliary(result, output)
    _plot_rank_solve(result, output)
    _plot_derivatives(result, output)
    summary["figures"] = {
        name: _sha256(output / name)
        for name in (
            "grid_and_auxiliary_convergence.png",
            "rank_and_solve_convergence.png",
            "derivative_sequences.png",
        )
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
