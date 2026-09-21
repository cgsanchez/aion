#!/usr/bin/env python3
"""Authenticate and analyze one Chapter 13 NQ4 stationary campaign."""

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


def _stationarity_groups(result: dict[str, Any]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in result["energy_stationarity"]:
        grouped[(float(row["field_au"]), str(row["branch"]), str(row["direction"]))].append(
            row
        )
    return list(grouped.values())


def _coarse_order(rows: list[dict[str, Any]]) -> float:
    selected = sorted(rows, key=lambda row: float(row["step"]), reverse=True)[:3]
    steps = np.asarray([float(row["step"]) for row in selected])
    errors = np.asarray(
        [float(row["central_energy_derivative_abs_au"]) for row in selected]
    )
    return float(np.polyfit(np.log(steps), np.log(errors), 1)[0])


def _level4_to_level5_errors(result: dict[str, Any]) -> list[float]:
    indexed = {
        (int(row["level"]), float(row["field_au"]), str(row["branch"])): float(
            row["energy_molecular_total_au"]
        )
        for row in result["grid_refinement"]
    }
    return [
        abs(indexed[(4, field, branch)] - indexed[(5, field, branch)])
        for field in (0.0, 0.03, 0.06)
        for branch in ("hartree", "kohn_sham_lda")
    ]


def _tolerance_energy_errors(result: dict[str, Any], policy: str) -> list[float]:
    indexed = {
        (str(row["branch"]), str(row["policy"])): float(
            row["energy_molecular_total_au"]
        )
        for row in result["stationary_tolerance_refinement"]
    }
    return [
        abs(indexed[(branch, policy)] - indexed[(branch, "tight")])
        for branch in ("hartree", "kohn_sham_lda")
    ]


def _analyze(result: dict[str, Any]) -> dict[str, Any]:
    scan = result["h3plus_main_scan"]["rows"]
    gauges = result["gauge_checks"]
    field_free = [
        result["field_free_recovery"][system][branch]
        for system in ("h2", "h3plus")
        for branch in ("hartree", "kohn_sham")
    ]
    groups = _stationarity_groups(result)
    metrics = {
        "field_free_energy_absolute_residual_max_au": max(
            float(row["energy_absolute_residual_au"]) for row in field_free
        ),
        "field_free_density_relative_residual_max": max(
            float(row["coefficient_density_relative_residual"]) for row in field_free
        ),
        "orbital_residual_max": max(float(row["orbital_residual"]) for row in scan),
        "density_fixed_point_residual_max": max(
            float(row["density_fixed_point_residual"]) for row in scan
        ),
        "commutator_residual_max": max(
            float(row["commutator_residual"]) for row in scan
        ),
        "metric_orthonormality_residual_max": max(
            float(row["metric_orthonormality_residual"]) for row in scan
        ),
        "metric_minimum_eigenvalue_min": min(
            float(row["metric_minimum_eigenvalue"]) for row in scan
        ),
        "particle_number_residual_max": max(
            float(row["particle_number_residual"]) for row in scan
        ),
        "occupation_spectrum_residual_max": max(
            float(row["occupation_spectrum_residual"]) for row in scan
        ),
        "closed_shell_density_polynomial_residual_max": max(
            float(row["closed_shell_density_polynomial_residual"]) for row in scan
        ),
        "double_counting_residual_max_au": max(
            float(row["double_counting_residual_au"]) for row in scan
        ),
        "density_real_minimum": min(float(row["density_real_minimum"]) for row in scan),
        "density_imaginary_max_abs": max(
            float(row["density_imaginary_max_abs"]) for row in scan
        ),
        "density_direct_factorized_residual_max": max(
            float(row["density_direct_factorized_residual"]) for row in scan
        ),
        "gauge_energy_absolute_residual_max_au": max(
            float(row["energy_absolute_residual_au"]) for row in gauges
        ),
        "gauge_physical_density_relative_residual_max": max(
            float(row["physical_density_relative_residual"]) for row in gauges
        ),
        "gauge_coefficient_density_covariance_residual_max": max(
            float(row["coefficient_density_covariance_relative_residual"])
            for row in gauges
        ),
        "gauge_lower_covariance_residual_max": max(
            float(row["lower_matrix_covariance_relative_residual"]) for row in gauges
        ),
        "stationarity_best_derivative_max_au": max(
            min(float(row["central_energy_derivative_abs_au"]) for row in rows)
            for rows in groups
        ),
        "stationarity_coarse_order_min": min(_coarse_order(rows) for rows in groups),
        "level4_to_level5_energy_residual_max_au": max(
            _level4_to_level5_errors(result)
        ),
        "loose_to_tight_energy_residual_max_au": max(
            _tolerance_energy_errors(result, "loose")
        ),
        "phase_spread_rms_max": max(
            float(row["phase_spread_rms_max"]) for row in scan
        ),
        "exact_minus_p0_mechanical_relative_frobenius_max": max(
            float(row["exact_minus_p0_mechanical_relative_frobenius"])
            for row in scan
        ),
        "exact_minus_p0_one_electron_energy_max_abs_au": max(
            abs(float(row["exact_minus_p0_one_electron_energy_au"])) for row in scan
        ),
    }
    thresholds = {
        "field_free_energy_absolute_residual_max_au": 2.0e-8,
        "field_free_density_relative_residual_max": 1.0e-7,
        "orbital_residual_max": 2.0e-12,
        "density_fixed_point_residual_max": 2.0e-12,
        "commutator_residual_max": 2.0e-12,
        "metric_orthonormality_residual_max": 2.0e-12,
        "metric_minimum_eigenvalue_min": 1.0e-3,
        "particle_number_residual_max": 2.0e-12,
        "occupation_spectrum_residual_max": 2.0e-12,
        "closed_shell_density_polynomial_residual_max": 2.0e-12,
        "double_counting_residual_max_au": 2.0e-12,
        "density_real_minimum": -2.0e-14,
        "density_imaginary_max_abs": 2.0e-14,
        "density_direct_factorized_residual_max": 2.0e-12,
        "gauge_energy_absolute_residual_max_au": 2.0e-12,
        "gauge_physical_density_relative_residual_max": 2.0e-12,
        "gauge_coefficient_density_covariance_residual_max": 2.0e-12,
        "gauge_lower_covariance_residual_max": 2.0e-12,
        "stationarity_best_derivative_max_au": 1.0e-8,
        "stationarity_coarse_order_min": 1.9,
        "level4_to_level5_energy_residual_max_au": 1.0e-8,
        "loose_to_tight_energy_residual_max_au": 1.0e-10,
        "phase_spread_rms_max": 5.0e-2,
        "exact_minus_p0_mechanical_relative_frobenius_max": 1.0e-2,
        "exact_minus_p0_one_electron_energy_max_abs_au": 1.0e-3,
    }
    lower_bound_metrics = {
        "metric_minimum_eigenvalue_min",
        "density_real_minimum",
        "stationarity_coarse_order_min",
        "phase_spread_rms_max",
        "exact_minus_p0_mechanical_relative_frobenius_max",
        "exact_minus_p0_one_electron_energy_max_abs_au",
    }
    checks = {
        name: (
            metrics[name] >= threshold
            if name in lower_bound_metrics
            else metrics[name] <= threshold
        )
        for name, threshold in thresholds.items()
    }
    checks["physical_h3plus_two_electrons"] = (
        result["realization"]["h3plus"]["charge"] == 1
        and result["realization"]["h3plus"]["electrons"] == 2
        and all(abs(float(row["particle_number"]) - 2.0) <= 2.0e-12 for row in scan)
    )
    checks["state_identities_are_separated"] = (
        set(result["realization"]["state_identity_labels"])
        == {"field_free_stationary", "finite_field_stationary"}
        and set(result["realization"]["not_executed_state_identities"])
        == {"adiabatically_prepared", "sudden_quench"}
    )
    return {
        "status": "analyzed_unreviewed",
        "metrics": metrics,
        "proposed_thresholds": thresholds,
        "checks": checks,
        "passed_proposed_thresholds": all(checks.values()),
        "interpretation_boundary": (
            "exact-minus-P0 values are one-electron diagnostics evaluated on exact "
            "stationary states, not self-consistent reduced-model NQ7 results"
        ),
    }


def _plot_field_scan(result: dict[str, Any], output: Path) -> None:
    rows = result["h3plus_main_scan"]["rows"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for branch, label in (("hartree", "Hartree"), ("kohn_sham_lda", "LDA KS")):
        selected = sorted(
            (row for row in rows if row["branch"] == branch),
            key=lambda row: float(row["field_au"]),
        )
        fields = np.asarray([float(row["field_au"]) for row in selected])
        energies = np.asarray([float(row["energy_molecular_total_au"]) for row in selected])
        axes[0].plot(fields, energies - energies[0], marker="o", label=label)
        axes[1].plot(
            fields,
            [abs(float(row["exact_minus_p0_one_electron_energy_au"])) for row in selected],
            marker="o",
            label=label,
        )
        axes[2].semilogy(
            fields,
            [max(float(row["density_fixed_point_residual"]), 1.0e-16) for row in selected],
            marker="o",
            label=f"{label} density",
        )
        axes[2].semilogy(
            fields,
            [max(float(row["orbital_residual"]), 1.0e-16) for row in selected],
            marker="x",
            linestyle="--",
            label=f"{label} orbital",
        )
    axes[0].set_title("Stationary magnetic response")
    axes[0].set_ylabel(r"$E(B)-E(0)$ (Ha)")
    axes[1].set_title("Exact-minus-P0 one-electron diagnostic")
    axes[1].set_ylabel("absolute contracted correction (Ha)")
    axes[2].set_title("Independent stationary residuals")
    axes[2].set_ylabel("relative residual")
    for axis in axes:
        axis.set_xlabel(r"$B_z$ (a.u.)")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "h3plus_stationary_field_scan.png", dpi=180)
    plt.close(figure)


def _plot_stationarity_and_gauge(result: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    grouped: dict[tuple[float, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in result["energy_stationarity"]:
        grouped[(float(row["field_au"]), str(row["branch"]), str(row["direction"]))].append(
            row
        )
    for (field, branch, direction), rows in grouped.items():
        selected = sorted(rows, key=lambda row: float(row["step"]), reverse=True)
        axes[0].loglog(
            [row["step"] for row in selected],
            [max(float(row["central_energy_derivative_abs_au"]), 1.0e-16) for row in selected],
            marker="o" if direction == "real" else "x",
            linestyle="-" if branch == "hartree" else "--",
            label=f"B={field:.2f} {branch.replace('_lda', '')} {direction}",
        )
    axes[0].set_title("Energy stationarity on the metric manifold")
    axes[0].set_xlabel("retracted coefficient step")
    axes[0].set_ylabel("absolute central derivative (Ha)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)

    gauge_rows = result["gauge_checks"]
    labels = [f"B={row['field_au']:.2f}\n{row['branch'].replace('_lda', '')}" for row in gauge_rows]
    positions = np.arange(len(labels))
    width = 0.23
    axes[1].bar(
        positions - width,
        [max(float(row["energy_absolute_residual_au"]), 1.0e-16) for row in gauge_rows],
        width,
        label="energy abs. (Ha)",
    )
    axes[1].bar(
        positions,
        [max(float(row["physical_density_relative_residual"]), 1.0e-16) for row in gauge_rows],
        width,
        label="physical density rel.",
    )
    axes[1].bar(
        positions + width,
        [
            max(float(row["lower_matrix_covariance_relative_residual"]), 1.0e-16)
            for row in gauge_rows
        ],
        width,
        label="lower matrix rel.",
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Independent gauge-representative solves")
    axes[1].set_ylabel("residual")
    axes[1].set_xticks(positions, labels)
    axes[1].grid(True, axis="y", which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "energy_stationarity_and_gauge.png", dpi=180)
    plt.close(figure)


def _plot_refinements(result: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    indexed = {
        (int(row["level"]), float(row["field_au"]), str(row["branch"])): row
        for row in result["grid_refinement"]
    }
    for field in (0.0, 0.03, 0.06):
        for branch in ("hartree", "kohn_sham_lda"):
            finest = float(indexed[(5, field, branch)]["energy_molecular_total_au"])
            rows = [indexed[(level, field, branch)] for level in (3, 4, 5)]
            axes[0].semilogy(
                [row["points"] for row in rows],
                [
                    max(abs(float(row["energy_molecular_total_au"]) - finest), 1.0e-16)
                    for row in rows
                ],
                marker="o",
                label=f"B={field:.2f} {branch.replace('_lda', '')}",
            )
    axes[0].set_title("Molecular-grid refinement")
    axes[0].set_xlabel("unpruned grid points")
    axes[0].set_ylabel("energy difference from level 5 (Ha)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)

    tolerance_rows = result["stationary_tolerance_refinement"]
    policy_order = ("loose", "standard", "tight")
    for branch in ("hartree", "kohn_sham_lda"):
        selected = {
            row["policy"]: row for row in tolerance_rows if row["branch"] == branch
        }
        tight = float(selected["tight"]["energy_molecular_total_au"])
        axes[1].semilogy(
            policy_order,
            [
                max(abs(float(selected[name]["energy_molecular_total_au"]) - tight), 1.0e-16)
                for name in policy_order
            ],
            marker="o",
            label=branch.replace("_lda", ""),
        )
    axes[1].set_title(r"Nonlinear-solve refinement at $B_z=0.06$")
    axes[1].set_xlabel("SCF residual policy")
    axes[1].set_ylabel("energy difference from tight solve (Ha)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "grid_and_stationary_refinement.png", dpi=180)
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
    if result.get("schema") != "aion.chapter13-nq4-stationary":
        raise RuntimeError("raw result has the wrong schema")
    summary = _analyze(result)
    summary["raw_root"] = str(raw)
    summary["raw_hashes"] = hashes
    _plot_field_scan(result, output)
    _plot_stationarity_and_gauge(result, output)
    _plot_refinements(result, output)
    summary["figures"] = {
        name: _sha256(output / name)
        for name in (
            "h3plus_stationary_field_scan.png",
            "energy_stationarity_and_gauge.png",
            "grid_and_stationary_refinement.png",
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
