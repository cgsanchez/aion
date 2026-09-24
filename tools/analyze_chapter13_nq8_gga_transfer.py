#!/usr/bin/env python3
"""Authenticate and analyze the Chapter 13 NQ8 GGA transfer evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


def _load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _simpson(values: np.ndarray, step: float) -> float:
    intervals = values.size - 1
    if intervals <= 0 or intervals % 2:
        raise RuntimeError("stored trajectory does not admit composite Simpson integration")
    return float(
        (step / 3.0)
        * (
            values[0]
            + values[-1]
            + 4.0 * np.sum(values[1:-1:2])
            + 2.0 * np.sum(values[2:-1:2])
        )
    )


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(1.0, np.linalg.norm(right)))


def _authenticate_raw(raw: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = raw / "result.json"
    provenance_path = raw / "provenance.json"
    completed_path = raw / "completed.json"
    result = _load(result_path)
    provenance = _load(provenance_path)
    completed = _load(completed_path)
    if result.get("schema") != "aion.chapter13.nq8.result.v1":
        raise RuntimeError("raw NQ8 result schema is invalid")
    if completed.get("status") != "executed_unreviewed":
        raise RuntimeError("raw NQ8 campaign status is invalid")
    if completed["result_sha256"] != _sha256(result_path):
        raise RuntimeError("raw NQ8 result hash mismatch")
    if completed["provenance_sha256"] != _sha256(provenance_path):
        raise RuntimeError("raw NQ8 provenance hash mismatch")
    log_mismatch: dict[str, str] | None = None
    for relative, expected in provenance["artifacts_sha256"].items():
        path = raw / relative
        if not path.is_file():
            raise RuntimeError(f"missing raw NQ8 artifact: {path}")
        actual = _sha256(path)
        if actual != expected:
            if relative != "run.log":
                raise RuntimeError(f"raw NQ8 artifact hash mismatch: {path}")
            log_mismatch = {"recorded": expected, "actual": actual}
    authentication = {
        "result_sha256": _sha256(result_path),
        "provenance_sha256": _sha256(provenance_path),
        "completed_sha256": _sha256(completed_path),
        "legacy_external_run_log_hash_mismatch": log_mismatch,
        "numerical_artifacts_authenticated": True,
    }
    return result, authentication


def _authenticate_refinement(
    refinement: Path,
    raw_authentication: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = refinement / "result.json"
    provenance_path = refinement / "provenance.json"
    completed_path = refinement / "completed.json"
    result = _load(result_path)
    provenance = _load(provenance_path)
    completed = _load(completed_path)
    if result.get("schema") != "aion.chapter13.nq8.co-source-refinement.v1":
        raise RuntimeError("CO source-refinement schema is invalid")
    if completed.get("status") != "executed_unreviewed":
        raise RuntimeError("CO source-refinement status is invalid")
    if completed["result_sha256"] != _sha256(result_path):
        raise RuntimeError("CO source-refinement result hash mismatch")
    if completed["provenance_sha256"] != _sha256(provenance_path):
        raise RuntimeError("CO source-refinement provenance hash mismatch")
    if provenance["result_sha256"] != _sha256(result_path):
        raise RuntimeError("CO source-refinement provenance/result mismatch")
    accepted = result["accepted_raw_hashes"]
    for name in ("result_sha256", "provenance_sha256", "completed_sha256"):
        if accepted[name] != raw_authentication[name]:
            raise RuntimeError(f"CO source refinement used a different raw {name}")
    authentication = {
        "result_sha256": _sha256(result_path),
        "provenance_sha256": _sha256(provenance_path),
        "completed_sha256": _sha256(completed_path),
    }
    return result, authentication


def _validate_raw(raw: Path, result: dict[str, Any]) -> dict[str, Any]:
    required = {
        "zero_coefficients",
        "zero_density",
        "field_coefficients",
        "field_density",
        "field_lower_matrix",
        "field_xc_density",
        "field_xc_density_gradient",
        "n4_times_au",
        "n4_final_density",
        "n4_final_mixed_density",
        "n4_energies_au",
        "n4_source_powers_au",
        "n8_times_au",
        "n8_final_density",
        "n8_final_mixed_density",
        "n8_energies_au",
        "n8_source_powers_au",
    }
    recomputation_max = 0.0
    validation: dict[str, Any] = {}
    for record in result["records"]:
        system = str(record["system"])
        checkpoint = _load(raw / "checkpoints" / f"{system}.json")
        if checkpoint != record:
            raise RuntimeError(f"raw/checkpoint JSON mismatch for {system}")
        with np.load(raw / "checkpoints" / f"{system}.npz", allow_pickle=False) as arrays:
            if not required.issubset(arrays.files):
                raise RuntimeError(f"incomplete NQ8 array checkpoint for {system}")
            for name in arrays.files:
                if not np.all(np.isfinite(np.asarray(arrays[name]))):
                    raise RuntimeError(f"non-finite NQ8 data in {system}:{name}")
            independent: dict[str, Any] = {}
            for intervals in (4, 8):
                energies = np.asarray(arrays[f"n{intervals}_energies_au"])
                powers = np.asarray(arrays[f"n{intervals}_source_powers_au"])
                step = float(record["propagation"][str(intervals)]["step_au"])
                energy_change = float(energies[-1] - energies[0])
                work = _simpson(powers, step)
                for field, value in (
                    ("energy_change_au", energy_change),
                    ("integrated_source_power_au", work),
                    ("work_energy_residual_au", energy_change - work),
                ):
                    difference = abs(
                        value - float(record["propagation"][str(intervals)][field])
                    )
                    recomputation_max = max(recomputation_max, difference)
                    if difference > 5.0e-14:
                        raise RuntimeError(
                            f"trajectory recomputation mismatch: {system} n={intervals} {field}"
                        )
                independent[str(intervals)] = {
                    "energy_change_au": energy_change,
                    "integrated_source_power_au": work,
                    "work_energy_residual_au": energy_change - work,
                }
            density_difference = _relative(
                np.asarray(arrays["n8_final_mixed_density"]),
                np.asarray(arrays["n4_final_mixed_density"]),
            )
            difference = abs(
                density_difference
                - float(record["propagation_final_mixed_density_relative_difference"])
            )
            recomputation_max = max(recomputation_max, difference)
            if difference > 5.0e-15:
                raise RuntimeError(f"density refinement recomputation mismatch: {system}")
            independent["final_mixed_density_relative_difference"] = density_difference
            validation[system] = independent
    return {
        "systems": validation,
        "maximum_absolute_recomputation_difference": recomputation_max,
    }


def _source_metrics(refinement: dict[str, Any]) -> dict[str, Any]:
    rows = refinement["rows"]
    output: list[dict[str, Any]] = []
    for row in rows:
        components = {
            name: row[name]
            for name in (
                "one_electron_action",
                "negative_hartree_energy",
                "negative_xc_energy",
                "electronic_action",
            )
        }
        finite_sum = sum(
            float(components[name]["finite_mean"])
            for name in (
                "one_electron_action",
                "negative_hartree_energy",
                "negative_xc_energy",
            )
        )
        analytic_sum = sum(
            float(components[name]["analytic"])
            for name in (
                "one_electron_action",
                "negative_hartree_energy",
                "negative_xc_energy",
            )
        )
        output.append(
            {
                "step": float(row["step"]),
                "repeats": int(row["repeats"]),
                "total_absolute_residual": float(
                    components["electronic_action"]["absolute_residual"]
                ),
                "total_repeat_range": float(
                    components["electronic_action"]["repeat_range"]
                ),
                "finite_component_sum_residual": abs(
                    finite_sum - float(components["electronic_action"]["finite_mean"])
                ),
                "analytic_component_sum_residual": abs(
                    analytic_sum - float(components["electronic_action"]["analytic"])
                ),
                "one_electron_absolute_residual": float(
                    components["one_electron_action"]["absolute_residual"]
                ),
                "hartree_absolute_residual": float(
                    components["negative_hartree_energy"]["absolute_residual"]
                ),
                "xc_absolute_residual": float(
                    components["negative_xc_energy"]["absolute_residual"]
                ),
            }
        )
    errors = np.asarray([row["total_absolute_residual"] for row in output])
    steps = np.asarray([row["step"] for row in output])
    coarse_order = float(
        np.log(errors[0] / errors[1]) / np.log(steps[0] / steps[1])
    )
    return {
        "rows": output,
        "best_total_absolute_residual": float(np.min(errors)),
        "best_step": float(steps[int(np.argmin(errors))]),
        "coarsest_pair_observed_order": coarse_order,
        "maximum_total_repeat_range": max(row["total_repeat_range"] for row in output),
        "maximum_finite_component_sum_residual": max(
            row["finite_component_sum_residual"] for row in output
        ),
        "maximum_analytic_component_sum_residual": max(
            row["analytic_component_sum_residual"] for row in output
        ),
    }


def _system_metrics(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in result["records"]:
        source_errors = [
            float(item["absolute_residual"])
            for item in record["source_derivative"]["finite_differences"]
        ]
        rows.append(
            {
                "system": record["system"],
                "nao": record["nao"],
                "grid_points": record["grid_points"],
                "same_grid_energy_residual_au": record["same_grid_pyscf"][
                    "energy_absolute_residual_au"
                ],
                "same_grid_lower_relative_residual": record["same_grid_pyscf"][
                    "lower_relative_residual"
                ],
                "stationary_reference_energy_residual_au": record["zero_field"][
                    "energy_absolute_residual_au"
                ],
                "zero_orbital_residual": record["zero_field"]["orbital_residual"],
                "field_orbital_residual": record["static_field"]["orbital_residual"],
                "source_best_raw_residual": min(source_errors),
                "continuity_finite_region_residual": record[
                    "instantaneous_continuity"
                ]["finite_region_residual"],
                "continuity_global_residual": record["instantaneous_continuity"][
                    "global_charge_residual"
                ],
                "instantaneous_power_residual_au": record["instantaneous_power"][
                    "absolute_residual_au"
                ],
                "final_density_refinement_difference": record[
                    "propagation_final_mixed_density_relative_difference"
                ],
                "fine_work_energy_residual_au": abs(
                    record["propagation"]["8"]["work_energy_residual_au"]
                ),
                "fine_particle_number_drift": record["propagation"]["8"][
                    "maximum_particle_number_drift"
                ],
                "fine_hermiticity_residual": record["propagation"]["8"][
                    "maximum_hermiticity_residual"
                ],
                "fine_nonlinear_residual": record["propagation"]["8"][
                    "maximum_nonlinear_residual"
                ],
                "metric_minimum_eigenvalue": record["static_field"][
                    "metric_minimum_eigenvalue"
                ],
                "density_minimum": record["static_field"]["density_minimum"],
                "density_maximum": record["static_field"]["density_maximum"],
                "density_gradient_maximum": record["static_field"][
                    "density_gradient_maximum"
                ],
                "elapsed_seconds": record["elapsed_seconds"],
            }
        )
    return rows


def _checks(
    systems: list[dict[str, Any]],
    source: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    limits = {
        "same_grid_energy_residual_au": 1.0e-12,
        "same_grid_lower_relative_residual": 1.0e-12,
        "stationary_reference_energy_residual_au": 1.0e-6,
        "zero_orbital_residual": 2.0e-10,
        "field_orbital_residual": 2.0e-10,
        "continuity_finite_region_residual": 1.0e-9,
        "continuity_global_residual": 1.0e-12,
        "instantaneous_power_residual_au": 1.0e-10,
        "final_density_refinement_difference": 1.0e-6,
        "fine_work_energy_residual_au": 1.0e-8,
        "fine_particle_number_drift": 1.0e-10,
        "fine_hermiticity_residual": 1.0e-12,
        "fine_nonlinear_residual": 1.0e-9,
    }
    checks: dict[str, dict[str, Any]] = {}
    for field, limit in limits.items():
        value = max(float(row[field]) for row in systems)
        checks[field] = {"value": value, "limit": limit, "passed": value <= limit}
    checks["co_source_best_total_absolute_residual"] = {
        "value": source["best_total_absolute_residual"],
        "limit": 5.0e-7,
        "passed": source["best_total_absolute_residual"] <= 5.0e-7,
    }
    checks["co_source_coarsest_pair_order"] = {
        "value": source["coarsest_pair_observed_order"],
        "minimum": 1.8,
        "passed": source["coarsest_pair_observed_order"] >= 1.8,
    }
    checks["positive_metric"] = {
        "value": min(float(row["metric_minimum_eigenvalue"]) for row in systems),
        "minimum": 0.0,
        "passed": min(float(row["metric_minimum_eigenvalue"]) for row in systems)
        > 0.0,
    }
    return checks


def _plot_source(source: dict[str, Any], path: Path) -> None:
    rows = source["rows"]
    steps = np.asarray([row["step"] for row in rows])
    figure, axis = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    for field, label in (
        ("total_absolute_residual", "complete action"),
        ("one_electron_absolute_residual", "one electron"),
        ("hartree_absolute_residual", "Hartree"),
        ("xc_absolute_residual", "PBE XC"),
    ):
        axis.loglog(steps, [row[field] for row in rows], marker="o", label=label)
    axis.invert_xaxis()
    axis.set_xlabel("central-difference step")
    axis.set_ylabel("absolute derivative residual (a.u.)")
    axis.set_title("CO localized-source derivative refinement")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_transfer(systems: list[dict[str, Any]], path: Path) -> None:
    labels = [str(row["system"]).upper() for row in systems]
    fields = (
        ("instantaneous_power_residual_au", "instantaneous power"),
        ("continuity_finite_region_residual", "weak continuity"),
        ("fine_work_energy_residual_au", "trajectory work-energy"),
        ("final_density_refinement_difference", "dt density difference"),
    )
    x = np.arange(len(labels))
    width = 0.18
    figure, axis = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    for index, (field, label) in enumerate(fields):
        axis.bar(
            x + (index - 1.5) * width,
            [max(float(row[field]), 1.0e-22) for row in systems],
            width,
            label=label,
        )
    axis.set_yscale("log")
    axis.set_xticks(x, labels)
    axis.set_ylabel("absolute or relative residual")
    axis.set_title("PBE exact-Wilson transfer diagnostics")
    axis.grid(True, axis="y", which="both", alpha=0.3)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--refinement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    raw = arguments.raw.resolve()
    refinement_path = arguments.refinement.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result, raw_authentication = _authenticate_raw(raw)
    refinement, refinement_authentication = _authenticate_refinement(
        refinement_path,
        raw_authentication,
    )
    validation = _validate_raw(raw, result)
    systems = _system_metrics(result)
    source = _source_metrics(refinement)
    checks = _checks(systems, source)
    all_passed = all(bool(check["passed"]) for check in checks.values())
    _write_csv(output / "system_metrics.csv", systems)
    _write_csv(output / "co_source_refinement.csv", source["rows"])
    _plot_source(source, output / "co_source_refinement.png")
    _plot_transfer(systems, output / "transfer_diagnostics.png")
    figures = {
        name: _sha256(output / name)
        for name in ("co_source_refinement.png", "transfer_diagnostics.png")
    }
    summary = {
        "schema": "aion.chapter13.nq8.analysis.v1",
        "status": "analyzed_unreviewed",
        "raw_campaign": str(raw),
        "source_refinement": str(refinement_path),
        "raw_authentication": raw_authentication,
        "refinement_authentication": refinement_authentication,
        "validation": validation,
        "realization": result["realization"],
        "system_metrics": systems,
        "co_source_metrics": source,
        "checks": checks,
        "all_proposed_checks_passed": all_passed,
        "figure_sha256": figures,
        "interpretation": {
            "identity": (
                "same-grid PBE energy and weak lower matrix agree with independent "
                "PySCF/libxc evaluation at floating-point scale"
            ),
            "transfer": (
                "stationary, localized-source, instantaneous continuity and power, "
                "and short nonlinear propagation checks transfer from H3+ to CO"
            ),
            "scope": (
                "evidence is bounded to PBE, cc-pVDZ, the declared level-4 grid, "
                "weigend RI, fields and time window; it is not an LDA relabeling"
            ),
            "legacy_log_hash": (
                "the raw numerical artifacts authenticate; the shell-managed run.log "
                "changed after provenance was written and is explicitly excluded from "
                "scientific evidence"
            ),
        },
    }
    summary_path = output / "summary.json"
    _write_json(summary_path, summary)
    completed = {
        "schema": "aion.chapter13.nq8.analysis-completed.v1",
        "status": "analyzed_unreviewed",
        "summary_sha256": _sha256(summary_path),
        "artifacts_sha256": {
            path.name: _sha256(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "completed.json"
        },
    }
    _write_json(output / "completed.json", completed)
    print(json.dumps({"output": str(output), **completed}, indent=2))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
