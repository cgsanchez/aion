#!/usr/bin/env python3
"""Authenticate and analyze one Chapter 13 NQ7 reduced-model campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from aion.config import BackendConfig
from aion.electromagnetism import AffineMagneticGauge, UniformMagneticField
from aion.electronic_structure import (
    AOGridPolicy,
    WilsonStationaryBranch,
    prepare_ao_quadrature,
    prepare_exact_wilson_stationary_factory,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_chapter13_nq7_reduced_models as campaign

_LEVELS = ("p0", "e1", "strict_c1", "density_resummed_c1")
_FIRST_ORDER_LEVELS = ("strict_c1", "density_resummed_c1")
_BRANCHES = ("hartree", "kohn_sham_lda")
_CASES = ("electric_field_free", "electric_static_b", "magnetic_induction")
_MAGNETIC_CASES = ("electric_static_b", "magnetic_induction")
_LEVEL_LABELS = {
    "p0": "P0",
    "e1": "E1",
    "strict_c1": "strict C1",
    "density_resummed_c1": "density-resummed C1",
}
_BRANCH_LABELS = {"hartree": "Hartree", "kohn_sham_lda": "LDA KS"}
_CASE_LABELS = {
    "electric_field_free": "uniform E, B=0",
    "electric_static_b": "uniform E, static B",
    "magnetic_induction": "time-dependent B",
}
_COLORS = {
    "p0": "#4c78a8",
    "e1": "#f58518",
    "strict_c1": "#7a5195",
    "density_resummed_c1": "#54a24b",
    "exact": "#2f2f2f",
}
_MARKERS = {"p0": "o", "e1": "s", "strict_c1": "^", "density_resummed_c1": "D"}
_FAILED_LAMBDA_PATTERN = re.compile(r"lambda_min=([+-]?[0-9.]+e[+-][0-9]+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _authenticate(raw: Path) -> dict[str, Any]:
    result_path = raw / "result.json"
    provenance_path = raw / "provenance.json"
    completed_path = raw / "completed.json"
    result = _load_json(result_path)
    provenance = _load_json(provenance_path)
    completed = _load_json(completed_path)
    if result.get("schema") != "aion.chapter13.nq7.result.v2":
        raise RuntimeError("raw NQ7 result has the wrong schema")
    if completed.get("status") != "complete_with_visible_domain_failures":
        raise RuntimeError("raw NQ7 campaign does not have its expected closed status")
    if completed["result_sha256"] != _sha256(result_path):
        raise RuntimeError("raw NQ7 result hash mismatch")
    if completed["provenance_sha256"] != _sha256(provenance_path):
        raise RuntimeError("raw NQ7 provenance hash mismatch")
    for relative, expected in provenance["artifacts_sha256"].items():
        path = raw / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"raw NQ7 artifact hash mismatch: {path}")
    for parent in result["accepted_inputs"].values():
        root = Path(parent["root"])
        for name in ("result", "provenance", "completed"):
            path = root / f"{name}.json"
            if _sha256(path) != parent[f"{name}_sha256"]:
                raise RuntimeError(f"accepted parent changed: {path}")
        review_path = Path(provenance["repository"]) / parent["review_record"]
        if _sha256(review_path) != parent["review_record_sha256"]:
            raise RuntimeError(f"accepted parent review changed: {review_path}")
        review = _load_json(review_path)
        if review.get("decision") != "accepted":
            raise RuntimeError(f"parent review is no longer accepted: {review_path}")
    return {
        "result_sha256": _sha256(result_path),
        "provenance_sha256": _sha256(provenance_path),
        "completed_sha256": _sha256(completed_path),
        "raw_artifact_count": len(provenance["artifacts_sha256"]),
    }


def _simpson(values: np.ndarray, interval: float) -> float:
    intervals = len(values) - 1
    if intervals <= 0 or intervals % 2:
        raise ValueError("Simpson integration requires a positive even interval count")
    return float(
        (interval / 3.0)
        * (values[0] + values[-1] + 4.0 * np.sum(values[1:-1:2]) + 2.0 * np.sum(values[2:-1:2]))
    )


def _relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(1.0, np.linalg.norm(right)))


def _validate_records(raw: Path, result: dict[str, Any]) -> dict[str, Any]:
    validation: dict[str, Any] = {}
    required_arrays = {
        "stationary": {
            "coefficients",
            "coefficient_density",
            "mixed_density",
            "lower_matrix",
            "density_grid",
            "matched_lower_matrix",
            "matched_density_grid",
        },
        "domain": {"metric", "density_zero", "density_first"},
        "dynamics": {
            "times_au",
            "final_mixed_density",
            "energies_au",
            "source_powers_au",
            "density_zero_minima",
            "density_assembled_minima",
            "metric_minimum_eigenvalues",
            "metric_condition_numbers",
        },
    }
    for phase, records in result["records"].items():
        record_ids = [str(record["run_id"]) for record in records]
        if len(record_ids) != len(set(record_ids)):
            raise RuntimeError(f"duplicate {phase} run IDs")
        complete = 0
        failed = 0
        recomputation_max = 0.0
        for record in records:
            run_id = str(record["run_id"])
            record_path = raw / "checkpoints" / phase / f"{run_id}.json"
            if _load_json(record_path) != record:
                raise RuntimeError(f"synthesized/raw record mismatch: {record_path}")
            npz_path = record_path.with_suffix(".npz")
            if record["status"] == "failed":
                failed += 1
                if npz_path.exists():
                    raise RuntimeError(f"failed record has arrays: {npz_path}")
                continue
            complete += 1
            with np.load(npz_path, allow_pickle=False) as arrays:
                if not required_arrays[phase].issubset(arrays.files):
                    raise RuntimeError(f"incomplete array archive: {npz_path}")
                for name in arrays.files:
                    if not np.all(np.isfinite(np.asarray(arrays[name]))):
                        raise RuntimeError(f"non-finite data in {npz_path}:{name}")
                if phase == "dynamics":
                    energies = np.asarray(arrays["energies_au"])
                    powers = np.asarray(arrays["source_powers_au"])
                    step = float(record["interval_au"])
                    independent = {
                        "energy_change_au": float(energies[-1] - energies[0]),
                        "integrated_source_power_au": _simpson(powers, step),
                    }
                    for field, value in independent.items():
                        difference = abs(value - float(record[field]))
                        recomputation_max = max(recomputation_max, difference)
                        if difference > 5.0e-15:
                            raise RuntimeError(f"dynamic recomputation failed: {run_id} {field}")
        expected = result["checkpoint_validation"][phase]
        if complete != expected["complete"] or failed != expected["failed"]:
            raise RuntimeError(f"{phase} status counts changed")
        validation[phase] = {
            "record_count": len(records),
            "complete": complete,
            "failed": failed,
            "dynamic_recomputation_max_abs": recomputation_max,
        }
    return validation


def _compute_exact_domain(raw: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for basis in campaign._DOMAIN_BASES:
        reference = campaign._domain_reference(raw, basis)
        quadrature = prepare_ao_quadrature(
            reference,
            BackendConfig(),
            grid_policy=AOGridPolicy.qualification(3),
            block_size=1024,
        )
        factory = prepare_exact_wilson_stationary_factory(
            quadrature,
            auxiliary_basis=campaign._AUXILIARY_BASIS,
            functional=campaign._FUNCTIONAL,
        )
        for field in campaign._DOMAIN_FIELDS:
            gauge = AffineMagneticGauge(
                UniformMagneticField((0.0, 0.0, field)),
                origin_au=campaign._GAUGE_ORIGIN,
            )
            overlap = np.asarray(factory.model(gauge, WilsonStationaryBranch.HARTREE).overlap)
            eigenvalues = np.linalg.eigvalsh(overlap)
            rows.append(
                {
                    "basis": basis,
                    "field_au": field,
                    "exact_metric_minimum_eigenvalue": float(eigenvalues[0]),
                    "exact_metric_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
                }
            )
    return rows


def _load_exact_domain(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "basis": row["basis"],
            "field_au": float(row["field_au"]),
            "exact_metric_minimum_eigenvalue": float(row["exact_metric_minimum_eigenvalue"]),
            "exact_metric_condition_number": float(row["exact_metric_condition_number"]),
        }
        for row in _read_csv(path)
    ]


def _validate_exact_domain(
    exact_rows: list[dict[str, Any]], domain_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    expected = {
        (basis, field) for basis in campaign._DOMAIN_BASES for field in campaign._DOMAIN_FIELDS
    }
    indexed = {(row["basis"], float(row["field_au"])): row for row in exact_rows}
    if set(indexed) != expected:
        raise RuntimeError("exact-domain metric matrix is incomplete")
    crosscheck_max = 0.0
    for record in domain_rows:
        if record["status"] != "complete":
            continue
        key = (record["basis"], float(record["field_au"]))
        crosscheck_max = max(
            crosscheck_max,
            abs(
                float(record["exact_metric_minimum_eigenvalue"])
                - float(indexed[key]["exact_metric_minimum_eigenvalue"])
            ),
        )
    if crosscheck_max > 2.0e-13:
        raise RuntimeError("recomputed exact-domain metrics disagree with raw records")
    return {
        "point_count": len(exact_rows),
        "minimum_eigenvalue_min": min(
            float(row["exact_metric_minimum_eigenvalue"]) for row in exact_rows
        ),
        "condition_number_max": max(
            float(row["exact_metric_condition_number"]) for row in exact_rows
        ),
        "raw_crosscheck_max_abs": crosscheck_max,
    }


def _failed_lambda(record: dict[str, Any]) -> float | None:
    if record["status"] == "complete":
        return float(record["metric_minimum_eigenvalue"])
    match = _FAILED_LAMBDA_PATTERN.search(str(record.get("exception", "")))
    return None if match is None else float(match.group(1))


def _stationary_by_field(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = (
        "energy_error_au",
        "mixed_density_relative_error",
        "real_space_density_relative_error",
        "matched_lower_relative_error",
        "matched_density_relative_error",
        "source_mechanical_rate_error_au",
        "power_error_au",
    )
    for field in campaign._FIELDS:
        for level in _LEVELS:
            selected = [
                row for row in rows if float(row["field_au"]) == field and row["level"] == level
            ]
            aggregate: dict[str, Any] = {"field_au": field, "level": level}
            for metric in metrics:
                aggregate[f"{metric}_max_abs"] = max(abs(float(row[metric])) for row in selected)
            aggregate["orbital_residual_max"] = max(
                float(row["orbital_residual"]) for row in selected
            )
            aggregate["density_fixed_point_residual_max"] = max(
                float(row["density_fixed_point_residual"]) for row in selected
            )
            output.append(aggregate)
    return output


def _log_slope(
    rows: list[dict[str, Any]], level: str, metric: str, fields: tuple[float, ...]
) -> float:
    selected = [row for row in rows if row["level"] == level and float(row["field_au"]) in fields]
    selected.sort(key=lambda row: float(row["field_au"]))
    x = np.log(np.asarray([float(row["field_au"]) for row in selected]))
    y = np.log(np.asarray([max(float(row[metric]), 1.0e-30) for row in selected]))
    return float(np.polyfit(x, y, 1)[0])


def _domain_boundaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for basis in campaign._DOMAIN_BASES:
        for level in _LEVELS:
            selected = sorted(
                (row for row in rows if row["basis"] == basis and row["level"] == level),
                key=lambda row: float(row["field_au"]),
            )
            passing = [float(row["field_au"]) for row in selected if row["status"] == "complete"]
            failing = [float(row["field_au"]) for row in selected if row["status"] == "failed"]
            output.append(
                {
                    "basis": basis,
                    "level": level,
                    "largest_tested_passing_field_au": max(passing),
                    "smallest_tested_failing_field_au": min(failing) if failing else None,
                    "failure_count": len(failing),
                    "complete_count": len(passing),
                    "failure_type": "reduced_metric_nonpositive" if failing else None,
                    "lda_failures_where_metric_valid": sum(
                        row.get("lda_domain_status") == "failed" for row in selected
                    ),
                    "density_assembled_minimum": min(
                        float(row["density_assembled_minimum"])
                        for row in selected
                        if row["status"] == "complete"
                    ),
                }
            )
    return output


def _dynamic_analysis(
    raw: Path,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {str(row["run_id"]): row for row in rows}
    finest = [row for row in rows if int(row["intervals"]) == 32]
    refinement: list[dict[str, Any]] = []
    for case in _CASES:
        for branch in _BRANCHES:
            for level in _FIRST_ORDER_LEVELS:
                prefix = f"{case}_{branch}_{level}"
                coarse_path = raw / "checkpoints" / "dynamics" / f"{prefix}_n16.npz"
                fine_path = raw / "checkpoints" / "dynamics" / f"{prefix}_n32.npz"
                with (
                    np.load(coarse_path, allow_pickle=False) as coarse,
                    np.load(fine_path, allow_pickle=False) as fine,
                ):
                    timestep_difference = _relative_difference(
                        np.asarray(coarse["final_mixed_density"]),
                        np.asarray(fine["final_mixed_density"]),
                    )
                model_error = float(indexed[f"{prefix}_n32"]["final_mixed_density_relative_error"])
                refinement.append(
                    {
                        "case": case,
                        "branch": branch,
                        "level": level,
                        "n16_to_n32_final_density_difference": timestep_difference,
                        "n32_exact_model_error": model_error,
                        "model_error_to_timestep_difference_ratio": (
                            model_error / timestep_difference
                        ),
                    }
                )
    comparison: list[dict[str, Any]] = []
    for case in _CASES:
        for branch in _BRANCHES:
            paths = {
                level: raw / "checkpoints" / "dynamics" / f"{case}_{branch}_{level}_n32.npz"
                for level in _FIRST_ORDER_LEVELS
            }
            with (
                np.load(paths["strict_c1"], allow_pickle=False) as strict,
                np.load(paths["density_resummed_c1"], allow_pickle=False) as resummed,
            ):
                comparison.append(
                    {
                        "case": case,
                        "branch": branch,
                        "final_density_relative_difference": _relative_difference(
                            np.asarray(strict["final_mixed_density"]),
                            np.asarray(resummed["final_mixed_density"]),
                        ),
                        "energy_trajectory_relative_difference": _relative_difference(
                            np.asarray(strict["energies_au"]),
                            np.asarray(resummed["energies_au"]),
                        ),
                        "power_trajectory_relative_difference": _relative_difference(
                            np.asarray(strict["source_powers_au"]),
                            np.asarray(resummed["source_powers_au"]),
                        ),
                    }
                )
    return finest, refinement + comparison


def _analyze(
    stationary: list[dict[str, Any]],
    stationary_field: list[dict[str, Any]],
    domain: list[dict[str, Any]],
    boundaries: list[dict[str, Any]],
    exact_validation: dict[str, Any],
    finest: list[dict[str, Any]],
    dynamic_secondary: list[dict[str, Any]],
) -> dict[str, Any]:
    zero_field = [row for row in stationary if float(row["field_au"]) == 0.0]
    e1_zero_b = [
        row for row in finest if row["case"] == "electric_field_free" and row["level"] == "e1"
    ]
    c1_stationary = [row for row in stationary_field if row["level"] in _FIRST_ORDER_LEVELS]
    c1_finest = [row for row in finest if row["level"] in _FIRST_ORDER_LEVELS]
    magnetic_c1 = [row for row in c1_finest if row["case"] in _MAGNETIC_CASES]
    magnetic_refinement = [
        row
        for row in dynamic_secondary
        if "n16_to_n32_final_density_difference" in row and row["case"] in _MAGNETIC_CASES
    ]
    main_domain = [
        row for row in domain if row["basis"] == "cc-pvdz" and float(row["field_au"]) <= 0.06
    ]
    p0e1_stationary = {
        (float(row["field_au"]), row["level"]): row
        for row in stationary_field
        if row["level"] in {"p0", "e1"}
    }
    c1_improvements: list[float] = []
    for row in c1_stationary:
        if float(row["field_au"]) == 0.0:
            continue
        p0 = p0e1_stationary[(float(row["field_au"]), "p0")]
        c1_improvements.append(
            float(p0["mixed_density_relative_error_max_abs"])
            / float(row["mixed_density_relative_error_max_abs"])
        )
    finest_index = {(row["case"], row["branch"], row["level"]): row for row in finest}
    dynamic_improvements: list[float] = []
    for row in magnetic_c1:
        baselines = [
            float(
                finest_index[(row["case"], row["branch"], level)][
                    "final_mixed_density_relative_error"
                ]
            )
            for level in ("p0", "e1")
        ]
        dynamic_improvements.append(
            min(baselines) / float(row["final_mixed_density_relative_error"])
        )
    metrics = {
        "zero_field_energy_error_max_abs_au": max(
            abs(float(row["energy_error_au"])) for row in zero_field
        ),
        "zero_field_mixed_density_relative_error_max": max(
            float(row["mixed_density_relative_error"]) for row in zero_field
        ),
        "zero_field_matched_lower_relative_error_max": max(
            float(row["matched_lower_relative_error"]) for row in zero_field
        ),
        "e1_zero_b_final_density_relative_error_max": max(
            float(row["final_mixed_density_relative_error"]) for row in e1_zero_b
        ),
        "e1_zero_b_energy_trajectory_relative_error_max": max(
            float(row["energy_trajectory_relative_error"]) for row in e1_zero_b
        ),
        "e1_zero_b_power_trajectory_relative_error_max": max(
            float(row["power_trajectory_relative_error"]) for row in e1_zero_b
        ),
        "strict_c1_low_field_mixed_density_order": _log_slope(
            stationary_field,
            "strict_c1",
            "mixed_density_relative_error_max_abs",
            (0.001, 0.003, 0.01),
        ),
        "p0_low_field_mixed_density_order": _log_slope(
            stationary_field,
            "p0",
            "mixed_density_relative_error_max_abs",
            (0.001, 0.003, 0.01),
        ),
        "strict_c1_low_field_energy_order": _log_slope(
            stationary_field,
            "strict_c1",
            "energy_error_au_max_abs",
            (0.001, 0.003, 0.01),
        ),
        "stationary_c1_density_improvement_factor_min": min(c1_improvements),
        "dynamic_c1_improvement_over_best_p0_e1_min": min(dynamic_improvements),
        "c1_stationary_energy_error_max_abs_au": max(
            float(row["energy_error_au_max_abs"]) for row in c1_stationary
        ),
        "c1_stationary_mixed_density_relative_error_max": max(
            float(row["mixed_density_relative_error_max_abs"]) for row in c1_stationary
        ),
        "c1_stationary_matched_lower_relative_error_max": max(
            float(row["matched_lower_relative_error_max_abs"]) for row in c1_stationary
        ),
        "c1_dynamic_final_density_relative_error_max": max(
            float(row["final_mixed_density_relative_error"]) for row in magnetic_c1
        ),
        "c1_dynamic_power_balance_residual_max_au": max(
            float(row["power_balance_residual_au"]) for row in c1_finest
        ),
        "c1_dynamic_model_to_timestep_ratio_min": min(
            float(row["model_error_to_timestep_difference_ratio"]) for row in magnetic_refinement
        ),
        "c1_dynamic_cross_metric_residual_max": max(
            float(row["cross_metric_residual_max"]) for row in c1_finest
        ),
        "c1_dynamic_nonlinear_residual_max": max(
            float(row["nonlinear_residual_max"]) for row in c1_finest
        ),
        "exact_domain_metric_minimum_eigenvalue": exact_validation["minimum_eigenvalue_min"],
        "exact_domain_metric_condition_number_max": exact_validation["condition_number_max"],
        "reduced_domain_failure_count": sum(row["status"] == "failed" for row in domain),
        "lda_failure_count_where_reduced_metric_valid": sum(
            row.get("lda_domain_status") == "failed" for row in domain
        ),
        "valid_domain_density_minimum": min(
            float(row["density_assembled_minimum"]) for row in domain if row["status"] == "complete"
        ),
        "main_cc_pvdz_domain_failure_count": sum(row["status"] == "failed" for row in main_domain),
        "metric_correction_count": sum(bool(row["metric_correction_applied"]) for row in finest),
    }
    upper = {
        "zero_field_energy_error_max_abs_au": 1.0e-10,
        "zero_field_mixed_density_relative_error_max": 1.0e-10,
        "zero_field_matched_lower_relative_error_max": 1.0e-10,
        "e1_zero_b_final_density_relative_error_max": 2.0e-10,
        "e1_zero_b_energy_trajectory_relative_error_max": 2.0e-10,
        "e1_zero_b_power_trajectory_relative_error_max": 2.0e-10,
        "c1_stationary_energy_error_max_abs_au": 1.6e-3,
        "c1_stationary_mixed_density_relative_error_max": 1.4e-3,
        "c1_stationary_matched_lower_relative_error_max": 1.9e-3,
        "c1_dynamic_final_density_relative_error_max": 1.7e-3,
        "c1_dynamic_power_balance_residual_max_au": 5.0e-11,
        "c1_dynamic_cross_metric_residual_max": 1.0e-8,
        "c1_dynamic_nonlinear_residual_max": 1.0e-10,
        "lda_failure_count_where_reduced_metric_valid": 0.0,
        "main_cc_pvdz_domain_failure_count": 0.0,
        "metric_correction_count": 0.0,
    }
    lower = {
        "strict_c1_low_field_mixed_density_order": 1.8,
        "p0_low_field_mixed_density_order": 0.8,
        "strict_c1_low_field_energy_order": 1.8,
        "stationary_c1_density_improvement_factor_min": 10.0,
        "dynamic_c1_improvement_over_best_p0_e1_min": 5.0,
        "c1_dynamic_model_to_timestep_ratio_min": 100.0,
        "exact_domain_metric_minimum_eigenvalue": 1.0e-6,
        "reduced_domain_failure_count": 1.0,
        "valid_domain_density_minimum": -1.0e-15,
    }
    checks = {
        **{name: float(metrics[name]) <= bound for name, bound in upper.items()},
        **{name: float(metrics[name]) >= bound for name, bound in lower.items()},
        "all_domain_failures_are_visible_metric_failures": all(
            row["status"] == "complete"
            or (
                row.get("exception_type") == "FormulationError"
                and "metric" in str(row.get("exception", ""))
                and _failed_lambda(row) is not None
            )
            for row in domain
        ),
        "all_dynamics_use_uncorrected_congruence": all(
            row["density_update"] == "coefficient_congruence"
            and not row["metric_correction_applied"]
            for row in finest
        ),
    }

    strict_wins = 0
    resummed_wins = 0
    ties = 0
    for case in _MAGNETIC_CASES:
        for branch in _BRANCHES:
            strict = finest_index[(case, branch, "strict_c1")]
            resummed = finest_index[(case, branch, "density_resummed_c1")]
            for field in (
                "final_mixed_density_relative_error",
                "energy_trajectory_relative_error",
                "power_trajectory_relative_error",
                "power_balance_residual_au",
            ):
                left = float(strict[field])
                right = float(resummed[field])
                if math.isclose(left, right, rel_tol=1.0e-10, abs_tol=1.0e-15):
                    ties += 1
                elif left < right:
                    strict_wins += 1
                else:
                    resummed_wins += 1
    recommendation = {
        "working_action": "strict_c1",
        "reason": (
            "Strict C1 is the safer bounded working approximation: it is the "
            "controlled degree-one action, has the same observed metric and "
            "density domain as density-resummed C1, and the resummation does not "
            "provide a systematic accuracy or stability gain in this campaign."
        ),
        "magnetic_dynamic_metric_wins": {
            "strict_c1": strict_wins,
            "density_resummed_c1": resummed_wins,
            "ties": ties,
        },
        "qualification_domain": (
            "equilateral H3+, cc-pVDZ/weigend, lda,vwn, |B| <= 0.06 a.u. for "
            "stationary comparisons, and the declared 2 a.u. driven trajectories"
        ),
        "non_transferable_warning": (
            "The stress scan is strongly basis conditioned: the first sampled "
            "reduced-metric failure occurs at B=1.2 a.u. in STO-3G, at 0.1 a.u. "
            "for P0/E1 and 0.4 a.u. for C1 in cc-pVDZ, and already at 0.03 a.u. "
            "for every reduced level in aug-cc-pVDZ."
        ),
    }
    return {
        "status": "analyzed_unreviewed",
        "metrics": metrics,
        "proposed_upper_bounds": upper,
        "proposed_lower_bounds": lower,
        "checks": checks,
        "passed_proposed_thresholds": all(checks.values()),
        "domain_boundaries": boundaries,
        "recommendation": recommendation,
        "interpretation_boundaries": [
            (
                "Stationary energy errors remain second order for both P0 and C1 "
                "in this symmetric fixture; first-order C1 chiefly removes the "
                "leading state and lower-matrix errors and need not reduce the "
                "coefficient of every even-in-field energy remainder."
            ),
            (
                "Source-response and power errors are derivatives of the truncated "
                "action.  Differentiating an O(B^2) omitted action term produces an "
                "O(B) source error, so their order need not match the energy order."
            ),
            (
                "The exact Wilson metric stays positive at every stress point.  "
                "The visible failures are reduced-metric domain failures, not an "
                "exact-frame rank loss and not an LDA point-density failure."
            ),
            (
                "The Kohn--Sham branch qualifies the declared auxiliary adiabatic "
                "pure-LDA action, not an exact interacting transverse current."
            ),
            (
                "No clipping, metric repair, endpoint correction, or post-step "
                "projection is included in the qualified trajectories."
            ),
        ],
    }


def _plot_stationary(rows: list[dict[str, Any]], output: Path) -> None:
    panels = (
        ("energy_error_au_max_abs", "stationary energy", "absolute error (Ha)"),
        (
            "mixed_density_relative_error_max_abs",
            "self-consistent mixed density",
            "relative error",
        ),
        (
            "matched_lower_relative_error_max_abs",
            "matched-state lower matrix",
            "relative error",
        ),
        ("power_error_au_max_abs", "action-derived power", "absolute error (Ha/a.u.)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), sharex=True)
    for axis, (metric, title, ylabel) in zip(axes.flat, panels, strict=True):
        for level in _LEVELS:
            selected = sorted(
                (row for row in rows if row["level"] == level and float(row["field_au"]) > 0.0),
                key=lambda row: float(row["field_au"]),
            )
            axis.loglog(
                [row["field_au"] for row in selected],
                [max(float(row[metric]), 1.0e-18) for row in selected],
                color=_COLORS[level],
                marker=_MARKERS[level],
                label=_LEVEL_LABELS[level],
            )
        axis.set_title(title)
        axis.set_xlabel(r"magnetic field $B_z$ (a.u.)")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(ncol=2, fontsize=8)
    figure.suptitle("NQ7 stationary reduced-action error relative to exact Wilson")
    figure.tight_layout()
    figure.savefig(output / "stationary_field_errors.png", dpi=180)
    plt.close(figure)


def _plot_dynamics(rows: list[dict[str, Any]], output: Path) -> None:
    metrics = (
        ("final_mixed_density_relative_error", "final mixed density"),
        ("energy_trajectory_relative_error", "energy trajectory"),
        ("power_trajectory_relative_error", "power trajectory"),
    )
    groups = [(case, branch) for case in _CASES for branch in _BRANCHES]
    x = np.arange(len(groups), dtype=float)
    width = 0.19
    figure, axes = plt.subplots(3, 1, figsize=(13.2, 11.0), sharex=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        for index, level in enumerate(_LEVELS):
            values = [
                max(
                    float(
                        next(
                            row
                            for row in rows
                            if row["case"] == case
                            and row["branch"] == branch
                            and row["level"] == level
                        )[metric]
                    ),
                    1.0e-15,
                )
                for case, branch in groups
            ]
            axis.bar(
                x + (index - 1.5) * width,
                values,
                width,
                color=_COLORS[level],
                edgecolor="black" if level == "density_resummed_c1" else "none",
                linewidth=0.7,
                hatch="//" if level == "density_resummed_c1" else None,
                label=_LEVEL_LABELS[level],
            )
        axis.set_yscale("log")
        axis.set_ylabel("relative error")
        axis.set_title(title)
        axis.grid(True, axis="y", which="both", alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xticks(
        x,
        [f"{_CASE_LABELS[case]}\n{_BRANCH_LABELS[branch]}" for case, branch in groups],
        rotation=20,
        ha="right",
    )
    figure.suptitle(r"NQ7 finest trajectories ($\Delta t=0.0625$ a.u.)")
    figure.tight_layout()
    figure.savefig(output / "dynamic_model_errors.png", dpi=180)
    plt.close(figure)


def _plot_domain(
    rows: list[dict[str, Any]], exact_rows: list[dict[str, Any]], output: Path
) -> None:
    exact_index = defaultdict(list)
    for row in exact_rows:
        exact_index[row["basis"]].append(row)
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 5.0), sharey=True)
    for axis, basis in zip(axes, campaign._DOMAIN_BASES, strict=True):
        exact = sorted(exact_index[basis], key=lambda row: float(row["field_au"]))
        axis.plot(
            [row["field_au"] for row in exact],
            [row["exact_metric_minimum_eigenvalue"] for row in exact],
            color=_COLORS["exact"],
            marker="x",
            linewidth=2.0,
            label="exact Wilson",
        )
        for level in _LEVELS:
            selected = sorted(
                (row for row in rows if row["basis"] == basis and row["level"] == level),
                key=lambda row: float(row["field_au"]),
            )
            axis.plot(
                [row["field_au"] for row in selected],
                [_failed_lambda(row) for row in selected],
                color=_COLORS[level],
                marker=_MARKERS[level],
                label=_LEVEL_LABELS[level],
            )
        axis.axhline(0.0, color="0.35", linewidth=1.0, linestyle=":")
        axis.set_yscale("symlog", linthresh=1.0e-4)
        axis.set_title(basis)
        axis.set_xlabel(r"magnetic field $B_z$ (a.u.)")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"minimum overlap eigenvalue $\lambda_{\min}$")
    axes[0].legend(fontsize=7)
    figure.suptitle("NQ7 basis-dependent reduced-metric domain")
    figure.tight_layout()
    figure.savefig(output / "domain_metric_boundaries.png", dpi=180)
    plt.close(figure)


def _plot_timestep_separation(rows: list[dict[str, Any]], output: Path) -> None:
    selected = [
        row
        for row in rows
        if "n16_to_n32_final_density_difference" in row and row["case"] in _MAGNETIC_CASES
    ]
    labels = [
        f"{_CASE_LABELS[row['case']]}\n{_BRANCH_LABELS[row['branch']]}\n"
        f"{_LEVEL_LABELS[row['level']]}"
        for row in selected
    ]
    x = np.arange(len(selected))
    figure, axis = plt.subplots(figsize=(13.2, 5.6))
    axis.bar(
        x - 0.2,
        [row["n16_to_n32_final_density_difference"] for row in selected],
        0.4,
        color="#b8b8b8",
        edgecolor="#4a4a4a",
        label=r"$n=16$ vs $n=32$ timestep difference",
    )
    axis.bar(
        x + 0.2,
        [row["n32_exact_model_error"] for row in selected],
        0.4,
        color="#7a5195",
        label="reduced vs exact model error",
    )
    axis.set_yscale("log")
    axis.set_ylabel("final mixed-density relative difference")
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.grid(True, axis="y", which="both", alpha=0.25)
    axis.legend(fontsize=8)
    axis.set_title("NQ7 model differences are resolved from timestep uncertainty")
    figure.tight_layout()
    figure.savefig(output / "model_vs_timestep_error.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exact-domain-cache", type=Path)
    arguments = parser.parse_args()
    raw = arguments.raw.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis directory: {output}")
    output.mkdir(parents=True)

    raw_hashes = _authenticate(raw)
    result = _load_json(raw / "result.json")
    matrix_validation = _validate_records(raw, result)
    stationary = list(result["records"]["stationary"])
    domain = list(result["records"]["domain"])
    dynamics = list(result["records"]["dynamics"])
    if arguments.exact_domain_cache is None:
        exact_domain = _compute_exact_domain(raw)
        exact_domain_source: dict[str, Any] = {"mode": "independent_recomputation"}
    else:
        cache_path = arguments.exact_domain_cache.expanduser().resolve()
        exact_domain = _load_exact_domain(cache_path)
        exact_domain_source = {
            "mode": "authenticated_cache_reuse",
            "path": str(cache_path),
            "sha256": _sha256(cache_path),
        }
    exact_validation = _validate_exact_domain(exact_domain, domain)
    stationary_field = _stationary_by_field(stationary)
    boundaries = _domain_boundaries(domain)
    finest, dynamic_secondary = _dynamic_analysis(raw, dynamics)
    analysis = _analyze(
        stationary,
        stationary_field,
        domain,
        boundaries,
        exact_validation,
        finest,
        dynamic_secondary,
    )

    _write_csv(output / "stationary_records.csv", stationary)
    _write_csv(output / "stationary_field_errors.csv", stationary_field)
    _write_csv(output / "domain_records.csv", domain)
    _write_csv(output / "domain_boundaries.csv", boundaries)
    _write_csv(output / "exact_domain_metrics.csv", exact_domain)
    _write_csv(output / "dynamic_finest_records.csv", finest)
    _write_csv(output / "dynamic_secondary_comparisons.csv", dynamic_secondary)
    _plot_stationary(stationary_field, output)
    _plot_dynamics(finest, output)
    _plot_domain(domain, exact_domain, output)
    _plot_timestep_separation(dynamic_secondary, output)

    table_names = (
        "stationary_records.csv",
        "stationary_field_errors.csv",
        "domain_records.csv",
        "domain_boundaries.csv",
        "exact_domain_metrics.csv",
        "dynamic_finest_records.csv",
        "dynamic_secondary_comparisons.csv",
    )
    figure_names = (
        "stationary_field_errors.png",
        "dynamic_model_errors.png",
        "domain_metric_boundaries.png",
        "model_vs_timestep_error.png",
    )
    analysis["raw_root"] = str(raw)
    analysis["raw_hashes"] = raw_hashes
    analysis["matrix_validation"] = matrix_validation
    analysis["exact_domain_validation"] = exact_validation
    analysis["exact_domain_source"] = exact_domain_source
    analysis["tables_sha256"] = {name: _sha256(output / name) for name in table_names}
    analysis["figures_sha256"] = {name: _sha256(output / name) for name in figure_names}
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "passed_proposed_thresholds": analysis["passed_proposed_thresholds"],
                "summary_sha256": _sha256(summary_path),
            },
            indent=2,
        )
    )
    if not analysis["passed_proposed_thresholds"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
