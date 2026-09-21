#!/usr/bin/env python3
"""Verify and summarize one Chapter 13 NQ1 Wilson-density execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_THRESHOLDS = {
    "density_imaginary_max_abs": 1.0e-14,
    "density_real_minimum": -1.0e-14,
    "density_direct_factorized_relative": 1.0e-13,
    "overlap_direct_factorized_relative": 1.0e-13,
    "stable_overlap_oracle_relative": 1.0e-13,
    "finest_particle_integral_stable_abs": 5.0e-10,
    "gauge_density_relative_l2_residual": 1.0e-12,
    "gauge_particle_stable_absolute_residual": 1.0e-12,
    "coefficient_frame_relative_l2_residual": 1.0e-12,
    "matter_best_central_relative_l2_error": 1.0e-10,
    "matter_forward_decade_ratio_relative_error": 1.0e-6,
    "source_best_central_relative_l2_error": 5.0e-9,
    "source_best_central_maximum_absolute_error": 1.0e-10,
    "source_first_pair_order_minimum": 1.9,
    "direction_imaginary_max_abs": 1.0e-14,
    "source_particle_stable_absolute_residual": 5.0e-12,
}


def _pyplot() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/aion-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/aion-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _verify_hashes(root: Path) -> dict[str, str]:
    completed = _read_json(root / "completed.json")
    provenance = _read_json(root / "provenance.json")
    actual_result = _sha256(root / "result.json")
    actual_provenance = _sha256(root / "provenance.json")
    if completed["result_sha256"] != actual_result:
        raise ValueError("result.json hash does not match completed.json")
    if completed["provenance_sha256"] != actual_provenance:
        raise ValueError("provenance.json hash does not match completed.json")
    for name, expected in provenance["artifacts_sha256"].items():
        actual = _sha256(root / name)
        if actual != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    return {
        "result_sha256": actual_result,
        "provenance_sha256": actual_provenance,
        "completed_sha256": _sha256(root / "completed.json"),
    }


def _by_direction(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(str(row["direction"]), []).append(row)
    return result


def _complex_real(value: dict[str, Any]) -> float:
    return float(value["real"])


def _summarize(result: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_system: list[dict[str, Any]] = []
    for system in result["systems"]:
        grid = system["grid_refinement"]
        if [row["level"] for row in grid] != [1, 2, 3, 4, 5]:
            raise ValueError(f"{system['system']} does not contain grid levels 1--5")
        finest = grid[-1]
        matter_groups = _by_direction(system["matter_derivatives"])
        source_groups = _by_direction(system["source_derivatives"])
        matter: dict[str, Any] = {}
        for direction, rows in matter_groups.items():
            row_by_step = {float(row["step"]): row for row in rows}
            ratio = float(row_by_step[0.1]["forward_relative_l2_error"]) / float(
                row_by_step[0.01]["forward_relative_l2_error"]
            )
            matter[direction] = {
                "best_central_relative_l2_error": min(
                    float(row["central_relative_l2_error"]) for row in rows
                ),
                "forward_decade_ratio": ratio,
                "forward_decade_ratio_relative_error": abs(ratio / 10.0 - 1.0),
                "analytic_imaginary_max_abs": max(
                    float(row["analytic_imaginary_max_abs"]) for row in rows
                ),
            }
        source: dict[str, Any] = {}
        for direction, rows in source_groups.items():
            first, second = rows[0], rows[1]
            order = math.log(
                float(first["central_relative_l2_error"])
                / float(second["central_relative_l2_error"])
            ) / math.log(float(first["step"]) / float(second["step"]))
            source[direction] = {
                "best_central_relative_l2_error": min(
                    float(row["central_relative_l2_error"]) for row in rows
                ),
                "best_central_maximum_absolute_error": min(
                    float(row["central_maximum_absolute_error"]) for row in rows
                ),
                "first_pair_order": order,
                "analytic_imaginary_max_abs": max(
                    float(row["analytic_imaginary_max_abs"]) for row in rows
                ),
                "particle_stable_absolute_residual": abs(
                    _complex_real(first["analytic_particle_integral"])
                    - _complex_real(first["analytic_particle_stable_metric"])
                ),
            }
        per_system.append(
            {
                "system": system["system"],
                "finest_grid": finest,
                "maximum_density_imaginary_abs": max(
                    max(float(row["density_direct_imaginary_max_abs"]) for row in grid),
                    max(float(row["density_factorized_imaginary_max_abs"]) for row in grid),
                ),
                "minimum_density_real": min(
                    min(float(row["density_direct_real_minimum"]) for row in grid),
                    min(float(row["density_factorized_real_minimum"]) for row in grid),
                ),
                "maximum_density_direct_factorized_relative": max(
                    float(row["density_direct_factorized_relative"]) for row in grid
                ),
                "maximum_overlap_direct_factorized_relative": max(
                    float(row["overlap_direct_factorized_relative"]) for row in grid
                ),
                "maximum_stable_overlap_oracle_relative": max(
                    float(row["stable_overlap_oracle_relative"]) for row in grid
                ),
                "invariance": system["invariance"],
                "matter": matter,
                "source": source,
            }
        )

    observed = {
        "density_imaginary_max_abs": max(
            row["maximum_density_imaginary_abs"] for row in per_system
        ),
        "density_real_minimum": min(row["minimum_density_real"] for row in per_system),
        "density_direct_factorized_relative": max(
            row["maximum_density_direct_factorized_relative"] for row in per_system
        ),
        "overlap_direct_factorized_relative": max(
            row["maximum_overlap_direct_factorized_relative"] for row in per_system
        ),
        "stable_overlap_oracle_relative": max(
            row["maximum_stable_overlap_oracle_relative"] for row in per_system
        ),
        "finest_particle_integral_stable_abs": max(
            float(row["finest_grid"]["particle_integral_stable_abs"])
            for row in per_system
        ),
        "gauge_density_relative_l2_residual": max(
            float(row["invariance"]["electromagnetic_gauge"]["density_relative_l2_residual"])
            for row in per_system
        ),
        "gauge_particle_stable_absolute_residual": max(
            float(
                row["invariance"]["electromagnetic_gauge"][
                    "particle_stable_absolute_residual"
                ]
            )
            for row in per_system
        ),
        "coefficient_frame_relative_l2_residual": max(
            float(row["invariance"]["coefficient_frame"]["relative_l2_residual"])
            for row in per_system
        ),
        "matter_best_central_relative_l2_error": max(
            float(values["best_central_relative_l2_error"])
            for row in per_system
            for values in row["matter"].values()
        ),
        "matter_forward_decade_ratio_relative_error": max(
            float(values["forward_decade_ratio_relative_error"])
            for row in per_system
            for values in row["matter"].values()
        ),
        "source_best_central_relative_l2_error": max(
            float(values["best_central_relative_l2_error"])
            for row in per_system
            for values in row["source"].values()
        ),
        "source_best_central_maximum_absolute_error": max(
            float(values["best_central_maximum_absolute_error"])
            for row in per_system
            for values in row["source"].values()
        ),
        "source_first_pair_order_minimum": min(
            float(values["first_pair_order"])
            for row in per_system
            for values in row["source"].values()
        ),
        "direction_imaginary_max_abs": max(
            float(values["analytic_imaginary_max_abs"])
            for row in per_system
            for family in (row["matter"], row["source"])
            for values in family.values()
        ),
        "source_particle_stable_absolute_residual": max(
            float(values["particle_stable_absolute_residual"])
            for row in per_system
            for values in row["source"].values()
        ),
    }
    return observed, per_system


def _checks(observed: dict[str, float]) -> dict[str, bool]:
    checks = {
        name: (
            observed[name] >= threshold
            if name in {"density_real_minimum", "source_first_pair_order_minimum"}
            else observed[name] <= threshold
        )
        for name, threshold in _THRESHOLDS.items()
    }
    return checks


def _grid_figure(result: dict[str, Any], output: Path) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for system in result["systems"]:
        rows = system["grid_refinement"]
        axis.semilogy(
            [row["level"] for row in rows],
            [row["particle_integral_stable_abs"] for row in rows],
            marker="o",
            label=system["system"].upper(),
        )
    axis.axhline(
        _THRESHOLDS["finest_particle_integral_stable_abs"],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="NQ1 tolerance",
    )
    axis.set_xlabel("unpruned PySCF grid level")
    axis.set_ylabel(r"$|\int n_{\rm W}-\mathrm{Tr}(PS_{\rm stable})|$")
    axis.set_title("Wilson-density normalization refinement")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.savefig(output / "normalization_refinement.png", dpi=180)
    plt.close(figure)


def _derivative_figure(result: dict[str, Any], output: Path) -> None:
    plt = _pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    for column, system in enumerate(result["systems"]):
        matter = _by_direction(system["matter_derivatives"])
        for direction, rows in matter.items():
            steps = [row["step"] for row in rows]
            axes[0, column].loglog(
                steps,
                [row["forward_relative_l2_error"] for row in rows],
                marker="o",
                label=f"{direction}, forward",
            )
            axes[0, column].loglog(
                steps,
                [row["central_relative_l2_error"] for row in rows],
                marker="s",
                linestyle="--",
                label=f"{direction}, central",
            )
        source = _by_direction(system["source_derivatives"])
        for direction, rows in source.items():
            axes[1, column].loglog(
                [row["step"] for row in rows],
                [row["central_relative_l2_error"] for row in rows],
                marker="o",
                label=direction.replace("_", " "),
            )
        axes[0, column].set_title(f"{system['system'].upper()}: matter directions")
        axes[1, column].set_title(f"{system['system'].upper()}: fixed-history source")
        for row in range(2):
            axes[row, column].grid(True, which="both", alpha=0.25)
            axes[row, column].set_xlabel("finite-difference step")
            axes[row, column].set_ylabel("scaled L2 error")
            axes[row, column].invert_xaxis()
            axes[row, column].legend(fontsize=8)
    figure.savefig(output / "derivative_sequences.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    execution = arguments.execution.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite analysis directory: {output}")
    hashes = _verify_hashes(execution)
    result = _read_json(execution / "result.json")
    observed, per_system = _summarize(result)
    checks = _checks(observed)
    passed = all(checks.values())
    output.mkdir(parents=True)
    _grid_figure(result, output)
    _derivative_figure(result, output)
    summary = {
        "schema": "aion.chapter13-nq1-analysis",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "passed_proposed_thresholds": passed,
        "thresholds": _THRESHOLDS,
        "observed": observed,
        "checks": checks,
        "per_system": per_system,
        "raw_execution": str(execution),
        "raw_hashes": hashes,
        "analysis_timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "command": " ".join(shlex.quote(value) for value in (sys.executable, *sys.argv)),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "passed_proposed_thresholds": passed,
                "summary_sha256": _sha256(summary_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
