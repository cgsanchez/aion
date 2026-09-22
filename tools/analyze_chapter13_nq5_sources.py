#!/usr/bin/env python3
"""Authenticate and analyze one Chapter 13 NQ5 source campaign."""

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


def _by_branch(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["branch"])].append(row)
    return dict(grouped)


def _coarse_order(rows: list[dict[str, Any]]) -> float:
    selected = sorted(rows, key=lambda row: float(row["step"]), reverse=True)[:3]
    steps = np.asarray([float(row["step"]) for row in selected])
    errors = np.asarray([float(row["absolute_error"]) for row in selected])
    return float(np.polyfit(np.log(steps), np.log(errors), 1)[0])


def _level_difference(
    rows: list[dict[str, Any]],
    field: str,
    coarse: int,
    fine: int,
) -> list[float]:
    indexed = {
        (str(row["branch"]), int(row["grid_level"])): float(row[field])
        for row in rows
    }
    return [
        abs(indexed[(branch, coarse)] - indexed[(branch, fine)])
        for branch in ("hartree", "kohn_sham_lda")
    ]


def _refinement_difference(
    rows: list[dict[str, Any]],
    key: str,
    field: str,
    left: float,
    right: float,
) -> list[float]:
    indexed = {
        (str(row["branch"]), float(row[key])): float(row[field]) for row in rows
    }
    return [
        abs(indexed[(branch, left)] - indexed[(branch, right)])
        for branch in ("hartree", "kohn_sham_lda")
    ]


def _analyze(result: dict[str, Any]) -> dict[str, Any]:
    charge = result["charge"]
    currents = result["off_shell_weak_current"]
    finite = result["source_finite_differences"]
    shells = result["on_shell_decomposition"]
    wards = result["off_shell_ward"]
    continuity = result["weak_continuity"]
    paths = result["path_refinement"]
    ranks = result["ri_rank_refinement"]
    finite_groups = _by_branch(finite)

    closure_identity = [
        abs(
            float(row["closure_current_pairing"])
            + float(row["hartree_energy_source_direction"])
            + float(row["xc_energy_source_direction"])
        )
        for row in currents
    ]
    action_composition = [
        abs(
            float(row["fixed_coordinate_total_pairing"])
            - float(row["one_electron_source_pairing"])
            - float(row["closure_current_pairing"])
        )
        for row in currents
    ]
    level5_charge = [row for row in charge if int(row["grid_level"]) == 5]
    metrics = {
        "metric_particle_number_residual_max": max(
            float(row["electron_count_residual"]) for row in charge
        ),
        "level5_grid_metric_charge_residual_max": max(
            float(row["grid_metric_charge_residual"]) for row in level5_charge
        ),
        "charge_density_imaginary_max_abs": max(
            float(row["density_imaginary_max_abs"]) for row in charge
        ),
        "complete_action_source_best_error_max": max(
            min(float(row["absolute_error"]) for row in rows)
            for rows in finite_groups.values()
        ),
        "complete_action_source_coarse_order_min": min(
            _coarse_order(rows) for rows in finite_groups.values()
        ),
        "closure_action_identity_residual_max": max(closure_identity),
        "complete_action_composition_residual_max": max(action_composition),
        "off_shell_decomposition_residual_max": max(
            abs(float(row["on_shell_decomposition_residual"])) for row in currents
        ),
        "stationary_tangential_pairing_max_abs": max(
            abs(float(row["tangential_pairing"])) for row in shells
        ),
        "stationary_decomposition_residual_max": max(
            abs(float(row["on_shell_decomposition_residual"])) for row in shells
        ),
        "ward_residual_max": max(float(row["ward_residual"]) for row in wards),
        "finite_region_continuity_residual_max": max(
            float(row["finite_region_residual"]) for row in continuity
        ),
        "global_charge_residual_max": max(
            float(row["global_charge_residual"]) for row in continuity
        ),
        "level4_to_level5_current_difference_max": max(
            _level_difference(
                currents,
                "fixed_coordinate_total_pairing",
                4,
                5,
            )
        ),
        "path_order24_to40_current_difference_max": max(
            _refinement_difference(
                paths,
                "path_order",
                "fixed_coordinate_total_pairing",
                24.0,
                40.0,
            )
        ),
        "ri_threshold_1e6_to1e9_current_difference_max": max(
            _refinement_difference(
                ranks,
                "absolute_threshold",
                "fixed_coordinate_total_pairing",
                1.0e-6,
                1.0e-9,
            )
        ),
        "off_shell_matter_residual_relative_norm_min": min(
            float(row["lower_coefficient_residual_relative_norm"]) for row in wards
        ),
        "off_shell_embedding_pairing_min_abs": min(
            abs(float(row["complete_embedding_pairing"]))
            for row in currents
            if int(row["grid_level"]) == 4
        ),
        "closure_current_pairing_min_abs": min(
            abs(float(row["closure_current_pairing"]))
            for row in currents
            if int(row["grid_level"]) == 4
        ),
        "stationary_normal_subspace_pairing_min_abs": min(
            abs(float(row["normal_subspace_pairing"])) for row in shells
        ),
        "ward_source_pairing_min_abs": min(
            abs(float(row["source_pairing"])) for row in wards
        ),
    }
    upper_bounds = {
        "metric_particle_number_residual_max": 1.0e-11,
        "level5_grid_metric_charge_residual_max": 2.0e-9,
        "charge_density_imaginary_max_abs": 2.0e-14,
        "complete_action_source_best_error_max": 1.0e-9,
        "closure_action_identity_residual_max": 2.0e-14,
        "complete_action_composition_residual_max": 2.0e-14,
        "off_shell_decomposition_residual_max": 2.0e-14,
        "stationary_tangential_pairing_max_abs": 2.0e-12,
        "stationary_decomposition_residual_max": 2.0e-14,
        "ward_residual_max": 2.0e-12,
        "finite_region_continuity_residual_max": 2.0e-12,
        "global_charge_residual_max": 2.0e-12,
        "level4_to_level5_current_difference_max": 1.0e-9,
        "path_order24_to40_current_difference_max": 1.0e-12,
        "ri_threshold_1e6_to1e9_current_difference_max": 1.0e-12,
    }
    lower_bounds = {
        "complete_action_source_coarse_order_min": 1.9,
        "off_shell_matter_residual_relative_norm_min": 1.0e-1,
        "off_shell_embedding_pairing_min_abs": 1.0e-3,
        "closure_current_pairing_min_abs": 1.0e-6,
        "stationary_normal_subspace_pairing_min_abs": 1.0e-8,
        "ward_source_pairing_min_abs": 1.0e-3,
    }
    checks = {
        **{name: metrics[name] <= limit for name, limit in upper_bounds.items()},
        **{name: metrics[name] >= limit for name, limit in lower_bounds.items()},
    }
    charge_by_branch = _by_branch(charge)
    checks["grid_charge_converges_monotonically"] = all(
        all(
            right < left
            for left, right in zip(
                [
                    float(row["grid_metric_charge_residual"])
                    for row in sorted(rows, key=lambda row: int(row["grid_level"]))
                ],
                [
                    float(row["grid_metric_charge_residual"])
                    for row in sorted(rows, key=lambda row: int(row["grid_level"]))
                ][1:],
                strict=False,
            )
        )
        for rows in charge_by_branch.values()
    )
    checks["ri_rank_is_stable"] = len(
        {int(row["retained_rank"]) for row in ranks}
    ) == 1
    return {
        "status": "analyzed_unreviewed",
        "metrics": metrics,
        "proposed_upper_bound_thresholds": upper_bounds,
        "proposed_lower_bound_diagnostics": lower_bounds,
        "checks": checks,
        "passed_proposed_thresholds": all(checks.values()),
        "interpretation_boundaries": [
            (
                "The Kohn--Sham result is the auxiliary adiabatic-KS action source "
                "current, not an interacting physical transverse current."
            ),
            (
                "The reported complete current is a weak source pairing; no pointwise "
                "complete subspace-response current is claimed."
            ),
            (
                "The fixed-coordinate off-shell source includes a tangential "
                "coefficient-variation term.  The on-shell current is the "
                "ambient-minimal plus normal-subspace pairing after the tangential "
                "term is removed."
            ),
            "Graph, P0, E1, and C1 currents were not evaluated in NQ5.",
        ],
    }


def _plot_source_difference(result: dict[str, Any], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    styles = {
        "hartree": ("Hartree", "o", "-"),
        "kohn_sham_lda": ("LDA KS", "s", "--"),
    }
    for branch, rows in _by_branch(result["source_finite_differences"]).items():
        selected = sorted(rows, key=lambda row: float(row["step"]), reverse=True)
        label, marker, linestyle = styles[branch]
        axis.loglog(
            [float(row["step"]) for row in selected],
            [float(row["absolute_error"]) for row in selected],
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
    reference_steps = np.asarray((3.0e-2, 1.0e-2, 3.0e-3))
    axis.loglog(
        reference_steps,
        4.0e-7 * (reference_steps / reference_steps[0]) ** 2,
        color="0.4",
        linestyle=":",
        label=r"$O(\epsilon^2)$ guide",
    )
    axis.set_title("Complete fixed-history action-source derivative")
    axis.set_xlabel(r"central-difference step $\epsilon$")
    axis.set_ylabel("absolute derivative error")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "source_finite_difference.png", dpi=180)
    plt.close(figure)


def _plot_current_decomposition(result: dict[str, Any], output: Path) -> None:
    rows = {
        str(row["branch"]): row
        for row in result["off_shell_weak_current"]
        if int(row["grid_level"]) == 4
    }
    shells = {str(row["branch"]): row for row in result["on_shell_decomposition"]}
    branches = ("hartree", "kohn_sham_lda")
    labels = ("Hartree", "LDA KS")
    positions = np.arange(2)
    width = 0.25
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    axes[0].bar(
        positions - width / 2,
        [float(rows[branch]["fixed_coordinate_total_pairing"]) for branch in branches],
        width,
        label=r"fixed-coordinate $J_{\rm total}$",
    )
    axes[0].bar(
        positions + width / 2,
        [float(rows[branch]["tangential_pairing"]) for branch in branches],
        width,
        label=r"tangential $J_{\parallel}$",
    )
    axes[0].set_title("Off shell: large tangent contribution")
    axes[0].set_ylabel("weak pairing")
    axes[0].set_xticks(positions, labels)
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].bar(
        positions - width,
        [
            float(rows[branch]["fixed_coordinate_total_pairing"])
            - float(rows[branch]["tangential_pairing"])
            for branch in branches
        ],
        width,
        label=r"$J_{\rm total}-J_{\parallel}$",
    )
    axes[1].bar(
        positions,
        [float(rows[branch]["ambient_minimal_pairing"]) for branch in branches],
        width,
        label=r"ambient minimal",
    )
    axes[1].bar(
        positions + width,
        [float(rows[branch]["normal_subspace_pairing"]) for branch in branches],
        width,
        label=r"normal subspace",
    )
    axes[1].set_title("Off shell: Gauss decomposition")
    axes[1].set_ylabel("weak pairing")
    axes[1].set_xticks(positions, labels)
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].bar(
        positions - width,
        [float(shells[branch]["ambient_minimal_pairing"]) for branch in branches],
        width,
        label="ambient minimal",
    )
    axes[2].bar(
        positions,
        [float(shells[branch]["normal_subspace_pairing"]) for branch in branches],
        width,
        label="normal subspace",
    )
    axes[2].bar(
        positions + width,
        [float(shells[branch]["tangential_pairing"]) for branch in branches],
        width,
        label="tangential",
    )
    axes[2].set_title("Accepted stationary coefficient shell")
    axes[2].set_ylabel("weak pairing")
    axes[2].set_xticks(positions, labels)
    axes[2].axhline(0.0, color="black", linewidth=0.7)
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "current_gauss_decomposition.png", dpi=180)
    plt.close(figure)


def _plot_identities_and_refinement(result: dict[str, Any], output: Path) -> None:
    branches = ("hartree", "kohn_sham_lda")
    labels = ("Hartree", "LDA KS")
    positions = np.arange(2)
    width = 0.3
    wards = {str(row["branch"]): row for row in result["off_shell_ward"]}
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    axes[0].bar(
        positions - width / 2,
        [float(wards[branch]["source_pairing"]) for branch in branches],
        width,
        label="source variation",
    )
    axes[0].bar(
        positions + width / 2,
        [float(wards[branch]["matter_pairing"]) for branch in branches],
        width,
        label="matter variation",
    )
    axes[0].set_title("Deliberately off-shell Ward test")
    axes[0].set_ylabel("action-direction pairing")
    axes[0].set_xticks(positions, labels)
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    continuity = _by_branch(result["weak_continuity"])
    axes[1].semilogy(
        positions,
        [max(float(wards[branch]["ward_residual"]), 1.0e-18) for branch in branches],
        marker="o",
        label="Ward residual",
    )
    axes[1].semilogy(
        positions,
        [
            max(
                max(float(row["finite_region_residual"]) for row in continuity[branch]),
                1.0e-18,
            )
            for branch in branches
        ],
        marker="s",
        label="finite-region continuity",
    )
    axes[1].semilogy(
        positions,
        [
            max(
                max(float(row["global_charge_residual"]) for row in continuity[branch]),
                1.0e-18,
            )
            for branch in branches
        ],
        marker="^",
        label="global charge",
    )
    axes[1].set_title("Ward and continuity closure")
    axes[1].set_ylabel("absolute residual")
    axes[1].set_xticks(positions, labels)
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)

    current_rows = result["off_shell_weak_current"]
    for branch, label, marker in (
        ("hartree", "Hartree", "o"),
        ("kohn_sham_lda", "LDA KS", "s"),
    ):
        selected = sorted(
            (row for row in current_rows if row["branch"] == branch),
            key=lambda row: int(row["grid_level"]),
        )
        finest = float(selected[-1]["fixed_coordinate_total_pairing"])
        axes[2].semilogy(
            [int(row["grid_level"]) for row in selected],
            [
                max(
                    abs(float(row["fixed_coordinate_total_pairing"]) - finest),
                    1.0e-18,
                )
                for row in selected
            ],
            marker=marker,
            label=label,
        )
    axes[2].set_title("Grid refinement of weak source")
    axes[2].set_xlabel("unpruned PySCF grid level")
    axes[2].set_ylabel("difference from level 5")
    axes[2].set_xticks((3, 4, 5))
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "ward_continuity_and_refinement.png", dpi=180)
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
    if result.get("schema") != "aion.chapter13-nq5-sources":
        raise RuntimeError("raw result has the wrong schema")
    summary = _analyze(result)
    summary["raw_root"] = str(raw)
    summary["raw_hashes"] = hashes
    _plot_source_difference(result, output)
    _plot_current_decomposition(result, output)
    _plot_identities_and_refinement(result, output)
    figure_names = (
        "source_finite_difference.png",
        "current_gauss_decomposition.png",
        "ward_continuity_and_refinement.png",
    )
    summary["figures"] = {
        name: _sha256(output / name) for name in figure_names
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
