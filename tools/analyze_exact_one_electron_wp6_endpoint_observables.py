#!/usr/bin/env python3
"""Evaluate physical field-free endpoint observables for the WP6 H3 trajectory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import run_exact_one_electron_wp6_linear as linear
import run_exact_one_electron_wp6_mixed_magnus_n128_diagnostics as diagnostic
import scipy.linalg

from aion.electronic_structure import prepare_one_electron_ao_reference

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DEFAULT_MIXED = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "exact_one_electron_qualification/"
    "wp6_mixed_magnus_20260918T231057Z_38ce83eb2fe9"
)
_DEFAULT_N64 = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "exact_one_electron_qualification/"
    "wp6_mixed_magnus_n64_20260919T120325Z_57ef2fb6d40e"
)
_MODEL_LABELS = {
    "exact": "EX",
    "p0": "P0",
    "e1": "E1",
    "geometric_b1": "gB1",
    "full_b1": "B1",
    "complete_first_order": "C1",
}
_COLORS = {
    "exact": "#1f2933",
    "p0": "#3569a8",
    "e1": "#d97724",
    "geometric_b1": "#7a5ea8",
    "full_b1": "#3f8f72",
    "complete_first_order": "#b44f68",
}
_MARKERS = {
    "exact": "o",
    "p0": "s",
    "e1": "^",
    "geometric_b1": "D",
    "full_b1": "v",
    "complete_first_order": "P",
}
_LINESTYLES = {
    "exact": "-",
    "p0": "-",
    "e1": "--",
    "geometric_b1": "-",
    "full_b1": "--",
    "complete_first_order": "--",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-execution", type=Path, default=_DEFAULT_MIXED)
    parser.add_argument("--n64-execution", type=Path, default=_DEFAULT_N64)
    parser.add_argument("--n128-execution", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("endpoint observable table cannot be empty")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _reference_operators() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    fixture = diagnostic._json(linear._WP5_FIXTURE)
    coordinates = list(fixture["geometries"]["distorted"]["coordinates_au"])
    reference = prepare_one_electron_ao_reference(
        linear._h3_config(coordinates, "cc-pvdz", [0.0, 0.0, 0.0])
    )
    operators = reference.core_operators
    metric = np.asarray(operators.overlap, dtype=np.complex128)
    mechanical = np.asarray(
        operators.kinetic + operators.nuclear_attraction,
        dtype=np.complex128,
    )
    eigenvalues, eigenvectors = scipy.linalg.eigh(mechanical, metric)
    ground = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
    return metric, mechanical, ground, float(eigenvalues[0])


def _observable_rows(
    arrays_for_intervals: dict[int, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    metric, mechanical, ground, ground_energy = _reference_operators()
    prefix = "h3_distorted_level4_above_threshold"
    output: list[dict[str, Any]] = []
    for intervals, arrays in sorted(arrays_for_intervals.items()):
        exact = arrays[f"{prefix}__exact__n{intervals}__P"][-1]
        for model in linear._MODEL_NAMES:
            density = arrays[f"{prefix}__{model}__n{intervals}__P"][-1]
            energy = float(np.real(np.trace(density @ mechanical)))
            electron_number = float(np.real(np.trace(density @ metric)))
            survival = float(
                np.real(ground.conj().T @ metric @ density @ metric @ ground)
            )
            delta = density - exact
            distance_squared = float(
                np.real(np.trace(delta.conj().T @ metric @ delta @ metric))
            )
            output.append(
                {
                    "intervals": intervals,
                    "step_au": 4.0 / intervals,
                    "model": model,
                    "model_label": _MODEL_LABELS[model],
                    "electron_number": electron_number,
                    "field_free_ground_energy_ha": ground_energy,
                    "final_field_free_energy_ha": energy,
                    "absorbed_energy_ha": energy - ground_energy,
                    "ground_state_survival": survival,
                    "excitation_probability": 1.0 - survival,
                    "metric_hilbert_schmidt_distance_from_exact": math_sqrt_nonnegative(
                        distance_squared
                    ),
                }
            )
    return output


def math_sqrt_nonnegative(value: float) -> float:
    if value < -1.0e-12:
        raise ValueError(f"negative Hilbert-Schmidt norm square: {value}")
    return float(np.sqrt(max(0.0, value)))


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.8, 5.2), constrained_layout=True)
    for model in linear._MODEL_NAMES:
        selected = sorted(
            (row for row in rows if row["model"] == model),
            key=lambda row: int(row["intervals"]),
        )
        intervals = [int(row["intervals"]) for row in selected]
        style = {
            "color": _COLORS[model],
            "marker": _MARKERS[model],
            "linestyle": _LINESTYLES[model],
            "linewidth": 1.7,
            "markersize": 5.5,
            "label": _MODEL_LABELS[model],
        }
        axes[0].plot(intervals, [row["absorbed_energy_ha"] for row in selected], **style)
        axes[1].plot(
            intervals,
            [row["excitation_probability"] for row in selected],
            **style,
        )
        if model != "exact":
            axes[2].semilogy(
                intervals,
                [row["metric_hilbert_schmidt_distance_from_exact"] for row in selected],
                **style,
            )
    titles = (
        "A  Final absorbed energy",
        "B  Final excitation probability",
        "C  Final density distance from EX",
    )
    ylabels = (
        r"$E(T)-E_0$ (Ha)",
        r"$1-\langle\psi_0|\rho(T)|\psi_0\rangle$",
        "metric Hilbert-Schmidt distance",
    )
    reviewed_intervals = sorted({int(row["intervals"]) for row in rows})
    for axis, title, ylabel in zip(axes, titles, ylabels, strict=True):
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("time intervals over 4 a.u.")
        axis.set_ylabel(ylabel)
        axis.set_xticks(reviewed_intervals)
        axis.grid(True, which="both", color="#d7dde5", linewidth=0.6, alpha=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=6,
        frameon=False,
    )
    figure.suptitle(
        r"Distorted H$_3^{2+}$ above-threshold pulse: field-free endpoint observables",
        fontsize=15,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"endpoint_observables.{suffix}", dpi=220)
    plt.close(figure)


def main() -> None:
    args = _arguments()
    mixed_root = args.mixed_execution.resolve()
    n64_root = args.n64_execution.resolve()
    n128_root = args.n128_execution.resolve() if args.n128_execution is not None else None
    default_output_root = n128_root if n128_root is not None else n64_root
    output = (args.output or default_output_root / "endpoint_observables").resolve()
    output.mkdir(parents=True, exist_ok=True)
    mixed_result, mixed_arrays = diagnostic._authenticate_execution(mixed_root)
    n64_result, n64_arrays = diagnostic._authenticate_execution(n64_root)
    arrays_for_intervals = {16: mixed_arrays, 32: mixed_arrays, 64: n64_arrays}
    source_plan_ids = {
        "n16_n32": mixed_result["plan_id"],
        "n64": n64_result["plan_id"],
    }
    if n128_root is not None:
        n128_result, n128_arrays = diagnostic._authenticate_execution(n128_root)
        arrays_for_intervals[128] = n128_arrays
        source_plan_ids["n128"] = n128_result["plan_id"]
    rows = _observable_rows(arrays_for_intervals)
    _write_csv(output / "endpoint_observables.csv", rows)
    _plot(rows, output)
    summary = {
        "schema": "aion.exact-one-electron-wp6-endpoint-observables-analysis",
        "version": "1.0.0",
        "status": "derived_analysis_unreviewed",
        "scope": "distorted above-threshold field-free endpoint",
        "source_plan_ids": source_plan_ids,
        "definitions": {
            "energy": "Re Tr[P K0]",
            "ground_state_survival": "Re c0^dag S0 P S0 c0",
            "metric_hilbert_schmidt_distance": (
                "sqrt(Re Tr[deltaP^dag S0 deltaP S0])"
            ),
        },
        "row_count": len(rows),
    }
    diagnostic._write_json(output / "summary.json", summary)
    print(f"output_directory={output}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
