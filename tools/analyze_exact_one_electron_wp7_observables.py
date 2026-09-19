#!/usr/bin/env python3
"""Analyze and plot the authenticated WP7 observable-history campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

_DEFAULT_CPU = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "exact_one_electron_qualification/"
    "wp7_observables_cpu_20260919T145221Z_7d603105e1db"
)
_DEFAULT_GPU = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "exact_one_electron_qualification/"
    "wp7_observables_gpu_20260919T145459Z_7d603105e1db"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-execution", type=Path, default=_DEFAULT_CPU)
    parser.add_argument("--gpu-execution", type=Path, default=_DEFAULT_GPU)
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _positive(values: np.ndarray, floor: float = 1.0e-18) -> np.ndarray:
    return np.maximum(np.abs(values), floor)


def _cumulative_trapezoid(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    increments = 0.5 * (values[1:] + values[:-1]) * np.diff(times)
    return np.concatenate((np.zeros(1), np.cumsum(increments)))


def _orders(steps: np.ndarray, errors: np.ndarray) -> list[float]:
    return [
        math.log(float(errors[index] / errors[index + 1]))
        / math.log(float(steps[index] / steps[index + 1]))
        for index in range(len(errors) - 1)
    ]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfd",
            "axes.edgecolor": "#30343b",
            "axes.labelcolor": "#20242a",
            "axes.titlecolor": "#20242a",
            "axes.grid": True,
            "grid.color": "#d9dee5",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.8,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "savefig.dpi": 180,
        }
    )


def _plot_electric_history(data: np.ndarray, output: Path) -> None:
    time_au = data[:, 0]
    energy_change = data[:, 5] - data[0, 5]
    accumulated_work = _cumulative_trapezoid(data[:, 6], time_au)
    finite_rate = np.gradient(data[:, 5], time_au, edge_order=2)
    finite_defect = finite_rate - data[:, 6]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    axes[0].plot(time_au, energy_change, color="#245fa5", lw=2.0, label=r"$U(t)-U(0)$")
    axes[0].plot(
        time_au,
        accumulated_work,
        color="#d97721",
        lw=1.8,
        ls="--",
        label=r"$\int_0^t \mathbf{E}\!\cdot\!\mathbf{J}_{\rm src}\,dt$",
    )
    axes[0].set_ylabel("energy (Ha)")
    axes[0].set_title(r"Uniform-electric trajectory, exact Wilson model ($\Delta t=0.125$ a.u.)")
    axes[0].legend(loc="upper left")
    axes[1].semilogy(
        time_au,
        _positive(data[:, 7]),
        color="#245fa5",
        lw=1.8,
        label="instantaneous action identity",
    )
    axes[1].semilogy(
        time_au,
        _positive(finite_defect),
        color="#d97721",
        lw=1.5,
        ls="--",
        label="trajectory finite-difference defect",
    )
    axes[1].set_xlabel("time (a.u.)")
    axes[1].set_ylabel("absolute power defect (Ha/a.u.)")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_refinement(summaries: list[dict[str, Any]], output: Path) -> None:
    electric = sorted(
        (row for row in summaries if row["case"] == "electric"),
        key=lambda row: float(row["step_au"]),
        reverse=True,
    )
    magnetic = sorted(
        (row for row in summaries if row["case"] == "magnetic"),
        key=lambda row: float(row["step_au"]),
        reverse=True,
    )
    steps = np.asarray([row["step_au"] for row in electric], dtype=np.float64)
    work = np.abs(np.asarray([row["integrated_work_defect"] for row in electric], dtype=np.float64))
    rate = np.asarray(
        [row["maximum_interior_trajectory_power_defect"] for row in electric],
        dtype=np.float64,
    )
    magnetic_steps = np.asarray(
        [row["step_au"] for row in magnetic],
        dtype=np.float64,
    )
    continuity = np.asarray(
        [row["maximum_continuity_residual"] for row in magnetic],
        dtype=np.float64,
    )
    ward = np.asarray(
        [row["maximum_ward_residual"] for row in magnetic],
        dtype=np.float64,
    )
    charge = np.asarray(
        [row["maximum_total_charge_error"] for row in magnetic],
        dtype=np.float64,
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    axes[0].loglog(steps, work, "o-", color="#245fa5", label="measured")
    axes[0].loglog(
        steps,
        work[-1] * (steps / steps[-1]) ** 4,
        color="#59616d",
        ls=":",
        label=r"$\Delta t^4$ guide",
    )
    axes[0].set_xlabel(r"$\Delta t$ (a.u.)")
    axes[0].set_ylabel("absolute energy-work defect (Ha)")
    axes[0].set_title("Integrated power balance")
    axes[0].legend()
    axes[0].invert_xaxis()
    axes[0].set_xticks(steps, labels=[f"{value:g}" for value in steps])
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    axes[1].loglog(steps, rate, "o-", color="#d97721", label="measured")
    axes[1].loglog(
        steps,
        rate[-1] * (steps / steps[-1]) ** 2,
        color="#59616d",
        ls=":",
        label=r"$\Delta t^2$ guide",
    )
    axes[1].set_xlabel(r"$\Delta t$ (a.u.)")
    axes[1].set_ylabel("maximum power-rate defect (Ha/a.u.)")
    axes[1].set_title("Centered trajectory derivative")
    axes[1].legend()
    axes[1].invert_xaxis()
    axes[1].set_xticks(steps, labels=[f"{value:g}" for value in steps])
    axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    axes[2].loglog(
        magnetic_steps,
        _positive(continuity),
        "o-",
        color="#245fa5",
        label="continuity",
    )
    axes[2].loglog(
        magnetic_steps,
        _positive(ward),
        "s--",
        color="#d97721",
        label="Ward",
    )
    axes[2].loglog(
        magnetic_steps,
        _positive(charge),
        "^:",
        color="#6f7f37",
        label="total charge",
    )
    axes[2].set_xlabel(r"$\Delta t$ (a.u.)")
    axes[2].set_ylabel("maximum absolute residual")
    axes[2].set_title("Magnetic trajectory identities")
    axes[2].legend()
    axes[2].invert_xaxis()
    axes[2].set_xticks(
        magnetic_steps,
        labels=[f"{value:g}" for value in magnetic_steps],
    )
    axes[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_magnetic_history(data: np.ndarray, output: Path) -> None:
    time_au = data[:, 0]
    charge_drift = data[:, 1] - data[0, 1]
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.semilogy(
        time_au,
        _positive(data[:, 3]),
        color="#245fa5",
        lw=1.8,
        label="local continuity",
    )
    axis.semilogy(
        time_au,
        _positive(data[:, 4]),
        color="#d97721",
        lw=1.6,
        ls="--",
        label="Ward",
    )
    axis.semilogy(
        time_au,
        _positive(charge_drift),
        color="#6f7f37",
        lw=1.4,
        ls=":",
        label="total-charge drift",
    )
    axis.set_xlabel("time (a.u.)")
    axis.set_ylabel("absolute residual")
    axis.set_title(r"Time-dependent magnetic trajectory ($\Delta t=0.125$ a.u.)")
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    args = _arguments()
    cpu = args.cpu_execution.resolve()
    gpu = args.gpu_execution.resolve()
    output = (
        args.output_directory.resolve() if args.output_directory is not None else cpu / "analysis"
    )
    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    cpu_result = _json(cpu / "result.json")
    gpu_result = _json(gpu / "result.json")
    _style()
    with np.load(cpu / "arrays.npz") as cpu_arrays, np.load(gpu / "arrays.npz") as gpu_arrays:
        electric_fine = np.asarray(cpu_arrays["electric__n64__diagnostics"])
        magnetic_fine = np.asarray(cpu_arrays["magnetic__n64__diagnostics"])
        _plot_electric_history(electric_fine, figures / "electric_power_history.png")
        _plot_refinement(
            list(cpu_result["summaries"]),
            figures / "observable_refinement.png",
        )
        _plot_magnetic_history(
            magnetic_fine,
            figures / "magnetic_identity_history.png",
        )
        parity: dict[str, dict[str, float]] = {}
        for case in ("electric", "magnetic"):
            for suffix in ("density", "diagnostics"):
                name = f"{case}__n32__{suffix}"
                difference = np.asarray(gpu_arrays[name]) - np.asarray(cpu_arrays[name])
                parity[name] = {
                    "maximum_absolute_difference": float(np.max(np.abs(difference))),
                    "frobenius_difference": float(np.linalg.norm(difference)),
                }

    refinement_rows: list[dict[str, Any]] = []
    for row in cpu_result["summaries"]:
        refinement_rows.append(dict(row))
    with (tables / "refinement.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = sorted({key for row in refinement_rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(refinement_rows)

    electric = sorted(
        (row for row in refinement_rows if row["case"] == "electric"),
        key=lambda row: float(row["step_au"]),
        reverse=True,
    )
    steps = np.asarray([row["step_au"] for row in electric])
    work_errors = np.abs(np.asarray([row["integrated_work_defect"] for row in electric]))
    rate_errors = np.asarray([row["maximum_interior_trajectory_power_defect"] for row in electric])
    summary = {
        "schema": "aion.exact-one-electron-wp7-observables-analysis",
        "version": "1.0.0",
        "status": "analyzed_unreviewed",
        "cpu_execution": str(cpu),
        "gpu_execution": str(gpu),
        "input_hashes": {
            "cpu_completed": _sha256(cpu / "completed.json"),
            "cpu_result": _sha256(cpu / "result.json"),
            "cpu_arrays": _sha256(cpu / "arrays.npz"),
            "gpu_completed": _sha256(gpu / "completed.json"),
            "gpu_result": _sha256(gpu / "result.json"),
            "gpu_arrays": _sha256(gpu / "arrays.npz"),
        },
        "electric_integrated_work_orders": _orders(steps, work_errors),
        "electric_trajectory_power_orders": _orders(steps, rate_errors),
        "cpu_gpu_parity": parity,
        "finest_cpu_electric": electric[-1],
        "finest_cpu_magnetic": sorted(
            (row for row in refinement_rows if row["case"] == "magnetic"),
            key=lambda row: float(row["step_au"]),
            reverse=True,
        )[-1],
        "gpu_summaries": gpu_result["summaries"],
        "figures": [
            "figures/electric_power_history.png",
            "figures/observable_refinement.png",
            "figures/magnetic_identity_history.png",
        ],
        "table": "tables/refinement.csv",
    }
    _write_json(output / "analysis_summary.json", summary)
    print(str(output), flush=True)


if __name__ == "__main__":
    main()
