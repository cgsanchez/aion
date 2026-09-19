"""Build compact figures used by the exact one-electron qualification report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesis", type=Path, required=True)
    parser.add_argument("--wp6-n128", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _style() -> None:
    plt.rcParams.update(
        {
            "axes.grid": True,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "grid.alpha": 0.35,
            "figure.dpi": 160,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def _save(figure: Figure, output: Path, stem: str) -> None:
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(output / f"{stem}.png", bbox_inches="tight", dpi=220)
    plt.close(figure)


def _validity_figure(synthesis: Path, output: Path) -> None:
    rows = _rows(synthesis / "tables" / "magnetic_validity_global.csv")
    families = ("overlap", "kinetic", "nuclear_attraction", "mechanical")
    models = ("P0", "gB1", "B1", "C1")
    thresholds = ((1.0e-4, "#2864a8", "relative error $10^{-4}$"),
                  (1.0e-2, "#dd6b20", "relative error $10^{-2}$"))
    lookup = {
        (row["family"], row["model_label"], float(row["relative_threshold"])): row
        for row in rows
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), sharex=True, sharey=True)
    for axis, family in zip(axes.flat, families, strict=True):
        y = np.arange(len(models), dtype=float)
        for offset, (threshold, color, label) in zip((-0.11, 0.11), thresholds, strict=True):
            lows = []
            highs = []
            for model in models:
                row = lookup[(family, model, threshold)]
                lows.append(float(row["conservative_confirmed_below_au"]))
                highs.append(float(row["minimum_first_above_au"]))
            yy = y + offset
            for low, high, ypos in zip(lows, highs, yy, strict=True):
                axis.plot([low, high], [ypos, ypos], color=color, linewidth=2.0)
            axis.scatter(lows, yy, color=color, marker="o", s=36, zorder=3, label=label)
            axis.scatter(
                highs,
                yy,
                facecolors="white",
                edgecolors=color,
                marker="o",
                linewidths=1.5,
                s=36,
                zorder=3,
            )
        axis.set_title(family.replace("_", " "))
        axis.set_xscale("log")
        axis.set_yticks(y, models)
        axis.invert_yaxis()
        axis.set_xlim(6.0e-5, 2.0)
    axes[1, 0].set_xlabel("uniform magnetic-field magnitude $|B|$ (a.u.)")
    axes[1, 1].set_xlabel("uniform magnetic-field magnitude $|B|$ (a.u.)")
    figure.suptitle("Conservative all-system validity brackets (no interpolation)", y=1.01)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2)
    figure.text(
        0.5,
        -0.01,
        "filled marker: last sampled field confirmed below threshold; "
        "open marker: first sampled field above threshold",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.92))
    _save(figure, output, "global_validity_brackets")


def _dynamic_resolution_figure(synthesis: Path, output: Path) -> None:
    rows = _rows(synthesis / "tables" / "dynamic_endpoint_differences.csv")
    labels = [row["model_label"] for row in rows]
    x = np.arange(len(rows), dtype=float)
    energy_difference = np.array(
        [abs(float(row["absorbed_energy_difference_from_exact"])) for row in rows]
    )
    energy_uncertainty = np.array(
        [float(row["absorbed_energy_timestep_difference_uncertainty"]) for row in rows]
    )
    density_difference = np.array(
        [
            float(row["metric_hilbert_schmidt_distance_from_exact_difference_from_exact"])
            for row in rows
        ]
    )
    density_uncertainty = np.array(
        [
            float(
                row[
                    "metric_hilbert_schmidt_distance_from_exact_"
                    "timestep_difference_uncertainty"
                ]
            )
            for row in rows
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for axis, difference, uncertainty, title, ylabel in (
        (
            axes[0],
            energy_difference,
            energy_uncertainty,
            "Absorbed-energy difference",
            "absolute value (Ha)",
        ),
        (
            axes[1],
            density_difference,
            density_uncertainty,
            "Metric density distance",
            "distance / uncertainty",
        ),
    ):
        axis.scatter(
            x - 0.10,
            difference,
            marker="o",
            s=52,
            color="#2864a8",
            label="model minus EX",
        )
        axis.scatter(
            x + 0.10,
            uncertainty,
            marker="s",
            s=46,
            color="#dd6b20",
            label="$n=64$ to $128$ uncertainty",
        )
        axis.set_yscale("log")
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    figure.suptitle(
        "Distorted one-electron $H_3^{2+}$ above-threshold endpoint: "
        "physical separation versus timestep uncertainty",
        y=1.15,
    )
    figure.tight_layout()
    _save(figure, output, "dynamic_difference_resolution")


def _temporal_convergence_figure(wp6_n128: Path, output: Path) -> None:
    with (wp6_n128 / "result.json").open(encoding="utf-8") as handle:
        result = json.load(handle)
    label = {
        "exact": "EX",
        "p0": "P0",
        "e1": "E1",
        "geometric_b1": "gB1",
        "full_b1": "B1",
        "complete_first_order": "C1",
    }
    convergence = result["temporal_convergence"]
    labels = [label[row["model"]] for row in convergence]
    orders = np.array([float(row["measured_order"]) for row in convergence])
    stability = result["model_difference_stability"]
    stability_labels = [label[row["model"]] for row in stability]
    fractions = np.array(
        [float(row["refinement_fraction_of_fine_model_difference"]) for row in stability]
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    bars = axes[0].bar(labels, orders, color="#2864a8", width=0.68)
    axes[0].axhline(4.0, color="#333333", linestyle="--", linewidth=1.4, label="fourth order")
    axes[0].set_ylim(3.5, 4.05)
    axes[0].set_ylabel("measured global order")
    axes[0].set_title("32/64/128-interval density convergence")
    axes[0].legend(loc="lower right")
    for bar, value in zip(bars, orders, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.012,
            f"{value:.3f}",
            ha="center",
            va="top",
            fontsize=8,
        )

    axes[1].scatter(stability_labels, fractions, color="#dd6b20", s=58, zorder=3)
    axes[1].axhline(0.25, color="#333333", linestyle="--", linewidth=1.4)
    axes[1].set_yscale("log")
    axes[1].set_ylim(3.0e-6, 0.5)
    axes[1].set_ylabel("64-to-128 change / final model difference")
    axes[1].set_title("Model-minus-EX difference stability")
    axes[1].text(
        len(stability_labels) - 0.05,
        0.19,
        "declared limit: 0.25",
        ha="right",
        va="top",
        fontsize=9,
    )
    figure.suptitle(
        "Distorted one-electron $H_3^{2+}$ above-threshold trajectory",
        y=1.03,
    )
    figure.tight_layout()
    _save(figure, output, "temporal_convergence_and_stability")


def main() -> None:
    args = _arguments()
    synthesis = args.synthesis.resolve()
    wp6_n128 = args.wp6_n128.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _style()
    _validity_figure(synthesis, output)
    _dynamic_resolution_figure(synthesis, output)
    _temporal_convergence_figure(wp6_n128, output)


if __name__ == "__main__":
    main()
