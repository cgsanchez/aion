#!/usr/bin/env python3
"""H2 pure Peierls P0 LDA trajectory gauge comparison from PySCF matrices."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-aion")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/aion-cache")

import matplotlib.pyplot as plt
import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pyscf import dft, gto  # noqa: E402

from aion import (  # noqa: E402
    DEFAULT_GAUGE_LAMBDAS,
    P0SCEMSettings,
    PyscfP0LdaModel,
    PyscfP0Reference,
    p0_natom_from_rows,
    p0_row_series,
    run_p0_uniform_electric_gauge_comparison,
    summarize_p0_gauge_errors,
)


DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="lda,vwn")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--nsteps", type=int, default=10)
    parser.add_argument("--field-z", type=float, default=0.02)
    parser.add_argument("--mixing", type=float, default=0.7)
    parser.add_argument("--midpoint-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--max-iterations", type=int, default=18)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def build_reference(args: argparse.Namespace) -> PyscfP0Reference:
    mol = gto.M(
        atom="H 0 0 -0.37; H 0 0 0.37",
        basis=args.basis,
        unit="Angstrom",
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = args.xc
    mf.grids.level = args.grid_level
    mf.conv_tol = 1.0e-11
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return PyscfP0Reference.from_mean_field(mf)


def write_rows(
    path: Path,
    rows_by_gauge: dict[str, list[dict[str, float | int | str]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [row for rows in rows_by_gauge.values() for row in rows]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)


def plot_summary(
    path: Path,
    rows_by_gauge: dict[str, list[dict[str, float | int | str]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = p0_row_series(rows_by_gauge["length"], "time_au")
    natom = p0_natom_from_rows(rows_by_gauge["length"])

    ref_dipole = p0_row_series(rows_by_gauge["length"], "dipole_z")
    ref_energy = p0_row_series(rows_by_gauge["length"], "energy")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    ax_dipole, ax_energy, ax_pop, ax_error = axes.ravel()

    for label, rows in rows_by_gauge.items():
        dipole = p0_row_series(rows, "dipole_z")
        energy = p0_row_series(rows, "energy")
        ax_dipole.plot(times, dipole, label=label)
        ax_energy.plot(times, energy, label=label)
        if label != "length":
            ax_error.semilogy(
                times,
                np.maximum(np.abs(dipole - ref_dipole), 1.0e-18),
                label=f"{label} dipole",
            )
            ax_error.semilogy(
                times,
                np.maximum(np.abs(energy - ref_energy), 1.0e-18),
                "--",
                label=f"{label} energy",
            )

    for atom in range(natom):
        ax_pop.plot(
            times,
            p0_row_series(rows_by_gauge["length"], f"population_{atom}"),
            label=f"site {atom}",
        )

    ax_dipole.set_title("P0 LDA dipole")
    ax_dipole.set_xlabel("time / au")
    ax_dipole.set_ylabel("dipole z / au")
    ax_dipole.legend()

    ax_energy.set_title("P0 LDA material energy")
    ax_energy.set_xlabel("time / au")
    ax_energy.set_ylabel("energy / Ha")
    ax_energy.legend()

    ax_pop.set_title("Length-gauge source populations")
    ax_pop.set_xlabel("time / au")
    ax_pop.set_ylabel("population")
    ax_pop.legend()

    ax_error.set_title("Gauge differences vs length")
    ax_error.set_xlabel("time / au")
    ax_error.set_ylabel("absolute error")
    ax_error.legend()

    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.nsteps < 0:
        raise ValueError("nsteps must be nonnegative")
    if args.dt <= 0.0:
        raise ValueError("dt must be positive")

    reference = build_reference(args)
    model = PyscfP0LdaModel.from_reference(reference)
    settings = P0SCEMSettings(
        dt=args.dt,
        nsteps=args.nsteps,
        midpoint_tolerance=args.midpoint_tolerance,
        density_tolerance=args.density_tolerance,
        max_iterations=args.max_iterations,
        mixing=args.mixing,
    )
    rows_by_gauge = run_p0_uniform_electric_gauge_comparison(
        geometry_factory=lambda electric: reference.geometry(electric=electric),
        model=model,
        occupations=reference.occupations,
        coeff0=reference.initial_coefficients(),
        field=np.array([0.0, 0.0, args.field_z], dtype=float),
        settings=settings,
        gauge_lambdas=DEFAULT_GAUGE_LAMBDAS,
    )

    csv_path = args.output_dir / "h2_p0_lda_gauge_compare.csv"
    write_rows(csv_path, rows_by_gauge)
    plot_path = args.output_dir / "h2_p0_lda_gauge_compare.png"
    if not args.no_plot:
        plot_summary(plot_path, rows_by_gauge)
    summary = summarize_p0_gauge_errors(rows_by_gauge)

    print(
        "# h2 p0 lda gauge comparison",
        f"basis={args.basis}",
        f"xc={args.xc}",
        f"dt={args.dt}",
        f"nsteps={args.nsteps}",
        f"field_z={args.field_z}",
        f"csv={csv_path}",
        f"plot={plot_path if not args.no_plot else 'disabled'}",
    )
    for label, values in summary.items():
        print(
            label,
            "max_dipole_norm_error={max_dipole_norm_error:.3e}".format(**values),
            "max_energy_abs_error={max_energy_abs_error:.3e}".format(**values),
            "max_population_norm_error={max_population_norm_error:.3e}".format(**values),
            "max_orthonormality_error={max_orthonormality_error:.3e}".format(**values),
            "max_instantaneous_continuity_residual={max_instantaneous_continuity_residual:.3e}".format(**values),
            "max_power_residual={max_power_residual:.3e}".format(**values),
            "max_dipole_power_residual={max_dipole_power_residual:.3e}".format(**values),
        )


if __name__ == "__main__":
    main()
