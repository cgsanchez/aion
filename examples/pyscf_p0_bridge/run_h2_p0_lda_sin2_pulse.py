#!/usr/bin/env python3
"""H2 P0 LDA sin²-pulse gauge comparison from PySCF matrices."""

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
    run_p0_electric_gauge_comparison,
    sin2_uniform_electric_gauge,
    summarize_p0_gauge_errors,
)


DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="lda,vwn")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--t-final", type=float, default=None)
    parser.add_argument("--amplitude", type=float, default=1.0e-3)
    parser.add_argument("--omega", type=float, default=0.45)
    parser.add_argument("--cycles", type=float, default=4.0)
    parser.add_argument("--polarization", type=float, nargs=3, default=[0.0, 0.0, 1.0])
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
    polarization: np.ndarray,
    field_source,
) -> None:
    time = p0_row_series(rows_by_gauge["length"], "time_au")
    natom = p0_natom_from_rows(rows_by_gauge["length"])
    field_parallel = np.asarray(
        [np.dot(field_source.electric_field(float(t)), polarization) for t in time],
        dtype=float,
    )
    ref_dipole = np.column_stack(
        [p0_row_series(rows_by_gauge["length"], f"dipole_{axis}") for axis in "xyz"]
    )
    ref_energy = p0_row_series(rows_by_gauge["length"], "energy")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    ax_signal, ax_energy, ax_pop, ax_error = axes.ravel()
    for label, rows in rows_by_gauge.items():
        dipole = np.column_stack([p0_row_series(rows, f"dipole_{axis}") for axis in "xyz"])
        energy = p0_row_series(rows, "energy")
        ax_signal.plot(time, dipole @ polarization, label=label)
        ax_energy.plot(time, energy, label=label)
        if label != "length":
            ax_error.semilogy(
                time,
                np.maximum(np.linalg.norm(dipole - ref_dipole, axis=1), 1.0e-18),
                label=f"{label} dipole",
            )
            ax_error.semilogy(
                time,
                np.maximum(np.abs(energy - ref_energy), 1.0e-18),
                "--",
                label=f"{label} energy",
            )
    field_scale = max(1.0, np.max(np.abs(field_parallel)))
    ax_signal.plot(
        time,
        field_parallel / field_scale,
        color="0.4",
        linestyle=":",
        label="field / scale",
    )
    for atom in range(natom):
        ax_pop.plot(
            time,
            p0_row_series(rows_by_gauge["length"], f"population_{atom}"),
            label=f"site {atom}",
        )

    ax_signal.set_xlabel("time / au")
    ax_signal.set_ylabel("mu_parallel / au")
    ax_signal.set_title("P0 LDA sin²-pulse response")
    ax_signal.legend()

    ax_energy.set_xlabel("time / au")
    ax_energy.set_ylabel("energy / Ha")
    ax_energy.set_title("Material energy")
    ax_energy.legend()

    ax_pop.set_xlabel("time / au")
    ax_pop.set_ylabel("population")
    ax_pop.set_title("Length-gauge source populations")
    ax_pop.legend()

    ax_error.set_xlabel("time / au")
    ax_error.set_ylabel("absolute error")
    ax_error.set_title("Gauge differences vs length")
    ax_error.legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    polarization = np.asarray(args.polarization, dtype=float)
    norm = np.linalg.norm(polarization)
    if norm == 0.0:
        raise ValueError("polarization must be nonzero")
    polarization = polarization / norm
    pulse_duration = args.cycles * 2.0 * np.pi / args.omega
    t_final = args.t_final if args.t_final is not None else pulse_duration + 20.0
    nsteps = int(np.ceil(t_final / args.dt))

    reference = build_reference(args)
    model = PyscfP0LdaModel.from_reference(reference)
    settings = P0SCEMSettings(
        dt=args.dt,
        nsteps=nsteps,
        midpoint_tolerance=args.midpoint_tolerance,
        density_tolerance=args.density_tolerance,
        max_iterations=args.max_iterations,
        mixing=args.mixing,
    )
    electric_factory = lambda lambda_value: sin2_uniform_electric_gauge(
        amplitude=args.amplitude,
        omega=args.omega,
        cycles=args.cycles,
        polarization=polarization,
        lambda_value=lambda_value,
    )
    rows_by_gauge = run_p0_electric_gauge_comparison(
        geometry_factory=lambda electric: reference.geometry(electric=electric),
        electric_factory=electric_factory,
        model=model,
        occupations=reference.occupations,
        coeff0=reference.initial_coefficients(),
        settings=settings,
        gauge_lambdas=DEFAULT_GAUGE_LAMBDAS,
    )

    csv_path = args.output_dir / "h2_p0_lda_sin2_pulse.csv"
    plot_path = args.output_dir / "h2_p0_lda_sin2_pulse.png"
    write_rows(csv_path, rows_by_gauge)
    if not args.no_plot:
        plot_summary(plot_path, rows_by_gauge, polarization, electric_factory(0.0))
    summary = summarize_p0_gauge_errors(rows_by_gauge)

    print(
        "# h2 p0 lda sin2 pulse",
        f"basis={args.basis}",
        f"xc={args.xc}",
        f"dt={args.dt}",
        f"t_final={nsteps * args.dt}",
        f"duration={pulse_duration}",
        f"amplitude={args.amplitude}",
        f"omega={args.omega}",
        f"cycles={args.cycles}",
        f"csv={csv_path}",
        f"plot={plot_path if not args.no_plot else 'disabled'}",
    )
    for label, values in summary.items():
        print(
            label,
            "max_dipole_norm_error={max_dipole_norm_error:.3e}".format(**values),
            "max_energy_abs_error={max_energy_abs_error:.3e}".format(**values),
            "max_population_norm_error={max_population_norm_error:.3e}".format(**values),
            "max_power_residual={max_power_residual:.3e}".format(**values),
        )


if __name__ == "__main__":
    main()
