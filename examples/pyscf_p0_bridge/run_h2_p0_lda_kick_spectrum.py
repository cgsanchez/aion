#!/usr/bin/env python3
"""H2 P0 LDA velocity-gauge delta-kick trajectory and spectrum."""

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
    P0SCEMSettings,
    PyscfP0LdaModel,
    PyscfP0Reference,
    apply_p0_velocity_delta_kick,
    kick_spectrum,
    p0_row_series,
    run_p0_scem_trajectory,
    velocity_delta_kick_electric_gauge,
    write_kick_spectrum_csv,
)


DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="lda,vwn")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--t-final", type=float, default=100.0)
    parser.add_argument("--kick", type=float, default=1.0e-3)
    parser.add_argument("--polarization", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    parser.add_argument("--damping", type=float, default=0.004)
    parser.add_argument("--zero-pad-factor", type=int, default=10)
    parser.add_argument("--max-energy-ev", type=float, default=20.0)
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


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_result(
    path: Path,
    rows: list[dict[str, float | int | str]],
    spectrum,
    polarization: np.ndarray,
) -> None:
    time = p0_row_series(rows, "time_au")
    dipole = np.column_stack([p0_row_series(rows, f"dipole_{axis}") for axis in "xyz"])
    mu_parallel = dipole @ polarization

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), constrained_layout=True)
    axes[0].plot(time, mu_parallel - mu_parallel[0])
    axes[0].set_xlabel("time / au")
    axes[0].set_ylabel("Delta mu_parallel / au")
    axes[0].set_title("P0 LDA velocity-gauge kick response")

    axes[1].plot(spectrum.energy_ev, spectrum.strength)
    axes[1].set_xlabel("energy / eV")
    axes[1].set_ylabel("strength / arb.")
    axes[1].set_title("Damped kick spectrum")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.t_final <= 0.0:
        raise ValueError("t-final must be positive")
    polarization = np.asarray(args.polarization, dtype=float)
    norm = np.linalg.norm(polarization)
    if norm == 0.0:
        raise ValueError("polarization must be nonzero")
    polarization = polarization / norm

    reference = build_reference(args)
    model = PyscfP0LdaModel.from_reference(reference)
    field_free = reference.geometry()
    impulse = args.kick * polarization
    kicked_geometry = reference.geometry(
        electric=velocity_delta_kick_electric_gauge(impulse)
    )
    coeff0 = apply_p0_velocity_delta_kick(
        reference.initial_coefficients(),
        field_free,
        kicked_geometry,
    )
    nsteps = int(np.ceil(args.t_final / args.dt))
    settings = P0SCEMSettings(
        dt=args.dt,
        nsteps=nsteps,
        midpoint_tolerance=args.midpoint_tolerance,
        density_tolerance=args.density_tolerance,
        max_iterations=args.max_iterations,
        mixing=args.mixing,
    )
    rows = run_p0_scem_trajectory(
        label="velocity_kick",
        geometry=kicked_geometry,
        model=model,
        occupations=reference.occupations,
        coeff0=coeff0,
        settings=settings,
    )

    time = p0_row_series(rows, "time_au")
    dipole = np.column_stack([p0_row_series(rows, f"dipole_{axis}") for axis in "xyz"])
    response = dipole @ polarization
    spectrum = kick_spectrum(
        time,
        response,
        kick_au=args.kick,
        damping_ha=args.damping,
        zero_pad_factor=args.zero_pad_factor,
        max_energy_ev=args.max_energy_ev,
    )

    rows_path = args.output_dir / "h2_p0_lda_velocity_kick.csv"
    spectrum_path = args.output_dir / "h2_p0_lda_velocity_kick_spectrum.csv"
    plot_path = args.output_dir / "h2_p0_lda_velocity_kick_spectrum.png"
    write_rows(rows_path, rows)
    write_kick_spectrum_csv(spectrum_path, spectrum)
    if not args.no_plot:
        plot_result(plot_path, rows, spectrum, polarization)

    print(
        "# h2 p0 lda velocity kick",
        f"basis={args.basis}",
        f"xc={args.xc}",
        f"dt={args.dt}",
        f"t_final={nsteps * args.dt}",
        f"kick={args.kick}",
        f"rows={rows_path}",
        f"spectrum={spectrum_path}",
        f"plot={plot_path if not args.no_plot else 'disabled'}",
    )


if __name__ == "__main__":
    main()
