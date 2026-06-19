#!/usr/bin/env python3
"""CN RT-TDDFT sin^2 pulse for para-nitroaniline."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-aion")

import matplotlib.pyplot as plt
import numpy as np

from common import DEFAULT_EXCITATIONS_PATH, build_ground_state

from aion import (  # noqa: E402
    LengthGaugeCNRTTDDFT,
    Sin2Pulse,
    excitation_by_index,
    load_excitations,
    transition_polarization,
)


AU_FIELD_TO_V_PER_ANGSTROM = 51.4220674763
V_PER_ANGSTROM_TO_AU_FIELD = 1.0 / AU_FIELD_TO_V_PER_ANGSTROM


HEADER = [
    "step",
    "time_au",
    "electron_number",
    "orthonormality_error",
    "idempotency_error",
    "field_free_energy_ha",
    "field_coupling_energy_ha",
    "total_energy_ha",
    "mu_x_au",
    "mu_y_au",
    "mu_z_au",
    "field_x_au",
    "field_y_au",
    "field_z_au",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excitations", type=Path, default=DEFAULT_EXCITATIONS_PATH)
    parser.add_argument(
        "--root",
        type=int,
        default=None,
        help="Casida root to drive. Defaults to metadata selected_root.",
    )
    parser.add_argument("--field-vpa", type=float, default=0.5)
    parser.add_argument("--cycles", type=float, default=1.0)
    parser.add_argument("--post-time", type=float, default=50.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--energy-stride", type=int, default=10)
    parser.add_argument("--corrector-iterations", type=int, default=0)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--verbose", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def record_to_row(rec) -> list[float]:
    return [
        rec.step,
        rec.time,
        rec.electron_number,
        rec.orthonormality_error,
        rec.idempotency_error,
        np.nan if rec.field_free_energy is None else rec.field_free_energy,
        rec.field_coupling_energy,
        np.nan if rec.total_energy is None else rec.total_energy,
        *rec.dipole.tolist(),
        *rec.field.tolist(),
    ]


def plot_rows(rows: list[list[float]], pulse_duration: float, plot_path: Path) -> None:
    data = np.asarray(rows, dtype=float)
    time_au = data[:, 1]
    e0 = data[:, 5]
    ecoup = data[:, 6]
    etot = data[:, 7]
    mux, muy, muz = data[:, 8], data[:, 9], data[:, 10]
    ex, ey, ez = data[:, 11], data[:, 12], data[:, 13]
    orth, idem = data[:, 3], data[:, 4]
    finite_energy = np.isfinite(e0) & np.isfinite(etot)

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    if np.any(finite_energy):
        tref = time_au[finite_energy]
        axes[0].plot(tref, e0[finite_energy] - e0[finite_energy][0], label="E0 - E0(0)")
        axes[0].plot(
            tref,
            etot[finite_energy] - etot[finite_energy][0],
            label="Etot - Etot(0)",
        )
    axes[0].plot(time_au, ecoup, label="Ecoup")
    axes[0].axvline(pulse_duration, color="0.5", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Energy (Ha)")
    axes[0].legend(loc="best")

    axes[1].plot(time_au, mux, label="mu_x")
    axes[1].plot(time_au, muy, label="mu_y")
    axes[1].plot(time_au, muz, label="mu_z")
    axes[1].axvline(pulse_duration, color="0.5", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Electronic dipole (a.u.)")
    axes[1].legend(loc="best")

    axes[2].plot(time_au, ex, label="E_x")
    axes[2].plot(time_au, ey, label="E_y")
    axes[2].plot(time_au, ez, label="E_z")
    axes[2].axvline(pulse_duration, color="0.5", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("Field (a.u.)")
    axes[2].legend(loc="best")

    axes[3].semilogy(time_au, orth, label="||C^dag S C - I||")
    axes[3].semilogy(time_au, idem, label="||P S P - 2P||")
    axes[3].axvline(pulse_duration, color="0.5", linestyle="--", linewidth=1.0)
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("time (a.u.)")
    axes[3].legend(loc="best")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def label_float(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def main() -> None:
    args = parse_args()
    metadata, excitations = load_excitations(args.excitations)
    root = int(metadata["selected_root"] if args.root is None else args.root)
    driven = excitation_by_index(excitations, root)
    polarization = transition_polarization(driven)

    basis = str(metadata["basis"])
    xc = str(metadata["xc"])
    grid_level = int(metadata.get("grid_level", 1))
    field_au = args.field_vpa * V_PER_ANGSTROM_TO_AU_FIELD
    pulse = Sin2Pulse.from_cycles(
        amplitude=field_au,
        omega=driven.energy,
        cycles=args.cycles,
        polarization=polarization,
        phase=args.phase,
    )
    t_final = pulse.duration + args.post_time
    nsteps = int(np.ceil(t_final / args.dt))
    actual_t_final = nsteps * args.dt

    results_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(__file__).resolve().parent / "results" / "cn_sin2"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"root{root}_"
        f"{xc}_{basis.replace('-', '').replace('*', 'star')}_"
        f"field{label_float(args.field_vpa)}vpa_"
        f"cycles{label_float(args.cycles)}_"
        f"dt{label_float(args.dt)}"
    )
    csv_path = results_dir / f"para_nitroaniline_cn_sin2_{tag}.csv"
    plot_path = results_dir / f"para_nitroaniline_cn_sin2_{tag}.png"

    print(
        "# CN sin2 pulse para-nitroaniline",
        f"casida={args.excitations}",
        f"basis={basis}",
        f"xc={xc}",
        f"grid_level={grid_level}",
        f"root={driven.index}",
        f"omega={driven.energy:.10f} Ha",
        f"omega_ev={driven.energy_ev:.6f}",
        f"osc={driven.oscillator_strength:.6e}",
        f"polarization={polarization.tolist()}",
        f"field={field_au:.10e} au",
        f"field={args.field_vpa:.6f} V/Ang",
        f"cycles={args.cycles}",
        f"pulse_duration={pulse.duration:.6f}",
        f"post_time={args.post_time:.6f}",
        f"dt={args.dt}",
        f"nsteps={nsteps}",
        f"actual_t_final={actual_t_final:.6f}",
        f"energy_stride={args.energy_stride}",
        flush=True,
    )

    mf = build_ground_state(
        basis=basis,
        xc=xc,
        grid_level=grid_level,
        verbose=args.verbose,
    )
    rt = LengthGaugeCNRTTDDFT.from_ground_state(mf, pulse, origin=np.zeros(3))
    coeff0 = rt.initial_coefficients()

    rows = []
    start = time.perf_counter()
    last = None
    for _, rec in rt.propagate(
        coeff0,
        dt=args.dt,
        nsteps=nsteps,
        corrector_iterations=args.corrector_iterations,
        record_energy=True,
        energy_stride=args.energy_stride,
    ):
        rows.append(record_to_row(rec))
        last = rec
        if rec.step % max(1, nsteps // 10) == 0:
            print(
                "# progress",
                f"step={rec.step}/{nsteps}",
                f"time={rec.time:.3f}",
                f"N={rec.electron_number:.12f}",
                f"orth={rec.orthonormality_error:.3e}",
                f"idem={rec.idempotency_error:.3e}",
                f"mu_x={rec.dipole[0]:.8e}",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    print(f"# propagation_seconds {elapsed:.3f}", flush=True)
    if last is not None:
        print(
            "# final",
            f"step={last.step}",
            f"time={last.time:.6f}",
            f"N={last.electron_number:.12f}",
            f"orth={last.orthonormality_error:.6e}",
            f"idem={last.idempotency_error:.6e}",
            f"mu={last.dipole.tolist()}",
            flush=True,
        )

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
    plot_rows(rows, pulse.duration, plot_path)
    print(f"# wrote {csv_path}", flush=True)
    print(f"# wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
