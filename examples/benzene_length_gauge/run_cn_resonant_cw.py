#!/usr/bin/env python3
"""Long resonant continuous-wave CN RT-TDDFT run for benzene."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-aion")

import matplotlib.pyplot as plt
import numpy as np

from common import DEFAULT_EXCITATIONS_PATH, build_ground_state

from aion import (  # noqa: E402
    ContinuousWave,
    LengthGaugeCNRTTDDFT,
    excitation_by_index,
    load_excitations,
    transition_polarization,
)


AU_FIELD_TO_V_PER_ANGSTROM = 51.4220674763
V_PER_ANGSTROM_TO_AU_FIELD = 1.0 / AU_FIELD_TO_V_PER_ANGSTROM


def record_to_row(rec) -> list[float]:
    e0 = np.nan if rec.field_free_energy is None else rec.field_free_energy
    etot = np.nan if rec.total_energy is None else rec.total_energy
    return [
        rec.step,
        rec.time,
        rec.electron_number,
        rec.orthonormality_error,
        rec.idempotency_error,
        e0,
        rec.field_coupling_energy,
        etot,
        *rec.dipole.tolist(),
        *rec.field.tolist(),
    ]


def plot_rows(rows: list[list[float]], plot_path: Path) -> None:
    data = np.asarray(rows, dtype=float)
    time_au = data[:, 1]
    e0 = data[:, 5]
    ecoup = data[:, 6]
    etot = data[:, 7]
    mux, muy, muz = data[:, 8], data[:, 9], data[:, 10]
    orth = data[:, 3]
    idem = data[:, 4]
    finite_energy = np.isfinite(e0) & np.isfinite(etot)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    if np.any(finite_energy):
        tref = time_au[finite_energy]
        axes[0].plot(tref, e0[finite_energy] - e0[finite_energy][0], label="E0 - E0(0)")
        axes[0].plot(
            tref,
            etot[finite_energy] - etot[finite_energy][0],
            label="Etot - Etot(0)",
        )
    axes[0].plot(time_au, ecoup, label="Ecoup")
    axes[0].set_ylabel("Energy (Ha)")
    axes[0].legend(loc="best")

    axes[1].plot(time_au, mux, label="mu_x")
    axes[1].plot(time_au, muy, label="mu_y")
    axes[1].plot(time_au, muz, label="mu_z")
    axes[1].set_ylabel("Electronic dipole (a.u.)")
    axes[1].legend(loc="best")

    axes[2].semilogy(time_au, orth, label="||C^dag S C - I||")
    axes[2].semilogy(time_au, idem, label="||P S P - 2P||")
    axes[2].set_ylabel("Residual")
    axes[2].set_xlabel("time (a.u.)")
    axes[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def main() -> None:
    metadata, excitations = load_excitations(DEFAULT_EXCITATIONS_PATH)
    driven = excitation_by_index(excitations, int(metadata["selected_root"]))
    polarization = transition_polarization(driven)

    field_v_per_angstrom = 0.5
    field_au = field_v_per_angstrom * V_PER_ANGSTROM_TO_AU_FIELD
    dt = 0.1
    nsteps = 1000
    energy_stride = 10

    results_dir = Path(__file__).resolve().parent / "results" / "cn"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "benzene_cn_resonant_cw_dt0p1_t100.csv"
    plot_path = results_dir / "benzene_cn_resonant_cw_dt0p1_t100.png"

    print(
        "# CN resonant CW",
        f"root={driven.index}",
        f"omega={driven.energy:.10f} Ha",
        f"field={field_au:.10e} au",
        f"field={field_v_per_angstrom:.6f} V/Ang",
        f"dt={dt}",
        f"nsteps={nsteps}",
        f"energy_stride={energy_stride}",
        flush=True,
    )

    mf = build_ground_state(verbose=3)
    field = ContinuousWave(
        amplitude=field_au,
        omega=driven.energy,
        polarization=polarization,
    )
    rt = LengthGaugeCNRTTDDFT.from_ground_state(mf, field, origin=np.zeros(3))
    coeff0 = rt.initial_coefficients()

    header = [
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
    rows = []
    start = time.perf_counter()
    last = None
    for _, rec in rt.propagate(
        coeff0,
        dt=dt,
        nsteps=nsteps,
        record_energy=True,
        energy_stride=energy_stride,
    ):
        rows.append(record_to_row(rec))
        last = rec
        if rec.step % 100 == 0:
            print(
                "# progress",
                f"step={rec.step}/{nsteps}",
                f"time={rec.time:.3f}",
                f"N={rec.electron_number:.12f}",
                f"orth={rec.orthonormality_error:.3e}",
                f"idem={rec.idempotency_error:.3e}",
                f"mu_y={rec.dipole[1]:.8e}",
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
            f"mu_y={last.dipole[1]:.12e}",
            flush=True,
        )

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    plot_rows(rows, plot_path)
    print(f"# wrote {csv_path}", flush=True)
    print(f"# wrote {plot_path}", flush=True)


if __name__ == "__main__":
    main()
