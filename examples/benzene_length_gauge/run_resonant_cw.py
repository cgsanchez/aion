#!/usr/bin/env python3
"""Long resonant continuous-wave RT-TDDFT run for benzene."""

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
    LengthGaugeRTTDDFT,
    excitation_by_index,
    load_excitations,
    transition_polarization,
)


AU_FIELD_TO_V_PER_ANGSTROM = 51.4220674763
V_PER_ANGSTROM_TO_AU_FIELD = 1.0 / AU_FIELD_TO_V_PER_ANGSTROM


def main() -> None:
    metadata, excitations = load_excitations(DEFAULT_EXCITATIONS_PATH)
    driven = excitation_by_index(excitations, int(metadata["selected_root"]))
    polarization = transition_polarization(driven)

    field_v_per_angstrom = 0.5
    field_au = field_v_per_angstrom * V_PER_ANGSTROM_TO_AU_FIELD
    dt = 0.1
    nsteps = 1000
    record_stride = 10
    idempotency_stop = 1.0

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "benzene_resonant_cw_dt0p1_1000.csv"
    plot_path = results_dir / "benzene_resonant_cw_dt0p1_1000.png"

    print(
        "# resonant CW",
        f"root={driven.index}",
        f"omega={driven.energy:.10f} Ha",
        f"field={field_au:.10e} au",
        f"field={field_v_per_angstrom:.6f} V/Ang",
        f"dt={dt}",
        f"nsteps={nsteps}",
    )

    mf = build_ground_state(verbose=3)
    field = ContinuousWave(
        amplitude=field_au,
        omega=driven.energy,
        polarization=polarization,
    )
    rt = LengthGaugeRTTDDFT.from_ground_state(mf, field, origin=np.zeros(3))

    rho = rt.hermitian_part(rt.initial_density())
    rho_prev = rt.backward_euler_previous(rho, 0.0, dt)

    rows = []
    header = [
        "step",
        "time_au",
        "electron_number",
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

    def append_record(step: int, time_au: float, rho_now: np.ndarray):
        rec = rt.record(step, time_au, rho_now)
        rows.append(
            [
                rec.step,
                rec.time,
                rec.electron_number,
                rec.idempotency_error,
                rec.field_free_energy,
                rec.field_coupling_energy,
                rec.total_energy,
                *rec.dipole.tolist(),
                *rec.field.tolist(),
            ]
        )
        print(
            rec.step,
            f"{rec.time:.6f}",
            f"{rec.field_free_energy:.12f}",
            f"{rec.total_energy:.12f}",
            *(f"{x:.8e}" for x in rec.dipole),
        )
        return rec

    start = time.perf_counter()
    append_record(0, 0.0, rho)

    t = 0.0
    for step in range(1, nsteps + 1):
        rho_next = rho_prev + 2.0 * dt * rt.rhs(rho, t)
        rho_next = rt.hermitian_part(rho_next)
        t_next = step * dt

        if not np.isfinite(rho_next).all():
            print(f"# stopped: nonfinite density at step {step} time {t_next:.6f}")
            break

        if step % record_stride == 0 or step == nsteps:
            rec = append_record(step, t_next, rho_next)
            if not np.isfinite(
                [
                    rec.field_free_energy,
                    rec.field_coupling_energy,
                    rec.total_energy,
                    *rec.dipole.tolist(),
                ]
            ).all():
                print(f"# stopped: nonfinite diagnostic at step {step} time {t_next:.6f}")
                break
            if rec.idempotency_error > idempotency_stop:
                print(
                    "# stopped:",
                    f"idempotency_error={rec.idempotency_error:.6e}",
                    f"at step {step} time {t_next:.6f}",
                )
                break

        rho_prev, rho = rho, rho_next
        t = t_next

    elapsed = time.perf_counter() - start
    print(f"# propagation_seconds {elapsed:.3f}")

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"# wrote {csv_path}")

    data = np.asarray(rows, dtype=float)
    times = data[:, 1]
    e0 = data[:, 4]
    ecoup = data[:, 5]
    etot = data[:, 6]
    mux, muy, muz = data[:, 7], data[:, 8], data[:, 9]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(times, e0 - e0[0], label="E0 - E0(0)")
    axes[0].plot(times, etot - etot[0], label="Etot - Etot(0)")
    axes[0].plot(times, ecoup, label="Ecoup")
    axes[0].set_ylabel("Energy (Ha)")
    axes[0].legend(loc="best")

    axes[1].plot(times, mux, label="mu_x")
    axes[1].plot(times, muy, label="mu_y")
    axes[1].plot(times, muz, label="mu_z")
    axes[1].set_xlabel("time (a.u.)")
    axes[1].set_ylabel("Electronic dipole (a.u.)")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    print(f"# wrote {plot_path}")


if __name__ == "__main__":
    main()
