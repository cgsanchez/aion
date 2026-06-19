#!/usr/bin/env python3
"""Stability scan for resonant CW benzene RT-TDDFT timesteps."""

from __future__ import annotations

import csv
import json
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


def make_row(rec) -> list[float]:
    return [
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


def row_is_finite(row: list[float]) -> bool:
    return bool(np.isfinite(np.asarray(row[1:], dtype=float)).all())


def run_case(rt: LengthGaugeRTTDDFT, *, dt: float, t_final: float, out_path: Path):
    nsteps = int(round(t_final / dt))
    idempotency_stop = 1.0
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

    rho = rt.hermitian_part(rt.initial_density())
    rho_prev = rt.backward_euler_previous(rho, 0.0, dt)

    rows = []
    status = "completed"
    start = time.perf_counter()

    rec0 = rt.record(0, 0.0, rho)
    rows.append(make_row(rec0))

    t = 0.0
    last_rec = rec0
    for step in range(1, nsteps + 1):
        rho_next = rho_prev + 2.0 * dt * rt.rhs(rho, t)
        rho_next = rt.hermitian_part(rho_next)
        t_next = step * dt

        if not np.isfinite(rho_next).all():
            status = "nonfinite_density"
            break

        rec = rt.record(step, t_next, rho_next)
        row = make_row(rec)
        rows.append(row)
        last_rec = rec

        if not row_is_finite(row):
            status = "nonfinite_record"
            break
        if rec.idempotency_error > idempotency_stop:
            status = "idempotency_stop"
            break

        rho_prev, rho = rho, rho_next
        t = t_next

        if step % max(1, nsteps // 10) == 0:
            print(
                "# progress",
                f"dt={dt}",
                f"step={step}/{nsteps}",
                f"time={t_next:.3f}",
                f"idem={rec.idempotency_error:.3e}",
                flush=True,
            )

    elapsed = time.perf_counter() - start

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    return {
        "dt": dt,
        "t_final_requested": t_final,
        "nsteps_requested": nsteps,
        "status": status,
        "rows": len(rows),
        "last_step": int(last_rec.step),
        "last_time": float(last_rec.time),
        "last_idempotency_error": float(last_rec.idempotency_error),
        "last_electron_number": float(last_rec.electron_number),
        "elapsed_seconds": elapsed,
        "csv": str(out_path),
    }


def plot_case(csv_path: Path, plot_path: Path) -> None:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.asarray([data])

    time_au = data["time_au"]
    e0 = data["field_free_energy_ha"]
    ecoup = data["field_coupling_energy_ha"]
    etot = data["total_energy_ha"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(time_au, e0 - e0[0], label="E0 - E0(0)")
    axes[0].plot(time_au, etot - etot[0], label="Etot - Etot(0)")
    axes[0].plot(time_au, ecoup, label="Ecoup")
    axes[0].set_ylabel("Energy (Ha)")
    axes[0].legend(loc="best")

    axes[1].plot(time_au, data["mu_x_au"], label="mu_x")
    axes[1].plot(time_au, data["mu_y_au"], label="mu_y")
    axes[1].plot(time_au, data["mu_z_au"], label="mu_z")
    axes[1].set_ylabel("Dipole (a.u.)")
    axes[1].legend(loc="best")

    axes[2].semilogy(time_au, data["idempotency_error"])
    axes[2].set_ylabel("Idempotency residual")
    axes[2].set_xlabel("time (a.u.)")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def plot_summary(results_dir: Path, summaries: list[dict]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for summary in summaries:
        data = np.genfromtxt(summary["csv"], delimiter=",", names=True)
        if data.ndim == 0:
            data = np.asarray([data])
        label = f"dt={summary['dt']}"
        axes[0].plot(data["time_au"], data["mu_y_au"], label=label)
        axes[1].semilogy(data["time_au"], data["idempotency_error"], label=label)

    axes[0].set_ylabel("mu_y (a.u.)")
    axes[0].legend(loc="best")
    axes[1].set_ylabel("Idempotency residual")
    axes[1].set_xlabel("time (a.u.)")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(results_dir / "timestep_scan_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    metadata, excitations = load_excitations(DEFAULT_EXCITATIONS_PATH)
    driven = excitation_by_index(excitations, int(metadata["selected_root"]))
    polarization = transition_polarization(driven)

    field_v_per_angstrom = 0.5
    field_au = field_v_per_angstrom * V_PER_ANGSTROM_TO_AU_FIELD
    t_final = 10.0
    timesteps = [0.02, 0.01, 0.005]

    print(
        "# timestep scan",
        f"omega={driven.energy:.10f} Ha",
        f"field={field_au:.10e} au",
        f"field={field_v_per_angstrom:.6f} V/Ang",
        f"t_final={t_final}",
        flush=True,
    )

    mf = build_ground_state(verbose=3)
    field = ContinuousWave(
        amplitude=field_au,
        omega=driven.energy,
        polarization=polarization,
    )
    rt = LengthGaugeRTTDDFT.from_ground_state(mf, field, origin=np.zeros(3))

    results_dir = Path(__file__).resolve().parent / "results" / "timestep_scan"
    results_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for dt in timesteps:
        stem = f"benzene_resonant_cw_dt{str(dt).replace('.', 'p')}_t10"
        csv_path = results_dir / f"{stem}.csv"
        plot_path = results_dir / f"{stem}.png"
        print("# case_start", f"dt={dt}", flush=True)
        summary = run_case(rt, dt=dt, t_final=t_final, out_path=csv_path)
        summaries.append(summary)
        plot_case(csv_path, plot_path)
        print("# case_summary", json.dumps(summary, sort_keys=True), flush=True)
        print(f"# wrote {csv_path}", flush=True)
        print(f"# wrote {plot_path}", flush=True)

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    plot_summary(results_dir, summaries)
    print(f"# wrote {summary_path}", flush=True)
    print(f"# wrote {results_dir / 'timestep_scan_summary.png'}", flush=True)


if __name__ == "__main__":
    main()
