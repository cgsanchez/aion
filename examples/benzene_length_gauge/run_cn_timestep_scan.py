#!/usr/bin/env python3
"""Timestep scan for resonant CW benzene CN RT-TDDFT."""

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
    LengthGaugeCNRTTDDFT,
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


def run_case(
    rt: LengthGaugeCNRTTDDFT,
    *,
    coeff0: np.ndarray,
    dt: float,
    t_final: float,
    out_path: Path,
    energy_stride: int,
) -> dict:
    nsteps = int(round(t_final / dt))
    rows = []
    start = time.perf_counter()
    last = None
    status = "completed"

    try:
        for _, rec in rt.propagate(
            coeff0,
            dt=dt,
            nsteps=nsteps,
            record_energy=True,
            energy_stride=energy_stride,
        ):
            row = record_to_row(rec)
            rows.append(row)
            last = rec
            finite_required = [
                rec.time,
                rec.electron_number,
                rec.orthonormality_error,
                rec.idempotency_error,
                rec.field_coupling_energy,
                *rec.dipole.tolist(),
                *rec.field.tolist(),
            ]
            if not np.isfinite(np.asarray(finite_required, dtype=float)).all():
                status = "nonfinite_record"
                break
            if rec.step % max(1, nsteps // 10) == 0:
                print(
                    "# progress",
                    f"dt={dt}",
                    f"step={rec.step}/{nsteps}",
                    f"time={rec.time:.3f}",
                    f"orth={rec.orthonormality_error:.3e}",
                    f"idem={rec.idempotency_error:.3e}",
                    f"mu_y={rec.dipole[1]:.8e}",
                    flush=True,
                )
    except Exception as err:  # noqa: BLE001
        status = f"exception:{type(err).__name__}:{err}"

    elapsed = time.perf_counter() - start
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)

    if last is None:
        return {
            "dt": dt,
            "status": status,
            "rows": 0,
            "elapsed_seconds": elapsed,
            "csv": str(out_path),
        }

    return {
        "dt": dt,
        "t_final_requested": t_final,
        "nsteps_requested": nsteps,
        "status": status,
        "rows": len(rows),
        "last_step": int(last.step),
        "last_time": float(last.time),
        "last_electron_number": float(last.electron_number),
        "last_orthonormality_error": float(last.orthonormality_error),
        "last_idempotency_error": float(last.idempotency_error),
        "last_mu_y": float(last.dipole[1]),
        "elapsed_seconds": elapsed,
        "csv": str(out_path),
    }


def plot_summary(results_dir: Path, summaries: list[dict]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for summary in summaries:
        data = np.genfromtxt(summary["csv"], delimiter=",", names=True)
        if data.ndim == 0:
            data = np.asarray([data])
        label = f"dt={summary['dt']}"
        axes[0].plot(data["time_au"], data["mu_y_au"], label=label)
        axes[1].semilogy(data["time_au"], data["orthonormality_error"], label=label)
        axes[2].semilogy(data["time_au"], data["idempotency_error"], label=label)

    axes[0].set_ylabel("mu_y (a.u.)")
    axes[0].legend(loc="best")
    axes[1].set_ylabel("||C^dag S C - I||")
    axes[1].legend(loc="best")
    axes[2].set_ylabel("||P S P - 2P||")
    axes[2].set_xlabel("time (a.u.)")
    axes[2].legend(loc="best")
    fig.tight_layout()
    fig.savefig(results_dir / "cn_timestep_scan_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    metadata, excitations = load_excitations(DEFAULT_EXCITATIONS_PATH)
    driven = excitation_by_index(excitations, int(metadata["selected_root"]))
    polarization = transition_polarization(driven)

    field_v_per_angstrom = 0.5
    field_au = field_v_per_angstrom * V_PER_ANGSTROM_TO_AU_FIELD
    t_final = 100.0
    timesteps = [0.2, 0.1, 0.05]
    energy_stride = 10

    print(
        "# CN timestep scan",
        f"omega={driven.energy:.10f} Ha",
        f"field={field_au:.10e} au",
        f"field={field_v_per_angstrom:.6f} V/Ang",
        f"t_final={t_final}",
        f"timesteps={timesteps}",
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

    results_dir = Path(__file__).resolve().parent / "results" / "cn_timestep_scan"
    results_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for dt in timesteps:
        stem = f"benzene_cn_resonant_cw_dt{str(dt).replace('.', 'p')}_t100"
        csv_path = results_dir / f"{stem}.csv"
        print("# case_start", f"dt={dt}", flush=True)
        summary = run_case(
            rt,
            coeff0=coeff0,
            dt=dt,
            t_final=t_final,
            out_path=csv_path,
            energy_stride=energy_stride,
        )
        summaries.append(summary)
        print("# case_summary", json.dumps(summary, sort_keys=True), flush=True)
        print(f"# wrote {csv_path}", flush=True)

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    plot_summary(results_dir, summaries)
    print(f"# wrote {summary_path}", flush=True)
    print(f"# wrote {results_dir / 'cn_timestep_scan_summary.png'}", flush=True)


if __name__ == "__main__":
    main()
