#!/usr/bin/env python3
"""Benchmark CPU and GPU SCEM propagation with minimal host-transfer overhead."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from pyscf import dft, gto, lib
from pyscf.scf import chkfile

from aion import LengthGaugeCNRTTDDFT


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
PNA_CHKFILE = (
    REPO_ROOT
    / "examples"
    / "nitroaniline_casida"
    / "checkpoints"
    / "para_nitroaniline_pbe_ccpvdz_grid3.chk"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", choices=["h2", "pna"], default="h2")
    parser.add_argument("--backend", choices=["cpu", "gpu", "both"], default="both")
    parser.add_argument("--basis", default="cc-pVDZ")
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--chkfile", type=Path, default=PNA_CHKFILE)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--nsteps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--kick-z", type=float, default=1.0e-3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--midpoint-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--record-energy", action="store_true")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def zero_field(_: float) -> np.ndarray:
    return np.zeros(3)


def build_h2(args: argparse.Namespace):
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis=args.basis,
        unit="Angstrom",
        verbose=0,
    )
    mf = dft.RKS(mol, xc=args.xc).density_fit()
    mf.grids.level = args.grid_level
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("H2 SCF did not converge")
    return mf


def load_pna(args: argparse.Namespace):
    if not args.chkfile.exists():
        raise FileNotFoundError(args.chkfile)
    mol, scf_data = chkfile.load_scf(str(args.chkfile))
    mf = dft.RKS(mol).density_fit()
    mf.xc = args.xc
    mf.grids.level = args.grid_level
    mf.chkfile = str(args.chkfile)
    mf.converged = True
    mf.e_tot = float(scf_data["e_tot"])
    mf.mo_energy = np.asarray(scf_data["mo_energy"])
    mf.mo_coeff = np.asarray(scf_data["mo_coeff"])
    mf.mo_occ = np.asarray(scf_data["mo_occ"])
    return mf


def build_mean_field(args: argparse.Namespace):
    if args.molecule == "h2":
        return build_h2(args)
    if args.molecule == "pna":
        return load_pna(args)
    raise ValueError(args.molecule)


def timed_run(rt: LengthGaugeCNRTTDDFT, args: argparse.Namespace, label: str) -> dict:
    coeff0 = rt.initial_coefficients()
    coeff0 = rt.apply_delta_kick(coeff0, np.array([0.0, 0.0, args.kick_z]))

    if args.warmup_steps > 0:
        rt.run_scem(
            coeff0,
            dt=args.dt,
            nsteps=args.warmup_steps,
            midpoint_tolerance=args.midpoint_tolerance,
            density_tolerance=args.density_tolerance,
            max_iterations=args.max_iterations,
            record_energy=False,
        )
        rt.backend.synchronize()

    start = time.perf_counter()
    summary = rt.run_scem(
        coeff0,
        dt=args.dt,
        nsteps=args.nsteps,
        midpoint_tolerance=args.midpoint_tolerance,
        density_tolerance=args.density_tolerance,
        max_iterations=args.max_iterations,
        record_energy=args.record_energy,
    )
    rt.backend.synchronize()
    elapsed = time.perf_counter() - start
    rec = summary.record_final
    return {
        "label": label,
        "backend": rt.backend.name,
        "molecule": args.molecule,
        "basis": args.basis,
        "xc": args.xc,
        "grid_level": args.grid_level,
        "nao": rt.mol.nao_nr(),
        "nocc": rt.nocc,
        "nelectron": rt.mol.nelectron,
        "threads": int(lib.num_threads()),
        "dt_au": args.dt,
        "nsteps": args.nsteps,
        "warmup_steps": args.warmup_steps,
        "seconds": elapsed,
        "seconds_per_step": elapsed / max(1, args.nsteps),
        "fock_builds": summary.fock_builds,
        "fock_builds_per_step": summary.fock_builds / max(1, args.nsteps),
        "seconds_per_fock_build": elapsed / max(1, summary.fock_builds),
        "midpoint_iterations": summary.midpoint_iterations,
        "midpoint_iterations_per_step": summary.midpoint_iterations / max(1, args.nsteps),
        "final_time_au": rec.time,
        "final_dipole_x": rec.dipole[0],
        "final_dipole_y": rec.dipole[1],
        "final_dipole_z": rec.dipole[2],
        "final_electron_number": rec.electron_number,
        "final_orthonormality_error": rec.orthonormality_error,
        "final_idempotency_error": rec.idempotency_error,
        "final_field_free_energy": rec.field_free_energy,
        "final_total_energy": rec.total_energy,
    }


def write_outputs(rows: list[dict], args: argparse.Namespace) -> None:
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    lib.num_threads(args.threads)
    mf_cpu = build_mean_field(args)

    rows = []
    if args.backend in {"cpu", "both"}:
        rt_cpu = LengthGaugeCNRTTDDFT.from_ground_state(
            mf_cpu,
            zero_field,
            backend="cpu",
        )
        rows.append(timed_run(rt_cpu, args, "cpu"))
        print("# result", json.dumps(rows[-1], sort_keys=True), flush=True)

    if args.backend in {"gpu", "both"}:
        mf_gpu = mf_cpu.to_gpu()
        rt_gpu = LengthGaugeCNRTTDDFT.from_ground_state(
            mf_gpu,
            zero_field,
            backend="gpu",
        )
        rows.append(timed_run(rt_gpu, args, "gpu"))
        print("# result", json.dumps(rows[-1], sort_keys=True), flush=True)

    if len(rows) == 2:
        cpu = next(row for row in rows if row["label"] == "cpu")
        gpu = next(row for row in rows if row["label"] == "gpu")
        print(
            "# speedup",
            f"cpu_seconds_per_step={cpu['seconds_per_step']:.6f}",
            f"gpu_seconds_per_step={gpu['seconds_per_step']:.6f}",
            f"speedup={cpu['seconds_per_step'] / gpu['seconds_per_step']:.3f}",
            flush=True,
        )

    write_outputs(rows, args)


if __name__ == "__main__":
    main()
