#!/usr/bin/env python3
"""Small GPU4PySCF SCEM propagation smoke example for H2."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from pyscf import dft, gto, lib

from aion import LengthGaugeCNRTTDDFT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="cc-pVDZ")
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--nsteps", type=int, default=10)
    parser.add_argument("--kick-z", type=float, default=1.0e-3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def zero_field(_: float) -> np.ndarray:
    return np.zeros(3)


def build_cpu_reference(args: argparse.Namespace):
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
        raise RuntimeError("SCF did not converge")
    return mf


def main() -> None:
    args = parse_args()
    lib.num_threads(args.threads)

    mf_cpu = build_cpu_reference(args)
    mf_gpu = mf_cpu.to_gpu()
    rt = LengthGaugeCNRTTDDFT.from_ground_state(
        mf_gpu,
        zero_field,
        backend="gpu",
    )

    coeff0 = rt.initial_coefficients()
    coeff0 = rt.apply_delta_kick(coeff0, np.array([0.0, 0.0, args.kick_z]))

    rows = []
    start = time.perf_counter()
    for _, rec in rt.propagate_scem(
        coeff0,
        dt=args.dt,
        nsteps=args.nsteps,
        midpoint_tolerance=1.0e-9,
        density_tolerance=1.0e-9,
        max_iterations=12,
        record_energy=False,
    ):
        rows.append(
            {
                "step": rec.step,
                "time_au": rec.time,
                "dipole_x": rec.dipole[0],
                "dipole_y": rec.dipole[1],
                "dipole_z": rec.dipole[2],
                "electron_number": rec.electron_number,
                "orthonormality_error": rec.orthonormality_error,
                "idempotency_error": rec.idempotency_error,
                "midpoint_iterations": rec.midpoint_iterations,
                "fock_builds": rec.fock_builds,
                "hamiltonian_residual": rec.hamiltonian_residual,
                "density_residual": rec.density_residual,
            }
        )
    rt.backend.synchronize()
    elapsed = time.perf_counter() - start

    final = rows[-1]
    print(
        "# h2 gpu scem",
        f"basis={args.basis}",
        f"xc={args.xc}",
        f"dt={args.dt}",
        f"nsteps={args.nsteps}",
        f"seconds={elapsed:.6f}",
        f"seconds_per_step={elapsed / max(1, args.nsteps):.6f}",
        f"final_dipole_z={final['dipole_z']:.12e}",
        f"final_orthonormality_error={final['orthonormality_error']:.3e}",
        flush=True,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
