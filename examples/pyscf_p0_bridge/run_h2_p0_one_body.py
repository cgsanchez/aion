#!/usr/bin/env python3
"""H2 pure Peierls P0 one-body propagation from PySCF AO matrices."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-aion")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/aion-cache")

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pyscf import dft, gto  # noqa: E402

from aion import (  # noqa: E402
    PyscfP0Reference,
    UniformElectricGauge,
    VariableMetricSCEM,
    density_from_coefficients,
    p0_dipole_moment,
    p0_site_populations,
)


DEFAULT_OUTPUT = EXAMPLE_DIR / "results" / "h2_p0_one_body.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="lda,vwn")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--nsteps", type=int, default=10)
    parser.add_argument("--field-z", type=float, default=0.025)
    parser.add_argument("--gauge", choices=["length", "velocity"], default="length")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
    mf.grids.level = 0
    mf.conv_tol = 1.0e-11
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return PyscfP0Reference.from_mean_field(mf)


def electric_source(args: argparse.Namespace) -> UniformElectricGauge:
    field = np.array([0.0, 0.0, args.field_z], dtype=float)
    if args.gauge == "length":
        return UniformElectricGauge.length(
            field=lambda _t: field,
            field_integral=lambda t: field * t,
        )
    return UniformElectricGauge.velocity(
        field=lambda _t: field,
        field_integral=lambda t: field * t,
    )


def main() -> None:
    args = parse_args()
    reference = build_reference(args)
    geometry = reference.geometry(electric=electric_source(args))
    model = reference.linear_model()
    rt = VariableMetricSCEM(geometry, model, reference.occupations)

    coeff = reference.initial_coefficients()
    rows = []
    for step in range(args.nsteps + 1):
        time = step * args.dt
        rho = density_from_coefficients(coeff, reference.occupations)
        dipole = p0_dipole_moment(rho, geometry, time)
        populations = p0_site_populations(rho, geometry, time)
        rows.append(
            {
                "step": step,
                "time_au": time,
                "dipole_x": dipole[0],
                "dipole_y": dipole[1],
                "dipole_z": dipole[2],
                "population_0": populations[0],
                "population_1": populations[1],
                "orthonormality_error": rt.orthonormality_error(coeff, time),
            }
        )
        if step < args.nsteps:
            coeff = rt.step(
                coeff,
                time=time,
                dt=args.dt,
                midpoint_tolerance=1.0e-12,
                density_tolerance=None,
                max_iterations=8,
            ).coeff_next

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    final = rows[-1]
    print(
        "# h2 p0 one-body",
        f"basis={args.basis}",
        f"xc={args.xc}",
        f"gauge={args.gauge}",
        f"dt={args.dt}",
        f"nsteps={args.nsteps}",
        f"final_dipole_z={final['dipole_z']:.12e}",
        f"final_orthonormality_error={final['orthonormality_error']:.3e}",
        f"output={args.output}",
    )


if __name__ == "__main__":
    main()
