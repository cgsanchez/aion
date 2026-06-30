#!/usr/bin/env python3
"""H2O P0 LDA static-B symmetric/Landau gauge comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-aion")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/aion-cache")

import numpy as np
import scipy.linalg

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pyscf import dft, gto  # noqa: E402

from aion import (  # noqa: E402
    P0SCEMSettings,
    PyscfP0LdaModel,
    PyscfP0Reference,
    UniformMagneticGauge,
    run_p0_scem_trajectory,
    summarize_p0_gauge_errors,
    transform_p0_coefficients_between_gauges,
)


DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "results"
H2O_ATOM = (
    "O 0.000000 0.000000 0.000000; "
    "H 0.758602 0.000000 0.504284; "
    "H -0.758602 0.000000 0.504284"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="lda,vwn")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--nsteps", type=int, default=5)
    parser.add_argument("--b-field", type=float, nargs=3, default=[0.0, 0.0, 0.05])
    parser.add_argument("--origin", type=float, nargs=3, default=[0.1, -0.2, 0.0])
    parser.add_argument("--landau-u", type=float, nargs=3, default=[1.0, 0.0, 0.0])
    parser.add_argument("--mixing", type=float, default=0.7)
    parser.add_argument("--midpoint-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--max-iterations", type=int, default=18)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_reference(args: argparse.Namespace) -> PyscfP0Reference:
    mol = gto.M(
        atom=H2O_ATOM,
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


def metric_orthonormalize(coeff: np.ndarray, metric: np.ndarray) -> np.ndarray:
    overlap = coeff.conj().T @ metric @ coeff
    eig, vec = scipy.linalg.eigh(overlap, check_finite=False)
    invsqrt = (vec * eig**-0.5) @ vec.conj().T
    return coeff @ invsqrt


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


def main() -> None:
    args = parse_args()
    reference = build_reference(args)
    model = PyscfP0LdaModel.from_reference(reference)
    b_field = np.asarray(args.b_field, dtype=float)
    origin = np.asarray(args.origin, dtype=float)

    symmetric = reference.geometry(
        magnetic=UniformMagneticGauge(b_field, gauge="symmetric", origin=origin)
    )
    landau = reference.geometry(
        magnetic=UniformMagneticGauge(
            b_field,
            gauge="landau",
            origin=origin,
            landau_u=np.asarray(args.landau_u, dtype=float),
        )
    )
    coeff_symmetric = metric_orthonormalize(
        reference.initial_coefficients(),
        symmetric.metric(0.0),
    )
    coeff_landau = transform_p0_coefficients_between_gauges(
        coeff_symmetric,
        symmetric,
        landau,
    )
    settings = P0SCEMSettings(
        dt=args.dt,
        nsteps=args.nsteps,
        midpoint_tolerance=args.midpoint_tolerance,
        density_tolerance=args.density_tolerance,
        max_iterations=args.max_iterations,
        mixing=args.mixing,
    )
    rows_by_gauge = {
        "symmetric": run_p0_scem_trajectory(
            label="symmetric",
            geometry=symmetric,
            model=model,
            occupations=reference.occupations,
            coeff0=coeff_symmetric,
            settings=settings,
        ),
        "landau": run_p0_scem_trajectory(
            label="landau",
            geometry=landau,
            model=model,
            occupations=reference.occupations,
            coeff0=coeff_landau,
            settings=settings,
        ),
    }
    summary = summarize_p0_gauge_errors(rows_by_gauge, reference_gauge="symmetric")

    csv_path = args.output_dir / "h2o_p0_lda_static_b_gauge_compare.csv"
    summary_path = args.output_dir / "h2o_p0_lda_static_b_gauge_compare_summary.json"
    write_rows(csv_path, rows_by_gauge)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as handle:
        json.dump(
            {
                "basis": args.basis,
                "xc": args.xc,
                "dt": args.dt,
                "nsteps": args.nsteps,
                "b_field": b_field.tolist(),
                "origin": origin.tolist(),
                "landau_u": list(args.landau_u),
                "summary": summary,
            },
            handle,
            indent=2,
        )

    values = summary["landau"]
    print(
        "# h2o p0 lda static b gauge compare",
        f"basis={args.basis}",
        f"xc={args.xc}",
        f"dt={args.dt}",
        f"nsteps={args.nsteps}",
        "landau_max_dipole_error={max_dipole_norm_error:.3e}".format(**values),
        "landau_max_energy_error={max_energy_abs_error:.3e}".format(**values),
        "landau_max_population_error={max_population_norm_error:.3e}".format(**values),
        f"csv={csv_path}",
        f"summary={summary_path}",
    )


if __name__ == "__main__":
    main()
