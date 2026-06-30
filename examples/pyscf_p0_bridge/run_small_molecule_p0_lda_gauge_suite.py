#!/usr/bin/env python3
"""Small-molecule P0 LDA gauge-regression suite from PySCF matrices."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-aion")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/aion-cache")

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pyscf import dft, gto  # noqa: E402

from aion import (  # noqa: E402
    DEFAULT_GAUGE_LAMBDAS,
    P0SCEMSettings,
    PyscfP0LdaModel,
    PyscfP0Reference,
    run_p0_uniform_electric_gauge_comparison,
    summarize_p0_gauge_errors,
)


DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "results"


@dataclass(frozen=True)
class MoleculeCase:
    name: str
    atom: str
    field: np.ndarray


MOLECULES = {
    "h2": MoleculeCase(
        name="h2",
        atom="H 0 0 -0.37; H 0 0 0.37",
        field=np.array([0.0, 0.0, 0.02]),
    ),
    "co": MoleculeCase(
        name="co",
        atom="C 0 0 -0.640; O 0 0 0.488",
        field=np.array([0.0, 0.0, 0.02]),
    ),
    "n2": MoleculeCase(
        name="n2",
        atom="N 0 0 -0.550; N 0 0 0.550",
        field=np.array([0.0, 0.0, 0.02]),
    ),
    "h2o": MoleculeCase(
        name="h2o",
        atom=(
            "O 0.000000 0.000000 0.000000; "
            "H 0.758602 0.000000 0.504284; "
            "H -0.758602 0.000000 0.504284"
        ),
        field=np.array([0.011, -0.017, 0.023]),
    ),
    "ch4": MoleculeCase(
        name="ch4",
        atom=(
            "C 0.000000 0.000000 0.000000; "
            "H 0.629118 0.629118 0.629118; "
            "H -0.629118 -0.629118 0.629118; "
            "H -0.629118 0.629118 -0.629118; "
            "H 0.629118 -0.629118 -0.629118"
        ),
        field=np.array([0.011, -0.017, 0.023]),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--molecule",
        choices=sorted(MOLECULES),
        action="append",
        help="Molecule to run. Repeat to select several. Defaults to all.",
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="lda,vwn")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--nsteps", type=int, default=3)
    parser.add_argument("--field-scale", type=float, default=1.0)
    parser.add_argument("--mixing", type=float, default=0.7)
    parser.add_argument("--midpoint-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--max-iterations", type=int, default=18)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_reference(
    case: MoleculeCase,
    *,
    basis: str,
    xc: str,
    grid_level: int,
) -> PyscfP0Reference:
    mol = gto.M(
        atom=case.atom,
        basis=basis,
        unit="Angstrom",
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = grid_level
    mf.conv_tol = 1.0e-11
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF did not converge for {case.name}")
    return PyscfP0Reference.from_mean_field(mf)


def _merged_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    preferred = [
        "molecule",
        "gauge",
        "step",
        "time_au",
        "energy",
        "energy_derivative",
        "source_power",
        "dipole_power",
        "power_residual",
        "dipole_power_residual",
        "electron_count",
        "dipole_x",
        "dipole_y",
        "dipole_z",
        "orthonormality_error",
        "midpoint_iterations",
        "hamiltonian_residual",
        "density_residual",
        "max_abs_current",
        "instantaneous_continuity_residual_norm",
    ]
    seen = set(preferred)
    fieldnames = [name for name in preferred if any(name in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_merged_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    selected = args.molecule if args.molecule is not None else sorted(MOLECULES)
    settings = P0SCEMSettings(
        dt=args.dt,
        nsteps=args.nsteps,
        midpoint_tolerance=args.midpoint_tolerance,
        density_tolerance=args.density_tolerance,
        max_iterations=args.max_iterations,
        mixing=args.mixing,
    )

    all_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "basis": args.basis,
        "xc": args.xc,
        "grid_level": args.grid_level,
        "dt": args.dt,
        "nsteps": args.nsteps,
        "field_scale": args.field_scale,
        "gauge_lambdas": DEFAULT_GAUGE_LAMBDAS,
        "molecules": {},
    }

    for name in selected:
        case = MOLECULES[name]
        reference = build_reference(
            case,
            basis=args.basis,
            xc=args.xc,
            grid_level=args.grid_level,
        )
        model = PyscfP0LdaModel.from_reference(reference)
        rows_by_gauge = run_p0_uniform_electric_gauge_comparison(
            geometry_factory=lambda electric, ref=reference: ref.geometry(
                electric=electric
            ),
            model=model,
            occupations=reference.occupations,
            coeff0=reference.initial_coefficients(),
            field=args.field_scale * case.field,
            settings=settings,
            gauge_lambdas=DEFAULT_GAUGE_LAMBDAS,
        )
        for rows in rows_by_gauge.values():
            for row in rows:
                all_rows.append({"molecule": name, **row})

        gauge_summary = summarize_p0_gauge_errors(rows_by_gauge)
        summary["molecules"][name] = {
            "nao": reference.nao,
            "nocc": reference.nocc,
            "nelectron": reference.mol.nelectron,
            "natom": reference.anchors.natom,
            "field": (args.field_scale * case.field).tolist(),
            "gauge_summary": gauge_summary,
        }

        velocity = gauge_summary["velocity"]
        print(
            name,
            f"nao={reference.nao}",
            f"nocc={reference.nocc}",
            "velocity_max_dipole_error={max_dipole_norm_error:.3e}".format(
                **velocity
            ),
            "velocity_max_energy_error={max_energy_abs_error:.3e}".format(
                **velocity
            ),
            "velocity_max_population_error={max_population_norm_error:.3e}".format(
                **velocity
            ),
        )

    csv_path = args.output_dir / "small_molecule_p0_lda_gauge_suite.csv"
    json_path = args.output_dir / "small_molecule_p0_lda_gauge_suite_summary.json"
    write_rows(csv_path, all_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(
        "# small molecule p0 lda gauge suite",
        f"molecules={','.join(selected)}",
        f"basis={args.basis}",
        f"xc={args.xc}",
        f"dt={args.dt}",
        f"nsteps={args.nsteps}",
        f"csv={csv_path}",
        f"summary={json_path}",
    )


if __name__ == "__main__":
    main()
