#!/usr/bin/env python3
"""Small-molecule P0+E1 pure-DFT gauge-regression suite from PySCF matrices."""

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
    P0E1Model,
    P0SCEMSettings,
    PyscfP0DftModel,
    PyscfP0Reference,
    p0_row_series,
    pyscf_central_dipole_matrices,
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
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--nsteps", type=int, default=3)
    parser.add_argument("--field-scale", type=float, default=0.1)
    parser.add_argument("--mixing", type=float, default=0.7)
    parser.add_argument("--midpoint-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--max-iterations", type=int, default=18)
    parser.add_argument("--max-gauge-error", type=float, default=1.0e-7)
    parser.add_argument("--max-power-residual", type=float, default=1.0e-5)
    parser.add_argument("--no-fail", action="store_true")
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


def vector_series(rows: list[dict[str, object]], prefix: str) -> np.ndarray:
    return np.column_stack([p0_row_series(rows, f"{prefix}_{axis}") for axis in "xyz"])


def max_vector_difference(
    left_rows: list[dict[str, object]],
    left_prefix: str,
    right_rows: list[dict[str, object]],
    right_prefix: str,
) -> float:
    left = vector_series(left_rows, left_prefix)
    right = vector_series(right_rows, right_prefix)
    return float(np.max(np.linalg.norm(left - right, axis=1)))


def _merged_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    preferred = [
        "molecule",
        "model_level",
        "gauge",
        "step",
        "time_au",
        "energy",
        "e1_coupling_energy",
        "energy_derivative",
        "source_power",
        "dipole_power",
        "e1_dipole_power",
        "power_residual",
        "dipole_power_residual",
        "e1_power_residual",
        "electron_count",
        "dipole_x",
        "dipole_y",
        "dipole_z",
        "e1_dipole_x",
        "e1_dipole_y",
        "e1_dipole_z",
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


def max_e1_gauge_error(gauge_summary: dict[str, dict[str, float]]) -> float:
    values = []
    for label in ("mixed", "velocity"):
        values.extend(
            [
                gauge_summary[label]["max_energy_abs_error"],
                gauge_summary[label]["max_population_norm_error"],
                gauge_summary[label]["max_e1_dipole_norm_error"],
            ]
        )
    return float(max(values))


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
    failures: list[str] = []

    for name in selected:
        case = MOLECULES[name]
        reference = build_reference(
            case,
            basis=args.basis,
            xc=args.xc,
            grid_level=args.grid_level,
        )
        base_model = PyscfP0DftModel.from_reference(reference)
        e1_model = P0E1Model(base_model, pyscf_central_dipole_matrices(reference))
        field = args.field_scale * case.field

        p0_rows_by_gauge = run_p0_uniform_electric_gauge_comparison(
            geometry_factory=lambda electric, ref=reference: ref.geometry(
                electric=electric
            ),
            model=base_model,
            occupations=reference.occupations,
            coeff0=reference.initial_coefficients(),
            field=field,
            settings=settings,
            gauge_lambdas=DEFAULT_GAUGE_LAMBDAS,
        )
        e1_rows_by_gauge = run_p0_uniform_electric_gauge_comparison(
            geometry_factory=lambda electric, ref=reference: ref.geometry(
                electric=electric
            ),
            model=e1_model,
            occupations=reference.occupations,
            coeff0=reference.initial_coefficients(),
            field=field,
            settings=settings,
            gauge_lambdas=DEFAULT_GAUGE_LAMBDAS,
        )

        for level, rows_by_gauge in (
            ("p0", p0_rows_by_gauge),
            ("p0_e1", e1_rows_by_gauge),
        ):
            for rows in rows_by_gauge.values():
                for row in rows:
                    all_rows.append(
                        {
                            "molecule": name,
                            "model_level": level,
                            **row,
                        }
                    )

        p0_summary = summarize_p0_gauge_errors(p0_rows_by_gauge)
        e1_summary = summarize_p0_gauge_errors(e1_rows_by_gauge)
        p0_e1_dipole_shift = max_vector_difference(
            p0_rows_by_gauge["length"],
            "dipole",
            e1_rows_by_gauge["length"],
            "e1_dipole",
        )
        intrinsic_e1_dipole = max_vector_difference(
            e1_rows_by_gauge["length"],
            "dipole",
            e1_rows_by_gauge["length"],
            "e1_dipole",
        )
        e1_gauge_error = max_e1_gauge_error(e1_summary)
        e1_power_residual = max(
            values["max_e1_power_residual"] for values in e1_summary.values()
        )
        if e1_gauge_error > args.max_gauge_error:
            failures.append(
                f"{name}: E1 gauge error {e1_gauge_error:.3e} "
                f"> {args.max_gauge_error:.3e}"
            )
        if e1_power_residual > args.max_power_residual:
            failures.append(
                f"{name}: E1 power residual {e1_power_residual:.3e} "
                f"> {args.max_power_residual:.3e}"
            )

        summary["molecules"][name] = {
            "nao": reference.nao,
            "nocc": reference.nocc,
            "nelectron": reference.mol.nelectron,
            "natom": reference.anchors.natom,
            "field": field.tolist(),
            "p0_gauge_summary": p0_summary,
            "p0_e1_gauge_summary": e1_summary,
            "max_p0_site_vs_e1_total_dipole_norm": p0_e1_dipole_shift,
            "max_intrinsic_e1_dipole_norm": intrinsic_e1_dipole,
            "max_e1_gauge_error": e1_gauge_error,
            "max_e1_power_residual": e1_power_residual,
        }

        print(
            name,
            f"nao={reference.nao}",
            f"nocc={reference.nocc}",
            f"e1_gauge_error={e1_gauge_error:.3e}",
            f"e1_power_residual={e1_power_residual:.3e}",
            f"intrinsic_e1_dipole={intrinsic_e1_dipole:.3e}",
            f"p0_to_e1_dipole_shift={p0_e1_dipole_shift:.3e}",
        )

    csv_path = args.output_dir / "small_molecule_p0_e1_gauge_suite.csv"
    json_path = args.output_dir / "small_molecule_p0_e1_gauge_suite_summary.json"
    write_rows(csv_path, all_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(
        "# small molecule p0+e1 gauge suite",
        f"molecules={','.join(selected)}",
        f"basis={args.basis}",
        f"xc={args.xc}",
        f"dt={args.dt}",
        f"nsteps={args.nsteps}",
        f"field_scale={args.field_scale}",
        f"csv={csv_path}",
        f"summary={json_path}",
    )
    if failures:
        print("# threshold failures")
        for failure in failures:
            print(failure)
        if not args.no_fail:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
