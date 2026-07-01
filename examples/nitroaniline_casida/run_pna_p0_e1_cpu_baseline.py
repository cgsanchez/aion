#!/usr/bin/env python3
"""CPU P0/P0+E1 gauge baseline for the PNA PySCF checkpoint.

This is intentionally a short, fully diagnosed run.  It checks that the
current gauge-invariant P0 and P0+E1 machinery can use the stored PNA SCF
reference, and it provides a conservative timing estimate for planning longer
kick/spectrum trajectories.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pyscf import dft, lib
from pyscf.scf import chkfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-aion")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/aion-cache")

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aion import (  # noqa: E402
    DEFAULT_GAUGE_LAMBDAS,
    P0E1Model,
    P0SCEMSettings,
    PyscfP0DftModel,
    PyscfP0Reference,
    VariableMetricSCEM,
    coefficient_orthonormality_error,
    constant_uniform_electric_gauge,
    density_from_coefficients,
    electron_count,
    p0_row_series,
    pyscf_central_dipole_matrices,
    run_p0_scem_trajectory,
    summarize_p0_gauge_errors,
)


DEFAULT_CHKFILE = (
    EXAMPLE_DIR / "checkpoints" / "para_nitroaniline_pbe_ccpvdz_grid3.chk"
)
DEFAULT_GEOMETRY = EXAMPLE_DIR / "para_nitroaniline_optimized.xyz"
DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "results" / "p0_e1_cpu_baseline"


class CountingModel:
    """Count expensive base-model calls while preserving the model API."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.hamiltonian_calls = 0
        self.energy_calls = 0

    def hamiltonian(self, density: np.ndarray, t: float, geometry: Any) -> np.ndarray:
        self.hamiltonian_calls += 1
        return self.model.hamiltonian(density, t, geometry)

    def energy(self, density: np.ndarray, t: float, geometry: Any) -> float:
        self.energy_calls += 1
        return self.model.energy(density, t, geometry)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chkfile", type=Path, default=DEFAULT_CHKFILE)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--grid-level", type=int, default=3)
    parser.add_argument(
        "--no-density-fit",
        action="store_true",
        help="Use the ordinary PySCF J builder instead of density fitting.",
    )
    parser.add_argument(
        "--complex-density-for-veff",
        action="store_true",
        help=(
            "Pass the complex Hermitian AO density to PySCF get_veff. "
            "The default passes the real part, which is required by the "
            "CPU density-fitted J builder and is appropriate for pure LDA/GGA."
        ),
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--nsteps", type=int, default=2)
    parser.add_argument(
        "--field",
        type=float,
        default=1.0e-3,
        help="Constant uniform electric field amplitude in atomic units.",
    )
    parser.add_argument("--mixing", type=float, default=0.7)
    parser.add_argument("--midpoint-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument(
        "--propagation-only",
        action="store_true",
        help=(
            "Benchmark SCEM stepping without trajectory diagnostics. "
            "This is closer to the cost of a lean kick/spectrum run."
        ),
    )
    parser.add_argument(
        "--level",
        choices=("p0", "p0-e1"),
        action="append",
        help="Model level to run. Repeat to select both. Defaults to both.",
    )
    parser.add_argument(
        "--gauge",
        choices=tuple(DEFAULT_GAUGE_LAMBDAS),
        action="append",
        help="Gauge to run. Repeat to select several. Defaults to length/mixed/velocity.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_xyz(path: Path) -> list[tuple[str, np.ndarray]]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    natom = int(lines[0])
    atoms: list[tuple[str, np.ndarray]] = []
    for line in lines[2 : 2 + natom]:
        fields = line.split()
        atoms.append((fields[0], np.array([float(x) for x in fields[1:4]])))
    return atoms


def amino_to_nitro_direction(path: Path) -> np.ndarray:
    atoms = read_xyz(path)
    nitrogens = [xyz for sym, xyz in atoms if sym.upper() == "N"]
    if len(nitrogens) != 2:
        raise ValueError(f"expected exactly two nitrogens, found {len(nitrogens)}")
    amino = min(nitrogens, key=lambda xyz: xyz[0])
    nitro = max(nitrogens, key=lambda xyz: xyz[0])
    direction = nitro - amino
    return direction / np.linalg.norm(direction)


def load_mean_field(args: argparse.Namespace):
    mol, scf_data = chkfile.load_scf(str(args.chkfile))
    mf = dft.RKS(mol)
    if not args.no_density_fit:
        mf = mf.density_fit()
    mf.xc = args.xc
    mf.grids.level = args.grid_level
    mf.chkfile = str(args.chkfile)
    mf.converged = True
    mf.e_tot = float(scf_data["e_tot"])
    mf.mo_energy = np.asarray(scf_data["mo_energy"])
    mf.mo_coeff = np.asarray(scf_data["mo_coeff"])
    mf.mo_occ = np.asarray(scf_data["mo_occ"])
    return mf


def _merged_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    preferred = [
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
    names = [name for name in preferred if any(name in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_merged_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def vector_series(rows: list[dict[str, object]], prefix: str) -> np.ndarray:
    return np.column_stack([p0_row_series(rows, f"{prefix}_{axis}") for axis in "xyz"])


def row_summary(rows_by_gauge: dict[str, list[dict[str, object]]]) -> dict[str, Any]:
    summary = summarize_p0_gauge_errors(rows_by_gauge)
    for gauge, rows in rows_by_gauge.items():
        summary[gauge]["max_midpoint_iterations"] = int(
            np.max(p0_row_series(rows, "midpoint_iterations"))
        )
        summary[gauge]["max_energy_abs"] = float(
            np.max(np.abs(p0_row_series(rows, "energy")))
        )
        dipole = vector_series(rows, "dipole")
        summary[gauge]["max_dipole_norm"] = float(
            np.max(np.linalg.norm(dipole, axis=1))
        )
        if all(f"e1_dipole_{axis}" in rows[0] for axis in "xyz"):
            e1_dipole = vector_series(rows, "e1_dipole")
            summary[gauge]["max_e1_dipole_norm"] = float(
                np.max(np.linalg.norm(e1_dipole, axis=1))
            )
    return summary


def run_propagation_only(
    *,
    geometry: Any,
    model: Any,
    occupations: np.ndarray,
    coeff0: np.ndarray,
    settings: P0SCEMSettings,
) -> dict[str, float | int]:
    rt = VariableMetricSCEM(geometry, model, occupations)
    coeff = np.asarray(coeff0, dtype=np.complex128).copy()
    max_iterations = 0
    final_h_residual = float("nan")
    final_d_residual = float("nan")

    start = time.perf_counter()
    for step in range(1, settings.nsteps + 1):
        result = rt.step(
            coeff,
            time=(step - 1) * settings.dt,
            dt=settings.dt,
            midpoint_tolerance=settings.midpoint_tolerance,
            density_tolerance=settings.density_tolerance,
            max_iterations=settings.max_iterations,
            mixing=settings.mixing,
        )
        coeff = result.coeff_next
        max_iterations = max(max_iterations, result.iterations)
        final_h_residual = result.hamiltonian_residual
        final_d_residual = (
            float("nan")
            if result.density_residual is None
            else result.density_residual
        )
    elapsed = time.perf_counter() - start

    metric = geometry.metric(settings.nsteps * settings.dt)
    rho = density_from_coefficients(coeff, occupations)
    return {
        "seconds": elapsed,
        "seconds_per_step": elapsed / max(1, settings.nsteps),
        "max_midpoint_iterations": max_iterations,
        "final_hamiltonian_residual": final_h_residual,
        "final_density_residual": final_d_residual,
        "final_electron_count": electron_count(rho, metric),
        "final_orthonormality_error": coefficient_orthonormality_error(coeff, metric),
    }


def label_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main() -> None:
    args = parse_args()
    if args.dt <= 0.0:
        raise ValueError("dt must be positive")
    if args.nsteps < 0:
        raise ValueError("nsteps must be nonnegative")

    lib.num_threads(args.threads)
    settings = P0SCEMSettings(
        dt=args.dt,
        nsteps=args.nsteps,
        midpoint_tolerance=args.midpoint_tolerance,
        density_tolerance=args.density_tolerance,
        max_iterations=args.max_iterations,
        mixing=args.mixing,
    )
    levels = args.level if args.level is not None else ["p0", "p0-e1"]
    gauges = args.gauge if args.gauge is not None else list(DEFAULT_GAUGE_LAMBDAS)
    lambdas = {name: DEFAULT_GAUGE_LAMBDAS[name] for name in gauges}

    setup_start = time.perf_counter()
    direction = amino_to_nitro_direction(args.geometry)
    field = args.field * direction
    mf = load_mean_field(args)
    reference = PyscfP0Reference.from_mean_field(mf)
    base_model = PyscfP0DftModel.from_reference(
        reference,
        real_density_for_veff=not args.complex_density_for_veff,
    )
    central_dipoles0 = pyscf_central_dipole_matrices(reference)
    setup_seconds = time.perf_counter() - setup_start

    stem = (
        f"pna_p0_e1_cpu_dt{label_float(args.dt)}"
        f"_n{args.nsteps}_f{label_float(args.field)}"
    )
    if args.propagation_only:
        stem += "_propagation"
    csv_path = args.output_dir / f"{stem}.csv"
    json_path = args.output_dir / f"{stem}.json"

    all_rows: list[dict[str, object]] = []
    summary: dict[str, Any] = {
        "chkfile": str(args.chkfile),
        "geometry": str(args.geometry),
        "xc": args.xc,
        "grid_level": args.grid_level,
        "density_fit": not args.no_density_fit,
        "real_density_for_veff": not args.complex_density_for_veff,
        "threads": args.threads,
        "dt": args.dt,
        "nsteps": args.nsteps,
        "run_mode": "propagation_only" if args.propagation_only else "diagnostic",
        "field_amplitude_au": args.field,
        "field_direction_amino_to_nitro": direction,
        "field_vector_au": field,
        "nao": reference.nao,
        "nocc": reference.nocc,
        "nelectrons": float(np.sum(reference.occupations)),
        "setup_seconds": setup_seconds,
        "levels": {},
    }

    print(
        "PNA P0/P0+E1 CPU baseline: "
        f"nao={reference.nao} nocc={reference.nocc} "
        f"threads={args.threads} dt={args.dt} nsteps={args.nsteps}"
    )

    for level in levels:
        rows_by_gauge: dict[str, list[dict[str, object]]] = {}
        level_timings: dict[str, dict[str, float | int]] = {}
        propagation_summaries: dict[str, dict[str, float | int]] = {}
        for gauge_label, lambda_value in lambdas.items():
            counted_base = CountingModel(base_model)
            model: Any
            if level == "p0":
                model = counted_base
            else:
                model = P0E1Model(counted_base, central_dipoles0)

            geometry = reference.geometry(
                electric=constant_uniform_electric_gauge(
                    field,
                    lambda_value=lambda_value,
                )
            )
            if args.propagation_only:
                propagation_summary = run_propagation_only(
                    geometry=geometry,
                    model=model,
                    occupations=reference.occupations,
                    coeff0=reference.initial_coefficients(),
                    settings=settings,
                )
                elapsed = float(propagation_summary["seconds"])
                propagation_summaries[gauge_label] = propagation_summary
            else:
                start = time.perf_counter()
                rows = run_p0_scem_trajectory(
                    label=gauge_label,
                    geometry=geometry,
                    model=model,
                    occupations=reference.occupations,
                    coeff0=reference.initial_coefficients(),
                    settings=settings,
                )
                elapsed = time.perf_counter() - start
                for row in rows:
                    row["model_level"] = level
                rows_by_gauge[gauge_label] = rows
                all_rows.extend(rows)
            fock_like_calls = counted_base.hamiltonian_calls + counted_base.energy_calls
            level_timings[gauge_label] = {
                "seconds": elapsed,
                "seconds_per_step": elapsed / max(1, args.nsteps),
                "base_hamiltonian_calls": counted_base.hamiltonian_calls,
                "base_energy_calls": counted_base.energy_calls,
                "fock_like_calls_including_energy": fock_like_calls,
                "fock_like_calls_per_step": fock_like_calls / max(1, args.nsteps),
            }
            print(
                f"  {level:5s} {gauge_label:8s}: "
                f"{elapsed:.3f} s, {elapsed / max(1, args.nsteps):.3f} s/step, "
                f"H={counted_base.hamiltonian_calls}, E={counted_base.energy_calls}"
            )

        level_summary: dict[str, Any] = {"timings": level_timings}
        if args.propagation_only:
            level_summary["propagation_summary"] = propagation_summaries
        else:
            level_summary["gauge_summary"] = row_summary(rows_by_gauge)
        summary["levels"][level] = level_summary

    if all_rows:
        write_rows(csv_path, all_rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, default=json_default) + "\n")

    if all_rows:
        print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
