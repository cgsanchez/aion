#!/usr/bin/env python3
"""Lean GPU P0/P0+E1 SCEM baseline for the PNA checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUPY_CACHE_DIR", "/tmp/aion-cupy-cache")
os.environ.setdefault("CUDA_CACHE_PATH", "/tmp/aion-cuda-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-aion")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/aion-cache")

import numpy as np
from pyscf import dft, lib
from pyscf.scf import chkfile

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
PNA_DIR = PROJECT_ROOT / "examples" / "nitroaniline_casida"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aion import (  # noqa: E402
    DEFAULT_GAUGE_LAMBDAS,
    P0E1Model,
    P0SCEMSettings,
    PyscfP0DftGpuModel,
    PyscfP0Reference,
    VariableMetricSCEM,
    constant_uniform_electric_gauge,
    pyscf_central_dipole_matrices,
)


DEFAULT_CHKFILE = PNA_DIR / "checkpoints" / "para_nitroaniline_pbe_ccpvdz_grid3.chk"
DEFAULT_GEOMETRY = PNA_DIR / "para_nitroaniline_optimized.xyz"
DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "results" / "pna_p0_e1_gpu_baseline"


class CountingModel:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.hamiltonian_calls = 0

    def hamiltonian(self, density, t: float, geometry):
        self.hamiltonian_calls += 1
        return self.model.hamiltonian(density, t, geometry)

    def energy(self, density, t: float, geometry) -> float:
        return self.model.energy(density, t, geometry)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chkfile", type=Path, default=DEFAULT_CHKFILE)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--grid-level", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--nsteps", type=int, default=5)
    parser.add_argument("--field", type=float, default=1.0e-3)
    parser.add_argument("--gauge", choices=tuple(DEFAULT_GAUGE_LAMBDAS), default="velocity")
    parser.add_argument("--level", choices=("p0", "p0-e1"), default="p0-e1")
    parser.add_argument("--mixing", type=float, default=0.7)
    parser.add_argument("--midpoint-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_xyz(path: Path) -> list[tuple[str, np.ndarray]]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    natom = int(lines[0])
    atoms = []
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


def label_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def json_default(value: Any) -> Any:
    if hasattr(value, "get"):
        value = value.get()
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

    import cupy

    lib.num_threads(args.threads)
    direction = amino_to_nitro_direction(args.geometry)
    field = args.field * direction
    settings = P0SCEMSettings(
        dt=args.dt,
        nsteps=args.nsteps,
        midpoint_tolerance=args.midpoint_tolerance,
        density_tolerance=args.density_tolerance,
        max_iterations=args.max_iterations,
        mixing=args.mixing,
    )

    setup_start = time.perf_counter()
    mf_cpu = load_mean_field(args)
    mf_gpu = mf_cpu.to_gpu()
    reference = PyscfP0Reference.from_mean_field(mf_gpu)
    base_model = CountingModel(PyscfP0DftGpuModel.from_reference(reference))
    if args.level == "p0":
        model = base_model
    else:
        central_dipoles = cupy.asarray(pyscf_central_dipole_matrices(reference))
        model = P0E1Model(base_model, central_dipoles)
    geometry = reference.geometry(
        electric=constant_uniform_electric_gauge(
            field,
            lambda_value=DEFAULT_GAUGE_LAMBDAS[args.gauge],
        )
    )
    rt = VariableMetricSCEM(geometry, model, reference.occupations, backend="gpu")
    coeff = rt.backend.asarray(reference.initial_coefficients(), dtype=np.complex128)
    setup_seconds = time.perf_counter() - setup_start

    start = time.perf_counter()
    max_iterations = 0
    final_h_residual = float("nan")
    final_d_residual = float("nan")
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
    rt.backend.synchronize()
    elapsed = time.perf_counter() - start

    t_final = settings.nsteps * settings.dt
    rho = rt.density_from_coefficients(coeff)
    metric = rt.backend.asarray(geometry.metric(t_final), dtype=np.complex128)
    electron_count = rt.backend.real_float(rt.backend.trace(rho @ metric))
    orthonormality_error = rt.orthonormality_error(coeff, t_final)

    result_data = {
        "backend": "gpu",
        "level": args.level,
        "gauge": args.gauge,
        "chkfile": str(args.chkfile),
        "geometry": str(args.geometry),
        "xc": args.xc,
        "grid_level": args.grid_level,
        "threads": args.threads,
        "nao": reference.nao,
        "nocc": reference.nocc,
        "nelectrons": float(np.sum(reference.occupations)),
        "dt": args.dt,
        "nsteps": args.nsteps,
        "field_amplitude_au": args.field,
        "field_direction_amino_to_nitro": direction,
        "field_vector_au": field,
        "setup_seconds": setup_seconds,
        "seconds": elapsed,
        "seconds_per_step": elapsed / max(1, args.nsteps),
        "base_hamiltonian_calls": base_model.hamiltonian_calls,
        "base_hamiltonian_calls_per_step": base_model.hamiltonian_calls
        / max(1, args.nsteps),
        "max_midpoint_iterations": max_iterations,
        "final_hamiltonian_residual": final_h_residual,
        "final_density_residual": final_d_residual,
        "final_electron_count": electron_count,
        "final_orthonormality_error": orthonormality_error,
    }

    stem = (
        f"pna_{args.level}_gpu_{args.gauge}"
        f"_dt{label_float(args.dt)}_n{args.nsteps}_f{label_float(args.field)}"
    )
    output_path = args.output_dir / f"{stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result_data, indent=2, default=json_default) + "\n")

    print(json.dumps(result_data, sort_keys=True, default=json_default))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
