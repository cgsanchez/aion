#!/usr/bin/env python3
"""GPU P0/P0+E1 velocity-gauge delta-kick trajectory and spectra."""

from __future__ import annotations

import argparse
import csv
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
    P0E1Model,
    P0SCEMSettings,
    PyscfP0DftGpuModel,
    PyscfP0Reference,
    VariableMetricSCEM,
    apply_e1_central_delta_kick,
    apply_p0_velocity_delta_kick,
    kick_spectrum,
    operator_expectations,
    p0_dipole_operator_matrices,
    p0_e1_dipole_operator_matrices,
    p0_e2_second_moment_operator_matrices,
    pyscf_central_dipole_matrices,
    pyscf_central_second_moment_matrices,
    traceless_quadrupole_from_second_moment,
    velocity_delta_kick_electric_gauge,
    write_kick_spectrum_csv,
)


DEFAULT_CHKFILE = PNA_DIR / "checkpoints" / "para_nitroaniline_pbe_ccpvdz_grid3.chk"
DEFAULT_GEOMETRY = PNA_DIR / "para_nitroaniline_optimized.xyz"
DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "results" / "p0_e1_kick_spectrum"
AXES = ("x", "y", "z")
TENSOR_LABELS = tuple(a + b for a in AXES for b in AXES)


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
    parser.add_argument("--t-final", type=float, default=1000.0)
    parser.add_argument("--kick", type=float, default=1.0e-3)
    parser.add_argument("--level", choices=("p0", "p0-e1"), default="p0-e1")
    parser.add_argument(
        "--kick-level",
        choices=("p0", "p0-e1"),
        default=None,
        help="Impulse operator. Defaults to level.",
    )
    parser.add_argument(
        "--direction",
        choices=("x", "y", "z", "amino-to-nitro"),
        default="amino-to-nitro",
    )
    parser.add_argument(
        "--direction-vector",
        type=float,
        nargs=3,
        help="Explicit kick direction; overrides --direction.",
    )
    parser.add_argument("--midpoint-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--mixing", type=float, default=0.7)
    parser.add_argument("--diag-stride", type=int, default=100)
    parser.add_argument("--progress-stride", type=int, default=100)
    parser.add_argument("--checkpoint-stride", type=int, default=1000)
    parser.add_argument("--damping", type=float, default=0.003)
    parser.add_argument("--zero-pad-factor", type=int, default=10)
    parser.add_argument("--max-energy-ev", type=float, default=20.0)
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


def parse_direction(args: argparse.Namespace) -> np.ndarray:
    if args.direction_vector is not None:
        direction = np.asarray(args.direction_vector, dtype=float)
    elif args.direction == "amino-to-nitro":
        direction = amino_to_nitro_direction(args.geometry)
    else:
        direction = np.zeros(3)
        direction[AXES.index(args.direction)] = 1.0
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError("kick direction must be nonzero")
    return direction / norm


def label_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


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


def trajectory_fieldnames() -> list[str]:
    fields = [
        "step",
        "time_au",
        "p0_mu_x",
        "p0_mu_y",
        "p0_mu_z",
        "p0_mu_parallel",
        "e1_mu_x",
        "e1_mu_y",
        "e1_mu_z",
        "e1_mu_parallel",
    ]
    fields.extend([f"e2_moment_{label}" for label in TENSOR_LABELS])
    fields.extend([f"quadrupole_{label}" for label in TENSOR_LABELS])
    fields.extend(
        [
            "electron_count",
            "orthonormality_error",
            "midpoint_iterations",
            "hamiltonian_residual",
            "density_residual",
            "hamiltonian_calls",
            "seconds_elapsed",
        ]
    )
    return fields


def write_trajectory_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trajectory_fieldnames())
        writer.writeheader()


def append_trajectory_row(path: Path, row: dict[str, object]) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trajectory_fieldnames())
        writer.writerow(row)


def save_checkpoint(path: Path, *, backend, step: int, time_au: float, coeff) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        step=np.asarray(step, dtype=int),
        time_au=np.asarray(time_au, dtype=float),
        coeff=backend.asnumpy(coeff),
    )


def observable_row(
    *,
    step: int,
    time_au: float,
    rho,
    coeff,
    geometry,
    backend,
    p0_dipole_ops,
    e1_dipole_ops,
    e2_second_ops,
    direction: np.ndarray,
    midpoint_iterations: int,
    hamiltonian_residual: float | None,
    density_residual: float | None,
    hamiltonian_calls: int,
    seconds_elapsed: float,
    include_health: bool,
) -> dict[str, object]:
    p0_mu = operator_expectations(rho, p0_dipole_ops, backend=backend)
    e1_mu = operator_expectations(rho, e1_dipole_ops, backend=backend)
    second = operator_expectations(rho, e2_second_ops, backend=backend)
    quadrupole = traceless_quadrupole_from_second_moment(second)
    row: dict[str, object] = {
        "step": step,
        "time_au": time_au,
        "p0_mu_x": p0_mu[0],
        "p0_mu_y": p0_mu[1],
        "p0_mu_z": p0_mu[2],
        "p0_mu_parallel": float(np.dot(p0_mu, direction)),
        "e1_mu_x": e1_mu[0],
        "e1_mu_y": e1_mu[1],
        "e1_mu_z": e1_mu[2],
        "e1_mu_parallel": float(np.dot(e1_mu, direction)),
        "midpoint_iterations": midpoint_iterations,
        "hamiltonian_residual": hamiltonian_residual,
        "density_residual": density_residual,
        "hamiltonian_calls": hamiltonian_calls,
        "seconds_elapsed": seconds_elapsed,
    }
    for label, value in zip(TENSOR_LABELS, second.reshape(-1)):
        row[f"e2_moment_{label}"] = float(value)
    for label, value in zip(TENSOR_LABELS, quadrupole.reshape(-1)):
        row[f"quadrupole_{label}"] = float(value)
    if include_health:
        metric = backend.asarray(geometry.metric(time_au), dtype=np.complex128)
        row["electron_count"] = backend.real_float(backend.trace(rho @ metric))
        overlap = coeff.conj().T @ metric @ coeff
        eye = backend.eye(coeff.shape[1], dtype=np.complex128)
        row["orthonormality_error"] = backend.norm_float(overlap - eye)
    else:
        row["electron_count"] = None
        row["orthonormality_error"] = None
    return row


def main() -> None:
    args = parse_args()
    if args.dt <= 0.0:
        raise ValueError("dt must be positive")
    if args.t_final <= 0.0:
        raise ValueError("t-final must be positive")
    for name in ("diag_stride", "progress_stride", "checkpoint_stride", "zero_pad_factor"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    kick_level = args.level if args.kick_level is None else args.kick_level
    if kick_level == "p0-e1" and args.level != "p0-e1":
        raise ValueError("--kick-level p0-e1 requires --level p0-e1")
    nsteps = int(round(args.t_final / args.dt))
    if not np.isclose(nsteps * args.dt, args.t_final):
        raise ValueError("t-final must be an integer multiple of dt")

    import cupy

    lib.num_threads(args.threads)
    direction = parse_direction(args)
    impulse = args.kick * direction
    settings = P0SCEMSettings(
        dt=args.dt,
        nsteps=nsteps,
        midpoint_tolerance=args.midpoint_tolerance,
        density_tolerance=args.density_tolerance,
        max_iterations=args.max_iterations,
        mixing=args.mixing,
    )

    stem = (
        f"p0_e1_kick_{args.level}_{kick_level}"
        f"_t{label_float(args.t_final)}_dt{label_float(args.dt)}"
        f"_k{label_float(args.kick)}"
    )
    trajectory_path = args.output_dir / f"{stem}.csv"
    p0_spectrum_path = args.output_dir / f"{stem}_p0_spectrum.csv"
    e1_spectrum_path = args.output_dir / f"{stem}_e1_spectrum.csv"
    metadata_path = args.output_dir / f"{stem}.json"
    checkpoint_path = args.output_dir / f"{stem}_checkpoint.npz"

    setup_start = time.perf_counter()
    mf_gpu = load_mean_field(args).to_gpu()
    reference = PyscfP0Reference.from_mean_field(mf_gpu)
    base_model = CountingModel(PyscfP0DftGpuModel.from_reference(reference))
    central_dipoles = cupy.asarray(pyscf_central_dipole_matrices(reference))
    central_second = cupy.asarray(pyscf_central_second_moment_matrices(reference))
    model: Any = base_model if args.level == "p0" else P0E1Model(base_model, central_dipoles)

    free_geometry = reference.geometry()
    kicked_geometry = reference.geometry(
        electric=velocity_delta_kick_electric_gauge(impulse)
    )
    coeff0 = apply_p0_velocity_delta_kick(
        reference.initial_coefficients(),
        free_geometry,
        kicked_geometry,
    )
    coeff = cupy.asarray(coeff0, dtype=cupy.complex128)
    if kick_level == "p0-e1":
        coeff = apply_e1_central_delta_kick(
            coeff,
            impulse=impulse,
            central_dipoles0=central_dipoles,
            geometry=kicked_geometry,
            backend="gpu",
        )

    p0_dipole_ops = p0_dipole_operator_matrices(kicked_geometry, 0.0, backend="gpu")
    e1_dipole_ops = p0_e1_dipole_operator_matrices(
        central_dipoles,
        kicked_geometry,
        0.0,
        backend="gpu",
    )
    e2_second_ops = p0_e2_second_moment_operator_matrices(
        central_dipoles0=central_dipoles,
        central_second_moments0=central_second,
        geometry=kicked_geometry,
        t=0.0,
        backend="gpu",
    )
    rt = VariableMetricSCEM(
        kicked_geometry,
        model,
        reference.occupations,
        backend="gpu",
    )
    setup_seconds = time.perf_counter() - setup_start

    metadata: dict[str, object] = {
        "backend": "gpu",
        "integrator": "variable-metric SCEM",
        "level": args.level,
        "kick_level": kick_level,
        "chkfile": str(args.chkfile),
        "geometry": str(args.geometry),
        "xc": args.xc,
        "grid_level": args.grid_level,
        "threads": int(lib.num_threads()),
        "nao": reference.nao,
        "nocc": reference.nocc,
        "nelectrons": float(np.sum(reference.occupations)),
        "dt_au": args.dt,
        "t_final_au": args.t_final,
        "nsteps": nsteps,
        "kick_au": args.kick,
        "direction": direction.tolist(),
        "impulse_au": impulse.tolist(),
        "damping_ha": args.damping,
        "zero_pad_factor": args.zero_pad_factor,
        "max_energy_ev": args.max_energy_ev,
        "setup_seconds": setup_seconds,
        "trajectory_path": str(trajectory_path),
        "p0_spectrum_path": str(p0_spectrum_path),
        "e1_spectrum_path": str(e1_spectrum_path),
        "checkpoint_path": str(checkpoint_path),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    write_trajectory_header(trajectory_path)

    start = time.perf_counter()
    rho = rt.density_from_coefficients(coeff)
    times = [0.0]
    p0_parallel = []
    e1_parallel = []
    row = observable_row(
        step=0,
        time_au=0.0,
        rho=rho,
        coeff=coeff,
        geometry=kicked_geometry,
        backend=rt.backend,
        p0_dipole_ops=p0_dipole_ops,
        e1_dipole_ops=e1_dipole_ops,
        e2_second_ops=e2_second_ops,
        direction=direction,
        midpoint_iterations=0,
        hamiltonian_residual=None,
        density_residual=None,
        hamiltonian_calls=0,
        seconds_elapsed=0.0,
        include_health=True,
    )
    append_trajectory_row(trajectory_path, row)
    p0_parallel.append(float(row["p0_mu_parallel"]))
    e1_parallel.append(float(row["e1_mu_parallel"]))

    for step in range(1, nsteps + 1):
        t = (step - 1) * settings.dt
        t_next = step * settings.dt
        calls_before = base_model.hamiltonian_calls
        result = rt.step(
            coeff,
            time=t,
            dt=settings.dt,
            midpoint_tolerance=settings.midpoint_tolerance,
            density_tolerance=settings.density_tolerance,
            max_iterations=settings.max_iterations,
            mixing=settings.mixing,
        )
        coeff = result.coeff_next
        rho = result.rho_next
        calls_step = base_model.hamiltonian_calls - calls_before
        elapsed = time.perf_counter() - start
        include_health = step % args.diag_stride == 0 or step == nsteps
        row = observable_row(
            step=step,
            time_au=t_next,
            rho=rho,
            coeff=coeff,
            geometry=kicked_geometry,
            backend=rt.backend,
            p0_dipole_ops=p0_dipole_ops,
            e1_dipole_ops=e1_dipole_ops,
            e2_second_ops=e2_second_ops,
            direction=direction,
            midpoint_iterations=result.iterations,
            hamiltonian_residual=result.hamiltonian_residual,
            density_residual=result.density_residual,
            hamiltonian_calls=calls_step,
            seconds_elapsed=elapsed,
            include_health=include_health,
        )
        append_trajectory_row(trajectory_path, row)
        times.append(t_next)
        p0_parallel.append(float(row["p0_mu_parallel"]))
        e1_parallel.append(float(row["e1_mu_parallel"]))

        if step % args.checkpoint_stride == 0 or step == nsteps:
            rt.backend.synchronize()
            save_checkpoint(
                checkpoint_path,
                backend=rt.backend,
                step=step,
                time_au=t_next,
                coeff=coeff,
            )
            metadata.update(
                {
                    "last_step": step,
                    "seconds_elapsed": elapsed,
                    "base_hamiltonian_calls": base_model.hamiltonian_calls,
                }
            )
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

        if step % args.progress_stride == 0 or step == nsteps:
            print(
                "# progress",
                f"step={step}/{nsteps}",
                f"time_au={t_next:.3f}",
                f"seconds={elapsed:.1f}",
                f"sec_per_step={elapsed / step:.4f}",
                f"H_calls={base_model.hamiltonian_calls}",
                flush=True,
            )

    rt.backend.synchronize()
    total_seconds = time.perf_counter() - start
    p0_spectrum = kick_spectrum(
        times,
        p0_parallel,
        kick_au=args.kick,
        damping_ha=args.damping,
        zero_pad_factor=args.zero_pad_factor,
        max_energy_ev=args.max_energy_ev,
    )
    e1_spectrum = kick_spectrum(
        times,
        e1_parallel,
        kick_au=args.kick,
        damping_ha=args.damping,
        zero_pad_factor=args.zero_pad_factor,
        max_energy_ev=args.max_energy_ev,
    )
    write_kick_spectrum_csv(p0_spectrum_path, p0_spectrum)
    write_kick_spectrum_csv(e1_spectrum_path, e1_spectrum)
    metadata.update(
        {
            "completed": True,
            "seconds_total": total_seconds,
            "seconds_per_step": total_seconds / max(1, nsteps),
            "base_hamiltonian_calls": base_model.hamiltonian_calls,
            "base_hamiltonian_calls_per_step": base_model.hamiltonian_calls
            / max(1, nsteps),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        "# done",
        f"trajectory={trajectory_path}",
        f"p0_spectrum={p0_spectrum_path}",
        f"e1_spectrum={e1_spectrum_path}",
        f"seconds={total_seconds:.1f}",
        f"sec_per_step={total_seconds / max(1, nsteps):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
