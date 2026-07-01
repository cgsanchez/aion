#!/usr/bin/env python3
"""CPU/GPU validation for small-molecule P0/P0+E1 kick spectra."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUPY_CACHE_DIR", "/tmp/aion-cupy-cache")
os.environ.setdefault("CUDA_CACHE_PATH", "/tmp/aion-cuda-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-aion")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/aion-cache")

import numpy as np
from pyscf import dft, gto, lib

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aion import (  # noqa: E402
    P0E1Model,
    P0SCEMSettings,
    PyscfP0DftGpuModel,
    PyscfP0DftModel,
    PyscfP0Reference,
    VariableMetricSCEM,
    apply_e1_central_delta_kick,
    apply_p0_velocity_delta_kick,
    kick_spectrum,
    operator_expectations,
    p0_dipole_operator_matrices,
    p0_e1_dipole_operator_matrices,
    pyscf_central_dipole_matrices,
    velocity_delta_kick_electric_gauge,
)


DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "results" / "small_molecule_p0_e1_kick_validation"


@dataclass(frozen=True)
class MoleculeCase:
    name: str
    atom: str
    direction: np.ndarray


MOLECULES = {
    "h2": MoleculeCase(
        name="h2",
        atom="H 0 0 -0.37; H 0 0 0.37",
        direction=np.array([0.0, 0.0, 1.0]),
    ),
    "co": MoleculeCase(
        name="co",
        atom="C 0 0 -0.640; O 0 0 0.488",
        direction=np.array([0.0, 0.0, 1.0]),
    ),
    "n2": MoleculeCase(
        name="n2",
        atom="N 0 0 -0.550; N 0 0 0.550",
        direction=np.array([0.0, 0.0, 1.0]),
    ),
    "h2o": MoleculeCase(
        name="h2o",
        atom=(
            "O 0.000000 0.000000 0.000000; "
            "H 0.758602 0.000000 0.504284; "
            "H -0.758602 0.000000 0.504284"
        ),
        direction=np.array([0.011, -0.017, 0.023]),
    ),
}


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
    parser.add_argument(
        "--molecule",
        choices=sorted(MOLECULES),
        action="append",
        help="Molecule to run. Repeat to select several. Defaults to all.",
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--xc", default="pbe,pbe")
    parser.add_argument("--grid-level", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--nsteps", type=int, default=20)
    parser.add_argument("--kick", type=float, default=1.0e-3)
    parser.add_argument("--midpoint-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--mixing", type=float, default=0.7)
    parser.add_argument("--damping", type=float, default=0.003)
    parser.add_argument("--zero-pad-factor", type=int, default=10)
    parser.add_argument("--max-energy-ev", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_mean_field(case: MoleculeCase, args: argparse.Namespace):
    mol = gto.M(atom=case.atom, basis=args.basis, unit="Angstrom", verbose=0)
    mf = dft.RKS(mol).density_fit()
    mf.xc = args.xc
    mf.grids.level = args.grid_level
    mf.grids.prune = None
    mf.small_rho_cutoff = 0.0
    mf.conv_tol = 1.0e-11
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF did not converge for {case.name}")
    return mf


def as_backend_array(value, backend: str):
    if backend == "gpu":
        import cupy

        return cupy.asarray(value)
    return np.asarray(value)


def run_backend(
    *,
    reference: PyscfP0Reference,
    model,
    central_dipoles,
    backend: str,
    impulse: np.ndarray,
    direction: np.ndarray,
    settings: P0SCEMSettings,
    damping: float,
    zero_pad_factor: int,
    max_energy_ev: float,
) -> dict[str, Any]:
    geometry0 = reference.geometry()
    geometry = reference.geometry(electric=velocity_delta_kick_electric_gauge(impulse))
    coeff = apply_p0_velocity_delta_kick(
        reference.initial_coefficients(),
        geometry0,
        geometry,
    )
    coeff = as_backend_array(coeff, backend)
    coeff = apply_e1_central_delta_kick(
        coeff,
        impulse=impulse,
        central_dipoles0=central_dipoles,
        geometry=geometry,
        backend=backend,
    )

    p0_ops = p0_dipole_operator_matrices(geometry, 0.0, backend=backend)
    e1_ops = p0_e1_dipole_operator_matrices(
        central_dipoles,
        geometry,
        0.0,
        backend=backend,
    )
    rt = VariableMetricSCEM(geometry, model, reference.occupations, backend=backend)
    times = [0.0]
    p0_parallel = []
    e1_parallel = []
    iterations = []
    h_residual = []
    d_residual = []

    start = time.perf_counter()
    rho = rt.density_from_coefficients(coeff)
    p0_mu = operator_expectations(rho, p0_ops, backend=rt.backend)
    e1_mu = operator_expectations(rho, e1_ops, backend=rt.backend)
    p0_parallel.append(float(np.dot(p0_mu, direction)))
    e1_parallel.append(float(np.dot(e1_mu, direction)))

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
        rho = result.rho_next
        p0_mu = operator_expectations(rho, p0_ops, backend=rt.backend)
        e1_mu = operator_expectations(rho, e1_ops, backend=rt.backend)
        times.append(step * settings.dt)
        p0_parallel.append(float(np.dot(p0_mu, direction)))
        e1_parallel.append(float(np.dot(e1_mu, direction)))
        iterations.append(result.iterations)
        h_residual.append(result.hamiltonian_residual)
        d_residual.append(float("nan") if result.density_residual is None else result.density_residual)

    rt.backend.synchronize()
    seconds = time.perf_counter() - start
    metric = rt.backend.asarray(geometry.metric(settings.nsteps * settings.dt), dtype=np.complex128)
    electron_count = rt.backend.real_float(rt.backend.trace(rho @ metric))
    orthonormality = rt.orthonormality_error(coeff, settings.nsteps * settings.dt)
    p0_spec = kick_spectrum(
        times,
        p0_parallel,
        kick_au=float(np.linalg.norm(impulse)),
        damping_ha=damping,
        zero_pad_factor=zero_pad_factor,
        max_energy_ev=max_energy_ev,
    )
    e1_spec = kick_spectrum(
        times,
        e1_parallel,
        kick_au=float(np.linalg.norm(impulse)),
        damping_ha=damping,
        zero_pad_factor=zero_pad_factor,
        max_energy_ev=max_energy_ev,
    )
    base_model = getattr(model, "base_model", model)
    finite_density_residuals = np.asarray(
        [value for value in d_residual if np.isfinite(value)],
        dtype=float,
    )
    return {
        "time": np.asarray(times),
        "p0_parallel": np.asarray(p0_parallel),
        "e1_parallel": np.asarray(e1_parallel),
        "p0_spectrum": p0_spec,
        "e1_spectrum": e1_spec,
        "seconds": seconds,
        "seconds_per_step": seconds / max(1, settings.nsteps),
        "hamiltonian_calls": base_model.hamiltonian_calls,
        "hamiltonian_calls_per_step": base_model.hamiltonian_calls
        / max(1, settings.nsteps),
        "max_midpoint_iterations": int(max(iterations) if iterations else 0),
        "max_hamiltonian_residual": float(max(h_residual) if h_residual else 0.0),
        "max_density_residual": float(
            np.max(finite_density_residuals)
            if finite_density_residuals.size
            else float("nan")
        ),
        "final_electron_count": electron_count,
        "final_orthonormality_error": orthonormality,
    }


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def max_spectrum_abs_error(left, right) -> float:
    if left.strength.size == 0 and right.strength.size == 0:
        return 0.0
    if left.strength.shape != right.strength.shape:
        raise ValueError("spectrum grids do not match")
    return float(np.max(np.abs(left.strength - right.strength)))


def main() -> None:
    args = parse_args()
    lib.num_threads(args.threads)
    selected = args.molecule if args.molecule is not None else sorted(MOLECULES)
    settings = P0SCEMSettings(
        dt=args.dt,
        nsteps=args.nsteps,
        midpoint_tolerance=args.midpoint_tolerance,
        density_tolerance=args.density_tolerance,
        max_iterations=args.max_iterations,
        mixing=args.mixing,
    )

    rows = []
    details: dict[str, Any] = {
        "basis": args.basis,
        "xc": args.xc,
        "grid_level": args.grid_level,
        "dt": args.dt,
        "nsteps": args.nsteps,
        "kick": args.kick,
        "threads": int(lib.num_threads()),
        "molecules": {},
    }

    for name in selected:
        case = MOLECULES[name]
        direction = case.direction / np.linalg.norm(case.direction)
        impulse = args.kick * direction
        mf_cpu = build_mean_field(case, args)
        mf_gpu = mf_cpu.to_gpu()
        reference_cpu = PyscfP0Reference.from_mean_field(mf_cpu)
        reference_gpu = PyscfP0Reference.from_mean_field(mf_gpu)
        base_cpu = CountingModel(
            PyscfP0DftModel.from_reference(reference_cpu, real_density_for_veff=True)
        )
        base_gpu = CountingModel(PyscfP0DftGpuModel.from_reference(reference_gpu))
        central_cpu = pyscf_central_dipole_matrices(reference_cpu)
        central_gpu = as_backend_array(central_cpu, "gpu")
        model_cpu = P0E1Model(base_cpu, central_cpu)
        model_gpu = P0E1Model(base_gpu, central_gpu)

        cpu = run_backend(
            reference=reference_cpu,
            model=model_cpu,
            central_dipoles=central_cpu,
            backend="cpu",
            impulse=impulse,
            direction=direction,
            settings=settings,
            damping=args.damping,
            zero_pad_factor=args.zero_pad_factor,
            max_energy_ev=args.max_energy_ev,
        )
        gpu = run_backend(
            reference=reference_gpu,
            model=model_gpu,
            central_dipoles=central_gpu,
            backend="gpu",
            impulse=impulse,
            direction=direction,
            settings=settings,
            damping=args.damping,
            zero_pad_factor=args.zero_pad_factor,
            max_energy_ev=args.max_energy_ev,
        )

        p0_trace_error = float(np.max(np.abs(gpu["p0_parallel"] - cpu["p0_parallel"])))
        e1_trace_error = float(np.max(np.abs(gpu["e1_parallel"] - cpu["e1_parallel"])))
        p0_spectrum_error = max_spectrum_abs_error(
            gpu["p0_spectrum"],
            cpu["p0_spectrum"],
        )
        e1_spectrum_error = max_spectrum_abs_error(
            gpu["e1_spectrum"],
            cpu["e1_spectrum"],
        )
        speedup = cpu["seconds_per_step"] / gpu["seconds_per_step"]

        row = {
            "molecule": name,
            "nao": reference_cpu.nao,
            "nocc": reference_cpu.nocc,
            "cpu_seconds_per_step": cpu["seconds_per_step"],
            "gpu_seconds_per_step": gpu["seconds_per_step"],
            "speedup": speedup,
            "cpu_hamiltonian_calls_per_step": cpu["hamiltonian_calls_per_step"],
            "gpu_hamiltonian_calls_per_step": gpu["hamiltonian_calls_per_step"],
            "max_p0_trace_abs_error": p0_trace_error,
            "max_e1_trace_abs_error": e1_trace_error,
            "max_p0_spectrum_abs_error": p0_spectrum_error,
            "max_e1_spectrum_abs_error": e1_spectrum_error,
            "gpu_final_electron_count": gpu["final_electron_count"],
            "gpu_final_orthonormality_error": gpu["final_orthonormality_error"],
        }
        rows.append(row)
        details["molecules"][name] = {
            "direction": direction.tolist(),
            "summary": row,
            "cpu": {
                key: value
                for key, value in cpu.items()
                if isinstance(value, (float, int))
            },
            "gpu": {
                key: value
                for key, value in gpu.items()
                if isinstance(value, (float, int))
            },
        }
        print("# result", json.dumps(row, sort_keys=True), flush=True)

    summary_path = args.output_dir / "small_molecule_p0_e1_kick_validation.csv"
    metadata_path = args.output_dir / "small_molecule_p0_e1_kick_validation.json"
    write_summary_csv(summary_path, rows)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(details, indent=2) + "\n")
    print(f"Wrote {summary_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
