#!/usr/bin/env python3
"""Diatomic cc-pVDZ Casida vs real-time delta-kick spectra."""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-aion")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/aion-cache")

import matplotlib.pyplot as plt
import numpy as np

from common import EXAMPLE_DIR, label_float

from pyscf import dft, gto  # noqa: E402

from aion import (  # noqa: E402
    HARTREE_TO_EV,
    LengthGaugeCNRTTDDFT,
    casida_excitations,
    load_excitations,
    lowest_active_excitation,
    save_excitations,
    transition_polarization,
)


@dataclass(frozen=True)
class DiatomicSpec:
    label: str
    atom_a: str
    atom_b: str
    bond_angstrom: float

    @property
    def slug(self) -> str:
        return self.label.lower()

    def atom_block(self) -> str:
        half = 0.5 * self.bond_angstrom
        return f"""
{self.atom_a} 0.0 0.0 {-half:.10f}
{self.atom_b} 0.0 0.0 {half:.10f}
"""


DIATOMICS = {
    "h2": DiatomicSpec("H2", "H", "H", 0.7414),
    "lih": DiatomicSpec("LiH", "Li", "H", 1.5956),
    "co": DiatomicSpec("CO", "C", "O", 1.1282),
    "n2": DiatomicSpec("N2", "N", "N", 1.0977),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("molecule", choices=sorted(DIATOMICS))
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--grid-level", type=int, default=3)
    parser.add_argument("--nstates", type=int, default=10)
    parser.add_argument("--integrator", choices=["cn", "ep-pc1", "scem"], default="cn")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--t-final",
        type=float,
        nargs="+",
        default=[500.0, 1000.0],
        help="One or more total propagation times in atomic units.",
    )
    parser.add_argument(
        "--kick",
        type=float,
        default=1.0e-3,
        help="Electric-field impulse magnitude integral E(t) dt in a.u.",
    )
    parser.add_argument("--corrector-iterations", type=int, default=1)
    parser.add_argument("--max-corrector-iterations", type=int, default=6)
    parser.add_argument("--hamiltonian-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--density-tolerance", type=float, default=None)
    parser.add_argument("--final-midpoint-solve", action="store_true")
    parser.add_argument("--ep-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--ep-nonstrict-endpoint", action="store_true")
    parser.add_argument("--scem-midpoint-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--scem-density-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--scem-max-iterations", type=int, default=50)
    parser.add_argument("--scem-initial-mixing", type=float, default=1.0)
    parser.add_argument("--scem-minimum-mixing", type=float, default=0.1)
    parser.add_argument("--energy-stride", type=int, default=None)
    parser.add_argument(
        "--density-stride",
        type=int,
        default=None,
        help="If set, store orthonormal density checkpoints every N steps.",
    )
    parser.add_argument("--damping", type=float, default=0.004)
    parser.add_argument("--max-energy-ev", type=float, default=25.0)
    parser.add_argument("--casida-path", type=Path, default=None)
    parser.add_argument("--rerun-casida", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verbose", type=int, default=3)
    return parser.parse_args()


def basis_slug(basis: str) -> str:
    return basis.lower().replace("-", "").replace("*", "star")


def default_casida_path(spec: DiatomicSpec, basis: str, xc: str) -> Path:
    return EXAMPLE_DIR / f"{spec.slug}_{xc.lower()}_{basis_slug(basis)}_casida.json"


def default_output_dir(spec: DiatomicSpec) -> Path:
    return EXAMPLE_DIR / "results" / f"{spec.slug}_kick"


def build_ground_state(
    spec: DiatomicSpec,
    *,
    basis: str,
    xc: str,
    grid_level: int,
    verbose: int,
):
    mol = gto.M(
        atom=spec.atom_block(),
        basis=basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=verbose,
    )

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.conv_tol = 1.0e-11
    mf.grids.level = grid_level
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"{spec.label} ground-state SCF did not converge")
    return mf


def load_or_run_casida(args: argparse.Namespace, spec: DiatomicSpec, mf):
    casida_path = args.casida_path or default_casida_path(spec, args.basis, args.xc)
    if casida_path.exists() and not args.rerun_casida:
        metadata, excitations = load_excitations(casida_path)
        return casida_path, metadata, excitations

    _, excitations = casida_excitations(mf, nstates=args.nstates, conv_tol=1.0e-9)
    driven = lowest_active_excitation(
        excitations, oscillator_threshold=1.0e-7, dipole_threshold=1.0e-7
    )
    polarization = transition_polarization(driven)
    metadata = {
        "molecule": spec.label,
        "bond_length_angstrom": spec.bond_angstrom,
        "basis": args.basis,
        "xc": args.xc,
        "grid_level": args.grid_level,
        "charge": 0,
        "spin": 0,
        "nstates": args.nstates,
        "singlet": True,
        "selected_root": driven.index,
        "selected_energy": driven.energy,
        "selected_energy_ev": driven.energy_ev,
        "selected_oscillator_strength": driven.oscillator_strength,
        "selected_polarization": polarization.tolist(),
    }
    save_excitations(casida_path, excitations, metadata=metadata)
    return casida_path, metadata, excitations


def zero_field(_: float) -> np.ndarray:
    return np.zeros(3)


def output_stem(
    spec: DiatomicSpec,
    *,
    t_final: float,
    dt: float,
    kick: float,
    corrector_iterations: int,
    hamiltonian_tolerance: float | None,
    density_tolerance: float | None,
    integrator: str,
    ep_strict_endpoint: bool,
) -> str:
    if integrator == "ep-pc1":
        corrector_label = "_epstrict" if ep_strict_endpoint else "_epreuse"
    elif integrator == "scem":
        corrector_label = "_scem"
    elif hamiltonian_tolerance is not None or density_tolerance is not None:
        corrector_label = "_scn"
    elif corrector_iterations == 0:
        corrector_label = ""
    else:
        corrector_label = f"_corr{corrector_iterations}"
    return (
        f"{spec.slug}_kick_"
        f"t{label_float(t_final)}_dt{label_float(dt)}_k{label_float(kick)}"
        f"{corrector_label}"
    )


def propagation_label(args: argparse.Namespace) -> str:
    if args.integrator == "ep-pc1":
        return "_epstrict" if not args.ep_nonstrict_endpoint else "_epreuse"
    if args.integrator == "scem":
        return "_scem"
    if args.hamiltonian_tolerance is not None or args.density_tolerance is not None:
        return "_scn"
    if args.corrector_iterations == 0:
        return ""
    return f"_corr{args.corrector_iterations}"


def write_rows(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "time_au",
                "electron_number",
                "orthonormality_error",
                "idempotency_error",
                "mu_x_au",
                "mu_y_au",
                "mu_z_au",
                "field_free_energy_ha",
                "midpoint_iterations",
                "midpoint_converged",
                "hamiltonian_residual",
                "density_residual",
                "fock_builds",
            ]
        )
        writer.writerows(rows)


def write_spectrum(path: Path, omega: np.ndarray, strength: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["omega_ha", "energy_ev", "strength_arb"])
        for w, s in zip(omega, strength):
            writer.writerow([w, w * HARTREE_TO_EV, s])


def write_density_checkpoints(
    path: Path,
    *,
    time_au: list[float],
    pbar: list[np.ndarray],
    metadata: dict[str, str | float | int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time_au=np.asarray(time_au, dtype=float),
        pbar=np.asarray(pbar, dtype=np.complex128),
        metadata=np.asarray([metadata], dtype=object),
    )


def compute_spectrum(
    rows: list[list[float]],
    *,
    polarization: np.ndarray,
    kick: float,
    damping: float,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(rows, dtype=float)
    time_au = data[:, 1]
    dipole = data[:, 5:8]
    dt = time_au[1] - time_au[0]
    signal = (dipole - dipole[0]) @ polarization
    windowed = signal * np.exp(-damping * time_au)
    alpha = dt * np.fft.fft(windowed) / kick
    omega = 2.0 * np.pi * np.fft.fftfreq(time_au.size, d=dt)
    keep = omega > 0.0
    strength = omega[keep] * np.abs(alpha[keep].imag)
    return omega[keep], strength


def nearest_strength(
    omega: np.ndarray,
    strength: np.ndarray,
    target_omega: float,
) -> tuple[float, float]:
    idx = int(np.argmin(np.abs(omega - target_omega)))
    return float(omega[idx]), float(strength[idx])


def energy_drift(data: np.ndarray) -> float:
    if data.dtype.names is None:
        if data.shape[1] <= 8:
            return float("nan")
        energy = data[:, 8]
    else:
        if "field_free_energy_ha" not in data.dtype.names:
            return float("nan")
        energy = data["field_free_energy_ha"]
    finite = np.isfinite(energy)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    energy = energy[finite]
    return float(energy[-1] - energy[0])


def propagate_one(
    rt: LengthGaugeCNRTTDDFT,
    coeff0: np.ndarray,
    *,
    polarization: np.ndarray,
    kick: float,
    dt: float,
    t_final: float,
    corrector_iterations: int,
    max_corrector_iterations: int | None,
    hamiltonian_tolerance: float | None,
    density_tolerance: float | None,
    final_midpoint_solve: bool,
    integrator: str,
    ep_tolerance: float,
    ep_strict_endpoint: bool,
    scem_midpoint_tolerance: float,
    scem_density_tolerance: float | None,
    scem_max_iterations: int,
    scem_initial_mixing: float,
    scem_minimum_mixing: float,
    energy_stride: int | None,
    density_stride: int | None,
) -> tuple[list[list[float]], list[float], list[np.ndarray]]:
    if density_stride is not None and density_stride <= 0:
        raise ValueError("density_stride must be positive or None")

    coeff = rt.apply_delta_kick(coeff0, kick * polarization)
    nsteps = int(np.ceil(t_final / dt))
    rows = []
    if integrator == "cn":
        propagation = rt.propagate(
            coeff,
            dt=dt,
            nsteps=nsteps,
            corrector_iterations=corrector_iterations,
            max_corrector_iterations=max_corrector_iterations,
            hamiltonian_tolerance=hamiltonian_tolerance,
            density_tolerance=density_tolerance,
            final_midpoint_solve=final_midpoint_solve,
            record_energy=energy_stride is not None,
            energy_stride=energy_stride,
        )
    elif integrator == "ep-pc1":
        propagation = rt.propagate_ep_pc1(
            coeff,
            dt=dt,
            nsteps=nsteps,
            tolerance=ep_tolerance,
            max_corrector_iterations=max_corrector_iterations,
            strict_endpoint_hamiltonian=ep_strict_endpoint,
            record_energy=energy_stride is not None,
            energy_stride=energy_stride,
        )
    elif integrator == "scem":
        propagation = rt.propagate_scem(
            coeff,
            dt=dt,
            nsteps=nsteps,
            midpoint_tolerance=scem_midpoint_tolerance,
            density_tolerance=scem_density_tolerance,
            max_iterations=scem_max_iterations,
            initial_mixing=scem_initial_mixing,
            minimum_mixing=scem_minimum_mixing,
            record_energy=energy_stride is not None,
            energy_stride=energy_stride,
        )
    else:
        raise ValueError(f"unknown integrator {integrator}")

    density_times: list[float] = []
    density_pbar: list[np.ndarray] = []
    for coeff_step, rec in propagation:
        rows.append(
            [
                rec.step,
                rec.time,
                rec.electron_number,
                rec.orthonormality_error,
                rec.idempotency_error,
                *rec.dipole.tolist(),
                np.nan if rec.field_free_energy is None else rec.field_free_energy,
                rec.midpoint_iterations,
                1 if rec.midpoint_converged else 0,
                (
                    np.nan
                    if rec.hamiltonian_residual is None
                    else rec.hamiltonian_residual
                ),
                np.nan if rec.density_residual is None else rec.density_residual,
                rec.fock_builds,
            ]
        )
        if density_stride is not None and rec.step % density_stride == 0:
            rho_step = rt.density_from_coefficients(coeff_step)
            density_times.append(rec.time)
            density_pbar.append(rt.to_orthonormal_density(rho_step))
    return rows, density_times, density_pbar


def plot_comparison(
    path: Path,
    spectra: list[tuple[float, np.ndarray, np.ndarray]],
    rows_by_time: list[tuple[float, list[list[float]]]],
    excitations,
    *,
    selected_root: int,
    max_energy_ev: float,
    polarization: np.ndarray,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    for t_final, rows in rows_by_time:
        data = np.asarray(rows, dtype=float)
        time_au = data[:, 1]
        dipole_parallel = data[:, 5:8] @ polarization
        axes[0].plot(
            time_au,
            dipole_parallel - dipole_parallel[0],
            label=f"T={t_final:g} au",
        )
    axes[0].set_title(title)
    axes[0].set_xlabel("time (a.u.)")
    axes[0].set_ylabel("Delta mu_parallel (a.u.)")
    axes[0].legend(loc="best")

    for t_final, omega, strength in spectra:
        energy_ev = omega * HARTREE_TO_EV
        mask = energy_ev <= max_energy_ev
        scale = np.max(strength[mask]) if np.any(mask) else np.max(strength)
        if scale <= 0:
            scale = 1.0
        axes[1].plot(energy_ev[mask], strength[mask] / scale, label=f"RT T={t_final:g} au")

    projected = [
        max(0.0, float(exc.energy * abs(np.dot(exc.transition_dipole, polarization)) ** 2))
        for exc in excitations
    ]
    max_projected = max(projected, default=1.0)
    if max_projected <= 0:
        max_projected = 1.0
    for exc, projected_strength in zip(excitations, projected):
        if exc.energy_ev > max_energy_ev:
            continue
        height = projected_strength / max_projected
        if height < 1.0e-4:
            continue
        color = "C3" if exc.index == selected_root else "0.35"
        axes[1].vlines(exc.energy_ev, 0.0, height, color=color, linewidth=1.2)
        axes[1].text(
            exc.energy_ev,
            min(1.02, height + 0.04),
            str(exc.index),
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )
    axes[1].set_xlim(0.0, max_energy_ev)
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_xlabel("energy (eV)")
    axes[1].set_ylabel("normalized RT strength / Casida sticks")
    axes[1].legend(loc="best")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    spec = DIATOMICS[args.molecule]
    output_dir = args.output_dir or default_output_dir(spec)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"# {spec.label} kick spectrum",
        f"basis={args.basis}",
        f"xc={args.xc}",
        f"grid_level={args.grid_level}",
        f"bond={spec.bond_angstrom}",
        f"integrator={args.integrator}",
        f"dt={args.dt}",
        f"t_final={args.t_final}",
        f"kick={args.kick}",
        f"corrector_iterations={args.corrector_iterations}",
        f"max_corrector_iterations={args.max_corrector_iterations}",
        f"hamiltonian_tolerance={args.hamiltonian_tolerance}",
        f"density_tolerance={args.density_tolerance}",
        f"final_midpoint_solve={args.final_midpoint_solve}",
        f"ep_tolerance={args.ep_tolerance}",
        f"ep_strict_endpoint={not args.ep_nonstrict_endpoint}",
        f"scem_midpoint_tolerance={args.scem_midpoint_tolerance}",
        f"scem_density_tolerance={args.scem_density_tolerance}",
        f"scem_max_iterations={args.scem_max_iterations}",
        f"scem_initial_mixing={args.scem_initial_mixing}",
        f"scem_minimum_mixing={args.scem_minimum_mixing}",
        f"energy_stride={args.energy_stride}",
        f"density_stride={args.density_stride}",
        f"damping={args.damping}",
        flush=True,
    )

    mf = build_ground_state(
        spec,
        basis=args.basis,
        xc=args.xc,
        grid_level=args.grid_level,
        verbose=args.verbose,
    )
    casida_path, metadata, excitations = load_or_run_casida(args, spec, mf)
    driven = lowest_active_excitation(
        excitations, oscillator_threshold=1.0e-7, dipole_threshold=1.0e-7
    )
    polarization = transition_polarization(driven)

    print(f"# casida_path {casida_path}", flush=True)
    print("# root omega_Ha omega_eV osc_strength mux muy muz", flush=True)
    for exc in excitations:
        print(
            exc.index,
            f"{exc.energy:.10f}",
            f"{exc.energy_ev:.6f}",
            f"{exc.oscillator_strength:.6e}",
            *(f"{x:.10e}" for x in exc.transition_dipole),
            flush=True,
        )
    print(
        "# selected",
        driven.index,
        f"{driven.energy:.10f}",
        f"{driven.energy_ev:.6f}",
        f"polarization={polarization.tolist()}",
        flush=True,
    )

    rt = LengthGaugeCNRTTDDFT.from_ground_state(mf, zero_field, origin=np.zeros(3))
    coeff0 = rt.initial_coefficients()

    spectra = []
    rows_by_time = []
    for t_final in args.t_final:
        stem = output_stem(
            spec,
            t_final=t_final,
            dt=args.dt,
            kick=args.kick,
            corrector_iterations=args.corrector_iterations,
            hamiltonian_tolerance=args.hamiltonian_tolerance,
            density_tolerance=args.density_tolerance,
            integrator=args.integrator,
            ep_strict_endpoint=not args.ep_nonstrict_endpoint,
        )
        start = time.perf_counter()
        rows, density_times, density_pbar = propagate_one(
            rt,
            coeff0,
            polarization=polarization,
            kick=args.kick,
            dt=args.dt,
            t_final=t_final,
            corrector_iterations=args.corrector_iterations,
            max_corrector_iterations=args.max_corrector_iterations,
            hamiltonian_tolerance=args.hamiltonian_tolerance,
            density_tolerance=args.density_tolerance,
            final_midpoint_solve=args.final_midpoint_solve,
            integrator=args.integrator,
            ep_tolerance=args.ep_tolerance,
            ep_strict_endpoint=not args.ep_nonstrict_endpoint,
            scem_midpoint_tolerance=args.scem_midpoint_tolerance,
            scem_density_tolerance=args.scem_density_tolerance,
            scem_max_iterations=args.scem_max_iterations,
            scem_initial_mixing=args.scem_initial_mixing,
            scem_minimum_mixing=args.scem_minimum_mixing,
            energy_stride=args.energy_stride,
            density_stride=args.density_stride,
        )
        elapsed = time.perf_counter() - start
        omega, strength = compute_spectrum(
            rows,
            polarization=polarization,
            kick=args.kick,
            damping=args.damping,
        )
        rows_out = output_dir / f"{stem}.csv"
        spec_out = output_dir / f"{stem}_spectrum.csv"
        density_out = output_dir / f"{stem}_density.npz"
        write_rows(rows_out, rows)
        write_spectrum(spec_out, omega, strength)
        if args.density_stride is not None:
            write_density_checkpoints(
                density_out,
                time_au=density_times,
                pbar=density_pbar,
                metadata={
                    "molecule": spec.label,
                    "basis": args.basis,
                    "xc": args.xc,
                    "integrator": args.integrator,
                    "dt_au": args.dt,
                    "t_final_au": t_final,
                    "kick": args.kick,
                    "density_stride": args.density_stride,
                },
            )
        nearest_omega, nearest_value = nearest_strength(omega, strength, driven.energy)
        resolution_ha = 2.0 * np.pi / (len(rows) * args.dt)
        data = np.asarray(rows, dtype=float)
        h_res = data[:, 11]
        d_res = data[:, 12]
        fock_builds = data[1:, 13] if data.shape[0] > 1 else data[:, 13]
        finite_h = h_res[np.isfinite(h_res)]
        finite_d = d_res[np.isfinite(d_res)]
        print(
            "# rt",
            f"T={t_final:g}",
            f"steps={len(rows) - 1}",
            f"seconds={elapsed:.3f}",
            f"resolution={resolution_ha:.6e} Ha",
            f"resolution={resolution_ha * HARTREE_TO_EV:.4f} eV",
            f"nearest_selected_bin={nearest_omega:.10f} Ha",
            f"nearest_selected_bin_ev={nearest_omega * HARTREE_TO_EV:.6f}",
            f"nearest_strength={nearest_value:.6e}",
            f"N_min={np.min(data[:, 2]):.12f}",
            f"N_max={np.max(data[:, 2]):.12f}",
            f"orth_max={np.max(data[:, 3]):.3e}",
            f"idem_max={np.max(data[:, 4]):.3e}",
            f"energy_drift={energy_drift(data):.6e}",
            f"midpoint_iter_max={int(np.max(data[:, 9]))}",
            f"h_residual_max={np.max(finite_h) if finite_h.size else np.nan:.3e}",
            f"density_residual_max={np.max(finite_d) if finite_d.size else np.nan:.3e}",
            f"fock_builds_avg={np.mean(fock_builds):.3f}",
            f"fock_builds_max={int(np.max(fock_builds))}",
            f"rows={rows_out}",
            f"spectrum={spec_out}",
            f"density={density_out if args.density_stride is not None else None}",
            flush=True,
        )
        spectra.append((t_final, omega, strength))
        rows_by_time.append((t_final, rows))

    plot_out = output_dir / (
        f"{spec.slug}_kick_spectrum_"
        f"dt{label_float(args.dt)}_k{label_float(args.kick)}"
        f"{propagation_label(args)}.png"
    )
    plot_comparison(
        plot_out,
        spectra,
        rows_by_time,
        excitations,
        selected_root=int(metadata.get("selected_root", driven.index)),
        max_energy_ev=args.max_energy_ev,
        polarization=polarization,
        title=f"{spec.label} {args.xc}/{args.basis} delta kick",
    )
    print(f"# wrote {plot_out}", flush=True)


if __name__ == "__main__":
    main()
