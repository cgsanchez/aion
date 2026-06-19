#!/usr/bin/env python3
"""Timing comparison for RT-TDDFT Hamiltonian builds."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import pyscf  # noqa: F401
except ImportError:
    pyscf_src = REPO_ROOT.parent / "pyscf"
    if pyscf_src.exists() and str(pyscf_src) not in sys.path:
        sys.path.insert(0, str(pyscf_src))

from pyscf import dft, gto  # noqa: E402

from aion import ContinuousWave, LengthGaugeRTTDDFT  # noqa: E402


AU_FIELD_TO_V_PER_ANGSTROM = 51.4220674763


BENZENE_XYZ = """
C     0.0000000000    1.3970000000    0.0000000000
C     1.2096570000    0.6985000000    0.0000000000
C     1.2096570000   -0.6985000000    0.0000000000
C     0.0000000000   -1.3970000000    0.0000000000
C    -1.2096570000   -0.6985000000    0.0000000000
C    -1.2096570000    0.6985000000    0.0000000000
H     0.0000000000    2.4810000000    0.0000000000
H     2.1487570000    1.2405000000    0.0000000000
H     2.1487570000   -1.2405000000    0.0000000000
H     0.0000000000   -2.4810000000    0.0000000000
H    -2.1487570000   -1.2405000000    0.0000000000
H    -2.1487570000    1.2405000000    0.0000000000
"""


def linear_acene_xyz(nrings: int) -> str:
    """Return a simple planar linear-acene geometry with fused hexagons."""

    if nrings < 1:
        raise ValueError("nrings must be positive")
    cc = 1.397
    ch = 1.084
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    center_step = np.array([np.sqrt(3.0) * cc, 0.0, 0.0])

    carbons = []
    occurrences = []
    for iring in range(nrings):
        center = iring * center_step
        for angle in angles:
            xyz = center + np.array([cc * np.cos(angle), cc * np.sin(angle), 0.0])
            found = None
            for i, old in enumerate(carbons):
                if np.linalg.norm(xyz - old) < 1.0e-8:
                    found = i
                    break
            if found is None:
                carbons.append(xyz)
                occurrences.append(1)
            else:
                occurrences[found] += 1

    skeleton_center = np.mean(carbons, axis=0)
    hydrogens = []
    for carbon, occurrence in zip(carbons, occurrences):
        if occurrence > 1:
            continue
        direction = carbon - skeleton_center
        direction[2] = 0.0
        direction /= np.linalg.norm(direction)
        hydrogens.append(carbon + ch * direction)

    lines = []
    for carbon in carbons:
        lines.append(f"C {carbon[0]: .10f} {carbon[1]: .10f} {carbon[2]: .10f}")
    for hydrogen in hydrogens:
        lines.append(f"H {hydrogen[0]: .10f} {hydrogen[1]: .10f} {hydrogen[2]: .10f}")
    return "\n".join(lines)


def naphthalene_xyz() -> str:
    """Return a simple planar fused-hexagon naphthalene geometry."""

    return linear_acene_xyz(2)


def build_ground_state(name: str, atom: str):
    print(f"# building {name}", flush=True)
    mol = gto.M(
        atom=atom,
        basis="cc-pvdz",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=3,
    )
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.conv_tol = 1.0e-10
    t0 = time.perf_counter()
    mf.kernel()
    scf_seconds = time.perf_counter() - t0
    print(f"# scf_done {name} seconds={scf_seconds:.3f}", flush=True)
    return mf, scf_seconds


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def time_call(fn, *, repeat: int) -> tuple[object, list[float]]:
    result = None
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, times


def benchmark_system(name: str, atom: str, *, omega: float, field_au: float):
    mf, scf_seconds = build_ground_state(name, atom)
    field = ContinuousWave(
        amplitude=field_au,
        omega=omega,
        polarization=np.array([0.0, 1.0, 0.0]),
    )
    rt = LengthGaugeRTTDDFT.from_ground_state(mf, field, origin=np.zeros(3))
    rho = rt.hermitian_part(rt.initial_density())
    dt = 0.01
    repeat = 3

    # Warm up DFT grid/cache paths.
    rt.fock_matrix(rho, 0.0)
    rt.rhs(rho, 0.0)
    rt.record(0, 0.0, rho)

    rho_h = rt.hermitian_part(rho)
    veff, veff_times = time_call(
        lambda: mf.get_veff(mf.mol, rho_h),
        repeat=repeat,
    )
    _, external_times = time_call(
        lambda: rt.external_potential(0.0),
        repeat=repeat,
    )
    fock = rt.hcore + np.asarray(veff, dtype=np.complex128) + rt.external_potential(0.0)
    _, solve_matmul_times = time_call(
        lambda: rt.solve_s_left(fock) @ rho + rho @ rt.solve_s_right(fock),
        repeat=repeat,
    )
    _, fock_times = time_call(
        lambda: rt.fock_matrix(rho, 0.0),
        repeat=repeat,
    )
    _, rhs_times = time_call(
        lambda: rt.rhs(rho, 0.0),
        repeat=repeat,
    )
    _, energy_times = time_call(
        lambda: (rt.field_free_energy(rho), rt.field_coupling_energy(rho, 0.0)),
        repeat=repeat,
    )
    _, light_diag_times = time_call(
        lambda: (
            rt.electron_number(rho),
            rt.idempotency_error(rho),
            rt.electronic_dipole(rho),
        ),
        repeat=repeat,
    )

    rho_prev = rt.backward_euler_previous(rho, 0.0, dt)
    step_times = []
    for step in range(1, repeat + 1):
        t = (step - 1) * dt
        t0 = time.perf_counter()
        rho_next = rho_prev + 2.0 * dt * rt.rhs(rho, t)
        rho_next = rt.hermitian_part(rho_next)
        step_times.append(time.perf_counter() - t0)
        rho_prev, rho = rho, rho_next

    mo_occ = np.asarray(mf.mo_occ)
    nocc = int(np.count_nonzero(mo_occ > 0))
    nvir = int(mo_occ.size - nocc)
    nao = int(rt.s.shape[0])
    ngrids = int(getattr(mf.grids, "coords", np.empty((0, 3))).shape[0])

    return {
        "name": name,
        "atoms": int(mf.mol.natm),
        "electrons": int(mf.mol.nelectron),
        "basis": "cc-pvdz",
        "xc": "pbe",
        "nao": nao,
        "nmo": int(mo_occ.size),
        "nocc_spatial": nocc,
        "nvir_spatial": nvir,
        "casida_ov_dim": nocc * nvir,
        "grids": ngrids,
        "scf_seconds": scf_seconds,
        "timings_seconds": {
            "get_veff": mean_std(veff_times),
            "external_potential": mean_std(external_times),
            "s_solve_and_matmul": mean_std(solve_matmul_times),
            "fock_matrix": mean_std(fock_times),
            "rhs": mean_std(rhs_times),
            "energy_diagnostic": mean_std(energy_times),
            "light_diagnostics": mean_std(light_diag_times),
            "leapfrog_step_no_energy": mean_std(step_times),
        },
    }


def main() -> None:
    omega = 0.2630787689
    field_au = 0.5 / AU_FIELD_TO_V_PER_ANGSTROM
    systems = [
        ("benzene", BENZENE_XYZ),
        ("naphthalene", linear_acene_xyz(2)),
        ("anthracene", linear_acene_xyz(3)),
        ("tetracene", linear_acene_xyz(4)),
    ]

    results = []
    for name, atom in systems:
        result = benchmark_system(name, atom, omega=omega, field_au=field_au)
        results.append(result)
        print("# result", json.dumps(result, sort_keys=True), flush=True)

    ratios = {}
    if len(results) >= 2:
        ref = results[0]
        ratios = {}
        for result in results[1:]:
            ratios[result["name"]] = {
                "nao": result["nao"] / ref["nao"],
                "atoms": result["atoms"] / ref["atoms"],
                "grids": result["grids"] / ref["grids"],
                "get_veff": (
                    result["timings_seconds"]["get_veff"]["mean"]
                    / ref["timings_seconds"]["get_veff"]["mean"]
                ),
                "rhs": (
                    result["timings_seconds"]["rhs"]["mean"]
                    / ref["timings_seconds"]["rhs"]["mean"]
                ),
                "leapfrog_step_no_energy": (
                    result["timings_seconds"]["leapfrog_step_no_energy"]["mean"]
                    / ref["timings_seconds"]["leapfrog_step_no_energy"]["mean"]
                ),
                "energy_diagnostic": (
                    result["timings_seconds"]["energy_diagnostic"]["mean"]
                    / ref["timings_seconds"]["energy_diagnostic"]["mean"]
                ),
            }
        nao = np.asarray([r["nao"] for r in results], dtype=float)
        grids = np.asarray([r["grids"] for r in results], dtype=float)
        get_veff = np.asarray(
            [r["timings_seconds"]["get_veff"]["mean"] for r in results],
            dtype=float,
        )
        rhs = np.asarray(
            [r["timings_seconds"]["rhs"]["mean"] for r in results],
            dtype=float,
        )
        ratios["power_law_fit"] = {
            "get_veff_vs_nao": float(np.polyfit(np.log(nao), np.log(get_veff), 1)[0]),
            "rhs_vs_nao": float(np.polyfit(np.log(nao), np.log(rhs), 1)[0]),
            "get_veff_vs_grids": float(
                np.polyfit(np.log(grids), np.log(get_veff), 1)[0]
            ),
        }
        print("# ratios", json.dumps(ratios, sort_keys=True), flush=True)

    out_dir = Path(__file__).resolve().parent / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hamiltonian_scaling_acenes.json"
    out_path.write_text(
        json.dumps({"results": results, "ratios": ratios}, indent=2, sort_keys=True)
        + "\n"
    )
    print(f"# wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
