#!/usr/bin/env python3
"""Run and save naphthalene LR-TDDFT/Casida roots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
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

from aion import (  # noqa: E402
    casida_excitations,
    lowest_active_excitation,
    save_excitations,
    transition_polarization,
)


def linear_acene_xyz(nrings: int) -> str:
    """Return a simple planar linear-acene geometry with fused hexagons."""

    if nrings < 1:
        raise ValueError("nrings must be positive")
    cc = 1.397
    ch = 1.084
    angles = np.deg2rad([30.0, 90.0, 150.0, 210.0, 270.0, 330.0])
    center_step = np.array([np.sqrt(3.0) * cc, 0.0, 0.0])

    carbons: list[np.ndarray] = []
    occurrences: list[int] = []
    for iring in range(nrings):
        center = iring * center_step
        for angle in angles:
            xyz = center + np.array([cc * np.cos(angle), cc * np.sin(angle), 0.0])
            match = None
            for i, old in enumerate(carbons):
                if np.linalg.norm(xyz - old) < 1.0e-8:
                    match = i
                    break
            if match is None:
                carbons.append(xyz)
                occurrences.append(1)
            else:
                occurrences[match] += 1

    skeleton_center = np.mean(carbons, axis=0)
    hydrogens = []
    for carbon, occurrence in zip(carbons, occurrences, strict=True):
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


def main() -> None:
    out_path = EXAMPLE_DIR / "naphthalene_pbe_ccpvdz_casida.json"
    mol = gto.M(
        atom=linear_acene_xyz(2),
        basis="cc-pvdz",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=3,
    )

    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.conv_tol = 1.0e-10
    mf.max_cycle = 120
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("naphthalene ground-state SCF did not converge")

    homo = mol.nelectron // 2 - 1
    lumo = homo + 1
    gap = float(mf.mo_energy[lumo] - mf.mo_energy[homo])
    print(
        "# scf",
        f"energy={mf.e_tot:.12f}",
        f"homo={mf.mo_energy[homo]:.10f}",
        f"lumo={mf.mo_energy[lumo]:.10f}",
        f"gap={gap:.10f}",
    )

    _, excitations = casida_excitations(mf, nstates=8, conv_tol=1.0e-8)
    driven = lowest_active_excitation(excitations)
    polarization = transition_polarization(driven)

    metadata = {
        "molecule": "naphthalene",
        "basis": "cc-pvdz",
        "xc": "pbe",
        "charge": 0,
        "spin": 0,
        "nstates": 8,
        "singlet": True,
        "geometry": "simple planar fused-hexagon geometry; not optimized",
        "selected_root": driven.index,
        "selected_energy": driven.energy,
        "selected_energy_ev": driven.energy_ev,
        "selected_oscillator_strength": driven.oscillator_strength,
        "selected_polarization": polarization.tolist(),
    }
    save_excitations(out_path, excitations, metadata=metadata)

    print(f"# wrote {out_path}")
    print("# root omega_Ha omega_eV osc_strength mux muy muz |mu|")
    for exc in excitations:
        print(
            exc.index,
            f"{exc.energy:.10f}",
            f"{exc.energy_ev:.6f}",
            f"{exc.oscillator_strength:.6e}",
            *(f"{x:.10e}" for x in exc.transition_dipole),
            f"{exc.dipole_norm:.10e}",
        )
    print(
        "# selected",
        driven.index,
        f"{driven.energy:.10f}",
        f"{driven.energy_ev:.6f}",
        f"{driven.oscillator_strength:.6e}",
        *(f"{x:.10e}" for x in polarization),
    )
    print("# metadata", json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
