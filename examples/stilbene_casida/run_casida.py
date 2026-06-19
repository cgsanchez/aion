#!/usr/bin/env python3
"""Run and save trans-stilbene LR-TDDFT/Casida roots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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

from geometry import format_atom_block, read_xyz, trans_stilbene_xyz  # noqa: E402

from aion import (  # noqa: E402
    casida_excitations,
    lowest_active_excitation,
    save_excitations,
    transition_polarization,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--nstates", type=int, default=8)
    parser.add_argument("--conv-tol", type=float, default=1.0e-8)
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--oscillator-threshold", type=float, default=1.0e-3)
    parser.add_argument("--dipole-threshold", type=float, default=1.0e-4)
    parser.add_argument("--verbose", type=int, default=3)
    parser.add_argument(
        "--geometry",
        type=Path,
        default=EXAMPLE_DIR / "trans_stilbene_optimized.xyz",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def label(value: str) -> str:
    return value.lower().replace("-", "").replace("*", "star")


def main() -> None:
    args = build_parser().parse_args()
    out_path = args.output
    if out_path is None:
        out_path = (
            EXAMPLE_DIR / f"trans_stilbene_{label(args.xc)}_{label(args.basis)}_casida.json"
        )

    if args.geometry.exists():
        atom = format_atom_block(read_xyz(args.geometry))
        geometry_description = f"optimized geometry from {args.geometry.name}"
    else:
        atom = trans_stilbene_xyz()
        geometry_description = "simple generated planar trans-stilbene; not optimized"

    mol = gto.M(
        atom=atom,
        basis=args.basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=args.verbose,
    )

    mf = dft.RKS(mol)
    mf.xc = args.xc
    mf.conv_tol = 1.0e-10
    mf.max_cycle = 160
    mf.diis_space = 12
    mf.grids.level = args.grid_level
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("trans-stilbene ground-state SCF did not converge")

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
    if gap <= 1.0e-5:
        raise RuntimeError(
            "trans-stilbene SCF reference has non-positive or tiny HOMO-LUMO gap"
        )

    _, excitations = casida_excitations(
        mf,
        nstates=args.nstates,
        conv_tol=args.conv_tol,
    )
    driven = lowest_active_excitation(
        excitations,
        oscillator_threshold=args.oscillator_threshold,
        dipole_threshold=args.dipole_threshold,
    )
    polarization = transition_polarization(driven)

    metadata = {
        "molecule": "trans-stilbene",
        "basis": args.basis,
        "xc": args.xc,
        "charge": 0,
        "spin": 0,
        "nstates": args.nstates,
        "singlet": True,
        "grid_level": args.grid_level,
        "oscillator_threshold": args.oscillator_threshold,
        "dipole_threshold": args.dipole_threshold,
        "geometry": geometry_description,
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
