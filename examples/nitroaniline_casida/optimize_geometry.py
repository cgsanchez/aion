#!/usr/bin/env python3
"""Optimize para-nitroaniline geometry with PySCF and geomeTRIC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]

try:
    import pyscf  # noqa: F401
except ImportError:
    pyscf_src = REPO_ROOT.parent / "pyscf"
    if pyscf_src.exists() and str(pyscf_src) not in sys.path:
        sys.path.insert(0, str(pyscf_src))

from pyscf import dft, gto  # noqa: E402
from pyscf.geomopt import geometric_solver  # noqa: E402

from geometry import para_nitroaniline_xyz, write_xyz  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--maxsteps", type=int, default=60)
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXAMPLE_DIR / "para_nitroaniline_optimized.xyz",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mol = gto.M(
        atom=para_nitroaniline_xyz(),
        basis=args.basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=args.verbose,
    )

    mf = dft.RKS(mol)
    mf.xc = args.xc
    mf.conv_tol = 1.0e-9
    mf.max_cycle = 120
    mf.diis_space = 12
    mf.grids.level = args.grid_level

    opt_params = {
        "convergence_energy": 1.0e-6,
        "convergence_grms": 3.0e-4,
        "convergence_gmax": 4.5e-4,
        "convergence_drms": 1.2e-3,
        "convergence_dmax": 1.8e-3,
    }
    converged, opt_mol = geometric_solver.kernel(
        mf,
        assert_convergence=True,
        maxsteps=args.maxsteps,
        **opt_params,
    )

    coords = opt_mol.atom_coords(unit="Angstrom")
    atoms = [
        (opt_mol.atom_pure_symbol(i), np.asarray(coords[i], dtype=float))
        for i in range(opt_mol.natm)
    ]
    comment = (
        f"para-nitroaniline optimized with PySCF/geomeTRIC {args.xc}/{args.basis}; "
        f"converged={converged}"
    )
    write_xyz(args.output, atoms, comment)

    print(f"# optimizer_converged {converged}")
    print(f"# wrote {args.output}")


if __name__ == "__main__":
    main()

