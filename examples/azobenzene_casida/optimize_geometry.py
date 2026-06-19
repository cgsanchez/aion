#!/usr/bin/env python3
"""Optimize trans-azobenzene geometry with PySCF gradients and SciPy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]

try:
    import pyscf  # noqa: F401
except ImportError:
    pyscf_src = REPO_ROOT.parent / "pyscf"
    if pyscf_src.exists() and str(pyscf_src) not in sys.path:
        sys.path.insert(0, str(pyscf_src))

from pyscf import dft, gto  # noqa: E402

from geometry import (  # noqa: E402
    generated_trans_azobenzene_xyz,
    parse_xyz,
    write_xyz,
)


BOHR_TO_ANGSTROM = 0.529177210903


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--xc", default="pbe")
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--gtol", type=float, default=7.5e-4)
    parser.add_argument("--grid-level", type=int, default=2)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXAMPLE_DIR / "trans_azobenzene_optimized.xyz",
    )
    return parser


def make_mol(
    symbols: list[str],
    coords_bohr: np.ndarray,
    *,
    basis: str,
    verbose: int,
) -> gto.Mole:
    atom = [(sym, coord) for sym, coord in zip(symbols, coords_bohr, strict=True)]
    return gto.M(
        atom=atom,
        basis=basis,
        unit="Bohr",
        charge=0,
        spin=0,
        verbose=verbose,
    )


def make_scf(mol: gto.Mole, *, xc: str, grid_level: int) -> dft.rks.RKS:
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.conv_tol = 1.0e-8
    mf.max_cycle = 120
    mf.diis_space = 12
    mf.grids.level = grid_level
    return mf


def main() -> None:
    args = build_parser().parse_args()
    atoms = parse_xyz(generated_trans_azobenzene_xyz())
    symbols = [sym for sym, _ in atoms]
    coords_angstrom = np.array([xyz for _, xyz in atoms])
    x0 = (coords_angstrom / BOHR_TO_ANGSTROM).reshape(-1)
    eval_count = 0
    best: tuple[float, np.ndarray, np.ndarray] | None = None

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal best, eval_count
        eval_count += 1
        coords_bohr = x.reshape((-1, 3))
        mol = make_mol(symbols, coords_bohr, basis=args.basis, verbose=args.verbose)
        mf = make_scf(mol, xc=args.xc, grid_level=args.grid_level)
        energy = float(mf.kernel())
        if not mf.converged:
            print(
                f"# opt_eval {eval_count:03d} DIIS did not converge; trying Newton",
                flush=True,
            )
            mf = mf.newton()
            mf.conv_tol = 1.0e-8
            mf.max_cycle = 60
            energy = float(mf.kernel())
        if not mf.converged:
            raise RuntimeError(f"SCF did not converge at geometry evaluation {eval_count}")

        grad = np.asarray(mf.nuc_grad_method().kernel(), dtype=float)
        grad_norm = float(np.linalg.norm(grad))
        grad_max = float(np.max(np.abs(grad)))
        gap = float(mf.mo_energy[mol.nelectron // 2] - mf.mo_energy[mol.nelectron // 2 - 1])
        print(
            f"# opt_eval {eval_count:03d} energy {energy:.12f} "
            f"|grad| {grad_norm:.6e} max_grad {grad_max:.6e} gap {gap:.6e}",
            flush=True,
        )

        if best is None or energy < best[0]:
            best = (energy, coords_bohr.copy(), grad.copy())
        return energy, grad.reshape(-1)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": args.maxiter,
            "gtol": args.gtol,
            "ftol": 1.0e-9,
            "maxls": 10,
        },
    )

    if best is None:
        raise RuntimeError("optimizer did not evaluate the objective")

    energy, coords_bohr, grad = best
    coords_angstrom = coords_bohr * BOHR_TO_ANGSTROM
    optimized_atoms = [
        (sym, coord) for sym, coord in zip(symbols, coords_angstrom, strict=True)
    ]
    comment = (
        f"trans-azobenzene optimized with PySCF {args.xc}/{args.basis}; "
        f"energy={energy:.12f} Ha; max_grad={np.max(np.abs(grad)):.6e} Ha/Bohr"
    )
    write_xyz(args.output, optimized_atoms, comment)

    print(f"# optimizer_success {result.success}")
    print(f"# optimizer_message {result.message}")
    print(f"# best_energy {energy:.12f}")
    print(f"# best_max_grad {np.max(np.abs(grad)):.6e}")
    print(f"# wrote {args.output}")


if __name__ == "__main__":
    main()
