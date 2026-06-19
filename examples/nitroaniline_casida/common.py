"""Shared setup for para-nitroaniline TDDFT examples."""

from __future__ import annotations

import sys
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
DEFAULT_EXCITATIONS_PATH = EXAMPLE_DIR / "para_nitroaniline_pbe_321g_casida.json"
DEFAULT_GEOMETRY_PATH = EXAMPLE_DIR / "para_nitroaniline_optimized.xyz"


def ensure_local_imports() -> None:
    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    try:
        import pyscf  # noqa: F401
    except ImportError:
        pyscf_src = REPO_ROOT.parent / "pyscf"
        if pyscf_src.exists() and str(pyscf_src) not in sys.path:
            sys.path.insert(0, str(pyscf_src))


ensure_local_imports()

from pyscf import dft, gto  # noqa: E402

from geometry import format_atom_block, para_nitroaniline_xyz, read_xyz  # noqa: E402


def build_ground_state(
    *,
    basis: str,
    xc: str,
    grid_level: int = 1,
    geometry_path: Path = DEFAULT_GEOMETRY_PATH,
    verbose: int = 3,
):
    if geometry_path.exists():
        atom = format_atom_block(read_xyz(geometry_path))
    else:
        atom = para_nitroaniline_xyz()

    mol = gto.M(
        atom=atom,
        basis=basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=verbose,
    )

    mf = dft.RKS(mol)
    mf.xc = xc
    mf.conv_tol = 1.0e-10
    mf.max_cycle = 160
    mf.diis_space = 12
    mf.grids.level = grid_level
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("para-nitroaniline ground-state SCF did not converge")
    return mf

