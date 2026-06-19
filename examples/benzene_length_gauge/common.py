"""Shared setup for the benzene length-gauge examples."""

from __future__ import annotations

import sys
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
DEFAULT_EXCITATIONS_PATH = EXAMPLE_DIR / "casida_pbe_ccpvdz.json"


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


def build_ground_state(*, verbose: int = 3):
    mol = gto.M(
        atom=BENZENE_XYZ,
        basis="cc-pvdz",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=verbose,
    )

    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.conv_tol = 1.0e-10
    mf.kernel()
    return mf
