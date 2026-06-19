"""Shared utilities for small-molecule RT-TDDFT reference examples."""

from __future__ import annotations

import sys
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]


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


def label_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")

