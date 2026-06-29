"""Dense array backends for real-time propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class CPUBackend:
    """NumPy/SciPy dense linear algebra backend."""

    name: str = "cpu"
    is_gpu: bool = False

    def asarray(self, a: Any, dtype: Any = None):
        if hasattr(a, "get"):
            a = a.get()
        return np.asarray(a, dtype=dtype)

    def asnumpy(self, a: Any):
        if hasattr(a, "get"):
            a = a.get()
        return np.asarray(a)

    def copy(self, a: Any):
        return a.copy()

    def eye(self, n: int, dtype: Any = None):
        return np.eye(n, dtype=dtype)

    def exp(self, a: Any):
        return np.exp(a)

    def einsum(self, *args: Any, **kwargs: Any):
        return np.einsum(*args, **kwargs)

    def trace(self, a: Any):
        return np.trace(a)

    def norm(self, a: Any):
        return np.linalg.norm(a)

    def norm_float(self, a: Any) -> float:
        return float(np.linalg.norm(a))

    def real_float(self, a: Any) -> float:
        return float(np.asarray(a).real)

    def min_float(self, a: Any) -> float:
        return float(np.min(a))

    def eigh(self, a: Any):
        return scipy.linalg.eigh(np.asarray(a), check_finite=False)

    def solve(self, a: Any, b: Any):
        return scipy.linalg.solve(a, b, assume_a="gen", check_finite=False)

    def lu_factor(self, a: Any):
        return scipy.linalg.lu_factor(a, check_finite=False)

    def lu_solve(self, lu_and_piv: Any, b: Any):
        return scipy.linalg.lu_solve(lu_and_piv, b, check_finite=False)

    def synchronize(self) -> None:
        return None


class CuPyBackend:
    """CuPy dense linear algebra backend."""

    name = "gpu"
    is_gpu = True

    def __init__(self) -> None:
        import cupy

        self.cp = cupy

    def asarray(self, a: Any, dtype: Any = None):
        return self.cp.asarray(a, dtype=dtype)

    def asnumpy(self, a: Any):
        return self.cp.asnumpy(a) if hasattr(a, "get") else np.asarray(a)

    def copy(self, a: Any):
        return a.copy()

    def eye(self, n: int, dtype: Any = None):
        return self.cp.eye(n, dtype=dtype)

    def exp(self, a: Any):
        return self.cp.exp(a)

    def einsum(self, *args: Any, **kwargs: Any):
        return self.cp.einsum(*args, **kwargs)

    def trace(self, a: Any):
        return self.cp.trace(a)

    def norm(self, a: Any):
        return self.cp.linalg.norm(a)

    def norm_float(self, a: Any) -> float:
        return float(self.cp.asnumpy(self.cp.linalg.norm(a)))

    def real_float(self, a: Any) -> float:
        return float(self.cp.asnumpy(a).real) if hasattr(a, "get") else float(np.asarray(a).real)

    def min_float(self, a: Any) -> float:
        return float(self.cp.asnumpy(self.cp.min(a)))

    def eigh(self, a: Any):
        return self.cp.linalg.eigh(a)

    def solve(self, a: Any, b: Any):
        return self.cp.linalg.solve(a, b)

    def lu_factor(self, a: Any):
        import cupyx.scipy.linalg

        return cupyx.scipy.linalg.lu_factor(a)

    def lu_solve(self, lu_and_piv: Any, b: Any):
        import cupyx.scipy.linalg

        return cupyx.scipy.linalg.lu_solve(lu_and_piv, b)

    def synchronize(self) -> None:
        self.cp.cuda.Stream.null.synchronize()


def make_backend(backend: str | CPUBackend | CuPyBackend | None):
    """Construct an array backend from a name or return an existing backend."""

    if backend is None:
        return CPUBackend()
    if isinstance(backend, (CPUBackend, CuPyBackend)):
        return backend
    if backend == "cpu":
        return CPUBackend()
    if backend in {"gpu", "cupy"}:
        return CuPyBackend()
    raise ValueError(f"unknown array backend {backend!r}")
