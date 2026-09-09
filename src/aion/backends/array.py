"""Strict NumPy/CuPy array backends used by Aion numerical workspaces."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from aion.config import BackendConfig, BackendKind
from aion.errors import BackendError, DeviceResidencyError


@runtime_checkable
class ArrayBackend(Protocol):
    """Small backend boundary with explicit host transfer and residency checks."""

    @property
    def kind(self) -> BackendKind: ...

    @property
    def device_index(self) -> int | None: ...

    @property
    def namespace(self) -> Any: ...

    def asarray(self, value: object, *, dtype: Any | None = None) -> Any: ...

    def empty(self, shape: tuple[int, ...], *, dtype: Any) -> Any: ...

    def zeros(self, shape: tuple[int, ...], *, dtype: Any) -> Any: ...

    def to_host(self, value: object) -> np.ndarray: ...

    def is_resident(self, value: object) -> bool: ...

    def assert_resident(self, value: object, *, name: str = "array") -> None: ...

    def synchronize(self) -> None: ...


class NumPyBackend:
    """IEEE float64/complex128 CPU backend."""

    kind: BackendKind = BackendKind.CPU
    device_index: int | None = None

    @property
    def namespace(self) -> Any:
        return np

    def asarray(self, value: object, *, dtype: Any | None = None) -> np.ndarray:
        if hasattr(value, "get"):
            raise DeviceResidencyError(
                "implicit GPU-to-host transfer is forbidden; use backend.to_host at storage"
            )
        return np.asarray(value, dtype=dtype)

    def empty(self, shape: tuple[int, ...], *, dtype: Any) -> np.ndarray:
        return np.empty(shape, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], *, dtype: Any) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def to_host(self, value: object) -> np.ndarray:
        self.assert_resident(value)
        return np.asarray(value)

    def is_resident(self, value: object) -> bool:
        return isinstance(value, np.ndarray | np.generic)

    def assert_resident(self, value: object, *, name: str = "array") -> None:
        if not self.is_resident(value):
            raise DeviceResidencyError(f"{name} is not a NumPy host array or scalar")

    def synchronize(self) -> None:
        return None


class CuPyBackend:
    """Physical CUDA backend; construction fails instead of falling back."""

    kind: BackendKind = BackendKind.GPU

    def __init__(self, device_index: int = 0) -> None:
        try:
            import cupy as cp
        except Exception as exc:  # pragma: no cover - exercised without the GPU extra
            raise BackendError("GPU backend requested but CuPy cannot be imported") from exc
        try:
            count = int(cp.cuda.runtime.getDeviceCount())
        except Exception as exc:
            raise BackendError("GPU backend requested but CUDA initialization failed") from exc
        if device_index < 0 or device_index >= count:
            raise BackendError(
                f"GPU device {device_index} requested, but CUDA reports {count} device(s)"
            )
        self.device_index: int | None = device_index
        self._cp = cp
        try:
            with cp.cuda.Device(device_index):
                probe = cp.empty(1, dtype=cp.float64)
                probe.fill(0.0)
                cp.cuda.get_current_stream().synchronize()
        except Exception as exc:
            raise BackendError(
                f"GPU device {device_index} cannot execute the required float64 operations"
            ) from exc

    @property
    def namespace(self) -> Any:
        return self._cp

    def asarray(self, value: object, *, dtype: Any | None = None) -> Any:
        with self._cp.cuda.Device(self.device_index):
            return self._cp.asarray(value, dtype=dtype)

    def empty(self, shape: tuple[int, ...], *, dtype: Any) -> Any:
        with self._cp.cuda.Device(self.device_index):
            return self._cp.empty(shape, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], *, dtype: Any) -> Any:
        with self._cp.cuda.Device(self.device_index):
            return self._cp.zeros(shape, dtype=dtype)

    def to_host(self, value: object) -> np.ndarray:
        self.assert_resident(value)
        with self._cp.cuda.Device(self.device_index):
            return np.asarray(self._cp.asnumpy(value))

    def is_resident(self, value: object) -> bool:
        if not isinstance(value, self._cp.ndarray):
            return False
        return int(value.device.id) == self.device_index

    def assert_resident(self, value: object, *, name: str = "array") -> None:
        if not isinstance(value, self._cp.ndarray):
            raise DeviceResidencyError(f"{name} is not a CuPy device array")
        actual = int(value.device.id)
        if actual != self.device_index:
            raise DeviceResidencyError(
                f"{name} resides on CUDA device {actual}, expected device {self.device_index}"
            )

    def synchronize(self) -> None:
        with self._cp.cuda.Device(self.device_index):
            self._cp.cuda.get_current_stream().synchronize()


def make_backend(config: BackendConfig) -> ArrayBackend:
    """Construct exactly the configured backend, with no fallback path."""

    if config.kind is BackendKind.CPU:
        return NumPyBackend()
    if config.device_index is None:  # guarded by BackendConfig, kept for type narrowing
        raise BackendError("GPU backend requires an explicit resolved device index")
    return CuPyBackend(config.device_index)
