"""Mutable backend-local caches and scratch storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aion.backends.array import ArrayBackend
from aion.errors import BackendError


@dataclass(slots=True)
class Workspace:
    """The sole owner of mutable backend state for one simulation.

    A workspace is intentionally not frozen or shareable. Immutable references
    are copied to it once; hot-loop code then consumes only resident arrays.
    """

    backend: ArrayBackend
    arrays: dict[str, Any] = field(default_factory=dict)
    scratch: dict[str, Any] = field(default_factory=dict)
    caches: dict[str, Any] = field(default_factory=dict)
    electronic_model: Any | None = None

    def install_host_array(
        self,
        name: str,
        value: np.ndarray,
        *,
        dtype: Any | None = None,
    ) -> Any:
        if name in self.arrays:
            raise BackendError(f"workspace array {name!r} is already installed")
        resident = self.backend.asarray(value, dtype=dtype)
        self.backend.assert_resident(resident, name=name)
        self.arrays[name] = resident
        return resident

    def install_resident_array(self, name: str, value: Any) -> Any:
        """Register an already-resident derived array without a host round trip."""

        if name in self.arrays:
            raise BackendError(f"workspace array {name!r} is already installed")
        self.backend.assert_resident(value, name=name)
        self.arrays[name] = value
        return value

    def allocate(self, name: str, shape: tuple[int, ...], *, dtype: Any) -> Any:
        if name in self.scratch:
            raise BackendError(f"workspace scratch array {name!r} is already allocated")
        resident = self.backend.empty(shape, dtype=dtype)
        self.backend.assert_resident(resident, name=name)
        self.scratch[name] = resident
        return resident

    def require(self, name: str) -> Any:
        try:
            value = self.arrays[name]
        except KeyError as exc:
            raise BackendError(f"workspace array {name!r} is not installed") from exc
        self.backend.assert_resident(value, name=name)
        return value

    def assert_all_resident(self) -> None:
        for namespace, values in (("arrays", self.arrays), ("scratch", self.scratch)):
            for name, value in values.items():
                self.backend.assert_resident(value, name=f"{namespace}.{name}")
