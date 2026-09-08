"""Typed, definition-specific observable records independent of propagation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from aion.config.units import AtomicUnit, PhysicalDimension
from aion.errors import SchemaError

_DEFINITION_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$")
_FORMULATION_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")

type ObservableArray = npt.NDArray[np.float64] | npt.NDArray[np.complex128]


class SamplingLocation(StrEnum):
    ENDPOINT = "endpoint"
    MIDPOINT = "midpoint"
    EVENT_PRE = "event_pre"
    EVENT_POST = "event_post"


@dataclass(frozen=True, slots=True)
class ObservableDefinition:
    """Persistent identity, dimensions, shape, and ownership of an observable."""

    definition_id: str
    originating_formulation: str
    physical_dimension: PhysicalDimension
    unit: AtomicUnit
    shape: tuple[int, ...]
    sampling_location: SamplingLocation
    mathematical_definition: str
    decomposition_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _DEFINITION_ID_RE.fullmatch(self.definition_id):
            raise SchemaError("observable definition_id must be a namespaced lowercase identifier")
        if not _FORMULATION_ID_RE.fullmatch(self.originating_formulation):
            raise SchemaError("observable originating_formulation is invalid")
        if not self.mathematical_definition.strip():
            raise SchemaError("observable mathematical_definition cannot be empty")
        object.__setattr__(self, "shape", tuple(self.shape))
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in self.shape
        ):
            raise SchemaError("observable shape entries must be positive integers")
        object.__setattr__(self, "decomposition_labels", tuple(self.decomposition_labels))
        if len(self.decomposition_labels) != len(set(self.decomposition_labels)):
            raise SchemaError("observable decomposition labels must be unique")

    @property
    def hdf5_path(self) -> str:
        return "/observables/" + self.definition_id.replace(".", "/")


@dataclass(frozen=True, slots=True)
class ObservableRecord:
    """One immutable endpoint, midpoint, or event sample."""

    definition: ObservableDefinition
    step: int
    time_au: float
    values: ObservableArray

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise SchemaError("observable step must be a nonnegative integer")
        if isinstance(self.time_au, bool) or not isinstance(self.time_au, int | float):
            raise SchemaError("observable time_au must be a finite number")
        time = float(self.time_au)
        if not math.isfinite(time):
            raise SchemaError("observable time_au must be finite")
        object.__setattr__(self, "time_au", time)
        array = np.asarray(self.values)
        if array.dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
            raise SchemaError("observable values must be float64 or complex128")
        if array.shape != self.definition.shape:
            raise SchemaError(
                f"observable values have shape {array.shape}; expected {self.definition.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise SchemaError("observable values contain non-finite entries")
        immutable = np.array(array, copy=True, order="C")
        immutable.flags.writeable = False
        object.__setattr__(self, "values", immutable)
