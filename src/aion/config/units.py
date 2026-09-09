"""Atomic-unit, origin, and fixed-time-grid value types.

All field names carrying a numerical quantity include ``_au`` at the input
boundary.  These types do not perform implicit unit conversion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from aion.errors import ConfigurationError

type Vector3 = tuple[float, float, float]


class PhysicalDimension(StrEnum):
    """Dimensions used by configuration and HDF5 metadata."""

    DIMENSIONLESS = "dimensionless"
    METADATA = "metadata"
    TIME = "time"
    LENGTH = "length"
    ENERGY = "energy"
    ELECTRIC_FIELD = "electric_field"
    VECTOR_POTENTIAL_REDUCED = "vector_potential_reduced"
    DIPOLE = "electric_dipole"
    CHARGE = "electric_charge"
    CHARGE_FLOW_RATE = "electric_charge_flow_rate"
    CURRENT = "electric_current"
    POWER = "power"
    DENSITY_MATRIX = "ao_density_matrix"


class AtomicUnit(StrEnum):
    """Canonical unit labels written to Aion artifacts."""

    ONE = "1"
    TIME = "atomic_unit_of_time"
    LENGTH = "bohr"
    ENERGY = "hartree"
    ELECTRIC_FIELD = "atomic_unit_of_electric_field"
    VECTOR_POTENTIAL_REDUCED = "atomic_unit_of_reduced_vector_potential"
    DIPOLE = "electron_bohr"
    CHARGE = "elementary_charge"
    CHARGE_FLOW_RATE = "elementary_charge_per_atomic_unit_of_time"
    CURRENT = "electron_per_atomic_unit_of_time_times_bohr"
    POWER = "hartree_per_atomic_unit_of_time"


def finite_float(value: object, path: str) -> float:
    """Return a finite non-boolean float or raise a path-specific error."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{path} must be finite")
    return result


def vector3(value: object, path: str) -> Vector3:
    """Validate a three-component Cartesian vector."""

    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ConfigurationError(f"{path} must contain exactly three components")
    return (
        finite_float(value[0], f"{path}[0]"),
        finite_float(value[1], f"{path}[1]"),
        finite_float(value[2], f"{path}[2]"),
    )


@dataclass(frozen=True, slots=True)
class ElectromagneticOrigin:
    """Explicit Cartesian origin shared by all electromagnetic observables."""

    position_au: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_au", vector3(self.position_au, "position_au"))


@dataclass(frozen=True, slots=True)
class FixedTimeGrid:
    """Endpoint-inclusive uniform grid with authoritative integer indices."""

    start_au: float
    step_au: float
    intervals: int

    def __post_init__(self) -> None:
        start = finite_float(self.start_au, "time_grid.start_au")
        step = finite_float(self.step_au, "time_grid.step_au")
        if step <= 0.0:
            raise ConfigurationError("time_grid.step_au must be positive")
        if isinstance(self.intervals, bool) or not isinstance(self.intervals, int):
            raise ConfigurationError("time_grid.intervals must be an integer")
        if self.intervals < 1:
            raise ConfigurationError("time_grid.intervals must be at least one")
        object.__setattr__(self, "start_au", start)
        object.__setattr__(self, "step_au", step)

    @property
    def state_count(self) -> int:
        """Number of endpoint states, including both initial and final states."""

        return self.intervals + 1

    @property
    def end_au(self) -> float:
        """Final endpoint time computed from the integer grid."""

        return self.time_at(self.intervals)

    def time_at(self, step: int) -> float:
        """Return ``t_0 + step*dt`` after validating the endpoint index."""

        if isinstance(step, bool) or not isinstance(step, int):
            raise ConfigurationError("time-grid index must be an integer")
        if not 0 <= step <= self.intervals:
            raise ConfigurationError(f"time-grid index {step} is outside [0, {self.intervals}]")
        return self.start_au + step * self.step_au


@dataclass(frozen=True, slots=True)
class StepSchedule:
    """Integer-step sampling schedule; ``every=0`` disables periodic sampling."""

    every: int = 0
    include_initial: bool = True
    include_final: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.every, bool) or not isinstance(self.every, int):
            raise ConfigurationError("schedule.every must be an integer")
        if self.every < 0:
            raise ConfigurationError("schedule.every cannot be negative")
        if not isinstance(self.include_initial, bool) or not isinstance(self.include_final, bool):
            raise ConfigurationError("schedule endpoint flags must be booleans")

    def steps(self, grid: FixedTimeGrid) -> tuple[int, ...]:
        """Resolve the schedule to sorted, duplicate-free endpoint indices."""

        selected: set[int] = set()
        if self.every:
            selected.update(range(0, grid.intervals + 1, self.every))
        if self.include_initial:
            selected.add(0)
        if self.include_final:
            selected.add(grid.intervals)
        return tuple(sorted(selected))
