"""Typed top-level workflow boundary.

WP1 fixes these signatures without retaining any numerical draft behavior.
Later work packages provide the concrete objects and implementations.
"""

from __future__ import annotations

from os import PathLike
from typing import Protocol

from aion.config import BackendConfig, ReferenceConfig, SimulationConfig
from aion.errors import FeatureNotImplementedError

type PathInput = str | PathLike[str]


class PreparedReference(Protocol):
    @property
    def fingerprint_sha256(self) -> str: ...


class Simulation(Protocol):
    @property
    def simulation_id(self) -> str: ...


class Trajectory(Protocol):
    @property
    def run_id(self) -> str: ...


def prepare_reference(config: ReferenceConfig) -> PreparedReference:
    """Prepare an immutable reference (numerical implementation: WP2)."""

    del config
    raise FeatureNotImplementedError("reference preparation is scheduled for WP2")


def load_reference(
    path: PathInput,
    *,
    backend: BackendConfig | None = None,
) -> PreparedReference:
    """Load and authenticate a reference (numerical implementation: WP2)."""

    del path, backend
    raise FeatureNotImplementedError("reference loading is scheduled for WP2")


def build_simulation(
    config: SimulationConfig,
    reference: PreparedReference,
) -> Simulation:
    """Build a validated independent simulation (implementation: WP3)."""

    del config, reference
    raise FeatureNotImplementedError("simulation construction is scheduled for WP3")


def run(simulation: Simulation) -> Trajectory:
    """Execute one simulation process (implementation: WP5)."""

    del simulation
    raise FeatureNotImplementedError("simulation execution is scheduled for WP5")


def resume(checkpoint: PathInput) -> Trajectory:
    """Resume from an immutable checkpoint (implementation: WP5)."""

    del checkpoint
    raise FeatureNotImplementedError("checkpoint restart is scheduled for WP5")


def load_trajectory(path: PathInput) -> Trajectory:
    """Load a completed trajectory (implementation: WP5)."""

    del path
    raise FeatureNotImplementedError("trajectory loading is scheduled for WP5")
