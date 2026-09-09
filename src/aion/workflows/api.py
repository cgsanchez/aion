"""Typed top-level workflow boundary introduced incrementally by work package."""

from __future__ import annotations

from dataclasses import replace
from os import PathLike
from typing import Protocol

from aion.config import BackendConfig, ReferenceConfig, SimulationConfig
from aion.electronic_structure import PreparedReference as PreparedReference
from aion.errors import FeatureNotImplementedError

type PathInput = str | PathLike[str]


class Simulation(Protocol):
    @property
    def simulation_id(self) -> str: ...


class Trajectory(Protocol):
    @property
    def run_id(self) -> str: ...


def prepare_reference(config: ReferenceConfig) -> PreparedReference:
    """Run one validated RKS calculation and return an immutable reference."""

    from aion.electronic_structure import prepare_pyscf_reference

    if not isinstance(config, ReferenceConfig):
        raise TypeError("config must be ReferenceConfig")
    return prepare_pyscf_reference(config)


def load_reference(
    path: PathInput,
    *,
    backend: BackendConfig | None = None,
) -> PreparedReference:
    """Load and authenticate a portable reference without rerunning SCF."""

    from aion.electronic_structure import load_reference_data, validate_reference_runtime

    reference = load_reference_data(path)
    validate_reference_runtime(reference)
    if backend is None:
        return reference
    return PreparedReference(
        config=replace(reference.config, backend=backend),
        ground_state=reference.ground_state,
        grid=reference.grid,
        core_operators=reference.core_operators,
        anchor_topology=reference.anchor_topology,
        dependencies=reference.dependencies,
        preparation_backend=reference.preparation_backend,
    )


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
