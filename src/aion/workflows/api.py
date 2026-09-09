"""Typed top-level workflow boundary introduced incrementally by work package."""

from __future__ import annotations

from dataclasses import dataclass, replace
from os import PathLike
from typing import Any, Protocol

from aion.config import BackendConfig, ReferenceConfig, SimulationConfig
from aion.electromagnetism import (
    CompiledUniformSource,
    EventSchedule,
    compile_event_schedule,
    compile_source_for_reference,
)
from aion.electronic_structure import PreparedReference as PreparedReference
from aion.errors import FeatureNotImplementedError, FormulationError
from aion.formulations import (
    AODensity,
    Formulation,
    FormulationSourceSample,
    InstantaneousEvaluation,
    SourceSampling,
    build_formulation,
)
from aion.observables import ObservableCalculators, build_observable_calculators
from aion.propagation import (
    OrbitalState,
    PropagationStepResult,
    SCEMPropagator,
    build_initial_orbital_state,
    build_propagator,
)

type PathInput = str | PathLike[str]


class Simulation(Protocol):
    @property
    def simulation_id(self) -> str: ...


class Trajectory(Protocol):
    @property
    def run_id(self) -> str: ...


@dataclass(slots=True)
class BuiltSimulation:
    """Validated simulation binding with a reusable in-memory WP4 stepper."""

    config: SimulationConfig
    reference: PreparedReference
    source: CompiledUniformSource
    formulation: Formulation
    state: OrbitalState
    propagator: SCEMPropagator
    calculators: ObservableCalculators
    events: EventSchedule

    @property
    def workspace(self) -> Any:
        return self.formulation.context.workspace

    @property
    def simulation_id(self) -> str:
        return self.config.scientific_id

    @property
    def density(self) -> AODensity:
        return self.state.density()

    def source_sample(self, location: SourceSampling, index: int) -> FormulationSourceSample:
        return FormulationSourceSample.from_workspace(
            self.workspace,
            gauge=self.formulation.gauge,
            location=location,
            index=index,
        )

    def evaluate(
        self, location: SourceSampling, index: int, density: AODensity | None = None
    ) -> InstantaneousEvaluation:
        state = self.density if density is None else density
        source = self.source_sample(location, index)
        return self.formulation.evaluate(state, source)

    def step(self, state: OrbitalState | None = None) -> PropagationStepResult:
        """Advance one in-memory step without starting the WP5 runner."""

        current = self.state if state is None else state
        result = self.propagator.step(current)
        if state is None:
            self.state = result.state
        return result


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
) -> BuiltSimulation:
    """Build an independent source/formulation/state/workspace binding."""

    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be SimulationConfig")
    if not isinstance(reference, PreparedReference):
        raise TypeError("reference must be PreparedReference")
    if config.reference.fingerprint_sha256 != reference.fingerprint_sha256:
        raise FormulationError("simulation reference fingerprint does not match the reference")
    workspace = reference.create_workspace(config.backend)
    source = compile_source_for_reference(config.source, config.propagation.time_grid, reference)
    source.install(workspace)
    formulation = build_formulation(config.formulation, reference, workspace)
    state = build_initial_orbital_state(formulation, workspace)
    propagator = build_propagator(formulation, config.propagation, workspace)
    calculators = build_observable_calculators(
        config.output.schedules,
        config.formulation.kind,
        natom=reference.anchor_topology.natom,
        npair=reference.anchor_topology.pair_indices.shape[0],
    )
    events = compile_event_schedule(config.events, config.propagation.time_grid)
    workspace.assert_all_resident()
    return BuiltSimulation(
        config=config,
        reference=reference,
        source=source,
        formulation=formulation,
        state=state,
        propagator=propagator,
        calculators=calculators,
        events=events,
    )


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
