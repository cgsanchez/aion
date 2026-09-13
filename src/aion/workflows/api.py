"""Typed top-level workflow boundary introduced incrementally by work package."""

from __future__ import annotations

from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from aion.config import (
    BackendConfig,
    OutputConfig,
    ReferenceConfig,
    SimulationConfig,
    dumps_config,
)
from aion.electromagnetism import (
    CompiledUniformSource,
    EventSchedule,
    RuntimeSourceController,
    compile_event_schedule,
    compile_source_for_reference,
)
from aion.electronic_structure import PreparedReference as PreparedReference
from aion.errors import FormulationError
from aion.formulations import (
    AODensity,
    Formulation,
    FormulationSourceSample,
    InstantaneousEvaluation,
    SourceSampling,
    build_formulation,
)
from aion.io.trajectory import Trajectory as Trajectory
from aion.observables import ObservableCalculators, build_observable_calculators
from aion.propagation import (
    OrbitalState,
    PropagationStepResult,
    SCEMPropagator,
    build_initial_orbital_state,
    build_propagator,
)

if TYPE_CHECKING:
    from aion.workflows.runner import RunControl

type PathInput = str | PathLike[str]


class Simulation(Protocol):
    @property
    def simulation_id(self) -> str: ...


@dataclass(slots=True)
class BuiltSimulation:
    """Validated simulation binding with independent mutable run state."""

    config: SimulationConfig
    reference: PreparedReference
    source: CompiledUniformSource
    formulation: Formulation
    state: OrbitalState
    propagator: SCEMPropagator
    calculators: ObservableCalculators
    events: EventSchedule
    runtime_source: RuntimeSourceController
    original_toml: str

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
            velocity_fraction=self.formulation.gauge_velocity_fraction,
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
        """Advance one in-memory accepted step without invoking observers."""

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
    *,
    original_toml: str | None = None,
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
    events = compile_event_schedule(config.events, config.propagation.time_grid)
    runtime_source = RuntimeSourceController(workspace, events)
    formulation = build_formulation(config.formulation, reference, workspace)
    state = build_initial_orbital_state(formulation, workspace)
    propagator = build_propagator(formulation, config.propagation, workspace)
    calculators = build_observable_calculators(
        config.output.schedules,
        config.formulation.kind,
        natom=reference.anchor_topology.natom,
        npair=reference.anchor_topology.pair_indices.shape[0],
    )
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
        runtime_source=runtime_source,
        original_toml=dumps_config(config) if original_toml is None else original_toml,
    )


def run(
    simulation: BuiltSimulation,
    *,
    control: RunControl | None = None,
) -> Trajectory:
    """Execute one validated simulation in the current process."""

    from aion.workflows.runner import execute_simulation

    return execute_simulation(simulation, control=control)


def resume(
    checkpoint: PathInput,
    *,
    output: OutputConfig | PathInput | None = None,
) -> Trajectory:
    """Reconstruct a checkpoint and publish a child trajectory segment."""

    from aion.workflows.runner import resume_simulation

    resolved_output: OutputConfig | str | Path | None
    resolved_output = Path(output) if isinstance(output, PathLike) else output
    return resume_simulation(Path(checkpoint), output=resolved_output)


def load_trajectory(path: PathInput) -> Trajectory:
    """Load and validate a completed trajectory lazily."""

    from aion.io.trajectory import load_trajectory as load

    return load(Path(path))
