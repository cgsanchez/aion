"""One-process production runner with observers, checkpoints, and restart."""

from __future__ import annotations

import hashlib
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from aion._version import __version__
from aion.config import (
    OutputConfig,
    canonical_sha256,
)
from aion.electromagnetism import DiscreteKickEvent
from aion.electronic_structure import DependencyVersions, load_reference_data
from aion.errors import (
    CheckpointError,
    RunCancelledError,
    RunnerError,
)
from aion.formulations import CurrentLedger, EnergyLedger, FormulationSourceSample, SourceSampling
from aion.io.checkpoint import CheckpointData, load_checkpoint, save_checkpoint
from aion.io.status import FailureSummary, RunPhase, RunStatus, publish_status
from aion.io.trajectory import Trajectory, TrajectoryWriter
from aion.io.util import file_sha256
from aion.observables import SamplingLocation
from aion.propagation import OrbitalState, orbital_metric_residual

if TYPE_CHECKING:
    from aion.workflows.api import BuiltSimulation


_CURRENT_FIELDS = (
    "electronic_dipole",
    "fixed_nuclear_dipole",
    "total_dipole",
    "dipole_derivative_analytic",
    "primary_current",
    "source_current",
    "mechanical_paramagnetic_current",
    "mechanical_diamagnetic_current",
    "mechanical_total_current",
    "ambient_projected_mechanical_current",
    "site_charges",
    "site_charge_derivatives",
    "pair_currents_continuity",
    "pair_currents_p0_source",
    "p0_graph_current",
    "e1_intrinsic_polarization_current",
    "e1_phase_response_current",
    "e1_residual_source_current",
)

_ENERGY_ENDPOINT_FIELDS = (
    "energy_kinetic_canonical",
    "energy_kinetic_vector_potential_linear",
    "energy_kinetic_diamagnetic",
    "energy_kinetic_mechanical",
    "energy_electron_nuclear",
    "energy_hartree",
    "energy_exchange_correlation",
    "energy_nuclear_repulsion",
    "energy_electromagnetic_electronic_scalar",
    "energy_electromagnetic_fixed_nuclear_scalar",
    "energy_electromagnetic_scalar_total",
    "energy_e1_coupling",
    "energy_matter_total",
    "energy_generator_total",
    "energy_absorbed",
    "source_work_accumulated",
)


@dataclass(slots=True)
class RunControl:
    """Signal-safe cancellation flag and deterministic test hook."""

    cancel_requested: bool = False
    signal_number: int | None = None
    after_accepted_step: Callable[[int], None] | None = None

    def request_cancel(self, signal_number: int | None = None) -> None:
        self.cancel_requested = True
        self.signal_number = signal_number


@dataclass(frozen=True, slots=True)
class RestartContext:
    parent_run_id: str
    parent_checkpoint_sha256: str
    global_step_offset: int
    accumulated_source_work_au: float
    initial_matter_energy_au: float
    applied_event_identifiers: frozenset[str]
    boundary_coefficients: np.ndarray
    boundary_density: np.ndarray


def _git_provenance() -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]

    def command(*args: str) -> bytes:
        try:
            completed = subprocess.run(
                ("git", *args),
                cwd=root,
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return b""
        return completed.stdout

    commit = command("rev-parse", "HEAD").decode().strip()
    status = command("status", "--porcelain=v1")
    patch = command("diff", "--binary", "HEAD")
    return {
        "commit": commit or "unavailable",
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
    }


def collect_run_provenance(simulation: BuiltSimulation) -> dict[str, object]:
    """Collect reproducibility metadata without changing scientific identity."""

    dependency = DependencyVersions.current().as_mapping()
    backend = simulation.workspace.backend
    backend_data: dict[str, object] = {
        "kind": simulation.config.backend.kind.value,
        "precision": simulation.config.backend.precision.value,
        "device_index": simulation.config.backend.device_index,
    }
    if simulation.config.backend.kind.value == "gpu":
        xp = backend.namespace
        properties = xp.cuda.runtime.getDeviceProperties(backend.device_index)
        name = properties.get("name", "unknown")
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        backend_data.update(
            gpu_name=str(name),
            cuda_runtime=int(xp.cuda.runtime.runtimeGetVersion()),
            cuda_driver=int(xp.cuda.runtime.driverGetVersion()),
            cupy=str(xp.__version__),
        )
    lock = Path(__file__).resolve().parents[3] / "conda-linux-64.lock"
    return {
        "aion_version": __version__,
        "git": _git_provenance(),
        "python": sys.version,
        "dependencies": dependency,
        "backend": backend_data,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "environment": {
            "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
            "loaded_modules": os.environ.get("LOADEDMODULES", ""),
            "conda_lock_sha256": file_sha256(lock) if lock.exists() else None,
        },
    }


@dataclass(slots=True)
class _StatusTracker:
    path: Path
    run_id: str
    simulation_id: str
    total_steps: int
    started: float
    latest_checkpoint: str | None = None
    last_publication: float = 0.0

    def publish(
        self,
        phase: RunPhase,
        *,
        step: int,
        force: bool = False,
        failure: FailureSummary | None = None,
    ) -> None:
        now = time.monotonic()
        if not force and now - self.last_publication < 30.0:
            return
        elapsed = now - self.started
        completed = max(0, step)
        remaining = self.total_steps - completed
        eta = None if completed == 0 else elapsed * remaining / completed
        status = RunStatus(
            run_id=self.run_id,
            simulation_id=self.simulation_id,
            phase=phase,
            global_step=step,
            last_accepted_step=step,
            total_steps=self.total_steps,
            latest_checkpoint=self.latest_checkpoint,
            wall_time_seconds=elapsed,
            eta_seconds=eta,
            updated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            host=socket.gethostname(),
            pid=os.getpid(),
            failure=failure,
        )
        publish_status(self.path, status)
        self.last_publication = now


@contextmanager
def _signal_boundary(control: RunControl) -> Iterator[None]:
    previous: dict[signal.Signals, Any] = {}

    def handler(number: int, _frame: Any) -> None:
        control.request_cancel(number)

    try:
        for number in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[number] = signal.getsignal(number)
                signal.signal(number, handler)
            except ValueError:
                previous.clear()
                break
        yield
    finally:
        for number, old_handler in previous.items():
            signal.signal(number, old_handler)


def _host(simulation: BuiltSimulation, value: object) -> np.ndarray:
    return np.asarray(simulation.workspace.backend.to_host(value))


def _scalar(simulation: BuiltSimulation, value: object) -> float:
    array = _host(simulation, value)
    if array.shape != ():
        raise RunnerError("expected a scalar observable at the storage boundary")
    result = float(array)
    if not np.isfinite(result):
        raise RunnerError("observable scalar is non-finite")
    return result


def _observable_value(
    simulation: BuiltSimulation,
    field_name: str,
    value: object,
) -> np.ndarray:
    definition = simulation.calculators.definitions[field_name]
    array = _host(simulation, value)
    if definition.shape == (1,) and array.shape == ():
        array = array.reshape(1)
    if array.shape != definition.shape:
        raise RunnerError(
            f"observable {field_name} has shape {array.shape}; expected {definition.shape}"
        )
    if array.dtype.kind == "c":
        array = np.asarray(array, dtype=np.complex128)
    else:
        array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise RunnerError(f"observable {field_name} is non-finite")
    return array


def _ledger_values(
    simulation: BuiltSimulation,
    ledger: CurrentLedger | EnergyLedger,
    fields: tuple[str, ...],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for field_name in fields:
        value = getattr(ledger, field_name)
        if value is not None and field_name in simulation.calculators.definitions:
            result[field_name] = _observable_value(simulation, field_name, value)
    return result


def _freeze_source_sample(
    simulation: BuiltSimulation,
    sample: FormulationSourceSample,
) -> FormulationSourceSample:
    """Detach an event-side sample from mutable workspace source histories."""

    xp = simulation.workspace.backend.namespace
    return FormulationSourceSample(
        time_au=sample.time_au,
        gauge=sample.gauge,
        electric_field=xp.array(sample.electric_field, copy=True),
        electric_field_dot=xp.array(sample.electric_field_dot, copy=True),
        vector_potential_reduced=xp.array(sample.vector_potential_reduced, copy=True),
        vector_potential_reduced_dot=xp.array(sample.vector_potential_reduced_dot, copy=True),
        node_scalar_potential=xp.array(sample.node_scalar_potential, copy=True),
        pair_link=xp.array(sample.pair_link, copy=True),
        pair_link_dot=xp.array(sample.pair_link_dot, copy=True),
        pair_electromotive_potential=xp.array(sample.pair_electromotive_potential, copy=True),
    )


class _RunExecutor:
    def __init__(
        self,
        simulation: BuiltSimulation,
        *,
        run_id: str,
        reference_artifact_path: Path,
        restart: RestartContext | None,
        control: RunControl,
    ) -> None:
        self.simulation = simulation
        self.run_id = run_id
        self.reference_artifact_path = reference_artifact_path
        self.restart = restart
        self.control = control
        self.accumulated_work = 0.0 if restart is None else restart.accumulated_source_work_au
        self.initial_matter_energy: float | None = (
            None if restart is None else restart.initial_matter_energy_au
        )
        self.written_checkpoints: dict[int, Path] = {}
        self.recorded_source_steps: set[int] = set()
        self.recorded_dipole_steps: set[int] = set()
        self.recorded_energy_steps: set[int] = set()
        self.recorded_diagnostic_steps: set[int] = set()
        self.recorded_snapshot_steps: set[int] = set()
        self.warning_count = 0
        schedules = simulation.config.output.schedules
        grid = simulation.config.propagation.time_grid
        self.dipole_steps = frozenset(schedules.dipole_current.steps(grid))
        self.energy_steps = frozenset(schedules.energy.steps(grid))
        self.diagnostic_steps = frozenset(schedules.diagnostics.steps(grid))
        self.source_steps = frozenset(schedules.source.steps(grid))
        self.snapshot_steps = frozenset(schedules.matrix_snapshots.steps(grid))
        self.checkpoint_steps = frozenset(schedules.checkpoints.steps(grid)) | {
            simulation.state.step_index,
            grid.intervals,
        }
        event_fingerprint = canonical_sha256(
            {
                "schema": "aion.event-schedule",
                "version": "1.0.0",
                "events": [
                    {
                        "event_id": event.event_id,
                        "fingerprint_sha256": event.fingerprint_sha256,
                    }
                    for event in simulation.events.events
                ],
            }
        )
        self.writer = TrajectoryWriter(
            simulation.config.output.directory,
            run_id=run_id,
            config=simulation.config,
            original_toml=simulation.original_toml,
            reference_artifact_path=reference_artifact_path,
            source_fingerprint_sha256=simulation.source.fingerprint_sha256,
            source_definition_json=simulation.source.definition_json,
            event_fingerprint_sha256=event_fingerprint,
            definitions=dict(simulation.calculators.definitions),
            provenance=collect_run_provenance(simulation),
            pair_indices=simulation.reference.anchor_topology.pair_indices,
            pair_displacements_au=simulation.reference.anchor_topology.pair_displacements_au,
            parent_run_id=None if restart is None else restart.parent_run_id,
            parent_checkpoint_sha256=(
                None if restart is None else restart.parent_checkpoint_sha256
            ),
            global_step_offset=0 if restart is None else restart.global_step_offset,
        )
        self.status = _StatusTracker(
            path=simulation.config.output.directory / "status.json",
            run_id=run_id,
            simulation_id=simulation.simulation_id,
            total_steps=grid.intervals,
            started=time.monotonic(),
        )

    def _endpoint_evaluation(self) -> tuple[Any, Any]:
        step = self.simulation.state.step_index
        source = self.simulation.source_sample(SourceSampling.ENDPOINT, step)
        evaluation = self.simulation.formulation.evaluate(self.simulation.state.density(), source)
        return source, evaluation

    def _event_observables(self, evaluation: Any, source: Any) -> dict[str, np.ndarray]:
        current = self.simulation.formulation.currents(evaluation, source)
        energy = self.simulation.formulation.energy(
            evaluation,
            source,
            initial_matter_energy=self.initial_matter_energy,
            accumulated_source_work=self.accumulated_work,
        )
        values = _ledger_values(self.simulation, current, _CURRENT_FIELDS)
        values.update(_ledger_values(self.simulation, energy, _ENERGY_ENDPOINT_FIELDS))
        return values

    def _initialize_baseline(self) -> None:
        if self.initial_matter_energy is not None:
            return
        source, evaluation = self._endpoint_evaluation()
        energy = self.simulation.formulation.energy(evaluation, source)
        self.initial_matter_energy = _scalar(self.simulation, energy.energy_matter_total)

    def _apply_event(self, event: DiscreteKickEvent, sequence: int) -> None:
        source_before, evaluation_before = self._endpoint_evaluation()
        source_before = _freeze_source_sample(self.simulation, source_before)
        coefficients_before = _host(self.simulation, self.simulation.state.coefficients)
        density_before = _host(self.simulation, self.simulation.state.density().matrix)
        offset_before = np.array(
            self.simulation.runtime_source.vector_potential_offset_au,
            copy=True,
        )
        observables_before = self._event_observables(evaluation_before, source_before)
        matter_before = _scalar(
            self.simulation,
            self.simulation.formulation.energy(
                evaluation_before,
                source_before,
            ).energy_matter_total,
        )
        self.simulation.runtime_source.apply(event)
        try:
            source_after = self.simulation.source_sample(SourceSampling.ENDPOINT, event.step)
            coefficients_after_backend = self.simulation.formulation.apply_kick(
                self.simulation.state.coefficients,
                event.impulse_au,
                evaluation_before,
                source_before,
                source_after,
            )
            state_after = OrbitalState(
                coefficients_after_backend,
                self.simulation.state.occupations,
                self.simulation.workspace.backend,
                event.step,
            )
            evaluation_after = self.simulation.formulation.evaluate(
                state_after.density(), source_after
            )
            metric_residual = orbital_metric_residual(
                state_after.coefficients,
                evaluation_after.triple.metric,
                self.simulation.workspace.backend,
            )
            if metric_residual > 2.0e-11:
                raise RunnerError(f"post-event orbital metric residual is {metric_residual:.3e}")
            matter_after = _scalar(
                self.simulation,
                self.simulation.formulation.energy(
                    evaluation_after,
                    source_after,
                ).energy_matter_total,
            )
        except BaseException:
            self.simulation.runtime_source.undo(event)
            raise
        self.simulation.state = state_after
        work_increment = matter_after - matter_before
        self.accumulated_work += work_increment
        observables_after = self._event_observables(evaluation_after, source_after)
        self.writer.append_event(
            sequence=sequence,
            event_id=event.event_id,
            event_fingerprint_sha256=event.fingerprint_sha256,
            step=event.step,
            time_au=event.time_au,
            impulse_au=np.asarray(event.impulse_au, dtype=np.float64),
            offset_before_au=offset_before,
            offset_after_au=np.array(
                self.simulation.runtime_source.vector_potential_offset_au,
                copy=True,
            ),
            work_increment_au=work_increment,
            coefficients_before=coefficients_before,
            coefficients_after=_host(self.simulation, state_after.coefficients),
            density_before=density_before,
            density_after=_host(self.simulation, state_after.density().matrix),
            observables_before=observables_before,
            observables_after=observables_after,
        )

    def _process_events(self) -> None:
        step = self.simulation.state.step_index
        pending = self.simulation.events.pending_at(
            step,
            frozenset(self.simulation.runtime_source.applied_identifiers),
        )
        sequence_by_id = {
            event.idempotency_identifier: index
            for index, event in enumerate(self.simulation.events.events)
        }
        for event in pending:
            self._apply_event(event, sequence_by_id[event.idempotency_identifier])

    def _record_boundary(self, *, force_observables: bool = False) -> None:
        step = self.simulation.state.step_index
        grid = self.simulation.config.propagation.time_grid
        time_au = grid.time_at(step)
        source, evaluation = self._endpoint_evaluation()
        if (force_observables or step in self.source_steps) and (
            step not in self.recorded_source_steps
        ):
            self.writer.append_source(
                SamplingLocation.ENDPOINT,
                step=step,
                sample=source,
                host=lambda value: _host(self.simulation, value),
            )
            self.recorded_source_steps.add(step)
        if (force_observables or step in self.dipole_steps) and (
            step not in self.recorded_dipole_steps
        ):
            current = self.simulation.calculators.dipole_current.calculate(
                self.simulation.formulation,
                evaluation,
                source,
            )
            for field_name, value in _ledger_values(
                self.simulation, current, _CURRENT_FIELDS
            ).items():
                self.writer.append_observable(
                    field_name,
                    step=step,
                    time_au=time_au,
                    value=value,
                )
            self.recorded_dipole_steps.add(step)
        if (force_observables or step in self.energy_steps) and (
            step not in self.recorded_energy_steps
        ):
            energy = self.simulation.calculators.energy.calculate(
                self.simulation.formulation,
                evaluation,
                source,
                initial_matter_energy=self.initial_matter_energy,
                accumulated_source_work=self.accumulated_work,
            )
            for field_name, value in _ledger_values(
                self.simulation, energy, _ENERGY_ENDPOINT_FIELDS
            ).items():
                self.writer.append_observable(
                    field_name,
                    step=step,
                    time_au=time_au,
                    value=value,
                )
            absorbed = _scalar(self.simulation, energy.energy_absorbed)
            raw = absorbed - self.accumulated_work
            policy = self.simulation.config.validation
            scale = max(abs(absorbed), abs(self.accumulated_work))
            threshold = policy.absolute_tolerance + policy.relative_tolerance * scale
            normalized = raw / threshold
            self.writer.append_work_energy_residual(
                step=step,
                time_au=time_au,
                raw_au=raw,
                normalized=normalized,
            )
            if abs(raw) > threshold:
                message = (
                    f"work-energy residual {raw:.6e} Ha exceeds combined threshold "
                    f"{threshold:.6e} Ha"
                )
                self.writer.append_warning(
                    step=step,
                    category="work_energy",
                    message=message,
                )
                self.warning_count += 1
                if policy.strict_validation:
                    raise RunnerError(message)
            self.recorded_energy_steps.add(step)
        if (force_observables or step in self.diagnostic_steps) and (
            step not in self.recorded_diagnostic_steps
        ):
            diagnostics = self.simulation.calculators.diagnostics.calculate(
                self.simulation.formulation,
                evaluation,
                source,
            )
            self.writer.append_instantaneous_diagnostics(
                step=step,
                time_au=time_au,
                diagnostics=diagnostics,
                host=lambda value: _host(self.simulation, value),
            )
            self.recorded_diagnostic_steps.add(step)
        if step in self.snapshot_steps and step not in self.recorded_snapshot_steps:
            self.writer.append_matrix_snapshot(
                step=step,
                time_au=time_au,
                density=_host(self.simulation, self.simulation.state.density().matrix),
            )
            self.recorded_snapshot_steps.add(step)

    def _checkpoint(self, *, force: bool = False) -> Path | None:
        step = self.simulation.state.step_index
        if not force and step not in self.checkpoint_steps:
            return None
        if step in self.written_checkpoints:
            return self.written_checkpoints[step]
        self.status.publish(RunPhase.CHECKPOINTING, step=step, force=True)
        assert self.initial_matter_energy is not None
        checkpoint = CheckpointData(
            run_id=self.run_id,
            simulation_config=self.simulation.config,
            original_toml=self.simulation.original_toml,
            reference_artifact_path=self.reference_artifact_path,
            source_fingerprint_sha256=self.simulation.source.fingerprint_sha256,
            global_step=step,
            coefficients=_host(self.simulation, self.simulation.state.coefficients),
            occupations=_host(self.simulation, self.simulation.state.occupations),
            density=_host(self.simulation, self.simulation.state.density().matrix),
            accumulated_source_work_au=self.accumulated_work,
            initial_matter_energy_au=self.initial_matter_energy,
            applied_event_identifiers=frozenset(self.simulation.runtime_source.applied_identifiers),
            vector_potential_offset_au=self.simulation.runtime_source.vector_potential_offset_au,
            parent_run_id=None if self.restart is None else self.restart.parent_run_id,
            parent_checkpoint_sha256=(
                None if self.restart is None else self.restart.parent_checkpoint_sha256
            ),
            global_step_offset=0 if self.restart is None else self.restart.global_step_offset,
        )
        path = self.simulation.config.output.directory / f"checkpoint_{step:08d}.h5"
        sha256 = save_checkpoint(checkpoint, path)
        self.writer.append_checkpoint_link(
            step=step,
            time_au=self.simulation.config.propagation.time_grid.time_at(step),
            path=path,
            sha256=sha256,
        )
        self.written_checkpoints[step] = path
        self.status.latest_checkpoint = path.name
        self.status.publish(RunPhase.PROPAGATING, step=step, force=True)
        return path

    def _record_restart_boundary(self) -> None:
        if self.restart is None:
            return
        step = self.simulation.state.step_index
        coefficients = _host(self.simulation, self.simulation.state.coefficients)
        density = _host(self.simulation, self.simulation.state.density().matrix)
        if not np.array_equal(coefficients, self.restart.boundary_coefficients):
            raise CheckpointError("reconstructed restart coefficients differ from checkpoint")
        if not np.array_equal(density, self.restart.boundary_density):
            raise CheckpointError("reconstructed restart density differs from checkpoint")
        self.writer.write_restart_boundary(
            step=step,
            time_au=self.simulation.config.propagation.time_grid.time_at(step),
            coefficients=coefficients,
            density=density,
            accumulated_source_work_au=self.accumulated_work,
        )

    def _cancel(self) -> None:
        step = self.simulation.state.step_index
        self._record_boundary(force_observables=True)
        self._checkpoint(force=True)
        self.writer.set_summary(
            final_step=step,
            accumulated_source_work_au=self.accumulated_work,
        )
        number = self.control.signal_number
        message = "cancellation requested" if number is None else f"received signal {number}"
        self.writer.fail(
            error_type="RunCancelledError",
            message=message,
            phase=RunPhase.CANCELLED.value,
            step=step,
        )
        self.status.publish(RunPhase.CANCELLED, step=step, force=True)
        raise RunCancelledError(message)

    def execute(self) -> Trajectory:
        step = self.simulation.state.step_index
        self.status.publish(RunPhase.INITIALIZING, step=step, force=True)
        self._initialize_baseline()
        self._record_restart_boundary()
        self.status.publish(RunPhase.PROPAGATING, step=step, force=True)
        if self.control.cancel_requested:
            self._cancel()
        self._process_events()
        if self.control.cancel_requested:
            self._cancel()
        self._record_boundary(force_observables=True)
        self._checkpoint()
        if self.control.cancel_requested:
            self._cancel()
        grid = self.simulation.config.propagation.time_grid
        while self.simulation.state.step_index < grid.intervals:
            if self.control.cancel_requested:
                self._cancel()
            result = self.simulation.step()
            interval = result.start_step
            midpoint_power = self.simulation.formulation.power(
                result.midpoint_evaluation,
                result.midpoint_source,
            )
            rate = _scalar(self.simulation, midpoint_power.source_work_rate)
            increment = grid.step_au * rate
            self.accumulated_work += increment
            self.writer.append_work_interval(
                step=interval,
                time_au=result.midpoint_source.time_au,
                rate_au=rate,
                increment_au=increment,
                accumulated_au=self.accumulated_work,
            )
            self.writer.append_source(
                SamplingLocation.MIDPOINT,
                step=interval,
                sample=result.midpoint_source,
                host=lambda value: _host(self.simulation, value),
            )
            self.writer.append_midpoint_physics(
                step=interval,
                time_au=result.midpoint_source.time_au,
                matter_rate_au=_scalar(self.simulation, midpoint_power.energy_matter_rate_analytic),
                generator_rate_au=(
                    None
                    if midpoint_power.energy_generator_rate_analytic is None
                    else _scalar(
                        self.simulation,
                        midpoint_power.energy_generator_rate_analytic,
                    )
                ),
                source_rate_au=rate,
                ward_residual_au=_scalar(self.simulation, midpoint_power.energy_ward_residual),
            )
            if result.state.step_index in self.diagnostic_steps:
                self.writer.append_scem_diagnostics(
                    step=interval,
                    time_au=result.midpoint_source.time_au,
                    diagnostics=result.diagnostics,
                )
            if self.control.after_accepted_step is not None:
                self.control.after_accepted_step(result.state.step_index)
            if self.control.cancel_requested:
                self._cancel()
            self._process_events()
            if self.control.cancel_requested:
                self._cancel()
            self._record_boundary()
            self._checkpoint()
            if self.control.cancel_requested:
                self._cancel()
            self.status.publish(
                RunPhase.PROPAGATING,
                step=self.simulation.state.step_index,
            )
            self.writer.flush()
        self.writer.set_summary(
            final_step=self.simulation.state.step_index,
            accumulated_source_work_au=self.accumulated_work,
        )
        trajectory = self.writer.finalize()
        self.status.publish(
            RunPhase.COMPLETED,
            step=self.simulation.state.step_index,
            force=True,
        )
        return trajectory


def _validate_reference_artifact(simulation: BuiltSimulation) -> Path:
    path = simulation.config.reference.path.expanduser().resolve()
    try:
        disk_reference = load_reference_data(path)
    except Exception as exc:
        raise RunnerError(
            f"a production run requires an authenticated reference artifact at {path}"
        ) from exc
    if disk_reference.fingerprint_sha256 != simulation.reference.fingerprint_sha256:
        raise RunnerError("in-memory and on-disk references have different fingerprints")
    return path


def execute_simulation(
    simulation: BuiltSimulation,
    *,
    control: RunControl | None = None,
    restart: RestartContext | None = None,
    reference_artifact_path: Path | None = None,
) -> Trajectory:
    """Execute one simulation in the current process and publish one segment."""

    from aion.workflows.api import BuiltSimulation

    if not isinstance(simulation, BuiltSimulation):
        raise TypeError("run requires a BuiltSimulation")
    controller = RunControl() if control is None else control
    reference_path = (
        _validate_reference_artifact(simulation)
        if reference_artifact_path is None
        else reference_artifact_path.resolve()
    )
    run_id = uuid.uuid4().hex
    executor = _RunExecutor(
        simulation,
        run_id=run_id,
        reference_artifact_path=reference_path,
        restart=restart,
        control=controller,
    )
    with _signal_boundary(controller):
        try:
            return executor.execute()
        except RunCancelledError:
            raise
        except BaseException as exc:
            step = simulation.state.step_index
            with suppress(BaseException):
                executor._checkpoint(force=True)
            with suppress(BaseException):
                executor.writer.set_summary(
                    final_step=step,
                    accumulated_source_work_au=executor.accumulated_work,
                )
                executor.writer.fail(
                    error_type=type(exc).__name__,
                    message=str(exc),
                    phase=RunPhase.PROPAGATING.value,
                    step=step,
                )
            with suppress(BaseException):
                failure = FailureSummary(
                    error_type=type(exc).__name__,
                    message=str(exc) or type(exc).__name__,
                    phase=RunPhase.PROPAGATING,
                    step=step,
                )
                executor.status.publish(
                    RunPhase.FAILED,
                    step=step,
                    force=True,
                    failure=failure,
                )
            raise


def resume_simulation(
    checkpoint_path: str | Path,
    *,
    output: OutputConfig | str | Path | None = None,
) -> Trajectory:
    """Reconstruct a checkpoint on its configured backend and execute a child segment."""

    from aion.workflows.api import build_simulation

    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = load_checkpoint(path)
    reference = load_reference_data(checkpoint.reference_artifact_path)
    config = checkpoint.simulation_config
    if output is None:
        directory = path.parent / (f"resume_{checkpoint.global_step:08d}_{uuid.uuid4().hex[:8]}")
        output_config = replace(config.output, directory=directory)
    elif isinstance(output, OutputConfig):
        output_config = output
    else:
        output_config = replace(config.output, directory=Path(output))
    config = replace(config, output=output_config)
    simulation = build_simulation(
        config,
        reference,
        original_toml=checkpoint.original_toml,
    )
    if simulation.source.fingerprint_sha256 != checkpoint.source_fingerprint_sha256:
        raise CheckpointError("reconstructed source does not match checkpoint fingerprint")
    simulation.runtime_source.restore(checkpoint.applied_event_identifiers)
    if not np.array_equal(
        simulation.runtime_source.vector_potential_offset_au,
        checkpoint.vector_potential_offset_au,
    ):
        raise CheckpointError("reconstructed runtime source offset does not match checkpoint")
    backend = simulation.workspace.backend
    xp = backend.namespace
    simulation.state = OrbitalState(
        backend.asarray(checkpoint.coefficients, dtype=xp.complex128),
        backend.asarray(checkpoint.occupations, dtype=xp.float64),
        backend,
        checkpoint.global_step,
    )
    source = simulation.source_sample(SourceSampling.ENDPOINT, checkpoint.global_step)
    evaluation = simulation.formulation.evaluate(simulation.state.density(), source)
    residual = orbital_metric_residual(
        simulation.state.coefficients,
        evaluation.triple.metric,
        backend,
    )
    if residual > 2.0e-11:
        raise CheckpointError(
            f"reconstructed checkpoint state violates its metric: residual={residual:.3e}"
        )
    restart = RestartContext(
        parent_run_id=checkpoint.run_id,
        parent_checkpoint_sha256=file_sha256(path),
        global_step_offset=checkpoint.global_step,
        accumulated_source_work_au=checkpoint.accumulated_source_work_au,
        initial_matter_energy_au=checkpoint.initial_matter_energy_au,
        applied_event_identifiers=checkpoint.applied_event_identifiers,
        boundary_coefficients=checkpoint.coefficients,
        boundary_density=checkpoint.density,
    )
    return execute_simulation(
        simulation,
        restart=restart,
        reference_artifact_path=checkpoint.reference_artifact_path,
    )
