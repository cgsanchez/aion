"""Streaming execution, cancellation, checkpointing, and restart for Wilson runs."""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from aion._version import __version__
from aion.config import OutputConfig
from aion.electronic_structure import (
    DependencyVersions,
    evaluate_exact_wilson_power,
    load_reference_data,
    load_wilson_stationary_state,
)
from aion.errors import CheckpointError, RunCancelledError, RunnerError
from aion.io.status import FailureSummary, RunPhase
from aion.io.util import file_sha256
from aion.io.wilson_checkpoint import (
    WilsonCheckpointData,
    load_wilson_checkpoint,
    save_wilson_checkpoint,
)
from aion.io.wilson_trajectory import WilsonTrajectory, WilsonTrajectoryWriter
from aion.workflows.runner import RunControl, _signal_boundary, _StatusTracker
from aion.workflows.wilson import BuiltWilsonSimulation


@dataclass(frozen=True, slots=True)
class WilsonRestartContext:
    parent_run_id: str
    parent_checkpoint_sha256: str
    global_step_offset: int
    accumulated_source_work_au: float
    initial_molecular_energy_au: float
    observer_schedule_state: tuple[tuple[str, int], ...]
    boundary_density: np.ndarray


def _host(simulation: BuiltWilsonSimulation, value: object) -> np.ndarray:
    if np.isscalar(value):
        return np.asarray(value)
    return np.asarray(simulation.quadrature.backend.to_host(value))


def _scalar(simulation: BuiltWilsonSimulation, value: object) -> float:
    array = _host(simulation, value)
    if array.shape != ():
        raise RunnerError("expected a scalar Wilson observable")
    result = float(array)
    if not np.isfinite(result):
        raise RunnerError("Wilson observable scalar is non-finite")
    return result


def _provenance(simulation: BuiltWilsonSimulation) -> str:
    backend = simulation.quadrature.backend
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
    return json.dumps(
        {
            "aion_version": __version__,
            "python": sys.version,
            "dependencies": DependencyVersions.current().as_mapping(),
            "backend": backend_data,
            "host": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class _WilsonRunExecutor:
    def __init__(
        self,
        simulation: BuiltWilsonSimulation,
        *,
        run_id: str,
        reference_artifact_path: Path,
        stationary_state_artifact_path: Path,
        restart: WilsonRestartContext | None,
        control: RunControl,
    ) -> None:
        self.simulation = simulation
        self.run_id = run_id
        self.reference_artifact_path = reference_artifact_path
        self.stationary_state_artifact_path = stationary_state_artifact_path
        self.restart = restart
        self.control = control
        self.accumulated_work = 0.0 if restart is None else restart.accumulated_source_work_au
        self.initial_energy = (
            simulation.stationary_state.energies.molecular_total_au
            if restart is None
            else restart.initial_molecular_energy_au
        )
        self.last_checkpoint_step: int | None = None
        self.last_checkpoint_path: Path | None = None
        observer_names = (
            "source",
            "dipole_current",
            "energy",
            "diagnostics",
            "matrix_snapshots",
        )
        inherited = () if restart is None else restart.observer_schedule_state
        inherited_state = dict(inherited)
        self.last_recorded = {name: inherited_state.get(name, -1) for name in observer_names}
        schedules = simulation.config.output.schedules
        grid = simulation.config.propagation.time_grid
        self.schedule = {
            "source": schedules.source,
            "dipole_current": schedules.dipole_current,
            "energy": schedules.energy,
            "diagnostics": schedules.diagnostics,
            "matrix_snapshots": schedules.matrix_snapshots,
        }
        self.checkpoint_schedule = schedules.checkpoints
        self.writer = WilsonTrajectoryWriter(
            simulation.config.output.directory,
            run_id=run_id,
            config=simulation.config,
            original_toml=simulation.original_toml,
            reference_artifact_path=reference_artifact_path,
            stationary_state_artifact_path=stationary_state_artifact_path,
            source_fingerprint_sha256=simulation.source_provider.fingerprint_sha256,
            provenance_json=_provenance(simulation),
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

    def _scheduled(self, name: str, step: int) -> bool:
        schedule = self.schedule[name]
        final = self.simulation.config.propagation.time_grid.intervals
        return bool(
            (schedule.every and step % schedule.every == 0)
            or (step == 0 and schedule.include_initial)
            or (step == final and schedule.include_final)
        )

    def _append(self, name: str, value: object, *, unit: str, dimension: str) -> None:
        self.writer.append_series(
            name,
            step=self.simulation.boundary_index,
            time_au=self.simulation.current_time_au,
            value=_host(self.simulation, value),
            unit=unit,
            physical_dimension=dimension,
        )

    def _record_boundary(self, *, force: bool = False) -> None:
        step = self.simulation.boundary_index
        time_au = self.simulation.current_time_au
        if (force or self._scheduled("source", step)) and step != self.last_recorded["source"]:
            self.writer.append_source(
                step=step,
                sample=self.simulation.dynamic_cache.sample(time_au).source,
            )
            self.last_recorded["source"] = step

        need_current = force or self._scheduled("dipole_current", step)
        need_energy = force or self._scheduled("energy", step)
        need_charge = force or self._scheduled("diagnostics", step)
        observation = None
        if need_current or need_energy or need_charge:
            observation = self.simulation.observe_endpoint(
                include_energy=need_energy,
                include_identities=need_charge,
            )
        if (
            need_current
            and step != self.last_recorded["dipole_current"]
            and observation is not None
        ):
            for name, value in (
                ("dipole/electronic", observation.electronic_dipole_au),
                ("dipole/fixed_nuclear", observation.fixed_nuclear_dipole_au),
                ("dipole/molecular_total", observation.molecular_total_dipole_au),
            ):
                self._append(name, value, unit="electron_bohr", dimension="electric_dipole")
            self._append(
                "current/uniform_source",
                observation.uniform_source_current_au,
                unit="electron_per_atomic_unit_of_time_times_bohr",
                dimension="electric_current",
            )
            for name, value in (
                ("power/source", observation.source_power_au),
                ("power/matrix_mechanical", observation.matrix_mechanical_energy_rate_au),
                ("power/identity_residual", observation.power_identity_residual_au),
            ):
                self._append(
                    name,
                    value,
                    unit="hartree_per_atomic_unit_of_time",
                    dimension="power",
                )
            self.last_recorded["dipole_current"] = step
        if need_charge and step != self.last_recorded["diagnostics"] and observation is not None:
            for name, value, unit, dimension in (
                (
                    "charge/metric_particle_number",
                    observation.metric_particle_number,
                    "electron",
                    "particle_number",
                ),
                (
                    "charge/integrated_grid",
                    observation.integrated_charge_grid,
                    "elementary_charge",
                    "electric_charge",
                ),
                (
                    "charge/integrated_metric",
                    observation.integrated_charge_metric,
                    "elementary_charge",
                    "electric_charge",
                ),
                (
                    "charge/grid_metric_residual",
                    observation.charge_grid_metric_residual,
                    "elementary_charge",
                    "electric_charge_residual",
                ),
            ):
                self._append(name, value, unit=unit, dimension=dimension)
            if observation.identities is None:
                raise RunnerError("scheduled Wilson identity diagnostics were not evaluated")
            identities = observation.identities
            for name, value in (
                ("identity/ward_residual_abs", identities.ward_residual_abs),
                (
                    "identity/density_shell_residual_relative_norm",
                    identities.density_shell_residual_relative_norm,
                ),
                (
                    "identity/finite_region_continuity_residual_abs",
                    identities.finite_region_continuity_residual_abs,
                ),
                (
                    "identity/global_charge_residual_abs",
                    identities.global_charge_residual_abs,
                ),
            ):
                self._append(name, value, unit="1", dimension="identity_residual")
            self.last_recorded["diagnostics"] = step
        if need_energy and step != self.last_recorded["energy"] and observation is not None:
            if observation.energy is None:
                raise RunnerError("scheduled Wilson energy was not evaluated")
            for name in (
                "kinetic_au",
                "electron_nuclear_au",
                "one_electron_au",
                "hartree_au",
                "exchange_correlation_au",
                "nuclear_repulsion_au",
                "electronic_au",
                "molecular_total_au",
            ):
                self._append(
                    "energy/" + name.removesuffix("_au"),
                    getattr(observation.energy, name),
                    unit="hartree",
                    dimension="energy",
                )
            molecular = _scalar(self.simulation, observation.energy.molecular_total_au)
            absorbed = molecular - self.initial_energy
            residual = absorbed - self.accumulated_work
            self._append(
                "energy/absorbed",
                absorbed,
                unit="hartree",
                dimension="energy",
            )
            self._append(
                "work/accumulated_source",
                self.accumulated_work,
                unit="hartree",
                dimension="energy",
            )
            self._append(
                "work/energy_residual",
                residual,
                unit="hartree",
                dimension="energy_residual",
            )
            policy = self.simulation.config.validation
            threshold = policy.absolute_tolerance + policy.relative_tolerance * max(
                abs(absorbed),
                abs(self.accumulated_work),
            )
            if policy.strict_validation and abs(residual) > threshold:
                raise RunnerError(
                    f"Wilson work-energy residual {residual:.6e} Ha exceeds {threshold:.6e} Ha"
                )
            self.last_recorded["energy"] = step
        if (
            self._scheduled("matrix_snapshots", step)
            and step != self.last_recorded["matrix_snapshots"]
        ):
            self.writer.append_snapshot(
                step=step,
                time_au=time_au,
                density=_host(self.simulation, self.simulation.density),
            )
            self.last_recorded["matrix_snapshots"] = step

    def _observer_state(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self.last_recorded.items()))

    def _checkpoint(self, *, force: bool = False) -> Path | None:
        step = self.simulation.boundary_index
        final = self.simulation.config.propagation.time_grid.intervals
        scheduled = bool(
            (self.checkpoint_schedule.every and step % self.checkpoint_schedule.every == 0)
            or (step == 0 and self.checkpoint_schedule.include_initial)
            or (step == final and self.checkpoint_schedule.include_final)
            or step == self.simulation.propagator.initial_boundary_index
        )
        if not force and not scheduled:
            return None
        if step == self.last_checkpoint_step:
            return self.last_checkpoint_path
        self.status.publish(RunPhase.CHECKPOINTING, step=step, force=True)
        checkpoint = WilsonCheckpointData(
            run_id=self.run_id,
            simulation_config=self.simulation.config,
            original_toml=self.simulation.original_toml,
            reference_artifact_path=self.reference_artifact_path,
            stationary_state_artifact_path=self.stationary_state_artifact_path,
            source_fingerprint_sha256=self.simulation.source_provider.fingerprint_sha256,
            global_step=step,
            contravariant_density=_host(self.simulation, self.simulation.density),
            accumulated_source_work_au=self.accumulated_work,
            initial_molecular_energy_au=self.initial_energy,
            observer_schedule_state=self._observer_state(),
            parent_run_id=None if self.restart is None else self.restart.parent_run_id,
            parent_checkpoint_sha256=(
                None if self.restart is None else self.restart.parent_checkpoint_sha256
            ),
            global_step_offset=0 if self.restart is None else self.restart.global_step_offset,
        )
        path = self.simulation.config.output.directory / f"checkpoint_{step:08d}.h5"
        sha256 = save_wilson_checkpoint(checkpoint, path)
        self.writer.append_checkpoint_link(
            step=step,
            time_au=self.simulation.current_time_au,
            path=path,
            sha256=sha256,
        )
        self.last_checkpoint_step = step
        self.last_checkpoint_path = path
        self.status.latest_checkpoint = path.name
        self.status.publish(RunPhase.PROPAGATING, step=step, force=True)
        return path

    def _authenticate_restart(self) -> None:
        if self.restart is None:
            return
        density = _host(self.simulation, self.simulation.density)
        if not np.array_equal(density, self.restart.boundary_density):
            raise CheckpointError("reconstructed Wilson density differs from checkpoint")
        self.writer.authenticate_restart_boundary(
            step=self.simulation.boundary_index,
            density=density,
            accumulated_source_work_au=self.accumulated_work,
        )

    def _cancel(self) -> None:
        step = self.simulation.boundary_index
        self._record_boundary(force=True)
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

    def execute(self) -> WilsonTrajectory:
        step = self.simulation.boundary_index
        self.status.publish(RunPhase.INITIALIZING, step=step, force=True)
        self._authenticate_restart()
        self.status.publish(RunPhase.PROPAGATING, step=step, force=True)
        if self.control.cancel_requested:
            self._cancel()
        self._record_boundary(force=True)
        self._checkpoint()
        grid = self.simulation.config.propagation.time_grid
        while self.simulation.boundary_index < grid.intervals:
            if self.control.cancel_requested:
                self._cancel()
            result = self.simulation.step()
            minus = evaluate_exact_wilson_power(
                result.gauss_minus_evaluation,
                result.gauss_minus_contravariant_density,
            )
            plus = evaluate_exact_wilson_power(
                result.gauss_plus_evaluation,
                result.gauss_plus_contravariant_density,
            )
            powers = (
                _scalar(self.simulation, minus.source_power_au),
                _scalar(self.simulation, plus.source_power_au),
            )
            increment = 0.5 * grid.step_au * (powers[0] + powers[1])
            self.accumulated_work += increment
            self.writer.append_interval_work(
                step=result.start_boundary_index,
                gauss_times_au=(result.gauss_minus_time_au, result.gauss_plus_time_au),
                gauss_power_au=powers,
                increment_au=increment,
                accumulated_au=self.accumulated_work,
            )
            if self._scheduled("diagnostics", result.end_boundary_index):
                diagnostics = result.diagnostics
                cache = self.simulation.dynamic_cache.statistics
                self.writer.append_diagnostics(
                    step=result.end_boundary_index,
                    time_au=result.end_time_au,
                    values={
                        "nonlinear_iterations": float(diagnostics.nonlinear_iterations),
                        "nonlinear_residual": diagnostics.nonlinear_residual,
                        "cross_metric_residual": diagnostics.cross_metric_residual,
                        "trace_drift": diagnostics.trace_drift,
                        "trace_imaginary_abs": diagnostics.trace_imaginary_abs,
                        "occupation_spectrum_drift": diagnostics.occupation_spectrum_drift,
                        "metric_hermiticity_residual": (diagnostics.metric_hermiticity_residual),
                        "contravariant_hermiticity_residual": (
                            diagnostics.contravariant_hermiticity_residual
                        ),
                        "cache_spatial_entries": float(cache.spatial_entries),
                        "cache_sample_entries": float(cache.sample_entries),
                    },
                )
            if self.control.after_accepted_step is not None:
                self.control.after_accepted_step(result.end_boundary_index)
            if self.control.cancel_requested:
                self._cancel()
            self._record_boundary()
            self._checkpoint()
            self.status.publish(
                RunPhase.PROPAGATING,
                step=self.simulation.boundary_index,
            )
            self.writer.flush()
        self.writer.set_summary(
            final_step=self.simulation.boundary_index,
            accumulated_source_work_au=self.accumulated_work,
        )
        trajectory = self.writer.finalize()
        self.status.publish(
            RunPhase.COMPLETED,
            step=self.simulation.boundary_index,
            force=True,
        )
        return trajectory


def _validated_artifacts(simulation: BuiltWilsonSimulation) -> tuple[Path, Path]:
    reference_path = simulation.config.reference.path.expanduser().resolve()
    stationary_path = simulation.config.stationary_state.path.expanduser().resolve()
    try:
        disk_reference = load_reference_data(reference_path)
        disk_state = load_wilson_stationary_state(stationary_path)
    except Exception as exc:
        raise RunnerError("Wilson production run requires authenticated input artifacts") from exc
    if disk_reference.fingerprint_sha256 != simulation.reference.fingerprint_sha256:
        raise RunnerError("in-memory and on-disk Wilson references disagree")
    if disk_state.fingerprint_sha256 != simulation.stationary_state.fingerprint_sha256:
        raise RunnerError("in-memory and on-disk Wilson stationary states disagree")
    return reference_path, stationary_path


def execute_wilson_simulation(
    simulation: BuiltWilsonSimulation,
    *,
    control: RunControl | None = None,
    restart: WilsonRestartContext | None = None,
    reference_artifact_path: Path | None = None,
    stationary_state_artifact_path: Path | None = None,
) -> WilsonTrajectory:
    """Execute one exact-Wilson segment with bounded in-memory state."""

    if not isinstance(simulation, BuiltWilsonSimulation):
        raise TypeError("run requires a BuiltWilsonSimulation")
    controller = RunControl() if control is None else control
    if reference_artifact_path is None or stationary_state_artifact_path is None:
        validated_reference, validated_stationary = _validated_artifacts(simulation)
        reference_path = (
            validated_reference if reference_artifact_path is None else reference_artifact_path
        )
        stationary_path = (
            validated_stationary
            if stationary_state_artifact_path is None
            else stationary_state_artifact_path
        )
    else:
        reference_path = reference_artifact_path.resolve()
        stationary_path = stationary_state_artifact_path.resolve()
    run_id = uuid.uuid4().hex
    executor = _WilsonRunExecutor(
        simulation,
        run_id=run_id,
        reference_artifact_path=reference_path,
        stationary_state_artifact_path=stationary_path,
        restart=restart,
        control=controller,
    )
    with _signal_boundary(controller):
        try:
            return executor.execute()
        except RunCancelledError:
            raise
        except BaseException as exc:
            step = simulation.boundary_index
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
                executor.status.publish(
                    RunPhase.FAILED,
                    step=step,
                    force=True,
                    failure=FailureSummary(
                        error_type=type(exc).__name__,
                        message=str(exc) or type(exc).__name__,
                        phase=RunPhase.PROPAGATING,
                        step=step,
                    ),
                )
            raise


def resume_wilson_simulation(
    checkpoint_path: str | Path,
    *,
    output: OutputConfig | str | Path | None = None,
) -> WilsonTrajectory:
    """Authenticate a Wilson checkpoint and execute its child segment."""

    from aion.workflows.api import build_simulation

    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = load_wilson_checkpoint(path)
    reference = load_reference_data(checkpoint.reference_artifact_path)
    stationary = load_wilson_stationary_state(checkpoint.stationary_state_artifact_path)
    config = checkpoint.simulation_config
    if output is None:
        directory = path.parent / f"resume_{checkpoint.global_step:08d}_{uuid.uuid4().hex[:8]}"
        output_config = replace(config.output, directory=directory)
    elif isinstance(output, OutputConfig):
        output_config = output
    else:
        output_config = replace(config.output, directory=Path(output))
    config = replace(config, output=output_config)
    simulation = build_simulation(
        config,
        reference,
        stationary_state=stationary,
        original_toml=checkpoint.original_toml,
    )
    if not isinstance(simulation, BuiltWilsonSimulation):
        raise CheckpointError("Wilson checkpoint reconstructed the wrong runtime type")
    if simulation.source_provider.fingerprint_sha256 != checkpoint.source_fingerprint_sha256:
        raise CheckpointError("reconstructed Wilson source fingerprint disagrees")
    simulation.restore_boundary(
        checkpoint.contravariant_density,
        checkpoint.global_step,
    )
    restart = WilsonRestartContext(
        parent_run_id=checkpoint.run_id,
        parent_checkpoint_sha256=file_sha256(path),
        global_step_offset=checkpoint.global_step,
        accumulated_source_work_au=checkpoint.accumulated_source_work_au,
        initial_molecular_energy_au=checkpoint.initial_molecular_energy_au,
        observer_schedule_state=checkpoint.observer_schedule_state,
        boundary_density=checkpoint.contravariant_density,
    )
    return execute_wilson_simulation(
        simulation,
        restart=restart,
        reference_artifact_path=checkpoint.reference_artifact_path,
        stationary_state_artifact_path=checkpoint.stationary_state_artifact_path,
    )
