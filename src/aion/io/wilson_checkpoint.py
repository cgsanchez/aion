"""Authenticated density-native restart checkpoints for exact-Wilson runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from aion.config import WilsonSimulationConfig, canonical_sha256, dumps_config, loads_config
from aion.errors import CheckpointError, SchemaError
from aion.io.schemas import WILSON_CHECKPOINT_SCHEMA, validate_artifact
from aion.io.transaction import publish_hdf5, write_dataset
from aion.io.util import file_sha256, read_text, write_text

type PathInput = str | PathLike[str]


def _immutable(value: object, dtype: Any, name: str, *, ndim: int) -> np.ndarray:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise CheckpointError(f"checkpoint {name} must be a finite {ndim}-dimensional array")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, slots=True)
class WilsonCheckpointData:
    """Complete backend-neutral state for one accepted Wilson boundary."""

    run_id: str
    simulation_config: WilsonSimulationConfig
    original_toml: str
    reference_artifact_path: Path
    stationary_state_artifact_path: Path
    source_fingerprint_sha256: str
    global_step: int
    contravariant_density: np.ndarray
    accumulated_source_work_au: float
    initial_molecular_energy_au: float
    observer_schedule_state: tuple[tuple[str, int], ...]
    applied_event_identifiers: tuple[str, ...] = ()
    parent_run_id: str | None = None
    parent_checkpoint_sha256: str | None = None
    global_step_offset: int = 0
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise CheckpointError("Wilson checkpoint run_id cannot be empty")
        if not isinstance(self.simulation_config, WilsonSimulationConfig):
            raise CheckpointError("Wilson checkpoint requires a WilsonSimulationConfig")
        if not isinstance(self.original_toml, str) or not self.original_toml:
            raise CheckpointError("Wilson checkpoint original TOML cannot be empty")
        object.__setattr__(self, "reference_artifact_path", Path(self.reference_artifact_path))
        object.__setattr__(
            self,
            "stationary_state_artifact_path",
            Path(self.stationary_state_artifact_path),
        )
        if len(self.source_fingerprint_sha256) != 64:
            raise CheckpointError("Wilson checkpoint source fingerprint is invalid")
        grid = self.simulation_config.propagation.time_grid
        if (
            isinstance(self.global_step, bool)
            or not isinstance(self.global_step, int)
            or not 0 <= self.global_step <= grid.intervals
        ):
            raise CheckpointError("Wilson checkpoint step lies outside its time grid")
        density = _immutable(
            self.contravariant_density,
            np.complex128,
            "contravariant density",
            ndim=2,
        )
        if density.shape[0] == 0 or density.shape[0] != density.shape[1]:
            raise CheckpointError("Wilson checkpoint density must be nonempty and square")
        scale = max(1.0, float(np.linalg.norm(density)))
        if np.linalg.norm(density - density.conj().T) > 1.0e-11 * scale:
            raise CheckpointError("Wilson checkpoint density is not Hermitian")
        object.__setattr__(self, "contravariant_density", density)
        for name in ("accumulated_source_work_au", "initial_molecular_energy_au"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise CheckpointError(f"Wilson checkpoint {name} must be finite")
            object.__setattr__(self, name, value)
        schedule = tuple(self.observer_schedule_state)
        labels = [name for name, _ in schedule]
        if len(set(labels)) != len(labels):
            raise CheckpointError("Wilson checkpoint observer schedule labels are duplicated")
        for name, step in schedule:
            if not name or not isinstance(name, str):
                raise CheckpointError("Wilson checkpoint observer label is invalid")
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or not -1 <= step <= self.global_step
            ):
                raise CheckpointError("Wilson checkpoint observer step is invalid")
        object.__setattr__(self, "observer_schedule_state", tuple(sorted(schedule)))
        identifiers = tuple(sorted(set(self.applied_event_identifiers)))
        if any(not identifier for identifier in identifiers):
            raise CheckpointError("Wilson checkpoint event identifier is invalid")
        object.__setattr__(self, "applied_event_identifiers", identifiers)
        for name in ("global_step_offset",):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= self.global_step
            ):
                raise CheckpointError(f"Wilson checkpoint {name} is invalid")
        object.__setattr__(self, "artifact_id", canonical_sha256(self.scientific_mapping()))

    @property
    def time_au(self) -> float:
        return self.simulation_config.propagation.time_grid.time_at(self.global_step)

    def scientific_mapping(self) -> dict[str, object]:
        return {
            "schema": WILSON_CHECKPOINT_SCHEMA.name,
            "version": str(WILSON_CHECKPOINT_SCHEMA.version),
            "run_id": self.run_id,
            "simulation_id_sha256": self.simulation_config.scientific_id,
            "reference_fingerprint_sha256": (self.simulation_config.reference.fingerprint_sha256),
            "stationary_state_fingerprint_sha256": (
                self.simulation_config.stationary_state.fingerprint_sha256
            ),
            "source_fingerprint_sha256": self.source_fingerprint_sha256,
            "global_step": self.global_step,
            "contravariant_density": self.contravariant_density,
            "accumulated_source_work_au": self.accumulated_source_work_au,
            "initial_molecular_energy_au": self.initial_molecular_energy_au,
            "observer_schedule_state": list(self.observer_schedule_state),
            "applied_event_identifiers": list(self.applied_event_identifiers),
            "parent_run_id": self.parent_run_id,
            "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
            "global_step_offset": self.global_step_offset,
        }


def save_wilson_checkpoint(checkpoint: WilsonCheckpointData, path: PathInput) -> str:
    """Transactionally publish one immutable Wilson checkpoint."""

    if not isinstance(checkpoint, WilsonCheckpointData):
        raise TypeError("checkpoint must be WilsonCheckpointData")
    target = Path(path)

    def populate(handle: h5py.File) -> None:
        meta = handle["meta"]
        configuration = handle["configuration"]
        reference = handle["reference"]
        stationary = handle["stationary_state"]
        time = handle["time"]
        source = handle["source"]
        state = handle["state"]
        observers = handle["observers"]
        events = handle["events"]
        restart = handle["restart"]
        for group in (
            meta,
            configuration,
            reference,
            stationary,
            time,
            source,
            state,
            observers,
            events,
            restart,
        ):
            assert isinstance(group, h5py.Group)
        write_text(meta, "run_id", checkpoint.run_id, physical_dimension="run_identifier")
        write_text(
            meta,
            "simulation_id_sha256",
            checkpoint.simulation_config.scientific_id,
            physical_dimension="sha256_digest",
        )
        write_text(
            configuration,
            "normalized_toml",
            dumps_config(checkpoint.simulation_config),
            physical_dimension="configuration",
        )
        write_text(
            configuration,
            "original_toml",
            checkpoint.original_toml,
            physical_dimension="configuration",
        )
        write_text(
            reference,
            "artifact_path",
            str(checkpoint.reference_artifact_path),
            physical_dimension="filesystem_path",
        )
        write_text(
            reference,
            "fingerprint_sha256",
            checkpoint.simulation_config.reference.fingerprint_sha256,
            physical_dimension="sha256_digest",
        )
        write_text(
            stationary,
            "artifact_path",
            str(checkpoint.stationary_state_artifact_path),
            physical_dimension="filesystem_path",
        )
        write_text(
            stationary,
            "fingerprint_sha256",
            checkpoint.simulation_config.stationary_state.fingerprint_sha256,
            physical_dimension="sha256_digest",
        )
        write_dataset(
            time,
            "global_step",
            np.int64(checkpoint.global_step),
            unit="1",
            physical_dimension="time_step_index",
        )
        write_dataset(
            time,
            "time_au",
            np.float64(checkpoint.time_au),
            unit="atomic_unit_of_time",
            physical_dimension="time",
        )
        write_text(
            source,
            "fingerprint_sha256",
            checkpoint.source_fingerprint_sha256,
            physical_dimension="sha256_digest",
        )
        write_dataset(
            state,
            "contravariant_density",
            checkpoint.contravariant_density,
            unit="electron",
            physical_dimension="contravariant_ao_density_matrix",
            compressed=True,
        )
        write_dataset(
            state,
            "accumulated_source_work_au",
            np.float64(checkpoint.accumulated_source_work_au),
            unit="hartree",
            physical_dimension="energy",
        )
        write_dataset(
            state,
            "initial_molecular_energy_au",
            np.float64(checkpoint.initial_molecular_energy_au),
            unit="hartree",
            physical_dimension="energy",
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        write_dataset(
            observers,
            "labels",
            np.asarray(
                [name for name, _ in checkpoint.observer_schedule_state], dtype=string_dtype
            ),
            unit="1",
            physical_dimension="observer_identifier",
        )
        write_dataset(
            observers,
            "last_recorded_steps",
            np.asarray([step for _, step in checkpoint.observer_schedule_state], dtype=np.int64),
            unit="1",
            physical_dimension="time_step_index",
        )
        write_dataset(
            events,
            "applied_idempotency_identifiers",
            np.asarray(checkpoint.applied_event_identifiers, dtype=string_dtype),
            unit="1",
            physical_dimension="event_identifier",
        )
        restart.attrs["global_step_offset"] = np.int64(checkpoint.global_step_offset)
        restart.attrs["parent_run_id"] = checkpoint.parent_run_id or ""
        restart.attrs["parent_checkpoint_sha256"] = checkpoint.parent_checkpoint_sha256 or ""

    publish_hdf5(
        target,
        WILSON_CHECKPOINT_SCHEMA,
        artifact_id=checkpoint.artifact_id,
        populate=populate,
    )
    return file_sha256(target)


def _required(handle: h5py.File, paths: tuple[str, ...]) -> None:
    missing = [path for path in paths if path not in handle]
    if missing:
        raise SchemaError("Wilson checkpoint lacks required object(s): " + ", ".join(missing))


def load_wilson_checkpoint(path: PathInput) -> WilsonCheckpointData:
    """Load and authenticate one complete Wilson checkpoint."""

    artifact_path = Path(path)
    header = validate_artifact(artifact_path, expected_schema=WILSON_CHECKPOINT_SCHEMA)
    with h5py.File(artifact_path, "r") as handle:
        _required(
            handle,
            (
                "meta/run_id",
                "meta/simulation_id_sha256",
                "configuration/normalized_toml",
                "configuration/original_toml",
                "reference/artifact_path",
                "reference/fingerprint_sha256",
                "stationary_state/artifact_path",
                "stationary_state/fingerprint_sha256",
                "time/global_step",
                "time/time_au",
                "source/fingerprint_sha256",
                "state/contravariant_density",
                "state/accumulated_source_work_au",
                "state/initial_molecular_energy_au",
                "observers/labels",
                "observers/last_recorded_steps",
                "events/applied_idempotency_identifiers",
            ),
        )
        resolved = loads_config(read_text(handle["configuration/normalized_toml"]))
        if not isinstance(resolved.config, WilsonSimulationConfig):
            raise CheckpointError("Wilson checkpoint contains the wrong configuration type")
        config = resolved.config
        if read_text(handle["meta/simulation_id_sha256"]) != config.scientific_id:
            raise CheckpointError("Wilson checkpoint simulation identity mismatch")
        if read_text(handle["reference/fingerprint_sha256"]) != (
            config.reference.fingerprint_sha256
        ):
            raise CheckpointError("Wilson checkpoint reference identity mismatch")
        if read_text(handle["stationary_state/fingerprint_sha256"]) != (
            config.stationary_state.fingerprint_sha256
        ):
            raise CheckpointError("Wilson checkpoint stationary-state identity mismatch")
        step = int(handle["time/global_step"][()])
        if float(handle["time/time_au"][()]) != config.propagation.time_grid.time_at(step):
            raise CheckpointError("Wilson checkpoint time disagrees with its integer step")
        labels = tuple(str(value) for value in handle["observers/labels"].asstr()[...])
        steps = tuple(int(value) for value in handle["observers/last_recorded_steps"][...])
        if len(labels) != len(steps):
            raise CheckpointError("Wilson checkpoint observer arrays disagree")
        events = tuple(
            str(value) for value in handle["events/applied_idempotency_identifiers"].asstr()[...]
        )
        restart = handle["restart"]
        assert isinstance(restart, h5py.Group)
        checkpoint = WilsonCheckpointData(
            run_id=read_text(handle["meta/run_id"]),
            simulation_config=config,
            original_toml=read_text(handle["configuration/original_toml"]),
            reference_artifact_path=Path(read_text(handle["reference/artifact_path"])),
            stationary_state_artifact_path=Path(
                read_text(handle["stationary_state/artifact_path"])
            ),
            source_fingerprint_sha256=read_text(handle["source/fingerprint_sha256"]),
            global_step=step,
            contravariant_density=handle["state/contravariant_density"][...],
            accumulated_source_work_au=float(handle["state/accumulated_source_work_au"][()]),
            initial_molecular_energy_au=float(handle["state/initial_molecular_energy_au"][()]),
            observer_schedule_state=tuple(zip(labels, steps, strict=True)),
            applied_event_identifiers=events,
            parent_run_id=str(restart.attrs.get("parent_run_id", "")) or None,
            parent_checkpoint_sha256=(
                str(restart.attrs.get("parent_checkpoint_sha256", "")) or None
            ),
            global_step_offset=int(restart.attrs.get("global_step_offset", 0)),
        )
    if checkpoint.artifact_id != header.artifact_id:
        raise CheckpointError("Wilson checkpoint content fingerprint mismatch")
    return checkpoint
