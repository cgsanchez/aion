"""Immutable authenticated restart checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from aion.config import SimulationConfig, canonical_sha256, dumps_config, loads_config
from aion.errors import CheckpointError, SchemaError
from aion.io.schemas import CHECKPOINT_SCHEMA
from aion.io.transaction import publish_hdf5, write_dataset
from aion.io.util import file_sha256, read_text, write_text

type PathInput = str | PathLike[str]


def _immutable(value: object, dtype: Any, name: str) -> np.ndarray:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    if not np.all(np.isfinite(array)):
        raise CheckpointError(f"checkpoint {name} contains non-finite values")
    array.flags.writeable = False
    return np.asarray(array)


@dataclass(frozen=True, slots=True)
class CheckpointData:
    """Complete backend-neutral state needed to continue one simulation."""

    run_id: str
    simulation_config: SimulationConfig
    original_toml: str
    reference_artifact_path: Path
    source_fingerprint_sha256: str
    global_step: int
    coefficients: np.ndarray
    occupations: np.ndarray
    density: np.ndarray
    accumulated_source_work_au: float
    initial_matter_energy_au: float
    applied_event_identifiers: frozenset[str]
    vector_potential_offset_au: np.ndarray
    parent_run_id: str | None = None
    parent_checkpoint_sha256: str | None = None
    global_step_offset: int = 0
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise CheckpointError("checkpoint run_id cannot be empty")
        if not isinstance(self.simulation_config, SimulationConfig):
            raise CheckpointError("checkpoint requires a SimulationConfig")
        if not self.original_toml:
            raise CheckpointError("checkpoint original TOML cannot be empty")
        object.__setattr__(self, "reference_artifact_path", Path(self.reference_artifact_path))
        if self.source_fingerprint_sha256 == "" or len(self.source_fingerprint_sha256) != 64:
            raise CheckpointError("checkpoint source fingerprint is invalid")
        grid = self.simulation_config.propagation.time_grid
        if (
            isinstance(self.global_step, bool)
            or not isinstance(self.global_step, int)
            or not 0 <= self.global_step <= grid.intervals
        ):
            raise CheckpointError("checkpoint global step lies outside its time grid")
        coefficients = _immutable(self.coefficients, np.complex128, "coefficients")
        occupations = _immutable(self.occupations, np.float64, "occupations")
        if coefficients.ndim != 2 or occupations.shape != (coefficients.shape[1],):
            raise CheckpointError("checkpoint coefficient and occupation shapes disagree")
        if np.any(occupations <= 0.0):
            raise CheckpointError("checkpoint stores occupied orbitals only")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "occupations", occupations)
        density = _immutable(self.density, np.complex128, "density")
        if density.shape != (coefficients.shape[0], coefficients.shape[0]):
            raise CheckpointError("checkpoint density shape disagrees with its coefficients")
        hermitian_scale = max(1.0, float(np.linalg.norm(density)))
        if np.linalg.norm(density - density.conj().T) > 1.0e-12 * hermitian_scale:
            raise CheckpointError("checkpoint density is not Hermitian")
        reconstructed = (coefficients * occupations[None, :]) @ coefficients.conj().T
        if not np.allclose(density, reconstructed, rtol=1.0e-12, atol=1.0e-13):
            raise CheckpointError("checkpoint density disagrees with its occupied orbitals")
        object.__setattr__(self, "density", density)
        for name in ("accumulated_source_work_au", "initial_matter_energy_au"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise CheckpointError(f"checkpoint {name} must be finite")
            object.__setattr__(self, name, value)
        identifiers = frozenset(self.applied_event_identifiers)
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise CheckpointError("checkpoint event identifiers must be nonempty text")
        object.__setattr__(self, "applied_event_identifiers", identifiers)
        offset = _immutable(
            self.vector_potential_offset_au,
            np.float64,
            "vector-potential offset",
        )
        if offset.shape != (3,):
            raise CheckpointError("checkpoint vector-potential offset must have shape (3,)")
        object.__setattr__(self, "vector_potential_offset_au", offset)
        if isinstance(self.global_step_offset, bool) or self.global_step_offset < 0:
            raise CheckpointError("checkpoint global_step_offset must be nonnegative")
        for name in ("parent_run_id", "parent_checkpoint_sha256"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise CheckpointError(f"checkpoint {name} must be nonempty text when present")
        if self.parent_checkpoint_sha256 is not None and len(self.parent_checkpoint_sha256) != 64:
            raise CheckpointError("checkpoint parent checksum is invalid")
        object.__setattr__(
            self,
            "artifact_id",
            canonical_sha256(
                {
                    "schema": "aion.checkpoint-content",
                    "version": "1.0.0",
                    "run_id": self.run_id,
                    "simulation_id": self.simulation_config.scientific_id,
                    "source_fingerprint_sha256": self.source_fingerprint_sha256,
                    "global_step": self.global_step,
                    "coefficients": coefficients,
                    "occupations": occupations,
                    "density": density,
                    "accumulated_source_work_au": self.accumulated_source_work_au,
                    "initial_matter_energy_au": self.initial_matter_energy_au,
                    "applied_event_identifiers": sorted(identifiers),
                    "vector_potential_offset_au": offset,
                    "parent_run_id": self.parent_run_id,
                    "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
                    "global_step_offset": self.global_step_offset,
                }
            ),
        )

    @property
    def time_au(self) -> float:
        return self.simulation_config.propagation.time_grid.time_at(self.global_step)


def save_checkpoint(checkpoint: CheckpointData, path: PathInput) -> str:
    """Publish one immutable checkpoint and return its whole-file SHA-256."""

    target = Path(path)

    def populate(handle: h5py.File) -> None:
        meta = handle["meta"]
        configuration = handle["configuration"]
        reference = handle["reference"]
        time = handle["time"]
        source = handle["source"]
        state = handle["state"]
        events = handle["events"]
        restart = handle["restart"]
        for group in (meta, configuration, reference, time, source, state, events, restart):
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
            source,
            "vector_potential_offset_au",
            checkpoint.vector_potential_offset_au,
            unit="atomic_unit_of_reduced_vector_potential",
            physical_dimension="vector_potential_reduced",
        )
        write_dataset(
            state,
            "coefficients",
            checkpoint.coefficients,
            unit="1",
            physical_dimension="ao_occupied_coefficients",
            compressed=True,
        )
        write_dataset(
            state,
            "occupations",
            checkpoint.occupations,
            unit="electron",
            physical_dimension="orbital_occupation",
            compressed=True,
        )
        write_dataset(
            state,
            "density",
            checkpoint.density,
            unit="electron",
            physical_dimension="ao_density_matrix",
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
            "initial_matter_energy_au",
            np.float64(checkpoint.initial_matter_energy_au),
            unit="hartree",
            physical_dimension="energy",
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        write_dataset(
            events,
            "applied_idempotency_identifiers",
            np.asarray(sorted(checkpoint.applied_event_identifiers), dtype=string_dtype),
            unit="1",
            physical_dimension="event_identifier",
        )
        restart.attrs["global_step_offset"] = np.int64(checkpoint.global_step_offset)
        restart.attrs["parent_run_id"] = checkpoint.parent_run_id or ""
        restart.attrs["parent_checkpoint_sha256"] = checkpoint.parent_checkpoint_sha256 or ""

    publish_hdf5(
        target,
        CHECKPOINT_SCHEMA,
        artifact_id=checkpoint.artifact_id,
        populate=populate,
    )
    return file_sha256(target)


def _required(handle: h5py.File, paths: tuple[str, ...]) -> None:
    missing = [path for path in paths if path not in handle]
    if missing:
        raise SchemaError("checkpoint lacks required object(s): " + ", ".join(missing))


def load_checkpoint(path: PathInput) -> CheckpointData:
    """Load and authenticate a complete backend-neutral checkpoint."""

    from aion.io.schemas import validate_artifact

    artifact_path = Path(path)
    header = validate_artifact(artifact_path, expected_schema=CHECKPOINT_SCHEMA)
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
                "time/global_step",
                "time/time_au",
                "source/fingerprint_sha256",
                "source/vector_potential_offset_au",
                "state/coefficients",
                "state/occupations",
                "state/density",
                "state/accumulated_source_work_au",
                "state/initial_matter_energy_au",
                "events/applied_idempotency_identifiers",
            ),
        )
        resolved = loads_config(read_text(handle["configuration/normalized_toml"]))
        if not isinstance(resolved.config, SimulationConfig):
            raise CheckpointError("checkpoint contains a non-simulation configuration")
        config = resolved.config
        if read_text(handle["meta/simulation_id_sha256"]) != config.scientific_id:
            raise CheckpointError("checkpoint simulation identity mismatch")
        if read_text(handle["reference/fingerprint_sha256"]) != (
            config.reference.fingerprint_sha256
        ):
            raise CheckpointError("checkpoint reference fingerprint mismatch")
        step = int(handle["time/global_step"][()])
        time_au = float(handle["time/time_au"][()])
        if time_au != config.propagation.time_grid.time_at(step):
            raise CheckpointError("checkpoint time does not equal its authoritative integer step")
        restart = handle["restart"]
        assert isinstance(restart, h5py.Group)
        parent_run = str(restart.attrs.get("parent_run_id", "")) or None
        parent_sha = str(restart.attrs.get("parent_checkpoint_sha256", "")) or None
        identifiers = frozenset(
            str(value) for value in handle["events/applied_idempotency_identifiers"].asstr()[...]
        )
        checkpoint = CheckpointData(
            run_id=read_text(handle["meta/run_id"]),
            simulation_config=config,
            original_toml=read_text(handle["configuration/original_toml"]),
            reference_artifact_path=Path(read_text(handle["reference/artifact_path"])),
            source_fingerprint_sha256=read_text(handle["source/fingerprint_sha256"]),
            global_step=step,
            coefficients=handle["state/coefficients"][...],
            occupations=handle["state/occupations"][...],
            density=handle["state/density"][...],
            accumulated_source_work_au=float(handle["state/accumulated_source_work_au"][()]),
            initial_matter_energy_au=float(handle["state/initial_matter_energy_au"][()]),
            applied_event_identifiers=identifiers,
            vector_potential_offset_au=handle["source/vector_potential_offset_au"][...],
            parent_run_id=parent_run,
            parent_checkpoint_sha256=parent_sha,
            global_step_offset=int(restart.attrs.get("global_step_offset", 0)),
        )
    if checkpoint.artifact_id != header.artifact_id:
        raise CheckpointError("checkpoint content fingerprint mismatch")
    return checkpoint
