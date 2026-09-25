"""Streaming exact-Wilson trajectory writer and immutable lazy reader."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import h5py
import numpy as np

from aion.config import WilsonSimulationConfig, dumps_config
from aion.electromagnetism import UniformMagneticSourceSample
from aion.errors import TrajectoryError
from aion.io.schemas import WILSON_TRAJECTORY_SCHEMA, stamp_artifact, validate_artifact
from aion.io.transaction import write_dataset
from aion.io.util import file_sha256, read_text, write_text

type PathInput = str | PathLike[str]


@dataclass(frozen=True, slots=True)
class WilsonSeries:
    name: str
    steps: np.ndarray
    times_au: np.ndarray
    values: np.ndarray
    unit: str
    physical_dimension: str


@dataclass(frozen=True, slots=True)
class WilsonTrajectory:
    path: Path
    run_id: str
    simulation_id: str
    reference_fingerprint_sha256: str
    stationary_state_fingerprint_sha256: str
    source_fingerprint_sha256: str
    parent_run_id: str | None
    parent_checkpoint_sha256: str | None
    global_step_offset: int
    final_step: int
    accumulated_source_work_au: float
    series_names: tuple[str, ...]
    checkpoint_sha256: tuple[str, ...]
    complete: bool
    sha256: str

    def read_series(self, name: str) -> WilsonSeries:
        if name not in self.series_names:
            raise TrajectoryError(f"Wilson trajectory has no series {name!r}")
        with h5py.File(self.path, "r") as handle:
            group = handle[f"observables/{name}"]
            assert isinstance(group, h5py.Group)
            values = group["values"]
            assert isinstance(values, h5py.Dataset)
            return WilsonSeries(
                name=name,
                steps=np.asarray(group["step"][...], dtype=np.int64),
                times_au=np.asarray(group["time_au"][...], dtype=np.float64),
                values=np.asarray(values[...]),
                unit=str(values.attrs["unit"]),
                physical_dimension=str(values.attrs["physical_dimension"]),
            )


@dataclass(slots=True)
class _Stream:
    datasets: dict[str, h5py.Dataset]

    def append(self, values: dict[str, object]) -> None:
        if set(values) != set(self.datasets):
            raise TrajectoryError("Wilson stream record does not match its schema")
        for name, dataset in self.datasets.items():
            index = dataset.shape[0]
            dataset.resize(index + 1, axis=0)
            dataset[index] = values[name]


def _nested_group(root: h5py.Group, path: str) -> h5py.Group:
    result = root
    for part in path.split("/"):
        result = result.require_group(part)
    return result


def _create_stream(
    group: h5py.Group,
    fields: dict[str, tuple[tuple[int, ...], object, str, str]],
    *,
    compressed: bool = False,
) -> _Stream:
    datasets: dict[str, h5py.Dataset] = {}
    for name, (shape, dtype, unit, dimension) in fields.items():
        kwargs: dict[str, object] = {
            "shape": (0, *shape),
            "maxshape": (None, *shape),
            "dtype": dtype,
            "chunks": (1, *shape) if shape else (256,),
        }
        if compressed:
            kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
        dataset = group.create_dataset(name, **kwargs)
        dataset.attrs["unit"] = unit
        dataset.attrs["physical_dimension"] = dimension
        datasets[name] = dataset
    return _Stream(datasets)


class WilsonTrajectoryWriter:
    """Append accepted Wilson records to a private file and publish once."""

    def __init__(
        self,
        directory: PathInput,
        *,
        run_id: str,
        config: WilsonSimulationConfig,
        original_toml: str,
        reference_artifact_path: Path,
        stationary_state_artifact_path: Path,
        source_fingerprint_sha256: str,
        provenance_json: str,
        parent_run_id: str | None = None,
        parent_checkpoint_sha256: str | None = None,
        global_step_offset: int = 0,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.target = self.directory / "trajectory.h5"
        self.failed_target = self.directory / "trajectory.failed.h5"
        if self.target.exists() or self.failed_target.exists():
            raise TrajectoryError(
                f"run directory already contains a Wilson trajectory: {self.directory}"
            )
        self.partial = self.directory / f".trajectory.{run_id}.{uuid.uuid4().hex}.partial"
        self.run_id = run_id
        self._streams: dict[str, _Stream] = {}
        self._closed = False
        self._handle = h5py.File(self.partial, "x")
        try:
            stamp_artifact(
                self._handle,
                WILSON_TRAJECTORY_SCHEMA,
                complete=False,
                artifact_id=run_id,
            )
            self._initialize(
                config=config,
                original_toml=original_toml,
                reference_artifact_path=reference_artifact_path,
                stationary_state_artifact_path=stationary_state_artifact_path,
                source_fingerprint_sha256=source_fingerprint_sha256,
                provenance_json=provenance_json,
                parent_run_id=parent_run_id,
                parent_checkpoint_sha256=parent_checkpoint_sha256,
                global_step_offset=global_step_offset,
            )
        except BaseException:
            self._handle.close()
            self.partial.unlink(missing_ok=True)
            self._closed = True
            raise

    def _initialize(
        self,
        *,
        config: WilsonSimulationConfig,
        original_toml: str,
        reference_artifact_path: Path,
        stationary_state_artifact_path: Path,
        source_fingerprint_sha256: str,
        provenance_json: str,
        parent_run_id: str | None,
        parent_checkpoint_sha256: str | None,
        global_step_offset: int,
    ) -> None:
        meta = self._handle["meta"]
        configuration = self._handle["configuration"]
        reference = self._handle["reference"]
        stationary = self._handle["stationary_state"]
        time = self._handle["time"]
        source = self._handle["source"]
        restart = self._handle["restart"]
        for group in (meta, configuration, reference, stationary, time, source, restart):
            assert isinstance(group, h5py.Group)
        write_text(meta, "run_id", self.run_id, physical_dimension="run_identifier")
        write_text(
            meta,
            "simulation_id_sha256",
            config.scientific_id,
            physical_dimension="sha256_digest",
        )
        write_text(
            meta,
            "provenance_json",
            provenance_json,
            physical_dimension="execution_provenance",
        )
        write_text(
            configuration,
            "normalized_toml",
            dumps_config(config),
            physical_dimension="configuration",
        )
        write_text(
            configuration,
            "original_toml",
            original_toml,
            physical_dimension="configuration",
        )
        write_text(
            reference,
            "artifact_path",
            str(reference_artifact_path),
            physical_dimension="filesystem_path",
        )
        reference.attrs["fingerprint_sha256"] = config.reference.fingerprint_sha256
        write_text(
            stationary,
            "artifact_path",
            str(stationary_state_artifact_path),
            physical_dimension="filesystem_path",
        )
        stationary.attrs["fingerprint_sha256"] = config.stationary_state.fingerprint_sha256
        grid = config.propagation.time_grid
        time.attrs["start_au"] = grid.start_au
        time.attrs["step_au"] = grid.step_au
        time.attrs["intervals"] = grid.intervals
        source.attrs["source_fingerprint_sha256"] = source_fingerprint_sha256
        write_text(
            source,
            "definition_toml",
            dumps_config(config),
            physical_dimension="source_definition",
        )
        restart.attrs["parent_run_id"] = parent_run_id or ""
        restart.attrs["parent_checkpoint_sha256"] = parent_checkpoint_sha256 or ""
        restart.attrs["global_step_offset"] = np.int64(global_step_offset)
        restart.attrs["boundary_authenticated"] = np.uint8(parent_run_id is None)

    def append_series(
        self,
        name: str,
        *,
        step: int,
        time_au: float,
        value: object,
        unit: str,
        physical_dimension: str,
    ) -> None:
        array = np.asarray(value)
        if not np.all(np.isfinite(array)):
            raise TrajectoryError(f"Wilson observable {name!r} is non-finite")
        if name not in self._streams:
            root = self._handle["observables"]
            assert isinstance(root, h5py.Group)
            group = _nested_group(root, name)
            self._streams[name] = _create_stream(
                group,
                {
                    "step": ((), np.int64, "1", "time_step_index"),
                    "time_au": ((), np.float64, "atomic_unit_of_time", "time"),
                    "values": (array.shape, array.dtype, unit, physical_dimension),
                },
                compressed=array.ndim >= 2,
            )
        self._streams[name].append(
            {
                "step": np.int64(step),
                "time_au": np.float64(time_au),
                "values": array,
            }
        )

    def append_source(self, *, step: int, sample: UniformMagneticSourceSample) -> None:
        if "source/endpoint" not in self._streams:
            source = self._handle["source"]
            assert isinstance(source, h5py.Group)
            self._streams["source/endpoint"] = _create_stream(
                source.require_group("endpoint"),
                {
                    "step": ((), np.int64, "1", "time_step_index"),
                    "time_au": ((), np.float64, "atomic_unit_of_time", "time"),
                    "electric_field_origin_au": (
                        (3,),
                        np.float64,
                        "atomic_unit_of_electric_field",
                        "electric_field",
                    ),
                    "magnetic_field_au": (
                        (3,),
                        np.float64,
                        "atomic_unit_of_magnetic_field",
                        "magnetic_field",
                    ),
                    "magnetic_field_dot_au": (
                        (3,),
                        np.float64,
                        "atomic_unit_of_magnetic_field_per_atomic_unit_of_time",
                        "magnetic_field_rate",
                    ),
                },
            )
        self._streams["source/endpoint"].append(
            {
                "step": np.int64(step),
                "time_au": np.float64(sample.time_au),
                "electric_field_origin_au": np.asarray(
                    sample.electric_field_origin_au,
                    dtype=np.float64,
                ),
                "magnetic_field_au": np.asarray(
                    sample.field.magnetic_field_au,
                    dtype=np.float64,
                ),
                "magnetic_field_dot_au": np.asarray(
                    sample.magnetic_field_dot_au,
                    dtype=np.float64,
                ),
            }
        )

    def append_interval_work(
        self,
        *,
        step: int,
        gauss_times_au: tuple[float, float],
        gauss_power_au: tuple[float, float],
        increment_au: float,
        accumulated_au: float,
    ) -> None:
        if "work/gauss_interval" not in self._streams:
            diagnostics = self._handle["diagnostics"]
            assert isinstance(diagnostics, h5py.Group)
            self._streams["work/gauss_interval"] = _create_stream(
                diagnostics.require_group("gauss_interval_work"),
                {
                    "step": ((), np.int64, "1", "time_step_index"),
                    "gauss_times_au": (
                        (2,),
                        np.float64,
                        "atomic_unit_of_time",
                        "time",
                    ),
                    "gauss_power_au": (
                        (2,),
                        np.float64,
                        "hartree_per_atomic_unit_of_time",
                        "power",
                    ),
                    "increment_au": ((), np.float64, "hartree", "energy"),
                    "accumulated_au": ((), np.float64, "hartree", "energy"),
                },
            )
        self._streams["work/gauss_interval"].append(
            {
                "step": np.int64(step),
                "gauss_times_au": np.asarray(gauss_times_au, dtype=np.float64),
                "gauss_power_au": np.asarray(gauss_power_au, dtype=np.float64),
                "increment_au": np.float64(increment_au),
                "accumulated_au": np.float64(accumulated_au),
            }
        )

    def append_diagnostics(self, *, step: int, time_au: float, values: dict[str, float]) -> None:
        key = "propagation/step"
        if key not in self._streams:
            diagnostics = self._handle["diagnostics"]
            assert isinstance(diagnostics, h5py.Group)
            fields: dict[str, tuple[tuple[int, ...], object, str, str]] = {
                "step": ((), np.int64, "1", "time_step_index"),
                "time_au": ((), np.float64, "atomic_unit_of_time", "time"),
            }
            fields.update(
                {name: ((), np.float64, "1", "dimensionless_residual") for name in values}
            )
            self._streams[key] = _create_stream(
                diagnostics.require_group("propagation"),
                fields,
            )
        record: dict[str, object] = {
            "step": np.int64(step),
            "time_au": np.float64(time_au),
        }
        record.update({name: np.float64(value) for name, value in values.items()})
        self._streams[key].append(record)

    def append_snapshot(self, *, step: int, time_au: float, density: np.ndarray) -> None:
        key = "snapshot/density"
        if key not in self._streams:
            diagnostics = self._handle["diagnostics"]
            assert isinstance(diagnostics, h5py.Group)
            self._streams[key] = _create_stream(
                diagnostics.require_group("matrix_snapshots"),
                {
                    "step": ((), np.int64, "1", "time_step_index"),
                    "time_au": ((), np.float64, "atomic_unit_of_time", "time"),
                    "contravariant_density": (
                        density.shape,
                        np.complex128,
                        "electron",
                        "contravariant_ao_density_matrix",
                    ),
                },
                compressed=True,
            )
        self._streams[key].append(
            {
                "step": np.int64(step),
                "time_au": np.float64(time_au),
                "contravariant_density": density,
            }
        )

    def append_checkpoint_link(
        self,
        *,
        step: int,
        time_au: float,
        path: Path,
        sha256: str,
    ) -> None:
        key = "restart/checkpoints"
        if key not in self._streams:
            restart = self._handle["restart"]
            assert isinstance(restart, h5py.Group)
            text = h5py.string_dtype("utf-8")
            self._streams[key] = _create_stream(
                restart.require_group("checkpoints"),
                {
                    "step": ((), np.int64, "1", "time_step_index"),
                    "time_au": ((), np.float64, "atomic_unit_of_time", "time"),
                    "path": ((), text, "1", "filesystem_path"),
                    "sha256": ((), text, "1", "sha256_digest"),
                },
            )
        self._streams[key].append(
            {
                "step": np.int64(step),
                "time_au": np.float64(time_au),
                "path": str(path),
                "sha256": sha256,
            }
        )

    def authenticate_restart_boundary(
        self,
        *,
        step: int,
        density: np.ndarray,
        accumulated_source_work_au: float,
    ) -> None:
        restart = self._handle["restart"]
        assert isinstance(restart, h5py.Group)
        write_dataset(
            restart,
            "boundary_step",
            np.int64(step),
            unit="1",
            physical_dimension="time_step_index",
        )
        write_dataset(
            restart,
            "boundary_contravariant_density",
            density,
            unit="electron",
            physical_dimension="contravariant_ao_density_matrix",
            compressed=True,
        )
        write_dataset(
            restart,
            "boundary_accumulated_source_work_au",
            np.float64(accumulated_source_work_au),
            unit="hartree",
            physical_dimension="energy",
        )
        restart.attrs["boundary_authenticated"] = np.uint8(1)

    def set_summary(self, *, final_step: int, accumulated_source_work_au: float) -> None:
        meta = self._handle["meta"]
        assert isinstance(meta, h5py.Group)
        meta.attrs["final_step"] = np.int64(final_step)
        meta.attrs["accumulated_source_work_au"] = np.float64(accumulated_source_work_au)

    def flush(self) -> None:
        self._handle.flush()

    def finalize(self) -> WilsonTrajectory:
        if self._closed:
            raise TrajectoryError("Wilson trajectory writer is already closed")
        self._handle.flush()
        self._handle.close()
        self._closed = True
        with h5py.File(self.partial, "r+") as handle:
            handle.attrs.modify("complete", 1)
            handle.flush()
        validate_artifact(self.partial, expected_schema=WILSON_TRAJECTORY_SCHEMA)
        try:
            os.link(self.partial, self.target)
        except FileExistsError:
            raise TrajectoryError(
                f"refusing to overwrite Wilson trajectory {self.target}"
            ) from None
        self.partial.unlink()
        return load_wilson_trajectory(self.target)

    def fail(self, *, error_type: str, message: str, phase: str, step: int) -> Path:
        if self._closed:
            return self.failed_target
        meta = self._handle["meta"]
        assert isinstance(meta, h5py.Group)
        meta.attrs["failure_type"] = error_type
        meta.attrs["failure_message"] = message
        meta.attrs["failure_phase"] = phase
        meta.attrs["failure_step"] = np.int64(step)
        self._handle.flush()
        self._handle.close()
        self._closed = True
        try:
            os.link(self.partial, self.failed_target)
        except FileExistsError:
            raise TrajectoryError(
                f"refusing to overwrite failed Wilson trajectory {self.failed_target}"
            ) from None
        self.partial.unlink()
        return self.failed_target


def _series_names(root: h5py.Group) -> tuple[str, ...]:
    names: list[str] = []

    def visitor(name: str, value: h5py.Group | h5py.Dataset) -> None:
        if isinstance(value, h5py.Group) and {"step", "time_au", "values"}.issubset(value):
            names.append(name)

    root.visititems(visitor)
    return tuple(sorted(names))


def load_wilson_trajectory(path: PathInput) -> WilsonTrajectory:
    """Open and authenticate one completed Wilson trajectory lazily."""

    artifact_path = Path(path)
    validate_artifact(artifact_path, expected_schema=WILSON_TRAJECTORY_SCHEMA)
    with h5py.File(artifact_path, "r") as handle:
        meta = handle["meta"]
        reference = handle["reference"]
        stationary = handle["stationary_state"]
        source = handle["source"]
        restart = handle["restart"]
        observables = handle["observables"]
        for group in (meta, reference, stationary, source, restart, observables):
            assert isinstance(group, h5py.Group)
        if "final_step" not in meta.attrs or "accumulated_source_work_au" not in meta.attrs:
            raise TrajectoryError("Wilson trajectory lacks a completed summary")
        checkpoint_sha: tuple[str, ...] = ()
        if "checkpoints/sha256" in restart:
            checkpoint_sha = tuple(
                str(value) for value in restart["checkpoints/sha256"].asstr()[...]
            )
        return WilsonTrajectory(
            path=artifact_path,
            run_id=read_text(meta["run_id"]),
            simulation_id=read_text(meta["simulation_id_sha256"]),
            reference_fingerprint_sha256=str(reference.attrs["fingerprint_sha256"]),
            stationary_state_fingerprint_sha256=str(stationary.attrs["fingerprint_sha256"]),
            source_fingerprint_sha256=str(source.attrs["source_fingerprint_sha256"]),
            parent_run_id=str(restart.attrs.get("parent_run_id", "")) or None,
            parent_checkpoint_sha256=(
                str(restart.attrs.get("parent_checkpoint_sha256", "")) or None
            ),
            global_step_offset=int(restart.attrs.get("global_step_offset", 0)),
            final_step=int(meta.attrs["final_step"]),
            accumulated_source_work_au=float(meta.attrs["accumulated_source_work_au"]),
            series_names=_series_names(observables),
            checkpoint_sha256=checkpoint_sha,
            complete=True,
            sha256=file_sha256(artifact_path),
        )
