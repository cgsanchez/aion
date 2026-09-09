"""Streaming trajectory writer, immutable reader, and lineage stitching."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from itertools import pairwise
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from aion.config import AtomicUnit, PhysicalDimension, SimulationConfig, dumps_config
from aion.errors import SchemaError, TrajectoryError
from aion.formulations import FormulationSourceSample
from aion.io.schemas import TRAJECTORY_SCHEMA, stamp_artifact, validate_artifact
from aion.io.transaction import write_dataset
from aion.io.util import file_sha256, read_text, write_text
from aion.observables import ObservableDefinition, SamplingLocation
from aion.observables.calculators import InstantaneousDiagnostics
from aion.propagation import SCEMStepDiagnostics

type PathInput = str | PathLike[str]


@dataclass(frozen=True, slots=True)
class ObservableSeries:
    definition: ObservableDefinition
    steps: np.ndarray
    times_au: np.ndarray
    values: np.ndarray


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Validated immutable trajectory handle with lazy stream reads."""

    path: Path
    run_id: str
    simulation_id: str
    reference_fingerprint_sha256: str
    source_fingerprint_sha256: str
    parent_run_id: str | None
    parent_checkpoint_sha256: str | None
    global_step_offset: int
    final_step: int
    accumulated_source_work_au: float
    observable_ids: tuple[str, ...]
    event_steps: tuple[int, ...]
    checkpoint_sha256: tuple[str, ...]
    complete: bool
    sha256: str

    def read_observable(self, definition_id: str) -> ObservableSeries:
        if definition_id not in self.observable_ids:
            raise TrajectoryError(f"trajectory has no observable {definition_id!r}")
        path = "/observables/" + definition_id.replace(".", "/")
        with h5py.File(self.path, "r") as handle:
            group = handle[path]
            assert isinstance(group, h5py.Group)
            definition = _read_definition(group)
            return ObservableSeries(
                definition=definition,
                steps=np.asarray(group["step"][...], dtype=np.int64),
                times_au=np.asarray(group["time_au"][...], dtype=np.float64),
                values=np.asarray(group["values"][...]),
            )


@dataclass(slots=True)
class _AppendStream:
    datasets: dict[str, h5py.Dataset]

    def append(self, values: dict[str, object]) -> None:
        if set(values) != set(self.datasets):
            raise TrajectoryError("append record fields do not match the stream schema")
        for name, dataset in self.datasets.items():
            index = dataset.shape[0]
            dataset.resize(index + 1, axis=0)
            dataset[index] = values[name]


def _create_stream(
    group: h5py.Group,
    fields: dict[str, tuple[tuple[int, ...], object, str, str]],
    *,
    compressed: bool = False,
) -> _AppendStream:
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
    return _AppendStream(datasets)


def _nested_group(root: h5py.Group, relative_path: str) -> h5py.Group:
    group = root
    for part in relative_path.split("/"):
        group = group.require_group(part)
    return group


def _definition_path(definition: ObservableDefinition) -> str:
    return definition.definition_id.replace(".", "/")


def _write_definition_attributes(group: h5py.Group, definition: ObservableDefinition) -> None:
    group.attrs["definition_id"] = definition.definition_id
    group.attrs["originating_formulation"] = definition.originating_formulation
    group.attrs["sampling_location"] = definition.sampling_location.value
    group.attrs["mathematical_definition"] = definition.mathematical_definition
    group.attrs["decomposition_labels_json"] = json.dumps(definition.decomposition_labels)


def _read_definition(group: h5py.Group) -> ObservableDefinition:
    values = group["values"]
    return ObservableDefinition(
        definition_id=str(group.attrs["definition_id"]),
        originating_formulation=str(group.attrs["originating_formulation"]),
        physical_dimension=PhysicalDimension(str(values.attrs["physical_dimension"])),
        unit=AtomicUnit(str(values.attrs["unit"])),
        shape=tuple(int(size) for size in values.shape[1:]),
        sampling_location=SamplingLocation(str(group.attrs["sampling_location"])),
        mathematical_definition=str(group.attrs["mathematical_definition"]),
        decomposition_labels=tuple(json.loads(str(group.attrs["decomposition_labels_json"]))),
    )


class TrajectoryWriter:
    """Append accepted records to one private partial file and publish once."""

    def __init__(
        self,
        directory: PathInput,
        *,
        run_id: str,
        config: SimulationConfig,
        original_toml: str,
        reference_artifact_path: Path,
        source_fingerprint_sha256: str,
        source_definition_json: str,
        event_fingerprint_sha256: str,
        definitions: dict[str, ObservableDefinition],
        provenance: dict[str, object],
        pair_indices: np.ndarray,
        pair_displacements_au: np.ndarray,
        parent_run_id: str | None = None,
        parent_checkpoint_sha256: str | None = None,
        global_step_offset: int = 0,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.target = self.directory / "trajectory.h5"
        self.failed_target = self.directory / "trajectory.failed.h5"
        if self.target.exists() or self.failed_target.exists():
            raise TrajectoryError(f"run directory already contains a trajectory: {self.directory}")
        self.partial = self.directory / f".trajectory.{run_id}.{uuid.uuid4().hex}.partial"
        self.run_id = run_id
        self._definitions = definitions
        self._observable_streams: dict[str, _AppendStream] = {}
        self._streams: dict[str, _AppendStream] = {}
        self._closed = False
        self._handle = h5py.File(self.partial, "x")
        try:
            stamp_artifact(
                self._handle,
                TRAJECTORY_SCHEMA,
                complete=False,
                artifact_id=run_id,
            )
            self._initialize(
                config=config,
                original_toml=original_toml,
                reference_artifact_path=reference_artifact_path,
                source_fingerprint_sha256=source_fingerprint_sha256,
                source_definition_json=source_definition_json,
                event_fingerprint_sha256=event_fingerprint_sha256,
                provenance=provenance,
                pair_indices=pair_indices,
                pair_displacements_au=pair_displacements_au,
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
        config: SimulationConfig,
        original_toml: str,
        reference_artifact_path: Path,
        source_fingerprint_sha256: str,
        source_definition_json: str,
        event_fingerprint_sha256: str,
        provenance: dict[str, object],
        pair_indices: np.ndarray,
        pair_displacements_au: np.ndarray,
        parent_run_id: str | None,
        parent_checkpoint_sha256: str | None,
        global_step_offset: int,
    ) -> None:
        meta = self._handle["meta"]
        configuration = self._handle["configuration"]
        reference = self._handle["reference"]
        time = self._handle["time"]
        source = self._handle["source"]
        restart = self._handle["restart"]
        for group in (meta, configuration, reference, time, source, restart):
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
            json.dumps(provenance, sort_keys=True, separators=(",", ":")),
            physical_dimension="execution_provenance",
        )
        write_text(
            meta,
            "storage_policy_json",
            json.dumps(
                {
                    "appendable_streams": "whole_record_chunks",
                    "dense_arrays": "gzip_level_4_shuffle",
                    "scalar_vector_streams": "uncompressed",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            physical_dimension="storage_policy",
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
        write_dataset(
            reference,
            "pair_indices",
            pair_indices,
            unit="1",
            physical_dimension="atom_index_pair",
        )
        write_dataset(
            reference,
            "pair_displacements_au",
            pair_displacements_au,
            unit="bohr",
            physical_dimension="length",
        )
        grid = config.propagation.time_grid
        time.attrs["start_au"] = grid.start_au
        time.attrs["step_au"] = grid.step_au
        time.attrs["intervals"] = grid.intervals
        endpoint = grid.start_au + np.arange(grid.state_count, dtype=np.float64) * grid.step_au
        midpoint = (
            grid.start_au + (np.arange(grid.intervals, dtype=np.float64) + 0.5) * grid.step_au
        )
        write_dataset(
            time,
            "endpoint_au",
            endpoint,
            unit="atomic_unit_of_time",
            physical_dimension="time",
        )
        write_dataset(
            time,
            "midpoint_au",
            midpoint,
            unit="atomic_unit_of_time",
            physical_dimension="time",
        )
        source.attrs["source_fingerprint_sha256"] = source_fingerprint_sha256
        source.attrs["event_fingerprint_sha256"] = event_fingerprint_sha256
        write_text(
            source,
            "definition_json",
            source_definition_json,
            physical_dimension="source_definition",
        )
        restart.attrs["parent_run_id"] = parent_run_id or ""
        restart.attrs["parent_checkpoint_sha256"] = parent_checkpoint_sha256 or ""
        restart.attrs["global_step_offset"] = np.int64(global_step_offset)
        restart.attrs["boundary_authenticated"] = np.uint8(parent_run_id is None)

    def append_observable(
        self,
        field_name: str,
        *,
        step: int,
        time_au: float,
        value: np.ndarray,
    ) -> None:
        definition = self._definitions[field_name]
        if field_name not in self._observable_streams:
            root = self._handle["observables"]
            assert isinstance(root, h5py.Group)
            group = _nested_group(root, _definition_path(definition))
            _write_definition_attributes(group, definition)
            self._observable_streams[field_name] = _create_stream(
                group,
                {
                    "step": ((), np.int64, "1", "time_step_index"),
                    "time_au": ((), np.float64, "atomic_unit_of_time", "time"),
                    "values": (
                        definition.shape,
                        value.dtype,
                        definition.unit.value,
                        definition.physical_dimension.value,
                    ),
                },
            )
        self._observable_streams[field_name].append(
            {"step": np.int64(step), "time_au": np.float64(time_au), "values": value}
        )

    def append_source(
        self,
        location: SamplingLocation,
        *,
        step: int,
        sample: FormulationSourceSample,
        host: Any,
    ) -> None:
        name = f"source.{location.value}"
        if name not in self._streams:
            root = self._handle["source"]
            assert isinstance(root, h5py.Group)
            group = root.create_group(location.value)
            natom = sample.node_scalar_potential.shape[0]
            npair = sample.pair_link.shape[0]
            fields: dict[str, tuple[tuple[int, ...], object, str, str]] = {
                "step": ((), np.int64, "1", "time_step_index"),
                "time_au": ((), np.float64, "atomic_unit_of_time", "time"),
                "electric_field": (
                    (3,),
                    np.float64,
                    "atomic_unit_of_electric_field",
                    "electric_field",
                ),
                "electric_field_dot": (
                    (3,),
                    np.float64,
                    "atomic_unit_of_electric_field_per_time",
                    "electric_field_time_derivative",
                ),
                "vector_potential_reduced": (
                    (3,),
                    np.float64,
                    "atomic_unit_of_reduced_vector_potential",
                    "vector_potential_reduced",
                ),
                "vector_potential_reduced_dot": (
                    (3,),
                    np.float64,
                    "atomic_unit_of_electric_field",
                    "vector_potential_reduced_time_derivative",
                ),
                "node_scalar_potential": (
                    (natom,),
                    np.float64,
                    "hartree_per_elementary_charge",
                    "electric_scalar_potential",
                ),
                "pair_link": (
                    (npair,),
                    np.float64,
                    "atomic_unit_of_action_per_elementary_charge",
                    "oriented_vector_potential_line_integral",
                ),
                "pair_link_dot": (
                    (npair,),
                    np.float64,
                    "hartree_per_elementary_charge",
                    "oriented_link_time_derivative",
                ),
                "pair_electromotive_potential": (
                    (npair,),
                    np.float64,
                    "hartree_per_elementary_charge",
                    "pair_electromotive_potential",
                ),
            }
            self._streams[name] = _create_stream(group, fields)
        self._streams[name].append(
            {
                "step": np.int64(step),
                "time_au": np.float64(sample.time_au),
                "electric_field": host(sample.electric_field),
                "electric_field_dot": host(sample.electric_field_dot),
                "vector_potential_reduced": host(sample.vector_potential_reduced),
                "vector_potential_reduced_dot": host(sample.vector_potential_reduced_dot),
                "node_scalar_potential": host(sample.node_scalar_potential),
                "pair_link": host(sample.pair_link),
                "pair_link_dot": host(sample.pair_link_dot),
                "pair_electromotive_potential": host(sample.pair_electromotive_potential),
            }
        )

    def append_work_interval(
        self,
        *,
        step: int,
        time_au: float,
        rate_au: float,
        increment_au: float,
        accumulated_au: float,
    ) -> None:
        name = "source.work"
        if name not in self._streams:
            root = self._handle["source"]
            assert isinstance(root, h5py.Group)
            self._streams[name] = _create_stream(
                root.create_group("work"),
                {
                    "step": ((), np.int64, "1", "interval_index"),
                    "time_au": ((), np.float64, "atomic_unit_of_time", "time"),
                    "rate_au": (
                        (),
                        np.float64,
                        "hartree_per_atomic_unit_of_time",
                        "power",
                    ),
                    "increment_au": ((), np.float64, "hartree", "energy"),
                    "accumulated_au": ((), np.float64, "hartree", "energy"),
                },
            )
        self._streams[name].append(
            {
                "step": np.int64(step),
                "time_au": np.float64(time_au),
                "rate_au": np.float64(rate_au),
                "increment_au": np.float64(increment_au),
                "accumulated_au": np.float64(accumulated_au),
            }
        )

    def append_scem_diagnostics(
        self,
        *,
        step: int,
        time_au: float,
        diagnostics: SCEMStepDiagnostics,
    ) -> None:
        name = "diagnostics.scem"
        if name not in self._streams:
            root = self._handle["diagnostics"]
            assert isinstance(root, h5py.Group)
            fields: dict[str, tuple[tuple[int, ...], object, str, str]] = {
                "step": ((), np.int64, "1", "interval_index"),
                "time_au": ((), np.float64, "atomic_unit_of_time", "time"),
                "iterations": ((), np.int64, "1", "iteration_count"),
                "density_residual": ((), np.float64, "1", "normalized_residual"),
                "hamiltonian_residual": ((), np.float64, "1", "normalized_residual"),
                "final_damping": ((), np.float64, "1", "mixing_fraction"),
                "predictor_present": ((), np.uint8, "1", "boolean"),
                "maximum_correction_norm": ((), np.float64, "1", "matrix_norm"),
                "initial_metric_residual": ((), np.float64, "1", "normalized_residual"),
                "final_metric_residual": ((), np.float64, "1", "normalized_residual"),
                "maximum_hermitian_cleanup_norm": (
                    (),
                    np.float64,
                    "1",
                    "normalized_residual",
                ),
            }
            for prefix in ("predictor", "half_step", "full_step"):
                for suffix in (
                    "raw_metric_residual",
                    "corrected_metric_residual",
                    "correction_norm",
                ):
                    fields[f"{prefix}_{suffix}"] = (
                        (),
                        np.float64,
                        "1",
                        "normalized_residual" if "residual" in suffix else "matrix_norm",
                    )
            self._streams[name] = _create_stream(root.create_group("scem"), fields)
        predictor = diagnostics.predictor_link
        values: dict[str, object] = {
            "step": np.int64(step),
            "time_au": np.float64(time_au),
            "iterations": np.int64(diagnostics.iterations),
            "density_residual": np.float64(diagnostics.density_residual),
            "hamiltonian_residual": np.float64(diagnostics.hamiltonian_residual),
            "final_damping": np.float64(diagnostics.final_damping),
            "predictor_present": np.uint8(predictor is not None),
            "maximum_correction_norm": np.float64(diagnostics.maximum_correction_norm),
            "initial_metric_residual": np.float64(diagnostics.initial_metric_residual),
            "final_metric_residual": np.float64(diagnostics.final_metric_residual),
            "maximum_hermitian_cleanup_norm": np.float64(
                diagnostics.maximum_hermitian_cleanup_norm
            ),
        }
        for prefix, link in (
            ("predictor", predictor),
            ("half_step", diagnostics.half_step_link),
            ("full_step", diagnostics.full_step_link),
        ):
            values[f"{prefix}_raw_metric_residual"] = np.float64(
                0.0 if link is None else link.raw_metric_residual
            )
            values[f"{prefix}_corrected_metric_residual"] = np.float64(
                0.0 if link is None else link.corrected_metric_residual
            )
            values[f"{prefix}_correction_norm"] = np.float64(
                0.0 if link is None else link.correction_norm
            )
        self._streams[name].append(values)

    def append_instantaneous_diagnostics(
        self,
        *,
        step: int,
        time_au: float,
        diagnostics: InstantaneousDiagnostics,
        host: Any,
    ) -> None:
        name = "diagnostics.instantaneous"
        if name not in self._streams:
            root = self._handle["diagnostics"]
            assert isinstance(root, h5py.Group)
            fields: dict[str, tuple[tuple[int, ...], object, str, str]] = {
                "step": ((), np.int64, "1", "time_step_index"),
                "time_au": ((), np.float64, "atomic_unit_of_time", "time"),
                "electron_count": ((), np.float64, "electron", "particle_count"),
                "density_hermiticity_residual": (
                    (),
                    np.float64,
                    "1",
                    "normalized_residual",
                ),
                "metric_compatibility_residual": (
                    (),
                    np.float64,
                    "1",
                    "normalized_residual",
                ),
                "source_current_dipole_derivative_residual": (
                    (3,),
                    np.float64,
                    "electron_per_atomic_unit_of_time_times_bohr",
                    "electric_current",
                ),
            }
            if diagnostics.continuity_residual is not None:
                fields["continuity_residual"] = (
                    tuple(diagnostics.continuity_residual.shape),
                    np.float64,
                    "elementary_charge_per_atomic_unit_of_time",
                    "electric_charge_flow_rate",
                )
            self._streams[name] = _create_stream(root.create_group("instantaneous"), fields)
        values: dict[str, object] = {
            "step": np.int64(step),
            "time_au": np.float64(time_au),
            "electron_count": np.float64(np.asarray(host(diagnostics.electron_count))),
            "density_hermiticity_residual": np.float64(
                np.asarray(host(diagnostics.density_hermiticity_residual))
            ),
            "metric_compatibility_residual": np.float64(
                np.asarray(host(diagnostics.metric_compatibility_residual))
            ),
            "source_current_dipole_derivative_residual": host(
                diagnostics.source_current_dipole_derivative_residual
            ),
        }
        if diagnostics.continuity_residual is not None:
            values["continuity_residual"] = host(diagnostics.continuity_residual)
        self._streams[name].append(values)

    def append_midpoint_physics(
        self,
        *,
        step: int,
        time_au: float,
        matter_rate_au: float,
        generator_rate_au: float | None,
        source_rate_au: float,
        ward_residual_au: float,
    ) -> None:
        name = "diagnostics.midpoint_physics"
        if name not in self._streams:
            root = self._handle["diagnostics"]
            assert isinstance(root, h5py.Group)
            self._streams[name] = _create_stream(
                root.create_group("midpoint_physics"),
                {
                    "step": ((), np.int64, "1", "interval_index"),
                    "time_au": ((), np.float64, "atomic_unit_of_time", "time"),
                    "matter_rate_au": (
                        (),
                        np.float64,
                        "hartree_per_atomic_unit_of_time",
                        "power",
                    ),
                    "generator_rate_present": ((), np.uint8, "1", "boolean"),
                    "generator_rate_au": (
                        (),
                        np.float64,
                        "hartree_per_atomic_unit_of_time",
                        "power",
                    ),
                    "source_rate_au": (
                        (),
                        np.float64,
                        "hartree_per_atomic_unit_of_time",
                        "power",
                    ),
                    "ward_residual_au": (
                        (),
                        np.float64,
                        "hartree_per_atomic_unit_of_time",
                        "power",
                    ),
                },
            )
        self._streams[name].append(
            {
                "step": np.int64(step),
                "time_au": np.float64(time_au),
                "matter_rate_au": np.float64(matter_rate_au),
                "generator_rate_present": np.uint8(generator_rate_au is not None),
                "generator_rate_au": np.float64(
                    0.0 if generator_rate_au is None else generator_rate_au
                ),
                "source_rate_au": np.float64(source_rate_au),
                "ward_residual_au": np.float64(ward_residual_au),
            }
        )

    def append_matrix_snapshot(
        self,
        *,
        step: int,
        time_au: float,
        density: np.ndarray,
    ) -> None:
        name = "observables.matrix_snapshot"
        if name not in self._streams:
            root = self._handle["observables"]
            assert isinstance(root, h5py.Group)
            group = root.require_group("state").create_group("ao_density_snapshot")
            group.attrs["definition_id"] = "state.ao_density_snapshot"
            group.attrs["originating_formulation"] = "state"
            group.attrs["sampling_location"] = SamplingLocation.ENDPOINT.value
            group.attrs["mathematical_definition"] = "lower-index AO density P=C f C^dagger"
            group.attrs["decomposition_labels_json"] = "[]"
            self._streams[name] = _create_stream(
                group,
                {
                    "step": ((), np.int64, "1", "time_step_index"),
                    "time_au": ((), np.float64, "atomic_unit_of_time", "time"),
                    "values": (
                        density.shape,
                        np.complex128,
                        "1",
                        "ao_density_matrix",
                    ),
                },
                compressed=True,
            )
        self._streams[name].append(
            {"step": np.int64(step), "time_au": np.float64(time_au), "values": density}
        )

    def append_work_energy_residual(
        self,
        *,
        step: int,
        time_au: float,
        raw_au: float,
        normalized: float,
    ) -> None:
        name = "diagnostics.work_energy"
        if name not in self._streams:
            root = self._handle["diagnostics"]
            assert isinstance(root, h5py.Group)
            self._streams[name] = _create_stream(
                root.create_group("work_energy"),
                {
                    "step": ((), np.int64, "1", "time_step_index"),
                    "time_au": ((), np.float64, "atomic_unit_of_time", "time"),
                    "raw_au": ((), np.float64, "hartree", "energy"),
                    "normalized": ((), np.float64, "1", "normalized_residual"),
                },
            )
        self._streams[name].append(
            {
                "step": np.int64(step),
                "time_au": np.float64(time_au),
                "raw_au": np.float64(raw_au),
                "normalized": np.float64(normalized),
            }
        )

    def append_warning(self, *, step: int, category: str, message: str) -> None:
        name = "diagnostics.warnings"
        if name not in self._streams:
            root = self._handle["diagnostics"]
            assert isinstance(root, h5py.Group)
            string = h5py.string_dtype(encoding="utf-8")
            self._streams[name] = _create_stream(
                root.create_group("warnings"),
                {
                    "step": ((), np.int64, "1", "time_step_index"),
                    "category": ((), string, "1", "warning_category"),
                    "message": ((), string, "1", "warning_message"),
                },
            )
        self._streams[name].append(
            {"step": np.int64(step), "category": category, "message": message}
        )

    def append_event(
        self,
        *,
        sequence: int,
        event_id: str,
        event_fingerprint_sha256: str,
        step: int,
        time_au: float,
        impulse_au: np.ndarray,
        offset_before_au: np.ndarray,
        offset_after_au: np.ndarray,
        work_increment_au: float,
        coefficients_before: np.ndarray,
        coefficients_after: np.ndarray,
        density_before: np.ndarray,
        density_after: np.ndarray,
        observables_before: dict[str, np.ndarray],
        observables_after: dict[str, np.ndarray],
    ) -> None:
        root = self._handle["events"]
        assert isinstance(root, h5py.Group)
        records = root.require_group("records")
        group = records.create_group(f"{sequence:08d}")
        group.attrs["event_id"] = event_id
        group.attrs["event_fingerprint_sha256"] = event_fingerprint_sha256
        group.attrs["step"] = np.int64(step)
        group.attrs["time_au"] = np.float64(time_au)
        for name, value, unit, dimension in (
            ("impulse_au", impulse_au, "atomic_unit_of_momentum", "electric_field_impulse"),
            (
                "vector_potential_offset_before_au",
                offset_before_au,
                "atomic_unit_of_reduced_vector_potential",
                "vector_potential_reduced",
            ),
            (
                "vector_potential_offset_after_au",
                offset_after_au,
                "atomic_unit_of_reduced_vector_potential",
                "vector_potential_reduced",
            ),
            ("work_increment_au", work_increment_au, "hartree", "energy"),
        ):
            write_dataset(group, name, value, unit=unit, physical_dimension=dimension)
        for side, coefficients, density, observables in (
            ("pre", coefficients_before, density_before, observables_before),
            ("post", coefficients_after, density_after, observables_after),
        ):
            side_group = group.create_group(side)
            write_dataset(
                side_group,
                "coefficients",
                coefficients,
                unit="1",
                physical_dimension="ao_occupied_coefficients",
                compressed=True,
            )
            write_dataset(
                side_group,
                "density",
                density,
                unit="electron",
                physical_dimension="ao_density_matrix",
                compressed=True,
            )
            observable_group = side_group.create_group("observables")
            for field_name, value in observables.items():
                definition = self._definitions[field_name]
                target = _nested_group(observable_group, _definition_path(definition))
                _write_definition_attributes(target, definition)
                write_dataset(
                    target,
                    "values",
                    value,
                    unit=definition.unit.value,
                    physical_dimension=definition.physical_dimension.value,
                )

    def append_checkpoint_link(
        self,
        *,
        step: int,
        time_au: float,
        path: Path,
        sha256: str,
    ) -> None:
        name = "restart.checkpoints"
        if name not in self._streams:
            restart = self._handle["restart"]
            assert isinstance(restart, h5py.Group)
            string = h5py.string_dtype(encoding="utf-8")
            self._streams[name] = _create_stream(
                restart.create_group("checkpoints"),
                {
                    "step": ((), np.int64, "1", "time_step_index"),
                    "time_au": ((), np.float64, "atomic_unit_of_time", "time"),
                    "path": ((), string, "1", "filesystem_path"),
                    "sha256": ((), string, "1", "sha256_digest"),
                },
            )
        self._streams[name].append(
            {
                "step": np.int64(step),
                "time_au": np.float64(time_au),
                "path": str(path),
                "sha256": sha256,
            }
        )

    def authenticate_restart_boundary(self) -> None:
        restart = self._handle["restart"]
        assert isinstance(restart, h5py.Group)
        restart.attrs.modify("boundary_authenticated", np.uint8(1))

    def write_restart_boundary(
        self,
        *,
        step: int,
        time_au: float,
        coefficients: np.ndarray,
        density: np.ndarray,
        accumulated_source_work_au: float,
    ) -> None:
        restart = self._handle["restart"]
        assert isinstance(restart, h5py.Group)
        if "boundary" in restart:
            raise TrajectoryError("restart boundary has already been written")
        group = restart.create_group("boundary")
        write_dataset(
            group,
            "step",
            np.int64(step),
            unit="1",
            physical_dimension="time_step_index",
        )
        write_dataset(
            group,
            "time_au",
            np.float64(time_au),
            unit="atomic_unit_of_time",
            physical_dimension="time",
        )
        write_dataset(
            group,
            "coefficients",
            coefficients,
            unit="1",
            physical_dimension="ao_occupied_coefficients",
            compressed=True,
        )
        write_dataset(
            group,
            "density",
            density,
            unit="electron",
            physical_dimension="ao_density_matrix",
            compressed=True,
        )
        write_dataset(
            group,
            "accumulated_source_work_au",
            np.float64(accumulated_source_work_au),
            unit="hartree",
            physical_dimension="energy",
        )
        self._handle.flush()
        self.authenticate_restart_boundary()

    def set_summary(self, *, final_step: int, accumulated_source_work_au: float) -> None:
        meta = self._handle["meta"]
        assert isinstance(meta, h5py.Group)
        meta.attrs["final_step"] = np.int64(final_step)
        meta.attrs["accumulated_source_work_au"] = np.float64(accumulated_source_work_au)

    def flush(self) -> None:
        self._handle.flush()

    def finalize(self) -> Trajectory:
        if self._closed:
            raise TrajectoryError("trajectory writer is already closed")
        self._handle.flush()
        self._handle.close()
        validate_artifact(self.partial, expected_schema=TRAJECTORY_SCHEMA, require_complete=False)
        with h5py.File(self.partial, "r+") as handle:
            handle.attrs.modify("complete", np.uint8(1))
            handle.flush()
        validate_artifact(self.partial, expected_schema=TRAJECTORY_SCHEMA, require_complete=True)
        self._publish(self.target)
        self._closed = True
        return load_trajectory(self.target)

    def fail(
        self,
        *,
        error_type: str,
        message: str,
        phase: str,
        step: int,
    ) -> Path:
        if self._closed:
            raise TrajectoryError("trajectory writer is already closed")
        meta = self._handle["meta"]
        assert isinstance(meta, h5py.Group)
        failure = meta.create_group("failure")
        write_text(failure, "error_type", error_type, physical_dimension="error_class")
        write_text(failure, "message", message, physical_dimension="error_message")
        write_text(failure, "phase", phase, physical_dimension="run_phase")
        write_dataset(
            failure,
            "step",
            np.int64(step),
            unit="1",
            physical_dimension="time_step_index",
        )
        self._handle.flush()
        self._handle.close()
        validate_artifact(self.partial, expected_schema=TRAJECTORY_SCHEMA, require_complete=False)
        self._publish(self.failed_target)
        self._closed = True
        return self.failed_target

    def _publish(self, target: Path) -> None:
        try:
            os.link(self.partial, target)
        except FileExistsError:
            raise TrajectoryError(f"refusing to overwrite trajectory artifact {target}") from None
        self.partial.unlink()


def _attribute_text(group: h5py.Group, name: str) -> str:
    value = group.attrs.get(name, "")
    if isinstance(value, bytes | np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _observable_ids(handle: h5py.File) -> tuple[str, ...]:
    root = handle["observables"]
    assert isinstance(root, h5py.Group)
    identifiers: list[str] = []

    def visitor(_name: str, item: h5py.Group | h5py.Dataset) -> None:
        if isinstance(item, h5py.Group) and "definition_id" in item.attrs:
            identifiers.append(str(item.attrs["definition_id"]))

    root.visititems(visitor)
    return tuple(sorted(identifiers))


def load_trajectory(path: PathInput, *, allow_incomplete: bool = False) -> Trajectory:
    """Load a trajectory; incomplete failure artifacts require explicit opt-in."""

    artifact_path = Path(path)
    header = validate_artifact(
        artifact_path,
        expected_schema=TRAJECTORY_SCHEMA,
        require_complete=not allow_incomplete,
    )
    with h5py.File(artifact_path, "r") as handle:
        for required in (
            "meta/run_id",
            "meta/simulation_id_sha256",
            "configuration/normalized_toml",
            "reference/artifact_path",
            "time/endpoint_au",
            "time/midpoint_au",
            "source/definition_json",
            "meta/storage_policy_json",
        ):
            if required not in handle:
                raise SchemaError(f"trajectory lacks required object /{required}")
        meta = handle["meta"]
        reference = handle["reference"]
        source = handle["source"]
        restart = handle["restart"]
        assert isinstance(meta, h5py.Group)
        assert isinstance(reference, h5py.Group)
        assert isinstance(source, h5py.Group)
        assert isinstance(restart, h5py.Group)
        run_id = read_text(handle["meta/run_id"])
        if header.artifact_id != run_id:
            raise TrajectoryError("trajectory artifact ID does not match its run ID")
        parent_run = _attribute_text(restart, "parent_run_id") or None
        parent_checkpoint = _attribute_text(restart, "parent_checkpoint_sha256") or None
        if int(restart.attrs.get("boundary_authenticated", 0)) != 1:
            raise TrajectoryError("resumed trajectory boundary was not authenticated")
        if "final_step" not in meta.attrs or "accumulated_source_work_au" not in meta.attrs:
            raise TrajectoryError("completed trajectory lacks its final summary")
        final_step = int(meta.attrs["final_step"])
        accumulated_work = float(meta.attrs["accumulated_source_work_au"])
        intervals = int(handle["time"].attrs["intervals"])
        if not 0 <= final_step <= intervals or not np.isfinite(accumulated_work):
            raise TrajectoryError("completed trajectory has an invalid final summary")
        event_steps: tuple[int, ...] = ()
        if "records" in handle["events"]:
            records = handle["events/records"]
            assert isinstance(records, h5py.Group)
            event_steps = tuple(int(records[name].attrs["step"]) for name in sorted(records.keys()))
        checkpoint_hashes: tuple[str, ...] = ()
        if "checkpoints" in restart:
            checkpoint_hashes = tuple(
                str(value) for value in restart["checkpoints/sha256"].asstr()[...]
            )
        return Trajectory(
            path=artifact_path,
            run_id=run_id,
            simulation_id=read_text(handle["meta/simulation_id_sha256"]),
            reference_fingerprint_sha256=_attribute_text(reference, "fingerprint_sha256"),
            source_fingerprint_sha256=_attribute_text(source, "source_fingerprint_sha256"),
            parent_run_id=parent_run,
            parent_checkpoint_sha256=parent_checkpoint,
            global_step_offset=int(restart.attrs.get("global_step_offset", 0)),
            final_step=final_step,
            accumulated_source_work_au=accumulated_work,
            observable_ids=_observable_ids(handle),
            event_steps=event_steps,
            checkpoint_sha256=checkpoint_hashes,
            complete=header.complete,
            sha256=file_sha256(artifact_path),
        )


def stitch_observable(
    trajectories: tuple[Trajectory, ...],
    definition_id: str,
) -> ObservableSeries:
    """Join one exact stream across authenticated parent/child segments."""

    if not trajectories:
        raise TrajectoryError("at least one trajectory is required for stitching")
    for parent, child in pairwise(trajectories):
        if child.parent_run_id != parent.run_id:
            raise TrajectoryError("trajectory lineage is not a parent/child chain")
        if child.simulation_id != parent.simulation_id:
            raise TrajectoryError("trajectory segments use different simulation identities")
        if child.parent_checkpoint_sha256 not in parent.checkpoint_sha256:
            raise TrajectoryError("child checkpoint checksum is absent from its parent segment")
    series = [trajectory.read_observable(definition_id) for trajectory in trajectories]
    definition = series[0].definition
    for item in series:
        if item.definition != definition:
            raise TrajectoryError("stitched observable definitions differ")
    steps = np.array(series[0].steps, copy=True)
    times = np.array(series[0].times_au, copy=True)
    values = np.array(series[0].values, copy=True)
    for index in range(1, len(series)):
        item = series[index]
        child = trajectories[index]
        boundary = child.global_step_offset
        parent_boundary = np.flatnonzero(steps == boundary)
        if parent_boundary.size != 1:
            raise TrajectoryError("parent stream lacks one unique restart-boundary sample")
        if item.steps.size == 0 or int(item.steps[0]) != boundary:
            raise TrajectoryError("child stream does not begin at its restart boundary")
        position = int(parent_boundary[0])
        if float(item.times_au[0]) != float(times[position]):
            raise TrajectoryError("restart-boundary times differ")
        event_at_boundary = boundary in child.event_steps
        if not event_at_boundary and not np.array_equal(item.values[0], values[position]):
            raise TrajectoryError("shared restart-boundary observable does not authenticate")
        keep = steps < boundary if event_at_boundary else steps <= boundary
        start = 0 if event_at_boundary else 1
        steps = np.concatenate((steps[keep], item.steps[start:]))
        times = np.concatenate((times[keep], item.times_au[start:]))
        values = np.concatenate((values[keep], item.values[start:]))
    if not steps.size:
        return ObservableSeries(
            definition,
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty((0, *definition.shape), dtype=np.float64),
        )
    return ObservableSeries(
        definition,
        steps,
        times,
        values,
    )
