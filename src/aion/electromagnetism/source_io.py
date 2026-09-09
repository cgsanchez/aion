"""Persistence of authoritative fixed-grid compiled source histories."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import h5py
import numpy as np

from aion.config import FixedTimeGrid
from aion.electromagnetism.compiled import (
    CompiledUniformSource,
    PhysicalSourceSeries,
    ProjectedGaugeSeries,
)
from aion.errors import SourceCompilationError
from aion.io import SOURCE_HISTORY_SCHEMA, publish_hdf5, validate_artifact, write_dataset

type PathInput = str | PathLike[str]

_REQUIRED_SOURCE_PATHS = (
    "meta/compiled_source_kind",
    "configuration/definition_json",
    "time/endpoint_au",
    "time/midpoint_au",
    "source/atom_coordinates_au",
    "source/electromagnetic_origin_au",
    "source/pair_indices",
    *tuple(
        f"source/{location}/{section}/{dataset}"
        for location in ("endpoint", "midpoint")
        for section, datasets in (
            (
                "physical",
                (
                    "electric_field",
                    "vector_potential_reduced",
                    "vector_potential_reduced_dot",
                ),
            ),
            (
                "length_gauge",
                (
                    "node_scalar_potential",
                    "pair_link",
                    "pair_link_dot",
                    "pair_electromotive_potential",
                ),
            ),
            (
                "velocity_gauge",
                (
                    "node_scalar_potential",
                    "pair_link",
                    "pair_link_dot",
                    "pair_electromotive_potential",
                ),
            ),
        )
        for dataset in datasets
    ),
)


def _write_text(group: h5py.Group, name: str, value: str, dimension: str) -> None:
    write_dataset(
        group,
        name,
        np.bytes_(value),
        unit="1",
        physical_dimension=dimension,
    )


def _write_physical(group: h5py.Group, series: PhysicalSourceSeries) -> None:
    for name, value, unit, dimension in (
        (
            "electric_field",
            series.electric_field,
            "atomic_unit_of_electric_field",
            "electric_field",
        ),
        (
            "vector_potential_reduced",
            series.vector_potential_reduced,
            "atomic_unit_of_reduced_vector_potential",
            "vector_potential_reduced",
        ),
        (
            "vector_potential_reduced_dot",
            series.vector_potential_reduced_dot,
            "atomic_unit_of_electric_field",
            "vector_potential_reduced_time_derivative",
        ),
    ):
        write_dataset(group, name, value, unit=unit, physical_dimension=dimension)


def _write_gauge(group: h5py.Group, series: ProjectedGaugeSeries) -> None:
    for name, value, unit, dimension in (
        (
            "node_scalar_potential",
            series.node_scalar_potential,
            "hartree_per_elementary_charge",
            "electric_scalar_potential",
        ),
        (
            "pair_link",
            series.pair_link,
            "atomic_unit_of_action_per_elementary_charge",
            "oriented_vector_potential_line_integral",
        ),
        (
            "pair_link_dot",
            series.pair_link_dot,
            "hartree_per_elementary_charge",
            "oriented_link_time_derivative",
        ),
        (
            "pair_electromotive_potential",
            series.pair_electromotive_potential,
            "hartree_per_elementary_charge",
            "pair_electromotive_potential",
        ),
    ):
        write_dataset(group, name, value, unit=unit, physical_dimension=dimension)


def save_compiled_source(source: CompiledUniformSource, path: PathInput) -> None:
    """Transactionally publish a compiled source history without overwrite."""

    def populate(handle: h5py.File) -> None:
        meta = handle["meta"]
        configuration = handle["configuration"]
        time = handle["time"]
        source_group = handle["source"]
        assert isinstance(meta, h5py.Group)
        assert isinstance(configuration, h5py.Group)
        assert isinstance(time, h5py.Group)
        assert isinstance(source_group, h5py.Group)
        _write_text(meta, "compiled_source_kind", "uniform", "source_kind")
        _write_text(
            configuration,
            "definition_json",
            source.definition_json,
            "source_definition",
        )
        time.attrs["start_au"] = source.time_grid.start_au
        time.attrs["step_au"] = source.time_grid.step_au
        time.attrs["intervals"] = source.time_grid.intervals
        write_dataset(
            time,
            "endpoint_au",
            source.endpoint.times_au,
            unit="atomic_unit_of_time",
            physical_dimension="time",
        )
        write_dataset(
            time,
            "midpoint_au",
            source.midpoint.times_au,
            unit="atomic_unit_of_time",
            physical_dimension="time",
        )
        source_group.attrs["source_fingerprint_sha256"] = source.fingerprint_sha256
        write_dataset(
            source_group,
            "atom_coordinates_au",
            source.atom_coordinates_au,
            unit="bohr",
            physical_dimension="length",
        )
        write_dataset(
            source_group,
            "electromagnetic_origin_au",
            source.electromagnetic_origin_au,
            unit="bohr",
            physical_dimension="length",
        )
        write_dataset(
            source_group,
            "pair_indices",
            source.pair_indices,
            unit="1",
            physical_dimension="atom_index_pair",
        )
        for location, physical, length, velocity in (
            (
                "endpoint",
                source.endpoint,
                source.length_endpoint,
                source.velocity_endpoint,
            ),
            (
                "midpoint",
                source.midpoint,
                source.length_midpoint,
                source.velocity_midpoint,
            ),
        ):
            location_group = source_group.create_group(location)
            _write_physical(location_group.create_group("physical"), physical)
            _write_gauge(location_group.create_group("length_gauge"), length)
            _write_gauge(location_group.create_group("velocity_gauge"), velocity)

    publish_hdf5(
        Path(path),
        SOURCE_HISTORY_SCHEMA,
        artifact_id=source.fingerprint_sha256,
        populate=populate,
    )


def _physical(group: h5py.Group, times: np.ndarray) -> PhysicalSourceSeries:
    return PhysicalSourceSeries(
        times_au=times,
        electric_field=group["electric_field"][...],
        vector_potential_reduced=group["vector_potential_reduced"][...],
        vector_potential_reduced_dot=group["vector_potential_reduced_dot"][...],
    )


def _gauge(group: h5py.Group) -> ProjectedGaugeSeries:
    return ProjectedGaugeSeries(
        node_scalar_potential=group["node_scalar_potential"][...],
        pair_link=group["pair_link"][...],
        pair_link_dot=group["pair_link_dot"][...],
        pair_electromotive_potential=group["pair_electromotive_potential"][...],
    )


def load_compiled_source(path: PathInput) -> CompiledUniformSource:
    """Load and authenticate an immutable compiled source artifact."""

    artifact_path = Path(path)
    header = validate_artifact(artifact_path, expected_schema=SOURCE_HISTORY_SCHEMA)
    with h5py.File(artifact_path, "r") as handle:
        missing = [name for name in _REQUIRED_SOURCE_PATHS if name not in handle]
        if missing:
            raise SourceCompilationError(
                "compiled-source artifact lacks required object(s): " + ", ".join(missing)
            )
        kind_value = handle["meta/compiled_source_kind"][()]
        kind = (
            bytes(kind_value).decode("utf-8")
            if isinstance(kind_value, bytes | np.bytes_)
            else str(kind_value)
        )
        if kind != "uniform":
            raise SourceCompilationError(f"compiled source kind is {kind!r}; expected 'uniform'")
        time = handle["time"]
        grid = FixedTimeGrid(
            start_au=float(time.attrs["start_au"]),
            step_au=float(time.attrs["step_au"]),
            intervals=int(time.attrs["intervals"]),
        )
        endpoint_times = time["endpoint_au"][...]
        midpoint_times = time["midpoint_au"][...]
        root = handle["source"]
        endpoint = root["endpoint"]
        midpoint = root["midpoint"]
        definition_value = handle["configuration/definition_json"][()]
        definition_text = (
            bytes(definition_value).decode("utf-8")
            if isinstance(definition_value, bytes | np.bytes_)
            else str(definition_value)
        )
        compiled = CompiledUniformSource(
            time_grid=grid,
            atom_coordinates_au=root["atom_coordinates_au"][...],
            electromagnetic_origin_au=root["electromagnetic_origin_au"][...],
            pair_indices=root["pair_indices"][...],
            endpoint=_physical(endpoint["physical"], endpoint_times),
            midpoint=_physical(midpoint["physical"], midpoint_times),
            length_endpoint=_gauge(endpoint["length_gauge"]),
            length_midpoint=_gauge(midpoint["length_gauge"]),
            velocity_endpoint=_gauge(endpoint["velocity_gauge"]),
            velocity_midpoint=_gauge(midpoint["velocity_gauge"]),
            definition_json=definition_text,
        )
        stored_fingerprint = root.attrs["source_fingerprint_sha256"]
        stored_fingerprint = (
            bytes(stored_fingerprint).decode("utf-8")
            if isinstance(stored_fingerprint, bytes | np.bytes_)
            else str(stored_fingerprint)
        )
    if compiled.fingerprint_sha256 != stored_fingerprint:
        raise SourceCompilationError("compiled-source content fingerprint mismatch")
    if header.artifact_id != compiled.fingerprint_sha256:
        raise SourceCompilationError("artifact ID does not authenticate the compiled source")
    return compiled
