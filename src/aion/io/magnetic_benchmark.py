"""Transactional persistence for static uniform-magnetic benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from aion.config import (
    AtomicUnit,
    BackendConfig,
    BackendKind,
    PhysicalDimension,
    Precision,
)
from aion.electromagnetism import (
    AffineMagneticGauge,
    MagneticGaugeKind,
    UniformMagneticField,
)
from aion.electronic_structure import AOGridKind, AOGridPolicy, LocalPotentialIdentity
from aion.errors import MagneticBenchmarkError, SchemaError
from aion.io.schemas import MAGNETIC_BENCHMARK_SCHEMA, validate_artifact
from aion.io.transaction import publish_hdf5, write_dataset
from aion.io.util import file_sha256, read_text, write_text
from aion.workflows.magnetic_benchmark import (
    MagneticBenchmarkConfig,
    MagneticBenchmarkResult,
    MagneticDiagnostic,
    MagneticMatrixFamily,
    MagneticMatrixRecord,
    MagneticValidationPolicy,
)


def magnetic_benchmark_configuration_json(config: MagneticBenchmarkConfig) -> str:
    """Return the deterministic, human-readable serialization of a config."""

    return json.dumps(config.as_mapping(), sort_keys=True, separators=(",", ":"))


def save_magnetic_benchmark(
    result: MagneticBenchmarkResult,
    path: str | Path,
) -> str:
    """Publish a complete immutable result and return the file SHA-256."""

    if not isinstance(result, MagneticBenchmarkResult):
        raise TypeError("result must be MagneticBenchmarkResult")

    def populate(handle: h5py.File) -> None:
        meta = _group(handle, "meta")
        configuration = _group(handle, "configuration")
        reference = _group(handle, "reference")
        grid = _group(handle, "grid")
        fields = _group(handle, "fields")
        diagnostics = _group(handle, "diagnostics")
        write_text(meta, "result_id", result.result_id, physical_dimension="sha256_digest")
        write_text(
            meta,
            "configuration_id",
            result.config.scientific_id,
            physical_dimension="sha256_digest",
        )
        write_text(
            meta,
            "ao_provenance_json",
            result.ao_provenance_json,
            physical_dimension="execution_provenance",
        )
        write_text(
            configuration,
            "normalized_json",
            magnetic_benchmark_configuration_json(result.config),
            physical_dimension="magnetic_benchmark_configuration",
        )
        write_text(
            reference,
            "fingerprint_sha256",
            result.reference_fingerprint_sha256,
            physical_dimension="sha256_digest",
        )
        write_text(
            grid,
            "fingerprint_sha256",
            result.grid_fingerprint_sha256,
            physical_dimension="sha256_digest",
        )
        write_text(grid, "kind", result.grid_kind, physical_dimension="grid_kind")
        write_text(grid, "pruning", result.grid_pruning, physical_dimension="grid_pruning")
        write_dataset(
            grid,
            "level",
            np.int64(result.grid_level),
            unit="1",
            physical_dimension="grid_level",
        )
        write_dataset(
            grid,
            "npoints",
            np.int64(result.grid_npoints),
            unit="1",
            physical_dimension="grid_point_count",
        )
        for record in result.matrices:
            parent, name = _record_parent(fields, record.path)
            dataset = write_dataset(
                parent,
                name,
                record.values,
                unit=record.unit,
                physical_dimension=record.physical_dimension,
                compressed=True,
            )
            dataset.attrs["definition"] = record.definition
        write_text(
            diagnostics,
            "values_json",
            _diagnostics_json(result.diagnostics),
            physical_dimension="magnetic_benchmark_diagnostics",
        )

    publish_hdf5(
        path,
        MAGNETIC_BENCHMARK_SCHEMA,
        artifact_id=result.result_id,
        populate=populate,
    )
    return file_sha256(path)


def load_magnetic_benchmark(path: str | Path) -> MagneticBenchmarkResult:
    """Load and content-authenticate a magnetic benchmark artifact."""

    target = Path(path)
    header = validate_artifact(target, expected_schema=MAGNETIC_BENCHMARK_SCHEMA)
    try:
        with h5py.File(target, "r") as handle:
            config = _config_from_json(read_text(handle["configuration/normalized_json"]))
            if read_text(handle["meta/configuration_id"]) != config.scientific_id:
                raise MagneticBenchmarkError(
                    "magnetic benchmark configuration fingerprint mismatch"
                )
            result = MagneticBenchmarkResult(
                config=config,
                reference_fingerprint_sha256=read_text(
                    handle["reference/fingerprint_sha256"]
                ),
                grid_fingerprint_sha256=read_text(handle["grid/fingerprint_sha256"]),
                grid_kind=read_text(handle["grid/kind"]),
                grid_level=int(handle["grid/level"][()]),
                grid_pruning=read_text(handle["grid/pruning"]),
                grid_npoints=int(handle["grid/npoints"][()]),
                ao_provenance_json=read_text(handle["meta/ao_provenance_json"]),
                matrices=_read_matrix_records(_group(handle, "fields")),
                diagnostics=_diagnostics_from_json(
                    read_text(handle["diagnostics/values_json"])
                ),
            )
            stored_result_id = read_text(handle["meta/result_id"])
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, MagneticBenchmarkError | SchemaError):
            raise
        raise MagneticBenchmarkError("magnetic benchmark artifact content is invalid") from exc
    if stored_result_id != result.result_id:
        raise MagneticBenchmarkError("magnetic benchmark numerical content fingerprint mismatch")
    if header.artifact_id != result.result_id:
        raise MagneticBenchmarkError("magnetic benchmark artifact fingerprint mismatch")
    return result


def _config_from_json(text: str) -> MagneticBenchmarkConfig:
    try:
        data = json.loads(text)
        if not isinstance(data, dict) or data.get("schema") != (
            "aion.magnetic-benchmark-config"
        ):
            raise MagneticBenchmarkError("expected magnetic benchmark configuration")
        if data.get("version") != "1.0.0":
            raise MagneticBenchmarkError("unsupported magnetic benchmark configuration version")
        fields = tuple(
            UniformMagneticField(tuple(value)) for value in data["magnetic_fields_au"]
        )
        gauge_data = data["gauges"]
        if not isinstance(gauge_data, list) or len(gauge_data) != len(fields):
            raise MagneticBenchmarkError("stored fields and gauges do not match")
        gauges = tuple(
            AffineMagneticGauge(
                field=magnetic_field,
                kind=MagneticGaugeKind(value["kind"]),
                origin_au=tuple(value["origin_au"]),
                landau_axis=(
                    None
                    if value["landau_axis"] is None
                    else tuple(value["landau_axis"])
                ),
            )
            for magnetic_field, value in zip(fields, gauge_data, strict=True)
        )
        backend_data = data["backend"]
        backend = BackendConfig(
            kind=BackendKind(backend_data["kind"]),
            precision=Precision(backend_data["precision"]),
            device_index=backend_data.get("device_index"),
        )
        grid_data = data["grid_policy"]
        grid_policy = AOGridPolicy(
            kind=AOGridKind(grid_data["kind"]),
            level=grid_data["level"],
        )
        identities = tuple(_identity_from_mapping(value) for value in data["local_potentials"])
        validation_data = data["validation"]
        config = MagneticBenchmarkConfig(
            magnetic_fields=fields,
            gauges=gauges,
            backend=backend,
            grid_policy=grid_policy,
            families=tuple(MagneticMatrixFamily(value) for value in data["families"]),
            local_potential_identities=identities,
            block_size=data["block_size"],
            memory_budget_bytes=data["memory_budget_bytes"],
            charge=data["charge"],
            mass=data["mass"],
            hbar=data["hbar"],
            include_direct_oracle=data["include_direct_oracle"],
            validation=MagneticValidationPolicy(
                hermiticity_tolerance=validation_data["hermiticity_tolerance"],
                direct_oracle_tolerance=validation_data["direct_oracle_tolerance"],
                adjoint_pair_tolerance=validation_data["adjoint_pair_tolerance"],
                e1_closure_tolerance=validation_data["e1_closure_tolerance"],
                fail_on_violation=validation_data["fail_on_violation"],
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, MagneticBenchmarkError):
            raise
        raise MagneticBenchmarkError("magnetic benchmark configuration JSON is invalid") from exc
    if config.as_mapping() != data:
        raise MagneticBenchmarkError("stored magnetic benchmark configuration is inconsistent")
    return config


def _identity_from_mapping(value: dict[str, Any]) -> LocalPotentialIdentity:
    identity = LocalPotentialIdentity(
        name=value["name"],
        version=value["version"],
        provenance=tuple(tuple(item) for item in value["provenance"]),
        unit=AtomicUnit(value["unit"]),
        physical_dimension=PhysicalDimension(value["physical_dimension"]),
    )
    if value.get("fingerprint_sha256") != identity.fingerprint_sha256:
        raise MagneticBenchmarkError("local-potential identity fingerprint mismatch")
    return identity


def _read_matrix_records(group: h5py.Group) -> tuple[MagneticMatrixRecord, ...]:
    records: list[MagneticMatrixRecord] = []

    def visitor(name: str, value: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(value, h5py.Dataset):
            return
        definition = _text_attribute(value, "definition")
        records.append(
            MagneticMatrixRecord(
                path=name,
                values=value[...],
                unit=_text_attribute(value, "unit"),
                physical_dimension=_text_attribute(value, "physical_dimension"),
                definition=definition,
            )
        )

    group.visititems(visitor)
    return tuple(sorted(records, key=lambda value: value.path))


def _diagnostics_json(values: tuple[MagneticDiagnostic, ...]) -> str:
    return json.dumps(
        [
            {
                "name": value.name,
                "value": value.value,
                "tolerance": value.tolerance,
                "unit": value.unit,
                "physical_dimension": value.physical_dimension,
            }
            for value in values
        ],
        sort_keys=True,
        separators=(",", ":"),
    )


def _diagnostics_from_json(text: str) -> tuple[MagneticDiagnostic, ...]:
    try:
        data = json.loads(text)
        if not isinstance(data, list):
            raise MagneticBenchmarkError("stored magnetic diagnostics must be a list")
        values = tuple(
            MagneticDiagnostic(
                name=value["name"],
                value=value["value"],
                tolerance=value["tolerance"],
                unit=value["unit"],
                physical_dimension=value["physical_dimension"],
            )
            for value in data
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, MagneticBenchmarkError):
            raise
        raise MagneticBenchmarkError("stored magnetic diagnostics are invalid") from exc
    if _diagnostics_json(values) != json.dumps(data, sort_keys=True, separators=(",", ":")):
        raise MagneticBenchmarkError("stored magnetic diagnostics are inconsistent")
    return values


def _record_parent(root: h5py.Group, path: str) -> tuple[h5py.Group, str]:
    parts = path.split("/")
    parent = root
    for part in parts[:-1]:
        parent = parent.require_group(part)
    return parent, parts[-1]


def _group(handle: h5py.File | h5py.Group, name: str) -> h5py.Group:
    value = handle[name]
    if not isinstance(value, h5py.Group):
        raise SchemaError(f"{value.name} must be an HDF5 group")
    return value


def _text_attribute(dataset: h5py.Dataset, name: str) -> str:
    if name not in dataset.attrs:
        raise SchemaError(f"dataset {dataset.name} lacks required {name!r} metadata")
    value = dataset.attrs[name]
    result = bytes(value).decode("utf-8") if isinstance(value, bytes | np.bytes_) else str(value)
    if not result:
        raise SchemaError(f"dataset {dataset.name} attribute {name!r} cannot be empty")
    return result
