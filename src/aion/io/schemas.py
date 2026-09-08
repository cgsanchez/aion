"""Version-one HDF5 schema declarations and strict structural validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import h5py
import numpy as np

from aion.errors import (
    IncompatibleSchemaVersionError,
    IncompleteArtifactError,
    SchemaError,
)
from aion.io.versions import SchemaVersion


class ArtifactKind(StrEnum):
    REFERENCE = "reference"
    TRAJECTORY = "trajectory"
    SOURCE_HISTORY = "source_history"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True, slots=True)
class ArtifactSchema:
    name: str
    version: SchemaVersion
    kind: ArtifactKind
    required_groups: tuple[str, ...]


REFERENCE_SCHEMA = ArtifactSchema(
    name="aion.reference",
    version=SchemaVersion(1, 0, 0),
    kind=ArtifactKind.REFERENCE,
    required_groups=("meta", "configuration", "reference"),
)

TRAJECTORY_SCHEMA = ArtifactSchema(
    name="aion.trajectory",
    version=SchemaVersion(1, 0, 0),
    kind=ArtifactKind.TRAJECTORY,
    required_groups=(
        "meta",
        "configuration",
        "reference",
        "time",
        "source",
        "observables",
        "diagnostics",
        "events",
        "restart",
    ),
)

SOURCE_HISTORY_SCHEMA = ArtifactSchema(
    name="aion.source-history",
    version=SchemaVersion(1, 0, 0),
    kind=ArtifactKind.SOURCE_HISTORY,
    required_groups=("meta", "configuration", "time", "source"),
)

CHECKPOINT_SCHEMA = ArtifactSchema(
    name="aion.checkpoint",
    version=SchemaVersion(1, 0, 0),
    kind=ArtifactKind.CHECKPOINT,
    required_groups=(
        "meta",
        "configuration",
        "reference",
        "time",
        "source",
        "state",
        "events",
        "restart",
    ),
)

SCHEMAS: dict[str, ArtifactSchema] = {
    schema.name: schema
    for schema in (
        REFERENCE_SCHEMA,
        TRAJECTORY_SCHEMA,
        SOURCE_HISTORY_SCHEMA,
        CHECKPOINT_SCHEMA,
    )
}


@dataclass(frozen=True, slots=True)
class ArtifactHeader:
    schema: ArtifactSchema
    artifact_version: SchemaVersion
    complete: bool
    artifact_id: str


def stamp_artifact(
    handle: h5py.File,
    schema: ArtifactSchema,
    *,
    complete: bool,
    artifact_id: str,
) -> None:
    """Stamp an open new HDF5 file and create its required empty groups.

    This helper establishes fixtures and future writer foundations.  It does
    not publish, overwrite, rename, or otherwise implement transactional I/O.
    """

    if handle.keys() or handle.attrs:
        raise SchemaError("artifact stamping requires a file with no existing groups")
    handle.attrs["schema_name"] = schema.name
    handle.attrs["schema_version"] = str(schema.version)
    handle.attrs["artifact_kind"] = schema.kind.value
    handle.attrs["complete"] = np.uint8(complete)
    handle.attrs["unit_system"] = "atomic"
    handle.attrs["index_base"] = np.uint8(0)
    handle.attrs["artifact_id"] = artifact_id
    for name in schema.required_groups:
        handle.create_group(name)


def validate_artifact(
    path: str | Path,
    *,
    expected_schema: ArtifactSchema | None = None,
    require_complete: bool = True,
) -> ArtifactHeader:
    """Validate a structured HDF5 artifact and return its typed header."""

    artifact_path = Path(path)
    try:
        with h5py.File(artifact_path, "r") as handle:
            return validate_open_artifact(
                handle,
                expected_schema=expected_schema,
                require_complete=require_complete,
            )
    except OSError as exc:
        raise SchemaError(f"cannot open HDF5 artifact {artifact_path}") from exc


def validate_open_artifact(
    handle: h5py.File,
    *,
    expected_schema: ArtifactSchema | None = None,
    require_complete: bool = True,
) -> ArtifactHeader:
    """Validate an already-open HDF5 artifact without taking ownership of it."""

    schema_name = _text_attribute(handle, "schema_name")
    try:
        schema = SCHEMAS[schema_name]
    except KeyError as exc:
        raise SchemaError(f"unknown Aion artifact schema {schema_name!r}") from exc
    if expected_schema is not None and schema.name != expected_schema.name:
        raise SchemaError(f"artifact schema is {schema.name!r}; expected {expected_schema.name!r}")
    artifact_version = SchemaVersion.parse(_text_attribute(handle, "schema_version"))
    if artifact_version.major != schema.version.major:
        raise IncompatibleSchemaVersionError(
            f"cannot read {schema.name} major version {artifact_version.major}; "
            f"this Aion reader supports major version {schema.version.major}"
        )
    if not schema.version.can_read(artifact_version):
        raise IncompatibleSchemaVersionError(
            f"cannot read newer {schema.name} schema {artifact_version}; "
            f"this Aion reader supports through {schema.version}"
        )
    kind = _text_attribute(handle, "artifact_kind")
    if kind != schema.kind.value:
        raise SchemaError(
            f"artifact_kind is {kind!r}; schema {schema.name!r} requires {schema.kind.value!r}"
        )
    if _text_attribute(handle, "unit_system") != "atomic":
        raise SchemaError("Aion 0.2 HDF5 artifacts require unit_system='atomic'")
    if _integer_attribute(handle, "index_base") != 0:
        raise SchemaError("Aion HDF5 integer indices are zero-based")
    complete_value = _integer_attribute(handle, "complete")
    if complete_value not in (0, 1):
        raise SchemaError("complete must be the integer 0 or 1")
    complete = bool(complete_value)
    if require_complete and not complete:
        raise IncompleteArtifactError("artifact is not marked complete")
    actual_groups = set(handle.keys())
    required_groups = set(schema.required_groups)
    missing = sorted(required_groups - actual_groups)
    unknown = sorted(actual_groups - required_groups)
    if missing:
        raise SchemaError(f"artifact lacks required top-level group(s): {', '.join(missing)}")
    if unknown:
        raise SchemaError(f"artifact has unknown top-level group(s): {', '.join(unknown)}")
    for name in schema.required_groups:
        if not isinstance(handle[name], h5py.Group):
            raise SchemaError(f"top-level object /{name} must be a group")
    _validate_dataset_metadata(handle)
    if schema is TRAJECTORY_SCHEMA:
        _validate_observable_namespace(handle["observables"])
    return ArtifactHeader(
        schema=schema,
        artifact_version=artifact_version,
        complete=complete,
        artifact_id=_text_attribute(handle, "artifact_id"),
    )


def _validate_dataset_metadata(handle: h5py.File) -> None:
    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        for attribute in ("unit", "physical_dimension"):
            if attribute not in obj.attrs:
                raise SchemaError(f"dataset /{name} lacks required {attribute!r} metadata")
            _decode_text(obj.attrs[attribute], f"dataset /{name} attribute {attribute}")

    handle.visititems(visitor)


def _validate_observable_namespace(group: h5py.Group) -> None:
    forbidden = {"current", "dipole", "energy"}
    for name in forbidden:
        if name in group and isinstance(group[name], h5py.Dataset):
            raise SchemaError(
                f"generic observable dataset /observables/{name} is forbidden; "
                "use a definition-specific group and identifier"
            )


def _text_attribute(handle: h5py.File, name: str) -> str:
    if name not in handle.attrs:
        raise SchemaError(f"artifact lacks required attribute {name!r}")
    return _decode_text(handle.attrs[name], f"attribute {name}")


def _decode_text(value: object, path: str) -> str:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in {"S", "U", "O"}:
        raise SchemaError(f"{path} must be scalar text")
    item = array.reshape(-1)[0]
    result = bytes(item).decode("utf-8") if isinstance(item, bytes | np.bytes_) else str(item)
    if not result:
        raise SchemaError(f"{path} cannot be empty")
    return result


def _integer_attribute(handle: h5py.File, name: str) -> int:
    if name not in handle.attrs:
        raise SchemaError(f"artifact lacks required attribute {name!r}")
    value = np.asarray(handle.attrs[name])
    if value.size != 1 or value.dtype.kind not in {"i", "u"}:
        raise SchemaError(f"artifact attribute {name!r} must be a scalar integer")
    return int(value.reshape(-1)[0])
