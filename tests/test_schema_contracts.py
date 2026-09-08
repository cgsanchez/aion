from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from aion.errors import IncompatibleSchemaVersionError, IncompleteArtifactError, SchemaError
from aion.io import (
    REFERENCE_SCHEMA,
    TRAJECTORY_SCHEMA,
    SchemaVersion,
    stamp_artifact,
    validate_artifact,
)

pytestmark = pytest.mark.fast


def stamped(path: Path, *, trajectory: bool = False, complete: bool = True) -> Path:
    schema = TRAJECTORY_SCHEMA if trajectory else REFERENCE_SCHEMA
    with h5py.File(path, "w") as handle:
        stamp_artifact(handle, schema, complete=complete, artifact_id="fixture-1")
    return path


def test_reference_and_trajectory_schema_fixtures_validate(tmp_path: Path) -> None:
    reference = stamped(tmp_path / "reference.h5")
    trajectory = stamped(tmp_path / "trajectory.h5", trajectory=True)
    assert validate_artifact(reference).schema is REFERENCE_SCHEMA
    assert validate_artifact(trajectory).schema is TRAJECTORY_SCHEMA


def test_compatible_patch_is_accepted_and_new_major_fails_clearly(
    tmp_path: Path,
) -> None:
    path = stamped(tmp_path / "reference.h5")
    with h5py.File(path, "r+") as handle:
        handle.attrs["schema_version"] = "1.0.7"
    assert str(validate_artifact(path).artifact_version) == "1.0.7"
    with h5py.File(path, "r+") as handle:
        handle.attrs["schema_version"] = "2.0.0"
    with pytest.raises(IncompatibleSchemaVersionError, match="major version 2"):
        validate_artifact(path)


def test_schema_reader_accepts_older_minor_but_not_newer_minor() -> None:
    reader = SchemaVersion(1, 3, 0)
    assert reader.can_read(SchemaVersion(1, 2, 9))
    assert not reader.can_read(SchemaVersion(1, 4, 0))
    assert not reader.can_read(SchemaVersion(2, 0, 0))


def test_incomplete_artifact_requires_explicit_opt_in(tmp_path: Path) -> None:
    path = stamped(tmp_path / "partial.h5", complete=False)
    with pytest.raises(IncompleteArtifactError):
        validate_artifact(path)
    assert not validate_artifact(path, require_complete=False).complete


def test_dataset_unit_and_dimension_metadata_are_mandatory(tmp_path: Path) -> None:
    path = stamped(tmp_path / "reference.h5")
    with h5py.File(path, "r+") as handle:
        handle["reference"].create_dataset("overlap", data=np.eye(2))
    with pytest.raises(SchemaError, match="unit"):
        validate_artifact(path)
    with h5py.File(path, "r+") as handle:
        dataset = handle["reference/overlap"]
        dataset.attrs["unit"] = "1"
        dataset.attrs["physical_dimension"] = "overlap"
    validate_artifact(path)


@pytest.mark.parametrize("generic_name", ["current", "dipole", "energy"])
def test_generic_observable_datasets_are_forbidden(
    tmp_path: Path,
    generic_name: str,
) -> None:
    path = stamped(tmp_path / f"{generic_name}.h5", trajectory=True)
    with h5py.File(path, "r+") as handle:
        dataset = handle["observables"].create_dataset(generic_name, data=np.zeros(2))
        dataset.attrs["unit"] = "1"
        dataset.attrs["physical_dimension"] = "unknown"
    with pytest.raises(SchemaError, match="definition-specific"):
        validate_artifact(path)


def test_unknown_top_level_groups_fail(tmp_path: Path) -> None:
    path = stamped(tmp_path / "reference.h5")
    with h5py.File(path, "r+") as handle:
        handle.create_group("mystery")
    with pytest.raises(SchemaError, match="unknown top-level"):
        validate_artifact(path)
