"""Transactional publication primitives for immutable HDF5 artifacts."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path

import h5py

from aion.errors import SchemaError
from aion.io.schemas import ArtifactSchema, stamp_artifact, validate_artifact


def publish_hdf5(
    path: str | Path,
    schema: ArtifactSchema,
    *,
    artifact_id: str,
    populate: Callable[[h5py.File], None],
) -> None:
    """Publish a complete immutable HDF5 artifact without overwriting.

    Data are built and validated in a same-directory ``.partial`` file. A hard
    link publishes the closed complete file atomically and, unlike POSIX rename,
    fails if the destination already exists. The private partial name is then
    unlinked. Completed artifacts are never opened writable by this routine.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise SchemaError(f"refusing to overwrite completed artifact {target}")
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    try:
        with h5py.File(partial, "x") as handle:
            stamp_artifact(handle, schema, complete=False, artifact_id=artifact_id)
            populate(handle)
            handle.flush()
        validate_artifact(partial, expected_schema=schema, require_complete=False)
        with h5py.File(partial, "r+") as handle:
            handle.attrs.modify("complete", 1)
            handle.flush()
        validate_artifact(partial, expected_schema=schema, require_complete=True)
        try:
            os.link(partial, target)
        except FileExistsError:
            raise SchemaError(f"refusing to overwrite completed artifact {target}") from None
        partial.unlink()
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def write_dataset(
    group: h5py.Group,
    name: str,
    data: object,
    *,
    unit: str,
    physical_dimension: str,
    compressed: bool = False,
) -> h5py.Dataset:
    """Create one dataset carrying mandatory physical metadata."""

    if name in group:
        raise SchemaError(f"dataset {group.name}/{name} already exists")
    kwargs: dict[str, object] = {}
    if compressed:
        shape = getattr(data, "shape", ())
        if shape:
            kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
    dataset = group.create_dataset(name, data=data, **kwargs)
    dataset.attrs["unit"] = unit
    dataset.attrs["physical_dimension"] = physical_dimension
    return dataset
