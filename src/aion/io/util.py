"""Small storage-boundary helpers shared by run artifacts."""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import h5py
import numpy as np

from aion.io.transaction import write_dataset


def write_text(
    group: h5py.Group,
    name: str,
    value: str,
    *,
    physical_dimension: str,
) -> h5py.Dataset:
    return write_dataset(
        group,
        name,
        np.bytes_(value),
        unit="1",
        physical_dimension=physical_dimension,
    )


def read_text(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    if isinstance(value, bytes | np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    """Durably replace one small non-scientific text record."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_text(path: str | Path, text: str) -> None:
    """Durably publish a small immutable text artifact without overwriting."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact {target}")
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite immutable artifact {target}") from None
    finally:
        if temporary.exists():
            temporary.unlink()
