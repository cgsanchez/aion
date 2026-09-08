"""Lossless canonical SHA-256 hashing for scientific identities."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path

import numpy as np

from aion.errors import ConfigurationError


def canonical_bytes(value: object) -> bytes:
    """Encode supported values without losing type, float, shape, or dtype data."""

    output = bytearray()
    _encode(value, output, "root")
    return bytes(output)


def canonical_sha256(value: object) -> str:
    """Return the lowercase SHA-256 digest of :func:`canonical_bytes`."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _field(tag: bytes, payload: bytes, output: bytearray) -> None:
    output.extend(tag)
    output.extend(struct.pack(">Q", len(payload)))
    output.extend(payload)


def _encode(value: object, output: bytearray, path: str) -> None:
    if value is None:
        output.extend(b"N")
    elif isinstance(value, Enum):
        _encode(value.value, output, path)
    elif isinstance(value, bool):
        output.extend(b"B1" if value else b"B0")
    elif isinstance(value, int):
        _field(b"I", str(value).encode("ascii"), output)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigurationError(f"{path} contains a non-finite float")
        _field(b"F", value.hex().encode("ascii"), output)
    elif isinstance(value, str):
        _field(b"S", value.encode("utf-8"), output)
    elif isinstance(value, bytes):
        _field(b"Y", value, output)
    elif isinstance(value, Path):
        _field(b"P", str(value).encode("utf-8"), output)
    elif isinstance(value, np.ndarray):
        _encode_array(value, output, path)
    elif isinstance(value, Mapping):
        output.extend(b"M")
        keys = list(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise ConfigurationError(f"{path} contains a non-string mapping key")
        output.extend(struct.pack(">Q", len(keys)))
        for key in sorted(keys):
            _encode(key, output, f"{path}.<key>")
            _encode(value[key], output, f"{path}.{key}")
    elif isinstance(value, Sequence):
        output.extend(b"Q")
        output.extend(struct.pack(">Q", len(value)))
        for index, item in enumerate(value):
            _encode(item, output, f"{path}[{index}]")
    else:
        raise ConfigurationError(
            f"{path} has unsupported canonical-hash type {type(value).__name__}"
        )


def _encode_array(value: np.ndarray, output: bytearray, path: str) -> None:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ConfigurationError(f"{path} uses an object dtype, which is not canonical")
    output.extend(b"A")
    _encode(tuple(int(size) for size in array.shape), output, f"{path}.shape")
    if array.dtype.kind in "biufc":
        if not np.all(np.isfinite(array)):
            raise ConfigurationError(f"{path} contains a non-finite array value")
        dtype = array.dtype.newbyteorder("<")
        canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
        _encode(dtype.str, output, f"{path}.dtype")
        _field(b"D", canonical.tobytes(order="C"), output)
        return
    if array.dtype.kind in "SU":
        _encode(array.dtype.str, output, f"{path}.dtype")
        _encode(tuple(str(item) for item in array.reshape(-1)), output, f"{path}.values")
        return
    raise ConfigurationError(f"{path} uses unsupported dtype {array.dtype}")
