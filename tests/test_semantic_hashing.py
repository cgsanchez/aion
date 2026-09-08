from __future__ import annotations

import math

import numpy as np
import pytest

from aion.config import canonical_bytes, canonical_sha256
from aion.errors import ConfigurationError

pytestmark = pytest.mark.fast


def test_mapping_order_is_canonical_but_sequence_order_is_not() -> None:
    assert canonical_sha256({"a": 1, "b": 2.0}) == canonical_sha256({"b": 2.0, "a": 1})
    assert canonical_sha256([1, 2]) != canonical_sha256([2, 1])


def test_float_encoding_is_lossless_and_type_distinct() -> None:
    value = 0.1
    adjacent = math.nextafter(value, math.inf)
    assert canonical_sha256(value) != canonical_sha256(adjacent)
    assert canonical_sha256(1) != canonical_sha256(1.0)
    assert canonical_sha256(0.0) != canonical_sha256(-0.0)


def test_array_hash_includes_dtype_shape_and_canonical_bytes() -> None:
    values = np.array([1.0, 2.0], dtype=np.float64)
    assert canonical_sha256(values) != canonical_sha256(values.astype(np.float32))
    assert canonical_sha256(values) != canonical_sha256(values.reshape(1, 2))
    assert canonical_sha256(values) == canonical_sha256(values.astype(">f8"))
    changed = values.copy()
    changed[1] = math.nextafter(2.0, math.inf)
    assert canonical_sha256(values) != canonical_sha256(changed)


def test_canonical_hash_rejects_nonfinite_and_object_arrays() -> None:
    with pytest.raises(ConfigurationError, match="non-finite"):
        canonical_bytes(float("nan"))
    with pytest.raises(ConfigurationError, match="object dtype"):
        canonical_bytes(np.array([object()], dtype=object))
    with pytest.raises(ConfigurationError, match="non-finite array"):
        canonical_bytes(np.array([np.inf], dtype=np.float64))
