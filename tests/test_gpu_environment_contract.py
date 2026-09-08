from __future__ import annotations

import os

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind, Precision

pytestmark = pytest.mark.gpu


def test_managed_physical_gpu_and_fp64_contract() -> None:
    """Qualify the WP1 GPU environment without exercising future Aion numerics."""

    import cupy as cp

    assert os.environ.get("AION_GPU_LAUNCHER") == "1"
    assert cp.cuda.runtime.getDeviceCount() >= 1
    backend = BackendConfig(BackendKind.GPU, Precision.FP64_COMPLEX128, device_index=0)
    with cp.cuda.Device(backend.device_index):
        values = cp.asarray([1.0, 2.0], dtype=cp.float64)
        assert isinstance(values, cp.ndarray)
        assert values.dtype == cp.float64
        assert np.array_equal(cp.asnumpy(values), np.array([1.0, 2.0]))
        cp.cuda.get_current_stream().synchronize()
