from __future__ import annotations

import numpy as np
import pytest

from aion.backends import NumPyBackend, Workspace
from aion.errors import BackendError, DeviceResidencyError

pytestmark = pytest.mark.fast


def test_numpy_backend_rejects_implicit_device_transfer() -> None:
    class DeviceLike:
        def get(self) -> np.ndarray:
            return np.ones(1)

    backend = NumPyBackend()
    with pytest.raises(DeviceResidencyError, match="implicit GPU-to-host"):
        backend.asarray(DeviceLike())
    with pytest.raises(DeviceResidencyError, match="not a NumPy"):
        backend.assert_resident([1.0])


def test_workspace_owns_resident_arrays_and_disallows_replacement() -> None:
    backend = NumPyBackend()
    workspace = Workspace(backend)
    installed = workspace.install_host_array("overlap", np.eye(2))
    assert installed is workspace.require("overlap")
    workspace.allocate("trial", (2, 2), dtype=np.complex128)
    workspace.assert_all_resident()
    with pytest.raises(BackendError, match="already installed"):
        workspace.install_host_array("overlap", np.eye(2))
    with pytest.raises(BackendError, match="already allocated"):
        workspace.allocate("trial", (2, 2), dtype=np.complex128)
    with pytest.raises(BackendError, match="not installed"):
        workspace.require("missing")
