from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aion.backends import CuPyBackend
from aion.config import BackendConfig, BackendKind
from aion.electromagnetism import UniformMagneticField
from aion.electronic_structure import prepare_pyscf_reference
from aion.workflows import (
    MagneticBenchmarkConfig,
    load_magnetic_benchmark,
    run_magnetic_benchmark,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.gpu


def test_gpu_workflow_and_artifact_match_cpu_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    field = UniformMagneticField((0.011, -0.007, 0.005))
    cpu_config = MagneticBenchmarkConfig(
        (field,),
        backend=BackendConfig(),
        block_size=4096,
    )
    gpu_config = replace(
        cpu_config,
        backend=BackendConfig(BackendKind.GPU, device_index=0),
    )
    cpu = run_magnetic_benchmark(reference, cpu_config)
    path = tmp_path / "gpu-magnetic-benchmark.h5"
    progress = []
    transfer_count = 0
    original_to_host = CuPyBackend.to_host

    def audited_to_host(self: CuPyBackend, value: object) -> np.ndarray:
        nonlocal transfer_count
        assert progress and progress[-1].stage == "finalize"
        transfer_count += 1
        return original_to_host(self, value)

    monkeypatch.setattr(CuPyBackend, "to_host", audited_to_host)
    gpu = run_magnetic_benchmark(
        reference,
        gpu_config,
        output_path=path,
        progress=progress.append,
    )
    loaded = load_magnetic_benchmark(path)
    assert loaded.result_id == gpu.result_id
    assert loaded.config.backend.kind is BackendKind.GPU
    assert transfer_count > len(gpu.matrices)
    assert '"evaluator":"gpu4pyscf.dft.numint.eval_ao"' in (loaded.ao_provenance_json)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert f'"{variable}":"1"' in loaded.ao_provenance_json
    assert tuple(value.path for value in cpu.matrices) == tuple(
        value.path for value in gpu.matrices
    )
    for expected, actual in zip(cpu.matrices, gpu.matrices, strict=True):
        np.testing.assert_allclose(actual.values, expected.values, atol=3.0e-11, rtol=3.0e-11)
    assert not any(value.passed is False for value in gpu.diagnostics)
