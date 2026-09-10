from __future__ import annotations

import numpy as np
import pytest

from aion.config import BackendConfig, BackendKind
from aion.electronic_structure import prepare_ao_quadrature, prepare_pyscf_reference
from test_reference_integration import molecular_config

pytestmark = pytest.mark.gpu


def test_gpu4pyscf_ao_blocks_are_resident_and_match_cpu() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    cpu = prepare_ao_quadrature(reference, BackendConfig(), block_size=257)
    gpu = prepare_ao_quadrature(
        reference,
        BackendConfig(BackendKind.GPU, device_index=0),
        block_size=257,
    )
    assert gpu.provenance.evaluator == "gpu4pyscf.dft.numint.eval_ao"
    assert gpu.provenance.device_index == 0
    for cpu_block, gpu_block in zip(cpu.blocks(), gpu.blocks(), strict=True):
        for value in (
            gpu_block.coordinates_au,
            gpu_block.weights_au,
            gpu_block.values,
            gpu_block.gradients,
        ):
            gpu.backend.assert_resident(value)
        np.testing.assert_allclose(
            gpu.backend.to_host(gpu_block.values), cpu_block.values, atol=2.0e-14, rtol=2.0e-14
        )
        np.testing.assert_allclose(
            gpu.backend.to_host(gpu_block.gradients),
            cpu_block.gradients,
            atol=3.0e-13,
            rtol=3.0e-13,
        )
