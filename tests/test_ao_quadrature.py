from __future__ import annotations

import numpy as np
import pytest

from aion.config import BackendConfig
from aion.electronic_structure import (
    AOGridKind,
    AOGridPolicy,
    estimate_ao_block_bytes,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
)
from aion.errors import ConfigurationError
from test_reference_integration import molecular_config

pytestmark = pytest.mark.integration


def _contract_blocks(quadrature: object, nao: int) -> tuple[np.ndarray, np.ndarray]:
    overlap = np.zeros((nao, nao))
    momentum = np.zeros((3, nao, nao), dtype=np.complex128)
    for block in quadrature.blocks():  # type: ignore[attr-defined]
        overlap += np.einsum(
            "p,pm,pn->mn", block.weights_au, block.values, block.values, optimize=True
        )
        momentum += -1j * np.einsum(
            "p,pm,xpn->xmn",
            block.weights_au,
            block.values,
            block.gradients,
            optimize=True,
        )
    return overlap, momentum


@pytest.mark.parametrize("name", ("h2", "lih"))
def test_reference_blocks_reconstruct_overlap_and_momentum_without_scf(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = prepare_pyscf_reference(molecular_config(name))

    from pyscf.dft.rks import RKS

    def forbidden_scf(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("AO quadrature reconstruction must not rerun SCF")

    monkeypatch.setattr(RKS, "kernel", forbidden_scf)
    quadrature = prepare_ao_quadrature(reference, BackendConfig(), block_size=257)
    assert quadrature.grid.kind is AOGridKind.REFERENCE
    assert quadrature.grid.fingerprint_sha256 == reference.grid.fingerprint_sha256
    assert quadrature.provenance.evaluator == "pyscf.dft.numint.eval_ao"
    overlap, momentum = _contract_blocks(quadrature, reference.core_operators.nao)
    overlap_scale = max(1.0, np.linalg.norm(reference.core_operators.overlap))
    momentum_scale = max(1.0, np.linalg.norm(reference.core_operators.canonical_momentum))
    # The stored level-1 DFT grid is operational rather than an analytic-
    # integral grid. This measured bound is the declared reference-grid floor.
    assert np.linalg.norm(overlap - reference.core_operators.overlap) / overlap_scale < 7.0e-5
    assert (
        np.linalg.norm(momentum - reference.core_operators.canonical_momentum) / momentum_scale
        < 7.0e-5
    )


@pytest.mark.parametrize("name", ("h2", "lih"))
def test_level4_unpruned_grid_reaches_analytic_overlap_and_momentum_floor(name: str) -> None:
    reference = prepare_pyscf_reference(molecular_config(name))
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(4),
        block_size=1024,
    )
    overlap, momentum = _contract_blocks(quadrature, reference.core_operators.nao)
    overlap_scale = max(1.0, np.linalg.norm(reference.core_operators.overlap))
    momentum_scale = max(1.0, np.linalg.norm(reference.core_operators.canonical_momentum))
    assert np.linalg.norm(overlap - reference.core_operators.overlap) / overlap_scale < 1.0e-7
    assert (
        np.linalg.norm(momentum - reference.core_operators.canonical_momentum) / momentum_scale
        < 1.0e-7
    )


def test_blocks_have_common_layout_and_concatenate_in_grid_order() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    quadrature = prepare_ao_quadrature(reference, BackendConfig(), block_size=113)
    blocks = tuple(quadrature.blocks())
    assert len(blocks) > 1
    assert [block.index for block in blocks] == list(range(len(blocks)))
    assert blocks[0].start == 0
    assert blocks[-1].stop == quadrature.grid.npoints
    coordinates = np.concatenate([block.coordinates_au for block in blocks])
    weights = np.concatenate([block.weights_au for block in blocks])
    values = np.concatenate([block.values for block in blocks])
    gradients = np.concatenate([block.gradients for block in blocks], axis=1)
    np.testing.assert_array_equal(coordinates, quadrature.grid.coordinates_au)
    np.testing.assert_array_equal(weights, quadrature.grid.weights_au)
    assert values.shape == (quadrature.grid.npoints, reference.core_operators.nao)
    assert gradients.shape == (3, quadrature.grid.npoints, reference.core_operators.nao)


def test_block_size_invariance_and_generated_unpruned_grid_metadata() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    small = prepare_ao_quadrature(reference, BackendConfig(), block_size=97)
    large = prepare_ao_quadrature(reference, BackendConfig(), block_size=521)
    small_matrices = _contract_blocks(small, reference.core_operators.nao)
    large_matrices = _contract_blocks(large, reference.core_operators.nao)
    for left, right in zip(small_matrices, large_matrices, strict=True):
        np.testing.assert_allclose(left, right, atol=2.0e-14, rtol=2.0e-14)

    qualification = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(0),
        block_size=257,
    )
    assert qualification.grid.kind is AOGridKind.QUALIFICATION
    assert qualification.grid.level == 0
    assert qualification.grid.pruning == "none"
    assert qualification.grid.fingerprint_sha256 != reference.grid.fingerprint_sha256


def test_block_and_memory_policy_validation() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    required = estimate_ao_block_bytes(128, reference.core_operators.nao)
    accepted = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        block_size=128,
        memory_budget_bytes=required,
    )
    assert accepted.provenance.estimated_block_bytes == required
    with pytest.raises(ConfigurationError, match="exceeding"):
        prepare_ao_quadrature(
            reference,
            BackendConfig(),
            block_size=128,
            memory_budget_bytes=required - 1,
        )
    with pytest.raises(ConfigurationError):
        AOGridPolicy.qualification(-1)
