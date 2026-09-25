from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aion.config import (
    AtomConfig,
    BackendConfig,
    ElectromagneticOrigin,
    OneElectronReferenceConfig,
)
from aion.electronic_structure import (
    AOGridPolicy,
    AOPruningKind,
    atom_pair_block_residuals,
    evaluate_zero_field_one_electron,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
)

pytestmark = pytest.mark.integration

_FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "exact_one_electron"


def _config(name: str) -> OneElectronReferenceConfig:
    fixture = json.loads(
        (_FIXTURE_DIRECTORY / f"{name}_sto3g.fixture.json").read_text(encoding="utf-8")
    )
    values = fixture["config"]
    return OneElectronReferenceConfig(
        atoms=tuple(
            AtomConfig(atom["symbol"], tuple(atom["position_au"])) for atom in values["atoms"]
        ),
        basis=values["basis"],
        electromagnetic_origin=ElectromagneticOrigin(tuple(values["electromagnetic_origin_au"])),
    )


def _evaluate(
    name: str,
    level: int,
    pruning: AOPruningKind = AOPruningKind.NONE,
) -> tuple[object, object, object]:
    reference = prepare_one_electron_ao_reference(_config(name))
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(level, pruning=pruning),
        block_size=1024,
    )
    return reference, quadrature, evaluate_zero_field_one_electron(quadrature)


def test_oh_fixture_reconstructs_deterministically() -> None:
    fixture = json.loads((_FIXTURE_DIRECTORY / "oh_sto3g.fixture.json").read_text(encoding="utf-8"))
    reference = prepare_one_electron_ao_reference(_config("oh"))
    assert fixture["status"] == "immutable_input_no_numerical_claim"
    assert reference.config.scientific_id == fixture["config_id"]
    assert reference.fingerprint_sha256 == fixture["reference_fingerprint_sha256"]
    assert list(reference.basis_metadata.ao_labels) == fixture["basis_metadata"]["ao_labels"]
    np.testing.assert_array_equal(reference.anchor_topology.ao_to_atom, fixture["ao_to_atom"])
    assert reference.core_operators.fingerprint_sha256 == fixture["fingerprints"]["core_operators"]


def test_complete_hh_zero_field_contraction_and_atom_pair_floors() -> None:
    reference, _, result = _evaluate("hh", 3)
    comparisons = (
        result.overlap,
        result.kinetic,
        result.nuclear_attraction,
        result.mechanical,
    )
    assert max(value.relative_frobenius_residual for value in comparisons) < 6.0e-9
    assert result.canonical_momentum.relative_frobenius_residual < 1.0e-9
    assert result.opposite_momentum_sign_relative_residual > 0.9
    np.testing.assert_array_equal(
        result.mechanical.analytic,
        result.kinetic.analytic + result.nuclear_attraction.analytic,
    )
    np.testing.assert_array_equal(
        result.mechanical.quadrature,
        result.kinetic.quadrature + result.nuclear_attraction.quadrature,
    )

    blocks = atom_pair_block_residuals(
        result.mechanical,
        reference.anchor_topology.ao_to_atom,
    )
    np.testing.assert_array_equal(blocks.pair_indices, ((0, 0), (0, 1), (1, 0), (1, 1)))
    assert max(blocks.relative_frobenius) < 2.0e-9
    np.testing.assert_allclose(
        blocks.absolute_frobenius[1],
        blocks.absolute_frobenius[2],
        atol=1.0e-14,
        rtol=0.0,
    )


def test_explicit_nwchem_pruning_is_an_independent_grid_choice() -> None:
    _, unpruned, unpruned_result = _evaluate("oh", 3)
    _, pruned, pruned_result = _evaluate("oh", 3, AOPruningKind.NWCHEM)
    assert unpruned.grid.pruning == "none"
    assert pruned.grid.pruning == "nwchem"
    assert pruned.grid.npoints < unpruned.grid.npoints
    assert pruned.grid.fingerprint_sha256 != unpruned.grid.fingerprint_sha256
    for result in (unpruned_result, pruned_result):
        assert result.overlap.relative_frobenius_residual < 3.0e-8
        assert result.kinetic.relative_frobenius_residual < 8.0e-8
        assert result.nuclear_attraction.relative_frobenius_residual < 2.0e-8
        assert result.mechanical.relative_frobenius_residual < 5.0e-8
        assert result.canonical_momentum.relative_frobenius_residual < 2.0e-7


def test_oh_all_wp1_matrix_families_converge_and_momentum_sign_is_resolved() -> None:
    evaluated = tuple(_evaluate("oh", level)[2] for level in (1, 2, 3, 4))
    names = ("overlap", "kinetic", "nuclear_attraction", "mechanical")
    for name in names:
        residuals = np.asarray(
            [getattr(result, name).relative_frobenius_residual for result in evaluated]
        )
        assert np.all(np.diff(residuals) < 0.0), (name, residuals)
    momentum = np.asarray(
        [result.canonical_momentum.relative_frobenius_residual for result in evaluated]
    )
    assert np.all(np.diff(momentum) < 0.0)

    finest = evaluated[-1]
    assert (
        max(
            finest.overlap.relative_frobenius_residual,
            finest.kinetic.relative_frobenius_residual,
            finest.nuclear_attraction.relative_frobenius_residual,
            finest.mechanical.relative_frobenius_residual,
            finest.canonical_momentum.relative_frobenius_residual,
        )
        < 3.0e-8
    )
    assert finest.opposite_momentum_sign_relative_residual > 1.9
    assert (
        finest.opposite_momentum_sign_relative_residual
        > 1.0e6 * finest.canonical_momentum.relative_frobenius_residual
    )
    assert (
        max(
            finest.overlap.hermiticity_residual,
            finest.kinetic.hermiticity_residual,
            finest.nuclear_attraction.hermiticity_residual,
            finest.mechanical.hermiticity_residual,
        )
        < 3.0e-16
    )
    assert (
        finest.canonical_momentum.hermiticity_residual
        < 4.0 * finest.canonical_momentum.relative_frobenius_residual
    )
