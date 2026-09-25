from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aion.config import AtomConfig, BackendConfig, OneElectronReferenceConfig
from aion.electronic_structure import (
    AOGridPolicy,
    evaluate_zero_field_overlap_kinetic,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
    reconstruct_one_electron_molecule,
)
from aion.errors import ConfigurationError

pytestmark = pytest.mark.integration

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "exact_one_electron" / "hh_sto3g.fixture.json"


def _config() -> OneElectronReferenceConfig:
    return OneElectronReferenceConfig(
        atoms=(
            AtomConfig("H", (0.0, 0.0, -0.7)),
            AtomConfig("H", (0.0, 0.0, 0.7)),
        ),
        basis="sto-3g",
    )


def test_g0_fixture_reconstructs_without_scf(monkeypatch: pytest.MonkeyPatch) -> None:
    from pyscf.dft.rks import RKS

    def forbidden_scf(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("occupancy-independent AO preparation must not invoke SCF")

    monkeypatch.setattr(RKS, "kernel", forbidden_scf)
    reference = prepare_one_electron_ao_reference(_config())
    molecule = reconstruct_one_electron_molecule(reference)
    fixture = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))

    assert fixture["status"] == "immutable_input_no_numerical_claim"
    assert reference.config.scientific_id == fixture["config_id"]
    assert reference.fingerprint_sha256 == fixture["reference_fingerprint_sha256"]
    assert reference.basis_metadata.ao_labels == tuple(fixture["basis_metadata"]["ao_labels"])
    np.testing.assert_array_equal(reference.anchor_topology.ao_to_atom, fixture["ao_to_atom"])
    for name in ("overlap", "kinetic", "nuclear_attraction", "position"):
        np.testing.assert_array_equal(
            getattr(reference.core_operators, name), fixture["analytic_matrices"][name]
        )
    momentum = np.asarray(fixture["analytic_matrices"]["canonical_momentum_real"]) + 1j * (
        np.asarray(fixture["analytic_matrices"]["canonical_momentum_imag"])
    )
    np.testing.assert_array_equal(reference.core_operators.canonical_momentum, momentum)
    assert molecule.nao_nr() == 2


def test_g0_fixture_is_deterministic() -> None:
    left = prepare_one_electron_ao_reference(_config())
    right = prepare_one_electron_ao_reference(_config())
    assert left.fingerprint_sha256 == right.fingerprint_sha256
    assert left.core_operators.fingerprint_sha256 == right.core_operators.fingerprint_sha256
    assert left.basis_metadata.fingerprint_sha256 == right.basis_metadata.fingerprint_sha256


def test_occupancy_independent_reference_requires_an_explicit_grid() -> None:
    reference = prepare_one_electron_ao_reference(_config())
    with pytest.raises(ConfigurationError, match="no stored DFT grid"):
        prepare_ao_quadrature(reference, BackendConfig(), grid_policy=AOGridPolicy.reference())


def test_wp1_hh_overlap_and_kinetic_converge_to_analytic_matrices() -> None:
    reference = prepare_one_electron_ao_reference(_config())
    results = tuple(
        evaluate_zero_field_overlap_kinetic(
            prepare_ao_quadrature(
                reference,
                BackendConfig(),
                grid_policy=AOGridPolicy.qualification(level),
                block_size=512,
            )
        )
        for level in (0, 2, 4)
    )
    overlap_residuals = np.asarray([value.overlap.relative_frobenius_residual for value in results])
    kinetic_residuals = np.asarray([value.kinetic.relative_frobenius_residual for value in results])
    assert np.all(np.diff(overlap_residuals) < 0.0)
    assert np.all(np.diff(kinetic_residuals) < 0.0)
    assert overlap_residuals[-1] < 2.0e-11
    assert kinetic_residuals[-1] < 3.0e-10
    assert (
        max(
            results[-1].overlap.hermiticity_residual,
            results[-1].kinetic.hermiticity_residual,
        )
        < 2.0e-16
    )
