from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

import aion
from aion.cli import main
from aion.config import (
    AtomConfig,
    BackendConfig,
    ElectronicStructureConfig,
    FormulationKind,
    GridPruning,
    MoleculeConfig,
    ReferenceConfig,
    ReferenceOutputConfig,
    XCFamily,
    dumps_config,
)
from aion.electronic_structure import (
    PreparedReference,
    load_reference_data,
    momentum_grid_residual,
    prepare_pyscf_reference,
)
from aion.errors import ReferencePreparationError, SchemaError, UnsupportedConfigurationError

pytestmark = pytest.mark.integration


def molecular_config(name: str, output: Path | None = None) -> ReferenceConfig:
    if name == "h2":
        atoms = (AtomConfig("H", (0.0, 0.0, -0.7)), AtomConfig("H", (0.0, 0.0, 0.7)))
    elif name == "lih":
        atoms = (AtomConfig("Li", (0.0, 0.0, -1.5)), AtomConfig("H", (0.0, 0.0, 1.5)))
    else:
        raise ValueError(name)
    return ReferenceConfig(
        molecule=MoleculeConfig(atoms=atoms),
        electronic_structure=ElectronicStructureConfig(
            basis="sto-3g",
            functional="pbe",
            xc_family=XCFamily.GGA,
            grid_level=1,
            scf_energy_tolerance_au=1.0e-11,
        ),
        backend=BackendConfig(),
        output=ReferenceOutputConfig(Path("reference.h5") if output is None else output),
    )


@pytest.mark.parametrize("name", ("h2", "lih"))
def test_h2_lih_reference_operators_topology_and_reconstruction(name: str) -> None:
    reference = prepare_pyscf_reference(molecular_config(name))
    assert isinstance(reference, PreparedReference)
    assert reference.supported_formulations == tuple(FormulationKind)
    assert not reference.ground_state.density.flags.writeable
    assert not reference.grid.coordinates_au.flags.writeable
    assert not reference.core_operators.canonical_momentum.flags.writeable
    overlap = reference.core_operators.overlap
    momentum = reference.core_operators.canonical_momentum
    assert np.linalg.norm(momentum - momentum.conj().transpose(0, 2, 1)) < 1.0e-13
    assert np.linalg.eigvalsh(overlap).min() > 0.0
    assert np.einsum("ij,ji->", reference.ground_state.density, overlap).real == pytest.approx(
        reference.ground_state.electron_count, abs=1.0e-11
    )
    workspace_a = reference.create_workspace(BackendConfig())
    workspace_b = reference.create_workspace(BackendConfig())
    assert workspace_a is not workspace_b
    assert workspace_a.electronic_model is not workspace_b.electronic_model
    assert workspace_a.caches["reference_fingerprint_sha256"] == reference.fingerprint_sha256
    assert np.array_equal(workspace_a.require("grid.coordinates_au"), reference.grid.coordinates_au)
    from pyscf import dft

    qualification_grid = dft.Grids(workspace_a.electronic_model.mol)
    qualification_grid.level = 4
    qualification_grid.build()
    residual = momentum_grid_residual(
        workspace_a.electronic_model.mol,
        qualification_grid.coords,
        qualification_grid.weights,
        momentum,
    )
    assert residual < 1.0e-7


def test_reference_transactional_roundtrip_rebuild_and_tamper_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "reference.h5"
    reference = prepare_pyscf_reference(molecular_config("h2", path))
    reference.save()
    assert tuple(tmp_path.iterdir()) == (path,)
    loaded = load_reference_data(path)
    assert loaded.fingerprint_sha256 == reference.fingerprint_sha256
    assert loaded.core_operators.fingerprint_sha256 == reference.core_operators.fingerprint_sha256
    assert loaded.grid.fingerprint_sha256 == reference.grid.fingerprint_sha256
    assert np.array_equal(loaded.grid.coordinates_au, reference.grid.coordinates_au)
    from pyscf.dft.rks import RKS

    def forbidden_scf(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("workspace reconstruction must not rerun SCF")

    monkeypatch.setattr(RKS, "kernel", forbidden_scf)
    rebuilt = loaded.create_workspace(BackendConfig()).electronic_model
    assert rebuilt.converged
    assert np.array_equal(rebuilt.grids.coords, reference.grid.coordinates_au)
    with pytest.raises(SchemaError, match="refusing to overwrite"):
        reference.save()
    with h5py.File(path, "r+") as handle:
        handle["reference/operators/kinetic"][0, 0] += 1.0e-6
    with pytest.raises(ReferencePreparationError, match="fingerprint mismatch"):
        load_reference_data(path)


def test_density_fitted_reference_round_trip_reconstructs_the_declared_auxiliary_basis(
    tmp_path: Path,
) -> None:
    base = molecular_config("h2", tmp_path / "density-fitted.reference.h5")
    config = replace(
        base,
        electronic_structure=replace(
            base.electronic_structure,
            density_fitting=True,
            auxiliary_basis="weigend",
        ),
    )
    reference = prepare_pyscf_reference(config)
    reference.save()
    loaded = load_reference_data(config.output.artifact_path)
    model = loaded.create_workspace(BackendConfig()).electronic_model
    assert model.with_df.auxbasis == "weigend"
    assert loaded.config.electronic_structure.density_fitting
    assert loaded.config.electronic_structure.auxiliary_basis == "weigend"


def test_reference_preparation_honors_an_explicit_unpruned_grid() -> None:
    base = molecular_config("h2")
    reference = prepare_pyscf_reference(
        replace(
            base,
            electronic_structure=replace(
                base.electronic_structure,
                grid_pruning=GridPruning.NONE,
            ),
        )
    )
    assert reference.grid.pruning == "none"
    workspace = reference.create_workspace(BackendConfig())
    assert workspace.electronic_model.grids.prune is None


def test_declared_functional_family_and_hybrids_fail_before_scf() -> None:
    config = molecular_config("h2")
    mismatched = ReferenceConfig(
        molecule=config.molecule,
        electronic_structure=ElectronicStructureConfig(
            basis="sto-3g", functional="pbe", xc_family=XCFamily.LDA, grid_level=0
        ),
    )
    with pytest.raises(UnsupportedConfigurationError, match="configuration declares LDA"):
        prepare_pyscf_reference(mismatched)
    hybrid = ReferenceConfig(
        molecule=config.molecule,
        electronic_structure=ElectronicStructureConfig(
            basis="sto-3g", functional="pbe0", xc_family=XCFamily.GGA, grid_level=0
        ),
    )
    with pytest.raises(UnsupportedConfigurationError, match="hybrid"):
        prepare_pyscf_reference(hybrid)


def test_public_prepare_and_cli_publish_the_same_reference_contract(tmp_path: Path) -> None:
    api_reference = aion.prepare_reference(molecular_config("h2"))
    assert isinstance(api_reference, PreparedReference)
    artifact = tmp_path / "cli-reference.h5"
    config = molecular_config("h2", artifact)
    config_path = tmp_path / "reference.toml"
    config_path.write_text(dumps_config(config), encoding="utf-8")
    assert main(["prepare", str(config_path)]) == 0
    loaded = aion.load_reference(artifact)
    assert isinstance(loaded, PreparedReference)
    assert loaded.config.scientific_id == api_reference.config.scientific_id
    assert loaded.core_operators.fingerprint_sha256 == (
        api_reference.core_operators.fingerprint_sha256
    )
    assert loaded.grid.fingerprint_sha256 == api_reference.grid.fingerprint_sha256
    assert loaded.ground_state.energy_total_au == pytest.approx(
        api_reference.ground_state.energy_total_au, abs=2.0e-12
    )
    assert np.allclose(
        loaded.ground_state.density,
        api_reference.ground_state.density,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
