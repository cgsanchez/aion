"""Occupancy-independent PySCF AO references for one-electron qualification."""

from __future__ import annotations

from typing import Any

import numpy as np

from aion.config import OneElectronReferenceConfig
from aion.electronic_structure.data import (
    AOBasisMetadata,
    DependencyVersions,
    OneElectronAOReference,
)
from aion.electronic_structure.operators import build_anchor_topology, build_core_operators
from aion.errors import ReferencePreparationError, UnsupportedConfigurationError


def prepare_one_electron_ao_reference(
    config: OneElectronReferenceConfig,
) -> OneElectronAOReference:
    """Build immutable analytic AO data without constructing or running an SCF model."""

    if not isinstance(config, OneElectronReferenceConfig):
        raise TypeError("config must be OneElectronReferenceConfig")
    molecule = _build_molecule(config)
    core = build_core_operators(molecule)
    topology = build_anchor_topology(molecule, core.nuclei)
    return OneElectronAOReference(
        config=config,
        core_operators=core,
        anchor_topology=topology,
        basis_metadata=_basis_metadata(molecule),
        dependencies=DependencyVersions.current(),
    )


def reconstruct_one_electron_molecule(reference: OneElectronAOReference) -> Any:
    """Rebuild and authenticate the live PySCF integral container without SCF."""

    if not isinstance(reference, OneElectronAOReference):
        raise TypeError("reference must be OneElectronAOReference")
    _validate_runtime(reference.dependencies)
    molecule = _build_molecule(reference.config)
    core = build_core_operators(molecule)
    if core.fingerprint_sha256 != reference.core_operators.fingerprint_sha256:
        raise ReferencePreparationError(
            "live molecule does not reproduce the one-electron core-operator fingerprint"
        )
    topology = build_anchor_topology(molecule, core.nuclei)
    if topology.fingerprint_sha256 != reference.anchor_topology.fingerprint_sha256:
        raise ReferencePreparationError(
            "live molecule does not reproduce the one-electron anchor-topology fingerprint"
        )
    metadata = _basis_metadata(molecule)
    if metadata.fingerprint_sha256 != reference.basis_metadata.fingerprint_sha256:
        raise ReferencePreparationError(
            "live molecule does not reproduce the one-electron AO-basis fingerprint"
        )
    return molecule


def _build_molecule(config: OneElectronReferenceConfig) -> Any:
    from pyscf import gto

    molecule = gto.Mole()
    molecule.atom = [
        (atom.symbol, tuple(float(value) for value in atom.position_au))
        for atom in config.atoms
    ]
    molecule.unit = "Bohr"
    molecule.basis = config.basis
    molecule.charge = 0
    # PySCF requires an electron-spin parity even though no electronic state
    # is constructed here.  This derived value affects neither AO functions
    # nor the one-electron integrals used by the qualification reference.
    molecule.spin = sum(int(gto.charge(atom.symbol)) for atom in config.atoms) % 2
    molecule.cart = False
    molecule.symmetry = False
    molecule.verbose = 0
    molecule.build()
    if molecule.ecp:
        raise UnsupportedConfigurationError(
            "PySCF resolved an ECP/nonlocal ionic model; the one-electron reference "
            "requires all-electron local nuclei"
        )
    if bool(molecule.cart):
        raise UnsupportedConfigurationError(
            "the one-electron reference requires real spherical Gaussian AOs"
        )
    return molecule


def _basis_metadata(molecule: Any) -> AOBasisMetadata:
    nbas = int(molecule.nbas)
    return AOBasisMetadata(
        ao_labels=tuple(str(value) for value in molecule.ao_labels()),
        shell_to_atom=np.asarray(
            [molecule.bas_atom(index) for index in range(nbas)], dtype=np.int64
        ),
        shell_angular_momenta=np.asarray(
            [molecule.bas_angular(index) for index in range(nbas)], dtype=np.int64
        ),
        shell_primitive_counts=np.asarray(
            [molecule.bas_nprim(index) for index in range(nbas)], dtype=np.int64
        ),
        shell_contraction_counts=np.asarray(
            [molecule.bas_nctr(index) for index in range(nbas)], dtype=np.int64
        ),
        ao_locations=np.asarray(molecule.ao_loc_nr(cart=False), dtype=np.int64),
        spherical=True,
    )


def _validate_runtime(expected: DependencyVersions) -> None:
    actual = DependencyVersions.current()
    expected_mapping = expected.as_mapping()
    actual_mapping = actual.as_mapping()
    mismatched = [
        name for name, value in expected_mapping.items() if value != actual_mapping[name]
    ]
    if mismatched:
        details = ", ".join(
            f"{name}={expected_mapping[name]!r} (current {actual_mapping[name]!r})"
            for name in mismatched
        )
        raise ReferencePreparationError(
            f"one-electron AO reference dependency mismatch: {details}"
        )
