"""Construction and independent qualification of reference AO operators."""

from __future__ import annotations

from typing import Any

import numpy as np

from aion.electronic_structure.data import (
    AnchorTopologyBundle,
    CoreOperatorBundle,
    E1OperatorBundle,
    NuclearData,
)
from aion.errors import ReferencePreparationError


def build_core_operators(molecule: Any) -> CoreOperatorBundle:
    """Build formulation-neutral all-electron AO operators from a PySCF molecule."""

    overlap = np.asarray(molecule.intor("int1e_ovlp"), dtype=np.float64)
    kinetic = np.asarray(molecule.intor("int1e_kin"), dtype=np.float64)
    nuclear_attraction = np.asarray(molecule.intor("int1e_nuc"), dtype=np.float64)
    with molecule.with_common_origin((0.0, 0.0, 0.0)):
        position = np.asarray(molecule.intor("int1e_r", comp=3), dtype=np.float64)
    gradient_bra = np.asarray(molecule.intor("int1e_ipovlp", comp=3), dtype=np.float64)
    # libcint int1e_ipovlp is (nabla phi_mu | phi_nu). Integration by parts gives
    # <phi_mu|p|phi_nu> = i (nabla phi_mu|phi_nu) for p=-i*nabla and hbar=1.
    momentum = 1j * gradient_bra
    adjoint = np.swapaxes(momentum.conj(), 1, 2)
    residual = float(np.linalg.norm(momentum - adjoint) / max(1.0, np.linalg.norm(momentum)))
    if residual > 1.0e-10:
        raise ReferencePreparationError(
            f"canonical momentum integrals are not Hermitian: residual={residual:.3e}"
        )
    momentum = 0.5 * (momentum + adjoint)
    nuclei = NuclearData(
        symbols=tuple(molecule.atom_symbol(index) for index in range(molecule.natm)),
        charges=np.asarray(molecule.atom_charges(), dtype=np.float64),
        coordinates_au=np.asarray(molecule.atom_coords(unit="Bohr"), dtype=np.float64),
    )
    return CoreOperatorBundle(
        overlap=overlap,
        kinetic=kinetic,
        nuclear_attraction=nuclear_attraction,
        position=position,
        canonical_momentum=momentum,
        nuclei=nuclei,
    )


def build_anchor_topology(molecule: Any, nuclei: NuclearData) -> AnchorTopologyBundle:
    """Build AO anchors, once-oriented atom pairs, incidence, and selectors."""

    nao = int(molecule.nao_nr())
    natom = int(molecule.natm)
    ao_to_atom = np.empty(nao, dtype=np.int64)
    for atom_index, row in enumerate(molecule.aoslice_by_atom()):
        start, stop = int(row[2]), int(row[3])
        ao_to_atom[start:stop] = atom_index
    pair_indices = np.asarray(
        [(a, b) for a in range(natom) for b in range(a + 1, natom)], dtype=np.int64
    ).reshape(-1, 2)
    pair_displacements = np.empty((pair_indices.shape[0], 3), dtype=np.float64)
    incidence = np.zeros((natom, pair_indices.shape[0]), dtype=np.float64)
    for pair_index, (a, b) in enumerate(pair_indices):
        pair_displacements[pair_index] = nuclei.coordinates_au[a] - nuclei.coordinates_au[b]
        # For I_ab > 0 flowing a -> b, div(I) is +I at a and -I at b.
        incidence[a, pair_index] = 1.0
        incidence[b, pair_index] = -1.0
    projectors = np.equal(np.arange(natom)[:, None], ao_to_atom[None, :]).astype(np.float64)
    return AnchorTopologyBundle(
        ao_to_atom=ao_to_atom,
        pair_indices=pair_indices,
        pair_displacements_au=pair_displacements,
        incidence=incidence,
        site_projector_diagonals=projectors,
    )


def build_e1_operators(
    core: CoreOperatorBundle, topology: AnchorTopologyBundle, *, charge: float = -1.0
) -> E1OperatorBundle:
    """Build the field-free pair-central dipoles from immutable core data."""

    ao_coordinates = core.nuclei.coordinates_au[topology.ao_to_atom]
    centers = 0.5 * (ao_coordinates[:, None, :] + ao_coordinates[None, :, :])
    central = np.empty_like(core.position, dtype=np.complex128)
    for axis in range(3):
        raw = float(charge) * (core.position[axis] - centers[:, :, axis] * core.overlap)
        central[axis] = 0.5 * (raw + raw.conj().T)
    return E1OperatorBundle(
        central_dipoles=central,
        source_operator_fingerprint_sha256=core.fingerprint_sha256,
    )


def momentum_grid_residual(
    molecule: Any,
    coordinates_au: np.ndarray,
    weights_au: np.ndarray,
    canonical_momentum: np.ndarray,
) -> float:
    """Compare analytic momentum with an independent real-space derivative integral.

    PySCF's AO derivative evaluator acts on the ket in this contraction, whereas
    the stored analytic integral is built from libcint's bra derivative. Their
    agreement therefore qualifies both the sign and Hermitian orientation.
    """

    coordinates = np.asarray(coordinates_au, dtype=np.float64)
    weights = np.asarray(weights_au, dtype=np.float64)
    ao = np.asarray(molecule.eval_gto("GTOval_sph", coordinates), dtype=np.float64)
    gradients = np.asarray(molecule.eval_gto("GTOval_ip_sph", coordinates), dtype=np.float64)
    numerical = -1j * np.einsum("g,gm,xgn->xmn", weights, ao, gradients, optimize=True)
    scale = max(1.0, float(np.linalg.norm(canonical_momentum)))
    residual = float(np.linalg.norm(numerical - canonical_momentum) / scale)
    if not np.isfinite(residual):
        raise ReferencePreparationError("momentum derivative qualification is non-finite")
    return residual
