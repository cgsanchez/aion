"""PySCF restricted-Kohn--Sham reference preparation and reconstruction."""

from __future__ import annotations

from typing import Any

import numpy as np

from aion.backends import ArrayBackend, Workspace, make_backend
from aion.config import BackendConfig, BackendKind, GridPruning, ReferenceConfig, XCFamily
from aion.electronic_structure.data import (
    DependencyVersions,
    GroundState,
    PreparedReference,
    QuadratureGrid,
)
from aion.electronic_structure.operators import build_anchor_topology, build_core_operators
from aion.errors import BackendError, ReferencePreparationError, UnsupportedConfigurationError


def _build_molecule(config: ReferenceConfig) -> Any:
    from pyscf import gto

    molecule = gto.Mole()
    molecule.atom = [
        (atom.symbol, tuple(float(value) for value in atom.position_au))
        for atom in config.molecule.atoms
    ]
    molecule.unit = "Bohr"
    molecule.basis = config.electronic_structure.basis
    molecule.charge = config.molecule.charge
    molecule.spin = config.molecule.spin
    molecule.symmetry = False
    molecule.verbose = 0
    molecule.build()
    if molecule.ecp:
        raise UnsupportedConfigurationError(
            "PySCF resolved an ECP/nonlocal ionic model; "
            "Aion 0.2 requires all-electron local nuclei"
        )
    if molecule.nelectron % 2:
        raise UnsupportedConfigurationError(
            "restricted closed-shell RKS requires an even electron count"
        )
    return molecule


def _validate_functional(config: ReferenceConfig) -> None:
    from pyscf.dft import libxc

    functional = config.electronic_structure.functional
    try:
        actual_type = str(libxc.xc_type(functional)).upper()
        hybrid = bool(libxc.is_hybrid_xc(functional))
        nonlocal_correlation = bool(libxc.is_nlc(functional))
    except Exception as exc:
        raise UnsupportedConfigurationError(
            f"PySCF/libxc cannot resolve functional {functional!r}"
        ) from exc
    expected = config.electronic_structure.xc_family.value.upper()
    if actual_type != expected:
        raise UnsupportedConfigurationError(
            f"functional {functional!r} is {actual_type}, but the configuration declares {expected}"
        )
    if hybrid:
        raise UnsupportedConfigurationError(
            "hybrid and exact-exchange functionals are out of scope"
        )
    if nonlocal_correlation:
        raise UnsupportedConfigurationError("nonlocal correlation functionals are out of scope")
    if config.electronic_structure.xc_family not in {XCFamily.LDA, XCFamily.GGA}:
        raise UnsupportedConfigurationError("only pure LDA and GGA functionals are supported")


def _build_cpu_mean_field(config: ReferenceConfig, molecule: Any) -> Any:
    from pyscf import dft

    mean_field = dft.RKS(molecule)
    if config.electronic_structure.density_fitting:
        auxiliary_basis = config.electronic_structure.auxiliary_basis
        assert auxiliary_basis is not None
        mean_field = mean_field.density_fit(auxbasis=auxiliary_basis)
    mean_field.xc = config.electronic_structure.functional
    mean_field.grids.level = config.electronic_structure.grid_level
    if config.electronic_structure.grid_pruning is GridPruning.NONE:
        mean_field.grids.prune = None
    mean_field.conv_tol = config.electronic_structure.scf_energy_tolerance_au
    mean_field.max_cycle = config.electronic_structure.scf_max_iterations
    return mean_field


def _to_host(value: object, backend: ArrayBackend) -> np.ndarray:
    if hasattr(value, "get"):
        return backend.to_host(value)
    return np.asarray(value)


def prepare_pyscf_reference(config: ReferenceConfig) -> PreparedReference:
    """Run exactly one validated CPU or GPU RKS calculation and freeze its result."""

    _validate_functional(config)
    molecule = _build_molecule(config)
    core = build_core_operators(molecule)
    topology = build_anchor_topology(molecule, core.nuclei)
    backend = make_backend(config.backend)
    mean_field = _build_cpu_mean_field(config, molecule)
    mean_field.grids.build(with_non0tab=True)
    canonical_grid_coordinates = np.array(mean_field.grids.coords, copy=True)
    canonical_grid_weights = np.array(mean_field.grids.weights, copy=True)
    if config.backend.kind is BackendKind.GPU:
        try:
            mean_field = mean_field.to_gpu()
            mean_field.grids.coords = backend.asarray(canonical_grid_coordinates)
            mean_field.grids.weights = backend.asarray(canonical_grid_weights)
        except Exception as exc:
            raise BackendError(
                "GPU RKS preparation requested but GPU4PySCF initialization failed"
            ) from exc
    try:
        energy = float(mean_field.kernel())
    except Exception as exc:
        raise ReferencePreparationError("PySCF RKS reference calculation failed") from exc
    if not bool(mean_field.converged):
        raise ReferencePreparationError(
            f"PySCF RKS did not converge in {config.electronic_structure.scf_max_iterations} cycles"
        )
    coefficients = _to_host(mean_field.mo_coeff, backend)
    occupations = _to_host(mean_field.mo_occ, backend)
    orbital_energies = _to_host(mean_field.mo_energy, backend)
    density = _to_host(mean_field.make_rdm1(), backend)
    coords = _to_host(mean_field.grids.coords, backend)
    weights = _to_host(mean_field.grids.weights, backend)
    overlap_count = float(np.einsum("ij,ji->", density, core.overlap).real)
    pruning = getattr(getattr(mean_field.grids, "prune", None), "__name__", "none")
    reference = PreparedReference(
        config=config,
        ground_state=GroundState(
            coefficients=coefficients,
            occupations=occupations,
            orbital_energies_au=orbital_energies,
            density=density,
            energy_total_au=energy,
            electron_count=overlap_count,
        ),
        grid=QuadratureGrid(
            coordinates_au=coords,
            weights_au=weights,
            level=config.electronic_structure.grid_level,
            pruning=pruning,
        ),
        core_operators=core,
        anchor_topology=topology,
        dependencies=DependencyVersions.current(),
        preparation_backend=(
            "cpu"
            if config.backend.kind is BackendKind.CPU
            else f"gpu:{config.backend.device_index}"
        ),
    )
    backend.synchronize()
    return reference


def validate_reference_runtime(reference: PreparedReference) -> None:
    """Require the dependency stack recorded by a reference before live reconstruction."""

    current = DependencyVersions.current()
    expected = reference.dependencies.as_mapping()
    actual = current.as_mapping()
    mismatched = [name for name in expected if expected[name] != actual[name]]
    if mismatched:
        details = ", ".join(
            f"{name}={expected[name]!r} (current {actual[name]!r})" for name in mismatched
        )
        raise ReferencePreparationError(f"prepared-reference dependency mismatch: {details}")


def reconstruct_mean_field(
    reference: PreparedReference,
    backend_config: BackendConfig,
    *,
    backend: ArrayBackend | None = None,
) -> Any:
    """Rebuild a live PySCF/GPU4PySCF model from immutable data without SCF."""

    validate_reference_runtime(reference)
    if backend is not None and backend.kind is not backend_config.kind:
        raise BackendError("reconstruction backend does not match BackendConfig.kind")
    if (
        backend is not None
        and backend_config.kind is BackendKind.GPU
        and backend.device_index != backend_config.device_index
    ):
        raise BackendError("reconstruction backend does not match BackendConfig.device_index")
    _validate_functional(reference.config)
    molecule = _build_molecule(reference.config)
    reconstructed_core = build_core_operators(molecule)
    if reconstructed_core.fingerprint_sha256 != reference.core_operators.fingerprint_sha256:
        raise ReferencePreparationError(
            "live molecule does not reproduce the stored core-operator fingerprint"
        )
    reconstructed_topology = build_anchor_topology(molecule, reconstructed_core.nuclei)
    if reconstructed_topology.fingerprint_sha256 != reference.anchor_topology.fingerprint_sha256:
        raise ReferencePreparationError(
            "live molecule does not reproduce the stored anchor-topology fingerprint"
        )
    mean_field = _build_cpu_mean_field(reference.config, molecule)
    mean_field.grids.coords = np.array(reference.grid.coordinates_au, copy=True)
    mean_field.grids.weights = np.array(reference.grid.weights_au, copy=True)
    mean_field.grids.non0tab = mean_field.grids.make_mask(molecule, mean_field.grids.coords)
    mean_field.mo_coeff = np.array(reference.ground_state.coefficients, copy=True)
    mean_field.mo_occ = np.array(reference.ground_state.occupations, copy=True)
    mean_field.mo_energy = np.array(reference.ground_state.orbital_energies_au, copy=True)
    mean_field.e_tot = reference.ground_state.energy_total_au
    mean_field.converged = True
    if backend_config.kind is BackendKind.GPU:
        selected_backend = make_backend(backend_config) if backend is None else backend
        if selected_backend.kind is not BackendKind.GPU:
            raise BackendError("GPU reconstruction received a non-GPU array backend")
        try:
            mean_field = mean_field.to_gpu()
            mean_field.grids.coords = selected_backend.asarray(reference.grid.coordinates_au)
            mean_field.grids.weights = selected_backend.asarray(reference.grid.weights_au)
        except Exception as exc:
            raise BackendError("GPU4PySCF reconstruction failed") from exc
    return mean_field


def create_reference_workspace(
    reference: PreparedReference,
    backend_config: object | None = None,
) -> Workspace:
    """Transfer immutable reference arrays once and reconstruct the live model."""

    if backend_config is None:
        selected = reference.config.backend
    elif isinstance(backend_config, BackendConfig):
        selected = backend_config
    else:
        raise TypeError("backend_config must be BackendConfig or None")
    backend = make_backend(selected)
    workspace = Workspace(backend=backend)
    arrays = {
        "ground_state.coefficients": reference.ground_state.coefficients,
        "ground_state.occupations": reference.ground_state.occupations,
        "ground_state.orbital_energies_au": reference.ground_state.orbital_energies_au,
        "ground_state.density": reference.ground_state.density,
        "grid.coordinates_au": reference.grid.coordinates_au,
        "grid.weights_au": reference.grid.weights_au,
        "operators.overlap": reference.core_operators.overlap,
        "operators.kinetic": reference.core_operators.kinetic,
        "operators.nuclear_attraction": reference.core_operators.nuclear_attraction,
        "operators.position": reference.core_operators.position,
        "operators.canonical_momentum": reference.core_operators.canonical_momentum,
        "nuclei.charges": reference.core_operators.nuclei.charges,
        "nuclei.coordinates_au": reference.core_operators.nuclei.coordinates_au,
        "anchors.ao_to_atom": reference.anchor_topology.ao_to_atom,
        "anchors.pair_indices": reference.anchor_topology.pair_indices,
        "anchors.pair_displacements_au": reference.anchor_topology.pair_displacements_au,
        "anchors.incidence": reference.anchor_topology.incidence,
        "anchors.site_projector_diagonals": reference.anchor_topology.site_projector_diagonals,
    }
    for name, value in arrays.items():
        workspace.install_host_array(name, value)
    workspace.electronic_model = reconstruct_mean_field(reference, selected, backend=backend)
    workspace.caches["reference_fingerprint_sha256"] = reference.fingerprint_sha256
    workspace.assert_all_resident()
    return workspace
