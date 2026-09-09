"""Immutable prepared-reference numerical records."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from os import PathLike
from typing import TYPE_CHECKING, Any

import numpy as np

from aion.config import FormulationKind, ReferenceConfig, canonical_sha256
from aion.errors import ReferencePreparationError

if TYPE_CHECKING:
    from aion.backends import Workspace
    from aion.config import BackendConfig

REFERENCE_CONSTRUCTION_SCHEMA = "aion.prepared-reference"
REFERENCE_CONSTRUCTION_VERSION = "1.0.0"
CORE_OPERATOR_SCHEMA = "aion.core-operators"
CORE_OPERATOR_VERSION = "1.0.0"
ANCHOR_TOPOLOGY_SCHEMA = "aion.anchor-topology"
ANCHOR_TOPOLOGY_VERSION = "1.0.0"
E1_OPERATOR_SCHEMA = "aion.e1-operators"
E1_OPERATOR_VERSION = "1.0.0"
GRID_SCHEMA = "aion.pyscf-grid"
GRID_VERSION = "1.0.0"


def immutable_array(
    value: object,
    *,
    dtype: Any | None = None,
    ndim: int | None = None,
    name: str,
) -> np.ndarray:
    """Return an owning, C-contiguous, finite, read-only NumPy array."""

    array = np.array(value, dtype=dtype, order="C", copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ReferencePreparationError(f"{name} must have {ndim} dimensions")
    if array.dtype.kind in "biufc" and not np.all(np.isfinite(array)):
        raise ReferencePreparationError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def _hermiticity_residual(value: np.ndarray) -> float:
    adjoint = np.swapaxes(value.conj(), -1, -2)
    return float(np.linalg.norm(value - adjoint) / max(1.0, np.linalg.norm(value)))


@dataclass(frozen=True, slots=True)
class DependencyVersions:
    python: str
    numpy: str
    scipy: str
    pyscf: str
    h5py: str
    platform: str

    @classmethod
    def current(cls) -> DependencyVersions:
        import h5py
        import pyscf
        import scipy

        return cls(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            pyscf=pyscf.__version__,
            h5py=h5py.__version__,
            platform=sys.platform,
        )

    def as_mapping(self) -> dict[str, str]:
        return {
            "python": self.python,
            "numpy": self.numpy,
            "scipy": self.scipy,
            "pyscf": self.pyscf,
            "h5py": self.h5py,
            "platform": self.platform,
        }


@dataclass(frozen=True, slots=True)
class NuclearData:
    symbols: tuple[str, ...]
    charges: np.ndarray
    coordinates_au: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbols", tuple(self.symbols))
        charges = immutable_array(self.charges, dtype=np.float64, ndim=1, name="nuclear charges")
        coordinates = immutable_array(
            self.coordinates_au,
            dtype=np.float64,
            ndim=2,
            name="nuclear coordinates",
        )
        if coordinates.shape != (charges.size, 3):
            raise ReferencePreparationError("nuclear coordinates must have shape (natom, 3)")
        if len(self.symbols) != charges.size:
            raise ReferencePreparationError("nuclear symbols and charges have different lengths")
        object.__setattr__(self, "charges", charges)
        object.__setattr__(self, "coordinates_au", coordinates)


@dataclass(frozen=True, slots=True)
class GroundState:
    coefficients: np.ndarray
    occupations: np.ndarray
    orbital_energies_au: np.ndarray
    density: np.ndarray
    energy_total_au: float
    electron_count: float

    def __post_init__(self) -> None:
        coefficients = immutable_array(
            self.coefficients, dtype=np.float64, ndim=2, name="ground-state coefficients"
        )
        occupations = immutable_array(
            self.occupations, dtype=np.float64, ndim=1, name="ground-state occupations"
        )
        orbital_energies = immutable_array(
            self.orbital_energies_au,
            dtype=np.float64,
            ndim=1,
            name="orbital energies",
        )
        density = immutable_array(
            self.density, dtype=np.float64, ndim=2, name="ground-state AO density"
        )
        nao, nmo = coefficients.shape
        if occupations.shape != (nmo,) or orbital_energies.shape != (nmo,):
            raise ReferencePreparationError("MO coefficient, occupation, and energy sizes disagree")
        if density.shape != (nao, nao):
            raise ReferencePreparationError("ground-state density has the wrong AO shape")
        if _hermiticity_residual(density) > 1.0e-12:
            raise ReferencePreparationError("ground-state AO density is not Hermitian")
        energy = float(self.energy_total_au)
        electron_count = float(self.electron_count)
        if not np.isfinite(energy) or not np.isfinite(electron_count):
            raise ReferencePreparationError("ground-state scalar data must be finite")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "occupations", occupations)
        object.__setattr__(self, "orbital_energies_au", orbital_energies)
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "energy_total_au", energy)
        object.__setattr__(self, "electron_count", electron_count)


@dataclass(frozen=True, slots=True)
class QuadratureGrid:
    coordinates_au: np.ndarray
    weights_au: np.ndarray
    level: int
    pruning: str
    fingerprint_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        coordinates = immutable_array(
            self.coordinates_au, dtype=np.float64, ndim=2, name="grid coordinates"
        )
        weights = immutable_array(self.weights_au, dtype=np.float64, ndim=1, name="grid weights")
        if coordinates.shape != (weights.size, 3):
            raise ReferencePreparationError("grid coordinates must have shape (npoint, 3)")
        if weights.size == 0:
            raise ReferencePreparationError("quadrature grid cannot be empty")
        object.__setattr__(self, "coordinates_au", coordinates)
        object.__setattr__(self, "weights_au", weights)
        fingerprint = canonical_sha256(
            {
                "schema": GRID_SCHEMA,
                "version": GRID_VERSION,
                "level": self.level,
                "pruning": self.pruning,
                "coordinates_au": coordinates,
                "weights_au": weights,
            }
        )
        object.__setattr__(self, "fingerprint_sha256", fingerprint)


@dataclass(frozen=True, slots=True)
class CoreOperatorBundle:
    overlap: np.ndarray
    kinetic: np.ndarray
    nuclear_attraction: np.ndarray
    position: np.ndarray
    canonical_momentum: np.ndarray
    nuclei: NuclearData
    momentum_convention: str = "p=i*hbar*(nabla_bra|ket), hbar=1"
    fingerprint_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        overlap = immutable_array(self.overlap, dtype=np.float64, ndim=2, name="overlap")
        nao = overlap.shape[0]
        if overlap.shape != (nao, nao):
            raise ReferencePreparationError("overlap must be square")
        values: dict[str, np.ndarray] = {"overlap": overlap}
        for name in ("kinetic", "nuclear_attraction"):
            value = immutable_array(
                getattr(self, name), dtype=np.float64, ndim=2, name=name.replace("_", " ")
            )
            if value.shape != (nao, nao):
                raise ReferencePreparationError(f"{name} has the wrong AO shape")
            values[name] = value
        position = immutable_array(self.position, dtype=np.float64, ndim=3, name="position")
        momentum = immutable_array(
            self.canonical_momentum,
            dtype=np.complex128,
            ndim=3,
            name="canonical momentum",
        )
        if position.shape != (3, nao, nao) or momentum.shape != (3, nao, nao):
            raise ReferencePreparationError("Cartesian AO operators must have shape (3, nao, nao)")
        for name, value in (*values.items(), ("position", position), ("momentum", momentum)):
            if _hermiticity_residual(value) > 1.0e-10:
                raise ReferencePreparationError(f"{name} operator is not Hermitian")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "canonical_momentum", momentum)
        fingerprint = canonical_sha256(
            {
                "schema": CORE_OPERATOR_SCHEMA,
                "version": CORE_OPERATOR_VERSION,
                "overlap": overlap,
                "kinetic": values["kinetic"],
                "nuclear_attraction": values["nuclear_attraction"],
                "position": position,
                "canonical_momentum": momentum,
                "momentum_convention": self.momentum_convention,
                "nuclear_symbols": self.nuclei.symbols,
                "nuclear_charges": self.nuclei.charges,
                "nuclear_coordinates_au": self.nuclei.coordinates_au,
            }
        )
        object.__setattr__(self, "fingerprint_sha256", fingerprint)

    @property
    def nao(self) -> int:
        return int(self.overlap.shape[0])


@dataclass(frozen=True, slots=True)
class AnchorTopologyBundle:
    ao_to_atom: np.ndarray
    pair_indices: np.ndarray
    pair_displacements_au: np.ndarray
    incidence: np.ndarray
    site_projector_diagonals: np.ndarray
    fingerprint_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        ao_to_atom = immutable_array(
            self.ao_to_atom, dtype=np.int64, ndim=1, name="AO-to-atom anchors"
        )
        pairs = immutable_array(self.pair_indices, dtype=np.int64, ndim=2, name="atom pairs")
        displacements = immutable_array(
            self.pair_displacements_au,
            dtype=np.float64,
            ndim=2,
            name="pair displacements",
        )
        incidence = immutable_array(self.incidence, dtype=np.float64, ndim=2, name="incidence")
        projectors = immutable_array(
            self.site_projector_diagonals,
            dtype=np.float64,
            ndim=2,
            name="site projector diagonals",
        )
        if pairs.shape[1:] != (2,) or displacements.shape != (pairs.shape[0], 3):
            raise ReferencePreparationError("pair topology shapes are inconsistent")
        natom = projectors.shape[0]
        if projectors.shape[1:] != (ao_to_atom.size,):
            raise ReferencePreparationError("site projectors have the wrong AO shape")
        if incidence.shape != (natom, pairs.shape[0]):
            raise ReferencePreparationError("incidence matrix has the wrong topology shape")
        if ao_to_atom.size == 0 or np.any(ao_to_atom < 0) or np.any(ao_to_atom >= natom):
            raise ReferencePreparationError("AO-to-atom anchors contain invalid indices")
        if pairs.size and (np.any(pairs[:, 0] >= pairs[:, 1]) or np.any(pairs < 0)):
            raise ReferencePreparationError("atom pairs must be stored once with a < b")
        expected_pairs = np.asarray(
            [(a, b) for a in range(natom) for b in range(a + 1, natom)],
            dtype=np.int64,
        ).reshape(-1, 2)
        if not np.array_equal(pairs, expected_pairs):
            raise ReferencePreparationError("atom pairs must contain every a < b pair in order")
        expected_incidence = np.zeros_like(incidence)
        for pair_index, (a, b) in enumerate(pairs):
            expected_incidence[a, pair_index] = 1.0
            expected_incidence[b, pair_index] = -1.0
        if not np.array_equal(incidence, expected_incidence):
            raise ReferencePreparationError(
                "incidence does not use the required current orientation"
            )
        expected_projectors = np.equal(np.arange(natom)[:, None], ao_to_atom[None, :]).astype(
            np.float64
        )
        if not np.array_equal(projectors, expected_projectors):
            raise ReferencePreparationError("site projector diagonals disagree with AO anchors")
        object.__setattr__(self, "ao_to_atom", ao_to_atom)
        object.__setattr__(self, "pair_indices", pairs)
        object.__setattr__(self, "pair_displacements_au", displacements)
        object.__setattr__(self, "incidence", incidence)
        object.__setattr__(self, "site_projector_diagonals", projectors)
        object.__setattr__(
            self,
            "fingerprint_sha256",
            canonical_sha256(
                {
                    "schema": ANCHOR_TOPOLOGY_SCHEMA,
                    "version": ANCHOR_TOPOLOGY_VERSION,
                    "ao_to_atom": ao_to_atom,
                    "pair_indices": pairs,
                    "pair_displacements_au": displacements,
                    "incidence": incidence,
                    "site_projector_diagonals": projectors,
                    "link_orientation": "b_to_a_for_a_less_than_b",
                    "positive_current": "a_to_b",
                }
            ),
        )

    @property
    def natom(self) -> int:
        return int(self.site_projector_diagonals.shape[0])


@dataclass(frozen=True, slots=True)
class E1OperatorBundle:
    """Independently fingerprinted field-free central-dipole operators."""

    central_dipoles: np.ndarray
    source_operator_fingerprint_sha256: str
    fingerprint_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        dipoles = immutable_array(
            self.central_dipoles,
            dtype=np.complex128,
            ndim=3,
            name="E1 central dipoles",
        )
        if dipoles.shape[0] != 3 or dipoles.shape[1] != dipoles.shape[2]:
            raise ReferencePreparationError("E1 central dipoles must have shape (3, nao, nao)")
        if _hermiticity_residual(dipoles) > 1.0e-12:
            raise ReferencePreparationError("E1 central dipoles are not Hermitian")
        object.__setattr__(self, "central_dipoles", dipoles)
        object.__setattr__(
            self,
            "fingerprint_sha256",
            canonical_sha256(
                {
                    "schema": E1_OPERATOR_SCHEMA,
                    "version": E1_OPERATOR_VERSION,
                    "central_dipoles": dipoles,
                    "source_operator_fingerprint_sha256": (self.source_operator_fingerprint_sha256),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class PreparedReference:
    """Portable immutable static electronic problem shared by formulations."""

    config: ReferenceConfig
    ground_state: GroundState
    grid: QuadratureGrid
    core_operators: CoreOperatorBundle
    anchor_topology: AnchorTopologyBundle
    dependencies: DependencyVersions
    preparation_backend: str
    fingerprint_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.ground_state.density.shape != self.core_operators.overlap.shape:
            raise ReferencePreparationError("ground state and operator AO dimensions disagree")
        configured_symbols = tuple(atom.symbol for atom in self.config.molecule.atoms)
        configured_coordinates = np.asarray(
            [atom.position_au for atom in self.config.molecule.atoms], dtype=np.float64
        )
        if configured_symbols != self.core_operators.nuclei.symbols or not np.array_equal(
            configured_coordinates, self.core_operators.nuclei.coordinates_au
        ):
            raise ReferencePreparationError("configured molecule and stored nuclear data disagree")
        overlap_count = float(
            np.einsum("ij,ji->", self.ground_state.density, self.core_operators.overlap).real
        )
        if abs(overlap_count - self.ground_state.electron_count) > 1.0e-8:
            raise ReferencePreparationError(
                "ground-state density has an inconsistent electron count"
            )
        if abs(float(np.sum(self.ground_state.occupations)) - overlap_count) > 1.0e-8:
            raise ReferencePreparationError("ground-state occupations have an inconsistent count")
        reconstructed_density = np.einsum(
            "mi,i,ni->mn",
            self.ground_state.coefficients,
            self.ground_state.occupations,
            self.ground_state.coefficients.conj(),
            optimize=True,
        )
        density_residual = float(
            np.linalg.norm(reconstructed_density - self.ground_state.density)
            / max(1.0, np.linalg.norm(self.ground_state.density))
        )
        if density_residual > 1.0e-10:
            raise ReferencePreparationError(
                "ground-state density disagrees with coefficients and occupations"
            )
        orbital_metric = (
            self.ground_state.coefficients.conj().T
            @ self.core_operators.overlap
            @ self.ground_state.coefficients
        )
        metric_residual = float(
            np.linalg.norm(orbital_metric - np.eye(orbital_metric.shape[0]))
            / max(1.0, np.linalg.norm(orbital_metric))
        )
        if metric_residual > 1.0e-9:
            raise ReferencePreparationError("ground-state orbitals are not overlap-orthonormal")
        fingerprint = canonical_sha256(
            {
                "schema": REFERENCE_CONSTRUCTION_SCHEMA,
                "version": REFERENCE_CONSTRUCTION_VERSION,
                "config": self.config.scientific_mapping(),
                "dependencies": self.dependencies.as_mapping(),
                "ground_state": {
                    "coefficients": self.ground_state.coefficients,
                    "occupations": self.ground_state.occupations,
                    "orbital_energies_au": self.ground_state.orbital_energies_au,
                    "density": self.ground_state.density,
                    "energy_total_au": self.ground_state.energy_total_au,
                    "electron_count": self.ground_state.electron_count,
                },
                "grid_fingerprint_sha256": self.grid.fingerprint_sha256,
                "core_operator_fingerprint_sha256": self.core_operators.fingerprint_sha256,
                "anchor_topology_fingerprint_sha256": self.anchor_topology.fingerprint_sha256,
            }
        )
        object.__setattr__(self, "fingerprint_sha256", fingerprint)

    @property
    def supported_formulations(self) -> tuple[FormulationKind, ...]:
        return tuple(FormulationKind)

    def save(self, path: str | PathLike[str] | None = None) -> None:
        """Transactionally publish this reference; implemented lazily to avoid I/O cycles."""

        from aion.electronic_structure.reference_io import save_reference

        target = self.config.output.artifact_path if path is None else path
        save_reference(self, target)

    def create_workspace(self, backend_config: BackendConfig | None = None) -> Workspace:
        """Reconstruct live backend objects without running SCF again."""

        from aion.electronic_structure.pyscf_rks import create_reference_workspace

        return create_reference_workspace(self, backend_config)
