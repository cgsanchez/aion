"""AO-to-atom anchor data for Wilson/Peierls geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AOAnchors:
    """Map AO basis functions to atom-centered gauge anchors."""

    atom_coords: np.ndarray
    ao_to_atom: np.ndarray

    def __post_init__(self) -> None:
        coords = np.asarray(self.atom_coords, dtype=float)
        ao_to_atom = np.asarray(self.ao_to_atom, dtype=int)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("atom_coords must have shape (natom, 3)")
        if ao_to_atom.ndim != 1:
            raise ValueError("ao_to_atom must be a one-dimensional array")
        if ao_to_atom.size == 0:
            raise ValueError("ao_to_atom must not be empty")
        if np.any(ao_to_atom < 0) or np.any(ao_to_atom >= coords.shape[0]):
            raise ValueError("ao_to_atom contains invalid atom indices")
        object.__setattr__(self, "atom_coords", coords)
        object.__setattr__(self, "ao_to_atom", ao_to_atom)

    @classmethod
    def from_mol(cls, mol) -> "AOAnchors":
        """Build anchors from a PySCF molecule."""

        coords = np.asarray(mol.atom_coords(unit="Bohr"), dtype=float)
        ao_to_atom = np.empty(mol.nao_nr(), dtype=int)
        for atom_index, row in enumerate(mol.aoslice_by_atom()):
            p0, p1 = int(row[2]), int(row[3])
            ao_to_atom[p0:p1] = atom_index
        return cls(coords, ao_to_atom)

    @property
    def natom(self) -> int:
        return int(self.atom_coords.shape[0])

    @property
    def nao(self) -> int:
        return int(self.ao_to_atom.size)

    @property
    def row_atoms(self) -> np.ndarray:
        return self.ao_to_atom[:, None]

    @property
    def col_atoms(self) -> np.ndarray:
        return self.ao_to_atom[None, :]

    def lift_site_matrix(self, site_matrix: np.ndarray) -> np.ndarray:
        """Expand an atom-site matrix to AO block structure."""

        matrix = np.asarray(site_matrix)
        if matrix.shape != (self.natom, self.natom):
            raise ValueError(f"site_matrix must have shape {(self.natom, self.natom)}")
        return matrix[self.row_atoms, self.col_atoms]

    def lift_site_vector(self, site_vector: np.ndarray) -> np.ndarray:
        """Expand an atom-site vector to AO entries."""

        vector = np.asarray(site_vector)
        if vector.shape != (self.natom,):
            raise ValueError(f"site_vector must have shape {(self.natom,)}")
        return vector[self.ao_to_atom]

    def site_projector_diagonal(self, atom_index: int) -> np.ndarray:
        """Return the diagonal of the AO projector for one atom site."""

        if atom_index < 0 or atom_index >= self.natom:
            raise ValueError("atom_index out of range")
        return (self.ao_to_atom == atom_index).astype(float)
