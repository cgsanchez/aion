"""Small wrappers around PySCF linear-response TDDFT."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class Excitation:
    """One linear-response excitation from PySCF TDDFT/TDA."""

    index: int
    energy: float
    oscillator_strength: float
    transition_dipole: np.ndarray

    @property
    def energy_ev(self) -> float:
        return self.energy * HARTREE_TO_EV

    @property
    def dipole_norm(self) -> float:
        return float(np.linalg.norm(self.transition_dipole))

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "energy": self.energy,
            "energy_ev": self.energy_ev,
            "oscillator_strength": self.oscillator_strength,
            "transition_dipole": self.transition_dipole.tolist(),
            "dipole_norm": self.dipole_norm,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Excitation":
        return cls(
            index=int(data["index"]),
            energy=float(data["energy"]),
            oscillator_strength=float(data["oscillator_strength"]),
            transition_dipole=np.asarray(data["transition_dipole"], dtype=float),
        )


def casida_excitations(
    mf,
    *,
    nstates: int = 10,
    singlet: bool = True,
    conv_tol: float | None = None,
) -> tuple[object, list[Excitation]]:
    """Run PySCF LR-TDDFT and collect energies plus dipole diagnostics.

    For non-hybrid RKS references, PySCF's ``mf.TDDFT()`` dispatches to its
    CasidaTDDFT implementation. For hybrids, PySCF uses its general full TDDFT
    solver behind the same public API.
    """

    td = mf.TDDFT()
    td.nstates = nstates
    td.singlet = singlet
    if conv_tol is not None:
        td.conv_tol = conv_tol
    td.kernel()

    transition_dipoles = np.asarray(td.transition_dipole(), dtype=float)
    oscillator_strengths = np.asarray(td.oscillator_strength(), dtype=float)
    energies = np.asarray(td.e, dtype=float)

    excitations = [
        Excitation(
            index=i,
            energy=float(energy),
            oscillator_strength=float(oscillator_strengths[i]),
            transition_dipole=transition_dipoles[i],
        )
        for i, energy in enumerate(energies)
    ]
    return td, excitations


def lowest_active_excitation(
    excitations: list[Excitation],
    *,
    oscillator_threshold: float = 1.0e-6,
    dipole_threshold: float = 1.0e-6,
) -> Excitation:
    """Return the lowest excitation that can be driven by an electric dipole."""

    for excitation in excitations:
        if (
            excitation.oscillator_strength > oscillator_threshold
            and excitation.dipole_norm > dipole_threshold
        ):
            return excitation
    raise ValueError("no dipole-active excitation found in the supplied roots")


def excitation_by_index(excitations: list[Excitation], index: int) -> Excitation:
    """Return one excitation by its saved root index."""

    for excitation in excitations:
        if excitation.index == index:
            return excitation
    raise ValueError(f"excitation index {index} was not found")


def transition_polarization(excitation: Excitation) -> np.ndarray:
    """Unit vector along a transition dipole."""

    norm = excitation.dipole_norm
    if norm == 0:
        raise ValueError("transition dipole is zero")
    return excitation.transition_dipole / norm


def save_excitations(
    path: str | Path,
    excitations: list[Excitation],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write LR-TDDFT excitation data to a JSON file."""

    payload = {
        "metadata": {} if metadata is None else metadata,
        "excitations": [excitation.to_dict() for excitation in excitations],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_excitations(path: str | Path) -> tuple[dict[str, Any], list[Excitation]]:
    """Read LR-TDDFT excitation data from a JSON file."""

    payload = json.loads(Path(path).read_text())
    metadata = dict(payload.get("metadata", {}))
    excitations = [
        Excitation.from_dict(item) for item in payload.get("excitations", [])
    ]
    return metadata, excitations
