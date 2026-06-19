"""Geometry helpers for para-nitroaniline examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def unit_from_degrees(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def rotate_xy(vec: np.ndarray, angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    rot = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rot @ vec


def para_nitroaniline_xyz() -> str:
    """Return a simple planar para-nitroaniline geometry in Angstrom."""

    cc = 1.397
    ch = 1.084
    c_nitro = 1.47
    no = 1.23
    c_amino = 1.40
    nh = 1.01

    atoms: list[tuple[str, np.ndarray]] = []
    angles = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    carbons = [cc * unit_from_degrees(angle) for angle in angles]
    for carbon in carbons:
        atoms.append(("C", carbon))

    # Nitro group on C0.
    nitro_dir = unit_from_degrees(0.0)
    nitro_n = carbons[0] + c_nitro * nitro_dir
    atoms.append(("N", nitro_n))
    atoms.append(("O", nitro_n + no * rotate_xy(nitro_dir, 60.0)))
    atoms.append(("O", nitro_n + no * rotate_xy(nitro_dir, -60.0)))

    # Amino group on the para carbon C3.
    amino_dir = unit_from_degrees(180.0)
    amino_n = carbons[3] + c_amino * amino_dir
    atoms.append(("N", amino_n))
    atoms.append(("H", amino_n + nh * rotate_xy(amino_dir, 60.0)))
    atoms.append(("H", amino_n + nh * rotate_xy(amino_dir, -60.0)))

    # Ring hydrogens on the four unsubstituted carbons.
    for i in (1, 2, 4, 5):
        direction = carbons[i] / np.linalg.norm(carbons[i])
        atoms.append(("H", carbons[i] + ch * direction))

    return format_atom_block(atoms)


def parse_xyz(text: str) -> list[tuple[str, np.ndarray]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].isdigit():
        natom = int(lines[0])
        lines = lines[2 : 2 + natom]

    atoms = []
    for line in lines:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"invalid XYZ line: {line!r}")
        atoms.append((fields[0], np.array([float(x) for x in fields[1:4]])))
    return atoms


def format_atom_block(atoms: list[tuple[str, np.ndarray]]) -> str:
    return "\n".join(
        f"{sym} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}"
        for sym, xyz in atoms
    )


def format_xyz(atoms: list[tuple[str, np.ndarray]], comment: str) -> str:
    return f"{len(atoms)}\n{comment}\n{format_atom_block(atoms)}\n"


def read_xyz(path: Path) -> list[tuple[str, np.ndarray]]:
    return parse_xyz(path.read_text())


def write_xyz(path: Path, atoms: list[tuple[str, np.ndarray]], comment: str) -> None:
    path.write_text(format_xyz(atoms, comment))

