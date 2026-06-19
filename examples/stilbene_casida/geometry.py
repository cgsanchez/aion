"""Geometry helpers for the stilbene examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def unit_from_degrees(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def trans_stilbene_xyz() -> str:
    """Return a simple planar trans-stilbene geometry in Angstrom."""

    c_double = 1.34
    c_sp2_caryl = 1.47
    cc = 1.397
    ch = 1.084
    ch_vinyl = 1.09

    atoms: list[tuple[str, np.ndarray]] = []
    left_vinyl = np.array([-0.5 * c_double, 0.0, 0.0])
    right_vinyl = np.array([0.5 * c_double, 0.0, 0.0])
    atoms.append(("C", left_vinyl))
    atoms.append(("C", right_vinyl))
    atoms.append(("H", left_vinyl + ch_vinyl * unit_from_degrees(210.0)))
    atoms.append(("H", right_vinyl + ch_vinyl * unit_from_degrees(30.0)))

    def add_phenyl(vinyl_carbon: np.ndarray, bond_angle: float) -> None:
        bond_dir = unit_from_degrees(bond_angle)
        ipso = vinyl_carbon + c_sp2_caryl * bond_dir
        radial = -bond_dir
        center = ipso - cc * radial
        radial_angle = np.rad2deg(np.arctan2(radial[1], radial[0]))
        ring_angles = np.deg2rad(
            [
                radial_angle,
                radial_angle + 60.0,
                radial_angle + 120.0,
                radial_angle + 180.0,
                radial_angle + 240.0,
                radial_angle + 300.0,
            ]
        )
        carbons = [
            center + np.array([cc * np.cos(angle), cc * np.sin(angle), 0.0])
            for angle in ring_angles
        ]
        for carbon in carbons:
            atoms.append(("C", carbon))
        for i, carbon in enumerate(carbons):
            if i == 0:
                continue
            direction = carbon - center
            direction /= np.linalg.norm(direction)
            atoms.append(("H", carbon + ch * direction))

    add_phenyl(left_vinyl, 150.0)
    add_phenyl(right_vinyl, 330.0)
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

