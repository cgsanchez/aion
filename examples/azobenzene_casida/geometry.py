"""Geometry helpers for the azobenzene Casida example."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def generated_trans_azobenzene_xyz(phenyl_twist_degrees: float = 30.0) -> str:
    """Return a simple generated trans-azobenzene geometry in Angstrom.

    The phenyl rings are twisted around the C-N axes to avoid the artificial
    near-metallic behavior of the fully planar generated structure. This is
    only a starting geometry; production roots should use an optimized XYZ.
    """

    nn = 1.25
    cn = 1.42
    cc = 1.397
    ch = 1.084

    atoms: list[tuple[str, np.ndarray]] = []
    atoms.append(("N", np.array([-0.5 * nn, 0.0, 0.0])))
    atoms.append(("N", np.array([0.5 * nn, 0.0, 0.0])))

    def rotate_about_x(xyz: np.ndarray, angle_degrees: float) -> np.ndarray:
        angle = np.deg2rad(angle_degrees)
        rot = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        )
        return rot @ xyz

    def add_ring(center: np.ndarray, ipso_angle: float, twist_degrees: float) -> None:
        angles = np.deg2rad(
            [
                ipso_angle,
                ipso_angle + 60.0,
                ipso_angle + 120.0,
                ipso_angle + 180.0,
                ipso_angle + 240.0,
                ipso_angle + 300.0,
            ]
        )
        carbons = [
            center
            + rotate_about_x(
                np.array([cc * np.cos(angle), cc * np.sin(angle), 0.0]),
                twist_degrees,
            )
            for angle in angles
        ]
        for carbon in carbons:
            atoms.append(("C", carbon))
        for i, carbon in enumerate(carbons):
            if i == 0:
                continue
            direction = carbon - center
            direction /= np.linalg.norm(direction)
            atoms.append(("H", carbon + ch * direction))

    n_right = np.array([0.5 * nn, 0.0, 0.0])
    ipso_right = n_right + np.array([cn, 0.0, 0.0])
    center_right = ipso_right + np.array([cc, 0.0, 0.0])
    add_ring(center_right, 180.0, phenyl_twist_degrees)

    n_left = np.array([-0.5 * nn, 0.0, 0.0])
    ipso_left = n_left - np.array([cn, 0.0, 0.0])
    center_left = ipso_left - np.array([cc, 0.0, 0.0])
    add_ring(center_left, 0.0, -phenyl_twist_degrees)

    return format_xyz(atoms)


def parse_xyz(text: str) -> list[tuple[str, np.ndarray]]:
    """Parse either a plain atom block or an XYZ file."""

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


def format_xyz(atoms: list[tuple[str, np.ndarray]], comment: str | None = None) -> str:
    """Format a plain atom block or full XYZ if a comment is provided."""

    body = "\n".join(
        f"{sym} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}"
        for sym, xyz in atoms
    )
    if comment is None:
        return body
    return f"{len(atoms)}\n{comment}\n{body}\n"


def read_xyz(path: Path) -> list[tuple[str, np.ndarray]]:
    return parse_xyz(path.read_text())


def write_xyz(path: Path, atoms: list[tuple[str, np.ndarray]], comment: str) -> None:
    path.write_text(format_xyz(atoms, comment=comment))

