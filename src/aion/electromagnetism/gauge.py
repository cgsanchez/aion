"""Consistent uniform LG/VG and projected P0 node/link gauge data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from aion.electromagnetism.sources import UniformPotentialSample
from aion.errors import SourceCompilationError


class UniformGauge(StrEnum):
    LENGTH = "length"
    VELOCITY = "velocity"


@dataclass(frozen=True, slots=True)
class UniformGaugeSample:
    gauge: UniformGauge
    electric_field: np.ndarray
    vector_potential_reduced: np.ndarray
    vector_potential_reduced_dot: np.ndarray
    node_scalar_potential: np.ndarray
    pair_link: np.ndarray
    pair_link_dot: np.ndarray
    pair_electromotive_potential: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "electric_field",
            "vector_potential_reduced",
            "vector_potential_reduced_dot",
            "node_scalar_potential",
            "pair_link",
            "pair_link_dot",
            "pair_electromotive_potential",
        ):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if not np.all(np.isfinite(value)):
                raise SourceCompilationError(f"{name} contains non-finite values")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def derive_uniform_gauge_sample(
    physical: UniformPotentialSample,
    gauge: UniformGauge,
    atom_coordinates_au: np.ndarray,
    electromagnetic_origin_au: np.ndarray,
    pair_indices: np.ndarray,
) -> UniformGaugeSample:
    """Derive LG or VG from one physical uniform vector-potential sample."""

    coordinates = np.asarray(atom_coordinates_au, dtype=np.float64)
    origin = np.asarray(electromagnetic_origin_au, dtype=np.float64)
    pairs = np.asarray(pair_indices, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
        raise SourceCompilationError("atom coordinates must have shape (natom, 3)")
    if origin.shape != (3,) or pairs.ndim != 2 or pairs.shape[1:] != (2,):
        raise SourceCompilationError("origin/pair topology has an invalid shape")
    field = physical.electric_field
    displacements = (
        coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]]
        if pairs.size
        else np.empty((0, 3), dtype=np.float64)
    )
    if gauge is UniformGauge.LENGTH:
        vector = np.zeros(3, dtype=np.float64)
        vector_dot = np.zeros(3, dtype=np.float64)
        scalar = -(coordinates - origin[None, :]) @ field
        links = np.zeros(pairs.shape[0], dtype=np.float64)
        link_dots = np.zeros(pairs.shape[0], dtype=np.float64)
    elif gauge is UniformGauge.VELOCITY:
        vector = np.asarray(physical.vector_potential_reduced, dtype=np.float64)
        vector_dot = np.asarray(physical.vector_potential_reduced_dot, dtype=np.float64)
        scalar = np.zeros(coordinates.shape[0], dtype=np.float64)
        links = displacements @ vector
        link_dots = displacements @ vector_dot
    else:  # pragma: no cover - closed enum, defensive for foreign callers
        raise SourceCompilationError(f"unsupported uniform gauge {gauge!r}")
    emf = -link_dots - scalar[pairs[:, 0]] + scalar[pairs[:, 1]]
    return UniformGaugeSample(
        gauge=gauge,
        electric_field=np.array(field, copy=True),
        vector_potential_reduced=np.array(vector, copy=True),
        vector_potential_reduced_dot=np.array(vector_dot, copy=True),
        node_scalar_potential=np.array(scalar, copy=True),
        pair_link=np.array(links, copy=True),
        pair_link_dot=np.array(link_dots, copy=True),
        pair_electromotive_potential=np.array(emf, copy=True),
    )
