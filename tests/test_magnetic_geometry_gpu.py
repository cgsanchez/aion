from __future__ import annotations

import numpy as np
import pytest

from aion.backends import CuPyBackend, NumPyBackend
from aion.electromagnetism import (
    AffineMagneticGauge,
    MagneticGaugeKind,
    UniformMagneticField,
    anchored_vectors,
    build_magnetic_pair_geometry,
    endpoint_links,
    triangle_factors,
    triangle_fluxes,
)

pytestmark = pytest.mark.gpu


def test_magnetic_geometry_is_gpu_resident_and_matches_numpy() -> None:
    cpu = NumPyBackend()
    gpu = CuPyBackend(0)
    atoms = np.array([[0.2, -0.3, 0.1], [1.4, 0.6, -0.2]])
    mapping = np.array([0, 0, 1])
    points = np.array([[0.1, 0.2, 0.3], [-0.7, 0.5, 1.1]])
    field = UniformMagneticField((0.09, -0.07, 0.11))
    landau_axis = (field.magnetic_field_au[1], -field.magnetic_field_au[0], 0.0)
    gauge = AffineMagneticGauge(
        field,
        kind=MagneticGaugeKind.LANDAU,
        origin_au=(0.17, -0.31, 0.23),
        landau_axis=landau_axis,
    )
    cpu_geometry = build_magnetic_pair_geometry(atoms, mapping, cpu)
    gpu_geometry = build_magnetic_pair_geometry(atoms, mapping, gpu)
    for value in (
        gpu_geometry.ao_anchor_coordinates_au,
        gpu_geometry.pair_midpoints_au,
        gpu_geometry.pair_displacements_au,
    ):
        gpu.assert_resident(value)

    cpu_results = (
        triangle_fluxes(points, cpu_geometry, field, cpu),
        triangle_factors(points, cpu_geometry, field, cpu).exact,
        anchored_vectors(points, cpu_geometry, field, cpu),
        endpoint_links(gauge, cpu_geometry, cpu),
    )
    gpu_results = (
        triangle_fluxes(points, gpu_geometry, field, gpu),
        triangle_factors(points, gpu_geometry, field, gpu).exact,
        anchored_vectors(points, gpu_geometry, field, gpu),
        endpoint_links(gauge, gpu_geometry, gpu),
    )
    for expected, actual in zip(cpu_results, gpu_results, strict=True):
        gpu.assert_resident(actual)
        np.testing.assert_allclose(gpu.to_host(actual), expected, atol=5.0e-15, rtol=5.0e-15)
