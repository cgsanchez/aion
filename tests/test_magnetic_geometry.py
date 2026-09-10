from __future__ import annotations

import numpy as np
import pytest

from aion.backends import NumPyBackend
from aion.config import AtomicUnit, PhysicalDimension
from aion.electromagnetism import (
    TESLA_PER_ATOMIC_UNIT_MAGNETIC_FIELD,
    AffineMagneticGauge,
    MagneticGaugeKind,
    UniformMagneticField,
    affine_gauge_difference_potential,
    anchored_vectors,
    build_magnetic_pair_geometry,
    endpoint_line_integrals,
    endpoint_links,
    triangle_factors,
    triangle_fluxes,
    triangle_phases,
)
from aion.errors import ConfigurationError


def _fixture() -> tuple[NumPyBackend, object, np.ndarray]:
    backend = NumPyBackend()
    atoms = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    geometry = build_magnetic_pair_geometry(atoms, np.array([0, 1]), backend)
    points = np.array([[1.0, 3.0, 0.0], [-0.4, 0.7, 1.2]])
    return backend, geometry, points


def test_magnetic_units_and_explicit_tesla_conversion() -> None:
    assert PhysicalDimension.MAGNETIC_FIELD.value == "magnetic_field"
    assert AtomicUnit.MAGNETIC_FIELD.value == "atomic_unit_of_magnetic_field"
    field = UniformMagneticField.from_tesla(
        (TESLA_PER_ATOMIC_UNIT_MAGNETIC_FIELD, 0.0, 0.0)
    )
    assert field.magnetic_field_au == (1.0, 0.0, 0.0)
    assert field.magnetic_field_tesla[0] == TESLA_PER_ATOMIC_UNIT_MAGNETIC_FIELD
    assert field.magnitude_au == 1.0


def test_oriented_fixture_fixes_phase_vector_and_endpoint_signs() -> None:
    backend, geometry, points = _fixture()
    field = UniformMagneticField((0.0, 0.0, 4.0))
    gauge = AffineMagneticGauge(field)

    flux = triangle_fluxes(points[:1], geometry, field, backend)
    phase = triangle_phases(
        points[:1], geometry, field, backend, charge=-1.0, hbar=2.0
    )
    vectors = anchored_vectors(points[:1], geometry, field, backend)
    np.testing.assert_allclose(flux[0, 0, 1], -12.0)
    np.testing.assert_allclose(phase[0, 0, 1], 6.0)
    np.testing.assert_allclose(vectors[0, 0], (6.0, 2.0, 0.0))
    np.testing.assert_allclose(vectors[0, 1], (6.0, -2.0, 0.0))

    integrals = gauge.anchor_to_point_line_integrals(
        geometry.ao_anchor_coordinates_au, points[:1], backend
    )
    gradients = gauge.anchor_to_point_line_integral_gradients(
        geometry.ao_anchor_coordinates_au, points[:1], backend
    )
    np.testing.assert_allclose(integrals[0], (12.0, 0.0))
    np.testing.assert_allclose(endpoint_line_integrals(gauge, geometry, backend), 0.0)
    point_a = gauge.vector_potential(points[:1], backend)
    np.testing.assert_allclose(gradients[0] - point_a, vectors[0])

    direct_factor = np.exp(1j * (-1.0 / 2.0) * (integrals[:, 1] - integrals[:, 0]))
    factorized = endpoint_links(
        gauge, geometry, backend, charge=-1.0, hbar=2.0
    )[0, 1] * np.exp(1j * phase[:, 0, 1])
    np.testing.assert_allclose(direct_factor, factorized, atol=2.0e-15)


def test_pair_and_field_reversal_same_anchor_and_taylor_coefficients() -> None:
    backend, geometry, points = _fixture()
    field = UniformMagneticField((0.3, -0.4, 0.2))
    negative = UniformMagneticField(tuple(-x for x in field.magnetic_field_au))
    flux = triangle_fluxes(points, geometry, field, backend)
    factors = triangle_factors(points, geometry, field, backend)
    reversed_factors = triangle_factors(points, geometry, negative, backend)

    np.testing.assert_allclose(flux, -np.swapaxes(flux, 1, 2), atol=2.0e-16)
    np.testing.assert_allclose(np.diagonal(flux, axis1=1, axis2=2), 0.0, atol=0.0)
    np.testing.assert_allclose(factors.exact, reversed_factors.exact.conj(), atol=2.0e-16)
    np.testing.assert_allclose(factors.first, -reversed_factors.first, atol=0.0)
    np.testing.assert_allclose(factors.second, reversed_factors.second, atol=0.0)

    phase = triangle_phases(points, geometry, field, backend)
    np.testing.assert_allclose(factors.first, 1j * phase, atol=0.0)
    np.testing.assert_allclose(factors.second, -0.5 * phase**2, atol=0.0)
    remainder = factors.exact - (1.0 + factors.first + factors.second)
    assert float(np.max(np.abs(remainder))) <= float(np.max(np.abs(phase) ** 3)) / 6.0


def test_bond_parallel_field_annuls_every_triangle_phase() -> None:
    backend, geometry, points = _fixture()
    field = UniformMagneticField((0.71, 0.0, 0.0))
    phases = triangle_phases(points, geometry, field, backend)
    np.testing.assert_allclose(phases, 0.0, atol=0.0, rtol=0.0)
    assert np.linalg.norm(anchored_vectors(points, geometry, field, backend)) > 0.0


def test_symmetric_and_landau_gauges_differ_by_analytic_gradient() -> None:
    backend, geometry, points = _fixture()
    field = UniformMagneticField((0.3, -0.4, 0.2))
    axis = np.array((field.magnetic_field_au[1], -field.magnetic_field_au[0], 0.0))
    symmetric = AffineMagneticGauge(
        field, origin_au=(0.17, -0.31, 0.23)
    )
    landau = AffineMagneticGauge(
        field,
        kind=MagneticGaugeKind.LANDAU,
        origin_au=(-0.21, 0.37, -0.16),
        landau_axis=tuple(axis),
    )
    step = 1.0e-6
    numerical_gradient = np.empty_like(points)
    for cartesian in range(3):
        displacement = np.zeros(3)
        displacement[cartesian] = step
        plus = affine_gauge_difference_potential(landau, symmetric, points + displacement, backend)
        minus = affine_gauge_difference_potential(landau, symmetric, points - displacement, backend)
        numerical_gradient[:, cartesian] = (plus - minus) / (2.0 * step)
    np.testing.assert_allclose(
        numerical_gradient,
        landau.vector_potential(points, backend) - symmetric.vector_potential(points, backend),
        atol=2.0e-10,
    )

    chi = affine_gauge_difference_potential(
        landau, symmetric, geometry.ao_anchor_coordinates_au, backend
    )
    link_ratio = endpoint_links(landau, geometry, backend) / endpoint_links(
        symmetric, geometry, backend
    )
    expected = np.exp(-1j * (chi[:, None] - chi[None, :]))
    np.testing.assert_allclose(link_ratio, expected, atol=3.0e-15)


def test_rigid_rotation_covariance() -> None:
    backend = NumPyBackend()
    angle = 0.43
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    atoms = np.array([[0.2, -0.1, 0.4], [1.1, 0.7, -0.3]])
    points = np.array([[0.6, 0.2, 0.8], [-0.3, 0.9, -0.2]])
    field_vector = np.array([0.13, -0.09, 0.17])
    geometry = build_magnetic_pair_geometry(atoms, np.array([0, 1]), backend)
    rotated_geometry = build_magnetic_pair_geometry(
        atoms @ rotation.T, np.array([0, 1]), backend
    )
    field = UniformMagneticField(tuple(field_vector))
    rotated_field = UniformMagneticField(tuple(field_vector @ rotation.T))
    np.testing.assert_allclose(
        triangle_fluxes(points, geometry, field, backend),
        triangle_fluxes(points @ rotation.T, rotated_geometry, rotated_field, backend),
        atol=3.0e-17,
    )
    np.testing.assert_allclose(
        anchored_vectors(points @ rotation.T, rotated_geometry, rotated_field, backend),
        anchored_vectors(points, geometry, field, backend) @ rotation.T,
        atol=3.0e-17,
    )


def test_three_center_loop_link_is_gauge_independent_and_orientation_sensitive() -> None:
    backend = NumPyBackend()
    vertices = np.array(
        [
            (0.0, 0.0, 0.0),
            (1.75, 0.0, 0.0),
            (0.38, 1.22, 0.0),
        ]
    )
    field = UniformMagneticField((0.0, 0.0, 0.73))
    area_vector = 0.5 * np.sum(
        np.cross(vertices, np.roll(vertices, -1, axis=0)),
        axis=0,
    )
    flux = float(np.dot(field.magnetic_field_au, area_vector))
    gauges = (
        AffineMagneticGauge(field, origin_au=(0.21, -0.17, 0.33)),
        AffineMagneticGauge(
            field,
            kind=MagneticGaugeKind.LANDAU,
            origin_au=(-0.13, 0.29, -0.24),
            landau_axis=(1.0, 0.0, 0.0),
        ),
    )
    for gauge in gauges:
        forward_integrals = gauge.straight_line_integrals(
            vertices,
            np.roll(vertices, -1, axis=0),
            backend,
        )
        reverse_integrals = gauge.straight_line_integrals(
            vertices,
            np.roll(vertices, 1, axis=0),
            backend,
        )
        forward_link = np.exp(-1j * np.sum(forward_integrals))
        reverse_link = np.exp(-1j * np.sum(reverse_integrals))
        assert forward_link == pytest.approx(np.exp(-1j * flux), abs=5.0e-16)
        assert reverse_link == pytest.approx(forward_link.conjugate(), abs=5.0e-16)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: UniformMagneticField((0.0, np.inf, 0.0)),
        lambda: AffineMagneticGauge(
            UniformMagneticField((0.0, 0.0, 1.0)),
            kind=MagneticGaugeKind.LANDAU,
        ),
        lambda: AffineMagneticGauge(
            UniformMagneticField((0.0, 0.0, 1.0)),
            kind=MagneticGaugeKind.LANDAU,
            landau_axis=(0.0, 0.0, 1.0),
        ),
    ],
)
def test_invalid_magnetic_values_fail(factory: object) -> None:
    with pytest.raises(ConfigurationError):
        factory()  # type: ignore[operator]


def test_invalid_geometry_and_particle_parameters_fail() -> None:
    backend, geometry, points = _fixture()
    field = UniformMagneticField((0.0, 0.0, 1.0))
    with pytest.raises(ConfigurationError):
        build_magnetic_pair_geometry(np.zeros((2, 2)), np.array([0, 1]), backend)
    with pytest.raises(ConfigurationError):
        build_magnetic_pair_geometry(np.zeros((2, 3)), np.array([0, 2]), backend)
    with pytest.raises(ConfigurationError):
        triangle_phases(points, geometry, field, backend, hbar=0.0)
