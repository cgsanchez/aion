from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aion.backends import NumPyBackend
from aion.config import (
    AtomConfig,
    BackendConfig,
    ElectromagneticOrigin,
    OneElectronReferenceConfig,
)
from aion.electromagnetism import UniformMagneticField, UniformMagneticSourceSample
from aion.electronic_structure import (
    AOGridPolicy,
    evaluate_exact_magnetic_field_source_direction,
    evaluate_exact_temporal_source_direction,
    evaluate_exact_wilson_one_electron_sample,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
)
from aion.formulations import (
    EOMTriple,
    OneElectronActionMatrixDirection,
    exact_endpoint_link_action_direction,
    exact_internal_magnetic_action_direction,
    exact_magnetic_endpoint_action_direction,
    exact_magnetic_field_action_direction,
    exact_pure_gauge_action_direction,
    exact_site_scalar_action_direction,
    exact_wilson_one_electron_triple,
    one_electron_velocity_density,
    restricted_one_electron_action_directional_derivative,
    restricted_one_electron_action_full_directional_derivative,
    restricted_one_electron_action_value,
)

_FIXTURE = Path(__file__).parent / "fixtures/exact_one_electron/hh_sto3g.fixture.json"


def _quadrature() -> object:
    values = json.loads(_FIXTURE.read_text(encoding="utf-8"))["config"]
    reference = prepare_one_electron_ao_reference(
        OneElectronReferenceConfig(
            atoms=tuple(
                AtomConfig(atom["symbol"], tuple(atom["position_au"])) for atom in values["atoms"]
            ),
            basis=values["basis"],
            electromagnetic_origin=ElectromagneticOrigin(
                tuple(values["electromagnetic_origin_au"])
            ),
        )
    )
    return prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(3),
        block_size=1024,
    )


def _state(metric: np.ndarray) -> np.ndarray:
    coefficient = np.asarray(((0.71 + 0.19j,), (-0.23 + 0.41j,)))
    coefficient /= np.sqrt((coefficient.conj().T @ metric @ coefficient).real.item())
    return coefficient @ coefficient.conj().T


def _direction(connection: np.ndarray) -> OneElectronActionMatrixDirection:
    zero = np.zeros_like(connection)
    return OneElectronActionMatrixDirection(zero, zero, connection)


def test_wp7_restricted_action_direction_matches_fixed_history_difference() -> None:
    backend = NumPyBackend()
    rng = np.random.default_rng(8817)

    def hermitian() -> np.ndarray:
        value = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        return value + value.conj().T

    metric = hermitian() + 7.0 * np.eye(3)
    mechanical = hermitian()
    connection = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    metric_direction = hermitian()
    mechanical_direction = hermitian()
    connection_direction = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    coefficients = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    coefficient_dots = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    occupations = np.asarray((1.0, 0.37))
    density = np.einsum(
        "mi,i,ni->mn", coefficients, occupations, coefficients.conj(), optimize=True
    )
    velocity_density = np.einsum(
        "mi,i,ni->mn",
        coefficient_dots,
        occupations,
        coefficients.conj(),
        optimize=True,
    )
    direction = OneElectronActionMatrixDirection(
        metric_direction,
        mechanical_direction,
        connection_direction,
    )
    analytic = restricted_one_electron_action_directional_derivative(
        density,
        velocity_density,
        direction,
        backend,
    )
    step = 2.0e-5

    def value(sign: float) -> object:
        triple = EOMTriple(
            metric + sign * step * metric_direction,
            mechanical + sign * step * mechanical_direction,
            connection + sign * step * connection_direction,
        )
        return restricted_one_electron_action_value(
            density,
            velocity_density,
            triple,
            backend,
        ).total

    finite_difference = (value(1.0) - value(-1.0)) / (2.0 * step)
    np.testing.assert_allclose(analytic.total, finite_difference, atol=2.0e-10, rtol=2.0e-10)
    np.testing.assert_allclose(
        analytic.total,
        analytic.metric_kinematic + analytic.connection_kinematic + analytic.mechanical,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.integration
def test_wp7_exact_electric_source_derivative_reduces_to_dipole() -> None:
    quadrature = _quadrature()
    source = UniformMagneticSourceSample(
        0.31,
        UniformMagneticField((0.0, 0.0, 0.0)),
        origin_au=(0.11, -0.07, 0.05),
    )
    sample = evaluate_exact_wilson_one_electron_sample(quadrature, source)
    triple = exact_wilson_one_electron_triple(sample)
    density = _state(sample.metric)
    velocity_density = one_electron_velocity_density(
        density,
        triple,
        quadrature.backend,
    )
    electric_direction = np.asarray((0.37, -0.23, 0.19))
    response = evaluate_exact_temporal_source_direction(
        quadrature,
        sample,
        electric_origin_direction_au=electric_direction,
    )
    derivative = restricted_one_electron_action_directional_derivative(
        density,
        velocity_density,
        _direction(response.connection),
        quadrature.backend,
    )
    operators = quadrature.reference.core_operators
    relative_position = operators.position - (
        np.asarray(source.origin_au)[:, None, None] * operators.overlap[None, :, :]
    )
    dipole = -np.real(np.einsum("ij,xji->x", density, relative_position, optimize=True))
    np.testing.assert_allclose(
        derivative.total,
        electric_direction @ dipole,
        atol=4.0e-13,
        rtol=4.0e-13,
    )
    np.testing.assert_allclose(
        response.connection,
        response.site_scalar_connection + response.internal_electric_connection,
        atol=4.0e-15,
        rtol=4.0e-15,
    )
    assert response.decomposition_residual < 4.0e-15

    step = 2.0e-4

    def action(sign: float) -> object:
        displaced = UniformMagneticSourceSample(
            source.time_au,
            source.field,
            electric_field_origin_au=tuple(sign * step * electric_direction),
            origin_au=source.origin_au,
        )
        rebuilt = evaluate_exact_wilson_one_electron_sample(quadrature, displaced)
        return restricted_one_electron_action_value(
            density,
            velocity_density,
            exact_wilson_one_electron_triple(rebuilt),
            quadrature.backend,
        ).total

    finite_difference = (action(1.0) - action(-1.0)) / (2.0 * step)
    np.testing.assert_allclose(
        derivative.total,
        finite_difference,
        atol=2.0e-11,
        rtol=2.0e-10,
    )


@pytest.mark.integration
def test_wp7_exact_electric_and_magnetic_rate_directions_are_linear() -> None:
    quadrature = _quadrature()
    source = UniformMagneticSourceSample(
        0.41,
        UniformMagneticField((0.0, 0.0, 0.031)),
        magnetic_field_dot_au=(0.003, -0.004, 0.007),
        electric_field_origin_au=(0.013, -0.009, 0.017),
        origin_au=(0.11, -0.07, 0.05),
    )
    sample = evaluate_exact_wilson_one_electron_sample(quadrature, source)
    electric = np.asarray((0.17, -0.11, 0.07))
    magnetic_rate = np.asarray((-0.09, 0.05, 0.13))
    electric_response = evaluate_exact_temporal_source_direction(
        quadrature,
        sample,
        electric_origin_direction_au=electric,
    )
    magnetic_response = evaluate_exact_temporal_source_direction(
        quadrature,
        sample,
        magnetic_field_rate_direction_au=magnetic_rate,
    )
    combined = evaluate_exact_temporal_source_direction(
        quadrature,
        sample,
        electric_origin_direction_au=electric,
        magnetic_field_rate_direction_au=magnetic_rate,
    )
    np.testing.assert_allclose(
        combined.connection,
        electric_response.connection + magnetic_response.connection,
        atol=3.0e-14,
        rtol=3.0e-14,
    )
    np.testing.assert_allclose(
        combined.metric_rate,
        electric_response.metric_rate + magnetic_response.metric_rate,
        atol=3.0e-14,
        rtol=3.0e-14,
    )
    assert electric_response.decomposition_residual < 4.0e-14
    assert magnetic_response.decomposition_residual < 4.0e-14

    triple = exact_wilson_one_electron_triple(sample)
    density = _state(sample.metric)
    velocity_density = one_electron_velocity_density(
        density,
        triple,
        quadrature.backend,
    )
    analytic = restricted_one_electron_action_directional_derivative(
        density,
        velocity_density,
        _direction(magnetic_response.connection),
        quadrature.backend,
    )
    step = 2.0e-4
    base_rate = np.asarray(source.magnetic_field_dot_au)

    def action_and_metric_rate(sign: float) -> tuple[object, np.ndarray]:
        displaced = UniformMagneticSourceSample(
            source.time_au,
            source.field,
            magnetic_field_dot_au=tuple(base_rate + sign * step * magnetic_rate),
            electric_field_origin_au=source.electric_field_origin_au,
            origin_au=source.origin_au,
        )
        rebuilt = evaluate_exact_wilson_one_electron_sample(quadrature, displaced)
        action_value = restricted_one_electron_action_value(
            density,
            velocity_density,
            exact_wilson_one_electron_triple(rebuilt),
            quadrature.backend,
        ).total
        return action_value, rebuilt.connection.metric_dot

    plus_action, plus_metric_rate = action_and_metric_rate(1.0)
    minus_action, minus_metric_rate = action_and_metric_rate(-1.0)
    np.testing.assert_allclose(
        analytic.total,
        (plus_action - minus_action) / (2.0 * step),
        atol=2.0e-11,
        rtol=3.0e-10,
    )
    np.testing.assert_allclose(
        magnetic_response.metric_rate,
        (plus_metric_rate - minus_metric_rate) / (2.0 * step),
        atol=2.0e-11,
        rtol=3.0e-10,
    )


@pytest.mark.integration
def test_wp7_site_scalar_and_oriented_link_action_directions() -> None:
    quadrature = _quadrature()
    source = UniformMagneticSourceSample(
        0.41,
        UniformMagneticField((0.0, 0.0, 0.031)),
        magnetic_field_dot_au=(0.003, -0.004, 0.007),
        electric_field_origin_au=(0.013, -0.009, 0.017),
        origin_au=(0.11, -0.07, 0.05),
    )
    sample = evaluate_exact_wilson_one_electron_sample(quadrature, source)
    triple = exact_wilson_one_electron_triple(sample)
    density = _state(sample.metric)
    velocity_density = one_electron_velocity_density(
        density,
        triple,
        quadrature.backend,
    )
    mapping = np.asarray(quadrature.reference.anchor_topology.ao_to_atom)
    scalar_values = np.asarray((0.31, -0.17))
    scalar_direction = exact_site_scalar_action_direction(
        sample,
        scalar_values,
        mapping,
        quadrature.backend,
    )
    scalar_derivative = restricted_one_electron_action_directional_derivative(
        density,
        velocity_density,
        scalar_direction,
        quadrature.backend,
    )
    charges = np.empty(2)
    for site in range(2):
        diagonal = mapping == site
        operator = 0.5 * (diagonal[:, None] * sample.metric + sample.metric * diagonal[None, :])
        charges[site] = -np.real(np.einsum("ij,ji->", density, operator))
    np.testing.assert_allclose(
        scalar_derivative.total,
        -scalar_values @ charges,
        atol=3.0e-14,
        rtol=3.0e-14,
    )
    np.testing.assert_allclose(charges.sum(), -1.0, atol=3.0e-14, rtol=3.0e-14)

    links = np.asarray(((0.0, 0.29), (-0.29, 0.0)))
    link_direction = exact_endpoint_link_action_direction(
        sample,
        links,
        mapping,
        quadrature.backend,
    )
    link_derivative = restricted_one_electron_action_directional_derivative(
        density,
        velocity_density,
        link_direction,
        quadrature.backend,
    )
    ao_links = links[mapping[:, None], mapping[None, :]]
    prefactor = 1j * sample.static_result.charge / sample.static_result.hbar
    step = 2.0e-4

    def endpoint_action(sign: float) -> object:
        phase = np.exp(sign * step * prefactor * ao_links)
        perturbed = EOMTriple(
            phase * sample.metric,
            phase * sample.mechanical,
            phase * sample.connection.connection,
        )
        return restricted_one_electron_action_value(
            density,
            velocity_density,
            perturbed,
            quadrature.backend,
        ).total

    finite_difference = (endpoint_action(1.0) - endpoint_action(-1.0)) / (2.0 * step)
    np.testing.assert_allclose(
        link_derivative.total,
        finite_difference,
        atol=2.0e-10,
        rtol=2.0e-9,
    )
    assert abs(float(link_derivative.metric_kinematic)) > 1.0e-8
    assert abs(float(link_derivative.mechanical)) > 1.0e-8


@pytest.mark.integration
def test_wp7_exact_finite_magnetic_source_direction_matches_fixed_history_difference() -> None:
    quadrature = _quadrature()
    field = np.asarray((0.017, -0.013, 0.031))
    source = UniformMagneticSourceSample(
        0.41,
        UniformMagneticField(tuple(field)),
        magnetic_field_dot_au=(0.003, -0.004, 0.007),
        electric_field_origin_au=(0.013, -0.009, 0.017),
        origin_au=(0.11, -0.07, 0.05),
    )
    sample = evaluate_exact_wilson_one_electron_sample(quadrature, source)
    magnetic_direction = np.asarray((0.13, -0.07, 0.09))
    response = evaluate_exact_magnetic_field_source_direction(
        quadrature,
        sample,
        magnetic_direction,
    )
    total_direction = exact_magnetic_field_action_direction(
        response,
        quadrature.backend,
    )
    endpoint_direction = exact_magnetic_endpoint_action_direction(
        response,
        quadrature.backend,
    )
    internal_direction = exact_internal_magnetic_action_direction(
        response,
        quadrature.backend,
    )
    for total, endpoint, internal in (
        (
            total_direction.metric,
            endpoint_direction.metric,
            internal_direction.metric,
        ),
        (
            total_direction.mechanical,
            endpoint_direction.mechanical,
            internal_direction.mechanical,
        ),
        (
            total_direction.connection,
            endpoint_direction.connection,
            internal_direction.connection,
        ),
    ):
        np.testing.assert_allclose(total, endpoint + internal, atol=3.0e-14, rtol=3.0e-14)

    step = 2.0e-5

    def displaced(sign: float) -> object:
        displaced_source = UniformMagneticSourceSample(
            source.time_au,
            UniformMagneticField(tuple(field + sign * step * magnetic_direction)),
            magnetic_field_dot_au=source.magnetic_field_dot_au,
            electric_field_origin_au=source.electric_field_origin_au,
            origin_au=source.origin_au,
        )
        return evaluate_exact_wilson_one_electron_sample(quadrature, displaced_source)

    plus = displaced(1.0)
    minus = displaced(-1.0)
    for analytic, plus_value, minus_value in (
        (total_direction.metric, plus.metric, minus.metric),
        (total_direction.mechanical, plus.mechanical, minus.mechanical),
        (
            total_direction.connection,
            plus.connection.connection,
            minus.connection.connection,
        ),
    ):
        finite_difference = (plus_value - minus_value) / (2.0 * step)
        np.testing.assert_allclose(
            analytic,
            finite_difference,
            atol=3.0e-9,
            rtol=4.0e-8,
        )

    triple = exact_wilson_one_electron_triple(sample)
    density = _state(sample.metric)
    velocity_density = one_electron_velocity_density(
        density,
        triple,
        quadrature.backend,
    )
    analytic_action = restricted_one_electron_action_directional_derivative(
        density,
        velocity_density,
        total_direction,
        quadrature.backend,
    )

    def action(displaced_sample: object) -> object:
        return restricted_one_electron_action_value(
            density,
            velocity_density,
            exact_wilson_one_electron_triple(displaced_sample),
            quadrature.backend,
        ).total

    finite_action = (action(plus) - action(minus)) / (2.0 * step)
    np.testing.assert_allclose(
        analytic_action.total,
        finite_action,
        atol=3.0e-9,
        rtol=4.0e-8,
    )


@pytest.mark.integration
def test_wp7_exact_off_shell_pure_gauge_ward_identity() -> None:
    quadrature = _quadrature()
    source = UniformMagneticSourceSample(
        0.41,
        UniformMagneticField((0.017, -0.013, 0.031)),
        magnetic_field_dot_au=(0.003, -0.004, 0.007),
        electric_field_origin_au=(0.013, -0.009, 0.017),
        origin_au=(0.11, -0.07, 0.05),
    )
    sample = evaluate_exact_wilson_one_electron_sample(quadrature, source)
    triple = exact_wilson_one_electron_triple(sample)
    density = _state(sample.metric)
    velocity_density = np.asarray(((0.13 + 0.29j, -0.17 + 0.07j), (0.11 - 0.19j, -0.23 + 0.31j)))
    mapping = np.asarray(quadrature.reference.anchor_topology.ao_to_atom)
    gauge_values = np.asarray((0.37, -0.21))
    gauge_rates = np.asarray((-0.16, 0.29))
    direction = exact_pure_gauge_action_direction(
        sample,
        density,
        velocity_density,
        gauge_values,
        gauge_rates,
        mapping,
        quadrature.backend,
    )
    ward = restricted_one_electron_action_full_directional_derivative(
        density,
        velocity_density,
        triple,
        direction.matrix,
        direction.history,
        quadrature.backend,
    )
    assert abs(float(ward.source.total)) > 1.0e-6
    np.testing.assert_allclose(
        ward.total.total,
        0.0,
        atol=3.0e-15,
        rtol=0.0,
    )

    prefactor = 1j * sample.static_result.charge / sample.static_result.hbar
    eta = prefactor * gauge_values[mapping]
    eta_dot = prefactor * gauge_rates[mapping]

    def transformed_action(amplitude: float) -> object:
        transform = np.diag(np.exp(amplitude * eta))
        transform_dagger = transform.conj().T
        rate = np.diag(eta_dot)
        transformed_density = transform @ density @ transform_dagger
        transformed_velocity = (
            transform @ (velocity_density + amplitude * rate @ density) @ transform_dagger
        )
        transformed_triple = EOMTriple(
            transform @ sample.metric @ transform_dagger,
            transform @ sample.mechanical @ transform_dagger,
            transform
            @ (sample.connection.connection - amplitude * sample.metric @ rate)
            @ transform_dagger,
        )
        return restricted_one_electron_action_value(
            transformed_density,
            transformed_velocity,
            transformed_triple,
            quadrature.backend,
        ).total

    reference_action = restricted_one_electron_action_value(
        density,
        velocity_density,
        triple,
        quadrature.backend,
    ).total
    np.testing.assert_allclose(
        transformed_action(0.23),
        reference_action,
        atol=4.0e-15,
        rtol=4.0e-15,
    )
    step = 2.0e-5
    finite_difference = (transformed_action(step) - transformed_action(-step)) / (2.0 * step)
    np.testing.assert_allclose(
        ward.total.total,
        finite_difference,
        atol=2.0e-10,
        rtol=0.0,
    )
