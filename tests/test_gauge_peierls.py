from __future__ import annotations

import numpy as np

from aion.gauge import (
    AOAnchors,
    PeierlsGeometry,
    UniformElectricGauge,
    UniformMagneticGauge,
)


def _spd_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return a.conj().T @ a + np.eye(n)


def _constant_vector(vector):
    vector = np.asarray(vector, dtype=float)
    return lambda _t: vector


def test_peierls_metric_and_time_connection_are_metric_compatible():
    rng = np.random.default_rng(123)
    anchors = AOAnchors(
        atom_coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.3, -0.2, 0.1],
                [-0.4, 0.9, 0.3],
            ]
        ),
        ao_to_atom=np.array([0, 0, 1, 2]),
    )
    s0 = _spd_matrix(rng, anchors.nao)
    electric_field = np.array([0.02, -0.03, 0.01])
    electric = UniformElectricGauge(
        field=_constant_vector(electric_field),
        field_integral=lambda t: electric_field * t,
        lambda_value=lambda _t: 0.37,
        lambda_derivative=lambda _t: 0.0,
    )
    geometry = PeierlsGeometry(anchors, s0, electric=electric)

    t = 0.8
    theta_site = geometry.site_theta(t)
    assert np.linalg.norm(theta_site.T.conj() - theta_site) < 1.0e-14

    metric = geometry.metric(t)
    assert np.linalg.norm(metric - metric.conj().T) < 1.0e-12

    omega = geometry.time_connection(t)
    metric_dot = geometry.ordinary_metric_dot(t)
    assert np.linalg.norm(omega + omega.conj().T - metric_dot) < 1.0e-12


def test_site_transport_uses_exact_scalar_potential_integral():
    anchors = AOAnchors(
        atom_coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.4, -0.3, 0.2],
                [-0.2, 0.8, 0.5],
            ]
        ),
        ao_to_atom=np.array([0, 1, 2]),
    )
    s0 = np.eye(anchors.nao)
    e = np.array([0.04, -0.02, 0.01])
    electric = UniformElectricGauge(
        field=_constant_vector(e),
        field_integral=lambda t: e * t,
        lambda_value=lambda t: 0.25 + 0.1 * t,
        lambda_derivative=lambda _t: 0.1,
    )
    geometry = PeierlsGeometry(anchors, s0, electric=electric)

    t0 = 0.3
    t1 = 1.7
    primitive0 = -(
        1.0 - electric.lam(t0)
    ) * np.einsum("x,ax->a", electric.impulse(t0), anchors.atom_coords)
    primitive1 = -(
        1.0 - electric.lam(t1)
    ) * np.einsum("x,ax->a", electric.impulse(t1), anchors.atom_coords)
    expected = np.exp(
        -(1j * geometry.charge / geometry.hbar) * (primitive1 - primitive0)
    )

    assert np.linalg.norm(geometry.site_transport(t0, t1) - expected) < 1.0e-14


def _site_phase_equivalence(
    reference: PeierlsGeometry,
    transformed: PeierlsGeometry,
    t: float,
) -> np.ndarray:
    delta = (
        transformed.site_bond_line_integrals(t)
        - reference.site_bond_line_integrals(t)
    )
    site_lambda = delta[:, 0]
    assert np.linalg.norm(delta - (site_lambda[:, None] - site_lambda[None, :])) < 1.0e-13
    return np.exp((1j * reference.charge / reference.hbar) * site_lambda)


def test_uniform_b_symmetric_origin_shift_is_site_gauge_equivalent():
    rng = np.random.default_rng(456)
    anchors = AOAnchors(
        atom_coords=np.array(
            [
                [0.2, -0.1, 0.0],
                [1.1, 0.4, -0.2],
                [-0.3, 1.0, 0.5],
            ]
        ),
        ao_to_atom=np.array([0, 1, 1, 2]),
    )
    s0 = _spd_matrix(rng, anchors.nao)
    b = np.array([0.0, 0.0, 0.15])
    geom0 = PeierlsGeometry(
        anchors,
        s0,
        magnetic=UniformMagneticGauge(b, gauge="symmetric", origin=np.zeros(3)),
    )
    geom1 = PeierlsGeometry(
        anchors,
        s0,
        magnetic=UniformMagneticGauge(
            b,
            gauge="symmetric",
            origin=np.array([0.3, -0.2, 0.1]),
        ),
    )

    site_g = _site_phase_equivalence(geom0, geom1, t=0.0)
    ao_g = anchors.lift_site_vector(site_g)
    transformed_metric = ao_g[:, None] * geom0.metric(0.0) * ao_g.conj()[None, :]
    assert np.linalg.norm(geom1.metric(0.0) - transformed_metric) < 1.0e-12


def test_uniform_b_symmetric_and_landau_gauges_are_site_gauge_equivalent():
    rng = np.random.default_rng(789)
    anchors = AOAnchors(
        atom_coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.3, 0.2],
                [-0.2, 1.2, -0.1],
            ]
        ),
        ao_to_atom=np.array([0, 1, 2]),
    )
    s0 = _spd_matrix(rng, anchors.nao)
    b = np.array([0.0, 0.0, 0.2])
    origin = np.array([0.1, -0.4, 0.0])
    symmetric = PeierlsGeometry(
        anchors,
        s0,
        magnetic=UniformMagneticGauge(b, gauge="symmetric", origin=origin),
    )
    landau = PeierlsGeometry(
        anchors,
        s0,
        magnetic=UniformMagneticGauge(
            b,
            gauge="landau",
            origin=origin,
            landau_u=np.array([1.0, 0.0, 0.0]),
        ),
    )

    site_g = _site_phase_equivalence(symmetric, landau, t=0.0)
    ao_g = anchors.lift_site_vector(site_g)
    transformed_metric = (
        ao_g[:, None] * symmetric.metric(0.0) * ao_g.conj()[None, :]
    )
    assert np.linalg.norm(landau.metric(0.0) - transformed_metric) < 1.0e-12


def test_uniform_b_affine_point_and_straight_line_operations():
    field = np.array([0.13, -0.08, 0.11])
    origin = np.array([0.21, -0.34, 0.17])
    landau_u = np.array([field[1], -field[0], 0.0])
    starts = np.array(
        [
            [0.24, -0.17, 0.31],
            [-0.42, 0.56, 0.13],
        ]
    )
    ends = np.array(
        [
            [1.12, 0.37, -0.29],
            [0.33, -0.22, 0.71],
        ]
    )

    gauges = (
        UniformMagneticGauge(field, gauge="symmetric", origin=origin),
        UniformMagneticGauge(
            field,
            gauge="landau",
            origin=origin,
            landau_u=landau_u,
        ),
    )
    for gauge in gauges:
        if gauge.gauge == "symmetric":
            expected_a = 0.5 * np.cross(field, ends - origin)
        else:
            assert gauge.landau_u is not None
            expected_a = np.einsum(
                "px,x->p",
                ends - origin,
                gauge.landau_u,
            )[:, None] * np.cross(field, gauge.landau_u)[None, :]
        assert np.linalg.norm(gauge.vector_potential(ends) - expected_a) < 1.0e-14

        midpoint = 0.5 * (starts + ends)
        displacement = ends - starts
        expected_integral = np.einsum(
            "px,px->p",
            gauge.vector_potential(midpoint),
            displacement,
        )
        actual_integral = gauge.straight_line_integrals(starts, ends)
        assert np.linalg.norm(actual_integral - expected_integral) < 1.0e-14

        analytic_gradient = gauge.straight_line_integral_gradients(starts, ends)
        finite_difference = np.empty_like(analytic_gradient)
        step = 1.0e-6
        for component in range(3):
            shift = np.zeros(3)
            shift[component] = step
            plus = gauge.straight_line_integrals(starts, ends + shift)
            minus = gauge.straight_line_integrals(starts, ends - shift)
            finite_difference[:, component] = (plus - minus) / (2.0 * step)
        assert np.linalg.norm(analytic_gradient - finite_difference) < 1.0e-9

        coords = np.vstack((starts, ends[:1]))
        pairwise = gauge.straight_line_integrals(
            coords[None, :, :],
            coords[:, None, :],
        )
        assert np.linalg.norm(gauge.bond_line_integrals(coords) - pairwise) < 1.0e-14
