from __future__ import annotations

import numpy as np
import scipy.linalg

from aion import (
    AOAnchors,
    ConnectionCayleySCEM,
    LinearOneBodyModel,
    P0E1Model,
    PeierlsGeometry,
    UniformElectricGauge,
    coefficient_derivative,
    density_from_coefficients,
    dressed_central_dipole_matrices,
    p0_e1_time_connection_residual,
    p0_e1_uniform_electric_potential,
    transform_p0_coefficients_between_gauges,
)


def _constant_vector(vector):
    value = np.asarray(vector, dtype=float)
    return lambda _t: value


def _hermitian_random(rng: np.random.Generator, size: int) -> np.ndarray:
    value = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    return 0.5 * (value + value.conj().T)


def _problem(seed: int = 7719):
    rng = np.random.default_rng(seed)
    anchors = AOAnchors(
        atom_coords=np.array([[0.0, 0.0, 0.0], [1.1, -0.3, 0.4]]),
        ao_to_atom=np.array([0, 0, 1]),
    )
    factor = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    overlap = factor.conj().T @ factor + np.eye(3)
    hamiltonian = _hermitian_random(rng, 3)
    central = np.stack([_hermitian_random(rng, 3) for _ in range(3)])
    field = np.array([0.04, -0.025, 0.03])
    electric = UniformElectricGauge.velocity(
        field=_constant_vector(field),
        field_integral=lambda t: field * t,
    )
    geometry = PeierlsGeometry(anchors, overlap, electric=electric)
    model = P0E1Model(LinearOneBodyModel(hamiltonian), central)
    occupations = np.array([2.0])
    return rng, geometry, model, central, field, occupations


def _orthonormal_coefficients(rng, metric, nocc=1):
    coeff = rng.normal(size=(metric.shape[0], nocc)) + 1j * rng.normal(
        size=(metric.shape[0], nocc)
    )
    gram = coeff.conj().T @ metric @ coeff
    eig, vec = scipy.linalg.eigh(gram, check_finite=False)
    return coeff @ ((vec * eig**-0.5) @ vec.conj().T)


def test_e1_connection_residual_is_antihermitian_and_equivalent_to_potential():
    _, geometry, _, central, _, _ = _problem()
    time = 0.37
    residual = p0_e1_time_connection_residual(central, geometry, time)
    potential = p0_e1_uniform_electric_potential(central, geometry, time)

    np.testing.assert_allclose(residual + residual.conj().T, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(
        -1j * geometry.hbar * residual,
        potential,
        atol=1.0e-13,
    )


def test_velocity_gauge_full_connection_is_the_p0_e1_basis_derivative():
    _, geometry, model, central, field, occupations = _problem()
    rt = ConnectionCayleySCEM(geometry, model, occupations)
    time = 0.29
    metric = geometry.metric(time)
    dressed = dressed_central_dipole_matrices(central, geometry, time)
    ao_coords = geometry.anchors.atom_coords[geometry.anchors.ao_to_atom]
    displacement = ao_coords[:, None, :] - ao_coords[None, :, :]
    endpoint = (
        -0.5j
        * geometry.charge
        / geometry.hbar
        * np.einsum("x,ijx,ij->ij", field, displacement, metric)
    )
    spread = -(1j / geometry.hbar) * np.einsum("x,xij->ij", field, dressed)

    np.testing.assert_allclose(
        rt.lower_time_connection(time),
        endpoint + spread,
        atol=2.0e-13,
    )


def test_connection_link_satisfies_the_endpoint_metric_identity():
    _, geometry, model, _, _, occupations = _problem()
    rt = ConnectionCayleySCEM(geometry, model, occupations)
    start = 0.17
    target = 0.41
    result = rt.connection_link(start, target)

    assert result.raw_metric_error > result.corrected_metric_error
    assert result.corrected_metric_error < 2.0e-12
    np.testing.assert_allclose(
        result.matrix.conj().T @ geometry.metric(target) @ result.matrix,
        geometry.metric(start),
        atol=2.0e-12,
    )


def test_connection_cayley_recovers_the_full_p0_e1_instantaneous_equation():
    rng, geometry, model, _, _, occupations = _problem()
    time = 0.31
    coeff = _orthonormal_coefficients(rng, geometry.metric(time))
    density = density_from_coefficients(coeff, occupations)
    full_hamiltonian = model.hamiltonian(density, time, geometry)
    expected = coefficient_derivative(coeff, full_hamiltonian, geometry, time)
    rt = ConnectionCayleySCEM(geometry, model, occupations)

    errors = []
    for dt in (2.0e-3, 1.0e-3):
        plus = rt.step(
            coeff,
            time=time,
            dt=dt,
            midpoint_tolerance=1.0e-13,
            density_tolerance=None,
        ).coeff_next
        minus = rt.step(
            coeff,
            time=time,
            dt=-dt,
            midpoint_tolerance=1.0e-13,
            density_tolerance=None,
        ).coeff_next
        finite_difference = (plus - minus) / (2.0 * dt)
        errors.append(np.linalg.norm(finite_difference - expected))

    assert errors[1] < errors[0] / 3.5
    assert errors[1] < 3.0e-5


def test_connection_cayley_step_preserves_metric_orthonormality():
    rng, geometry, model, _, _, occupations = _problem()
    coeff = _orthonormal_coefficients(rng, geometry.metric(0.0))
    rt = ConnectionCayleySCEM(geometry, model, occupations)
    result = rt.step(
        coeff,
        time=0.0,
        dt=0.1,
        midpoint_tolerance=1.0e-13,
        density_tolerance=None,
    )

    assert rt.orthonormality_error(result.coeff_next, 0.1) < 2.0e-12
    assert result.left_connection_metric_error < 2.0e-12
    assert result.right_connection_metric_error < 2.0e-12


def test_connection_cayley_p0_e1_is_covariant_between_length_and_velocity_gauges():
    rng, velocity, model, central, field, occupations = _problem()
    length = PeierlsGeometry(
        velocity.anchors,
        velocity.overlap0,
        electric=UniformElectricGauge.length(
            field=_constant_vector(field),
            field_integral=lambda t: field * t,
        ),
    )
    coeff0 = _orthonormal_coefficients(rng, velocity.overlap0)
    rt_length = ConnectionCayleySCEM(length, model, occupations)
    rt_velocity = ConnectionCayleySCEM(velocity, model, occupations)
    coeff_length = coeff0.copy()
    coeff_velocity = coeff0.copy()
    dt = 0.02

    for step in range(5):
        kwargs = {
            "time": step * dt,
            "dt": dt,
            "midpoint_tolerance": 1.0e-13,
            "density_tolerance": None,
        }
        coeff_length = rt_length.step(coeff_length, **kwargs).coeff_next
        coeff_velocity = rt_velocity.step(coeff_velocity, **kwargs).coeff_next

    transformed_velocity = transform_p0_coefficients_between_gauges(
        coeff_velocity,
        velocity,
        length,
        time=5 * dt,
    )
    assert np.linalg.norm(coeff_length - transformed_velocity) < 2.0e-8

    rho_length = density_from_coefficients(coeff_length, occupations)
    rho_velocity = density_from_coefficients(coeff_velocity, occupations)
    dipole_length = np.asarray(
        [
            np.trace(rho_length @ matrix).real
            for matrix in dressed_central_dipole_matrices(central, length, 5 * dt)
        ]
    )
    dipole_velocity = np.asarray(
        [
            np.trace(rho_velocity @ matrix).real
            for matrix in dressed_central_dipole_matrices(central, velocity, 5 * dt)
        ]
    )
    assert np.linalg.norm(dipole_length - dipole_velocity) < 2.0e-8


def test_connection_cayley_linear_step_is_time_reversible_to_local_order():
    rng, geometry, model, _, _, occupations = _problem()
    time = 0.23
    dt = 0.02
    coeff = _orthonormal_coefficients(rng, geometry.metric(time))
    rt = ConnectionCayleySCEM(geometry, model, occupations)
    forward = rt.step(
        coeff,
        time=time,
        dt=dt,
        midpoint_tolerance=1.0e-13,
        density_tolerance=None,
    ).coeff_next
    backward = rt.step(
        forward,
        time=time + dt,
        dt=-dt,
        midpoint_tolerance=1.0e-13,
        density_tolerance=None,
    ).coeff_next

    assert np.linalg.norm(backward - coeff) < 2.0e-9
