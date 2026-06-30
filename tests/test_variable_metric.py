from __future__ import annotations

import numpy as np
import scipy.linalg

from aion.gauge import AOAnchors, PeierlsGeometry, UniformElectricGauge
from aion.matrix_models import (
    LinearOneBodyModel,
    SiteHubbardModel,
    density_from_coefficients,
    site_populations,
)
from aion.variable_metric import VariableMetricSCEM


def _constant_vector(vector):
    vector = np.asarray(vector, dtype=float)
    return lambda _t: vector


def _spd_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return a.conj().T @ a + np.eye(n)


def _hermitian_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return 0.5 * (a + a.conj().T)


def _orthonormal_coefficients(
    rng: np.random.Generator,
    metric: np.ndarray,
    nocc: int,
) -> np.ndarray:
    eig, vec = scipy.linalg.eigh(metric, check_finite=False)
    invsqrt = (vec * eig**-0.5) @ vec.conj().T
    q, _ = np.linalg.qr(
        rng.normal(size=(metric.shape[0], nocc))
        + 1j * rng.normal(size=(metric.shape[0], nocc))
    )
    return invsqrt @ q[:, :nocc]


def _velocity_geometry(anchors: AOAnchors, s0: np.ndarray) -> PeierlsGeometry:
    e = np.array([0.05, -0.01, 0.0])
    electric = UniformElectricGauge.velocity(
        field=_constant_vector(e),
        field_integral=lambda t: e * t,
    )
    return PeierlsGeometry(anchors, s0, electric=electric)


def test_variable_metric_scem_preserves_endpoint_metric_for_linear_model():
    rng = np.random.default_rng(2468)
    anchors = AOAnchors(
        atom_coords=np.array([[0.0, 0.0, 0.0], [1.2, 0.4, 0.0]]),
        ao_to_atom=np.array([0, 0, 1]),
    )
    s0 = _spd_matrix(rng, anchors.nao)
    h0 = _hermitian_matrix(rng, anchors.nao)
    geometry = _velocity_geometry(anchors, s0)
    model = LinearOneBodyModel(h0)
    occupations = np.array([2.0])
    coeff = _orthonormal_coefficients(rng, geometry.metric(0.0), nocc=1)

    rt = VariableMetricSCEM(geometry, model, occupations)
    result = rt.step(
        coeff,
        time=0.0,
        dt=0.2,
        midpoint_tolerance=1.0e-12,
        density_tolerance=None,
    )

    assert result.iterations == 1
    assert result.hamiltonian_residual < 1.0e-12
    assert rt.orthonormality_error(result.coeff_next, 0.2) < 1.0e-11


def test_site_hubbard_model_converges_and_preserves_endpoint_metric():
    rng = np.random.default_rng(13579)
    anchors = AOAnchors(
        atom_coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.2, 1.1, 0.0],
            ]
        ),
        ao_to_atom=np.array([0, 1, 1, 2]),
    )
    s0 = _spd_matrix(rng, anchors.nao)
    h0 = _hermitian_matrix(rng, anchors.nao)
    geometry = _velocity_geometry(anchors, s0)
    occupations = np.array([2.0, 2.0])
    coeff = _orthonormal_coefficients(rng, geometry.metric(0.0), nocc=2)
    rho = density_from_coefficients(coeff, occupations)
    reference = site_populations(rho, geometry, 0.0)
    model = SiteHubbardModel(h0, hubbard_u=0.15, reference_populations=reference)

    rt = VariableMetricSCEM(geometry, model, occupations)
    result = rt.step(
        coeff,
        time=0.0,
        dt=0.1,
        midpoint_tolerance=1.0e-10,
        density_tolerance=1.0e-10,
        max_iterations=20,
        mixing=0.7,
    )

    assert result.converged
    assert result.iterations >= 1
    assert rt.orthonormality_error(result.coeff_next, 0.1) < 1.0e-10


def test_length_and_velocity_gauge_site_populations_match_for_one_body_model():
    rng = np.random.default_rng(97531)
    anchors = AOAnchors(
        atom_coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [-0.1, 1.1, 0.0]]),
        ao_to_atom=np.array([0, 1, 2]),
    )
    s0 = _spd_matrix(rng, anchors.nao)
    h0 = _hermitian_matrix(rng, anchors.nao)
    e = np.array([0.03, -0.02, 0.0])
    length = PeierlsGeometry(
        anchors,
        s0,
        electric=UniformElectricGauge.length(
            field=_constant_vector(e),
            field_integral=lambda t: e * t,
        ),
    )
    velocity = PeierlsGeometry(
        anchors,
        s0,
        electric=UniformElectricGauge.velocity(
            field=_constant_vector(e),
            field_integral=lambda t: e * t,
        ),
    )
    occupations = np.array([2.0])
    coeff0 = _orthonormal_coefficients(rng, s0, nocc=1)

    coeff_length = coeff0.copy()
    coeff_velocity = coeff0.copy()
    rt_length = VariableMetricSCEM(length, LinearOneBodyModel(h0), occupations)
    rt_velocity = VariableMetricSCEM(velocity, LinearOneBodyModel(h0), occupations)

    dt = 0.02
    for step in range(5):
        coeff_length = rt_length.step(
            coeff_length,
            time=step * dt,
            dt=dt,
            midpoint_tolerance=1.0e-12,
            density_tolerance=None,
        ).coeff_next
        coeff_velocity = rt_velocity.step(
            coeff_velocity,
            time=step * dt,
            dt=dt,
            midpoint_tolerance=1.0e-12,
            density_tolerance=None,
        ).coeff_next

    rho_length = density_from_coefficients(coeff_length, occupations)
    rho_velocity = density_from_coefficients(coeff_velocity, occupations)
    pop_length = site_populations(rho_length, length, 5 * dt)
    pop_velocity = site_populations(rho_velocity, velocity, 5 * dt)

    assert np.linalg.norm(pop_length - pop_velocity) < 5.0e-8
