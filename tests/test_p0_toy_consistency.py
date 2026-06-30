from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from aion import (
    AOAnchors,
    LinearOneBodyModel,
    PeierlsGeometry,
    UniformElectricGauge,
    UniformMagneticGauge,
    VariableMetricSCEM,
    density_from_coefficients,
    p0_dipole_moment,
    p0_graph_currents,
    p0_site_populations,
    p0_source_power,
)


def _toy_anchors() -> AOAnchors:
    return AOAnchors(
        atom_coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.15, 0.1, 0.0],
                [-0.25, 0.95, 0.15],
                [0.85, 1.25, -0.1],
            ],
            dtype=float,
        ),
        ao_to_atom=np.array([0, 1, 2, 3]),
    )


def _toy_matrices(anchors: AOAnchors) -> tuple[np.ndarray, np.ndarray]:
    coords = anchors.atom_coords
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    s0 = np.exp(-0.65 * distance**2)
    np.fill_diagonal(s0, 1.0)
    h0 = np.diag([-0.42, -0.12, 0.03, 0.18]) - 0.35 * np.exp(-0.75 * distance)
    np.fill_diagonal(h0, np.diag([-0.42, -0.12, 0.03, 0.18]))
    return s0.astype(np.complex128), h0.astype(np.complex128)


def _orthonormal_coefficients(metric: np.ndarray, nocc: int) -> np.ndarray:
    eig, vec = scipy.linalg.eigh(metric, check_finite=False)
    invsqrt = (vec * eig**-0.5) @ vec.conj().T
    rng = np.random.default_rng(1729 + 17 * metric.shape[0] + nocc)
    trial = (
        rng.normal(size=(metric.shape[0], nocc))
        + 1j * rng.normal(size=(metric.shape[0], nocc))
    )
    q, _ = np.linalg.qr(trial)
    return invsqrt @ q[:, :nocc]


def _ground_state_coefficients(h0: np.ndarray, s0: np.ndarray, nocc: int) -> np.ndarray:
    _, coeff = scipy.linalg.eigh(h0, s0, check_finite=False)
    return coeff[:, :nocc]


def _propagate(
    geometry: PeierlsGeometry,
    h0: np.ndarray,
    coeff: np.ndarray,
    occupations: np.ndarray,
    *,
    dt: float,
    nsteps: int,
) -> np.ndarray:
    rt = VariableMetricSCEM(geometry, LinearOneBodyModel(h0), occupations)
    current = coeff.copy()
    time = 0.0 if dt > 0.0 else nsteps * abs(dt)
    for step in range(nsteps):
        current = rt.step(
            current,
            time=time + step * dt,
            dt=dt,
            midpoint_tolerance=1.0e-12,
            density_tolerance=None,
            max_iterations=8,
        ).coeff_next
    return current


def _sinusoidal_electric(lambda_value: float) -> UniformElectricGauge:
    direction = np.array([1.0, -0.3, 0.2], dtype=float)
    direction = direction / np.linalg.norm(direction)
    amplitude = 0.055
    omega = 0.8

    return UniformElectricGauge(
        field=lambda t: amplitude * np.sin(omega * t) * direction,
        field_integral=lambda t: amplitude * (1.0 - np.cos(omega * t)) / omega * direction,
        lambda_value=lambda _t: lambda_value,
        lambda_derivative=lambda _t: 0.0,
    )


def _site_phase_equivalence(
    reference: PeierlsGeometry,
    transformed: PeierlsGeometry,
    t: float,
) -> np.ndarray:
    delta = transformed.site_bond_line_integrals(t) - reference.site_bond_line_integrals(t)
    site_lambda = delta[:, 0]
    assert np.linalg.norm(delta - (site_lambda[:, None] - site_lambda[None, :])) < 1.0e-12
    return np.exp((1j * reference.charge / reference.hbar) * site_lambda)


def _observables(
    geometry: PeierlsGeometry,
    h0: np.ndarray,
    coeff: np.ndarray,
    occupations: np.ndarray,
    t: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    rho = density_from_coefficients(coeff, occupations)
    model = LinearOneBodyModel(h0)
    h = model.hamiltonian(rho, t, geometry)
    return (
        p0_site_populations(rho, geometry, t),
        p0_dipole_moment(rho, geometry, t),
        model.energy(rho, t, geometry),
        p0_graph_currents(rho, h, geometry, t),
    )


def test_static_magnetic_propagation_is_independent_of_gauge_choice():
    anchors = _toy_anchors()
    s0, h0 = _toy_matrices(anchors)
    occupations = np.array([1.0, 0.7])
    b = np.array([0.0, 0.0, 0.18])
    origin = np.array([0.2, -0.1, 0.05])
    reference = PeierlsGeometry(
        anchors,
        s0,
        magnetic=UniformMagneticGauge(b, gauge="symmetric", origin=origin),
    )
    gauges = [
        PeierlsGeometry(
            anchors,
            s0,
            magnetic=UniformMagneticGauge(
                b,
                gauge="symmetric",
                origin=np.array([-0.15, 0.25, 0.0]),
            ),
        ),
        PeierlsGeometry(
            anchors,
            s0,
            magnetic=UniformMagneticGauge(
                b,
                gauge="landau",
                origin=origin,
                landau_u=np.array([1.0, 0.0, 0.0]),
            ),
        ),
        PeierlsGeometry(
            anchors,
            s0,
            magnetic=UniformMagneticGauge(
                b,
                gauge="landau",
                origin=origin,
                landau_u=np.array([0.0, 1.0, 0.0]),
            ),
        ),
    ]
    coeff0 = _orthonormal_coefficients(reference.metric(0.0), occupations.size)
    coeff_reference = coeff0.copy()
    coeff_reference = _propagate(
        reference,
        h0,
        coeff_reference,
        occupations,
        dt=0.04,
        nsteps=8,
    )
    ref_obs = _observables(reference, h0, coeff_reference, occupations, 0.32)

    for geometry in gauges:
        site_g = _site_phase_equivalence(reference, geometry, 0.0)
        coeff = geometry.anchors.lift_site_vector(site_g)[:, None] * coeff0
        coeff = _propagate(geometry, h0, coeff, occupations, dt=0.04, nsteps=8)
        obs = _observables(geometry, h0, coeff, occupations, 0.32)
        assert np.linalg.norm(obs[0] - ref_obs[0]) < 1.0e-11
        assert np.linalg.norm(obs[1] - ref_obs[1]) < 1.0e-11
        assert abs(obs[2] - ref_obs[2]) < 1.0e-11
        assert np.linalg.norm(obs[3] - ref_obs[3]) < 1.0e-10


@dataclass(frozen=True)
class _BondSource:
    natom: int
    row_atom: int
    col_atom: int
    value: float

    def bond_line_integrals(self, _coords: np.ndarray, _t: float | None = None) -> np.ndarray:
        acal = np.zeros((self.natom, self.natom), dtype=float)
        acal[self.row_atom, self.col_atom] = self.value
        acal[self.col_atom, self.row_atom] = -self.value
        return acal

    def bond_line_integral_dots(self, coords: np.ndarray, _t: float | None = None) -> np.ndarray:
        return np.zeros((np.asarray(coords).shape[0], np.asarray(coords).shape[0]))

    def site_scalar_potential(self, coords: np.ndarray, _t: float | None = None) -> np.ndarray:
        return np.zeros(np.asarray(coords).shape[0])

    def site_scalar_integral(
        self,
        coords: np.ndarray,
        _t0: float,
        _t1: float,
    ) -> np.ndarray:
        return np.zeros(np.asarray(coords).shape[0])

    def bond_electromotive_forces(self, coords: np.ndarray, _t: float | None = None) -> np.ndarray:
        return np.zeros((np.asarray(coords).shape[0], np.asarray(coords).shape[0]))


def test_orthogonal_p0_current_matches_finite_difference_source_derivative():
    anchors = AOAnchors(
        atom_coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 0.9, 0.0]]),
        ao_to_atom=np.array([0, 1, 2]),
    )
    h0 = np.array(
        [
            [-0.5, -0.2 + 0.05j, 0.04 - 0.02j],
            [-0.2 - 0.05j, -0.1, -0.08 + 0.03j],
            [0.04 + 0.02j, -0.08 - 0.03j, 0.15],
        ],
        dtype=np.complex128,
    )
    occupations = np.array([1.0, 0.6])
    coeff = _orthonormal_coefficients(np.eye(3), occupations.size)
    rho = density_from_coefficients(coeff, occupations)
    a, b = 1, 2
    acal0 = 0.17
    eps = 1.0e-6
    geometry = PeierlsGeometry(
        anchors,
        np.eye(3),
        magnetic=_BondSource(anchors.natom, a, b, acal0),
    )
    model = LinearOneBodyModel(h0)
    h = model.hamiltonian(rho, 0.0, geometry)
    current = p0_graph_currents(rho, h, geometry, 0.0)[a, b]

    e_plus = model.energy(
        rho,
        0.0,
        PeierlsGeometry(
            anchors,
            np.eye(3),
            magnetic=_BondSource(anchors.natom, a, b, acal0 + eps),
        ),
    )
    e_minus = model.energy(
        rho,
        0.0,
        PeierlsGeometry(
            anchors,
            np.eye(3),
            magnetic=_BondSource(anchors.natom, a, b, acal0 - eps),
        ),
    )
    finite_difference = (e_plus - e_minus) / (2.0 * eps)

    assert np.isclose(current, finite_difference, atol=1.0e-10)


def test_variable_metric_scem_is_time_reversible_for_linear_midpoint_steps():
    anchors = _toy_anchors()
    s0, h0 = _toy_matrices(anchors)
    geometry = PeierlsGeometry(
        anchors,
        s0,
        electric=_sinusoidal_electric(lambda_value=0.45),
    )
    occupations = np.array([1.0, 0.8])
    coeff0 = _orthonormal_coefficients(geometry.metric(0.0), occupations.size)
    coeff_forward = _propagate(geometry, h0, coeff0, occupations, dt=0.03, nsteps=10)
    coeff_back = coeff_forward.copy()
    rt = VariableMetricSCEM(geometry, LinearOneBodyModel(h0), occupations)
    for step in range(10):
        coeff_back = rt.step(
            coeff_back,
            time=0.30 - step * 0.03,
            dt=-0.03,
            midpoint_tolerance=1.0e-12,
            density_tolerance=None,
            max_iterations=8,
        ).coeff_next

    assert np.linalg.norm(coeff_back - coeff0) < 2.0e-11


def test_zero_field_ground_state_is_stationary():
    anchors = _toy_anchors()
    s0, h0 = _toy_matrices(anchors)
    geometry = PeierlsGeometry(anchors, s0)
    occupations = np.array([2.0, 1.0])
    coeff0 = _ground_state_coefficients(h0, s0, occupations.size)
    coeff = _propagate(geometry, h0, coeff0, occupations, dt=0.05, nsteps=16)
    rho0 = density_from_coefficients(coeff0, occupations)
    rho = density_from_coefficients(coeff, occupations)
    model = LinearOneBodyModel(h0)
    h = model.hamiltonian(rho, 0.8, geometry)

    assert (
        np.linalg.norm(
            p0_site_populations(rho, geometry, 0.8)
            - p0_site_populations(rho0, geometry, 0.0)
        )
        < 1.0e-11
    )
    assert (
        abs(model.energy(rho, 0.8, geometry) - model.energy(rho0, 0.0, geometry))
        < 1.0e-11
    )
    assert (
        np.linalg.norm(
            p0_dipole_moment(rho, geometry, 0.8)
            - p0_dipole_moment(rho0, geometry, 0.0)
        )
        < 1.0e-11
    )
    assert np.linalg.norm(p0_graph_currents(rho, h, geometry, 0.8)) < 1.0e-11
    assert p0_source_power(p0_graph_currents(rho, h, geometry, 0.8), geometry, 0.8) == 0.0


def test_p0_observable_timestep_convergence_for_smooth_drive():
    anchors = _toy_anchors()
    s0, h0 = _toy_matrices(anchors)
    occupations = np.array([1.0, 0.8])
    geometry = PeierlsGeometry(
        anchors,
        s0,
        electric=_sinusoidal_electric(lambda_value=0.35),
    )
    coeff0 = _orthonormal_coefficients(geometry.metric(0.0), occupations.size)

    def final_dipole(dt: float) -> np.ndarray:
        coeff = _propagate(
            geometry,
            h0,
            coeff0,
            occupations,
            dt=dt,
            nsteps=round(1.2 / dt),
        )
        rho = density_from_coefficients(coeff, occupations)
        return p0_dipole_moment(rho, geometry, 1.2)

    ref = final_dipole(0.015)
    err_coarse = np.linalg.norm(final_dipole(0.06) - ref)
    err_medium = np.linalg.norm(final_dipole(0.03) - ref)

    assert err_medium < 0.4 * err_coarse
