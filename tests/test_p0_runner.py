from __future__ import annotations

import numpy as np
import scipy.linalg

from aion import (
    AOAnchors,
    LinearOneBodyModel,
    P0SCEMSettings,
    PeierlsGeometry,
    apply_p0_velocity_delta_kick,
    p0_dipole_moment,
    run_p0_electric_gauge_comparison,
    sin2_uniform_electric_gauge,
    summarize_p0_gauge_errors,
    velocity_delta_kick_electric_gauge,
)


def _toy_problem():
    anchors = AOAnchors(
        atom_coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.1, -0.2, 0.1],
                [-0.3, 0.9, 0.2],
            ]
        ),
        ao_to_atom=np.array([0, 1, 2]),
    )
    s0 = np.array(
        [
            [1.0, 0.08, -0.03],
            [0.08, 1.0, 0.04],
            [-0.03, 0.04, 1.0],
        ],
        dtype=np.complex128,
    )
    h0 = np.array(
        [
            [-0.5, -0.12, 0.03],
            [-0.12, -0.1, -0.07],
            [0.03, -0.07, 0.15],
        ],
        dtype=np.complex128,
    )
    _, coeff = scipy.linalg.eigh(h0, s0, check_finite=False)
    return anchors, s0, h0, np.array([1.0, 0.7]), coeff[:, :2]


def test_sin2_uniform_electric_gauge_impulse_matches_field_derivative():
    electric = sin2_uniform_electric_gauge(
        amplitude=0.03,
        omega=0.4,
        cycles=2.5,
        polarization=np.array([1.0, -0.4, 0.2]),
        lambda_value=0.5,
        t0=0.3,
        phase=0.2,
    )
    for time in [0.6, 4.0, 12.0]:
        eps = 1.0e-6
        derivative = (electric.impulse(time + eps) - electric.impulse(time - eps)) / (
            2.0 * eps
        )
        assert np.linalg.norm(derivative - electric.electric_field(time)) < 1.0e-9

    assert np.linalg.norm(electric.electric_field(0.1)) == 0.0
    assert np.linalg.norm(electric.impulse(0.1)) == 0.0
    assert np.linalg.norm(electric.electric_field(100.0)) == 0.0
    assert np.linalg.norm(electric.impulse(100.0) - electric.impulse(50.0)) < 1.0e-14


def test_five_cycle_sin2_pulse_has_zero_field_and_vector_potential_at_end():
    omega = 0.20304185038149578
    duration = 5.0 * 2.0 * np.pi / omega
    electric = sin2_uniform_electric_gauge(
        amplitude=0.025,
        omega=omega,
        cycles=5.0,
        polarization=np.array([0.0, 0.0, 1.0]),
        lambda_value=1.0,
    )

    assert np.linalg.norm(electric.electric_field(duration)) < 1.0e-14
    assert np.linalg.norm(electric.vector_potential(duration)) < 1.0e-14
    assert np.linalg.norm(electric.electric_field(np.nextafter(duration, np.inf))) == 0.0
    assert (
        np.linalg.norm(electric.vector_potential(np.nextafter(duration, np.inf)))
        < 1.0e-14
    )


def test_velocity_delta_kick_transforms_coefficients_to_post_kick_metric():
    anchors, s0, _h0, occupations, coeff = _toy_problem()
    field_free = PeierlsGeometry(anchors, s0)
    impulse = np.array([0.02, -0.01, 0.03])
    kicked = PeierlsGeometry(
        anchors,
        s0,
        electric=velocity_delta_kick_electric_gauge(impulse),
    )

    coeff_kicked = apply_p0_velocity_delta_kick(coeff, field_free, kicked)

    assert (
        np.linalg.norm(coeff_kicked.conj().T @ kicked.metric(0.0) @ coeff_kicked - np.eye(2))
        < 1.0e-12
    )
    rho = (coeff_kicked * occupations[None, :]) @ coeff_kicked.conj().T
    dipole = p0_dipole_moment(rho, kicked, 0.0)
    assert dipole.shape == (3,)


def test_sin2_pulse_source_observables_are_gauge_covariant_for_linear_p0_model():
    anchors, s0, h0, occupations, coeff0 = _toy_problem()
    model = LinearOneBodyModel(h0)
    settings = P0SCEMSettings(
        dt=0.03,
        nsteps=5,
        midpoint_tolerance=1.0e-12,
        density_tolerance=None,
        max_iterations=8,
        mixing=1.0,
    )

    rows_by_gauge = run_p0_electric_gauge_comparison(
        geometry_factory=lambda electric: PeierlsGeometry(anchors, s0, electric=electric),
        electric_factory=lambda lambda_value: sin2_uniform_electric_gauge(
            amplitude=0.02,
            omega=0.35,
            cycles=1.5,
            polarization=np.array([1.0, 0.3, -0.2]),
            lambda_value=lambda_value,
        ),
        model=model,
        occupations=occupations,
        coeff0=coeff0,
        settings=settings,
    )
    summary = summarize_p0_gauge_errors(rows_by_gauge)

    assert summary["mixed"]["max_dipole_norm_error"] < 1.0e-11
    assert summary["velocity"]["max_dipole_norm_error"] < 1.0e-11
    assert summary["mixed"]["max_energy_abs_error"] < 1.0e-11
    assert summary["velocity"]["max_energy_abs_error"] < 1.0e-11
