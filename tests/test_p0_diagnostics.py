from __future__ import annotations

import numpy as np
import scipy.linalg

from aion import (
    AOAnchors,
    LinearOneBodyModel,
    PeierlsGeometry,
    UniformElectricGauge,
    density_from_coefficients,
    record_p0_observables,
    summarize_p0_gauge_errors,
)


def _orthonormal_coefficients(metric: np.ndarray, nocc: int) -> np.ndarray:
    eig, vec = scipy.linalg.eigh(metric, check_finite=False)
    invsqrt = (vec * eig**-0.5) @ vec.conj().T
    trial = np.array(
        [
            [1.0, 0.2],
            [0.1, 1.1],
            [0.4, -0.3],
        ],
        dtype=np.complex128,
    )
    q, _ = np.linalg.qr(trial)
    return invsqrt @ q[:, :nocc]


def test_record_p0_observables_reports_ward_diagnostics_for_linear_model():
    anchors = AOAnchors(
        atom_coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.2, 0.1, 0.0],
                [-0.1, 0.8, 0.2],
            ]
        ),
        ao_to_atom=np.array([0, 1, 2]),
    )
    s0 = np.array(
        [
            [1.0, 0.07, -0.03],
            [0.07, 1.0, 0.05],
            [-0.03, 0.05, 1.0],
        ],
        dtype=np.complex128,
    )
    h0 = np.array(
        [
            [-0.45, -0.18, 0.04],
            [-0.18, -0.08, -0.11],
            [0.04, -0.11, 0.13],
        ],
        dtype=np.complex128,
    )
    field = np.array([0.025, -0.015, 0.01])
    electric = UniformElectricGauge(
        field=lambda _t: field,
        field_integral=lambda t: field * t,
        lambda_value=lambda _t: 0.4,
        lambda_derivative=lambda _t: 0.0,
    )
    geometry = PeierlsGeometry(anchors, s0, electric=electric)
    model = LinearOneBodyModel(h0)
    occupations = np.array([1.0, 0.8])
    coeff = _orthonormal_coefficients(geometry.metric(0.0), occupations.size)
    rho = density_from_coefficients(coeff, occupations)

    row = record_p0_observables(
        step=3,
        time=0.37,
        coeff=coeff,
        geometry=geometry,
        model=model,
        occupations=occupations,
        midpoint_iterations=2,
        hamiltonian_residual=1.0e-13,
    )

    assert row["step"] == 3
    assert row["midpoint_iterations"] == 2
    assert (
        abs(row["electron_count"] - np.trace(rho @ geometry.metric(0.37)).real)
        < 1.0e-13
    )
    assert row["instantaneous_continuity_residual_norm"] < 1.0e-13
    assert abs(row["dipole_power_residual"]) < 1.0e-13
    assert abs(row["power_residual"]) < 2.0e-10
    assert row["source_current_dipole_derivative_residual_norm"] < 1.0e-13
    assert abs(row["source_current_power_residual"]) < 2.0e-10


def test_summarize_p0_gauge_errors_uses_source_observables():
    rows = [
        {
            "time_au": 0.0,
            "energy": -1.0,
            "dipole_x": 0.0,
            "dipole_y": 0.0,
            "dipole_z": 0.0,
            "population_0": 1.0,
            "charge_0": -1.0,
            "current_balance_0": 0.0,
            "orthonormality_error": 0.0,
            "instantaneous_continuity_residual_norm": 0.0,
            "power_residual": 0.0,
            "dipole_power_residual": 0.0,
            "max_abs_current": 0.0,
        },
        {
            "time_au": 1.0,
            "energy": -0.9,
            "dipole_x": 0.1,
            "dipole_y": 0.0,
            "dipole_z": 0.0,
            "population_0": 1.0,
            "charge_0": -1.0,
            "current_balance_0": 0.0,
            "orthonormality_error": 1.0e-14,
            "instantaneous_continuity_residual_norm": 0.0,
            "power_residual": 1.0e-12,
            "dipole_power_residual": 2.0e-12,
            "max_abs_current": 0.0,
        },
        {
            "time_au": 2.0,
            "energy": -0.8,
            "dipole_x": 0.2,
            "dipole_y": 0.0,
            "dipole_z": 0.0,
            "population_0": 1.0,
            "charge_0": -1.0,
            "current_balance_0": 0.0,
            "orthonormality_error": 2.0e-14,
            "instantaneous_continuity_residual_norm": 0.0,
            "power_residual": 2.0e-12,
            "dipole_power_residual": 3.0e-12,
            "max_abs_current": 0.0,
        },
    ]
    shifted = [dict(row) for row in rows]
    shifted[2]["dipole_x"] = 0.25

    summary = summarize_p0_gauge_errors({"length": rows, "velocity": shifted})

    assert summary["length"]["max_dipole_norm_error"] == 0.0
    assert np.isclose(summary["velocity"]["max_dipole_norm_error"], 0.05)
    assert summary["velocity"]["max_population_norm_error"] == 0.0
