from __future__ import annotations

import numpy as np
import pytest

from aion import (
    PyscfP0Reference,
    UniformElectricGauge,
    apply_e1_central_delta_kick,
    coefficient_orthonormality_error,
    density_from_coefficients,
    operator_expectations,
    p0_dipole_moment,
    p0_dipole_operator_matrices,
    p0_e1_dipole_moment,
    p0_e1_dipole_operator_matrices,
    pyscf_central_dipole_matrices,
    apply_p0_velocity_delta_kick,
    velocity_delta_kick_electric_gauge,
)


def _pyscf_modules():
    pytest.importorskip("pyscf")
    from pyscf import dft, gto

    return dft, gto


def _h2_reference() -> PyscfP0Reference:
    dft, gto = _pyscf_modules()
    mol = gto.M(
        atom="H 0 0 -0.37; H 0 0 0.37",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "lda,vwn"
    mf.grids.level = 0
    mf.conv_tol = 1.0e-11
    mf.kernel()
    assert mf.converged
    return PyscfP0Reference.from_mean_field(mf)


def test_p0_dipole_operator_expectations_match_observable_function():
    reference = _h2_reference()
    impulse = np.array([0.0, 0.0, 1.0e-3])
    geometry = reference.geometry(electric=velocity_delta_kick_electric_gauge(impulse))
    rho = reference.initial_density()
    operators = p0_dipole_operator_matrices(geometry, 0.0)

    from_ops = operator_expectations(rho, operators)
    direct = p0_dipole_moment(rho, geometry, 0.0)

    assert np.linalg.norm(from_ops - direct) < 1.0e-12


def test_p0_e1_dipole_operator_expectations_match_observable_function():
    reference = _h2_reference()
    field = np.array([0.003, -0.001, 0.002])
    geometry = reference.geometry(
        electric=UniformElectricGauge(
            field=lambda _t: np.zeros(3),
            field_integral=lambda _t: field,
            lambda_value=lambda _t: 1.0,
            lambda_derivative=lambda _t: 0.0,
        )
    )
    rho = reference.initial_density()
    central = pyscf_central_dipole_matrices(reference)
    operators = p0_e1_dipole_operator_matrices(central, geometry, 0.0)

    from_ops = operator_expectations(rho, operators)
    direct = p0_e1_dipole_moment(rho, central, geometry, 0.0)

    assert np.linalg.norm(from_ops - direct) < 1.0e-12


def test_e1_central_delta_kick_preserves_metric_orthonormality():
    reference = _h2_reference()
    impulse = np.array([0.0, 0.0, 1.0e-3])
    geometry = reference.geometry(electric=velocity_delta_kick_electric_gauge(impulse))
    central = pyscf_central_dipole_matrices(reference)
    coeff = apply_p0_velocity_delta_kick(
        reference.initial_coefficients(),
        reference.geometry(),
        geometry,
    )
    coeff = apply_e1_central_delta_kick(
        coeff,
        impulse=impulse,
        central_dipoles0=central,
        geometry=geometry,
    )

    assert coefficient_orthonormality_error(coeff, geometry.metric(0.0)) < 1.0e-12


def test_e1_central_delta_kick_zero_impulse_is_identity():
    reference = _h2_reference()
    geometry = reference.geometry()
    central = pyscf_central_dipole_matrices(reference)
    coeff = reference.initial_coefficients()
    kicked = apply_e1_central_delta_kick(
        coeff,
        impulse=np.zeros(3),
        central_dipoles0=central,
        geometry=geometry,
    )

    assert np.linalg.norm(kicked - coeff) < 1.0e-12
