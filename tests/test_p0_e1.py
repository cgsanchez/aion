from __future__ import annotations

import numpy as np
import pytest

from aion import (
    P0E1Model,
    P0SCEMSettings,
    PyscfP0DftModel,
    PyscfP0Reference,
    UniformElectricGauge,
    central_dipole_matrices,
    density_from_coefficients,
    dressed_central_dipole_matrices,
    p0_e1_dipole_moment,
    p0_e1_uniform_electric_potential,
    p0_pair_scalar_potential_matrix,
    pyscf_central_dipole_matrices,
    run_p0_uniform_electric_gauge_comparison,
    summarize_p0_gauge_errors,
    transform_p0_coefficients_between_gauges,
)


def _pyscf_modules():
    pytest.importorskip("pyscf")
    from pyscf import dft, gto

    return dft, gto


def _constant_vector(vector: np.ndarray):
    vector = np.asarray(vector, dtype=float)
    return lambda _t: vector


def _water_reference() -> PyscfP0Reference:
    dft, gto = _pyscf_modules()
    mol = gto.M(
        atom=(
            "O 0.000000 0.000000 0.000000; "
            "H 0.758602 0.000000 0.504284; "
            "H -0.758602 0.000000 0.504284"
        ),
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


def test_central_dipole_matrices_match_definition_and_are_hermitian():
    reference = _water_reference()
    position = np.asarray(reference.mol.intor("int1e_r", comp=3), dtype=np.complex128)
    dipoles = central_dipole_matrices(
        anchors=reference.anchors,
        overlap0=reference.overlap0,
        position_matrices=position,
        charge=-1.0,
    )
    dipoles_from_reference = pyscf_central_dipole_matrices(reference)

    assert np.linalg.norm(dipoles - dipoles_from_reference) < 1.0e-14
    for axis in range(3):
        assert np.linalg.norm(dipoles[axis] - dipoles[axis].conj().T) < 1.0e-14


def test_p0_e1_length_gauge_reconstructs_ordinary_ao_dipole_coupling():
    reference = _water_reference()
    field = np.array([0.011, -0.017, 0.023])
    electric = UniformElectricGauge.length(
        field=_constant_vector(field),
        field_integral=lambda t: field * t,
    )
    geometry = reference.geometry(electric=electric)
    central_dipoles = pyscf_central_dipole_matrices(reference)
    time = 0.37

    scalar_pair = p0_pair_scalar_potential_matrix(geometry, time)
    spread = p0_e1_uniform_electric_potential(central_dipoles, geometry, time)
    position = np.asarray(reference.mol.intor("int1e_r", comp=3), dtype=np.complex128)
    ordinary_length = -reference.geometry().charge * np.einsum(
        "x,xij->ij",
        field,
        position,
    )

    assert np.linalg.norm((scalar_pair + spread) - ordinary_length) < 1.0e-10


def test_dressed_central_dipoles_are_hermitian_in_mixed_gauge():
    reference = _water_reference()
    field = np.array([0.011, -0.017, 0.023])
    electric = UniformElectricGauge(
        field=_constant_vector(field),
        field_integral=lambda t: field * t,
        lambda_value=lambda _t: 0.5,
        lambda_derivative=lambda _t: 0.0,
    )
    geometry = reference.geometry(electric=electric)
    central_dipoles = pyscf_central_dipole_matrices(reference)
    dressed = dressed_central_dipole_matrices(central_dipoles, geometry, 0.37)

    for axis in range(3):
        assert np.linalg.norm(dressed[axis] - dressed[axis].conj().T) < 1.0e-14


def test_p0_e1_model_zero_field_reduces_to_base_model():
    reference = _water_reference()
    geometry = reference.geometry()
    base_model = PyscfP0DftModel.from_reference(reference)
    central_dipoles = pyscf_central_dipole_matrices(reference)
    model = P0E1Model(base_model, central_dipoles)
    density = reference.initial_density()

    assert (
        np.linalg.norm(
            model.hamiltonian(density, 0.0, geometry)
            - base_model.hamiltonian(density, 0.0, geometry)
        )
        < 1.0e-14
    )
    assert (
        abs(
            model.energy(density, 0.0, geometry)
            - base_model.energy(density, 0.0, geometry)
        )
        < 1.0e-12
    )


def test_p0_e1_dipole_reconstructs_ordinary_ao_dipole_in_length_gauge():
    reference = _water_reference()
    geometry = reference.geometry()
    central_dipoles = pyscf_central_dipole_matrices(reference)
    density = reference.initial_density()
    position = np.asarray(reference.mol.intor("int1e_r", comp=3), dtype=np.complex128)
    ordinary_dipole = np.asarray(
        [
            reference.geometry().charge * np.trace(density @ position[axis]).real
            for axis in range(3)
        ]
    )

    assert (
        np.linalg.norm(
            p0_e1_dipole_moment(density, central_dipoles, geometry, 0.0)
            - ordinary_dipole
        )
        < 1.0e-10
    )


def test_p0_e1_short_trajectory_is_gauge_covariant_for_water():
    reference = _water_reference()
    base_model = PyscfP0DftModel.from_reference(reference)
    central_dipoles = pyscf_central_dipole_matrices(reference)
    model = P0E1Model(base_model, central_dipoles)
    field = np.array([0.0011, -0.0017, 0.0023])
    settings = P0SCEMSettings(
        dt=0.01,
        nsteps=2,
        midpoint_tolerance=1.0e-9,
        density_tolerance=1.0e-9,
        max_iterations=18,
        mixing=0.7,
    )

    rows_by_gauge = run_p0_uniform_electric_gauge_comparison(
        geometry_factory=lambda electric: reference.geometry(electric=electric),
        model=model,
        occupations=reference.occupations,
        coeff0=reference.initial_coefficients(),
        field=field,
        settings=settings,
    )
    summary = summarize_p0_gauge_errors(rows_by_gauge)

    assert summary["mixed"]["max_population_norm_error"] < 1.0e-8
    assert summary["velocity"]["max_population_norm_error"] < 1.0e-8
    assert summary["mixed"]["max_energy_abs_error"] < 1.0e-8
    assert summary["velocity"]["max_energy_abs_error"] < 1.0e-8


def test_p0_e1_dipole_is_gauge_invariant_for_transformed_density():
    reference = _water_reference()
    field = np.array([0.011, -0.017, 0.023])
    central_dipoles = pyscf_central_dipole_matrices(reference)
    length = reference.geometry(
        electric=UniformElectricGauge.length(
            field=_constant_vector(field),
            field_integral=lambda t: field * t,
        )
    )
    velocity = reference.geometry(
        electric=UniformElectricGauge.velocity(
            field=_constant_vector(field),
            field_integral=lambda t: field * t,
        )
    )
    time = 0.4
    coeff_length = reference.initial_coefficients()
    coeff_velocity = transform_p0_coefficients_between_gauges(
        coeff_length,
        length,
        velocity,
        time=time,
    )
    rho_length = density_from_coefficients(coeff_length, reference.occupations)
    rho_velocity = density_from_coefficients(coeff_velocity, reference.occupations)

    assert (
        np.linalg.norm(
            p0_e1_dipole_moment(rho_length, central_dipoles, length, time)
            - p0_e1_dipole_moment(rho_velocity, central_dipoles, velocity, time)
        )
        < 1.0e-10
    )
