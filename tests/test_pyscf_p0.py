from __future__ import annotations

import numpy as np
import pytest

from aion import (
    PyscfP0LdaModel,
    PyscfP0Reference,
    UniformElectricGauge,
    VariableMetricSCEM,
    density_from_coefficients,
    p0_dipole_moment,
    p0_site_populations,
    record_p0_observables,
    summarize_p0_gauge_errors,
)


def _pyscf_modules():
    pytest.importorskip("pyscf")
    from pyscf import dft, gto, scf

    return dft, gto, scf


def _h2_mol():
    _, gto, _ = _pyscf_modules()
    return gto.M(
        atom="H 0 0 -0.37; H 0 0 0.37",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )


def _lda_h2_reference() -> PyscfP0Reference:
    dft, _, _ = _pyscf_modules()
    mf = dft.RKS(_h2_mol())
    mf.xc = "lda,vwn"
    mf.grids.level = 0
    mf.conv_tol = 1.0e-11
    mf.kernel()
    assert mf.converged
    return PyscfP0Reference.from_mean_field(mf)


def _constant_vector(vector: np.ndarray):
    vector = np.asarray(vector, dtype=float)
    return lambda _t: vector


def test_pyscf_p0_reference_extracts_ao_data_and_initial_state():
    _, _, scf = _pyscf_modules()
    mf = scf.RHF(_h2_mol())
    mf.conv_tol = 1.0e-12
    mf.kernel()
    assert mf.converged

    reference = PyscfP0Reference.from_mean_field(mf)
    density = reference.initial_density()
    coeff = reference.initial_coefficients()

    assert reference.anchors.natom == 2
    assert reference.anchors.nao == reference.mol.nao_nr()
    assert reference.overlap0.shape == (reference.nao, reference.nao)
    assert reference.hcore0.shape == (reference.nao, reference.nao)
    assert (
        np.linalg.norm(
            coeff.conj().T @ reference.overlap0 @ coeff - np.eye(reference.nocc)
        )
        < 1.0e-10
    )
    assert (
        abs(np.trace(density @ reference.overlap0).real - reference.mol.nelectron)
        < 1.0e-10
    )


def test_pyscf_p0_one_body_scem_runs_with_real_gaussian_matrices():
    reference = _lda_h2_reference()
    e = np.array([0.0, 0.0, 0.025])
    length = reference.geometry(
        electric=UniformElectricGauge.length(
            field=_constant_vector(e),
            field_integral=lambda t: e * t,
        )
    )
    velocity = reference.geometry(
        electric=UniformElectricGauge.velocity(
            field=_constant_vector(e),
            field_integral=lambda t: e * t,
        )
    )
    coeff_length = reference.initial_coefficients()
    coeff_velocity = reference.initial_coefficients()
    rt_length = VariableMetricSCEM(length, reference.linear_model(), reference.occupations)
    rt_velocity = VariableMetricSCEM(velocity, reference.linear_model(), reference.occupations)

    dt = 0.02
    for step in range(4):
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

    t_final = 4 * dt
    rho_length = density_from_coefficients(coeff_length, reference.occupations)
    rho_velocity = density_from_coefficients(coeff_velocity, reference.occupations)
    assert (
        np.linalg.norm(
            p0_site_populations(rho_length, length, t_final)
            - p0_site_populations(rho_velocity, velocity, t_final)
        )
        < 1.0e-10
    )
    assert (
        np.linalg.norm(
            p0_dipole_moment(rho_length, length, t_final)
            - p0_dipole_moment(rho_velocity, velocity, t_final)
        )
        < 1.0e-10
    )


def test_pyscf_p0_lda_zero_source_matches_pyscf_fock_and_energy():
    reference = _lda_h2_reference()
    geometry = reference.geometry()
    model = PyscfP0LdaModel.from_reference(reference)
    density = reference.initial_density()

    h = model.hamiltonian(density, 0.0, geometry)
    expected = reference.hcore0 + np.asarray(
        reference.mf.get_veff(reference.mol, density),
        dtype=np.complex128,
    )
    expected_energy = float(
        np.real(reference.mf.energy_tot(dm=density, h1e=reference.hcore0))
    )

    assert np.linalg.norm(h - expected) < 1.0e-10
    assert abs(model.energy(density, 0.0, geometry) - expected_energy) < 1.0e-10


def test_pyscf_p0_lda_scem_step_is_gauge_covariant_for_h2():
    reference = _lda_h2_reference()
    model = PyscfP0LdaModel.from_reference(reference)
    e = np.array([0.0, 0.0, 0.02])
    length = reference.geometry(
        electric=UniformElectricGauge.length(
            field=_constant_vector(e),
            field_integral=lambda t: e * t,
        )
    )
    velocity = reference.geometry(
        electric=UniformElectricGauge.velocity(
            field=_constant_vector(e),
            field_integral=lambda t: e * t,
        )
    )

    coeff_length = reference.initial_coefficients()
    coeff_velocity = reference.initial_coefficients()
    rt_length = VariableMetricSCEM(length, model, reference.occupations)
    rt_velocity = VariableMetricSCEM(velocity, model, reference.occupations)

    step_kwargs = dict(
        dt=0.02,
        midpoint_tolerance=1.0e-9,
        density_tolerance=1.0e-9,
        max_iterations=12,
        mixing=0.7,
    )
    coeff_length = rt_length.step(coeff_length, time=0.0, **step_kwargs).coeff_next
    coeff_velocity = rt_velocity.step(coeff_velocity, time=0.0, **step_kwargs).coeff_next
    rho_length = density_from_coefficients(coeff_length, reference.occupations)
    rho_velocity = density_from_coefficients(coeff_velocity, reference.occupations)

    assert rt_length.orthonormality_error(coeff_length, 0.02) < 1.0e-12
    assert rt_velocity.orthonormality_error(coeff_velocity, 0.02) < 1.0e-12
    assert (
        np.linalg.norm(
            p0_site_populations(rho_length, length, 0.02)
            - p0_site_populations(rho_velocity, velocity, 0.02)
        )
        < 1.0e-9
    )
    assert (
        np.linalg.norm(
            p0_dipole_moment(rho_length, length, 0.02)
            - p0_dipole_moment(rho_velocity, velocity, 0.02)
        )
        < 1.0e-9
    )


def test_pyscf_p0_lda_short_trajectory_diagnostics_are_gauge_covariant_for_h2():
    reference = _lda_h2_reference()
    model = PyscfP0LdaModel.from_reference(reference)
    field = np.array([0.0, 0.0, 0.02])

    def run(label: str, lambda_value: float):
        electric = UniformElectricGauge(
            field=_constant_vector(field),
            field_integral=lambda t: field * t,
            lambda_value=lambda _t: lambda_value,
            lambda_derivative=lambda _t: 0.0,
        )
        geometry = reference.geometry(electric=electric)
        rt = VariableMetricSCEM(geometry, model, reference.occupations)
        coeff = reference.initial_coefficients()
        rows = [
            {
                "gauge": label,
                **record_p0_observables(
                    step=0,
                    time=0.0,
                    coeff=coeff,
                    geometry=geometry,
                    model=model,
                    occupations=reference.occupations,
                ),
            }
        ]
        dt = 0.02
        for step in range(1, 4):
            result = rt.step(
                coeff,
                time=(step - 1) * dt,
                dt=dt,
                midpoint_tolerance=1.0e-9,
                density_tolerance=1.0e-9,
                max_iterations=14,
                mixing=0.7,
            )
            coeff = result.coeff_next
            rows.append(
                {
                    "gauge": label,
                    **record_p0_observables(
                        step=step,
                        time=step * dt,
                        coeff=coeff,
                        geometry=geometry,
                        model=model,
                        occupations=reference.occupations,
                        midpoint_iterations=result.iterations,
                        hamiltonian_residual=result.hamiltonian_residual,
                        density_residual=result.density_residual,
                    ),
                }
            )
        return rows

    rows_by_gauge = {
        "length": run("length", 0.0),
        "velocity": run("velocity", 1.0),
    }
    summary = summarize_p0_gauge_errors(rows_by_gauge)

    assert summary["velocity"]["max_dipole_norm_error"] < 1.0e-8
    assert summary["velocity"]["max_population_norm_error"] < 1.0e-8
    assert summary["velocity"]["max_energy_abs_error"] < 1.0e-8
    assert summary["length"]["max_orthonormality_error"] < 1.0e-11
    assert summary["velocity"]["max_orthonormality_error"] < 1.0e-11
    assert summary["length"]["max_instantaneous_continuity_residual"] < 1.0e-8
    assert summary["velocity"]["max_instantaneous_continuity_residual"] < 1.0e-8
    assert "source_power" in rows_by_gauge["length"][0]
    assert "current_0_1" in rows_by_gauge["length"][0]


def test_pyscf_p0_lda_rejects_non_lda_functional():
    dft, _, _ = _pyscf_modules()
    mf = dft.RKS(_h2_mol())
    mf.xc = "pbe"
    mf.grids.level = 0
    mf.kernel()
    reference = PyscfP0Reference.from_mean_field(mf)
    with pytest.raises(NotImplementedError, match="LDA only"):
        PyscfP0LdaModel.from_reference(reference)
