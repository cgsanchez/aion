from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from aion import (
    P0SCEMSettings,
    PyscfP0DftModel,
    PyscfP0LdaModel,
    PyscfP0Reference,
    UniformElectricGauge,
    UniformMagneticGauge,
    VariableMetricSCEM,
    density_from_coefficients,
    p0_dipole_moment,
    p0_site_populations,
    run_p0_scem_trajectory,
    record_p0_observables,
    run_p0_uniform_electric_gauge_comparison,
    summarize_p0_gauge_errors,
    transform_p0_coefficients_between_gauges,
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


def _pbe_h2_reference() -> PyscfP0Reference:
    dft, _, _ = _pyscf_modules()
    mf = dft.RKS(_h2_mol())
    mf.xc = "pbe,pbe"
    mf.grids.level = 0
    mf.grids.prune = None
    mf.small_rho_cutoff = 0.0
    mf.conv_tol = 1.0e-11
    mf.kernel()
    assert mf.converged
    return PyscfP0Reference.from_mean_field(mf)


def _lda_reference(atom: str) -> PyscfP0Reference:
    dft, gto, _ = _pyscf_modules()
    mol = gto.M(
        atom=atom,
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


def _constant_vector(vector: np.ndarray):
    vector = np.asarray(vector, dtype=float)
    return lambda _t: vector


def _metric_orthonormalize(coeff: np.ndarray, metric: np.ndarray) -> np.ndarray:
    overlap = coeff.conj().T @ metric @ coeff
    eig, vec = scipy.linalg.eigh(overlap, check_finite=False)
    invsqrt = (vec * eig**-0.5) @ vec.conj().T
    return coeff @ invsqrt


def _matrix_trace_product(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.einsum("ij,ji->", left, right).real)


def _real_symmetric_direction(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(size, size))
    direction = 0.5 * (raw + raw.T)
    return direction / np.linalg.norm(direction)


def _complex_hermitian_direction(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.normal(size=(size, size))
    imag = rng.normal(size=(size, size))
    direction = 0.5 * (real + real.T) + 0.5j * (imag - imag.T)
    return direction / np.linalg.norm(direction)


def _assert_matrix_is_functional_derivative(
    *,
    functional,
    density: np.ndarray,
    potential: np.ndarray,
    direction: np.ndarray,
    tolerance: float,
) -> None:
    expected = _matrix_trace_product(direction, potential)
    eps_values = (1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5)
    errors = []
    for eps in eps_values:
        fd = (
            functional(density + eps * direction)
            - functional(density - eps * direction)
        ) / (2.0 * eps)
        errors.append(abs(fd - expected))

    best_error = min(errors)
    assert best_error < tolerance, (
        f"functional derivative mismatch: expected {expected:.16e}, "
        f"best error {best_error:.3e}, errors {errors}"
    )


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


def test_pyscf_p0_dft_accepts_pure_gga_and_rejects_hybrids():
    reference = _pbe_h2_reference()
    model = PyscfP0DftModel.from_reference(reference)
    geometry = reference.geometry()
    density = reference.initial_density()

    assert np.isfinite(model.energy(density, 0.0, geometry))

    reference.mf.xc = "pbe0"
    with pytest.raises(NotImplementedError, match="hybrid"):
        PyscfP0DftModel.from_reference(reference)


def test_pyscf_pbe_veff_is_bare_hxc_energy_derivative():
    reference = _pbe_h2_reference()
    density = reference.initial_density()
    veff = np.asarray(
        reference.mf.get_veff(reference.mol, density),
        dtype=np.complex128,
    )

    def hxc_energy(dm: np.ndarray) -> float:
        total = reference.mf.energy_tot(dm=dm, h1e=reference.hcore0)
        return float(np.real(total)) - _matrix_trace_product(dm, reference.hcore0)

    for direction in (
        _real_symmetric_direction(reference.nao, seed=1401),
        _complex_hermitian_direction(reference.nao, seed=1402),
    ):
        _assert_matrix_is_functional_derivative(
            functional=hxc_energy,
            density=density,
            potential=veff,
            direction=direction,
            tolerance=2.0e-7,
        )


def test_pyscf_p0_pbe_veff_is_dressed_hxc_energy_derivative():
    reference = _pbe_h2_reference()
    model = PyscfP0DftModel.from_reference(reference)
    field = np.array([0.011, -0.017, 0.023])
    electric = UniformElectricGauge(
        field=_constant_vector(field),
        field_integral=lambda t: field * t,
        lambda_value=lambda _t: 0.65,
        lambda_derivative=lambda _t: 0.0,
    )
    magnetic = UniformMagneticGauge(
        np.array([0.007, -0.011, 0.019]),
        gauge="symmetric",
        origin=np.array([0.13, -0.07, 0.05]),
    )
    geometry = reference.geometry(electric=electric, magnetic=magnetic)
    time = 0.43
    theta = geometry.theta(time)
    density0 = reference.initial_density()
    density = theta * density0
    hcore_p0 = geometry.dress_matrix(reference.hcore0, time)
    veff_p0 = model.hamiltonian(density, time, geometry) - hcore_p0

    def hxc_energy(rho: np.ndarray) -> float:
        return model.energy(rho, time, geometry) - _matrix_trace_product(
            rho,
            hcore_p0,
        )

    for direction in (
        _real_symmetric_direction(reference.nao, seed=2401),
        _complex_hermitian_direction(reference.nao, seed=2402),
    ):
        _assert_matrix_is_functional_derivative(
            functional=hxc_energy,
            density=density,
            potential=veff_p0,
            direction=direction,
            tolerance=2.0e-7,
        )


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


@pytest.mark.parametrize(
    ("atom", "field"),
    [
        (
            "O 0.000000 0.000000 0.000000; "
            "H 0.758602 0.000000 0.504284; "
            "H -0.758602 0.000000 0.504284",
            np.array([0.011, -0.017, 0.023]),
        ),
        (
            "C 0.000000 0.000000 0.000000; "
            "H 0.629118 0.629118 0.629118; "
            "H -0.629118 -0.629118 0.629118; "
            "H -0.629118 0.629118 -0.629118; "
            "H 0.629118 -0.629118 -0.629118",
            np.array([0.011, -0.017, 0.023]),
        ),
    ],
)
def test_pyscf_p0_lda_short_trajectory_is_gauge_covariant_for_nonlinear_molecules(
    atom: str,
    field: np.ndarray,
):
    reference = _lda_reference(atom)
    model = PyscfP0LdaModel.from_reference(reference)
    settings = P0SCEMSettings(
        dt=0.02,
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

    assert summary["mixed"]["max_dipole_norm_error"] < 1.0e-8
    assert summary["velocity"]["max_dipole_norm_error"] < 1.0e-8
    assert summary["mixed"]["max_population_norm_error"] < 1.0e-8
    assert summary["velocity"]["max_population_norm_error"] < 1.0e-8
    assert summary["mixed"]["max_energy_abs_error"] < 1.0e-8
    assert summary["velocity"]["max_energy_abs_error"] < 1.0e-8


def test_pyscf_p0_lda_static_magnetic_symmetric_and_landau_gauges_are_covariant():
    reference = _lda_reference(
        "O 0.000000 0.000000 0.000000; "
        "H 0.758602 0.000000 0.504284; "
        "H -0.758602 0.000000 0.504284"
    )
    model = PyscfP0LdaModel.from_reference(reference)
    magnetic_field = np.array([0.0, 0.0, 0.05])
    origin = np.array([0.1, -0.2, 0.0])
    symmetric = reference.geometry(
        magnetic=UniformMagneticGauge(
            magnetic_field,
            gauge="symmetric",
            origin=origin,
        )
    )
    landau = reference.geometry(
        magnetic=UniformMagneticGauge(
            magnetic_field,
            gauge="landau",
            origin=origin,
            landau_u=np.array([1.0, 0.0, 0.0]),
        )
    )
    coeff_symmetric = _metric_orthonormalize(
        reference.initial_coefficients(),
        symmetric.metric(0.0),
    )
    coeff_landau = transform_p0_coefficients_between_gauges(
        coeff_symmetric,
        symmetric,
        landau,
    )
    settings = P0SCEMSettings(
        dt=0.02,
        nsteps=2,
        midpoint_tolerance=1.0e-9,
        density_tolerance=1.0e-9,
        max_iterations=18,
        mixing=0.7,
    )

    rows_by_gauge = {
        "symmetric": run_p0_scem_trajectory(
            label="symmetric",
            geometry=symmetric,
            model=model,
            occupations=reference.occupations,
            coeff0=coeff_symmetric,
            settings=settings,
        ),
        "landau": run_p0_scem_trajectory(
            label="landau",
            geometry=landau,
            model=model,
            occupations=reference.occupations,
            coeff0=coeff_landau,
            settings=settings,
        ),
    }
    summary = summarize_p0_gauge_errors(rows_by_gauge, reference_gauge="symmetric")

    assert summary["landau"]["max_dipole_norm_error"] < 1.0e-8
    assert summary["landau"]["max_population_norm_error"] < 1.0e-8
    assert summary["landau"]["max_energy_abs_error"] < 1.0e-8
    assert summary["landau"]["max_orthonormality_error"] < 1.0e-11


def test_pyscf_p0_lda_rejects_non_lda_functional():
    dft, _, _ = _pyscf_modules()
    mf = dft.RKS(_h2_mol())
    mf.xc = "pbe"
    mf.grids.level = 0
    mf.kernel()
    reference = PyscfP0Reference.from_mean_field(mf)
    with pytest.raises(NotImplementedError, match="LDA only"):
        PyscfP0LdaModel.from_reference(reference)
