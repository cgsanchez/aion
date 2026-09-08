from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from aion import (
    P0E1Model,
    PyscfP0DftModel,
    PyscfP0Reference,
    UniformElectricGauge,
    coefficient_derivative,
    density_derivative_from_coefficients,
    density_from_coefficients,
    p0_continuity_residual,
    p0_e1_dipole_derivative,
    p0_e1_dipole_moment,
    p0_e1_intrinsic_dipole_derivative,
    p0_e1_uniform_electric_potential,
    p0_e1_uniform_residual_current,
    p0_graph_current_vector,
    p0_graph_currents,
    p0_pair_scalar_potential_matrix,
    p0_site_charge_derivative,
    pyscf_central_dipole_matrices,
    record_p0_observables,
    transform_p0_coefficients_between_gauges,
)


@pytest.fixture(scope="module")
def lih_reference() -> PyscfP0Reference:
    pytest.importorskip("pyscf")
    from pyscf import dft, gto

    half_bond = 0.5 * 1.5956
    mol = gto.M(
        atom=f"Li 0 0 {-half_bond}; H 0 0 {half_bond}",
        basis="cc-pvdz",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "lda,vwn"
    mf.grids.level = 0
    mf.grids.prune = None
    mf.small_rho_cutoff = 0.0
    mf.conv_tol = 1.0e-11
    mf.kernel()
    assert mf.converged
    return PyscfP0Reference.from_mean_field(mf)


def _constant_source(
    field: np.ndarray,
    *,
    lambda_value,
    lambda_derivative,
) -> UniformElectricGauge:
    field = np.asarray(field, dtype=float)
    return UniformElectricGauge(
        field=lambda _t: field,
        field_integral=lambda t: field * t,
        lambda_value=lambda_value,
        lambda_derivative=lambda_derivative,
    )


def _excited_test_coefficients(reference: PyscfP0Reference) -> np.ndarray:
    coeff = reference.initial_coefficients()
    virtual = np.asarray(reference.mf.mo_coeff)[:, reference.nocc]
    coeff[:, -1] = coeff[:, -1] + 0.06j * virtual
    gram = coeff.conj().T @ reference.overlap0 @ coeff
    eig, vec = scipy.linalg.eigh(gram, check_finite=False)
    invsqrt = (vec * eig**-0.5) @ vec.conj().T
    return coeff @ invsqrt


def _vector(row: dict[str, object], prefix: str) -> np.ndarray:
    return np.asarray([row[f"{prefix}_{axis}"] for axis in "xyz"], dtype=float)


def _intrinsic_dipole_for_theta(
    density: np.ndarray,
    theta: np.ndarray,
    central_dipoles: np.ndarray,
) -> np.ndarray:
    values = []
    for axis in range(3):
        matrix = theta * central_dipoles[axis]
        matrix = 0.5 * (matrix + matrix.conj().T)
        values.append(np.trace(density @ matrix).real)
    return np.asarray(values)


def test_lih_e1_is_nonzero_and_reconstructs_length_gauge(
    lih_reference: PyscfP0Reference,
):
    field = np.array([0.0023, -0.0017, 0.0031])
    source = UniformElectricGauge.length(
        field=lambda _t: field,
        field_integral=lambda t: field * t,
    )
    geometry = lih_reference.geometry(electric=source)
    central = pyscf_central_dipole_matrices(lih_reference)
    position = np.asarray(
        lih_reference.mol.intor("int1e_r", comp=3),
        dtype=np.complex128,
    )
    reconstructed = p0_pair_scalar_potential_matrix(
        geometry,
        0.37,
    ) + p0_e1_uniform_electric_potential(central, geometry, 0.37)
    ordinary = -geometry.charge * np.einsum("x,xij->ij", field, position)

    assert np.linalg.norm(central) > 1.0
    assert np.linalg.norm(reconstructed - ordinary) < 1.0e-10

    density = lih_reference.initial_density()
    expected_dipole = np.asarray(
        [geometry.charge * np.trace(density @ position[a]).real for a in range(3)]
    )
    actual_dipole = p0_e1_dipole_moment(density, central, geometry, 0.37)
    assert np.linalg.norm(actual_dipole - expected_dipole) < 1.0e-10


def test_lih_euler_lagrange_current_and_action_split(
    lih_reference: PyscfP0Reference,
):
    field = np.array([0.0023, -0.0017, 0.0031])
    source = _constant_source(
        field,
        lambda_value=lambda _t: 0.4,
        lambda_derivative=lambda _t: 0.0,
    )
    geometry = lih_reference.geometry(electric=source)
    central = pyscf_central_dipole_matrices(lih_reference)
    model = P0E1Model(PyscfP0DftModel.from_reference(lih_reference), central)
    coeff = _excited_test_coefficients(lih_reference)
    time = 0.37
    density = density_from_coefficients(coeff, lih_reference.occupations)
    hamiltonian = model.hamiltonian(density, time, geometry)
    coeff_dot = coefficient_derivative(coeff, hamiltonian, geometry, time)
    density_dot = density_derivative_from_coefficients(
        coeff,
        coeff_dot,
        lih_reference.occupations,
    )
    charge_dot = p0_site_charge_derivative(
        density,
        density_dot,
        geometry,
        time,
    )

    intrinsic = p0_e1_intrinsic_dipole_derivative(
        density=density,
        density_dot=density_dot,
        central_dipoles0=central,
        geometry=geometry,
        t=time,
    )
    residual = p0_e1_uniform_residual_current(
        density=density,
        density_dot=density_dot,
        central_dipoles0=central,
        geometry=geometry,
        t=time,
        electric_field=field,
    )
    phase_response = residual - intrinsic

    eps = 1.0e-6
    intrinsic_fd = (
        p0_e1_dipole_moment(
            density + eps * density_dot,
            central,
            geometry,
            time + eps,
            include_site=False,
        )
        - p0_e1_dipole_moment(
            density - eps * density_dot,
            central,
            geometry,
            time - eps,
            include_site=False,
        )
    ) / (2.0 * eps)

    ao_coords = geometry.anchors.atom_coords[geometry.anchors.ao_to_atom]
    delta_r = ao_coords[:, None, :] - ao_coords[None, :, :]
    theta = geometry.theta(time)
    phase_fd = np.empty(3)
    for beta in range(3):
        perturbation = np.exp(
            1j
            * geometry.charge
            / geometry.hbar
            * eps
            * delta_r[:, :, beta]
        )
        plus = _intrinsic_dipole_for_theta(
            density,
            theta * perturbation,
            central,
        )
        minus = _intrinsic_dipole_for_theta(
            density,
            theta / perturbation,
            central,
        )
        phase_fd[beta] = np.dot(field, plus - minus) / (2.0 * eps)

    base_hamiltonian = hamiltonian - model.e1_hamiltonian(time, geometry)
    full_graph = p0_graph_currents(density, hamiltonian, geometry, time)
    split_graph = p0_graph_currents(
        density,
        hamiltonian,
        geometry,
        time,
        source_hamiltonian=base_hamiltonian,
    )
    full_current = p0_graph_current_vector(full_graph, geometry)
    split_current = p0_graph_current_vector(split_graph, geometry)
    dipole_dot = p0_e1_dipole_derivative(
        charge_derivative=charge_dot,
        density=density,
        density_dot=density_dot,
        central_dipoles0=central,
        geometry=geometry,
        t=time,
    )

    assert np.linalg.norm(intrinsic - intrinsic_fd) < 1.0e-8
    assert np.linalg.norm(phase_response - phase_fd) < 1.0e-8
    assert np.linalg.norm(residual - (intrinsic_fd + phase_fd)) < 2.0e-8
    assert np.linalg.norm(full_current - split_current - phase_response) < 1.0e-11
    assert np.linalg.norm(split_current + residual - dipole_dot) < 1.0e-11
    assert np.linalg.norm(full_current + intrinsic - dipole_dot) < 1.0e-11
    assert np.linalg.norm(p0_continuity_residual(charge_dot, full_graph)) < 1.0e-11


def test_lih_total_current_is_covariant_for_static_and_time_dependent_gauges(
    lih_reference: PyscfP0Reference,
):
    field = np.array([0.0023, -0.0017, 0.0031])
    length_source = UniformElectricGauge.length(
        field=lambda _t: field,
        field_integral=lambda t: field * t,
    )
    length_geometry = lih_reference.geometry(electric=length_source)
    central = pyscf_central_dipole_matrices(lih_reference)
    model = P0E1Model(PyscfP0DftModel.from_reference(lih_reference), central)
    coeff_length = _excited_test_coefficients(lih_reference)
    time = 0.73

    sources = {
        "mixed": _constant_source(
            field,
            lambda_value=lambda _t: 0.4,
            lambda_derivative=lambda _t: 0.0,
        ),
        "velocity": _constant_source(
            field,
            lambda_value=lambda _t: 1.0,
            lambda_derivative=lambda _t: 0.0,
        ),
        "time_dependent": _constant_source(
            field,
            lambda_value=lambda t: 0.45 + 0.20 * np.sin(0.6 * t),
            lambda_derivative=lambda t: 0.12 * np.cos(0.6 * t),
        ),
    }
    reference_row = record_p0_observables(
        step=0,
        time=time,
        coeff=coeff_length,
        geometry=length_geometry,
        model=model,
        occupations=lih_reference.occupations,
    )

    for source in sources.values():
        geometry = lih_reference.geometry(electric=source)
        coeff = transform_p0_coefficients_between_gauges(
            coeff_length,
            length_geometry,
            geometry,
            time=time,
        )
        row = record_p0_observables(
            step=0,
            time=time,
            coeff=coeff,
            geometry=geometry,
            model=model,
            occupations=lih_reference.occupations,
        )

        assert abs(float(row["energy"]) - float(reference_row["energy"])) < 1.0e-9
        assert abs(
            float(row["electron_count"]) - float(reference_row["electron_count"])
        ) < 1.0e-10
        assert np.linalg.norm(
            _vector(row, "e1_dipole") - _vector(reference_row, "e1_dipole")
        ) < 1.0e-9
        assert np.linalg.norm(
            _vector(row, "source_current")
            - _vector(reference_row, "source_current")
        ) < 1.0e-9
        assert float(row["source_current_dipole_derivative_residual_norm"]) < 1.0e-10
