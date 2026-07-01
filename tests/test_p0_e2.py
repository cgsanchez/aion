from __future__ import annotations

import numpy as np
import pytest

from aion import (
    UniformElectricGauge,
    density_from_coefficients,
    p0_e2_full_second_moment_matrices,
    p0_e2_second_moment,
    pyscf_central_dipole_matrices,
    pyscf_central_second_moment_matrices,
    traceless_quadrupole_from_second_moment,
)


def _pyscf_modules():
    pytest.importorskip("pyscf")
    from pyscf import dft, gto

    return dft, gto


def _constant_vector(vector: np.ndarray):
    vector = np.asarray(vector, dtype=float)
    return lambda _t: vector


def _water_reference():
    from aion import PyscfP0Reference

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


def test_p0_e2_reconstructs_ordinary_second_moment_operator():
    reference = _water_reference()
    geometry = reference.geometry()
    central_dipoles = pyscf_central_dipole_matrices(reference)
    central_second = pyscf_central_second_moment_matrices(reference)
    full = p0_e2_full_second_moment_matrices(
        central_dipoles0=central_dipoles,
        central_second_moments0=central_second,
        geometry=geometry,
        t=0.0,
    )
    rr = np.asarray(reference.mol.intor("int1e_rr", comp=9), dtype=np.complex128)
    ordinary = geometry.charge * rr.reshape(3, 3, reference.nao, reference.nao)

    assert np.linalg.norm(full - ordinary) < 1.0e-10


def test_p0_e2_second_moment_matches_ordinary_expectation():
    reference = _water_reference()
    geometry = reference.geometry()
    density = density_from_coefficients(
        reference.initial_coefficients(),
        reference.occupations,
    )
    central_dipoles = pyscf_central_dipole_matrices(reference)
    central_second = pyscf_central_second_moment_matrices(reference)
    full = p0_e2_full_second_moment_matrices(
        central_dipoles0=central_dipoles,
        central_second_moments0=central_second,
        geometry=geometry,
        t=0.0,
    )
    moment = p0_e2_second_moment(density, full)

    rr = np.asarray(reference.mol.intor("int1e_rr", comp=9), dtype=np.complex128)
    ordinary = geometry.charge * rr.reshape(3, 3, reference.nao, reference.nao)
    expected = np.empty((3, 3), dtype=float)
    for a in range(3):
        for b in range(3):
            expected[a, b] = np.trace(density @ ordinary[a, b]).real

    assert np.linalg.norm(moment - expected) < 1.0e-10


def test_p0_e2_dressed_second_moment_is_hermitian_in_mixed_gauge():
    from aion import dressed_central_second_moment_matrices

    reference = _water_reference()
    field = np.array([0.011, -0.017, 0.023])
    geometry = reference.geometry(
        electric=UniformElectricGauge(
            field=_constant_vector(field),
            field_integral=lambda t: field * t,
            lambda_value=lambda _t: 0.5,
            lambda_derivative=lambda _t: 0.0,
        )
    )
    central_second = pyscf_central_second_moment_matrices(reference)
    dressed = dressed_central_second_moment_matrices(central_second, geometry, 0.37)

    for a in range(3):
        for b in range(3):
            assert np.linalg.norm(dressed[a, b] - dressed[a, b].conj().T) < 1.0e-14


def test_traceless_quadrupole_map_has_zero_trace():
    moment = np.array(
        [
            [1.2, 0.1, -0.2],
            [0.1, -0.4, 0.3],
            [-0.2, 0.3, 2.1],
        ]
    )
    quadrupole = traceless_quadrupole_from_second_moment(moment)

    assert abs(np.trace(quadrupole)) < 1.0e-14
    assert np.linalg.norm(quadrupole - quadrupole.T) < 1.0e-14
