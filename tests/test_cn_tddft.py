from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg

from aion.cn_tddft import LengthGaugeCNRTTDDFT, MidpointConvergenceError


def test_cn_step_preserves_s_metric_for_fixed_hamiltonian():
    rng = np.random.default_rng(1234)
    nao = 16
    nocc = 5

    a = rng.normal(size=(nao, nao)) + 1j * rng.normal(size=(nao, nao))
    s = a.conj().T @ a + np.eye(nao)

    b = rng.normal(size=(nao, nao)) + 1j * rng.normal(size=(nao, nao))
    h = 0.5 * (b + b.conj().T)

    c0 = rng.normal(size=(nao, nocc)) + 1j * rng.normal(size=(nao, nocc))
    metric = c0.conj().T @ s @ c0
    eig, vec = scipy.linalg.eigh(metric, check_finite=False)
    c = c0 @ ((vec * eig**-0.5) @ vec.conj().T)

    rt = LengthGaugeCNRTTDDFT.__new__(LengthGaugeCNRTTDDFT)
    rt.s = s
    rt.hbar = 1.0
    rt.nocc = nocc

    for _ in range(100):
        c = rt.cn_step(c, h, 0.2)

    err = np.linalg.norm(c.conj().T @ s @ c - np.eye(nocc))
    assert err < 1.0e-11


def test_delta_kick_preserves_s_metric():
    rng = np.random.default_rng(4321)
    nao = 12
    nocc = 4

    a = rng.normal(size=(nao, nao))
    s = a.T @ a + np.eye(nao)

    rint = rng.normal(size=(3, nao, nao))
    rint = 0.5 * (rint + np.swapaxes(rint, 1, 2))

    c0 = rng.normal(size=(nao, nocc)) + 1j * rng.normal(size=(nao, nocc))
    metric = c0.conj().T @ s @ c0
    eig, vec = scipy.linalg.eigh(metric, check_finite=False)
    c = c0 @ ((vec * eig**-0.5) @ vec.conj().T)

    rt = LengthGaugeCNRTTDDFT.__new__(LengthGaugeCNRTTDDFT)
    rt.s = s.astype(np.complex128)
    rt._s_cho = scipy.linalg.cho_factor(rt.s, lower=True, check_finite=False)
    s_eval, s_evec = scipy.linalg.eigh(rt.s, check_finite=False)
    rt.s_sqrt = (s_evec * s_eval**0.5) @ s_evec.conj().T
    rt.s_invsqrt = (s_evec * s_eval**-0.5) @ s_evec.conj().T
    rt.hbar = 1.0
    rt.charge = -1.0
    rt.nocc = nocc
    rt.dipole_position = rint.astype(np.complex128)

    kicked = rt.apply_delta_kick(c, np.array([1.0e-3, -2.0e-3, 1.5e-3]))

    err = np.linalg.norm(kicked.conj().T @ s @ kicked - np.eye(nocc))
    assert err < 1.0e-11


def test_exponential_step_preserves_s_metric_for_fixed_hamiltonian():
    rng = np.random.default_rng(2468)
    nao = 14
    nocc = 4

    a = rng.normal(size=(nao, nao)) + 1j * rng.normal(size=(nao, nao))
    s = a.conj().T @ a + np.eye(nao)
    s_eval, s_evec = scipy.linalg.eigh(s, check_finite=False)

    b = rng.normal(size=(nao, nao)) + 1j * rng.normal(size=(nao, nao))
    h = 0.5 * (b + b.conj().T)

    c0 = rng.normal(size=(nao, nocc)) + 1j * rng.normal(size=(nao, nocc))
    metric = c0.conj().T @ s @ c0
    eig, vec = scipy.linalg.eigh(metric, check_finite=False)
    c = c0 @ ((vec * eig**-0.5) @ vec.conj().T)

    rt = LengthGaugeCNRTTDDFT.__new__(LengthGaugeCNRTTDDFT)
    rt.s = s
    rt.s_sqrt = (s_evec * s_eval**0.5) @ s_evec.conj().T
    rt.s_invsqrt = (s_evec * s_eval**-0.5) @ s_evec.conj().T
    rt.hbar = 1.0
    rt.nocc = nocc

    for _ in range(20):
        c = rt.exponential_step(c, h, 0.3)

    err = np.linalg.norm(c.conj().T @ s @ c - np.eye(nocc))
    assert err < 1.0e-11


def test_scem_constant_hamiltonian_matches_frozen_exponential():
    rng = np.random.default_rng(1357)
    nao = 10
    nocc = 3

    a = rng.normal(size=(nao, nao)) + 1j * rng.normal(size=(nao, nao))
    s = a.conj().T @ a + np.eye(nao)
    s_eval, s_evec = scipy.linalg.eigh(s, check_finite=False)

    b = rng.normal(size=(nao, nao)) + 1j * rng.normal(size=(nao, nao))
    h = 0.5 * (b + b.conj().T)

    c0 = rng.normal(size=(nao, nocc)) + 1j * rng.normal(size=(nao, nocc))
    metric = c0.conj().T @ s @ c0
    eig, vec = scipy.linalg.eigh(metric, check_finite=False)
    c = c0 @ ((vec * eig**-0.5) @ vec.conj().T)

    rt = LengthGaugeCNRTTDDFT.__new__(LengthGaugeCNRTTDDFT)
    rt.s = s
    rt.s_sqrt = (s_evec * s_eval**0.5) @ s_evec.conj().T
    rt.s_invsqrt = (s_evec * s_eval**-0.5) @ s_evec.conj().T
    rt.hbar = 1.0
    rt.nocc = nocc
    rt.occ = np.full(nocc, 2.0)
    rt.hamiltonian_from_density = lambda _rho, _time: (h, None)

    result = rt.scem_step(
        c,
        time=0.0,
        dt=0.37,
        h_guess=h,
        midpoint_tolerance=1.0e-12,
        density_tolerance=1.0e-12,
        max_iterations=4,
    )
    expected = rt.exponential_step(c, h, 0.37)

    assert result.iterations == 1
    assert result.fock_builds == 1
    assert result.hamiltonian_residual < 1.0e-12
    assert np.linalg.norm(result.coeff_next - expected) < 1.0e-12
    err = np.linalg.norm(result.coeff_next.conj().T @ s @ result.coeff_next - np.eye(nocc))
    assert err < 1.0e-11


def test_run_scem_matches_propagate_scem_for_constant_hamiltonian():
    rng = np.random.default_rng(24601)
    nao = 8
    nocc = 2

    a = rng.normal(size=(nao, nao)) + 1j * rng.normal(size=(nao, nao))
    s = a.conj().T @ a + np.eye(nao)
    s_eval, s_evec = scipy.linalg.eigh(s, check_finite=False)

    b = rng.normal(size=(nao, nao)) + 1j * rng.normal(size=(nao, nao))
    h = 0.5 * (b + b.conj().T)

    c0 = rng.normal(size=(nao, nocc)) + 1j * rng.normal(size=(nao, nocc))
    metric = c0.conj().T @ s @ c0
    eig, vec = scipy.linalg.eigh(metric, check_finite=False)
    c = c0 @ ((vec * eig**-0.5) @ vec.conj().T)

    rt = LengthGaugeCNRTTDDFT.__new__(LengthGaugeCNRTTDDFT)
    rt.s = s
    rt.s_sqrt = (s_evec * s_eval**0.5) @ s_evec.conj().T
    rt.s_invsqrt = (s_evec * s_eval**-0.5) @ s_evec.conj().T
    rt.hbar = 1.0
    rt.nocc = nocc
    rt.occ = np.full(nocc, 2.0)
    rt.occ_backend = rt.occ
    rt.idempotency_factor = 2.0
    rt.charge = -1.0
    rt.field = lambda _t: np.zeros(3)
    rt.reorthonormalize_every = None
    rt.reorthonormalize_tolerance = None
    rt.dipole_position = np.zeros((3, nao, nao), dtype=np.complex128)
    rt.hamiltonian_from_density = lambda _rho, _time: (h, None)

    propagated = list(
        rt.propagate_scem(
            c,
            dt=0.11,
            nsteps=4,
            midpoint_tolerance=1.0e-12,
            density_tolerance=1.0e-12,
        )
    )
    summary = rt.run_scem(
        c,
        dt=0.11,
        nsteps=4,
        midpoint_tolerance=1.0e-12,
        density_tolerance=1.0e-12,
    )

    assert summary.steps == 4
    assert summary.fock_builds == 5
    assert summary.midpoint_iterations == 4
    assert np.linalg.norm(summary.coeff_final - propagated[-1][0]) < 1.0e-12
    assert abs(summary.record_final.electron_number - propagated[-1][1].electron_number) < 1.0e-12


def test_scem_rejects_unconverged_midpoint():
    rng = np.random.default_rng(9753)
    nao = 8
    nocc = 2

    a = rng.normal(size=(nao, nao)) + 1j * rng.normal(size=(nao, nao))
    s = a.conj().T @ a + np.eye(nao)
    s_eval, s_evec = scipy.linalg.eigh(s, check_finite=False)

    h_guess = np.zeros((nao, nao), dtype=np.complex128)
    h_built = np.eye(nao, dtype=np.complex128)

    c0 = rng.normal(size=(nao, nocc)) + 1j * rng.normal(size=(nao, nocc))
    metric = c0.conj().T @ s @ c0
    eig, vec = scipy.linalg.eigh(metric, check_finite=False)
    c = c0 @ ((vec * eig**-0.5) @ vec.conj().T)

    rt = LengthGaugeCNRTTDDFT.__new__(LengthGaugeCNRTTDDFT)
    rt.s = s
    rt.s_sqrt = (s_evec * s_eval**0.5) @ s_evec.conj().T
    rt.s_invsqrt = (s_evec * s_eval**-0.5) @ s_evec.conj().T
    rt.hbar = 1.0
    rt.nocc = nocc
    rt.occ = np.full(nocc, 2.0)
    rt.hamiltonian_from_density = lambda _rho, _time: (h_built, None)

    with pytest.raises(MidpointConvergenceError):
        rt.scem_step(
            c,
            time=0.0,
            dt=0.5,
            h_guess=h_guess,
            midpoint_tolerance=1.0e-14,
            density_tolerance=None,
            max_iterations=1,
        )
