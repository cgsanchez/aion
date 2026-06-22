from __future__ import annotations

import numpy as np
import scipy.linalg

from aion.cn_tddft import LengthGaugeCNRTTDDFT


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
