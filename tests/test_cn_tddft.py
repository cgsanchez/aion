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
