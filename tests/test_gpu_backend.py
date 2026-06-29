from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.linalg

from aion.backends import make_backend
from aion.cn_tddft import LengthGaugeCNRTTDDFT


pytestmark = pytest.mark.skipif(
    os.environ.get("AION_GPU_LAUNCHER") != "1",
    reason="GPU tests are enabled only through tools/gpu-python",
)


def _require_cupy():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device visible")
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device is not usable: {exc}")
    return cupy


def test_gpu_scem_constant_hamiltonian_matches_cpu_reference():
    cupy = _require_cupy()
    backend = make_backend("gpu")
    rng = np.random.default_rng(8642)
    nao = 9
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
    rt.backend = backend
    rt.s = backend.asarray(s, dtype=np.complex128)
    rt.s_sqrt = backend.asarray(
        (s_evec * s_eval**0.5) @ s_evec.conj().T,
        dtype=np.complex128,
    )
    rt.s_invsqrt = backend.asarray(
        (s_evec * s_eval**-0.5) @ s_evec.conj().T,
        dtype=np.complex128,
    )
    rt.hbar = 1.0
    rt.nocc = nocc
    rt.occ = np.full(nocc, 2.0)
    h_gpu = backend.asarray(h, dtype=np.complex128)
    rt.hamiltonian_from_density = lambda _rho, _time: (h_gpu, None)

    result = rt.scem_step(
        backend.asarray(c, dtype=np.complex128),
        time=0.0,
        dt=0.23,
        h_guess=h_gpu,
        midpoint_tolerance=1.0e-12,
        density_tolerance=1.0e-12,
        max_iterations=4,
    )
    backend.synchronize()

    h_tilde = (s_evec * s_eval**-0.5) @ s_evec.conj().T @ h @ (
        s_evec * s_eval**-0.5
    ) @ s_evec.conj().T
    eps, u = scipy.linalg.eigh(0.5 * (h_tilde + h_tilde.conj().T), check_finite=False)
    c_orth = ((s_evec * s_eval**0.5) @ s_evec.conj().T) @ c
    expected_orth = u @ (
        np.exp(-1j * 0.23 * eps)[:, None] * (u.conj().T @ c_orth)
    )
    expected = ((s_evec * s_eval**-0.5) @ s_evec.conj().T) @ expected_orth

    assert result.iterations == 1
    assert result.fock_builds == 1
    assert np.linalg.norm(cupy.asnumpy(result.coeff_next) - expected) < 1.0e-10


def test_gpu4pyscf_scem_smoke_step_pbe_h2():
    _require_cupy()
    pytest.importorskip("gpu4pyscf")
    from pyscf import dft, gto

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = dft.RKS(mol, xc="pbe").density_fit()
    mf.grids.level = 0
    mf = mf.to_gpu()
    mf.kernel()
    assert mf.converged

    rt = LengthGaugeCNRTTDDFT.from_ground_state(
        mf,
        lambda _t: np.zeros(3),
        backend="gpu",
    )
    coeff0 = rt.initial_coefficients()
    records = list(
        rt.propagate_scem(
            coeff0,
            dt=0.05,
            nsteps=1,
            midpoint_tolerance=1.0e-9,
            density_tolerance=1.0e-9,
            max_iterations=8,
            record_energy=False,
        )
    )
    rt.backend.synchronize()

    assert len(records) == 2
    _, rec = records[-1]
    assert rec.midpoint_converged
    assert rec.fock_builds >= 1
    assert abs(rec.electron_number - mol.nelectron) < 1.0e-8
    assert rec.orthonormality_error < 1.0e-8


def test_gpu4pyscf_scem_matches_cpu_observables_for_short_h2_kick():
    _require_cupy()
    pytest.importorskip("gpu4pyscf")
    from pyscf import dft, gto

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf_cpu = dft.RKS(mol, xc="pbe").density_fit()
    mf_cpu.grids.level = 0
    mf_cpu.kernel()
    assert mf_cpu.converged

    mf_gpu = mf_cpu.to_gpu()
    assert mf_gpu.converged

    field = lambda _t: np.zeros(3)
    rt_cpu = LengthGaugeCNRTTDDFT.from_ground_state(mf_cpu, field, backend="cpu")
    rt_gpu = LengthGaugeCNRTTDDFT.from_ground_state(mf_gpu, field, backend="gpu")

    impulse = np.array([0.0, 0.0, 1.0e-3])
    coeff_cpu = rt_cpu.apply_delta_kick(rt_cpu.initial_coefficients(), impulse)
    coeff_gpu = rt_gpu.apply_delta_kick(rt_gpu.initial_coefficients(), impulse)

    records_cpu = list(
        rt_cpu.propagate_scem(
            coeff_cpu,
            dt=0.05,
            nsteps=3,
            midpoint_tolerance=1.0e-9,
            density_tolerance=1.0e-9,
            max_iterations=8,
            record_energy=False,
        )
    )
    records_gpu = list(
        rt_gpu.propagate_scem(
            coeff_gpu,
            dt=0.05,
            nsteps=3,
            midpoint_tolerance=1.0e-9,
            density_tolerance=1.0e-9,
            max_iterations=8,
            record_energy=False,
        )
    )
    rt_gpu.backend.synchronize()

    rec_cpu = records_cpu[-1][1]
    rec_gpu = records_gpu[-1][1]
    assert np.linalg.norm(rec_gpu.dipole - rec_cpu.dipole) < 1.0e-7
    assert abs(rec_gpu.electron_number - rec_cpu.electron_number) < 1.0e-8
    assert abs(rec_gpu.idempotency_error - rec_cpu.idempotency_error) < 1.0e-7
    assert rec_gpu.orthonormality_error < 1.0e-8


def test_gpu4pyscf_run_scem_matches_propagate_scem_final_record():
    _require_cupy()
    pytest.importorskip("gpu4pyscf")
    from pyscf import dft, gto

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = dft.RKS(mol, xc="pbe").density_fit()
    mf.grids.level = 0
    mf.kernel()
    assert mf.converged

    rt = LengthGaugeCNRTTDDFT.from_ground_state(
        mf.to_gpu(),
        lambda _t: np.zeros(3),
        backend="gpu",
    )
    coeff0 = rt.apply_delta_kick(
        rt.initial_coefficients(),
        np.array([0.0, 0.0, 1.0e-3]),
    )
    propagated = list(
        rt.propagate_scem(
            coeff0,
            dt=0.05,
            nsteps=2,
            midpoint_tolerance=1.0e-9,
            density_tolerance=1.0e-9,
            max_iterations=8,
            record_energy=False,
        )
    )
    summary = rt.run_scem(
        coeff0,
        dt=0.05,
        nsteps=2,
        midpoint_tolerance=1.0e-9,
        density_tolerance=1.0e-9,
        max_iterations=8,
        record_energy=False,
    )
    rt.backend.synchronize()

    final_rec = propagated[-1][1]
    assert summary.steps == 2
    assert summary.fock_builds >= 3
    assert np.linalg.norm(summary.record_final.dipole - final_rec.dipole) < 1.0e-8
    assert abs(summary.record_final.electron_number - final_rec.electron_number) < 1.0e-8
    assert summary.record_final.orthonormality_error < 1.0e-8
