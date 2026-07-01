from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.linalg

from aion import (
    P0E1Model,
    PyscfP0DftGpuModel,
    PyscfP0DftModel,
    PyscfP0Reference,
    UniformElectricGauge,
    UniformMagneticGauge,
    VariableMetricSCEM,
    pyscf_central_dipole_matrices,
)
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


def _h2_mol():
    from pyscf import gto

    return gto.M(
        atom="H 0 0 -0.37; H 0 0 0.37",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )


def _pbe_h2_references_for_gpu():
    from pyscf import dft

    mf_cpu = dft.RKS(_h2_mol()).density_fit()
    mf_cpu.xc = "pbe,pbe"
    mf_cpu.grids.level = 0
    mf_cpu.grids.prune = None
    mf_cpu.small_rho_cutoff = 0.0
    mf_cpu.conv_tol = 1.0e-11
    mf_cpu.kernel()
    assert mf_cpu.converged
    mf_gpu = mf_cpu.to_gpu()
    assert mf_gpu.converged
    return (
        PyscfP0Reference.from_mean_field(mf_cpu),
        PyscfP0Reference.from_mean_field(mf_gpu),
    )


def _asnumpy(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def _matrix_trace_product(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.einsum("ij,ji->", left, _asnumpy(right)).real)


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


def test_gpu4pyscf_p0_e1_scem_matches_cpu_h2_short_step():
    cupy = _require_cupy()
    pytest.importorskip("gpu4pyscf")
    from pyscf import dft, gto

    mol = gto.M(
        atom="H 0 0 -0.37; H 0 0 0.37",
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

    reference_cpu = PyscfP0Reference.from_mean_field(mf_cpu)
    reference_gpu = PyscfP0Reference.from_mean_field(mf_gpu)
    base_cpu = PyscfP0DftModel.from_reference(
        reference_cpu,
        real_density_for_veff=True,
    )
    base_gpu = PyscfP0DftGpuModel.from_reference(reference_gpu)
    central_cpu = pyscf_central_dipole_matrices(reference_cpu)
    central_gpu = cupy.asarray(central_cpu)
    model_cpu = P0E1Model(base_cpu, central_cpu)
    model_gpu = P0E1Model(base_gpu, central_gpu)

    field = np.array([0.003, -0.001, 0.002])
    electric = UniformElectricGauge(
        field=lambda _t: field,
        field_integral=lambda t: field * t,
        lambda_value=lambda _t: 0.4,
        lambda_derivative=lambda _t: 0.0,
    )
    rt_cpu = VariableMetricSCEM(
        reference_cpu.geometry(electric=electric),
        model_cpu,
        reference_cpu.occupations,
        backend="cpu",
    )
    rt_gpu = VariableMetricSCEM(
        reference_gpu.geometry(electric=electric),
        model_gpu,
        reference_gpu.occupations,
        backend="gpu",
    )

    result_cpu = rt_cpu.step(
        reference_cpu.initial_coefficients(),
        time=0.0,
        dt=0.02,
        midpoint_tolerance=1.0e-9,
        density_tolerance=1.0e-9,
        max_iterations=8,
        mixing=0.7,
    )
    result_gpu = rt_gpu.step(
        cupy.asarray(reference_gpu.initial_coefficients(), dtype=cupy.complex128),
        time=0.0,
        dt=0.02,
        midpoint_tolerance=1.0e-9,
        density_tolerance=1.0e-9,
        max_iterations=8,
        mixing=0.7,
    )
    rt_gpu.backend.synchronize()

    assert result_cpu.converged
    assert result_gpu.converged
    assert np.linalg.norm(cupy.asnumpy(result_gpu.rho_next) - result_cpu.rho_next) < 1.0e-7
    assert rt_gpu.orthonormality_error(result_gpu.coeff_next, 0.02) < 1.0e-8


def test_gpu4pyscf_pbe_veff_is_bare_hxc_energy_derivative():
    cupy = _require_cupy()
    pytest.importorskip("gpu4pyscf")
    reference_cpu, reference_gpu = _pbe_h2_references_for_gpu()
    model_cpu = PyscfP0DftModel.from_reference(
        reference_cpu,
        real_density_for_veff=True,
    )
    model_gpu = PyscfP0DftGpuModel.from_reference(reference_gpu)
    geometry_cpu = reference_cpu.geometry()
    geometry_gpu = reference_gpu.geometry()
    density = reference_cpu.initial_density()
    veff_cpu = model_cpu.veff0(density, 0.0, geometry_cpu)
    veff_gpu = cupy.asnumpy(model_gpu.veff0(density, 0.0, geometry_gpu))

    assert np.linalg.norm(veff_gpu - veff_cpu) < 1.0e-8

    def hxc_energy(dm: np.ndarray) -> float:
        return model_gpu.energy(dm, 0.0, geometry_gpu) - _matrix_trace_product(
            dm,
            reference_cpu.hcore0,
        )

    for direction in (
        _real_symmetric_direction(reference_cpu.nao, seed=3401),
        _complex_hermitian_direction(reference_cpu.nao, seed=3402),
    ):
        _assert_matrix_is_functional_derivative(
            functional=hxc_energy,
            density=density,
            potential=veff_gpu,
            direction=direction,
            tolerance=1.0e-6,
        )


def test_gpu4pyscf_p0_pbe_veff_is_dressed_hxc_energy_derivative():
    cupy = _require_cupy()
    pytest.importorskip("gpu4pyscf")
    reference_cpu, reference_gpu = _pbe_h2_references_for_gpu()
    model_cpu = PyscfP0DftModel.from_reference(
        reference_cpu,
        real_density_for_veff=True,
    )
    model_gpu = PyscfP0DftGpuModel.from_reference(reference_gpu)
    field = np.array([0.011, -0.017, 0.023])
    electric = UniformElectricGauge(
        field=lambda _t: field,
        field_integral=lambda t: field * t,
        lambda_value=lambda _t: 0.65,
        lambda_derivative=lambda _t: 0.0,
    )
    magnetic = UniformMagneticGauge(
        np.array([0.007, -0.011, 0.019]),
        gauge="symmetric",
        origin=np.array([0.13, -0.07, 0.05]),
    )
    geometry_cpu = reference_cpu.geometry(electric=electric, magnetic=magnetic)
    geometry_gpu = reference_gpu.geometry(electric=electric, magnetic=magnetic)
    time = 0.43
    density0 = reference_cpu.initial_density()
    density = geometry_cpu.theta(time) * density0
    hcore_p0 = geometry_cpu.dress_matrix(reference_cpu.hcore0, time)
    veff_cpu = model_cpu.hamiltonian(density, time, geometry_cpu) - hcore_p0
    veff_gpu = (
        cupy.asnumpy(model_gpu.hamiltonian(density, time, geometry_gpu))
        - hcore_p0
    )

    assert np.linalg.norm(veff_gpu - veff_cpu) < 1.0e-8

    def hxc_energy(rho: np.ndarray) -> float:
        return model_gpu.energy(rho, time, geometry_gpu) - _matrix_trace_product(
            rho,
            hcore_p0,
        )

    for direction in (
        _real_symmetric_direction(reference_cpu.nao, seed=4401),
        _complex_hermitian_direction(reference_cpu.nao, seed=4402),
    ):
        _assert_matrix_is_functional_derivative(
            functional=hxc_energy,
            density=density,
            potential=veff_gpu,
            direction=direction,
            tolerance=1.0e-6,
        )
