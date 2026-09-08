from __future__ import annotations

import numpy as np
import pytest

from aion import LengthGaugeCNRTTDDFT, VelocityGaugeCNRTTDDFT


@pytest.fixture(scope="module")
def h2_reference():
    pytest.importorskip("pyscf")
    from pyscf import dft, gto

    mol = gto.M(
        atom="H 0 0 -0.37; H 0 0 0.37",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "lda,vwn"
    mf.grids.level = 0
    mf.conv_tol = 1.0e-12
    mf.kernel()
    assert mf.converged
    return mf


def test_pemmaraju_velocity_interaction_contains_linear_and_a2_terms(h2_reference):
    vector = np.array([0.013, -0.021, 0.034])
    rt = VelocityGaugeCNRTTDDFT.from_ground_state(
        h2_reference,
        field=lambda _t: np.zeros(3),
        vector_potential=lambda _t: vector,
    )

    expected_momentum = 1j * h2_reference.mol.intor("int1e_ipovlp", comp=3)
    expected = -rt.charge * np.einsum(
        "x,xij->ij", vector, expected_momentum
    ) + 0.5 * rt.charge**2 * np.dot(vector, vector) * rt.s

    assert np.linalg.norm(rt.canonical_momentum - expected_momentum) < 1.0e-13
    assert np.linalg.norm(rt.external_potential(0.7) - expected) < 1.0e-13
    for component in rt.canonical_momentum:
        assert np.linalg.norm(component - component.conj().T) < 1.0e-13


def test_from_field_integral_uses_paper_vector_potential_sign(h2_reference):
    electric = np.array([0.02, -0.01, 0.03])
    rt = VelocityGaugeCNRTTDDFT.from_field_integral(
        h2_reference,
        field=lambda _t: electric,
        field_integral=lambda t: t * electric,
    )

    assert np.allclose(rt.vector_potential_at(0.4), -0.4 * electric)


def test_zero_vector_potential_reduces_to_length_driver_at_zero_field(h2_reference):
    zero = lambda _t: np.zeros(3)
    length = LengthGaugeCNRTTDDFT.from_ground_state(h2_reference, zero)
    velocity = VelocityGaugeCNRTTDDFT.from_ground_state(
        h2_reference,
        field=zero,
        vector_potential=zero,
    )
    rho = length.initial_density()

    h_length, _ = length.hamiltonian_from_density(rho, 0.0)
    h_velocity, _ = velocity.hamiltonian_from_density(rho, 0.0)
    assert np.linalg.norm(h_velocity - h_length) < 1.0e-13

    coeff_length = list(
        length.propagate_scem(
            length.initial_coefficients(),
            dt=0.05,
            nsteps=1,
            midpoint_tolerance=1.0e-12,
            density_tolerance=None,
        )
    )[-1][0]
    coeff_velocity = list(
        velocity.propagate_scem(
            velocity.initial_coefficients(),
            dt=0.05,
            nsteps=1,
            midpoint_tolerance=1.0e-12,
            density_tolerance=None,
        )
    )[-1][0]
    assert np.linalg.norm(coeff_velocity - coeff_length) < 1.0e-12


def test_velocity_delta_kick_is_carried_by_vector_potential(h2_reference):
    impulse = np.array([1.0e-3, -2.0e-3, 3.0e-3])
    rt = VelocityGaugeCNRTTDDFT.from_delta_kick(h2_reference, impulse)
    coeff = rt.initial_coefficients()
    kicked = rt.apply_delta_kick(coeff, impulse)

    assert np.array_equal(rt.vector_potential_at(-1.0e-12), np.zeros(3))
    assert np.allclose(rt.vector_potential_at(0.0), -impulse)
    assert np.allclose(rt.vector_potential_at(1.0e-12), -impulse)
    assert np.array_equal(rt.field(-1.0e-12), np.zeros(3))
    assert np.array_equal(rt.field(0.0), np.zeros(3))
    assert np.array_equal(rt.field(1.0e-12), np.zeros(3))
    assert np.array_equal(kicked, coeff)
    assert kicked is not coeff
    with pytest.raises(ValueError, match="post-kick vector potential"):
        rt.apply_delta_kick(coeff, impulse, time=-1.0e-12)
    with pytest.raises(ValueError, match="post-kick vector potential"):
        rt.apply_delta_kick(coeff, 2.0 * impulse)


def test_velocity_delta_kick_supports_nonzero_kick_time(h2_reference):
    impulse = np.array([0.0, 0.0, 2.0e-3])
    kick_time = 1.25
    rt = VelocityGaugeCNRTTDDFT.from_delta_kick(
        h2_reference,
        impulse,
        kick_time=kick_time,
    )

    assert np.array_equal(
        rt.vector_potential_at(kick_time - 1.0e-12),
        np.zeros(3),
    )
    assert np.allclose(rt.vector_potential_at(kick_time), -impulse)
    assert np.allclose(rt.vector_potential_at(kick_time + 1.0e-12), -impulse)
    assert np.array_equal(
        rt.apply_delta_kick(rt.initial_coefficients(), impulse),
        rt.initial_coefficients(),
    )


def test_bare_length_and_velocity_kicks_use_their_native_representations(
    h2_reference,
):
    impulse = np.array([0.0, 0.0, 2.0e-3])
    zero = lambda _time: np.zeros(3)
    length = LengthGaugeCNRTTDDFT.from_ground_state(h2_reference, zero)
    velocity = VelocityGaugeCNRTTDDFT.from_delta_kick(h2_reference, impulse)
    coeff0 = length.initial_coefficients()

    coeff_length = length.apply_delta_kick(coeff0, impulse)
    coeff_velocity = velocity.apply_delta_kick(coeff0, impulse)

    # The length representation carries the impulse in its orbitals.  The
    # velocity representation carries the same impulse only in the step of a.
    assert np.linalg.norm(coeff_length - coeff0) > 1.0e-8
    assert np.array_equal(coeff_velocity, coeff0)
    identity = np.eye(length.nocc)
    assert (
        np.linalg.norm(coeff_length.conj().T @ length.s @ coeff_length - identity)
        < 1.0e-12
    )
    assert (
        np.linalg.norm(
            coeff_velocity.conj().T @ velocity.s @ coeff_velocity - identity
        )
        < 1.0e-12
    )


def test_velocity_record_contains_mechanical_current(h2_reference):
    impulse = np.array([0.0, 0.0, 2.0e-3])
    rt = VelocityGaugeCNRTTDDFT.from_delta_kick(h2_reference, impulse)
    coeff = rt.initial_coefficients()
    rho = rt.density_from_coefficients(coeff)
    record = rt.record(0, 0.0, coeff, rho, record_energy=False)

    expected_mechanical = rt.canonical_momentum - (
        rt.charge
        * rt.vector_potential_at(0.0)[:, None, None]
        * rt.s[None, :, :]
    )
    expected_current = rt.charge * np.einsum(
        "ij,xji->x", rho, expected_mechanical
    ).real

    assert np.allclose(record.vector_potential, -impulse)
    assert np.allclose(record.current, expected_current)
    assert np.allclose(
        record.current,
        h2_reference.mol.nelectron * impulse,
        atol=1.0e-13,
    )


def test_nonlocal_ionic_reference_is_rejected_before_propagation():
    class ECPMolecule:
        def has_ecp(self):
            return True

    class ECPMeanField:
        mol = ECPMolecule()

    with pytest.raises(NotImplementedError, match="Eq. \\(6\\)"):
        VelocityGaugeCNRTTDDFT(
            ECPMeanField(),
            field=lambda _t: np.zeros(3),
            vector_potential=lambda _t: np.zeros(3),
        )


def test_nonlocal_exact_exchange_reference_is_rejected(h2_reference):
    from pyscf import dft

    hybrid = dft.RKS(h2_reference.mol)
    hybrid.xc = "pbe0"
    with pytest.raises(NotImplementedError, match="exact exchange"):
        VelocityGaugeCNRTTDDFT(
            hybrid,
            field=lambda _t: np.zeros(3),
            vector_potential=lambda _t: np.zeros(3),
            real_density_for_veff=False,
        )
