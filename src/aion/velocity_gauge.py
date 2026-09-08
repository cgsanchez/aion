"""Conventional static-AO velocity-gauge real-time TDDFT.

This module implements the localized-basis velocity-gauge Hamiltonian of
Pemmaraju et al., Comput. Phys. Commun. 226, 30-38 (2018), specialized to
finite, all-electron PySCF references with local ionic potentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Callable

import numpy as np

from .cn_tddft import CNPropagationRecord, LengthGaugeCNRTTDDFT


VectorPotential = Callable[[float], np.ndarray]


def _as_vector3(value, *, name: str) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must return shape (3,)")
    return array


@dataclass
class VelocityGaugePropagationRecord(CNPropagationRecord):
    """Diagnostics for one conventional velocity-gauge propagation step."""

    vector_potential: np.ndarray = dataclass_field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    current: np.ndarray = dataclass_field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )


class VelocityGaugeCNRTTDDFT(LengthGaugeCNRTTDDFT):
    r"""Fixed-basis velocity-gauge RT-TDDFT for a uniform electric field.

    The input ``vector_potential`` is the reduced vector potential
    :math:`\mathbf a=\mathbf A/c` in atomic units.  For an electric field it is

    .. math::

        \mathbf a(t) = -\int^t \mathbf E(t')\,dt'.

    For a particle of charge ``q`` and unit mass, the field-dependent matrix is

    .. math::

        V_\mathrm{VG}(t) = -q\,\mathbf a(t)\cdot\mathbf p
          + \tfrac12 q^2 |\mathbf a(t)|^2 S,

    with :math:`\mathbf p=-i\hbar\nabla`.  This is Eq. (13) of Pemmaraju et
    al. expressed for a nonorthogonal Gaussian AO basis.  The Hartree and
    semi-local XC terms are built exactly as in the existing length-gauge
    driver, and all inherited CN, EP-PC1, and SCEM propagators remain
    available.

    PySCF ECP/pseudopotential references are deliberately rejected.  Their
    nonlocal ionic operator requires the additional gauge transformation in
    Eq. (6) of the paper; omitting it would not implement the stated method.
    """

    def __init__(
        self,
        mf,
        field: Callable[[float], np.ndarray],
        vector_potential: VectorPotential,
        origin: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        self._validate_local_ionic_reference(mf)
        self._validate_semilocal_reference(mf)
        self._vector_potential = vector_potential
        super().__init__(mf, field, origin, **kwargs)

        # libcint's int1e_ipovlp is (nabla |) = (| -nabla).  Consequently
        # <mu|p|nu> = i hbar (nabla mu|nu).
        gradient_bra = np.asarray(
            self.mol.intor("int1e_ipovlp", comp=3),
            dtype=np.complex128,
        )
        expected_shape = (3, self.s.shape[0], self.s.shape[1])
        if gradient_bra.shape != expected_shape:
            raise ValueError(
                "int1e_ipovlp returned shape "
                f"{gradient_bra.shape}, expected {expected_shape}"
            )
        momentum_host = 1j * self.hbar * gradient_bra
        adjoint = np.swapaxes(momentum_host.conj(), 1, 2)
        hermiticity_error = np.linalg.norm(momentum_host - adjoint)
        hermiticity_scale = max(1.0, np.linalg.norm(momentum_host))
        if hermiticity_error > 1.0e-10 * hermiticity_scale:
            raise ValueError(
                "canonical momentum integrals are not Hermitian: "
                f"relative error={hermiticity_error / hermiticity_scale:.3e}"
            )
        momentum_host = 0.5 * (momentum_host + adjoint)
        self.canonical_momentum = self.backend.asarray(
            momentum_host,
            dtype=np.complex128,
        )

    @classmethod
    def from_ground_state(
        cls,
        mf,
        field: Callable[[float], np.ndarray],
        vector_potential: VectorPotential,
        origin: np.ndarray | None = None,
        **kwargs,
    ) -> "VelocityGaugeCNRTTDDFT":
        """Construct a velocity-gauge driver from a converged reference."""

        if not getattr(mf, "converged", False):
            raise ValueError("mean-field object is not converged")
        return cls(mf, field, vector_potential, origin, **kwargs)

    @classmethod
    def from_field_integral(
        cls,
        mf,
        field: Callable[[float], np.ndarray],
        field_integral: Callable[[float], np.ndarray],
        origin: np.ndarray | None = None,
        **kwargs,
    ) -> "VelocityGaugeCNRTTDDFT":
        r"""Construct from ``E(t)`` and its analytic impulse ``int E dt``.

        The reduced vector potential is formed with the paper's convention
        :math:`\mathbf a(t)=-\int^t\mathbf E(t')dt'`.
        """

        def vector_potential(time: float) -> np.ndarray:
            return -_as_vector3(field_integral(time), name="field_integral(t)")

        return cls.from_ground_state(
            mf,
            field,
            vector_potential,
            origin,
            **kwargs,
        )

    @classmethod
    def from_delta_kick(
        cls,
        mf,
        electric_field_impulse: np.ndarray,
        origin: np.ndarray | None = None,
        *,
        kick_time: float = 0.0,
        **kwargs,
    ) -> "VelocityGaugeCNRTTDDFT":
        r"""Construct a velocity-gauge source with an impulsive electric field.

        If ``K = int E dt``, the reduced vector potential is the right-continuous
        step

        .. math::

            \mathbf a(t) = -\mathbf K\,\Theta(t-t_k).

        Thus its distributional derivative gives
        :math:`\mathbf E(t)=\mathbf K\delta(t-t_k)`.  The callable ``field``
        returns only the regular (zero) part of that electric field; the
        impulse is represented exactly by the jump in ``vector_potential``.

        The velocity-gauge occupied coefficients are continuous across the
        kick.  Start post-kick propagation at ``kick_time`` with the unchanged
        field-free coefficients; do not also apply the length-gauge orbital
        exponential.
        """

        impulse = np.asarray(electric_field_impulse, dtype=float)
        if impulse.shape != (3,):
            raise ValueError("electric_field_impulse must have shape (3,)")
        if not np.all(np.isfinite(impulse)):
            raise ValueError("electric_field_impulse must contain finite values")
        kick_time = float(kick_time)
        if not np.isfinite(kick_time):
            raise ValueError("kick_time must be finite")
        impulse = impulse.copy()
        zero = np.zeros(3, dtype=float)

        def vector_potential(time: float) -> np.ndarray:
            return zero if time < kick_time else -impulse

        driver = cls.from_ground_state(
            mf,
            field=lambda _time: zero,
            vector_potential=vector_potential,
            origin=origin,
            **kwargs,
        )
        driver._delta_kick_impulse = impulse
        driver._delta_kick_time = kick_time
        return driver

    def vector_potential_at(self, time: float) -> np.ndarray:
        """Return the reduced vector potential ``A(t)/c`` in atomic units."""

        return _as_vector3(self._vector_potential(time), name="vector_potential(t)")

    def external_potential(self, t: float) -> np.ndarray:
        """Return the linear and quadratic velocity-gauge kinetic terms."""

        backend = self._array_backend()
        vector_host = self.vector_potential_at(t)
        vector = backend.asarray(vector_host, dtype=float)
        linear = -self.charge * backend.einsum(
            "x,xij->ij",
            vector,
            self.canonical_momentum,
        )
        quadratic = (
            0.5
            * self.charge**2
            * float(np.dot(vector_host, vector_host))
            * self.s
        )
        return self.hermitian_part(linear + quadratic)

    def mechanical_momentum(self, time: float) -> np.ndarray:
        r"""Return ``p - q a(t)`` as three AO operator matrices."""

        backend = self._array_backend()
        vector = backend.asarray(self.vector_potential_at(time), dtype=float)
        return self.canonical_momentum - (
            self.charge * vector[:, None, None] * self.s[None, :, :]
        )

    def electronic_current(self, rho: np.ndarray, time: float) -> np.ndarray:
        r"""Return the total electronic current ``q Tr[rho (p-q a)]``.

        This is the molecular (volume-integrated) current, with unit electron
        mass.  It equals the time derivative of the electronic dipole in the
        complete-basis, local-potential limit.  A periodic current density would
        additionally divide by the unit-cell volume.
        """

        backend = self._array_backend()
        rho = backend.asarray(rho, dtype=np.complex128)
        if rho.shape != self.s.shape:
            raise ValueError(f"rho must have shape {self.s.shape}")
        current = self.charge * backend.einsum(
            "ij,xji->x",
            rho,
            self.mechanical_momentum(time),
        ).real
        return backend.asnumpy(current)

    def apply_delta_kick(
        self,
        coeff: np.ndarray,
        electric_field_impulse: np.ndarray,
        *,
        time: float | None = None,
    ) -> np.ndarray:
        """Return post-kick coefficients when the source already carries the kick.

        Unlike a length-gauge kick, no orbital exponential is applied.  This
        method verifies that the configured post-kick vector potential is
        ``-integral E dt`` and then returns an independent coefficient copy.
        """

        impulse = np.asarray(electric_field_impulse, dtype=float)
        if impulse.shape != (3,):
            raise ValueError("electric_field_impulse must have shape (3,)")
        if time is None:
            time = getattr(self, "_delta_kick_time", 0.0)
        actual = self.vector_potential_at(time)
        if not np.allclose(actual, -impulse, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError(
                "the configured post-kick vector potential does not equal "
                "-electric_field_impulse at the requested time"
            )
        backend = self._array_backend()
        coeff = backend.asarray(coeff, dtype=np.complex128)
        if coeff.shape != (self.s.shape[0], self.nocc):
            raise ValueError(f"coeff must have shape {(self.s.shape[0], self.nocc)}")
        return backend.copy(coeff)

    def record(
        self,
        step: int,
        time: float,
        coeff: np.ndarray,
        rho: np.ndarray,
        *,
        record_energy: bool,
        midpoint_iterations: int = 0,
        midpoint_converged: bool = True,
        hamiltonian_residual: float | None = None,
        density_residual: float | None = None,
        fock_builds: int = 0,
    ) -> VelocityGaugePropagationRecord:
        """Return the common diagnostics plus vector potential and current."""

        base = super().record(
            step,
            time,
            coeff,
            rho,
            record_energy=record_energy,
            midpoint_iterations=midpoint_iterations,
            midpoint_converged=midpoint_converged,
            hamiltonian_residual=hamiltonian_residual,
            density_residual=density_residual,
            fock_builds=fock_builds,
        )
        return VelocityGaugePropagationRecord(
            **vars(base),
            vector_potential=self.vector_potential_at(time),
            current=self.electronic_current(rho, time),
        )

    @staticmethod
    def _validate_local_ionic_reference(mf) -> None:
        mol = getattr(mf, "mol", None)
        if mol is None:
            raise TypeError("mean-field object must expose a molecular 'mol' object")

        has_ecp = getattr(mol, "has_ecp", None)
        if callable(has_ecp):
            has_ecp = bool(has_ecp())
        else:
            has_ecp = bool(has_ecp) or bool(getattr(mol, "_ecp", {}))
        has_pseudo = bool(getattr(mol, "_pseudo", {}))
        if has_ecp or has_pseudo:
            raise NotImplementedError(
                "velocity-gauge ECP/pseudopotential references require the "
                "phase-transformed nonlocal ionic operator of Pemmaraju et al. "
                "Eq. (6), which is not implemented"
            )

    @staticmethod
    def _validate_semilocal_reference(mf) -> None:
        xc = getattr(mf, "xc", None)
        numint = getattr(mf, "_numint", None)
        if xc is None or numint is None:
            raise NotImplementedError(
                "the Pemmaraju velocity-gauge path currently requires a "
                "pure LDA/GGA PySCF RKS reference"
            )
        xc_type = str(numint._xc_type(xc)).upper()
        if xc_type not in {"LDA", "GGA"}:
            raise NotImplementedError(
                "the Pemmaraju velocity-gauge path currently supports pure "
                f"LDA/GGA functionals, got {xc_type}"
            )
        _, alpha, hybrid = numint.rsh_and_hybrid_coeff(xc, spin=mf.mol.spin)
        if max(abs(float(alpha)), abs(float(hybrid))) > 1.0e-14:
            raise NotImplementedError(
                "the Pemmaraju velocity-gauge path does not yet support "
                "nonlocal exact exchange"
            )
