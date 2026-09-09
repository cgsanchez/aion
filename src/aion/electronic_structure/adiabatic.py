"""Backend-resident adiabatic pure-RKS Hamiltonian and energy contractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aion.backends import Workspace
from aion.errors import FormulationError


def hermitian_part(value: Any) -> Any:
    """Return the Hermitian part without changing array backend."""

    return 0.5 * (value + value.conj().T)


def expectation(density: Any, operator: Any, namespace: Any) -> Any:
    """Return ``Re Tr[P O]`` as a backend-resident scalar."""

    return namespace.real(namespace.einsum("ij,ji->", density, operator, optimize=True))


@dataclass(frozen=True, slots=True)
class DFTMatrixBuild:
    """One field-free pure-DFT matrix build in the bare AO frame."""

    density: Any
    hamiltonian: Any
    effective_potential: Any
    energy_hartree_hint: Any | None
    energy_exchange_correlation_hint: Any | None


@dataclass(frozen=True, slots=True)
class DFTInternalEnergy:
    """Canonical field-free DFT decomposition, all scalars backend resident."""

    energy_kinetic_canonical: Any
    energy_electron_nuclear: Any
    energy_hartree: Any
    energy_exchange_correlation: Any
    energy_nuclear_repulsion: Any
    energy_internal_total: Any


@dataclass(frozen=True, slots=True)
class AdiabaticPureRKS:
    """Validated PySCF/GPU4PySCF pure LDA/GGA functional bridge.

    Real Gaussian AOs make the real-space density depend only on the real
    symmetric part of a Hermitian AO density.  The complete complex density is
    retained in all operator contractions; only its real part is supplied to
    PySCF's real LDA/GGA quadrature and Coulomb builders.
    """

    workspace: Workspace

    @property
    def backend(self) -> Any:
        return self.workspace.backend

    @property
    def namespace(self) -> Any:
        return self.backend.namespace

    @property
    def nao(self) -> int:
        return int(self.workspace.require("operators.overlap").shape[0])

    def _density(self, value: Any) -> Any:
        self.backend.assert_resident(value, name="AO density")
        if value.shape != (self.nao, self.nao):
            raise FormulationError(
                f"AO density has shape {value.shape}; expected {(self.nao, self.nao)}"
            )
        return value

    def build(self, density: Any) -> DFTMatrixBuild:
        """Build ``F[P]=T+V_eN+J[P]+V_xc[P]`` in the field-free AO frame."""

        rho = self._density(density)
        xp = self.namespace
        rho_for_pyscf = xp.asarray(xp.real(hermitian_part(rho)), dtype=xp.float64)
        mean_field = self.workspace.electronic_model
        if mean_field is None:
            raise FormulationError("workspace has no reconstructed electronic model")
        try:
            raw_effective = mean_field.get_veff(mean_field.mol, rho_for_pyscf)
        except Exception as exc:
            raise FormulationError("pure-RKS effective-potential construction failed") from exc
        raw_hartree = getattr(raw_effective, "ecoul", None)
        raw_xc = getattr(raw_effective, "exc", None)
        effective = self.backend.asarray(raw_effective, dtype=np.complex128)
        self.backend.assert_resident(effective, name="effective potential")
        hcore = self.backend.asarray(
            self.workspace.require("operators.kinetic")
            + self.workspace.require("operators.nuclear_attraction"),
            dtype=np.complex128,
        )
        hamiltonian = hermitian_part(hcore + effective)
        return DFTMatrixBuild(
            density=rho,
            hamiltonian=hamiltonian,
            effective_potential=effective,
            energy_hartree_hint=(
                None if raw_hartree is None else self.backend.asarray(raw_hartree, dtype=np.float64)
            ),
            energy_exchange_correlation_hint=(
                None if raw_xc is None else self.backend.asarray(raw_xc, dtype=np.float64)
            ),
        )

    def energy(self, build: DFTMatrixBuild) -> DFTInternalEnergy:
        """Contract the qualified field-free DFT energy decomposition."""

        rho = self._density(build.density)
        xp = self.namespace
        kinetic = expectation(rho, self.workspace.require("operators.kinetic"), xp)
        electron_nuclear = expectation(
            rho, self.workspace.require("operators.nuclear_attraction"), xp
        )
        hartree = build.energy_hartree_hint
        xc = build.energy_exchange_correlation_hint
        if hartree is None:
            mean_field = self.workspace.electronic_model
            assert mean_field is not None
            rho_real = xp.asarray(xp.real(hermitian_part(rho)), dtype=xp.float64)
            coulomb = self.backend.asarray(
                mean_field.get_j(mean_field.mol, rho_real), dtype=np.complex128
            )
            hartree = 0.5 * expectation(rho, coulomb, xp)
        if xc is None:
            raise FormulationError("PySCF effective potential did not expose XC energy")
        mean_field = self.workspace.electronic_model
        if mean_field is None:
            raise FormulationError("workspace has no reconstructed electronic model")
        nuclear_repulsion = self.backend.asarray(
            float(mean_field.mol.energy_nuc()), dtype=np.float64
        )
        total = kinetic + electron_nuclear + hartree + xc + nuclear_repulsion
        return DFTInternalEnergy(
            energy_kinetic_canonical=kinetic,
            energy_electron_nuclear=electron_nuclear,
            energy_hartree=hartree,
            energy_exchange_correlation=xc,
            energy_nuclear_repulsion=nuclear_repulsion,
            energy_internal_total=total,
        )

    def energy_rate(self, build: DFTMatrixBuild, density_dot: Any) -> Any:
        """Analytic DFT functional derivative ``Re Tr[F[P] Pdot]``."""

        rho_dot = self._density(density_dot)
        return expectation(rho_dot, build.hamiltonian, self.namespace)
