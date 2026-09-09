"""Typed backend-resident contracts shared by all formulations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np

from aion.backends import ArrayBackend, Workspace
from aion.config import FormulationKind, GaugeRepresentation
from aion.electronic_structure import DFTMatrixBuild
from aion.errors import FormulationError


class SourceSampling(StrEnum):
    ENDPOINT = "endpoint"
    MIDPOINT = "midpoint"


@dataclass(frozen=True, slots=True)
class AODensity:
    """A lower-index AO density using ``P=C f C^dagger``."""

    matrix: Any
    backend: ArrayBackend

    def __post_init__(self) -> None:
        self.backend.assert_resident(self.matrix, name="AO density")
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            raise FormulationError("AO density must be a square matrix")
        if self.matrix.dtype != np.dtype(np.complex128):
            raise FormulationError("AO density must use complex128 precision")
        xp = self.backend.namespace
        if not bool(self.backend.scalar_to_float(xp.all(xp.isfinite(self.matrix)))):
            raise FormulationError("AO density contains non-finite values")
        residual = self.hermiticity_residual()
        residual_value = self.backend.scalar_to_float(residual)
        if residual_value > 1.0e-11:
            raise FormulationError(f"AO density is not Hermitian: residual={residual_value:.3e}")

    @classmethod
    def from_matrix(cls, value: object, backend: ArrayBackend) -> AODensity:
        return cls(backend.asarray(value, dtype=np.complex128), backend)

    @classmethod
    def from_coefficients(
        cls, coefficients: Any, occupations: Any, backend: ArrayBackend
    ) -> AODensity:
        backend.assert_resident(coefficients, name="AO coefficients")
        backend.assert_resident(occupations, name="occupations")
        if coefficients.ndim != 2 or occupations.shape != (coefficients.shape[1],):
            raise FormulationError("coefficient and occupation shapes are inconsistent")
        xp = backend.namespace
        matrix = xp.einsum(
            "mi,i,ni->mn",
            coefficients,
            occupations,
            coefficients.conj(),
            optimize=True,
        )
        return cls(xp.asarray(matrix, dtype=xp.complex128), backend)

    def hermiticity_residual(self) -> Any:
        xp = self.backend.namespace
        scale = xp.maximum(xp.asarray(1.0), xp.linalg.norm(self.matrix))
        return xp.linalg.norm(self.matrix - self.matrix.conj().T) / scale


@dataclass(frozen=True, slots=True)
class FormulationSourceSample:
    """One gauge-resolved source sample resident on the simulation backend."""

    time_au: float
    gauge: GaugeRepresentation
    electric_field: Any
    electric_field_dot: Any
    vector_potential_reduced: Any
    vector_potential_reduced_dot: Any
    node_scalar_potential: Any
    pair_link: Any
    pair_link_dot: Any
    pair_electromotive_potential: Any

    @classmethod
    def from_workspace(
        cls,
        workspace: Workspace,
        *,
        gauge: GaugeRepresentation,
        location: SourceSampling,
        index: int,
        prefix: str = "source",
    ) -> FormulationSourceSample:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise FormulationError("source sample index must be a nonnegative integer")
        loc = location.value
        times = workspace.require(f"{prefix}.{loc}.times_au")
        if index >= times.shape[0]:
            raise FormulationError(
                f"source sample index {index} is outside [0, {times.shape[0] - 1}]"
            )
        gauge_prefix = f"{prefix}.{gauge.value}_{loc}"
        physical_vector = workspace.require(f"{prefix}.{loc}.vector_potential_reduced")[index]
        physical_vector_dot = workspace.require(f"{prefix}.{loc}.vector_potential_reduced_dot")[
            index
        ]
        xp = workspace.backend.namespace
        if gauge is GaugeRepresentation.LENGTH:
            vector = xp.zeros((3,), dtype=xp.float64)
            vector_dot = xp.zeros((3,), dtype=xp.float64)
        else:
            vector = physical_vector
            vector_dot = physical_vector_dot
        return cls(
            time_au=workspace.backend.scalar_to_float(times[index]),
            gauge=gauge,
            electric_field=workspace.require(f"{prefix}.{loc}.electric_field")[index],
            electric_field_dot=workspace.require(f"{prefix}.{loc}.electric_field_dot")[index],
            vector_potential_reduced=vector,
            vector_potential_reduced_dot=vector_dot,
            node_scalar_potential=workspace.require(f"{gauge_prefix}.node_scalar_potential")[index],
            pair_link=workspace.require(f"{gauge_prefix}.pair_link")[index],
            pair_link_dot=workspace.require(f"{gauge_prefix}.pair_link_dot")[index],
            pair_electromotive_potential=workspace.require(
                f"{gauge_prefix}.pair_electromotive_potential"
            )[index],
        )

    def assert_resident(self, backend: ArrayBackend) -> None:
        if not math.isfinite(self.time_au):
            raise FormulationError("source sample time is not finite")
        for name in (
            "electric_field",
            "electric_field_dot",
            "vector_potential_reduced",
            "vector_potential_reduced_dot",
            "node_scalar_potential",
            "pair_link",
            "pair_link_dot",
            "pair_electromotive_potential",
        ):
            backend.assert_resident(getattr(self, name), name=f"source.{name}")


@dataclass(frozen=True, slots=True)
class EOMTriple:
    """Normative coefficient equation ``i hbar(S Cdot+omega C)=H C``."""

    metric: Any
    hamiltonian_eom: Any
    connection: Any


@dataclass(frozen=True, slots=True)
class InstantaneousEvaluation:
    """One reusable matrix build plus analytic state derivatives."""

    triple: EOMTriple
    density: AODensity
    density_dot: Any
    field_free_density: Any
    field_free_density_dot: Any
    field_free_dft: DFTMatrixBuild
    hamiltonian_dynamic: Any
    source_hamiltonian: Any
    metric_dot: Any
    d_site_metric: Any
    theta: Any | None = None
    theta_dot: Any | None = None
    e1_potential: Any | None = None
    dressed_central_dipoles: Any | None = None
    dressed_central_dipole_dots: Any | None = None
    dressed_central_dipole_vector_derivatives: Any | None = None


@dataclass(frozen=True, slots=True)
class CurrentLedger:
    """Definition-qualified dipoles, primary/source currents, and diagnostics."""

    electronic_dipole: Any
    fixed_nuclear_dipole: Any
    total_dipole: Any
    dipole_derivative_analytic: Any
    primary_current: Any
    source_current: Any
    source_work_rate: Any
    mechanical_paramagnetic_current: Any
    mechanical_diamagnetic_current: Any
    mechanical_total_current: Any
    ambient_projected_mechanical_current: Any | None = None
    site_charges: Any | None = None
    site_charge_derivatives: Any | None = None
    pair_currents_continuity: Any | None = None
    pair_currents_p0_source: Any | None = None
    p0_graph_current: Any | None = None
    e1_intrinsic_polarization_current: Any | None = None
    e1_phase_response_current: Any | None = None
    e1_residual_source_current: Any | None = None
    continuity_residual: Any | None = None
    source_current_dipole_derivative_residual: Any | None = None


@dataclass(frozen=True, slots=True)
class EnergyLedger:
    """Complete qualified instantaneous energy and power record."""

    energy_kinetic_canonical: Any
    energy_kinetic_vector_potential_linear: Any
    energy_kinetic_diamagnetic: Any
    energy_kinetic_mechanical: Any
    energy_electron_nuclear: Any
    energy_hartree: Any
    energy_exchange_correlation: Any
    energy_nuclear_repulsion: Any
    energy_electromagnetic_electronic_scalar: Any
    energy_electromagnetic_fixed_nuclear_scalar: Any
    energy_electromagnetic_scalar_total: Any
    energy_e1_coupling: Any
    energy_matter_total: Any
    energy_generator_total: Any
    energy_absorbed: Any | None
    source_work_accumulated: Any | None
    energy_matter_rate_analytic: Any
    energy_generator_rate_analytic: Any | None
    source_work_rate: Any
    energy_ward_residual: Any


@runtime_checkable
class Formulation(Protocol):
    @property
    def kind(self) -> FormulationKind: ...

    @property
    def gauge(self) -> GaugeRepresentation: ...

    @property
    def context(self) -> Any: ...

    def evaluate(
        self, density: AODensity, source: FormulationSourceSample
    ) -> InstantaneousEvaluation: ...

    def currents(
        self, evaluation: InstantaneousEvaluation, source: FormulationSourceSample
    ) -> CurrentLedger: ...

    def energy(
        self,
        evaluation: InstantaneousEvaluation,
        source: FormulationSourceSample,
        *,
        initial_matter_energy: Any | None = None,
        accumulated_source_work: Any | None = None,
    ) -> EnergyLedger: ...

    @property
    def observable_dependencies(self) -> frozenset[str]: ...
