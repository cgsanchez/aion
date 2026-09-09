"""Backend-resident state and immutable SCEM result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from aion.backends import ArrayBackend
from aion.errors import StateIntegrityError
from aion.formulations import AODensity, FormulationSourceSample, InstantaneousEvaluation


@runtime_checkable
class ElectronicState(Protocol):
    """State seam retained for a later density-matrix propagator."""

    @property
    def step_index(self) -> int: ...

    @property
    def backend(self) -> ArrayBackend: ...

    def density(self) -> AODensity: ...


@dataclass(frozen=True, slots=True)
class OrbitalState:
    """Occupied AO coefficients and their complete spin occupations."""

    coefficients: Any
    occupations: Any
    backend: ArrayBackend
    step_index: int = 0

    def __post_init__(self) -> None:
        self.backend.assert_resident(self.coefficients, name="orbital coefficients")
        self.backend.assert_resident(self.occupations, name="orbital occupations")
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int):
            raise StateIntegrityError("state step_index must be an integer")
        if self.step_index < 0:
            raise StateIntegrityError("state step_index cannot be negative")
        if self.coefficients.ndim != 2 or self.occupations.shape != (self.coefficients.shape[1],):
            raise StateIntegrityError("orbital coefficient and occupation shapes disagree")
        if self.coefficients.shape[1] == 0:
            raise StateIntegrityError("orbital state must contain at least one occupied orbital")
        if self.coefficients.dtype != np.dtype(np.complex128):
            raise StateIntegrityError("orbital coefficients must use complex128 precision")
        if self.occupations.dtype != np.dtype(np.float64):
            raise StateIntegrityError("orbital occupations must use float64 precision")
        xp = self.backend.namespace
        finite = xp.all(xp.isfinite(self.coefficients)) & xp.all(xp.isfinite(self.occupations))
        if not bool(self.backend.scalar_to_float(finite)):
            raise StateIntegrityError("orbital state contains non-finite values")
        if not bool(self.backend.scalar_to_float(xp.all(self.occupations > 0.0))):
            raise StateIntegrityError("orbital state stores occupied orbitals only")

    @property
    def orbital_count(self) -> int:
        return int(self.coefficients.shape[1])

    @property
    def electron_count(self) -> float:
        return self.backend.scalar_to_float(self.backend.namespace.sum(self.occupations))

    def density(self) -> AODensity:
        return AODensity.from_coefficients(self.coefficients, self.occupations, self.backend)


@dataclass(frozen=True, slots=True)
class LinkDiagnostics:
    """Cross-metric diagnostics for one raw/corrected transport link."""

    raw_metric_residual: float
    corrected_metric_residual: float
    correction_norm: float
    correction_applied: bool


@dataclass(frozen=True, slots=True)
class SCEMStepDiagnostics:
    """Complete nonlinear and geometric diagnostics for one accepted step."""

    iterations: int
    density_residual: float
    hamiltonian_residual: float
    final_damping: float
    density_residual_history: tuple[float, ...]
    hamiltonian_residual_history: tuple[float, ...]
    damping_history: tuple[float, ...]
    predictor_link: LinkDiagnostics | None
    half_step_link: LinkDiagnostics
    full_step_link: LinkDiagnostics
    maximum_correction_norm: float
    initial_metric_residual: float
    final_metric_residual: float
    maximum_hermitian_cleanup_norm: float
    metric_correction_diagnostic_mode: bool
    converged: bool = True


@dataclass(frozen=True, slots=True)
class PropagationStepResult:
    """One accepted boundary state plus the converged midpoint intermediates."""

    state: OrbitalState
    midpoint_evaluation: InstantaneousEvaluation
    midpoint_source: FormulationSourceSample
    diagnostics: SCEMStepDiagnostics

    @property
    def start_step(self) -> int:
        return self.state.step_index - 1
