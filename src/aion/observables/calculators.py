"""Typed dependency-declared, independently scheduled observable calculators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from aion.config import FixedTimeGrid, FormulationKind, ObservableSchedules, StepSchedule
from aion.electronic_structure import expectation
from aion.formulations import (
    CurrentLedger,
    EnergyLedger,
    Formulation,
    FormulationSourceSample,
    InstantaneousEvaluation,
)
from aion.observables.definitions import observable_definition_catalog
from aion.observables.records import ObservableDefinition


class ObservableDependency(StrEnum):
    DENSITY = "density"
    DENSITY_DOT = "density_dot"
    EOM_TRIPLE = "eom_triple"
    FIELD_FREE_DFT = "field_free_dft"
    SOURCE = "source"
    P0_TOPOLOGY = "p0_topology"
    E1_TENSORS = "e1_tensors"


@dataclass(frozen=True, slots=True)
class InstantaneousDiagnostics:
    electron_count: Any
    density_hermiticity_residual: Any
    metric_compatibility_residual: Any
    continuity_residual: Any | None
    source_current_dipole_derivative_residual: Any


@dataclass(frozen=True, slots=True)
class DipoleCurrentCalculator:
    schedule: StepSchedule
    dependencies: frozenset[ObservableDependency] = frozenset(
        {
            ObservableDependency.DENSITY,
            ObservableDependency.DENSITY_DOT,
            ObservableDependency.EOM_TRIPLE,
            ObservableDependency.SOURCE,
        }
    )

    def is_scheduled(self, step: int, grid: FixedTimeGrid) -> bool:
        return step in self.schedule.steps(grid)

    def calculate(
        self,
        formulation: Formulation,
        evaluation: InstantaneousEvaluation,
        source: FormulationSourceSample,
    ) -> CurrentLedger:
        return formulation.currents(evaluation, source)


@dataclass(frozen=True, slots=True)
class EnergyCalculator:
    schedule: StepSchedule
    dependencies: frozenset[ObservableDependency] = frozenset(
        {
            ObservableDependency.DENSITY,
            ObservableDependency.DENSITY_DOT,
            ObservableDependency.FIELD_FREE_DFT,
            ObservableDependency.SOURCE,
        }
    )

    def is_scheduled(self, step: int, grid: FixedTimeGrid) -> bool:
        return step in self.schedule.steps(grid)

    def calculate(
        self,
        formulation: Formulation,
        evaluation: InstantaneousEvaluation,
        source: FormulationSourceSample,
        *,
        initial_matter_energy: Any | None = None,
        accumulated_source_work: Any | None = None,
    ) -> EnergyLedger:
        return formulation.energy(
            evaluation,
            source,
            initial_matter_energy=initial_matter_energy,
            accumulated_source_work=accumulated_source_work,
        )


@dataclass(frozen=True, slots=True)
class DiagnosticCalculator:
    schedule: StepSchedule
    dependencies: frozenset[ObservableDependency] = frozenset(
        {
            ObservableDependency.DENSITY,
            ObservableDependency.DENSITY_DOT,
            ObservableDependency.EOM_TRIPLE,
            ObservableDependency.SOURCE,
        }
    )

    def is_scheduled(self, step: int, grid: FixedTimeGrid) -> bool:
        return step in self.schedule.steps(grid)

    def calculate(
        self,
        formulation: Formulation,
        evaluation: InstantaneousEvaluation,
        source: FormulationSourceSample,
    ) -> InstantaneousDiagnostics:
        current = formulation.currents(evaluation, source)
        context = formulation.context
        xp = context.namespace
        compatibility = (
            evaluation.triple.connection
            + (evaluation.triple.connection.conj().T)
            - evaluation.metric_dot
        )
        compatibility_scale = xp.maximum(xp.asarray(1.0), xp.linalg.norm(evaluation.metric_dot))
        return InstantaneousDiagnostics(
            electron_count=expectation(evaluation.density.matrix, evaluation.triple.metric, xp),
            density_hermiticity_residual=evaluation.density.hermiticity_residual(),
            metric_compatibility_residual=xp.linalg.norm(compatibility) / compatibility_scale,
            continuity_residual=current.continuity_residual,
            source_current_dipole_derivative_residual=(
                current.source_current_dipole_derivative_residual
            ),
        )


@dataclass(frozen=True, slots=True)
class ObservableCalculators:
    dipole_current: DipoleCurrentCalculator
    energy: EnergyCalculator
    diagnostics: DiagnosticCalculator
    definitions: Mapping[str, ObservableDefinition]


def build_observable_calculators(
    schedules: ObservableSchedules,
    formulation: FormulationKind,
    *,
    natom: int,
    npair: int,
) -> ObservableCalculators:
    """Bind each family to its own integer-step schedule."""

    return ObservableCalculators(
        dipole_current=DipoleCurrentCalculator(schedules.dipole_current),
        energy=EnergyCalculator(schedules.energy),
        diagnostics=DiagnosticCalculator(schedules.diagnostics),
        definitions=MappingProxyType(
            observable_definition_catalog(formulation, natom=natom, npair=npair)
        ),
    )
