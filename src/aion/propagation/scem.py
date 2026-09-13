"""One shared density-fixed-point SCEM engine for both transport geometries."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

from aion.backends import Workspace
from aion.config import PropagationConfig
from aion.errors import (
    MetricConstraintError,
    MidpointConvergenceError,
    PropagationError,
)
from aion.formulations import AODensity, Formulation, InstantaneousEvaluation
from aion.propagation.linalg import (
    density_from_orbitals,
    hamiltonian_residual,
    hermitian_cleanup,
    metric_density_residual,
    metric_roundoff_limit,
    orbital_metric_residual,
)
from aion.propagation.transports import (
    ConnectionAwareTransport,
    FixedMetricTransport,
    StepSourceSamples,
    TransportTarget,
)
from aion.propagation.types import (
    LinkDiagnostics,
    OrbitalState,
    PropagationStepResult,
    SCEMStepDiagnostics,
)


@dataclass(slots=True)
class PropagationScratch:
    """Stable backend-resident buffers allocated once per simulation."""

    density_trial: Any
    density_output: Any
    density_mixed: Any
    coefficients_midpoint: Any
    coefficients_endpoint: Any
    identity: Any

    @classmethod
    def create(cls, workspace: Workspace, *, nao: int, norbital: int) -> PropagationScratch:
        xp = workspace.backend.namespace
        density_shape = (nao, nao)
        coefficient_shape = (nao, norbital)
        trial = workspace.allocate("propagation.density_trial", density_shape, dtype=xp.complex128)
        output = workspace.allocate(
            "propagation.density_output", density_shape, dtype=xp.complex128
        )
        mixed = workspace.allocate("propagation.density_mixed", density_shape, dtype=xp.complex128)
        midpoint = workspace.allocate(
            "propagation.coefficients_midpoint", coefficient_shape, dtype=xp.complex128
        )
        endpoint = workspace.allocate(
            "propagation.coefficients_endpoint", coefficient_shape, dtype=xp.complex128
        )
        identity = workspace.allocate("propagation.identity", density_shape, dtype=xp.complex128)
        identity[...] = xp.eye(nao, dtype=xp.complex128)
        return cls(trial, output, mixed, midpoint, endpoint, identity)


@dataclass(slots=True)
class DampedPicardController:
    """Residual-monitored damping policy shared by every SCEM transport."""

    minimum_damping: float
    damping: float = 1.0
    previous_residual: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_damping) or not 0.0 < self.minimum_damping <= 1.0:
            raise PropagationError("minimum Picard damping must lie in (0, 1]")

    def observe(self, residual: float) -> float:
        if not math.isfinite(residual) or residual < 0.0:
            raise PropagationError("Picard residual must be finite and nonnegative")
        if self.previous_residual is not None and residual > self.previous_residual:
            self.damping = max(self.minimum_damping, 0.5 * self.damping)
        self.previous_residual = residual
        return self.damping


def _stable_evaluation(
    evaluation: InstantaneousEvaluation,
    *,
    backend: Any,
) -> InstantaneousEvaluation:
    """Detach accepted midpoint records from reusable hot-loop buffers."""

    xp = backend.namespace
    density_matrix = xp.array(evaluation.density.matrix, dtype=xp.complex128, copy=True)
    field_free_density = xp.array(evaluation.field_free_density, dtype=xp.complex128, copy=True)
    stable_density = AODensity.from_matrix(density_matrix, backend)
    stable_dft = replace(evaluation.field_free_dft, density=field_free_density)
    return replace(
        evaluation,
        density=stable_density,
        field_free_density=field_free_density,
        field_free_dft=stable_dft,
    )


def _link_residual_to_enforce(diagnostics: LinkDiagnostics) -> float:
    return (
        diagnostics.corrected_metric_residual
        if diagnostics.correction_applied
        else diagnostics.raw_metric_residual
    )


@dataclass(slots=True)
class SCEMPropagator:
    """Self-consistent exponential midpoint with pluggable transport geometry."""

    formulation: Formulation
    config: PropagationConfig
    workspace: Workspace
    transport: FixedMetricTransport | ConnectionAwareTransport
    scratch: PropagationScratch

    @property
    def backend(self) -> Any:
        return self.workspace.backend

    def _evaluate(self, density: Any, source: Any) -> InstantaneousEvaluation:
        return self.formulation.evaluate(AODensity.from_matrix(density, self.backend), source)

    def _clean_density(self, density: Any, *, out: Any, name: str) -> tuple[Any, float]:
        return hermitian_cleanup(
            density,
            threshold=self.config.hermitian_cleanup_threshold,
            backend=self.backend,
            name=name,
            out=out,
        )

    def _check_link(self, diagnostics: LinkDiagnostics, *, size: int, name: str) -> None:
        residual = _link_residual_to_enforce(diagnostics)
        limit = metric_roundoff_limit(size)
        if residual > limit:
            raise MetricConstraintError(
                f"{name} cross-metric residual {residual:.3e} exceeds "
                f"FP64 roundoff envelope {limit:.3e}"
            )

    def step(self, state: OrbitalState) -> PropagationStepResult:
        """Advance exactly one interval; an unconverged midpoint is never returned."""

        if not isinstance(state, OrbitalState):
            raise TypeError("SCEM state must be OrbitalState")
        if state.backend is not self.backend:
            raise PropagationError("orbital state belongs to a different backend")
        step_index = state.step_index
        grid = self.config.time_grid
        if step_index >= grid.intervals:
            raise PropagationError(
                f"state step {step_index} has no following interval in the fixed grid"
            )
        samples = StepSourceSamples.from_workspace(
            self.workspace,
            gauge=self.formulation.gauge,
            velocity_fraction=self.formulation.gauge_velocity_fraction,
            step_index=step_index,
        )
        prepared = self.transport.prepare(samples)
        metric_limit = metric_roundoff_limit(prepared.start_metric.shape[0])
        initial_metric_residual = orbital_metric_residual(
            state.coefficients, prepared.start_metric, self.backend
        )
        if initial_metric_residual > metric_limit:
            raise MetricConstraintError(
                f"input orbital metric residual {initial_metric_residual:.3e} exceeds "
                f"FP64 roundoff envelope {metric_limit:.3e}"
            )

        predictor_coefficients, predictor_diagnostics = prepared.predictor_coefficients(
            state.coefficients,
            identity=self.scratch.identity,
            out=self.scratch.coefficients_midpoint,
        )
        if predictor_diagnostics is not None and not prepared.metric_correction_diagnostic_mode:
            self._check_link(
                predictor_diagnostics,
                size=prepared.start_metric.shape[0],
                name="predictor",
            )
        seed_coefficients = (
            state.coefficients if predictor_coefficients is None else predictor_coefficients
        )
        density_from_orbitals(
            seed_coefficients,
            state.occupations,
            backend=self.backend,
            out=self.scratch.density_trial,
        )
        trial_density, cleanup = self._clean_density(
            self.scratch.density_trial,
            out=self.scratch.density_trial,
            name="initial midpoint density",
        )
        maximum_cleanup = cleanup
        evaluation_in = self._evaluate(trial_density, samples.midpoint)
        electron_count = state.electron_count
        density_history: list[float] = []
        hamiltonian_history: list[float] = []
        damping_history: list[float] = []
        controller = DampedPicardController(self.config.minimum_damping)
        half_application = None
        evaluation_out = evaluation_in

        for iteration in range(1, self.config.max_iterations + 1):
            half_application = prepared.apply(
                state.coefficients,
                evaluation_in,
                TransportTarget.MIDPOINT,
                approximation=self.config.rational_approximation,
                identity=self.scratch.identity,
                out=self.scratch.coefficients_midpoint,
            )
            if not prepared.metric_correction_diagnostic_mode:
                self._check_link(
                    half_application.diagnostics,
                    size=prepared.start_metric.shape[0],
                    name="SCEM half-step",
                )
            density_from_orbitals(
                half_application.coefficients,
                state.occupations,
                backend=self.backend,
                out=self.scratch.density_output,
            )
            output_density, cleanup = self._clean_density(
                self.scratch.density_output,
                out=self.scratch.density_output,
                name="SCEM output midpoint density",
            )
            maximum_cleanup = max(maximum_cleanup, cleanup)
            evaluation_out = self._evaluate(output_density, samples.midpoint)
            density_residual = metric_density_residual(
                output_density,
                trial_density,
                prepared.midpoint_metric,
                electron_count=electron_count,
                backend=self.backend,
            )
            h_residual = hamiltonian_residual(
                evaluation_out.triple.hamiltonian_eom,
                evaluation_in.triple.hamiltonian_eom,
                self.backend,
            )
            damping = controller.observe(density_residual)
            density_history.append(density_residual)
            hamiltonian_history.append(h_residual)
            damping_history.append(damping)

            if density_residual <= self.config.density_tolerance:
                full_application = prepared.apply(
                    state.coefficients,
                    evaluation_out,
                    TransportTarget.ENDPOINT,
                    approximation=self.config.rational_approximation,
                    identity=self.scratch.identity,
                    out=self.scratch.coefficients_endpoint,
                )
                if not prepared.metric_correction_diagnostic_mode:
                    self._check_link(
                        full_application.diagnostics,
                        size=prepared.start_metric.shape[0],
                        name="SCEM full-step",
                    )
                final_metric_residual = orbital_metric_residual(
                    full_application.coefficients,
                    prepared.endpoint_metric,
                    self.backend,
                )
                if (
                    not prepared.metric_correction_diagnostic_mode
                    and final_metric_residual > metric_limit
                ):
                    raise MetricConstraintError(
                        f"output orbital metric residual {final_metric_residual:.3e} exceeds "
                        f"FP64 roundoff envelope {metric_limit:.3e}"
                    )
                xp = self.backend.namespace
                stable_coefficients = xp.array(
                    full_application.coefficients, dtype=xp.complex128, copy=True
                )
                next_state = OrbitalState(
                    stable_coefficients,
                    state.occupations,
                    self.backend,
                    step_index + 1,
                )
                stable_midpoint = _stable_evaluation(evaluation_out, backend=self.backend)
                links = [half_application.diagnostics, full_application.diagnostics]
                if predictor_diagnostics is not None:
                    links.append(predictor_diagnostics)
                diagnostics = SCEMStepDiagnostics(
                    iterations=iteration,
                    density_residual=density_residual,
                    hamiltonian_residual=h_residual,
                    final_damping=damping,
                    density_residual_history=tuple(density_history),
                    hamiltonian_residual_history=tuple(hamiltonian_history),
                    damping_history=tuple(damping_history),
                    predictor_link=predictor_diagnostics,
                    half_step_link=half_application.diagnostics,
                    full_step_link=full_application.diagnostics,
                    maximum_correction_norm=max(link.correction_norm for link in links),
                    initial_metric_residual=initial_metric_residual,
                    final_metric_residual=final_metric_residual,
                    maximum_hermitian_cleanup_norm=maximum_cleanup,
                    metric_correction_diagnostic_mode=(prepared.metric_correction_diagnostic_mode),
                )
                return PropagationStepResult(
                    state=next_state,
                    midpoint_evaluation=stable_midpoint,
                    midpoint_source=samples.midpoint,
                    diagnostics=diagnostics,
                )

            if damping == 1.0:
                self.scratch.density_trial[...] = output_density
                trial_density = self.scratch.density_trial
                evaluation_in = evaluation_out
            else:
                self.scratch.density_mixed[...] = (
                    1.0 - damping
                ) * trial_density + damping * output_density
                trial_density, cleanup = self._clean_density(
                    self.scratch.density_mixed,
                    out=self.scratch.density_trial,
                    name="damped SCEM midpoint density",
                )
                maximum_cleanup = max(maximum_cleanup, cleanup)
                evaluation_in = self._evaluate(trial_density, samples.midpoint)

        assert half_application is not None
        raise MidpointConvergenceError(
            "SCEM midpoint density did not converge: "
            f"step={step_index} iterations={self.config.max_iterations} "
            f"density_residual={density_history[-1]:.6e} "
            f"hamiltonian_residual={hamiltonian_history[-1]:.6e}",
            step_index=step_index,
            iterations=self.config.max_iterations,
            density_residual=density_history[-1],
            hamiltonian_residual=hamiltonian_history[-1],
        )

    def propagate(
        self,
        state: OrbitalState,
        *,
        intervals: int | None = None,
    ) -> Iterator[PropagationStepResult]:
        """Yield accepted in-memory steps without performing WP5 I/O or event handling."""

        remaining = self.config.time_grid.intervals - state.step_index
        count = remaining if intervals is None else intervals
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise PropagationError("propagation interval count must be a nonnegative integer")
        if count > remaining:
            raise PropagationError("requested propagation extends beyond the fixed time grid")
        current = state
        for _ in range(count):
            result = self.step(current)
            current = result.state
            yield result
