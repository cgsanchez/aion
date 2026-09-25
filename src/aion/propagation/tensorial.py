"""Tensorial density propagation on an evolving AO manifold.

The stateful contravariant-density engine implements the accepted nonlinear
congruence algorithm.  Earlier mixed-similarity experiments remain available
as explicit experimental records and functions for numerical comparison.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import InitVar, dataclass, field
from typing import Any

from aion.backends import ArrayBackend
from aion.config import RationalApproximation
from aion.errors import PropagationError
from aion.formulations import EOMTriple
from aion.propagation.linalg import cross_metric_residual, rational_map, relative_frobenius


@dataclass(frozen=True, slots=True)
class ExperimentalGaussMagnusHistory:
    """Exact endpoint metrics and EOM triples at both Gauss nodes per step."""

    endpoint_metrics: tuple[Any, ...]
    gauss_minus_triples: tuple[EOMTriple, ...]
    gauss_plus_triples: tuple[EOMTriple, ...]
    interval_au: float
    hbar: float = 1.0

    def __post_init__(self) -> None:
        intervals = len(self.gauss_minus_triples)
        if intervals == 0:
            raise PropagationError("Gauss--Magnus history must contain at least one interval")
        if len(self.gauss_plus_triples) != intervals:
            raise PropagationError("Gauss--Magnus node histories must have equal length")
        if len(self.endpoint_metrics) != intervals + 1:
            raise PropagationError(
                "Gauss--Magnus history requires one more endpoint metric than interval"
            )
        if not math.isfinite(self.interval_au) or self.interval_au <= 0.0:
            raise PropagationError("Gauss--Magnus interval must be finite and positive")
        if not math.isfinite(self.hbar) or self.hbar <= 0.0:
            raise PropagationError("Gauss--Magnus hbar must be finite and positive")


@dataclass(frozen=True, slots=True)
class MixedDensityStepDiagnostics:
    """Uncorrected geometric and algebraic residuals for one Magnus step."""

    cross_metric_residual: float
    trace_drift: float
    trace_imaginary_abs: float
    input_idempotency_residual: float
    output_idempotency_residual: float
    metric_hermiticity_residual: float
    contravariant_hermiticity_residual: float


@dataclass(frozen=True, slots=True)
class ExperimentalMixedDensityPropagation:
    """Boundary mixed densities, raw links, and diagnostics for a trajectory."""

    mixed_densities: tuple[Any, ...]
    contravariant_densities: tuple[Any, ...]
    links: tuple[Any, ...]
    diagnostics: tuple[MixedDensityStepDiagnostics, ...]
    metric_correction_applied: bool = False


@dataclass(frozen=True, slots=True)
class NonlinearGaussMagnusPolicy:
    """Fixed-point controls for the implicit nonlinear Gauss nodes."""

    tolerance: float = 1.0e-10
    maximum_iterations: int = 40

    def __post_init__(self) -> None:
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise PropagationError("nonlinear tolerance must be finite and positive")
        if (
            isinstance(self.maximum_iterations, bool)
            or not isinstance(self.maximum_iterations, int)
            or self.maximum_iterations <= 0
        ):
            raise PropagationError("maximum nonlinear iterations must be positive")


@dataclass(frozen=True, slots=True)
class NonlinearMixedDensityStepDiagnostics:
    """Uncorrected diagnostics for one converged nonlinear Magnus step."""

    start_time_au: float
    end_time_au: float
    nonlinear_iterations: int
    nonlinear_residual: float
    cross_metric_residual: float
    trace_drift: float
    trace_imaginary_abs: float
    occupation_spectrum_drift: float
    metric_hermiticity_residual: float
    contravariant_hermiticity_residual: float


@dataclass(frozen=True, slots=True)
class NonlinearMixedDensityPropagation:
    """Boundary states and converged Gauss nodes of a nonlinear trajectory."""

    times_au: tuple[float, ...]
    mixed_densities: tuple[Any, ...]
    contravariant_densities: tuple[Any, ...]
    gauss_minus_mixed_densities: tuple[Any, ...]
    gauss_plus_mixed_densities: tuple[Any, ...]
    links: tuple[Any, ...]
    diagnostics: tuple[NonlinearMixedDensityStepDiagnostics, ...]
    metric_correction_applied: bool = False


@dataclass(frozen=True, slots=True)
class NonlinearContravariantDensityPropagation:
    """Boundary and Gauss-node densities from uncorrected congruence links."""

    times_au: tuple[float, ...]
    contravariant_densities: tuple[Any, ...]
    mixed_densities: tuple[Any, ...]
    gauss_minus_contravariant_densities: tuple[Any, ...]
    gauss_plus_contravariant_densities: tuple[Any, ...]
    links: tuple[Any, ...]
    diagnostics: tuple[NonlinearMixedDensityStepDiagnostics, ...]
    metric_correction_applied: bool = False
    density_update: str = "coefficient_congruence"


@dataclass(frozen=True, slots=True)
class NonlinearContravariantDensityStep[EvaluationT]:
    """One accepted nonlinear congruence step and its converged node data."""

    start_boundary_index: int
    end_boundary_index: int
    start_time_au: float
    end_time_au: float
    gauss_minus_time_au: float
    gauss_plus_time_au: float
    contravariant_density: Any
    mixed_density: Any
    gauss_minus_contravariant_density: Any
    gauss_plus_contravariant_density: Any
    gauss_minus_evaluation: EvaluationT
    gauss_plus_evaluation: EvaluationT
    link: Any
    diagnostics: NonlinearMixedDensityStepDiagnostics
    metric_correction_applied: bool = False
    density_update: str = "coefficient_congruence"


@dataclass(slots=True)
class NonlinearContravariantDensityPropagator[EvaluationT]:
    """Stateful accepted fourth-order nonlinear contravariant-density engine.

    The object owns exactly one accepted boundary density and fixed-size
    nonlinear scratch.  A successful :meth:`step` commits one congruence
    update.  Failed node solves leave the accepted state and boundary index
    unchanged.
    """

    initial_contravariant_density: InitVar[Any]
    initial_time_au: float
    interval_au: float
    metric_provider: Callable[[float], Any]
    evaluation_provider: Callable[[float, Any], EvaluationT]
    eom_extractor: Callable[[EvaluationT], EOMTriple]
    backend: ArrayBackend
    policy: NonlinearGaussMagnusPolicy = field(default_factory=NonlinearGaussMagnusPolicy)
    hbar: float = 1.0
    initial_boundary_index: int = 0
    initial_occupation_spectrum: InitVar[Any | None] = None
    _current_contravariant_density: Any = field(init=False, repr=False)
    _initial_occupation_spectrum: Any = field(init=False, repr=False)
    _boundary_index: int = field(init=False, repr=False)
    _dimension: int = field(init=False, repr=False)

    def __post_init__(
        self,
        initial_contravariant_density: Any,
        initial_occupation_spectrum: Any | None,
    ) -> None:
        _validate_square(
            initial_contravariant_density,
            self.backend,
            "initial contravariant density",
        )
        start_time = float(self.initial_time_au)
        step = float(self.interval_au)
        if not math.isfinite(start_time):
            raise PropagationError("initial time must be finite")
        if not math.isfinite(step) or step <= 0.0:
            raise PropagationError("propagation interval must be finite and positive")
        if not isinstance(self.policy, NonlinearGaussMagnusPolicy):
            raise TypeError("policy must be a NonlinearGaussMagnusPolicy")
        if not math.isfinite(self.hbar) or self.hbar <= 0.0:
            raise PropagationError("hbar must be finite and positive")
        if (
            isinstance(self.initial_boundary_index, bool)
            or not isinstance(self.initial_boundary_index, int)
            or self.initial_boundary_index < 0
        ):
            raise PropagationError("initial boundary index must be a nonnegative integer")
        if not callable(self.metric_provider):
            raise TypeError("metric_provider must be callable")
        if not callable(self.evaluation_provider):
            raise TypeError("evaluation_provider must be callable")
        if not callable(self.eom_extractor):
            raise TypeError("eom_extractor must be callable")

        object.__setattr__(self, "initial_time_au", start_time)
        object.__setattr__(self, "interval_au", step)
        xp = self.backend.namespace
        dimension = initial_contravariant_density.shape[0]
        current = xp.array(
            initial_contravariant_density,
            dtype=xp.complex128,
            copy=True,
        )
        initial_metric = self.metric_provider(start_time)
        _validate_matrix_shape(
            initial_metric,
            dimension,
            self.backend,
            "initial endpoint metric",
        )
        object.__setattr__(self, "_dimension", dimension)
        object.__setattr__(self, "_current_contravariant_density", current)
        if initial_occupation_spectrum is None:
            spectrum = xp.linalg.eigvals(current @ initial_metric)
        else:
            spectrum = xp.asarray(initial_occupation_spectrum, dtype=xp.complex128)
            self.backend.assert_resident(spectrum, name="initial occupation spectrum")
            if spectrum.shape != (dimension,):
                raise PropagationError("initial occupation spectrum has an incompatible dimension")
        object.__setattr__(self, "_initial_occupation_spectrum", spectrum)
        object.__setattr__(self, "_boundary_index", self.initial_boundary_index)

    @property
    def boundary_index(self) -> int:
        return self._boundary_index

    @property
    def current_time_au(self) -> float:
        offset = self._boundary_index - self.initial_boundary_index
        return self.initial_time_au + offset * self.interval_au

    @property
    def current_contravariant_density(self) -> Any:
        """Return the backend-resident accepted state owned by this propagator."""

        return self._current_contravariant_density

    def current_mixed_density(self) -> Any:
        """Return ``P S`` at the current exact endpoint metric."""

        metric = self.metric_provider(self.current_time_au)
        _validate_matrix_shape(
            metric,
            self._dimension,
            self.backend,
            "current endpoint metric",
        )
        return self._current_contravariant_density @ metric

    def _triple(self, evaluation: EvaluationT) -> EOMTriple:
        triple = self.eom_extractor(evaluation)
        _validate_triple(triple, self.backend, expected_dimension=self._dimension)
        return triple

    def step(self) -> NonlinearContravariantDensityStep[EvaluationT]:
        """Solve and atomically commit one nonlinear two-node Magnus step."""

        backend = self.backend
        xp = backend.namespace
        step = self.interval_au
        left_index = self._boundary_index
        right_index = left_index + 1
        left_time = self.current_time_au
        right_time = left_time + step
        c_minus = 0.5 - math.sqrt(3.0) / 6.0
        c_plus = 0.5 + math.sqrt(3.0) / 6.0
        minus_time = left_time + c_minus * step
        plus_time = left_time + c_plus * step
        start_metric = self.metric_provider(left_time)
        target_metric = self.metric_provider(right_time)
        _validate_matrix_shape(
            start_metric,
            self._dimension,
            backend,
            "start endpoint metric",
        )
        _validate_matrix_shape(
            target_metric,
            self._dimension,
            backend,
            "target endpoint metric",
        )
        current = self._current_contravariant_density
        guess_minus = xp.array(current, dtype=xp.complex128, copy=True)
        guess_plus = xp.array(current, dtype=xp.complex128, copy=True)
        converged = False
        nonlinear_residual = math.inf
        final_minus: EvaluationT | None = None
        final_plus: EvaluationT | None = None
        nonlinear_iterations = 0

        for iteration in range(1, self.policy.maximum_iterations + 1):
            nonlinear_iterations = iteration
            try:
                minus_evaluation = self.evaluation_provider(minus_time, guess_minus)
                plus_evaluation = self.evaluation_provider(plus_time, guess_plus)
            except Exception as exc:
                raise PropagationError(
                    "nonlinear congruence node evaluation failed at interval "
                    f"{left_index}, iteration {iteration}"
                ) from exc
            minus_generator = mixed_eom_generator(
                self._triple(minus_evaluation),
                backend,
                hbar=self.hbar,
            )
            plus_generator = mixed_eom_generator(
                self._triple(plus_evaluation),
                backend,
                hbar=self.hbar,
            )
            minus_link = partial_fourth_order_gauss_magnus_link(
                minus_generator,
                plus_generator,
                step,
                c_minus,
                backend,
            )
            plus_link = partial_fourth_order_gauss_magnus_link(
                minus_generator,
                plus_generator,
                step,
                c_plus,
                backend,
            )
            proposed_minus = minus_link @ current @ minus_link.conj().T
            proposed_plus = plus_link @ current @ plus_link.conj().T
            nonlinear_residual = max(
                relative_frobenius(proposed_minus - guess_minus, proposed_minus, backend),
                relative_frobenius(proposed_plus - guess_plus, proposed_plus, backend),
            )
            guess_minus = proposed_minus
            guess_plus = proposed_plus
            if nonlinear_residual <= self.policy.tolerance:
                final_minus = self.evaluation_provider(minus_time, guess_minus)
                final_plus = self.evaluation_provider(plus_time, guess_plus)
                check_minus_generator = mixed_eom_generator(
                    self._triple(final_minus),
                    backend,
                    hbar=self.hbar,
                )
                check_plus_generator = mixed_eom_generator(
                    self._triple(final_plus),
                    backend,
                    hbar=self.hbar,
                )
                check_minus_link = partial_fourth_order_gauss_magnus_link(
                    check_minus_generator,
                    check_plus_generator,
                    step,
                    c_minus,
                    backend,
                )
                check_plus_link = partial_fourth_order_gauss_magnus_link(
                    check_minus_generator,
                    check_plus_generator,
                    step,
                    c_plus,
                    backend,
                )
                check_minus = check_minus_link @ current @ check_minus_link.conj().T
                check_plus = check_plus_link @ current @ check_plus_link.conj().T
                nonlinear_residual = max(
                    relative_frobenius(check_minus - guess_minus, check_minus, backend),
                    relative_frobenius(check_plus - guess_plus, check_plus, backend),
                )
                guess_minus = check_minus
                guess_plus = check_plus
                if nonlinear_residual <= self.policy.tolerance:
                    converged = True
                    break

        if not converged or final_minus is None or final_plus is None:
            raise PropagationError(
                "nonlinear congruence Gauss-node solve failed at interval "
                f"{left_index} after {self.policy.maximum_iterations} iterations; "
                f"residual {nonlinear_residual:.3e}"
            )

        final_minus = self.evaluation_provider(minus_time, guess_minus)
        final_plus = self.evaluation_provider(plus_time, guess_plus)
        link = fourth_order_gauss_magnus_link(
            self._triple(final_minus),
            self._triple(final_plus),
            step,
            backend,
            hbar=self.hbar,
        )
        input_mixed = current @ start_metric
        input_trace = xp.trace(input_mixed)
        next_density = link @ current @ link.conj().T
        mixed = next_density @ target_metric
        output_trace = xp.trace(mixed)
        trace_scale = xp.maximum(xp.asarray(1.0), xp.abs(input_trace))
        spectrum = xp.linalg.eigvals(mixed)
        spectrum_scale = xp.maximum(
            xp.asarray(1.0),
            xp.linalg.norm(self._initial_occupation_spectrum),
        )
        real_spectrum_drift = (
            xp.linalg.norm(
                xp.sort(xp.real(spectrum)) - xp.sort(xp.real(self._initial_occupation_spectrum))
            )
            / spectrum_scale
        )
        imaginary_spectrum_drift = (
            xp.maximum(
                xp.max(xp.abs(xp.imag(spectrum))),
                xp.max(xp.abs(xp.imag(self._initial_occupation_spectrum))),
            )
            / spectrum_scale
        )
        diagnostics = NonlinearMixedDensityStepDiagnostics(
            start_time_au=left_time,
            end_time_au=right_time,
            nonlinear_iterations=nonlinear_iterations,
            nonlinear_residual=nonlinear_residual,
            cross_metric_residual=cross_metric_residual(
                link,
                start_metric,
                target_metric,
                backend,
            ),
            trace_drift=backend.scalar_to_float(xp.abs(output_trace - input_trace) / trace_scale),
            trace_imaginary_abs=backend.scalar_to_float(xp.abs(xp.imag(output_trace))),
            occupation_spectrum_drift=backend.scalar_to_float(
                xp.maximum(real_spectrum_drift, imaginary_spectrum_drift)
            ),
            metric_hermiticity_residual=mixed_metric_hermiticity_residual(
                mixed,
                target_metric,
                backend,
            ),
            contravariant_hermiticity_residual=relative_frobenius(
                next_density - next_density.conj().T,
                next_density,
                backend,
            ),
        )

        accepted_density = xp.array(next_density, dtype=xp.complex128, copy=True)
        result = NonlinearContravariantDensityStep(
            start_boundary_index=left_index,
            end_boundary_index=right_index,
            start_time_au=left_time,
            end_time_au=right_time,
            gauss_minus_time_au=minus_time,
            gauss_plus_time_au=plus_time,
            contravariant_density=accepted_density,
            mixed_density=xp.array(mixed, dtype=xp.complex128, copy=True),
            gauss_minus_contravariant_density=xp.array(
                guess_minus,
                dtype=xp.complex128,
                copy=True,
            ),
            gauss_plus_contravariant_density=xp.array(
                guess_plus,
                dtype=xp.complex128,
                copy=True,
            ),
            gauss_minus_evaluation=final_minus,
            gauss_plus_evaluation=final_plus,
            link=xp.array(link, dtype=xp.complex128, copy=True),
            diagnostics=diagnostics,
        )
        object.__setattr__(self, "_current_contravariant_density", accepted_density)
        object.__setattr__(self, "_boundary_index", right_index)
        return result


def contravariant_to_mixed_density(
    contravariant_density: Any,
    metric: Any,
    backend: ArrayBackend,
) -> Any:
    """Raise one AO-density index: ``D^mu_nu = P^{mu lambda} S_lambda nu``."""

    _validate_square_pair(contravariant_density, metric, backend, "contravariant density")
    return contravariant_density @ metric


def mixed_to_contravariant_density(
    mixed_density: Any,
    metric: Any,
    backend: ArrayBackend,
) -> Any:
    """Recover ``P = D S^-1`` by a linear solve, without forming an inverse."""

    _validate_square_pair(mixed_density, metric, backend, "mixed density")
    xp = backend.namespace
    try:
        result = xp.linalg.solve(metric.T, mixed_density.T).T
    except Exception as exc:
        raise PropagationError("mixed-density metric solve failed") from exc
    _assert_finite(result, backend, "contravariant density")
    return result


def mixed_eom_generator(
    triple: EOMTriple,
    backend: ArrayBackend,
    *,
    hbar: float = 1.0,
) -> Any:
    r"""Return ``G=S^-1(-omega-i K/hbar)`` for ``Cdot=G C``."""

    if not isinstance(triple, EOMTriple):
        raise TypeError("triple must be an EOMTriple")
    if not math.isfinite(hbar) or hbar <= 0.0:
        raise PropagationError("hbar must be finite and positive")
    _validate_triple(triple, backend)
    xp = backend.namespace
    try:
        result = xp.linalg.solve(
            triple.metric,
            -triple.connection - (1j / float(hbar)) * triple.hamiltonian_eom,
        )
    except Exception as exc:
        raise PropagationError("mixed EOM generator solve failed") from exc
    _assert_finite(result, backend, "mixed EOM generator")
    return result


def fourth_order_gauss_magnus_link(
    gauss_minus: EOMTriple,
    gauss_plus: EOMTriple,
    interval_au: float,
    backend: ArrayBackend,
    *,
    hbar: float = 1.0,
) -> Any:
    r"""Build the uncorrected fourth-order two-node Gauss--Magnus link.

    With the earlier Gauss node denoted by ``-`` and the later by ``+``,

    ``Omega = h/2 (G_- + G_+) - sqrt(3) h^2/12 [G_-, G_+]``.

    The exponential is evaluated with the fourth-order-compatible diagonal
    ``[2/2]`` Pade map.  No endpoint-metric correction is applied.
    """

    interval = float(interval_au)
    if not math.isfinite(interval) or interval <= 0.0:
        raise PropagationError("Gauss--Magnus interval must be finite and positive")
    left = mixed_eom_generator(gauss_minus, backend, hbar=hbar)
    right = mixed_eom_generator(gauss_plus, backend, hbar=hbar)
    if left.shape != right.shape:
        raise PropagationError("Gauss-node generators have incompatible shapes")
    commutator = left @ right - right @ left
    omega = (
        0.5 * interval * (left + right) - math.sqrt(3.0) * interval * interval * commutator / 12.0
    )
    return rational_map(
        omega,
        1.0,
        RationalApproximation.PADE_22,
        backend=backend,
    )


def partial_fourth_order_gauss_magnus_link(
    gauss_minus_generator: Any,
    gauss_plus_generator: Any,
    interval_au: float,
    fraction: float,
    backend: ArrayBackend,
) -> Any:
    r"""Transport to a fractional step using the Gauss-node interpolant.

    The two complete generators define the linear interpolant
    ``G(s)=a+b*s`` on ``s in [0,1]``.  Its fourth-order Magnus exponent from
    zero to ``fraction=c`` is

    ``h(c*a+c^2*b/2)-h^2*c^3*[a,b]/12``.

    At ``c=1`` this is algebraically the usual two-node fourth-order Gauss--
    Magnus exponent.  Fractional links close the nonlinear node fixed point;
    no endpoint-metric correction or orthogonalization is introduced.
    """

    interval = float(interval_au)
    coordinate = float(fraction)
    if not math.isfinite(interval) or interval <= 0.0:
        raise PropagationError("Gauss--Magnus interval must be finite and positive")
    if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 1.0:
        raise PropagationError("fractional Gauss--Magnus coordinate must lie in [0, 1]")
    _validate_square_pair(
        gauss_minus_generator,
        gauss_plus_generator,
        backend,
        "Gauss-node generator",
    )
    slope = math.sqrt(3.0) * (gauss_plus_generator - gauss_minus_generator)
    intercept = gauss_minus_generator - (0.5 - math.sqrt(3.0) / 6.0) * slope
    commutator = intercept @ slope - slope @ intercept
    omega = (
        interval * (coordinate * intercept + 0.5 * coordinate * coordinate * slope)
        - (interval * interval * coordinate**3 / 12.0) * commutator
    )
    return rational_map(
        omega,
        1.0,
        RationalApproximation.PADE_22,
        backend=backend,
    )


def similarity_transport(mixed_density: Any, link: Any, backend: ArrayBackend) -> Any:
    """Return ``link D link^-1`` using a right-side linear solve."""

    _validate_square_pair(mixed_density, link, backend, "mixed density")
    xp = backend.namespace
    left = link @ mixed_density
    try:
        result = xp.linalg.solve(link.T, left.T).T
    except Exception as exc:
        raise PropagationError("mixed-density similarity solve failed") from exc
    _assert_finite(result, backend, "transported mixed density")
    return result


def mixed_idempotency_residual(mixed_density: Any, backend: ArrayBackend) -> float:
    """Return the relative projector defect ``||D^2-D||``."""

    _validate_square(mixed_density, backend, "mixed density")
    return relative_frobenius(
        mixed_density @ mixed_density - mixed_density,
        mixed_density,
        backend,
    )


def mixed_metric_hermiticity_residual(
    mixed_density: Any,
    metric: Any,
    backend: ArrayBackend,
) -> float:
    r"""Return the mixed-index Hermiticity defect ``||D^dag S-S D||``."""

    _validate_square_pair(mixed_density, metric, backend, "mixed density")
    reference = metric @ mixed_density
    return relative_frobenius(mixed_density.conj().T @ metric - reference, reference, backend)


def propagate_experimental_mixed_density(
    initial_mixed_density: Any,
    history: ExperimentalGaussMagnusHistory,
    backend: ArrayBackend,
) -> ExperimentalMixedDensityPropagation:
    """Propagate a mixed AO density by raw fourth-order Magnus similarities."""

    if not isinstance(history, ExperimentalGaussMagnusHistory):
        raise TypeError("history must be an ExperimentalGaussMagnusHistory")
    _validate_square(initial_mixed_density, backend, "initial mixed density")
    xp = backend.namespace
    dimension = initial_mixed_density.shape[0]
    current = xp.array(initial_mixed_density, dtype=xp.complex128, copy=True)
    mixed_densities = [xp.array(current, dtype=xp.complex128, copy=True)]
    links: list[Any] = []
    diagnostics: list[MixedDensityStepDiagnostics] = []
    initial_metric = history.endpoint_metrics[0]
    _validate_matrix_shape(initial_metric, dimension, backend, "initial endpoint metric")
    contravariant = mixed_to_contravariant_density(current, initial_metric, backend)
    contravariant_densities = [xp.array(contravariant, dtype=xp.complex128, copy=True)]

    for index, (gauss_minus, gauss_plus) in enumerate(
        zip(history.gauss_minus_triples, history.gauss_plus_triples, strict=True)
    ):
        start_metric = history.endpoint_metrics[index]
        target_metric = history.endpoint_metrics[index + 1]
        _validate_matrix_shape(start_metric, dimension, backend, "start endpoint metric")
        _validate_matrix_shape(target_metric, dimension, backend, "target endpoint metric")
        for triple in (gauss_minus, gauss_plus):
            _validate_triple(triple, backend, expected_dimension=dimension)

        input_trace = xp.trace(current)
        input_idempotency = mixed_idempotency_residual(current, backend)
        link = fourth_order_gauss_magnus_link(
            gauss_minus,
            gauss_plus,
            history.interval_au,
            backend,
            hbar=history.hbar,
        )
        current = similarity_transport(current, link, backend)
        contravariant = mixed_to_contravariant_density(current, target_metric, backend)
        output_trace = xp.trace(current)
        trace_scale = xp.maximum(xp.asarray(1.0), xp.abs(input_trace))
        trace_drift = backend.scalar_to_float(xp.abs(output_trace - input_trace) / trace_scale)
        trace_imaginary_abs = backend.scalar_to_float(xp.abs(xp.imag(output_trace)))
        contravariant_hermiticity = relative_frobenius(
            contravariant - contravariant.conj().T,
            contravariant,
            backend,
        )
        diagnostics.append(
            MixedDensityStepDiagnostics(
                cross_metric_residual=cross_metric_residual(
                    link,
                    start_metric,
                    target_metric,
                    backend,
                ),
                trace_drift=trace_drift,
                trace_imaginary_abs=trace_imaginary_abs,
                input_idempotency_residual=input_idempotency,
                output_idempotency_residual=mixed_idempotency_residual(current, backend),
                metric_hermiticity_residual=mixed_metric_hermiticity_residual(
                    current,
                    target_metric,
                    backend,
                ),
                contravariant_hermiticity_residual=contravariant_hermiticity,
            )
        )
        mixed_densities.append(xp.array(current, dtype=xp.complex128, copy=True))
        contravariant_densities.append(xp.array(contravariant, dtype=xp.complex128, copy=True))
        links.append(xp.array(link, dtype=xp.complex128, copy=True))

    return ExperimentalMixedDensityPropagation(
        mixed_densities=tuple(mixed_densities),
        contravariant_densities=tuple(contravariant_densities),
        links=tuple(links),
        diagnostics=tuple(diagnostics),
    )


def propagate_nonlinear_mixed_density(
    initial_mixed_density: Any,
    *,
    initial_time_au: float,
    interval_au: float,
    intervals: int,
    metric_provider: Callable[[float], Any],
    eom_provider: Callable[[float, Any], EOMTriple],
    backend: ArrayBackend,
    policy: NonlinearGaussMagnusPolicy | None = None,
    hbar: float = 1.0,
) -> NonlinearMixedDensityPropagation:
    r"""Propagate ``D=PS`` with self-consistent fourth-order Gauss nodes.

    At each interval the two nonlinear generators are evaluated on mixed
    densities obtained by partial Magnus similarities from the left endpoint.
    The node densities and generators are iterated to the declared tolerance.
    The accepted endpoint update remains the uncorrected similarity
    ``D_{n+1}=U D_n U^{-1}``; failure to converge raises visibly rather than
    projecting or correcting the state.
    """

    _validate_square(initial_mixed_density, backend, "initial mixed density")
    start_time = float(initial_time_au)
    step = float(interval_au)
    if not math.isfinite(start_time):
        raise PropagationError("initial time must be finite")
    if not math.isfinite(step) or step <= 0.0:
        raise PropagationError("propagation interval must be finite and positive")
    if isinstance(intervals, bool) or not isinstance(intervals, int) or intervals <= 0:
        raise PropagationError("interval count must be a positive integer")
    if not math.isfinite(hbar) or hbar <= 0.0:
        raise PropagationError("hbar must be finite and positive")
    selected = NonlinearGaussMagnusPolicy() if policy is None else policy
    if not isinstance(selected, NonlinearGaussMagnusPolicy):
        raise TypeError("policy must be a NonlinearGaussMagnusPolicy")

    xp = backend.namespace
    dimension = initial_mixed_density.shape[0]
    current = xp.array(initial_mixed_density, dtype=xp.complex128, copy=True)
    initial_spectrum = xp.linalg.eigvals(current)
    times = tuple(start_time + index * step for index in range(intervals + 1))
    mixed = [xp.array(current, dtype=xp.complex128, copy=True)]
    initial_metric = metric_provider(times[0])
    _validate_matrix_shape(initial_metric, dimension, backend, "initial endpoint metric")
    contravariant = mixed_to_contravariant_density(current, initial_metric, backend)
    contravariant_values = [xp.array(contravariant, dtype=xp.complex128, copy=True)]
    minus_nodes: list[Any] = []
    plus_nodes: list[Any] = []
    links: list[Any] = []
    diagnostics: list[NonlinearMixedDensityStepDiagnostics] = []
    c_minus = 0.5 - math.sqrt(3.0) / 6.0
    c_plus = 0.5 + math.sqrt(3.0) / 6.0

    for index in range(intervals):
        left_time = times[index]
        right_time = times[index + 1]
        minus_time = left_time + c_minus * step
        plus_time = left_time + c_plus * step
        start_metric = metric_provider(left_time)
        minus_metric = metric_provider(minus_time)
        plus_metric = metric_provider(plus_time)
        target_metric = metric_provider(right_time)
        _validate_matrix_shape(start_metric, dimension, backend, "start endpoint metric")
        _validate_matrix_shape(minus_metric, dimension, backend, "minus-node metric")
        _validate_matrix_shape(plus_metric, dimension, backend, "plus-node metric")
        _validate_matrix_shape(target_metric, dimension, backend, "target endpoint metric")

        # A frozen-generator predictor does not respect a changing metric and
        # can leave the Hermitian density domain before the nonlinear solve
        # begins.  Re-expressing the accepted left contravariant density in
        # each exact node metric is a domain-preserving initial guess only;
        # it neither changes nor projects an accepted trajectory state.
        start_contravariant = mixed_to_contravariant_density(
            current,
            start_metric,
            backend,
        )
        guess_minus = start_contravariant @ minus_metric
        guess_plus = start_contravariant @ plus_metric
        converged = False
        nonlinear_residual = math.inf
        final_minus: EOMTriple | None = None
        final_plus: EOMTriple | None = None
        nonlinear_iterations = 0

        for iteration in range(1, selected.maximum_iterations + 1):
            nonlinear_iterations = iteration
            minus_triple = eom_provider(minus_time, guess_minus)
            plus_triple = eom_provider(plus_time, guess_plus)
            minus_generator = mixed_eom_generator(minus_triple, backend, hbar=hbar)
            plus_generator = mixed_eom_generator(plus_triple, backend, hbar=hbar)
            proposed_minus = similarity_transport(
                current,
                partial_fourth_order_gauss_magnus_link(
                    minus_generator,
                    plus_generator,
                    step,
                    c_minus,
                    backend,
                ),
                backend,
            )
            proposed_plus = similarity_transport(
                current,
                partial_fourth_order_gauss_magnus_link(
                    minus_generator,
                    plus_generator,
                    step,
                    c_plus,
                    backend,
                ),
                backend,
            )
            nonlinear_residual = max(
                relative_frobenius(proposed_minus - guess_minus, proposed_minus, backend),
                relative_frobenius(proposed_plus - guess_plus, proposed_plus, backend),
            )
            guess_minus = proposed_minus
            guess_plus = proposed_plus
            if nonlinear_residual <= selected.tolerance:
                final_minus = eom_provider(minus_time, guess_minus)
                final_plus = eom_provider(plus_time, guess_plus)
                check_minus_generator = mixed_eom_generator(
                    final_minus,
                    backend,
                    hbar=hbar,
                )
                check_plus_generator = mixed_eom_generator(
                    final_plus,
                    backend,
                    hbar=hbar,
                )
                check_minus = similarity_transport(
                    current,
                    partial_fourth_order_gauss_magnus_link(
                        check_minus_generator,
                        check_plus_generator,
                        step,
                        c_minus,
                        backend,
                    ),
                    backend,
                )
                check_plus = similarity_transport(
                    current,
                    partial_fourth_order_gauss_magnus_link(
                        check_minus_generator,
                        check_plus_generator,
                        step,
                        c_plus,
                        backend,
                    ),
                    backend,
                )
                nonlinear_residual = max(
                    relative_frobenius(check_minus - guess_minus, check_minus, backend),
                    relative_frobenius(check_plus - guess_plus, check_plus, backend),
                )
                guess_minus = check_minus
                guess_plus = check_plus
                if nonlinear_residual <= selected.tolerance:
                    converged = True
                    break

        if not converged or final_minus is None or final_plus is None:
            raise PropagationError(
                "nonlinear Gauss-node solve failed at interval "
                f"{index} after {selected.maximum_iterations} iterations; "
                f"residual {nonlinear_residual:.3e}"
            )
        # Re-evaluate on the accepted node states so the endpoint generator
        # is paired with the stored nonlinear node solution.
        final_minus = eom_provider(minus_time, guess_minus)
        final_plus = eom_provider(plus_time, guess_plus)
        link = fourth_order_gauss_magnus_link(
            final_minus,
            final_plus,
            step,
            backend,
            hbar=hbar,
        )
        input_trace = xp.trace(current)
        current = similarity_transport(current, link, backend)
        output_trace = xp.trace(current)
        contravariant = mixed_to_contravariant_density(current, target_metric, backend)
        trace_scale = xp.maximum(xp.asarray(1.0), xp.abs(input_trace))
        spectrum = xp.linalg.eigvals(current)
        spectrum_scale = xp.maximum(xp.asarray(1.0), xp.linalg.norm(initial_spectrum))
        real_spectrum_drift = (
            xp.linalg.norm(xp.sort(xp.real(spectrum)) - xp.sort(xp.real(initial_spectrum)))
            / spectrum_scale
        )
        imaginary_spectrum_drift = (
            xp.maximum(
                xp.max(xp.abs(xp.imag(spectrum))),
                xp.max(xp.abs(xp.imag(initial_spectrum))),
            )
            / spectrum_scale
        )
        spectrum_drift = backend.scalar_to_float(
            xp.maximum(real_spectrum_drift, imaginary_spectrum_drift)
        )
        diagnostics.append(
            NonlinearMixedDensityStepDiagnostics(
                start_time_au=left_time,
                end_time_au=right_time,
                nonlinear_iterations=nonlinear_iterations,
                nonlinear_residual=nonlinear_residual,
                cross_metric_residual=cross_metric_residual(
                    link,
                    start_metric,
                    target_metric,
                    backend,
                ),
                trace_drift=backend.scalar_to_float(
                    xp.abs(output_trace - input_trace) / trace_scale
                ),
                trace_imaginary_abs=backend.scalar_to_float(xp.abs(xp.imag(output_trace))),
                occupation_spectrum_drift=spectrum_drift,
                metric_hermiticity_residual=mixed_metric_hermiticity_residual(
                    current,
                    target_metric,
                    backend,
                ),
                contravariant_hermiticity_residual=relative_frobenius(
                    contravariant - contravariant.conj().T,
                    contravariant,
                    backend,
                ),
            )
        )
        mixed.append(xp.array(current, dtype=xp.complex128, copy=True))
        contravariant_values.append(xp.array(contravariant, dtype=xp.complex128, copy=True))
        minus_nodes.append(xp.array(guess_minus, dtype=xp.complex128, copy=True))
        plus_nodes.append(xp.array(guess_plus, dtype=xp.complex128, copy=True))
        links.append(xp.array(link, dtype=xp.complex128, copy=True))

    return NonlinearMixedDensityPropagation(
        times_au=times,
        mixed_densities=tuple(mixed),
        contravariant_densities=tuple(contravariant_values),
        gauss_minus_mixed_densities=tuple(minus_nodes),
        gauss_plus_mixed_densities=tuple(plus_nodes),
        links=tuple(links),
        diagnostics=tuple(diagnostics),
    )


def propagate_nonlinear_contravariant_density(
    initial_contravariant_density: Any,
    *,
    initial_time_au: float,
    interval_au: float,
    intervals: int,
    metric_provider: Callable[[float], Any],
    eom_provider: Callable[[float, Any], EOMTriple],
    backend: ArrayBackend,
    policy: NonlinearGaussMagnusPolicy | None = None,
    hbar: float = 1.0,
) -> NonlinearContravariantDensityPropagation:
    r"""Propagate ``P`` by self-consistent Magnus-link congruences.

    The coefficient link uses the same two nonlinear Gauss nodes as the
    mixed-similarity algorithm, while the discrete density update is
    ``P_{n+1}=U P_n U^dagger``.  Hermiticity and positivity of the nonlinear
    action argument are therefore structural.  No orthogonalization, metric
    correction, or post-step density projection is applied.  ``D=P S`` is
    reconstructed only to expose particle-number and occupation defects due
    to the finite-step endpoint-metric error.
    """

    if isinstance(intervals, bool) or not isinstance(intervals, int) or intervals <= 0:
        raise PropagationError("interval count must be a positive integer")
    selected = NonlinearGaussMagnusPolicy() if policy is None else policy
    xp = backend.namespace
    propagator = NonlinearContravariantDensityPropagator[EOMTriple](
        initial_contravariant_density=initial_contravariant_density,
        initial_time_au=initial_time_au,
        interval_au=interval_au,
        metric_provider=metric_provider,
        evaluation_provider=eom_provider,
        eom_extractor=_identity_eom_triple,
        backend=backend,
        policy=selected,
        hbar=hbar,
    )
    times = tuple(
        propagator.initial_time_au + index * propagator.interval_au
        for index in range(intervals + 1)
    )
    contravariant_values = [
        xp.array(
            propagator.current_contravariant_density,
            dtype=xp.complex128,
            copy=True,
        )
    ]
    mixed_values = [
        xp.array(
            propagator.current_mixed_density(),
            dtype=xp.complex128,
            copy=True,
        )
    ]
    minus_nodes: list[Any] = []
    plus_nodes: list[Any] = []
    links: list[Any] = []
    diagnostics: list[NonlinearMixedDensityStepDiagnostics] = []

    for _ in range(intervals):
        result = propagator.step()
        contravariant_values.append(result.contravariant_density)
        mixed_values.append(result.mixed_density)
        minus_nodes.append(result.gauss_minus_contravariant_density)
        plus_nodes.append(result.gauss_plus_contravariant_density)
        links.append(result.link)
        diagnostics.append(result.diagnostics)

    return NonlinearContravariantDensityPropagation(
        times_au=times,
        contravariant_densities=tuple(contravariant_values),
        mixed_densities=tuple(mixed_values),
        gauss_minus_contravariant_densities=tuple(minus_nodes),
        gauss_plus_contravariant_densities=tuple(plus_nodes),
        links=tuple(links),
        diagnostics=tuple(diagnostics),
    )


def _identity_eom_triple(value: EOMTriple) -> EOMTriple:
    return value


def _validate_square(value: Any, backend: ArrayBackend, name: str) -> None:
    backend.assert_resident(value, name=name)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise PropagationError(f"{name} must be a square matrix")
    _assert_finite(value, backend, name)


def _validate_square_pair(
    value: Any,
    other: Any,
    backend: ArrayBackend,
    name: str,
) -> None:
    _validate_square(value, backend, name)
    _validate_matrix_shape(other, value.shape[0], backend, "metric or link")


def _validate_matrix_shape(
    value: Any,
    dimension: int,
    backend: ArrayBackend,
    name: str,
) -> None:
    _validate_square(value, backend, name)
    if value.shape != (dimension, dimension):
        raise PropagationError(f"{name} has an incompatible shape")


def _validate_triple(
    triple: EOMTriple,
    backend: ArrayBackend,
    *,
    expected_dimension: int | None = None,
) -> None:
    if not isinstance(triple, EOMTriple):
        raise TypeError("Gauss-node value must be an EOMTriple")
    dimension = triple.metric.shape[0] if getattr(triple.metric, "ndim", 0) == 2 else -1
    for name, value in (
        ("Gauss-node metric", triple.metric),
        ("Gauss-node mechanical matrix", triple.hamiltonian_eom),
        ("Gauss-node connection", triple.connection),
    ):
        _validate_matrix_shape(value, dimension, backend, name)
    if expected_dimension is not None and dimension != expected_dimension:
        raise PropagationError("Gauss-node triple has an incompatible shape")


def _assert_finite(value: Any, backend: ArrayBackend, name: str) -> None:
    xp = backend.namespace
    finite = backend.scalar_to_float(xp.asarray(xp.all(xp.isfinite(value)), dtype=xp.float64))
    if not bool(finite):
        raise PropagationError(f"{name} contains non-finite values")
