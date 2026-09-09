"""Fixed-metric and analytic-connection SCEM transport constructions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from aion.backends import ArrayBackend, Workspace
from aion.config import FormulationKind, GaugeRepresentation, RationalApproximation
from aion.errors import MetricConstraintError, PropagationError
from aion.formulations import (
    Formulation,
    FormulationSourceSample,
    InstantaneousEvaluation,
    SourceSampling,
    e1_tensors,
    p0_geometry,
)
from aion.propagation.linalg import (
    metric_roundoff_limit,
    pull_lower_matrix,
    rational_map,
    relative_frobenius,
    right_cholesky_metric_link,
    uncorrected_link_diagnostics,
)
from aion.propagation.types import LinkDiagnostics


class TransportTarget(StrEnum):
    MIDPOINT = "midpoint"
    ENDPOINT = "endpoint"


@dataclass(frozen=True, slots=True)
class StepSourceSamples:
    """The three source fibers needed by one fixed-grid interval."""

    start: FormulationSourceSample
    midpoint: FormulationSourceSample
    endpoint: FormulationSourceSample
    physical_vector_start: Any
    physical_vector_midpoint: Any
    physical_vector_endpoint: Any

    @classmethod
    def from_workspace(
        cls,
        workspace: Workspace,
        *,
        gauge: GaugeRepresentation,
        step_index: int,
    ) -> StepSourceSamples:
        return cls(
            start=FormulationSourceSample.from_workspace(
                workspace,
                gauge=gauge,
                location=SourceSampling.ENDPOINT,
                index=step_index,
            ),
            midpoint=FormulationSourceSample.from_workspace(
                workspace,
                gauge=gauge,
                location=SourceSampling.MIDPOINT,
                index=step_index,
            ),
            endpoint=FormulationSourceSample.from_workspace(
                workspace,
                gauge=gauge,
                location=SourceSampling.ENDPOINT,
                index=step_index + 1,
            ),
            physical_vector_start=workspace.require("source.endpoint.vector_potential_reduced")[
                step_index
            ],
            physical_vector_midpoint=workspace.require("source.midpoint.vector_potential_reduced")[
                step_index
            ],
            physical_vector_endpoint=workspace.require("source.endpoint.vector_potential_reduced")[
                step_index + 1
            ],
        )


@dataclass(frozen=True, slots=True)
class TransportApplication:
    coefficients: Any
    diagnostics: LinkDiagnostics


class PreparedStepTransport(Protocol):
    start_metric: Any
    midpoint_metric: Any
    endpoint_metric: Any
    metric_correction_diagnostic_mode: bool

    def predictor_coefficients(
        self,
        coefficients: Any,
        *,
        identity: Any,
        out: Any,
    ) -> tuple[Any | None, LinkDiagnostics | None]: ...

    def apply(
        self,
        coefficients: Any,
        evaluation: InstantaneousEvaluation,
        target: TransportTarget,
        *,
        approximation: RationalApproximation,
        identity: Any,
        out: Any,
    ) -> TransportApplication: ...


def _assert_midpoint_metric(
    evaluation: InstantaneousEvaluation,
    expected: Any,
    backend: ArrayBackend,
) -> None:
    residual = relative_frobenius(evaluation.triple.metric - expected, expected, backend)
    if residual > metric_roundoff_limit(expected.shape[0]):
        raise MetricConstraintError(
            f"formulation midpoint metric changed inside SCEM: residual={residual:.3e}"
        )


@dataclass(frozen=True, slots=True)
class PreparedFixedMetricTransport:
    backend: ArrayBackend
    start_metric: Any
    midpoint_metric: Any
    endpoint_metric: Any
    half_interval_au: float
    full_interval_au: float
    hbar: float
    metric_correction_diagnostic_mode: bool = False

    def predictor_coefficients(
        self,
        coefficients: Any,
        *,
        identity: Any,
        out: Any,
    ) -> tuple[None, None]:
        del coefficients, identity, out
        return None, None

    def apply(
        self,
        coefficients: Any,
        evaluation: InstantaneousEvaluation,
        target: TransportTarget,
        *,
        approximation: RationalApproximation,
        identity: Any,
        out: Any,
    ) -> TransportApplication:
        _assert_midpoint_metric(evaluation, self.midpoint_metric, self.backend)
        xp = self.backend.namespace
        generator = xp.linalg.solve(
            self.midpoint_metric,
            -evaluation.triple.connection - (1j / self.hbar) * evaluation.triple.hamiltonian_eom,
        )
        interval = (
            self.half_interval_au if target is TransportTarget.MIDPOINT else self.full_interval_au
        )
        link = rational_map(
            generator,
            interval,
            approximation,
            backend=self.backend,
            identity=identity,
        )
        diagnostics = uncorrected_link_diagnostics(link, self.start_metric, self.backend)
        out[...] = link @ coefficients
        return TransportApplication(out, diagnostics)


@dataclass(frozen=True, slots=True)
class FixedMetricTransport:
    formulation: Formulation
    interval_au: float

    def prepare(self, samples: StepSourceSamples) -> PreparedFixedMetricTransport:
        del samples
        context = self.formulation.context
        xp = context.namespace
        metric = xp.asarray(context.workspace.require("operators.overlap"), dtype=xp.complex128)
        return PreparedFixedMetricTransport(
            backend=context.backend,
            start_metric=metric,
            midpoint_metric=metric,
            endpoint_metric=metric,
            half_interval_au=0.5 * self.interval_au,
            full_interval_au=self.interval_au,
            hbar=context.hbar,
        )


@dataclass(frozen=True, slots=True)
class _CovariantKinematics:
    metric: Any
    connection: Any
    sigma_diagonal: Any


def _covariant_kinematics(
    formulation: Formulation,
    source: FormulationSourceSample,
) -> _CovariantKinematics:
    context = formulation.context
    geometry = p0_geometry(
        context.workspace.require("operators.overlap"),
        context.workspace.require("anchors.ao_to_atom"),
        context.pairs,
        source.node_scalar_potential,
        source.pair_link,
        source.pair_link_dot,
        natom=context.natom,
        charge=context.charge,
        hbar=context.hbar,
        backend=context.backend,
    )
    connection = geometry.connection
    if formulation.kind is FormulationKind.P0_E1:
        e1 = e1_tensors(
            context.central_dipoles(),
            geometry.theta,
            geometry.theta_dot,
            context.workspace.require("anchors.ao_to_atom"),
            context.workspace.require("nuclei.coordinates_au"),
            source.electric_field,
            charge=context.charge,
            hbar=context.hbar,
            backend=context.backend,
        )
        connection = connection + e1.connection
    return _CovariantKinematics(geometry.metric, connection, geometry.sigma_diagonal)


def _site_transport(
    formulation: Formulation,
    start_vector: Any,
    target_vector: Any,
) -> Any:
    """Exact Abelian site transport for the compiled uniform gauge family."""

    context = formulation.context
    xp = context.namespace
    if formulation.gauge is GaugeRepresentation.VELOCITY:
        return xp.ones(
            (context.workspace.require("operators.overlap").shape[0],),
            dtype=xp.complex128,
        )
    if formulation.gauge is not GaugeRepresentation.LENGTH:
        raise PropagationError(f"unsupported site-transport gauge {formulation.gauge!r}")
    coordinates = context.workspace.require("nuclei.coordinates_au")
    ao_to_atom = context.workspace.require("anchors.ao_to_atom")
    origin = xp.asarray(
        context.reference.config.molecule.electromagnetic_origin.position_au,
        dtype=xp.float64,
    )
    relative = coordinates - origin[None, :]
    phase_integral = relative @ (target_vector - start_vector)
    atom_transport = xp.exp((-1j * context.charge / context.hbar) * phase_integral)
    return atom_transport[ao_to_atom]


@dataclass(frozen=True, slots=True)
class PreparedConnectionAwareTransport:
    formulation: Formulation
    backend: ArrayBackend
    start_metric: Any
    midpoint_metric: Any
    endpoint_metric: Any
    midpoint_metric_parallel: Any
    endpoint_metric_parallel: Any
    midpoint_transport: Any
    endpoint_transport: Any
    midpoint_connection: Any
    midpoint_sigma: Any
    half_interval_au: float
    full_interval_au: float
    hbar: float
    apply_metric_correction: bool

    @property
    def metric_correction_diagnostic_mode(self) -> bool:
        return not self.apply_metric_correction

    def _residual_connection(self, connection: Any) -> Any:
        site = self.midpoint_metric * self.midpoint_sigma[None, :]
        return pull_lower_matrix(connection - site, self.midpoint_transport)

    def _correct(
        self,
        raw: Any,
        target_metric_parallel: Any,
    ) -> tuple[Any, LinkDiagnostics]:
        return right_cholesky_metric_link(
            raw,
            self.start_metric,
            target_metric_parallel,
            backend=self.backend,
            apply_correction=self.apply_metric_correction,
        )

    def predictor_coefficients(
        self,
        coefficients: Any,
        *,
        identity: Any,
        out: Any,
    ) -> tuple[Any, LinkDiagnostics]:
        xp = self.backend.namespace
        residual = self._residual_connection(self.midpoint_connection)
        generator = xp.linalg.solve(self.midpoint_metric_parallel, -residual)
        raw = rational_map(
            generator,
            self.half_interval_au,
            RationalApproximation.CAYLEY_11,
            backend=self.backend,
            identity=identity,
        )
        link, diagnostics = self._correct(raw, self.midpoint_metric_parallel)
        out[...] = self.midpoint_transport[:, None] * (link @ coefficients)
        return out, diagnostics

    def apply(
        self,
        coefficients: Any,
        evaluation: InstantaneousEvaluation,
        target: TransportTarget,
        *,
        approximation: RationalApproximation,
        identity: Any,
        out: Any,
    ) -> TransportApplication:
        _assert_midpoint_metric(evaluation, self.midpoint_metric, self.backend)
        connection_residual = self._residual_connection(evaluation.triple.connection)
        hamiltonian_parallel = pull_lower_matrix(
            evaluation.triple.hamiltonian_eom, self.midpoint_transport
        )
        xp = self.backend.namespace
        generator = xp.linalg.solve(
            self.midpoint_metric_parallel,
            -connection_residual - (1j / self.hbar) * hamiltonian_parallel,
        )
        if target is TransportTarget.MIDPOINT:
            interval = self.half_interval_au
            target_metric = self.midpoint_metric_parallel
            site_transport = self.midpoint_transport
        else:
            interval = self.full_interval_au
            target_metric = self.endpoint_metric_parallel
            site_transport = self.endpoint_transport
        raw = rational_map(
            generator,
            interval,
            approximation,
            backend=self.backend,
            identity=identity,
        )
        link, diagnostics = self._correct(raw, target_metric)
        out[...] = site_transport[:, None] * (link @ coefficients)
        return TransportApplication(out, diagnostics)


@dataclass(frozen=True, slots=True)
class ConnectionAwareTransport:
    formulation: Formulation
    interval_au: float
    apply_metric_correction: bool = True

    def prepare(self, samples: StepSourceSamples) -> PreparedConnectionAwareTransport:
        start = _covariant_kinematics(self.formulation, samples.start)
        midpoint = _covariant_kinematics(self.formulation, samples.midpoint)
        endpoint = _covariant_kinematics(self.formulation, samples.endpoint)
        site_mid = _site_transport(
            self.formulation,
            samples.physical_vector_start,
            samples.physical_vector_midpoint,
        )
        site_end = _site_transport(
            self.formulation,
            samples.physical_vector_start,
            samples.physical_vector_endpoint,
        )
        midpoint_parallel = pull_lower_matrix(midpoint.metric, site_mid)
        endpoint_parallel = pull_lower_matrix(endpoint.metric, site_end)
        return PreparedConnectionAwareTransport(
            formulation=self.formulation,
            backend=self.formulation.context.backend,
            start_metric=start.metric,
            midpoint_metric=midpoint.metric,
            endpoint_metric=endpoint.metric,
            midpoint_metric_parallel=midpoint_parallel,
            endpoint_metric_parallel=endpoint_parallel,
            midpoint_transport=site_mid,
            endpoint_transport=site_end,
            midpoint_connection=midpoint.connection,
            midpoint_sigma=midpoint.sigma_diagonal,
            half_interval_au=0.5 * self.interval_au,
            full_interval_au=self.interval_au,
            hbar=self.formulation.context.hbar,
            apply_metric_correction=self.apply_metric_correction,
        )
