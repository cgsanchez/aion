"""Reusable construction and in-memory stepping for exact-Wilson dynamics."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aion.config import (
    ExactWilsonActionConfig,
    WilsonGridKind,
    WilsonSimulationConfig,
    WilsonStationaryBranch,
    dumps_config,
)
from aion.electromagnetism import (
    AffineElectromagneticSourceProvider,
    UniformMagneticSourceSample,
    build_affine_electromagnetic_source,
)
from aion.electronic_structure import (
    AOGridKind,
    AOGridPolicy,
    AOPruningKind,
    AOQuadrature,
    ExactWilsonDynamicEvaluation,
    ExactWilsonDynamicSample,
    ExactWilsonStationaryFactory,
    PreparedExactWilsonDynamicSpatialAction,
    PreparedReference,
    RIMetricRankPolicy,
    WilsonStationaryStateData,
    auxiliary_space_fingerprint,
    prepare_ao_quadrature,
    prepare_exact_wilson_dynamic_spatial_action,
    prepare_exact_wilson_stationary_factory,
)
from aion.errors import FormulationError, UnsupportedConfigurationError, WilsonStateError
from aion.observables import (
    ExactWilsonEndpointObservation,
    evaluate_exact_wilson_endpoint_observation,
)
from aion.propagation import (
    NonlinearContravariantDensityPropagator,
    NonlinearContravariantDensityStep,
    NonlinearGaussMagnusPolicy,
)

type _SpatialKey = tuple[object, ...]
type _SampleKey = tuple[object, ...]


def _spatial_key(source: UniformMagneticSourceSample) -> _SpatialKey:
    gauge = source.gauge
    return (
        gauge.field.magnetic_field_au,
        gauge.kind.value,
        gauge.origin_au,
        gauge.landau_axis,
    )


def _sample_key(source: UniformMagneticSourceSample) -> _SampleKey:
    return (
        source.time_au,
        source.field.magnetic_field_au,
        source.magnetic_field_dot_au,
        source.electric_field_origin_au,
        source.origin_au,
        source.gauge_kind.value,
        source.landau_axis,
    )


@dataclass(frozen=True, slots=True)
class WilsonDynamicCacheStatistics:
    spatial_entries: int
    sample_entries: int
    maximum_entries: int
    spatial_hits: int
    spatial_misses: int
    sample_hits: int
    sample_misses: int


@dataclass(slots=True)
class ExactWilsonDynamicCache:
    """Strictly bounded LRU cache of density-independent dynamic action data."""

    factory: ExactWilsonStationaryFactory
    source_provider: AffineElectromagneticSourceProvider
    branch: WilsonStationaryBranch
    maximum_entries: int
    _spatial: OrderedDict[_SpatialKey, PreparedExactWilsonDynamicSpatialAction] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )
    _samples: OrderedDict[_SampleKey, ExactWilsonDynamicSample] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )
    _spatial_hits: int = field(default=0, init=False, repr=False)
    _spatial_misses: int = field(default=0, init=False, repr=False)
    _sample_hits: int = field(default=0, init=False, repr=False)
    _sample_misses: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.factory, ExactWilsonStationaryFactory):
            raise TypeError("factory must be an ExactWilsonStationaryFactory")
        if not isinstance(self.source_provider, AffineElectromagneticSourceProvider):
            raise TypeError("source_provider must satisfy AffineElectromagneticSourceProvider")
        if (
            isinstance(self.maximum_entries, bool)
            or not isinstance(self.maximum_entries, int)
            or self.maximum_entries < 1
        ):
            raise ValueError("maximum_entries must be a positive integer")

    @staticmethod
    def _retain_lru(cache: OrderedDict[Any, Any], key: object, value: object, limit: int) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def sample(self, time_au: float) -> ExactWilsonDynamicSample:
        """Return the exact source/action sample at one arbitrary time."""

        source = self.source_provider.sample(time_au)
        sample_key = _sample_key(source)
        cached_sample = self._samples.get(sample_key)
        if cached_sample is not None:
            self._sample_hits += 1
            self._samples.move_to_end(sample_key)
            return cached_sample
        self._sample_misses += 1

        spatial_key = _spatial_key(source)
        spatial = self._spatial.get(spatial_key)
        if spatial is None:
            self._spatial_misses += 1
            spatial = prepare_exact_wilson_dynamic_spatial_action(
                self.factory,
                source.gauge,
            )
            self._retain_lru(
                self._spatial,
                spatial_key,
                spatial,
                self.maximum_entries,
            )
        else:
            self._spatial_hits += 1
            self._spatial.move_to_end(spatial_key)

        sample = spatial.sample(source, self.branch)
        self._retain_lru(
            self._samples,
            sample_key,
            sample,
            self.maximum_entries,
        )
        return sample

    def evaluate(self, time_au: float, density: object) -> ExactWilsonDynamicEvaluation:
        """Evaluate a fresh nonlinear action for ``density`` at ``time_au``."""

        return self.sample(time_au).evaluate(density)

    @property
    def statistics(self) -> WilsonDynamicCacheStatistics:
        return WilsonDynamicCacheStatistics(
            spatial_entries=len(self._spatial),
            sample_entries=len(self._samples),
            maximum_entries=self.maximum_entries,
            spatial_hits=self._spatial_hits,
            spatial_misses=self._spatial_misses,
            sample_hits=self._sample_hits,
            sample_misses=self._sample_misses,
        )


@dataclass(slots=True)
class BuiltWilsonSimulation:
    """Authenticated exact-Wilson runtime with one mutable accepted density."""

    config: WilsonSimulationConfig
    reference: PreparedReference
    stationary_state: WilsonStationaryStateData
    quadrature: AOQuadrature
    factory: ExactWilsonStationaryFactory
    source_provider: AffineElectromagneticSourceProvider
    dynamic_cache: ExactWilsonDynamicCache
    propagator: NonlinearContravariantDensityPropagator[ExactWilsonDynamicEvaluation]
    original_toml: str

    @property
    def simulation_id(self) -> str:
        return self.config.scientific_id

    @property
    def density(self) -> Any:
        return self.propagator.current_contravariant_density

    @property
    def current_time_au(self) -> float:
        return self.propagator.current_time_au

    @property
    def boundary_index(self) -> int:
        return self.propagator.boundary_index

    def evaluate(
        self, time_au: float | None = None, density: object | None = None
    ) -> ExactWilsonDynamicEvaluation:
        selected_time = self.current_time_au if time_au is None else float(time_au)
        selected_density = self.density if density is None else density
        return self.dynamic_cache.evaluate(selected_time, selected_density)

    def observe_endpoint(
        self,
        *,
        include_energy: bool = False,
        include_identities: bool = False,
    ) -> ExactWilsonEndpointObservation:
        evaluation = self.evaluate()
        return evaluate_exact_wilson_endpoint_observation(
            evaluation,
            self.density,
            include_energy=include_energy,
            include_identities=include_identities,
        )

    def step(self) -> NonlinearContravariantDensityStep[ExactWilsonDynamicEvaluation]:
        return self.propagator.step()

    def restore_boundary(self, density: object, boundary_index: int) -> None:
        """Replace the accepted state from one authenticated restart boundary."""

        grid = self.config.propagation.time_grid
        if (
            isinstance(boundary_index, bool)
            or not isinstance(boundary_index, int)
            or not 0 <= boundary_index <= grid.intervals
        ):
            raise WilsonStateError("restart boundary index lies outside the simulation grid")
        restored = self.quadrature.backend.asarray(
            density,
            dtype=self.quadrature.backend.namespace.complex128,
        )
        expected = self.stationary_state.contravariant_density.shape
        if restored.shape != expected:
            raise WilsonStateError(
                f"restart density has shape {restored.shape}; expected {expected}"
            )
        self.propagator = _build_propagator(
            self.config,
            self.quadrature,
            self.dynamic_cache,
            restored,
            boundary_index=boundary_index,
            initial_occupation_spectrum=self.stationary_state.occupation_spectrum,
        )


def _grid_policy(config: WilsonSimulationConfig) -> AOGridPolicy:
    numerics = config.numerics
    if numerics.grid_kind is WilsonGridKind.REFERENCE:
        return AOGridPolicy.reference()
    assert numerics.grid_level is not None
    return AOGridPolicy(
        kind=AOGridKind.QUALIFICATION,
        level=numerics.grid_level,
        pruning=AOPruningKind(numerics.grid_pruning.value),
    )


def _validate_stationary_link(
    config: WilsonSimulationConfig,
    reference: PreparedReference,
    state: WilsonStationaryStateData,
) -> None:
    if config.reference.fingerprint_sha256 != reference.fingerprint_sha256:
        raise FormulationError("simulation reference fingerprint does not match the reference")
    if state.reference_fingerprint_sha256 != reference.fingerprint_sha256:
        raise WilsonStateError("stationary state does not belong to the supplied reference")
    if config.stationary_state.fingerprint_sha256 != state.fingerprint_sha256:
        raise WilsonStateError("stationary-state content fingerprint does not match its link")
    if state.config.action != config.action:
        raise WilsonStateError("stationary and simulation action configurations disagree")
    if state.config.numerics != config.numerics:
        raise WilsonStateError("stationary and simulation numerical realizations disagree")
    if state.config.source != config.source:
        raise WilsonStateError("stationary and simulation source configurations disagree")
    if state.config.source_time_au != config.propagation.time_grid.start_au:
        raise WilsonStateError("stationary source time is not the simulation start time")
    if state.config.backend.precision != config.backend.precision:
        raise WilsonStateError("stationary and simulation backend precision disagree")


def _build_propagator(
    config: WilsonSimulationConfig,
    quadrature: AOQuadrature,
    dynamic_cache: ExactWilsonDynamicCache,
    initial_density: object,
    *,
    boundary_index: int,
    initial_occupation_spectrum: object,
) -> NonlinearContravariantDensityPropagator[ExactWilsonDynamicEvaluation]:
    grid = config.propagation.time_grid
    return NonlinearContravariantDensityPropagator[ExactWilsonDynamicEvaluation](
        initial_contravariant_density=initial_density,
        initial_time_au=grid.time_at(boundary_index),
        interval_au=grid.step_au,
        metric_provider=lambda time: dynamic_cache.sample(time).one_electron.metric,
        evaluation_provider=dynamic_cache.evaluate,
        eom_extractor=lambda evaluation: evaluation.triple,
        backend=quadrature.backend,
        policy=NonlinearGaussMagnusPolicy(
            tolerance=config.propagation.nonlinear_tolerance,
            maximum_iterations=config.propagation.maximum_iterations,
        ),
        initial_boundary_index=boundary_index,
        initial_occupation_spectrum=quadrature.backend.asarray(
            initial_occupation_spectrum,
            dtype=quadrature.backend.namespace.complex128,
        ),
    )


def build_wilson_simulation(
    config: WilsonSimulationConfig,
    reference: PreparedReference,
    stationary_state: WilsonStationaryStateData,
    *,
    original_toml: str | None = None,
) -> BuiltWilsonSimulation:
    """Build and authenticate one reusable exact-Wilson simulation runtime."""

    if not isinstance(config, WilsonSimulationConfig):
        raise TypeError("config must be WilsonSimulationConfig")
    if not isinstance(reference, PreparedReference):
        raise TypeError("reference must be PreparedReference")
    if not isinstance(stationary_state, WilsonStationaryStateData):
        raise TypeError("stationary_state must be WilsonStationaryStateData")
    if not isinstance(config.action, ExactWilsonActionConfig):
        raise UnsupportedConfigurationError(
            "reduced Wilson production dynamics remain disabled until P2-6 qualification"
        )
    _validate_stationary_link(config, reference, stationary_state)

    quadrature = prepare_ao_quadrature(
        reference,
        config.backend,
        grid_policy=_grid_policy(config),
        block_size=config.numerics.block_size,
        memory_budget_bytes=config.numerics.memory_budget_bytes,
    )
    if quadrature.grid.fingerprint_sha256 != stationary_state.grid_fingerprint_sha256:
        raise WilsonStateError("rebuilt AO grid fingerprint disagrees with stationary state")
    rank_policy = RIMetricRankPolicy(
        relative_threshold=config.numerics.ri_relative_threshold,
        absolute_threshold=config.numerics.ri_absolute_threshold,
        maximum_rank=config.numerics.ri_maximum_rank,
    )
    factory = prepare_exact_wilson_stationary_factory(
        quadrature,
        auxiliary_basis=config.numerics.auxiliary_basis,
        functional=reference.config.electronic_structure.functional,
        rank_policy=rank_policy,
    )
    if (
        auxiliary_space_fingerprint(factory.hartree_evaluator)
        != stationary_state.auxiliary_space_fingerprint_sha256
    ):
        raise WilsonStateError(
            "rebuilt auxiliary-space fingerprint disagrees with stationary state"
        )

    source_provider = build_affine_electromagnetic_source(
        config.source,
        reference.electromagnetic_origin_au,
    )
    if source_provider.fingerprint_sha256 != stationary_state.source_fingerprint_sha256:
        raise WilsonStateError("rebuilt source fingerprint disagrees with stationary state")
    initial_source = source_provider.sample(config.propagation.time_grid.start_au)
    if initial_source != stationary_state.source_sample:
        raise WilsonStateError("simulation source at its start disagrees with stationary state")

    dynamic_cache = ExactWilsonDynamicCache(
        factory=factory,
        source_provider=source_provider,
        branch=config.action.branch,
        maximum_entries=config.numerics.dynamic_cache_entries,
    )
    initial_sample = dynamic_cache.sample(config.propagation.time_grid.start_au)
    rebuilt_metric = quadrature.backend.to_host(initial_sample.one_electron.metric)
    metric_residual = float(
        np.linalg.norm(rebuilt_metric - stationary_state.metric)
        / max(1.0, float(np.linalg.norm(stationary_state.metric)))
    )
    if metric_residual > 1.0e-11:
        raise WilsonStateError(
            f"rebuilt initial metric disagrees with stationary state ({metric_residual:.3e})"
        )

    initial_density = quadrature.backend.asarray(
        stationary_state.contravariant_density,
        dtype=quadrature.backend.namespace.complex128,
    )
    propagator = _build_propagator(
        config,
        quadrature,
        dynamic_cache,
        initial_density,
        boundary_index=0,
        initial_occupation_spectrum=stationary_state.occupation_spectrum,
    )
    quadrature.backend.assert_resident(initial_density, name="initial Wilson density")
    return BuiltWilsonSimulation(
        config=config,
        reference=reference,
        stationary_state=stationary_state,
        quadrature=quadrature,
        factory=factory,
        source_provider=source_provider,
        dynamic_cache=dynamic_cache,
        propagator=propagator,
        original_toml=dumps_config(config) if original_toml is None else original_toml,
    )
