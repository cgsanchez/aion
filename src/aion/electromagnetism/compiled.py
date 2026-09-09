"""Deterministic endpoint/midpoint compilation of prescribed uniform sources."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aion.backends import Workspace
from aion.config import FixedTimeGrid, Sin2VectorPotentialPulseConfig, canonical_sha256
from aion.electromagnetism.gauge import UniformGauge, derive_uniform_gauge_sample
from aion.electromagnetism.sources import (
    Sin2VectorPotentialPulse,
    UniformPotentialSample,
    UniformPotentialSource,
)
from aion.errors import SourceCompilationError

COMPILED_SOURCE_SCHEMA = "aion.compiled-uniform-source"
COMPILED_SOURCE_VERSION = "1.0.0"


def _immutable(value: object, dtype: Any, name: str) -> np.ndarray:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    if array.dtype.kind in "iufc" and not np.all(np.isfinite(array)):
        raise SourceCompilationError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return np.asarray(array)


def _normalized_definition(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _normalized_definition(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise SourceCompilationError("source definition mappings require string keys")
        return {key: _normalized_definition(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalized_definition(item) for item in value]
    if value is None or isinstance(value, str | bool | int | float):
        if isinstance(value, float) and not math.isfinite(value):
            raise SourceCompilationError("source definition contains a non-finite value")
        return value
    raise SourceCompilationError(
        f"source definition contains unsupported value {type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class PhysicalSourceSeries:
    times_au: np.ndarray
    electric_field: np.ndarray
    vector_potential_reduced: np.ndarray
    vector_potential_reduced_dot: np.ndarray

    def __post_init__(self) -> None:
        times = _immutable(self.times_au, np.float64, "source times")
        field = _immutable(self.electric_field, np.float64, "electric field")
        vector = _immutable(self.vector_potential_reduced, np.float64, "reduced vector potential")
        vector_dot = _immutable(
            self.vector_potential_reduced_dot,
            np.float64,
            "reduced vector-potential derivative",
        )
        expected = (times.size, 3)
        if (
            times.ndim != 1
            or field.shape != expected
            or vector.shape != expected
            or vector_dot.shape != expected
        ):
            raise SourceCompilationError("physical source-series shapes are inconsistent")
        object.__setattr__(self, "times_au", times)
        object.__setattr__(self, "electric_field", field)
        object.__setattr__(self, "vector_potential_reduced", vector)
        object.__setattr__(self, "vector_potential_reduced_dot", vector_dot)


@dataclass(frozen=True, slots=True)
class ProjectedGaugeSeries:
    node_scalar_potential: np.ndarray
    pair_link: np.ndarray
    pair_link_dot: np.ndarray
    pair_electromotive_potential: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "node_scalar_potential",
            "pair_link",
            "pair_link_dot",
            "pair_electromotive_potential",
        ):
            value = _immutable(getattr(self, name), np.float64, name.replace("_", " "))
            if value.ndim != 2:
                raise SourceCompilationError(f"{name} must be a two-dimensional time series")
            object.__setattr__(self, name, value)
        count = self.node_scalar_potential.shape[0]
        if any(
            value.shape[0] != count
            for value in (
                self.pair_link,
                self.pair_link_dot,
                self.pair_electromotive_potential,
            )
        ):
            raise SourceCompilationError("projected gauge series have inconsistent time counts")


@dataclass(frozen=True, slots=True)
class CompiledUniformSource:
    """Authoritative immutable source arrays for all fixed-grid sample locations."""

    time_grid: FixedTimeGrid
    atom_coordinates_au: np.ndarray
    electromagnetic_origin_au: np.ndarray
    pair_indices: np.ndarray
    endpoint: PhysicalSourceSeries
    midpoint: PhysicalSourceSeries
    length_endpoint: ProjectedGaugeSeries
    length_midpoint: ProjectedGaugeSeries
    velocity_endpoint: ProjectedGaugeSeries
    velocity_midpoint: ProjectedGaugeSeries
    definition_json: str
    fingerprint_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            parsed_definition = json.loads(self.definition_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SourceCompilationError("compiled source definition is not valid JSON") from exc
        if not isinstance(parsed_definition, dict):
            raise SourceCompilationError("compiled source definition JSON must contain an object")
        canonical_definition = json.dumps(
            parsed_definition, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if canonical_definition != self.definition_json:
            raise SourceCompilationError("compiled source definition JSON is not canonical")
        coordinates = _immutable(self.atom_coordinates_au, np.float64, "source atom coordinates")
        origin = _immutable(self.electromagnetic_origin_au, np.float64, "source EM origin")
        pairs = _immutable(self.pair_indices, np.int64, "source pair indices")
        if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
            raise SourceCompilationError("source atom coordinates must have shape (natom, 3)")
        if origin.shape != (3,) or pairs.ndim != 2 or pairs.shape[1:] != (2,):
            raise SourceCompilationError("source origin/pair topology has an invalid shape")
        if self.endpoint.times_au.size != self.time_grid.state_count:
            raise SourceCompilationError("endpoint series does not match the fixed time grid")
        if self.midpoint.times_au.size != self.time_grid.intervals:
            raise SourceCompilationError("midpoint series does not match the fixed time grid")
        expected_endpoint_times, expected_midpoint_times = _times(self.time_grid)
        if not np.array_equal(self.endpoint.times_au, expected_endpoint_times):
            raise SourceCompilationError("endpoint times do not equal the authoritative fixed grid")
        if not np.array_equal(self.midpoint.times_au, expected_midpoint_times):
            raise SourceCompilationError("midpoint times do not equal the authoritative fixed grid")
        natom = coordinates.shape[0]
        npair = pairs.shape[0]
        for location, physical, length, velocity in (
            ("endpoint", self.endpoint, self.length_endpoint, self.velocity_endpoint),
            ("midpoint", self.midpoint, self.length_midpoint, self.velocity_midpoint),
        ):
            count = physical.times_au.size
            for gauge_name, gauge in (("length", length), ("velocity", velocity)):
                if gauge.node_scalar_potential.shape != (count, natom):
                    raise SourceCompilationError(
                        f"{location} {gauge_name} node series has the wrong shape"
                    )
                for value in (
                    gauge.pair_link,
                    gauge.pair_link_dot,
                    gauge.pair_electromotive_potential,
                ):
                    if value.shape != (count, npair):
                        raise SourceCompilationError(
                            f"{location} {gauge_name} pair series has the wrong shape"
                        )
        object.__setattr__(self, "atom_coordinates_au", coordinates)
        object.__setattr__(self, "electromagnetic_origin_au", origin)
        object.__setattr__(self, "pair_indices", pairs)
        fingerprint = canonical_sha256(
            {
                "schema": COMPILED_SOURCE_SCHEMA,
                "version": COMPILED_SOURCE_VERSION,
                "time_grid": {
                    "start_au": self.time_grid.start_au,
                    "step_au": self.time_grid.step_au,
                    "intervals": self.time_grid.intervals,
                },
                "atom_coordinates_au": coordinates,
                "electromagnetic_origin_au": origin,
                "pair_indices": pairs,
                "definition_json": self.definition_json,
                "endpoint": _physical_mapping(self.endpoint),
                "midpoint": _physical_mapping(self.midpoint),
                "length_endpoint": _gauge_mapping(self.length_endpoint),
                "length_midpoint": _gauge_mapping(self.length_midpoint),
                "velocity_endpoint": _gauge_mapping(self.velocity_endpoint),
                "velocity_midpoint": _gauge_mapping(self.velocity_midpoint),
            }
        )
        object.__setattr__(self, "fingerprint_sha256", fingerprint)

    def install(self, workspace: Workspace, *, prefix: str = "source") -> None:
        """Transfer all compiled arrays once to an existing backend workspace."""

        arrays: dict[str, np.ndarray] = {
            "endpoint.times_au": self.endpoint.times_au,
            "endpoint.electric_field": self.endpoint.electric_field,
            "endpoint.vector_potential_reduced": self.endpoint.vector_potential_reduced,
            "endpoint.vector_potential_reduced_dot": self.endpoint.vector_potential_reduced_dot,
            "midpoint.times_au": self.midpoint.times_au,
            "midpoint.electric_field": self.midpoint.electric_field,
            "midpoint.vector_potential_reduced": self.midpoint.vector_potential_reduced,
            "midpoint.vector_potential_reduced_dot": self.midpoint.vector_potential_reduced_dot,
        }
        for gauge_name, series in (
            ("length_endpoint", self.length_endpoint),
            ("length_midpoint", self.length_midpoint),
            ("velocity_endpoint", self.velocity_endpoint),
            ("velocity_midpoint", self.velocity_midpoint),
        ):
            arrays[f"{gauge_name}.node_scalar_potential"] = series.node_scalar_potential
            arrays[f"{gauge_name}.pair_link"] = series.pair_link
            arrays[f"{gauge_name}.pair_link_dot"] = series.pair_link_dot
            arrays[f"{gauge_name}.pair_electromotive_potential"] = (
                series.pair_electromotive_potential
            )
        for name, value in arrays.items():
            workspace.install_host_array(f"{prefix}.{name}", value)
        workspace.caches[f"{prefix}.fingerprint_sha256"] = self.fingerprint_sha256


def _physical_mapping(series: PhysicalSourceSeries) -> dict[str, np.ndarray]:
    return {
        "times_au": series.times_au,
        "electric_field": series.electric_field,
        "vector_potential_reduced": series.vector_potential_reduced,
        "vector_potential_reduced_dot": series.vector_potential_reduced_dot,
    }


def _gauge_mapping(series: ProjectedGaugeSeries) -> dict[str, np.ndarray]:
    return {
        "node_scalar_potential": series.node_scalar_potential,
        "pair_link": series.pair_link,
        "pair_link_dot": series.pair_link_dot,
        "pair_electromotive_potential": series.pair_electromotive_potential,
    }


def _times(grid: FixedTimeGrid) -> tuple[np.ndarray, np.ndarray]:
    endpoint_steps = np.arange(grid.state_count, dtype=np.float64)
    interval_steps = np.arange(grid.intervals, dtype=np.float64) + 0.5
    return (
        grid.start_au + endpoint_steps * grid.step_au,
        grid.start_au + interval_steps * grid.step_au,
    )


def _check_boundary_alignment(source: UniformPotentialSource, grid: FixedTimeGrid) -> None:
    tolerance = 64.0 * np.finfo(np.float64).eps
    for boundary in source.support_boundaries_au:
        coordinate = (boundary - grid.start_au) / grid.step_au
        nearest = round(coordinate)
        scale = max(1.0, abs(coordinate))
        if abs(coordinate - nearest) > tolerance * scale:
            raise SourceCompilationError(
                f"source boundary t={boundary:.17g} is not aligned to the fixed time grid"
            )
        if nearest < 0 or nearest > grid.intervals:
            raise SourceCompilationError(
                f"source boundary t={boundary:.17g} lies outside the fixed time grid"
            )


def _evaluate_reproducibly(
    source: UniformPotentialSource, times: np.ndarray
) -> list[UniformPotentialSample]:
    first = [source.sample(float(time)) for time in times]
    second = [source.sample(float(time)) for time in times]
    for index, (left, right) in enumerate(zip(first, second, strict=True)):
        if not np.array_equal(
            left.vector_potential_reduced, right.vector_potential_reduced
        ) or not np.array_equal(
            left.vector_potential_reduced_dot, right.vector_potential_reduced_dot
        ):
            raise SourceCompilationError(
                f"source is stateful or non-deterministic at compiled sample {index}"
            )
    return first


def _physical_series(
    times: np.ndarray, samples: list[UniformPotentialSample]
) -> PhysicalSourceSeries:
    return PhysicalSourceSeries(
        times_au=times,
        electric_field=np.asarray([sample.electric_field for sample in samples]),
        vector_potential_reduced=np.asarray(
            [sample.vector_potential_reduced for sample in samples]
        ),
        vector_potential_reduced_dot=np.asarray(
            [sample.vector_potential_reduced_dot for sample in samples]
        ),
    )


def _projected_series(
    samples: list[UniformPotentialSample],
    gauge: UniformGauge,
    coordinates: np.ndarray,
    origin: np.ndarray,
    pairs: np.ndarray,
) -> ProjectedGaugeSeries:
    projected = [
        derive_uniform_gauge_sample(sample, gauge, coordinates, origin, pairs) for sample in samples
    ]
    return ProjectedGaugeSeries(
        node_scalar_potential=np.asarray([sample.node_scalar_potential for sample in projected]),
        pair_link=np.asarray([sample.pair_link for sample in projected]),
        pair_link_dot=np.asarray([sample.pair_link_dot for sample in projected]),
        pair_electromotive_potential=np.asarray(
            [sample.pair_electromotive_potential for sample in projected]
        ),
    )


def _validate_gauge_identities(compiled: CompiledUniformSource) -> None:
    pairs = compiled.pair_indices
    displacements = (
        compiled.atom_coordinates_au[pairs[:, 0]] - compiled.atom_coordinates_au[pairs[:, 1]]
        if pairs.size
        else np.empty((0, 3))
    )
    for location, physical, length, velocity in (
        ("endpoint", compiled.endpoint, compiled.length_endpoint, compiled.velocity_endpoint),
        ("midpoint", compiled.midpoint, compiled.length_midpoint, compiled.velocity_midpoint),
    ):
        expected_emf = physical.electric_field @ displacements.T
        for gauge_name, gauge in (("length", length), ("velocity", velocity)):
            error = float(
                np.max(np.abs(gauge.pair_electromotive_potential - expected_emf), initial=0.0)
            )
            if error > 2.0e-13:
                raise SourceCompilationError(
                    f"{location} {gauge_name}-gauge EMF identity failed: {error:.3e}"
                )
        derivative_error = float(
            np.max(
                np.abs(physical.electric_field + physical.vector_potential_reduced_dot),
                initial=0.0,
            )
        )
        if derivative_error != 0.0:
            raise SourceCompilationError(f"{location} potential/field identity is inconsistent")


def compile_uniform_source(
    source: UniformPotentialSource,
    time_grid: FixedTimeGrid,
    atom_coordinates_au: np.ndarray,
    electromagnetic_origin_au: np.ndarray,
    pair_indices: np.ndarray,
) -> CompiledUniformSource:
    """Freeze physical and gauge-derived data at every endpoint and midpoint."""

    if not isinstance(source, UniformPotentialSource):
        raise SourceCompilationError("source does not satisfy the uniform potential protocol")
    _check_boundary_alignment(source, time_grid)
    endpoint_times, midpoint_times = _times(time_grid)
    endpoint_samples = _evaluate_reproducibly(source, endpoint_times)
    midpoint_samples = _evaluate_reproducibly(source, midpoint_times)
    coordinates = np.asarray(atom_coordinates_au, dtype=np.float64)
    origin = np.asarray(electromagnetic_origin_au, dtype=np.float64)
    pairs = np.asarray(pair_indices, dtype=np.int64)
    compiled = CompiledUniformSource(
        time_grid=time_grid,
        atom_coordinates_au=coordinates,
        electromagnetic_origin_au=origin,
        pair_indices=pairs,
        endpoint=_physical_series(endpoint_times, endpoint_samples),
        midpoint=_physical_series(midpoint_times, midpoint_samples),
        length_endpoint=_projected_series(
            endpoint_samples, UniformGauge.LENGTH, coordinates, origin, pairs
        ),
        length_midpoint=_projected_series(
            midpoint_samples, UniformGauge.LENGTH, coordinates, origin, pairs
        ),
        velocity_endpoint=_projected_series(
            endpoint_samples, UniformGauge.VELOCITY, coordinates, origin, pairs
        ),
        velocity_midpoint=_projected_series(
            midpoint_samples, UniformGauge.VELOCITY, coordinates, origin, pairs
        ),
        definition_json=_source_definition_json(source),
    )
    _validate_gauge_identities(compiled)
    return compiled


def _source_definition_json(source: UniformPotentialSource) -> str:
    definition = _normalized_definition(source.scientific_mapping())
    if not isinstance(definition, dict):
        raise SourceCompilationError("source scientific_mapping() must return a mapping")
    return json.dumps(definition, sort_keys=True, separators=(",", ":"), allow_nan=False)


def pulse_aligned_time_grid(
    pulse: Sin2VectorPotentialPulse | Sin2VectorPotentialPulseConfig,
    maximum_step_au: float,
    *,
    post_pulse_intervals: int = 0,
) -> FixedTimeGrid:
    """Choose an actual step no larger than requested and align pulse endpoints."""

    source = (
        pulse if isinstance(pulse, Sin2VectorPotentialPulse) else Sin2VectorPotentialPulse(pulse)
    )
    maximum_step = float(maximum_step_au)
    if not math.isfinite(maximum_step) or maximum_step <= 0.0:
        raise SourceCompilationError("maximum_step_au must be finite and positive")
    if (
        isinstance(post_pulse_intervals, bool)
        or not isinstance(post_pulse_intervals, int)
        or post_pulse_intervals < 0
    ):
        raise SourceCompilationError("post_pulse_intervals must be a nonnegative integer")
    pulse_intervals = math.ceil(source.duration_au / maximum_step)
    actual_step = source.duration_au / pulse_intervals
    return FixedTimeGrid(
        start_au=source.start_time_au,
        step_au=actual_step,
        intervals=pulse_intervals + post_pulse_intervals,
    )
