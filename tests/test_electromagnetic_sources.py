from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest

from aion.backends import NumPyBackend, Workspace
from aion.config import FixedTimeGrid, KickEventConfig, Sin2VectorPotentialPulseConfig
from aion.electromagnetism import (
    AdditiveUniformSource,
    GatedUniformSource,
    Sin2VectorPotentialPulse,
    UniformPotentialSample,
    ZeroUniformSource,
    compile_event_schedule,
    compile_uniform_source,
    load_compiled_source,
    pulse_aligned_time_grid,
    save_compiled_source,
)
from aion.errors import SchemaError, SourceCompilationError

pytestmark = pytest.mark.fast

COORDINATES = np.asarray(((0.2, -0.1, 0.4), (1.1, 0.3, -0.5), (-0.7, 0.8, 0.6)))
PAIRS = np.asarray(((0, 1), (0, 2), (1, 2)), dtype=np.int64)
ORIGIN = np.asarray((0.13, -0.27, 0.09))


def _pulse() -> Sin2VectorPotentialPulse:
    return Sin2VectorPotentialPulse(
        Sin2VectorPotentialPulseConfig(
            peak_electric_field_au=0.025,
            angular_frequency_au=0.489597,
            cycles=5,
            polarization=(1.0, -2.0, 0.5),
            start_time_au=0.0,
            carrier_phase_rad=0.37,
            require_zero_impulse=True,
        )
    )


def test_pulse_duration_endpoints_peak_and_aligned_grid() -> None:
    pulse = _pulse()
    assert pulse.duration_au == pytest.approx(10.0 * math.pi / 0.489597)
    assert np.array_equal(pulse.sample(pulse.start_time_au).vector_potential_reduced, np.zeros(3))
    assert np.array_equal(pulse.sample(pulse.end_time_au).vector_potential_reduced, np.zeros(3))
    assert np.array_equal(pulse.impulse_au, np.zeros(3))
    dense_times = np.linspace(pulse.start_time_au, pulse.end_time_au, 100_001)
    fields = np.asarray(
        [np.linalg.norm(pulse.sample(float(time)).electric_field) for time in dense_times]
    )
    assert fields.max() <= pulse.config.peak_electric_field_au * (1.0 + 2.0e-12)
    assert fields.max() == pytest.approx(pulse.config.peak_electric_field_au, rel=2.0e-8)
    grid = pulse_aligned_time_grid(pulse, 0.05)
    assert grid.step_au <= 0.05
    assert grid.end_au == pytest.approx(pulse.end_time_au, abs=2.0e-14)
    time = 0.371 * pulse.duration_au
    epsilon = 1.0e-5
    field_dot_fd = (
        pulse.sample(time + epsilon).electric_field - pulse.sample(time - epsilon).electric_field
    ) / (2.0 * epsilon)
    assert np.allclose(
        pulse.sample(time).electric_field_dot,
        field_dot_fd,
        rtol=2.0e-8,
        atol=2.0e-10,
    )


def test_compilation_derives_identical_lg_vg_field_and_emf_at_all_samples() -> None:
    pulse = _pulse()
    grid = pulse_aligned_time_grid(pulse, 0.1)
    compiled = compile_uniform_source(pulse, grid, COORDINATES, ORIGIN, PAIRS)
    displacements = COORDINATES[PAIRS[:, 0]] - COORDINATES[PAIRS[:, 1]]
    for physical, length, velocity in (
        (compiled.endpoint, compiled.length_endpoint, compiled.velocity_endpoint),
        (compiled.midpoint, compiled.length_midpoint, compiled.velocity_midpoint),
    ):
        expected = physical.electric_field @ displacements.T
        assert np.array_equal(physical.electric_field, -physical.vector_potential_reduced_dot)
        assert np.array_equal(physical.electric_field_dot, -physical.vector_potential_reduced_ddot)
        assert np.allclose(length.pair_electromotive_potential, expected, atol=2.0e-15)
        assert np.allclose(velocity.pair_electromotive_potential, expected, atol=2.0e-15)
        assert np.array_equal(length.pair_link, np.zeros_like(length.pair_link))
        assert np.array_equal(
            velocity.node_scalar_potential, np.zeros_like(velocity.node_scalar_potential)
        )
        expected_phi = -(COORDINATES[None, :, :] - ORIGIN) @ physical.electric_field[:, :, None]
        assert np.allclose(length.node_scalar_potential, expected_phi[..., 0], atol=2.0e-15)
    assert not compiled.endpoint.electric_field.flags.writeable
    assert not compiled.velocity_midpoint.pair_link.flags.writeable


@dataclass(frozen=True)
class LinearEnvelope:
    semantic_id: str = "test.linear"

    def value(self, time_au: float) -> float:
        return 2.0 + 3.0 * time_au

    def derivative(self, time_au: float) -> float:
        return 3.0

    def second_derivative(self, time_au: float) -> float:
        return 0.0


@dataclass(frozen=True)
class ConstantPotential:
    vector: tuple[float, float, float]

    @property
    def support_boundaries_au(self) -> tuple[float, ...]:
        return ()

    def sample(self, time_au: float) -> UniformPotentialSample:
        del time_au
        return UniformPotentialSample(np.asarray(self.vector), np.zeros(3), np.zeros(3))

    def scientific_mapping(self) -> dict[str, object]:
        return {"kind": "test.constant", "vector": list(self.vector)}


def test_addition_and_temporal_gate_apply_before_projection_with_product_rule() -> None:
    source = AdditiveUniformSource(
        (ConstantPotential((1.0, 0.0, 0.0)), ConstantPotential((0.0, 2.0, 0.0)))
    )
    gated = GatedUniformSource(source, LinearEnvelope())
    sample = gated.sample(0.5)
    assert np.array_equal(sample.vector_potential_reduced, np.asarray((3.5, 7.0, 0.0)))
    assert np.array_equal(sample.vector_potential_reduced_dot, np.asarray((3.0, 6.0, 0.0)))
    assert np.array_equal(sample.electric_field, np.asarray((-3.0, -6.0, 0.0)))
    assert np.array_equal(sample.electric_field_dot, np.zeros(3))


def test_off_grid_pulse_boundary_is_rejected() -> None:
    pulse = _pulse()
    grid = FixedTimeGrid(0.0, 0.11, math.ceil(pulse.duration_au / 0.11))
    with pytest.raises(SourceCompilationError, match="not aligned"):
        compile_uniform_source(pulse, grid, COORDINATES, ORIGIN, PAIRS)


def test_stateful_provider_is_rejected_during_precompilation() -> None:
    class StatefulSource:
        def __init__(self) -> None:
            self.calls = 0

        @property
        def support_boundaries_au(self) -> tuple[float, ...]:
            return ()

        def sample(self, time_au: float) -> UniformPotentialSample:
            del time_au
            self.calls += 1
            return UniformPotentialSample(
                np.asarray((float(self.calls), 0.0, 0.0)), np.zeros(3), np.zeros(3)
            )

        def scientific_mapping(self) -> dict[str, object]:
            return {"kind": "test.stateful"}

    with pytest.raises(SourceCompilationError, match="stateful or non-deterministic"):
        compile_uniform_source(
            StatefulSource(),
            FixedTimeGrid(0.0, 0.1, 2),
            COORDINATES,
            ORIGIN,
            PAIRS,
        )


def test_compiled_source_round_trip_authentication_and_workspace_residency(
    tmp_path: Path,
) -> None:
    grid = FixedTimeGrid(0.0, 0.1, 4)
    compiled = compile_uniform_source(ZeroUniformSource(), grid, COORDINATES, ORIGIN, PAIRS)
    path = tmp_path / "source.h5"
    save_compiled_source(compiled, path)
    loaded = load_compiled_source(path)
    assert loaded.fingerprint_sha256 == compiled.fingerprint_sha256
    workspace = Workspace(NumPyBackend())
    loaded.install(workspace)
    workspace.assert_all_resident()
    with pytest.raises(SchemaError, match="refusing to overwrite"):
        save_compiled_source(compiled, path)
    with h5py.File(path, "r+") as handle:
        handle["source/endpoint/physical/electric_field"][0, 0] = 1.0
    with pytest.raises(SourceCompilationError, match="fingerprint mismatch"):
        load_compiled_source(path)


def test_events_are_exactly_aligned_and_restart_idempotent() -> None:
    grid = FixedTimeGrid(0.25, 0.1, 10)
    configs = (
        KickEventConfig("kick-x", 0, (0.01, 0.0, 0.0)),
        KickEventConfig("kick-z", 4, (0.0, 0.0, -0.02)),
    )
    schedule = compile_event_schedule(configs, grid)
    kick = schedule.events[1]
    assert kick.time_au == grid.time_at(4)
    assert schedule.pending_at(4) == (kick,)
    assert schedule.pending_at(4, frozenset((kick.idempotency_identifier,))) == ()
    assert kick.idempotency_identifier == schedule.events[1].idempotency_identifier
