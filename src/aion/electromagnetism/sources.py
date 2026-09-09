"""Potential-first uniform electromagnetic source definitions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.optimize import brentq

from aion.config import Sin2VectorPotentialPulseConfig
from aion.errors import SourceCompilationError

PULSE_ALGORITHM_VERSION = "sin2-vector-potential-1.0.0"


def _vector3(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise SourceCompilationError(f"{name} must be a finite Cartesian vector")
    result = np.array(result, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class UniformPotentialSample:
    """Uniform reduced vector potential and two analytic time derivatives."""

    vector_potential_reduced: np.ndarray
    vector_potential_reduced_dot: np.ndarray
    vector_potential_reduced_ddot: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vector_potential_reduced",
            _vector3(self.vector_potential_reduced, "vector_potential_reduced"),
        )
        object.__setattr__(
            self,
            "vector_potential_reduced_dot",
            _vector3(
                self.vector_potential_reduced_dot,
                "vector_potential_reduced_dot",
            ),
        )
        object.__setattr__(
            self,
            "vector_potential_reduced_ddot",
            _vector3(
                self.vector_potential_reduced_ddot,
                "vector_potential_reduced_ddot",
            ),
        )

    @property
    def electric_field(self) -> np.ndarray:
        result = -self.vector_potential_reduced_dot
        result.setflags(write=False)
        return result

    @property
    def electric_field_dot(self) -> np.ndarray:
        result = -self.vector_potential_reduced_ddot
        result.setflags(write=False)
        return result


@runtime_checkable
class UniformPotentialSource(Protocol):
    """Analytically evaluable prescribed source, independent of electronic state."""

    def sample(self, time_au: float) -> UniformPotentialSample: ...

    def scientific_mapping(self) -> dict[str, object]: ...

    @property
    def support_boundaries_au(self) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class ZeroUniformSource:
    @property
    def support_boundaries_au(self) -> tuple[float, ...]:
        return ()

    def sample(self, time_au: float) -> UniformPotentialSample:
        if not math.isfinite(float(time_au)):
            raise SourceCompilationError("source evaluation time must be finite")
        return UniformPotentialSample(np.zeros(3), np.zeros(3), np.zeros(3))

    def scientific_mapping(self) -> dict[str, object]:
        return {"kind": "zero_uniform_source", "version": "1.0.0"}


@dataclass(frozen=True, slots=True)
class Sin2VectorPotentialPulse:
    """Compact pulse normalized by its analytic peak electric field."""

    config: Sin2VectorPotentialPulseConfig
    polarization_unit: np.ndarray = field(init=False, repr=False)
    duration_au: float = field(init=False)
    vector_potential_amplitude_au: float = field(init=False)
    derivative_peak_unscaled: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        polarization = np.asarray(self.config.polarization, dtype=np.float64)
        polarization /= np.linalg.norm(polarization)
        polarization.setflags(write=False)
        duration = 2.0 * math.pi * self.config.cycles / self.config.angular_frequency_au
        derivative_peak = _maximum_absolute_pulse_derivative(
            self.config.angular_frequency_au,
            self.config.cycles,
            self.config.carrier_phase_rad,
            duration,
        )
        if derivative_peak <= 0.0 or not math.isfinite(derivative_peak):
            raise SourceCompilationError("cannot normalize a pulse with zero field derivative")
        amplitude = self.config.peak_electric_field_au / derivative_peak
        object.__setattr__(self, "polarization_unit", polarization)
        object.__setattr__(self, "duration_au", duration)
        object.__setattr__(self, "derivative_peak_unscaled", derivative_peak)
        object.__setattr__(self, "vector_potential_amplitude_au", amplitude)
        if self.config.require_zero_impulse and np.linalg.norm(self.impulse_au) > 1.0e-14:
            raise SourceCompilationError("pulse does not satisfy its configured zero-impulse check")

    @property
    def start_time_au(self) -> float:
        return self.config.start_time_au

    @property
    def end_time_au(self) -> float:
        return self.start_time_au + self.duration_au

    @property
    def support_boundaries_au(self) -> tuple[float, ...]:
        return (self.start_time_au, self.end_time_au)

    @property
    def impulse_au(self) -> np.ndarray:
        # Integral E dt = -[a(end)-a(start)]. Boundary values are clamped exactly.
        return np.zeros(3, dtype=np.float64)

    @property
    def peak_electric_field_au(self) -> float:
        return self.vector_potential_amplitude_au * self.derivative_peak_unscaled

    def sample(self, time_au: float) -> UniformPotentialSample:
        time = float(time_au)
        if not math.isfinite(time):
            raise SourceCompilationError("source evaluation time must be finite")
        if time <= self.start_time_au or time >= self.end_time_au:
            return UniformPotentialSample(np.zeros(3), np.zeros(3), np.zeros(3))
        relative = time - self.start_time_au
        value, derivative = _pulse_scalar_and_derivative(
            relative,
            self.duration_au,
            self.config.angular_frequency_au,
            self.config.carrier_phase_rad,
        )
        second_derivative = _pulse_second_derivative(
            relative,
            self.duration_au,
            self.config.angular_frequency_au,
            self.config.carrier_phase_rad,
        )
        return UniformPotentialSample(
            self.vector_potential_amplitude_au * value * self.polarization_unit,
            self.vector_potential_amplitude_au * derivative * self.polarization_unit,
            self.vector_potential_amplitude_au * second_derivative * self.polarization_unit,
        )

    def scientific_mapping(self) -> dict[str, object]:
        return {
            "kind": "sin2_vector_potential_pulse",
            "algorithm_version": PULSE_ALGORITHM_VERSION,
            "configuration": self.config.as_mapping(),
            "duration_au": self.duration_au,
            "polarization_unit": self.polarization_unit.tolist(),
            "vector_potential_amplitude_au": self.vector_potential_amplitude_au,
        }


def _pulse_scalar_and_derivative(
    relative_time: float,
    duration: float,
    angular_frequency: float,
    phase: float,
) -> tuple[float, float]:
    argument = math.pi * relative_time / duration
    envelope = math.sin(argument) ** 2
    envelope_dot = (math.pi / duration) * math.sin(2.0 * argument)
    carrier_argument = angular_frequency * relative_time + phase
    carrier = math.sin(carrier_argument)
    carrier_dot = angular_frequency * math.cos(carrier_argument)
    return envelope * carrier, envelope_dot * carrier + envelope * carrier_dot


def _pulse_second_derivative(
    relative_time: float,
    duration: float,
    angular_frequency: float,
    phase: float,
) -> float:
    argument = math.pi * relative_time / duration
    envelope = math.sin(argument) ** 2
    envelope_dot = (math.pi / duration) * math.sin(2.0 * argument)
    envelope_ddot = (2.0 * math.pi**2 / duration**2) * math.cos(2.0 * argument)
    carrier_argument = angular_frequency * relative_time + phase
    sine = math.sin(carrier_argument)
    cosine = math.cos(carrier_argument)
    return (
        envelope_ddot * sine
        + 2.0 * envelope_dot * angular_frequency * cosine
        - envelope * angular_frequency**2 * sine
    )


def _maximum_absolute_pulse_derivative(
    angular_frequency: float,
    cycles: int,
    phase: float,
    duration: float,
) -> float:
    # f' is a short trigonometric polynomial. Bracket every stationary point
    # on a deterministic dense mesh, polish with Brent, and evaluate endpoints.
    samples = max(1024, 256 * cycles)
    mesh = np.linspace(0.0, duration, samples + 1, dtype=np.float64)
    second = np.asarray(
        [_pulse_second_derivative(float(t), duration, angular_frequency, phase) for t in mesh]
    )
    candidates = [0.0, duration]
    for left, right, f_left, f_right in zip(
        mesh[:-1], mesh[1:], second[:-1], second[1:], strict=True
    ):
        if f_left == 0.0:
            candidates.append(float(left))
        if f_left * f_right < 0.0:
            candidates.append(
                float(
                    brentq(
                        _pulse_second_derivative,
                        float(left),
                        float(right),
                        args=(duration, angular_frequency, phase),
                        xtol=1.0e-14,
                        rtol=4.0 * np.finfo(np.float64).eps,
                    )
                )
            )
    return max(
        abs(_pulse_scalar_and_derivative(t, duration, angular_frequency, phase)[1])
        for t in candidates
    )


@dataclass(frozen=True, slots=True)
class AdditiveUniformSource:
    """Sum providers before any gauge projection or Wilson dressing."""

    providers: tuple[UniformPotentialSource, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "providers", tuple(self.providers))
        if not self.providers:
            raise SourceCompilationError("an additive source requires at least one provider")
        if not all(isinstance(provider, UniformPotentialSource) for provider in self.providers):
            raise SourceCompilationError(
                "all additive providers must satisfy UniformPotentialSource"
            )

    @property
    def support_boundaries_au(self) -> tuple[float, ...]:
        return tuple(
            sorted(
                {value for provider in self.providers for value in provider.support_boundaries_au}
            )
        )

    def sample(self, time_au: float) -> UniformPotentialSample:
        values = [provider.sample(time_au) for provider in self.providers]
        return UniformPotentialSample(
            sum((value.vector_potential_reduced for value in values), start=np.zeros(3)),
            sum((value.vector_potential_reduced_dot for value in values), start=np.zeros(3)),
            sum((value.vector_potential_reduced_ddot for value in values), start=np.zeros(3)),
        )

    def scientific_mapping(self) -> dict[str, object]:
        return {
            "kind": "additive_uniform_source",
            "version": "1.0.0",
            "providers": [provider.scientific_mapping() for provider in self.providers],
        }


@runtime_checkable
class ScalarEnvelope(Protocol):
    semantic_id: str

    def value(self, time_au: float) -> float: ...

    def derivative(self, time_au: float) -> float: ...

    def second_derivative(self, time_au: float) -> float: ...


@dataclass(frozen=True, slots=True)
class GatedUniformSource:
    """Apply an analytic scalar envelope to potentials with the product rule."""

    source: UniformPotentialSource
    envelope: ScalarEnvelope

    @property
    def support_boundaries_au(self) -> tuple[float, ...]:
        return self.source.support_boundaries_au

    def sample(self, time_au: float) -> UniformPotentialSample:
        sample = self.source.sample(time_au)
        value = float(self.envelope.value(time_au))
        derivative = float(self.envelope.derivative(time_au))
        second_derivative = float(self.envelope.second_derivative(time_au))
        if not all(math.isfinite(item) for item in (value, derivative, second_derivative)):
            raise SourceCompilationError("temporal envelope returned a non-finite value")
        return UniformPotentialSample(
            value * sample.vector_potential_reduced,
            derivative * sample.vector_potential_reduced
            + value * sample.vector_potential_reduced_dot,
            second_derivative * sample.vector_potential_reduced
            + 2.0 * derivative * sample.vector_potential_reduced_dot
            + value * sample.vector_potential_reduced_ddot,
        )

    def scientific_mapping(self) -> dict[str, object]:
        return {
            "kind": "gated_uniform_source",
            "version": "1.0.0",
            "source": self.source.scientific_mapping(),
            "envelope_semantic_id": self.envelope.semantic_id,
        }
