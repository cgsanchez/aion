"""Analytic Maxwell-consistent affine Wilson source providers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from aion.config import (
    AffineElectromagneticSourceConfig,
    Sin2VectorPotentialPulseConfig,
    WilsonMagneticGaugeKind,
    ZeroSourceConfig,
    canonical_sha256,
)
from aion.config.units import Vector3, vector3
from aion.electromagnetism.magnetic import (
    MagneticGaugeKind,
    UniformMagneticField,
    UniformMagneticSourceSample,
)
from aion.electromagnetism.sources import (
    Sin2VectorPotentialPulse,
    UniformPotentialSource,
    ZeroUniformSource,
)
from aion.errors import SourceCompilationError


@runtime_checkable
class AffineElectromagneticSourceProvider(Protocol):
    """Stateless analytic source sampled at arbitrary integration nodes."""

    def sample(self, time_au: float) -> UniformMagneticSourceSample: ...

    def scientific_mapping(self) -> dict[str, object]: ...

    @property
    def fingerprint_sha256(self) -> str: ...


@dataclass(frozen=True, slots=True)
class AnalyticAffineElectromagneticSource:
    """Uniform electric source plus a linearly time-dependent magnetic field."""

    config: AffineElectromagneticSourceConfig
    origin_au: Vector3
    electric_provider: UniformPotentialSource = field(init=False, repr=False)
    fingerprint_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, AffineElectromagneticSourceConfig):
            raise TypeError("config must be an AffineElectromagneticSourceConfig")
        origin = vector3(self.origin_au, "origin_au")
        object.__setattr__(self, "origin_au", origin)
        electric: UniformPotentialSource
        if isinstance(self.config.electric, ZeroSourceConfig):
            electric = ZeroUniformSource()
        elif isinstance(self.config.electric, Sin2VectorPotentialPulseConfig):
            electric = Sin2VectorPotentialPulse(self.config.electric)
        else:  # pragma: no cover - protected by the typed configuration boundary
            raise SourceCompilationError("unsupported affine electric source")
        object.__setattr__(self, "electric_provider", electric)
        self._validate_landau_axis()
        object.__setattr__(
            self,
            "fingerprint_sha256",
            canonical_sha256(self.scientific_mapping()),
        )

    def _validate_landau_axis(self) -> None:
        if self.config.magnetic_gauge is WilsonMagneticGaugeKind.SYMMETRIC:
            return
        assert self.config.landau_axis is not None
        axis = np.asarray(self.config.landau_axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm == 0.0:
            raise SourceCompilationError("Landau axis cannot be zero")
        axis /= norm
        for name, vector in (
            ("magnetic field", self.config.magnetic_field_reference_au),
            ("magnetic-field rate", self.config.magnetic_field_rate_au),
        ):
            values = np.asarray(vector, dtype=np.float64)
            scale = max(1.0, float(np.linalg.norm(values)))
            if abs(float(values @ axis)) > 1.0e-12 * scale:
                raise SourceCompilationError(f"Landau axis must be perpendicular to {name}")

    def scientific_mapping(self) -> dict[str, object]:
        return {
            "schema": "aion.analytic-affine-electromagnetic-source",
            "version": "1.0.0",
            "origin_au": list(self.origin_au),
            "configuration": self.config.as_mapping(),
            "electric_representation": "uniform_scalar_potential_from_potential_first_field",
            "magnetic_time_law": "linear_about_explicit_reference_time",
        }

    def sample(self, time_au: float) -> UniformMagneticSourceSample:
        time = float(time_au)
        if not math.isfinite(time):
            raise SourceCompilationError("source evaluation time must be finite")
        delta = time - self.config.magnetic_reference_time_au
        magnetic = vector3(
            tuple(
                reference + delta * rate
                for reference, rate in zip(
                    self.config.magnetic_field_reference_au,
                    self.config.magnetic_field_rate_au,
                    strict=True,
                )
            ),
            "magnetic_field_au",
        )
        electric = self.electric_provider.sample(time).electric_field
        offset = np.asarray(self.config.electric_field_origin_offset_au, dtype=np.float64)
        electric_origin = vector3(
            tuple(float(value) for value in offset + electric),
            "electric_field_origin_au",
        )
        gauge_kind = MagneticGaugeKind(self.config.magnetic_gauge.value)
        return UniformMagneticSourceSample(
            time_au=time,
            field=UniformMagneticField(magnetic),
            magnetic_field_dot_au=self.config.magnetic_field_rate_au,
            electric_field_origin_au=electric_origin,
            origin_au=self.origin_au,
            gauge_kind=gauge_kind,
            landau_axis=self.config.landau_axis,
        )


def build_affine_electromagnetic_source(
    config: AffineElectromagneticSourceConfig,
    origin_au: object,
) -> AnalyticAffineElectromagneticSource:
    """Build the immutable analytic provider used by Wilson workflows."""

    return AnalyticAffineElectromagneticSource(
        config=config,
        origin_au=vector3(origin_au, "origin_au"),
    )
