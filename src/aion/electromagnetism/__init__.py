"""Potential-first physical sources, gauge projection, compilation, and events."""

from aion.electromagnetism.api import compile_source_for_reference
from aion.electromagnetism.compiled import (
    CompiledUniformSource,
    PhysicalSourceSeries,
    ProjectedGaugeSeries,
    compile_uniform_source,
    pulse_aligned_time_grid,
)
from aion.electromagnetism.events import (
    DiscreteKickEvent,
    EventSchedule,
    compile_event_schedule,
)
from aion.electromagnetism.gauge import UniformGauge, UniformGaugeSample
from aion.electromagnetism.source_io import load_compiled_source, save_compiled_source
from aion.electromagnetism.sources import (
    AdditiveUniformSource,
    GatedUniformSource,
    ScalarEnvelope,
    Sin2VectorPotentialPulse,
    UniformPotentialSample,
    UniformPotentialSource,
    ZeroUniformSource,
)

__all__ = [
    "AdditiveUniformSource",
    "CompiledUniformSource",
    "DiscreteKickEvent",
    "EventSchedule",
    "GatedUniformSource",
    "PhysicalSourceSeries",
    "ProjectedGaugeSeries",
    "ScalarEnvelope",
    "Sin2VectorPotentialPulse",
    "UniformGauge",
    "UniformGaugeSample",
    "UniformPotentialSample",
    "UniformPotentialSource",
    "ZeroUniformSource",
    "compile_event_schedule",
    "compile_source_for_reference",
    "compile_uniform_source",
    "load_compiled_source",
    "pulse_aligned_time_grid",
    "save_compiled_source",
]
