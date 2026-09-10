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
from aion.electromagnetism.magnetic import (
    TESLA_PER_ATOMIC_UNIT_MAGNETIC_FIELD,
    AffineMagneticGauge,
    MagneticGaugeKind,
    MagneticPairGeometry,
    TriangleFactors,
    UniformMagneticField,
    affine_gauge_difference_potential,
    anchored_vectors,
    build_magnetic_pair_geometry,
    endpoint_line_integrals,
    endpoint_links,
    triangle_factors,
    triangle_fluxes,
    triangle_phases,
)
from aion.electromagnetism.runtime import RuntimeSourceController
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
    "TESLA_PER_ATOMIC_UNIT_MAGNETIC_FIELD",
    "AdditiveUniformSource",
    "AffineMagneticGauge",
    "CompiledUniformSource",
    "DiscreteKickEvent",
    "EventSchedule",
    "GatedUniformSource",
    "MagneticGaugeKind",
    "MagneticPairGeometry",
    "PhysicalSourceSeries",
    "ProjectedGaugeSeries",
    "RuntimeSourceController",
    "ScalarEnvelope",
    "Sin2VectorPotentialPulse",
    "TriangleFactors",
    "UniformGauge",
    "UniformGaugeSample",
    "UniformMagneticField",
    "UniformPotentialSample",
    "UniformPotentialSource",
    "ZeroUniformSource",
    "affine_gauge_difference_potential",
    "anchored_vectors",
    "build_magnetic_pair_geometry",
    "compile_event_schedule",
    "compile_source_for_reference",
    "compile_uniform_source",
    "endpoint_line_integrals",
    "endpoint_links",
    "load_compiled_source",
    "pulse_aligned_time_grid",
    "save_compiled_source",
    "triangle_factors",
    "triangle_fluxes",
    "triangle_phases",
]
