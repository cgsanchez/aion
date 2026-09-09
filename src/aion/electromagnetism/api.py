"""Bridges between strict source configuration and expert source objects."""

from __future__ import annotations

import numpy as np

from aion.config import (
    CompiledSourceConfig,
    FixedTimeGrid,
    PythonProviderSourceConfig,
    Sin2VectorPotentialPulseConfig,
    SourceConfig,
    ZeroSourceConfig,
)
from aion.electromagnetism.compiled import CompiledUniformSource, compile_uniform_source
from aion.electromagnetism.sources import (
    Sin2VectorPotentialPulse,
    UniformPotentialSource,
    ZeroUniformSource,
)
from aion.errors import SourceCompilationError


def compile_source_for_reference(
    config: SourceConfig,
    time_grid: FixedTimeGrid,
    reference: object,
) -> CompiledUniformSource:
    """Compile a configured source against one prepared reference topology."""

    from aion.electronic_structure import PreparedReference

    if not isinstance(reference, PreparedReference):
        raise TypeError("reference must be a PreparedReference")
    provider: UniformPotentialSource
    if isinstance(config, ZeroSourceConfig):
        provider = ZeroUniformSource()
    elif isinstance(config, Sin2VectorPotentialPulseConfig):
        provider = Sin2VectorPotentialPulse(config)
    elif isinstance(config, CompiledSourceConfig):
        from aion.electromagnetism.source_io import load_compiled_source

        compiled = load_compiled_source(config.path)
        if compiled.fingerprint_sha256 != config.fingerprint_sha256:
            raise SourceCompilationError("compiled-source configuration fingerprint mismatch")
        if compiled.time_grid != time_grid:
            raise SourceCompilationError("compiled source uses a different fixed time grid")
        if (
            compiled.pair_indices.shape != reference.anchor_topology.pair_indices.shape
            or not (compiled.pair_indices == reference.anchor_topology.pair_indices).all()
        ):
            raise SourceCompilationError("compiled source uses a different reference topology")
        if not np.array_equal(
            compiled.atom_coordinates_au,
            reference.core_operators.nuclei.coordinates_au,
        ):
            raise SourceCompilationError("compiled source uses different nuclear coordinates")
        if not np.array_equal(
            compiled.electromagnetic_origin_au,
            np.asarray(reference.config.molecule.electromagnetic_origin.position_au),
        ):
            raise SourceCompilationError("compiled source uses a different EM origin")
        return compiled
    elif isinstance(config, PythonProviderSourceConfig):
        raise SourceCompilationError(
            "a Python provider object must be explicitly supplied to compile_uniform_source; "
            "configuration alone never imports or executes callbacks"
        )
    else:  # pragma: no cover - exhaustive over the public union
        raise SourceCompilationError("unsupported source configuration")
    return compile_uniform_source(
        provider,
        time_grid,
        reference.core_operators.nuclei.coordinates_au,
        np.asarray(reference.config.molecule.electromagnetic_origin.position_au),
        reference.anchor_topology.pair_indices,
    )
