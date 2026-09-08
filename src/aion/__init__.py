"""Aion 0.2 public API.

Only symbols listed in ``__all__`` are covered by the public API contract.
Numerical implementations are introduced behind these contracts in later
work packages.
"""

from aion._version import __version__
from aion.config import (
    BackendConfig,
    FormulationConfig,
    ReferenceConfig,
    SimulationConfig,
    load_config,
)
from aion.workflows import (
    build_simulation,
    load_reference,
    load_trajectory,
    prepare_reference,
    resume,
    run,
)

__all__ = [
    "BackendConfig",
    "FormulationConfig",
    "ReferenceConfig",
    "SimulationConfig",
    "__version__",
    "build_simulation",
    "load_config",
    "load_reference",
    "load_trajectory",
    "prepare_reference",
    "resume",
    "run",
]
