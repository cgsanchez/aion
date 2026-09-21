"""Construction boundary for immutable formulation objects."""

from __future__ import annotations

from aion.backends import Workspace
from aion.config import FormulationConfig, FormulationKind
from aion.electronic_structure import PreparedReference
from aion.errors import FormulationError
from aion.formulations.bare import BareLengthGauge, BareVelocityGauge
from aion.formulations.base import FormulationContext
from aion.formulations.covariant import P0, P0E1
from aion.formulations.types import Formulation


def build_formulation(
    config: FormulationConfig,
    reference: PreparedReference,
    workspace: Workspace,
) -> Formulation:
    """Bind one validated formulation to one independent backend workspace."""

    if not isinstance(config, FormulationConfig):
        raise TypeError("config must be FormulationConfig")
    if not isinstance(reference, PreparedReference):
        raise TypeError("reference must be PreparedReference")
    context = FormulationContext.create(reference, workspace)
    if config.kind is FormulationKind.BARE_LENGTH_GAUGE:
        return BareLengthGauge(context)
    if config.kind is FormulationKind.BARE_VELOCITY_GAUGE:
        return BareVelocityGauge(context)
    assert config.gauge is not None
    if config.kind is FormulationKind.P0:
        return P0(context, config.gauge, config.resolved_velocity_fraction)
    if config.kind is FormulationKind.P0_E1:
        return P0E1(context, config.gauge, config.resolved_velocity_fraction)
    raise FormulationError(f"unsupported formulation kind {config.kind!r}")
