"""Mutable per-run source offsets generated only by exact boundary events."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aion.backends import Workspace
from aion.electromagnetism.events import DiscreteKickEvent, EventSchedule
from aion.errors import SourceCompilationError


@dataclass(slots=True)
class RuntimeSourceController:
    """Apply persistent vector-potential jumps to compiled resident source arrays.

    The immutable :class:`CompiledUniformSource` remains the authenticated base
    source.  This controller records the cumulative event offset and changes only
    the mutable arrays belonging to one simulation workspace.
    """

    workspace: Workspace
    schedule: EventSchedule
    vector_potential_offset_au: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    applied_identifiers: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        offset = np.asarray(self.vector_potential_offset_au, dtype=np.float64)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise SourceCompilationError("runtime vector-potential offset must be a finite vector")
        self.vector_potential_offset_au = np.array(offset, copy=True)

    def apply(self, event: DiscreteKickEvent) -> None:
        """Apply one idempotent post-boundary jump ``Delta a=-impulse``."""

        identifier = event.idempotency_identifier
        if identifier in self.applied_identifiers:
            return
        if not any(
            candidate.idempotency_identifier == identifier for candidate in self.schedule.events
        ):
            raise SourceCompilationError(f"event {event.event_id!r} is not in this simulation")
        delta_host = -np.asarray(event.impulse_au, dtype=np.float64)
        backend = self.workspace.backend
        xp = backend.namespace
        delta = backend.asarray(delta_host, dtype=xp.float64)
        step = event.step
        self.workspace.require("source.endpoint.vector_potential_reduced")[step:] += delta
        if step < self.workspace.require("source.midpoint.vector_potential_reduced").shape[0]:
            self.workspace.require("source.midpoint.vector_potential_reduced")[step:] += delta

        pair_displacements = self.workspace.require("anchors.pair_displacements_au")
        link_delta = pair_displacements @ delta
        self.workspace.require("source.velocity_endpoint.pair_link")[step:] += link_delta
        if step < self.workspace.require("source.velocity_midpoint.pair_link").shape[0]:
            self.workspace.require("source.velocity_midpoint.pair_link")[step:] += link_delta

        self.vector_potential_offset_au += delta_host
        self.applied_identifiers.add(identifier)

    def undo(self, event: DiscreteKickEvent) -> None:
        """Roll back the most recently attempted event before state publication."""

        identifier = event.idempotency_identifier
        if identifier not in self.applied_identifiers:
            raise SourceCompilationError(f"event {event.event_id!r} is not currently applied")
        later = [
            candidate
            for candidate in self.schedule.events
            if candidate.idempotency_identifier in self.applied_identifiers
            and candidate.step > event.step
        ]
        if later:
            raise SourceCompilationError("runtime events may only be undone in reverse order")
        delta_host = np.asarray(event.impulse_au, dtype=np.float64)
        backend = self.workspace.backend
        xp = backend.namespace
        delta = backend.asarray(delta_host, dtype=xp.float64)
        step = event.step
        self.workspace.require("source.endpoint.vector_potential_reduced")[step:] += delta
        if step < self.workspace.require("source.midpoint.vector_potential_reduced").shape[0]:
            self.workspace.require("source.midpoint.vector_potential_reduced")[step:] += delta
        link_delta = self.workspace.require("anchors.pair_displacements_au") @ delta
        self.workspace.require("source.velocity_endpoint.pair_link")[step:] += link_delta
        if step < self.workspace.require("source.velocity_midpoint.pair_link").shape[0]:
            self.workspace.require("source.velocity_midpoint.pair_link")[step:] += link_delta
        self.vector_potential_offset_au += delta_host
        self.applied_identifiers.remove(identifier)

    def restore(self, identifiers: frozenset[str]) -> None:
        """Reconstruct runtime source state from authenticated applied event IDs."""

        known = {event.idempotency_identifier for event in self.schedule.events}
        unknown = identifiers - known
        if unknown:
            raise SourceCompilationError(
                "checkpoint contains unknown applied event identifier(s): "
                + ", ".join(sorted(unknown))
            )
        for event in self.schedule.events:
            if event.idempotency_identifier in identifiers:
                self.apply(event)
