"""Exact, boundary-aligned, restart-idempotent discrete event definitions."""

from __future__ import annotations

from dataclasses import dataclass, field

from aion.config import FixedTimeGrid, KickEventConfig, canonical_sha256
from aion.errors import SourceCompilationError


@dataclass(frozen=True, slots=True)
class DiscreteKickEvent:
    event_id: str
    step: int
    time_au: float
    impulse_au: tuple[float, float, float]
    fingerprint_sha256: str = field(init=False)

    @classmethod
    def from_config(cls, config: KickEventConfig, grid: FixedTimeGrid) -> DiscreteKickEvent:
        if config.step > grid.intervals:
            raise SourceCompilationError(f"kick {config.event_id!r} lies outside the time grid")
        return cls(
            event_id=config.event_id,
            step=config.step,
            time_au=grid.time_at(config.step),
            impulse_au=config.impulse_au,
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fingerprint_sha256",
            canonical_sha256(
                {
                    "schema": "aion.discrete-kick-event",
                    "version": "1.0.0",
                    "event_id": self.event_id,
                    "step": self.step,
                    "time_au": self.time_au,
                    "impulse_au": self.impulse_au,
                }
            ),
        )

    @property
    def idempotency_identifier(self) -> str:
        return f"{self.event_id}:{self.fingerprint_sha256}"


@dataclass(frozen=True, slots=True)
class EventSchedule:
    events: tuple[DiscreteKickEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        identifiers = [event.event_id for event in self.events]
        if len(identifiers) != len(set(identifiers)):
            raise SourceCompilationError("discrete event identifiers must be unique")

    def pending_at(
        self, step: int, applied_idempotency_identifiers: frozenset[str] = frozenset()
    ) -> tuple[DiscreteKickEvent, ...]:
        return tuple(
            event
            for event in self.events
            if event.step == step
            and event.idempotency_identifier not in applied_idempotency_identifiers
        )


def compile_event_schedule(
    configs: tuple[KickEventConfig, ...], grid: FixedTimeGrid
) -> EventSchedule:
    return EventSchedule(tuple(DiscreteKickEvent.from_config(config, grid) for config in configs))
