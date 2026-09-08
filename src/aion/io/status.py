"""Strict data model for the non-authoritative detached-run ``status.json``."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from aion.config.units import finite_float
from aion.errors import SchemaError
from aion.io.versions import SchemaVersion

STATUS_SCHEMA = "aion.status"
STATUS_SCHEMA_VERSION = SchemaVersion(1, 0, 0)


class RunPhase(StrEnum):
    INITIALIZING = "initializing"
    PREPARING = "preparing"
    PROPAGATING = "propagating"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class FailureSummary:
    error_type: str
    message: str
    phase: RunPhase
    step: int | None = None

    def __post_init__(self) -> None:
        if not self.error_type or not self.message:
            raise SchemaError("status failure type and message cannot be empty")
        if self.step is not None and self.step < 0:
            raise SchemaError("status failure step cannot be negative")

    def as_mapping(self) -> dict[str, object]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "phase": self.phase.value,
            "step": self.step,
        }


@dataclass(frozen=True, slots=True)
class RunStatus:
    run_id: str
    simulation_id: str
    phase: RunPhase
    global_step: int
    last_accepted_step: int
    total_steps: int
    wall_time_seconds: float
    updated_at_utc: str
    host: str
    pid: int
    latest_checkpoint: str | None = None
    eta_seconds: float | None = None
    failure: FailureSummary | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "simulation_id", "updated_at_utc", "host"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise SchemaError(f"status.{name} must be nonempty text")
        if len(self.simulation_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.simulation_id
        ):
            raise SchemaError("status.simulation_id must be a lowercase SHA-256 digest")
        for name in ("global_step", "last_accepted_step", "total_steps", "pid"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchemaError(f"status.{name} must be a nonnegative integer")
        if self.global_step > self.total_steps:
            raise SchemaError("status.global_step cannot exceed status.total_steps")
        if self.last_accepted_step > self.global_step:
            raise SchemaError("status.last_accepted_step cannot exceed status.global_step")
        wall_time = finite_float(self.wall_time_seconds, "status.wall_time_seconds")
        if wall_time < 0.0:
            raise SchemaError("status.wall_time_seconds cannot be negative")
        object.__setattr__(self, "wall_time_seconds", wall_time)
        if self.eta_seconds is not None:
            eta = finite_float(self.eta_seconds, "status.eta_seconds")
            if eta < 0.0:
                raise SchemaError("status.eta_seconds cannot be negative")
            object.__setattr__(self, "eta_seconds", eta)
        if self.phase is RunPhase.FAILED and self.failure is None:
            raise SchemaError("failed status requires a failure summary")
        if self.phase is not RunPhase.FAILED and self.failure is not None:
            raise SchemaError("a failure summary is valid only for failed status")

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema": STATUS_SCHEMA,
            "schema_version": str(STATUS_SCHEMA_VERSION),
            "run_id": self.run_id,
            "simulation_id": self.simulation_id,
            "phase": self.phase.value,
            "global_step": self.global_step,
            "last_accepted_step": self.last_accepted_step,
            "total_steps": self.total_steps,
            "latest_checkpoint": self.latest_checkpoint,
            "wall_time_seconds": self.wall_time_seconds,
            "eta_seconds": self.eta_seconds,
            "updated_at_utc": self.updated_at_utc,
            "host": self.host,
            "pid": self.pid,
            "failure": self.failure.as_mapping() if self.failure is not None else None,
        }


def dumps_status(status: RunStatus) -> str:
    return json.dumps(status.as_mapping(), sort_keys=True, separators=(",", ":")) + "\n"


def loads_status(text: str) -> RunStatus:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid status JSON: {exc}") from exc
    data = _mapping(raw, "status")
    expected = {
        "schema",
        "schema_version",
        "run_id",
        "simulation_id",
        "phase",
        "global_step",
        "last_accepted_step",
        "total_steps",
        "latest_checkpoint",
        "wall_time_seconds",
        "eta_seconds",
        "updated_at_utc",
        "host",
        "pid",
        "failure",
    }
    unknown = set(data) - expected
    missing = expected - set(data)
    if unknown or missing:
        raise SchemaError(
            "status fields differ from schema; "
            f"unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    if data["schema"] != STATUS_SCHEMA:
        raise SchemaError(f"status schema must be {STATUS_SCHEMA!r}")
    artifact_version = SchemaVersion.parse(_text(data["schema_version"], "schema_version"))
    if not STATUS_SCHEMA_VERSION.can_read(artifact_version):
        raise SchemaError(
            f"incompatible status schema {artifact_version}; reader is {STATUS_SCHEMA_VERSION}"
        )
    failure = _parse_failure(data["failure"])
    try:
        phase = RunPhase(_text(data["phase"], "phase"))
    except ValueError as exc:
        raise SchemaError(f"unknown status phase {data['phase']!r}") from exc
    return RunStatus(
        run_id=_text(data["run_id"], "run_id"),
        simulation_id=_text(data["simulation_id"], "simulation_id"),
        phase=phase,
        global_step=_integer(data["global_step"], "global_step"),
        last_accepted_step=_integer(data["last_accepted_step"], "last_accepted_step"),
        total_steps=_integer(data["total_steps"], "total_steps"),
        latest_checkpoint=_optional_text(data["latest_checkpoint"], "latest_checkpoint"),
        wall_time_seconds=_number(data["wall_time_seconds"], "wall_time_seconds"),
        eta_seconds=_optional_number(data["eta_seconds"], "eta_seconds"),
        updated_at_utc=_text(data["updated_at_utc"], "updated_at_utc"),
        host=_text(data["host"], "host"),
        pid=_integer(data["pid"], "pid"),
        failure=failure,
    )


def read_status(path: str | Path) -> RunStatus:
    return loads_status(Path(path).read_text(encoding="utf-8"))


def _parse_failure(value: object) -> FailureSummary | None:
    if value is None:
        return None
    data = _mapping(value, "failure")
    expected = {"error_type", "message", "phase", "step"}
    if set(data) != expected:
        raise SchemaError("failure summary fields differ from the status schema")
    try:
        phase = RunPhase(_text(data["phase"], "failure.phase"))
    except ValueError as exc:
        raise SchemaError(f"unknown failure phase {data['phase']!r}") from exc
    step_value = data["step"]
    return FailureSummary(
        error_type=_text(data["error_type"], "failure.error_type"),
        message=_text(data["message"], "failure.message"),
        phase=phase,
        step=_integer(step_value, "failure.step") if step_value is not None else None,
    )


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise SchemaError(f"{path} must be an object")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{path} must be nonempty text")
    return value


def _optional_text(value: object, path: str) -> str | None:
    return None if value is None else _text(value, path)


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{path} must be an integer")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SchemaError(f"{path} must be a number")
    return float(value)


def _optional_number(value: object, path: str) -> float | None:
    return None if value is None else _number(value, path)
