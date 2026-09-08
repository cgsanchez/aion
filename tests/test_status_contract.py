from __future__ import annotations

import json

import pytest

from aion.errors import SchemaError
from aion.io import (
    FailureSummary,
    RunPhase,
    RunStatus,
    dumps_status,
    loads_status,
)

pytestmark = pytest.mark.fast


def status(**changes: object) -> RunStatus:
    values: dict[str, object] = {
        "run_id": "run-001",
        "simulation_id": "a" * 64,
        "phase": RunPhase.PROPAGATING,
        "global_step": 20,
        "last_accepted_step": 19,
        "total_steps": 100,
        "wall_time_seconds": 14.2,
        "eta_seconds": 57.0,
        "updated_at_utc": "2026-09-08T12:00:00Z",
        "host": "node1",
        "pid": 1234,
        "latest_checkpoint": "checkpoint_10.h5",
        "failure": None,
    }
    values.update(changes)
    return RunStatus(**values)  # type: ignore[arg-type]


def test_status_json_round_trip_is_strict_and_deterministic() -> None:
    original = status()
    text = dumps_status(original)
    assert loads_status(text) == original
    assert dumps_status(loads_status(text)) == text


def test_unknown_status_field_fails() -> None:
    data = json.loads(dumps_status(status()))
    data["mystery"] = 1
    with pytest.raises(SchemaError, match="unknown"):
        loads_status(json.dumps(data))


def test_failed_status_requires_failure_summary() -> None:
    with pytest.raises(SchemaError, match="requires"):
        status(phase=RunPhase.FAILED)
    failed = status(
        phase=RunPhase.FAILED,
        failure=FailureSummary(
            error_type="MidpointConvergenceError",
            message="density residual did not converge",
            phase=RunPhase.PROPAGATING,
            step=20,
        ),
    )
    assert loads_status(dumps_status(failed)) == failed


def test_status_step_ordering_is_validated() -> None:
    with pytest.raises(SchemaError, match="last_accepted"):
        status(global_step=5, last_accepted_step=6)
