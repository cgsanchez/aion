"""Lightweight manifests connecting independent immutable trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py

from aion.config import SimulationConfig, canonical_sha256, loads_config
from aion.errors import TrajectoryError
from aion.io.trajectory import Trajectory, load_trajectory
from aion.io.util import publish_text, read_text


@dataclass(frozen=True, slots=True)
class ComparisonMember:
    label: str
    trajectory: Trajectory
    formulation: str
    gauge: str
    velocity_fraction: float


@dataclass(frozen=True, slots=True)
class ComparisonManifest:
    path: Path
    comparison_id: str
    reference_fingerprint_sha256: str
    source_fingerprint_sha256: str
    event_fingerprint_sha256: str
    members: tuple[ComparisonMember, ...]


def _metadata(trajectory: Trajectory) -> tuple[SimulationConfig, str]:
    with h5py.File(trajectory.path, "r") as handle:
        resolved = loads_config(read_text(handle["configuration/normalized_toml"]))
        if not isinstance(resolved.config, SimulationConfig):
            raise TrajectoryError("comparison member contains a non-simulation configuration")
        source = handle["source"]
        assert isinstance(source, h5py.Group)
        event_fingerprint = str(source.attrs.get("event_fingerprint_sha256", ""))
    if len(event_fingerprint) != 64:
        raise TrajectoryError("comparison member lacks an event-schedule fingerprint")
    return resolved.config, event_fingerprint


def create_comparison_manifest(
    trajectories: dict[str, Trajectory | str | Path],
    path: str | Path,
) -> ComparisonManifest:
    """Validate comparability and publish a manifest without co-propagation."""

    if len(trajectories) < 2:
        raise TrajectoryError("a comparison requires at least two trajectories")
    if any(not label.strip() for label in trajectories):
        raise TrajectoryError("comparison labels must be nonempty")
    resolved_members: list[tuple[str, Trajectory, SimulationConfig, str]] = []
    for label, value in sorted(trajectories.items()):
        trajectory = load_trajectory(value) if isinstance(value, str | Path) else value
        config, event_fingerprint = _metadata(trajectory)
        resolved_members.append((label, trajectory, config, event_fingerprint))
    if len({item[1].run_id for item in resolved_members}) != len(resolved_members):
        raise TrajectoryError("a trajectory run may occur only once in a comparison")

    first_trajectory = resolved_members[0][1]
    first_config = resolved_members[0][2]
    first_event = resolved_members[0][3]
    expected = (
        first_trajectory.reference_fingerprint_sha256,
        first_trajectory.source_fingerprint_sha256,
        first_event,
        first_config.propagation.time_grid,
        first_config.output.schedules,
        first_trajectory.final_step,
    )
    for _label, trajectory, config, event_fingerprint in resolved_members[1:]:
        actual = (
            trajectory.reference_fingerprint_sha256,
            trajectory.source_fingerprint_sha256,
            event_fingerprint,
            config.propagation.time_grid,
            config.output.schedules,
            trajectory.final_step,
        )
        if actual != expected:
            raise TrajectoryError(
                "comparison members must share reference, source/events, time grid, "
                "observation schedules, and final step"
            )

    identity_members = [
        {
            "label": label,
            "run_id": trajectory.run_id,
            "trajectory_sha256": trajectory.sha256,
        }
        for label, trajectory, _config, _event in resolved_members
    ]
    comparison_id = canonical_sha256(
        {
            "schema": "aion.comparison-manifest",
            "schema_version": "1.0.0",
            "reference_fingerprint_sha256": expected[0],
            "source_fingerprint_sha256": expected[1],
            "event_fingerprint_sha256": expected[2],
            "time_grid": {
                "start_time_au": first_config.propagation.time_grid.start_au,
                "time_step_au": first_config.propagation.time_grid.step_au,
                "intervals": first_config.propagation.time_grid.intervals,
            },
            "observation_schedules": first_config.output.schedules.as_mapping(),
            "members": identity_members,
        }
    )
    members_list: list[ComparisonMember] = []
    for label, trajectory, config, _event in resolved_members:
        gauge = config.formulation.gauge
        if gauge is None:  # guarded by FormulationConfig, retained for type narrowing
            raise TrajectoryError("comparison member has no resolved gauge representation")
        members_list.append(
            ComparisonMember(
                label=label,
                trajectory=trajectory,
                formulation=config.formulation.kind.value,
                gauge=gauge.value,
                velocity_fraction=config.formulation.resolved_velocity_fraction,
            )
        )
    members = tuple(members_list)
    payload = {
        "schema": "aion.comparison-manifest",
        "schema_version": "1.0.0",
        "comparison_id": comparison_id,
        "shared": {
            "reference_fingerprint_sha256": expected[0],
            "source_fingerprint_sha256": expected[1],
            "event_fingerprint_sha256": expected[2],
            "start_time_au": first_config.propagation.time_grid.start_au,
            "time_step_au": first_config.propagation.time_grid.step_au,
            "intervals": first_config.propagation.time_grid.intervals,
            "observation_schedules": first_config.output.schedules.as_mapping(),
        },
        "members": [
            {
                "label": member.label,
                "formulation": member.formulation,
                "gauge": member.gauge,
                "velocity_fraction": member.velocity_fraction,
                "run_id": member.trajectory.run_id,
                "simulation_id": member.trajectory.simulation_id,
                "trajectory_path": str(member.trajectory.path.resolve()),
                "trajectory_sha256": member.trajectory.sha256,
            }
            for member in members
        ],
    }
    target = Path(path)
    try:
        publish_text(target, json.dumps(payload, sort_keys=True, indent=2) + "\n")
    except FileExistsError as exc:
        raise TrajectoryError(str(exc)) from exc
    return ComparisonManifest(
        path=target,
        comparison_id=comparison_id,
        reference_fingerprint_sha256=str(expected[0]),
        source_fingerprint_sha256=str(expected[1]),
        event_fingerprint_sha256=str(expected[2]),
        members=members,
    )
