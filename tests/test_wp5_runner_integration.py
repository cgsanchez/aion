from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from aion.cli import main
from aion.config import (
    BackendConfig,
    FixedTimeGrid,
    FormulationConfig,
    FormulationKind,
    GaugeRepresentation,
    IntegratorKind,
    KickEventConfig,
    ObservableSchedules,
    OutputConfig,
    PropagationConfig,
    ReferenceLinkConfig,
    SimulationConfig,
    StepSchedule,
    ZeroSourceConfig,
    dumps_config,
)
from aion.errors import MidpointConvergenceError, RunCancelledError, TrajectoryError
from aion.io.checkpoint import load_checkpoint
from aion.io.export import export_trajectory_csv
from aion.io.status import RunPhase, read_status
from aion.io.trajectory import load_trajectory, stitch_observable
from aion.workflows import (
    RunControl,
    build_simulation,
    create_comparison_manifest,
    resume,
    run,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def runner_references(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    from aion.electronic_structure import prepare_pyscf_reference

    root = tmp_path_factory.mktemp("wp5-references")
    result: dict[str, object] = {}
    for name in ("h2", "lih"):
        path = root / f"{name}.reference.h5"
        reference = prepare_pyscf_reference(molecular_config(name, path))
        reference.save()
        result[name] = reference
    return result


def _config(
    reference: object,
    output: Path,
    kind: FormulationKind,
    gauge: GaugeRepresentation | None = None,
    *,
    backend: BackendConfig | None = None,
    kick_step: int = 2,
    velocity_fraction: float | None = None,
) -> SimulationConfig:
    integrator = (
        IntegratorKind.FIXED_METRIC_SCEM
        if kind in {FormulationKind.BARE_LENGTH_GAUGE, FormulationKind.BARE_VELOCITY_GAUGE}
        else IntegratorKind.CONNECTION_AWARE_SCEM
    )
    schedules = ObservableSchedules(
        dipole_current=StepSchedule(every=2),
        energy=StepSchedule(every=3),
        diagnostics=StepSchedule(every=2),
        source=StepSchedule(every=3),
        checkpoints=StepSchedule(every=2),
    )
    return SimulationConfig(
        reference=ReferenceLinkConfig(
            reference.fingerprint_sha256,
            reference.config.output.artifact_path,
        ),
        formulation=FormulationConfig(kind, gauge, velocity_fraction),
        source=ZeroSourceConfig(),
        propagation=PropagationConfig(
            FixedTimeGrid(0.0, 0.05, 4),
            integrator,
            density_tolerance=1.0e-11,
        ),
        backend=BackendConfig() if backend is None else backend,
        events=(KickEventConfig("kick-z", kick_step, (0.0, 0.0, 1.0e-3)),),
        output=OutputConfig(output, schedules),
    )


@pytest.mark.parametrize(
    ("kind", "gauge"),
    (
        (FormulationKind.BARE_LENGTH_GAUGE, None),
        (FormulationKind.BARE_VELOCITY_GAUGE, None),
        (FormulationKind.P0, GaugeRepresentation.LENGTH),
        (FormulationKind.P0_E1, GaugeRepresentation.VELOCITY),
    ),
)
def test_runner_records_exact_schedules_events_checkpoints_and_status(
    runner_references: dict[str, object],
    tmp_path: Path,
    kind: FormulationKind,
    gauge: GaugeRepresentation | None,
) -> None:
    reference = runner_references["lih" if kind is FormulationKind.P0_E1 else "h2"]
    output = tmp_path / f"{kind.value}-{gauge}"
    simulation = build_simulation(_config(reference, output, kind, gauge), reference)
    dipole_id = simulation.calculators.definitions["electronic_dipole"].definition_id
    primary_current_id = simulation.calculators.definitions["primary_current"].definition_id
    energy_id = simulation.calculators.definitions["energy_matter_total"].definition_id
    trajectory = run(simulation)

    assert trajectory.complete
    assert trajectory.final_step == 4
    assert trajectory.event_steps == (2,)
    assert trajectory.read_observable(dipole_id).steps.tolist() == [0, 2, 4]
    assert trajectory.read_observable(energy_id).steps.tolist() == [0, 3, 4]
    assert read_status(output / "status.json").phase is RunPhase.COMPLETED
    assert sorted(path.name for path in output.glob("checkpoint_*.h5")) == [
        "checkpoint_00000000.h5",
        "checkpoint_00000002.h5",
        "checkpoint_00000004.h5",
    ]
    with h5py.File(trajectory.path, "r") as handle:
        assert handle["source/midpoint/step"][...].tolist() == [0, 1, 2, 3]
        assert handle["source/endpoint/step"][...].tolist() == [0, 3, 4]
        assert handle["source/work/step"][...].tolist() == [0, 1, 2, 3]
        assert "state/ao_density_snapshot" not in handle["observables"]
        event = handle["events/records/00000000"]
        assert event["pre/coefficients"].compression == "gzip"
        assert event["post/density"].compression == "gzip"
        if kind in {
            FormulationKind.BARE_LENGTH_GAUGE,
            FormulationKind.BARE_VELOCITY_GAUGE,
        }:
            event_current_path = "observables/" + primary_current_id.replace(".", "/")
            current_before = event[f"pre/{event_current_path}/values"][...]
            current_after = event[f"post/{event_current_path}/values"][...]
            # With kappa=integral(E dt), the electron kick convention gives
            # Delta a=-kappa in VG and exp(-i kappa.r) in LG.  Both therefore
            # produce the same positive charge-current jump along +z.
            assert current_after[2] > current_before[2]
        assert float(event["work_increment_au"][()]) == pytest.approx(
            trajectory.accumulated_source_work_au,
            abs=1.0e-9,
        )
    with h5py.File(output / "checkpoint_00000004.h5", "r") as handle:
        assert handle["state/coefficients"].compression == "gzip"
    with pytest.raises(TrajectoryError, match="already contains"):
        run(build_simulation(_config(reference, output, kind, gauge), reference))


@pytest.mark.parametrize(
    ("kind", "gauge"),
    (
        (FormulationKind.BARE_LENGTH_GAUGE, None),
        (FormulationKind.BARE_VELOCITY_GAUGE, None),
        (FormulationKind.P0_E1, GaugeRepresentation.LENGTH),
        (FormulationKind.P0_E1, GaugeRepresentation.VELOCITY),
    ),
)
def test_interruption_restart_event_idempotence_and_stitching_equal_full_run(
    runner_references: dict[str, object],
    tmp_path: Path,
    kind: FormulationKind,
    gauge: GaugeRepresentation | None,
) -> None:
    reference = runner_references["lih" if kind is FormulationKind.P0_E1 else "h2"]
    suffix = kind.value if gauge is None else f"{kind.value}-{gauge.value}"

    full_config = _config(reference, tmp_path / f"full-{suffix}", kind, gauge)
    full_simulation = build_simulation(full_config, reference)
    definition_id = full_simulation.calculators.definitions["electronic_dipole"].definition_id
    full = run(full_simulation)

    interrupted_config = replace(
        full_config,
        output=replace(full_config.output, directory=tmp_path / f"interrupted-{suffix}"),
    )
    interrupted_simulation = build_simulation(interrupted_config, reference)
    control = RunControl()

    def stop_at_kick_boundary(step: int) -> None:
        if step == 2:
            control.request_cancel()

    control.after_accepted_step = stop_at_kick_boundary
    with pytest.raises(RunCancelledError):
        run(interrupted_simulation, control=control)

    failed_path = interrupted_config.output.directory / "trajectory.failed.h5"
    parent = load_trajectory(failed_path, allow_incomplete=True)
    assert not parent.complete
    assert parent.final_step == 2
    assert parent.event_steps == ()
    assert not tuple(interrupted_config.output.directory.glob("*.partial"))
    checkpoint_path = interrupted_config.output.directory / "checkpoint_00000002.h5"
    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint.applied_event_identifiers == frozenset()

    resumed_directory = tmp_path / f"resumed-{suffix}"
    child = resume(checkpoint_path, output=resumed_directory)
    assert child.parent_run_id == parent.run_id
    assert child.parent_checkpoint_sha256 in parent.checkpoint_sha256
    assert child.event_steps == (2,)
    assert load_checkpoint(resumed_directory / "checkpoint_00000002.h5").applied_event_identifiers

    full_final = load_checkpoint(full_config.output.directory / "checkpoint_00000004.h5")
    child_final = load_checkpoint(resumed_directory / "checkpoint_00000004.h5")
    assert np.allclose(child_final.coefficients, full_final.coefficients, atol=2.0e-12)
    assert np.allclose(
        (child_final.coefficients * child_final.occupations[None, :])
        @ child_final.coefficients.conj().T,
        (full_final.coefficients * full_final.occupations[None, :])
        @ full_final.coefficients.conj().T,
        atol=2.0e-12,
    )
    assert child.accumulated_source_work_au == pytest.approx(
        full.accumulated_source_work_au,
        abs=2.0e-12,
    )
    stitched = stitch_observable((parent, child), definition_id)
    uninterrupted = full.read_observable(definition_id)
    assert np.array_equal(stitched.steps, uninterrupted.steps)
    assert np.array_equal(stitched.times_au, uninterrupted.times_au)
    assert np.allclose(stitched.values, uninterrupted.values, atol=2.0e-11)

    exported = export_trajectory_csv(
        full,
        tmp_path / f"csv-{suffix}",
        observable_ids=(definition_id,),
    )
    assert [path.name for path in exported] == [
        f"{definition_id.replace('.', '__')}.csv",
        "export_manifest.json",
    ]
    header = exported[0].read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("global_step,time_au,value_0")

    with (
        h5py.File(full.path, "r") as full_handle,
        h5py.File(parent.path, "r") as parent_handle,
        h5py.File(child.path, "r") as child_handle,
    ):
        full_work = full_handle["source/work"]
        parent_work = parent_handle["source/work"]
        child_work = child_handle["source/work"]
        for name in ("step", "time_au", "rate_au", "increment_au", "accumulated_au"):
            combined = np.concatenate((parent_work[name][...], child_work[name][...]))
            assert np.allclose(combined, full_work[name][...], atol=2.0e-12)
        assert len(parent_handle["events"]) == 0
        assert len(child_handle["events/records"]) == 1
        assert len(full_handle["events/records"]) == 1


def test_covariant_runs_publish_an_independent_comparison_manifest(
    runner_references: dict[str, object],
    tmp_path: Path,
) -> None:
    reference = runner_references["lih"]
    trajectories = {}
    specifications = (
        ("length", GaugeRepresentation.LENGTH, None),
        ("mixed", GaugeRepresentation.MIXED, 0.5),
        ("velocity", GaugeRepresentation.VELOCITY, None),
    )
    simulations = {}
    for label, gauge, fraction in specifications:
        simulation = build_simulation(
            _config(
                reference,
                tmp_path / label,
                FormulationKind.P0_E1,
                gauge,
                velocity_fraction=fraction,
            ),
            reference,
        )
        simulations[label] = simulation
        trajectories[label] = run(simulation)
    manifest = create_comparison_manifest(trajectories, tmp_path / "comparison.json")
    assert len(manifest.comparison_id) == 64
    assert {member.gauge for member in manifest.members} == {"length", "mixed", "velocity"}
    assert {member.label: member.velocity_fraction for member in manifest.members} == {
        "length": 0.0,
        "mixed": 0.5,
        "velocity": 1.0,
    }
    assert (tmp_path / "comparison.json").is_file()
    for field, tolerance in (
        ("electronic_dipole", 2.0e-10),
        ("primary_current", 2.0e-9),
        ("energy_matter_total", 2.0e-10),
    ):
        baseline = trajectories["length"].read_observable(
            simulations["length"].calculators.definitions[field].definition_id
        )
        for label in ("mixed", "velocity"):
            series = trajectories[label].read_observable(
                simulations[label].calculators.definitions[field].definition_id
            )
            assert np.array_equal(baseline.steps, series.steps)
            assert np.allclose(baseline.values, series.values, atol=tolerance)
    with pytest.raises(TrajectoryError, match="overwrite"):
        create_comparison_manifest(trajectories, tmp_path / "comparison.json")


def test_nonlinear_failure_publishes_only_accepted_state_and_status(
    runner_references: dict[str, object],
    tmp_path: Path,
) -> None:
    reference = runner_references["h2"]
    base = _config(
        reference,
        tmp_path / "failure",
        FormulationKind.BARE_LENGTH_GAUGE,
    )
    config = replace(
        base,
        propagation=replace(
            base.propagation,
            density_tolerance=1.0e-30,
            max_iterations=1,
        ),
        events=(),
    )
    with pytest.raises(MidpointConvergenceError):
        run(build_simulation(config, reference))
    assert not (config.output.directory / "trajectory.h5").exists()
    failed = load_trajectory(
        config.output.directory / "trajectory.failed.h5",
        allow_incomplete=True,
    )
    assert failed.final_step == 0
    assert load_checkpoint(config.output.directory / "checkpoint_00000000.h5").global_step == 0
    status = read_status(config.output.directory / "status.json")
    assert status.phase is RunPhase.FAILED
    assert status.failure is not None
    assert status.failure.error_type == "MidpointConvergenceError"


def test_run_inspect_and_export_cli_use_the_typed_workflow(
    runner_references: dict[str, object],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference = runner_references["h2"]
    config = _config(
        reference,
        tmp_path / "cli-run",
        FormulationKind.BARE_LENGTH_GAUGE,
    )
    config_path = tmp_path / "simulation.toml"
    config_path.write_text(dumps_config(config), encoding="utf-8")
    assert main(["run", str(config_path)]) == 0
    trajectory = config.output.directory / "trajectory.h5"
    assert trajectory.is_file()
    assert main(["inspect", str(trajectory)]) == 0
    assert main(["export", str(trajectory), str(tmp_path / "cli-csv")]) == 0
    resumed_output = tmp_path / "cli-resumed"
    assert (
        main(
            [
                "resume",
                str(config.output.directory / "checkpoint_00000002.h5"),
                "--output",
                str(resumed_output),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "scientific_id =" in output
    assert '"artifact_kind": "trajectory"' in output
    assert (tmp_path / "cli-csv/export_manifest.json").is_file()
    assert (resumed_output / "trajectory.h5").is_file()


def test_midpoint_work_does_not_force_complete_energy_evaluation(
    runner_references: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = runner_references["h2"]
    base = _config(
        reference,
        tmp_path / "scheduled-energy",
        FormulationKind.BARE_LENGTH_GAUGE,
    )
    schedules = replace(base.output.schedules, energy=StepSchedule(every=0))
    config = replace(base, events=(), output=replace(base.output, schedules=schedules))
    simulation = build_simulation(config, reference)
    model = simulation.formulation.context.electronic_model
    model_type = type(model)
    original = model_type.energy
    calls = 0

    def counted_energy(self: object, *args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(model_type, "energy", counted_energy)
    trajectory = run(simulation)
    assert trajectory.final_step == 4
    # Initial baseline, scheduled initial sample, and scheduled final sample.
    # The four accepted midpoint work records use Formulation.power(), not
    # complete component evaluation.
    assert calls == 3
