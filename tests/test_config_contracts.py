from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from aion.config import (
    AtomConfig,
    BackendConfig,
    BackendKind,
    CompiledSourceConfig,
    ElectromagneticOrigin,
    ElectronicStructureConfig,
    FixedTimeGrid,
    FormulationConfig,
    FormulationKind,
    GaugeRepresentation,
    IntegratorKind,
    KickEventConfig,
    MetadataConfig,
    MoleculeConfig,
    ObservableSchedules,
    OutputConfig,
    PropagationConfig,
    PythonProviderSourceConfig,
    ReferenceConfig,
    ReferenceLinkConfig,
    ReferenceOutputConfig,
    SimulationConfig,
    Sin2VectorPotentialPulseConfig,
    SourceConfig,
    StepSchedule,
    ValidationConfig,
    XCFamily,
    ZeroSourceConfig,
    dumps_config,
    loads_config,
)
from aion.errors import (
    ConfigurationError,
    UnknownConfigurationFieldError,
    UnsupportedConfigurationError,
)

pytestmark = pytest.mark.fast

_DIGEST = "a" * 64


def reference_config() -> ReferenceConfig:
    return ReferenceConfig(
        molecule=MoleculeConfig(
            atoms=(
                AtomConfig("H", (0.0, 0.0, -0.7)),
                AtomConfig("H", (0.0, 0.0, 0.7)),
            ),
            electromagnetic_origin=ElectromagneticOrigin((0.0, 0.0, 0.2)),
        ),
        electronic_structure=ElectronicStructureConfig(
            basis="cc-pvdz",
            functional="pbe",
            xc_family=XCFamily.GGA,
        ),
    )


def simulation_config() -> SimulationConfig:
    return SimulationConfig(
        reference=ReferenceLinkConfig(_DIGEST),
        formulation=FormulationConfig(FormulationKind.BARE_LENGTH_GAUGE),
        source=Sin2VectorPotentialPulseConfig(
            peak_electric_field_au=1.0e-3,
            angular_frequency_au=0.31,
            cycles=5,
            polarization=(0.0, 0.0, 1.0),
        ),
        propagation=PropagationConfig(
            FixedTimeGrid(0.0, 0.05, 100),
            IntegratorKind.FIXED_METRIC_SCEM,
        ),
    )


def test_reference_resolved_toml_round_trip_is_identical() -> None:
    original = reference_config()
    text = dumps_config(original)
    first = loads_config(text)
    second = loads_config(first.normalized_toml)
    assert first.config == original
    assert second.config == original
    assert second.normalized_toml == first.normalized_toml
    assert second.scientific_id == first.scientific_id


def test_simulation_resolved_toml_round_trip_is_identical() -> None:
    original = simulation_config()
    first = loads_config(dumps_config(original))
    second = loads_config(first.normalized_toml)
    assert first.config == original
    assert second.config == original
    assert second.scientific_id == first.scientific_id


@pytest.mark.parametrize(
    ("source", "formulation", "integrator"),
    [
        (
            ZeroSourceConfig(),
            FormulationKind.BARE_VELOCITY_GAUGE,
            IntegratorKind.FIXED_METRIC_SCEM,
        ),
        (
            CompiledSourceConfig(_DIGEST, Path("compiled-source.h5")),
            FormulationKind.P0,
            IntegratorKind.CONNECTION_AWARE_SCEM,
        ),
        (
            PythonProviderSourceConfig("projectile.v1", "b" * 64),
            FormulationKind.P0_E1,
            IntegratorKind.CONNECTION_AWARE_SCEM,
        ),
    ],
)
def test_all_source_and_formulation_variants_round_trip(
    source: SourceConfig,
    formulation: FormulationKind,
    integrator: IntegratorKind,
) -> None:
    baseline = simulation_config()
    config = replace(
        baseline,
        source=source,
        formulation=FormulationConfig(formulation),
        propagation=replace(baseline.propagation, integrator=integrator),
    )
    resolved = loads_config(dumps_config(config))
    assert resolved.config == config
    assert resolved.scientific_id == config.scientific_id


def test_unknown_and_source_irrelevant_fields_fail() -> None:
    text = dumps_config(simulation_config())
    with pytest.raises(UnknownConfigurationFieldError, match="mystery"):
        loads_config(text + "\nmystery = 4\n")
    bad_source = text.replace(
        'kind = "sin2_vector_potential_pulse"',
        'kind = "zero"',
    )
    with pytest.raises(UnknownConfigurationFieldError, match="unknown or unused"):
        loads_config(bad_source)


def test_backend_rejects_unused_device_for_cpu() -> None:
    with pytest.raises(ConfigurationError, match="unused"):
        BackendConfig(BackendKind.CPU, device_index=0)


def test_formulation_gauge_is_explicit_and_bare_mismatches_fail() -> None:
    covariant = FormulationConfig(FormulationKind.P0_E1, GaugeRepresentation.VELOCITY)
    assert covariant.as_mapping() == {"kind": "p0_e1", "gauge": "velocity"}
    with pytest.raises(UnsupportedConfigurationError, match="requires the length"):
        FormulationConfig(FormulationKind.BARE_LENGTH_GAUGE, GaugeRepresentation.VELOCITY)


def test_programmatic_boundary_rejects_untyped_nested_values() -> None:
    with pytest.raises(ConfigurationError, match="AtomConfig"):
        MoleculeConfig(atoms=(("H", (0.0, 0.0, 0.0)),))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="wrong typed contract"):
        replace(reference_config(), backend="gpu")  # type: ignore[arg-type]


def test_configuration_is_deeply_immutable() -> None:
    config = replace(
        simulation_config(),
        events=(KickEventConfig("kick", 0, (0.0, 0.0, 0.1)),),
    )
    with pytest.raises(FrozenInstanceError):
        config.reference = ReferenceLinkConfig("b" * 64)  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.events[0] = config.events[0]  # type: ignore[index]


def test_operational_values_do_not_change_simulation_identity() -> None:
    original = simulation_config()
    changed = replace(
        original,
        reference=replace(original.reference, path=Path("elsewhere/reference.h5")),
        source=replace(
            CompiledSourceConfig(_DIGEST, Path("one/source.h5")),
            path=Path("two/source.h5"),
        ),
        output=OutputConfig(
            directory=Path("another/run"),
            schedules=ObservableSchedules(
                dipole_current=StepSchedule(every=7),
                energy=StepSchedule(every=11),
            ),
        ),
        metadata=MetadataConfig(
            label="rerun",
            timestamp_utc="2026-09-08T12:00:00Z",
            host="different-host",
        ),
    )
    baseline_compiled = replace(
        original,
        source=CompiledSourceConfig(_DIGEST, Path("one/source.h5")),
    )
    assert changed.scientific_id == baseline_compiled.scientific_id


def test_gpu_device_index_does_not_change_identity_but_backend_does() -> None:
    original = simulation_config()
    gpu0 = replace(original, backend=BackendConfig(BackendKind.GPU, device_index=0))
    gpu1 = replace(original, backend=BackendConfig(BackendKind.GPU, device_index=1))
    assert gpu0.scientific_id == gpu1.scientific_id
    assert original.scientific_id != gpu0.scientific_id


def test_physics_and_algorithm_values_change_simulation_identity() -> None:
    original = simulation_config()
    assert (
        original.scientific_id
        != replace(
            original,
            source=replace(original.source, angular_frequency_au=0.32),
        ).scientific_id
    )
    assert (
        original.scientific_id
        != replace(
            original,
            propagation=replace(
                original.propagation,
                time_grid=FixedTimeGrid(0.0, 0.04, 125),
            ),
        ).scientific_id
    )
    assert (
        original.scientific_id
        != replace(
            original,
            validation=ValidationConfig(strict_validation=True),
        ).scientific_id
    )


def test_reference_paths_and_metadata_do_not_change_reference_identity() -> None:
    original = reference_config()
    changed = replace(
        original,
        backend=BackendConfig(BackendKind.GPU, device_index=3),
        output=ReferenceOutputConfig(Path("elsewhere/reference.h5")),
        metadata=MetadataConfig(label="other", host="node2"),
    )
    assert changed.scientific_id == original.scientific_id


def test_support_validation_rejects_wrong_integrator_and_spin() -> None:
    original = simulation_config()
    with pytest.raises(UnsupportedConfigurationError, match="requires"):
        replace(
            original,
            propagation=replace(
                original.propagation,
                integrator=IntegratorKind.CONNECTION_AWARE_SCEM,
            ),
        )
    with pytest.raises(UnsupportedConfigurationError, match="closed-shell"):
        replace(reference_config().molecule, spin=1)


def test_event_ids_and_grid_alignment_are_strict() -> None:
    config = replace(
        simulation_config(),
        events=(KickEventConfig("late", 100, (0.0, 0.0, 0.1)),),
    )
    text = dumps_config(config).replace("intervals = 100", "intervals = 99")
    with pytest.raises(ConfigurationError, match="outside"):
        loads_config(text)


def test_fixed_time_grid_is_endpoint_inclusive_and_integer_authoritative() -> None:
    grid = FixedTimeGrid(start_au=1.0, step_au=0.05, intervals=4)
    assert grid.state_count == 5
    assert grid.end_au == pytest.approx(1.2)
    assert StepSchedule(every=3).steps(grid) == (0, 3, 4)
    with pytest.raises(ConfigurationError, match="outside"):
        grid.time_at(5)
