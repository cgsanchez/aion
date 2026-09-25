from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from aion.config import (
    AffineElectromagneticSourceConfig,
    BackendConfig,
    ExactWilsonActionConfig,
    FixedTimeGrid,
    MetadataConfig,
    OutputConfig,
    RationalApproximation,
    ReducedWilsonActionConfig,
    ReducedWilsonLevel,
    ReferenceLinkConfig,
    Sin2VectorPotentialPulseConfig,
    WilsonGridKind,
    WilsonGridPruning,
    WilsonIntegratorKind,
    WilsonMagneticGaugeKind,
    WilsonNumericsConfig,
    WilsonPropagationConfig,
    WilsonSimulationConfig,
    WilsonStationaryBranch,
    WilsonStationaryConfig,
    WilsonStationaryOutputConfig,
    WilsonStationaryPolicyConfig,
    WilsonStationaryStateLinkConfig,
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

_REFERENCE_DIGEST = "a" * 64
_STATE_DIGEST = "b" * 64


def affine_source() -> AffineElectromagneticSourceConfig:
    return AffineElectromagneticSourceConfig(
        electric=Sin2VectorPotentialPulseConfig(
            peak_electric_field_au=1.0e-3,
            angular_frequency_au=0.31,
            cycles=5,
            polarization=(0.0, 0.0, 1.0),
        ),
        electric_field_origin_offset_au=(1.0e-4, -2.0e-4, 3.0e-4),
        magnetic_field_reference_au=(0.0, 0.0, 2.0e-4),
        magnetic_field_rate_au=(0.0, 0.0, 1.0e-6),
        magnetic_reference_time_au=0.25,
        magnetic_gauge=WilsonMagneticGaugeKind.SYMMETRIC,
    )


def numerics() -> WilsonNumericsConfig:
    return WilsonNumericsConfig(
        grid_kind=WilsonGridKind.QUALIFICATION,
        grid_level=4,
        grid_pruning=WilsonGridPruning.NONE,
        block_size=4096,
        auxiliary_basis="weigend",
        ri_relative_threshold=0.0,
        ri_absolute_threshold=1.0e-7,
        ri_maximum_rank=None,
        memory_budget_bytes=2**30,
    )


def stationary_config() -> WilsonStationaryConfig:
    return WilsonStationaryConfig(
        reference=ReferenceLinkConfig(_REFERENCE_DIGEST, Path("reference.h5")),
        action=ExactWilsonActionConfig(WilsonStationaryBranch.KOHN_SHAM_LDA),
        numerics=numerics(),
        source=affine_source(),
        source_time_au=0.0,
        stationary=WilsonStationaryPolicyConfig(
            maximum_iterations=100,
            density_tolerance=1.0e-10,
            orbital_tolerance=1.0e-9,
            energy_tolerance_au=1.0e-11,
            damping=0.5,
            diis_start_iteration=2,
            diis_space=8,
        ),
        backend=BackendConfig(),
        output=WilsonStationaryOutputConfig(Path("stationary.h5")),
    )


def simulation_config() -> WilsonSimulationConfig:
    return WilsonSimulationConfig(
        reference=ReferenceLinkConfig(_REFERENCE_DIGEST, Path("reference.h5")),
        stationary_state=WilsonStationaryStateLinkConfig(
            _STATE_DIGEST,
            Path("stationary.h5"),
        ),
        action=ExactWilsonActionConfig(WilsonStationaryBranch.KOHN_SHAM_LDA),
        numerics=numerics(),
        source=affine_source(),
        propagation=WilsonPropagationConfig(
            FixedTimeGrid(0.0, 0.05, 100),
            WilsonIntegratorKind.NONLINEAR_GAUSS_MAGNUS,
            RationalApproximation.PADE_22,
            nonlinear_tolerance=1.0e-11,
            maximum_iterations=40,
        ),
        backend=BackendConfig(),
        output=OutputConfig(Path("run")),
    )


@pytest.mark.parametrize("config", [stationary_config(), simulation_config()])
def test_wilson_configurations_have_stable_strict_toml_round_trips(
    config: WilsonStationaryConfig | WilsonSimulationConfig,
) -> None:
    first = loads_config(dumps_config(config))
    second = loads_config(first.normalized_toml)
    assert first.config == config
    assert second.config == config
    assert second.normalized_toml == first.normalized_toml
    assert second.scientific_id == config.scientific_id


def test_operational_paths_and_metadata_do_not_change_scientific_identity() -> None:
    stationary = stationary_config()
    moved_stationary = replace(
        stationary,
        reference=replace(stationary.reference, path=Path("elsewhere/reference.h5")),
        output=WilsonStationaryOutputConfig(Path("elsewhere/state.h5")),
        metadata=MetadataConfig(label="moved"),
    )
    assert moved_stationary.scientific_id == stationary.scientific_id

    simulation = simulation_config()
    moved_simulation = replace(
        simulation,
        reference=replace(simulation.reference, path=Path("elsewhere/reference.h5")),
        stationary_state=replace(
            simulation.stationary_state,
            path=Path("elsewhere/state.h5"),
        ),
        output=OutputConfig(Path("elsewhere/run")),
        metadata=MetadataConfig(host="other-host"),
    )
    assert moved_simulation.scientific_id == simulation.scientific_id


def test_wilson_unknown_fields_and_unqualified_algorithm_choices_fail() -> None:
    text = dumps_config(simulation_config())
    with pytest.raises(UnknownConfigurationFieldError, match="mystery"):
        loads_config(
            text.replace('schema_version = "1.0.0"', 'schema_version = "1.0.0"\nmystery = 4')
        )
    with pytest.raises(UnsupportedConfigurationError, match="pade_22"):
        loads_config(
            text.replace(
                'rational_approximation = "pade_22"', 'rational_approximation = "cayley_11"'
            )
        )


def test_reduced_action_is_explicit_and_reduced_gga_is_rejected() -> None:
    reduced = replace(
        stationary_config(),
        action=ReducedWilsonActionConfig(
            WilsonStationaryBranch.KOHN_SHAM_LDA,
            ReducedWilsonLevel.STRICT_C1,
        ),
        source=replace(affine_source(), electric=ZeroSourceConfig()),
    )
    assert loads_config(dumps_config(reduced)).config == reduced
    with pytest.raises(UnsupportedConfigurationError, match="reduced Wilson GGA"):
        ReducedWilsonActionConfig(
            WilsonStationaryBranch.KOHN_SHAM_GGA,
            ReducedWilsonLevel.E1,
        )


def test_affine_source_magnetic_gauge_fields_are_strict() -> None:
    with pytest.raises(ConfigurationError, match="landau_axis"):
        loads_config(
            dumps_config(simulation_config()).replace(
                'magnetic_gauge = "symmetric"',
                'magnetic_gauge = "symmetric"\nlandau_axis = [1.0, 0.0, 0.0]',
            )
        )
