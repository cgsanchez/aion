from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from aion.config import (
    AffineElectromagneticSourceConfig,
    BackendConfig,
    ExactWilsonActionConfig,
    ReferenceLinkConfig,
    WilsonGridKind,
    WilsonGridPruning,
    WilsonMagneticGaugeKind,
    WilsonNumericsConfig,
    WilsonStationaryBranch,
    WilsonStationaryConfig,
    WilsonStationaryOutputConfig,
    WilsonStationaryPolicyConfig,
    ZeroSourceConfig,
)
from aion.electromagnetism import build_affine_electromagnetic_source
from aion.electronic_structure import (
    StationarySCFIteration,
    WilsonStationaryEnergyComponents,
    WilsonStationaryResiduals,
    WilsonStationaryStateData,
    load_wilson_stationary_state,
    save_wilson_stationary_state,
)
from aion.errors import SchemaError, WilsonStateError

pytestmark = pytest.mark.fast

_REFERENCE_DIGEST = "a" * 64


def stationary_state() -> WilsonStationaryStateData:
    source_config = AffineElectromagneticSourceConfig(
        electric=ZeroSourceConfig(),
        electric_field_origin_offset_au=(0.01, -0.02, 0.03),
        magnetic_field_reference_au=(0.0, 0.0, 0.04),
        magnetic_field_rate_au=(0.0, 0.0, 0.0),
        magnetic_reference_time_au=0.0,
        magnetic_gauge=WilsonMagneticGaugeKind.SYMMETRIC,
    )
    config = WilsonStationaryConfig(
        reference=ReferenceLinkConfig(_REFERENCE_DIGEST, Path("reference.h5")),
        action=ExactWilsonActionConfig(WilsonStationaryBranch.KOHN_SHAM_LDA),
        numerics=WilsonNumericsConfig(
            grid_kind=WilsonGridKind.QUALIFICATION,
            grid_level=4,
            grid_pruning=WilsonGridPruning.NONE,
            block_size=2048,
            dynamic_cache_entries=4,
            auxiliary_basis="weigend",
            ri_relative_threshold=0.0,
            ri_absolute_threshold=1.0e-7,
            ri_maximum_rank=None,
        ),
        source=source_config,
        source_time_au=0.0,
        stationary=WilsonStationaryPolicyConfig(
            maximum_iterations=80,
            density_tolerance=2.0e-10,
            orbital_tolerance=2.0e-10,
            energy_tolerance_au=2.0e-11,
            damping=0.5,
            diis_start_iteration=2,
            diis_space=8,
        ),
        backend=BackendConfig(),
        output=WilsonStationaryOutputConfig(Path("state.h5")),
    )
    source = build_affine_electromagnetic_source(source_config, (0.1, -0.2, 0.3)).sample(0.0)
    metric = np.eye(2, dtype=np.complex128)
    coefficient = np.asarray(((np.sqrt(0.8),), (1j * np.sqrt(0.2),)))
    occupations = np.asarray((2.0,))
    density = (coefficient * occupations[None, :]) @ coefficient.conj().T
    return WilsonStationaryStateData(
        config=config,
        reference_fingerprint_sha256=_REFERENCE_DIGEST,
        grid_fingerprint_sha256="b" * 64,
        auxiliary_space_fingerprint_sha256="c" * 64,
        source_sample=source,
        coefficients=coefficient,
        occupations=occupations,
        contravariant_density=density,
        mixed_density=density @ metric,
        metric=metric,
        orbital_frequency_matrix=np.asarray(((-0.3,),)),
        active_orbital_energies_au=np.asarray((-0.3,)),
        complete_orbital_spectrum_au=np.asarray((-0.3, 0.2)),
        occupation_spectrum=np.asarray((2.0, 0.0)),
        energies=WilsonStationaryEnergyComponents(
            kinetic_au=1.2,
            electron_nuclear_au=-2.4,
            one_electron_au=-1.2,
            hartree_au=0.7,
            exchange_correlation_au=-0.2,
            nuclear_repulsion_au=0.3,
            electronic_au=-0.7,
            molecular_total_au=-0.4,
        ),
        residuals=WilsonStationaryResiduals(
            orbital=1.0e-12,
            density_fixed_point=2.0e-12,
            commutator=3.0e-12,
            metric_orthonormality=4.0e-13,
            particle_number=2.0,
            particle_number_residual=1.0e-14,
            occupation_spectrum_imaginary_max_abs=2.0e-15,
            occupation_spectrum_residual=3.0e-14,
            closed_shell_density_polynomial=4.0e-14,
            orbital_frequency_occupation_commutator=5.0e-14,
            orbital_energy_sum_au=-0.6,
            double_counting_reconstructed_electronic_energy_au=-0.7,
            double_counting_reconstructed_molecular_energy_au=-0.4,
            double_counting_residual_au=6.0e-14,
            metric_minimum_eigenvalue=1.0,
            metric_condition_number=1.0,
            converged=True,
        ),
        iterations=(
            StationarySCFIteration(
                iteration=1,
                energy_molecular_total_au=-0.4,
                energy_change_au=None,
                density_fixed_point_residual=2.0e-12,
                commutator_residual=3.0e-12,
                diis_dimension=0,
                used_diis=False,
            ),
        ),
    )


def test_stationary_state_round_trip_is_authenticated_and_immutable(tmp_path: Path) -> None:
    original = stationary_state()
    path = tmp_path / "state.h5"
    save_wilson_stationary_state(original, path)
    loaded = load_wilson_stationary_state(path)
    assert loaded.fingerprint_sha256 == original.fingerprint_sha256
    assert loaded.action_fingerprint_sha256 == original.action_fingerprint_sha256
    assert loaded.source_fingerprint_sha256 == original.source_fingerprint_sha256
    assert loaded.config == original.config
    np.testing.assert_array_equal(loaded.contravariant_density, original.contravariant_density)
    np.testing.assert_array_equal(
        loaded.orbital_frequency_matrix, original.orbital_frequency_matrix
    )
    assert not loaded.contravariant_density.flags.writeable
    assert not loaded.metric.flags.writeable


def test_stationary_state_identity_tracks_scientific_content_not_output_path() -> None:
    original = stationary_state()
    moved = replace(
        original,
        config=replace(
            original.config,
            output=WilsonStationaryOutputConfig(Path("elsewhere/state.h5")),
        ),
    )
    changed_energy = replace(
        original,
        energies=replace(
            original.energies,
            kinetic_au=original.energies.kinetic_au + 0.1,
            one_electron_au=original.energies.one_electron_au + 0.1,
            electronic_au=original.energies.electronic_au + 0.1,
            molecular_total_au=original.energies.molecular_total_au + 0.1,
        ),
    )
    assert moved.fingerprint_sha256 == original.fingerprint_sha256
    assert changed_energy.fingerprint_sha256 != original.fingerprint_sha256


def test_stationary_state_writer_refuses_overwrite(tmp_path: Path) -> None:
    state = stationary_state()
    path = tmp_path / "state.h5"
    save_wilson_stationary_state(state, path)
    with pytest.raises(SchemaError, match="refusing to overwrite"):
        save_wilson_stationary_state(state, path)


def test_stationary_state_reader_rejects_fingerprint_tampering(tmp_path: Path) -> None:
    state = stationary_state()
    path = tmp_path / "state.h5"
    save_wilson_stationary_state(state, path)
    with h5py.File(path, "r+") as handle:
        handle["meta/state_fingerprint_sha256"][...] = np.bytes_("f" * 64)
    with pytest.raises(WilsonStateError, match="content fingerprint mismatch"):
        load_wilson_stationary_state(path)
