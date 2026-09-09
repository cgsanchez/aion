from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from aion.errors import SchemaError, SpectroscopyError
from aion.io.schemas import CASIDA_SCHEMA, KICK_SPECTRUM_SCHEMA, validate_artifact
from aion.io.util import file_sha256
from aion.spectroscopy import (
    BaselineKind,
    CasidaConfig,
    CasidaResult,
    KickSpectrumConfig,
    SpectrumSourceLink,
    TransformConfig,
    compute_kick_spectrum,
    load_casida_result,
    load_kick_spectrum,
    positive_frequency_transform,
    save_casida_result,
    save_kick_spectrum,
    select_resonance,
)

pytestmark = pytest.mark.fast


def _source(tmp_path: Path) -> SpectrumSourceLink:
    parent = tmp_path / "parent.dat"
    parent.write_bytes(b"immutable parent\n")
    return SpectrumSourceLink(
        kind="analytic_fixture",
        path=parent,
        sha256=file_sha256(parent),
        identifier="sinusoid-v1",
        description="analytic damped sinusoid",
    )


def _analytic_spectrum(tmp_path: Path, *, padding: int = 4) -> object:
    time = np.linspace(0.0, 80.0, 1601)
    omega0 = 0.7
    dipole = np.zeros((time.size, 3))
    current = np.zeros_like(dipole)
    dipole[:, 2] = np.sin(omega0 * time)
    current[:, 2] = omega0 * np.cos(omega0 * time)
    return compute_kick_spectrum(
        time_au=time,
        dipole_au=dipole,
        current_au=current,
        impulse_au=np.array([0.0, 0.0, 0.01]),
        baseline_dipole_au=np.zeros(3),
        config=KickSpectrumConfig(
            transform=TransformConfig(
                damping_energy_au=0.04,
                zero_padding_factor=padding,
                maximum_energy_au=2.0,
            ),
            baseline=BaselineKind.NONE,
        ),
        source=_source(tmp_path),
        event_id="analytic-kick",
        dipole_definition_id="dipole.electronic.analytic",
        current_definition_id="current.primary.analytic",
    )


def test_positive_frequency_transform_matches_declared_trapezoidal_integral() -> None:
    time = np.linspace(3.25, 23.25, 401)
    signal = np.column_stack(
        (
            np.sin(0.73 * (time - time[0])),
            np.cos(1.11 * (time - time[0])) + 0.2,
        )
    )
    config = TransformConfig(
        damping_energy_au=0.031,
        zero_padding_factor=3,
        maximum_energy_au=3.0,
    )
    result = positive_frequency_transform(time, signal, config)
    indices = np.array([0, 7, 17, result.omega_au.size - 1])
    tau = time - time[0]
    for index in indices:
        omega = result.omega_au[index]
        integrand = (
            signal
            * np.exp(-config.damping_energy_au * tau)[:, None]
            * np.exp(1j * omega * tau)[:, None]
        )
        direct = np.trapezoid(integrand, x=time, axis=0)
        assert result.values[index] == pytest.approx(direct, rel=2.0e-13, abs=2.0e-13)


def test_padding_interpolates_but_does_not_change_intrinsic_resolution() -> None:
    time = np.linspace(0.0, 60.0, 601)
    signal = np.sin(0.61 * time)
    native = positive_frequency_transform(time, signal, TransformConfig())
    padded = positive_frequency_transform(
        time,
        signal,
        TransformConfig(zero_padding_factor=8),
    )
    assert padded.intrinsic_resolution_au == native.intrinsic_resolution_au
    assert padded.native_grid_spacing_au == native.native_grid_spacing_au
    assert padded.padded_grid_spacing_au == pytest.approx(native.padded_grid_spacing_au / 8.0)
    assert padded.nyquist_energy_au == native.nyquist_energy_au
    peak = padded.omega_au[np.argmax(padded.values.imag)]
    assert peak == pytest.approx(0.61, abs=padded.padded_grid_spacing_au)


def test_exponential_damping_has_declared_lorentzian_fwhm() -> None:
    time = np.linspace(0.0, 800.0, 8001)
    eta = 0.025
    result = positive_frequency_transform(
        time,
        np.sin(0.8 * time),
        TransformConfig(
            damping_energy_au=eta,
            zero_padding_factor=8,
            maximum_energy_au=1.2,
        ),
    )
    profile = result.values.imag
    peak = int(np.argmax(profile))
    half = profile[peak] / 2.0
    left = np.flatnonzero(profile[:peak] <= half)[-1]
    right = peak + np.flatnonzero(profile[peak:] <= half)[0]
    width = result.omega_au[right] - result.omega_au[left]
    assert width == pytest.approx(2.0 * eta, abs=2.5 * result.padded_grid_spacing_au)


def test_current_domain_identity_includes_both_finite_record_endpoints(
    tmp_path: Path,
) -> None:
    spectrum = _analytic_spectrum(tmp_path)
    assert spectrum.polarizability_current_au is not None
    assert spectrum.current_domain_residual_au is not None
    peak = int(np.argmax(spectrum.absorption_strength_parallel_au))
    window = slice(max(0, peak - 4), peak + 5)
    residual = np.linalg.norm(spectrum.current_domain_residual_au[window, 2])
    signal = np.linalg.norm(spectrum.polarizability_dipole_au[window, 2])
    assert residual / signal < 2.0e-4


def test_kick_spectrum_round_trip_is_immutable_and_parent_qualified(tmp_path: Path) -> None:
    spectrum = _analytic_spectrum(tmp_path)
    path = tmp_path / "spectrum.h5"
    checksum = save_kick_spectrum(spectrum, path)
    assert checksum == file_sha256(path)
    header = validate_artifact(path, expected_schema=KICK_SPECTRUM_SCHEMA)
    loaded = load_kick_spectrum(path)
    assert header.artifact_id == loaded.result_id == spectrum.result_id
    assert loaded.source.sha256 == file_sha256(spectrum.source.path)
    assert loaded.source.metadata_json == spectrum.source.metadata_json
    assert loaded.config.as_mapping() == spectrum.config.as_mapping()
    assert np.array_equal(loaded.polarizability_dipole_au, spectrum.polarizability_dipole_au)
    assert np.array_equal(
        loaded.polarizability_current_au,
        spectrum.polarizability_current_au,
    )
    with pytest.raises(SchemaError, match="overwrite"):
        save_kick_spectrum(spectrum, path)
    with h5py.File(path, "r+") as handle:
        handle["response/absorption_strength_parallel_au"][3] += 1.0e-4
    with pytest.raises(SpectroscopyError, match="absorption strength"):
        load_kick_spectrum(path)


def _casida_fixture() -> CasidaResult:
    energy = np.array([0.20, 0.35, 0.70])
    dipole = np.array([[0.0, 0.0, 0.5], [0.4, 0.0, 0.0], [0.0, 0.2, 0.0]])
    strength = (2.0 / 3.0) * energy * np.sum(dipole * dipole, axis=1)
    direction = dipole / np.linalg.norm(dipole, axis=1)[:, None]
    return CasidaResult(
        config=CasidaConfig(nstates=3),
        reference_fingerprint_sha256="a" * 64,
        method="analytic Casida fixture",
        pyscf_version="fixture",
        excitation_energy_au=energy,
        oscillator_strength=strength,
        transition_dipole_au=dipole,
        transition_direction=direction,
        converged=np.ones(3, dtype=np.bool_),
    )


def test_polarized_resonance_selection_and_casida_round_trip(tmp_path: Path) -> None:
    result = _casida_fixture()
    selection = select_resonance(
        result,
        [1.0, 0.0, 0.0],
        brightness_threshold=1.0e-4,
    )
    assert selection.selected_root == 1
    explicit = select_resonance(result, [0.0, 0.0, 2.0], root_index=0)
    assert explicit.user_selected
    reference = tmp_path / "reference.h5"
    reference.write_bytes(b"reference parent\n")
    path = tmp_path / "casida.h5"
    checksum = save_casida_result(
        result,
        path,
        selection=selection,
        reference_artifact=reference,
    )
    assert checksum == file_sha256(path)
    validate_artifact(path, expected_schema=CASIDA_SCHEMA)
    loaded, loaded_selection = load_casida_result(path)
    assert loaded.result_id == result.result_id
    assert loaded_selection is not None
    assert loaded_selection.as_mapping() == selection.as_mapping()
    with h5py.File(path, "r") as handle:
        assert handle["reference/artifact_sha256"][()].decode() == file_sha256(reference)
    with pytest.raises(SchemaError, match="overwrite"):
        save_casida_result(result, path, selection=selection)


def test_invalid_transform_and_ambiguous_resonance_policies_fail() -> None:
    with pytest.raises(SpectroscopyError, match="uniform"):
        positive_frequency_transform([0.0, 0.1, 0.21], [0.0, 1.0, 0.0])
    result = _casida_fixture()
    with pytest.raises(SpectroscopyError, match="exactly one"):
        select_resonance(
            result,
            [0.0, 0.0, 1.0],
            brightness_threshold=0.0,
            root_index=0,
        )
