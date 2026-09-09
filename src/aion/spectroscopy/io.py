"""Immutable HDF5 persistence for Casida and kick-spectrum results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from aion.config import canonical_sha256
from aion.errors import SpectroscopyError
from aion.io.schemas import CASIDA_SCHEMA, KICK_SPECTRUM_SCHEMA, validate_artifact
from aion.io.transaction import publish_hdf5, write_dataset
from aion.io.util import file_sha256, read_text, write_text
from aion.spectroscopy.types import (
    BaselineKind,
    CasidaConfig,
    CasidaResult,
    KickSpectrum,
    KickSpectrumConfig,
    QuadratureKind,
    ResonanceSelection,
    SpectrumSourceLink,
    TransformConfig,
    WindowKind,
    configuration_json,
)


def _mapping(text: str, expected_schema: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpectroscopyError("spectroscopy configuration JSON is invalid") from exc
    if not isinstance(value, dict) or value.get("schema") != expected_schema:
        raise SpectroscopyError(f"expected {expected_schema!r} configuration")
    return value


def _transform_config(value: dict[str, Any]) -> TransformConfig:
    if value.get("schema") != "aion.transform-config":
        raise SpectroscopyError("kick spectrum lacks its transform configuration")
    config = TransformConfig(
        damping_energy_au=value["damping_energy_au"],
        zero_padding_factor=value["zero_padding_factor"],
        maximum_energy_au=value["maximum_energy_au"],
        window=WindowKind(value["window"]),
        quadrature=QuadratureKind(value["quadrature"]),
    )
    if config.as_mapping() != value:
        raise SpectroscopyError("stored transform configuration is inconsistent")
    return config


def save_casida_result(
    result: CasidaResult,
    path: str | Path,
    *,
    selection: ResonanceSelection | None = None,
    reference_artifact: str | Path | None = None,
) -> str:
    """Publish structured Casida roots without overwriting an existing result."""

    if selection is not None and selection.casida_result_id != result.result_id:
        raise SpectroscopyError("resonance selection belongs to a different Casida result")
    reference_path: Path | None = None
    reference_sha256: str | None = None
    if reference_artifact is not None:
        reference_path = Path(reference_artifact).expanduser().resolve()
        if not reference_path.is_file():
            raise SpectroscopyError(f"reference artifact does not exist: {reference_path}")
        reference_sha256 = file_sha256(reference_path)
    artifact_id = canonical_sha256(
        {
            "schema": "aion.casida-artifact-content",
            "version": "1.0.0",
            "result_id": result.result_id,
            "selection": None if selection is None else selection.as_mapping(),
            "reference_artifact_sha256": reference_sha256,
        }
    )

    def populate(handle: h5py.File) -> None:
        meta = handle["meta"]
        configuration = handle["configuration"]
        reference = handle["reference"]
        roots = handle["roots"]
        selected = handle["selection"]
        for group in (meta, configuration, reference, roots, selected):
            assert isinstance(group, h5py.Group)
        write_text(meta, "result_id", result.result_id, physical_dimension="sha256_digest")
        write_text(meta, "method", result.method, physical_dimension="method_description")
        write_text(meta, "pyscf_version", result.pyscf_version, physical_dimension="version")
        write_text(
            configuration,
            "analysis_json",
            configuration_json(result.config),
            physical_dimension="analysis_configuration",
        )
        write_text(
            reference,
            "fingerprint_sha256",
            result.reference_fingerprint_sha256,
            physical_dimension="sha256_digest",
        )
        if reference_path is not None and reference_sha256 is not None:
            write_text(
                reference,
                "artifact_path",
                str(reference_path),
                physical_dimension="filesystem_path",
            )
            write_text(
                reference,
                "artifact_sha256",
                reference_sha256,
                physical_dimension="sha256_digest",
            )
        nroot = result.root_count
        write_dataset(
            roots,
            "index",
            np.arange(nroot, dtype=np.int64),
            unit="1",
            physical_dimension="zero_based_root_index",
        )
        write_dataset(
            roots,
            "excitation_energy_au",
            result.excitation_energy_au,
            unit="hartree",
            physical_dimension="energy",
        )
        write_dataset(
            roots,
            "oscillator_strength",
            result.oscillator_strength,
            unit="1",
            physical_dimension="dimensionless_oscillator_strength",
        )
        write_dataset(
            roots,
            "transition_dipole_au",
            result.transition_dipole_au,
            unit="elementary_charge_times_bohr",
            physical_dimension="transition_dipole",
        )
        write_dataset(
            roots,
            "transition_direction",
            result.transition_direction,
            unit="1",
            physical_dimension="cartesian_unit_direction",
        )
        write_dataset(
            roots,
            "converged",
            result.converged.astype(np.uint8),
            unit="1",
            physical_dimension="boolean",
        )
        selected.attrs["present"] = np.uint8(selection is not None)
        if selection is not None:
            write_dataset(
                selected,
                "polarization",
                selection.polarization,
                unit="1",
                physical_dimension="cartesian_unit_direction",
            )
            write_dataset(
                selected,
                "projected_oscillator_strength",
                selection.projected_oscillator_strength,
                unit="1",
                physical_dimension="polarized_oscillator_strength",
            )
            write_dataset(
                selected,
                "selected_root",
                np.int64(selection.selected_root),
                unit="1",
                physical_dimension="zero_based_root_index",
            )
            selected.attrs["user_selected"] = np.uint8(selection.user_selected)
            selected.attrs["brightness_threshold_present"] = np.uint8(
                selection.brightness_threshold is not None
            )
            if selection.brightness_threshold is not None:
                write_dataset(
                    selected,
                    "brightness_threshold",
                    np.float64(selection.brightness_threshold),
                    unit="1",
                    physical_dimension="polarized_oscillator_strength",
                )

    publish_hdf5(path, CASIDA_SCHEMA, artifact_id=artifact_id, populate=populate)
    return file_sha256(path)


def load_casida_result(path: str | Path) -> tuple[CasidaResult, ResonanceSelection | None]:
    """Load and content-authenticate a structured Casida artifact."""

    target = Path(path)
    header = validate_artifact(target, expected_schema=CASIDA_SCHEMA)
    with h5py.File(target, "r") as handle:
        config_data = _mapping(
            read_text(handle["configuration/analysis_json"]), "aion.casida-config"
        )
        config = CasidaConfig(
            nstates=config_data["nstates"],
            convergence_tolerance=config_data["convergence_tolerance"],
            singlet=config_data["singlet"],
        )
        if config.as_mapping() != config_data:
            raise SpectroscopyError("stored Casida configuration is inconsistent")
        result = CasidaResult(
            config=config,
            reference_fingerprint_sha256=read_text(handle["reference/fingerprint_sha256"]),
            method=read_text(handle["meta/method"]),
            pyscf_version=read_text(handle["meta/pyscf_version"]),
            excitation_energy_au=handle["roots/excitation_energy_au"][...],
            oscillator_strength=handle["roots/oscillator_strength"][...],
            transition_dipole_au=handle["roots/transition_dipole_au"][...],
            transition_direction=handle["roots/transition_direction"][...],
            converged=np.asarray(handle["roots/converged"][...], dtype=np.bool_),
        )
        if read_text(handle["meta/result_id"]) != result.result_id:
            raise SpectroscopyError("Casida numerical content fingerprint mismatch")
        selection_group = handle["selection"]
        assert isinstance(selection_group, h5py.Group)
        selection: ResonanceSelection | None = None
        if bool(int(selection_group.attrs.get("present", 0))):
            threshold = (
                float(selection_group["brightness_threshold"][()])
                if bool(int(selection_group.attrs.get("brightness_threshold_present", 0)))
                else None
            )
            selection = ResonanceSelection(
                casida_result_id=result.result_id,
                polarization=selection_group["polarization"][...],
                projected_oscillator_strength=selection_group["projected_oscillator_strength"][...],
                selected_root=int(selection_group["selected_root"][()]),
                brightness_threshold=threshold,
                user_selected=bool(int(selection_group.attrs.get("user_selected", 0))),
            )
        reference_sha256 = (
            read_text(handle["reference/artifact_sha256"])
            if "artifact_sha256" in handle["reference"]
            else None
        )
    artifact_id = canonical_sha256(
        {
            "schema": "aion.casida-artifact-content",
            "version": "1.0.0",
            "result_id": result.result_id,
            "selection": None if selection is None else selection.as_mapping(),
            "reference_artifact_sha256": reference_sha256,
        }
    )
    if artifact_id != header.artifact_id:
        raise SpectroscopyError("Casida artifact content fingerprint mismatch")
    return result, selection


def save_kick_spectrum(spectrum: KickSpectrum, path: str | Path) -> str:
    """Publish one immutable spectrum linked to its exact parent checksum."""

    if not spectrum.source.path.is_file():
        raise SpectroscopyError(f"spectrum parent does not exist: {spectrum.source.path}")
    if file_sha256(spectrum.source.path) != spectrum.source.sha256:
        raise SpectroscopyError("spectrum parent changed after analysis")

    def populate(handle: h5py.File) -> None:
        meta = handle["meta"]
        configuration = handle["configuration"]
        source = handle["source"]
        time = handle["time"]
        frequency = handle["frequency"]
        response = handle["response"]
        for group in (meta, configuration, source, time, frequency, response):
            assert isinstance(group, h5py.Group)
        write_text(meta, "result_id", spectrum.result_id, physical_dimension="sha256_digest")
        write_text(
            configuration,
            "analysis_json",
            configuration_json(spectrum.config),
            physical_dimension="analysis_configuration",
        )
        for text_name, text_value, text_dimension in (
            ("kind", spectrum.source.kind, "source_artifact_kind"),
            ("path", str(spectrum.source.path), "filesystem_path"),
            ("sha256", spectrum.source.sha256, "sha256_digest"),
            ("identifier", spectrum.source.identifier, "source_identifier"),
            ("description", spectrum.source.description, "description"),
            ("metadata_json", spectrum.source.metadata_json, "source_metadata"),
            ("event_id", spectrum.event_id, "event_identifier"),
            ("dipole_definition_id", spectrum.dipole_definition_id, "observable_identifier"),
        ):
            write_text(
                source,
                text_name,
                text_value,
                physical_dimension=text_dimension,
            )
        if spectrum.current_definition_id is not None:
            write_text(
                source,
                "current_definition_id",
                spectrum.current_definition_id,
                physical_dimension="observable_identifier",
            )
        for scalar_name, scalar_value, scalar_unit, scalar_dimension in (
            ("sample_count", spectrum.sample_count, "1", "sample_count"),
            ("nfft", spectrum.nfft, "1", "fft_length"),
            ("start_time_au", spectrum.start_time_au, "atomic_unit_of_time", "time"),
            ("duration_au", spectrum.duration_au, "atomic_unit_of_time", "duration"),
            ("time_step_au", spectrum.time_step_au, "atomic_unit_of_time", "time_step"),
        ):
            write_dataset(
                time,
                scalar_name,
                scalar_value,
                unit=scalar_unit,
                physical_dimension=scalar_dimension,
            )
        for frequency_name, frequency_value, frequency_dimension in (
            ("omega_au", spectrum.omega_au, "angular_frequency_energy"),
            ("intrinsic_resolution_au", spectrum.intrinsic_resolution_au, "energy_resolution"),
            ("native_grid_spacing_au", spectrum.native_grid_spacing_au, "frequency_grid_spacing"),
            ("padded_grid_spacing_au", spectrum.padded_grid_spacing_au, "frequency_grid_spacing"),
            ("nyquist_energy_au", spectrum.nyquist_energy_au, "nyquist_energy"),
        ):
            write_dataset(
                frequency,
                frequency_name,
                frequency_value,
                unit="hartree",
                physical_dimension=frequency_dimension,
            )
        for response_name, response_value, response_unit, response_dimension in (
            (
                "impulse_au",
                spectrum.impulse_au,
                "atomic_unit_of_momentum",
                "electric_field_impulse",
            ),
            ("input_polarization", spectrum.input_polarization, "1", "cartesian_unit_direction"),
            (
                "impulse_amplitude_au",
                spectrum.impulse_amplitude_au,
                "atomic_unit_of_momentum",
                "electric_field_impulse",
            ),
            (
                "baseline_dipole_au",
                spectrum.baseline_dipole_au,
                "elementary_charge_times_bohr",
                "dipole",
            ),
            (
                "polarizability_dipole_au",
                spectrum.polarizability_dipole_au,
                "bohr_cubed",
                "polarizability",
            ),
            (
                "polarizability_parallel_au",
                spectrum.polarizability_parallel_au,
                "bohr_cubed",
                "polarizability",
            ),
            (
                "absorption_strength_parallel_au",
                spectrum.absorption_strength_parallel_au,
                "hartree_times_bohr_cubed",
                "directional_absorption_strength",
            ),
            (
                "oscillator_strength_density_parallel_au",
                spectrum.oscillator_strength_density_parallel_au,
                "hartree_inverse",
                "oscillator_strength_density",
            ),
        ):
            write_dataset(
                response,
                response_name,
                response_value,
                unit=response_unit,
                physical_dimension=response_dimension,
            )
        response.attrs["current_domain_present"] = np.uint8(
            spectrum.polarizability_current_au is not None
        )
        if (
            spectrum.polarizability_current_au is not None
            and spectrum.current_domain_residual_au is not None
        ):
            write_dataset(
                response,
                "polarizability_current_au",
                spectrum.polarizability_current_au,
                unit="bohr_cubed",
                physical_dimension="polarizability",
            )
            write_dataset(
                response,
                "current_domain_residual_au",
                spectrum.current_domain_residual_au,
                unit="bohr_cubed",
                physical_dimension="polarizability_residual",
            )

    publish_hdf5(
        path,
        KICK_SPECTRUM_SCHEMA,
        artifact_id=spectrum.result_id,
        populate=populate,
    )
    return file_sha256(path)


def load_kick_spectrum(path: str | Path) -> KickSpectrum:
    """Load and content-authenticate an immutable kick-spectrum artifact."""

    target = Path(path)
    header = validate_artifact(target, expected_schema=KICK_SPECTRUM_SCHEMA)
    with h5py.File(target, "r") as handle:
        config_data = _mapping(
            read_text(handle["configuration/analysis_json"]),
            "aion.kick-spectrum-config",
        )
        config = KickSpectrumConfig(
            transform=_transform_config(config_data["transform"]),
            baseline=BaselineKind(config_data["baseline"]),
        )
        if config.as_mapping() != config_data:
            raise SpectroscopyError("stored kick-spectrum configuration is inconsistent")
        source = SpectrumSourceLink(
            kind=read_text(handle["source/kind"]),
            path=Path(read_text(handle["source/path"])),
            sha256=read_text(handle["source/sha256"]),
            identifier=read_text(handle["source/identifier"]),
            description=read_text(handle["source/description"]),
            metadata_json=read_text(handle["source/metadata_json"]),
        )
        response = handle["response"]
        assert isinstance(response, h5py.Group)
        current_present = bool(int(response.attrs.get("current_domain_present", 0)))
        spectrum = KickSpectrum(
            config=config,
            source=source,
            event_id=read_text(handle["source/event_id"]),
            dipole_definition_id=read_text(handle["source/dipole_definition_id"]),
            current_definition_id=(
                read_text(handle["source/current_definition_id"])
                if "current_definition_id" in handle["source"]
                else None
            ),
            impulse_au=response["impulse_au"][...],
            input_polarization=response["input_polarization"][...],
            impulse_amplitude_au=float(response["impulse_amplitude_au"][()]),
            baseline_dipole_au=response["baseline_dipole_au"][...],
            sample_count=int(handle["time/sample_count"][()]),
            nfft=int(handle["time/nfft"][()]),
            start_time_au=float(handle["time/start_time_au"][()]),
            duration_au=float(handle["time/duration_au"][()]),
            time_step_au=float(handle["time/time_step_au"][()]),
            intrinsic_resolution_au=float(handle["frequency/intrinsic_resolution_au"][()]),
            native_grid_spacing_au=float(handle["frequency/native_grid_spacing_au"][()]),
            padded_grid_spacing_au=float(handle["frequency/padded_grid_spacing_au"][()]),
            nyquist_energy_au=float(handle["frequency/nyquist_energy_au"][()]),
            omega_au=handle["frequency/omega_au"][...],
            polarizability_dipole_au=response["polarizability_dipole_au"][...],
            polarizability_current_au=(
                response["polarizability_current_au"][...] if current_present else None
            ),
            polarizability_parallel_au=response["polarizability_parallel_au"][...],
            absorption_strength_parallel_au=response["absorption_strength_parallel_au"][...],
            oscillator_strength_density_parallel_au=response[
                "oscillator_strength_density_parallel_au"
            ][...],
            current_domain_residual_au=(
                response["current_domain_residual_au"][...] if current_present else None
            ),
        )
        if read_text(handle["meta/result_id"]) != spectrum.result_id:
            raise SpectroscopyError("kick-spectrum numerical content fingerprint mismatch")
    if header.artifact_id != spectrum.result_id:
        raise SpectroscopyError("kick-spectrum artifact content fingerprint mismatch")
    return spectrum
