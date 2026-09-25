"""Transactional HDF5 persistence for immutable Wilson stationary states."""

from __future__ import annotations

from dataclasses import fields
from os import PathLike
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np

from aion.config import WilsonStationaryConfig, dumps_config, loads_config
from aion.config.units import vector3
from aion.electromagnetism import (
    MagneticGaugeKind,
    UniformMagneticField,
    UniformMagneticSourceSample,
)
from aion.electronic_structure.wilson_state import (
    WilsonStationaryEnergyComponents,
    WilsonStationaryResiduals,
    WilsonStationaryStateData,
)
from aion.electronic_structure.wilson_stationary import StationarySCFIteration
from aion.errors import SchemaError, WilsonStateError
from aion.io import (
    WILSON_STATIONARY_STATE_SCHEMA,
    publish_hdf5,
    validate_artifact,
    write_dataset,
)

type PathInput = str | PathLike[str]

_ARRAY_DATASETS = {
    "coefficients": ("1", "ao_occupied_coefficients"),
    "occupations": ("electron", "orbital_occupation"),
    "contravariant_density": ("electron", "contravariant_ao_density_matrix"),
    "mixed_density": ("electron", "mixed_ao_density_matrix"),
    "metric": ("1", "overlap"),
    "orbital_frequency_matrix": ("hartree", "occupied_orbital_frequency_matrix"),
    "active_orbital_energies_au": ("hartree", "energy"),
    "complete_orbital_spectrum_au": ("hartree", "energy"),
    "occupation_spectrum": ("electron", "mixed_density_occupation_spectrum"),
}


def _write_text(group: h5py.Group, name: str, value: str, *, dimension: str) -> None:
    write_dataset(
        group,
        name,
        np.bytes_(value),
        unit="1",
        physical_dimension=dimension,
    )


def _text(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    if isinstance(value, bytes | np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def save_wilson_stationary_state(
    state: WilsonStationaryStateData,
    path: PathInput,
) -> None:
    """Publish one complete immutable density-native stationary artifact."""

    if not isinstance(state, WilsonStationaryStateData):
        raise TypeError("state must be a WilsonStationaryStateData")

    def populate(handle: h5py.File) -> None:
        meta = handle["meta"]
        configuration = handle["configuration"]
        reference = handle["reference"]
        source = handle["source"]
        state_group = handle["state"]
        observables = handle["observables"]
        diagnostics = handle["diagnostics"]
        for group in (
            meta,
            configuration,
            reference,
            source,
            state_group,
            observables,
            diagnostics,
        ):
            assert isinstance(group, h5py.Group)

        _write_text(
            meta,
            "state_fingerprint_sha256",
            state.fingerprint_sha256,
            dimension="sha256_digest",
        )
        _write_text(
            meta,
            "action_fingerprint_sha256",
            state.action_fingerprint_sha256,
            dimension="sha256_digest",
        )
        _write_text(
            configuration,
            "normalized_toml",
            dumps_config(state.config),
            dimension="configuration",
        )
        _write_text(
            configuration,
            "scientific_id_sha256",
            state.config.scientific_id,
            dimension="sha256_digest",
        )
        for name, value in (
            ("reference_fingerprint_sha256", state.reference_fingerprint_sha256),
            ("grid_fingerprint_sha256", state.grid_fingerprint_sha256),
            (
                "auxiliary_space_fingerprint_sha256",
                state.auxiliary_space_fingerprint_sha256,
            ),
        ):
            _write_text(reference, name, value, dimension="sha256_digest")

        _write_text(
            source,
            "source_fingerprint_sha256",
            state.source_fingerprint_sha256,
            dimension="sha256_digest",
        )
        sample = state.source_sample
        source.attrs["gauge_kind"] = sample.gauge_kind.value
        source.attrs["has_landau_axis"] = np.uint8(sample.landau_axis is not None)
        write_dataset(
            source,
            "time_au",
            sample.time_au,
            unit="atomic_unit_of_time",
            physical_dimension="time",
        )
        for source_name, source_value, source_dimension in (
            ("magnetic_field_au", sample.field.magnetic_field_au, "magnetic_field"),
            ("magnetic_field_dot_au", sample.magnetic_field_dot_au, "magnetic_field_rate"),
            (
                "electric_field_origin_au",
                sample.electric_field_origin_au,
                "electric_field",
            ),
            ("origin_au", sample.origin_au, "length"),
        ):
            write_dataset(
                source,
                source_name,
                np.asarray(source_value, dtype=np.float64),
                unit=(
                    "bohr" if source_name == "origin_au" else "atomic_unit_of_" + source_dimension
                ),
                physical_dimension=source_dimension,
            )
        if sample.landau_axis is not None:
            write_dataset(
                source,
                "landau_axis",
                np.asarray(sample.landau_axis, dtype=np.float64),
                unit="1",
                physical_dimension="direction",
            )

        for array_name, (array_unit, array_dimension) in _ARRAY_DATASETS.items():
            write_dataset(
                state_group,
                array_name,
                getattr(state, array_name),
                unit=array_unit,
                physical_dimension=array_dimension,
            )

        energy = observables.create_group("energy_components")
        for energy_name, energy_value in state.energies.as_mapping().items():
            write_dataset(
                energy,
                energy_name,
                energy_value,
                unit="hartree",
                physical_dimension="energy",
            )

        residuals = diagnostics.create_group("stationary_residuals")
        for residual_name, residual_value in state.residuals.as_mapping().items():
            if residual_name == "converged":
                write_dataset(
                    residuals,
                    residual_name,
                    np.uint8(residual_value),
                    unit="1",
                    physical_dimension="boolean",
                )
            else:
                residual_unit = "hartree" if residual_name.endswith("_au") else "1"
                residual_dimension = (
                    "energy" if residual_name.endswith("_au") else "relative_residual"
                )
                write_dataset(
                    residuals,
                    residual_name,
                    residual_value,
                    unit=residual_unit,
                    physical_dimension=residual_dimension,
                )

        iterations = diagnostics.create_group("iterations")
        records = state.iterations
        write_dataset(
            iterations,
            "iteration",
            np.asarray([item.iteration for item in records], dtype=np.int64),
            unit="1",
            physical_dimension="iteration_index",
        )
        write_dataset(
            iterations,
            "energy_molecular_total_au",
            np.asarray([item.energy_molecular_total_au for item in records]),
            unit="hartree",
            physical_dimension="energy",
        )
        energy_changes = np.asarray(
            [np.nan if item.energy_change_au is None else item.energy_change_au for item in records]
        )
        write_dataset(
            iterations,
            "energy_change_au",
            energy_changes,
            unit="hartree",
            physical_dimension="energy",
        )
        write_dataset(
            iterations,
            "has_energy_change",
            np.asarray([item.energy_change_au is not None for item in records], dtype=np.uint8),
            unit="1",
            physical_dimension="boolean",
        )
        for name in ("density_fixed_point_residual", "commutator_residual"):
            write_dataset(
                iterations,
                name,
                np.asarray([getattr(item, name) for item in records]),
                unit="1",
                physical_dimension="relative_residual",
            )
        write_dataset(
            iterations,
            "diis_dimension",
            np.asarray([item.diis_dimension for item in records], dtype=np.int64),
            unit="1",
            physical_dimension="subspace_dimension",
        )
        write_dataset(
            iterations,
            "used_diis",
            np.asarray([item.used_diis for item in records], dtype=np.uint8),
            unit="1",
            physical_dimension="boolean",
        )

    publish_hdf5(
        Path(path),
        WILSON_STATIONARY_STATE_SCHEMA,
        artifact_id=state.fingerprint_sha256,
        populate=populate,
    )


def load_wilson_stationary_state(path: PathInput) -> WilsonStationaryStateData:
    """Load and reauthenticate one density-native Wilson stationary state."""

    artifact_path = Path(path)
    header = validate_artifact(
        artifact_path,
        expected_schema=WILSON_STATIONARY_STATE_SCHEMA,
        require_complete=True,
    )
    with h5py.File(artifact_path, "r") as handle:
        normalized = _required_text(handle, "configuration/normalized_toml")
        resolved = loads_config(normalized)
        if not isinstance(resolved.config, WilsonStationaryConfig):
            raise SchemaError("Wilson stationary artifact contains the wrong configuration type")
        recorded_config_id = _required_text(handle, "configuration/scientific_id_sha256")
        if recorded_config_id != resolved.config.scientific_id:
            raise WilsonStateError("stationary configuration scientific ID mismatch")
        source = handle["source"]
        assert isinstance(source, h5py.Group)
        try:
            gauge_kind = MagneticGaugeKind(str(source.attrs["gauge_kind"]))
            has_landau_axis = bool(int(source.attrs["has_landau_axis"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise SchemaError("Wilson stationary source metadata is incomplete") from exc
        landau_axis = (
            vector3(
                _required_array(handle, "source/landau_axis").tolist(),
                "source.landau_axis",
            )
            if has_landau_axis
            else None
        )
        sample = UniformMagneticSourceSample(
            time_au=float(cast(Any, _required_scalar(handle, "source/time_au"))),
            field=UniformMagneticField(
                vector3(
                    _required_array(handle, "source/magnetic_field_au").tolist(),
                    "source.magnetic_field_au",
                )
            ),
            magnetic_field_dot_au=vector3(
                _required_array(handle, "source/magnetic_field_dot_au").tolist(),
                "source.magnetic_field_dot_au",
            ),
            electric_field_origin_au=vector3(
                _required_array(handle, "source/electric_field_origin_au").tolist(),
                "source.electric_field_origin_au",
            ),
            origin_au=vector3(
                _required_array(handle, "source/origin_au").tolist(),
                "source.origin_au",
            ),
            gauge_kind=gauge_kind,
            landau_axis=landau_axis,
        )
        energy_values = {
            item.name: float(
                cast(
                    Any,
                    _required_scalar(handle, f"observables/energy_components/{item.name}"),
                )
            )
            for item in fields(WilsonStationaryEnergyComponents)
        }
        residual_values: dict[str, float | bool] = {}
        for item in fields(WilsonStationaryResiduals):
            value = _required_scalar(handle, f"diagnostics/stationary_residuals/{item.name}")
            residual_values[item.name] = (
                bool(int(cast(Any, value))) if item.name == "converged" else float(cast(Any, value))
            )
        iterations = _load_iterations(handle)
        state = WilsonStationaryStateData(
            config=resolved.config,
            reference_fingerprint_sha256=_required_text(
                handle,
                "reference/reference_fingerprint_sha256",
            ),
            grid_fingerprint_sha256=_required_text(
                handle,
                "reference/grid_fingerprint_sha256",
            ),
            auxiliary_space_fingerprint_sha256=_required_text(
                handle,
                "reference/auxiliary_space_fingerprint_sha256",
            ),
            source_sample=sample,
            coefficients=_required_array(handle, "state/coefficients"),
            occupations=_required_array(handle, "state/occupations"),
            contravariant_density=_required_array(handle, "state/contravariant_density"),
            mixed_density=_required_array(handle, "state/mixed_density"),
            metric=_required_array(handle, "state/metric"),
            orbital_frequency_matrix=_required_array(
                handle,
                "state/orbital_frequency_matrix",
            ),
            active_orbital_energies_au=_required_array(
                handle,
                "state/active_orbital_energies_au",
            ),
            complete_orbital_spectrum_au=_required_array(
                handle,
                "state/complete_orbital_spectrum_au",
            ),
            occupation_spectrum=_required_array(handle, "state/occupation_spectrum"),
            energies=WilsonStationaryEnergyComponents(**cast(Any, energy_values)),
            residuals=WilsonStationaryResiduals(**cast(Any, residual_values)),
            iterations=iterations,
        )
        recorded_state = _required_text(handle, "meta/state_fingerprint_sha256")
        recorded_action = _required_text(handle, "meta/action_fingerprint_sha256")
        recorded_source = _required_text(handle, "source/source_fingerprint_sha256")
    if state.fingerprint_sha256 != recorded_state:
        raise WilsonStateError("Wilson stationary state content fingerprint mismatch")
    if state.action_fingerprint_sha256 != recorded_action:
        raise WilsonStateError("Wilson stationary action fingerprint mismatch")
    if state.source_fingerprint_sha256 != recorded_source:
        raise WilsonStateError("Wilson stationary source fingerprint mismatch")
    if header.artifact_id != state.fingerprint_sha256:
        raise WilsonStateError("artifact ID does not authenticate the Wilson stationary state")
    return state


def _load_iterations(handle: h5py.File) -> tuple[StationarySCFIteration, ...]:
    root = "diagnostics/iterations"
    iteration = _required_array(handle, f"{root}/iteration")
    energy = _required_array(handle, f"{root}/energy_molecular_total_au")
    change = _required_array(handle, f"{root}/energy_change_au")
    has_change = _required_array(handle, f"{root}/has_energy_change")
    density = _required_array(handle, f"{root}/density_fixed_point_residual")
    commutator = _required_array(handle, f"{root}/commutator_residual")
    diis_dimension = _required_array(handle, f"{root}/diis_dimension")
    used_diis = _required_array(handle, f"{root}/used_diis")
    count = len(iteration)
    values = (energy, change, has_change, density, commutator, diis_dimension, used_diis)
    if any(len(value) != count for value in values):
        raise SchemaError("Wilson stationary iteration arrays have different lengths")
    return tuple(
        StationarySCFIteration(
            iteration=int(iteration[index]),
            energy_molecular_total_au=float(energy[index]),
            energy_change_au=(float(change[index]) if bool(has_change[index]) else None),
            density_fixed_point_residual=float(density[index]),
            commutator_residual=float(commutator[index]),
            diis_dimension=int(diis_dimension[index]),
            used_diis=bool(used_diis[index]),
        )
        for index in range(count)
    )


def _required_text(handle: h5py.File, path: str) -> str:
    if path not in handle or not isinstance(handle[path], h5py.Dataset):
        raise SchemaError(f"Wilson stationary artifact lacks dataset /{path}")
    return _text(handle[path])


def _required_array(handle: h5py.File, path: str) -> np.ndarray:
    if path not in handle or not isinstance(handle[path], h5py.Dataset):
        raise SchemaError(f"Wilson stationary artifact lacks dataset /{path}")
    return np.asarray(handle[path][...])


def _required_scalar(handle: h5py.File, path: str) -> object:
    if path not in handle or not isinstance(handle[path], h5py.Dataset):
        raise SchemaError(f"Wilson stationary artifact lacks dataset /{path}")
    value = handle[path][()]
    if np.asarray(value).shape != ():
        raise SchemaError(f"Wilson stationary dataset /{path} must be scalar")
    return value
