"""Versioned transactional HDF5 persistence for prepared references."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path

import h5py
import numpy as np

from aion.config import ReferenceConfig, dumps_config, loads_config
from aion.electronic_structure.data import (
    AnchorTopologyBundle,
    CoreOperatorBundle,
    DependencyVersions,
    GroundState,
    NuclearData,
    PreparedReference,
    QuadratureGrid,
)
from aion.errors import ReferencePreparationError, SchemaError
from aion.io import REFERENCE_SCHEMA, publish_hdf5, validate_artifact, write_dataset

type PathInput = str | PathLike[str]

_REQUIRED_REFERENCE_PATHS = (
    "meta/dependencies_json",
    "meta/preparation_backend",
    "configuration/normalized_toml",
    "configuration/scientific_id_sha256",
    "reference/ground_state/coefficients",
    "reference/ground_state/occupations",
    "reference/ground_state/orbital_energies_au",
    "reference/ground_state/density",
    "reference/ground_state/energy_total_au",
    "reference/ground_state/electron_count",
    "reference/grid/coordinates_au",
    "reference/grid/weights_au",
    "reference/operators/overlap",
    "reference/operators/kinetic",
    "reference/operators/nuclear_attraction",
    "reference/operators/position",
    "reference/operators/canonical_momentum",
    "reference/nuclei/symbols",
    "reference/nuclei/charges",
    "reference/nuclei/coordinates_au",
    "reference/anchors/ao_to_atom",
    "reference/anchors/pair_indices",
    "reference/anchors/pair_displacements_au",
    "reference/anchors/incidence",
    "reference/anchors/site_projector_diagonals",
)


def _write_text(group: h5py.Group, name: str, value: str, *, dimension: str) -> None:
    write_dataset(
        group,
        name,
        np.bytes_(value),
        unit="1",
        physical_dimension=dimension,
    )


def save_reference(reference: PreparedReference, path: PathInput) -> None:
    """Publish one immutable authenticated ``aion.reference`` artifact."""

    def populate(handle: h5py.File) -> None:
        meta = handle["meta"]
        configuration = handle["configuration"]
        root = handle["reference"]
        assert isinstance(meta, h5py.Group)
        assert isinstance(configuration, h5py.Group)
        assert isinstance(root, h5py.Group)
        _write_text(
            meta,
            "dependencies_json",
            json.dumps(reference.dependencies.as_mapping(), sort_keys=True, separators=(",", ":")),
            dimension="software_metadata",
        )
        _write_text(
            meta,
            "preparation_backend",
            reference.preparation_backend,
            dimension="execution_metadata",
        )
        _write_text(
            configuration,
            "normalized_toml",
            dumps_config(reference.config),
            dimension="configuration",
        )
        _write_text(
            configuration,
            "scientific_id_sha256",
            reference.config.scientific_id,
            dimension="sha256_digest",
        )
        root.attrs["reference_fingerprint_sha256"] = reference.fingerprint_sha256
        root.attrs["grid_fingerprint_sha256"] = reference.grid.fingerprint_sha256
        root.attrs["core_operator_fingerprint_sha256"] = reference.core_operators.fingerprint_sha256
        root.attrs["anchor_topology_fingerprint_sha256"] = (
            reference.anchor_topology.fingerprint_sha256
        )

        state = root.create_group("ground_state")
        write_dataset(
            state,
            "coefficients",
            reference.ground_state.coefficients,
            unit="1",
            physical_dimension="ao_mo_coefficients",
        )
        write_dataset(
            state,
            "occupations",
            reference.ground_state.occupations,
            unit="electron",
            physical_dimension="orbital_occupation",
        )
        write_dataset(
            state,
            "orbital_energies_au",
            reference.ground_state.orbital_energies_au,
            unit="hartree",
            physical_dimension="energy",
        )
        write_dataset(
            state,
            "density",
            reference.ground_state.density,
            unit="electron",
            physical_dimension="ao_density_matrix",
        )
        write_dataset(
            state,
            "energy_total_au",
            reference.ground_state.energy_total_au,
            unit="hartree",
            physical_dimension="energy",
        )
        write_dataset(
            state,
            "electron_count",
            reference.ground_state.electron_count,
            unit="electron",
            physical_dimension="particle_count",
        )

        grid = root.create_group("grid")
        grid.attrs["level"] = reference.grid.level
        grid.attrs["pruning"] = reference.grid.pruning
        write_dataset(
            grid,
            "coordinates_au",
            reference.grid.coordinates_au,
            unit="bohr",
            physical_dimension="length",
        )
        write_dataset(
            grid,
            "weights_au",
            reference.grid.weights_au,
            unit="bohr^3",
            physical_dimension="volume",
        )

        operators = root.create_group("operators")
        operators.attrs["momentum_convention"] = reference.core_operators.momentum_convention
        for name, value, unit, dimension in (
            ("overlap", reference.core_operators.overlap, "1", "overlap"),
            ("kinetic", reference.core_operators.kinetic, "hartree", "energy_operator"),
            (
                "nuclear_attraction",
                reference.core_operators.nuclear_attraction,
                "hartree",
                "energy_operator",
            ),
            ("position", reference.core_operators.position, "bohr", "length_operator"),
            (
                "canonical_momentum",
                reference.core_operators.canonical_momentum,
                "atomic_unit_of_momentum",
                "momentum_operator",
            ),
        ):
            write_dataset(operators, name, value, unit=unit, physical_dimension=dimension)

        nuclei = root.create_group("nuclei")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        write_dataset(
            nuclei,
            "symbols",
            np.asarray(reference.core_operators.nuclei.symbols, dtype=string_dtype),
            unit="1",
            physical_dimension="element_symbol",
        )
        write_dataset(
            nuclei,
            "charges",
            reference.core_operators.nuclei.charges,
            unit="elementary_charge",
            physical_dimension="charge",
        )
        write_dataset(
            nuclei,
            "coordinates_au",
            reference.core_operators.nuclei.coordinates_au,
            unit="bohr",
            physical_dimension="length",
        )

        anchors = root.create_group("anchors")
        for name, value, unit, dimension in (
            ("ao_to_atom", reference.anchor_topology.ao_to_atom, "1", "atom_index"),
            ("pair_indices", reference.anchor_topology.pair_indices, "1", "atom_index_pair"),
            (
                "pair_displacements_au",
                reference.anchor_topology.pair_displacements_au,
                "bohr",
                "length",
            ),
            ("incidence", reference.anchor_topology.incidence, "1", "graph_incidence"),
            (
                "site_projector_diagonals",
                reference.anchor_topology.site_projector_diagonals,
                "1",
                "ao_selector",
            ),
        ):
            write_dataset(anchors, name, value, unit=unit, physical_dimension=dimension)

    publish_hdf5(
        Path(path),
        REFERENCE_SCHEMA,
        artifact_id=reference.fingerprint_sha256,
        populate=populate,
    )


def _text(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    if isinstance(value, bytes | np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _attribute_text(group: h5py.Group, name: str) -> str:
    if name not in group.attrs:
        raise SchemaError(f"group {group.name} lacks required attribute {name!r}")
    value = group.attrs[name]
    return bytes(value).decode("utf-8") if isinstance(value, bytes | np.bytes_) else str(value)


def load_reference_data(path: PathInput) -> PreparedReference:
    """Load all portable data and verify every construction fingerprint."""

    artifact_path = Path(path)
    header = validate_artifact(
        artifact_path, expected_schema=REFERENCE_SCHEMA, require_complete=True
    )
    with h5py.File(artifact_path, "r") as handle:
        missing = [name for name in _REQUIRED_REFERENCE_PATHS if name not in handle]
        if missing:
            raise SchemaError("reference artifact lacks required object(s): " + ", ".join(missing))
        normalized = _text(handle["configuration/normalized_toml"])
        resolved = loads_config(normalized)
        if not isinstance(resolved.config, ReferenceConfig):
            raise SchemaError("reference artifact contains a non-reference configuration")
        expected_scientific_id = _text(handle["configuration/scientific_id_sha256"])
        if resolved.config.scientific_id != expected_scientific_id:
            raise ReferencePreparationError("reference configuration scientific ID mismatch")
        dependency_mapping = json.loads(_text(handle["meta/dependencies_json"]))
        dependencies = DependencyVersions(**dependency_mapping)
        state = GroundState(
            coefficients=handle["reference/ground_state/coefficients"][...],
            occupations=handle["reference/ground_state/occupations"][...],
            orbital_energies_au=handle["reference/ground_state/orbital_energies_au"][...],
            density=handle["reference/ground_state/density"][...],
            energy_total_au=float(handle["reference/ground_state/energy_total_au"][()]),
            electron_count=float(handle["reference/ground_state/electron_count"][()]),
        )
        grid_group = handle["reference/grid"]
        grid = QuadratureGrid(
            coordinates_au=grid_group["coordinates_au"][...],
            weights_au=grid_group["weights_au"][...],
            level=int(grid_group.attrs["level"]),
            pruning=_attribute_text(grid_group, "pruning"),
        )
        nuclei_group = handle["reference/nuclei"]
        symbol_values = nuclei_group["symbols"].asstr()[...]
        nuclei = NuclearData(
            symbols=tuple(str(value) for value in symbol_values),
            charges=nuclei_group["charges"][...],
            coordinates_au=nuclei_group["coordinates_au"][...],
        )
        operator_group = handle["reference/operators"]
        operators = CoreOperatorBundle(
            overlap=operator_group["overlap"][...],
            kinetic=operator_group["kinetic"][...],
            nuclear_attraction=operator_group["nuclear_attraction"][...],
            position=operator_group["position"][...],
            canonical_momentum=operator_group["canonical_momentum"][...],
            nuclei=nuclei,
            momentum_convention=_attribute_text(operator_group, "momentum_convention"),
        )
        anchor_group = handle["reference/anchors"]
        anchors = AnchorTopologyBundle(
            ao_to_atom=anchor_group["ao_to_atom"][...],
            pair_indices=anchor_group["pair_indices"][...],
            pair_displacements_au=anchor_group["pair_displacements_au"][...],
            incidence=anchor_group["incidence"][...],
            site_projector_diagonals=anchor_group["site_projector_diagonals"][...],
        )
        root = handle["reference"]
        expected_fingerprints = {
            "grid": _attribute_text(root, "grid_fingerprint_sha256"),
            "core operators": _attribute_text(root, "core_operator_fingerprint_sha256"),
            "anchor topology": _attribute_text(root, "anchor_topology_fingerprint_sha256"),
            "reference": _attribute_text(root, "reference_fingerprint_sha256"),
        }
        preparation_backend = _text(handle["meta/preparation_backend"])
    actual_bundles = {
        "grid": grid.fingerprint_sha256,
        "core operators": operators.fingerprint_sha256,
        "anchor topology": anchors.fingerprint_sha256,
    }
    for name, actual in actual_bundles.items():
        if actual != expected_fingerprints[name]:
            raise ReferencePreparationError(f"{name} fingerprint mismatch")
    reference = PreparedReference(
        config=resolved.config,
        ground_state=state,
        grid=grid,
        core_operators=operators,
        anchor_topology=anchors,
        dependencies=dependencies,
        preparation_backend=preparation_backend,
    )
    if reference.fingerprint_sha256 != expected_fingerprints["reference"]:
        raise ReferencePreparationError("prepared-reference content fingerprint mismatch")
    if header.artifact_id != reference.fingerprint_sha256:
        raise ReferencePreparationError("artifact ID does not authenticate the prepared reference")
    return reference
