"""Strict TOML parsing and deterministic resolved serialization."""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeGuard

from aion.config.models import (
    CONFIG_SCHEMA_VERSION,
    REFERENCE_CONFIG_SCHEMA,
    SIMULATION_CONFIG_SCHEMA,
    WILSON_SIMULATION_CONFIG_SCHEMA,
    WILSON_STATIONARY_CONFIG_SCHEMA,
    AffineElectromagneticSourceConfig,
    AionConfig,
    AtomConfig,
    BackendConfig,
    BackendKind,
    CompiledSourceConfig,
    ElectronicStructureConfig,
    ExactWilsonActionConfig,
    FormulationConfig,
    FormulationKind,
    GaugeRepresentation,
    IntegratorKind,
    KickEventConfig,
    MetadataConfig,
    MoleculeConfig,
    NuclearModel,
    ObservableSchedules,
    OutputConfig,
    Precision,
    PropagationConfig,
    PythonProviderSourceConfig,
    RationalApproximation,
    ReducedWilsonActionConfig,
    ReducedWilsonLevel,
    ReferenceConfig,
    ReferenceLinkConfig,
    ReferenceOutputConfig,
    SimulationConfig,
    Sin2VectorPotentialPulseConfig,
    SourceKind,
    SpinTreatment,
    ValidationConfig,
    WilsonActionConfig,
    WilsonActionKind,
    WilsonGridKind,
    WilsonGridPruning,
    WilsonIntegratorKind,
    WilsonMagneticGaugeKind,
    WilsonNumericsConfig,
    WilsonPropagationConfig,
    WilsonSimulationConfig,
    WilsonSourceKind,
    WilsonStationaryBranch,
    WilsonStationaryConfig,
    WilsonStationaryOutputConfig,
    WilsonStationaryPolicyConfig,
    WilsonStationaryStateLinkConfig,
    XCFamily,
    ZeroSourceConfig,
)
from aion.config.units import ElectromagneticOrigin, FixedTimeGrid, StepSchedule, vector3
from aion.errors import ConfigurationError, UnknownConfigurationFieldError

_BARE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """Original TOML plus its complete normalized immutable configuration."""

    config: AionConfig
    original_toml: str
    normalized_toml: str

    @property
    def scientific_id(self) -> str:
        return self.config.scientific_id


def load_config(path: str | Path) -> ResolvedConfig:
    """Read, strictly validate, and resolve an Aion TOML configuration."""

    config_path = Path(path)
    return loads_config(config_path.read_text(encoding="utf-8"))


def loads_config(text: str) -> ResolvedConfig:
    """Strictly parse a reference or simulation configuration document."""

    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"invalid TOML: {exc}") from exc
    root = _table(parsed, "input")
    schema = _string(root.get("schema"), "schema")
    version = _string(root.get("schema_version"), "schema_version")
    if version != CONFIG_SCHEMA_VERSION:
        raise ConfigurationError(
            f"schema_version must be {CONFIG_SCHEMA_VERSION!r}, received {version!r}"
        )
    if schema == REFERENCE_CONFIG_SCHEMA:
        config: AionConfig = _parse_reference(root)
    elif schema == SIMULATION_CONFIG_SCHEMA:
        config = _parse_simulation(root)
    elif schema == WILSON_STATIONARY_CONFIG_SCHEMA:
        config = _parse_wilson_stationary(root)
    elif schema == WILSON_SIMULATION_CONFIG_SCHEMA:
        config = _parse_wilson_simulation(root)
    else:
        raise ConfigurationError(f"unsupported configuration schema {schema!r}")
    return ResolvedConfig(
        config=config,
        original_toml=text,
        normalized_toml=dumps_config(config),
    )


def dumps_config(config: AionConfig) -> str:
    """Serialize a fully resolved configuration to deterministic TOML."""

    lines: list[str] = []
    _emit_table(config.as_mapping(), (), lines, emit_header=False)
    return "\n".join(lines).rstrip() + "\n"


def _parse_reference(root: Mapping[str, object]) -> ReferenceConfig:
    _fields(
        root,
        "input",
        {
            "schema",
            "schema_version",
            "molecule",
            "electronic_structure",
            "backend",
            "output",
            "metadata",
        },
        {"schema", "schema_version", "molecule", "electronic_structure"},
    )
    molecule_data = _table(root["molecule"], "molecule")
    _fields(
        molecule_data,
        "molecule",
        {"charge", "spin", "electromagnetic_origin_au", "atoms"},
        {"atoms"},
    )
    atom_values = _array(molecule_data["atoms"], "molecule.atoms")
    atoms: list[AtomConfig] = []
    for index, value in enumerate(atom_values):
        atom = _table(value, f"molecule.atoms[{index}]")
        _fields(
            atom,
            f"molecule.atoms[{index}]",
            {"symbol", "position_au"},
            {"symbol", "position_au"},
        )
        atoms.append(
            AtomConfig(
                symbol=_string(atom["symbol"], f"molecule.atoms[{index}].symbol"),
                position_au=vector3(atom["position_au"], f"molecule.atoms[{index}].position_au"),
            )
        )
    origin = ElectromagneticOrigin(
        vector3(
            molecule_data.get("electromagnetic_origin_au", [0.0, 0.0, 0.0]),
            "molecule.electromagnetic_origin_au",
        )
    )
    molecule = MoleculeConfig(
        atoms=tuple(atoms),
        charge=_integer(molecule_data.get("charge", 0), "molecule.charge"),
        spin=_integer(molecule_data.get("spin", 0), "molecule.spin"),
        electromagnetic_origin=origin,
    )

    electronic_data = _table(root["electronic_structure"], "electronic_structure")
    _fields(
        electronic_data,
        "electronic_structure",
        {
            "basis",
            "functional",
            "xc_family",
            "grid_level",
            "density_fitting",
            "auxiliary_basis",
            "scf_energy_tolerance_au",
            "scf_max_iterations",
            "spin_treatment",
            "nuclear_model",
        },
        {"basis", "functional", "xc_family"},
    )
    electronic = ElectronicStructureConfig(
        basis=_string(electronic_data["basis"], "electronic_structure.basis"),
        functional=_string(electronic_data["functional"], "electronic_structure.functional"),
        xc_family=_enum(
            XCFamily,
            electronic_data["xc_family"],
            "electronic_structure.xc_family",
        ),
        grid_level=_integer(
            electronic_data.get("grid_level", 3), "electronic_structure.grid_level"
        ),
        density_fitting=_boolean(
            electronic_data.get("density_fitting", False),
            "electronic_structure.density_fitting",
        ),
        auxiliary_basis=(
            None
            if electronic_data.get("auxiliary_basis") is None
            else _string(
                electronic_data["auxiliary_basis"],
                "electronic_structure.auxiliary_basis",
            )
        ),
        scf_energy_tolerance_au=_number(
            electronic_data.get("scf_energy_tolerance_au", 1.0e-10),
            "electronic_structure.scf_energy_tolerance_au",
        ),
        scf_max_iterations=_integer(
            electronic_data.get("scf_max_iterations", 100),
            "electronic_structure.scf_max_iterations",
        ),
        spin_treatment=_enum(
            SpinTreatment,
            electronic_data.get("spin_treatment", SpinTreatment.RESTRICTED.value),
            "electronic_structure.spin_treatment",
        ),
        nuclear_model=_enum(
            NuclearModel,
            electronic_data.get("nuclear_model", NuclearModel.ALL_ELECTRON_LOCAL.value),
            "electronic_structure.nuclear_model",
        ),
    )
    return ReferenceConfig(
        molecule=molecule,
        electronic_structure=electronic,
        backend=_parse_backend(root.get("backend", {})),
        output=_parse_reference_output(root.get("output", {})),
        metadata=_parse_metadata(root.get("metadata", {})),
    )


def _parse_simulation(root: Mapping[str, object]) -> SimulationConfig:
    _fields(
        root,
        "input",
        {
            "schema",
            "schema_version",
            "reference",
            "formulation",
            "source",
            "propagation",
            "backend",
            "events",
            "validation",
            "output",
            "metadata",
        },
        {"schema", "schema_version", "reference", "formulation", "source", "propagation"},
    )
    reference_data = _table(root["reference"], "reference")
    _fields(
        reference_data,
        "reference",
        {"fingerprint_sha256", "path"},
        {"fingerprint_sha256"},
    )
    reference = ReferenceLinkConfig(
        fingerprint_sha256=_string(
            reference_data["fingerprint_sha256"], "reference.fingerprint_sha256"
        ),
        path=Path(_string(reference_data.get("path", "reference.h5"), "reference.path")),
    )
    formulation_data = _table(root["formulation"], "formulation")
    _fields(
        formulation_data,
        "formulation",
        {"kind", "gauge", "velocity_fraction"},
        {"kind"},
    )
    formulation_kind = _enum(FormulationKind, formulation_data["kind"], "formulation.kind")
    formulation = FormulationConfig(
        formulation_kind,
        (
            _enum(GaugeRepresentation, formulation_data["gauge"], "formulation.gauge")
            if "gauge" in formulation_data
            else None
        ),
        (
            _number(
                formulation_data["velocity_fraction"],
                "formulation.velocity_fraction",
            )
            if "velocity_fraction" in formulation_data
            else None
        ),
    )
    source = _parse_source(root["source"])
    propagation = _parse_propagation(root["propagation"])
    events_data = _table(root.get("events", {}), "events")
    _fields(events_data, "events", {"kicks"})
    kicks: list[KickEventConfig] = []
    for index, value in enumerate(_array(events_data.get("kicks", []), "events.kicks")):
        kick = _table(value, f"events.kicks[{index}]")
        _fields(
            kick,
            f"events.kicks[{index}]",
            {"event_id", "step", "impulse_au"},
            {"event_id", "step", "impulse_au"},
        )
        kicks.append(
            KickEventConfig(
                event_id=_string(kick["event_id"], f"events.kicks[{index}].event_id"),
                step=_integer(kick["step"], f"events.kicks[{index}].step"),
                impulse_au=vector3(kick["impulse_au"], f"events.kicks[{index}].impulse_au"),
            )
        )
    return SimulationConfig(
        reference=reference,
        formulation=formulation,
        source=source,
        propagation=propagation,
        backend=_parse_backend(root.get("backend", {})),
        events=tuple(kicks),
        validation=_parse_validation(root.get("validation", {})),
        output=_parse_output(root.get("output", {})),
        metadata=_parse_metadata(root.get("metadata", {})),
    )


def _parse_reference_link(value: object, path: str = "reference") -> ReferenceLinkConfig:
    data = _table(value, path)
    _fields(data, path, {"fingerprint_sha256", "path"}, {"fingerprint_sha256"})
    return ReferenceLinkConfig(
        fingerprint_sha256=_string(
            data["fingerprint_sha256"],
            f"{path}.fingerprint_sha256",
        ),
        path=Path(_string(data.get("path", "reference.h5"), f"{path}.path")),
    )


def _parse_wilson_action(value: object) -> WilsonActionConfig:
    data = _table(value, "action")
    kind = _enum(WilsonActionKind, data.get("kind"), "action.kind")
    if kind is WilsonActionKind.EXACT:
        _fields(data, "action", {"kind", "branch"}, {"kind", "branch"})
        return ExactWilsonActionConfig(
            _enum(WilsonStationaryBranch, data["branch"], "action.branch")
        )
    _fields(
        data,
        "action",
        {"kind", "branch", "level"},
        {"kind", "branch", "level"},
    )
    return ReducedWilsonActionConfig(
        _enum(WilsonStationaryBranch, data["branch"], "action.branch"),
        _enum(ReducedWilsonLevel, data["level"], "action.level"),
    )


def _parse_wilson_numerics(value: object) -> WilsonNumericsConfig:
    data = _table(value, "numerics")
    required = {
        "grid_kind",
        "grid_pruning",
        "block_size",
        "auxiliary_basis",
        "ri_relative_threshold",
        "ri_absolute_threshold",
    }
    _fields(
        data,
        "numerics",
        required | {"grid_level", "ri_maximum_rank", "memory_budget_bytes"},
        required,
    )
    return WilsonNumericsConfig(
        grid_kind=_enum(WilsonGridKind, data["grid_kind"], "numerics.grid_kind"),
        grid_level=(
            _integer(data["grid_level"], "numerics.grid_level") if "grid_level" in data else None
        ),
        grid_pruning=_enum(
            WilsonGridPruning,
            data["grid_pruning"],
            "numerics.grid_pruning",
        ),
        block_size=_integer(data["block_size"], "numerics.block_size"),
        auxiliary_basis=_string(data["auxiliary_basis"], "numerics.auxiliary_basis"),
        ri_relative_threshold=_number(
            data["ri_relative_threshold"],
            "numerics.ri_relative_threshold",
        ),
        ri_absolute_threshold=_number(
            data["ri_absolute_threshold"],
            "numerics.ri_absolute_threshold",
        ),
        ri_maximum_rank=(
            _integer(data["ri_maximum_rank"], "numerics.ri_maximum_rank")
            if "ri_maximum_rank" in data
            else None
        ),
        memory_budget_bytes=(
            _integer(data["memory_budget_bytes"], "numerics.memory_budget_bytes")
            if "memory_budget_bytes" in data
            else None
        ),
    )


def _parse_affine_electromagnetic_source(
    value: object,
) -> AffineElectromagneticSourceConfig:
    data = _table(value, "source")
    required = {
        "kind",
        "electric_field_origin_offset_au",
        "magnetic_field_reference_au",
        "magnetic_field_rate_au",
        "magnetic_reference_time_au",
        "magnetic_gauge",
        "electric",
    }
    _fields(data, "source", required | {"landau_axis"}, required)
    kind = _enum(WilsonSourceKind, data["kind"], "source.kind")
    if kind is not WilsonSourceKind.AFFINE_ELECTROMAGNETIC:
        raise AssertionError("unhandled Wilson source kind")
    electric = _parse_source(data["electric"])
    if not isinstance(electric, ZeroSourceConfig | Sin2VectorPotentialPulseConfig):
        raise ConfigurationError("source.electric must be zero or sin2_vector_potential_pulse")
    return AffineElectromagneticSourceConfig(
        electric=electric,
        electric_field_origin_offset_au=vector3(
            data["electric_field_origin_offset_au"],
            "source.electric_field_origin_offset_au",
        ),
        magnetic_field_reference_au=vector3(
            data["magnetic_field_reference_au"],
            "source.magnetic_field_reference_au",
        ),
        magnetic_field_rate_au=vector3(
            data["magnetic_field_rate_au"],
            "source.magnetic_field_rate_au",
        ),
        magnetic_reference_time_au=_number(
            data["magnetic_reference_time_au"],
            "source.magnetic_reference_time_au",
        ),
        magnetic_gauge=_enum(
            WilsonMagneticGaugeKind,
            data["magnetic_gauge"],
            "source.magnetic_gauge",
        ),
        landau_axis=(
            vector3(data["landau_axis"], "source.landau_axis") if "landau_axis" in data else None
        ),
    )


def _parse_wilson_stationary_policy(value: object) -> WilsonStationaryPolicyConfig:
    data = _table(value, "stationary")
    fields = {
        "maximum_iterations",
        "density_tolerance",
        "orbital_tolerance",
        "energy_tolerance_au",
        "damping",
        "diis_start_iteration",
        "diis_space",
    }
    _fields(data, "stationary", fields, fields)
    return WilsonStationaryPolicyConfig(
        maximum_iterations=_integer(
            data["maximum_iterations"],
            "stationary.maximum_iterations",
        ),
        density_tolerance=_number(
            data["density_tolerance"],
            "stationary.density_tolerance",
        ),
        orbital_tolerance=_number(
            data["orbital_tolerance"],
            "stationary.orbital_tolerance",
        ),
        energy_tolerance_au=_number(
            data["energy_tolerance_au"],
            "stationary.energy_tolerance_au",
        ),
        damping=_number(data["damping"], "stationary.damping"),
        diis_start_iteration=_integer(
            data["diis_start_iteration"],
            "stationary.diis_start_iteration",
        ),
        diis_space=_integer(data["diis_space"], "stationary.diis_space"),
    )


def _parse_wilson_propagation(value: object) -> WilsonPropagationConfig:
    data = _table(value, "propagation")
    fields = {
        "start_time_au",
        "time_step_au",
        "intervals",
        "integrator",
        "rational_approximation",
        "nonlinear_tolerance",
        "maximum_iterations",
    }
    _fields(
        data,
        "propagation",
        fields,
        fields - {"start_time_au"},
    )
    return WilsonPropagationConfig(
        time_grid=FixedTimeGrid(
            _number(data.get("start_time_au", 0.0), "propagation.start_time_au"),
            _number(data["time_step_au"], "propagation.time_step_au"),
            _integer(data["intervals"], "propagation.intervals"),
        ),
        integrator=_enum(
            WilsonIntegratorKind,
            data["integrator"],
            "propagation.integrator",
        ),
        rational_approximation=_enum(
            RationalApproximation,
            data["rational_approximation"],
            "propagation.rational_approximation",
        ),
        nonlinear_tolerance=_number(
            data["nonlinear_tolerance"],
            "propagation.nonlinear_tolerance",
        ),
        maximum_iterations=_integer(
            data["maximum_iterations"],
            "propagation.maximum_iterations",
        ),
    )


def _parse_wilson_stationary(root: Mapping[str, object]) -> WilsonStationaryConfig:
    fields = {
        "schema",
        "schema_version",
        "reference",
        "action",
        "numerics",
        "source",
        "source_time_au",
        "stationary",
        "backend",
        "output",
        "metadata",
    }
    _fields(root, "input", fields, fields - {"metadata"})
    output = _table(root["output"], "output")
    _fields(output, "output", {"artifact_path"}, {"artifact_path"})
    return WilsonStationaryConfig(
        reference=_parse_reference_link(root["reference"]),
        action=_parse_wilson_action(root["action"]),
        numerics=_parse_wilson_numerics(root["numerics"]),
        source=_parse_affine_electromagnetic_source(root["source"]),
        source_time_au=_number(root["source_time_au"], "source_time_au"),
        stationary=_parse_wilson_stationary_policy(root["stationary"]),
        backend=_parse_backend(root["backend"]),
        output=WilsonStationaryOutputConfig(
            Path(_string(output["artifact_path"], "output.artifact_path"))
        ),
        metadata=_parse_metadata(root.get("metadata", {})),
    )


def _parse_wilson_simulation(root: Mapping[str, object]) -> WilsonSimulationConfig:
    fields = {
        "schema",
        "schema_version",
        "reference",
        "stationary_state",
        "action",
        "numerics",
        "source",
        "propagation",
        "backend",
        "validation",
        "output",
        "metadata",
    }
    _fields(
        root,
        "input",
        fields,
        fields - {"validation", "metadata"},
    )
    state = _table(root["stationary_state"], "stationary_state")
    _fields(
        state,
        "stationary_state",
        {"fingerprint_sha256", "path"},
        {"fingerprint_sha256", "path"},
    )
    return WilsonSimulationConfig(
        reference=_parse_reference_link(root["reference"]),
        stationary_state=WilsonStationaryStateLinkConfig(
            _string(
                state["fingerprint_sha256"],
                "stationary_state.fingerprint_sha256",
            ),
            Path(_string(state["path"], "stationary_state.path")),
        ),
        action=_parse_wilson_action(root["action"]),
        numerics=_parse_wilson_numerics(root["numerics"]),
        source=_parse_affine_electromagnetic_source(root["source"]),
        propagation=_parse_wilson_propagation(root["propagation"]),
        backend=_parse_backend(root["backend"]),
        validation=_parse_validation(root.get("validation", {})),
        output=_parse_output(root["output"]),
        metadata=_parse_metadata(root.get("metadata", {})),
    )


def _parse_source(
    value: object,
) -> (
    ZeroSourceConfig
    | Sin2VectorPotentialPulseConfig
    | CompiledSourceConfig
    | PythonProviderSourceConfig
):
    data = _table(value, "source")
    kind = _enum(SourceKind, data.get("kind"), "source.kind")
    if kind is SourceKind.ZERO:
        _fields(data, "source", {"kind"}, {"kind"})
        return ZeroSourceConfig()
    if kind is SourceKind.SIN2_VECTOR_POTENTIAL_PULSE:
        allowed = {
            "kind",
            "peak_electric_field_au",
            "angular_frequency_au",
            "cycles",
            "polarization",
            "start_time_au",
            "carrier_phase_rad",
            "require_zero_impulse",
        }
        required = {
            "kind",
            "peak_electric_field_au",
            "angular_frequency_au",
            "cycles",
            "polarization",
        }
        _fields(data, "source", allowed, required)
        return Sin2VectorPotentialPulseConfig(
            peak_electric_field_au=_number(
                data["peak_electric_field_au"], "source.peak_electric_field_au"
            ),
            angular_frequency_au=_number(
                data["angular_frequency_au"], "source.angular_frequency_au"
            ),
            cycles=_integer(data["cycles"], "source.cycles"),
            polarization=vector3(data["polarization"], "source.polarization"),
            start_time_au=_number(data.get("start_time_au", 0.0), "source.start_time_au"),
            carrier_phase_rad=_number(
                data.get("carrier_phase_rad", 0.0), "source.carrier_phase_rad"
            ),
            require_zero_impulse=_boolean(
                data.get("require_zero_impulse", False), "source.require_zero_impulse"
            ),
        )
    if kind is SourceKind.COMPILED:
        _fields(
            data,
            "source",
            {"kind", "fingerprint_sha256", "path"},
            {"kind", "fingerprint_sha256", "path"},
        )
        return CompiledSourceConfig(
            fingerprint_sha256=_string(data["fingerprint_sha256"], "source.fingerprint_sha256"),
            path=Path(_string(data["path"], "source.path")),
        )
    _fields(
        data,
        "source",
        {"kind", "semantic_id", "compiled_fingerprint_sha256"},
        {"kind", "semantic_id", "compiled_fingerprint_sha256"},
    )
    return PythonProviderSourceConfig(
        semantic_id=_string(data["semantic_id"], "source.semantic_id"),
        compiled_fingerprint_sha256=_string(
            data["compiled_fingerprint_sha256"], "source.compiled_fingerprint_sha256"
        ),
    )


def _parse_propagation(value: object) -> PropagationConfig:
    data = _table(value, "propagation")
    _fields(
        data,
        "propagation",
        {
            "start_time_au",
            "time_step_au",
            "intervals",
            "integrator",
            "rational_approximation",
            "density_tolerance",
            "max_iterations",
            "minimum_damping",
            "hermitian_cleanup_threshold",
        },
        {"time_step_au", "intervals", "integrator"},
    )
    grid = FixedTimeGrid(
        start_au=_number(data.get("start_time_au", 0.0), "propagation.start_time_au"),
        step_au=_number(data["time_step_au"], "propagation.time_step_au"),
        intervals=_integer(data["intervals"], "propagation.intervals"),
    )
    return PropagationConfig(
        time_grid=grid,
        integrator=_enum(IntegratorKind, data["integrator"], "propagation.integrator"),
        rational_approximation=_enum(
            RationalApproximation,
            data.get("rational_approximation", RationalApproximation.PADE_22.value),
            "propagation.rational_approximation",
        ),
        density_tolerance=_number(
            data.get("density_tolerance", 1.0e-10), "propagation.density_tolerance"
        ),
        max_iterations=_integer(data.get("max_iterations", 50), "propagation.max_iterations"),
        minimum_damping=_number(data.get("minimum_damping", 0.1), "propagation.minimum_damping"),
        hermitian_cleanup_threshold=_number(
            data.get("hermitian_cleanup_threshold", 1.0e-12),
            "propagation.hermitian_cleanup_threshold",
        ),
    )


def _parse_backend(value: object) -> BackendConfig:
    data = _table(value, "backend")
    _fields(data, "backend", {"kind", "precision", "device_index"})
    return BackendConfig(
        kind=_enum(BackendKind, data.get("kind", BackendKind.CPU.value), "backend.kind"),
        precision=_enum(
            Precision,
            data.get("precision", Precision.FP64_COMPLEX128.value),
            "backend.precision",
        ),
        device_index=(
            _integer(data["device_index"], "backend.device_index")
            if "device_index" in data
            else None
        ),
    )


def _parse_reference_output(value: object) -> ReferenceOutputConfig:
    data = _table(value, "output")
    _fields(data, "output", {"artifact_path"})
    return ReferenceOutputConfig(
        Path(_string(data.get("artifact_path", "reference.h5"), "output.artifact_path"))
    )


def _parse_output(value: object) -> OutputConfig:
    data = _table(value, "output")
    _fields(data, "output", {"directory", "schedules"})
    schedules_data = _table(data.get("schedules", {}), "output.schedules")
    names = {
        "dipole_current",
        "energy",
        "diagnostics",
        "source",
        "checkpoints",
        "matrix_snapshots",
    }
    _fields(schedules_data, "output.schedules", names)
    defaults = ObservableSchedules()
    schedules = ObservableSchedules(
        **{
            name: _parse_schedule(schedules_data.get(name, {}), getattr(defaults, name), name)
            for name in names
        }
    )
    return OutputConfig(
        directory=Path(_string(data.get("directory", "run"), "output.directory")),
        schedules=schedules,
    )


def _parse_schedule(value: object, default: StepSchedule, name: str) -> StepSchedule:
    path = f"output.schedules.{name}"
    data = _table(value, path)
    _fields(data, path, {"every", "include_initial", "include_final"})
    return StepSchedule(
        every=_integer(data.get("every", default.every), f"{path}.every"),
        include_initial=_boolean(
            data.get("include_initial", default.include_initial), f"{path}.include_initial"
        ),
        include_final=_boolean(
            data.get("include_final", default.include_final), f"{path}.include_final"
        ),
    )


def _parse_validation(value: object) -> ValidationConfig:
    data = _table(value, "validation")
    _fields(
        data,
        "validation",
        {"strict_validation", "absolute_tolerance", "relative_tolerance"},
    )
    return ValidationConfig(
        strict_validation=_boolean(
            data.get("strict_validation", False), "validation.strict_validation"
        ),
        absolute_tolerance=_number(
            data.get("absolute_tolerance", 1.0e-9), "validation.absolute_tolerance"
        ),
        relative_tolerance=_number(
            data.get("relative_tolerance", 1.0e-7), "validation.relative_tolerance"
        ),
    )


def _parse_metadata(value: object) -> MetadataConfig:
    data = _table(value, "metadata")
    _fields(data, "metadata", {"label", "timestamp_utc", "host"})
    return MetadataConfig(
        label=_optional_string(data.get("label"), "metadata.label"),
        timestamp_utc=_optional_string(data.get("timestamp_utc"), "metadata.timestamp_utc"),
        host=_optional_string(data.get("host"), "metadata.host"),
    )


def _fields(
    data: Mapping[str, object],
    path: str,
    allowed: Set[str],
    required: Set[str] = frozenset(),
) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise UnknownConfigurationFieldError(
            f"{path} contains unknown or unused field(s): {', '.join(unknown)}"
        )
    missing = sorted(required - set(data))
    if missing:
        raise ConfigurationError(f"{path} lacks required field(s): {', '.join(missing)}")


def _table(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{path} must be a TOML table")
    return value


def _array(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{path} must be a TOML array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{path} must be a nonempty string")
    return value


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path} must be an integer")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{path} must be a number")
    return float(value)


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{path} must be a boolean")
    return value


def _enum[EnumT: Enum](enum_type: type[EnumT], value: object, path: str) -> EnumT:
    text = _string(value, path)
    try:
        return enum_type(text)
    except ValueError as exc:
        choices = ", ".join(str(member.value) for member in enum_type)
        raise ConfigurationError(f"{path} must be one of: {choices}") from exc


def _emit_table(
    data: Mapping[str, object],
    prefix: tuple[str, ...],
    lines: list[str],
    *,
    emit_header: bool,
) -> None:
    if emit_header:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(f"[{'.'.join(_key(part) for part in prefix)}]")
    for key, value in data.items():
        if _is_inline(value):
            lines.append(f"{_key(key)} = {_toml_value(value)}")
    for key, value in data.items():
        if isinstance(value, Mapping):
            _emit_table(value, (*prefix, key), lines, emit_header=True)
    for key, value in data.items():
        if _is_table_array(value):
            for item in value:
                if lines and lines[-1] != "":
                    lines.append("")
                path = ".".join(_key(part) for part in (*prefix, key))
                lines.append(f"[[{path}]]")
                for item_key, item_value in item.items():
                    if not _is_inline(item_value):
                        raise ConfigurationError("nested tables in arrays are not supported")
                    lines.append(f"{_key(item_key)} = {_toml_value(item_value)}")


def _is_inline(value: object) -> bool:
    if isinstance(value, str | bool | int | float):
        return True
    return isinstance(value, list) and not any(isinstance(item, Mapping) for item in value)


def _is_table_array(value: object) -> TypeGuard[list[Mapping[str, object]]]:
    return (
        isinstance(value, list) and bool(value) and all(isinstance(item, Mapping) for item in value)
    )


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise ConfigurationError(f"cannot serialize value of type {type(value).__name__} to TOML")


def _key(value: str) -> str:
    return value if _BARE_KEY.fullmatch(value) else json.dumps(value)
