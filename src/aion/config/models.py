"""Immutable configuration contracts for Aion 0.2.

The dataclasses in this module are the authoritative programmatic API.  TOML
is only a strict serialization boundary implemented in :mod:`aion.config.toml`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from aion.config.hashing import canonical_sha256
from aion.config.units import (
    ElectromagneticOrigin,
    FixedTimeGrid,
    StepSchedule,
    Vector3,
    finite_float,
    vector3,
)
from aion.errors import ConfigurationError, UnsupportedConfigurationError

REFERENCE_CONFIG_SCHEMA = "aion.reference-input"
SIMULATION_CONFIG_SCHEMA = "aion.simulation-input"
CONFIG_SCHEMA_VERSION = "1.0.0"

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$|^[A-Z]{1,3}$")


class BackendKind(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


class Precision(StrEnum):
    FP64_COMPLEX128 = "float64_complex128"


class SpinTreatment(StrEnum):
    RESTRICTED = "restricted"


class NuclearModel(StrEnum):
    ALL_ELECTRON_LOCAL = "all_electron_local"


class XCFamily(StrEnum):
    LDA = "lda"
    GGA = "gga"


class FormulationKind(StrEnum):
    BARE_LENGTH_GAUGE = "bare_length_gauge"
    BARE_VELOCITY_GAUGE = "bare_velocity_gauge"
    P0 = "p0"
    P0_E1 = "p0_e1"


class GaugeRepresentation(StrEnum):
    """Gauge representation selected for one formulation simulation."""

    LENGTH = "length"
    MIXED = "mixed"
    VELOCITY = "velocity"


class IntegratorKind(StrEnum):
    FIXED_METRIC_SCEM = "fixed_metric_scem"
    CONNECTION_AWARE_SCEM = "connection_aware_scem"


class RationalApproximation(StrEnum):
    CAYLEY_11 = "cayley_11"
    PADE_22 = "pade_22"


class SourceKind(StrEnum):
    ZERO = "zero"
    SIN2_VECTOR_POTENTIAL_PULSE = "sin2_vector_potential_pulse"
    COMPILED = "compiled"
    PYTHON_PROVIDER = "python_provider"


def _identifier(value: str, path: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ConfigurationError(
            f"{path} must start with a letter and contain only letters, digits, '.', '_', or '-'"
        )
    return value


def _digest(value: str, path: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise ConfigurationError(f"{path} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class AtomConfig:
    """One fixed nucleus in input coordinates, expressed in bohr."""

    symbol: str
    position_au: Vector3

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not _ELEMENT_RE.fullmatch(self.symbol):
            raise ConfigurationError(f"invalid element symbol {self.symbol!r}")
        object.__setattr__(self, "position_au", vector3(self.position_au, "atom.position_au"))

    def as_mapping(self) -> dict[str, object]:
        return {"symbol": self.symbol, "position_au": list(self.position_au)}


@dataclass(frozen=True, slots=True)
class MoleculeConfig:
    """Fixed-nucleus finite molecular geometry and electromagnetic origin."""

    atoms: tuple[AtomConfig, ...]
    charge: int = 0
    spin: int = 0
    electromagnetic_origin: ElectromagneticOrigin = field(default_factory=ElectromagneticOrigin)

    def __post_init__(self) -> None:
        if not isinstance(self.atoms, tuple | list):
            raise ConfigurationError("molecule.atoms must be a sequence of AtomConfig values")
        object.__setattr__(self, "atoms", tuple(self.atoms))
        if not self.atoms:
            raise ConfigurationError("molecule.atoms must contain at least one atom")
        if not all(isinstance(atom, AtomConfig) for atom in self.atoms):
            raise ConfigurationError("molecule.atoms must contain only AtomConfig values")
        if not isinstance(self.electromagnetic_origin, ElectromagneticOrigin):
            raise ConfigurationError(
                "molecule.electromagnetic_origin must be an ElectromagneticOrigin"
            )
        for name, value in (("charge", self.charge), ("spin", self.spin)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigurationError(f"molecule.{name} must be an integer")
        if self.spin != 0:
            raise UnsupportedConfigurationError(
                "Aion 0.2 validates closed-shell restricted RKS only; molecule.spin must be 0"
            )

    def as_mapping(self) -> dict[str, object]:
        return {
            "charge": self.charge,
            "spin": self.spin,
            "electromagnetic_origin_au": list(self.electromagnetic_origin.position_au),
            "atoms": [atom.as_mapping() for atom in self.atoms],
        }


@dataclass(frozen=True, slots=True)
class ElectronicStructureConfig:
    """Validated first-release PySCF RKS preparation domain."""

    basis: str
    functional: str
    xc_family: XCFamily
    grid_level: int = 3
    density_fitting: bool = False
    auxiliary_basis: str | None = None
    scf_energy_tolerance_au: float = 1.0e-10
    scf_max_iterations: int = 100
    spin_treatment: SpinTreatment = SpinTreatment.RESTRICTED
    nuclear_model: NuclearModel = NuclearModel.ALL_ELECTRON_LOCAL

    def __post_init__(self) -> None:
        if not isinstance(self.xc_family, XCFamily):
            raise ConfigurationError("electronic_structure.xc_family is invalid")
        if not isinstance(self.spin_treatment, SpinTreatment):
            raise ConfigurationError("electronic_structure.spin_treatment is invalid")
        if not isinstance(self.nuclear_model, NuclearModel):
            raise ConfigurationError("electronic_structure.nuclear_model is invalid")
        if not isinstance(self.basis, str) or not self.basis.strip():
            raise ConfigurationError("electronic_structure.basis cannot be empty")
        if not isinstance(self.functional, str) or not self.functional.strip():
            raise ConfigurationError("electronic_structure.functional cannot be empty")
        if isinstance(self.grid_level, bool) or not isinstance(self.grid_level, int):
            raise ConfigurationError("electronic_structure.grid_level must be an integer")
        if self.grid_level < 0:
            raise ConfigurationError("electronic_structure.grid_level cannot be negative")
        if not isinstance(self.density_fitting, bool):
            raise ConfigurationError("electronic_structure.density_fitting must be a boolean")
        if self.density_fitting:
            if not isinstance(self.auxiliary_basis, str) or not self.auxiliary_basis.strip():
                raise ConfigurationError(
                    "electronic_structure.auxiliary_basis must be explicit when "
                    "density fitting is enabled"
                )
        elif self.auxiliary_basis is not None:
            raise ConfigurationError(
                "electronic_structure.auxiliary_basis is unused when density fitting is disabled"
            )
        tolerance = finite_float(
            self.scf_energy_tolerance_au,
            "electronic_structure.scf_energy_tolerance_au",
        )
        if tolerance <= 0.0:
            raise ConfigurationError(
                "electronic_structure.scf_energy_tolerance_au must be positive"
            )
        object.__setattr__(self, "scf_energy_tolerance_au", tolerance)
        if isinstance(self.scf_max_iterations, bool) or not isinstance(
            self.scf_max_iterations, int
        ):
            raise ConfigurationError("electronic_structure.scf_max_iterations must be an integer")
        if self.scf_max_iterations < 1:
            raise ConfigurationError("electronic_structure.scf_max_iterations must be at least one")
        if self.spin_treatment is not SpinTreatment.RESTRICTED:
            raise UnsupportedConfigurationError("only restricted spin treatment is supported")
        if self.nuclear_model is not NuclearModel.ALL_ELECTRON_LOCAL:
            raise UnsupportedConfigurationError(
                "pseudopotentials, ECPs, and nonlocal ionic operators are out of scope"
            )

    def as_mapping(self) -> dict[str, object]:
        return {
            "basis": self.basis,
            "functional": self.functional,
            "xc_family": self.xc_family.value,
            "grid_level": self.grid_level,
            "density_fitting": self.density_fitting,
            "auxiliary_basis": self.auxiliary_basis,
            "scf_energy_tolerance_au": self.scf_energy_tolerance_au,
            "scf_max_iterations": self.scf_max_iterations,
            "spin_treatment": self.spin_treatment.value,
            "nuclear_model": self.nuclear_model.value,
        }


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Execution backend selection without silent fallback semantics."""

    kind: BackendKind = BackendKind.CPU
    precision: Precision = Precision.FP64_COMPLEX128
    device_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BackendKind):
            raise ConfigurationError("backend.kind is invalid")
        if not isinstance(self.precision, Precision):
            raise ConfigurationError("backend.precision is invalid")
        if self.precision is not Precision.FP64_COMPLEX128:
            raise UnsupportedConfigurationError(
                "Aion 0.2 supports float64/complex128 execution only"
            )
        if self.kind is BackendKind.CPU:
            if self.device_index is not None:
                raise ConfigurationError("backend.device_index is unused for a CPU backend")
            return
        if self.device_index is None:
            object.__setattr__(self, "device_index", 0)
        elif isinstance(self.device_index, bool) or not isinstance(self.device_index, int):
            raise ConfigurationError("backend.device_index must be an integer")
        elif self.device_index < 0:
            raise ConfigurationError("backend.device_index cannot be negative")

    def as_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": self.kind.value,
            "precision": self.precision.value,
        }
        if self.device_index is not None:
            result["device_index"] = self.device_index
        return result

    def scientific_mapping(self) -> dict[str, object]:
        return {"kind": self.kind.value, "precision": self.precision.value}


@dataclass(frozen=True, slots=True)
class ReferenceOutputConfig:
    artifact_path: Path = Path("reference.h5")

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_path", Path(self.artifact_path))

    def as_mapping(self) -> dict[str, object]:
        return {"artifact_path": str(self.artifact_path)}


@dataclass(frozen=True, slots=True)
class MetadataConfig:
    """Human/execution metadata excluded from scientific identity."""

    label: str | None = None
    timestamp_utc: str | None = None
    host: str | None = None

    def __post_init__(self) -> None:
        for name in ("label", "timestamp_utc", "host"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ConfigurationError(f"metadata.{name} must be a nonempty string")

    def as_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for name in ("label", "timestamp_utc", "host"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    """Complete, resolved input for an immutable prepared reference."""

    molecule: MoleculeConfig
    electronic_structure: ElectronicStructureConfig
    backend: BackendConfig = field(default_factory=BackendConfig)
    output: ReferenceOutputConfig = field(default_factory=ReferenceOutputConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)

    def __post_init__(self) -> None:
        expected_types = (
            ("molecule", self.molecule, MoleculeConfig),
            ("electronic_structure", self.electronic_structure, ElectronicStructureConfig),
            ("backend", self.backend, BackendConfig),
            ("output", self.output, ReferenceOutputConfig),
            ("metadata", self.metadata, MetadataConfig),
        )
        for name, value, expected in expected_types:
            if not isinstance(value, expected):
                raise ConfigurationError(f"reference.{name} has the wrong typed contract")

    @property
    def scientific_id(self) -> str:
        return canonical_sha256(self.scientific_mapping())

    def scientific_mapping(self) -> dict[str, object]:
        return {
            "schema": REFERENCE_CONFIG_SCHEMA,
            "schema_version": CONFIG_SCHEMA_VERSION,
            "molecule": self.molecule.as_mapping(),
            "electronic_structure": self.electronic_structure.as_mapping(),
            "precision": self.backend.precision.value,
        }

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema": REFERENCE_CONFIG_SCHEMA,
            "schema_version": CONFIG_SCHEMA_VERSION,
            "molecule": self.molecule.as_mapping(),
            "electronic_structure": self.electronic_structure.as_mapping(),
            "backend": self.backend.as_mapping(),
            "output": self.output.as_mapping(),
            "metadata": self.metadata.as_mapping(),
        }


@dataclass(frozen=True, slots=True)
class ReferenceLinkConfig:
    """Authenticated reference input; its path is operational, its digest scientific."""

    fingerprint_sha256: str
    path: Path = Path("reference.h5")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fingerprint_sha256",
            _digest(self.fingerprint_sha256, "reference.fingerprint_sha256"),
        )
        object.__setattr__(self, "path", Path(self.path))

    def as_mapping(self) -> dict[str, object]:
        return {
            "fingerprint_sha256": self.fingerprint_sha256,
            "path": str(self.path),
        }


@dataclass(frozen=True, slots=True)
class FormulationConfig:
    kind: FormulationKind
    gauge: GaugeRepresentation | None = None
    velocity_fraction: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FormulationKind):
            raise ConfigurationError("formulation.kind is invalid")
        fraction = self.velocity_fraction
        if fraction is not None:
            fraction = finite_float(fraction, "formulation.velocity_fraction")
            if not 0.0 <= fraction <= 1.0:
                raise ConfigurationError(
                    "formulation.velocity_fraction must lie in the closed interval [0, 1]"
                )
            object.__setattr__(self, "velocity_fraction", fraction)
        gauge = self.gauge
        if gauge is None:
            if fraction is None:
                gauge = (
                    GaugeRepresentation.VELOCITY
                    if self.kind is FormulationKind.BARE_VELOCITY_GAUGE
                    else GaugeRepresentation.LENGTH
                )
            elif fraction == 0.0:
                gauge = GaugeRepresentation.LENGTH
            elif fraction == 1.0:
                gauge = GaugeRepresentation.VELOCITY
            else:
                gauge = GaugeRepresentation.MIXED
            object.__setattr__(self, "gauge", gauge)
        if not isinstance(gauge, GaugeRepresentation):
            raise ConfigurationError("formulation.gauge is invalid")
        required = {
            FormulationKind.BARE_LENGTH_GAUGE: GaugeRepresentation.LENGTH,
            FormulationKind.BARE_VELOCITY_GAUGE: GaugeRepresentation.VELOCITY,
        }.get(self.kind)
        if required is not None and gauge is not required:
            raise UnsupportedConfigurationError(
                f"{self.kind.value} requires the {required.value}-gauge representation"
            )
        if gauge is GaugeRepresentation.MIXED:
            if self.kind not in {FormulationKind.P0, FormulationKind.P0_E1}:
                raise UnsupportedConfigurationError(
                    "mixed gauge is implemented only for the covariant P0/P0+E1 formulations"
                )
            if fraction is None or fraction in {0.0, 1.0}:
                raise ConfigurationError(
                    "mixed gauge requires a velocity_fraction strictly between zero and one"
                )
        else:
            expected = 0.0 if gauge is GaugeRepresentation.LENGTH else 1.0
            if fraction is not None and fraction != expected:
                raise ConfigurationError(
                    f"{gauge.value} gauge requires velocity_fraction={expected:.1f}"
                )
            object.__setattr__(self, "velocity_fraction", None)

    @property
    def resolved_velocity_fraction(self) -> float:
        """Return the fraction of the physical reduced vector potential."""

        assert self.gauge is not None
        if self.gauge is GaugeRepresentation.LENGTH:
            return 0.0
        if self.gauge is GaugeRepresentation.VELOCITY:
            return 1.0
        assert self.velocity_fraction is not None
        return self.velocity_fraction

    def as_mapping(self) -> dict[str, object]:
        assert self.gauge is not None
        result: dict[str, object] = {"kind": self.kind.value, "gauge": self.gauge.value}
        if self.gauge is GaugeRepresentation.MIXED:
            result["velocity_fraction"] = self.resolved_velocity_fraction
        return result


@dataclass(frozen=True, slots=True)
class ZeroSourceConfig:
    kind: SourceKind = field(default=SourceKind.ZERO, init=False)

    def as_mapping(self) -> dict[str, object]:
        return {"kind": self.kind.value}


@dataclass(frozen=True, slots=True)
class Sin2VectorPotentialPulseConfig:
    """Compactly supported potential-first uniform pulse definition."""

    peak_electric_field_au: float
    angular_frequency_au: float
    cycles: int
    polarization: Vector3
    start_time_au: float = 0.0
    carrier_phase_rad: float = 0.0
    require_zero_impulse: bool = False
    kind: SourceKind = field(default=SourceKind.SIN2_VECTOR_POTENTIAL_PULSE, init=False)

    def __post_init__(self) -> None:
        for name in (
            "peak_electric_field_au",
            "angular_frequency_au",
            "start_time_au",
            "carrier_phase_rad",
        ):
            object.__setattr__(self, name, finite_float(getattr(self, name), f"source.{name}"))
        if self.peak_electric_field_au <= 0.0:
            raise ConfigurationError("source.peak_electric_field_au must be positive")
        if self.angular_frequency_au <= 0.0:
            raise ConfigurationError("source.angular_frequency_au must be positive")
        if isinstance(self.cycles, bool) or not isinstance(self.cycles, int) or self.cycles < 1:
            raise ConfigurationError("source.cycles must be a positive integer")
        polarization = vector3(self.polarization, "source.polarization")
        norm2 = sum(component * component for component in polarization)
        if norm2 <= 0.0:
            raise ConfigurationError("source.polarization cannot be the zero vector")
        object.__setattr__(self, "polarization", polarization)
        if not isinstance(self.require_zero_impulse, bool):
            raise ConfigurationError("source.require_zero_impulse must be a boolean")

    def as_mapping(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "peak_electric_field_au": self.peak_electric_field_au,
            "angular_frequency_au": self.angular_frequency_au,
            "cycles": self.cycles,
            "polarization": list(self.polarization),
            "start_time_au": self.start_time_au,
            "carrier_phase_rad": self.carrier_phase_rad,
            "require_zero_impulse": self.require_zero_impulse,
        }


@dataclass(frozen=True, slots=True)
class CompiledSourceConfig:
    fingerprint_sha256: str
    path: Path
    kind: SourceKind = field(default=SourceKind.COMPILED, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fingerprint_sha256",
            _digest(self.fingerprint_sha256, "source.fingerprint_sha256"),
        )
        object.__setattr__(self, "path", Path(self.path))

    def as_mapping(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "fingerprint_sha256": self.fingerprint_sha256,
            "path": str(self.path),
        }

    def scientific_mapping(self) -> dict[str, object]:
        return {"kind": self.kind.value, "fingerprint_sha256": self.fingerprint_sha256}


@dataclass(frozen=True, slots=True)
class PythonProviderSourceConfig:
    semantic_id: str
    compiled_fingerprint_sha256: str
    kind: SourceKind = field(default=SourceKind.PYTHON_PROVIDER, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_id", _identifier(self.semantic_id, "source.semantic_id"))
        object.__setattr__(
            self,
            "compiled_fingerprint_sha256",
            _digest(
                self.compiled_fingerprint_sha256,
                "source.compiled_fingerprint_sha256",
            ),
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "semantic_id": self.semantic_id,
            "compiled_fingerprint_sha256": self.compiled_fingerprint_sha256,
        }


type SourceConfig = (
    ZeroSourceConfig
    | Sin2VectorPotentialPulseConfig
    | CompiledSourceConfig
    | PythonProviderSourceConfig
)


def source_scientific_mapping(source: SourceConfig) -> dict[str, object]:
    if isinstance(source, CompiledSourceConfig):
        return source.scientific_mapping()
    return source.as_mapping()


@dataclass(frozen=True, slots=True)
class KickEventConfig:
    """An exact, idempotent event applied on an integer state boundary."""

    event_id: str
    step: int
    impulse_au: Vector3

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _identifier(self.event_id, "kick.event_id"))
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ConfigurationError("kick.step must be a nonnegative integer")
        object.__setattr__(self, "impulse_au", vector3(self.impulse_au, "kick.impulse_au"))

    def as_mapping(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "step": self.step,
            "impulse_au": list(self.impulse_au),
        }


@dataclass(frozen=True, slots=True)
class PropagationConfig:
    time_grid: FixedTimeGrid
    integrator: IntegratorKind
    rational_approximation: RationalApproximation = RationalApproximation.PADE_22
    density_tolerance: float = 1.0e-10
    max_iterations: int = 50
    minimum_damping: float = 0.1
    hermitian_cleanup_threshold: float = 1.0e-12

    def __post_init__(self) -> None:
        if not isinstance(self.time_grid, FixedTimeGrid):
            raise ConfigurationError("propagation.time_grid must be a FixedTimeGrid")
        if not isinstance(self.integrator, IntegratorKind):
            raise ConfigurationError("propagation.integrator is invalid")
        if not isinstance(self.rational_approximation, RationalApproximation):
            raise ConfigurationError("propagation.rational_approximation is invalid")
        for name in (
            "density_tolerance",
            "minimum_damping",
            "hermitian_cleanup_threshold",
        ):
            value = finite_float(getattr(self, name), f"propagation.{name}")
            object.__setattr__(self, name, value)
            if value <= 0.0:
                raise ConfigurationError(f"propagation.{name} must be positive")
        if self.minimum_damping > 1.0:
            raise ConfigurationError("propagation.minimum_damping cannot exceed one")
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int):
            raise ConfigurationError("propagation.max_iterations must be an integer")
        if self.max_iterations < 1:
            raise ConfigurationError("propagation.max_iterations must be at least one")

    def as_mapping(self) -> dict[str, object]:
        return {
            "start_time_au": self.time_grid.start_au,
            "time_step_au": self.time_grid.step_au,
            "intervals": self.time_grid.intervals,
            "integrator": self.integrator.value,
            "rational_approximation": self.rational_approximation.value,
            "density_tolerance": self.density_tolerance,
            "max_iterations": self.max_iterations,
            "minimum_damping": self.minimum_damping,
            "hermitian_cleanup_threshold": self.hermitian_cleanup_threshold,
        }


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    strict_validation: bool = False
    absolute_tolerance: float = 1.0e-9
    relative_tolerance: float = 1.0e-7

    def __post_init__(self) -> None:
        if not isinstance(self.strict_validation, bool):
            raise ConfigurationError("validation.strict_validation must be a boolean")
        for name in ("absolute_tolerance", "relative_tolerance"):
            value = finite_float(getattr(self, name), f"validation.{name}")
            if value <= 0.0:
                raise ConfigurationError(f"validation.{name} must be positive")
            object.__setattr__(self, name, value)

    def as_mapping(self) -> dict[str, object]:
        return {
            "strict_validation": self.strict_validation,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
        }


@dataclass(frozen=True, slots=True)
class ObservableSchedules:
    dipole_current: StepSchedule = field(default_factory=lambda: StepSchedule(every=1))
    energy: StepSchedule = field(default_factory=StepSchedule)
    diagnostics: StepSchedule = field(default_factory=lambda: StepSchedule(every=1))
    source: StepSchedule = field(default_factory=lambda: StepSchedule(every=1))
    checkpoints: StepSchedule = field(default_factory=StepSchedule)
    matrix_snapshots: StepSchedule = field(
        default_factory=lambda: StepSchedule(
            every=0,
            include_initial=False,
            include_final=False,
        )
    )

    def __post_init__(self) -> None:
        for name in (
            "dipole_current",
            "energy",
            "diagnostics",
            "source",
            "checkpoints",
            "matrix_snapshots",
        ):
            if not isinstance(getattr(self, name), StepSchedule):
                raise ConfigurationError(f"output.schedules.{name} must be a StepSchedule")

    def as_mapping(self) -> dict[str, object]:
        return {
            name: _schedule_mapping(getattr(self, name))
            for name in (
                "dipole_current",
                "energy",
                "diagnostics",
                "source",
                "checkpoints",
                "matrix_snapshots",
            )
        }


def _schedule_mapping(schedule: StepSchedule) -> dict[str, object]:
    return {
        "every": schedule.every,
        "include_initial": schedule.include_initial,
        "include_final": schedule.include_final,
    }


@dataclass(frozen=True, slots=True)
class OutputConfig:
    directory: Path = Path("run")
    schedules: ObservableSchedules = field(default_factory=ObservableSchedules)

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if not isinstance(self.schedules, ObservableSchedules):
            raise ConfigurationError("output.schedules must be ObservableSchedules")

    def as_mapping(self) -> dict[str, object]:
        return {
            "directory": str(self.directory),
            "schedules": self.schedules.as_mapping(),
        }


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Complete, resolved scientific and execution configuration for one run."""

    reference: ReferenceLinkConfig
    formulation: FormulationConfig
    source: SourceConfig
    propagation: PropagationConfig
    backend: BackendConfig = field(default_factory=BackendConfig)
    events: tuple[KickEventConfig, ...] = ()
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)

    def __post_init__(self) -> None:
        expected_types = (
            ("reference", self.reference, ReferenceLinkConfig),
            ("formulation", self.formulation, FormulationConfig),
            ("propagation", self.propagation, PropagationConfig),
            ("backend", self.backend, BackendConfig),
            ("validation", self.validation, ValidationConfig),
            ("output", self.output, OutputConfig),
            ("metadata", self.metadata, MetadataConfig),
        )
        for name, value, expected in expected_types:
            if not isinstance(value, expected):
                raise ConfigurationError(f"simulation.{name} has the wrong typed contract")
        if not isinstance(
            self.source,
            (
                ZeroSourceConfig,
                Sin2VectorPotentialPulseConfig,
                CompiledSourceConfig,
                PythonProviderSourceConfig,
            ),
        ):
            raise ConfigurationError("simulation.source has the wrong typed contract")
        object.__setattr__(self, "events", tuple(self.events))
        if not all(isinstance(event, KickEventConfig) for event in self.events):
            raise ConfigurationError("simulation.events must contain KickEventConfig values")
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ConfigurationError("kick event IDs must be unique")
        for event in self.events:
            if event.step > self.propagation.time_grid.intervals:
                raise ConfigurationError(
                    f"kick {event.event_id!r} lies outside the fixed time grid"
                )
        fixed = {
            FormulationKind.BARE_LENGTH_GAUGE,
            FormulationKind.BARE_VELOCITY_GAUGE,
        }
        required_integrator = (
            IntegratorKind.FIXED_METRIC_SCEM
            if self.formulation.kind in fixed
            else IntegratorKind.CONNECTION_AWARE_SCEM
        )
        if self.propagation.integrator is not required_integrator:
            raise UnsupportedConfigurationError(
                f"{self.formulation.kind.value} requires {required_integrator.value}"
            )

    @property
    def scientific_id(self) -> str:
        return canonical_sha256(self.scientific_mapping())

    def scientific_mapping(self) -> dict[str, object]:
        return {
            "schema": SIMULATION_CONFIG_SCHEMA,
            "schema_version": CONFIG_SCHEMA_VERSION,
            "reference_fingerprint_sha256": self.reference.fingerprint_sha256,
            "formulation": self.formulation.as_mapping(),
            "source": source_scientific_mapping(self.source),
            "propagation": self.propagation.as_mapping(),
            "backend": self.backend.scientific_mapping(),
            "events": [event.as_mapping() for event in self.events],
            "validation": self.validation.as_mapping(),
        }

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema": SIMULATION_CONFIG_SCHEMA,
            "schema_version": CONFIG_SCHEMA_VERSION,
            "reference": self.reference.as_mapping(),
            "formulation": self.formulation.as_mapping(),
            "source": self.source.as_mapping(),
            "propagation": self.propagation.as_mapping(),
            "backend": self.backend.as_mapping(),
            "events": {"kicks": [event.as_mapping() for event in self.events]},
            "validation": self.validation.as_mapping(),
            "output": self.output.as_mapping(),
            "metadata": self.metadata.as_mapping(),
        }


type AionConfig = ReferenceConfig | SimulationConfig
