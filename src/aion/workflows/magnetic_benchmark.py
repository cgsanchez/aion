"""Reusable expert workflow for the static uniform-magnetic matrix benchmark."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from aion.config import BackendConfig, canonical_sha256
from aion.electromagnetism import AffineMagneticGauge, UniformMagneticField
from aion.electronic_structure import (
    AOGridPolicy,
    LocalPotentialIdentity,
    LocalPotentialMagneticResult,
    LocalPotentialProvider,
    MagneticOneElectronResult,
    MagneticSpatialConnectionResult,
    PreparedReference,
    evaluate_local_potential_magnetic_matrices,
    evaluate_magnetic_one_electron_matrices,
    evaluate_magnetic_spatial_connections,
    prepare_ao_quadrature,
)
from aion.electronic_structure.data import immutable_array
from aion.errors import ConfigurationError, MagneticBenchmarkError

_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


class MagneticMatrixFamily(StrEnum):
    """Matrix families selectable by the reusable benchmark workflow."""

    ONE_ELECTRON = "one_electron"
    SPATIAL_CONNECTION = "spatial_connection"
    LOCAL_POTENTIALS = "local_potentials"


@dataclass(frozen=True, slots=True)
class MagneticValidationPolicy:
    """Predeclared internal-identity tolerances for one workflow execution."""

    hermiticity_tolerance: float = 1.0e-10
    direct_oracle_tolerance: float = 1.0e-9
    adjoint_pair_tolerance: float = 1.0e-10
    e1_closure_tolerance: float = 1.0e-5
    fail_on_violation: bool = True

    def __post_init__(self) -> None:
        for name in (
            "hermiticity_tolerance",
            "direct_oracle_tolerance",
            "adjoint_pair_tolerance",
            "e1_closure_tolerance",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ConfigurationError(f"{name} must be a positive finite number")
            checked = float(value)
            if not math.isfinite(checked) or checked <= 0.0:
                raise ConfigurationError(f"{name} must be a positive finite number")
            object.__setattr__(self, name, checked)
        if not isinstance(self.fail_on_violation, bool):
            raise ConfigurationError("fail_on_violation must be boolean")

    def as_mapping(self) -> dict[str, object]:
        return {
            "hermiticity_tolerance": self.hermiticity_tolerance,
            "direct_oracle_tolerance": self.direct_oracle_tolerance,
            "adjoint_pair_tolerance": self.adjoint_pair_tolerance,
            "e1_closure_tolerance": self.e1_closure_tolerance,
            "fail_on_violation": self.fail_on_violation,
        }


@dataclass(frozen=True, slots=True)
class MagneticBenchmarkConfig:
    """Complete normalized definition of one static benchmark execution."""

    magnetic_fields: tuple[UniformMagneticField, ...]
    gauges: tuple[AffineMagneticGauge, ...] = ()
    backend: BackendConfig = field(default_factory=BackendConfig)
    grid_policy: AOGridPolicy = field(default_factory=AOGridPolicy.reference)
    families: tuple[MagneticMatrixFamily, ...] = (
        MagneticMatrixFamily.ONE_ELECTRON,
        MagneticMatrixFamily.SPATIAL_CONNECTION,
    )
    local_potential_identities: tuple[LocalPotentialIdentity, ...] = ()
    block_size: int = 2048
    memory_budget_bytes: int | None = None
    charge: float = -1.0
    mass: float = 1.0
    hbar: float = 1.0
    include_direct_oracle: bool = True
    validation: MagneticValidationPolicy = field(default_factory=MagneticValidationPolicy)

    def __post_init__(self) -> None:
        fields = tuple(self.magnetic_fields)
        if not fields or not all(isinstance(value, UniformMagneticField) for value in fields):
            raise ConfigurationError("benchmark requires at least one UniformMagneticField")
        object.__setattr__(self, "magnetic_fields", fields)
        gauges = tuple(self.gauges)
        if not gauges:
            gauges = tuple(AffineMagneticGauge(value) for value in fields)
        if len(gauges) != len(fields):
            raise ConfigurationError("benchmark gauges must match magnetic fields")
        for magnetic_field, gauge in zip(fields, gauges, strict=True):
            if not isinstance(gauge, AffineMagneticGauge) or gauge.field != magnetic_field:
                raise ConfigurationError("each benchmark gauge must represent its field")
        object.__setattr__(self, "gauges", gauges)
        if not isinstance(self.backend, BackendConfig):
            raise TypeError("backend must be BackendConfig")
        if not isinstance(self.grid_policy, AOGridPolicy):
            raise TypeError("grid_policy must be AOGridPolicy")
        requested_families = tuple(self.families)
        if not requested_families or not all(
            isinstance(value, MagneticMatrixFamily) for value in requested_families
        ):
            raise ConfigurationError("benchmark families are invalid or empty")
        families = tuple(sorted(set(requested_families), key=lambda value: value.value))
        if (
            MagneticMatrixFamily.LOCAL_POTENTIALS in families
            and not self.local_potential_identities
        ):
            raise ConfigurationError("local-potential family requires declared identities")
        if (
            MagneticMatrixFamily.LOCAL_POTENTIALS not in families
            and self.local_potential_identities
        ):
            raise ConfigurationError("local-potential identities require the matching family")
        object.__setattr__(self, "families", families)
        identities = tuple(self.local_potential_identities)
        if not all(isinstance(value, LocalPotentialIdentity) for value in identities):
            raise ConfigurationError("local-potential identities are invalid")
        fingerprints = [value.fingerprint_sha256 for value in identities]
        if len(set(fingerprints)) != len(fingerprints):
            raise ConfigurationError("local-potential identities must be unique")
        object.__setattr__(self, "local_potential_identities", identities)
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, int):
            raise ConfigurationError("benchmark block_size must be an integer")
        if self.block_size <= 0:
            raise ConfigurationError("benchmark block_size must be positive")
        if self.memory_budget_bytes is not None and (
            isinstance(self.memory_budget_bytes, bool)
            or not isinstance(self.memory_budget_bytes, int)
            or self.memory_budget_bytes <= 0
        ):
            raise ConfigurationError("benchmark memory budget must be a positive integer")
        for name in ("charge", "mass", "hbar"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ConfigurationError(f"benchmark {name} must be finite")
            checked = float(value)
            if not math.isfinite(checked) or (name != "charge" and checked <= 0.0):
                raise ConfigurationError(f"benchmark {name} is outside its valid domain")
            object.__setattr__(self, name, checked)
        if not isinstance(self.include_direct_oracle, bool):
            raise ConfigurationError("include_direct_oracle must be boolean")
        if not isinstance(self.validation, MagneticValidationPolicy):
            raise TypeError("validation must be MagneticValidationPolicy")

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema": "aion.magnetic-benchmark-config",
            "version": "1.0.0",
            "magnetic_fields_au": [list(value.magnetic_field_au) for value in self.magnetic_fields],
            "gauges": [_gauge_mapping(value) for value in self.gauges],
            "backend": self.backend.as_mapping(),
            "grid_policy": {
                "kind": self.grid_policy.kind.value,
                "level": self.grid_policy.level,
            },
            "families": [value.value for value in self.families],
            "local_potentials": [
                _identity_mapping(value) for value in self.local_potential_identities
            ],
            "block_size": self.block_size,
            "memory_budget_bytes": self.memory_budget_bytes,
            "charge": self.charge,
            "mass": self.mass,
            "hbar": self.hbar,
            "include_direct_oracle": self.include_direct_oracle,
            "validation": self.validation.as_mapping(),
        }

    @property
    def scientific_id(self) -> str:
        return canonical_sha256(self.as_mapping())


@dataclass(frozen=True, slots=True)
class MagneticMatrixRecord:
    """One immutable named matrix with complete unit and definition metadata."""

    path: str
    values: np.ndarray
    unit: str
    physical_dimension: str
    definition: str

    def __post_init__(self) -> None:
        parts = self.path.split("/")
        if not parts or any(not _PATH_COMPONENT.fullmatch(value) for value in parts):
            raise MagneticBenchmarkError(f"invalid matrix record path {self.path!r}")
        values = immutable_array(self.values, name=f"matrix {self.path}")
        if values.ndim not in (2, 3):
            raise MagneticBenchmarkError("magnetic matrix records must be rank two or three")
        if not self.unit or not self.physical_dimension or not self.definition:
            raise MagneticBenchmarkError("matrix metadata strings must be nonempty")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class MagneticDiagnostic:
    """One normalized scalar diagnostic and optional acceptance threshold."""

    name: str
    value: float
    tolerance: float | None = None
    unit: str = "1"
    physical_dimension: str = "dimensionless"

    def __post_init__(self) -> None:
        if not self.name or not _PATH_COMPONENT.fullmatch(self.name.replace("/", ".")):
            raise MagneticBenchmarkError(f"invalid diagnostic name {self.name!r}")
        value = float(self.value)
        if not math.isfinite(value):
            raise MagneticBenchmarkError(f"diagnostic {self.name} is non-finite")
        object.__setattr__(self, "value", value)
        if self.tolerance is not None:
            tolerance = float(self.tolerance)
            if not math.isfinite(tolerance) or tolerance <= 0.0:
                raise MagneticBenchmarkError("diagnostic tolerance must be positive")
            object.__setattr__(self, "tolerance", tolerance)
        if not self.unit or not self.physical_dimension:
            raise MagneticBenchmarkError("diagnostic unit metadata must be nonempty")

    @property
    def passed(self) -> bool | None:
        return None if self.tolerance is None else self.value <= self.tolerance


@dataclass(frozen=True, slots=True)
class MagneticProgress:
    """One synchronous progress event emitted at a reusable workflow boundary."""

    stage: str
    completed: int
    total: int

    def __post_init__(self) -> None:
        if not self.stage:
            raise MagneticBenchmarkError("progress stage must be nonempty")
        if not 0 <= self.completed <= self.total or self.total <= 0:
            raise MagneticBenchmarkError("progress counts are inconsistent")


@dataclass(frozen=True, slots=True)
class MagneticBenchmarkResult:
    """Immutable host result used by storage, inspection, and campaigns."""

    config: MagneticBenchmarkConfig
    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str
    grid_kind: str
    grid_level: int
    grid_pruning: str
    grid_npoints: int
    ao_provenance_json: str
    matrices: tuple[MagneticMatrixRecord, ...]
    diagnostics: tuple[MagneticDiagnostic, ...]
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, MagneticBenchmarkConfig):
            raise TypeError("config must be MagneticBenchmarkConfig")
        for name in ("reference_fingerprint_sha256", "grid_fingerprint_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise MagneticBenchmarkError(f"{name} is not a lowercase SHA-256 digest")
        if not isinstance(self.grid_kind, str) or not self.grid_kind:
            raise MagneticBenchmarkError("grid_kind must be nonempty")
        if (
            isinstance(self.grid_level, bool)
            or not isinstance(self.grid_level, int)
            or self.grid_level < 0
        ):
            raise MagneticBenchmarkError("grid_level must be a nonnegative integer")
        if not isinstance(self.grid_pruning, str) or not self.grid_pruning:
            raise MagneticBenchmarkError("grid_pruning must be nonempty")
        if (
            isinstance(self.grid_npoints, bool)
            or not isinstance(self.grid_npoints, int)
            or self.grid_npoints <= 0
        ):
            raise MagneticBenchmarkError("grid_npoints must be positive")
        matrices = tuple(sorted(self.matrices, key=lambda value: value.path))
        diagnostics = tuple(self.diagnostics)
        paths = [value.path for value in matrices]
        if len(set(paths)) != len(paths):
            raise MagneticBenchmarkError("magnetic matrix record paths must be unique")
        if not matrices:
            raise MagneticBenchmarkError("magnetic benchmark result contains no matrices")
        try:
            provenance = json.loads(self.ao_provenance_json)
        except json.JSONDecodeError as exc:
            raise MagneticBenchmarkError("AO provenance is not valid JSON") from exc
        if not isinstance(provenance, dict):
            raise MagneticBenchmarkError("AO provenance must be a JSON object")
        normalized_provenance = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "ao_provenance_json", normalized_provenance)
        object.__setattr__(self, "matrices", matrices)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(
            self,
            "result_id",
            canonical_sha256(
                {
                    "schema": "aion.magnetic-benchmark-result",
                    "version": "1.0.0",
                    "config": self.config.as_mapping(),
                    "reference_fingerprint_sha256": self.reference_fingerprint_sha256,
                    "grid_fingerprint_sha256": self.grid_fingerprint_sha256,
                    "grid_kind": self.grid_kind,
                    "grid_level": self.grid_level,
                    "grid_pruning": self.grid_pruning,
                    "grid_npoints": self.grid_npoints,
                    "ao_provenance_json": normalized_provenance,
                    "matrices": [
                        {
                            "path": value.path,
                            "values": value.values,
                            "unit": value.unit,
                            "physical_dimension": value.physical_dimension,
                            "definition": value.definition,
                        }
                        for value in matrices
                    ],
                    "diagnostics": [
                        {
                            "name": value.name,
                            "value": value.value,
                            "tolerance": value.tolerance,
                            "unit": value.unit,
                            "physical_dimension": value.physical_dimension,
                        }
                        for value in diagnostics
                    ],
                }
            ),
        )

    def matrix(self, path: str) -> np.ndarray:
        for record in self.matrices:
            if record.path == path:
                return record.values
        raise KeyError(path)


def run_magnetic_benchmark(
    reference: PreparedReference,
    config: MagneticBenchmarkConfig,
    *,
    local_potential_providers: Sequence[LocalPotentialProvider] = (),
    output_path: str | Path | None = None,
    progress: Callable[[MagneticProgress], None] | None = None,
) -> MagneticBenchmarkResult:
    """Run a static benchmark from an existing reference; never invoke SCF/dynamics."""

    if not isinstance(reference, PreparedReference):
        raise TypeError("reference must be PreparedReference")
    if not isinstance(config, MagneticBenchmarkConfig):
        raise TypeError("config must be MagneticBenchmarkConfig")
    providers = tuple(local_potential_providers)
    if tuple(value.identity for value in providers) != config.local_potential_identities:
        raise MagneticBenchmarkError(
            "local providers do not match the identities frozen in benchmark config"
        )
    _emit(progress, "prepare", 0, 1)
    quadrature = prepare_ao_quadrature(
        reference,
        config.backend,
        grid_policy=config.grid_policy,
        block_size=config.block_size,
        memory_budget_bytes=config.memory_budget_bytes,
    )
    _emit(progress, "prepare", 1, 1)
    core: tuple[MagneticOneElectronResult, ...] = ()
    spatial: tuple[MagneticSpatialConnectionResult, ...] = ()
    local: tuple[LocalPotentialMagneticResult, ...] = ()

    def block_callback(stage: str) -> Callable[[int, int], None]:
        def report(completed: int, total: int) -> None:
            _emit(progress, stage, completed, total)

        return report

    if MagneticMatrixFamily.ONE_ELECTRON in config.families:
        core = evaluate_magnetic_one_electron_matrices(
            quadrature,
            config.magnetic_fields,
            direct_gauges=config.gauges,
            charge=config.charge,
            mass=config.mass,
            hbar=config.hbar,
            memory_budget_bytes=config.memory_budget_bytes,
            include_direct_oracle=config.include_direct_oracle,
            block_callback=block_callback("one_electron"),
        )
    if MagneticMatrixFamily.SPATIAL_CONNECTION in config.families:
        spatial = evaluate_magnetic_spatial_connections(
            quadrature,
            config.magnetic_fields,
            direct_gauges=config.gauges,
            charge=config.charge,
            hbar=config.hbar,
            memory_budget_bytes=config.memory_budget_bytes,
            include_direct_oracle=config.include_direct_oracle,
            block_callback=block_callback("spatial_connection"),
        )
    if MagneticMatrixFamily.LOCAL_POTENTIALS in config.families:
        local = evaluate_local_potential_magnetic_matrices(
            quadrature,
            config.magnetic_fields,
            providers,
            direct_gauges=config.gauges,
            charge=config.charge,
            hbar=config.hbar,
            memory_budget_bytes=config.memory_budget_bytes,
            include_direct_oracle=config.include_direct_oracle,
            block_callback=block_callback("local_potentials"),
        )
    _emit(progress, "finalize", 0, 1)
    matrices = _matrix_records(
        quadrature.backend,
        core,
        spatial,
        local,
        field_count=len(config.magnetic_fields),
        provider_count=len(providers),
    )
    diagnostics = _diagnostics(
        reference,
        core,
        spatial,
        local,
        config.validation,
        quadrature.backend,
    )
    diagnostics += _matrix_norm_diagnostics(matrices, reference)
    diagnostics += _field_reversal_diagnostics(
        matrices,
        config.magnetic_fields,
        config.validation.hermiticity_tolerance,
    )
    failed = [value for value in diagnostics if value.passed is False]
    if failed and config.validation.fail_on_violation:
        summary = ", ".join(
            f"{value.name}={value.value:.3e}>{value.tolerance:.3e}"
            for value in failed
            if value.tolerance is not None
        )
        raise MagneticBenchmarkError(f"magnetic validation failed: {summary}")
    provenance = {
        "evaluator": quadrature.provenance.evaluator,
        "backend": quadrature.provenance.backend,
        "device_index": quadrature.provenance.device_index,
        "block_size": quadrature.provenance.block_size,
        "memory_budget_bytes": quadrature.provenance.memory_budget_bytes,
        "estimated_block_bytes": quadrature.provenance.estimated_block_bytes,
        "dependencies": dict(quadrature.provenance.dependencies),
        "thread_limits": dict(quadrature.provenance.thread_limits),
    }
    result = MagneticBenchmarkResult(
        config=config,
        reference_fingerprint_sha256=reference.fingerprint_sha256,
        grid_fingerprint_sha256=quadrature.grid.fingerprint_sha256,
        grid_kind=quadrature.grid.kind.value,
        grid_level=quadrature.grid.level,
        grid_pruning=quadrature.grid.pruning,
        grid_npoints=quadrature.grid.npoints,
        ao_provenance_json=json.dumps(provenance, sort_keys=True, separators=(",", ":")),
        matrices=matrices,
        diagnostics=diagnostics,
    )
    if output_path is not None:
        from aion.io.magnetic_benchmark import save_magnetic_benchmark

        save_magnetic_benchmark(result, output_path)
    _emit(progress, "finalize", 1, 1)
    return result


def inspect_magnetic_benchmark(result: MagneticBenchmarkResult) -> dict[str, object]:
    """Return a concise scalar summary without replacing authoritative matrices."""

    return {
        "result_id": result.result_id,
        "reference_fingerprint_sha256": result.reference_fingerprint_sha256,
        "grid_fingerprint_sha256": result.grid_fingerprint_sha256,
        "grid_npoints": result.grid_npoints,
        "backend": result.config.backend.as_mapping(),
        "field_count": len(result.config.magnetic_fields),
        "matrix_count": len(result.matrices),
        "diagnostic_count": len(result.diagnostics),
        "failed_diagnostics": [value.name for value in result.diagnostics if value.passed is False],
    }


def save_magnetic_benchmark(result: MagneticBenchmarkResult, path: str | Path) -> str:
    """Publish a result through the branch-local versioned artifact contract."""

    from aion.io.magnetic_benchmark import save_magnetic_benchmark as save

    return save(result, path)


def load_magnetic_benchmark(path: str | Path) -> MagneticBenchmarkResult:
    """Load and content-authenticate a result artifact."""

    from aion.io.magnetic_benchmark import load_magnetic_benchmark as load

    return load(path)


def _matrix_records(
    backend: Any,
    core: Sequence[Any],
    spatial: Sequence[Any],
    local: Sequence[Any],
    *,
    field_count: int,
    provider_count: int,
) -> tuple[MagneticMatrixRecord, ...]:
    records: list[MagneticMatrixRecord] = []

    def add(path: str, values: Any, unit: str, dimension: str, definition: str) -> None:
        records.append(
            MagneticMatrixRecord(
                path,
                backend.to_host(values),
                unit,
                dimension,
                definition,
            )
        )

    for index, result in enumerate(core):
        root = f"{index:04d}"
        add(f"{root}/endpoint_link", result.endpoint_link, "1", "wilson_link", "Theta")
        for family_name in ("overlap", "nuclear_attraction"):
            hierarchy = getattr(result, family_name)
            unit = "1" if family_name == "overlap" else "hartree"
            dimension = "overlap" if family_name == "overlap" else "energy_operator"
            for name in (
                "zero",
                "quadrature_zero",
                "first_F",
                "second_F2",
                "exact_grid",
                "exact",
                "b1",
                "b2",
            ):
                add(
                    f"{root}/{family_name}/{name}",
                    getattr(hierarchy, name),
                    unit,
                    dimension,
                    f"{family_name}.{name}",
                )
        kinetic = result.kinetic
        for name in (
            "zero",
            "quadrature_zero",
            "first_F",
            "first_pC",
            "first_Cp",
            "first",
            "second_F2",
            "second_F_pC",
            "second_F_Cp",
            "second_C2",
            "second",
            "exact_grid",
            "exact",
            "b1",
            "b2",
        ):
            add(
                f"{root}/kinetic/{name}",
                getattr(kinetic, name),
                "hartree",
                "energy_operator",
                f"kinetic.{name}",
            )
        for sector_name, sectors in (
            ("quadrature_exact_sectors", kinetic.quadrature_exact_sectors),
            ("exact_sectors", kinetic.exact_sectors),
        ):
            for name in ("pp", "pC", "Cp", "C2"):
                add(
                    f"{root}/kinetic/{sector_name}/{name}",
                    getattr(sectors, name),
                    "hartree",
                    "energy_operator",
                    f"kinetic.{sector_name}.{name}",
                )
        for name in ("overlap", "kinetic", "nuclear_attraction"):
            add(
                f"{root}/lower_exact/{name}",
                getattr(result.lower_exact, name),
                "1" if name == "overlap" else "hartree",
                "overlap" if name == "overlap" else "energy_operator",
                f"endpoint_link*{name}.exact",
            )
        if result.direct_oracle is not None:
            for name in ("overlap", "kinetic", "nuclear_attraction"):
                add(
                    f"{root}/direct_one_electron/{name}",
                    getattr(result.direct_oracle.lower_grid, name),
                    "1" if name == "overlap" else "hartree",
                    "overlap" if name == "overlap" else "energy_operator",
                    f"direct_gauge.{name}.lower_grid",
                )
    for index, result in enumerate(spatial):
        root = f"{index:04d}/spatial_connection"
        hierarchy = result.spatial_connection
        for name in (
            "zero",
            "quadrature_zero",
            "first_F",
            "first_C",
            "first",
            "second_F2",
            "second_FC",
            "second",
            "exact_grid",
            "exact",
            "b1",
            "b2",
        ):
            add(
                f"{root}/{name}",
                getattr(hierarchy, name),
                "bohr^-1",
                "spatial_connection",
                f"omega_spatial.{name}",
            )
        add(
            f"{root}/e1_first_C_closure",
            result.e1_first_C_closure,
            "bohr^-1",
            "spatial_connection",
            "i*(d cross B)/(2*hbar)",
        )
        add(
            f"{root}/lower_exact",
            result.lower_exact,
            "bohr^-1",
            "spatial_connection",
            "Theta*omega_spatial.exact",
        )
        if result.direct_oracle is not None:
            add(
                f"{root}/direct_lower_grid",
                result.direct_oracle.lower_grid,
                "bohr^-1",
                "spatial_connection",
                "direct covariant-derivative lower grid matrix",
            )
    if local:
        if provider_count <= 0 or len(local) != field_count * provider_count:
            raise MagneticBenchmarkError("local result cardinality is inconsistent")
        for flat_index, result in enumerate(local):
            field_index = flat_index // provider_count
            provider_index = flat_index % provider_count
            root = f"{field_index:04d}/local/{provider_index:04d}_{result.identity.name}"
            for name in (
                "zero",
                "quadrature_zero",
                "first_F",
                "second_F2",
                "exact_grid",
                "exact",
                "b1",
                "b2",
            ):
                add(
                    f"{root}/{name}",
                    getattr(result.hierarchy, name),
                    "hartree",
                    "energy_operator",
                    f"local.{result.identity.name}.{name}",
                )
            if result.direct_lower_grid is not None:
                add(
                    f"{root}/direct_lower_grid",
                    result.direct_lower_grid,
                    "hartree",
                    "energy_operator",
                    f"direct_gauge.local.{result.identity.name}",
                )
    return tuple(records)


def _diagnostics(
    reference: PreparedReference,
    core: Sequence[Any],
    spatial: Sequence[Any],
    local: Sequence[Any],
    policy: MagneticValidationPolicy,
    backend: Any,
) -> tuple[MagneticDiagnostic, ...]:
    values: list[MagneticDiagnostic] = []

    def add(
        name: str,
        value: float,
        tolerance: float | None = None,
        *,
        unit: str = "1",
        physical_dimension: str = "dimensionless",
    ) -> None:
        values.append(
            MagneticDiagnostic(
                name,
                value,
                tolerance,
                unit,
                physical_dimension,
            )
        )

    for index, result in enumerate(core):
        prefix = f"field{index:04d}"
        for name in ("overlap", "kinetic", "nuclear_attraction"):
            hierarchy = getattr(result, name)
            add(
                f"{prefix}.{name}.hermiticity",
                _adjoint_residual(hierarchy.exact, backend, sign=1.0),
                policy.hermiticity_tolerance,
            )
            add(
                f"{prefix}.{name}.zero_grid",
                _relative(hierarchy.quadrature_zero, hierarchy.zero, backend),
            )
            add(
                f"{prefix}.{name}.B1_error",
                _relative(hierarchy.b1, hierarchy.exact, backend),
            )
            add(
                f"{prefix}.{name}.B2_error",
                _relative(hierarchy.b2, hierarchy.exact, backend),
            )
        add(
            f"{prefix}.kinetic.pC_adjoint",
            _relative(
                result.kinetic.first_pC.conj().T,
                result.kinetic.first_Cp,
                backend,
            ),
            policy.adjoint_pair_tolerance,
        )
        first_c = backend.to_host(result.kinetic.first_C)
        first_f = backend.to_host(result.kinetic.first_F)
        c_norm = float(np.linalg.norm(first_c))
        f_norm = float(np.linalg.norm(first_f))
        add(f"{prefix}.kinetic.F_over_C", f_norm / max(c_norm, np.finfo(float).tiny))
        metric_eigenvalues = np.linalg.eigvalsh(backend.to_host(result.overlap.exact))
        add(f"{prefix}.metric.minimum_eigenvalue", float(metric_eigenvalues.min()))
        add(f"{prefix}.metric.maximum_eigenvalue", float(metric_eigenvalues.max()))
        add(
            f"{prefix}.metric.condition_number",
            float(metric_eigenvalues.max() / metric_eigenvalues.min()),
        )
        if result.direct_oracle is not None:
            for name in ("overlap", "kinetic", "nuclear_attraction"):
                add(
                    f"{prefix}.{name}.direct_oracle",
                    _relative(
                        getattr(result.direct_oracle.lower_grid, name),
                        getattr(result.lower_exact_grid, name),
                        backend,
                    ),
                    policy.direct_oracle_tolerance,
                )
        anchors = reference.anchor_topology.ao_to_atom
        onsite = anchors[:, None] == anchors[None, :]
        add(
            f"{prefix}.kinetic.first_F.onsite_norm",
            float(np.linalg.norm(first_f * onsite)),
            unit="hartree",
            physical_dimension="energy_operator_norm",
        )
        add(
            f"{prefix}.kinetic.first_F.intersite_norm",
            float(np.linalg.norm(first_f * ~onsite)),
            unit="hartree",
            physical_dimension="energy_operator_norm",
        )
        add(
            f"{prefix}.kinetic.first_C.onsite_norm",
            float(np.linalg.norm(first_c * onsite)),
            unit="hartree",
            physical_dimension="energy_operator_norm",
        )
        add(
            f"{prefix}.kinetic.first_C.intersite_norm",
            float(np.linalg.norm(first_c * ~onsite)),
            unit="hartree",
            physical_dimension="energy_operator_norm",
        )
    for index, result in enumerate(spatial):
        prefix = f"field{index:04d}.spatial_connection"
        hierarchy = result.spatial_connection
        for name in ("zero", "first_F", "first_C", "second_F2", "second_FC", "exact"):
            add(
                f"{prefix}.{name}.antihermiticity",
                _adjoint_residual(getattr(hierarchy, name), backend, sign=-1.0),
                policy.hermiticity_tolerance,
            )
        add(
            f"{prefix}.e1_C_closure",
            _relative(hierarchy.first_C, result.e1_first_C_closure, backend),
            policy.e1_closure_tolerance,
        )
        add(f"{prefix}.B1_error", _relative(hierarchy.b1, hierarchy.exact, backend))
        add(f"{prefix}.B2_error", _relative(hierarchy.b2, hierarchy.exact, backend))
        if result.direct_oracle is not None:
            add(
                f"{prefix}.direct_oracle",
                _relative(
                    result.direct_oracle.lower_grid,
                    result.lower_exact_grid,
                    backend,
                ),
                policy.direct_oracle_tolerance,
            )
    for flat_index, result in enumerate(local):
        prefix = f"local{flat_index:04d}.{result.identity.name}"
        add(
            f"{prefix}.hermiticity",
            _adjoint_residual(result.hierarchy.exact, backend, sign=1.0),
            policy.hermiticity_tolerance,
        )
        if result.direct_lower_grid is not None:
            add(
                f"{prefix}.direct_oracle",
                _relative(result.direct_lower_grid, result.lower_exact_grid, backend),
                policy.direct_oracle_tolerance,
            )
    return tuple(values)


def _matrix_norm_diagnostics(
    records: Sequence[MagneticMatrixRecord],
    reference: PreparedReference,
) -> tuple[MagneticDiagnostic, ...]:
    anchors = reference.anchor_topology.ao_to_atom
    same_anchor = anchors[:, None] == anchors[None, :]
    values: list[MagneticDiagnostic] = []
    for record in records:
        matrix = record.values
        if matrix.shape[-2:] != same_anchor.shape:
            raise MagneticBenchmarkError(
                f"matrix {record.path!r} is incompatible with reference AO anchors"
            )
        mask = same_anchor if matrix.ndim == 2 else same_anchor[None, :, :]
        root = f"matrix/{record.path}"
        dimension = f"{record.physical_dimension}_norm"
        values.extend(
            (
                MagneticDiagnostic(
                    f"{root}/frobenius",
                    float(np.linalg.norm(matrix)),
                    unit=record.unit,
                    physical_dimension=dimension,
                ),
                MagneticDiagnostic(
                    f"{root}/maximum_abs",
                    float(np.max(np.abs(matrix))),
                    unit=record.unit,
                    physical_dimension=dimension,
                ),
                MagneticDiagnostic(
                    f"{root}/same_anchor_frobenius",
                    float(np.linalg.norm(matrix * mask)),
                    unit=record.unit,
                    physical_dimension=dimension,
                ),
                MagneticDiagnostic(
                    f"{root}/intersite_frobenius",
                    float(np.linalg.norm(matrix * ~mask)),
                    unit=record.unit,
                    physical_dimension=dimension,
                ),
            )
        )
    return tuple(values)


def _field_reversal_diagnostics(
    records: Sequence[MagneticMatrixRecord],
    fields: Sequence[UniformMagneticField],
    tolerance: float,
) -> tuple[MagneticDiagnostic, ...]:
    by_path = {value.path: value for value in records}
    values: list[MagneticDiagnostic] = []
    for positive_index, magnetic_field in enumerate(fields):
        positive = np.asarray(magnetic_field.magnetic_field_au)
        if np.linalg.norm(positive) == 0.0:
            continue
        for negative_index in range(positive_index + 1, len(fields)):
            negative = np.asarray(fields[negative_index].magnetic_field_au)
            if not np.array_equal(negative, -positive):
                continue
            positive_root = f"{positive_index:04d}/"
            negative_root = f"{negative_index:04d}/"
            for path, positive_record in by_path.items():
                if not path.startswith(positive_root) or not path.endswith(
                    ("/exact", "/b1", "/b2")
                ):
                    continue
                suffix = path.removeprefix(positive_root)
                negative_record = by_path.get(negative_root + suffix)
                if negative_record is None:
                    continue
                values.append(
                    MagneticDiagnostic(
                        name=(f"field_reversal/{positive_index:04d}_{negative_index:04d}/{suffix}"),
                        value=_host_relative(
                            negative_record.values,
                            positive_record.values.conj(),
                        ),
                        tolerance=tolerance,
                    )
                )
    return tuple(values)


def _host_relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(1.0, np.linalg.norm(right)))


def _relative(left: Any, right: Any, backend: Any) -> float:
    left_host = backend.to_host(left)
    right_host = backend.to_host(right)
    return float(np.linalg.norm(left_host - right_host) / max(1.0, np.linalg.norm(right_host)))


def _adjoint_residual(value: Any, backend: Any, *, sign: float) -> float:
    array = backend.to_host(value)
    adjoint = array.conj().swapaxes(-1, -2)
    return float(np.linalg.norm(array - sign * adjoint) / max(1.0, np.linalg.norm(array)))


def _emit(
    callback: Callable[[MagneticProgress], None] | None,
    stage: str,
    completed: int,
    total: int,
) -> None:
    if callback is not None:
        callback(MagneticProgress(stage, completed, total))


def _identity_mapping(value: LocalPotentialIdentity) -> dict[str, object]:
    return {
        "name": value.name,
        "version": value.version,
        "provenance": [list(item) for item in value.provenance],
        "unit": value.unit.value,
        "physical_dimension": value.physical_dimension.value,
        "fingerprint_sha256": value.fingerprint_sha256,
    }


def _gauge_mapping(value: AffineMagneticGauge) -> dict[str, object]:
    return {
        "kind": value.kind.value,
        "origin_au": list(value.origin_au),
        "landau_axis": None if value.landau_axis is None else list(value.landau_axis),
    }
