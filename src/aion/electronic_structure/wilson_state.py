"""Portable immutable exact/reduced Wilson stationary-state records."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from os import PathLike
from typing import Any, cast

import numpy as np

from aion.backends import ArrayBackend
from aion.config import WilsonStationaryConfig, canonical_sha256
from aion.electromagnetism import (
    UniformMagneticSourceSample,
    build_affine_electromagnetic_source,
)
from aion.electronic_structure.data import immutable_array
from aion.electronic_structure.ri_wilson_hartree import RIWilsonHartreeEvaluator
from aion.electronic_structure.wilson_stationary import (
    ExactWilsonStationaryState,
    StationarySCFIteration,
    WilsonStationaryActionProtocol,
)
from aion.errors import ReferencePreparationError, WilsonStateError

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _digest(value: str, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise WilsonStateError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise WilsonStateError(f"{name} must be finite")
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise WilsonStateError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise WilsonStateError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class WilsonStationaryEnergyComponents:
    kinetic_au: float
    electron_nuclear_au: float
    one_electron_au: float
    hartree_au: float
    exchange_correlation_au: float
    nuclear_repulsion_au: float
    electronic_au: float
    molecular_total_au: float

    def __post_init__(self) -> None:
        for name in (
            "kinetic_au",
            "electron_nuclear_au",
            "one_electron_au",
            "hartree_au",
            "exchange_correlation_au",
            "nuclear_repulsion_au",
            "electronic_au",
            "molecular_total_au",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), f"energy.{name}"))
        identities = (
            (self.kinetic_au + self.electron_nuclear_au, self.one_electron_au),
            (
                self.one_electron_au + self.hartree_au + self.exchange_correlation_au,
                self.electronic_au,
            ),
            (self.electronic_au + self.nuclear_repulsion_au, self.molecular_total_au),
        )
        for reconstructed, recorded in identities:
            scale = max(1.0, abs(reconstructed), abs(recorded))
            if abs(reconstructed - recorded) > 2.0e-11 * scale:
                raise WilsonStateError("stationary energy components are inconsistent")

    def as_mapping(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WilsonStationaryResiduals:
    orbital: float
    density_fixed_point: float
    commutator: float
    metric_orthonormality: float
    particle_number: float
    particle_number_residual: float
    occupation_spectrum_imaginary_max_abs: float
    occupation_spectrum_residual: float
    closed_shell_density_polynomial: float
    orbital_frequency_occupation_commutator: float
    orbital_energy_sum_au: float
    double_counting_reconstructed_electronic_energy_au: float
    double_counting_reconstructed_molecular_energy_au: float
    double_counting_residual_au: float
    metric_minimum_eigenvalue: float
    metric_condition_number: float
    converged: bool

    def __post_init__(self) -> None:
        for name in (
            "orbital",
            "density_fixed_point",
            "commutator",
            "metric_orthonormality",
            "particle_number",
            "particle_number_residual",
            "occupation_spectrum_imaginary_max_abs",
            "occupation_spectrum_residual",
            "closed_shell_density_polynomial",
            "orbital_frequency_occupation_commutator",
            "orbital_energy_sum_au",
            "double_counting_reconstructed_electronic_energy_au",
            "double_counting_reconstructed_molecular_energy_au",
            "double_counting_residual_au",
            "metric_minimum_eigenvalue",
            "metric_condition_number",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), f"residuals.{name}"))
        if not isinstance(self.converged, bool):
            raise WilsonStateError("residuals.converged must be boolean")
        if not self.converged:
            raise WilsonStateError("a completed stationary state must be converged")
        if self.metric_minimum_eigenvalue <= 0.0:
            raise WilsonStateError("stationary metric must be positive definite")
        if self.metric_condition_number < 1.0:
            raise WilsonStateError("stationary metric condition number cannot be below one")

    def as_mapping(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WilsonStationaryStateData:
    """Backend-neutral immutable state authenticated by its scientific content."""

    config: WilsonStationaryConfig
    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str
    auxiliary_space_fingerprint_sha256: str
    source_sample: UniformMagneticSourceSample
    coefficients: np.ndarray
    occupations: np.ndarray
    contravariant_density: np.ndarray
    mixed_density: np.ndarray
    metric: np.ndarray
    orbital_frequency_matrix: np.ndarray
    active_orbital_energies_au: np.ndarray
    complete_orbital_spectrum_au: np.ndarray
    occupation_spectrum: np.ndarray
    energies: WilsonStationaryEnergyComponents
    residuals: WilsonStationaryResiduals
    iterations: tuple[StationarySCFIteration, ...]
    source_fingerprint_sha256: str = field(init=False)
    action_fingerprint_sha256: str = field(init=False)
    fingerprint_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, WilsonStationaryConfig):
            raise WilsonStateError("state config must be a WilsonStationaryConfig")
        reference = _digest(
            self.reference_fingerprint_sha256,
            "reference_fingerprint_sha256",
        )
        if reference != self.config.reference.fingerprint_sha256:
            raise WilsonStateError("stationary state reference fingerprint disagrees with config")
        object.__setattr__(self, "reference_fingerprint_sha256", reference)
        for name in (
            "grid_fingerprint_sha256",
            "auxiliary_space_fingerprint_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if not isinstance(self.source_sample, UniformMagneticSourceSample):
            raise WilsonStateError("source_sample must be a UniformMagneticSourceSample")
        if self.source_sample.time_au != self.config.source_time_au:
            raise WilsonStateError("stationary source sample time disagrees with config")
        provider = build_affine_electromagnetic_source(
            self.config.source,
            self.source_sample.origin_au,
        )
        expected_sample = provider.sample(self.config.source_time_au)
        if _source_mapping(expected_sample) != _source_mapping(self.source_sample):
            raise WilsonStateError("stationary source sample disagrees with analytic config")
        object.__setattr__(
            self,
            "source_fingerprint_sha256",
            provider.fingerprint_sha256,
        )

        arrays = {
            "coefficients": (self.coefficients, np.complex128, 2),
            "occupations": (self.occupations, np.float64, 1),
            "contravariant_density": (self.contravariant_density, np.complex128, 2),
            "mixed_density": (self.mixed_density, np.complex128, 2),
            "metric": (self.metric, np.complex128, 2),
            "orbital_frequency_matrix": (self.orbital_frequency_matrix, np.complex128, 2),
            "active_orbital_energies_au": (
                self.active_orbital_energies_au,
                np.float64,
                1,
            ),
            "complete_orbital_spectrum_au": (
                self.complete_orbital_spectrum_au,
                np.float64,
                1,
            ),
            "occupation_spectrum": (self.occupation_spectrum, np.float64, 1),
        }
        try:
            for name, (value, dtype, ndim) in arrays.items():
                object.__setattr__(
                    self,
                    name,
                    immutable_array(
                        value,
                        dtype=dtype,
                        ndim=ndim,
                        name=name.replace("_", " "),
                    ),
                )
        except ReferencePreparationError as exc:
            raise WilsonStateError(str(exc)) from exc
        nao = self.metric.shape[0]
        if self.metric.shape != (nao, nao) or nao == 0:
            raise WilsonStateError("stationary metric must be a nonempty square matrix")
        noccupied = self.coefficients.shape[1]
        if self.coefficients.shape[0] != nao or self.occupations.shape != (noccupied,):
            raise WilsonStateError("stationary coefficient and occupation shapes disagree")
        square_names = (
            "contravariant_density",
            "mixed_density",
            "metric",
        )
        if any(getattr(self, name).shape != (nao, nao) for name in square_names):
            raise WilsonStateError("stationary AO matrix shapes disagree")
        if self.orbital_frequency_matrix.shape != (noccupied, noccupied):
            raise WilsonStateError("orbital frequency matrix has the wrong shape")
        if self.active_orbital_energies_au.shape != (noccupied,):
            raise WilsonStateError("active orbital energies have the wrong shape")
        if self.complete_orbital_spectrum_au.shape != (nao,):
            raise WilsonStateError("complete orbital spectrum has the wrong shape")
        if self.occupation_spectrum.shape != (nao,):
            raise WilsonStateError("occupation spectrum has the wrong shape")
        for name in ("metric", "contravariant_density", "orbital_frequency_matrix"):
            value = getattr(self, name)
            residual = np.linalg.norm(value - value.conj().T) / max(
                1.0,
                float(np.linalg.norm(value)),
            )
            if residual > 1.0e-10:
                raise WilsonStateError(f"{name} is not Hermitian")
        mixed_residual = np.linalg.norm(
            self.mixed_density - self.contravariant_density @ self.metric
        ) / max(1.0, float(np.linalg.norm(self.mixed_density)))
        if mixed_residual > 1.0e-10:
            raise WilsonStateError("mixed density disagrees with P S")
        coefficient_density = (
            self.coefficients * self.occupations[None, :]
        ) @ self.coefficients.conj().T
        coefficient_residual = np.linalg.norm(
            self.contravariant_density - coefficient_density
        ) / max(1.0, float(np.linalg.norm(self.contravariant_density)))
        if coefficient_residual > 1.0e-10:
            raise WilsonStateError("contravariant density disagrees with occupied coefficients")
        spectrum = np.sort(np.linalg.eigvals(self.mixed_density).real)[::-1]
        spectrum_residual = np.linalg.norm(spectrum - self.occupation_spectrum) / max(
            1.0,
            float(np.linalg.norm(self.occupation_spectrum)),
        )
        if spectrum_residual > 1.0e-10:
            raise WilsonStateError("stored occupation spectrum disagrees with mixed density")
        if not isinstance(self.energies, WilsonStationaryEnergyComponents):
            raise WilsonStateError("stationary energies have the wrong type")
        if not isinstance(self.residuals, WilsonStationaryResiduals):
            raise WilsonStateError("stationary residuals have the wrong type")
        object.__setattr__(self, "iterations", tuple(self.iterations))
        if not self.iterations or not all(
            isinstance(value, StationarySCFIteration) for value in self.iterations
        ):
            raise WilsonStateError("stationary iterations must be nonempty SCF records")
        for index, record in enumerate(self.iterations, start=1):
            if record.iteration != index:
                raise WilsonStateError("stationary iteration indices must be contiguous")
            for name in (
                "energy_molecular_total_au",
                "density_fixed_point_residual",
                "commutator_residual",
            ):
                _finite(getattr(record, name), f"iterations[{index}].{name}")
            if record.energy_change_au is not None:
                _finite(record.energy_change_au, f"iterations[{index}].energy_change_au")
            if record.diis_dimension < 0 or not isinstance(record.used_diis, bool):
                raise WilsonStateError("stationary iteration DIIS metadata is invalid")

        action_fingerprint = canonical_sha256(
            {
                "schema": "aion.wilson-action-realization",
                "version": "1.0.0",
                "reference_fingerprint_sha256": self.reference_fingerprint_sha256,
                "grid_fingerprint_sha256": self.grid_fingerprint_sha256,
                "auxiliary_space_fingerprint_sha256": (self.auxiliary_space_fingerprint_sha256),
                "source_fingerprint_sha256": self.source_fingerprint_sha256,
                "source_sample": _source_mapping(self.source_sample),
                "action": self.config.action.as_mapping(),
                "numerics": self.config.numerics.as_mapping(),
            }
        )
        object.__setattr__(self, "action_fingerprint_sha256", action_fingerprint)
        object.__setattr__(
            self,
            "fingerprint_sha256",
            canonical_sha256(self.scientific_mapping()),
        )

    def scientific_mapping(self) -> dict[str, object]:
        return {
            "schema": "aion.wilson-stationary-state-content",
            "version": "1.0.0",
            "configuration_scientific_id_sha256": self.config.scientific_id,
            "reference_fingerprint_sha256": self.reference_fingerprint_sha256,
            "grid_fingerprint_sha256": self.grid_fingerprint_sha256,
            "auxiliary_space_fingerprint_sha256": self.auxiliary_space_fingerprint_sha256,
            "source_fingerprint_sha256": self.source_fingerprint_sha256,
            "action_fingerprint_sha256": self.action_fingerprint_sha256,
            "source_sample": _source_mapping(self.source_sample),
            "coefficients": self.coefficients,
            "occupations": self.occupations,
            "contravariant_density": self.contravariant_density,
            "mixed_density": self.mixed_density,
            "metric": self.metric,
            "orbital_frequency_matrix": self.orbital_frequency_matrix,
            "active_orbital_energies_au": self.active_orbital_energies_au,
            "complete_orbital_spectrum_au": self.complete_orbital_spectrum_au,
            "occupation_spectrum": self.occupation_spectrum,
            "energies": self.energies.as_mapping(),
            "residuals": self.residuals.as_mapping(),
            "iterations": [asdict(value) for value in self.iterations],
        }

    def save(self, path: str | PathLike[str] | None = None) -> None:
        """Transactionally publish this stationary state."""

        from aion.electronic_structure.wilson_state_io import (
            save_wilson_stationary_state,
        )

        target = self.config.output.artifact_path if path is None else path
        save_wilson_stationary_state(self, target)


def auxiliary_space_fingerprint(evaluator: RIWilsonHartreeEvaluator) -> str:
    """Authenticate one realized RI auxiliary metric and retained subspace."""

    if not isinstance(evaluator, RIWilsonHartreeEvaluator):
        raise TypeError("evaluator must be an RIWilsonHartreeEvaluator")
    backend = evaluator.quadrature.backend
    return canonical_sha256(
        {
            "schema": "aion.ri-wilson-auxiliary-space",
            "version": "1.0.0",
            "provenance": asdict(evaluator.provenance),
            "metric": backend.to_host(evaluator.metric),
            "metric_inverse": backend.to_host(evaluator.metric_inverse),
            "retained_projector": backend.to_host(evaluator.retained_projector),
        }
    )


def capture_wilson_stationary_state[ActionT: WilsonStationaryActionProtocol](
    config: WilsonStationaryConfig,
    state: ExactWilsonStationaryState[ActionT],
    source_sample: UniformMagneticSourceSample,
    *,
    grid_fingerprint_sha256: str,
    auxiliary_space_fingerprint_sha256: str,
    backend: ArrayBackend,
) -> WilsonStationaryStateData:
    """Copy one converged backend state into the portable immutable contract."""

    if not isinstance(state, ExactWilsonStationaryState):
        raise TypeError("state must be an ExactWilsonStationaryState")
    action: Any = state.action
    if getattr(action, "branch", None) is not config.action.branch:
        raise WilsonStateError("stationary action branch disagrees with configuration")
    configured_level = getattr(config.action, "level", None)
    if getattr(action, "level", configured_level) is not configured_level:
        raise WilsonStateError("stationary reduced level disagrees with configuration")
    for name, expected in (
        ("reference_fingerprint_sha256", config.reference.fingerprint_sha256),
        ("grid_fingerprint_sha256", grid_fingerprint_sha256),
    ):
        actual = getattr(action, name, expected)
        if actual != expected:
            raise WilsonStateError(f"stationary action {name} disagrees with configuration")
    if (
        backend.kind is not config.backend.kind
        or backend.device_index != config.backend.device_index
    ):
        raise WilsonStateError("stationary execution backend disagrees with configuration")

    def host(value: object) -> np.ndarray:
        return backend.to_host(value)

    def scalar(value: object) -> float:
        if backend.is_resident(value):
            return backend.scalar_to_float(value)
        return _finite(value, "stationary scalar")

    energies = WilsonStationaryEnergyComponents(
        kinetic_au=scalar(action.energy_kinetic_au),
        electron_nuclear_au=scalar(action.energy_electron_nuclear_au),
        one_electron_au=scalar(action.energy_one_electron_au),
        hartree_au=scalar(action.energy_hartree_au),
        exchange_correlation_au=scalar(action.energy_exchange_correlation_au),
        nuclear_repulsion_au=scalar(action.energy_nuclear_repulsion_au),
        electronic_au=scalar(action.energy_electronic_au),
        molecular_total_au=scalar(action.energy_molecular_total_au),
    )
    residuals = WilsonStationaryResiduals(
        orbital=state.orbital_residual,
        density_fixed_point=state.density_fixed_point_residual,
        commutator=state.commutator_residual,
        metric_orthonormality=state.metric_orthonormality_residual,
        particle_number=state.particle_number,
        particle_number_residual=state.particle_number_residual,
        occupation_spectrum_imaginary_max_abs=(state.occupation_spectrum_imaginary_max_abs),
        occupation_spectrum_residual=state.occupation_spectrum_residual,
        closed_shell_density_polynomial=state.closed_shell_density_polynomial_residual,
        orbital_frequency_occupation_commutator=(
            state.orbital_frequency_occupation_commutator_residual
        ),
        orbital_energy_sum_au=scalar(state.orbital_energy_sum_au),
        double_counting_reconstructed_electronic_energy_au=scalar(
            state.double_counting_reconstructed_electronic_energy_au
        ),
        double_counting_reconstructed_molecular_energy_au=scalar(
            state.double_counting_reconstructed_molecular_energy_au
        ),
        double_counting_residual_au=state.double_counting_residual_au,
        metric_minimum_eigenvalue=state.metric_minimum_eigenvalue,
        metric_condition_number=state.metric_condition_number,
        converged=state.converged,
    )
    return WilsonStationaryStateData(
        config=config,
        reference_fingerprint_sha256=config.reference.fingerprint_sha256,
        grid_fingerprint_sha256=grid_fingerprint_sha256,
        auxiliary_space_fingerprint_sha256=auxiliary_space_fingerprint_sha256,
        source_sample=source_sample,
        coefficients=host(state.coefficients),
        occupations=host(state.occupations),
        contravariant_density=host(state.coefficient_density),
        mixed_density=host(state.mixed_density),
        metric=host(action.overlap),
        orbital_frequency_matrix=host(state.orbital_frequency_matrix),
        active_orbital_energies_au=host(state.active_orbital_energies_au),
        complete_orbital_spectrum_au=host(state.complete_orbital_spectrum_au),
        occupation_spectrum=state.occupation_spectrum,
        energies=energies,
        residuals=residuals,
        iterations=state.iterations,
    )


def _source_mapping(sample: UniformMagneticSourceSample) -> dict[str, object]:
    return {
        "time_au": sample.time_au,
        "magnetic_field_au": list(sample.field.magnetic_field_au),
        "magnetic_field_dot_au": list(sample.magnetic_field_dot_au),
        "electric_field_origin_au": list(sample.electric_field_origin_au),
        "origin_au": list(sample.origin_au),
        "gauge_kind": sample.gauge_kind.value,
        "landau_axis": None if sample.landau_axis is None else list(sample.landau_axis),
    }
