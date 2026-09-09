"""Immutable numerical contracts for reusable molecular spectroscopy."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from aion.config import canonical_sha256
from aion.errors import SpectroscopyError


class WindowKind(StrEnum):
    """Explicit finite-record windows supported by the transform."""

    RECTANGULAR = "rectangular"


class QuadratureKind(StrEnum):
    """Time quadrature used to approximate the continuous transform."""

    TRAPEZOIDAL = "trapezoidal"


class BaselineKind(StrEnum):
    """Explicit baseline convention for a kick-induced dipole change."""

    EVENT_PRE = "event_pre"
    FIRST_SAMPLE = "first_sample"
    NONE = "none"


def _finite_scalar(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SpectroscopyError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and " if positive else ""
        raise SpectroscopyError(f"{name} must be {qualifier}finite")
    return result


def _array(
    value: object,
    dtype: Any,
    name: str,
    *,
    ndim: int | None = None,
) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    if ndim is not None and result.ndim != ndim:
        raise SpectroscopyError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(result)):
        raise SpectroscopyError(f"{name} contains non-finite values")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class TransformConfig:
    """Complete, reproducible positive-frequency transform configuration."""

    damping_energy_au: float = 0.0
    zero_padding_factor: int = 1
    maximum_energy_au: float | None = None
    window: WindowKind = WindowKind.RECTANGULAR
    quadrature: QuadratureKind = QuadratureKind.TRAPEZOIDAL

    def __post_init__(self) -> None:
        damping = _finite_scalar(self.damping_energy_au, "damping energy")
        if damping < 0.0:
            raise SpectroscopyError("damping energy cannot be negative")
        object.__setattr__(self, "damping_energy_au", damping)
        if (
            isinstance(self.zero_padding_factor, bool)
            or not isinstance(self.zero_padding_factor, int)
            or self.zero_padding_factor < 1
        ):
            raise SpectroscopyError("zero-padding factor must be a positive integer")
        if self.maximum_energy_au is not None:
            object.__setattr__(
                self,
                "maximum_energy_au",
                _finite_scalar(self.maximum_energy_au, "maximum energy", positive=True),
            )
        if not isinstance(self.window, WindowKind):
            raise SpectroscopyError("window must be a WindowKind")
        if not isinstance(self.quadrature, QuadratureKind):
            raise SpectroscopyError("quadrature must be a QuadratureKind")

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema": "aion.transform-config",
            "version": "1.0.0",
            "fourier_exponent_sign": 1,
            "normalization": "continuous_time_trapezoidal",
            "time_origin": "first_sample",
            "window": self.window.value,
            "damping_energy_au": self.damping_energy_au,
            "damping_fwhm_au": 2.0 * self.damping_energy_au,
            "zero_padding_factor": self.zero_padding_factor,
            "maximum_energy_au": self.maximum_energy_au,
            "quadrature": self.quadrature.value,
        }


@dataclass(frozen=True, slots=True)
class KickSpectrumConfig:
    """Postprocessing choices specific to an impulsive dipole response."""

    transform: TransformConfig = field(default_factory=TransformConfig)
    baseline: BaselineKind = BaselineKind.EVENT_PRE

    def __post_init__(self) -> None:
        if not isinstance(self.transform, TransformConfig):
            raise SpectroscopyError("kick-spectrum transform has the wrong type")
        if not isinstance(self.baseline, BaselineKind):
            raise SpectroscopyError("kick-spectrum baseline must be a BaselineKind")

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema": "aion.kick-spectrum-config",
            "version": "1.0.0",
            "baseline": self.baseline.value,
            "transform": self.transform.as_mapping(),
        }


@dataclass(frozen=True, slots=True)
class FourierTransformResult:
    """Positive-frequency samples of one continuous-time transform."""

    config: TransformConfig
    omega_au: np.ndarray
    values: np.ndarray
    sample_count: int
    nfft: int
    start_time_au: float
    duration_au: float
    time_step_au: float
    intrinsic_resolution_au: float
    native_grid_spacing_au: float
    padded_grid_spacing_au: float
    nyquist_energy_au: float

    def __post_init__(self) -> None:
        omega = _array(self.omega_au, np.float64, "transform frequencies", ndim=1)
        values = _array(self.values, np.complex128, "transform values")
        if values.ndim < 1 or values.shape[0] != omega.size:
            raise SpectroscopyError("transform values must have frequency as their first axis")
        if omega.size == 0 or np.any(omega <= 0.0) or np.any(np.diff(omega) <= 0.0):
            raise SpectroscopyError("transform frequencies must be nonempty and strictly positive")
        if self.sample_count < 2 or self.nfft < self.sample_count:
            raise SpectroscopyError("transform sample-count/NFFT metadata is inconsistent")
        if self.nfft != self.config.zero_padding_factor * self.sample_count:
            raise SpectroscopyError("transform NFFT disagrees with zero-padding factor")
        for name in (
            "duration_au",
            "time_step_au",
            "intrinsic_resolution_au",
            "native_grid_spacing_au",
            "padded_grid_spacing_au",
            "nyquist_energy_au",
        ):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name, positive=True))
        object.__setattr__(
            self,
            "start_time_au",
            _finite_scalar(self.start_time_au, "transform start time"),
        )
        object.__setattr__(self, "omega_au", omega)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class SpectrumSourceLink:
    """Checksum-qualified immutable parent for a derived spectrum."""

    kind: str
    path: Path
    sha256: str
    identifier: str
    description: str = ""
    metadata_json: str = "{}"

    def __post_init__(self) -> None:
        if not self.kind.strip() or not self.identifier.strip():
            raise SpectroscopyError("spectrum source kind and identifier must be nonempty")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise SpectroscopyError("spectrum source checksum must be lowercase SHA-256")
        try:
            metadata = json.loads(self.metadata_json)
        except json.JSONDecodeError as exc:
            raise SpectroscopyError("spectrum source metadata must be valid JSON") from exc
        if not isinstance(metadata, dict):
            raise SpectroscopyError("spectrum source metadata must be a JSON object")
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(
            self,
            "metadata_json",
            json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "path": str(self.path),
            "sha256": self.sha256,
            "identifier": self.identifier,
            "description": self.description,
            "metadata_json": self.metadata_json,
        }

    def identity_mapping(self) -> dict[str, object]:
        """Return path-independent parent identity for scientific hashing."""

        return {
            "kind": self.kind,
            "sha256": self.sha256,
            "identifier": self.identifier,
            "description": self.description,
            "metadata_json": self.metadata_json,
        }


@dataclass(frozen=True, slots=True)
class KickSpectrum:
    """One polarization column of a molecular kick-response tensor."""

    config: KickSpectrumConfig
    source: SpectrumSourceLink
    event_id: str
    dipole_definition_id: str
    current_definition_id: str | None
    impulse_au: np.ndarray
    input_polarization: np.ndarray
    impulse_amplitude_au: float
    baseline_dipole_au: np.ndarray
    sample_count: int
    nfft: int
    start_time_au: float
    duration_au: float
    time_step_au: float
    intrinsic_resolution_au: float
    native_grid_spacing_au: float
    padded_grid_spacing_au: float
    nyquist_energy_au: float
    omega_au: np.ndarray
    polarizability_dipole_au: np.ndarray
    polarizability_current_au: np.ndarray | None
    polarizability_parallel_au: np.ndarray
    absorption_strength_parallel_au: np.ndarray
    oscillator_strength_density_parallel_au: np.ndarray
    current_domain_residual_au: np.ndarray | None
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        impulse = _array(self.impulse_au, np.float64, "kick impulse", ndim=1)
        polarization = _array(
            self.input_polarization, np.float64, "kick input polarization", ndim=1
        )
        baseline = _array(self.baseline_dipole_au, np.float64, "dipole baseline", ndim=1)
        if impulse.shape != (3,) or polarization.shape != (3,) or baseline.shape != (3,):
            raise SpectroscopyError("kick vectors and dipole baseline must have shape (3,)")
        amplitude = _finite_scalar(self.impulse_amplitude_au, "kick amplitude", positive=True)
        if not np.isclose(np.linalg.norm(impulse), amplitude, rtol=2.0e-14, atol=0.0):
            raise SpectroscopyError("kick amplitude disagrees with its impulse vector")
        if not np.allclose(polarization, impulse / amplitude, rtol=2.0e-14, atol=2.0e-15):
            raise SpectroscopyError("kick polarization disagrees with its impulse direction")
        omega = _array(self.omega_au, np.float64, "spectrum frequencies", ndim=1)
        alpha = _array(
            self.polarizability_dipole_au,
            np.complex128,
            "dipole-domain polarizability",
            ndim=2,
        )
        parallel = _array(
            self.polarizability_parallel_au,
            np.complex128,
            "parallel polarizability",
            ndim=1,
        )
        absorption = _array(
            self.absorption_strength_parallel_au,
            np.float64,
            "parallel absorption strength",
            ndim=1,
        )
        oscillator = _array(
            self.oscillator_strength_density_parallel_au,
            np.float64,
            "parallel oscillator-strength density",
            ndim=1,
        )
        if omega.size == 0 or np.any(omega <= 0.0) or np.any(np.diff(omega) <= 0.0):
            raise SpectroscopyError("spectrum frequencies must be nonempty and increasing")
        if alpha.shape != (omega.size, 3):
            raise SpectroscopyError("polarizability must have shape (nfrequency, 3)")
        if any(value.shape != omega.shape for value in (parallel, absorption, oscillator)):
            raise SpectroscopyError("parallel spectrum arrays must match the frequency grid")
        expected_parallel = alpha @ polarization
        expected_absorption = omega * expected_parallel.imag
        expected_oscillator = (2.0 / np.pi) * expected_absorption
        if not np.array_equal(parallel, expected_parallel):
            raise SpectroscopyError("parallel polarizability disagrees with the response tensor")
        if not np.array_equal(absorption, expected_absorption):
            raise SpectroscopyError("absorption strength disagrees with the polarizability")
        if not np.array_equal(oscillator, expected_oscillator):
            raise SpectroscopyError("oscillator-strength density disagrees with absorption")
        current = self.polarizability_current_au
        residual = self.current_domain_residual_au
        if (current is None) != (residual is None) or (current is None) != (
            self.current_definition_id is None
        ):
            raise SpectroscopyError("current-domain spectrum fields must be present together")
        if current is not None and residual is not None:
            current = _array(current, np.complex128, "current-domain polarizability", ndim=2)
            residual = _array(residual, np.complex128, "current-domain residual", ndim=2)
            if current.shape != alpha.shape or residual.shape != alpha.shape:
                raise SpectroscopyError("current-domain arrays must match dipole polarizability")
            if not np.array_equal(residual, current - alpha):
                raise SpectroscopyError("current-domain residual disagrees with polarizabilities")
        if not self.event_id.strip() or not self.dipole_definition_id.strip():
            raise SpectroscopyError("event and dipole definition identifiers must be nonempty")
        if self.sample_count < 2 or self.nfft != self.config.transform.zero_padding_factor * (
            self.sample_count
        ):
            raise SpectroscopyError("spectrum sample-count/NFFT metadata is inconsistent")
        for name in (
            "duration_au",
            "time_step_au",
            "intrinsic_resolution_au",
            "native_grid_spacing_au",
            "padded_grid_spacing_au",
            "nyquist_energy_au",
        ):
            object.__setattr__(self, name, _finite_scalar(getattr(self, name), name, positive=True))
        expected_duration = (self.sample_count - 1) * self.time_step_au
        expected_resolution = 2.0 * np.pi / expected_duration
        expected_native = 2.0 * np.pi / (self.sample_count * self.time_step_au)
        expected_padded = 2.0 * np.pi / (self.nfft * self.time_step_au)
        expected_nyquist = np.pi / self.time_step_au
        for actual, expected, label in (
            (self.duration_au, expected_duration, "duration"),
            (self.intrinsic_resolution_au, expected_resolution, "intrinsic resolution"),
            (self.native_grid_spacing_au, expected_native, "native grid spacing"),
            (self.padded_grid_spacing_au, expected_padded, "padded grid spacing"),
            (self.nyquist_energy_au, expected_nyquist, "Nyquist energy"),
        ):
            if not np.isclose(actual, expected, rtol=2.0e-14, atol=0.0):
                raise SpectroscopyError(f"spectrum {label} metadata is inconsistent")
        object.__setattr__(self, "start_time_au", _finite_scalar(self.start_time_au, "start time"))
        for name, value in (
            ("impulse_au", impulse),
            ("input_polarization", polarization),
            ("baseline_dipole_au", baseline),
            ("omega_au", omega),
            ("polarizability_dipole_au", alpha),
            ("polarizability_parallel_au", parallel),
            ("absorption_strength_parallel_au", absorption),
            ("oscillator_strength_density_parallel_au", oscillator),
            ("polarizability_current_au", current),
            ("current_domain_residual_au", residual),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "impulse_amplitude_au", amplitude)
        object.__setattr__(
            self,
            "result_id",
            canonical_sha256(
                {
                    "schema": "aion.kick-spectrum-content",
                    "version": "1.0.0",
                    "config": self.config.as_mapping(),
                    "source": self.source.identity_mapping(),
                    "event_id": self.event_id,
                    "dipole_definition_id": self.dipole_definition_id,
                    "current_definition_id": self.current_definition_id,
                    "impulse_au": impulse,
                    "baseline_dipole_au": baseline,
                    "sample_count": self.sample_count,
                    "nfft": self.nfft,
                    "start_time_au": self.start_time_au,
                    "duration_au": self.duration_au,
                    "time_step_au": self.time_step_au,
                    "intrinsic_resolution_au": self.intrinsic_resolution_au,
                    "native_grid_spacing_au": self.native_grid_spacing_au,
                    "padded_grid_spacing_au": self.padded_grid_spacing_au,
                    "nyquist_energy_au": self.nyquist_energy_au,
                    "omega_au": omega,
                    "polarizability_dipole_au": alpha,
                    "polarizability_current_au": current,
                    "polarizability_parallel_au": parallel,
                    "absorption_strength_parallel_au": absorption,
                    "oscillator_strength_density_parallel_au": oscillator,
                    "current_domain_residual_au": residual,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class CasidaConfig:
    """Numerical choices for the PySCF Casida-TDDFT wrapper."""

    nstates: int
    convergence_tolerance: float = 1.0e-9
    singlet: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.nstates, bool) or not isinstance(self.nstates, int) or self.nstates < 1:
            raise SpectroscopyError("Casida nstates must be a positive integer")
        object.__setattr__(
            self,
            "convergence_tolerance",
            _finite_scalar(
                self.convergence_tolerance,
                "Casida convergence tolerance",
                positive=True,
            ),
        )
        if self.singlet is not True:
            raise SpectroscopyError("only singlet Casida response is validated in Aion 0.2")

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema": "aion.casida-config",
            "version": "1.0.0",
            "nstates": self.nstates,
            "convergence_tolerance": self.convergence_tolerance,
            "singlet": self.singlet,
            "oscillator_strength_gauge": "length",
        }


@dataclass(frozen=True, slots=True)
class CasidaResult:
    """Structured length-dipole Casida roots tied to one prepared reference."""

    config: CasidaConfig
    reference_fingerprint_sha256: str
    method: str
    pyscf_version: str
    excitation_energy_au: np.ndarray
    oscillator_strength: np.ndarray
    transition_dipole_au: np.ndarray
    transition_direction: np.ndarray
    converged: np.ndarray
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if len(self.reference_fingerprint_sha256) != 64:
            raise SpectroscopyError("Casida reference fingerprint is invalid")
        if not self.method.strip() or not self.pyscf_version.strip():
            raise SpectroscopyError("Casida method and PySCF version must be nonempty")
        energy = _array(self.excitation_energy_au, np.float64, "excitation energies", ndim=1)
        strength = _array(self.oscillator_strength, np.float64, "oscillator strengths", ndim=1)
        dipole = _array(self.transition_dipole_au, np.float64, "transition dipoles", ndim=2)
        direction = _array(self.transition_direction, np.float64, "transition directions", ndim=2)
        converged = _array(self.converged, np.bool_, "Casida convergence flags", ndim=1)
        nroot = energy.size
        if nroot == 0 or nroot > self.config.nstates:
            raise SpectroscopyError("Casida returned an invalid number of roots")
        if strength.shape != (nroot,) or dipole.shape != (nroot, 3):
            raise SpectroscopyError("Casida root arrays have inconsistent shapes")
        if direction.shape != dipole.shape or converged.shape != (nroot,):
            raise SpectroscopyError("Casida direction/convergence arrays have inconsistent shapes")
        if np.any(energy <= 0.0) or np.any(np.diff(energy) < 0.0):
            raise SpectroscopyError("Casida excitation energies must be positive and ordered")
        if np.any(strength < -1.0e-13):
            raise SpectroscopyError("Casida oscillator strengths cannot be negative")
        norms = np.linalg.norm(dipole, axis=1)
        length_strength = (2.0 / 3.0) * energy * np.square(norms)
        if not np.allclose(strength, length_strength, rtol=2.0e-11, atol=2.0e-13):
            raise SpectroscopyError(
                "Casida oscillator strengths disagree with length-gauge transition dipoles"
            )
        expected = np.zeros_like(dipole)
        active = norms > 0.0
        expected[active] = dipole[active] / norms[active, None]
        if not np.allclose(direction, expected, rtol=2.0e-13, atol=2.0e-14):
            raise SpectroscopyError("transition directions disagree with transition dipoles")
        for name, value in (
            ("excitation_energy_au", energy),
            ("oscillator_strength", strength),
            ("transition_dipole_au", dipole),
            ("transition_direction", direction),
            ("converged", converged),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "result_id",
            canonical_sha256(
                {
                    "schema": "aion.casida-content",
                    "version": "1.0.0",
                    "config": self.config.as_mapping(),
                    "reference_fingerprint_sha256": self.reference_fingerprint_sha256,
                    "method": self.method,
                    "pyscf_version": self.pyscf_version,
                    "excitation_energy_au": energy,
                    "oscillator_strength": strength,
                    "transition_dipole_au": dipole,
                    "converged": converged,
                }
            ),
        )

    @property
    def root_count(self) -> int:
        return int(self.excitation_energy_au.size)


@dataclass(frozen=True, slots=True)
class ResonanceSelection:
    """Explicit polarization-resolved root selection and all candidates."""

    casida_result_id: str
    polarization: np.ndarray
    projected_oscillator_strength: np.ndarray
    selected_root: int
    brightness_threshold: float | None
    user_selected: bool

    def __post_init__(self) -> None:
        polarization = _array(self.polarization, np.float64, "selection polarization", ndim=1)
        strengths = _array(
            self.projected_oscillator_strength,
            np.float64,
            "projected oscillator strengths",
            ndim=1,
        )
        if polarization.shape != (3,) or not np.isclose(np.linalg.norm(polarization), 1.0):
            raise SpectroscopyError("selection polarization must be a unit Cartesian vector")
        if (
            isinstance(self.selected_root, bool)
            or not isinstance(self.selected_root, int)
            or not 0 <= self.selected_root < strengths.size
        ):
            raise SpectroscopyError("selected Casida root is outside the candidate array")
        if self.brightness_threshold is not None:
            threshold = _finite_scalar(self.brightness_threshold, "brightness threshold")
            if threshold < 0.0:
                raise SpectroscopyError("brightness threshold cannot be negative")
            object.__setattr__(self, "brightness_threshold", threshold)
        object.__setattr__(self, "polarization", polarization)
        object.__setattr__(self, "projected_oscillator_strength", strengths)

    def as_mapping(self) -> dict[str, object]:
        return {
            "casida_result_id": self.casida_result_id,
            "polarization": self.polarization.tolist(),
            "projected_oscillator_strength": self.projected_oscillator_strength.tolist(),
            "selected_root": self.selected_root,
            "brightness_threshold": self.brightness_threshold,
            "user_selected": self.user_selected,
        }


def configuration_json(config: TransformConfig | KickSpectrumConfig | CasidaConfig) -> str:
    """Return the canonical human-readable configuration stored in artifacts."""

    return json.dumps(config.as_mapping(), sort_keys=True, separators=(",", ":"))
