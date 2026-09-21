"""Variational pure-LDA action on an exact straight-Wilson density.

The XC energy, lower matrix, and optional fixed-coefficient-history source
direction returned here are descendants of one discrete molecular-grid sum.
PySCF/libxc supplies only the pointwise pure-LDA energy per particle and its
density derivative.  Aion owns the complex Wilson density, its source
response, and every grid contraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aion.electromagnetism import build_magnetic_pair_geometry
from aion.electronic_structure.ao_quadrature import AOQuadrature
from aion.electronic_structure.data import PreparedReference
from aion.electronic_structure.wilson_density import StraightLineVectorPotentialDirection
from aion.errors import (
    ConfigurationError,
    FormulationError,
    UnsupportedConfigurationError,
)


@dataclass(frozen=True, slots=True)
class WilsonLDAProvenance:
    """Numerical identity of one pointwise LDA realization."""

    functional: str
    functional_family: str
    spin: str
    pointwise_engine: str
    libxc_version: str
    pointwise_engine_backend: str
    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class WilsonLDAResult:
    """One energy/matrix/source package from a common Wilson-LDA action.

    Every numerical array is resident on the selected Aion backend.  The
    optional source fields are present only when a fixed-history real vector-
    potential direction was requested.
    """

    energy: Any
    lower_xc_matrix: Any
    density: Any
    energy_per_particle: Any
    density_derivative: Any
    source_energy_direction: Any | None
    source_density_direction: Any | None
    electron_count_grid: Any
    lower_hermiticity_residual: float
    density_imaginary_max_abs: float
    density_real_minimum: float
    source_density_imaginary_max_abs: float | None
    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str
    backend: str
    device_index: int | None
    functional: str
    coefficient_frame_applied: bool
    charge: float
    hbar: float


@dataclass(slots=True)
class WilsonLDAEvaluator:
    r"""Prepared restricted pure-LDA evaluator for one AO quadrature.

    The discrete action is

    ``E_xc,Q = sum_g w_g n_g epsilon_xc(n_g)``.

    Its lower matrix and fixed-history source derivative are evaluated as

    ``V_ij = sum_g w_g v_xc(n_g) chi_i(g)^* chi_j(g)`` and
    ``delta E = sum_g w_g v_xc(n_g) delta n_g``.

    The pointwise call to PySCF/libxc is deliberately host-side.  On a GPU,
    only the real density block crosses to the host and the returned
    pointwise fields cross back; all Wilson and quadrature contractions and
    all returned arrays remain device resident.
    """

    quadrature: AOQuadrature
    functional: str
    imaginary_relative_tolerance: float = 2.0e-10
    negative_density_relative_tolerance: float = 2.0e-12
    provenance: WilsonLDAProvenance = field(init=False)
    _numint: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.quadrature, AOQuadrature):
            raise TypeError("quadrature must be an AOQuadrature")
        if not isinstance(self.quadrature.reference, PreparedReference):
            raise UnsupportedConfigurationError(
                "Wilson LDA requires an authenticated prepared DFT reference"
            )
        if not isinstance(self.functional, str) or not self.functional.strip():
            raise ConfigurationError("functional must be a nonempty explicit name")
        imaginary_tolerance = _nonnegative_scalar(
            self.imaginary_relative_tolerance,
            "imaginary_relative_tolerance",
        )
        negative_tolerance = _nonnegative_scalar(
            self.negative_density_relative_tolerance,
            "negative_density_relative_tolerance",
        )
        object.__setattr__(self, "imaginary_relative_tolerance", imaginary_tolerance)
        object.__setattr__(
            self,
            "negative_density_relative_tolerance",
            negative_tolerance,
        )

        from pyscf.dft import libxc, numint

        try:
            family = str(libxc.xc_type(self.functional)).upper()
            hybrid = bool(libxc.is_hybrid_xc(self.functional))
            nonlocal_correlation = bool(libxc.is_nlc(self.functional))
        except Exception as exc:
            raise UnsupportedConfigurationError(
                f"PySCF/libxc cannot resolve functional {self.functional!r}"
            ) from exc
        if family != "LDA":
            raise UnsupportedConfigurationError(
                f"functional {self.functional!r} is {family}; WilsonLDAEvaluator "
                "accepts pure LDA only"
            )
        if hybrid or nonlocal_correlation:
            raise UnsupportedConfigurationError(
                "hybrid, exact-exchange, and nonlocal-correlation functionals are out of scope"
            )
        object.__setattr__(self, "_numint", numint.NumInt())
        object.__setattr__(
            self,
            "provenance",
            WilsonLDAProvenance(
                functional=self.functional,
                functional_family=family,
                spin="restricted_unpolarized",
                pointwise_engine="pyscf.dft.numint.NumInt.eval_xc_eff",
                libxc_version=str(libxc.libxc_version()),
                pointwise_engine_backend="cpu",
                reference_fingerprint_sha256=(
                    self.quadrature.reference.fingerprint_sha256
                ),
                grid_fingerprint_sha256=self.quadrature.grid.fingerprint_sha256,
            ),
        )

    @property
    def backend(self) -> Any:
        return self.quadrature.backend

    @property
    def namespace(self) -> Any:
        return self.backend.namespace

    def evaluate(
        self,
        coefficient_density: object,
        vector_potential: StraightLineVectorPotentialDirection,
        *,
        source_direction: StraightLineVectorPotentialDirection | None = None,
        coefficient_frame: object | None = None,
        charge: float = -1.0,
        hbar: float = 1.0,
    ) -> WilsonLDAResult:
        """Evaluate the action and an optional fixed-history source direction.

        ``coefficient_frame`` applies a fixed square change ``chi' = chi A``.
        The caller must supply the corresponding contravariant density.  This
        explicit boundary allows general complex, nonunitary frame-covariance
        tests without introducing a second action implementation.
        """

        if not callable(getattr(vector_potential, "straight_line_integrals", None)):
            raise TypeError("vector_potential must provide straight_line_integrals")
        if source_direction is not None and not callable(
            getattr(source_direction, "straight_line_integrals", None)
        ):
            raise TypeError("source_direction must provide straight_line_integrals")
        checked_charge = _finite_scalar(charge, "charge")
        checked_hbar = _positive_scalar(hbar, "hbar")
        density_matrix = self._validated_density(coefficient_density)
        frame_change = self._validated_frame(coefficient_frame)

        backend = self.backend
        xp = self.namespace
        reference = self.quadrature.reference
        nao = reference.core_operators.nao
        npoint = self.quadrature.grid.npoints
        density_values = backend.zeros((npoint,), dtype=xp.float64)
        energy_per_particle = backend.zeros((npoint,), dtype=xp.float64)
        density_derivative = backend.zeros((npoint,), dtype=xp.float64)
        source_density = (
            None
            if source_direction is None
            else backend.zeros((npoint,), dtype=xp.float64)
        )
        lower = backend.zeros((nao, nao), dtype=xp.complex128)
        energy = backend.asarray(0.0, dtype=xp.float64)
        particle_number = backend.asarray(0.0, dtype=xp.float64)
        source_energy = (
            None
            if source_direction is None
            else backend.asarray(0.0, dtype=xp.float64)
        )

        geometry = build_magnetic_pair_geometry(
            reference.core_operators.nuclei.coordinates_au,
            reference.anchor_topology.ao_to_atom,
            backend,
        )
        anchors = geometry.ao_anchor_coordinates_au
        prefactor = 1j * checked_charge / checked_hbar
        density_imaginary_maximum = 0.0
        density_minimum = math.inf
        source_imaginary_maximum = 0.0 if source_direction is not None else None

        for block in self.quadrature.blocks():
            line = vector_potential.straight_line_integrals(
                anchors[None, :, :],
                block.coordinates_au[:, None, :],
                backend,
            )
            frame = xp.exp(prefactor * line) * block.values
            if frame_change is not None:
                frame = frame @ frame_change
            complex_density = xp.einsum(
                "mn,pm,pn->p",
                density_matrix,
                frame,
                frame.conj(),
                optimize=True,
            )
            real_density = xp.real(complex_density)
            block_imaginary = backend.scalar_to_float(
                xp.max(xp.abs(xp.imag(complex_density)))
            )
            block_real_scale = backend.scalar_to_float(
                xp.maximum(xp.asarray(1.0), xp.max(xp.abs(real_density)))
            )
            if block_imaginary > self.imaginary_relative_tolerance * block_real_scale:
                raise FormulationError(
                    "Wilson density has an unresolved imaginary component "
                    f"{block_imaginary:.3e}"
                )
            block_minimum = backend.scalar_to_float(xp.min(real_density))
            if block_minimum < -self.negative_density_relative_tolerance * block_real_scale:
                raise FormulationError(
                    "Wilson density is negative beyond its declared numerical tolerance: "
                    f"{block_minimum:.3e}"
                )
            density_imaginary_maximum = max(density_imaginary_maximum, block_imaginary)
            density_minimum = min(density_minimum, block_minimum)

            exc_host, potential_host = self._evaluate_pointwise(
                backend.to_host(real_density)
            )
            exc = backend.asarray(exc_host, dtype=xp.float64)
            potential = backend.asarray(potential_host, dtype=xp.float64)
            backend.assert_resident(exc, name="LDA energy per particle")
            backend.assert_resident(potential, name="LDA density derivative")

            density_values[block.start : block.stop] = real_density
            energy_per_particle[block.start : block.stop] = exc
            density_derivative[block.start : block.stop] = potential
            energy += xp.einsum(
                "p,p,p->",
                block.weights_au,
                real_density,
                exc,
                optimize=True,
            )
            particle_number += xp.einsum(
                "p,p->",
                block.weights_au,
                real_density,
                optimize=True,
            )
            lower += xp.einsum(
                "p,p,pi,pj->ij",
                block.weights_au,
                potential,
                frame.conj(),
                frame,
                optimize=True,
            )

            if source_direction is not None:
                direction_line = source_direction.straight_line_integrals(
                    anchors[None, :, :],
                    block.coordinates_au[:, None, :],
                    backend,
                )
                frame_direction = prefactor * direction_line * (
                    xp.exp(prefactor * line) * block.values
                )
                if frame_change is not None:
                    frame_direction = frame_direction @ frame_change
                complex_direction = xp.einsum(
                    "mn,pm,pn->p",
                    density_matrix,
                    frame_direction,
                    frame.conj(),
                    optimize=True,
                ) + xp.einsum(
                    "mn,pm,pn->p",
                    density_matrix,
                    frame,
                    frame_direction.conj(),
                    optimize=True,
                )
                real_direction = xp.real(complex_direction)
                block_source_imaginary = backend.scalar_to_float(
                    xp.max(xp.abs(xp.imag(complex_direction)))
                )
                direction_scale = backend.scalar_to_float(
                    xp.maximum(xp.asarray(1.0), xp.max(xp.abs(real_direction)))
                )
                if (
                    block_source_imaginary
                    > self.imaginary_relative_tolerance * direction_scale
                ):
                    raise FormulationError(
                        "Wilson source-density direction has an unresolved imaginary "
                        f"component {block_source_imaginary:.3e}"
                    )
                assert source_imaginary_maximum is not None
                source_imaginary_maximum = max(
                    source_imaginary_maximum,
                    block_source_imaginary,
                )
                assert source_density is not None
                source_density[block.start : block.stop] = real_direction
                assert source_energy is not None
                source_energy += xp.einsum(
                    "p,p,p->",
                    block.weights_au,
                    potential,
                    real_direction,
                    optimize=True,
                )

        lower_scale = xp.maximum(xp.asarray(1.0), xp.linalg.norm(lower))
        lower_hermiticity = backend.scalar_to_float(
            xp.linalg.norm(lower - lower.conj().T) / lower_scale
        )
        backend.synchronize()
        return WilsonLDAResult(
            energy=energy,
            lower_xc_matrix=lower,
            density=density_values,
            energy_per_particle=energy_per_particle,
            density_derivative=density_derivative,
            source_energy_direction=source_energy,
            source_density_direction=source_density,
            electron_count_grid=particle_number,
            lower_hermiticity_residual=lower_hermiticity,
            density_imaginary_max_abs=density_imaginary_maximum,
            density_real_minimum=density_minimum,
            source_density_imaginary_max_abs=source_imaginary_maximum,
            reference_fingerprint_sha256=reference.fingerprint_sha256,
            grid_fingerprint_sha256=self.quadrature.grid.fingerprint_sha256,
            backend=self.quadrature.backend_config.kind.value,
            device_index=self.quadrature.backend_config.device_index,
            functional=self.functional,
            coefficient_frame_applied=frame_change is not None,
            charge=checked_charge,
            hbar=checked_hbar,
        )

    def _evaluate_pointwise(self, density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        try:
            values = self._numint.eval_xc_eff(
                self.functional,
                density,
                deriv=1,
                xctype="LDA",
                spin=0,
            )
        except Exception as error:
            raise FormulationError(
                f"PySCF/libxc failed for pure-LDA functional {self.functional!r}"
            ) from error
        energy_per_particle = np.asarray(values[0], dtype=np.float64)
        derivative = np.asarray(values[1], dtype=np.float64)
        potential = derivative[0] if derivative.ndim == 2 else derivative
        if (
            energy_per_particle.shape != density.shape
            or potential.shape != density.shape
            or not np.all(np.isfinite(energy_per_particle))
            or not np.all(np.isfinite(potential))
        ):
            raise FormulationError("PySCF/libxc returned invalid pure-LDA pointwise data")
        return energy_per_particle, potential

    def _validated_density(self, value: object) -> Any:
        xp = self.namespace
        dimension = self.quadrature.reference.core_operators.nao
        density = self.backend.asarray(value, dtype=xp.complex128)
        self.backend.assert_resident(density, name="contravariant coefficient density")
        if density.shape != (dimension, dimension):
            raise ConfigurationError(
                f"coefficient_density has shape {density.shape}; expected "
                f"{(dimension, dimension)}"
            )
        if not _control_bool(xp.all(xp.isfinite(density)), self.backend):
            raise ConfigurationError("coefficient_density contains non-finite values")
        scale = xp.maximum(xp.asarray(1.0), xp.linalg.norm(density))
        residual = self.backend.scalar_to_float(
            xp.linalg.norm(density - density.conj().T) / scale
        )
        if residual > 1.0e-11:
            raise ConfigurationError("coefficient_density must be Hermitian")
        return density

    def _validated_frame(self, value: object | None) -> Any | None:
        if value is None:
            return None
        xp = self.namespace
        dimension = self.quadrature.reference.core_operators.nao
        frame = self.backend.asarray(value, dtype=xp.complex128)
        self.backend.assert_resident(frame, name="coefficient frame change")
        if frame.shape != (dimension, dimension):
            raise ConfigurationError(
                f"coefficient_frame has shape {frame.shape}; expected "
                f"{(dimension, dimension)}"
            )
        if not _control_bool(xp.all(xp.isfinite(frame)), self.backend):
            raise ConfigurationError("coefficient_frame contains non-finite values")
        singular_values = xp.linalg.svd(frame, compute_uv=False)
        smallest = self.backend.scalar_to_float(xp.min(singular_values))
        largest = self.backend.scalar_to_float(xp.max(singular_values))
        if smallest <= 1.0e-12 * max(1.0, largest):
            raise ConfigurationError("coefficient_frame must be nonsingular")
        return frame


def prepare_wilson_lda(
    quadrature: AOQuadrature,
    functional: str,
    *,
    imaginary_relative_tolerance: float = 2.0e-10,
    negative_density_relative_tolerance: float = 2.0e-12,
) -> WilsonLDAEvaluator:
    """Prepare one explicitly labelled restricted pure-LDA grid action."""

    return WilsonLDAEvaluator(
        quadrature=quadrature,
        functional=functional,
        imaginary_relative_tolerance=imaginary_relative_tolerance,
        negative_density_relative_tolerance=negative_density_relative_tolerance,
    )


def _control_bool(value: object, backend: Any) -> bool:
    return bool(backend.scalar_to_float(value))


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{name} must be a finite number")
    return result


def _positive_scalar(value: float, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ConfigurationError(f"{name} must be positive")
    return result


def _nonnegative_scalar(value: float, name: str) -> float:
    result = _finite_scalar(value, name)
    if result < 0.0:
        raise ConfigurationError(f"{name} must be nonnegative")
    return result
