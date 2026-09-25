"""Variational pure-GGA action on an exact straight-Wilson density.

The energy, weak lower matrix, and optional fixed-history source derivative
returned here are exact derivatives of one discrete molecular-grid action.
PySCF/libxc supplies only pointwise pure-GGA functional data.  Aion owns the
complex Wilson density and density gradient, their source directions, and
all grid contractions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from aion.electromagnetism import build_magnetic_pair_geometry
from aion.electronic_structure.ao_quadrature import AOQuadrature
from aion.electronic_structure.data import PreparedReference
from aion.errors import (
    ConfigurationError,
    FormulationError,
    UnsupportedConfigurationError,
)


class DifferentiableStraightLineVectorPotential(Protocol):
    """Vector-potential data needed by a Wilson-GGA action."""

    def straight_line_integrals(
        self,
        starts_au: object,
        ends_au: object,
        backend: Any,
    ) -> Any: ...

    def straight_line_integral_gradients(
        self,
        starts_au: object,
        ends_au: object,
        backend: Any,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class WilsonGGAProvenance:
    """Numerical identity of one quadrature--Wilson pure-GGA realization."""

    functional: str
    functional_family: str
    realization: str
    spin: str
    pointwise_engine: str
    libxc_version: str
    pointwise_engine_backend: str
    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class WilsonGGAResult:
    """Energy, weak matrix, and source data from one Wilson-GGA action."""

    energy: Any
    lower_xc_matrix: Any
    density: Any
    density_gradient: Any
    energy_per_particle: Any
    density_derivative: Any
    density_gradient_derivative: Any
    source_energy_direction: Any | None
    source_density_direction: Any | None
    source_density_gradient_direction: Any | None
    electron_count_grid: Any
    lower_hermiticity_residual: float
    density_imaginary_max_abs: float
    density_gradient_imaginary_max_abs: float
    density_real_minimum: float
    source_density_imaginary_max_abs: float | None
    source_density_gradient_imaginary_max_abs: float | None
    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str
    backend: str
    device_index: int | None
    functional: str
    realization: str
    coefficient_frame_applied: bool
    charge: float
    hbar: float


@dataclass(slots=True)
class WilsonGGAEvaluator:
    r"""Prepared restricted pure-GGA evaluator for one AO quadrature.

    With ``g=grad(n)``, the discrete action and its two derivatives are

    ``E_Q = sum_p w_p f(n_p, g_p)``,

    ``V_ij = sum_p w_p [f_n N_ij + f_g . grad(N_ij)]``, and

    ``delta E_Q = sum_p w_p [f_n delta n + f_g . delta g]``.

    All three use exactly the same nodes, weights, Wilson-dressed AO values,
    and Wilson-dressed AO gradients.  No numerical divergence is formed.
    """

    quadrature: AOQuadrature
    functional: str
    imaginary_relative_tolerance: float = 2.0e-10
    negative_density_relative_tolerance: float = 2.0e-12
    provenance: WilsonGGAProvenance = field(init=False)
    _numint: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.quadrature, AOQuadrature):
            raise TypeError("quadrature must be an AOQuadrature")
        if not isinstance(self.quadrature.reference, PreparedReference):
            raise UnsupportedConfigurationError(
                "Wilson GGA requires an authenticated prepared DFT reference"
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
        if family != "GGA":
            raise UnsupportedConfigurationError(
                f"functional {self.functional!r} is {family}; WilsonGGAEvaluator "
                "accepts pure GGA only"
            )
        if hybrid or nonlocal_correlation:
            raise UnsupportedConfigurationError(
                "hybrid, exact-exchange, and nonlocal-correlation functionals are out of scope"
            )
        object.__setattr__(self, "_numint", numint.NumInt())
        object.__setattr__(
            self,
            "provenance",
            WilsonGGAProvenance(
                functional=self.functional,
                functional_family=family,
                realization="quadrature--Wilson GGA",
                spin="restricted_unpolarized",
                pointwise_engine="pyscf.dft.numint.NumInt.eval_xc_eff",
                libxc_version=str(libxc.libxc_version()),
                pointwise_engine_backend="cpu",
                reference_fingerprint_sha256=(self.quadrature.reference.fingerprint_sha256),
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
        vector_potential: DifferentiableStraightLineVectorPotential,
        *,
        source_direction: DifferentiableStraightLineVectorPotential | None = None,
        coefficient_frame: object | None = None,
        charge: float = -1.0,
        hbar: float = 1.0,
    ) -> WilsonGGAResult:
        """Evaluate the action and an optional fixed-history source direction."""

        _validate_vector_potential(vector_potential, "vector_potential")
        if source_direction is not None:
            _validate_vector_potential(source_direction, "source_direction")
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
        density_gradients = backend.zeros((3, npoint), dtype=xp.float64)
        energy_per_particle = backend.zeros((npoint,), dtype=xp.float64)
        density_derivative = backend.zeros((npoint,), dtype=xp.float64)
        gradient_derivative = backend.zeros((3, npoint), dtype=xp.float64)
        source_density = (
            None if source_direction is None else backend.zeros((npoint,), dtype=xp.float64)
        )
        source_gradient = (
            None if source_direction is None else backend.zeros((3, npoint), dtype=xp.float64)
        )
        lower = backend.zeros((nao, nao), dtype=xp.complex128)
        energy = backend.asarray(0.0, dtype=xp.float64)
        particle_number = backend.asarray(0.0, dtype=xp.float64)
        source_energy = None if source_direction is None else backend.asarray(0.0, dtype=xp.float64)

        geometry = build_magnetic_pair_geometry(
            reference.core_operators.nuclei.coordinates_au,
            reference.anchor_topology.ao_to_atom,
            backend,
        )
        anchors = geometry.ao_anchor_coordinates_au
        prefactor = 1j * checked_charge / checked_hbar
        density_imaginary_maximum = 0.0
        gradient_imaginary_maximum = 0.0
        density_minimum = math.inf
        source_imaginary_maximum = 0.0 if source_direction is not None else None
        source_gradient_imaginary_maximum = 0.0 if source_direction is not None else None

        for block in self.quadrature.blocks():
            line = vector_potential.straight_line_integrals(
                anchors[None, :, :],
                block.coordinates_au[:, None, :],
                backend,
            )
            line_gradient = vector_potential.straight_line_integral_gradients(
                anchors[None, :, :],
                block.coordinates_au[:, None, :],
                backend,
            )
            phase = xp.exp(prefactor * line)
            frame = phase * block.values
            frame_gradient = phase[None, :, :] * (
                block.gradients
                + prefactor * xp.moveaxis(line_gradient, -1, 0) * block.values[None, :, :]
            )
            if frame_change is not None:
                frame = frame @ frame_change
                frame_gradient = xp.einsum(
                    "xpi,ij->xpj", frame_gradient, frame_change, optimize=True
                )

            complex_density = xp.einsum(
                "mn,pm,pn->p", density_matrix, frame, frame.conj(), optimize=True
            )
            complex_gradient = xp.einsum(
                "mn,xpm,pn->xp",
                density_matrix,
                frame_gradient,
                frame.conj(),
                optimize=True,
            ) + xp.einsum(
                "mn,pm,xpn->xp",
                density_matrix,
                frame,
                frame_gradient.conj(),
                optimize=True,
            )
            real_density = xp.real(complex_density)
            real_gradient = xp.real(complex_gradient)
            block_imaginary = backend.scalar_to_float(xp.max(xp.abs(xp.imag(complex_density))))
            block_gradient_imaginary = backend.scalar_to_float(
                xp.max(xp.abs(xp.imag(complex_gradient)))
            )
            block_real_scale = backend.scalar_to_float(
                xp.maximum(xp.asarray(1.0), xp.max(xp.abs(real_density)))
            )
            block_gradient_scale = backend.scalar_to_float(
                xp.maximum(xp.asarray(1.0), xp.max(xp.abs(real_gradient)))
            )
            if block_imaginary > self.imaginary_relative_tolerance * block_real_scale:
                raise FormulationError(
                    f"Wilson density has an unresolved imaginary component {block_imaginary:.3e}"
                )
            if block_gradient_imaginary > self.imaginary_relative_tolerance * block_gradient_scale:
                raise FormulationError(
                    "Wilson density gradient has an unresolved imaginary component "
                    f"{block_gradient_imaginary:.3e}"
                )
            block_minimum = backend.scalar_to_float(xp.min(real_density))
            if block_minimum < -self.negative_density_relative_tolerance * block_real_scale:
                raise FormulationError(
                    "Wilson density is negative beyond its declared numerical tolerance: "
                    f"{block_minimum:.3e}"
                )
            density_imaginary_maximum = max(density_imaginary_maximum, block_imaginary)
            gradient_imaginary_maximum = max(gradient_imaginary_maximum, block_gradient_imaginary)
            density_minimum = min(density_minimum, block_minimum)

            exc_host, potential_host, gradient_potential_host = self._evaluate_pointwise(
                backend.to_host(real_density),
                backend.to_host(real_gradient),
            )
            exc = backend.asarray(exc_host, dtype=xp.float64)
            potential = backend.asarray(potential_host, dtype=xp.float64)
            gradient_potential = backend.asarray(gradient_potential_host, dtype=xp.float64)
            for name, value in (
                ("GGA energy per particle", exc),
                ("GGA density derivative", potential),
                ("GGA density-gradient derivative", gradient_potential),
            ):
                backend.assert_resident(value, name=name)

            density_values[block.start : block.stop] = real_density
            density_gradients[:, block.start : block.stop] = real_gradient
            energy_per_particle[block.start : block.stop] = exc
            density_derivative[block.start : block.stop] = potential
            gradient_derivative[:, block.start : block.stop] = gradient_potential
            energy += xp.einsum("p,p,p->", block.weights_au, real_density, exc, optimize=True)
            particle_number += xp.einsum("p,p->", block.weights_au, real_density, optimize=True)
            lower += xp.einsum(
                "p,p,pi,pj->ij",
                block.weights_au,
                potential,
                frame.conj(),
                frame,
                optimize=True,
            )
            lower += xp.einsum(
                "p,xp,xpi,pj->ij",
                block.weights_au,
                gradient_potential,
                frame_gradient.conj(),
                frame,
                optimize=True,
            ) + xp.einsum(
                "p,xp,pi,xpj->ij",
                block.weights_au,
                gradient_potential,
                frame.conj(),
                frame_gradient,
                optimize=True,
            )

            if source_direction is not None:
                direction_line = source_direction.straight_line_integrals(
                    anchors[None, :, :],
                    block.coordinates_au[:, None, :],
                    backend,
                )
                direction_line_gradient = source_direction.straight_line_integral_gradients(
                    anchors[None, :, :],
                    block.coordinates_au[:, None, :],
                    backend,
                )
                untransformed_direction = prefactor * direction_line * (phase * block.values)
                untransformed_gradient_direction = (
                    prefactor
                    * direction_line[None, :, :]
                    * (
                        phase[None, :, :]
                        * (
                            block.gradients
                            + prefactor
                            * xp.moveaxis(line_gradient, -1, 0)
                            * block.values[None, :, :]
                        )
                    )
                    + phase[None, :, :]
                    * prefactor
                    * xp.moveaxis(direction_line_gradient, -1, 0)
                    * block.values[None, :, :]
                )
                if frame_change is None:
                    frame_direction = untransformed_direction
                    gradient_direction = untransformed_gradient_direction
                else:
                    frame_direction = untransformed_direction @ frame_change
                    gradient_direction = xp.einsum(
                        "xpi,ij->xpj",
                        untransformed_gradient_direction,
                        frame_change,
                        optimize=True,
                    )
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
                complex_gradient_direction = (
                    xp.einsum(
                        "mn,xpm,pn->xp",
                        density_matrix,
                        gradient_direction,
                        frame.conj(),
                        optimize=True,
                    )
                    + xp.einsum(
                        "mn,xpm,pn->xp",
                        density_matrix,
                        frame_gradient,
                        frame_direction.conj(),
                        optimize=True,
                    )
                    + xp.einsum(
                        "mn,pm,xpn->xp",
                        density_matrix,
                        frame_direction,
                        frame_gradient.conj(),
                        optimize=True,
                    )
                    + xp.einsum(
                        "mn,pm,xpn->xp",
                        density_matrix,
                        frame,
                        gradient_direction.conj(),
                        optimize=True,
                    )
                )
                real_direction = xp.real(complex_direction)
                real_gradient_direction = xp.real(complex_gradient_direction)
                block_source_imaginary = backend.scalar_to_float(
                    xp.max(xp.abs(xp.imag(complex_direction)))
                )
                block_source_gradient_imaginary = backend.scalar_to_float(
                    xp.max(xp.abs(xp.imag(complex_gradient_direction)))
                )
                direction_scale = backend.scalar_to_float(
                    xp.maximum(xp.asarray(1.0), xp.max(xp.abs(real_direction)))
                )
                gradient_direction_scale = backend.scalar_to_float(
                    xp.maximum(xp.asarray(1.0), xp.max(xp.abs(real_gradient_direction)))
                )
                if block_source_imaginary > self.imaginary_relative_tolerance * direction_scale:
                    raise FormulationError(
                        "Wilson source-density direction has an unresolved imaginary "
                        f"component {block_source_imaginary:.3e}"
                    )
                if (
                    block_source_gradient_imaginary
                    > self.imaginary_relative_tolerance * gradient_direction_scale
                ):
                    raise FormulationError(
                        "Wilson source density-gradient direction has an unresolved "
                        f"imaginary component {block_source_gradient_imaginary:.3e}"
                    )
                assert source_imaginary_maximum is not None
                assert source_gradient_imaginary_maximum is not None
                source_imaginary_maximum = max(source_imaginary_maximum, block_source_imaginary)
                source_gradient_imaginary_maximum = max(
                    source_gradient_imaginary_maximum,
                    block_source_gradient_imaginary,
                )
                assert source_density is not None
                assert source_gradient is not None
                assert source_energy is not None
                source_density[block.start : block.stop] = real_direction
                source_gradient[:, block.start : block.stop] = real_gradient_direction
                source_energy += xp.einsum(
                    "p,p,p->",
                    block.weights_au,
                    potential,
                    real_direction,
                    optimize=True,
                ) + xp.einsum(
                    "p,xp,xp->",
                    block.weights_au,
                    gradient_potential,
                    real_gradient_direction,
                    optimize=True,
                )

        lower_scale = xp.maximum(xp.asarray(1.0), xp.linalg.norm(lower))
        lower_hermiticity = backend.scalar_to_float(
            xp.linalg.norm(lower - lower.conj().T) / lower_scale
        )
        backend.synchronize()
        return WilsonGGAResult(
            energy=energy,
            lower_xc_matrix=lower,
            density=density_values,
            density_gradient=density_gradients,
            energy_per_particle=energy_per_particle,
            density_derivative=density_derivative,
            density_gradient_derivative=gradient_derivative,
            source_energy_direction=source_energy,
            source_density_direction=source_density,
            source_density_gradient_direction=source_gradient,
            electron_count_grid=particle_number,
            lower_hermiticity_residual=lower_hermiticity,
            density_imaginary_max_abs=density_imaginary_maximum,
            density_gradient_imaginary_max_abs=gradient_imaginary_maximum,
            density_real_minimum=density_minimum,
            source_density_imaginary_max_abs=source_imaginary_maximum,
            source_density_gradient_imaginary_max_abs=(source_gradient_imaginary_maximum),
            reference_fingerprint_sha256=reference.fingerprint_sha256,
            grid_fingerprint_sha256=self.quadrature.grid.fingerprint_sha256,
            backend=self.quadrature.backend_config.kind.value,
            device_index=self.quadrature.backend_config.device_index,
            functional=self.functional,
            realization=self.provenance.realization,
            coefficient_frame_applied=frame_change is not None,
            charge=checked_charge,
            hbar=checked_hbar,
        )

    def _evaluate_pointwise(
        self,
        density: np.ndarray,
        density_gradient: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = np.concatenate((density[None, :], density_gradient), axis=0)
        try:
            values = self._numint.eval_xc_eff(
                self.functional,
                rho,
                deriv=1,
                xctype="GGA",
                spin=0,
            )
        except Exception as error:
            raise FormulationError(
                f"PySCF/libxc failed for pure-GGA functional {self.functional!r}"
            ) from error
        energy_per_particle = np.asarray(values[0], dtype=np.float64)
        derivative = np.asarray(values[1], dtype=np.float64)
        if (
            energy_per_particle.shape != density.shape
            or derivative.shape != rho.shape
            or not np.all(np.isfinite(energy_per_particle))
            or not np.all(np.isfinite(derivative))
        ):
            raise FormulationError("PySCF/libxc returned invalid pure-GGA pointwise data")
        return energy_per_particle, derivative[0], derivative[1:4]

    def _validated_density(self, value: object) -> Any:
        xp = self.namespace
        dimension = self.quadrature.reference.core_operators.nao
        density = self.backend.asarray(value, dtype=xp.complex128)
        self.backend.assert_resident(density, name="contravariant coefficient density")
        if density.shape != (dimension, dimension):
            raise ConfigurationError(
                f"coefficient_density has shape {density.shape}; expected {(dimension, dimension)}"
            )
        if not _control_bool(xp.all(xp.isfinite(density)), self.backend):
            raise ConfigurationError("coefficient_density contains non-finite values")
        scale = xp.maximum(xp.asarray(1.0), xp.linalg.norm(density))
        residual = self.backend.scalar_to_float(xp.linalg.norm(density - density.conj().T) / scale)
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
                f"coefficient_frame has shape {frame.shape}; expected {(dimension, dimension)}"
            )
        if not _control_bool(xp.all(xp.isfinite(frame)), self.backend):
            raise ConfigurationError("coefficient_frame contains non-finite values")
        singular_values = xp.linalg.svd(frame, compute_uv=False)
        smallest = self.backend.scalar_to_float(xp.min(singular_values))
        largest = self.backend.scalar_to_float(xp.max(singular_values))
        if smallest <= 1.0e-12 * max(1.0, largest):
            raise ConfigurationError("coefficient_frame must be nonsingular")
        return frame


def prepare_wilson_gga(
    quadrature: AOQuadrature,
    functional: str,
    *,
    imaginary_relative_tolerance: float = 2.0e-10,
    negative_density_relative_tolerance: float = 2.0e-12,
) -> WilsonGGAEvaluator:
    """Prepare one explicitly labelled quadrature--Wilson pure-GGA action."""

    return WilsonGGAEvaluator(
        quadrature=quadrature,
        functional=functional,
        imaginary_relative_tolerance=imaginary_relative_tolerance,
        negative_density_relative_tolerance=negative_density_relative_tolerance,
    )


def _validate_vector_potential(value: object, name: str) -> None:
    for method in ("straight_line_integrals", "straight_line_integral_gradients"):
        if not callable(getattr(value, method, None)):
            raise TypeError(f"{name} must provide {method}")


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
