"""Smooth real vector-potential variations for weak current qualification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aion.backends import ArrayBackend
from aion.config.units import Vector3, finite_float, vector3
from aion.errors import ConfigurationError

from .magnetic import AffineMagneticGauge


@dataclass(frozen=True, slots=True)
class AffineGaugeDifferenceVariation:
    r"""Pure-gauge direction ``alpha=A_target-A_reference``.

    Both affine representatives must have the same physical magnetic field,
    so this direction has zero curl. It is useful for fixed-coefficient
    source derivatives and is not an electromagnetic gauge transformation of
    the matter coefficients.
    """

    target: AffineMagneticGauge
    reference: AffineMagneticGauge

    def __post_init__(self) -> None:
        if not isinstance(self.target, AffineMagneticGauge) or not isinstance(
            self.reference, AffineMagneticGauge
        ):
            raise ConfigurationError("gauge-difference endpoints must be affine gauges")
        if self.target.field != self.reference.field:
            raise ConfigurationError("gauge-difference endpoints must have the same field")

    def vector_potential(self, points_au: object, backend: ArrayBackend) -> Any:
        """Evaluate the curl-free affine vector-potential difference."""

        return self.target.vector_potential(points_au, backend) - self.reference.vector_potential(
            points_au,
            backend,
        )

    def straight_line_integrals(
        self,
        starts_au: object,
        ends_au: object,
        backend: ArrayBackend,
    ) -> Any:
        """Return the exact straight-path line-integral difference."""

        return self.target.straight_line_integrals(
            starts_au,
            ends_au,
            backend,
        ) - self.reference.straight_line_integrals(starts_au, ends_au, backend)

    def straight_line_integral_gradients(
        self,
        starts_au: object,
        ends_au: object,
        backend: ArrayBackend,
    ) -> Any:
        """Return the endpoint gradient of the line-integral difference."""

        return self.target.straight_line_integral_gradients(
            starts_au,
            ends_au,
            backend,
        ) - self.reference.straight_line_integral_gradients(
            starts_au,
            ends_au,
            backend,
        )


@dataclass(frozen=True, slots=True)
class GaussianVectorPotentialVariation:
    r"""A smooth localized test variation ``alpha(r)=a exp(-beta|r-c|^2)``.

    The field is used only as a real source direction, not as a propagated
    electromagnetic source.  Straight-path line integrals and their endpoint
    gradients are evaluated with a declared Gauss--Legendre order.
    """

    amplitude_au: Vector3
    center_au: Vector3 = (0.0, 0.0, 0.0)
    exponent_au_inverse2: float = 1.0
    path_quadrature_order: int = 20

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "amplitude_au",
            vector3(self.amplitude_au, "amplitude_au"),
        )
        object.__setattr__(self, "center_au", vector3(self.center_au, "center_au"))
        exponent = finite_float(self.exponent_au_inverse2, "exponent_au_inverse2")
        if exponent <= 0.0:
            raise ConfigurationError("exponent_au_inverse2 must be positive")
        object.__setattr__(self, "exponent_au_inverse2", exponent)
        order = self.path_quadrature_order
        if isinstance(order, bool) or not isinstance(order, int) or order < 2:
            raise ConfigurationError("path_quadrature_order must be an integer >= 2")

    def vector_potential(self, points_au: object, backend: ArrayBackend) -> Any:
        """Evaluate the test vector field at Cartesian points."""

        points = _cartesian_array(points_au, backend, "points_au")
        xp = backend.namespace
        center = backend.asarray(self.center_au, dtype=xp.float64)
        amplitude = backend.asarray(self.amplitude_au, dtype=xp.float64)
        relative = points - center
        envelope = xp.exp(
            -self.exponent_au_inverse2
            * xp.einsum("...x,...x->...", relative, relative, optimize=True)
        )
        return envelope[..., None] * amplitude

    def jacobian(self, points_au: object, backend: ArrayBackend) -> Any:
        r"""Return ``partial alpha_i / partial r_j`` at Cartesian points."""

        points = _cartesian_array(points_au, backend, "points_au")
        xp = backend.namespace
        center = backend.asarray(self.center_au, dtype=xp.float64)
        alpha = self.vector_potential(points, backend)
        logarithmic_gradient = -2.0 * self.exponent_au_inverse2 * (points - center)
        return alpha[..., :, None] * logarithmic_gradient[..., None, :]

    def straight_line_integrals(
        self,
        starts_au: object,
        ends_au: object,
        backend: ArrayBackend,
    ) -> Any:
        """Numerically integrate ``alpha.dl`` along broadcast straight paths."""

        starts, ends = _broadcast_paths(starts_au, ends_au, backend)
        xp = backend.namespace
        displacement = ends - starts
        result = xp.zeros(starts.shape[:-1], dtype=xp.float64)
        nodes, weights = _unit_legendre(self.path_quadrature_order, backend)
        for index in range(self.path_quadrature_order):
            point = starts + nodes[index] * displacement
            alpha = self.vector_potential(point, backend)
            result += weights[index] * xp.einsum(
                "...x,...x->...",
                displacement,
                alpha,
                optimize=True,
            )
        return result

    def straight_line_integral_gradients(
        self,
        starts_au: object,
        ends_au: object,
        backend: ArrayBackend,
    ) -> Any:
        """Differentiate each line integral with respect to its end point."""

        starts, ends = _broadcast_paths(starts_au, ends_au, backend)
        xp = backend.namespace
        displacement = ends - starts
        result = xp.zeros(starts.shape, dtype=xp.float64)
        nodes, weights = _unit_legendre(self.path_quadrature_order, backend)
        for index in range(self.path_quadrature_order):
            node = nodes[index]
            point = starts + node * displacement
            alpha = self.vector_potential(point, backend)
            jacobian = self.jacobian(point, backend)
            result += weights[index] * (
                alpha
                + node
                * xp.einsum(
                    "...i,...ij->...j",
                    displacement,
                    jacobian,
                    optimize=True,
                )
            )
        return result


@dataclass(frozen=True, slots=True)
class GaussianScalarGaugeVariation:
    r"""Gradient test ``alpha=grad lambda`` from a localized scalar.

    ``lambda(r)=amplitude*exp(-beta*|r-center|^2)``.  Its straight-line
    integral is evaluated exactly from the endpoint values.  The object is
    both a smooth vector-potential test and the spatial weight used by the
    weak finite-region continuity audit.
    """

    amplitude: float = 1.0
    center_au: Vector3 = (0.0, 0.0, 0.0)
    exponent_au_inverse2: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "amplitude", finite_float(self.amplitude, "amplitude"))
        object.__setattr__(self, "center_au", vector3(self.center_au, "center_au"))
        exponent = finite_float(self.exponent_au_inverse2, "exponent_au_inverse2")
        if exponent <= 0.0:
            raise ConfigurationError("exponent_au_inverse2 must be positive")
        object.__setattr__(self, "exponent_au_inverse2", exponent)

    def scalar_field(self, points_au: object, backend: ArrayBackend) -> Any:
        """Evaluate the real scalar gauge test ``lambda``."""

        points = _cartesian_array(points_au, backend, "points_au")
        xp = backend.namespace
        center = backend.asarray(self.center_au, dtype=xp.float64)
        relative = points - center
        return self.amplitude * xp.exp(
            -self.exponent_au_inverse2
            * xp.einsum("...x,...x->...", relative, relative, optimize=True)
        )

    def vector_potential(self, points_au: object, backend: ArrayBackend) -> Any:
        """Evaluate ``grad lambda`` at Cartesian points."""

        points = _cartesian_array(points_au, backend, "points_au")
        xp = backend.namespace
        center = backend.asarray(self.center_au, dtype=xp.float64)
        return (
            -2.0
            * self.exponent_au_inverse2
            * (points - center)
            * self.scalar_field(points, backend)[..., None]
        )

    def straight_line_integrals(
        self,
        starts_au: object,
        ends_au: object,
        backend: ArrayBackend,
    ) -> Any:
        """Return ``lambda(end)-lambda(start)`` exactly."""

        starts, ends = _broadcast_paths(starts_au, ends_au, backend)
        return self.scalar_field(ends, backend) - self.scalar_field(starts, backend)

    def straight_line_integral_gradients(
        self,
        starts_au: object,
        ends_au: object,
        backend: ArrayBackend,
    ) -> Any:
        """Return the endpoint gradient of the exact path integral."""

        _, ends = _broadcast_paths(starts_au, ends_au, backend)
        return self.vector_potential(ends, backend)


@dataclass(frozen=True, slots=True)
class PerturbedVectorPotential:
    """Linear source curve ``A_epsilon=A_0+epsilon*alpha``.

    Both operands are required to provide vector-potential values,
    straight-line integrals, and endpoint gradients.  This class is used to
    finite-difference the same grid action whose analytic weak derivative is
    evaluated at ``epsilon=0``.
    """

    base: Any
    direction: Any
    amplitude: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "amplitude", finite_float(self.amplitude, "amplitude"))
        for label, value in (("base", self.base), ("direction", self.direction)):
            for method in (
                "vector_potential",
                "straight_line_integrals",
                "straight_line_integral_gradients",
            ):
                if not callable(getattr(value, method, None)):
                    raise ConfigurationError(f"{label} must provide {method}")

    def vector_potential(self, points_au: object, backend: ArrayBackend) -> Any:
        return self.base.vector_potential(points_au, backend) + self.amplitude * (
            self.direction.vector_potential(points_au, backend)
        )

    def straight_line_integrals(
        self,
        starts_au: object,
        ends_au: object,
        backend: ArrayBackend,
    ) -> Any:
        return self.base.straight_line_integrals(starts_au, ends_au, backend) + (
            self.amplitude * self.direction.straight_line_integrals(starts_au, ends_au, backend)
        )

    def straight_line_integral_gradients(
        self,
        starts_au: object,
        ends_au: object,
        backend: ArrayBackend,
    ) -> Any:
        return self.base.straight_line_integral_gradients(
            starts_au, ends_au, backend
        ) + self.amplitude * self.direction.straight_line_integral_gradients(
            starts_au, ends_au, backend
        )


@dataclass(frozen=True, slots=True)
class AffineVectorFieldVariation:
    r"""General affine vector test ``alpha(r)=offset+M(r-origin)``.

    The class supplies exact straight-segment integrals and endpoint
    gradients.  It is useful for pairing a variational current with the
    uniform-plus-induction electric field of a time-dependent uniform
    magnetic source; the name deliberately does not assign a physical gauge
    role to the test field.
    """

    offset_au: Vector3 = (0.0, 0.0, 0.0)
    matrix_au: tuple[Vector3, Vector3, Vector3] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    origin_au: Vector3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset_au", vector3(self.offset_au, "offset_au"))
        object.__setattr__(self, "origin_au", vector3(self.origin_au, "origin_au"))
        rows = tuple(
            vector3(row, f"matrix_au[{index}]") for index, row in enumerate(self.matrix_au)
        )
        if len(rows) != 3:
            raise ConfigurationError("matrix_au must contain three Cartesian rows")
        object.__setattr__(self, "matrix_au", rows)

    def vector_potential(self, points_au: object, backend: ArrayBackend) -> Any:
        """Evaluate the affine vector test at Cartesian points."""

        points = _cartesian_array(points_au, backend, "points_au")
        xp = backend.namespace
        origin = backend.asarray(self.origin_au, dtype=xp.float64)
        offset = backend.asarray(self.offset_au, dtype=xp.float64)
        matrix = backend.asarray(self.matrix_au, dtype=xp.float64)
        return offset + (points - origin) @ matrix.T

    def straight_line_integrals(
        self,
        starts_au: object,
        ends_au: object,
        backend: ArrayBackend,
    ) -> Any:
        """Integrate the affine field exactly along each straight segment."""

        starts, ends = _broadcast_paths(starts_au, ends_au, backend)
        displacement = ends - starts
        midpoint = 0.5 * (starts + ends)
        return backend.namespace.einsum(
            "...x,...x->...",
            self.vector_potential(midpoint, backend),
            displacement,
            optimize=True,
        )

    def straight_line_integral_gradients(
        self,
        starts_au: object,
        ends_au: object,
        backend: ArrayBackend,
    ) -> Any:
        """Differentiate the exact segment integral with respect to its end."""

        starts, ends = _broadcast_paths(starts_au, ends_au, backend)
        xp = backend.namespace
        matrix = backend.asarray(self.matrix_au, dtype=xp.float64)
        return self.vector_potential(starts, backend) + 0.5 * (
            (ends - starts) @ (matrix + matrix.T).T
        )

    @classmethod
    def from_uniform_magnetic_electric_field(
        cls,
        *,
        electric_field_origin_au: object,
        magnetic_field_dot_au: object,
        origin_au: object,
    ) -> AffineVectorFieldVariation:
        r"""Build ``E(r)=E_o+1/2 (r-o) cross Bdot`` exactly."""

        electric = vector3(electric_field_origin_au, "electric_field_origin_au")
        magnetic_dot = vector3(magnetic_field_dot_au, "magnetic_field_dot_au")
        bx, by, bz = magnetic_dot
        matrix = (
            (0.0, 0.5 * bz, -0.5 * by),
            (-0.5 * bz, 0.0, 0.5 * bx),
            (0.5 * by, -0.5 * bx, 0.0),
        )
        return cls(
            offset_au=electric,
            matrix_au=matrix,
            origin_au=vector3(origin_au, "origin_au"),
        )


def _cartesian_array(value: object, backend: ArrayBackend, name: str) -> Any:
    xp = backend.namespace
    result = backend.asarray(value, dtype=xp.float64)
    backend.assert_resident(result, name=name)
    if result.ndim == 0 or result.shape[-1] != 3:
        raise ConfigurationError(f"{name} must end in a Cartesian axis")
    finite = backend.scalar_to_float(xp.asarray(xp.all(xp.isfinite(result)), dtype=xp.float64))
    if not bool(finite):
        raise ConfigurationError(f"{name} must contain only finite values")
    return result


def _broadcast_paths(
    starts_au: object,
    ends_au: object,
    backend: ArrayBackend,
) -> tuple[Any, Any]:
    starts = _cartesian_array(starts_au, backend, "starts_au")
    ends = _cartesian_array(ends_au, backend, "ends_au")
    try:
        broadcast = backend.namespace.broadcast_arrays(starts, ends)
    except ValueError as exc:
        raise ConfigurationError("starts_au and ends_au cannot be broadcast") from exc
    return (broadcast[0], broadcast[1])


def _unit_legendre(order: int, backend: ArrayBackend) -> tuple[Any, Any]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    xp = backend.namespace
    return (
        backend.asarray(0.5 * (nodes + 1.0), dtype=xp.float64),
        backend.asarray(0.5 * weights, dtype=xp.float64),
    )
