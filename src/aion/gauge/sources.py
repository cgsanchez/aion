"""Uniform electromagnetic source parametrizations for Peierls geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np


VectorFunction = Callable[[float], np.ndarray]
ScalarFunction = Callable[[float], float]


def _as_vector3(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return array


def _as_cartesian_array(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have final dimension 3")
    return array


def _pair_displacements(coords: np.ndarray) -> np.ndarray:
    return coords[:, None, :] - coords[None, :, :]


@dataclass(frozen=True)
class UniformElectricGauge:
    """Uniform electric field represented by a length/velocity gauge family."""

    field: VectorFunction
    field_integral: VectorFunction
    lambda_value: ScalarFunction = lambda _t: 0.0
    lambda_derivative: ScalarFunction = lambda _t: 0.0

    @classmethod
    def length(
        cls,
        field: VectorFunction,
        field_integral: VectorFunction,
    ) -> "UniformElectricGauge":
        return cls(field=field, field_integral=field_integral)

    @classmethod
    def velocity(
        cls,
        field: VectorFunction,
        field_integral: VectorFunction,
    ) -> "UniformElectricGauge":
        return cls(
            field=field,
            field_integral=field_integral,
            lambda_value=lambda _t: 1.0,
            lambda_derivative=lambda _t: 0.0,
        )

    def electric_field(self, t: float) -> np.ndarray:
        return _as_vector3(self.field(t), name="field(t)")

    def impulse(self, t: float) -> np.ndarray:
        return _as_vector3(self.field_integral(t), name="field_integral(t)")

    def lam(self, t: float) -> float:
        return float(self.lambda_value(t))

    def lam_dot(self, t: float) -> float:
        return float(self.lambda_derivative(t))

    def vector_potential(self, t: float) -> np.ndarray:
        return -self.lam(t) * self.impulse(t)

    def vector_potential_dot(self, t: float) -> np.ndarray:
        return -self.lam_dot(t) * self.impulse(t) - self.lam(t) * self.electric_field(t)

    def site_scalar_potential(self, coords: np.ndarray, t: float) -> np.ndarray:
        effective_field = (
            -(1.0 - self.lam(t)) * self.electric_field(t)
            + self.lam_dot(t) * self.impulse(t)
        )
        return np.asarray(coords, dtype=float) @ effective_field

    def site_scalar_primitive(self, coords: np.ndarray, t: float) -> np.ndarray:
        """Return an exact primitive whose time derivative is ``Phi_a(t)``."""

        return np.asarray(coords, dtype=float) @ (
            -(1.0 - self.lam(t)) * self.impulse(t)
        )

    def site_scalar_integral(
        self,
        coords: np.ndarray,
        t0: float,
        t1: float,
    ) -> np.ndarray:
        return self.site_scalar_primitive(coords, t1) - self.site_scalar_primitive(
            coords,
            t0,
        )

    def bond_line_integrals(self, coords: np.ndarray, t: float) -> np.ndarray:
        displacements = _pair_displacements(np.asarray(coords, dtype=float))
        return np.einsum("x,abx->ab", self.vector_potential(t), displacements)

    def bond_line_integral_dots(self, coords: np.ndarray, t: float) -> np.ndarray:
        displacements = _pair_displacements(np.asarray(coords, dtype=float))
        return np.einsum("x,abx->ab", self.vector_potential_dot(t), displacements)

    def bond_electromotive_forces(self, coords: np.ndarray, t: float) -> np.ndarray:
        displacements = _pair_displacements(np.asarray(coords, dtype=float))
        return np.einsum("x,abx->ab", self.electric_field(t), displacements)


@dataclass(frozen=True)
class UniformMagneticGauge:
    """Static uniform magnetic field in symmetric or Landau gauge."""

    magnetic_field: np.ndarray
    gauge: Literal["symmetric", "landau"] = "symmetric"
    origin: np.ndarray | None = None
    landau_u: np.ndarray | None = None

    def __post_init__(self) -> None:
        field = _as_vector3(self.magnetic_field, name="magnetic_field")
        origin = np.zeros(3) if self.origin is None else _as_vector3(self.origin, name="origin")
        if self.gauge not in {"symmetric", "landau"}:
            raise ValueError("gauge must be 'symmetric' or 'landau'")
        landau_u = None
        if self.gauge == "landau":
            if self.landau_u is None:
                raise ValueError("landau_u is required for Landau gauge")
            landau_u = _as_vector3(self.landau_u, name="landau_u")
            norm = np.linalg.norm(landau_u)
            if norm == 0.0:
                raise ValueError("landau_u must be nonzero")
            landau_u = landau_u / norm
            field_norm = np.linalg.norm(field)
            if field_norm > 0.0:
                perpendicular = abs(float(np.dot(field, landau_u))) / field_norm
                if perpendicular > 1.0e-10:
                    raise ValueError("landau_u must be perpendicular to magnetic_field")
        object.__setattr__(self, "magnetic_field", field)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "landau_u", landau_u)

    @property
    def affine_matrix(self) -> np.ndarray:
        """Return ``M`` such that ``A(r) = M @ (r - origin)``.

        Both supported gauges are affine in position.  Exposing the affine
        map makes the straight-line Wilson integral and its endpoint
        derivative analytic without imposing a coordinate-axis convention.
        """

        bx, by, bz = self.magnetic_field
        cross_matrix = np.array(
            [
                [0.0, -bz, by],
                [bz, 0.0, -bx],
                [-by, bx, 0.0],
            ]
        )
        if self.gauge == "symmetric":
            return 0.5 * cross_matrix

        assert self.landau_u is not None
        return np.outer(np.cross(self.magnetic_field, self.landau_u), self.landau_u)

    def vector_potential(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the selected vector potential at Cartesian points."""

        points = _as_cartesian_array(points, name="points")
        return (points - self.origin) @ self.affine_matrix.T

    def straight_line_integrals(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
    ) -> np.ndarray:
        """Integrate ``A . dl`` from ``starts`` to ``ends``.

        The leading dimensions of ``starts`` and ``ends`` are broadcast.  An
        affine vector potential is integrated exactly by its midpoint value.
        """

        starts = _as_cartesian_array(starts, name="starts")
        ends = _as_cartesian_array(ends, name="ends")
        starts, ends = np.broadcast_arrays(starts, ends)
        displacements = ends - starts
        midpoints = 0.5 * (starts + ends)
        return np.einsum(
            "...x,...x->...",
            self.vector_potential(midpoints),
            displacements,
        )

    def straight_line_integral_gradients(
        self,
        starts: np.ndarray,
        ends: np.ndarray,
    ) -> np.ndarray:
        """Return the gradient with respect to each straight-line endpoint."""

        starts = _as_cartesian_array(starts, name="starts")
        ends = _as_cartesian_array(ends, name="ends")
        starts, ends = np.broadcast_arrays(starts, ends)
        displacements = ends - starts
        symmetric_part = self.affine_matrix + self.affine_matrix.T
        return self.vector_potential(starts) + 0.5 * (
            displacements @ symmetric_part.T
        )

    def anchor_to_point_line_integrals(
        self,
        anchors: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        """Return straight Wilson integrals with shape ``(npoint, nanchor)``."""

        anchors = _as_cartesian_array(anchors, name="anchors")
        points = _as_cartesian_array(points, name="points")
        if anchors.ndim != 2 or points.ndim != 2:
            raise ValueError("anchors and points must have shape (n, 3)")
        return self.straight_line_integrals(
            anchors[None, :, :],
            points[:, None, :],
        )

    def anchor_to_point_line_integral_gradients(
        self,
        anchors: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        """Return endpoint gradients with shape ``(npoint, nanchor, 3)``."""

        anchors = _as_cartesian_array(anchors, name="anchors")
        points = _as_cartesian_array(points, name="points")
        if anchors.ndim != 2 or points.ndim != 2:
            raise ValueError("anchors and points must have shape (n, 3)")
        return self.straight_line_integral_gradients(
            anchors[None, :, :],
            points[:, None, :],
        )

    def bond_line_integrals(self, coords: np.ndarray, _t: float | None = None) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        return self.straight_line_integrals(
            coords[None, :, :],
            coords[:, None, :],
        )

    def bond_line_integral_dots(self, coords: np.ndarray, _t: float | None = None) -> np.ndarray:
        return np.zeros((np.asarray(coords).shape[0], np.asarray(coords).shape[0]))

    def site_scalar_potential(self, coords: np.ndarray, _t: float | None = None) -> np.ndarray:
        return np.zeros(np.asarray(coords).shape[0])

    def site_scalar_integral(
        self,
        coords: np.ndarray,
        _t0: float,
        _t1: float,
    ) -> np.ndarray:
        return np.zeros(np.asarray(coords).shape[0])

    def bond_electromotive_forces(self, coords: np.ndarray, _t: float | None = None) -> np.ndarray:
        return np.zeros((np.asarray(coords).shape[0], np.asarray(coords).shape[0]))
