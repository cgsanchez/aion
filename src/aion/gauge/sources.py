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
    def length(cls, field: VectorFunction) -> "UniformElectricGauge":
        zeros = lambda _t: np.zeros(3)
        return cls(field=field, field_integral=zeros)

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

    def bond_line_integrals(self, coords: np.ndarray, _t: float | None = None) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        displacements = _pair_displacements(coords)
        if self.gauge == "symmetric":
            left = coords[None, :, :] - self.origin[None, None, :]
            right = coords[:, None, :] - self.origin[None, None, :]
            triangles = np.cross(left, right)
            return 0.5 * np.einsum("x,abx->ab", self.magnetic_field, triangles)

        assert self.landau_u is not None
        midpoint = 0.5 * (coords[:, None, :] + coords[None, :, :])
        prefactor = np.einsum("abx,x->ab", midpoint - self.origin, self.landau_u)
        vector = np.cross(self.magnetic_field, self.landau_u)
        return prefactor * np.einsum("x,abx->ab", vector, displacements)

    def bond_line_integral_dots(self, coords: np.ndarray, _t: float | None = None) -> np.ndarray:
        return np.zeros((np.asarray(coords).shape[0], np.asarray(coords).shape[0]))

    def site_scalar_potential(self, coords: np.ndarray, _t: float | None = None) -> np.ndarray:
        return np.zeros(np.asarray(coords).shape[0])

    def bond_electromotive_forces(self, coords: np.ndarray, _t: float | None = None) -> np.ndarray:
        return np.zeros((np.asarray(coords).shape[0], np.asarray(coords).shape[0]))
