"""Finite-dimensional action values and source-direction contractions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from aion.backends import ArrayBackend
from aion.errors import FormulationError
from aion.formulations.types import EOMTriple


@dataclass(frozen=True, slots=True)
class OneElectronActionMatrixDirection:
    """Directional derivatives ``(delta S, delta K, delta omega_t)``."""

    metric: Any
    mechanical: Any
    connection: Any


@dataclass(frozen=True, slots=True)
class OneElectronActionContraction:
    """Resolved real contributions to an action value or source derivative."""

    metric_kinematic: Any
    connection_kinematic: Any
    mechanical: Any
    total: Any


def one_electron_velocity_density(
    density: Any,
    triple: EOMTriple,
    backend: ArrayBackend,
    *,
    hbar: float = 1.0,
) -> Any:
    r"""Return ``R=Cdot f C^dagger=G P`` on the unperturbed EOM shell."""

    checked_hbar = _positive_hbar(hbar)
    dimension = _validate_triple(triple, backend)
    _validate_square(density, dimension, backend, "density")
    xp = backend.namespace
    try:
        generator = xp.linalg.solve(
            triple.metric,
            -triple.connection - (1j / checked_hbar) * triple.hamiltonian_eom,
        )
    except Exception as exc:
        raise FormulationError("one-electron action generator solve failed") from exc
    result = generator @ density
    _assert_finite(result, backend, "velocity density")
    return result


def restricted_one_electron_action_value(
    density: Any,
    velocity_density: Any,
    triple: EOMTriple,
    backend: ArrayBackend,
    *,
    hbar: float = 1.0,
) -> OneElectronActionContraction:
    r"""Evaluate the real restricted one-electron Lagrangian at one time.

    ``density`` is ``P=C f C^dagger`` and ``velocity_density`` is
    ``R=Cdot f C^dagger``.  No equation of motion is used by this contraction;
    the coefficient history is held fixed exactly as required for a source
    derivative.
    """

    checked_hbar = _positive_hbar(hbar)
    dimension = _validate_triple(triple, backend)
    _validate_square(density, dimension, backend, "density")
    _validate_square(velocity_density, dimension, backend, "velocity density")
    return _contract(
        density,
        velocity_density,
        triple.metric,
        triple.hamiltonian_eom,
        triple.connection,
        checked_hbar,
        backend,
    )


def restricted_one_electron_action_directional_derivative(
    density: Any,
    velocity_density: Any,
    direction: OneElectronActionMatrixDirection,
    backend: ArrayBackend,
    *,
    hbar: float = 1.0,
) -> OneElectronActionContraction:
    r"""Contract a fixed-history source direction of the restricted action.

    For a real source parameter ``lambda``, this evaluates

    ``(i hbar/2) Tr[(R-R^dag) S_lambda
                    +P(omega_lambda-omega_lambda^dag)]
      -Re Tr[P K_lambda]``.

    The direction must descend from the same discretized action as the EOM.
    This function does not manufacture a current operator or differentiate
    the electronic state.
    """

    if not isinstance(direction, OneElectronActionMatrixDirection):
        raise TypeError("direction must be a OneElectronActionMatrixDirection")
    checked_hbar = _positive_hbar(hbar)
    dimension = _validate_direction(direction, backend)
    _validate_square(density, dimension, backend, "density")
    _validate_square(velocity_density, dimension, backend, "velocity density")
    return _contract(
        density,
        velocity_density,
        direction.metric,
        direction.mechanical,
        direction.connection,
        checked_hbar,
        backend,
    )


def _contract(
    density: Any,
    velocity_density: Any,
    metric: Any,
    mechanical: Any,
    connection: Any,
    hbar: float,
    backend: ArrayBackend,
) -> OneElectronActionContraction:
    xp = backend.namespace
    velocity_antihermitian = velocity_density - velocity_density.conj().T
    connection_antihermitian = connection - connection.conj().T
    metric_term = xp.real(
        0.5j * hbar * xp.einsum("ij,ji->", velocity_antihermitian, metric, optimize=True)
    )
    connection_term = xp.real(
        0.5j * hbar * xp.einsum("ij,ji->", density, connection_antihermitian, optimize=True)
    )
    mechanical_term = -xp.real(xp.einsum("ij,ji->", density, mechanical, optimize=True))
    total = metric_term + connection_term + mechanical_term
    for name, value in (
        ("metric action contraction", metric_term),
        ("connection action contraction", connection_term),
        ("mechanical action contraction", mechanical_term),
        ("total action contraction", total),
    ):
        _assert_finite(value, backend, name)
    return OneElectronActionContraction(
        metric_kinematic=metric_term,
        connection_kinematic=connection_term,
        mechanical=mechanical_term,
        total=total,
    )


def _validate_triple(triple: EOMTriple, backend: ArrayBackend) -> int:
    if not isinstance(triple, EOMTriple):
        raise TypeError("triple must be an EOMTriple")
    dimension = _matrix_dimension(triple.metric, backend, "metric")
    _validate_square(
        triple.hamiltonian_eom,
        dimension,
        backend,
        "mechanical matrix",
    )
    _validate_square(triple.connection, dimension, backend, "connection")
    return dimension


def _validate_direction(
    direction: OneElectronActionMatrixDirection,
    backend: ArrayBackend,
) -> int:
    dimension = _matrix_dimension(direction.metric, backend, "metric direction")
    _validate_square(
        direction.mechanical,
        dimension,
        backend,
        "mechanical direction",
    )
    _validate_square(
        direction.connection,
        dimension,
        backend,
        "connection direction",
    )
    return dimension


def _matrix_dimension(value: Any, backend: ArrayBackend, name: str) -> int:
    backend.assert_resident(value, name=name)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] == 0:
        raise FormulationError(f"{name} must be a nonempty square matrix")
    _assert_finite(value, backend, name)
    return int(value.shape[0])


def _validate_square(
    value: Any,
    dimension: int,
    backend: ArrayBackend,
    name: str,
) -> None:
    backend.assert_resident(value, name=name)
    if value.shape != (dimension, dimension):
        raise FormulationError(f"{name} has an incompatible shape")
    _assert_finite(value, backend, name)


def _assert_finite(value: Any, backend: ArrayBackend, name: str) -> None:
    xp = backend.namespace
    finite = backend.scalar_to_float(xp.asarray(xp.all(xp.isfinite(value)), dtype=xp.float64))
    if not bool(finite):
        raise FormulationError(f"{name} contains non-finite values")


def _positive_hbar(value: float) -> float:
    if isinstance(value, bool):
        raise FormulationError("hbar must be finite and positive")
    checked = float(value)
    if not math.isfinite(checked) or checked <= 0.0:
        raise FormulationError("hbar must be finite and positive")
    return checked
