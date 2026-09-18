"""Uniform-electric exact internal connection and E1 response tensors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aion.electronic_structure.ao_quadrature import AOQuadrature
from aion.electronic_structure.operators import build_e1_operators
from aion.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class UniformElectricE1Tensor:
    """Analytic and independently quadratured uniform-electric E1 tensors.

    The central dipole includes the signed particle charge.  The connection
    derivative is therefore ``-i d_alpha / hbar``.  Arrays remain resident on
    the quadrature backend.
    """

    central_dipoles: Any
    quadrature_central_dipoles: Any
    connection_derivatives: Any
    quadrature_connection_derivatives: Any
    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str
    backend: str
    device_index: int | None
    charge: float
    hbar: float


def evaluate_uniform_electric_e1_tensor(
    quadrature: AOQuadrature,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> UniformElectricE1Tensor:
    """Evaluate the analytic E1 tensor and an independent real-space oracle."""

    checked_charge, checked_hbar = _inputs(quadrature, charge, hbar)
    backend = quadrature.backend
    xp = backend.namespace
    reference = quadrature.reference
    analytic = build_e1_operators(
        reference.core_operators,
        reference.anchor_topology,
        charge=checked_charge,
    )
    analytic_dipoles = backend.asarray(analytic.central_dipoles, dtype=xp.complex128)

    nao = reference.core_operators.nao
    quadrature_dipoles = backend.zeros((3, nao, nao), dtype=xp.complex128)
    anchors = backend.asarray(
        reference.core_operators.nuclei.coordinates_au[
            reference.anchor_topology.ao_to_atom
        ],
        dtype=xp.float64,
    )
    pair_centers = 0.5 * (anchors[:, None, :] + anchors[None, :, :])
    for block in quadrature.blocks():
        pair_position = xp.einsum(
            "p,pm,pn,px->xmn",
            block.weights_au,
            block.values.conj(),
            block.values,
            block.coordinates_au,
            optimize=True,
        )
        pair_overlap = xp.einsum(
            "p,pm,pn->mn",
            block.weights_au,
            block.values.conj(),
            block.values,
            optimize=True,
        )
        quadrature_dipoles += checked_charge * (
            pair_position - xp.moveaxis(pair_centers, -1, 0) * pair_overlap[None, :, :]
        )

    connection = (-1j / checked_hbar) * analytic_dipoles
    quadrature_connection = (-1j / checked_hbar) * quadrature_dipoles
    backend.synchronize()
    return UniformElectricE1Tensor(
        central_dipoles=analytic_dipoles,
        quadrature_central_dipoles=quadrature_dipoles,
        connection_derivatives=connection,
        quadrature_connection_derivatives=quadrature_connection,
        reference_fingerprint_sha256=reference.fingerprint_sha256,
        grid_fingerprint_sha256=quadrature.grid.fingerprint_sha256,
        backend=quadrature.backend_config.kind.value,
        device_index=quadrature.backend_config.device_index,
        charge=checked_charge,
        hbar=checked_hbar,
    )


def evaluate_exact_uniform_electric_internal_connections(
    quadrature: AOQuadrature,
    electric_fields_au: Sequence[Sequence[float]],
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> tuple[Any, ...]:
    r"""Directly quadrature the exact uniform-electric internal connections.

    For a pair midpoint ``R_ij`` the exact internal generator is
    ``G_ij(r) = (r - R_ij) . E``.  It is evaluated here for each supplied
    source rather than contracted from :class:`UniformElectricE1Tensor`, so a
    centred source finite difference exercises an independent parent path.
    """

    checked_charge, checked_hbar = _inputs(quadrature, charge, hbar)
    fields = tuple(_field(value) for value in electric_fields_au)
    if not fields:
        raise ConfigurationError("electric_fields_au cannot be empty")

    backend = quadrature.backend
    xp = backend.namespace
    reference = quadrature.reference
    fields_array = backend.asarray(np.asarray(fields), dtype=xp.float64)
    nao = reference.core_operators.nao
    output = [backend.zeros((nao, nao), dtype=xp.complex128) for _ in fields]
    anchors = backend.asarray(
        reference.core_operators.nuclei.coordinates_au[
            reference.anchor_topology.ao_to_atom
        ],
        dtype=xp.float64,
    )
    pair_centers = 0.5 * (anchors[:, None, :] + anchors[None, :, :])
    center_projections = xp.einsum("fx,mnx->fmn", fields_array, pair_centers, optimize=True)
    prefactor = -1j * checked_charge / checked_hbar

    for block in quadrature.blocks():
        point_projections = xp.einsum(
            "fx,px->fp", fields_array, block.coordinates_au, optimize=True
        )
        generators = point_projections[:, :, None, None] - center_projections[:, None, :, :]
        values = xp.einsum(
            "p,pm,pn,fpmn->fmn",
            block.weights_au,
            block.values.conj(),
            block.values,
            generators,
            optimize=True,
        )
        for index, value in enumerate(output):
            value += prefactor * values[index]

    backend.synchronize()
    return tuple(output)


def _inputs(quadrature: AOQuadrature, charge: float, hbar: float) -> tuple[float, float]:
    if not isinstance(quadrature, AOQuadrature):
        raise TypeError("quadrature must be an AOQuadrature")
    checked_charge = _finite_scalar(charge, "charge")
    checked_hbar = _finite_scalar(hbar, "hbar")
    if checked_hbar <= 0.0:
        raise ConfigurationError("hbar must be positive")
    return checked_charge, checked_hbar


def _field(value: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ConfigurationError("every electric field must be a finite length-three vector")
    return (float(array[0]), float(array[1]), float(array[2]))


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be finite")
    checked = float(value)
    if not np.isfinite(checked):
        raise ConfigurationError(f"{name} must be finite")
    return checked
