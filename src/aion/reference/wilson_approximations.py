"""Magnetic-amplitude derivatives and one-electron Wilson truncations.

The endpoint link is retained exactly.  Only the gauge-invariant internal
uniform-field primitives are expanded under a bookkeeping parameter
``lambda_B``:

``F(lambda_B) = exp(lambda_B * g)`` and ``C_i(lambda_B) = lambda_B * C_i``.

The resulting coefficients are assembled consistently in the overlap,
kinetic, nuclear-potential, and mechanical matrices.  Truncation through order
one is therefore action-complete ``B1[1e]`` relative to the declared static
independent-electron ``S/T/V`` closure.  Order two is the corresponding
quadratic uniform-field ``B^(2,0)[1e]`` truncation (called ``B2[1e]`` in the
milestone plan); it is not a magnetic-gradient expansion.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Any

import numpy as np

from aion.gauge import AOAnchors, UniformMagneticGauge
from .wilson_grid import (
    GaugeMetadata,
    GridMetadata,
    KineticSectors,
    OneElectronMatrices,
    WilsonGridResult,
    analytic_bare_matrices,
    build_unpruned_grid,
    endpoint_link_matrix,
    uniform_anchored_vectors,
    uniform_triangle_flux,
)


@dataclass(frozen=True)
class MagneticTaylorOrder:
    """One coefficient of the internal ``lambda_B`` Taylor expansion.

    ``barred`` stores ``(1 / order!) d^order X / d lambda_B^order`` at zero.
    The supplied physical magnetic-field vector is already included, so the
    coefficient is evaluated at the physical bookkeeping value
    ``lambda_B = 1`` when truncation orders are summed.
    """

    order: int
    barred: OneElectronMatrices
    kinetic_sectors: KineticSectors


@dataclass(frozen=True)
class UniformMagneticTaylorResult:
    """Analytic internal-field coefficients and exact-link truncations."""

    orders: tuple[MagneticTaylorOrder, ...]
    quadrature_zeroth: OneElectronMatrices
    theta: np.ndarray
    ao_anchors: np.ndarray
    gauge: GaugeMetadata
    grid: GridMetadata
    charge: float
    hbar: float
    mass: float

    @property
    def max_order(self) -> int:
        return len(self.orders) - 1

    def coefficient(self, order: int) -> MagneticTaylorOrder:
        """Return the coefficient of ``lambda_B**order``."""

        checked = self._checked_order(order)
        return self.orders[checked]

    def directional_derivative(self, order: int) -> OneElectronMatrices:
        """Return the derivative at zero along the supplied field vector.

        For derivatives with respect to a scalar field amplitude, supply a
        unit direction as the magnetic field.  The second-order return value
        is the second derivative, not its Taylor coefficient.
        """

        checked = self._checked_order(order)
        return _scale_matrices(
            self.orders[checked].barred,
            float(factorial(checked)),
        )

    def barred_through(self, order: int) -> OneElectronMatrices:
        """Return the barred one-electron truncation through ``order``."""

        checked = self._checked_order(order)
        return _sum_matrices(entry.barred for entry in self.orders[: checked + 1])

    def lower_through(self, order: int) -> OneElectronMatrices:
        """Return the truncation with the physical endpoint link kept exact."""

        return _dress_matrices(self.theta, self.barred_through(order))

    @property
    def p0_barred(self) -> OneElectronMatrices:
        return self.barred_through(0)

    @property
    def b1_one_electron_barred(self) -> OneElectronMatrices:
        return self.barred_through(1)

    @property
    def b2_one_electron_barred(self) -> OneElectronMatrices:
        return self.barred_through(2)

    @property
    def p0_lower(self) -> OneElectronMatrices:
        return self.lower_through(0)

    @property
    def b1_one_electron_lower(self) -> OneElectronMatrices:
        return self.lower_through(1)

    @property
    def b2_one_electron_lower(self) -> OneElectronMatrices:
        return self.lower_through(2)

    def _checked_order(self, order: int) -> int:
        if not isinstance(order, (int, np.integer)):
            raise TypeError("order must be an integer")
        checked = int(order)
        if not 0 <= checked <= self.max_order:
            raise ValueError(f"order must lie between 0 and {self.max_order}")
        return checked


@dataclass(frozen=True)
class GIAOOneElectronDerivatives:
    """Independent libcint GIAO derivatives in matched Wilson conventions.

    Every matrix has shape ``(3, nao, nao)``; the leading index differentiates
    with respect to the Cartesian magnetic-field component.  ``lower`` is the
    full London/GIAO derivative, ``endpoint`` is ``Theta' X(0)``, and
    ``barred = lower - endpoint`` is directly comparable with derivatives of
    the internal Wilson matrices returned by this module.

    The underlying libcint kernels use atomic units, electron charge ``-1``,
    symmetric gauge, and gauge origin zero.  These restrictions are part of
    the result's contract rather than configurable parameters.
    """

    lower: OneElectronMatrices
    endpoint: OneElectronMatrices
    barred: OneElectronMatrices


def pyscf_giao_one_electron_derivatives(
    mol: Any,
) -> GIAOOneElectronDerivatives:
    """Return independent Cartesian GIAO derivatives for the ``S/T/V`` closure.

    PySCF/libcint's kinetic derivative requires both the derivative of the
    Gaussian phase and the explicit orbital angular-momentum contribution.
    With the Wilson conventions used here the complete Cartesian kernels are

    ``-i int1e_igovlp``,
    ``-i (int1e_igkin + int1e_giao_irjxp / 2)``, and
    ``-i int1e_ignuc``.

    The endpoint derivative is removed analytically using the AO-center map,
    leaving the gauge-invariant barred derivative used by the Taylor oracle.
    """

    anchors = AOAnchors.from_mol(mol)
    ao_anchors = np.asarray(anchors.atom_coords[anchors.ao_to_atom], dtype=float)
    overlap = -1j * np.asarray(mol.intor("int1e_igovlp", comp=3))
    kinetic = -1j * (
        np.asarray(mol.intor("int1e_igkin", comp=3))
        + 0.5 * np.asarray(mol.intor("int1e_giao_irjxp", comp=3))
    )
    potential = -1j * np.asarray(mol.intor("int1e_ignuc", comp=3))
    lower = _make_matrices(overlap, kinetic, potential)

    bare = analytic_bare_matrices(mol, hbar=1.0, mass=1.0)
    theta_derivative = symmetric_endpoint_link_cartesian_derivatives_at_zero(
        ao_anchors,
    )
    endpoint = _make_matrices(
        theta_derivative * bare.overlap[None, :, :],
        theta_derivative * bare.kinetic[None, :, :],
        theta_derivative * bare.potential[None, :, :],
    )
    return GIAOOneElectronDerivatives(
        lower=lower,
        endpoint=endpoint,
        barred=_subtract_matrices(lower, endpoint),
    )


def symmetric_endpoint_link_cartesian_derivatives_at_zero(
    ao_anchors: np.ndarray,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> np.ndarray:
    """Return ``d Theta_ij / d B_a`` at zero in symmetric gauge, origin zero.

    The result has shape ``(3, nao, nao)`` and uses the same ket-to-bra path
    orientation as :func:`endpoint_link_matrix`.
    """

    anchors = _cartesian_rows(ao_anchors, name="ao_anchors")
    if not np.isfinite(charge):
        raise ValueError("charge must be finite")
    if not np.isfinite(hbar) or hbar <= 0.0:
        raise ValueError("hbar must be positive and finite")
    derivatives = []
    for direction in np.eye(3):
        gauge = UniformMagneticGauge(direction, gauge="symmetric", origin=np.zeros(3))
        line_integrals = gauge.straight_line_integrals(
            anchors[None, :, :],
            anchors[:, None, :],
        )
        derivatives.append((1j * charge / hbar) * line_integrals)
    return np.asarray(derivatives, dtype=np.complex128)


def uniform_triangle_factor_taylor_coefficients(
    points: np.ndarray,
    ao_anchors: np.ndarray,
    magnetic_field: np.ndarray,
    *,
    max_order: int = 2,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> tuple[np.ndarray, ...]:
    """Return coefficients of ``exp(i q lambda_B Phi / hbar)`` through order.

    Entry ``n`` is ``g**n / n!`` with ``g = i q Phi / hbar``.  Consequently,
    ``n!`` times that entry is the exact ``n``th primitive derivative at
    ``lambda_B = 0``.
    """

    checked_order = _validate_max_order(max_order)
    if not np.isfinite(charge):
        raise ValueError("charge must be finite")
    if not np.isfinite(hbar) or hbar <= 0.0:
        raise ValueError("hbar must be positive and finite")
    flux = uniform_triangle_flux(points, ao_anchors, magnetic_field)
    generator = (1j * float(charge) / float(hbar)) * flux
    return tuple(
        generator**order / float(factorial(order))
        for order in range(checked_order + 1)
    )


def uniform_anchored_vector_taylor_coefficients(
    points: np.ndarray,
    ao_anchors: np.ndarray,
    magnetic_field: np.ndarray,
    *,
    max_order: int = 2,
) -> tuple[np.ndarray, ...]:
    """Return coefficients of ``C_i(lambda_B) = lambda_B C_i``."""

    checked_order = _validate_max_order(max_order)
    linear = uniform_anchored_vectors(points, ao_anchors, magnetic_field)
    zero = np.zeros_like(linear)
    return tuple(
        linear if order == 1 else np.array(zero, copy=True)
        for order in range(checked_order + 1)
    )


def evaluate_uniform_magnetic_taylor_matrices(
    mol: Any,
    gauge: UniformMagneticGauge,
    *,
    max_order: int = 2,
    grid_level: int = 3,
    grid: tuple[np.ndarray, np.ndarray] | None = None,
    block_size: int = 2048,
    charge: float = -1.0,
    hbar: float = 1.0,
    mass: float = 1.0,
) -> UniformMagneticTaylorResult:
    """Differentiate uniform-field barred matrices under the spatial integral.

    Order zero uses analytic PySCF integrals.  The returned
    ``quadrature_zeroth`` retains the matching grid value so finite-field exact
    matrices can be zero-field corrected when testing asymptotic scaling.
    Orders one and two contain every contribution generated by ``F`` and
    ``C_i`` in the declared one-electron ``S/T/V`` closure.
    """

    checked_order = _validate_max_order(max_order)
    if not isinstance(gauge, UniformMagneticGauge):
        raise TypeError("gauge must be a UniformMagneticGauge")
    if not isinstance(block_size, (int, np.integer)) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if not np.isfinite(charge):
        raise ValueError("charge must be finite")
    if not np.isfinite(hbar) or hbar <= 0.0:
        raise ValueError("hbar must be positive and finite")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass must be positive and finite")
    if getattr(mol, "has_ecp", lambda: False)():
        raise NotImplementedError("the nuclear grid potential does not support ECPs")

    try:
        from pyscf.dft import numint
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError("PySCF is required; install the 'aion[pyscf]' extra") from exc

    if grid is None:
        coords, weights = build_unpruned_grid(mol, level=grid_level)
        recorded_level: int | None = int(grid_level)
    else:
        if len(grid) != 2:
            raise ValueError("grid must be a (coords, weights) pair")
        coords = _cartesian_rows(grid[0], name="grid coordinates")
        weights = np.asarray(grid[1], dtype=float)
        recorded_level = None
    if weights.shape != (coords.shape[0],):
        raise ValueError("grid weights must have shape (npoint,)")
    if coords.shape[0] == 0:
        raise ValueError("grid must contain at least one point")
    if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(weights)):
        raise ValueError("grid coordinates and weights must be finite")

    anchors = AOAnchors.from_mol(mol)
    ao_anchors = np.asarray(anchors.atom_coords[anchors.ao_to_atom], dtype=float)
    nao = anchors.nao
    theta = endpoint_link_matrix(gauge, ao_anchors, charge=charge, hbar=hbar)
    accumulators = [_matrix_accumulators(nao) for _ in range(checked_order + 1)]
    sector_accumulators = [
        {name: _zero_matrix(nao) for name in ("pp", "p_c", "c_p", "c_c")}
        for _ in range(checked_order + 1)
    ]
    nuclei = np.asarray(mol.atom_coords(unit="Bohr"), dtype=float)
    nuclear_charges = np.asarray(mol.atom_charges(), dtype=float)

    for start in range(0, coords.shape[0], int(block_size)):
        stop = min(start + int(block_size), coords.shape[0])
        block_coords = coords[start:stop]
        block_weights = weights[start:stop]
        ao_data = np.asarray(numint.eval_ao(mol, block_coords, deriv=1))
        values = np.asarray(ao_data[0], dtype=np.complex128)
        gradients = np.asarray(
            np.moveaxis(ao_data[1:4], 0, -1),
            dtype=np.complex128,
        )
        potential = _nuclear_attraction(block_coords, nuclei, nuclear_charges)
        factor_coefficients = uniform_triangle_factor_taylor_coefficients(
            block_coords,
            ao_anchors,
            gauge.magnetic_field,
            max_order=checked_order,
            charge=charge,
            hbar=hbar,
        )
        anchored = uniform_anchored_vectors(
            block_coords,
            ao_anchors,
            gauge.magnetic_field,
        )
        bare_momentum = -1j * hbar * gradients
        anchored_momentum = charge * anchored * values[:, :, None]

        for order in range(checked_order + 1):
            factor = factor_coefficients[order]
            accumulator = accumulators[order]
            sectors = sector_accumulators[order]
            accumulator["overlap"] += _factorized_pair(
                values,
                values,
                block_weights,
                factor,
            )
            accumulator["potential"] += _factorized_pair(
                values,
                values,
                block_weights * potential,
                factor,
            )

            kinetic_scale = 1.0 / (2.0 * mass)
            pp = kinetic_scale * _factorized_vector_pair(
                bare_momentum,
                bare_momentum,
                block_weights,
                factor,
            )
            p_c = _zero_matrix(nao)
            c_p = _zero_matrix(nao)
            c_c = _zero_matrix(nao)
            if order >= 1:
                previous_factor = factor_coefficients[order - 1]
                p_c = kinetic_scale * _factorized_vector_pair(
                    bare_momentum,
                    anchored_momentum,
                    block_weights,
                    previous_factor,
                )
                c_p = kinetic_scale * _factorized_vector_pair(
                    anchored_momentum,
                    bare_momentum,
                    block_weights,
                    previous_factor,
                )
            if order >= 2:
                previous_second_factor = factor_coefficients[order - 2]
                c_c = kinetic_scale * _factorized_vector_pair(
                    anchored_momentum,
                    anchored_momentum,
                    block_weights,
                    previous_second_factor,
                )
            sectors["pp"] += pp
            sectors["p_c"] += p_c
            sectors["c_p"] += c_p
            sectors["c_c"] += c_c
            accumulator["kinetic"] += pp + p_c + c_p + c_c

    quadrature_orders = tuple(
        _finish_matrices(accumulator) for accumulator in accumulators
    )
    bare = analytic_bare_matrices(mol, hbar=hbar, mass=mass)
    p0_sectors = KineticSectors(
        pp=np.array(bare.kinetic, copy=True),
        p_c=_zero_matrix(nao),
        c_p=_zero_matrix(nao),
        c_c=_zero_matrix(nao),
    )
    orders: list[MagneticTaylorOrder] = []
    for order in range(checked_order + 1):
        kinetic_sectors = KineticSectors(**sector_accumulators[order])
        if order == 0:
            orders.append(
                MagneticTaylorOrder(
                    order=0,
                    barred=bare,
                    kinetic_sectors=p0_sectors,
                )
            )
        else:
            coefficient = quadrature_orders[order]
            orders.append(
                MagneticTaylorOrder(
                    order=order,
                    barred=_make_matrices(
                        coefficient.overlap,
                        kinetic_sectors.total,
                        coefficient.potential,
                    ),
                    kinetic_sectors=kinetic_sectors,
                )
            )

    gauge_metadata = GaugeMetadata(
        magnetic_field=np.array(gauge.magnetic_field, copy=True),
        gauge=gauge.gauge,
        origin=np.array(gauge.origin, copy=True),
        landau_u=(
            None if gauge.landau_u is None else np.array(gauge.landau_u, copy=True)
        ),
    )
    grid_metadata = GridMetadata(
        npoints=int(coords.shape[0]),
        level=recorded_level,
        pruning="none" if grid is None else "caller supplied",
        block_size=int(block_size),
        weight_sum=float(np.sum(weights)),
    )
    return UniformMagneticTaylorResult(
        orders=tuple(orders),
        quadrature_zeroth=quadrature_orders[0],
        theta=theta,
        ao_anchors=ao_anchors,
        gauge=gauge_metadata,
        grid=grid_metadata,
        charge=float(charge),
        hbar=float(hbar),
        mass=float(mass),
    )


def zero_field_corrected_barred_matrices(
    field_result: WilsonGridResult,
    zero_result: WilsonGridResult,
) -> OneElectronMatrices:
    """Replace the grid zero-field constant by the analytic bare matrices."""

    if field_result.barred.overlap.shape != zero_result.barred.overlap.shape:
        raise ValueError("field and zero-field results must have matching dimensions")
    return _add_matrices(
        field_result.bare,
        _subtract_matrices(field_result.barred, zero_result.barred),
    )


def _validate_max_order(max_order: int) -> int:
    if not isinstance(max_order, (int, np.integer)):
        raise TypeError("max_order must be an integer")
    checked = int(max_order)
    if not 0 <= checked <= 2:
        raise ValueError("max_order must lie between 0 and 2")
    return checked


def _factorized_pair(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
    factor: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "p,pi,pj,pij->ij",
        weights,
        left.conj(),
        right,
        factor,
        optimize="greedy",
    )


def _factorized_vector_pair(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
    factor: np.ndarray,
) -> np.ndarray:
    result = np.zeros((left.shape[1], right.shape[1]), dtype=np.complex128)
    for component in range(3):
        result += _factorized_pair(
            left[:, :, component],
            right[:, :, component],
            weights,
            factor,
        )
    return result


def _nuclear_attraction(
    points: np.ndarray,
    nuclei: np.ndarray,
    nuclear_charges: np.ndarray,
) -> np.ndarray:
    distances = np.linalg.norm(points[:, None, :] - nuclei[None, :, :], axis=2)
    if np.any(distances == 0.0):
        raise ValueError("the grid contains a nuclear position where V_nuc is singular")
    return -np.sum(nuclear_charges[None, :] / distances, axis=1)


def _matrix_accumulators(nao: int) -> dict[str, np.ndarray]:
    return {
        "overlap": _zero_matrix(nao),
        "kinetic": _zero_matrix(nao),
        "potential": _zero_matrix(nao),
    }


def _finish_matrices(accumulator: dict[str, np.ndarray]) -> OneElectronMatrices:
    return _make_matrices(
        accumulator["overlap"],
        accumulator["kinetic"],
        accumulator["potential"],
    )


def _make_matrices(
    overlap: np.ndarray,
    kinetic: np.ndarray,
    potential: np.ndarray,
) -> OneElectronMatrices:
    return OneElectronMatrices(
        overlap=overlap,
        kinetic=kinetic,
        potential=potential,
        mechanical=kinetic + potential,
    )


def _sum_matrices(matrices: Any) -> OneElectronMatrices:
    entries = tuple(matrices)
    if not entries:
        raise ValueError("at least one matrix collection is required")
    return _make_matrices(
        sum((entry.overlap for entry in entries), np.zeros_like(entries[0].overlap)),
        sum((entry.kinetic for entry in entries), np.zeros_like(entries[0].kinetic)),
        sum(
            (entry.potential for entry in entries),
            np.zeros_like(entries[0].potential),
        ),
    )


def _scale_matrices(
    matrices: OneElectronMatrices,
    scale: float,
) -> OneElectronMatrices:
    return _make_matrices(
        scale * matrices.overlap,
        scale * matrices.kinetic,
        scale * matrices.potential,
    )


def _add_matrices(
    left: OneElectronMatrices,
    right: OneElectronMatrices,
) -> OneElectronMatrices:
    return _make_matrices(
        left.overlap + right.overlap,
        left.kinetic + right.kinetic,
        left.potential + right.potential,
    )


def _subtract_matrices(
    left: OneElectronMatrices,
    right: OneElectronMatrices,
) -> OneElectronMatrices:
    return _make_matrices(
        left.overlap - right.overlap,
        left.kinetic - right.kinetic,
        left.potential - right.potential,
    )


def _dress_matrices(
    theta: np.ndarray,
    barred: OneElectronMatrices,
) -> OneElectronMatrices:
    return _make_matrices(
        theta * barred.overlap,
        theta * barred.kinetic,
        theta * barred.potential,
    )


def _zero_matrix(nao: int) -> np.ndarray:
    return np.zeros((nao, nao), dtype=np.complex128)


def _cartesian_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    return array
