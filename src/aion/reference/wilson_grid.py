"""Real-space reference matrices for Wilson-dressed Gaussian orbitals.

This module implements the static, fixed-center, one-electron qualification
problem for a spatially uniform magnetic field.  It deliberately assembles
the same lower matrices through two geometrically independent routes:

``direct``
    Dress every AO with its straight-line Wilson phase and apply the
    covariant momentum using the pointwise vector potential and the explicit
    endpoint derivative of that phase.

``factorized``
    Remove the endpoint link and integrate the gauge-invariant uniform-field
    triangle factor and center-anchored magnetic vectors.

Both routes share the spatial quadrature and bare PySCF AO samples.  Agreement
therefore qualifies phase, orientation, and momentum conventions, while
zero-field comparisons with analytic PySCF integrals separately qualify the
grid itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aion.gauge import AOAnchors, UniformMagneticGauge
from .wilson_measures import PhaseSpreadMoments


@dataclass(frozen=True)
class OneElectronMatrices:
    """Overlap and one-electron mechanical lower matrices."""

    overlap: np.ndarray
    kinetic: np.ndarray
    potential: np.ndarray
    mechanical: np.ndarray


@dataclass(frozen=True)
class KineticSectors:
    """Gauge-invariant barred kinetic sectors retaining the exact ``F``."""

    pp: np.ndarray
    p_c: np.ndarray
    c_p: np.ndarray
    c_c: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.pp + self.p_c + self.c_p + self.c_c

    @property
    def T_pp(self) -> np.ndarray:
        return self.pp

    @property
    def T_pC(self) -> np.ndarray:
        return self.p_c

    @property
    def T_Cp(self) -> np.ndarray:
        return self.c_p

    @property
    def T_CC(self) -> np.ndarray:
        return self.c_c


@dataclass(frozen=True)
class DiagnosticCalculation:
    """One named internal-geometry calculation in barred and lower form."""

    barred: OneElectronMatrices
    lower: OneElectronMatrices
    kinetic_sectors: KineticSectors


@dataclass(frozen=True)
class WilsonDiagnostics:
    """The four diagnostic sector choices defined by the qualification note.

    The middle two entries are diagnostic sector deletions, not named magnetic
    hierarchy levels or action-complete dynamical theories.
    """

    p0: DiagnosticCalculation
    form_factor_only: DiagnosticCalculation
    anchored_vector_only: DiagnosticCalculation
    full_exact: DiagnosticCalculation


@dataclass(frozen=True)
class GaugeMetadata:
    """Serializable description of the affine magnetic gauge."""

    magnetic_field: np.ndarray
    gauge: str
    origin: np.ndarray
    landau_u: np.ndarray | None


@dataclass(frozen=True)
class GridMetadata:
    """Numerical quadrature metadata retained with every result."""

    npoints: int
    level: int | None
    pruning: str
    block_size: int
    weight_sum: float


@dataclass(frozen=True)
class WilsonGridResult:
    """Complete result of one uniform-field Wilson grid evaluation."""

    bare: OneElectronMatrices
    direct: OneElectronMatrices
    barred: OneElectronMatrices
    factorized: OneElectronMatrices
    theta: np.ndarray
    kinetic_sectors: KineticSectors
    diagnostics: WilsonDiagnostics
    phase_spread_moments: PhaseSpreadMoments
    ao_anchors: np.ndarray
    gauge: GaugeMetadata
    grid: GridMetadata
    charge: float
    hbar: float
    mass: float

    @property
    def S0(self) -> np.ndarray:
        return self.bare.overlap

    @property
    def T0(self) -> np.ndarray:
        return self.bare.kinetic

    @property
    def V0(self) -> np.ndarray:
        return self.bare.potential

    @property
    def K0(self) -> np.ndarray:
        return self.bare.mechanical


def build_unpruned_grid(mol: Any, *, level: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Build an unpruned PySCF atom-centered integration grid."""

    if not isinstance(level, (int, np.integer)) or level < 0:
        raise ValueError("level must be a nonnegative integer")
    try:
        from pyscf.dft import gen_grid
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError("PySCF is required; install the 'aion[pyscf]' extra") from exc

    grids = gen_grid.Grids(mol)
    grids.level = int(level)
    grids.prune = None
    grids.build()
    return np.asarray(grids.coords, dtype=float), np.asarray(grids.weights, dtype=float)


def endpoint_link_matrix(
    gauge: UniformMagneticGauge,
    ao_anchors: np.ndarray,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> np.ndarray:
    """Return ``Theta_ij`` for paths oriented from ket anchor ``j`` to ``i``."""

    anchors = _cartesian_rows(ao_anchors, name="ao_anchors")
    if hbar <= 0.0:
        raise ValueError("hbar must be positive")
    line_integrals = gauge.straight_line_integrals(
        anchors[None, :, :],
        anchors[:, None, :],
    )
    return np.exp((1j * charge / hbar) * line_integrals)


def uniform_triangle_flux(
    points: np.ndarray,
    ao_anchors: np.ndarray,
    magnetic_field: np.ndarray,
) -> np.ndarray:
    """Return ``Phi_ij(r)`` for the oriented triangle ``j -> r -> i -> j``."""

    points = _cartesian_rows(points, name="points")
    anchors = _cartesian_rows(ao_anchors, name="ao_anchors")
    field = _vector3(magnetic_field, name="magnetic_field")
    anchor_difference = anchors[:, None, :] - anchors[None, :, :]
    midpoints = 0.5 * (anchors[:, None, :] + anchors[None, :, :])
    phase_vectors = 0.5 * np.cross(anchor_difference, field)
    point_term = np.einsum("px,ijx->pij", points, phase_vectors)
    midpoint_term = np.einsum("ijx,ijx->ij", midpoints, phase_vectors)
    return point_term - midpoint_term[None, :, :]


def uniform_anchored_vectors(
    points: np.ndarray,
    ao_anchors: np.ndarray,
    magnetic_field: np.ndarray,
) -> np.ndarray:
    """Return ``C_i(r) = 1/2 (r - R_i) x B``."""

    points = _cartesian_rows(points, name="points")
    anchors = _cartesian_rows(ao_anchors, name="ao_anchors")
    field = _vector3(magnetic_field, name="magnetic_field")
    return 0.5 * np.cross(points[:, None, :] - anchors[None, :, :], field)


def evaluate_uniform_magnetic_matrices(
    mol: Any,
    gauge: UniformMagneticGauge,
    *,
    grid_level: int = 3,
    grid: tuple[np.ndarray, np.ndarray] | None = None,
    block_size: int = 2048,
    charge: float = -1.0,
    hbar: float = 1.0,
    mass: float = 1.0,
) -> WilsonGridResult:
    """Evaluate exact direct and endpoint-factorized one-electron matrices.

    Parameters use atomic units by default.  ``grid`` may provide an explicit
    ``(coords, weights)`` pair; otherwise an unpruned PySCF grid at
    ``grid_level`` is built.  Pair-resolved arrays are limited to one spatial
    block at a time.
    """

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
    theta = endpoint_link_matrix(
        gauge,
        ao_anchors,
        charge=charge,
        hbar=hbar,
    )

    direct_acc = _matrix_accumulators(nao)
    barred_acc = _matrix_accumulators(nao)
    sector_acc = {name: _zero_matrix(nao) for name in ("pp", "p_c", "c_p", "c_c")}
    no_flux_sector_acc = {
        name: _zero_matrix(nao) for name in ("p_c", "c_p", "c_c")
    }
    phase_spread_numerator = np.zeros((nao, nao), dtype=float)
    phase_spread_denominator = np.zeros((nao, nao), dtype=float)
    nuclei = np.asarray(mol.atom_coords(unit="Bohr"), dtype=float)
    nuclear_charges = np.asarray(mol.atom_charges(), dtype=float)

    for start in range(0, coords.shape[0], int(block_size)):
        stop = min(start + int(block_size), coords.shape[0])
        block_coords = coords[start:stop]
        block_weights = weights[start:stop]
        ao_data = np.asarray(numint.eval_ao(mol, block_coords, deriv=1))
        values = np.asarray(ao_data[0], dtype=np.complex128)
        gradients = np.asarray(np.moveaxis(ao_data[1:4], 0, -1), dtype=np.complex128)
        potential = _nuclear_attraction(block_coords, nuclei, nuclear_charges)

        _accumulate_direct_block(
            direct_acc,
            values,
            gradients,
            block_coords,
            block_weights,
            potential,
            ao_anchors,
            gauge,
            charge=charge,
            hbar=hbar,
            mass=mass,
        )
        _accumulate_factorized_block(
            barred_acc,
            sector_acc,
            no_flux_sector_acc,
            phase_spread_numerator,
            phase_spread_denominator,
            values,
            gradients,
            block_coords,
            block_weights,
            potential,
            ao_anchors,
            gauge.magnetic_field,
            charge=charge,
            hbar=hbar,
            mass=mass,
        )

    direct = _finish_matrices(direct_acc)
    barred = _finish_matrices(barred_acc)
    factorized = OneElectronMatrices(
        overlap=theta * barred.overlap,
        kinetic=theta * barred.kinetic,
        potential=theta * barred.potential,
        mechanical=theta * barred.mechanical,
    )
    bare = analytic_bare_matrices(mol, hbar=hbar, mass=mass)
    kinetic_sectors = KineticSectors(**sector_acc)
    no_flux_kinetic_sectors = KineticSectors(
        pp=np.array(bare.kinetic, copy=True),
        p_c=no_flux_sector_acc["p_c"],
        c_p=no_flux_sector_acc["c_p"],
        c_c=no_flux_sector_acc["c_c"],
    )
    diagnostics = _finish_diagnostics(
        theta,
        bare,
        barred,
        kinetic_sectors,
        no_flux_kinetic_sectors,
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
    return WilsonGridResult(
        bare=bare,
        direct=direct,
        barred=barred,
        factorized=factorized,
        theta=theta,
        kinetic_sectors=kinetic_sectors,
        diagnostics=diagnostics,
        phase_spread_moments=PhaseSpreadMoments(
            numerator=phase_spread_numerator,
            denominator=phase_spread_denominator,
        ),
        ao_anchors=ao_anchors,
        gauge=gauge_metadata,
        grid=grid_metadata,
        charge=float(charge),
        hbar=float(hbar),
        mass=float(mass),
    )


def _accumulate_direct_block(
    accumulator: dict[str, np.ndarray],
    values: np.ndarray,
    gradients: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    potential: np.ndarray,
    ao_anchors: np.ndarray,
    gauge: UniformMagneticGauge,
    *,
    charge: float,
    hbar: float,
    mass: float,
) -> None:
    line_integrals = gauge.anchor_to_point_line_integrals(ao_anchors, points)
    line_gradients = gauge.anchor_to_point_line_integral_gradients(
        ao_anchors,
        points,
    )
    wilson = np.exp((1j * charge / hbar) * line_integrals)
    dressed_values = wilson * values

    pointwise_a = gauge.vector_potential(points)[:, None, :]
    phase_residual = line_gradients - pointwise_a
    bare_momentum = -1j * hbar * gradients
    dressed_momentum = wilson[:, :, None] * (
        bare_momentum + charge * phase_residual * values[:, :, None]
    )

    accumulator["overlap"] += _ordinary_pair(dressed_values, dressed_values, weights)
    kinetic = _ordinary_vector_pair(dressed_momentum, dressed_momentum, weights)
    accumulator["kinetic"] += kinetic / (2.0 * mass)
    accumulator["potential"] += _ordinary_pair(
        dressed_values,
        dressed_values,
        weights * potential,
    )


def _accumulate_factorized_block(
    accumulator: dict[str, np.ndarray],
    sectors: dict[str, np.ndarray],
    no_flux_sectors: dict[str, np.ndarray],
    phase_spread_numerator: np.ndarray,
    phase_spread_denominator: np.ndarray,
    values: np.ndarray,
    gradients: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    potential: np.ndarray,
    ao_anchors: np.ndarray,
    magnetic_field: np.ndarray,
    *,
    charge: float,
    hbar: float,
    mass: float,
) -> None:
    flux = uniform_triangle_flux(points, ao_anchors, magnetic_field)
    triangle_factor = np.exp((1j * charge / hbar) * flux)
    anchored = uniform_anchored_vectors(points, ao_anchors, magnetic_field)
    bare_momentum = -1j * hbar * gradients
    anchored_momentum = charge * anchored * values[:, :, None]
    total_momentum = bare_momentum + anchored_momentum

    magnitudes = np.abs(values)
    phase_spread_denominator += magnitudes.T @ (weights[:, None] * magnitudes)
    phase_spread_numerator += np.einsum(
        "p,pi,pj,pij->ij",
        weights,
        magnitudes,
        magnitudes,
        np.abs(triangle_factor - 1.0) ** 2,
        optimize="greedy",
    )

    accumulator["overlap"] += _factorized_pair(
        values,
        values,
        weights,
        triangle_factor,
    )
    accumulator["potential"] += _factorized_pair(
        values,
        values,
        weights * potential,
        triangle_factor,
    )
    accumulator["kinetic"] += _factorized_vector_pair(
        total_momentum,
        total_momentum,
        weights,
        triangle_factor,
    ) / (2.0 * mass)

    scale = 1.0 / (2.0 * mass)
    sectors["pp"] += scale * _factorized_vector_pair(
        bare_momentum,
        bare_momentum,
        weights,
        triangle_factor,
    )
    sectors["p_c"] += scale * _factorized_vector_pair(
        bare_momentum,
        anchored_momentum,
        weights,
        triangle_factor,
    )
    sectors["c_p"] += scale * _factorized_vector_pair(
        anchored_momentum,
        bare_momentum,
        weights,
        triangle_factor,
    )
    sectors["c_c"] += scale * _factorized_vector_pair(
        anchored_momentum,
        anchored_momentum,
        weights,
        triangle_factor,
    )
    no_flux_sectors["p_c"] += scale * _ordinary_vector_pair(
        bare_momentum,
        anchored_momentum,
        weights,
    )
    no_flux_sectors["c_p"] += scale * _ordinary_vector_pair(
        anchored_momentum,
        bare_momentum,
        weights,
    )
    no_flux_sectors["c_c"] += scale * _ordinary_vector_pair(
        anchored_momentum,
        anchored_momentum,
        weights,
    )


def _ordinary_pair(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return left.conj().T @ (weights[:, None] * right)


def _ordinary_vector_pair(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    result = np.zeros((left.shape[1], right.shape[1]), dtype=np.complex128)
    for component in range(3):
        result += _ordinary_pair(left[:, :, component], right[:, :, component], weights)
    return result


def _factorized_pair(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
    triangle_factor: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "p,pi,pj,pij->ij",
        weights,
        left.conj(),
        right,
        triangle_factor,
        optimize="greedy",
    )


def _factorized_vector_pair(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
    triangle_factor: np.ndarray,
) -> np.ndarray:
    result = np.zeros((left.shape[1], right.shape[1]), dtype=np.complex128)
    for component in range(3):
        result += _factorized_pair(
            left[:, :, component],
            right[:, :, component],
            weights,
            triangle_factor,
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


def analytic_bare_matrices(
    mol: Any,
    *,
    hbar: float,
    mass: float,
) -> OneElectronMatrices:
    """Return analytic field-free overlap, kinetic, and nuclear matrices."""

    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=np.complex128)
    kinetic = (hbar**2 / mass) * np.asarray(
        mol.intor_symmetric("int1e_kin"),
        dtype=np.complex128,
    )
    potential = np.asarray(mol.intor_symmetric("int1e_nuc"), dtype=np.complex128)
    return OneElectronMatrices(
        overlap=overlap,
        kinetic=kinetic,
        potential=potential,
        mechanical=kinetic + potential,
    )


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


def _finish_diagnostics(
    theta: np.ndarray,
    bare: OneElectronMatrices,
    exact_barred: OneElectronMatrices,
    exact_sectors: KineticSectors,
    no_flux_sectors: KineticSectors,
) -> WilsonDiagnostics:
    nao = theta.shape[0]
    p0_sectors = KineticSectors(
        pp=np.array(bare.kinetic, copy=True),
        p_c=_zero_matrix(nao),
        c_p=_zero_matrix(nao),
        c_c=_zero_matrix(nao),
    )
    form_factor_sectors = KineticSectors(
        pp=np.array(exact_sectors.pp, copy=True),
        p_c=_zero_matrix(nao),
        c_p=_zero_matrix(nao),
        c_c=_zero_matrix(nao),
    )
    form_factor_barred = _make_matrices(
        exact_barred.overlap,
        form_factor_sectors.total,
        exact_barred.potential,
    )
    anchored_vector_barred = _make_matrices(
        bare.overlap,
        no_flux_sectors.total,
        bare.potential,
    )
    return WilsonDiagnostics(
        p0=_diagnostic_calculation(theta, bare, p0_sectors),
        form_factor_only=_diagnostic_calculation(
            theta,
            form_factor_barred,
            form_factor_sectors,
        ),
        anchored_vector_only=_diagnostic_calculation(
            theta,
            anchored_vector_barred,
            no_flux_sectors,
        ),
        full_exact=_diagnostic_calculation(
            theta,
            exact_barred,
            exact_sectors,
        ),
    )


def _diagnostic_calculation(
    theta: np.ndarray,
    barred: OneElectronMatrices,
    kinetic_sectors: KineticSectors,
) -> DiagnosticCalculation:
    return DiagnosticCalculation(
        barred=barred,
        lower=_dress_matrices(theta, barred),
        kinetic_sectors=kinetic_sectors,
    )


def _dress_matrices(
    theta: np.ndarray,
    barred: OneElectronMatrices,
) -> OneElectronMatrices:
    return OneElectronMatrices(
        overlap=theta * barred.overlap,
        kinetic=theta * barred.kinetic,
        potential=theta * barred.potential,
        mechanical=theta * barred.mechanical,
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


def _zero_matrix(nao: int) -> np.ndarray:
    return np.zeros((nao, nao), dtype=np.complex128)


def _vector3(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return array


def _cartesian_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    return array
