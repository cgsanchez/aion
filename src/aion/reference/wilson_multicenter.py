"""Angular-channel and center-loop utilities for Wilson qualification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np

from aion.gauge import AOAnchors, UniformMagneticGauge

from .wilson_grid import (
    OneElectronMatrices,
    analytic_bare_matrices,
    endpoint_link_matrix,
)
from .wilson_measures import GeneralizedSpectrum


@dataclass(frozen=True)
class MetricDiagnostics:
    """Eigenvalue and conditioning diagnostics for one Hermitian metric."""

    eigenvalues: np.ndarray
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float | None
    positive_definite: bool
    floor: float


@dataclass(frozen=True)
class PositivityLossBracket:
    """First sampled bracket in which a metric loses positive definiteness."""

    lower_field: float
    lower_minimum_eigenvalue: float
    upper_field: float
    upper_minimum_eigenvalue: float
    floor: float


@dataclass(frozen=True)
class AnalyticWilsonOverlap:
    """Analytic finite-field overlap from Gaussian AO-pair Fourier integrals."""

    barred: np.ndarray
    lower: np.ndarray
    theta: np.ndarray
    atom_pair_wavevectors: np.ndarray


@dataclass(frozen=True)
class EigenbranchTracking:
    """Continuity-matched eigenvalue branches for a Hermitian matrix path."""

    eigenvalues: np.ndarray
    assignments: np.ndarray
    adjacent_overlaps: np.ndarray
    ambiguous_steps: tuple[int, ...]
    degeneracy_tolerance: float


def build_second_row_diatomic(
    system: Literal["N2", "CO"],
    *,
    bond_length_bohr: float,
    basis: str,
    axis: Literal["x", "y", "z"] = "z",
) -> Any:
    """Build a centered N2 or CO framework along one Cartesian axis."""

    if system not in {"N2", "CO"}:
        raise ValueError("system must be 'N2' or 'CO'")
    if not np.isfinite(bond_length_bohr) or bond_length_bohr <= 0.0:
        raise ValueError("bond_length_bohr must be positive and finite")
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    try:
        from pyscf import gto
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError("PySCF is required for multicenter qualification") from exc

    direction = np.zeros(3)
    direction[{"x": 0, "y": 1, "z": 2}[axis]] = 1.0
    offset = 0.5 * float(bond_length_bohr) * direction
    symbols = ("N", "N") if system == "N2" else ("C", "O")
    return gto.M(
        atom=[(symbols[0], -offset), (symbols[1], offset)],
        basis=basis,
        unit="Bohr",
        spin=0,
        verbose=0,
    )


def diatomic_axial_ao_channels(
    mol: Any,
    *,
    axis: Literal["x", "y", "z"] = "z",
) -> dict[str, np.ndarray]:
    """Classify real spherical AOs by ``|m|`` about a Cartesian bond axis.

    The current classifier covers the s, p, and d labels used by STO-3G and
    cc-pVDZ in this milestone.  Returned integer arrays partition every AO
    into ``sigma``, ``pi``, and, when present, ``delta`` channels.
    """

    if axis != "z":
        raise NotImplementedError(
            "the real-spherical d-shell classifier currently supports axis='z'"
        )
    channels: dict[str, list[int]] = {"sigma": [], "pi": [], "delta": []}
    for index, label in enumerate(mol.ao_labels(fmt=False)):
        shell = str(label[2])
        component = str(label[3])
        angular = _shell_angular_momentum(shell)
        channel = _cartesian_axial_channel(angular, component, axis=axis)
        channels[channel].append(index)
    result = {
        name: np.asarray(indices, dtype=int)
        for name, indices in channels.items()
        if indices
    }
    covered = np.sort(np.concatenate(tuple(result.values())))
    if not np.array_equal(covered, np.arange(mol.nao_nr())):
        raise RuntimeError("axial-channel classification did not partition the AOs")
    return result


def equilateral_h3_geometry(*, side_bohr: float = 1.65) -> np.ndarray:
    """Return a centered equilateral H3 triangle in the xy plane."""

    if not np.isfinite(side_bohr) or side_bohr <= 0.0:
        raise ValueError("side_bohr must be positive and finite")
    side = float(side_bohr)
    return np.array(
        [
            [-0.5 * side, -np.sqrt(3.0) * side / 6.0, 0.0],
            [0.5 * side, -np.sqrt(3.0) * side / 6.0, 0.0],
            [0.0, np.sqrt(3.0) * side / 3.0, 0.0],
        ]
    )


def distorted_h3_geometry() -> np.ndarray:
    """Return a centered scalene H3 triangle in the xy plane."""

    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.75, 0.0, 0.0],
            [0.38, 1.22, 0.0],
        ]
    )
    return coordinates - np.mean(coordinates, axis=0)


def build_h3_framework(
    geometry: np.ndarray,
    *,
    basis: str = "sto-3g",
) -> Any:
    """Build the physical two-electron H3+ nuclear framework."""

    coordinates = _cartesian_rows(geometry, name="geometry")
    if coordinates.shape != (3, 3):
        raise ValueError("H3 geometry must have shape (3, 3)")
    try:
        from pyscf import gto
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError("PySCF is required for multicenter qualification") from exc
    return gto.M(
        atom=[("H", coordinate) for coordinate in coordinates],
        basis=basis,
        unit="Bohr",
        charge=1,
        spin=0,
        verbose=0,
    )


def oriented_polygon_area_vector(
    vertices: np.ndarray,
    order: Iterable[int] | None = None,
) -> np.ndarray:
    """Return the translation-invariant oriented area vector of a polygon."""

    points = _ordered_vertices(vertices, order)
    origin = points[0]
    shifted = points - origin
    return 0.5 * np.sum(np.cross(shifted, np.roll(shifted, -1, axis=0)), axis=0)


def uniform_loop_flux(
    vertices: np.ndarray,
    magnetic_field: np.ndarray,
    order: Iterable[int] | None = None,
) -> float:
    """Return ``B`` dotted into the oriented polygon area vector."""

    field = _vector3(magnetic_field, name="magnetic_field")
    return float(np.dot(field, oriented_polygon_area_vector(vertices, order)))


def uniform_loop_link_product(
    gauge: UniformMagneticGauge,
    vertices: np.ndarray,
    order: Iterable[int] | None = None,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> complex:
    """Return the ordered product of straight endpoint links around a loop."""

    if not isinstance(gauge, UniformMagneticGauge):
        raise TypeError("gauge must be a UniformMagneticGauge")
    if not np.isfinite(charge):
        raise ValueError("charge must be finite")
    if not np.isfinite(hbar) or hbar <= 0.0:
        raise ValueError("hbar must be positive and finite")
    points = _ordered_vertices(vertices, order)
    ends = np.roll(points, -1, axis=0)
    integrals = gauge.straight_line_integrals(points, ends)
    return complex(np.exp((1j * charge / hbar) * np.sum(integrals)))


def p0_one_electron_matrices(
    mol: Any,
    gauge: UniformMagneticGauge,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
    mass: float = 1.0,
) -> OneElectronMatrices:
    """Return analytic P0 lower matrices with exact endpoint links."""

    anchors = AOAnchors.from_mol(mol)
    ao_anchors = np.asarray(anchors.atom_coords[anchors.ao_to_atom], dtype=float)
    theta = endpoint_link_matrix(gauge, ao_anchors, charge=charge, hbar=hbar)
    bare = analytic_bare_matrices(mol, hbar=hbar, mass=mass)
    return _make_matrices(
        theta * bare.overlap,
        theta * bare.kinetic,
        theta * bare.potential,
    )


def analytic_b1_one_electron_matrices(
    mol: Any,
    gauge: UniformMagneticGauge,
) -> OneElectronMatrices:
    """Return ``B1[1e]`` from independent analytic GIAO derivatives.

    The internal first derivatives use the convention-matched libcint kernels
    accepted in Milestone 5.  They are contracted with the physical Cartesian
    field while the endpoint link is retained exactly.  Atomic-unit electron
    conventions ``q=-1`` and ``hbar=m=1`` are therefore part of this helper's
    contract.
    """

    if not isinstance(gauge, UniformMagneticGauge):
        raise TypeError("gauge must be a UniformMagneticGauge")
    from .wilson_approximations import pyscf_giao_one_electron_derivatives

    anchors = AOAnchors.from_mol(mol)
    ao_anchors = np.asarray(anchors.atom_coords[anchors.ao_to_atom], dtype=float)
    theta = endpoint_link_matrix(gauge, ao_anchors, charge=-1.0, hbar=1.0)
    bare = analytic_bare_matrices(mol, hbar=1.0, mass=1.0)
    derivatives = pyscf_giao_one_electron_derivatives(mol).barred
    field = np.asarray(gauge.magnetic_field, dtype=float)
    barred_overlap = bare.overlap + np.tensordot(
        field,
        derivatives.overlap,
        axes=(0, 0),
    )
    barred_kinetic = bare.kinetic + np.tensordot(
        field,
        derivatives.kinetic,
        axes=(0, 0),
    )
    barred_potential = bare.potential + np.tensordot(
        field,
        derivatives.potential,
        axes=(0, 0),
    )
    return _make_matrices(
        theta * barred_overlap,
        theta * barred_kinetic,
        theta * barred_potential,
    )


def analytic_uniform_wilson_overlap(
    mol: Any,
    gauge: UniformMagneticGauge,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> AnalyticWilsonOverlap:
    r"""Evaluate the exact uniform-field Wilson overlap analytically.

    For AO anchors ``R_i`` and ``R_j``, the endpoint-removed triangle factor
    is the plane wave

    ``F_ij(r) = exp(i k_ij . (r - (R_i + R_j)/2))``

    with ``k_ij = q (R_i - R_j) x B / (2 hbar)``.  PySCF/libcint evaluates
    the Fourier transform of each contracted Gaussian AO-pair product
    analytically.  This route shares neither the real-space grid nor its AO
    samples with :func:`evaluate_uniform_magnetic_matrices`.
    """

    if not isinstance(gauge, UniformMagneticGauge):
        raise TypeError("gauge must be a UniformMagneticGauge")
    if not np.isfinite(charge):
        raise ValueError("charge must be finite")
    if not np.isfinite(hbar) or hbar <= 0.0:
        raise ValueError("hbar must be positive and finite")
    try:
        from pyscf.gto.ft_ao import ft_aopair
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError("PySCF is required for analytic Gaussian overlaps") from exc

    anchors = AOAnchors.from_mol(mol)
    atom_coordinates = np.asarray(anchors.atom_coords, dtype=float)
    mapping = np.asarray(anchors.ao_to_atom, dtype=int)
    ao_anchors = atom_coordinates[mapping]
    theta = endpoint_link_matrix(gauge, ao_anchors, charge=charge, hbar=hbar)
    barred = np.zeros((anchors.nao, anchors.nao), dtype=np.complex128)
    wavevectors = np.zeros(
        (atom_coordinates.shape[0], atom_coordinates.shape[0], 3),
        dtype=float,
    )

    pairs = []
    fourier_arguments = []
    for atom_i, center_i in enumerate(atom_coordinates):
        for atom_j, center_j in enumerate(atom_coordinates):
            midpoint = 0.5 * (center_i + center_j)
            wavevector = (
                float(charge)
                / (2.0 * float(hbar))
                * np.cross(center_i - center_j, gauge.magnetic_field)
            )
            wavevectors[atom_i, atom_j] = wavevector
            pairs.append((atom_i, atom_j, midpoint, wavevector))
            fourier_arguments.append(-wavevector)

    pair_fourier = np.asarray(
        ft_aopair(mol, np.asarray(fourier_arguments)),
        dtype=np.complex128,
    )
    for index, (atom_i, atom_j, midpoint, wavevector) in enumerate(pairs):
        rows = np.flatnonzero(mapping == atom_i)
        columns = np.flatnonzero(mapping == atom_j)
        midpoint_phase = np.exp(-1j * np.dot(wavevector, midpoint))
        barred[np.ix_(rows, columns)] = (
            midpoint_phase * pair_fourier[index][np.ix_(rows, columns)]
        )

    return AnalyticWilsonOverlap(
        barred=barred,
        lower=theta * barred,
        theta=theta,
        atom_pair_wavevectors=wavevectors,
    )


def affine_gauge_endpoint_congruence(
    source: UniformMagneticGauge,
    target: UniformMagneticGauge,
    ao_anchors: np.ndarray,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> np.ndarray:
    """Return the diagonal AO congruence between two affine gauges of one B."""

    if not isinstance(source, UniformMagneticGauge) or not isinstance(
        target, UniformMagneticGauge
    ):
        raise TypeError("source and target must be UniformMagneticGauge objects")
    if not np.allclose(
        source.magnetic_field,
        target.magnetic_field,
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError("source and target must represent the same magnetic field")
    if not np.isfinite(charge):
        raise ValueError("charge must be finite")
    if not np.isfinite(hbar) or hbar <= 0.0:
        raise ValueError("hbar must be positive and finite")
    anchors = _cartesian_rows(ao_anchors, name="ao_anchors")
    quadratic = target.affine_matrix - source.affine_matrix
    scale = max(1.0, float(np.linalg.norm(quadratic, ord="fro")))
    if np.linalg.norm(quadratic - quadratic.T, ord="fro") > (
        128.0 * np.finfo(float).eps * scale
    ):
        raise ValueError("the affine gauge difference is not a scalar gradient")
    linear = (
        -target.affine_matrix @ target.origin
        + source.affine_matrix @ source.origin
    )
    gauge_function = (
        0.5 * np.einsum("ix,xy,iy->i", anchors, quadratic, anchors)
        + anchors @ linear
    )
    return np.diag(np.exp((1j * charge / hbar) * gauge_function))


def metric_diagnostics(
    metric: np.ndarray,
    *,
    floor: float = 0.0,
) -> MetricDiagnostics:
    """Return Hermitian metric eigenvalues without regularizing indefiniteness."""

    declared_floor = float(floor)
    if not np.isfinite(declared_floor) or declared_floor < 0.0:
        raise ValueError("floor must be nonnegative and finite")
    matrix = np.asarray(metric, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("metric must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("metric must be finite")
    scale = max(1.0, float(np.linalg.norm(matrix, ord="fro")))
    tolerance = 512.0 * np.finfo(float).eps * matrix.shape[0] * scale
    residual = float(np.linalg.norm(matrix - matrix.conj().T, ord="fro"))
    if residual > tolerance:
        raise ValueError("metric must be Hermitian at its floating-point scale")
    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    positive = bool(eigenvalues[0] > declared_floor)
    condition = (
        float(eigenvalues[-1] / eigenvalues[0]) if positive else None
    )
    return MetricDiagnostics(
        eigenvalues=eigenvalues,
        minimum_eigenvalue=float(eigenvalues[0]),
        maximum_eigenvalue=float(eigenvalues[-1]),
        condition_number=condition,
        positive_definite=positive,
        floor=declared_floor,
    )


def track_hermitian_eigenbranches(
    matrices: Iterable[np.ndarray],
    *,
    degeneracy_tolerance: float = 1.0e-8,
) -> EigenbranchTracking:
    """Track eigenvectors by maximum adjacent overlap along a matrix path.

    The returned branches are a continuity diagnostic, not unique state labels
    inside a degenerate subspace.  ``ambiguous_steps`` identifies steps for
    which either adjacent spectrum contains a gap no larger than the declared
    degeneracy tolerance.
    """

    tolerance = float(degeneracy_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("degeneracy_tolerance must be nonnegative and finite")
    path = tuple(np.asarray(matrix, dtype=np.complex128) for matrix in matrices)
    if not path:
        raise ValueError("at least one matrix is required")
    shape = path[0].shape
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] == 0:
        raise ValueError("matrices must be nonempty and square")
    if any(matrix.shape != shape for matrix in path):
        raise ValueError("all matrices must have the same shape")

    eigenpairs = []
    for matrix in path:
        if not np.all(np.isfinite(matrix)):
            raise ValueError("matrices must be finite")
        scale = max(1.0, float(np.linalg.norm(matrix, ord="fro")))
        residual = float(np.linalg.norm(matrix - matrix.conj().T, ord="fro"))
        if residual > 512.0 * np.finfo(float).eps * shape[0] * scale:
            raise ValueError("matrices must be Hermitian at floating-point scale")
        eigenpairs.append(np.linalg.eigh(0.5 * (matrix + matrix.conj().T)))
    return _track_eigenpairs(
        tuple(eigenpairs),
        np.eye(shape[0]),
        degeneracy_tolerance=tolerance,
    )


def track_generalized_eigenbranches(
    spectra: Iterable[GeneralizedSpectrum],
    comparison_metric: np.ndarray,
    *,
    degeneracy_tolerance: float = 1.0e-8,
) -> EigenbranchTracking:
    """Track generalized eigenvectors using one fixed positive metric.

    ``comparison_metric`` is normally the bare overlap ``S0``. It provides a
    common coefficient-space inner product even though every spectrum was
    solved with its own field-dependent metric. Branches remain nonunique
    inside subspaces flagged by ``ambiguous_steps``.
    """

    path = tuple(spectra)
    if not path:
        raise ValueError("at least one generalized spectrum is required")
    if any(not isinstance(spectrum, GeneralizedSpectrum) for spectrum in path):
        raise TypeError("spectra must contain GeneralizedSpectrum objects")
    eigenpairs = tuple(
        (
            np.asarray(spectrum.eigenvalues, dtype=float),
            np.asarray(spectrum.eigenvectors, dtype=np.complex128),
        )
        for spectrum in path
    )
    return _track_eigenpairs(
        eigenpairs,
        comparison_metric,
        degeneracy_tolerance=degeneracy_tolerance,
    )


def _track_eigenpairs(
    eigenpairs: tuple[tuple[np.ndarray, np.ndarray], ...],
    comparison_metric: np.ndarray,
    *,
    degeneracy_tolerance: float,
) -> EigenbranchTracking:
    tolerance = float(degeneracy_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("degeneracy_tolerance must be nonnegative and finite")
    dimension = eigenpairs[0][0].size
    metric = np.asarray(comparison_metric, dtype=np.complex128)
    if metric.shape != (dimension, dimension) or not np.all(np.isfinite(metric)):
        raise ValueError("comparison_metric must be finite with shape (n, n)")
    metric_scale = max(1.0, float(np.linalg.norm(metric, ord="fro")))
    metric_residual = float(np.linalg.norm(metric - metric.conj().T, ord="fro"))
    if metric_residual > 512.0 * np.finfo(float).eps * dimension * metric_scale:
        raise ValueError("comparison_metric must be Hermitian")
    metric = 0.5 * (metric + metric.conj().T)
    if np.linalg.eigvalsh(metric)[0] <= 0.0:
        raise ValueError("comparison_metric must be positive definite")

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - project dependency
        raise ImportError("SciPy is required for eigenbranch assignment") from exc

    branch_values = []
    assignments = []
    adjacent_overlaps = []
    ambiguous_steps = []
    previous_values: np.ndarray | None = None
    previous_vectors: np.ndarray | None = None
    for step, (values, vectors) in enumerate(eigenpairs):
        if values.shape != (dimension,) or vectors.shape != (dimension, dimension):
            raise ValueError("every eigenpair must have shapes (n,) and (n, n)")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
            raise ValueError("eigenpairs must be finite")
        if previous_vectors is None:
            branch_values.append(values)
            assignments.append(np.arange(dimension, dtype=int))
            adjacent_overlaps.append(np.ones(dimension))
            previous_values = values
            previous_vectors = vectors
            continue

        previous_norms = _column_metric_norms(previous_vectors, metric)
        current_norms = _column_metric_norms(vectors, metric)
        raw_overlap = previous_vectors.conj().T @ metric @ vectors
        overlap_matrix = np.abs(raw_overlap) / np.outer(
            previous_norms,
            current_norms,
        )
        rows, columns = linear_sum_assignment(-overlap_matrix)
        assignment = np.empty(dimension, dtype=int)
        assignment[rows] = columns
        reordered_values = values[assignment]
        reordered_vectors = vectors[:, assignment]
        overlaps = np.diag(
            previous_vectors.conj().T @ metric @ reordered_vectors
        )
        overlap_magnitudes = np.abs(overlaps) / (
            previous_norms * current_norms[assignment]
        )
        nonzero = np.abs(overlaps) > np.finfo(float).eps
        reordered_vectors[:, nonzero] *= np.exp(
            -1j * np.angle(overlaps[nonzero])
        )

        assert previous_values is not None
        previous_gaps = np.abs(np.diff(np.sort(previous_values)))
        current_gaps = np.abs(np.diff(values))
        if (
            np.any(previous_gaps <= tolerance)
            or np.any(current_gaps <= tolerance)
        ):
            ambiguous_steps.append(step)

        branch_values.append(reordered_values)
        assignments.append(assignment)
        adjacent_overlaps.append(overlap_magnitudes)
        previous_values = reordered_values
        previous_vectors = reordered_vectors

    return EigenbranchTracking(
        eigenvalues=np.asarray(branch_values),
        assignments=np.asarray(assignments),
        adjacent_overlaps=np.asarray(adjacent_overlaps),
        ambiguous_steps=tuple(ambiguous_steps),
        degeneracy_tolerance=tolerance,
    )


def _column_metric_norms(vectors: np.ndarray, metric: np.ndarray) -> np.ndarray:
    norms_squared = np.real(
        np.einsum("ij,ij->j", vectors.conj(), metric @ vectors)
    )
    scale = max(1.0, float(np.max(np.abs(norms_squared), initial=0.0)))
    floor = 512.0 * np.finfo(float).eps * vectors.shape[0] * scale
    if np.any(norms_squared <= floor):
        raise ValueError("eigenvectors must have positive comparison-metric norm")
    return np.sqrt(norms_squared)


def first_positivity_loss_bracket(
    fields: Iterable[float],
    minimum_eigenvalues: Iterable[float],
    *,
    floor: float = 0.0,
) -> PositivityLossBracket | None:
    """Return the first ascending-field transition from positive to nonpositive."""

    declared_floor = float(floor)
    if not np.isfinite(declared_floor) or declared_floor < 0.0:
        raise ValueError("floor must be nonnegative and finite")
    pairs = sorted(
        (float(field), float(value))
        for field, value in zip(fields, minimum_eigenvalues, strict=True)
    )
    if not pairs:
        raise ValueError("at least one field/eigenvalue pair is required")
    if any(
        not np.isfinite(field) or not np.isfinite(value) or field < 0.0
        for field, value in pairs
    ):
        raise ValueError(
            "fields and eigenvalues must be finite with fields nonnegative"
        )
    for lower, upper in zip(pairs[:-1], pairs[1:], strict=True):
        if lower[1] > declared_floor and upper[1] <= declared_floor:
            return PositivityLossBracket(
                lower_field=lower[0],
                lower_minimum_eigenvalue=lower[1],
                upper_field=upper[0],
                upper_minimum_eigenvalue=upper[1],
                floor=declared_floor,
            )
    return None


def _shell_angular_momentum(shell: str) -> int:
    for symbol, angular in (("s", 0), ("p", 1), ("d", 2)):
        if symbol in shell:
            return angular
    raise NotImplementedError(f"unsupported AO shell label {shell!r}")


def _cartesian_axial_channel(
    angular: int,
    component: str,
    *,
    axis: str,
) -> str:
    if angular == 0:
        return "sigma"
    if angular == 1:
        return "sigma" if component == axis else "pi"
    if angular != 2:
        raise NotImplementedError("only s, p, and d channels are supported")
    axial_square = f"{axis}^2"
    if component in {axial_square, "z^2" if axis == "z" else axial_square}:
        return "sigma"
    if axis in component and component not in {"x2-y2"}:
        return "pi"
    return "delta"


def _ordered_vertices(
    vertices: np.ndarray,
    order: Iterable[int] | None,
) -> np.ndarray:
    points = _cartesian_rows(vertices, name="vertices")
    if points.shape[0] < 3:
        raise ValueError("a loop requires at least three vertices")
    if order is None:
        indices = np.arange(points.shape[0], dtype=int)
    else:
        indices = np.asarray(tuple(order), dtype=int)
        if indices.shape != (points.shape[0],):
            raise ValueError("order must contain one entry per vertex")
        if not np.array_equal(np.sort(indices), np.arange(points.shape[0])):
            raise ValueError("order must be a permutation of vertex indices")
    return points[indices]


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


def _cartesian_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape (n, 3)")
    return array


def _vector3(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape (3,)")
    return array
