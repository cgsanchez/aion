"""Numerically explicit measures for Wilson/P0 matrix qualification.

Every normalization that can hide a small denominator requires a caller-
supplied floor.  Campaign code can therefore derive the floor from grid
refinement instead of silently inheriting a physical significance threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class PhaseSpreadMoments:
    """Raw numerator and denominator of the phase-spread measure."""

    numerator: np.ndarray
    denominator: np.ndarray


@dataclass(frozen=True)
class PhaseSpreadMeasure:
    """Evaluated ``epsilon_F`` with an explicit zero-overlap policy."""

    values: np.ndarray
    defined: np.ndarray
    numerator: np.ndarray
    denominator: np.ndarray
    denominator_floor: float


@dataclass(frozen=True)
class ElementwiseMatrixError:
    """Elementwise exact-minus-reference errors and normalizations."""

    difference: np.ndarray
    absolute: np.ndarray
    relative: np.ndarray
    relative_defined: np.ndarray
    diagonal_scaled: np.ndarray
    diagonal_scale: np.ndarray
    floor: float


@dataclass(frozen=True)
class AtomPairBlockError:
    """Rotation-stable measures for one atom-pair AO block."""

    atom_a: int
    atom_b: int
    absolute_frobenius: float
    relative_frobenius: float
    reference_frobenius: float
    difference_singular_values: np.ndarray
    exact_singular_values: np.ndarray
    reference_singular_values: np.ndarray
    floor: float


@dataclass(frozen=True)
class MatrixNumericalFloor:
    """Frobenius floor inferred from refinement and an optional oracle."""

    value: float
    refinement_change: float
    analytic_error: float | None


@dataclass(frozen=True)
class GeneralizedSpectrum:
    """Hermitian generalized eigensystem with metric diagnostics."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    metric_eigenvalues: np.ndarray
    metric_condition_number: float
    metric_floor: float


@dataclass(frozen=True)
class SpectralSubspaceComparison:
    """Principal-angle comparison of equal-dimensional spectral subspaces."""

    cosines: np.ndarray
    principal_angles: np.ndarray
    largest_principal_angle: float
    chordal_distance: float
    comparison_metric_floor: float


class MetricNotPositiveDefiniteError(np.linalg.LinAlgError):
    """Raised when a generalized spectrum is undefined at the declared floor."""


def phase_spread(
    moments: PhaseSpreadMoments,
    *,
    denominator_floor: float,
) -> PhaseSpreadMeasure:
    """Evaluate the phase-spread measure with undefined entries set to NaN.

    An entry is defined only when its absolute-overlap denominator is strictly
    greater than ``denominator_floor``.  The floor is mandatory so a report
    cannot silently turn a zero-overlap policy into a physical threshold.
    """

    floor = _nonnegative_floor(denominator_floor, name="denominator_floor")
    numerator = np.asarray(moments.numerator, dtype=float)
    denominator = np.asarray(moments.denominator, dtype=float)
    if numerator.shape != denominator.shape:
        raise ValueError("phase-spread numerator and denominator shapes must match")
    if not np.all(np.isfinite(numerator)) or not np.all(np.isfinite(denominator)):
        raise ValueError("phase-spread moments must be finite")

    numerator = _clip_quadrature_nonnegative(numerator, name="numerator")
    denominator = _clip_quadrature_nonnegative(denominator, name="denominator")
    defined = denominator > floor
    values = np.full(denominator.shape, np.nan, dtype=float)
    values[defined] = np.sqrt(numerator[defined] / denominator[defined])
    return PhaseSpreadMeasure(
        values=values,
        defined=defined,
        numerator=numerator,
        denominator=denominator,
        denominator_floor=floor,
    )


def elementwise_matrix_error(
    exact: np.ndarray,
    reference: np.ndarray,
    *,
    floor: float,
) -> ElementwiseMatrixError:
    """Return absolute, relative, and diagonal-scaled elementwise errors.

    Relative errors are NaN where ``abs(reference) <= floor``.  The
    diagonal-scaled denominator is
    ``max(sqrt(abs(X_ii X_jj)), floor)`` and is therefore always defined.
    """

    declared_floor = _positive_floor(floor, name="floor")
    exact_matrix = _square_matrix(exact, name="exact")
    reference_matrix = _square_matrix(reference, name="reference")
    if exact_matrix.shape != reference_matrix.shape:
        raise ValueError("exact and reference matrix shapes must match")

    difference = exact_matrix - reference_matrix
    absolute = np.abs(difference)
    reference_absolute = np.abs(reference_matrix)
    relative_defined = reference_absolute > declared_floor
    relative = np.full(reference_matrix.shape, np.nan, dtype=float)
    relative[relative_defined] = (
        absolute[relative_defined] / reference_absolute[relative_defined]
    )
    diagonal = np.abs(np.diag(reference_matrix))
    diagonal_scale = np.sqrt(diagonal[:, None] * diagonal[None, :])
    diagonal_scaled = absolute / np.maximum(diagonal_scale, declared_floor)
    return ElementwiseMatrixError(
        difference=difference,
        absolute=absolute,
        relative=relative,
        relative_defined=relative_defined,
        diagonal_scaled=diagonal_scaled,
        diagonal_scale=diagonal_scale,
        floor=declared_floor,
    )


def atom_pair_block_error(
    exact: np.ndarray,
    reference: np.ndarray,
    ao_to_atom: np.ndarray,
    atom_a: int,
    atom_b: int,
    *,
    floor: float,
) -> AtomPairBlockError:
    """Return Frobenius and singular-value measures for an atom-pair block."""

    declared_floor = _positive_floor(floor, name="floor")
    exact_matrix = _square_matrix(exact, name="exact")
    reference_matrix = _square_matrix(reference, name="reference")
    if exact_matrix.shape != reference_matrix.shape:
        raise ValueError("exact and reference matrix shapes must match")
    mapping = np.asarray(ao_to_atom, dtype=int)
    if mapping.shape != (exact_matrix.shape[0],):
        raise ValueError("ao_to_atom must have shape (nao,)")
    if np.any(mapping < 0):
        raise ValueError("ao_to_atom must contain nonnegative indices")
    if not isinstance(atom_a, (int, np.integer)) or not isinstance(
        atom_b, (int, np.integer)
    ):
        raise TypeError("atom indices must be integers")

    rows = np.flatnonzero(mapping == int(atom_a))
    columns = np.flatnonzero(mapping == int(atom_b))
    if rows.size == 0 or columns.size == 0:
        raise ValueError("each requested atom must own at least one AO")
    index = np.ix_(rows, columns)
    exact_block = exact_matrix[index]
    reference_block = reference_matrix[index]
    difference_block = exact_block - reference_block
    absolute = float(np.linalg.norm(difference_block, ord="fro"))
    reference_norm = float(np.linalg.norm(reference_block, ord="fro"))
    return AtomPairBlockError(
        atom_a=int(atom_a),
        atom_b=int(atom_b),
        absolute_frobenius=absolute,
        relative_frobenius=absolute / max(reference_norm, declared_floor),
        reference_frobenius=reference_norm,
        difference_singular_values=scipy.linalg.svdvals(
            difference_block,
            check_finite=False,
        ),
        exact_singular_values=scipy.linalg.svdvals(
            exact_block,
            check_finite=False,
        ),
        reference_singular_values=scipy.linalg.svdvals(
            reference_block,
            check_finite=False,
        ),
        floor=declared_floor,
    )


def estimate_frobenius_floor(
    finer: np.ndarray,
    finest: np.ndarray,
    *,
    analytic_reference: np.ndarray | None = None,
) -> MatrixNumericalFloor:
    """Estimate a matrix floor from two refinements and an optional oracle.

    The reported floor is the maximum of the two-finest-grid change and, when
    supplied, the finest-grid error against an analytic reference.
    """

    finer_matrix = _square_matrix(finer, name="finer")
    finest_matrix = _square_matrix(finest, name="finest")
    if finer_matrix.shape != finest_matrix.shape:
        raise ValueError("finer and finest matrix shapes must match")
    refinement_change = float(np.linalg.norm(finest_matrix - finer_matrix, ord="fro"))
    analytic_error = None
    if analytic_reference is not None:
        analytic_matrix = _square_matrix(analytic_reference, name="analytic_reference")
        if analytic_matrix.shape != finest_matrix.shape:
            raise ValueError("analytic reference shape must match grid matrices")
        analytic_error = float(
            np.linalg.norm(finest_matrix - analytic_matrix, ord="fro")
        )
    value = refinement_change if analytic_error is None else max(
        refinement_change,
        analytic_error,
    )
    return MatrixNumericalFloor(
        value=value,
        refinement_change=refinement_change,
        analytic_error=analytic_error,
    )


def generalized_hermitian_spectrum(
    operator: np.ndarray,
    metric: np.ndarray,
    *,
    metric_floor: float,
) -> GeneralizedSpectrum:
    """Solve ``operator u = metric u epsilon`` when the metric is resolved SPD."""

    floor = _nonnegative_floor(metric_floor, name="metric_floor")
    operator_hermitian = _validated_hermitian(operator, name="operator")
    metric_hermitian = _validated_hermitian(metric, name="metric")
    if operator_hermitian.shape != metric_hermitian.shape:
        raise ValueError("operator and metric shapes must match")

    metric_eigenvalues = scipy.linalg.eigvalsh(
        metric_hermitian,
        check_finite=False,
    )
    if metric_eigenvalues[0] <= floor:
        raise MetricNotPositiveDefiniteError(
            "metric minimum eigenvalue "
            f"{metric_eigenvalues[0]:.16g} is not above floor {floor:.16g}"
        )
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        operator_hermitian,
        metric_hermitian,
        check_finite=False,
    )
    return GeneralizedSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        metric_eigenvalues=metric_eigenvalues,
        metric_condition_number=float(metric_eigenvalues[-1] / metric_eigenvalues[0]),
        metric_floor=floor,
    )


def eigenvalue_clusters(
    eigenvalues: np.ndarray,
    *,
    absolute_gap: float,
    relative_gap: float = 0.0,
) -> tuple[np.ndarray, ...]:
    """Group sorted eigenvalues using an explicitly declared gap policy."""

    values = np.asarray(eigenvalues, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("eigenvalues must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("eigenvalues must be finite")
    if np.any(np.diff(values) < 0.0):
        raise ValueError("eigenvalues must be sorted in nondecreasing order")
    absolute = _nonnegative_floor(absolute_gap, name="absolute_gap")
    relative = _nonnegative_floor(relative_gap, name="relative_gap")

    clusters: list[np.ndarray] = []
    start = 0
    for index, gap in enumerate(np.diff(values), start=1):
        scale = max(1.0, abs(values[index - 1]), abs(values[index]))
        if gap > absolute + relative * scale:
            clusters.append(np.arange(start, index, dtype=int))
            start = index
    clusters.append(np.arange(start, values.size, dtype=int))
    return tuple(clusters)


def compare_spectral_subspaces(
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
    comparison_metric: np.ndarray,
    *,
    metric_floor: float,
) -> SpectralSubspaceComparison:
    """Compare two subspaces through principal angles in one declared metric.

    This operation is invariant under independent unitary rotations of the
    supplied vectors inside either subspace.  Callers comparing exact and P0
    eigenspaces must explicitly choose the common comparison metric.
    """

    floor = _nonnegative_floor(metric_floor, name="metric_floor")
    metric = _validated_hermitian(comparison_metric, name="comparison_metric")
    metric_eigenvalues = scipy.linalg.eigvalsh(metric, check_finite=False)
    if metric_eigenvalues[0] <= floor:
        raise MetricNotPositiveDefiniteError(
            "comparison metric is not positive definite above the declared floor"
        )
    left = _vector_block(left_vectors, metric.shape[0], name="left_vectors")
    right = _vector_block(right_vectors, metric.shape[0], name="right_vectors")
    if left.shape[1] != right.shape[1]:
        raise ValueError("spectral subspaces must have equal dimension")

    left_orthonormal = _metric_orthonormalize(left, metric, name="left_vectors")
    right_orthonormal = _metric_orthonormalize(right, metric, name="right_vectors")
    overlap = left_orthonormal.conj().T @ metric @ right_orthonormal
    cosines = np.clip(
        scipy.linalg.svdvals(overlap, check_finite=False),
        0.0,
        1.0,
    )
    angles = np.arccos(cosines)
    chordal = float(np.sqrt(np.sum(np.maximum(0.0, 1.0 - cosines**2))))
    return SpectralSubspaceComparison(
        cosines=cosines,
        principal_angles=angles,
        largest_principal_angle=float(np.max(angles)),
        chordal_distance=chordal,
        comparison_metric_floor=floor,
    )


def _square_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


def _validated_hermitian(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = _square_matrix(value, name=name)
    scale = max(1.0, float(np.linalg.norm(matrix, ord="fro")))
    tolerance = 512.0 * np.finfo(float).eps * matrix.shape[0] * scale
    residual = float(np.linalg.norm(matrix - matrix.conj().T, ord="fro"))
    if residual > tolerance:
        raise ValueError(
            f"{name} is not Hermitian at its floating-point scale: "
            f"residual {residual:.16g} exceeds {tolerance:.16g}"
        )
    return 0.5 * (matrix + matrix.conj().T)


def _vector_block(value: np.ndarray, nrow: int, *, name: str) -> np.ndarray:
    vectors = np.asarray(value, dtype=np.complex128)
    if vectors.ndim != 2 or vectors.shape[0] != nrow or vectors.shape[1] == 0:
        raise ValueError(f"{name} must have shape ({nrow}, nvector) with nvector > 0")
    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"{name} must be finite")
    return vectors


def _metric_orthonormalize(
    vectors: np.ndarray,
    metric: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    gram = _validated_hermitian(vectors.conj().T @ metric @ vectors, name=f"{name} Gram")
    eigenvalues, eigenvectors = scipy.linalg.eigh(gram, check_finite=False)
    rank_floor = (
        512.0
        * np.finfo(float).eps
        * gram.shape[0]
        * max(1.0, float(eigenvalues[-1]))
    )
    if eigenvalues[0] <= rank_floor:
        raise ValueError(f"{name} does not span a numerically independent subspace")
    inverse_square_root = (
        eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]
    ) @ eigenvectors.conj().T
    return vectors @ inverse_square_root


def _clip_quadrature_nonnegative(value: np.ndarray, *, name: str) -> np.ndarray:
    scale = max(1.0, float(np.max(np.abs(value), initial=0.0)))
    tolerance = 64.0 * np.finfo(float).eps * scale
    if np.any(value < -tolerance):
        raise ValueError(f"phase-spread {name} must be nonnegative")
    return np.maximum(value, 0.0)


def _nonnegative_floor(value: float, *, name: str) -> float:
    floor = float(value)
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return floor


def _positive_floor(value: float, *, name: str) -> float:
    floor = _nonnegative_floor(value, name=name)
    if floor == 0.0:
        raise ValueError(f"{name} must be positive")
    return floor
