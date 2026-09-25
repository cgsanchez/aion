"""Gauge-covariant static model decompositions for exact Wilson pair analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import eigh, subspace_angles

from aion.config import BackendConfig
from aion.electronic_structure.data import (
    OneElectronAOReference,
    PreparedReference,
    immutable_array,
)
from aion.electronic_structure.magnetic_matrices import (
    ExactStaticMagneticOneElectronResult,
    MagneticOneElectronResult,
    OneElectronLowerMatrices,
)
from aion.electronic_structure.pyscf_rks import reconstruct_mean_field


@dataclass(frozen=True, slots=True)
class StaticMagneticModelSet:
    """P0, diagnostic sector deletions, and exact matrices in one representation."""

    p0: OneElectronLowerMatrices
    form_factor_only: OneElectronLowerMatrices
    anchored_vector_only: OneElectronLowerMatrices
    exact: OneElectronLowerMatrices


@dataclass(frozen=True, slots=True)
class StaticMagneticDiagnosticModels:
    """Static models as endpoint-removed amplitudes and endpoint-dressed matrices.

    The two middle models are diagnostic sector deletions. They are not
    asserted to define gauge-covariant action-level approximations for general
    time-dependent dynamics.
    """

    barred: StaticMagneticModelSet
    lower: StaticMagneticModelSet


@dataclass(frozen=True, slots=True)
class StaticMagneticFirstOrderModelSet:
    """Named action-model matrices in the static magnetic Maxwell sector.

    ``complete_first`` is equal to ``full_b1`` in this sector because the
    electric internal increment vanishes.  The separate field is retained so
    that the sector-reduction identity is represented explicitly in the API.
    """

    p0: OneElectronLowerMatrices
    geometric_b1: OneElectronLowerMatrices
    full_b1: OneElectronLowerMatrices
    complete_first: OneElectronLowerMatrices
    exact: OneElectronLowerMatrices


@dataclass(frozen=True, slots=True)
class StaticMagneticFirstOrderModels:
    """Endpoint-removed and endpoint-dressed first-order model hierarchy."""

    barred: StaticMagneticFirstOrderModelSet
    lower: StaticMagneticFirstOrderModelSet


@dataclass(frozen=True, slots=True)
class MatrixPartitionChanges:
    """Stable changes in onsite-diagonal, same-anchor, and intersite partitions."""

    labels: tuple[str, ...]
    absolute_frobenius: np.ndarray
    relative_frobenius: np.ndarray
    maximum_absolute_element: np.ndarray

    def __post_init__(self) -> None:
        if self.labels != (
            "onsite_diagonal",
            "same_anchor_offdiagonal",
            "intersite",
        ):
            raise ValueError("matrix partition labels are not canonical")
        for name in (
            "absolute_frobenius",
            "relative_frobenius",
            "maximum_absolute_element",
        ):
            value = immutable_array(
                getattr(self, name), dtype=np.float64, ndim=1, name=name.replace("_", " ")
            )
            if value.shape != (3,):
                raise ValueError("matrix partition diagnostics must have three entries")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class AngularChannelChange:
    """One atom-pair and angular-momentum block change."""

    bra_atom: int
    ket_atom: int
    bra_angular_momentum: int
    ket_angular_momentum: int
    classification: str
    absolute_frobenius: float
    relative_frobenius: float
    maximum_absolute_element: float
    singular_values: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "bra_atom",
            "ket_atom",
            "bra_angular_momentum",
            "ket_angular_momentum",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.classification not in ("onsite", "same_anchor_cross_l", "intersite"):
            raise ValueError("invalid angular-channel classification")
        for name in (
            "absolute_frobenius",
            "relative_frobenius",
            "maximum_absolute_element",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite")
            object.__setattr__(self, name, value)
        singular_values = immutable_array(
            self.singular_values,
            dtype=np.float64,
            ndim=1,
            name="angular-channel singular values",
        )
        object.__setattr__(self, "singular_values", singular_values)


@dataclass(frozen=True, slots=True)
class GeneralizedSpectralComparison:
    """Generalized eigenvalue shifts and cumulative coefficient-subspace angles."""

    reference_eigenvalues: np.ndarray
    candidate_eigenvalues: np.ndarray
    eigenvalue_shifts: np.ndarray
    cumulative_subspace_angles_rad: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "reference_eigenvalues",
            "candidate_eigenvalues",
            "eigenvalue_shifts",
            "cumulative_subspace_angles_rad",
        ):
            value = immutable_array(
                getattr(self, name), dtype=np.float64, ndim=1, name=name.replace("_", " ")
            )
            object.__setattr__(self, name, value)
        size = self.reference_eigenvalues.size
        if (
            size == 0
            or self.candidate_eigenvalues.shape != (size,)
            or self.eigenvalue_shifts.shape != (size,)
            or self.cumulative_subspace_angles_rad.shape != (max(0, size - 1),)
        ):
            raise ValueError("generalized spectral comparison dimensions disagree")


@dataclass(frozen=True, slots=True)
class MetricSpectrum:
    """Hermitian metric eigenvalues and positivity diagnostics."""

    eigenvalues: np.ndarray
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    hermiticity_residual: float
    positive: bool

    def __post_init__(self) -> None:
        eigenvalues = immutable_array(
            self.eigenvalues,
            dtype=np.float64,
            ndim=1,
            name="metric eigenvalues",
        )
        if eigenvalues.size == 0:
            raise ValueError("metric spectrum must not be empty")
        object.__setattr__(self, "eigenvalues", eigenvalues)


def static_magnetic_diagnostic_models(
    result: MagneticOneElectronResult | ExactStaticMagneticOneElectronResult,
) -> StaticMagneticDiagnosticModels:
    """Build the four exact-versus-P0 static diagnostic models.

    P0 retains only the endpoint link. ``form_factor_only`` retains the exact
    triangle factor while deleting anchored vectors. ``anchored_vector_only``
    retains the exact anchored-vector terms with the triangle factor set to
    one. ``exact`` retains both effects.
    """

    if not isinstance(result, MagneticOneElectronResult | ExactStaticMagneticOneElectronResult):
        raise TypeError("result must be a supported static magnetic result")
    p0 = OneElectronLowerMatrices(
        overlap=result.overlap.zero,
        kinetic=result.kinetic.zero,
        nuclear_attraction=result.nuclear_attraction.zero,
    )
    form_factor_only = OneElectronLowerMatrices(
        overlap=result.overlap.exact,
        kinetic=result.kinetic.exact_sectors.pp,
        nuclear_attraction=result.nuclear_attraction.exact,
    )
    anchored_pC = (
        result.kinetic.anchored_pC
        if isinstance(result, ExactStaticMagneticOneElectronResult)
        else result.kinetic.first_pC
    )
    anchored_Cp = (
        result.kinetic.anchored_Cp
        if isinstance(result, ExactStaticMagneticOneElectronResult)
        else result.kinetic.first_Cp
    )
    anchored_C2 = (
        result.kinetic.anchored_C2
        if isinstance(result, ExactStaticMagneticOneElectronResult)
        else result.kinetic.second_C2
    )
    anchored_vector_only = OneElectronLowerMatrices(
        overlap=result.overlap.zero,
        kinetic=(result.kinetic.zero + anchored_pC + anchored_Cp + anchored_C2),
        nuclear_attraction=result.nuclear_attraction.zero,
    )
    exact = OneElectronLowerMatrices(
        overlap=result.overlap.exact,
        kinetic=result.kinetic.exact,
        nuclear_attraction=result.nuclear_attraction.exact,
    )
    barred = StaticMagneticModelSet(
        p0=p0,
        form_factor_only=form_factor_only,
        anchored_vector_only=anchored_vector_only,
        exact=exact,
    )
    return StaticMagneticDiagnosticModels(
        barred=barred,
        lower=StaticMagneticModelSet(
            p0=_dress(result.endpoint_link, p0),
            form_factor_only=_dress(result.endpoint_link, form_factor_only),
            anchored_vector_only=_dress(result.endpoint_link, anchored_vector_only),
            exact=_dress(result.endpoint_link, exact),
        ),
    )


def static_magnetic_first_order_models(
    result: MagneticOneElectronResult,
) -> StaticMagneticFirstOrderModels:
    """Assemble P0, gB1, B1, C1, and exact static magnetic matrices.

    The endpoint link is exact in every named model.  Only endpoint-removed
    internal amplitudes are truncated.  Geometric B1 adds the first magnetic
    metric increment to P0 while leaving the mechanical matrix at P0.  Full
    B1 additionally adds the complete first magnetic mechanical increment.
    In the static electric-free Maxwell sector C1 reduces identically to B1.

    The exact-only WP3 result cannot be accepted here because it deliberately
    omits the first-order contractions required to define these action models.
    """

    if not isinstance(result, MagneticOneElectronResult):
        raise TypeError("result must be a MagneticOneElectronResult")

    p0 = OneElectronLowerMatrices(
        overlap=result.overlap.zero,
        kinetic=result.kinetic.zero,
        nuclear_attraction=result.nuclear_attraction.zero,
    )
    geometric_b1 = OneElectronLowerMatrices(
        overlap=result.overlap.b1,
        kinetic=result.kinetic.zero,
        nuclear_attraction=result.nuclear_attraction.zero,
    )
    full_b1 = OneElectronLowerMatrices(
        overlap=result.overlap.b1,
        kinetic=result.kinetic.b1,
        nuclear_attraction=result.nuclear_attraction.b1,
    )
    exact = OneElectronLowerMatrices(
        overlap=result.overlap.exact,
        kinetic=result.kinetic.exact,
        nuclear_attraction=result.nuclear_attraction.exact,
    )
    barred = StaticMagneticFirstOrderModelSet(
        p0=p0,
        geometric_b1=geometric_b1,
        full_b1=full_b1,
        complete_first=full_b1,
        exact=exact,
    )
    return StaticMagneticFirstOrderModels(
        barred=barred,
        lower=StaticMagneticFirstOrderModelSet(
            p0=_dress(result.endpoint_link, p0),
            geometric_b1=_dress(result.endpoint_link, geometric_b1),
            full_b1=_dress(result.endpoint_link, full_b1),
            complete_first=_dress(result.endpoint_link, full_b1),
            exact=_dress(result.endpoint_link, exact),
        ),
    )


def ao_angular_momenta(
    reference: PreparedReference | OneElectronAOReference,
) -> np.ndarray:
    """Expand authenticated shell angular momenta to one value per AO."""

    if not isinstance(reference, PreparedReference | OneElectronAOReference):
        raise TypeError("reference must be PreparedReference or OneElectronAOReference")
    if isinstance(reference, OneElectronAOReference):
        angular_momenta = reference.basis_metadata.shell_angular_momenta
        ao_locations = reference.basis_metadata.ao_locations
    else:
        molecule = reconstruct_mean_field(reference, BackendConfig()).mol
        angular_momenta = np.asarray(
            [molecule.bas_angular(shell) for shell in range(molecule.nbas)],
            dtype=np.int64,
        )
        ao_locations = np.asarray(molecule.ao_loc_nr(cart=False), dtype=np.int64)
    values = np.empty(reference.core_operators.nao, dtype=np.int64)
    for shell, angular_momentum in enumerate(angular_momenta):
        start = int(ao_locations[shell])
        stop = int(ao_locations[shell + 1])
        values[start:stop] = int(angular_momentum)
    values.setflags(write=False)
    return values


def matrix_partition_changes(
    candidate: object,
    reference: object,
    ao_to_atom: object,
    *,
    floor: float = np.finfo(np.float64).tiny,
) -> MatrixPartitionChanges:
    """Resolve a matrix change into canonical anchor partitions."""

    candidate_array, reference_array, mapping, checked_floor = _change_inputs(
        candidate, reference, ao_to_atom, floor
    )
    size = candidate_array.shape[0]
    diagonal = np.eye(size, dtype=bool)
    same_anchor = mapping[:, None] == mapping[None, :]
    masks = (diagonal, same_anchor & ~diagonal, ~same_anchor)
    difference = candidate_array - reference_array
    absolute: list[float] = []
    relative: list[float] = []
    maximum: list[float] = []
    for mask in masks:
        delta = difference[mask]
        baseline = reference_array[mask]
        absolute_norm = float(np.linalg.norm(delta))
        absolute.append(absolute_norm)
        relative.append(absolute_norm / max(float(np.linalg.norm(baseline)), checked_floor))
        maximum.append(float(np.max(np.abs(delta))) if delta.size else 0.0)
    return MatrixPartitionChanges(
        labels=("onsite_diagonal", "same_anchor_offdiagonal", "intersite"),
        absolute_frobenius=np.asarray(absolute),
        relative_frobenius=np.asarray(relative),
        maximum_absolute_element=np.asarray(maximum),
    )


def angular_channel_changes(
    candidate: object,
    reference: object,
    ao_to_atom: object,
    angular_momenta: object,
    *,
    floor: float = np.finfo(np.float64).tiny,
) -> tuple[AngularChannelChange, ...]:
    """Return every nonempty atom-pair/angular-channel block change."""

    candidate_array, reference_array, mapping, checked_floor = _change_inputs(
        candidate, reference, ao_to_atom, floor
    )
    angular = np.asarray(angular_momenta, dtype=np.int64)
    if angular.shape != mapping.shape or np.any(angular < 0):
        raise ValueError("angular_momenta must match ao_to_atom and be nonnegative")
    records: list[AngularChannelChange] = []
    for bra_atom in np.unique(mapping):
        bra_atom_mask = mapping == bra_atom
        for ket_atom in np.unique(mapping):
            ket_atom_mask = mapping == ket_atom
            for bra_l in np.unique(angular[bra_atom_mask]):
                rows = np.flatnonzero(bra_atom_mask & (angular == bra_l))
                for ket_l in np.unique(angular[ket_atom_mask]):
                    columns = np.flatnonzero(ket_atom_mask & (angular == ket_l))
                    candidate_block = candidate_array[np.ix_(rows, columns)]
                    reference_block = reference_array[np.ix_(rows, columns)]
                    difference = candidate_block - reference_block
                    absolute = float(np.linalg.norm(difference))
                    classification = (
                        "intersite"
                        if bra_atom != ket_atom
                        else "onsite"
                        if bra_l == ket_l
                        else "same_anchor_cross_l"
                    )
                    records.append(
                        AngularChannelChange(
                            bra_atom=int(bra_atom),
                            ket_atom=int(ket_atom),
                            bra_angular_momentum=int(bra_l),
                            ket_angular_momentum=int(ket_l),
                            classification=classification,
                            absolute_frobenius=absolute,
                            relative_frobenius=absolute
                            / max(float(np.linalg.norm(reference_block)), checked_floor),
                            maximum_absolute_element=float(np.max(np.abs(difference))),
                            singular_values=np.linalg.svd(difference, compute_uv=False),
                        )
                    )
    return tuple(records)


def diagonal_scaled_element_change(
    candidate: object,
    reference: object,
    *,
    floor: float,
) -> np.ndarray:
    """Return element changes normalized by geometric diagonal scales."""

    candidate_array = np.asarray(candidate, dtype=np.complex128)
    reference_array = np.asarray(reference, dtype=np.complex128)
    checked_floor = _positive_floor(floor)
    if (
        candidate_array.ndim != 2
        or candidate_array.shape[0] != candidate_array.shape[1]
        or reference_array.shape != candidate_array.shape
    ):
        raise ValueError("candidate and reference must be equal square matrices")
    diagonal = np.abs(np.diag(reference_array))
    scale = np.maximum(np.sqrt(diagonal[:, None] * diagonal[None, :]), checked_floor)
    result = np.asarray(np.abs(candidate_array - reference_array) / scale, dtype=np.float64)
    result.setflags(write=False)
    return result


def compare_generalized_spectra(
    candidate_hamiltonian: object,
    candidate_metric: object,
    reference_hamiltonian: object,
    reference_metric: object,
) -> GeneralizedSpectralComparison:
    """Compare Hermitian-definite spectra and cumulative coefficient subspaces.

    Coefficient-space principal angles are meaningful only within the same AO
    ordering and gauge representative. Cumulative subspaces are reported so a
    near-degenerate cluster can be interpreted at a gap boundary rather than
    through unstable individual eigenvectors.
    """

    candidate_h = _square_complex(candidate_hamiltonian, "candidate_hamiltonian")
    candidate_s = _square_complex(candidate_metric, "candidate_metric")
    reference_h = _square_complex(reference_hamiltonian, "reference_hamiltonian")
    reference_s = _square_complex(reference_metric, "reference_metric")
    shape = candidate_h.shape
    if any(value.shape != shape for value in (candidate_s, reference_h, reference_s)):
        raise ValueError("all generalized spectral matrices must have the same shape")
    candidate_values, candidate_vectors = eigh(candidate_h, candidate_s)
    reference_values, reference_vectors = eigh(reference_h, reference_s)
    angles = np.asarray(
        [
            float(
                np.max(
                    subspace_angles(
                        candidate_vectors[:, :dimension],
                        reference_vectors[:, :dimension],
                    )
                )
            )
            for dimension in range(1, shape[0])
        ],
        dtype=np.float64,
    )
    return GeneralizedSpectralComparison(
        reference_eigenvalues=reference_values,
        candidate_eigenvalues=candidate_values,
        eigenvalue_shifts=candidate_values - reference_values,
        cumulative_subspace_angles_rad=angles,
    )


def metric_spectrum(
    metric: object,
    *,
    positivity_tolerance: float = 0.0,
) -> MetricSpectrum:
    """Return explicit positivity and conditioning data for a Hermitian metric.

    No eigenvalue clipping or regularization is performed.  The tolerance is
    used only to classify the reported minimum eigenvalue.
    """

    matrix = _square_complex(metric, "metric")
    tolerance = float(positivity_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("positivity_tolerance must be finite and nonnegative")
    scale = max(1.0, float(np.linalg.norm(matrix)))
    hermiticity = float(np.linalg.norm(matrix - matrix.conj().T) / scale)
    values = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))
    minimum = float(values[0])
    maximum = float(values[-1])
    condition = float(np.linalg.cond(matrix))
    return MetricSpectrum(
        eigenvalues=values,
        minimum_eigenvalue=minimum,
        maximum_eigenvalue=maximum,
        condition_number=condition,
        hermiticity_residual=hermiticity,
        positive=minimum > tolerance,
    )


def _dress(endpoint_link: Any, matrices: OneElectronLowerMatrices) -> OneElectronLowerMatrices:
    return OneElectronLowerMatrices(
        overlap=endpoint_link * matrices.overlap,
        kinetic=endpoint_link * matrices.kinetic,
        nuclear_attraction=endpoint_link * matrices.nuclear_attraction,
    )


def _change_inputs(
    candidate: object,
    reference: object,
    ao_to_atom: object,
    floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    candidate_array = _square_complex(candidate, "candidate")
    reference_array = _square_complex(reference, "reference")
    if reference_array.shape != candidate_array.shape:
        raise ValueError("candidate and reference matrices must have the same shape")
    mapping = np.asarray(ao_to_atom, dtype=np.int64)
    if mapping.shape != (candidate_array.shape[0],) or np.any(mapping < 0):
        raise ValueError("ao_to_atom must match the matrix dimension")
    return candidate_array, reference_array, mapping, _positive_floor(floor)


def _square_complex(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.complex128)
    if result.ndim != 2 or result.shape[0] != result.shape[1] or result.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _positive_floor(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("floor must be a positive finite number")
    checked = float(value)
    if not np.isfinite(checked) or checked <= 0.0:
        raise ValueError("floor must be a positive finite number")
    return checked
