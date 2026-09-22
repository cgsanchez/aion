"""Variational Coulomb-metric RI Hartree action for straight-Wilson frames.

The stored energy, lower Coulomb matrix, and fixed-coefficient source
direction in this module are descendants of one discrete three-index action.
PySCF supplies the analytic auxiliary Coulomb metric and the Coulomb
potential of each auxiliary Gaussian on the molecular grid.  Aion contracts
those potentials with the complete complex Wilson pair field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aion.electromagnetism import build_magnetic_pair_geometry
from aion.electronic_structure.ao_quadrature import AOQuadrature
from aion.electronic_structure.data import immutable_array
from aion.electronic_structure.wilson_density import (
    StraightLineVectorPotentialDirection,
)
from aion.errors import ConfigurationError, FormulationError, ReferencePreparationError


@dataclass(frozen=True, slots=True)
class RIMetricRankPolicy:
    """Declared retained subspace of the auxiliary Coulomb metric.

    Eigenvectors are retained when their eigenvalue is strictly larger than
    ``max(absolute_threshold, relative_threshold * lambda_max)``.  An
    optional maximum rank keeps the largest surviving eigenvalues.
    """

    relative_threshold: float = 0.0
    absolute_threshold: float = 1.0e-7
    maximum_rank: int | None = None

    def __post_init__(self) -> None:
        relative = _nonnegative_scalar(self.relative_threshold, "relative_threshold")
        absolute = _nonnegative_scalar(self.absolute_threshold, "absolute_threshold")
        if relative == 0.0 and absolute == 0.0:
            raise ConfigurationError("at least one RI metric threshold must be positive")
        if self.maximum_rank is not None and (
            isinstance(self.maximum_rank, bool)
            or not isinstance(self.maximum_rank, int)
            or self.maximum_rank <= 0
        ):
            raise ConfigurationError("maximum_rank must be a positive integer or None")
        object.__setattr__(self, "relative_threshold", relative)
        object.__setattr__(self, "absolute_threshold", absolute)


@dataclass(frozen=True, slots=True)
class RIAuxiliaryProvenance:
    """Immutable numerical identity of one auxiliary realization."""

    auxiliary_basis: str
    auxiliary_functions: int
    metric_rank: int
    metric_dimension: int
    metric_cutoff: float
    metric_minimum_retained_eigenvalue: float
    metric_maximum_eigenvalue: float
    metric_retained_condition_number: float
    relative_threshold: float
    absolute_threshold: float
    maximum_rank: int | None
    point_charge_exponent: float
    potential_cache: str
    potential_cache_bytes: int
    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class RIWilsonHartreeResult:
    """One evaluation of the discrete RI--Wilson Hartree action.

    All numerical arrays are resident on the evaluator backend.  The
    optional source fields are present only when a fixed-history vector-
    potential direction was requested.
    """

    energy: Any
    lower_coulomb_matrix: Any
    moment: Any
    fitted_coefficients: Any
    three_index: Any
    source_energy_direction: Any | None
    source_moment_direction: Any | None
    three_index_source_direction: Any | None
    pair_counting_residual: float
    lower_hermiticity_residual: float
    moment_imaginary_max_abs: float
    source_moment_imaginary_max_abs: float | None
    retained_solve_relative_residual: float
    discarded_moment_relative_norm: float
    stationary_energy_residual: float
    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str
    backend: str
    device_index: int | None
    charge: float
    hbar: float
    auxiliary_basis: str
    auxiliary_rank: int
    self_interaction_included: bool = True


@dataclass(frozen=True, slots=True)
class PreparedRIWilsonHartreeAction:
    """Source-fixed RI--Wilson tensor reused across nonlinear evaluations.

    The three-index tensor and its optional source direction depend on the
    electromagnetic source and quadrature, but not on the coefficient
    density.  Preparing them once is therefore an exact reuse of the same
    discrete action, not a frozen-density approximation.
    """

    evaluator: RIWilsonHartreeEvaluator
    three_index: Any
    three_index_source_direction: Any | None
    charge: float
    hbar: float

    def evaluate(self, coefficient_density: object) -> RIWilsonHartreeResult:
        """Contract the prepared action with one contravariant density."""

        density = self.evaluator._validated_density(coefficient_density)
        return self.evaluator._contract_action(
            density,
            self.three_index,
            self.three_index_source_direction,
            self.charge,
            self.hbar,
        )


@dataclass(slots=True)
class RIWilsonHartreeEvaluator:
    r"""Prepared Coulomb-metric RI evaluator for a fixed AO quadrature.

    The discrete action is

    ``E_H = 1/2 b.T M_R^+ b``,

    with ``b_P = sum_ij P^{ji} B_{P,ij}``.  ``M_R^+`` is the declared
    retained-rank inverse of the analytic auxiliary Coulomb metric.  The
    same grid three-index tensor ``B`` generates the energy and lower
    matrix.  A requested source direction differentiates that same tensor.
    """

    quadrature: AOQuadrature
    auxiliary_basis: str
    rank_policy: RIMetricRankPolicy = field(default_factory=RIMetricRankPolicy)
    cache_auxiliary_potentials: bool = True
    potential_cache_memory_budget_bytes: int | None = None
    point_charge_exponent: float = 1.0e16
    metric: Any = field(init=False, repr=False)
    metric_inverse: Any = field(init=False, repr=False)
    retained_projector: Any = field(init=False, repr=False)
    provenance: RIAuxiliaryProvenance = field(init=False)
    _auxiliary_molecule: Any = field(init=False, repr=False)
    _potential_cache_host: np.ndarray | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.quadrature, AOQuadrature):
            raise TypeError("quadrature must be an AOQuadrature")
        if not isinstance(self.auxiliary_basis, str) or not self.auxiliary_basis.strip():
            raise ConfigurationError("auxiliary_basis must be a nonempty explicit name")
        if not isinstance(self.rank_policy, RIMetricRankPolicy):
            raise TypeError("rank_policy must be an RIMetricRankPolicy")
        if not isinstance(self.cache_auxiliary_potentials, bool):
            raise ConfigurationError("cache_auxiliary_potentials must be boolean")
        if self.potential_cache_memory_budget_bytes is not None and (
            isinstance(self.potential_cache_memory_budget_bytes, bool)
            or not isinstance(self.potential_cache_memory_budget_bytes, int)
            or self.potential_cache_memory_budget_bytes <= 0
        ):
            raise ConfigurationError(
                "potential_cache_memory_budget_bytes must be a positive integer"
            )
        exponent = _positive_scalar(self.point_charge_exponent, "point_charge_exponent")
        object.__setattr__(self, "point_charge_exponent", exponent)

        from pyscf import df

        try:
            auxiliary_molecule = df.addons.make_auxmol(
                self.quadrature.pyscf_molecule,
                self.auxiliary_basis,
            )
            metric_host = np.asarray(
                auxiliary_molecule.intor("int2c2e"),
                dtype=np.float64,
            )
        except Exception as exc:
            raise ReferencePreparationError(
                f"failed to prepare auxiliary Coulomb data for {self.auxiliary_basis!r}"
            ) from exc
        if metric_host.ndim != 2 or metric_host.shape[0] != metric_host.shape[1]:
            raise ReferencePreparationError("auxiliary Coulomb metric is not square")
        metric_hermiticity = np.linalg.norm(metric_host - metric_host.T) / max(
            1.0, np.linalg.norm(metric_host)
        )
        if metric_hermiticity > 1.0e-12:
            raise ReferencePreparationError("auxiliary Coulomb metric is not symmetric")
        eigenvalues, eigenvectors = np.linalg.eigh(metric_host)
        maximum = float(eigenvalues[-1])
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ReferencePreparationError("auxiliary Coulomb metric is not positive")
        cutoff = max(
            self.rank_policy.absolute_threshold,
            self.rank_policy.relative_threshold * maximum,
        )
        retained = np.flatnonzero(eigenvalues > cutoff)
        if (
            self.rank_policy.maximum_rank is not None
            and retained.size > self.rank_policy.maximum_rank
        ):
            retained = retained[-self.rank_policy.maximum_rank :]
        if retained.size == 0:
            raise ReferencePreparationError("RI metric rank policy retained no directions")
        retained_vectors = eigenvectors[:, retained]
        retained_values = eigenvalues[retained]
        inverse_host = (retained_vectors / retained_values[None, :]) @ retained_vectors.T
        projector_host = retained_vectors @ retained_vectors.T
        backend = self.quadrature.backend
        xp = backend.namespace
        object.__setattr__(self, "metric", backend.asarray(metric_host, dtype=xp.float64))
        object.__setattr__(
            self,
            "metric_inverse",
            backend.asarray(inverse_host, dtype=xp.float64),
        )
        object.__setattr__(
            self,
            "retained_projector",
            backend.asarray(projector_host, dtype=xp.float64),
        )
        object.__setattr__(self, "_auxiliary_molecule", auxiliary_molecule)

        cache_bytes = self.quadrature.grid.npoints * int(metric_host.shape[0]) * 8
        if (
            self.cache_auxiliary_potentials
            and self.potential_cache_memory_budget_bytes is not None
            and cache_bytes > self.potential_cache_memory_budget_bytes
        ):
            raise ConfigurationError(
                "auxiliary-potential cache requires "
                f"{cache_bytes} bytes, exceeding "
                f"potential_cache_memory_budget_bytes={self.potential_cache_memory_budget_bytes}"
            )
        cache = self._build_potential_cache() if self.cache_auxiliary_potentials else None
        object.__setattr__(self, "_potential_cache_host", cache)
        object.__setattr__(
            self,
            "provenance",
            RIAuxiliaryProvenance(
                auxiliary_basis=self.auxiliary_basis,
                auxiliary_functions=int(metric_host.shape[0]),
                metric_rank=int(retained.size),
                metric_dimension=int(metric_host.shape[0]),
                metric_cutoff=float(cutoff),
                metric_minimum_retained_eigenvalue=float(retained_values[0]),
                metric_maximum_eigenvalue=maximum,
                metric_retained_condition_number=float(maximum / retained_values[0]),
                relative_threshold=self.rank_policy.relative_threshold,
                absolute_threshold=self.rank_policy.absolute_threshold,
                maximum_rank=self.rank_policy.maximum_rank,
                point_charge_exponent=exponent,
                potential_cache="host" if cache is not None else "blocked_recompute",
                potential_cache_bytes=cache_bytes if cache is not None else 0,
                reference_fingerprint_sha256=self.quadrature.reference.fingerprint_sha256,
                grid_fingerprint_sha256=self.quadrature.grid.fingerprint_sha256,
            ),
        )

    @property
    def backend(self) -> Any:
        return self.quadrature.backend

    @property
    def namespace(self) -> Any:
        return self.backend.namespace

    def evaluate(
        self,
        coefficient_density: object,
        vector_potential: StraightLineVectorPotentialDirection,
        *,
        source_direction: StraightLineVectorPotentialDirection | None = None,
        charge: float = -1.0,
        hbar: float = 1.0,
    ) -> RIWilsonHartreeResult:
        """Evaluate one action package and optional fixed-history direction."""

        action = self.prepare_action(
            vector_potential,
            source_direction=source_direction,
            charge=charge,
            hbar=hbar,
        )
        return action.evaluate(coefficient_density)

    def prepare_action(
        self,
        vector_potential: StraightLineVectorPotentialDirection,
        *,
        source_direction: StraightLineVectorPotentialDirection | None = None,
        charge: float = -1.0,
        hbar: float = 1.0,
    ) -> PreparedRIWilsonHartreeAction:
        """Prepare density-independent RI--Wilson tensors for one source."""

        line_method = getattr(vector_potential, "straight_line_integrals", None)
        if not callable(line_method):
            raise TypeError("vector_potential must provide straight_line_integrals")
        if source_direction is not None and not callable(
            getattr(source_direction, "straight_line_integrals", None)
        ):
            raise TypeError("source_direction must provide straight_line_integrals")
        checked_charge = _finite_scalar(charge, "charge")
        checked_hbar = _positive_scalar(hbar, "hbar")
        three_index, three_index_direction = self._build_three_index(
            vector_potential,
            source_direction,
            checked_charge,
            checked_hbar,
        )
        return PreparedRIWilsonHartreeAction(
            evaluator=self,
            three_index=three_index,
            three_index_source_direction=three_index_direction,
            charge=checked_charge,
            hbar=checked_hbar,
        )

    def _build_potential_cache(self) -> np.ndarray:
        npoint = self.quadrature.grid.npoints
        naux = int(self.metric.shape[0])
        cache = np.empty((npoint, naux), dtype=np.float64)
        for start in range(0, npoint, self.quadrature.block_size):
            stop = min(start + self.quadrature.block_size, npoint)
            cache[start:stop] = self._auxiliary_potentials_host(
                self.quadrature.grid.coordinates_au[start:stop]
            )
        return immutable_array(cache, dtype=np.float64, ndim=2, name="auxiliary potentials")

    def _auxiliary_potentials_host(self, coordinates: np.ndarray) -> np.ndarray:
        from pyscf import gto

        fake_charges = gto.fakemol_for_charges(
            np.asarray(coordinates, dtype=np.float64),
            expnt=self.point_charge_exponent,
        )
        values = gto.intor_cross(
            "int2c2e",
            self._auxiliary_molecule,
            fake_charges,
        ).T
        expected = (coordinates.shape[0], int(self.metric.shape[0]))
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise ReferencePreparationError(
                f"auxiliary Coulomb potentials have shape {values.shape}; expected {expected}"
            )
        return np.asarray(values, dtype=np.float64)

    def _potential_block(self, start: int, stop: int) -> Any:
        if self._potential_cache_host is None:
            host = self._auxiliary_potentials_host(self.quadrature.grid.coordinates_au[start:stop])
        else:
            host = self._potential_cache_host[start:stop]
        result = self.backend.asarray(host, dtype=self.namespace.float64)
        self.backend.assert_resident(result, name="auxiliary Coulomb potentials")
        return result

    def _build_three_index(
        self,
        vector_potential: StraightLineVectorPotentialDirection,
        source_direction: StraightLineVectorPotentialDirection | None,
        charge: float,
        hbar: float,
    ) -> tuple[Any, Any | None]:
        backend = self.backend
        xp = self.namespace
        reference = self.quadrature.reference
        nao = reference.core_operators.nao
        naux = int(self.metric.shape[0])
        three_index = backend.zeros((naux, nao, nao), dtype=xp.complex128)
        direction = (
            None
            if source_direction is None
            else backend.zeros((naux, nao, nao), dtype=xp.complex128)
        )
        geometry = build_magnetic_pair_geometry(
            reference.core_operators.nuclei.coordinates_au,
            reference.anchor_topology.ao_to_atom,
            backend,
        )
        anchors = geometry.ao_anchor_coordinates_au
        prefactor = 1j * charge / hbar
        for block in self.quadrature.blocks():
            line = vector_potential.straight_line_integrals(
                anchors[None, :, :],
                block.coordinates_au[:, None, :],
                backend,
            )
            dressed = xp.exp(prefactor * line) * block.values
            auxiliary = self._potential_block(block.start, block.stop)
            three_index += xp.einsum(
                "p,pP,pi,pj->Pij",
                block.weights_au,
                auxiliary,
                dressed.conj(),
                dressed,
                optimize=True,
            )
            if source_direction is not None:
                direction_line = source_direction.straight_line_integrals(
                    anchors[None, :, :],
                    block.coordinates_au[:, None, :],
                    backend,
                )
                dressed_direction = prefactor * direction_line * dressed
                assert direction is not None
                direction += xp.einsum(
                    "p,pP,pi,pj->Pij",
                    block.weights_au,
                    auxiliary,
                    dressed_direction.conj(),
                    dressed,
                    optimize=True,
                ) + xp.einsum(
                    "p,pP,pi,pj->Pij",
                    block.weights_au,
                    auxiliary,
                    dressed.conj(),
                    dressed_direction,
                    optimize=True,
                )
        return three_index, direction

    def _contract_action(
        self,
        density: Any,
        three_index: Any,
        three_index_direction: Any | None,
        charge: float,
        hbar: float,
    ) -> RIWilsonHartreeResult:
        xp = self.namespace
        complex_moment = xp.einsum(
            "ji,Pij->P",
            density,
            three_index,
            optimize=True,
        )
        moment_imaginary = self.backend.scalar_to_float(xp.max(xp.abs(xp.imag(complex_moment))))
        _require_real_contraction(complex_moment, moment_imaginary, self.backend, "RI moment")
        moment = xp.real(complex_moment)
        coefficients = self.metric_inverse @ moment
        lower = xp.einsum(
            "P,Pij->ij",
            coefficients,
            three_index,
            optimize=True,
        )
        energy = 0.5 * xp.einsum("P,P->", moment, coefficients, optimize=True)
        pair_value = xp.real(xp.einsum("ij,ji->", lower, density, optimize=True))
        pair_residual = self.backend.scalar_to_float(xp.abs(pair_value - 2.0 * energy))
        lower_scale = xp.maximum(xp.asarray(1.0), xp.linalg.norm(lower))
        lower_hermiticity = self.backend.scalar_to_float(
            xp.linalg.norm(lower - lower.conj().T) / lower_scale
        )
        projected_moment = self.retained_projector @ moment
        solve_scale = xp.maximum(xp.asarray(1.0), xp.linalg.norm(projected_moment))
        solve_residual = self.backend.scalar_to_float(
            xp.linalg.norm(self.metric @ coefficients - projected_moment) / solve_scale
        )
        moment_scale = xp.maximum(xp.asarray(1.0), xp.linalg.norm(moment))
        discarded = self.backend.scalar_to_float(
            xp.linalg.norm(moment - projected_moment) / moment_scale
        )
        stationary_objective = xp.einsum(
            "P,P->", coefficients, moment, optimize=True
        ) - 0.5 * xp.einsum("P,PQ,Q->", coefficients, self.metric, coefficients, optimize=True)
        stationary_residual = self.backend.scalar_to_float(xp.abs(stationary_objective - energy))

        source_energy = None
        source_moment = None
        source_imaginary = None
        if three_index_direction is not None:
            complex_source_moment = xp.einsum(
                "ji,Pij->P",
                density,
                three_index_direction,
                optimize=True,
            )
            source_imaginary = self.backend.scalar_to_float(
                xp.max(xp.abs(xp.imag(complex_source_moment)))
            )
            _require_real_contraction(
                complex_source_moment,
                source_imaginary,
                self.backend,
                "RI source moment",
            )
            source_moment = xp.real(complex_source_moment)
            source_energy = xp.einsum(
                "P,P->",
                coefficients,
                source_moment,
                optimize=True,
            )
        self.backend.synchronize()
        return RIWilsonHartreeResult(
            energy=energy,
            lower_coulomb_matrix=lower,
            moment=moment,
            fitted_coefficients=coefficients,
            three_index=three_index,
            source_energy_direction=source_energy,
            source_moment_direction=source_moment,
            three_index_source_direction=three_index_direction,
            pair_counting_residual=pair_residual,
            lower_hermiticity_residual=lower_hermiticity,
            moment_imaginary_max_abs=moment_imaginary,
            source_moment_imaginary_max_abs=source_imaginary,
            retained_solve_relative_residual=solve_residual,
            discarded_moment_relative_norm=discarded,
            stationary_energy_residual=stationary_residual,
            reference_fingerprint_sha256=self.quadrature.reference.fingerprint_sha256,
            grid_fingerprint_sha256=self.quadrature.grid.fingerprint_sha256,
            backend=self.quadrature.backend_config.kind.value,
            device_index=self.quadrature.backend_config.device_index,
            charge=charge,
            hbar=hbar,
            auxiliary_basis=self.auxiliary_basis,
            auxiliary_rank=self.provenance.metric_rank,
        )

    def _validated_density(self, value: object) -> Any:
        xp = self.namespace
        dimension = self.quadrature.reference.core_operators.nao
        density = self.backend.asarray(value, dtype=xp.complex128)
        self.backend.assert_resident(density, name="contravariant coefficient density")
        if density.shape != (dimension, dimension):
            raise ConfigurationError(
                f"coefficient_density has shape {density.shape}; expected {(dimension, dimension)}"
            )
        if not _control_bool(xp.all(xp.isfinite(density)), self.backend):
            raise ConfigurationError("coefficient_density contains non-finite values")
        scale = xp.maximum(xp.asarray(1.0), xp.linalg.norm(density))
        residual = self.backend.scalar_to_float(xp.linalg.norm(density - density.conj().T) / scale)
        if residual > 1.0e-11:
            raise ConfigurationError(
                "coefficient_density Hermiticity residual "
                f"{residual:.3e} exceeds 1.000e-11"
            )
        return density


def prepare_ri_wilson_hartree(
    quadrature: AOQuadrature,
    auxiliary_basis: str,
    *,
    rank_policy: RIMetricRankPolicy | None = None,
    cache_auxiliary_potentials: bool = True,
    potential_cache_memory_budget_bytes: int | None = None,
    point_charge_exponent: float = 1.0e16,
) -> RIWilsonHartreeEvaluator:
    """Prepare one explicitly labelled RI--Wilson Hartree realization."""

    return RIWilsonHartreeEvaluator(
        quadrature=quadrature,
        auxiliary_basis=auxiliary_basis,
        rank_policy=RIMetricRankPolicy() if rank_policy is None else rank_policy,
        cache_auxiliary_potentials=cache_auxiliary_potentials,
        potential_cache_memory_budget_bytes=potential_cache_memory_budget_bytes,
        point_charge_exponent=point_charge_exponent,
    )


def _require_real_contraction(
    value: Any,
    imaginary_maximum: float,
    backend: Any,
    name: str,
) -> None:
    xp = backend.namespace
    scale = backend.scalar_to_float(xp.maximum(xp.asarray(1.0), xp.linalg.norm(xp.real(value))))
    if imaginary_maximum > 2.0e-10 * scale:
        raise FormulationError(
            f"{name} has an unresolved imaginary component {imaginary_maximum:.3e}"
        )


def _control_bool(value: object, backend: Any) -> bool:
    return bool(backend.scalar_to_float(value))


def _finite_scalar(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{name} must be a finite number")
    return result


def _positive_scalar(value: float, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ConfigurationError(f"{name} must be positive")
    return result


def _nonnegative_scalar(value: float, name: str) -> float:
    result = _finite_scalar(value, name)
    if result < 0.0:
        raise ConfigurationError(f"{name} must be nonnegative")
    return result
