"""Self-consistent exact-Wilson Hartree and pure-LDA stationary states."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import numpy as np
from scipy.linalg import eigh

from aion.config import BackendKind
from aion.electromagnetism import AffineMagneticGauge
from aion.electronic_structure.adiabatic import expectation, hermitian_part
from aion.electronic_structure.ao_quadrature import AOQuadrature
from aion.electronic_structure.data import PreparedReference
from aion.electronic_structure.magnetic_matrices import (
    ExactStaticMagneticOneElectronResult,
    evaluate_exact_static_magnetic_one_electron_matrices,
)
from aion.electronic_structure.ri_wilson_hartree import (
    PreparedRIWilsonHartreeAction,
    RIMetricRankPolicy,
    RIWilsonHartreeEvaluator,
    RIWilsonHartreeResult,
    prepare_ri_wilson_hartree,
)
from aion.electronic_structure.wilson_lda import (
    WilsonLDAEvaluator,
    WilsonLDAResult,
    prepare_wilson_lda,
)
from aion.errors import ConfigurationError, FormulationError, UnsupportedConfigurationError


class WilsonStationaryBranch(StrEnum):
    """Implemented exact-Wilson nonlinear stationary branches."""

    HARTREE = "hartree"
    KOHN_SHAM_LDA = "kohn_sham_lda"


@dataclass(frozen=True, slots=True)
class StationarySCFPolicy:
    """Declared fixed-point, Pulay, and residual controls."""

    maximum_iterations: int = 100
    density_tolerance: float = 2.0e-10
    orbital_tolerance: float = 2.0e-10
    energy_tolerance_au: float = 2.0e-11
    damping: float = 0.5
    diis_start_iteration: int = 2
    diis_space: int = 8

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_iterations, bool)
            or not isinstance(self.maximum_iterations, int)
            or self.maximum_iterations <= 0
        ):
            raise ConfigurationError("maximum_iterations must be a positive integer")
        for name in ("density_tolerance", "orbital_tolerance", "energy_tolerance_au"):
            value = _positive_scalar(getattr(self, name), name)
            object.__setattr__(self, name, value)
        damping = _finite_scalar(self.damping, "damping")
        if not 0.0 < damping <= 1.0:
            raise ConfigurationError("damping must lie in (0, 1]")
        object.__setattr__(self, "damping", damping)
        if (
            isinstance(self.diis_start_iteration, bool)
            or not isinstance(self.diis_start_iteration, int)
            or self.diis_start_iteration < 1
        ):
            raise ConfigurationError("diis_start_iteration must be a positive integer")
        if (
            isinstance(self.diis_space, bool)
            or not isinstance(self.diis_space, int)
            or self.diis_space < 2
        ):
            raise ConfigurationError("diis_space must be an integer of at least two")


@dataclass(frozen=True, slots=True)
class ExactWilsonActionEvaluation:
    """One complete instantaneous nonlinear action evaluation."""

    coefficient_density: Any
    overlap: Any
    kinetic_matrix: Any
    nuclear_attraction_matrix: Any
    one_electron_matrix: Any
    lower_mechanical_matrix: Any
    hartree: RIWilsonHartreeResult
    exchange_correlation: WilsonLDAResult | None
    energy_kinetic_au: Any
    energy_electron_nuclear_au: Any
    energy_one_electron_au: Any
    energy_hartree_au: Any
    energy_exchange_correlation_au: Any
    energy_nuclear_repulsion_au: Any
    energy_electronic_au: Any
    energy_molecular_total_au: Any
    xc_potential_contraction_au: Any
    branch: WilsonStationaryBranch
    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str
    backend: str
    device_index: int | None


@dataclass(frozen=True, slots=True)
class StationarySCFIteration:
    """One visible nonlinear iteration diagnostic."""

    iteration: int
    energy_molecular_total_au: float
    energy_change_au: float | None
    density_fixed_point_residual: float
    commutator_residual: float
    diis_dimension: int
    used_diis: bool


class WilsonStationaryActionProtocol(Protocol):
    """Action fields required by the common SCF solver and diagnostics."""

    @property
    def overlap(self) -> Any: ...

    @property
    def lower_mechanical_matrix(self) -> Any: ...

    @property
    def energy_hartree_au(self) -> Any: ...

    @property
    def energy_exchange_correlation_au(self) -> Any: ...

    @property
    def energy_nuclear_repulsion_au(self) -> Any: ...

    @property
    def energy_molecular_total_au(self) -> Any: ...

    @property
    def xc_potential_contraction_au(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class ExactWilsonStationaryState[ActionT: WilsonStationaryActionProtocol]:
    """One converged nonlinear stationary generalized-eigenproblem."""

    coefficients: Any
    occupations: Any
    coefficient_density: Any
    mixed_density: Any
    orbital_frequency_matrix: Any
    active_orbital_energies_au: Any
    complete_orbital_spectrum_au: Any
    action: ActionT
    iterations: tuple[StationarySCFIteration, ...]
    orbital_residual: float
    density_fixed_point_residual: float
    commutator_residual: float
    metric_orthonormality_residual: float
    particle_number: float
    particle_number_residual: float
    occupation_spectrum: np.ndarray
    occupation_spectrum_imaginary_max_abs: float
    occupation_spectrum_residual: float
    closed_shell_density_polynomial_residual: float
    orbital_frequency_occupation_commutator_residual: float
    orbital_energy_sum_au: Any
    double_counting_reconstructed_electronic_energy_au: Any
    double_counting_reconstructed_molecular_energy_au: Any
    double_counting_residual_au: float
    metric_minimum_eigenvalue: float
    metric_condition_number: float
    converged: bool = True


@dataclass(slots=True)
class ExactWilsonStationaryModel:
    """Source-fixed exact-Wilson nonlinear action and stationary solver."""

    quadrature: AOQuadrature
    gauge: AffineMagneticGauge
    branch: WilsonStationaryBranch
    one_electron: ExactStaticMagneticOneElectronResult
    hartree_evaluator: RIWilsonHartreeEvaluator
    hartree_action: PreparedRIWilsonHartreeAction
    lda_evaluator: WilsonLDAEvaluator | None
    nuclear_repulsion_au: float

    @property
    def backend(self) -> Any:
        return self.quadrature.backend

    @property
    def namespace(self) -> Any:
        return self.backend.namespace

    @property
    def overlap(self) -> Any:
        return self.one_electron.lower_exact.overlap

    @property
    def one_electron_matrix(self) -> Any:
        return self.one_electron.lower_exact.mechanical

    def for_branch(
        self,
        branch: WilsonStationaryBranch | str,
        *,
        functional: str = "lda,vwn",
    ) -> ExactWilsonStationaryModel:
        """Reuse the source-fixed geometric actions for another nonlinear branch."""

        try:
            selected = WilsonStationaryBranch(branch)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"unsupported stationary branch {branch!r}") from exc
        lda = None
        if selected is WilsonStationaryBranch.KOHN_SHAM_LDA:
            if (
                self.lda_evaluator is not None
                and self.lda_evaluator.provenance.functional == functional
            ):
                lda = self.lda_evaluator
            else:
                lda = prepare_wilson_lda(self.quadrature, functional)
        return ExactWilsonStationaryModel(
            quadrature=self.quadrature,
            gauge=self.gauge,
            branch=selected,
            one_electron=self.one_electron,
            hartree_evaluator=self.hartree_evaluator,
            hartree_action=self.hartree_action,
            lda_evaluator=lda,
            nuclear_repulsion_au=self.nuclear_repulsion_au,
        )

    def evaluate(self, coefficient_density: object) -> ExactWilsonActionEvaluation:
        """Assemble one complete energy/lower-matrix package."""

        xp = self.namespace
        density = self.hartree_evaluator._validated_density(coefficient_density)
        hartree = self.hartree_action.evaluate(density)
        xc = (
            None
            if self.lda_evaluator is None
            else self.lda_evaluator.evaluate(density, self.gauge)
        )
        one_electron = self.one_electron.lower_exact
        kinetic = one_electron.kinetic
        nuclear = one_electron.nuclear_attraction
        lower = one_electron.mechanical + hartree.lower_coulomb_matrix
        xc_energy = self.backend.asarray(0.0, dtype=xp.float64)
        xc_contraction = self.backend.asarray(0.0, dtype=xp.float64)
        if xc is not None:
            lower = lower + xc.lower_xc_matrix
            xc_energy = xc.energy
            xc_contraction = expectation(density, xc.lower_xc_matrix, xp)
        lower = hermitian_part(lower)
        kinetic_energy = expectation(density, kinetic, xp)
        nuclear_energy = expectation(density, nuclear, xp)
        one_electron_energy = kinetic_energy + nuclear_energy
        electronic = one_electron_energy + hartree.energy + xc_energy
        nuclear_repulsion = self.backend.asarray(
            self.nuclear_repulsion_au,
            dtype=xp.float64,
        )
        molecular = electronic + nuclear_repulsion
        self.backend.synchronize()
        return ExactWilsonActionEvaluation(
            coefficient_density=density,
            overlap=self.overlap,
            kinetic_matrix=kinetic,
            nuclear_attraction_matrix=nuclear,
            one_electron_matrix=one_electron.mechanical,
            lower_mechanical_matrix=lower,
            hartree=hartree,
            exchange_correlation=xc,
            energy_kinetic_au=kinetic_energy,
            energy_electron_nuclear_au=nuclear_energy,
            energy_one_electron_au=one_electron_energy,
            energy_hartree_au=hartree.energy,
            energy_exchange_correlation_au=xc_energy,
            energy_nuclear_repulsion_au=nuclear_repulsion,
            energy_electronic_au=electronic,
            energy_molecular_total_au=molecular,
            xc_potential_contraction_au=xc_contraction,
            branch=self.branch,
            reference_fingerprint_sha256=(
                self.quadrature.reference.fingerprint_sha256
            ),
            grid_fingerprint_sha256=self.quadrature.grid.fingerprint_sha256,
            backend=self.quadrature.backend_config.kind.value,
            device_index=self.quadrature.backend_config.device_index,
        )

    def solve(
        self,
        *,
        policy: StationarySCFPolicy | None = None,
        initial_coefficients: object | None = None,
    ) -> ExactWilsonStationaryState[ExactWilsonActionEvaluation]:
        """Solve the static nonlinear generalized eigenproblem on the CPU.

        The generalized eigensystem is solved directly with ``scipy.linalg``;
        no full-AO Lowdin or Cholesky propagation transform is introduced.
        Pulay extrapolation acts only on lower mechanical matrices.
        """

        return solve_wilson_stationary_model(
            self,
            policy=policy,
            initial_coefficients=initial_coefficients,
        )


class WilsonStationaryModelProtocol[ActionT: WilsonStationaryActionProtocol](Protocol):
    """Structural action interface consumed by the common stationary solver."""

    @property
    def quadrature(self) -> AOQuadrature: ...

    @property
    def backend(self) -> Any: ...

    @property
    def namespace(self) -> Any: ...

    @property
    def overlap(self) -> Any: ...

    def evaluate(self, coefficient_density: object) -> ActionT: ...


def solve_wilson_stationary_model[ActionT: WilsonStationaryActionProtocol](
    model: WilsonStationaryModelProtocol[ActionT],
    *,
    policy: StationarySCFPolicy | None = None,
    initial_coefficients: object | None = None,
) -> ExactWilsonStationaryState[ActionT]:
    """Solve any action-compatible Wilson generalized eigenproblem on CPU.

    The generalized eigensystem is solved directly with ``scipy.linalg``;
    no full-AO Lowdin or Cholesky propagation transform is introduced. Pulay
    extrapolation acts only on lower mechanical matrices. Exact and reduced
    Wilson actions therefore share one nonlinear numerical algorithm.
    """

    if model.quadrature.backend_config.kind is not BackendKind.CPU:
        raise UnsupportedConfigurationError(
            "the stationary reference solver is CPU-only; GPU action evaluation "
            "is available separately"
        )
    selected_policy = StationarySCFPolicy() if policy is None else policy
    if not isinstance(selected_policy, StationarySCFPolicy):
        raise TypeError("policy must be a StationarySCFPolicy")
    reference = model.quadrature.reference
    if not isinstance(reference, PreparedReference):
        raise UnsupportedConfigurationError(
            "stationary SCF requires a prepared many-electron reference"
        )
    stored_occupations = np.asarray(
        reference.ground_state.occupations,
        dtype=np.float64,
    )
    occupied = np.flatnonzero(stored_occupations > 0.0)
    if occupied.size == 0 or not np.array_equal(occupied, np.arange(occupied.size)):
        raise UnsupportedConfigurationError(
            "stationary solver requires contiguous positive occupations first"
        )
    occupations = stored_occupations[occupied]
    if not np.allclose(occupations, 2.0, atol=1.0e-13, rtol=0.0):
        raise UnsupportedConfigurationError(
            "stationary solver currently requires fully doubly occupied orbitals"
        )
    overlap = np.asarray(model.overlap, dtype=np.complex128)
    _validate_positive_metric(overlap)
    if initial_coefficients is None:
        coefficients = np.asarray(
            reference.ground_state.coefficients[:, occupied],
            dtype=np.complex128,
        )
    else:
        coefficients = np.asarray(initial_coefficients, dtype=np.complex128)
    expected = (overlap.shape[0], occupied.size)
    if coefficients.shape != expected or not np.all(np.isfinite(coefficients)):
        raise ConfigurationError(
            f"initial_coefficients must be finite with shape {expected}"
        )
    coefficients = _metric_orthonormalize(coefficients, overlap)
    density = _coefficient_density(coefficients, occupations)
    diis = _PulayHistory(selected_policy.diis_space)
    records: list[StationarySCFIteration] = []
    previous_energy: float | None = None

    for iteration in range(1, selected_policy.maximum_iterations + 1):
        action = model.evaluate(density)
        lower = np.asarray(action.lower_mechanical_matrix, dtype=np.complex128)
        _, complete_coefficients = _generalized_eigensystem(lower, overlap)
        physical_coefficients = complete_coefficients[:, : occupied.size]
        physical_density = _coefficient_density(physical_coefficients, occupations)
        density_residual = _mixed_density_residual(
            physical_density,
            density,
            overlap,
        )
        commutator = _commutator_residual(lower, density, overlap)
        energy = float(action.energy_molecular_total_au)
        energy_change = None if previous_energy is None else abs(energy - previous_energy)
        error_matrix = lower @ density @ overlap - overlap @ density @ lower
        diis.add(lower, error_matrix)
        use_diis = iteration >= selected_policy.diis_start_iteration and diis.size >= 2
        records.append(
            StationarySCFIteration(
                iteration=iteration,
                energy_molecular_total_au=energy,
                energy_change_au=energy_change,
                density_fixed_point_residual=density_residual,
                commutator_residual=commutator,
                diis_dimension=diis.size,
                used_diis=use_diis,
            )
        )

        candidate = physical_density
        candidate_action = model.evaluate(candidate)
        candidate_lower = np.asarray(
            candidate_action.lower_mechanical_matrix,
            dtype=np.complex128,
        )
        candidate_orbital_residual, frequency = _orbital_residual(
            candidate_lower,
            overlap,
            physical_coefficients,
        )
        check_values, check_coefficients = _generalized_eigensystem(
            candidate_lower,
            overlap,
        )
        check_density = _coefficient_density(
            check_coefficients[:, : occupied.size],
            occupations,
        )
        candidate_density_residual = _mixed_density_residual(
            check_density,
            candidate,
            overlap,
        )
        candidate_energy_change = abs(
            float(candidate_action.energy_molecular_total_au) - energy
        )
        if (
            candidate_density_residual <= selected_policy.density_tolerance
            and candidate_orbital_residual <= selected_policy.orbital_tolerance
            and candidate_energy_change <= selected_policy.energy_tolerance_au
        ):
            return _build_stationary_state(
                model,
                candidate_action,
                physical_coefficients,
                occupations,
                frequency,
                check_values,
                candidate_density_residual,
                candidate_orbital_residual,
                tuple(records),
            )

        step_lower = diis.extrapolate() if use_diis else lower
        _, step_coefficients = _generalized_eigensystem(step_lower, overlap)
        step_density = _coefficient_density(
            step_coefficients[:, : occupied.size],
            occupations,
        )
        mixing = 1.0 if use_diis else selected_policy.damping
        density = hermitian_part(
            (1.0 - mixing) * density + mixing * step_density
        )
        previous_energy = energy

    raise FormulationError(
        "Wilson stationary SCF did not converge in "
        f"{selected_policy.maximum_iterations} iterations; final density residual "
        f"{records[-1].density_fixed_point_residual:.3e}, commutator residual "
        f"{records[-1].commutator_residual:.3e}"
    )


@dataclass(slots=True)
class ExactWilsonStationaryFactory:
    """Reusable quadrature, RI, and LDA data for a family of static sources."""

    quadrature: AOQuadrature
    hartree_evaluator: RIWilsonHartreeEvaluator
    lda_evaluator: WilsonLDAEvaluator
    nuclear_repulsion_au: float
    charge: float
    mass: float
    hbar: float

    def model(
        self,
        gauge: AffineMagneticGauge,
        branch: WilsonStationaryBranch | str,
    ) -> ExactWilsonStationaryModel:
        """Create one source-fixed stationary model without rebuilding shared data."""

        if not isinstance(gauge, AffineMagneticGauge):
            raise TypeError("gauge must be an AffineMagneticGauge")
        try:
            selected_branch = WilsonStationaryBranch(branch)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"unsupported stationary branch {branch!r}") from exc
        one_electron = evaluate_exact_static_magnetic_one_electron_matrices(
            self.quadrature,
            (gauge.field,),
            direct_gauges=(gauge,),
            charge=self.charge,
            mass=self.mass,
            hbar=self.hbar,
            include_direct_oracle=False,
        )[0]
        hartree_action = self.hartree_evaluator.prepare_action(
            gauge,
            charge=self.charge,
            hbar=self.hbar,
        )
        return ExactWilsonStationaryModel(
            quadrature=self.quadrature,
            gauge=gauge,
            branch=selected_branch,
            one_electron=one_electron,
            hartree_evaluator=self.hartree_evaluator,
            hartree_action=hartree_action,
            lda_evaluator=(
                None
                if selected_branch is WilsonStationaryBranch.HARTREE
                else self.lda_evaluator
            ),
            nuclear_repulsion_au=self.nuclear_repulsion_au,
        )


def prepare_exact_wilson_stationary_factory(
    quadrature: AOQuadrature,
    *,
    auxiliary_basis: str,
    functional: str = "lda,vwn",
    rank_policy: RIMetricRankPolicy | None = None,
    charge: float = -1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> ExactWilsonStationaryFactory:
    """Prepare source-independent nonlinear data once for a static scan."""

    if not isinstance(quadrature, AOQuadrature):
        raise TypeError("quadrature must be an AOQuadrature")
    if not isinstance(quadrature.reference, PreparedReference):
        raise UnsupportedConfigurationError(
            "stationary models require a prepared many-electron reference"
        )
    checked_charge = _finite_scalar(charge, "charge")
    checked_mass = _positive_scalar(mass, "mass")
    checked_hbar = _positive_scalar(hbar, "hbar")
    hartree_evaluator = prepare_ri_wilson_hartree(
        quadrature,
        auxiliary_basis,
        rank_policy=rank_policy,
    )
    return ExactWilsonStationaryFactory(
        quadrature=quadrature,
        hartree_evaluator=hartree_evaluator,
        lda_evaluator=prepare_wilson_lda(quadrature, functional),
        nuclear_repulsion_au=float(quadrature.pyscf_molecule.energy_nuc()),
        charge=checked_charge,
        mass=checked_mass,
        hbar=checked_hbar,
    )


def prepare_exact_wilson_stationary_model(
    quadrature: AOQuadrature,
    gauge: AffineMagneticGauge,
    branch: WilsonStationaryBranch | str,
    *,
    auxiliary_basis: str,
    functional: str = "lda,vwn",
    rank_policy: RIMetricRankPolicy | None = None,
    charge: float = -1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> ExactWilsonStationaryModel:
    """Prepare one source-fixed exact-Wilson stationary action."""

    if not isinstance(gauge, AffineMagneticGauge):
        raise TypeError("gauge must be an AffineMagneticGauge")
    factory = prepare_exact_wilson_stationary_factory(
        quadrature,
        auxiliary_basis=auxiliary_basis,
        functional=functional,
        rank_policy=rank_policy,
        charge=charge,
        mass=mass,
        hbar=hbar,
    )
    model = factory.model(gauge, branch)
    _validate_positive_metric(
        quadrature.backend.to_host(model.overlap)
        if quadrature.backend_config.kind is BackendKind.GPU
        else np.asarray(model.overlap)
    )
    return model


@dataclass(slots=True)
class _PulayHistory:
    maximum_size: int
    matrices: list[np.ndarray] = field(default_factory=list)
    errors: list[np.ndarray] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.matrices)

    def add(self, matrix: np.ndarray, error: np.ndarray) -> None:
        self.matrices.append(np.array(matrix, copy=True))
        self.errors.append(np.array(error, copy=True))
        if self.size > self.maximum_size:
            self.matrices.pop(0)
            self.errors.pop(0)

    def extrapolate(self) -> np.ndarray:
        dimension = self.size
        system = np.empty((dimension + 1, dimension + 1), dtype=np.float64)
        system[:-1, :-1] = [
            [float(np.vdot(left, right).real) for right in self.errors]
            for left in self.errors
        ]
        system[-1, :-1] = -1.0
        system[:-1, -1] = -1.0
        system[-1, -1] = 0.0
        scale = max(1.0, float(np.max(np.abs(system[:-1, :-1]))))
        system[:-1, :-1] /= scale
        target = np.zeros((dimension + 1,), dtype=np.float64)
        target[-1] = -1.0
        try:
            coefficients = np.linalg.solve(system, target)[:-1]
        except np.linalg.LinAlgError:
            return self.matrices[-1]
        result = sum(
            (coefficient * matrix for coefficient, matrix in zip(
                coefficients, self.matrices, strict=True
            )),
            np.zeros_like(self.matrices[0]),
        )
        return np.asarray(hermitian_part(result), dtype=np.complex128)


def _build_stationary_state[ActionT: WilsonStationaryActionProtocol](
    model: WilsonStationaryModelProtocol[ActionT],
    action: ActionT,
    coefficients: np.ndarray,
    occupations: np.ndarray,
    frequency: np.ndarray,
    complete_spectrum: np.ndarray,
    density_residual: float,
    orbital_residual: float,
    records: tuple[StationarySCFIteration, ...],
) -> ExactWilsonStationaryState[ActionT]:
    xp = model.namespace
    overlap = np.asarray(action.overlap, dtype=np.complex128)
    lower = np.asarray(action.lower_mechanical_matrix, dtype=np.complex128)
    density = _coefficient_density(coefficients, occupations)
    mixed = density @ overlap
    commutator = _commutator_residual(lower, density, overlap)
    orthonormality = float(
        np.linalg.norm(coefficients.conj().T @ overlap @ coefficients - np.eye(len(occupations)))
    )
    particle_number = float(np.trace(mixed).real)
    target_particle_number = float(np.sum(occupations))
    spectrum_complex = np.linalg.eigvals(mixed)
    spectrum = np.sort(spectrum_complex.real)[::-1]
    expected_spectrum = np.sort(
        np.concatenate((occupations, np.zeros(overlap.shape[0] - len(occupations))))
    )[::-1]
    spectrum_residual = float(
        np.linalg.norm(spectrum - expected_spectrum)
        / max(1.0, float(np.linalg.norm(expected_spectrum)))
    )
    polynomial = float(
        np.linalg.norm(mixed @ mixed - 2.0 * mixed)
        / max(1.0, float(np.linalg.norm(mixed)))
    )
    occupation_matrix = np.diag(occupations)
    frequency_commutator = float(
        np.linalg.norm(frequency @ occupation_matrix - occupation_matrix @ frequency)
        / max(1.0, float(np.linalg.norm(frequency)))
    )
    orbital_sum = np.einsum("i,ii->", occupations, frequency, optimize=True).real
    reconstructed_electronic = (
        orbital_sum
        - float(action.energy_hartree_au)
        + float(action.energy_exchange_correlation_au)
        - float(action.xc_potential_contraction_au)
    )
    reconstructed_molecular = reconstructed_electronic + float(
        action.energy_nuclear_repulsion_au
    )
    double_counting = abs(
        reconstructed_molecular - float(action.energy_molecular_total_au)
    )
    metric_eigenvalues = np.linalg.eigvalsh(overlap)
    backend_coefficients = model.backend.asarray(coefficients, dtype=xp.complex128)
    backend_occupations = model.backend.asarray(occupations, dtype=xp.float64)
    backend_density = model.backend.asarray(density, dtype=xp.complex128)
    backend_mixed = model.backend.asarray(mixed, dtype=xp.complex128)
    backend_frequency = model.backend.asarray(frequency, dtype=xp.complex128)
    backend_active_energies = model.backend.asarray(
        np.linalg.eigvalsh(frequency), dtype=xp.float64
    )
    backend_complete_spectrum = model.backend.asarray(complete_spectrum, dtype=xp.float64)
    backend_orbital_sum = model.backend.asarray(orbital_sum, dtype=xp.float64)
    backend_reconstructed_electronic = model.backend.asarray(
        reconstructed_electronic,
        dtype=xp.float64,
    )
    backend_reconstructed_molecular = model.backend.asarray(
        reconstructed_molecular,
        dtype=xp.float64,
    )
    return ExactWilsonStationaryState(
        coefficients=backend_coefficients,
        occupations=backend_occupations,
        coefficient_density=backend_density,
        mixed_density=backend_mixed,
        orbital_frequency_matrix=backend_frequency,
        active_orbital_energies_au=backend_active_energies,
        complete_orbital_spectrum_au=backend_complete_spectrum,
        action=action,
        iterations=records,
        orbital_residual=orbital_residual,
        density_fixed_point_residual=density_residual,
        commutator_residual=commutator,
        metric_orthonormality_residual=orthonormality,
        particle_number=particle_number,
        particle_number_residual=abs(particle_number - target_particle_number),
        occupation_spectrum=np.asarray(spectrum, dtype=np.float64),
        occupation_spectrum_imaginary_max_abs=float(np.max(np.abs(spectrum_complex.imag))),
        occupation_spectrum_residual=spectrum_residual,
        closed_shell_density_polynomial_residual=polynomial,
        orbital_frequency_occupation_commutator_residual=frequency_commutator,
        orbital_energy_sum_au=backend_orbital_sum,
        double_counting_reconstructed_electronic_energy_au=(
            backend_reconstructed_electronic
        ),
        double_counting_reconstructed_molecular_energy_au=(
            backend_reconstructed_molecular
        ),
        double_counting_residual_au=double_counting,
        metric_minimum_eigenvalue=float(metric_eigenvalues[0]),
        metric_condition_number=float(metric_eigenvalues[-1] / metric_eigenvalues[0]),
    )


def _generalized_eigensystem(
    lower: np.ndarray,
    overlap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values, coefficients = eigh(
        np.asarray(hermitian_part(lower), dtype=np.complex128),
        np.asarray(hermitian_part(overlap), dtype=np.complex128),
        check_finite=True,
    )
    return np.asarray(values, dtype=np.float64), np.asarray(
        coefficients,
        dtype=np.complex128,
    )


def _coefficient_density(coefficients: np.ndarray, occupations: np.ndarray) -> np.ndarray:
    return np.asarray(
        np.einsum(
            "mi,i,ni->mn",
            coefficients,
            occupations,
            coefficients.conj(),
            optimize=True,
        ),
        dtype=np.complex128,
    )


def _metric_orthonormalize(coefficients: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    gram = np.asarray(
        hermitian_part(coefficients.conj().T @ overlap @ coefficients),
        dtype=np.complex128,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if eigenvalues[0] <= 1.0e-12 * max(1.0, float(eigenvalues[-1])):
        raise ConfigurationError("initial occupied coefficient frame is rank deficient")
    inverse_square_root = (
        eigenvectors / np.sqrt(eigenvalues)[None, :]
    ) @ eigenvectors.conj().T
    return np.asarray(coefficients @ inverse_square_root, dtype=np.complex128)


def _mixed_density_residual(
    candidate: np.ndarray,
    reference: np.ndarray,
    overlap: np.ndarray,
) -> float:
    candidate_mixed = candidate @ overlap
    difference = (candidate - reference) @ overlap
    return float(
        np.linalg.norm(difference) / max(1.0, float(np.linalg.norm(candidate_mixed)))
    )


def _commutator_residual(
    lower: np.ndarray,
    density: np.ndarray,
    overlap: np.ndarray,
) -> float:
    residual = lower @ density @ overlap - overlap @ density @ lower
    scale = max(
        1.0,
        float(np.linalg.norm(lower @ density @ overlap)),
        float(np.linalg.norm(overlap @ density @ lower)),
    )
    return float(np.linalg.norm(residual) / scale)


def _orbital_residual(
    lower: np.ndarray,
    overlap: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[float, np.ndarray]:
    frequency = np.asarray(
        hermitian_part(coefficients.conj().T @ lower @ coefficients),
        dtype=np.complex128,
    )
    residual = lower @ coefficients - overlap @ coefficients @ frequency
    scale = max(1.0, float(np.linalg.norm(lower @ coefficients)))
    return float(np.linalg.norm(residual) / scale), frequency


def _validate_positive_metric(overlap: np.ndarray) -> None:
    matrix = np.asarray(overlap, dtype=np.complex128)
    residual = np.linalg.norm(matrix - matrix.conj().T) / max(
        1.0,
        float(np.linalg.norm(matrix)),
    )
    if residual > 2.0e-10:
        raise FormulationError("stationary overlap is not Hermitian")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] <= 0.0:
        raise FormulationError(
            "stationary overlap is outside the admitted positive-metric domain: "
            f"lambda_min={eigenvalues[0]:.3e}"
        )


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
