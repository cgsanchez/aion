"""Variational P0, E1, strict-C1, and density-resummed Wilson actions.

The exact-Wilson implementation is the reference model.  This module builds
the reduced nonlinear descendants defined in Chapter 13 from the same AO
quadrature, RI Coulomb metric, pure-LDA datum, endpoint links, and analytic
one-electron first magnetic derivatives.  Approximation is applied to the
action before matter differentiation:

``P0`` and ``E1`` use the zeroth-order pair density ``N0``;
``STRICT_C1`` uses the degree-zero and degree-one closure coefficients; and
``DENSITY_RESUMMED_C1`` evaluates the unexpanded closure at ``n0 + n1``.

The last two are intentionally distinct action labels even though they share
the same first-order one-electron geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from aion.electromagnetism import (
    AffineMagneticGauge,
    UniformMagneticSourceSample,
    build_magnetic_pair_geometry,
    endpoint_links,
    triangle_phases,
)
from aion.electronic_structure.adiabatic import expectation, hermitian_part
from aion.electronic_structure.electric_matrices import (
    UniformElectricE1Tensor,
    evaluate_uniform_electric_e1_tensor,
)
from aion.electronic_structure.magnetic_matrices import (
    MagneticOneElectronFirstDerivatives,
    evaluate_magnetic_one_electron_first_derivatives,
)
from aion.electronic_structure.wilson_lda import WilsonLDAEvaluator
from aion.electronic_structure.wilson_stationary import (
    ExactWilsonStationaryFactory,
    ExactWilsonStationaryState,
    StationarySCFPolicy,
    WilsonStationaryBranch,
    solve_wilson_stationary_model,
)
from aion.errors import ConfigurationError, FormulationError
from aion.formulations import EOMTriple, one_electron_velocity_density


class ReducedWilsonLevel(StrEnum):
    """Action-level electromagnetic descendants qualified by NQ7."""

    P0 = "p0"
    E1 = "e1"
    STRICT_C1 = "strict_c1"
    DENSITY_RESUMMED_C1 = "density_resummed_c1"

    @property
    def retains_electric_increment(self) -> bool:
        return self is not ReducedWilsonLevel.P0

    @property
    def retains_first_magnetic_order(self) -> bool:
        return self in (
            ReducedWilsonLevel.STRICT_C1,
            ReducedWilsonLevel.DENSITY_RESUMMED_C1,
        )


@dataclass(frozen=True, slots=True)
class ReducedWilsonDensityResult:
    """Zeroth, first, and assembled real-space density data."""

    density_zero: Any
    density_first: Any
    density_assembled: Any
    electron_count_zero: Any
    electron_count_first: Any
    electron_count_assembled: Any
    density_zero_minimum: float
    density_assembled_minimum: float
    density_zero_imaginary_max_abs: float
    density_first_imaginary_max_abs: float


@dataclass(frozen=True, slots=True)
class ReducedWilsonHartreeResult:
    """RI Hartree coefficients and matter derivative of one reduced action."""

    energy_zero: Any
    energy_first: Any
    energy: Any
    lower_zero: Any
    lower_first: Any
    lower_matrix: Any
    moment_zero: Any
    moment_first: Any
    fitted_zero: Any
    fitted_first: Any
    pair_counting_residual: float


@dataclass(frozen=True, slots=True)
class ReducedWilsonLDAResult:
    """Pure-LDA energy and lower matrix of one reduced action."""

    energy_zero: Any
    energy_first: Any
    energy: Any
    lower_zero: Any
    lower_first: Any
    lower_matrix: Any
    potential_zero: Any
    potential_first: Any
    kernel_zero: Any
    potential_contraction: Any


@dataclass(frozen=True, slots=True)
class ReducedWilsonClosureResult:
    """Matched density, Hartree, and optional LDA descendants."""

    density: ReducedWilsonDensityResult
    hartree: ReducedWilsonHartreeResult
    exchange_correlation: ReducedWilsonLDAResult | None
    lower_matrix: Any
    energy_hartree_au: Any
    energy_exchange_correlation_au: Any
    energy_total_au: Any
    xc_potential_contraction_au: Any


@dataclass(frozen=True, slots=True)
class ReducedWilsonOneElectronData:
    """One selected endpoint-dressed one-electron action level."""

    metric: Any
    kinetic: Any
    nuclear_attraction: Any
    mechanical: Any
    barred_metric: Any
    barred_kinetic: Any
    barred_nuclear_attraction: Any
    endpoint_link: Any
    metric_minimum_eigenvalue: float
    metric_condition_number: float
    metric_positive: bool


@dataclass(frozen=True, slots=True)
class ReducedWilsonTimeConnection:
    """Connection reconstructed from the selected metric and E1 residual."""

    connection: Any
    metric_dot: Any
    endpoint_link_dot: Any
    site_connection: Any
    covariant_metric_rate: Any
    electric_residual: Any
    metric_compatibility_residual: float


@dataclass(frozen=True, slots=True)
class ReducedWilsonSourceResponse:
    """Fixed-coefficient-history derivative along one physical source rate."""

    one_electron_matrix_rate: Any
    density_zero_rate: Any
    density_first_rate: Any
    hartree_energy_rate_au: Any
    exchange_correlation_energy_rate_au: Any
    closure_energy_rate_au: Any
    mechanical_energy_rate_au: Any


@dataclass(frozen=True, slots=True)
class ReducedWilsonPowerObservation:
    """On-shell mechanical-energy rate of one reduced nonlinear action."""

    source_response: ReducedWilsonSourceResponse
    velocity_density: Any
    transport_energy_rate_au: Any
    source_power_au: Any


@dataclass(frozen=True, slots=True)
class ReducedWilsonActionEvaluation:
    """Complete instantaneous reduced nonlinear action evaluation."""

    coefficient_density: Any
    overlap: Any
    kinetic_matrix: Any
    nuclear_attraction_matrix: Any
    one_electron_matrix: Any
    lower_mechanical_matrix: Any
    closure: ReducedWilsonClosureResult
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
    level: ReducedWilsonLevel


@dataclass(frozen=True, slots=True)
class ReducedWilsonDynamicEvaluation:
    """Action evaluation and EOM triple for one reduced model sample."""

    sample: ReducedWilsonModel
    action: ReducedWilsonActionEvaluation
    triple: EOMTriple


@dataclass(slots=True)
class PreparedReducedWilsonClosure:
    """Density-independent N0/N1 and RI data for one magnetic source."""

    factory: PreparedReducedWilsonFactory
    gauge: AffineMagneticGauge
    endpoint_link: Any
    three_index_zero: Any
    three_index_first: Any

    @property
    def backend(self) -> Any:
        return self.factory.exact_factory.quadrature.backend

    @property
    def namespace(self) -> Any:
        return self.backend.namespace

    def evaluate(
        self,
        coefficient_density: object,
        level: ReducedWilsonLevel | str,
        branch: WilsonStationaryBranch | str,
    ) -> ReducedWilsonClosureResult:
        """Evaluate the closure energy and its unrestricted matter derivative."""

        selected_level = _level(level)
        selected_branch = _branch(branch)
        density = self.factory.exact_factory.hartree_evaluator._validated_density(
            coefficient_density
        )
        density_result, lda = self._density_and_lda(
            density,
            selected_level,
            selected_branch,
        )
        hartree = self._hartree(density, selected_level)
        xp = self.namespace
        zero = self.backend.asarray(0.0, dtype=xp.float64)
        lower = hartree.lower_matrix
        xc_energy = zero
        xc_contraction = zero
        if lda is not None:
            lower = lower + lda.lower_matrix
            xc_energy = lda.energy
            xc_contraction = lda.potential_contraction
        lower = hermitian_part(lower)
        return ReducedWilsonClosureResult(
            density=density_result,
            hartree=hartree,
            exchange_correlation=lda,
            lower_matrix=lower,
            energy_hartree_au=hartree.energy,
            energy_exchange_correlation_au=xc_energy,
            energy_total_au=hartree.energy + xc_energy,
            xc_potential_contraction_au=xc_contraction,
        )

    def source_energy_rates(
        self,
        coefficient_density: object,
        level: ReducedWilsonLevel | str,
        branch: WilsonStationaryBranch | str,
        source: UniformMagneticSourceSample,
        endpoint_link_dot: Any,
    ) -> tuple[Any, Any, Any, Any]:
        """Differentiate the selected closure at fixed coefficient history."""

        selected_level = _level(level)
        selected_branch = _branch(branch)
        density = self.factory.exact_factory.hartree_evaluator._validated_density(
            coefficient_density
        )
        density_result, lda = self._density_and_lda(
            density,
            selected_level,
            selected_branch,
        )
        quadrature = self.factory.exact_factory.quadrature
        evaluator = self.factory.exact_factory.hartree_evaluator
        backend = self.backend
        xp = self.namespace
        geometry = build_magnetic_pair_geometry(
            quadrature.reference.core_operators.nuclei.coordinates_au,
            quadrature.reference.anchor_topology.ao_to_atom,
            backend,
        )
        npoint = quadrature.grid.npoints
        density_zero_rate = backend.zeros((npoint,), dtype=xp.float64)
        density_first_rate = backend.zeros((npoint,), dtype=xp.float64)
        three_zero_rate = backend.zeros(
            self.three_index_zero.shape,
            dtype=xp.complex128,
        )
        three_first_rate = backend.zeros(
            self.three_index_first.shape,
            dtype=xp.complex128,
        )

        for block in quadrature.blocks():
            bare_pair = block.values.conj()[:, :, None] * block.values[:, None, :]
            phase = triangle_phases(
                block.coordinates_au,
                geometry,
                self.gauge.field,
                backend,
                charge=self.factory.exact_factory.charge,
                hbar=self.factory.exact_factory.hbar,
            )
            phase_rate = triangle_phases(
                block.coordinates_au,
                geometry,
                source.gauge_rate.field,
                backend,
                charge=self.factory.exact_factory.charge,
                hbar=self.factory.exact_factory.hbar,
            )
            pair_zero = self.endpoint_link[None, :, :] * bare_pair
            pair_zero_rate = endpoint_link_dot[None, :, :] * bare_pair
            pair_first_rate = (1j * phase_rate) * pair_zero + (1j * phase) * pair_zero_rate
            complex_zero_rate = xp.einsum(
                "ji,pij->p",
                density,
                pair_zero_rate,
                optimize=True,
            )
            complex_first_rate = xp.einsum(
                "ji,pij->p",
                density,
                pair_first_rate,
                optimize=True,
            )
            _require_real(complex_zero_rate, backend, "zeroth-order density source rate")
            _require_real(complex_first_rate, backend, "first-order density source rate")
            density_zero_rate[block.start : block.stop] = xp.real(complex_zero_rate)
            density_first_rate[block.start : block.stop] = xp.real(complex_first_rate)
            auxiliary = evaluator._potential_block(block.start, block.stop)
            three_zero_rate += xp.einsum(
                "p,pP,pij->Pij",
                block.weights_au,
                auxiliary,
                pair_zero_rate,
                optimize=True,
            )
            three_first_rate += xp.einsum(
                "p,pP,pij->Pij",
                block.weights_au,
                auxiliary,
                pair_first_rate,
                optimize=True,
            )

        b0 = xp.real(xp.einsum("ji,Pij->P", density, self.three_index_zero, optimize=True))
        b1 = xp.real(xp.einsum("ji,Pij->P", density, self.three_index_first, optimize=True))
        b0_rate = xp.real(xp.einsum("ji,Pij->P", density, three_zero_rate, optimize=True))
        b1_rate = xp.real(xp.einsum("ji,Pij->P", density, three_first_rate, optimize=True))
        c0 = evaluator.metric_inverse @ b0
        c0_rate = evaluator.metric_inverse @ b0_rate
        if selected_level in (ReducedWilsonLevel.P0, ReducedWilsonLevel.E1):
            hartree_rate = xp.einsum("P,P->", b0_rate, c0, optimize=True)
        elif selected_level is ReducedWilsonLevel.STRICT_C1:
            hartree_rate = (
                xp.einsum("P,P->", b0_rate, c0, optimize=True)
                + xp.einsum("P,P->", b1_rate, c0, optimize=True)
                + xp.einsum("P,P->", b1, c0_rate, optimize=True)
            )
        else:
            moment = b0 + b1
            moment_rate = b0_rate + b1_rate
            fitted = evaluator.metric_inverse @ moment
            hartree_rate = xp.einsum(
                "P,P->",
                moment_rate,
                fitted,
                optimize=True,
            )

        xc_rate = backend.asarray(0.0, dtype=xp.float64)
        if lda is not None:
            weights = backend.asarray(quadrature.grid.weights_au, dtype=xp.float64)
            if selected_level in (ReducedWilsonLevel.P0, ReducedWilsonLevel.E1):
                xc_rate = xp.einsum(
                    "p,p,p->",
                    weights,
                    lda.potential_zero,
                    density_zero_rate,
                    optimize=True,
                )
            elif selected_level is ReducedWilsonLevel.STRICT_C1:
                xc_rate = (
                    xp.einsum(
                        "p,p,p->",
                        weights,
                        lda.potential_zero,
                        density_zero_rate,
                        optimize=True,
                    )
                    + xp.einsum(
                        "p,p,p,p->",
                        weights,
                        lda.kernel_zero,
                        density_result.density_first,
                        density_zero_rate,
                        optimize=True,
                    )
                    + xp.einsum(
                        "p,p,p->",
                        weights,
                        lda.potential_zero,
                        density_first_rate,
                        optimize=True,
                    )
                )
            else:
                xc_rate = xp.einsum(
                    "p,p,p->",
                    weights,
                    lda.potential_zero + lda.potential_first,
                    density_zero_rate + density_first_rate,
                    optimize=True,
                )
        return density_zero_rate, density_first_rate, hartree_rate, xc_rate

    def _hartree(
        self,
        density: Any,
        level: ReducedWilsonLevel,
    ) -> ReducedWilsonHartreeResult:
        evaluator = self.factory.exact_factory.hartree_evaluator
        xp = self.namespace
        b0_complex = xp.einsum("ji,Pij->P", density, self.three_index_zero, optimize=True)
        b1_complex = xp.einsum("ji,Pij->P", density, self.three_index_first, optimize=True)
        _require_real(b0_complex, self.backend, "zeroth-order RI moment")
        _require_real(b1_complex, self.backend, "first-order RI moment")
        b0 = xp.real(b0_complex)
        b1 = xp.real(b1_complex)
        c0 = evaluator.metric_inverse @ b0
        c1 = evaluator.metric_inverse @ b1
        e0 = 0.5 * xp.einsum("P,P->", b0, c0, optimize=True)
        e1 = xp.einsum("P,P->", b1, c0, optimize=True)
        v00 = xp.einsum("P,Pij->ij", c0, self.three_index_zero, optimize=True)
        z = self.backend.zeros(v00.shape, dtype=xp.complex128)
        if level in (ReducedWilsonLevel.P0, ReducedWilsonLevel.E1):
            energy = e0
            first = self.backend.asarray(0.0, dtype=xp.float64)
            lower_first = z
            lower = v00
            fitted_first = self.backend.zeros(c0.shape, dtype=xp.float64)
        elif level is ReducedWilsonLevel.STRICT_C1:
            v10 = xp.einsum("P,Pij->ij", c0, self.three_index_first, optimize=True)
            v01 = xp.einsum("P,Pij->ij", c1, self.three_index_zero, optimize=True)
            first = e1
            energy = e0 + e1
            lower_first = v10 + v01
            lower = v00 + lower_first
            fitted_first = c1
        else:
            three = self.three_index_zero + self.three_index_first
            moment = b0 + b1
            fitted = evaluator.metric_inverse @ moment
            energy = 0.5 * xp.einsum("P,P->", moment, fitted, optimize=True)
            lower = xp.einsum("P,Pij->ij", fitted, three, optimize=True)
            first = energy - e0
            lower_first = lower - v00
            fitted_first = fitted - c0
        lower = hermitian_part(lower)
        lower_first = hermitian_part(lower_first)
        pair_value = xp.real(xp.einsum("ij,ji->", lower, density, optimize=True))
        pair_residual = self.backend.scalar_to_float(xp.abs(pair_value - 2.0 * energy))
        return ReducedWilsonHartreeResult(
            energy_zero=e0,
            energy_first=first,
            energy=energy,
            lower_zero=hermitian_part(v00),
            lower_first=lower_first,
            lower_matrix=lower,
            moment_zero=b0,
            moment_first=b1,
            fitted_zero=c0,
            fitted_first=fitted_first,
            pair_counting_residual=pair_residual,
        )

    def _density_and_lda(
        self,
        density: Any,
        level: ReducedWilsonLevel,
        branch: WilsonStationaryBranch,
    ) -> tuple[ReducedWilsonDensityResult, ReducedWilsonLDAResult | None]:
        quadrature = self.factory.exact_factory.quadrature
        backend = self.backend
        xp = self.namespace
        lda_evaluator = self.factory.exact_factory.lda_evaluator
        point_count = quadrature.grid.npoints
        nao = quadrature.reference.core_operators.nao
        n0_values = backend.zeros((point_count,), dtype=xp.float64)
        n1_values = backend.zeros((point_count,), dtype=xp.float64)
        n0_integral = backend.asarray(0.0, dtype=xp.float64)
        n1_integral = backend.asarray(0.0, dtype=xp.float64)
        xc0 = backend.asarray(0.0, dtype=xp.float64)
        xc1 = backend.asarray(0.0, dtype=xp.float64)
        lower0 = backend.zeros((nao, nao), dtype=xp.complex128)
        lower1 = backend.zeros((nao, nao), dtype=xp.complex128)
        potential0_values = backend.zeros((point_count,), dtype=xp.float64)
        potential1_values = backend.zeros((point_count,), dtype=xp.float64)
        kernel_values = backend.zeros((point_count,), dtype=xp.float64)
        geometry = build_magnetic_pair_geometry(
            quadrature.reference.core_operators.nuclei.coordinates_au,
            quadrature.reference.anchor_topology.ao_to_atom,
            backend,
        )
        maximum_imag0 = 0.0
        maximum_imag1 = 0.0
        minimum0 = math.inf
        minimum_assembled = math.inf

        for block in quadrature.blocks():
            bare_pair = block.values.conj()[:, :, None] * block.values[:, None, :]
            phase = triangle_phases(
                block.coordinates_au,
                geometry,
                self.gauge.field,
                backend,
                charge=self.factory.exact_factory.charge,
                hbar=self.factory.exact_factory.hbar,
            )
            pair0 = self.endpoint_link[None, :, :] * bare_pair
            pair1 = (1j * phase) * pair0
            complex0 = xp.einsum("ji,pij->p", density, pair0, optimize=True)
            complex1 = xp.einsum("ji,pij->p", density, pair1, optimize=True)
            real0 = xp.real(complex0)
            real1 = xp.real(complex1)
            real_assembled = real0 + real1
            imag0 = backend.scalar_to_float(xp.max(xp.abs(xp.imag(complex0))))
            imag1 = backend.scalar_to_float(xp.max(xp.abs(xp.imag(complex1))))
            maximum_imag0 = max(maximum_imag0, imag0)
            maximum_imag1 = max(maximum_imag1, imag1)
            minimum0 = min(minimum0, backend.scalar_to_float(xp.min(real0)))
            minimum_assembled = min(
                minimum_assembled,
                backend.scalar_to_float(xp.min(real_assembled)),
            )
            n0_values[block.start : block.stop] = real0
            n1_values[block.start : block.stop] = real1
            n0_integral += xp.einsum("p,p->", block.weights_au, real0, optimize=True)
            n1_integral += xp.einsum("p,p->", block.weights_au, real1, optimize=True)

            if branch is WilsonStationaryBranch.KOHN_SHAM_LDA:
                _validate_density_domain(real0, lda_evaluator, backend, "n0")
                if level is ReducedWilsonLevel.DENSITY_RESUMMED_C1:
                    _validate_density_domain(
                        real_assembled,
                        lda_evaluator,
                        backend,
                        "n0+n1",
                    )
                    exc0_host, potential0_host, kernel0_host = (
                        lda_evaluator.evaluate_pointwise_with_kernel(backend.to_host(real0))
                    )
                    exc_resummed_host, potential_resummed_host, _ = (
                        lda_evaluator.evaluate_pointwise_with_kernel(
                            backend.to_host(real_assembled)
                        )
                    )
                    exc0 = backend.asarray(exc0_host, dtype=xp.float64)
                    potential0 = backend.asarray(potential0_host, dtype=xp.float64)
                    kernel0 = backend.asarray(kernel0_host, dtype=xp.float64)
                    exc_resummed = backend.asarray(exc_resummed_host, dtype=xp.float64)
                    potential_resummed = backend.asarray(
                        potential_resummed_host,
                        dtype=xp.float64,
                    )
                    xc0_block = xp.einsum(
                        "p,p,p->",
                        block.weights_au,
                        real0,
                        exc0,
                        optimize=True,
                    )
                    xc_resummed_block = xp.einsum(
                        "p,p,p->",
                        block.weights_au,
                        real_assembled,
                        exc_resummed,
                        optimize=True,
                    )
                    lower0_block = xp.einsum(
                        "p,p,pij->ij",
                        block.weights_au,
                        potential0,
                        pair0,
                        optimize=True,
                    )
                    lower_resummed_block = xp.einsum(
                        "p,p,pij->ij",
                        block.weights_au,
                        potential_resummed,
                        pair0 + pair1,
                        optimize=True,
                    )
                    xc0 += xc0_block
                    xc1 += xc_resummed_block - xc0_block
                    lower0 += lower0_block
                    lower1 += lower_resummed_block - lower0_block
                    potential0_values[block.start : block.stop] = potential0
                    potential1_values[block.start : block.stop] = potential_resummed - potential0
                    kernel_values[block.start : block.stop] = kernel0
                else:
                    exc_host, potential_host, kernel_host = (
                        lda_evaluator.evaluate_pointwise_with_kernel(backend.to_host(real0))
                    )
                    exc = backend.asarray(exc_host, dtype=xp.float64)
                    potential0 = backend.asarray(potential_host, dtype=xp.float64)
                    kernel0 = backend.asarray(kernel_host, dtype=xp.float64)
                    potential0_values[block.start : block.stop] = potential0
                    kernel_values[block.start : block.stop] = kernel0
                    xc0 += xp.einsum("p,p,p->", block.weights_au, real0, exc, optimize=True)
                    lower0 += xp.einsum(
                        "p,p,pij->ij",
                        block.weights_au,
                        potential0,
                        pair0,
                        optimize=True,
                    )
                    if level is ReducedWilsonLevel.STRICT_C1:
                        potential1 = kernel0 * real1
                        potential1_values[block.start : block.stop] = potential1
                        xc1 += xp.einsum(
                            "p,p,p->",
                            block.weights_au,
                            potential0,
                            real1,
                            optimize=True,
                        )
                        lower1 += xp.einsum(
                            "p,p,pij->ij",
                            block.weights_au,
                            potential0,
                            pair1,
                            optimize=True,
                        ) + xp.einsum(
                            "p,p,pij->ij",
                            block.weights_au,
                            potential1,
                            pair0,
                            optimize=True,
                        )

        density_result = ReducedWilsonDensityResult(
            density_zero=n0_values,
            density_first=n1_values,
            density_assembled=n0_values + n1_values,
            electron_count_zero=n0_integral,
            electron_count_first=n1_integral,
            electron_count_assembled=n0_integral + n1_integral,
            density_zero_minimum=minimum0,
            density_assembled_minimum=minimum_assembled,
            density_zero_imaginary_max_abs=maximum_imag0,
            density_first_imaginary_max_abs=maximum_imag1,
        )
        if branch is WilsonStationaryBranch.HARTREE:
            return density_result, None

        xc_zero = xc0
        xc_first = xc1
        xc_energy = xc0 + xc1
        xc_lower_first = hermitian_part(lower1)
        xc_lower = hermitian_part(lower0 + lower1)
        contraction = expectation(density, xc_lower, xp)
        return density_result, ReducedWilsonLDAResult(
            energy_zero=xc_zero,
            energy_first=xc_first,
            energy=xc_energy,
            lower_zero=hermitian_part(lower0),
            lower_first=xc_lower_first,
            lower_matrix=xc_lower,
            potential_zero=potential0_values,
            potential_first=potential1_values,
            kernel_zero=kernel_values,
            potential_contraction=contraction,
        )


@dataclass(frozen=True, slots=True)
class PreparedReducedWilsonFactory:
    """Source-independent data shared by every reduced action level."""

    exact_factory: ExactWilsonStationaryFactory
    magnetic_first_derivatives: MagneticOneElectronFirstDerivatives
    electric_e1: UniformElectricE1Tensor

    def spatial_action(
        self,
        gauge: AffineMagneticGauge,
        level: ReducedWilsonLevel | str,
    ) -> PreparedReducedWilsonSpatialAction:
        """Prepare density-independent one-electron and closure data."""

        if not isinstance(gauge, AffineMagneticGauge):
            raise TypeError("gauge must be an AffineMagneticGauge")
        selected = _level(level)
        one = self._one_electron(gauge, selected)
        closure = self._closure(gauge)
        return PreparedReducedWilsonSpatialAction(
            factory=self,
            gauge=gauge,
            level=selected,
            one_electron=one,
            closure=closure,
        )

    def model(
        self,
        gauge: AffineMagneticGauge,
        level: ReducedWilsonLevel | str,
        branch: WilsonStationaryBranch | str,
    ) -> ReducedWilsonModel:
        """Build a static-source model (zero electric field and ``Bdot``)."""

        source = UniformMagneticSourceSample(
            time_au=0.0,
            field=gauge.field,
            origin_au=gauge.origin_au,
            gauge_kind=gauge.kind,
            landau_axis=gauge.landau_axis,
        )
        return self.spatial_action(gauge, level).sample(source, branch)

    def _one_electron(
        self,
        gauge: AffineMagneticGauge,
        level: ReducedWilsonLevel,
    ) -> ReducedWilsonOneElectronData:
        quadrature = self.exact_factory.quadrature
        backend = quadrature.backend
        xp = backend.namespace
        reference = quadrature.reference
        geometry = build_magnetic_pair_geometry(
            reference.core_operators.nuclei.coordinates_au,
            reference.anchor_topology.ao_to_atom,
            backend,
        )
        endpoint = endpoint_links(
            gauge,
            geometry,
            backend,
            charge=self.exact_factory.charge,
            hbar=self.exact_factory.hbar,
        )
        overlap0 = backend.asarray(reference.core_operators.overlap, dtype=xp.complex128)
        kinetic0 = backend.asarray(reference.core_operators.kinetic, dtype=xp.complex128)
        nuclear0 = backend.asarray(
            reference.core_operators.nuclear_attraction,
            dtype=xp.complex128,
        )
        if level.retains_first_magnetic_order:
            field = backend.asarray(gauge.field.magnetic_field_au, dtype=xp.float64)
            overlap_bar = overlap0 + xp.einsum(
                "x,xij->ij",
                field,
                self.magnetic_first_derivatives.metric,
                optimize=True,
            )
            kinetic_bar = kinetic0 + xp.einsum(
                "x,xij->ij",
                field,
                self.magnetic_first_derivatives.kinetic,
                optimize=True,
            )
            nuclear_bar = nuclear0 + xp.einsum(
                "x,xij->ij",
                field,
                self.magnetic_first_derivatives.nuclear_attraction_triangle,
                optimize=True,
            )
        else:
            overlap_bar = overlap0
            kinetic_bar = kinetic0
            nuclear_bar = nuclear0
        metric = hermitian_part(endpoint * overlap_bar)
        kinetic = hermitian_part(endpoint * kinetic_bar)
        nuclear = hermitian_part(endpoint * nuclear_bar)
        eigenvalues = np.linalg.eigvalsh(backend.to_host(metric))
        positive = bool(eigenvalues[0] > 0.0)
        condition = float(eigenvalues[-1] / eigenvalues[0]) if positive else math.inf
        return ReducedWilsonOneElectronData(
            metric=metric,
            kinetic=kinetic,
            nuclear_attraction=nuclear,
            mechanical=kinetic + nuclear,
            barred_metric=overlap_bar,
            barred_kinetic=kinetic_bar,
            barred_nuclear_attraction=nuclear_bar,
            endpoint_link=endpoint,
            metric_minimum_eigenvalue=float(eigenvalues[0]),
            metric_condition_number=condition,
            metric_positive=positive,
        )

    def _closure(self, gauge: AffineMagneticGauge) -> PreparedReducedWilsonClosure:
        quadrature = self.exact_factory.quadrature
        evaluator = self.exact_factory.hartree_evaluator
        backend = quadrature.backend
        xp = backend.namespace
        reference = quadrature.reference
        geometry = build_magnetic_pair_geometry(
            reference.core_operators.nuclei.coordinates_au,
            reference.anchor_topology.ao_to_atom,
            backend,
        )
        endpoint = endpoint_links(
            gauge,
            geometry,
            backend,
            charge=self.exact_factory.charge,
            hbar=self.exact_factory.hbar,
        )
        naux = int(evaluator.metric.shape[0])
        nao = reference.core_operators.nao
        b0 = backend.zeros((naux, nao, nao), dtype=xp.complex128)
        b1 = backend.zeros((naux, nao, nao), dtype=xp.complex128)
        for block in quadrature.blocks():
            bare_pair = block.values.conj()[:, :, None] * block.values[:, None, :]
            phase = triangle_phases(
                block.coordinates_au,
                geometry,
                gauge.field,
                backend,
                charge=self.exact_factory.charge,
                hbar=self.exact_factory.hbar,
            )
            pair0 = endpoint[None, :, :] * bare_pair
            pair1 = (1j * phase) * pair0
            auxiliary = evaluator._potential_block(block.start, block.stop)
            b0 += xp.einsum(
                "p,pP,pij->Pij",
                block.weights_au,
                auxiliary,
                pair0,
                optimize=True,
            )
            b1 += xp.einsum(
                "p,pP,pij->Pij",
                block.weights_au,
                auxiliary,
                pair1,
                optimize=True,
            )
        return PreparedReducedWilsonClosure(
            factory=self,
            gauge=gauge,
            endpoint_link=endpoint,
            three_index_zero=b0,
            three_index_first=b1,
        )


@dataclass(frozen=True, slots=True)
class PreparedReducedWilsonSpatialAction:
    """Reduced spatial action reused for temporal samples at fixed ``B``."""

    factory: PreparedReducedWilsonFactory
    gauge: AffineMagneticGauge
    level: ReducedWilsonLevel
    one_electron: ReducedWilsonOneElectronData
    closure: PreparedReducedWilsonClosure

    def sample(
        self,
        source: UniformMagneticSourceSample,
        branch: WilsonStationaryBranch | str,
    ) -> ReducedWilsonModel:
        if not isinstance(source, UniformMagneticSourceSample):
            raise TypeError("source must be a UniformMagneticSourceSample")
        if source.gauge != self.gauge:
            raise ConfigurationError("temporal source does not match spatial gauge")
        connection = _time_connection(self, source)
        return ReducedWilsonModel(
            spatial=self,
            source=source,
            branch=_branch(branch),
            connection=connection,
        )


@dataclass(frozen=True, slots=True)
class ReducedWilsonModel:
    """One source-fixed reduced nonlinear action and EOM provider."""

    spatial: PreparedReducedWilsonSpatialAction
    source: UniformMagneticSourceSample
    branch: WilsonStationaryBranch
    connection: ReducedWilsonTimeConnection

    @property
    def backend(self) -> Any:
        return self.spatial.factory.exact_factory.quadrature.backend

    @property
    def quadrature(self) -> Any:
        return self.spatial.factory.exact_factory.quadrature

    @property
    def namespace(self) -> Any:
        return self.backend.namespace

    @property
    def overlap(self) -> Any:
        return self.spatial.one_electron.metric

    @property
    def level(self) -> ReducedWilsonLevel:
        return self.spatial.level

    def evaluate(self, coefficient_density: object) -> ReducedWilsonActionEvaluation:
        if not self.spatial.one_electron.metric_positive:
            raise FormulationError(
                "reduced Wilson metric is outside the admitted positive domain: "
                f"lambda_min={self.spatial.one_electron.metric_minimum_eigenvalue:.3e}"
            )
        exact_factory = self.spatial.factory.exact_factory
        density = exact_factory.hartree_evaluator._validated_density(coefficient_density)
        closure = self.spatial.closure.evaluate(density, self.level, self.branch)
        one = self.spatial.one_electron
        lower = hermitian_part(one.mechanical + closure.lower_matrix)
        xp = self.namespace
        kinetic_energy = expectation(density, one.kinetic, xp)
        nuclear_energy = expectation(density, one.nuclear_attraction, xp)
        one_energy = kinetic_energy + nuclear_energy
        electronic = one_energy + closure.energy_total_au
        nuclear_repulsion = self.backend.asarray(
            exact_factory.nuclear_repulsion_au,
            dtype=xp.float64,
        )
        return ReducedWilsonActionEvaluation(
            coefficient_density=density,
            overlap=one.metric,
            kinetic_matrix=one.kinetic,
            nuclear_attraction_matrix=one.nuclear_attraction,
            one_electron_matrix=one.mechanical,
            lower_mechanical_matrix=lower,
            closure=closure,
            energy_kinetic_au=kinetic_energy,
            energy_electron_nuclear_au=nuclear_energy,
            energy_one_electron_au=one_energy,
            energy_hartree_au=closure.energy_hartree_au,
            energy_exchange_correlation_au=closure.energy_exchange_correlation_au,
            energy_nuclear_repulsion_au=nuclear_repulsion,
            energy_electronic_au=electronic,
            energy_molecular_total_au=electronic + nuclear_repulsion,
            xc_potential_contraction_au=closure.xc_potential_contraction_au,
            branch=self.branch,
            level=self.level,
        )

    def dynamic_evaluation(
        self,
        coefficient_density: object,
    ) -> ReducedWilsonDynamicEvaluation:
        action = self.evaluate(coefficient_density)
        triple = EOMTriple(
            metric=self.overlap,
            hamiltonian_eom=action.lower_mechanical_matrix,
            connection=self.connection.connection,
        )
        return ReducedWilsonDynamicEvaluation(sample=self, action=action, triple=triple)

    def source_response(
        self,
        coefficient_density: object,
    ) -> ReducedWilsonSourceResponse:
        """Differentiate the selected mechanical action along this source rate."""

        density = self.spatial.factory.exact_factory.hartree_evaluator._validated_density(
            coefficient_density
        )
        factory = self.spatial.factory
        one = self.spatial.one_electron
        xp = self.namespace
        field_rate = self.backend.asarray(
            self.source.magnetic_field_dot_au,
            dtype=xp.float64,
        )
        barred_kinetic_rate = self.backend.zeros(
            one.barred_kinetic.shape,
            dtype=xp.complex128,
        )
        barred_nuclear_rate = self.backend.zeros(
            one.barred_nuclear_attraction.shape,
            dtype=xp.complex128,
        )
        if self.level.retains_first_magnetic_order:
            barred_kinetic_rate = xp.einsum(
                "x,xij->ij",
                field_rate,
                factory.magnetic_first_derivatives.kinetic,
                optimize=True,
            )
            barred_nuclear_rate = xp.einsum(
                "x,xij->ij",
                field_rate,
                factory.magnetic_first_derivatives.nuclear_attraction_triangle,
                optimize=True,
            )
        endpoint = one.endpoint_link
        endpoint_rate = self.connection.endpoint_link_dot
        one_electron_rate = hermitian_part(
            endpoint_rate * (one.barred_kinetic + one.barred_nuclear_attraction)
            + endpoint * (barred_kinetic_rate + barred_nuclear_rate)
        )
        density_zero_rate, density_first_rate, hartree_rate, xc_rate = (
            self.spatial.closure.source_energy_rates(
                density,
                self.level,
                self.branch,
                self.source,
                endpoint_rate,
            )
        )
        closure_rate = hartree_rate + xc_rate
        mechanical_rate = expectation(density, one_electron_rate, xp) + closure_rate
        return ReducedWilsonSourceResponse(
            one_electron_matrix_rate=one_electron_rate,
            density_zero_rate=density_zero_rate,
            density_first_rate=density_first_rate,
            hartree_energy_rate_au=hartree_rate,
            exchange_correlation_energy_rate_au=xc_rate,
            closure_energy_rate_au=closure_rate,
            mechanical_energy_rate_au=mechanical_rate,
        )

    def power(
        self,
        coefficient_density: object,
    ) -> ReducedWilsonPowerObservation:
        """Evaluate the on-shell rate of the selected mechanical energy."""

        dynamic = self.dynamic_evaluation(coefficient_density)
        density = dynamic.action.coefficient_density
        response = self.source_response(density)
        xp = self.namespace
        try:
            gamma = xp.linalg.solve(dynamic.triple.metric, dynamic.triple.connection)
        except Exception as exc:
            raise FormulationError("reduced-Wilson power connection solve failed") from exc
        lower = dynamic.action.lower_mechanical_matrix
        transport_operator = -gamma.conj().T @ lower - lower @ gamma
        transport_rate = expectation(density, transport_operator, xp)
        velocity = one_electron_velocity_density(
            density,
            dynamic.triple,
            self.backend,
            hbar=self.spatial.factory.exact_factory.hbar,
        )
        return ReducedWilsonPowerObservation(
            source_response=response,
            velocity_density=velocity,
            transport_energy_rate_au=transport_rate,
            source_power_au=response.mechanical_energy_rate_au + transport_rate,
        )

    def solve(
        self,
        *,
        policy: StationarySCFPolicy | None = None,
        initial_coefficients: object | None = None,
    ) -> ExactWilsonStationaryState[ReducedWilsonActionEvaluation]:
        """Solve this reduced action with the common Wilson SCF algorithm."""

        return solve_wilson_stationary_model(
            self,
            policy=policy,
            initial_coefficients=initial_coefficients,
        )


def prepare_reduced_wilson_factory(
    exact_factory: ExactWilsonStationaryFactory,
) -> PreparedReducedWilsonFactory:
    """Prepare the common analytic first-order data for all reduced levels."""

    if not isinstance(exact_factory, ExactWilsonStationaryFactory):
        raise TypeError("exact_factory must be an ExactWilsonStationaryFactory")
    return PreparedReducedWilsonFactory(
        exact_factory=exact_factory,
        magnetic_first_derivatives=evaluate_magnetic_one_electron_first_derivatives(
            exact_factory.quadrature,
            charge=exact_factory.charge,
            mass=exact_factory.mass,
            hbar=exact_factory.hbar,
        ),
        electric_e1=evaluate_uniform_electric_e1_tensor(
            exact_factory.quadrature,
            charge=exact_factory.charge,
            hbar=exact_factory.hbar,
        ),
    )


def _time_connection(
    spatial: PreparedReducedWilsonSpatialAction,
    source: UniformMagneticSourceSample,
) -> ReducedWilsonTimeConnection:
    factory = spatial.factory
    exact_factory = factory.exact_factory
    quadrature = exact_factory.quadrature
    backend = quadrature.backend
    xp = backend.namespace
    reference = quadrature.reference
    geometry = build_magnetic_pair_geometry(
        reference.core_operators.nuclei.coordinates_au,
        reference.anchor_topology.ao_to_atom,
        backend,
    )
    anchors = geometry.ao_anchor_coordinates_au
    endpoint = spatial.one_electron.endpoint_link
    endpoint_rate_line = source.gauge_rate.straight_line_integrals(
        anchors[None, :, :],
        anchors[:, None, :],
        backend,
    )
    prefactor = 1j * exact_factory.charge / exact_factory.hbar
    endpoint_dot = prefactor * endpoint_rate_line * endpoint
    barred_dot = backend.zeros(spatial.one_electron.barred_metric.shape, dtype=xp.complex128)
    if spatial.level.retains_first_magnetic_order:
        field_dot = backend.asarray(source.magnetic_field_dot_au, dtype=xp.float64)
        barred_dot = xp.einsum(
            "x,xij->ij",
            field_dot,
            factory.magnetic_first_derivatives.metric,
            optimize=True,
        )
    metric = spatial.one_electron.metric
    metric_dot = hermitian_part(
        endpoint_dot * spatial.one_electron.barred_metric + endpoint * barred_dot
    )
    scalar = source.scalar_potential(anchors, backend)
    sigma = xp.diag(prefactor * scalar)
    covariant_metric_rate = metric_dot + sigma @ metric - metric @ sigma
    electric_residual = backend.zeros(metric.shape, dtype=xp.complex128)
    if spatial.level.retains_electric_increment:
        electric_pair = source.electric_field(geometry.pair_midpoints_au, backend)
        electric_residual = (
            (-1j / exact_factory.hbar)
            * endpoint
            * xp.einsum(
                "ijx,xij->ij",
                electric_pair,
                factory.electric_e1.central_dipoles,
                optimize=True,
            )
        )
    connection = metric @ sigma + 0.5 * covariant_metric_rate + electric_residual
    scale = xp.maximum(xp.asarray(1.0), xp.linalg.norm(metric_dot))
    compatibility = backend.scalar_to_float(
        xp.linalg.norm(metric_dot - connection - connection.conj().T) / scale
    )
    return ReducedWilsonTimeConnection(
        connection=connection,
        metric_dot=metric_dot,
        endpoint_link_dot=endpoint_dot,
        site_connection=sigma,
        covariant_metric_rate=covariant_metric_rate,
        electric_residual=electric_residual,
        metric_compatibility_residual=compatibility,
    )


def _validate_density_domain(
    density: Any,
    evaluator: WilsonLDAEvaluator,
    backend: Any,
    label: str,
) -> None:
    xp = backend.namespace
    minimum = backend.scalar_to_float(xp.min(density))
    scale = backend.scalar_to_float(xp.maximum(xp.asarray(1.0), xp.max(xp.abs(density))))
    if minimum < -evaluator.negative_density_relative_tolerance * scale:
        raise FormulationError(
            f"{label} is outside the declared pure-LDA density domain: minimum={minimum:.3e}"
        )


def _require_real(value: Any, backend: Any, name: str) -> None:
    xp = backend.namespace
    imaginary = backend.scalar_to_float(xp.max(xp.abs(xp.imag(value))))
    scale = backend.scalar_to_float(xp.maximum(xp.asarray(1.0), xp.max(xp.abs(xp.real(value)))))
    if imaginary > 2.0e-10 * scale:
        raise FormulationError(f"{name} has unresolved imaginary part {imaginary:.3e}")


def _level(value: ReducedWilsonLevel | str) -> ReducedWilsonLevel:
    try:
        return ReducedWilsonLevel(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"unsupported reduced Wilson level {value!r}") from exc


def _branch(value: WilsonStationaryBranch | str) -> WilsonStationaryBranch:
    try:
        return WilsonStationaryBranch(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"unsupported stationary branch {value!r}") from exc
