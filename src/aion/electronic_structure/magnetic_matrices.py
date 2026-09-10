"""Exact and B1/B2 uniform-magnetic one-electron matrix contractions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aion.electromagnetism.magnetic import (
    AffineMagneticGauge,
    UniformMagneticField,
    anchored_vectors,
    build_magnetic_pair_geometry,
    endpoint_links,
    triangle_phases,
)
from aion.electronic_structure.ao_quadrature import AOQuadrature
from aion.electronic_structure.local_potentials import (
    NuclearAttractionProvider,
    bind_local_potential,
)
from aion.electronic_structure.pyscf_rks import reconstruct_mean_field
from aion.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class ScalarMagneticHierarchy:
    """Exact and amplitude-expanded barred matrices for one scalar operator."""

    zero: Any
    quadrature_zero: Any
    first_F: Any
    second_F2: Any
    exact_grid: Any
    exact: Any

    @property
    def first(self) -> Any:
        return self.first_F

    @property
    def second(self) -> Any:
        return self.second_F2

    @property
    def b1(self) -> Any:
        return self.zero + self.first_F

    @property
    def b2(self) -> Any:
        return self.b1 + self.second_F2


@dataclass(frozen=True, slots=True)
class KineticExactSectors:
    """Exact-factor kinetic sectors before their authoritative sum."""

    pp: Any
    pC: Any
    Cp: Any
    C2: Any

    @property
    def total(self) -> Any:
        return self.pp + self.pC + self.Cp + self.C2


@dataclass(frozen=True, slots=True)
class KineticMagneticHierarchy:
    """Exact and decomposed B1/B2 barred kinetic matrices."""

    zero: Any
    quadrature_zero: Any
    first_F: Any
    first_pC: Any
    first_Cp: Any
    second_F2: Any
    second_F_pC: Any
    second_F_Cp: Any
    second_C2: Any
    quadrature_exact_sectors: KineticExactSectors
    exact_sectors: KineticExactSectors
    exact_grid: Any
    exact: Any

    @property
    def first_C(self) -> Any:
        return self.first_pC + self.first_Cp

    @property
    def first(self) -> Any:
        return self.first_F + self.first_C

    @property
    def second_FC(self) -> Any:
        return self.second_F_pC + self.second_F_Cp

    @property
    def second(self) -> Any:
        return self.second_F2 + self.second_FC + self.second_C2

    @property
    def b1(self) -> Any:
        return self.zero + self.first

    @property
    def b2(self) -> Any:
        return self.b1 + self.second


@dataclass(frozen=True, slots=True)
class OneElectronLowerMatrices:
    """Endpoint-dressed lower matrices for the MB3 one-electron closure."""

    overlap: Any
    kinetic: Any
    nuclear_attraction: Any


@dataclass(frozen=True, slots=True)
class DirectGaugeOracle:
    """Matrices assembled directly from independently Wilson-dressed AOs."""

    gauge: AffineMagneticGauge
    lower_grid: OneElectronLowerMatrices


@dataclass(frozen=True, slots=True)
class MagneticOneElectronResult:
    """One field's complete exact/B1/B2 static one-electron result."""

    field: UniformMagneticField
    endpoint_link: Any
    overlap: ScalarMagneticHierarchy
    kinetic: KineticMagneticHierarchy
    nuclear_attraction: ScalarMagneticHierarchy
    direct_oracle: DirectGaugeOracle | None
    reference_fingerprint_sha256: str
    grid_fingerprint_sha256: str
    backend: str
    device_index: int | None
    charge: float
    mass: float
    hbar: float

    @property
    def lower_exact(self) -> OneElectronLowerMatrices:
        """Return exact-link lower matrices using stable corrected barred values."""

        return OneElectronLowerMatrices(
            overlap=self.endpoint_link * self.overlap.exact,
            kinetic=self.endpoint_link * self.kinetic.exact,
            nuclear_attraction=self.endpoint_link * self.nuclear_attraction.exact,
        )

    @property
    def lower_exact_grid(self) -> OneElectronLowerMatrices:
        """Return exact-link raw-grid matrices for comparison with the direct oracle."""

        return OneElectronLowerMatrices(
            overlap=self.endpoint_link * self.overlap.exact_grid,
            kinetic=self.endpoint_link * self.kinetic.exact_grid,
            nuclear_attraction=self.endpoint_link * self.nuclear_attraction.exact_grid,
        )


@dataclass(frozen=True, slots=True)
class GIAOOneElectronDerivatives:
    """Independent libcint lower, endpoint, and barred Cartesian B derivatives."""

    lower: OneElectronLowerMatrices
    endpoint: OneElectronLowerMatrices
    barred: OneElectronLowerMatrices


@dataclass(slots=True)
class _FieldAccumulators:
    overlap_zero: Any
    overlap_first: Any
    overlap_second: Any
    overlap_exact_correction: Any
    nuclear_zero: Any
    nuclear_first: Any
    nuclear_second: Any
    nuclear_exact_correction: Any
    kinetic_zero: Any
    kinetic_first_F: Any
    kinetic_first_pC: Any
    kinetic_first_Cp: Any
    kinetic_second_F2: Any
    kinetic_second_F_pC: Any
    kinetic_second_F_Cp: Any
    kinetic_second_C2: Any
    kinetic_exact_pp_correction: Any
    kinetic_exact_pC: Any
    kinetic_exact_Cp: Any
    kinetic_exact_C2: Any
    direct_overlap: Any
    direct_kinetic: Any
    direct_nuclear: Any


def evaluate_magnetic_one_electron_matrices(
    quadrature: AOQuadrature,
    magnetic_fields: Sequence[UniformMagneticField],
    *,
    direct_gauges: Sequence[AffineMagneticGauge] | None = None,
    charge: float = -1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    memory_budget_bytes: int | None = None,
    include_direct_oracle: bool = True,
) -> tuple[MagneticOneElectronResult, ...]:
    """Evaluate several fields while sampling each AO block exactly once.

    Raw quadrature matrices are retained for the independently assembled
    direct-gauge comparison. Authoritative exact barred matrices replace the
    raw zero-field constant with the stored analytic PySCF matrix.
    """

    if not isinstance(quadrature, AOQuadrature):
        raise TypeError("quadrature must be an AOQuadrature")
    fields = tuple(magnetic_fields)
    if not fields or not all(isinstance(field, UniformMagneticField) for field in fields):
        raise ConfigurationError("magnetic_fields must contain UniformMagneticField values")
    checked_charge = _finite_parameter(charge, "charge")
    checked_mass = _positive_parameter(mass, "mass")
    checked_hbar = _positive_parameter(hbar, "hbar")
    if not isinstance(include_direct_oracle, bool):
        raise ConfigurationError("include_direct_oracle must be boolean")
    if memory_budget_bytes is not None:
        if (
            isinstance(memory_budget_bytes, bool)
            or not isinstance(memory_budget_bytes, int)
            or memory_budget_bytes <= 0
        ):
            raise ConfigurationError("magnetic memory_budget_bytes must be a positive integer")
        required = estimate_magnetic_block_bytes(
            quadrature.block_size, quadrature.reference.core_operators.nao
        )
        if required > memory_budget_bytes:
            raise ConfigurationError(
                f"magnetic block requires {required} bytes, exceeding "
                f"memory_budget_bytes={memory_budget_bytes}"
            )

    if direct_gauges is None:
        origin = quadrature.reference.config.molecule.electromagnetic_origin.position_au
        gauges: tuple[AffineMagneticGauge, ...] = tuple(
            AffineMagneticGauge(field, origin_au=origin) for field in fields
        )
    else:
        gauges = tuple(direct_gauges)
        if len(gauges) != len(fields):
            raise ConfigurationError("direct_gauges must match magnetic_fields in length")
        for field, gauge in zip(fields, gauges, strict=True):
            if not isinstance(gauge, AffineMagneticGauge) or gauge.field != field:
                raise ConfigurationError("every direct gauge must represent its matching field")

    backend = quadrature.backend
    xp = backend.namespace
    reference = quadrature.reference
    geometry = build_magnetic_pair_geometry(
        reference.core_operators.nuclei.coordinates_au,
        reference.anchor_topology.ao_to_atom,
        backend,
    )
    nao = reference.core_operators.nao
    analytic_overlap = backend.asarray(reference.core_operators.overlap, dtype=xp.complex128)
    analytic_kinetic = backend.asarray(
        (checked_hbar * checked_hbar / checked_mass) * reference.core_operators.kinetic,
        dtype=xp.complex128,
    )
    nuclear_provider = bind_local_potential(
        NuclearAttractionProvider(), reference, backend
    )
    analytic_nuclear = nuclear_provider.zero_matrix_au
    accumulators = tuple(_new_accumulators(nao, backend) for _ in fields)

    for block in quadrature.blocks():
        values = block.values
        gradients = xp.moveaxis(block.gradients, 0, -1)
        bare_momentum = -1j * checked_hbar * gradients
        nuclear_potential = nuclear_provider.values_au(block.coordinates_au)
        for field, gauge, accumulator in zip(fields, gauges, accumulators, strict=True):
            phase = triangle_phases(
                block.coordinates_au,
                geometry,
                field,
                backend,
                charge=checked_charge,
                hbar=checked_hbar,
            )
            exact_minus_one = xp.expm1(1j * phase)
            exact_factor = 1.0 + exact_minus_one
            first_factor = 1j * phase
            second_factor = -0.5 * phase * phase
            anchored = anchored_vectors(block.coordinates_au, geometry, field, backend)
            anchored_momentum = checked_charge * anchored * values[:, :, None]

            accumulator.overlap_zero += _ordinary_pair(
                values, values, block.weights_au, xp
            )
            accumulator.overlap_first += _factorized_pair(
                values, values, block.weights_au, first_factor, xp
            )
            accumulator.overlap_second += _factorized_pair(
                values, values, block.weights_au, second_factor, xp
            )
            accumulator.overlap_exact_correction += _factorized_pair(
                values, values, block.weights_au, exact_minus_one, xp
            )

            potential_weights = block.weights_au * nuclear_potential
            accumulator.nuclear_zero += _ordinary_pair(
                values, values, potential_weights, xp
            )
            accumulator.nuclear_first += _factorized_pair(
                values, values, potential_weights, first_factor, xp
            )
            accumulator.nuclear_second += _factorized_pair(
                values, values, potential_weights, second_factor, xp
            )
            accumulator.nuclear_exact_correction += _factorized_pair(
                values, values, potential_weights, exact_minus_one, xp
            )

            kinetic_scale = 1.0 / (2.0 * checked_mass)
            accumulator.kinetic_zero += kinetic_scale * _ordinary_vector_pair(
                bare_momentum, bare_momentum, block.weights_au, xp
            )
            accumulator.kinetic_first_F += kinetic_scale * _factorized_vector_pair(
                bare_momentum, bare_momentum, block.weights_au, first_factor, xp
            )
            accumulator.kinetic_first_pC += kinetic_scale * _ordinary_vector_pair(
                bare_momentum, anchored_momentum, block.weights_au, xp
            )
            accumulator.kinetic_first_Cp += kinetic_scale * _ordinary_vector_pair(
                anchored_momentum, bare_momentum, block.weights_au, xp
            )
            accumulator.kinetic_second_F2 += kinetic_scale * _factorized_vector_pair(
                bare_momentum, bare_momentum, block.weights_au, second_factor, xp
            )
            accumulator.kinetic_second_F_pC += kinetic_scale * _factorized_vector_pair(
                bare_momentum, anchored_momentum, block.weights_au, first_factor, xp
            )
            accumulator.kinetic_second_F_Cp += kinetic_scale * _factorized_vector_pair(
                anchored_momentum, bare_momentum, block.weights_au, first_factor, xp
            )
            accumulator.kinetic_second_C2 += kinetic_scale * _ordinary_vector_pair(
                anchored_momentum, anchored_momentum, block.weights_au, xp
            )
            accumulator.kinetic_exact_pp_correction += (
                kinetic_scale
                * _factorized_vector_pair(
                    bare_momentum,
                    bare_momentum,
                    block.weights_au,
                    exact_minus_one,
                    xp,
                )
            )
            accumulator.kinetic_exact_pC += kinetic_scale * _factorized_vector_pair(
                bare_momentum, anchored_momentum, block.weights_au, exact_factor, xp
            )
            accumulator.kinetic_exact_Cp += kinetic_scale * _factorized_vector_pair(
                anchored_momentum, bare_momentum, block.weights_au, exact_factor, xp
            )
            accumulator.kinetic_exact_C2 += kinetic_scale * _factorized_vector_pair(
                anchored_momentum,
                anchored_momentum,
                block.weights_au,
                exact_factor,
                xp,
            )

            if include_direct_oracle:
                line_integrals = gauge.anchor_to_point_line_integrals(
                    geometry.ao_anchor_coordinates_au, block.coordinates_au, backend
                )
                line_gradients = gauge.anchor_to_point_line_integral_gradients(
                    geometry.ao_anchor_coordinates_au, block.coordinates_au, backend
                )
                point_vector_potential = gauge.vector_potential(
                    block.coordinates_au, backend
                )
                direct_residual = line_gradients - point_vector_potential[:, None, :]
                wilson = xp.exp((1j * checked_charge / checked_hbar) * line_integrals)
                dressed_values = wilson * values
                dressed_momentum = wilson[:, :, None] * (
                    bare_momentum
                    + checked_charge * direct_residual * values[:, :, None]
                )
                accumulator.direct_overlap += _ordinary_pair(
                    dressed_values, dressed_values, block.weights_au, xp
                )
                accumulator.direct_kinetic += kinetic_scale * _ordinary_vector_pair(
                    dressed_momentum, dressed_momentum, block.weights_au, xp
                )
                accumulator.direct_nuclear += _ordinary_pair(
                    dressed_values, dressed_values, potential_weights, xp
                )

    results: list[MagneticOneElectronResult] = []
    for field, gauge, accumulator in zip(fields, gauges, accumulators, strict=True):
        endpoint = endpoint_links(
            gauge,
            geometry,
            backend,
            charge=checked_charge,
            hbar=checked_hbar,
        )
        overlap_exact_grid = accumulator.overlap_zero + accumulator.overlap_exact_correction
        overlap_exact = analytic_overlap + accumulator.overlap_exact_correction
        nuclear_exact_grid = accumulator.nuclear_zero + accumulator.nuclear_exact_correction
        nuclear_exact = analytic_nuclear + accumulator.nuclear_exact_correction
        grid_sectors = KineticExactSectors(
            pp=accumulator.kinetic_zero + accumulator.kinetic_exact_pp_correction,
            pC=accumulator.kinetic_exact_pC,
            Cp=accumulator.kinetic_exact_Cp,
            C2=accumulator.kinetic_exact_C2,
        )
        corrected_sectors = KineticExactSectors(
            pp=analytic_kinetic + accumulator.kinetic_exact_pp_correction,
            pC=accumulator.kinetic_exact_pC,
            Cp=accumulator.kinetic_exact_Cp,
            C2=accumulator.kinetic_exact_C2,
        )
        overlap = ScalarMagneticHierarchy(
            zero=analytic_overlap,
            quadrature_zero=accumulator.overlap_zero,
            first_F=accumulator.overlap_first,
            second_F2=accumulator.overlap_second,
            exact_grid=overlap_exact_grid,
            exact=overlap_exact,
        )
        kinetic = KineticMagneticHierarchy(
            zero=analytic_kinetic,
            quadrature_zero=accumulator.kinetic_zero,
            first_F=accumulator.kinetic_first_F,
            first_pC=accumulator.kinetic_first_pC,
            first_Cp=accumulator.kinetic_first_Cp,
            second_F2=accumulator.kinetic_second_F2,
            second_F_pC=accumulator.kinetic_second_F_pC,
            second_F_Cp=accumulator.kinetic_second_F_Cp,
            second_C2=accumulator.kinetic_second_C2,
            quadrature_exact_sectors=grid_sectors,
            exact_sectors=corrected_sectors,
            exact_grid=grid_sectors.total,
            exact=corrected_sectors.total,
        )
        nuclear = ScalarMagneticHierarchy(
            zero=analytic_nuclear,
            quadrature_zero=accumulator.nuclear_zero,
            first_F=accumulator.nuclear_first,
            second_F2=accumulator.nuclear_second,
            exact_grid=nuclear_exact_grid,
            exact=nuclear_exact,
        )
        results.append(
            MagneticOneElectronResult(
                field=field,
                endpoint_link=endpoint,
                overlap=overlap,
                kinetic=kinetic,
                nuclear_attraction=nuclear,
                direct_oracle=(
                    DirectGaugeOracle(
                        gauge=gauge,
                        lower_grid=OneElectronLowerMatrices(
                            overlap=accumulator.direct_overlap,
                            kinetic=accumulator.direct_kinetic,
                            nuclear_attraction=accumulator.direct_nuclear,
                        ),
                    )
                    if include_direct_oracle
                    else None
                ),
                reference_fingerprint_sha256=reference.fingerprint_sha256,
                grid_fingerprint_sha256=quadrature.grid.fingerprint_sha256,
                backend=quadrature.backend_config.kind.value,
                device_index=quadrature.backend_config.device_index,
                charge=checked_charge,
                mass=checked_mass,
                hbar=checked_hbar,
            )
        )
    backend.synchronize()
    return tuple(results)


def pyscf_giao_one_electron_derivatives(
    reference: Any,
) -> GIAOOneElectronDerivatives:
    """Return independent electron-AU symmetric-gauge libcint B derivatives."""

    from aion.config import BackendConfig
    from aion.electronic_structure.data import PreparedReference

    if not isinstance(reference, PreparedReference):
        raise TypeError("reference must be a PreparedReference")
    model = reconstruct_mean_field(reference, BackendConfig())
    molecule = model.mol
    overlap_lower = -1j * np.asarray(molecule.intor("int1e_igovlp", comp=3))
    kinetic_lower = -1j * (
        np.asarray(molecule.intor("int1e_igkin", comp=3))
        + 0.5 * np.asarray(molecule.intor("int1e_giao_irjxp", comp=3))
    )
    nuclear_lower = -1j * np.asarray(molecule.intor("int1e_ignuc", comp=3))
    anchors = reference.core_operators.nuclei.coordinates_au[
        reference.anchor_topology.ao_to_atom
    ]
    endpoint_derivatives = []
    from aion.backends import NumPyBackend

    cpu = NumPyBackend()
    for direction in np.eye(3):
        field = UniformMagneticField(
            (float(direction[0]), float(direction[1]), float(direction[2]))
        )
        gauge = AffineMagneticGauge(field)
        line = gauge.straight_line_integrals(
            anchors[None, :, :], anchors[:, None, :], cpu
        )
        endpoint_derivatives.append(-1j * line)
    theta_derivative = np.asarray(endpoint_derivatives)
    overlap_endpoint = theta_derivative * reference.core_operators.overlap[None, :, :]
    kinetic_endpoint = theta_derivative * reference.core_operators.kinetic[None, :, :]
    nuclear_endpoint = (
        theta_derivative * reference.core_operators.nuclear_attraction[None, :, :]
    )
    lower = OneElectronLowerMatrices(overlap_lower, kinetic_lower, nuclear_lower)
    endpoint = OneElectronLowerMatrices(
        overlap_endpoint, kinetic_endpoint, nuclear_endpoint
    )
    barred = OneElectronLowerMatrices(
        overlap_lower - overlap_endpoint,
        kinetic_lower - kinetic_endpoint,
        nuclear_lower - nuclear_endpoint,
    )
    return GIAOOneElectronDerivatives(lower=lower, endpoint=endpoint, barred=barred)


def estimate_magnetic_block_bytes(block_size: int, nao: int) -> int:
    """Conservative peak temporary bytes for one field within one AO block."""

    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ConfigurationError("magnetic block_size must be a positive integer")
    if isinstance(nao, bool) or not isinstance(nao, int) or nao <= 0:
        raise ConfigurationError("nao must be a positive integer")
    return block_size * (64 * nao * nao + 256 * nao + 64)


def _new_accumulators(nao: int, backend: Any) -> _FieldAccumulators:
    def zero() -> Any:
        return backend.zeros((nao, nao), dtype=backend.namespace.complex128)

    return _FieldAccumulators(*(zero() for _ in range(23)))


def _ordinary_pair(left: Any, right: Any, weights: Any, xp: Any) -> Any:
    return xp.einsum("p,pm,pn->mn", weights, left.conj(), right, optimize=True)


def _factorized_pair(
    left: Any, right: Any, weights: Any, factor: Any, xp: Any
) -> Any:
    return xp.einsum(
        "p,pm,pn,pmn->mn", weights, left.conj(), right, factor, optimize=True
    )


def _ordinary_vector_pair(left: Any, right: Any, weights: Any, xp: Any) -> Any:
    return xp.einsum(
        "p,pmx,pnx->mn", weights, left.conj(), right, optimize=True
    )


def _factorized_vector_pair(
    left: Any, right: Any, weights: Any, factor: Any, xp: Any
) -> Any:
    return xp.einsum(
        "p,pmx,pnx,pmn->mn",
        weights,
        left.conj(),
        right,
        factor,
        optimize=True,
    )


def _finite_parameter(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be finite")
    try:
        checked = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be finite") from exc
    if not np.isfinite(checked):
        raise ConfigurationError(f"{name} must be finite")
    return checked


def _positive_parameter(value: float, name: str) -> float:
    checked = _finite_parameter(value, name)
    if checked <= 0.0:
        raise ConfigurationError(f"{name} must be positive")
    return checked
