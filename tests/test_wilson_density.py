from __future__ import annotations

import numpy as np
import pytest

from aion.config import BackendConfig
from aion.electromagnetism import (
    AffineGaugeDifferenceVariation,
    AffineMagneticGauge,
    MagneticGaugeKind,
    UniformMagneticField,
    affine_gauge_difference_potential,
    build_magnetic_pair_geometry,
)
from aion.electronic_structure import (
    AOGridPolicy,
    contract_wilson_density_block,
    estimate_wilson_density_block_bytes,
    evaluate_exact_static_magnetic_one_electron_matrices,
    evaluate_exact_uniform_magnetic_wilson_density,
    evaluate_exact_uniform_magnetic_wilson_density_matter_direction,
    evaluate_exact_uniform_magnetic_wilson_density_source_direction,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
)
from aion.errors import ConfigurationError
from test_reference_integration import molecular_config

pytestmark = pytest.mark.integration


def _prepared_h2() -> tuple[object, object]:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(2),
        block_size=257,
    )
    return reference, quadrature


def test_exact_wilson_density_direct_factorized_normalization_and_matrix_oracle() -> None:
    reference, quadrature = _prepared_h2()
    field = UniformMagneticField((0.013, -0.009, 0.017))
    gauge = AffineMagneticGauge(field, origin_au=(0.17, -0.31, 0.23))
    coefficient_density = reference.ground_state.density.astype(np.complex128)

    result = evaluate_exact_uniform_magnetic_wilson_density(
        quadrature,
        coefficient_density,
        gauge,
    )
    matrix_oracle = evaluate_exact_static_magnetic_one_electron_matrices(
        quadrature,
        (field,),
        direct_gauges=(gauge,),
    )[0]

    assert result.density_direct_factorized_residual < 2.0e-13
    assert result.overlap_direct_factorized_residual < 2.0e-13
    assert result.density_direct_imaginary_max_abs < 2.0e-15
    assert result.density_factorized_imaginary_max_abs < 2.0e-15
    assert result.density_direct_real_minimum >= -2.0e-15
    assert result.density_factorized_real_minimum >= -2.0e-15
    np.testing.assert_allclose(
        result.density_direct,
        result.density_factorized,
        atol=2.0e-13,
        rtol=2.0e-13,
    )
    np.testing.assert_allclose(
        result.overlap_factorized_grid,
        matrix_oracle.lower_exact_grid.overlap,
        atol=3.0e-13,
        rtol=3.0e-13,
    )
    np.testing.assert_allclose(
        result.overlap_stable,
        matrix_oracle.lower_exact.overlap,
        atol=3.0e-13,
        rtol=3.0e-13,
    )
    np.testing.assert_allclose(
        result.particle_number_direct_integral,
        result.particle_number_direct_metric,
        atol=2.0e-13,
        rtol=2.0e-13,
    )
    np.testing.assert_allclose(
        result.particle_number_factorized_integral,
        result.particle_number_factorized_metric,
        atol=2.0e-13,
        rtol=2.0e-13,
    )


def test_density_block_is_invariant_under_general_complex_coefficient_frame() -> None:
    reference, quadrature = _prepared_h2()
    backend = quadrature.backend
    gauge = AffineMagneticGauge(
        UniformMagneticField((0.013, -0.009, 0.017)),
        origin_au=(0.17, -0.31, 0.23),
    )
    geometry = build_magnetic_pair_geometry(
        reference.core_operators.nuclei.coordinates_au,
        reference.anchor_topology.ao_to_atom,
        backend,
    )
    block = next(quadrature.blocks())
    line = gauge.anchor_to_point_line_integrals(
        geometry.ao_anchor_coordinates_au,
        block.coordinates_au,
        backend,
    )
    frame = np.exp(-1j * line) * block.values
    density = reference.ground_state.density.astype(np.complex128)
    change = np.asarray(
        ((1.1 + 0.2j, -0.3 + 0.1j), (0.25 - 0.15j, 0.9 - 0.1j)),
        dtype=np.complex128,
    )
    inverse = np.linalg.inv(change)
    transformed_frame = frame @ change
    transformed_density = inverse @ density @ inverse.conj().T
    np.testing.assert_allclose(
        contract_wilson_density_block(frame, density, backend),
        contract_wilson_density_block(transformed_frame, transformed_density, backend),
        atol=2.0e-13,
        rtol=2.0e-13,
    )


@pytest.mark.parametrize("imaginary", (False, True))
def test_wilson_density_matter_direction_matches_coefficient_difference(
    imaginary: bool,
) -> None:
    reference, quadrature = _prepared_h2()
    gauge = AffineMagneticGauge(
        UniformMagneticField((0.013, -0.009, 0.017)),
        origin_au=(0.17, -0.31, 0.23),
    )
    coefficients = reference.ground_state.coefficients.astype(np.complex128)
    occupations = reference.ground_state.occupations
    direction = np.asarray(((0.17, -0.08), (0.11, 0.19)), dtype=np.complex128)
    if imaginary:
        direction *= 1j
    direction /= np.linalg.norm(direction)
    analytic = evaluate_exact_uniform_magnetic_wilson_density_matter_direction(
        quadrature,
        coefficients,
        occupations,
        direction,
        gauge,
    )
    step = 1.0e-4
    densities = []
    for sign in (1.0, -1.0):
        displaced = coefficients + sign * step * direction
        density = np.einsum(
            "mi,i,ni->mn",
            displaced,
            occupations,
            displaced.conj(),
            optimize=True,
        )
        densities.append(
            evaluate_exact_uniform_magnetic_wilson_density(
                quadrature,
                density,
                gauge,
            ).density_direct
        )
    finite = (densities[0] - densities[1]) / (2.0 * step)
    np.testing.assert_allclose(
        analytic.density_direction,
        finite,
        atol=3.0e-11,
        rtol=3.0e-11,
    )
    assert analytic.density_direction_imaginary_max_abs < 2.0e-15


def _density_with_line_perturbation(
    quadrature: object,
    density: np.ndarray,
    gauge: AffineMagneticGauge,
    direction: object,
    step: float,
) -> np.ndarray:
    backend = quadrature.backend
    reference = quadrature.reference
    geometry = build_magnetic_pair_geometry(
        reference.core_operators.nuclei.coordinates_au,
        reference.anchor_topology.ao_to_atom,
        backend,
    )
    result = np.empty((quadrature.grid.npoints,), dtype=np.complex128)
    for block in quadrature.blocks():
        base = gauge.anchor_to_point_line_integrals(
            geometry.ao_anchor_coordinates_au,
            block.coordinates_au,
            backend,
        )
        delta = direction.straight_line_integrals(
            geometry.ao_anchor_coordinates_au[None, :, :],
            block.coordinates_au[:, None, :],
            backend,
        )
        frame = np.exp(-1j * (base + step * delta)) * block.values
        result[block.start : block.stop] = contract_wilson_density_block(
            frame,
            density,
            backend,
        )
    return result


def test_wilson_density_physical_and_pure_gauge_source_directions() -> None:
    reference, quadrature = _prepared_h2()
    density = reference.ground_state.density.astype(np.complex128)
    field = UniformMagneticField((0.013, -0.009, 0.017))
    gauge = AffineMagneticGauge(field, origin_au=(0.17, -0.31, 0.23))
    physical = AffineMagneticGauge(
        UniformMagneticField((-0.4, 0.7, 0.2)),
        origin_au=gauge.origin_au,
    )
    landau = AffineMagneticGauge(
        field,
        kind=MagneticGaugeKind.LANDAU,
        origin_au=(-0.21, 0.37, -0.16),
        landau_axis=(field.magnetic_field_au[1], -field.magnetic_field_au[0], 0.0),
    )
    pure_gauge = AffineGaugeDifferenceVariation(landau, gauge)
    step = 1.0e-4
    for direction in (physical, pure_gauge):
        analytic = evaluate_exact_uniform_magnetic_wilson_density_source_direction(
            quadrature,
            density,
            gauge,
            direction,
        )
        plus = _density_with_line_perturbation(
            quadrature,
            density,
            gauge,
            direction,
            step,
        )
        minus = _density_with_line_perturbation(
            quadrature,
            density,
            gauge,
            direction,
            -step,
        )
        np.testing.assert_allclose(
            analytic.density_direction,
            (plus - minus) / (2.0 * step),
            atol=5.0e-11,
            rtol=5.0e-11,
        )
        np.testing.assert_allclose(
            analytic.particle_number_direction_integral,
            analytic.particle_number_direction_metric_grid,
            atol=2.0e-13,
            rtol=2.0e-13,
        )
        assert analytic.density_direction_imaginary_max_abs < 2.0e-15


def test_exact_wilson_density_is_invariant_between_affine_gauge_representatives() -> None:
    reference, quadrature = _prepared_h2()
    field = UniformMagneticField((0.013, -0.009, 0.017))
    symmetric = AffineMagneticGauge(field, origin_au=(0.17, -0.31, 0.23))
    landau = AffineMagneticGauge(
        field,
        kind=MagneticGaugeKind.LANDAU,
        origin_au=(-0.21, 0.37, -0.16),
        landau_axis=(field.magnetic_field_au[1], -field.magnetic_field_au[0], 0.0),
    )
    coefficient_density = reference.ground_state.density.astype(np.complex128)
    anchors = reference.core_operators.nuclei.coordinates_au[reference.anchor_topology.ao_to_atom]
    gauge_function_at_anchors = affine_gauge_difference_potential(
        landau,
        symmetric,
        anchors,
        quadrature.backend,
    )
    coefficient_unitary = np.exp(-1j * gauge_function_at_anchors)
    transformed_density = (
        coefficient_unitary[:, None] * coefficient_density * coefficient_unitary.conj()[None, :]
    )

    symmetric_result = evaluate_exact_uniform_magnetic_wilson_density(
        quadrature,
        coefficient_density,
        symmetric,
    )
    landau_result = evaluate_exact_uniform_magnetic_wilson_density(
        quadrature,
        transformed_density,
        landau,
    )
    np.testing.assert_allclose(
        symmetric_result.density_direct,
        landau_result.density_direct,
        atol=3.0e-13,
        rtol=3.0e-13,
    )
    np.testing.assert_allclose(
        symmetric_result.particle_number_direct_integral,
        landau_result.particle_number_direct_integral,
        atol=3.0e-13,
        rtol=3.0e-13,
    )


def test_wilson_density_rejects_nonhermitian_density_and_small_memory_budget() -> None:
    reference, quadrature = _prepared_h2()
    gauge = AffineMagneticGauge(UniformMagneticField((0.0, 0.0, 0.0)))
    density = reference.ground_state.density.astype(np.complex128)
    invalid = density.copy()
    invalid[0, 1] += 0.01j
    with pytest.raises(ConfigurationError, match="must be Hermitian"):
        evaluate_exact_uniform_magnetic_wilson_density(quadrature, invalid, gauge)

    required = estimate_wilson_density_block_bytes(
        quadrature.block_size,
        reference.core_operators.nao,
    )
    with pytest.raises(ConfigurationError, match="exceeding memory_budget_bytes"):
        evaluate_exact_uniform_magnetic_wilson_density(
            quadrature,
            density,
            gauge,
            memory_budget_bytes=required - 1,
        )
