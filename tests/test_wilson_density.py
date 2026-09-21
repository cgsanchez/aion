from __future__ import annotations

import numpy as np
import pytest

from aion.config import BackendConfig
from aion.electromagnetism import (
    AffineMagneticGauge,
    MagneticGaugeKind,
    UniformMagneticField,
    affine_gauge_difference_potential,
)
from aion.electronic_structure import (
    AOGridPolicy,
    estimate_wilson_density_block_bytes,
    evaluate_exact_static_magnetic_one_electron_matrices,
    evaluate_exact_uniform_magnetic_wilson_density,
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
    anchors = reference.core_operators.nuclei.coordinates_au[
        reference.anchor_topology.ao_to_atom
    ]
    gauge_function_at_anchors = affine_gauge_difference_potential(
        landau,
        symmetric,
        anchors,
        quadrature.backend,
    )
    coefficient_unitary = np.exp(-1j * gauge_function_at_anchors)
    transformed_density = (
        coefficient_unitary[:, None]
        * coefficient_density
        * coefficient_unitary.conj()[None, :]
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
