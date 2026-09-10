from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import eigvalsh

from aion.config import AtomConfig, BackendConfig, MoleculeConfig
from aion.electromagnetism import (
    AffineMagneticGauge,
    MagneticGaugeKind,
    UniformMagneticField,
)
from aion.electronic_structure import (
    AOGridPolicy,
    evaluate_magnetic_one_electron_matrices,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
    pyscf_giao_one_electron_derivatives,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.integration


def _matrix_families(result: object) -> tuple[object, object, object]:
    return result.overlap, result.kinetic, result.nuclear_attraction  # type: ignore[attr-defined]


def _assert_hermitian(matrix: np.ndarray, tolerance: float = 2.0e-11) -> None:
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=tolerance, rtol=tolerance)


def test_exact_factorized_and_direct_routes_agree_and_sectors_close() -> None:
    reference = prepare_pyscf_reference(molecular_config("lih"))
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(1),
        block_size=257,
    )
    field = UniformMagneticField((0.013, -0.009, 0.017))
    result = evaluate_magnetic_one_electron_matrices(quadrature, (field,))[0]
    assert result.direct_oracle is not None
    for name in ("overlap", "kinetic", "nuclear_attraction"):
        direct = getattr(result.direct_oracle.lower_grid, name)
        factorized = getattr(result.lower_exact_grid, name)
        np.testing.assert_allclose(direct, factorized, atol=2.0e-11, rtol=2.0e-11)
    np.testing.assert_allclose(
        result.kinetic.quadrature_exact_sectors.total,
        result.kinetic.exact_grid,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        result.kinetic.exact_sectors.total, result.kinetic.exact, atol=2.0e-13
    )
    np.testing.assert_allclose(
        result.kinetic.first_pC.conj().T,
        result.kinetic.first_Cp,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        result.kinetic.second_F_pC.conj().T,
        result.kinetic.second_F_Cp,
        atol=2.0e-11,
    )
    for hierarchy in _matrix_families(result):
        for matrix in (hierarchy.zero, hierarchy.exact, hierarchy.b1, hierarchy.b2):
            _assert_hermitian(matrix)


def test_zero_field_reversal_parity_and_multi_field_single_ao_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = prepare_pyscf_reference(molecular_config("lih"))
    quadrature = prepare_ao_quadrature(reference, BackendConfig(), block_size=521)
    direction = np.array([0.37, -0.51, 0.69])
    direction /= np.linalg.norm(direction)
    field = UniformMagneticField(tuple(0.006 * direction))
    negative = UniformMagneticField(tuple(-0.006 * direction))
    zero = UniformMagneticField((0.0, 0.0, 0.0))

    calls = 0
    original = type(quadrature).blocks

    def counted_blocks(self: object) -> object:
        nonlocal calls
        calls += 1
        return original(self)  # type: ignore[arg-type]

    monkeypatch.setattr(type(quadrature), "blocks", counted_blocks)
    positive_result, negative_result, zero_result = evaluate_magnetic_one_electron_matrices(
        quadrature, (field, negative, zero)
    )
    assert calls == 1
    for positive, reversed_value, at_zero in zip(
        _matrix_families(positive_result),
        _matrix_families(negative_result),
        _matrix_families(zero_result),
        strict=True,
    ):
        np.testing.assert_allclose(reversed_value.exact, positive.exact.conj(), atol=2.0e-12)
        np.testing.assert_allclose(reversed_value.first, -positive.first, atol=2.0e-12)
        np.testing.assert_allclose(reversed_value.second, positive.second, atol=2.0e-12)
        np.testing.assert_allclose(at_zero.exact, at_zero.zero, atol=0.0, rtol=0.0)


def test_h2_bond_parallel_field_isolates_C_terms() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    quadrature = prepare_ao_quadrature(reference, BackendConfig(), block_size=257)
    result = evaluate_magnetic_one_electron_matrices(
        quadrature, (UniformMagneticField((0.0, 0.0, 0.04)),)
    )[0]
    np.testing.assert_allclose(result.overlap.first_F, 0.0, atol=0.0)
    np.testing.assert_allclose(result.overlap.second_F2, 0.0, atol=0.0)
    np.testing.assert_allclose(result.nuclear_attraction.first_F, 0.0, atol=0.0)
    np.testing.assert_allclose(result.kinetic.first_F, 0.0, atol=0.0)
    np.testing.assert_allclose(result.kinetic.second_F2, 0.0, atol=0.0)
    np.testing.assert_allclose(result.kinetic.second_F_pC, 0.0, atol=0.0)
    np.testing.assert_allclose(result.kinetic.second_F_Cp, 0.0, atol=0.0)
    # The first-order C matrix is symmetry-suppressed for a pure 1s/1s
    # collinear STO-3G fixture, while the C geometry and C2 signal remain.
    assert np.linalg.norm(result.kinetic.first_C) < 1.0e-12
    assert np.linalg.norm(result.kinetic.second_C2) > 1.0e-8


def test_affine_gauges_give_equal_barred_matrices_and_congruent_spectra() -> None:
    reference = prepare_pyscf_reference(molecular_config("lih"))
    quadrature = prepare_ao_quadrature(reference, BackendConfig(), block_size=521)
    field = UniformMagneticField((0.013, -0.009, 0.017))
    axis = (field.magnetic_field_au[1], -field.magnetic_field_au[0], 0.0)
    gauges = (
        AffineMagneticGauge(field, origin_au=(0.17, -0.31, 0.23)),
        AffineMagneticGauge(
            field,
            kind=MagneticGaugeKind.LANDAU,
            origin_au=(-0.21, 0.37, -0.16),
            landau_axis=axis,
        ),
    )
    symmetric, landau = evaluate_magnetic_one_electron_matrices(
        quadrature, (field, field), direct_gauges=gauges
    )
    for left, right in zip(
        _matrix_families(symmetric), _matrix_families(landau), strict=True
    ):
        np.testing.assert_allclose(left.exact, right.exact, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(left.b1, right.b1, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(left.b2, right.b2, atol=0.0, rtol=0.0)
    for result in (symmetric, landau):
        assert result.direct_oracle is not None
        np.testing.assert_allclose(
            result.direct_oracle.lower_grid.kinetic,
            result.lower_exact_grid.kinetic,
            atol=2.0e-11,
        )
    symmetric_eigenvalues = eigvalsh(
        symmetric.lower_exact.kinetic + symmetric.lower_exact.nuclear_attraction,
        symmetric.lower_exact.overlap,
    )
    landau_eigenvalues = eigvalsh(
        landau.lower_exact.kinetic + landau.lower_exact.nuclear_attraction,
        landau.lower_exact.overlap,
    )
    np.testing.assert_allclose(symmetric_eigenvalues, landau_eigenvalues, atol=2.0e-13)


def test_finite_differences_and_B1_B2_remainders_have_declared_orders() -> None:
    reference = prepare_pyscf_reference(molecular_config("lih"))
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(2),
        block_size=1024,
    )
    direction = np.array([0.37, -0.51, 0.69])
    direction /= np.linalg.norm(direction)
    steps = np.array([5.0e-4, 1.0e-3, 2.0e-3, 4.0e-3])
    fields = [UniformMagneticField(tuple(step * direction)) for step in steps]
    fields.extend(
        (
            UniformMagneticField(tuple(direction)),
            UniformMagneticField(tuple(-1.0e-3 * direction)),
            UniformMagneticField((0.0, 0.0, 0.0)),
        )
    )
    results = evaluate_magnetic_one_electron_matrices(quadrature, fields)
    scaled, unit, minus, zero = results[:4], results[4], results[5], results[6]
    plus = scaled[1]
    for positive, negative, at_zero, unit_value in zip(
        _matrix_families(plus),
        _matrix_families(minus),
        _matrix_families(zero),
        _matrix_families(unit),
        strict=True,
    ):
        first_fd = (positive.exact - negative.exact) / (2.0e-3)
        second_fd = (positive.exact + negative.exact - 2.0 * at_zero.exact) / (2.0e-6)
        np.testing.assert_allclose(first_fd, unit_value.first, atol=5.0e-7, rtol=5.0e-7)
        np.testing.assert_allclose(second_fd, unit_value.second, atol=5.0e-7, rtol=5.0e-7)

    for family_index in range(3):
        b1_errors = np.array(
            [
                np.linalg.norm(
                    _matrix_families(result)[family_index].exact
                    - _matrix_families(result)[family_index].b1
                )
                for result in scaled
            ]
        )
        b2_errors = np.array(
            [
                np.linalg.norm(
                    _matrix_families(result)[family_index].exact
                    - _matrix_families(result)[family_index].b2
                )
                for result in scaled
            ]
        )
        assert np.polyfit(np.log(steps), np.log(b1_errors), 1)[0] == pytest.approx(2.0, abs=0.03)
        assert np.polyfit(np.log(steps), np.log(b2_errors), 1)[0] == pytest.approx(3.0, abs=0.04)


def test_displaced_lih_B1_matches_independent_libcint_giao_derivatives() -> None:
    config = molecular_config("lih")
    config = replace(
        config,
        molecule=MoleculeConfig(
            atoms=(
                AtomConfig("Li", (0.2, -0.1, -1.2)),
                AtomConfig("H", (0.4, 0.3, 1.7)),
            )
        ),
    )
    reference = prepare_pyscf_reference(config)
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(4),
        block_size=2048,
    )
    fields = tuple(
        UniformMagneticField((float(axis[0]), float(axis[1]), float(axis[2])))
        for axis in np.eye(3)
    )
    results = evaluate_magnetic_one_electron_matrices(quadrature, fields)
    giao = pyscf_giao_one_electron_derivatives(reference)
    for name in ("overlap", "kinetic", "nuclear_attraction"):
        calculated = np.asarray(
            [
                getattr(result, name).first_F
                if name != "kinetic"
                else result.kinetic.first
                for result in results
            ]
        )
        np.testing.assert_allclose(calculated, getattr(giao.barred, name), atol=3.0e-8, rtol=3.0e-8)
        np.testing.assert_allclose(
            getattr(giao.lower, name),
            getattr(giao.barred, name) + getattr(giao.endpoint, name),
            atol=2.0e-14,
            rtol=2.0e-14,
        )
        assert np.linalg.norm(getattr(giao.endpoint, name)) > 1.0e-3


def test_block_size_invariance() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    field = UniformMagneticField((0.011, -0.007, 0.005))
    small = evaluate_magnetic_one_electron_matrices(
        prepare_ao_quadrature(reference, BackendConfig(), block_size=97), (field,)
    )[0]
    large = evaluate_magnetic_one_electron_matrices(
        prepare_ao_quadrature(reference, BackendConfig(), block_size=1024), (field,)
    )[0]
    for left, right in zip(_matrix_families(small), _matrix_families(large), strict=True):
        for name in ("first", "second", "exact"):
            np.testing.assert_allclose(
                getattr(left, name), getattr(right, name), atol=2.0e-13, rtol=2.0e-13
            )


def test_direct_oracle_can_be_omitted_without_changing_authoritative_matrices() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    quadrature = prepare_ao_quadrature(reference, BackendConfig(), block_size=1024)
    field = UniformMagneticField((0.011, -0.007, 0.005))
    with_oracle = evaluate_magnetic_one_electron_matrices(quadrature, (field,))[0]
    without_oracle = evaluate_magnetic_one_electron_matrices(
        quadrature, (field,), include_direct_oracle=False
    )[0]
    assert with_oracle.direct_oracle is not None
    assert without_oracle.direct_oracle is None
    for left, right in zip(
        _matrix_families(with_oracle), _matrix_families(without_oracle), strict=True
    ):
        np.testing.assert_array_equal(left.exact, right.exact)
