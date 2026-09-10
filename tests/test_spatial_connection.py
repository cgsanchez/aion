from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aion.config import AtomConfig, BackendConfig, MoleculeConfig
from aion.electromagnetism import UniformMagneticField
from aion.electronic_structure import (
    AOGridPolicy,
    evaluate_magnetic_spatial_connections,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.integration


def _assert_antihermitian(matrix: np.ndarray, tolerance: float = 2.0e-11) -> None:
    np.testing.assert_allclose(
        matrix + matrix.conj().transpose(0, 2, 1), 0.0, atol=tolerance, rtol=tolerance
    )


def test_direct_and_factorized_spatial_connections_agree_and_are_antihermitian() -> None:
    reference = prepare_pyscf_reference(molecular_config("lih"))
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(1),
        block_size=257,
    )
    result = evaluate_magnetic_spatial_connections(
        quadrature, (UniformMagneticField((0.013, -0.009, 0.017)),)
    )[0]
    assert result.direct_oracle is not None
    np.testing.assert_allclose(
        result.direct_oracle.lower_grid,
        result.lower_exact_grid,
        atol=2.0e-11,
        rtol=2.0e-11,
    )
    hierarchy = result.spatial_connection
    for matrix in (
        hierarchy.zero,
        hierarchy.quadrature_zero,
        hierarchy.first_F,
        hierarchy.first_C,
        hierarchy.second_F2,
        hierarchy.second_FC,
        hierarchy.exact,
        hierarchy.b1,
        hierarchy.b2,
    ):
        _assert_antihermitian(matrix)


def test_momentum_and_E1_closures_reach_the_unpruned_grid_floor() -> None:
    reference = prepare_pyscf_reference(molecular_config("lih"))
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(4),
        block_size=2048,
    )
    result = evaluate_magnetic_spatial_connections(
        quadrature,
        (UniformMagneticField((0.013, -0.009, 0.017)),),
        include_direct_oracle=False,
    )[0]
    hierarchy = result.spatial_connection
    np.testing.assert_array_equal(
        hierarchy.zero, 1j * reference.core_operators.canonical_momentum
    )
    assert np.linalg.norm(hierarchy.quadrature_zero - hierarchy.zero) < 1.0e-8
    assert np.linalg.norm(hierarchy.first_C - result.e1_first_C_closure) < 2.0e-9


def test_reversal_finite_differences_and_truncation_orders() -> None:
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
    results = evaluate_magnetic_spatial_connections(
        quadrature, fields, include_direct_oracle=False
    )
    scaled, unit, negative, zero = results[:4], results[4], results[5], results[6]
    positive = scaled[1]
    np.testing.assert_allclose(
        negative.spatial_connection.exact,
        positive.spatial_connection.exact.conj(),
        atol=2.0e-12,
    )
    first_fd = (
        positive.spatial_connection.exact - negative.spatial_connection.exact
    ) / 2.0e-3
    second_fd = (
        positive.spatial_connection.exact
        + negative.spatial_connection.exact
        - 2.0 * zero.spatial_connection.exact
    ) / 2.0e-6
    np.testing.assert_allclose(
        first_fd, unit.spatial_connection.first, atol=5.0e-7, rtol=5.0e-7
    )
    np.testing.assert_allclose(
        second_fd, unit.spatial_connection.second, atol=5.0e-7, rtol=5.0e-7
    )
    b1_errors = np.asarray(
        [
            np.linalg.norm(result.spatial_connection.exact - result.spatial_connection.b1)
            for result in scaled
        ]
    )
    b2_errors = np.asarray(
        [
            np.linalg.norm(result.spatial_connection.exact - result.spatial_connection.b2)
            for result in scaled
        ]
    )
    assert np.polyfit(np.log(steps), np.log(b1_errors), 1)[0] == pytest.approx(2.0, abs=0.03)
    assert np.polyfit(np.log(steps), np.log(b2_errors), 1)[0] == pytest.approx(3.0, abs=0.04)


def test_s_only_h2_spatial_connection_is_rigid_rotation_covariant() -> None:
    angle = 0.43
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    atoms = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    field = np.array([0.013, -0.009, 0.017])
    base_config = molecular_config("h2")
    rotated_atoms = atoms @ rotation.T
    rotated_config = replace(
        base_config,
        molecule=MoleculeConfig(
            atoms=tuple(AtomConfig("H", tuple(position)) for position in rotated_atoms)
        ),
    )
    base_reference = prepare_pyscf_reference(base_config)
    rotated_reference = prepare_pyscf_reference(rotated_config)
    base_quadrature = prepare_ao_quadrature(
        base_reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(4),
        block_size=2048,
    )
    rotated_quadrature = prepare_ao_quadrature(
        rotated_reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(4),
        block_size=2048,
    )
    base = evaluate_magnetic_spatial_connections(
        base_quadrature,
        (UniformMagneticField(tuple(field)),),
        include_direct_oracle=False,
    )[0].spatial_connection
    rotated = evaluate_magnetic_spatial_connections(
        rotated_quadrature,
        (UniformMagneticField(tuple(field @ rotation.T)),),
        include_direct_oracle=False,
    )[0].spatial_connection
    for name in ("zero", "first_F", "first_C", "second_F2", "second_FC", "exact"):
        expected = np.einsum("xa,amn->xmn", rotation, getattr(base, name))
        np.testing.assert_allclose(getattr(rotated, name), expected, atol=2.0e-12, rtol=2.0e-12)


def test_spatial_connection_is_block_size_invariant_and_direct_oracle_is_optional() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    field = UniformMagneticField((0.011, -0.007, 0.005))
    small = evaluate_magnetic_spatial_connections(
        prepare_ao_quadrature(reference, BackendConfig(), block_size=97),
        (field,),
        include_direct_oracle=False,
    )[0]
    large = evaluate_magnetic_spatial_connections(
        prepare_ao_quadrature(reference, BackendConfig(), block_size=1024),
        (field,),
    )[0]
    assert small.direct_oracle is None
    assert large.direct_oracle is not None
    for name in ("first", "second", "exact"):
        np.testing.assert_allclose(
            getattr(small.spatial_connection, name),
            getattr(large.spatial_connection, name),
            atol=2.0e-13,
            rtol=2.0e-13,
        )
