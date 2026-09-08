from __future__ import annotations

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")

from aion.gauge import UniformMagneticGauge
from aion.reference import (
    build_unpruned_grid,
    evaluate_uniform_magnetic_matrices,
    evaluate_uniform_magnetic_taylor_matrices,
    generalized_hermitian_spectrum,
    pyscf_giao_one_electron_derivatives,
    uniform_anchored_vector_taylor_coefficients,
    uniform_anchored_vectors,
    uniform_triangle_factor_taylor_coefficients,
    uniform_triangle_flux,
    zero_field_corrected_barred_matrices,
)
from aion.reference.wilson_campaign import (
    PairConfiguration,
    build_pair_molecule,
    field_directions,
)


def _matrix_names():
    return ("overlap", "kinetic", "potential", "mechanical")


@pytest.fixture(scope="module")
def oh_molecule():
    return build_pair_molecule(PairConfiguration("O-H", 1.8, "sto-3g"))


@pytest.fixture(scope="module")
def oh_grid(oh_molecule):
    return build_unpruned_grid(oh_molecule, level=0)


@pytest.fixture(scope="module")
def oblique_direction(oh_molecule):
    return field_directions(oh_molecule)["oblique"]


def test_uniform_primitive_derivatives_match_multistep_central_differences():
    points = np.array(
        [
            [0.31, -0.27, 0.44],
            [-0.52, 0.63, 0.18],
            [1.21, -0.08, -0.37],
        ]
    )
    anchors = np.array(
        [
            [0.17, -0.33, 0.21],
            [1.14, 0.49, -0.28],
        ]
    )
    direction = np.array([0.37, -0.51, 0.69])
    direction /= np.linalg.norm(direction)
    coefficients = uniform_triangle_factor_taylor_coefficients(
        points,
        anchors,
        direction,
    )
    anchored_coefficients = uniform_anchored_vector_taylor_coefficients(
        points,
        anchors,
        direction,
    )

    np.testing.assert_allclose(coefficients[0], 1.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(anchored_coefficients[0], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        anchored_coefficients[1],
        uniform_anchored_vectors(points, anchors, direction),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(anchored_coefficients[2], 0.0, atol=0.0, rtol=0.0)

    first_errors = []
    second_errors = []
    anchored_first_errors = []
    anchored_second_errors = []
    for step in (4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4):
        plus_flux = uniform_triangle_flux(points, anchors, step * direction)
        minus_flux = uniform_triangle_flux(points, anchors, -step * direction)
        plus = np.exp(-1j * plus_flux)
        minus = np.exp(-1j * minus_flux)
        first_fd = (plus - minus) / (2.0 * step)
        second_coefficient_fd = (plus - 2.0 + minus) / (2.0 * step**2)
        first_errors.append(np.linalg.norm(first_fd - coefficients[1]))
        second_errors.append(np.linalg.norm(second_coefficient_fd - coefficients[2]))
        plus_anchored = uniform_anchored_vectors(points, anchors, step * direction)
        minus_anchored = uniform_anchored_vectors(points, anchors, -step * direction)
        anchored_first_errors.append(
            np.linalg.norm(
                (plus_anchored - minus_anchored) / (2.0 * step)
                - anchored_coefficients[1]
            )
        )
        anchored_second_errors.append(
            np.linalg.norm(
                (plus_anchored + minus_anchored) / (2.0 * step**2)
                - anchored_coefficients[2]
            )
        )

    assert first_errors[-1] < first_errors[0] / 10.0
    assert second_errors[-1] < second_errors[0] / 10.0
    assert first_errors[-1] < 2.0e-8
    assert second_errors[-1] < 2.0e-8
    assert max(anchored_first_errors) < 3.0e-13
    assert max(anchored_second_errors) < 3.0e-13


def test_matrix_derivatives_match_multistep_central_differences(
    oh_molecule,
    oh_grid,
    oblique_direction,
):
    taylor = evaluate_uniform_magnetic_taylor_matrices(
        oh_molecule,
        UniformMagneticGauge(oblique_direction),
        grid=oh_grid,
        block_size=257,
    )
    zero = evaluate_uniform_magnetic_matrices(
        oh_molecule,
        UniformMagneticGauge(np.zeros(3)),
        grid=oh_grid,
        block_size=257,
    )
    first = taylor.directional_derivative(1)
    second_coefficient = taylor.coefficient(2).barred

    errors: dict[str, list[tuple[float, float]]] = {
        name: [] for name in _matrix_names()
    }
    for step in (4.0e-3, 2.0e-3, 1.0e-3):
        plus = evaluate_uniform_magnetic_matrices(
            oh_molecule,
            UniformMagneticGauge(step * oblique_direction),
            grid=oh_grid,
            block_size=257,
        )
        minus = evaluate_uniform_magnetic_matrices(
            oh_molecule,
            UniformMagneticGauge(-step * oblique_direction),
            grid=oh_grid,
            block_size=257,
        )
        for name in _matrix_names():
            plus_matrix = getattr(plus.barred, name)
            minus_matrix = getattr(minus.barred, name)
            zero_matrix = getattr(zero.barred, name)
            first_fd = (plus_matrix - minus_matrix) / (2.0 * step)
            second_fd = (
                plus_matrix - 2.0 * zero_matrix + minus_matrix
            ) / (2.0 * step**2)
            errors[name].append(
                (
                    np.linalg.norm(first_fd - getattr(first, name), ord="fro"),
                    np.linalg.norm(
                        second_fd - getattr(second_coefficient, name),
                        ord="fro",
                    ),
                )
            )

    for name, matrix_errors in errors.items():
        first_errors = [entry[0] for entry in matrix_errors]
        second_errors = [entry[1] for entry in matrix_errors]
        assert first_errors[-1] < max(2.0e-8, first_errors[0] / 8.0), name
        assert second_errors[-1] < max(2.0e-7, second_errors[0] / 8.0), name


def test_barred_derivatives_match_independent_libcint_giao_kernels(oh_molecule):
    giao = pyscf_giao_one_electron_derivatives(oh_molecule)
    grid = build_unpruned_grid(oh_molecule, level=3)
    analytic_components = []
    for direction in np.eye(3):
        result = evaluate_uniform_magnetic_taylor_matrices(
            oh_molecule,
            UniformMagneticGauge(direction, gauge="symmetric", origin=np.zeros(3)),
            max_order=1,
            grid=grid,
            block_size=1024,
        )
        analytic_components.append(result.directional_derivative(1))

    for name in _matrix_names():
        analytic = np.asarray(
            [getattr(component, name) for component in analytic_components]
        )
        np.testing.assert_allclose(
            analytic,
            getattr(giao.barred, name),
            atol=1.0e-7,
            rtol=1.0e-7,
        )
        np.testing.assert_allclose(
            getattr(giao.lower, name),
            getattr(giao.barred, name) + getattr(giao.endpoint, name),
            atol=2.0e-14,
            rtol=2.0e-14,
        )


def test_taylor_orders_have_predicted_parity_and_complete_kinetic_sectors(
    oh_molecule,
    oh_grid,
    oblique_direction,
):
    positive = evaluate_uniform_magnetic_taylor_matrices(
        oh_molecule,
        UniformMagneticGauge(oblique_direction),
        grid=oh_grid,
        block_size=257,
    )
    negative = evaluate_uniform_magnetic_taylor_matrices(
        oh_molecule,
        UniformMagneticGauge(-oblique_direction),
        grid=oh_grid,
        block_size=257,
    )

    for order, sign in ((0, 1.0), (1, -1.0), (2, 1.0)):
        positive_order = positive.coefficient(order)
        negative_order = negative.coefficient(order)
        for name in _matrix_names():
            matrix = getattr(positive_order.barred, name)
            np.testing.assert_allclose(matrix, matrix.conj().T, atol=3.0e-12)
            np.testing.assert_allclose(
                getattr(negative_order.barred, name),
                sign * matrix,
                atol=3.0e-12,
            )
            if order % 2:
                assert np.linalg.norm(matrix.real, ord="fro") < 3.0e-12
            else:
                assert np.linalg.norm(matrix.imag, ord="fro") < 3.0e-12
        np.testing.assert_allclose(
            positive_order.kinetic_sectors.total,
            positive_order.barred.kinetic,
            atol=3.0e-12,
        )
        for sector_name in ("pp", "p_c", "c_p", "c_c"):
            positive_sector = getattr(positive_order.kinetic_sectors, sector_name)
            negative_sector = getattr(negative_order.kinetic_sectors, sector_name)
            np.testing.assert_allclose(
                negative_sector,
                sign * positive_sector,
                atol=3.0e-12,
            )
        np.testing.assert_allclose(
            positive_order.kinetic_sectors.p_c.conj().T,
            positive_order.kinetic_sectors.c_p,
            atol=3.0e-12,
        )

    np.testing.assert_allclose(
        positive.coefficient(1).kinetic_sectors.c_c,
        0.0,
        atol=0.0,
        rtol=0.0,
    )
    assert np.linalg.norm(positive.coefficient(1).kinetic_sectors.p_c) > 1.0e-8
    assert np.linalg.norm(positive.coefficient(1).kinetic_sectors.c_p) > 1.0e-8
    assert np.linalg.norm(positive.coefficient(2).kinetic_sectors.c_c) > 1.0e-8


def test_b1_and_b2_one_electron_errors_show_second_and_third_order_scaling(
    oh_molecule,
    oh_grid,
    oblique_direction,
):
    zero = evaluate_uniform_magnetic_matrices(
        oh_molecule,
        UniformMagneticGauge(np.zeros(3)),
        grid=oh_grid,
        block_size=257,
    )
    fields = np.array([5.0e-4, 1.0e-3, 2.0e-3, 4.0e-3])
    b1_errors = []
    b2_errors = []
    for field in fields:
        gauge = UniformMagneticGauge(field * oblique_direction)
        exact = evaluate_uniform_magnetic_matrices(
            oh_molecule,
            gauge,
            grid=oh_grid,
            block_size=257,
        )
        taylor = evaluate_uniform_magnetic_taylor_matrices(
            oh_molecule,
            gauge,
            grid=oh_grid,
            block_size=257,
        )
        corrected = zero_field_corrected_barred_matrices(exact, zero)
        b1_errors.append(
            np.linalg.norm(
                corrected.mechanical - taylor.b1_one_electron_barred.mechanical,
                ord="fro",
            )
        )
        b2_errors.append(
            np.linalg.norm(
                corrected.mechanical - taylor.b2_one_electron_barred.mechanical,
                ord="fro",
            )
        )

    b1_slope = np.polyfit(np.log(fields), np.log(b1_errors), 1)[0]
    b2_slope = np.polyfit(np.log(fields), np.log(b2_errors), 1)[0]
    assert b1_slope == pytest.approx(2.0, abs=0.06)
    assert b2_slope == pytest.approx(3.0, abs=0.08)
    assert b2_errors[-1] < b1_errors[-1] / 100.0


def test_exact_link_truncations_preserve_gauge_invariant_barred_data_and_spectra(
    oh_molecule,
    oh_grid,
    oblique_direction,
):
    field = 0.03 * oblique_direction
    symmetric = evaluate_uniform_magnetic_taylor_matrices(
        oh_molecule,
        UniformMagneticGauge(
            field,
            gauge="symmetric",
            origin=np.array([0.17, -0.31, 0.23]),
        ),
        grid=oh_grid,
        block_size=257,
    )
    landau = evaluate_uniform_magnetic_taylor_matrices(
        oh_molecule,
        UniformMagneticGauge(
            field,
            gauge="landau",
            origin=np.array([-0.21, 0.37, -0.16]),
            landau_u=np.array([field[1], -field[0], 0.0]),
        ),
        grid=oh_grid,
        block_size=257,
    )

    for order in (1, 2):
        symmetric_barred = symmetric.barred_through(order)
        landau_barred = landau.barred_through(order)
        for name in _matrix_names():
            np.testing.assert_allclose(
                getattr(symmetric_barred, name),
                getattr(landau_barred, name),
                atol=0.0,
                rtol=0.0,
            )
        symmetric_lower = symmetric.lower_through(order)
        landau_lower = landau.lower_through(order)
        symmetric_spectrum = generalized_hermitian_spectrum(
            symmetric_lower.mechanical,
            symmetric_lower.overlap,
            metric_floor=1.0e-10,
        )
        landau_spectrum = generalized_hermitian_spectrum(
            landau_lower.mechanical,
            landau_lower.overlap,
            metric_floor=1.0e-10,
        )
        np.testing.assert_allclose(
            symmetric_spectrum.eigenvalues,
            landau_spectrum.eigenvalues,
            atol=2.0e-13,
        )
