from __future__ import annotations

import numpy as np

from aion.backends import NumPyBackend
from aion.electromagnetism import AffineVectorFieldVariation


def test_affine_vector_field_segment_integral_and_gradient() -> None:
    backend = NumPyBackend()
    variation = AffineVectorFieldVariation(
        offset_au=(0.17, -0.23, 0.31),
        matrix_au=((0.2, -0.1, 0.4), (0.3, 0.5, -0.2), (-0.4, 0.1, 0.6)),
        origin_au=(0.11, -0.07, 0.19),
    )
    starts = np.asarray(((0.2, -0.4, 0.1), (-0.3, 0.5, 0.7)))
    ends = np.asarray(((0.8, 0.1, -0.2), (0.4, -0.6, 0.3)))
    values = variation.straight_line_integrals(starts, ends, backend)
    midpoint = 0.5 * (starts + ends)
    expected = np.einsum(
        "px,px->p",
        variation.vector_potential(midpoint, backend),
        ends - starts,
    )
    np.testing.assert_allclose(values, expected, atol=2.0e-16, rtol=2.0e-16)

    analytic = variation.straight_line_integral_gradients(starts, ends, backend)
    step = 1.0e-6
    finite = np.empty_like(analytic)
    for point in range(ends.shape[0]):
        for axis in range(3):
            displaced_plus = ends.copy()
            displaced_minus = ends.copy()
            displaced_plus[point, axis] += step
            displaced_minus[point, axis] -= step
            plus = variation.straight_line_integrals(starts, displaced_plus, backend)
            minus = variation.straight_line_integrals(starts, displaced_minus, backend)
            finite[point, axis] = (plus[point] - minus[point]) / (2.0 * step)
    np.testing.assert_allclose(analytic, finite, atol=2.0e-10, rtol=2.0e-10)


def test_affine_induction_field_matches_uniform_source_convention() -> None:
    backend = NumPyBackend()
    electric = np.asarray((0.13, -0.07, 0.19))
    magnetic_dot = np.asarray((0.03, -0.05, 0.11))
    origin = np.asarray((0.2, -0.3, 0.4))
    points = np.asarray(((0.8, -0.4, 0.1), (-0.7, 0.2, 0.9)))
    variation = AffineVectorFieldVariation.from_uniform_magnetic_electric_field(
        electric_field_origin_au=tuple(electric),
        magnetic_field_dot_au=tuple(magnetic_dot),
        origin_au=tuple(origin),
    )
    expected = electric + 0.5 * np.cross(points - origin, magnetic_dot)
    np.testing.assert_allclose(
        variation.vector_potential(points, backend),
        expected,
        atol=2.0e-16,
        rtol=2.0e-16,
    )
