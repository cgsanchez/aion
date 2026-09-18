from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aion.backends import NumPyBackend
from aion.config import (
    AtomConfig,
    BackendConfig,
    ElectromagneticOrigin,
    OneElectronReferenceConfig,
)
from aion.electromagnetism import (
    MagneticGaugeKind,
    UniformMagneticField,
    UniformMagneticSourceSample,
    affine_gauge_difference_potential,
)
from aion.electronic_structure import (
    AOGridPolicy,
    evaluate_exact_static_magnetic_one_electron_matrices,
    evaluate_exact_wilson_one_electron_sample,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
)
from aion.formulations import (
    exact_one_electron_model_triples,
    exact_wilson_one_electron_triple,
    prepare_exact_one_electron_model_context,
)
from aion.propagation import (
    LinearMatrixHistory,
    generalized_spectral_trajectory,
    propagate_linear_matrix_history,
)

pytestmark = pytest.mark.integration

_FIXTURE = Path(__file__).parent / "fixtures/exact_one_electron/hh_sto3g.fixture.json"


def _quadrature() -> object:
    values = json.loads(_FIXTURE.read_text(encoding="utf-8"))["config"]
    reference = prepare_one_electron_ao_reference(
        OneElectronReferenceConfig(
            atoms=tuple(
                AtomConfig(atom["symbol"], tuple(atom["position_au"]))
                for atom in values["atoms"]
            ),
            basis=values["basis"],
            electromagnetic_origin=ElectromagneticOrigin(
                tuple(values["electromagnetic_origin_au"])
            ),
        )
    )
    return prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(3),
        block_size=1024,
    )


def _relative(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value) / max(1.0, np.linalg.norm(reference)))


def test_wp6_source_includes_induction_field_in_every_affine_gauge() -> None:
    backend = NumPyBackend()
    points = np.asarray(
        ((0.17, -0.31, 0.23), (-0.41, 0.29, 0.37), (0.61, 0.13, -0.27))
    )
    field = np.asarray((0.0, 0.0, 0.07))
    field_dot = np.asarray((0.0, 0.0, -0.013))
    electric = np.asarray((0.011, -0.017, 0.019))
    origin = np.asarray((0.09, -0.07, 0.12))
    for kind, axis in (
        (MagneticGaugeKind.SYMMETRIC, None),
        (MagneticGaugeKind.LANDAU, (1.0, 0.0, 0.0)),
    ):
        source = UniformMagneticSourceSample(
            0.37,
            UniformMagneticField(tuple(field)),
            magnetic_field_dot_au=tuple(field_dot),
            electric_field_origin_au=tuple(electric),
            origin_au=tuple(origin),
            gauge_kind=kind,
            landau_axis=axis,
        )
        step = 2.0e-6
        gradient = np.empty_like(points)
        for component in range(3):
            shift = np.zeros(3)
            shift[component] = step
            gradient[:, component] = (
                source.scalar_potential(points + shift, backend)
                - source.scalar_potential(points - shift, backend)
            ) / (2.0 * step)
        reconstructed = -source.gauge_rate.vector_potential(points, backend) - gradient
        np.testing.assert_allclose(
            reconstructed,
            source.electric_field(points, backend),
            atol=4.0e-12,
            rtol=4.0e-12,
        )


def test_wp6_exact_time_connection_routes_metric_rate_and_compatibility() -> None:
    quadrature = _quadrature()
    source = UniformMagneticSourceSample(
        0.41,
        UniformMagneticField((0.0, 0.0, 0.031)),
        magnetic_field_dot_au=(0.0, 0.0, -0.007),
        electric_field_origin_au=(0.013, -0.009, 0.017),
        origin_au=(0.11, -0.07, 0.05),
    )
    sample = evaluate_exact_wilson_one_electron_sample(quadrature, source)
    time = sample.connection
    assert time.direct_factorized_connection_residual < 3.0e-14
    assert time.direct_factorized_metric_dot_residual < 3.0e-14
    assert time.metric_compatibility_residual < 2.0e-10

    step = 2.0e-5
    direction = np.asarray(source.magnetic_field_dot_au)
    current = np.asarray(source.field.magnetic_field_au)
    plus, minus = evaluate_exact_static_magnetic_one_electron_matrices(
        quadrature,
        (
            UniformMagneticField(tuple(current + step * direction)),
            UniformMagneticField(tuple(current - step * direction)),
        ),
        include_direct_oracle=False,
    )
    finite_difference = (plus.lower_exact.overlap - minus.lower_exact.overlap) / (
        2.0 * step
    )
    np.testing.assert_allclose(
        time.metric_dot,
        finite_difference,
        atol=3.0e-10,
        rtol=3.0e-8,
    )


def test_wp6_zero_magnetic_field_exact_connection_reduces_to_uniform_electric_e1() -> None:
    quadrature = _quadrature()
    electric = np.asarray((0.013, -0.009, 0.017))
    source = UniformMagneticSourceSample(
        0.27,
        UniformMagneticField((0.0, 0.0, 0.0)),
        electric_field_origin_au=tuple(electric),
        origin_au=(0.11, -0.07, 0.05),
    )
    sample = evaluate_exact_wilson_one_electron_sample(quadrature, source)
    position = quadrature.reference.core_operators.position
    overlap = quadrature.reference.core_operators.overlap
    relative_position = position - np.asarray(source.origin_au)[:, None, None] * overlap
    expected = 1j * np.einsum("x,xmn->mn", electric, relative_position, optimize=True)
    np.testing.assert_allclose(
        sample.connection.connection,
        expected,
        atol=3.0e-15,
        rtol=3.0e-15,
    )


def test_wp6_exact_triple_obeys_affine_gauge_covariance() -> None:
    quadrature = _quadrature()
    common = dict(
        time_au=0.41,
        field=UniformMagneticField((0.0, 0.0, 0.031)),
        magnetic_field_dot_au=(0.0, 0.0, -0.007),
        electric_field_origin_au=(0.013, -0.009, 0.017),
        origin_au=(0.11, -0.07, 0.05),
    )
    symmetric_source = UniformMagneticSourceSample(**common)
    landau_source = UniformMagneticSourceSample(
        **common,
        gauge_kind=MagneticGaugeKind.LANDAU,
        landau_axis=(1.0, 0.0, 0.0),
    )
    symmetric = evaluate_exact_wilson_one_electron_sample(quadrature, symmetric_source)
    landau = evaluate_exact_wilson_one_electron_sample(quadrature, landau_source)
    backend = quadrature.backend
    anchors = quadrature.reference.core_operators.nuclei.coordinates_au[
        quadrature.reference.anchor_topology.ao_to_atom
    ]
    gauge_lambda = affine_gauge_difference_potential(
        landau_source.gauge,
        symmetric_source.gauge,
        anchors,
        backend,
    )
    gauge_lambda_dot = affine_gauge_difference_potential(
        landau_source.gauge_rate,
        symmetric_source.gauge_rate,
        anchors,
        backend,
    )
    diagonal = np.exp(-1j * gauge_lambda)
    diagonal_dot = -1j * gauge_lambda_dot * diagonal

    def transform(matrix: np.ndarray) -> np.ndarray:
        return diagonal[:, None] * matrix * diagonal.conj()[None, :]

    np.testing.assert_allclose(landau.metric, transform(symmetric.metric), atol=3.0e-14)
    np.testing.assert_allclose(
        landau.mechanical,
        transform(symmetric.mechanical),
        atol=3.0e-13,
    )
    expected_connection = transform(symmetric.connection.connection) + (
        diagonal[:, None]
        * symmetric.metric
        * diagonal_dot.conj()[None, :]
    )
    assert _relative(
        landau.connection.connection - expected_connection,
        expected_connection,
    ) < 2.0e-10


def test_wp6_static_linear_propagation_matches_generalized_spectral_solution() -> None:
    quadrature = _quadrature()
    source = UniformMagneticSourceSample(
        0.0,
        UniformMagneticField((0.0, 0.0, 0.043)),
    )
    sample = evaluate_exact_wilson_one_electron_sample(quadrature, source)
    triple = exact_wilson_one_electron_triple(sample)
    np.testing.assert_allclose(triple.connection, 0.0, atol=0.0, rtol=0.0)
    initial = np.asarray(((1.0,), (0.31 + 0.17j,)), dtype=np.complex128)
    initial /= np.sqrt((initial.conj().T @ sample.metric @ initial).real.item())
    final_time = 1.2
    errors = []
    for intervals in (4, 8, 16):
        step = final_time / intervals
        history = LinearMatrixHistory(
            endpoint_metrics=(sample.metric,) * (intervals + 1),
            midpoint_triples=(triple,) * intervals,
            interval_au=step,
        )
        propagated = propagate_linear_matrix_history(
            initial,
            history,
            quadrature.backend,
        )
        spectral = generalized_spectral_trajectory(
            initial,
            sample.metric,
            sample.mechanical,
            (final_time,),
            quadrature.backend,
        )[0]
        errors.append(np.linalg.norm(propagated.coefficients[-1] - spectral))
        assert max(
            diagnostic.output_metric_residual for diagnostic in propagated.diagnostics
        ) < 2.0e-15
    assert errors[0] / errors[1] > 14.0
    assert errors[1] / errors[2] > 14.0

    first = LinearMatrixHistory(
        endpoint_metrics=(sample.metric,) * 5,
        midpoint_triples=(triple,) * 4,
        interval_au=final_time / 8.0,
    )
    second = LinearMatrixHistory(
        endpoint_metrics=(sample.metric,) * 5,
        midpoint_triples=(triple,) * 4,
        interval_au=final_time / 8.0,
    )
    uninterrupted = propagate_linear_matrix_history(
        initial,
        LinearMatrixHistory(
            endpoint_metrics=(sample.metric,) * 9,
            midpoint_triples=(triple,) * 8,
            interval_au=final_time / 8.0,
        ),
        quadrature.backend,
    )
    restarted_first = propagate_linear_matrix_history(initial, first, quadrature.backend)
    restarted_second = propagate_linear_matrix_history(
        restarted_first.coefficients[-1], second, quadrature.backend
    )
    np.testing.assert_array_equal(
        restarted_second.coefficients[-1],
        uninterrupted.coefficients[-1],
    )


def test_wp6_named_model_triples_are_action_consistent_and_e1_has_exact_limit() -> None:
    quadrature = _quadrature()
    context = prepare_exact_one_electron_model_context(quadrature)
    source = UniformMagneticSourceSample(
        0.27,
        UniformMagneticField((0.0, 0.0, 0.0)),
        electric_field_origin_au=(0.013, -0.009, 0.017),
        origin_au=(0.11, -0.07, 0.05),
    )
    sample = evaluate_exact_wilson_one_electron_sample(quadrature, source)
    models = exact_one_electron_model_triples(sample, context)
    for value in (
        models.exact,
        models.p0,
        models.e1,
        models.geometric_b1,
        models.full_b1,
        models.complete_first_order,
    ):
        residual = value.metric_dot - value.triple.connection - value.triple.connection.conj().T
        assert _relative(residual, value.metric_dot) < 3.0e-14
    np.testing.assert_allclose(
        models.e1.triple.metric,
        models.exact.triple.metric,
        atol=3.0e-15,
        rtol=3.0e-15,
    )
    np.testing.assert_allclose(
        models.e1.triple.hamiltonian_eom,
        models.exact.triple.hamiltonian_eom,
        atol=3.0e-15,
        rtol=3.0e-15,
    )
    np.testing.assert_allclose(
        models.e1.triple.connection,
        models.exact.triple.connection,
        atol=3.0e-15,
        rtol=3.0e-15,
    )
