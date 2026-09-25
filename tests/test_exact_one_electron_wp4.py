from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aion.config import (
    AtomConfig,
    BackendConfig,
    ElectromagneticOrigin,
    OneElectronReferenceConfig,
)
from aion.electromagnetism import UniformMagneticField
from aion.electronic_structure import (
    AOGridPolicy,
    evaluate_exact_uniform_electric_internal_connections,
    evaluate_magnetic_one_electron_first_derivatives,
    evaluate_magnetic_one_electron_matrices,
    evaluate_uniform_electric_e1_tensor,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
    static_magnetic_first_order_models,
)

pytestmark = pytest.mark.integration

_FIXTURE = Path(__file__).parent / "fixtures/exact_one_electron/hh_sto3g.fixture.json"


def _quadrature(fixture: Path = _FIXTURE) -> object:
    values = json.loads(fixture.read_text(encoding="utf-8"))["config"]
    reference = prepare_one_electron_ao_reference(
        OneElectronReferenceConfig(
            atoms=tuple(
                AtomConfig(atom["symbol"], tuple(atom["position_au"])) for atom in values["atoms"]
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


def test_wp4_uniform_e1_tensor_matches_position_integral_quadrature() -> None:
    quadrature = _quadrature()
    tensor = evaluate_uniform_electric_e1_tensor(quadrature)

    np.testing.assert_allclose(
        tensor.quadrature_central_dipoles,
        tensor.central_dipoles,
        atol=2.0e-9,
        rtol=2.0e-9,
    )
    np.testing.assert_allclose(
        tensor.quadrature_connection_derivatives,
        tensor.connection_derivatives,
        atol=2.0e-9,
        rtol=2.0e-9,
    )
    np.testing.assert_allclose(
        tensor.connection_derivatives + np.swapaxes(tensor.connection_derivatives.conj(), 1, 2),
        0.0,
        atol=2.0e-15,
        rtol=0.0,
    )


def test_wp4_uniform_e1_is_derivative_of_direct_exact_parent() -> None:
    quadrature = _quadrature()
    tensor = evaluate_uniform_electric_e1_tensor(quadrature)
    direction = np.asarray((0.31, -0.47, 0.67), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    steps = (4.0e-4, 2.0e-4, 1.0e-4)
    fields = [(0.0, 0.0, 0.0)]
    for step in steps:
        fields.extend((step * direction, -step * direction))
    connections = evaluate_exact_uniform_electric_internal_connections(
        quadrature,
        fields,
    )
    np.testing.assert_array_equal(connections[0], np.zeros_like(connections[0]))
    expected = np.einsum("x,xmn->mn", direction, tensor.connection_derivatives)
    for index, step in enumerate(steps):
        plus = connections[1 + 2 * index]
        minus = connections[2 + 2 * index]
        derivative = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(derivative, expected, atol=2.0e-9, rtol=2.0e-9)


def test_wp4_uniform_e1_and_site_term_reconstruct_charge_position() -> None:
    quadrature = _quadrature()
    tensor = evaluate_uniform_electric_e1_tensor(quadrature)
    reference = quadrature.reference
    charge = tensor.charge
    anchors = reference.core_operators.nuclei.coordinates_au[reference.anchor_topology.ao_to_atom]
    centers = 0.5 * (anchors[:, None, :] + anchors[None, :, :])
    reconstructed = (
        tensor.central_dipoles
        + charge * np.moveaxis(centers, -1, 0) * reference.core_operators.overlap[None, :, :]
    )
    np.testing.assert_allclose(
        reconstructed,
        charge * reference.core_operators.position,
        atol=2.0e-15,
        rtol=0.0,
    )


def test_wp4_magnetic_derivatives_match_exact_parent_and_pair_adjoint() -> None:
    quadrature = _quadrature(
        Path(__file__).parent / "fixtures/exact_one_electron/oh_sto3g.fixture.json"
    )
    derivatives = evaluate_magnetic_one_electron_first_derivatives(quadrature)
    direction = np.asarray((0.31, -0.47, 0.67), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    step = 2.0e-4
    plus, minus = evaluate_magnetic_one_electron_matrices(
        quadrature,
        (
            UniformMagneticField(tuple(step * direction)),
            UniformMagneticField(tuple(-step * direction)),
        ),
        include_direct_oracle=False,
    )
    comparisons = (
        (
            (plus.overlap.exact - minus.overlap.exact) / (2.0 * step),
            np.einsum("x,xmn->mn", direction, derivatives.metric),
        ),
        (
            (plus.kinetic.exact - minus.kinetic.exact) / (2.0 * step),
            np.einsum("x,xmn->mn", direction, derivatives.kinetic),
        ),
        (
            (plus.nuclear_attraction.exact - minus.nuclear_attraction.exact) / (2.0 * step),
            np.einsum("x,xmn->mn", direction, derivatives.nuclear_attraction_triangle),
        ),
    )
    for finite_difference, analytic in comparisons:
        np.testing.assert_allclose(
            finite_difference,
            analytic,
            atol=2.0e-7,
            rtol=2.0e-7,
        )
    for coefficients in (
        derivatives.metric,
        derivatives.kinetic_triangle,
        derivatives.kinetic_anchored_pC + derivatives.kinetic_anchored_Cp,
        derivatives.nuclear_attraction_triangle,
        derivatives.mechanical,
    ):
        np.testing.assert_allclose(
            coefficients,
            np.swapaxes(coefficients.conj(), 1, 2),
            atol=2.0e-11,
            rtol=2.0e-11,
        )


def test_wp4_static_model_remainders_have_declared_orders() -> None:
    quadrature = _quadrature(
        Path(__file__).parent / "fixtures/exact_one_electron/oh_sto3g.fixture.json"
    )
    direction = np.asarray((0.31, -0.47, 0.67), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    steps = np.asarray((5.0e-4, 1.0e-3, 2.0e-3, 4.0e-3))
    results = evaluate_magnetic_one_electron_matrices(
        quadrature,
        tuple(UniformMagneticField(tuple(step * direction)) for step in steps),
        include_direct_oracle=False,
    )
    errors: dict[str, list[float]] = {
        "p0_metric": [],
        "gb1_metric": [],
        "p0_mechanical": [],
        "gb1_mechanical": [],
        "b1_mechanical": [],
    }
    for result in results:
        models = static_magnetic_first_order_models(result).barred
        errors["p0_metric"].append(np.linalg.norm(models.exact.overlap - models.p0.overlap))
        errors["gb1_metric"].append(
            np.linalg.norm(models.exact.overlap - models.geometric_b1.overlap)
        )
        errors["p0_mechanical"].append(
            np.linalg.norm(models.exact.mechanical - models.p0.mechanical)
        )
        errors["gb1_mechanical"].append(
            np.linalg.norm(models.exact.mechanical - models.geometric_b1.mechanical)
        )
        errors["b1_mechanical"].append(
            np.linalg.norm(models.exact.mechanical - models.full_b1.mechanical)
        )

    expected = {
        "p0_metric": 1.0,
        "gb1_metric": 2.0,
        "p0_mechanical": 1.0,
        "gb1_mechanical": 1.0,
        "b1_mechanical": 2.0,
    }
    for name, values in errors.items():
        fitted = np.polyfit(np.log(steps), np.log(values), 1)[0]
        assert fitted == pytest.approx(expected[name], abs=0.04)
