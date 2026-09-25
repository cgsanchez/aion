from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from aion.config import BackendConfig
from aion.electromagnetism import (
    AffineGaugeDifferenceVariation,
    AffineMagneticGauge,
    MagneticGaugeKind,
    UniformMagneticField,
    affine_gauge_difference_potential,
)
from aion.electronic_structure import (
    AOGridPolicy,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
    prepare_wilson_lda,
    reconstruct_mean_field,
)
from aion.errors import UnsupportedConfigurationError
from test_reference_integration import molecular_config

pytestmark = pytest.mark.integration


@dataclass(frozen=True, slots=True)
class _AffineSourceCurve:
    base: Any
    direction: Any
    scale: float

    def straight_line_integrals(
        self,
        starts_au: object,
        ends_au: object,
        backend: Any,
    ) -> Any:
        return self.base.straight_line_integrals(
            starts_au, ends_au, backend
        ) + self.scale * self.direction.straight_line_integrals(starts_au, ends_au, backend)


def _prepared(level: int = 2) -> tuple[Any, Any, Any]:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(level),
        block_size=1024,
    )
    return reference, quadrature, prepare_wilson_lda(quadrature, "lda,vwn")


def _symmetric_gauge() -> AffineMagneticGauge:
    return AffineMagneticGauge(
        UniformMagneticField((0.013, -0.009, 0.017)),
        origin_au=(0.17, -0.31, 0.23),
    )


def _coefficient_density(coefficients: np.ndarray, occupations: np.ndarray) -> np.ndarray:
    return np.einsum(
        "mi,i,ni->mn",
        coefficients,
        occupations,
        coefficients.conj(),
        optimize=True,
    )


def test_zero_field_action_matches_independent_pyscf_same_grid() -> None:
    reference, quadrature, evaluator = _prepared(level=3)
    density = reference.ground_state.density.astype(np.complex128)
    zero = AffineMagneticGauge(UniformMagneticField((0.0, 0.0, 0.0)))
    result = evaluator.evaluate(density, zero)

    from pyscf import dft

    molecule = reconstruct_mean_field(reference, BackendConfig()).mol
    grids = dft.gen_grid.Grids(molecule)
    grids.coords = np.array(quadrature.grid.coordinates_au, copy=True)
    grids.weights = np.array(quadrature.grid.weights_au, copy=True)
    grids.non0tab = grids.make_mask(molecule, grids.coords)
    electron_count, energy, lower = dft.numint.NumInt().nr_rks(
        molecule,
        grids,
        "lda,vwn",
        density.real,
    )

    np.testing.assert_allclose(result.electron_count_grid, electron_count, atol=2.0e-12)
    np.testing.assert_allclose(result.energy, energy, atol=2.0e-12, rtol=2.0e-12)
    np.testing.assert_allclose(
        result.lower_xc_matrix,
        lower,
        atol=3.0e-12,
        rtol=3.0e-12,
    )
    assert result.lower_hermiticity_residual < 2.0e-13
    assert result.density_imaginary_max_abs < 2.0e-14
    assert result.density_real_minimum >= -2.0e-14
    assert evaluator.provenance.functional == "lda,vwn"
    assert evaluator.provenance.functional_family == "LDA"


@pytest.mark.parametrize("imaginary", (False, True))
def test_lower_matrix_is_unrestricted_matter_derivative(imaginary: bool) -> None:
    reference, _, evaluator = _prepared(level=2)
    coefficients = reference.ground_state.coefficients.astype(np.complex128)
    occupations = reference.ground_state.occupations
    raw_direction = np.asarray(((0.17, -0.08), (0.11, 0.19)), dtype=np.complex128)
    direction = (1j if imaginary else 1.0) * raw_direction / np.linalg.norm(raw_direction)
    density = _coefficient_density(coefficients, occupations)
    gauge = _symmetric_gauge()
    result = evaluator.evaluate(density, gauge)
    density_direction = np.einsum(
        "mi,i,ni->mn",
        direction,
        occupations,
        coefficients.conj(),
        optimize=True,
    ) + np.einsum(
        "mi,i,ni->mn",
        coefficients,
        occupations,
        direction.conj(),
        optimize=True,
    )
    analytic = np.einsum(
        "ij,ji->",
        result.lower_xc_matrix,
        density_direction,
        optimize=True,
    ).real

    step = 1.0e-4
    plus = _coefficient_density(coefficients + step * direction, occupations)
    minus = _coefficient_density(coefficients - step * direction, occupations)
    finite = (
        float(evaluator.evaluate(plus, gauge).energy)
        - float(evaluator.evaluate(minus, gauge).energy)
    ) / (2.0 * step)
    np.testing.assert_allclose(finite, analytic, atol=2.0e-9, rtol=2.0e-8)


@pytest.mark.parametrize("direction_kind", ("physical", "pure_gauge"))
def test_fixed_history_source_derivative(direction_kind: str) -> None:
    reference, _, evaluator = _prepared(level=2)
    density = reference.ground_state.density.astype(np.complex128)
    gauge = _symmetric_gauge()
    if direction_kind == "physical":
        direction: Any = AffineMagneticGauge(
            UniformMagneticField((-0.4, 0.7, 0.2)),
            origin_au=gauge.origin_au,
        )
    else:
        landau = AffineMagneticGauge(
            gauge.field,
            kind=MagneticGaugeKind.LANDAU,
            origin_au=(-0.21, 0.08, -0.19),
            landau_axis=(
                gauge.field.magnetic_field_au[1],
                -gauge.field.magnetic_field_au[0],
                0.0,
            ),
        )
        direction = AffineGaugeDifferenceVariation(landau, gauge)
    result = evaluator.evaluate(density, gauge, source_direction=direction)
    assert result.source_energy_direction is not None
    assert result.source_density_direction is not None
    assert result.source_density_imaginary_max_abs is not None
    assert result.source_density_imaginary_max_abs < 2.0e-13

    step = 1.0e-4
    plus = evaluator.evaluate(density, _AffineSourceCurve(gauge, direction, step))
    minus = evaluator.evaluate(density, _AffineSourceCurve(gauge, direction, -step))
    finite = (float(plus.energy) - float(minus.energy)) / (2.0 * step)
    np.testing.assert_allclose(
        finite,
        result.source_energy_direction,
        atol=3.0e-9,
        rtol=3.0e-8,
    )


def test_gauge_and_general_coefficient_frame_covariance() -> None:
    reference, quadrature, evaluator = _prepared(level=2)
    density = reference.ground_state.density.astype(np.complex128)
    symmetric = _symmetric_gauge()
    landau = AffineMagneticGauge(
        symmetric.field,
        kind=MagneticGaugeKind.LANDAU,
        origin_au=(-0.21, 0.08, -0.19),
        landau_axis=(
            symmetric.field.magnetic_field_au[1],
            -symmetric.field.magnetic_field_au[0],
            0.0,
        ),
    )
    anchors = reference.core_operators.nuclei.coordinates_au[reference.anchor_topology.ao_to_atom]
    gauge_function = np.asarray(
        affine_gauge_difference_potential(
            landau,
            symmetric,
            anchors,
            quadrature.backend,
        )
    )
    unitary = np.exp(-1j * gauge_function)
    landau_density = unitary[:, None] * density * unitary.conj()[None, :]
    symmetric_result = evaluator.evaluate(density, symmetric)
    landau_result = evaluator.evaluate(landau_density, landau)
    np.testing.assert_allclose(
        landau_result.energy,
        symmetric_result.energy,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        landau_result.lower_xc_matrix,
        unitary[:, None] * np.asarray(symmetric_result.lower_xc_matrix) * unitary.conj()[None, :],
        atol=3.0e-12,
        rtol=3.0e-12,
    )

    change = np.asarray(
        ((1.1 + 0.2j, -0.3 + 0.1j), (0.25 - 0.15j, 0.9 - 0.1j)),
        dtype=np.complex128,
    )
    inverse = np.linalg.inv(change)
    transformed_density = inverse @ density @ inverse.conj().T
    transformed = evaluator.evaluate(
        transformed_density,
        symmetric,
        coefficient_frame=change,
    )
    np.testing.assert_allclose(
        transformed.energy,
        symmetric_result.energy,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        transformed.lower_xc_matrix,
        change.conj().T @ symmetric_result.lower_xc_matrix @ change,
        atol=3.0e-12,
        rtol=3.0e-12,
    )
    assert transformed.coefficient_frame_applied


def test_non_lda_functional_is_rejected() -> None:
    _, quadrature, _ = _prepared(level=1)
    with pytest.raises(UnsupportedConfigurationError, match="pure LDA only"):
        prepare_wilson_lda(quadrature, "pbe")
