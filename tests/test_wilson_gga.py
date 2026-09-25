from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from aion.config import BackendConfig
from aion.electromagnetism import (
    AffineMagneticGauge,
    GaussianVectorPotentialVariation,
    MagneticGaugeKind,
    PerturbedVectorPotential,
    UniformMagneticField,
    affine_gauge_difference_potential,
)
from aion.electronic_structure import (
    AOGridPolicy,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
    prepare_wilson_gga,
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

    def straight_line_integrals(self, starts_au: object, ends_au: object, backend: Any) -> Any:
        return self.base.straight_line_integrals(
            starts_au, ends_au, backend
        ) + self.scale * self.direction.straight_line_integrals(starts_au, ends_au, backend)

    def straight_line_integral_gradients(
        self, starts_au: object, ends_au: object, backend: Any
    ) -> Any:
        return self.base.straight_line_integral_gradients(
            starts_au, ends_au, backend
        ) + self.scale * self.direction.straight_line_integral_gradients(
            starts_au, ends_au, backend
        )


def _prepared(level: int = 2) -> tuple[Any, Any, Any]:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(level),
        block_size=1024,
    )
    return reference, quadrature, prepare_wilson_gga(quadrature, "pbe")


def _gauge() -> AffineMagneticGauge:
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
        "pbe",
        density.real,
    )

    np.testing.assert_allclose(result.electron_count_grid, electron_count, atol=3.0e-11)
    np.testing.assert_allclose(result.energy, energy, atol=3.0e-11, rtol=3.0e-11)
    np.testing.assert_allclose(
        result.lower_xc_matrix,
        lower,
        atol=5.0e-11,
        rtol=5.0e-11,
    )
    assert result.lower_hermiticity_residual < 2.0e-13
    assert result.density_imaginary_max_abs < 2.0e-14
    assert result.density_gradient_imaginary_max_abs < 2.0e-14
    assert evaluator.provenance.functional_family == "GGA"
    assert evaluator.provenance.realization == "quadrature--Wilson GGA"


@pytest.mark.parametrize("imaginary", (False, True))
def test_weak_lower_matrix_is_complete_density_gradient_derivative(
    imaginary: bool,
) -> None:
    reference, _, evaluator = _prepared(level=2)
    coefficients = reference.ground_state.coefficients.astype(np.complex128)
    occupations = reference.ground_state.occupations
    raw_direction = np.asarray(((0.17, -0.08), (0.11, 0.19)), dtype=np.complex128)
    direction = (1j if imaginary else 1.0) * raw_direction / np.linalg.norm(raw_direction)
    density = _coefficient_density(coefficients, occupations)
    result = evaluator.evaluate(density, _gauge())
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
    analytic = np.einsum("ij,ji->", result.lower_xc_matrix, density_direction, optimize=True).real

    step = 1.0e-4
    plus = _coefficient_density(coefficients + step * direction, occupations)
    minus = _coefficient_density(coefficients - step * direction, occupations)
    finite = (
        float(evaluator.evaluate(plus, _gauge()).energy)
        - float(evaluator.evaluate(minus, _gauge()).energy)
    ) / (2.0 * step)
    np.testing.assert_allclose(finite, analytic, atol=5.0e-9, rtol=5.0e-8)


def test_fixed_history_source_derivative_includes_density_gradient_direction() -> None:
    reference, _, evaluator = _prepared(level=2)
    density = reference.ground_state.density.astype(np.complex128)
    gauge = _gauge()
    direction = AffineMagneticGauge(
        UniformMagneticField((-0.4, 0.7, 0.2)),
        origin_au=gauge.origin_au,
    )
    result = evaluator.evaluate(density, gauge, source_direction=direction)
    assert result.source_energy_direction is not None
    assert result.source_density_direction is not None
    assert result.source_density_gradient_direction is not None
    assert float(np.linalg.norm(result.source_density_gradient_direction)) > 1.0e-8

    step = 1.0e-4
    plus = evaluator.evaluate(density, _AffineSourceCurve(gauge, direction, step))
    minus = evaluator.evaluate(density, _AffineSourceCurve(gauge, direction, -step))
    finite = (float(plus.energy) - float(minus.energy)) / (2.0 * step)
    np.testing.assert_allclose(
        finite,
        result.source_energy_direction,
        atol=8.0e-9,
        rtol=8.0e-8,
    )


def test_localized_fixed_history_source_derivative() -> None:
    _, _, evaluator = _prepared(level=2)
    coefficient = np.asarray(((0.71 + 0.19j,), (-0.23 + 0.41j,)))
    density = 2.0 * coefficient @ coefficient.conj().T
    gauge = _gauge()
    direction = GaussianVectorPotentialVariation(
        amplitude_au=(0.19, -0.13, 0.07),
        center_au=(0.23, -0.17, 0.11),
        exponent_au_inverse2=0.41,
        path_quadrature_order=24,
    )
    result = evaluator.evaluate(density, gauge, source_direction=direction)
    assert result.source_energy_direction is not None

    step = 1.0e-4
    plus = evaluator.evaluate(density, PerturbedVectorPotential(gauge, direction, step))
    minus = evaluator.evaluate(density, PerturbedVectorPotential(gauge, direction, -step))
    finite = (float(plus.energy) - float(minus.energy)) / (2.0 * step)
    np.testing.assert_allclose(
        finite,
        result.source_energy_direction,
        atol=8.0e-9,
        rtol=8.0e-8,
    )


def test_gauge_and_general_coefficient_frame_covariance() -> None:
    reference, quadrature, evaluator = _prepared(level=2)
    density = reference.ground_state.density.astype(np.complex128)
    symmetric = _gauge()
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
        atol=3.0e-12,
        rtol=3.0e-12,
    )
    np.testing.assert_allclose(
        landau_result.lower_xc_matrix,
        unitary[:, None] * np.asarray(symmetric_result.lower_xc_matrix) * unitary.conj()[None, :],
        atol=5.0e-12,
        rtol=5.0e-12,
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
        atol=3.0e-12,
        rtol=3.0e-12,
    )
    np.testing.assert_allclose(
        transformed.lower_xc_matrix,
        change.conj().T @ symmetric_result.lower_xc_matrix @ change,
        atol=5.0e-12,
        rtol=5.0e-12,
    )
    np.testing.assert_allclose(
        transformed.density_gradient,
        symmetric_result.density_gradient,
        atol=3.0e-12,
        rtol=3.0e-12,
    )
    assert transformed.coefficient_frame_applied


def test_non_gga_functional_is_rejected() -> None:
    _, quadrature, _ = _prepared(level=1)
    with pytest.raises(UnsupportedConfigurationError, match="pure GGA only"):
        prepare_wilson_gga(quadrature, "lda,vwn")
