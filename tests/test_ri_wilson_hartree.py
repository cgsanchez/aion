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
    RIMetricRankPolicy,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
    prepare_ri_wilson_hartree,
    reconstruct_mean_field,
)
from aion.errors import ConfigurationError
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
    evaluator = prepare_ri_wilson_hartree(quadrature, "weigend")
    return reference, quadrature, evaluator


def _symmetric_gauge() -> AffineMagneticGauge:
    return AffineMagneticGauge(
        UniformMagneticField((0.013, -0.009, 0.017)),
        origin_au=(0.17, -0.31, 0.23),
    )


def test_field_free_grid_three_index_matches_independent_pyscf_coulomb_data() -> None:
    reference, quadrature, evaluator = _prepared(level=3)
    density = reference.ground_state.density.astype(np.complex128)
    zero = AffineMagneticGauge(UniformMagneticField((0.0, 0.0, 0.0)))
    result = evaluator.evaluate(density, zero)

    from pyscf import df

    molecule = reconstruct_mean_field(reference, BackendConfig()).mol
    auxiliary = df.addons.make_auxmol(molecule, "weigend")
    analytic_three_index = np.asarray(
        df.incore.aux_e2(molecule, auxiliary, intor="int3c2e", aosym="s1")
    ).transpose(2, 0, 1)
    metric_inverse = np.asarray(evaluator.metric_inverse)
    analytic_moment = np.einsum("ji,Pij->P", density, analytic_three_index, optimize=True).real
    analytic_coefficients = metric_inverse @ analytic_moment
    analytic_energy = 0.5 * analytic_moment @ analytic_coefficients
    analytic_lower = np.einsum(
        "P,Pij->ij", analytic_coefficients, analytic_three_index, optimize=True
    )

    grid_three_index = np.asarray(result.three_index)
    assert (
        np.linalg.norm(grid_three_index - analytic_three_index)
        / np.linalg.norm(analytic_three_index)
        < 5.0e-9
    )
    np.testing.assert_allclose(result.energy, analytic_energy, atol=2.0e-9, rtol=2.0e-9)
    np.testing.assert_allclose(
        result.lower_coulomb_matrix,
        analytic_lower,
        atol=5.0e-9,
        rtol=5.0e-9,
    )

    exact_eri = np.asarray(molecule.intor("int2e"))
    exact_lower = np.einsum("ijkl,lk->ij", exact_eri, density, optimize=True)
    exact_energy = 0.5 * np.einsum("ij,ji->", exact_lower, density, optimize=True).real
    assert abs(analytic_energy - exact_energy) < 2.0e-4
    assert result.pair_counting_residual < 2.0e-13
    assert result.lower_hermiticity_residual < 2.0e-13
    assert result.retained_solve_relative_residual < 5.0e-13
    assert result.stationary_energy_residual < 5.0e-13
    assert result.self_interaction_included
    assert quadrature.grid.npoints > 0


@pytest.mark.parametrize("imaginary", (False, True))
def test_ri_wilson_hartree_lower_matrix_is_unrestricted_matter_derivative(
    imaginary: bool,
) -> None:
    reference, _, evaluator = _prepared(level=2)
    coefficients = reference.ground_state.coefficients.astype(np.complex128)
    occupations = reference.ground_state.occupations
    raw_direction = np.asarray(((0.17, -0.08), (0.11, 0.19)), dtype=np.complex128)
    direction = (1j if imaginary else 1.0) * raw_direction / np.linalg.norm(raw_direction)
    density = np.einsum(
        "mi,i,ni->mn", coefficients, occupations, coefficients.conj(), optimize=True
    )
    gauge = _symmetric_gauge()
    result = evaluator.evaluate(density, gauge)
    density_direction = np.einsum(
        "mi,i,ni->mn", direction, occupations, coefficients.conj(), optimize=True
    ) + np.einsum("mi,i,ni->mn", coefficients, occupations, direction.conj(), optimize=True)
    analytic = np.einsum(
        "ij,ji->", result.lower_coulomb_matrix, density_direction, optimize=True
    ).real

    step = 1.0e-4
    energies: list[float] = []
    for sign in (1.0, -1.0):
        displaced = coefficients + sign * step * direction
        displaced_density = np.einsum(
            "mi,i,ni->mn", displaced, occupations, displaced.conj(), optimize=True
        )
        energies.append(float(evaluator.evaluate(displaced_density, gauge).energy))
    finite = (energies[0] - energies[1]) / (2.0 * step)
    np.testing.assert_allclose(finite, analytic, atol=2.0e-9, rtol=2.0e-8)


@pytest.mark.parametrize("direction_kind", ("physical", "pure_gauge"))
def test_ri_wilson_hartree_fixed_history_source_derivative(
    direction_kind: str,
) -> None:
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
            landau_axis=(gauge.field.magnetic_field_au[1], -gauge.field.magnetic_field_au[0], 0.0),
        )
        direction = AffineGaugeDifferenceVariation(landau, gauge)
    result = evaluator.evaluate(density, gauge, source_direction=direction)
    assert result.source_energy_direction is not None
    assert result.source_moment_direction is not None
    assert result.three_index_source_direction is not None
    assert result.source_moment_imaginary_max_abs is not None
    assert result.source_moment_imaginary_max_abs < 2.0e-13

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


def test_ri_wilson_hartree_gauge_and_general_coefficient_frame_invariance() -> None:
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
        landau_result.moment,
        symmetric_result.moment,
        atol=2.0e-12,
        rtol=2.0e-12,
    )

    change = np.asarray(
        ((1.1 + 0.2j, -0.3 + 0.1j), (0.25 - 0.15j, 0.9 - 0.1j)),
        dtype=np.complex128,
    )
    inverse = np.linalg.inv(change)
    transformed_density = inverse @ density @ inverse.conj().T
    transformed_three_index = np.einsum(
        "ia,Pij,jb->Pab",
        change.conj(),
        symmetric_result.three_index,
        change,
        optimize=True,
    )
    transformed_moment = np.einsum(
        "ba,Pab->P", transformed_density, transformed_three_index, optimize=True
    )
    np.testing.assert_allclose(
        transformed_moment,
        symmetric_result.moment,
        atol=2.0e-12,
        rtol=2.0e-12,
    )


def test_ri_metric_rank_and_cache_policies_are_explicit() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(1),
        block_size=257,
    )
    evaluator = prepare_ri_wilson_hartree(
        quadrature,
        "weigend",
        rank_policy=RIMetricRankPolicy(maximum_rank=5),
        cache_auxiliary_potentials=False,
    )
    assert evaluator.provenance.metric_rank == 5
    assert evaluator.provenance.potential_cache == "blocked_recompute"
    with pytest.raises(ConfigurationError, match="cache requires"):
        prepare_ri_wilson_hartree(
            quadrature,
            "weigend",
            potential_cache_memory_budget_bytes=8,
        )


def test_source_fixed_prepared_action_reuses_the_same_three_index_tensor() -> None:
    reference, _, evaluator = _prepared(level=2)
    density = reference.ground_state.density.astype(np.complex128)
    gauge = _symmetric_gauge()
    prepared = evaluator.prepare_action(gauge)
    first = prepared.evaluate(density)
    second = prepared.evaluate(0.97 * density)

    assert first.three_index is prepared.three_index
    assert second.three_index is prepared.three_index
    direct = evaluator.evaluate(density, gauge)
    np.testing.assert_allclose(first.energy, direct.energy, atol=2.0e-12, rtol=2.0e-12)
    np.testing.assert_allclose(
        first.lower_coulomb_matrix,
        direct.lower_coulomb_matrix,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
