from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aion.config import BackendConfig, XCFamily
from aion.electromagnetism import (
    AffineMagneticGauge,
    GaussianScalarGaugeVariation,
    GaussianVectorPotentialVariation,
    PerturbedVectorPotential,
    UniformMagneticField,
    UniformMagneticSourceSample,
)
from aion.electronic_structure import (
    AOGridPolicy,
    AOQuadrature,
    WilsonStationaryBranch,
    evaluate_exact_static_wilson_grid_one_electron_action,
    evaluate_exact_wilson_charge,
    evaluate_exact_wilson_one_electron_sample,
    evaluate_nonlinear_density_pure_gauge_ward,
    evaluate_nonlinear_pure_gauge_ward,
    evaluate_nonlinear_weak_continuity,
    evaluate_nonlinear_weak_current_pairing,
    evaluate_static_nonlinear_wilson_grid_action,
    prepare_ao_quadrature,
    prepare_exact_wilson_stationary_factory,
    prepare_pyscf_reference,
)
from aion.formulations import EOMTriple
from aion.formulations.exact_one_electron import exact_pure_gauge_action_direction
from test_reference_integration import molecular_config

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def quadrature() -> AOQuadrature:
    base = molecular_config("h2")
    config = replace(
        base,
        electronic_structure=replace(
            base.electronic_structure,
            functional="lda,vwn",
            xc_family=XCFamily.LDA,
            grid_level=2,
            density_fitting=True,
            auxiliary_basis="weigend",
        ),
    )
    reference = prepare_pyscf_reference(config)
    return prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(2),
        block_size=1024,
    )


def _model_sample(
    quadrature: AOQuadrature,
    branch: WilsonStationaryBranch,
) -> tuple[object, object]:
    gauge = AffineMagneticGauge(
        UniformMagneticField((0.0, 0.0, 0.017)),
        origin_au=(0.13, -0.19, 0.07),
    )
    factory = prepare_exact_wilson_stationary_factory(
        quadrature,
        auxiliary_basis="weigend",
        functional="lda,vwn",
    )
    model = factory.model(gauge, branch)
    sample = evaluate_exact_wilson_one_electron_sample(
        quadrature,
        UniformMagneticSourceSample(
            0.0,
            gauge.field,
            origin_au=gauge.origin_au,
            gauge_kind=gauge.kind,
            landau_axis=gauge.landau_axis,
        ),
    )
    return model, sample


def _normalized_density(metric: np.ndarray, occupation: float = 2.0) -> np.ndarray:
    coefficient = np.asarray(((0.71 + 0.19j,), (-0.23 + 0.41j,)))
    coefficient /= np.sqrt((coefficient.conj().T @ metric @ coefficient).real.item())
    return occupation * coefficient @ coefficient.conj().T


def test_scalar_gauge_variation_has_exact_endpoint_integral() -> None:
    from aion.backends import NumPyBackend

    variation = GaussianScalarGaugeVariation(
        amplitude=0.37,
        center_au=(0.11, -0.17, 0.23),
        exponent_au_inverse2=0.41,
    )
    starts = np.asarray(((0.1, -0.2, 0.3), (-0.4, 0.2, 0.1)))
    ends = np.asarray(((0.7, 0.1, -0.2), (0.3, -0.5, 0.4)))
    backend = NumPyBackend()
    np.testing.assert_allclose(
        variation.straight_line_integrals(starts, ends, backend),
        variation.scalar_field(ends, backend) - variation.scalar_field(starts, backend),
        atol=0.0,
        rtol=0.0,
    )
    step = 2.0e-5
    finite = np.empty_like(ends)
    for axis in range(3):
        displacement = np.zeros_like(ends)
        displacement[:, axis] = step
        finite[:, axis] = (
            variation.straight_line_integrals(starts, ends + displacement, backend)
            - variation.straight_line_integrals(starts, ends - displacement, backend)
        ) / (2.0 * step)
    np.testing.assert_allclose(
        variation.straight_line_integral_gradients(starts, ends, backend),
        finite,
        atol=3.0e-11,
        rtol=3.0e-10,
    )


@pytest.mark.parametrize(
    "branch",
    (WilsonStationaryBranch.HARTREE, WilsonStationaryBranch.KOHN_SHAM_LDA),
)
def test_complete_nonlinear_source_pairing_matches_fixed_history_action_difference(
    quadrature: AOQuadrature,
    branch: WilsonStationaryBranch,
) -> None:
    model, sample = _model_sample(quadrature, branch)
    grid = evaluate_exact_static_wilson_grid_one_electron_action(
        quadrature,
        model.gauge,
    )
    triple = EOMTriple(
        grid.overlap,
        grid.mechanical,
        np.zeros_like(grid.overlap),
    )
    density = _normalized_density(np.asarray(grid.overlap))
    velocity = np.asarray(((0.13 + 0.29j, -0.17 + 0.07j), (0.11 - 0.19j, -0.23 + 0.31j)))
    variation = GaussianVectorPotentialVariation(
        amplitude_au=(0.19, -0.13, 0.07),
        center_au=(0.23, -0.17, 0.11),
        exponent_au_inverse2=0.41,
        path_quadrature_order=24,
    )
    analytic = evaluate_nonlinear_weak_current_pairing(
        model,
        sample,
        density,
        velocity,
        variation,
        one_electron_triple=triple,
    )
    np.testing.assert_allclose(
        analytic.response.kinetic,
        analytic.response.kinetic_embedding + analytic.response.kinetic_explicit,
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(
        analytic.response.metric,
        analytic.response.frame_overlap + analytic.response.frame_overlap.conj().T,
        atol=2.0e-14,
        rtol=2.0e-14,
    )
    assert abs(float(analytic.closure_pairing)) > 1.0e-7

    errors: list[float] = []
    for step in (1.0e-3, 3.0e-4, 1.0e-4):
        values = []
        for sign in (1.0, -1.0):
            perturbed = PerturbedVectorPotential(
                model.gauge,
                variation,
                sign * step,
            )
            values.append(
                float(
                    evaluate_static_nonlinear_wilson_grid_action(
                        model,
                        density,
                        velocity,
                        perturbed,
                    ).electronic_action_value
                )
            )
        finite = (values[0] - values[1]) / (2.0 * step)
        errors.append(abs(finite - float(analytic.total_pairing)))
    assert errors[-1] < 2.0e-7
    assert min(errors) < 1.0e-8


def test_exact_charge_off_shell_ward_and_on_shell_weak_continuity(
    quadrature: AOQuadrature,
) -> None:
    model, sample = _model_sample(quadrature, WilsonStationaryBranch.KOHN_SHAM_LDA)
    grid = evaluate_exact_static_wilson_grid_one_electron_action(
        quadrature,
        model.gauge,
    )
    grid_triple = EOMTriple(
        grid.overlap,
        grid.mechanical,
        np.zeros_like(grid.overlap),
    )
    density = _normalized_density(np.asarray(grid.overlap))
    charge = evaluate_exact_wilson_charge(model, density)
    np.testing.assert_allclose(charge.integrated_charge_grid, -2.0, atol=3.0e-8, rtol=0.0)
    np.testing.assert_allclose(
        charge.integrated_charge_metric,
        -2.0,
        atol=3.0e-8,
        rtol=0.0,
    )

    coefficient = np.linalg.eigh(np.asarray(grid.overlap))[1][:, :1].astype(np.complex128)
    norm = np.sqrt((coefficient.conj().T @ grid.overlap @ coefficient).real.item())
    coefficient /= norm
    coefficient_velocity = np.asarray(((0.17 + 0.31j,), (-0.29 + 0.13j,)))
    gauge_test = GaussianScalarGaugeVariation(
        amplitude=0.37,
        center_au=(0.11, -0.17, 0.23),
        exponent_au_inverse2=0.41,
    )
    gauge_rate = GaussianScalarGaugeVariation(
        amplitude=-0.23,
        center_au=gauge_test.center_au,
        exponent_au_inverse2=gauge_test.exponent_au_inverse2,
    )
    ward = evaluate_nonlinear_pure_gauge_ward(
        model,
        sample,
        coefficient,
        coefficient_velocity,
        np.asarray((2.0,)),
        gauge_test,
        gauge_parameter_rate=gauge_rate,
    )
    assert ward.lower_coefficient_residual_relative_norm > 1.0e-3
    assert abs(float(ward.source_pairing)) > 1.0e-5
    np.testing.assert_allclose(ward.total_ward_residual, 0.0, atol=3.0e-12, rtol=0.0)
    orbital_density = 2.0 * coefficient @ coefficient.conj().T
    velocity_density = 2.0 * coefficient_velocity @ coefficient.conj().T
    density_ward = evaluate_nonlinear_density_pure_gauge_ward(
        model,
        sample,
        orbital_density,
        velocity_density,
        gauge_test,
        gauge_parameter_rate=gauge_rate,
    )
    np.testing.assert_allclose(
        density_ward.source_pairing,
        ward.source_pairing,
        atol=3.0e-13,
        rtol=3.0e-13,
    )
    np.testing.assert_allclose(
        density_ward.matter_pairing,
        ward.matter_pairing,
        atol=3.0e-13,
        rtol=3.0e-13,
    )
    np.testing.assert_allclose(
        density_ward.total_ward_residual,
        0.0,
        atol=3.0e-12,
        rtol=0.0,
    )

    dynamic_source = UniformMagneticSourceSample(
        0.5,
        model.gauge.field,
        magnetic_field_dot_au=(0.0, 0.0, 0.007),
        origin_au=model.gauge.origin_au,
    )
    dynamic_sample = evaluate_exact_wilson_one_electron_sample(
        quadrature,
        dynamic_source,
    )
    dynamic_current = evaluate_nonlinear_weak_current_pairing(
        model,
        dynamic_sample,
        orbital_density,
        velocity_density,
        gauge_test,
    )
    site_values = gauge_test.scalar_field(
        quadrature.reference.core_operators.nuclei.coordinates_au,
        model.backend,
    )
    pure_direction = exact_pure_gauge_action_direction(
        dynamic_sample,
        orbital_density,
        velocity_density,
        site_values,
        np.zeros_like(site_values),
        quadrature.reference.anchor_topology.ao_to_atom,
        model.backend,
    )
    np.testing.assert_allclose(
        dynamic_current.response.connection,
        pure_direction.matrix.connection,
        atol=3.0e-14,
        rtol=3.0e-14,
    )
    dynamic_continuity = evaluate_nonlinear_weak_continuity(
        model,
        dynamic_sample,
        orbital_density,
        gauge_test,
    )
    np.testing.assert_allclose(
        dynamic_continuity.finite_region_residual,
        0.0,
        atol=2.0e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        dynamic_continuity.global_charge_residual,
        0.0,
        atol=3.0e-12,
        rtol=0.0,
    )

    continuity = evaluate_nonlinear_weak_continuity(
        model,
        sample,
        density,
        gauge_test,
        one_electron_triple=grid_triple,
    )
    np.testing.assert_allclose(
        continuity.finite_region_residual,
        0.0,
        atol=5.0e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        continuity.global_charge_residual,
        0.0,
        atol=3.0e-12,
        rtol=0.0,
    )
