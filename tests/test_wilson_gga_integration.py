from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aion.config import BackendConfig, XCFamily
from aion.electromagnetism import (
    GaussianScalarGaugeVariation,
    GaussianVectorPotentialVariation,
    PerturbedVectorPotential,
    UniformMagneticField,
    UniformMagneticSourceSample,
)
from aion.electronic_structure import (
    AOGridPolicy,
    WilsonStationaryBranch,
    evaluate_exact_static_wilson_grid_one_electron_action,
    evaluate_exact_wilson_power,
    evaluate_nonlinear_weak_continuity,
    evaluate_nonlinear_weak_current_pairing,
    evaluate_static_nonlinear_wilson_grid_action,
    prepare_ao_quadrature,
    prepare_exact_wilson_dynamic_sample,
    prepare_exact_wilson_stationary_factory,
    prepare_pyscf_reference,
)
from aion.formulations import EOMTriple
from aion.propagation import (
    NonlinearGaussMagnusPolicy,
    propagate_nonlinear_contravariant_density,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def gga_factory() -> object:
    base = molecular_config("h2")
    config = replace(
        base,
        electronic_structure=replace(
            base.electronic_structure,
            functional="pbe",
            xc_family=XCFamily.GGA,
            grid_level=2,
            density_fitting=True,
            auxiliary_basis="weigend",
        ),
    )
    reference = prepare_pyscf_reference(config)
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(2),
        block_size=1024,
    )
    return prepare_exact_wilson_stationary_factory(
        quadrature,
        auxiliary_basis="weigend",
        functional="pbe",
    )


def _source(time_au: float = 0.3) -> UniformMagneticSourceSample:
    return UniformMagneticSourceSample(
        time_au=time_au,
        field=UniformMagneticField((0.0, 0.0, 0.017 + 0.001 * time_au)),
        magnetic_field_dot_au=(0.0, 0.0, 0.001),
        electric_field_origin_au=(0.004, -0.003, 0.002),
        origin_au=(0.13, -0.19, 0.07),
    )


def _normalized_density(metric: np.ndarray) -> np.ndarray:
    coefficient = np.asarray(((0.71 + 0.19j,), (-0.23 + 0.41j,)))
    coefficient /= np.sqrt((coefficient.conj().T @ metric @ coefficient).real.item())
    return 2.0 * coefficient @ coefficient.conj().T


def test_pure_gga_stationary_state_and_short_propagation(gga_factory: object) -> None:
    factory = gga_factory
    zero_source = UniformMagneticSourceSample(
        time_au=0.0,
        field=UniformMagneticField((0.0, 0.0, 0.0)),
    )
    zero_model = factory.model(
        zero_source.gauge,
        WilsonStationaryBranch.KOHN_SHAM_GGA,
    )
    state = zero_model.solve()
    assert state.orbital_residual < 2.0e-9
    assert state.density_fixed_point_residual < 2.0e-9
    assert state.commutator_residual < 2.0e-9
    assert state.double_counting_residual_au < 2.0e-11
    assert abs(
        float(state.action.energy_molecular_total_au)
        - factory.quadrature.reference.ground_state.energy_total_au
    ) < 5.0e-8
    assert state.action.exchange_correlation is not None
    assert state.action.exchange_correlation.realization == "quadrature--Wilson GGA"

    sample = prepare_exact_wilson_dynamic_sample(
        factory,
        _source(),
        WilsonStationaryBranch.KOHN_SHAM_GGA,
    )
    density = _normalized_density(np.asarray(sample.one_electron.metric))
    evaluation = sample.evaluate(density)
    power = evaluate_exact_wilson_power(evaluation, density)
    assert power.current.exchange_correlation is not None
    assert power.current.exchange_correlation.realization == "quadrature--Wilson GGA"
    np.testing.assert_allclose(
        power.matrix_mechanical_energy_rate_au,
        power.source_power_au,
        atol=3.0e-6,
        rtol=0.0,
    )

    cache: dict[float, object] = {}

    def dynamic(time_au: float) -> object:
        if time_au not in cache:
            cache[time_au] = prepare_exact_wilson_dynamic_sample(
                factory,
                _source(time_au),
                WilsonStationaryBranch.KOHN_SHAM_GGA,
            )
        return cache[time_au]

    trajectory = propagate_nonlinear_contravariant_density(
        density,
        initial_time_au=0.0,
        interval_au=0.05,
        intervals=2,
        metric_provider=lambda time: dynamic(time).one_electron.metric,
        eom_provider=lambda time, value: dynamic(time).evaluate(value).triple,
        backend=factory.quadrature.backend,
        policy=NonlinearGaussMagnusPolicy(tolerance=1.0e-10),
    )
    assert trajectory.metric_correction_applied is False
    assert max(
        item.contravariant_hermiticity_residual for item in trajectory.diagnostics
    ) < 2.0e-15


def test_pure_gga_source_difference_and_weak_continuity(gga_factory: object) -> None:
    factory = gga_factory
    dynamic = prepare_exact_wilson_dynamic_sample(
        factory,
        UniformMagneticSourceSample(
            time_au=0.0,
            field=UniformMagneticField((0.0, 0.0, 0.017)),
            origin_au=(0.13, -0.19, 0.07),
        ),
        WilsonStationaryBranch.KOHN_SHAM_GGA,
    )
    model = dynamic.model
    grid = evaluate_exact_static_wilson_grid_one_electron_action(
        factory.quadrature,
        model.gauge,
    )
    triple = EOMTriple(grid.overlap, grid.mechanical, np.zeros_like(grid.overlap))
    density = _normalized_density(np.asarray(grid.overlap))
    velocity = np.asarray(
        ((0.13 + 0.29j, -0.17 + 0.07j), (0.11 - 0.19j, -0.23 + 0.31j))
    )
    variation = GaussianVectorPotentialVariation(
        amplitude_au=(0.19, -0.13, 0.07),
        center_au=(0.23, -0.17, 0.11),
        exponent_au_inverse2=0.41,
        path_quadrature_order=24,
    )
    analytic = evaluate_nonlinear_weak_current_pairing(
        model,
        dynamic.one_electron,
        density,
        velocity,
        variation,
        one_electron_triple=triple,
    )
    errors: list[float] = []
    for step in (1.0e-3, 3.0e-4, 1.0e-4):
        values = [
            float(
                evaluate_static_nonlinear_wilson_grid_action(
                    model,
                    density,
                    velocity,
                    PerturbedVectorPotential(model.gauge, variation, sign * step),
                ).electronic_action_value
            )
            for sign in (1.0, -1.0)
        ]
        errors.append(
            abs((values[0] - values[1]) / (2.0 * step) - float(analytic.total_pairing))
        )
    assert errors[-1] < 3.0e-7
    assert min(errors) < 2.0e-8

    continuity = evaluate_nonlinear_weak_continuity(
        model,
        dynamic.one_electron,
        density,
        GaussianScalarGaugeVariation(
            amplitude=0.37,
            center_au=(0.11, -0.17, 0.23),
            exponent_au_inverse2=0.41,
        ),
    )
    np.testing.assert_allclose(
        continuity.finite_region_residual,
        0.0,
        atol=3.0e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        continuity.global_charge_residual,
        0.0,
        atol=2.0e-13,
        rtol=0.0,
    )
