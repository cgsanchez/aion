from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aion.config import BackendConfig, XCFamily
from aion.electromagnetism import UniformMagneticField, UniformMagneticSourceSample
from aion.electronic_structure import (
    AOGridPolicy,
    ExactWilsonDynamicSample,
    WilsonStationaryBranch,
    evaluate_exact_wilson_power,
    prepare_ao_quadrature,
    prepare_exact_wilson_dynamic_sample,
    prepare_exact_wilson_stationary_factory,
    prepare_pyscf_reference,
)
from aion.propagation import (
    NonlinearGaussMagnusPolicy,
    propagate_nonlinear_contravariant_density,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.integration


def test_dynamic_sample_and_nonlinear_power_identity() -> None:
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
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(2),
        block_size=1024,
    )
    factory = prepare_exact_wilson_stationary_factory(
        quadrature,
        auxiliary_basis="weigend",
        functional="lda,vwn",
    )
    source = UniformMagneticSourceSample(
        time_au=0.3,
        field=UniformMagneticField((0.0, 0.0, 0.017)),
        magnetic_field_dot_au=(0.002, -0.001, 0.003),
        electric_field_origin_au=(0.004, -0.003, 0.002),
        origin_au=(0.13, -0.19, 0.07),
    )
    sample = prepare_exact_wilson_dynamic_sample(
        factory,
        source,
        WilsonStationaryBranch.KOHN_SHAM_LDA,
    )
    metric = np.asarray(sample.one_electron.metric)
    coefficient = np.asarray(((0.71 + 0.19j,), (-0.23 + 0.41j,)))
    coefficient /= np.sqrt((coefficient.conj().T @ metric @ coefficient).real.item())
    density = 2.0 * coefficient @ coefficient.conj().T
    evaluation = sample.evaluate(density)
    np.testing.assert_allclose(
        evaluation.triple.hamiltonian_eom,
        evaluation.action.lower_mechanical_matrix,
        atol=0.0,
        rtol=0.0,
    )
    power = evaluate_exact_wilson_power(evaluation, density)
    assert abs(float(power.source_power_au)) > 1.0e-6
    np.testing.assert_allclose(
        power.matrix_mechanical_energy_rate_au,
        power.source_power_au,
        # The two routes deliberately combine stable analytic matrices with
        # an independent all-grid weak source response.  Grid refinement is
        # qualified by the NQ6 campaign; level 2 is only a bounded smoke test.
        atol=2.0e-6,
        rtol=0.0,
    )

    dynamic_cache: dict[float, ExactWilsonDynamicSample] = {}

    def dynamic(time_au: float) -> ExactWilsonDynamicSample:
        if time_au not in dynamic_cache:
            dynamic_cache[time_au] = prepare_exact_wilson_dynamic_sample(
                factory,
                UniformMagneticSourceSample(
                    time_au=time_au,
                    field=UniformMagneticField((0.0, 0.0, 0.017 + 0.001 * time_au)),
                    magnetic_field_dot_au=(0.0, 0.0, 0.001),
                    origin_au=(0.13, -0.19, 0.07),
                ),
                WilsonStationaryBranch.KOHN_SHAM_LDA,
            )
        return dynamic_cache[time_au]

    trajectory = propagate_nonlinear_contravariant_density(
        density,
        initial_time_au=0.0,
        interval_au=0.05,
        intervals=2,
        metric_provider=lambda time: dynamic(time).one_electron.metric,
        eom_provider=lambda time, value: dynamic(time).evaluate(value).triple,
        backend=quadrature.backend,
        policy=NonlinearGaussMagnusPolicy(tolerance=1.0e-10),
    )
    assert trajectory.metric_correction_applied is False
    assert trajectory.density_update == "coefficient_congruence"
    assert max(item.contravariant_hermiticity_residual for item in trajectory.diagnostics) < 2.0e-15
