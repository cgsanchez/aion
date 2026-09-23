from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aion.config import BackendConfig, XCFamily
from aion.electromagnetism import (
    AffineMagneticGauge,
    UniformMagneticField,
    UniformMagneticSourceSample,
)
from aion.electronic_structure import (
    AOGridPolicy,
    AOQuadrature,
    ReducedWilsonLevel,
    StationarySCFPolicy,
    WilsonStationaryBranch,
    prepare_ao_quadrature,
    prepare_exact_wilson_dynamic_spatial_action,
    prepare_exact_wilson_stationary_factory,
    prepare_pyscf_reference,
    prepare_reduced_wilson_factory,
)
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
    return prepare_ao_quadrature(
        prepare_pyscf_reference(config),
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(2),
        block_size=1024,
    )


def _factories(quadrature: AOQuadrature) -> tuple[object, object]:
    exact = prepare_exact_wilson_stationary_factory(
        quadrature,
        auxiliary_basis="weigend",
        functional="lda,vwn",
    )
    return exact, prepare_reduced_wilson_factory(exact)


@pytest.mark.parametrize(
    "branch",
    (WilsonStationaryBranch.HARTREE, WilsonStationaryBranch.KOHN_SHAM_LDA),
)
def test_all_reduced_actions_recover_exact_wilson_at_zero_field(
    quadrature: AOQuadrature,
    branch: WilsonStationaryBranch,
) -> None:
    exact_factory, reduced_factory = _factories(quadrature)
    gauge = AffineMagneticGauge(UniformMagneticField((0.0, 0.0, 0.0)))
    density = quadrature.reference.ground_state.density
    exact = exact_factory.model(gauge, branch).evaluate(density)

    for level in ReducedWilsonLevel:
        reduced = reduced_factory.model(gauge, level, branch).evaluate(density)
        np.testing.assert_allclose(reduced.overlap, exact.overlap, atol=2.0e-13, rtol=0.0)
        np.testing.assert_allclose(
            reduced.lower_mechanical_matrix,
            exact.lower_mechanical_matrix,
            atol=2.0e-10,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            reduced.energy_molecular_total_au,
            exact.energy_molecular_total_au,
            atol=2.0e-10,
            rtol=0.0,
        )
        assert reduced.closure.hartree.pair_counting_residual < 2.0e-12
        assert abs(float(reduced.closure.density.electron_count_first)) < 2.0e-13


def test_e1_connection_is_exact_for_zero_b_uniform_electric_field(
    quadrature: AOQuadrature,
) -> None:
    exact_factory, reduced_factory = _factories(quadrature)
    source = UniformMagneticSourceSample(
        time_au=0.37,
        field=UniformMagneticField((0.0, 0.0, 0.0)),
        electric_field_origin_au=(0.003, -0.002, 0.004),
        origin_au=(0.17, -0.11, 0.23),
    )
    exact = prepare_exact_wilson_dynamic_spatial_action(
        exact_factory,
        source.gauge,
    ).temporal_sample(source)
    reduced = reduced_factory.spatial_action(
        source.gauge,
        ReducedWilsonLevel.E1,
    ).sample(source, WilsonStationaryBranch.HARTREE)

    np.testing.assert_allclose(
        reduced.connection.connection,
        exact.connection.connection,
        atol=3.0e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        reduced.connection.metric_dot,
        exact.connection.metric_dot,
        atol=2.0e-13,
        rtol=0.0,
    )
    assert reduced.connection.metric_compatibility_residual < 2.0e-13


@pytest.mark.parametrize(
    "level",
    (ReducedWilsonLevel.STRICT_C1, ReducedWilsonLevel.DENSITY_RESUMMED_C1),
)
@pytest.mark.parametrize(
    "branch",
    (WilsonStationaryBranch.HARTREE, WilsonStationaryBranch.KOHN_SHAM_LDA),
)
def test_reduced_lower_matrix_is_unrestricted_action_derivative(
    quadrature: AOQuadrature,
    level: ReducedWilsonLevel,
    branch: WilsonStationaryBranch,
) -> None:
    _, reduced_factory = _factories(quadrature)
    gauge = AffineMagneticGauge(UniformMagneticField((0.004, -0.003, 0.002)))
    model = reduced_factory.model(gauge, level, branch)
    nao = quadrature.reference.core_operators.nao
    density = np.asarray(quadrature.reference.ground_state.density) + 0.2 * np.eye(nao)
    random = np.random.default_rng(8173).normal(size=(nao, nao)) + 1j * np.random.default_rng(
        8174
    ).normal(size=(nao, nao))
    direction = random + random.conj().T
    direction /= np.linalg.norm(direction)
    step = 2.0e-5

    plus = float(model.evaluate(density + step * direction).energy_electronic_au)
    minus = float(model.evaluate(density - step * direction).energy_electronic_au)
    numerical = (plus - minus) / (2.0 * step)
    analytic = np.einsum(
        "ij,ji->",
        np.asarray(model.evaluate(density).lower_mechanical_matrix),
        direction,
        optimize=True,
    ).real
    assert numerical == pytest.approx(float(analytic), abs=3.0e-8, rel=2.0e-8)


def test_strict_and_density_resummed_c1_separate_at_second_order(
    quadrature: AOQuadrature,
) -> None:
    _, reduced_factory = _factories(quadrature)
    nao = quadrature.reference.core_operators.nao
    generator = np.eye(nao, dtype=np.complex128)
    generator[0, -1] = 0.2j
    density = generator @ generator.conj().T
    differences: list[float] = []
    for scale in (0.004, 0.008):
        gauge = AffineMagneticGauge(UniformMagneticField((scale, -0.75 * scale, 0.5 * scale)))
        strict = reduced_factory.model(
            gauge,
            ReducedWilsonLevel.STRICT_C1,
            WilsonStationaryBranch.HARTREE,
        ).evaluate(density)
        resummed = reduced_factory.model(
            gauge,
            ReducedWilsonLevel.DENSITY_RESUMMED_C1,
            WilsonStationaryBranch.HARTREE,
        ).evaluate(density)
        differences.append(abs(float(resummed.energy_hartree_au - strict.energy_hartree_au)))

    assert differences[0] > 1.0e-12
    assert differences[1] / differences[0] == pytest.approx(4.0, rel=2.0e-2)


def test_reduced_factory_reuses_common_spatial_data(
    quadrature: AOQuadrature,
) -> None:
    _, reduced_factory = _factories(quadrature)
    gauge = AffineMagneticGauge(UniformMagneticField((0.004, -0.003, 0.002)))
    p0 = reduced_factory.spatial_action(gauge, ReducedWilsonLevel.P0)
    e1 = reduced_factory.spatial_action(gauge, ReducedWilsonLevel.E1)
    strict = reduced_factory.spatial_action(gauge, ReducedWilsonLevel.STRICT_C1)
    resummed = reduced_factory.spatial_action(
        gauge,
        ReducedWilsonLevel.DENSITY_RESUMMED_C1,
    )

    assert p0.one_electron is e1.one_electron
    assert strict.one_electron is resummed.one_electron
    assert p0.closure is e1.closure is strict.closure is resummed.closure


def test_c1_connection_is_metric_compatible_for_time_dependent_b(
    quadrature: AOQuadrature,
) -> None:
    _, reduced_factory = _factories(quadrature)
    source = UniformMagneticSourceSample(
        time_au=0.31,
        field=UniformMagneticField((0.006, -0.004, 0.003)),
        magnetic_field_dot_au=(0.002, 0.001, -0.003),
        electric_field_origin_au=(0.001, -0.002, 0.004),
        origin_au=(0.13, -0.19, 0.07),
    )
    model = reduced_factory.spatial_action(
        source.gauge,
        ReducedWilsonLevel.STRICT_C1,
    ).sample(source, WilsonStationaryBranch.KOHN_SHAM_LDA)
    assert model.connection.metric_compatibility_residual < 5.0e-13


@pytest.mark.parametrize(
    "level",
    (ReducedWilsonLevel.STRICT_C1, ReducedWilsonLevel.DENSITY_RESUMMED_C1),
)
def test_reduced_actions_use_common_stationary_solver(
    quadrature: AOQuadrature,
    level: ReducedWilsonLevel,
) -> None:
    _, reduced_factory = _factories(quadrature)
    model = reduced_factory.model(
        AffineMagneticGauge(UniformMagneticField((0.003, -0.002, 0.004))),
        level,
        WilsonStationaryBranch.KOHN_SHAM_LDA,
    )
    state = model.solve(
        policy=StationarySCFPolicy(
            maximum_iterations=80,
            density_tolerance=2.0e-9,
            orbital_tolerance=2.0e-9,
            energy_tolerance_au=2.0e-10,
        )
    )

    assert state.orbital_residual < 2.0e-9
    assert state.density_fixed_point_residual < 2.0e-9
    assert state.metric_orthonormality_residual < 2.0e-12
    assert state.particle_number_residual < 2.0e-12
    assert state.double_counting_residual_au < 3.0e-10


@pytest.mark.parametrize("level", tuple(ReducedWilsonLevel))
@pytest.mark.parametrize(
    "branch",
    (WilsonStationaryBranch.HARTREE, WilsonStationaryBranch.KOHN_SHAM_LDA),
)
def test_reduced_fixed_history_source_response_matches_finite_difference(
    quadrature: AOQuadrature,
    level: ReducedWilsonLevel,
    branch: WilsonStationaryBranch,
) -> None:
    _, reduced_factory = _factories(quadrature)
    field = np.asarray((0.006, -0.004, 0.003))
    field_rate = np.asarray((0.002, 0.001, -0.003))
    source = UniformMagneticSourceSample(
        time_au=0.31,
        field=UniformMagneticField(tuple(field)),
        magnetic_field_dot_au=tuple(field_rate),
        origin_au=(0.13, -0.19, 0.07),
    )
    model = reduced_factory.spatial_action(source.gauge, level).sample(source, branch)
    nao = quadrature.reference.core_operators.nao
    density = np.asarray(quadrature.reference.ground_state.density) + 0.2 * np.eye(nao)
    analytic = float(model.source_response(density).mechanical_energy_rate_au)
    step = 2.0e-4
    energies: list[float] = []
    for sign in (1.0, -1.0):
        displaced_gauge = AffineMagneticGauge(
            UniformMagneticField(tuple(field + sign * step * field_rate)),
            origin_au=source.origin_au,
        )
        displaced = reduced_factory.model(displaced_gauge, level, branch)
        energies.append(float(displaced.evaluate(density).energy_electronic_au))
    numerical = (energies[0] - energies[1]) / (2.0 * step)
    assert numerical == pytest.approx(analytic, abs=2.0e-8, rel=3.0e-7)


@pytest.mark.parametrize(
    "level",
    (ReducedWilsonLevel.E1, ReducedWilsonLevel.STRICT_C1),
)
def test_reduced_power_is_total_action_derivative_on_eom_shell(
    quadrature: AOQuadrature,
    level: ReducedWilsonLevel,
) -> None:
    _, reduced_factory = _factories(quadrature)
    field = np.asarray((0.004, -0.003, 0.002))
    field_rate = np.asarray((0.001, 0.002, -0.0015))
    source = UniformMagneticSourceSample(
        time_au=0.23,
        field=UniformMagneticField(tuple(field)),
        magnetic_field_dot_au=tuple(field_rate),
        electric_field_origin_au=(0.003, -0.002, 0.004),
        origin_au=(0.13, -0.19, 0.07),
    )
    branch = WilsonStationaryBranch.KOHN_SHAM_LDA
    model = reduced_factory.spatial_action(source.gauge, level).sample(source, branch)
    nao = quadrature.reference.core_operators.nao
    density = np.asarray(quadrature.reference.ground_state.density) + 0.2 * np.eye(nao)
    power = model.power(density)
    density_rate = np.asarray(power.velocity_density) + np.asarray(power.velocity_density).conj().T
    step = 2.0e-4
    energies: list[float] = []
    for sign in (1.0, -1.0):
        displaced_gauge = AffineMagneticGauge(
            UniformMagneticField(tuple(field + sign * step * field_rate)),
            origin_au=source.origin_au,
        )
        displaced = reduced_factory.model(displaced_gauge, level, branch)
        energies.append(
            float(displaced.evaluate(density + sign * step * density_rate).energy_electronic_au)
        )
    numerical = (energies[0] - energies[1]) / (2.0 * step)
    assert numerical == pytest.approx(
        float(power.source_power_au),
        abs=3.0e-8,
        rel=3.0e-7,
    )
