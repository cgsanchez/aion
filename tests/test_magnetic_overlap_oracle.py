from __future__ import annotations

import numpy as np
import pytest

from aion.config import (
    AtomConfig,
    BackendConfig,
    ElectronicStructureConfig,
    MoleculeConfig,
    ReferenceConfig,
    XCFamily,
)
from aion.electromagnetism import (
    AffineMagneticGauge,
    MagneticGaugeKind,
    UniformMagneticField,
)
from aion.electronic_structure import (
    AOGridPolicy,
    analytic_uniform_magnetic_overlap,
    evaluate_magnetic_one_electron_matrices,
    evaluate_magnetic_spatial_connections,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
)
from test_reference_integration import molecular_config

pytestmark = [pytest.mark.integration, pytest.mark.qualification]


def test_fourier_overlap_is_grid_independent_gauge_independent_and_exact_at_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    from pyscf.dft.rks import RKS

    def forbidden_scf(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("analytic overlap oracle must not rerun SCF")

    monkeypatch.setattr(RKS, "kernel", forbidden_scf)
    field = UniformMagneticField((0.09, -0.07, 0.11))
    symmetric = AffineMagneticGauge(field, origin_au=(0.17, -0.31, 0.23))
    landau = AffineMagneticGauge(
        field,
        kind=MagneticGaugeKind.LANDAU,
        origin_au=(-0.21, 0.37, -0.16),
        landau_axis=(-0.07, -0.09, 0.0),
    )
    analytic = analytic_uniform_magnetic_overlap(reference, symmetric)
    analytic_landau = analytic_uniform_magnetic_overlap(reference, landau)
    negative = analytic_uniform_magnetic_overlap(
        reference,
        AffineMagneticGauge(UniformMagneticField(tuple(-np.asarray(field.magnetic_field_au)))),
    )
    zero = analytic_uniform_magnetic_overlap(
        reference,
        AffineMagneticGauge(UniformMagneticField((0.0, 0.0, 0.0))),
    )
    np.testing.assert_allclose(analytic.barred, analytic_landau.barred, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        analytic.atom_pair_wavevectors_au,
        -analytic.atom_pair_wavevectors_au.swapaxes(0, 1),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(analytic.barred, analytic.barred.conj().T, atol=2.0e-15)
    np.testing.assert_allclose(analytic.lower, analytic.lower.conj().T, atol=2.0e-15)
    np.testing.assert_allclose(negative.barred, analytic.barred.conj(), atol=2.0e-15)
    np.testing.assert_allclose(
        zero.barred,
        reference.core_operators.overlap,
        atol=2.0e-14,
        rtol=2.0e-14,
    )

    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(4),
        block_size=2048,
    )
    grid_result = evaluate_magnetic_one_electron_matrices(
        quadrature,
        (field,),
        direct_gauges=(symmetric,),
    )[0]
    np.testing.assert_allclose(grid_result.overlap.exact, analytic.barred, atol=2.0e-12)
    np.testing.assert_allclose(grid_result.lower_exact.overlap, analytic.lower, atol=2.0e-12)


def test_three_center_exact_gram_remains_positive_when_p0_and_b1_metrics_fail() -> None:
    side = 1.65
    geometry = np.array(
        [
            (-0.5 * side, -np.sqrt(3.0) * side / 6.0, 0.0),
            (0.5 * side, -np.sqrt(3.0) * side / 6.0, 0.0),
            (0.0, np.sqrt(3.0) * side / 3.0, 0.0),
        ]
    )
    area_vector = 0.5 * np.sum(
        np.cross(geometry, np.roll(geometry, -1, axis=0)),
        axis=0,
    )
    area = float(np.linalg.norm(area_vector))
    reference = prepare_pyscf_reference(
        ReferenceConfig(
            molecule=MoleculeConfig(
                atoms=tuple(AtomConfig("H", tuple(position)) for position in geometry),
                charge=1,
            ),
            electronic_structure=ElectronicStructureConfig(
                basis="sto-3g",
                functional="pbe",
                xc_family=XCFamily.GGA,
                grid_level=1,
            ),
            backend=BackendConfig(),
        )
    )
    field = UniformMagneticField((0.0, 0.0, 2.0 / area))
    analytic = analytic_uniform_magnetic_overlap(reference, AffineMagneticGauge(field))
    p0_metric = analytic.endpoint_link * reference.core_operators.overlap
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(2),
        block_size=2048,
    )
    hierarchy = evaluate_magnetic_one_electron_matrices(
        quadrature,
        (field,),
        include_direct_oracle=False,
    )[0]
    b1_metric = hierarchy.endpoint_link * hierarchy.overlap.b1
    exact_grid_metric = hierarchy.lower_exact.overlap

    assert np.linalg.eigvalsh(analytic.lower).min() > 0.4
    assert np.linalg.eigvalsh(p0_metric).min() < -0.06
    assert np.linalg.eigvalsh(b1_metric).min() < -0.06
    np.testing.assert_allclose(exact_grid_metric, analytic.lower, atol=2.0e-6)


def test_consecutive_unpruned_grids_resolve_every_core_matrix_family() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    field = UniformMagneticField((0.09, -0.07, 0.11))
    levels = []
    for level in (2, 3, 4):
        quadrature = prepare_ao_quadrature(
            reference,
            BackendConfig(),
            grid_policy=AOGridPolicy.qualification(level),
            block_size=2048,
        )
        core = evaluate_magnetic_one_electron_matrices(
            quadrature,
            (field,),
            include_direct_oracle=False,
        )[0]
        spatial = evaluate_magnetic_spatial_connections(
            quadrature,
            (field,),
            include_direct_oracle=False,
        )[0]
        levels.append(
            (
                core.overlap.exact,
                core.kinetic.exact,
                core.nuclear_attraction.exact,
                spatial.spatial_connection.exact,
            )
        )
    level_23 = np.asarray(
        [np.linalg.norm(right - left) for left, right in zip(levels[0], levels[1], strict=True)]
    )
    level_34 = np.asarray(
        [np.linalg.norm(right - left) for left, right in zip(levels[1], levels[2], strict=True)]
    )
    assert np.all(level_34 < level_23)
    assert np.all(level_34 < np.array((1.0e-11, 1.0e-10, 1.0e-11, 1.0e-9)))


def test_physical_30_tesla_field_is_separate_from_stress_field_qualification() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    direction = np.array((0.37, -0.51, 0.69))
    direction /= np.linalg.norm(direction)
    field = UniformMagneticField.from_tesla(tuple(30.0 * direction))
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(4),
        block_size=2048,
    )
    result = evaluate_magnetic_one_electron_matrices(
        quadrature,
        (field,),
        include_direct_oracle=False,
    )[0]
    for hierarchy in (result.overlap, result.kinetic, result.nuclear_attraction):
        assert np.linalg.norm(hierarchy.exact - hierarchy.b1) > 1.0e-10
        assert np.linalg.norm(hierarchy.exact - hierarchy.b2) < 1.0e-15
