from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aion.backends import NumPyBackend
from aion.config import (
    AffineElectromagneticSourceConfig,
    BackendConfig,
    ExactWilsonActionConfig,
    ReferenceLinkConfig,
    WilsonGridKind,
    WilsonGridPruning,
    WilsonMagneticGaugeKind,
    WilsonNumericsConfig,
    WilsonStationaryConfig,
    WilsonStationaryOutputConfig,
    WilsonStationaryPolicyConfig,
    XCFamily,
    ZeroSourceConfig,
)
from aion.electromagnetism import (
    AffineMagneticGauge,
    MagneticGaugeKind,
    UniformMagneticField,
    affine_gauge_difference_potential,
    build_affine_electromagnetic_source,
)
from aion.electronic_structure import (
    AOGridPolicy,
    AOQuadrature,
    StationarySCFPolicy,
    WilsonStationaryBranch,
    auxiliary_space_fingerprint,
    capture_wilson_stationary_state,
    load_wilson_stationary_state,
    prepare_ao_quadrature,
    prepare_exact_wilson_stationary_model,
    prepare_pyscf_reference,
    save_wilson_stationary_state,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.integration


def _quadrature() -> AOQuadrature:
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


@pytest.mark.parametrize(
    "branch",
    (WilsonStationaryBranch.HARTREE, WilsonStationaryBranch.KOHN_SHAM_LDA),
)
@pytest.mark.parametrize("field_strength", (0.0, 0.02))
def test_stationary_solver_converges_with_independent_residuals_and_invariants(
    branch: WilsonStationaryBranch,
    field_strength: float,
    tmp_path: Path,
) -> None:
    quadrature = _quadrature()
    gauge = AffineMagneticGauge(UniformMagneticField((0.0, 0.0, field_strength)))
    model = prepare_exact_wilson_stationary_model(
        quadrature,
        gauge,
        branch,
        auxiliary_basis="weigend",
    )
    opposite = model.for_branch(
        WilsonStationaryBranch.HARTREE
        if branch is WilsonStationaryBranch.KOHN_SHAM_LDA
        else WilsonStationaryBranch.KOHN_SHAM_LDA
    )
    assert opposite.hartree_action is model.hartree_action
    assert opposite.one_electron is model.one_electron
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
    assert state.commutator_residual < 2.0e-9
    assert state.metric_orthonormality_residual < 2.0e-12
    assert state.particle_number == pytest.approx(2.0, abs=2.0e-12)
    assert state.occupation_spectrum_residual < 2.0e-12
    assert state.closed_shell_density_polynomial_residual < 2.0e-12
    assert state.double_counting_residual_au < 2.0e-11
    assert state.metric_minimum_eigenvalue > 0.0

    if branch is WilsonStationaryBranch.KOHN_SHAM_LDA and field_strength == 0.0:
        source_config = AffineElectromagneticSourceConfig(
            electric=ZeroSourceConfig(),
            electric_field_origin_offset_au=(0.0, 0.0, 0.0),
            magnetic_field_reference_au=(0.0, 0.0, field_strength),
            magnetic_field_rate_au=(0.0, 0.0, 0.0),
            magnetic_reference_time_au=0.0,
            magnetic_gauge=WilsonMagneticGaugeKind.SYMMETRIC,
        )
        config = WilsonStationaryConfig(
            reference=ReferenceLinkConfig(
                quadrature.reference.fingerprint_sha256,
                Path("reference.h5"),
            ),
            action=ExactWilsonActionConfig(branch),
            numerics=WilsonNumericsConfig(
                grid_kind=WilsonGridKind.QUALIFICATION,
                grid_level=2,
                grid_pruning=WilsonGridPruning.NONE,
                block_size=1024,
                auxiliary_basis="weigend",
                ri_relative_threshold=0.0,
                ri_absolute_threshold=1.0e-7,
                ri_maximum_rank=None,
            ),
            source=source_config,
            source_time_au=0.0,
            stationary=WilsonStationaryPolicyConfig(
                maximum_iterations=80,
                density_tolerance=2.0e-9,
                orbital_tolerance=2.0e-9,
                energy_tolerance_au=2.0e-10,
                damping=0.5,
                diis_start_iteration=2,
                diis_space=8,
            ),
            backend=BackendConfig(),
            output=WilsonStationaryOutputConfig(tmp_path / "stationary.h5"),
        )
        source = build_affine_electromagnetic_source(
            source_config,
            gauge.origin_au,
        ).sample(0.0)
        portable = capture_wilson_stationary_state(
            config,
            state,
            source,
            grid_fingerprint_sha256=quadrature.grid.fingerprint_sha256,
            auxiliary_space_fingerprint_sha256=auxiliary_space_fingerprint(model.hartree_evaluator),
            backend=quadrature.backend,
        )
        save_wilson_stationary_state(portable, config.output.artifact_path)
        loaded = load_wilson_stationary_state(config.output.artifact_path)
        assert loaded.fingerprint_sha256 == portable.fingerprint_sha256
        np.testing.assert_array_equal(
            loaded.contravariant_density,
            np.asarray(state.coefficient_density),
        )


def test_gauge_related_stationary_solutions_have_equal_energy_and_density() -> None:
    quadrature = _quadrature()
    field = UniformMagneticField((0.0, 0.0, 0.02))
    symmetric = AffineMagneticGauge(field, origin_au=(0.17, -0.11, 0.23))
    landau = AffineMagneticGauge(
        field,
        kind=MagneticGaugeKind.LANDAU,
        origin_au=(-0.13, 0.19, -0.07),
        landau_axis=(1.0, 0.0, 0.0),
    )
    symmetric_model = prepare_exact_wilson_stationary_model(
        quadrature,
        symmetric,
        WilsonStationaryBranch.KOHN_SHAM_LDA,
        auxiliary_basis="weigend",
    )
    symmetric_state = symmetric_model.solve()

    reference = quadrature.reference
    anchors = reference.core_operators.nuclei.coordinates_au[reference.anchor_topology.ao_to_atom]
    chi = affine_gauge_difference_potential(
        landau,
        symmetric,
        anchors,
        NumPyBackend(),
    )
    coefficient_phase = np.exp(-1j * np.asarray(chi))
    transformed_coefficients = coefficient_phase[:, None] * np.asarray(symmetric_state.coefficients)
    landau_model = prepare_exact_wilson_stationary_model(
        quadrature,
        landau,
        WilsonStationaryBranch.KOHN_SHAM_LDA,
        auxiliary_basis="weigend",
    )
    landau_state = landau_model.solve(initial_coefficients=transformed_coefficients)

    transformed_density = (
        coefficient_phase[:, None]
        * np.asarray(symmetric_state.coefficient_density)
        * coefficient_phase[None, :].conj()
    )
    np.testing.assert_allclose(
        landau_state.coefficient_density,
        transformed_density,
        atol=2.0e-9,
        rtol=2.0e-9,
    )
    np.testing.assert_allclose(
        landau_state.action.energy_molecular_total_au,
        symmetric_state.action.energy_molecular_total_au,
        atol=2.0e-10,
        rtol=2.0e-10,
    )


def test_zero_field_stationary_states_recover_independent_references() -> None:
    quadrature = _quadrature()
    zero = AffineMagneticGauge(UniformMagneticField((0.0, 0.0, 0.0)))
    ks_model = prepare_exact_wilson_stationary_model(
        quadrature,
        zero,
        WilsonStationaryBranch.KOHN_SHAM_LDA,
        auxiliary_basis="weigend",
    )
    ks_state = ks_model.solve()
    assert (
        abs(
            float(ks_state.action.energy_molecular_total_au)
            - quadrature.reference.ground_state.energy_total_au
        )
        < 5.0e-8
    )
    from pyscf import scf

    class RestrictedHartree(scf.hf.RHF):
        def get_veff(
            self,
            mol: object | None = None,
            dm: object | None = None,
            dm_last: object = 0,
            vhf_last: object = 0,
            hermi: int = 1,
        ) -> object:
            del dm_last, vhf_last
            return self.get_j(mol, dm, hermi)

    workspace = quadrature.reference.create_workspace(BackendConfig())
    assert workspace.electronic_model is not None
    hartree_reference = RestrictedHartree(workspace.electronic_model.mol).density_fit(
        auxbasis="weigend"
    )
    hartree_reference.conv_tol = 1.0e-12
    hartree_reference.verbose = 0
    hartree_reference.kernel()
    assert hartree_reference.converged

    hartree_model = prepare_exact_wilson_stationary_model(
        quadrature,
        zero,
        WilsonStationaryBranch.HARTREE,
        auxiliary_basis="weigend",
    )
    hartree_state = hartree_model.solve()
    assert (
        abs(float(hartree_state.action.energy_molecular_total_au) - float(hartree_reference.e_tot))
        < 5.0e-8
    )


@pytest.mark.parametrize("imaginary", (False, True))
def test_stationary_energy_is_flat_in_unrestricted_retracted_directions(
    imaginary: bool,
) -> None:
    quadrature = _quadrature()
    gauge = AffineMagneticGauge(UniformMagneticField((0.0, 0.0, 0.02)))
    model = prepare_exact_wilson_stationary_model(
        quadrature,
        gauge,
        WilsonStationaryBranch.KOHN_SHAM_LDA,
        auxiliary_basis="weigend",
    )
    state = model.solve()
    coefficients = np.asarray(state.coefficients)
    occupations = np.asarray(state.occupations)
    overlap = np.asarray(state.action.overlap)
    raw = np.random.default_rng(3107).normal(size=coefficients.shape)
    direction = (1j if imaginary else 1.0) * raw / np.linalg.norm(raw)

    derivatives: list[float] = []
    for step in (3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4):
        energies: list[float] = []
        for sign in (1.0, -1.0):
            displaced = coefficients + sign * step * direction
            gram = displaced.conj().T @ overlap @ displaced
            values, vectors = np.linalg.eigh(gram)
            retracted = displaced @ ((vectors / np.sqrt(values)[None, :]) @ vectors.conj().T)
            density = np.einsum(
                "mi,i,ni->mn",
                retracted,
                occupations,
                retracted.conj(),
                optimize=True,
            )
            energies.append(float(model.evaluate(density).energy_molecular_total_au))
        derivatives.append(abs((energies[0] - energies[1]) / (2.0 * step)))

    assert derivatives[-1] < 2.0e-8
    assert derivatives[-1] <= derivatives[0]
