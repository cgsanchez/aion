#!/usr/bin/env python3
"""Execute the Chapter 13 NQ4 nonlinear stationary-state campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np

from aion.backends import NumPyBackend
from aion.config import (
    AtomConfig,
    BackendConfig,
    ElectronicStructureConfig,
    MetadataConfig,
    MoleculeConfig,
    ReferenceConfig,
    ReferenceOutputConfig,
    XCFamily,
    dumps_config,
)
from aion.electromagnetism import (
    AffineMagneticGauge,
    MagneticGaugeKind,
    UniformMagneticField,
    affine_gauge_difference_potential,
)
from aion.electronic_structure import (
    AOGridPolicy,
    DependencyVersions,
    ExactWilsonStationaryFactory,
    ExactWilsonStationaryModel,
    ExactWilsonStationaryState,
    StationarySCFPolicy,
    WilsonStationaryBranch,
    evaluate_exact_uniform_magnetic_wilson_density,
    prepare_ao_quadrature,
    prepare_exact_wilson_stationary_factory,
    prepare_pyscf_reference,
)

_AUXILIARY_BASIS = "weigend"
_FUNCTIONAL = "lda,vwn"
_H3_FIELDS = (0.0, 0.001, 0.003, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
_GAUGE_FIELDS = (0.03, 0.06)
_GRID_FIELDS = (0.0, 0.03, 0.06)
_GRID_LEVELS = (3, 4, 5)
_STATIONARITY_STEPS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5)
_SYMMETRIC_ORIGIN = (0.17, -0.31, 0.23)
_LANDAU_ORIGIN = (-0.21, 0.37, -0.16)
_MAIN_POLICY = StationarySCFPolicy(
    maximum_iterations=160,
    density_tolerance=2.0e-12,
    orbital_tolerance=2.0e-12,
    energy_tolerance_au=2.0e-13,
)
_POLICIES = {
    "loose": StationarySCFPolicy(
        maximum_iterations=100,
        density_tolerance=2.0e-7,
        orbital_tolerance=2.0e-7,
        energy_tolerance_au=2.0e-8,
    ),
    "standard": StationarySCFPolicy(
        maximum_iterations=120,
        density_tolerance=2.0e-9,
        orbital_tolerance=2.0e-9,
        energy_tolerance_au=2.0e-10,
    ),
    "tight": _MAIN_POLICY,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return cast(str | bytes, completed.stdout)


def _relative(value: object, reference: object) -> float:
    candidate = np.asarray(value)
    baseline = np.asarray(reference)
    return float(
        np.linalg.norm(candidate - baseline)
        / max(1.0, float(np.linalg.norm(baseline)))
    )


def _configuration(system: str, artifact: Path, timestamp: str) -> ReferenceConfig:
    if system == "h2":
        atoms = (
            AtomConfig("H", (0.0, 0.0, -0.7)),
            AtomConfig("H", (0.0, 0.0, 0.7)),
        )
        charge = 0
        basis = "sto-3g"
        grid_level = 3
    elif system == "h3plus":
        atoms = (
            AtomConfig("H", (-0.7, -0.404145188432738, 0.0)),
            AtomConfig("H", (0.7, -0.404145188432738, 0.0)),
            AtomConfig("H", (0.0, 0.808290376865476, 0.0)),
        )
        charge = 1
        basis = "cc-pvdz"
        grid_level = 4
    else:
        raise ValueError(system)
    return ReferenceConfig(
        molecule=MoleculeConfig(atoms=atoms, charge=charge, spin=0),
        electronic_structure=ElectronicStructureConfig(
            basis=basis,
            functional=_FUNCTIONAL,
            xc_family=XCFamily.LDA,
            grid_level=grid_level,
            density_fitting=True,
            auxiliary_basis=_AUXILIARY_BASIS,
            scf_energy_tolerance_au=1.0e-12,
            scf_max_iterations=120,
        ),
        backend=BackendConfig(),
        output=ReferenceOutputConfig(artifact),
        metadata=MetadataConfig(
            label=f"chapter13-nq4-{system}-stationary",
            timestamp_utc=timestamp,
            host=platform.node(),
        ),
    )


def _density(coefficients: object, occupations: object) -> np.ndarray:
    return np.asarray(
        np.einsum(
            "mi,i,ni->mn",
            np.asarray(coefficients),
            np.asarray(occupations),
            np.asarray(coefficients).conj(),
            optimize=True,
        ),
        dtype=np.complex128,
    )


def _retract(
    coefficients: np.ndarray,
    direction: np.ndarray,
    overlap: np.ndarray,
    scale: float,
) -> np.ndarray:
    displaced = coefficients + scale * direction
    gram = displaced.conj().T @ overlap @ displaced
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.conj().T))
    inverse_square_root = (vectors / np.sqrt(values)[None, :]) @ vectors.conj().T
    return np.asarray(displaced @ inverse_square_root, dtype=np.complex128)


def _independent_hartree_reference(reference: Any) -> dict[str, Any]:
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

    workspace = reference.create_workspace(BackendConfig())
    assert workspace.electronic_model is not None
    mean_field = RestrictedHartree(workspace.electronic_model.mol).density_fit(
        auxbasis=_AUXILIARY_BASIS
    )
    mean_field.conv_tol = 1.0e-12
    mean_field.max_cycle = 120
    mean_field.verbose = 0
    mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("independent PySCF restricted-Hartree reference did not converge")
    return {
        "energy_molecular_total_au": float(mean_field.e_tot),
        "coefficient_density": np.asarray(mean_field.make_rdm1()),
        "cycles": int(getattr(mean_field, "cycles", -1)),
    }


def _state_record(
    system: str,
    field_strength: float,
    state: ExactWilsonStationaryState,
    elapsed_seconds: float,
    classification: str,
) -> dict[str, Any]:
    action = state.action
    xc = action.exchange_correlation
    return {
        "system": system,
        "branch": action.branch.value,
        "field_au": field_strength,
        "classification": classification,
        "energy_molecular_total_au": float(action.energy_molecular_total_au),
        "energy_electronic_au": float(action.energy_electronic_au),
        "energy_kinetic_au": float(action.energy_kinetic_au),
        "energy_electron_nuclear_au": float(action.energy_electron_nuclear_au),
        "energy_hartree_au": float(action.energy_hartree_au),
        "energy_exchange_correlation_au": float(
            action.energy_exchange_correlation_au
        ),
        "energy_nuclear_repulsion_au": float(action.energy_nuclear_repulsion_au),
        "iterations": len(state.iterations),
        "elapsed_seconds": elapsed_seconds,
        "orbital_residual": state.orbital_residual,
        "density_fixed_point_residual": state.density_fixed_point_residual,
        "commutator_residual": state.commutator_residual,
        "metric_orthonormality_residual": state.metric_orthonormality_residual,
        "particle_number": state.particle_number,
        "particle_number_residual": state.particle_number_residual,
        "occupation_spectrum": state.occupation_spectrum.tolist(),
        "occupation_spectrum_imaginary_max_abs": (
            state.occupation_spectrum_imaginary_max_abs
        ),
        "occupation_spectrum_residual": state.occupation_spectrum_residual,
        "closed_shell_density_polynomial_residual": (
            state.closed_shell_density_polynomial_residual
        ),
        "orbital_frequency_occupation_commutator_residual": (
            state.orbital_frequency_occupation_commutator_residual
        ),
        "double_counting_residual_au": state.double_counting_residual_au,
        "metric_minimum_eigenvalue": state.metric_minimum_eigenvalue,
        "metric_condition_number": state.metric_condition_number,
        "hartree_pair_counting_residual": action.hartree.pair_counting_residual,
        "hartree_stationary_energy_residual": action.hartree.stationary_energy_residual,
        "hartree_lower_hermiticity_residual": action.hartree.lower_hermiticity_residual,
        "xc_lower_hermiticity_residual": (
            0.0 if xc is None else xc.lower_hermiticity_residual
        ),
    }


def _one_electron_diagnostics(
    model: ExactWilsonStationaryModel,
    state: ExactWilsonStationaryState,
) -> dict[str, float]:
    one = model.one_electron
    endpoint = np.asarray(one.endpoint_link)
    exact = one.lower_exact
    p0_overlap = endpoint * np.asarray(one.overlap.zero)
    p0_kinetic = endpoint * np.asarray(one.kinetic.zero)
    p0_nuclear = endpoint * np.asarray(one.nuclear_attraction.zero)
    p0_mechanical = p0_kinetic + p0_nuclear
    exact_mechanical = np.asarray(exact.mechanical)
    density = np.asarray(state.coefficient_density)
    energy_correction = np.einsum(
        "ij,ji->",
        exact_mechanical - p0_mechanical,
        density,
        optimize=True,
    ).real
    return {
        "phase_spread_rms_max": float(np.max(np.asarray(one.phase_spread.rms))),
        "exact_minus_p0_overlap_relative_frobenius": _relative(
            exact.overlap, p0_overlap
        ),
        "exact_minus_p0_kinetic_relative_frobenius": _relative(
            exact.kinetic, p0_kinetic
        ),
        "exact_minus_p0_nuclear_relative_frobenius": _relative(
            exact.nuclear_attraction, p0_nuclear
        ),
        "exact_minus_p0_mechanical_relative_frobenius": _relative(
            exact_mechanical, p0_mechanical
        ),
        "exact_minus_p0_one_electron_energy_au": float(energy_correction),
    }


def _solve(
    model: ExactWilsonStationaryModel,
    *,
    policy: StationarySCFPolicy = _MAIN_POLICY,
    initial_coefficients: object | None = None,
) -> tuple[ExactWilsonStationaryState, float]:
    start = perf_counter()
    state = model.solve(policy=policy, initial_coefficients=initial_coefficients)
    return state, perf_counter() - start


def _field_free_recovery(
    system: str,
    reference: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.reference(),
        block_size=2048,
    )
    factory = prepare_exact_wilson_stationary_factory(
        quadrature,
        auxiliary_basis=_AUXILIARY_BASIS,
        functional=_FUNCTIONAL,
    )
    zero = AffineMagneticGauge(UniformMagneticField((0.0, 0.0, 0.0)))
    ks_state, ks_elapsed = _solve(
        factory.model(zero, WilsonStationaryBranch.KOHN_SHAM_LDA)
    )
    hartree_state, hartree_elapsed = _solve(
        factory.model(zero, WilsonStationaryBranch.HARTREE)
    )
    hartree_oracle = _independent_hartree_reference(reference)
    reference_density = np.asarray(reference.ground_state.density)
    record = {
        "grid_kind": quadrature.grid.kind.value,
        "grid_level": quadrature.grid.level,
        "grid_points": quadrature.grid.npoints,
        "kohn_sham": {
            "state": _state_record(
                system, 0.0, ks_state, ks_elapsed, "field_free_stationary"
            ),
            "pyscf_energy_molecular_total_au": reference.ground_state.energy_total_au,
            "energy_absolute_residual_au": abs(
                float(ks_state.action.energy_molecular_total_au)
                - reference.ground_state.energy_total_au
            ),
            "coefficient_density_relative_residual": _relative(
                ks_state.coefficient_density, reference_density
            ),
        },
        "hartree": {
            "state": _state_record(
                system, 0.0, hartree_state, hartree_elapsed, "field_free_stationary"
            ),
            "pyscf_energy_molecular_total_au": hartree_oracle[
                "energy_molecular_total_au"
            ],
            "energy_absolute_residual_au": abs(
                float(hartree_state.action.energy_molecular_total_au)
                - float(hartree_oracle["energy_molecular_total_au"])
            ),
            "coefficient_density_relative_residual": _relative(
                hartree_state.coefficient_density,
                hartree_oracle["coefficient_density"],
            ),
            "pyscf_cycles": hartree_oracle["cycles"],
        },
    }
    arrays = {
        f"{system}_field_free_ks_density": np.asarray(ks_state.coefficient_density),
        f"{system}_field_free_hartree_density": np.asarray(
            hartree_state.coefficient_density
        ),
    }
    return record, arrays


def _main_h3_scan(
    reference: Any,
) -> tuple[
    dict[str, Any],
    ExactWilsonStationaryFactory,
    dict[tuple[float, WilsonStationaryBranch], ExactWilsonStationaryState],
    dict[str, np.ndarray],
]:
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(4),
        block_size=2048,
    )
    factory = prepare_exact_wilson_stationary_factory(
        quadrature,
        auxiliary_basis=_AUXILIARY_BASIS,
        functional=_FUNCTIONAL,
    )
    states: dict[tuple[float, WilsonStationaryBranch], ExactWilsonStationaryState] = {}
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    previous: dict[WilsonStationaryBranch, np.ndarray | None] = {
        WilsonStationaryBranch.HARTREE: None,
        WilsonStationaryBranch.KOHN_SHAM_LDA: None,
    }
    for field_strength in _H3_FIELDS:
        gauge = AffineMagneticGauge(
            UniformMagneticField((0.0, 0.0, field_strength)),
            origin_au=_SYMMETRIC_ORIGIN,
        )
        ks_model = factory.model(gauge, WilsonStationaryBranch.KOHN_SHAM_LDA)
        models = {
            WilsonStationaryBranch.HARTREE: ks_model.for_branch(
                WilsonStationaryBranch.HARTREE
            ),
            WilsonStationaryBranch.KOHN_SHAM_LDA: ks_model,
        }
        for branch in (
            WilsonStationaryBranch.HARTREE,
            WilsonStationaryBranch.KOHN_SHAM_LDA,
        ):
            state, elapsed = _solve(
                models[branch],
                initial_coefficients=previous[branch],
            )
            previous[branch] = np.asarray(state.coefficients)
            states[(field_strength, branch)] = state
            density_result = evaluate_exact_uniform_magnetic_wilson_density(
                quadrature,
                state.coefficient_density,
                gauge,
            )
            row = _state_record(
                "h3plus",
                field_strength,
                state,
                elapsed,
                "finite_field_stationary" if field_strength else "field_free_stationary",
            )
            row.update(_one_electron_diagnostics(models[branch], state))
            row.update(
                {
                    "density_real_minimum": (
                        density_result.density_direct_real_minimum
                    ),
                    "density_imaginary_max_abs": (
                        density_result.density_direct_imaginary_max_abs
                    ),
                    "density_direct_factorized_residual": (
                        density_result.density_direct_factorized_residual
                    ),
                    "density_particle_number_residual": abs(
                        float(
                            np.asarray(
                                density_result.particle_number_stable_metric
                            ).real
                        )
                        - state.particle_number
                    ),
                }
            )
            rows.append(row)
            tag = f"h3plus_b{field_strength:.3f}_{branch.value}".replace(".", "p")
            arrays[f"{tag}_coefficients"] = np.asarray(state.coefficients)
            arrays[f"{tag}_density"] = np.asarray(state.coefficient_density)
            arrays[f"{tag}_overlap"] = np.asarray(state.action.overlap)
            arrays[f"{tag}_lower"] = np.asarray(state.action.lower_mechanical_matrix)
    return (
        {
            "grid_level": 4,
            "grid_pruning": quadrature.grid.pruning,
            "grid_points": quadrature.grid.npoints,
            "grid_fingerprint_sha256": quadrature.grid.fingerprint_sha256,
            "rows": rows,
        },
        factory,
        states,
        arrays,
    )


def _gauge_checks(
    reference: Any,
    factory: ExactWilsonStationaryFactory,
    states: dict[tuple[float, WilsonStationaryBranch], ExactWilsonStationaryState],
) -> list[dict[str, Any]]:
    anchors = reference.core_operators.nuclei.coordinates_au[
        reference.anchor_topology.ao_to_atom
    ]
    rows: list[dict[str, Any]] = []
    for field_strength in _GAUGE_FIELDS:
        field = UniformMagneticField((0.0, 0.0, field_strength))
        symmetric = AffineMagneticGauge(field, origin_au=_SYMMETRIC_ORIGIN)
        landau = AffineMagneticGauge(
            field,
            kind=MagneticGaugeKind.LANDAU,
            origin_au=_LANDAU_ORIGIN,
            landau_axis=(1.0, 0.0, 0.0),
        )
        chi = affine_gauge_difference_potential(
            landau,
            symmetric,
            anchors,
            NumPyBackend(),
        )
        phase = np.exp(-1j * np.asarray(chi))
        for branch in (
            WilsonStationaryBranch.HARTREE,
            WilsonStationaryBranch.KOHN_SHAM_LDA,
        ):
            baseline = states[(field_strength, branch)]
            initial = phase[:, None] * np.asarray(baseline.coefficients)
            model = factory.model(landau, branch)
            candidate, elapsed = _solve(model, initial_coefficients=initial)
            expected_density = (
                phase[:, None]
                * np.asarray(baseline.coefficient_density)
                * phase[None, :].conj()
            )
            expected_lower = (
                phase[:, None]
                * np.asarray(baseline.action.lower_mechanical_matrix)
                * phase[None, :].conj()
            )
            baseline_density = evaluate_exact_uniform_magnetic_wilson_density(
                factory.quadrature,
                baseline.coefficient_density,
                symmetric,
            )
            candidate_density = evaluate_exact_uniform_magnetic_wilson_density(
                factory.quadrature,
                candidate.coefficient_density,
                landau,
            )
            rows.append(
                {
                    "field_au": field_strength,
                    "branch": branch.value,
                    "elapsed_seconds": elapsed,
                    "energy_absolute_residual_au": abs(
                        float(candidate.action.energy_molecular_total_au)
                        - float(baseline.action.energy_molecular_total_au)
                    ),
                    "coefficient_density_covariance_relative_residual": _relative(
                        candidate.coefficient_density, expected_density
                    ),
                    "lower_matrix_covariance_relative_residual": _relative(
                        candidate.action.lower_mechanical_matrix, expected_lower
                    ),
                    "physical_density_relative_residual": _relative(
                        candidate_density.density_direct,
                        baseline_density.density_direct,
                    ),
                    "candidate_orbital_residual": candidate.orbital_residual,
                    "candidate_density_fixed_point_residual": (
                        candidate.density_fixed_point_residual
                    ),
                }
            )
    return rows


def _stationarity_checks(
    factory: ExactWilsonStationaryFactory,
    states: dict[tuple[float, WilsonStationaryBranch], ExactWilsonStationaryState],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field_strength in _GAUGE_FIELDS:
        gauge = AffineMagneticGauge(
            UniformMagneticField((0.0, 0.0, field_strength)),
            origin_au=_SYMMETRIC_ORIGIN,
        )
        for branch in (
            WilsonStationaryBranch.HARTREE,
            WilsonStationaryBranch.KOHN_SHAM_LDA,
        ):
            state = states[(field_strength, branch)]
            model = factory.model(gauge, branch)
            coefficients = np.asarray(state.coefficients)
            occupations = np.asarray(state.occupations)
            overlap = np.asarray(state.action.overlap)
            rng = np.random.default_rng(
                6400
                + round(field_strength * 1000)
                + (0 if branch.value == "hartree" else 1)
            )
            raw = rng.normal(size=coefficients.shape)
            for direction_name, direction in (
                ("real", raw.astype(np.complex128)),
                ("imaginary", 1j * raw),
            ):
                direction = direction / np.linalg.norm(direction)
                for step in _STATIONARITY_STEPS:
                    energies: list[float] = []
                    for sign in (1.0, -1.0):
                        displaced = _retract(
                            coefficients,
                            direction,
                            overlap,
                            sign * step,
                        )
                        action = model.evaluate(_density(displaced, occupations))
                        energies.append(float(action.energy_molecular_total_au))
                    derivative = (energies[0] - energies[1]) / (2.0 * step)
                    rows.append(
                        {
                            "field_au": field_strength,
                            "branch": branch.value,
                            "direction": direction_name,
                            "step": step,
                            "central_energy_derivative_abs_au": abs(derivative),
                        }
                    )
    return rows


def _grid_refinement(
    reference: Any,
    main_scan: dict[str, Any],
    states: dict[tuple[float, WilsonStationaryBranch], ExactWilsonStationaryState],
) -> list[dict[str, Any]]:
    main_rows = {
        (float(row["field_au"]), str(row["branch"])): row
        for row in main_scan["rows"]
    }
    rows: list[dict[str, Any]] = []
    for level in _GRID_LEVELS:
        if level == 4:
            for field_strength in _GRID_FIELDS:
                for branch in (
                    WilsonStationaryBranch.HARTREE,
                    WilsonStationaryBranch.KOHN_SHAM_LDA,
                ):
                    state = states[(field_strength, branch)]
                    source = main_rows[(field_strength, branch.value)]
                    rows.append(
                        {
                            "level": level,
                            "points": int(main_scan["grid_points"]),
                            "field_au": field_strength,
                            "branch": branch.value,
                            "energy_molecular_total_au": source[
                                "energy_molecular_total_au"
                            ],
                            "density_fixed_point_residual": (
                                state.density_fixed_point_residual
                            ),
                            "orbital_residual": state.orbital_residual,
                        }
                    )
            continue
        quadrature = prepare_ao_quadrature(
            reference,
            BackendConfig(),
            grid_policy=AOGridPolicy.qualification(level),
            block_size=2048,
        )
        factory = prepare_exact_wilson_stationary_factory(
            quadrature,
            auxiliary_basis=_AUXILIARY_BASIS,
            functional=_FUNCTIONAL,
        )
        for field_strength in _GRID_FIELDS:
            gauge = AffineMagneticGauge(
                UniformMagneticField((0.0, 0.0, field_strength)),
                origin_au=_SYMMETRIC_ORIGIN,
            )
            for branch in (
                WilsonStationaryBranch.HARTREE,
                WilsonStationaryBranch.KOHN_SHAM_LDA,
            ):
                state, _ = _solve(
                    factory.model(gauge, branch),
                    initial_coefficients=states[(field_strength, branch)].coefficients,
                )
                rows.append(
                    {
                        "level": level,
                        "points": quadrature.grid.npoints,
                        "field_au": field_strength,
                        "branch": branch.value,
                        "energy_molecular_total_au": float(
                            state.action.energy_molecular_total_au
                        ),
                        "density_fixed_point_residual": (
                            state.density_fixed_point_residual
                        ),
                        "orbital_residual": state.orbital_residual,
                    }
                )
    return rows


def _tolerance_refinement(
    factory: ExactWilsonStationaryFactory,
    states: dict[tuple[float, WilsonStationaryBranch], ExactWilsonStationaryState],
) -> list[dict[str, Any]]:
    field_strength = 0.06
    gauge = AffineMagneticGauge(
        UniformMagneticField((0.0, 0.0, field_strength)),
        origin_au=_SYMMETRIC_ORIGIN,
    )
    rows: list[dict[str, Any]] = []
    for branch in (
        WilsonStationaryBranch.HARTREE,
        WilsonStationaryBranch.KOHN_SHAM_LDA,
    ):
        main = states[(field_strength, branch)]
        for name, policy in _POLICIES.items():
            if name == "tight":
                state = main
                elapsed = 0.0
            else:
                state, elapsed = _solve(
                    factory.model(gauge, branch),
                    policy=policy,
                    initial_coefficients=reference_initial_coefficients(factory),
                )
            rows.append(
                {
                    "field_au": field_strength,
                    "branch": branch.value,
                    "policy": name,
                    "density_tolerance": policy.density_tolerance,
                    "orbital_tolerance": policy.orbital_tolerance,
                    "energy_tolerance_au": policy.energy_tolerance_au,
                    "iterations": len(state.iterations),
                    "elapsed_seconds": elapsed,
                    "energy_molecular_total_au": float(
                        state.action.energy_molecular_total_au
                    ),
                    "density_fixed_point_residual": (
                        state.density_fixed_point_residual
                    ),
                    "orbital_residual": state.orbital_residual,
                }
            )
    return rows


def reference_initial_coefficients(factory: ExactWilsonStationaryFactory) -> np.ndarray:
    """Return the occupied field-free reference frame for refinement restarts."""

    reference = factory.quadrature.reference
    occupied = np.flatnonzero(np.asarray(reference.ground_state.occupations) > 0.0)
    return np.asarray(reference.ground_state.coefficients[:, occupied])


def _small_finite_field(reference: Any) -> list[dict[str, Any]]:
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(3),
        block_size=1024,
    )
    factory = prepare_exact_wilson_stationary_factory(
        quadrature,
        auxiliary_basis=_AUXILIARY_BASIS,
        functional=_FUNCTIONAL,
    )
    gauge = AffineMagneticGauge(
        UniformMagneticField((0.0, 0.0, 0.02)),
        origin_au=_SYMMETRIC_ORIGIN,
    )
    rows = []
    for branch in (
        WilsonStationaryBranch.HARTREE,
        WilsonStationaryBranch.KOHN_SHAM_LDA,
    ):
        state, elapsed = _solve(factory.model(gauge, branch))
        rows.append(
            _state_record(
                "h2",
                0.02,
                state,
                elapsed,
                "finite_field_stationary",
            )
        )
    return rows


def _write_csv(output: Path, name: str, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with (output / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact directory: {output}")
    output.mkdir(parents=True)
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    repo = Path(__file__).resolve().parents[1]

    references: dict[str, Any] = {}
    for system in ("h2", "h3plus"):
        config = _configuration(system, output / f"{system}.reference.h5", timestamp)
        (output / f"{system}.reference.toml").write_text(
            dumps_config(config), encoding="utf-8"
        )
        reference = prepare_pyscf_reference(config)
        reference.save()
        references[system] = reference

    field_free: dict[str, Any] = {}
    raw_arrays: dict[str, np.ndarray] = {}
    for system, reference in references.items():
        record, arrays = _field_free_recovery(system, reference)
        field_free[system] = record
        raw_arrays.update(arrays)

    small_finite_field = _small_finite_field(references["h2"])
    main_scan, factory, states, scan_arrays = _main_h3_scan(references["h3plus"])
    raw_arrays.update(scan_arrays)
    gauge_checks = _gauge_checks(references["h3plus"], factory, states)
    stationarity = _stationarity_checks(factory, states)
    grid_refinement = _grid_refinement(references["h3plus"], main_scan, states)
    tolerance_refinement = _tolerance_refinement(factory, states)

    np.savez_compressed(output / "stationary_states.npz", **raw_arrays)
    result = {
        "schema": "aion.chapter13-nq4-stationary",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "equations": [
            "eq:wilson-hartree-ks-total-hartree-matrix",
            "eq:wilson-hartree-ks-total-ks-matrix",
            "eq:wilson-hartree-ks-hartree-mechanical-energy",
            "eq:wilson-hartree-ks-ks-mechanical-energy",
            "eq:wilson-hartree-ks-stationary-data",
            "eq:wilson-hartree-ks-stationary-density-data",
            "eq:wilson-hartree-ks-stationary-generalized-eigenproblem",
            "eq:wilson-hartree-ks-hartree-energy-double-counting-form",
            "eq:wilson-hartree-ks-ks-energy-double-counting-form",
        ],
        "realization": {
            "h2": {"charge": 0, "electrons": 2, "basis": "sto-3g"},
            "h3plus": {
                "charge": 1,
                "electrons": 2,
                "basis": "cc-pvdz",
                "geometry": "equilateral, side 1.4 bohr, molecular plane xy",
            },
            "functional": _FUNCTIONAL,
            "auxiliary_basis": _AUXILIARY_BASIS,
            "main_grid_level": 4,
            "main_grid_pruning": "none",
            "field_direction": "Bz",
            "field_interval_au": [min(_H3_FIELDS), max(_H3_FIELDS)],
            "fields_au": list(_H3_FIELDS),
            "symmetric_gauge_origin_au": list(_SYMMETRIC_ORIGIN),
            "landau_gauge_origin_au": list(_LANDAU_ORIGIN),
            "occupations": [2.0],
            "precision": "float64_complex128",
            "stationary_algorithm": (
                "direct Hermitian generalized eigensolve plus lower-matrix Pulay; "
                "occupied-frame metric retraction only"
            ),
            "main_scf_policy": asdict(_MAIN_POLICY),
            "state_identity_labels": [
                "field_free_stationary",
                "finite_field_stationary",
            ],
            "not_executed_state_identities": [
                "adiabatically_prepared",
                "sudden_quench",
            ],
        },
        "field_free_recovery": field_free,
        "h2_finite_field": small_finite_field,
        "h3plus_main_scan": main_scan,
        "gauge_checks": gauge_checks,
        "energy_stationarity": stationarity,
        "grid_refinement": grid_refinement,
        "stationary_tolerance_refinement": tolerance_refinement,
    }
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output, "h3plus_stationary_scan.csv", main_scan["rows"])
    _write_csv(output, "gauge_checks.csv", gauge_checks)
    _write_csv(output, "energy_stationarity.csv", stationarity)
    _write_csv(output, "grid_refinement.csv", grid_refinement)
    _write_csv(output, "stationary_tolerance_refinement.csv", tolerance_refinement)

    tracked_diff = _git(repo, "diff", "--binary", "HEAD", binary=True)
    assert isinstance(tracked_diff, bytes)
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    assert isinstance(status, str)
    head = _git(repo, "rev-parse", "HEAD")
    assert isinstance(head, str)
    source_paths = (
        repo / "tools/run_chapter13_nq4_stationary.py",
        repo / "src/aion/electronic_structure/wilson_stationary.py",
        repo / "src/aion/electronic_structure/ri_wilson_hartree.py",
        repo / "src/aion/electronic_structure/wilson_lda.py",
        repo / "src/aion/electronic_structure/magnetic_matrices.py",
        repo / "environment.yml",
        repo / "conda-linux-64.lock",
    )
    artifact_paths = tuple(
        sorted(
            path
            for path in output.iterdir()
            if path.is_file() and path.name not in {"provenance.json", "completed.json"}
        )
    )
    provenance = {
        "timestamp_utc": timestamp,
        "command": " ".join(shlex.quote(value) for value in (sys.executable, *sys.argv)),
        "repository": str(repo),
        "git_head": head.strip(),
        "git_status_porcelain": status.splitlines(),
        "git_tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "environment": {
            "dependencies": DependencyVersions.current().as_mapping(),
            "loaded_modules": os.environ.get("LOADEDMODULES", ""),
            "thread_limits": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "python_executable": sys.executable,
        },
        "source_sha256": {str(path): _sha256(path) for path in source_paths},
        "artifacts_sha256": {path.name: _sha256(path) for path in artifact_paths},
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    completed = {
        "schema": "aion.chapter13-nq4-completed",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "result_sha256": _sha256(result_path),
        "provenance_sha256": _sha256(provenance_path),
    }
    (output / "completed.json").write_text(
        json.dumps(completed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "status": "executed_unreviewed"}, indent=2))


if __name__ == "__main__":
    main()
