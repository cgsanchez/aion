#!/usr/bin/env python3
"""Execute the Chapter 13 NQ5 nonlinear source and static-observable campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from aion.config import BackendConfig
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
    DependencyVersions,
    RIMetricRankPolicy,
    WilsonStationaryBranch,
    evaluate_exact_static_wilson_grid_one_electron_action,
    evaluate_exact_wilson_charge,
    evaluate_exact_wilson_one_electron_sample,
    evaluate_nonlinear_pure_gauge_ward,
    evaluate_nonlinear_weak_continuity,
    evaluate_nonlinear_weak_current_pairing,
    evaluate_static_nonlinear_wilson_grid_action,
    load_reference_data,
    prepare_ao_quadrature,
    prepare_exact_wilson_stationary_factory,
)
from aion.formulations import EOMTriple, one_electron_velocity_density

_NQ4_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "chapter13_wilson_adiabatic_qualification/"
    "nq4_stationary_20260921T232232Z_6197d6828bb3"
)
_FIELD_STRENGTH = 0.03
_GAUGE_ORIGIN = (0.17, -0.31, 0.23)
_GRID_LEVELS = (3, 4, 5)
_SOURCE_STEPS = (3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5)
_PATH_ORDERS = (12, 24, 40)
_RANK_THRESHOLDS = (1.0e-6, 1.0e-7, 1.0e-9)
_FUNCTIONAL = "lda,vwn"
_AUXILIARY_BASIS = "weigend"
_BRANCHES = (
    WilsonStationaryBranch.HARTREE,
    WilsonStationaryBranch.KOHN_SHAM_LDA,
)


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


def _float(value: object) -> float:
    return float(np.asarray(value).real)


def _write_csv(output: Path, name: str, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with (output / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _variation(order: int = 24) -> GaussianVectorPotentialVariation:
    return GaussianVectorPotentialVariation(
        amplitude_au=(0.19, -0.13, 0.07),
        center_au=(0.23, -0.17, 0.11),
        exponent_au_inverse2=0.41,
        path_quadrature_order=order,
    )


def _source_sample(gauge: AffineMagneticGauge) -> UniformMagneticSourceSample:
    return UniformMagneticSourceSample(
        0.0,
        gauge.field,
        origin_au=gauge.origin_au,
        gauge_kind=gauge.kind,
        landau_axis=gauge.landau_axis,
    )


def _history(
    coefficients: np.ndarray,
    occupations: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coefficient_velocity = (
        rng.normal(size=coefficients.shape) + 1j * rng.normal(size=coefficients.shape)
    ) / np.sqrt(coefficients.shape[0])
    density = np.einsum(
        "mi,i,ni->mn",
        coefficients,
        occupations,
        coefficients.conj(),
        optimize=True,
    )
    velocity_density = np.einsum(
        "mi,i,ni->mn",
        coefficient_velocity,
        occupations,
        coefficients.conj(),
        optimize=True,
    )
    return density, velocity_density, coefficient_velocity


def _normalized_dynamic_density(metric: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coefficient = rng.normal(size=(metric.shape[0], 1)) + 1j * rng.normal(
        size=(metric.shape[0], 1)
    )
    coefficient /= np.sqrt(
        (coefficient.conj().T @ metric @ coefficient).real.item()
    )
    return 2.0 * coefficient @ coefficient.conj().T


def _current_record(branch: str, level: int, value: Any) -> dict[str, Any]:
    xc_direction = (
        0.0
        if value.exchange_correlation is None
        else _float(value.exchange_correlation.source_energy_direction)
    )
    return {
        "branch": branch,
        "grid_level": level,
        "one_electron_source_pairing": _float(value.one_electron_action.total),
        "hartree_energy_source_direction": _float(value.hartree.source_energy_direction),
        "xc_energy_source_direction": xc_direction,
        "closure_current_pairing": _float(value.closure_pairing),
        "fixed_coordinate_total_pairing": _float(value.total_pairing),
        "ambient_minimal_pairing": _float(value.ambient_minimal_pairing),
        "one_electron_embedding_pairing": _float(
            value.one_electron_embedding_pairing
        ),
        "complete_embedding_pairing": _float(value.embedding_pairing),
        "tangential_pairing": _float(value.tangential_pairing),
        "normal_subspace_pairing": _float(value.normal_subspace_pairing),
        "on_shell_pairing": _float(value.on_shell_pairing),
        "on_shell_decomposition_residual": _float(
            value.on_shell_decomposition_residual
        ),
    }


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

    nq4_completed = json.loads((_NQ4_ROOT / "completed.json").read_text(encoding="utf-8"))
    if nq4_completed["result_sha256"] != _sha256(_NQ4_ROOT / "result.json"):
        raise RuntimeError("accepted NQ4 result hash changed")
    if nq4_completed["provenance_sha256"] != _sha256(_NQ4_ROOT / "provenance.json"):
        raise RuntimeError("accepted NQ4 provenance hash changed")

    reference = load_reference_data(_NQ4_ROOT / "h3plus.reference.h5")
    accepted_states = np.load(_NQ4_ROOT / "stationary_states.npz", allow_pickle=False)
    gauge = AffineMagneticGauge(
        UniformMagneticField((0.0, 0.0, _FIELD_STRENGTH)),
        origin_au=_GAUGE_ORIGIN,
    )
    source = _source_sample(gauge)
    occupations = np.asarray((2.0,))
    coefficients = {
        branch: np.asarray(
            accepted_states[f"h3plus_b0p030_{branch.value}_coefficients"],
            dtype=np.complex128,
        )
        for branch in _BRANCHES
    }
    histories = {
        branch: _history(coefficients[branch], occupations, 7510 + index)
        for index, branch in enumerate(_BRANCHES)
    }

    quadratures: dict[int, Any] = {}
    factories: dict[int, Any] = {}
    samples: dict[int, Any] = {}
    grid_actions: dict[int, Any] = {}
    for level in _GRID_LEVELS:
        quadrature = prepare_ao_quadrature(
            reference,
            BackendConfig(),
            grid_policy=AOGridPolicy.qualification(level),
            block_size=2048,
        )
        quadratures[level] = quadrature
        factories[level] = prepare_exact_wilson_stationary_factory(
            quadrature,
            auxiliary_basis=_AUXILIARY_BASIS,
            functional=_FUNCTIONAL,
        )
        samples[level] = evaluate_exact_wilson_one_electron_sample(quadrature, source)
        grid_actions[level] = evaluate_exact_static_wilson_grid_one_electron_action(
            quadrature,
            gauge,
        )

    charge_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    source_fd_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    on_shell_rows: list[dict[str, Any]] = []
    ward_rows: list[dict[str, Any]] = []
    continuity_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}

    for level in _GRID_LEVELS:
        grid = grid_actions[level]
        grid_triple = EOMTriple(
            grid.overlap,
            grid.mechanical,
            np.zeros_like(np.asarray(grid.overlap)),
        )
        for branch in _BRANCHES:
            model = factories[level].model(gauge, branch)
            density, velocity, _ = histories[branch]
            charge = evaluate_exact_wilson_charge(model, density)
            charge_rows.append(
                {
                    "branch": branch.value,
                    "grid_level": level,
                    "grid_points": quadratures[level].grid.npoints,
                    "integrated_charge_grid": _float(charge.integrated_charge_grid),
                    "integrated_charge_stable_metric": _float(
                        charge.integrated_charge_metric
                    ),
                    "metric_particle_number": _float(charge.metric_particle_number),
                    "grid_metric_charge_residual": abs(
                        _float(charge.integrated_charge_grid)
                        - _float(charge.integrated_charge_metric)
                    ),
                    "electron_count_residual": abs(
                        _float(charge.integrated_charge_metric) + 2.0
                    ),
                    "density_imaginary_max_abs": (
                        charge.density_evaluation.density_direct_imaginary_max_abs
                    ),
                }
            )
            if level == 4:
                arrays[f"charge_density_{branch.value}"] = np.asarray(
                    charge.signed_charge_density
                )
            current = evaluate_nonlinear_weak_current_pairing(
                model,
                samples[level],
                density,
                velocity,
                _variation(),
                one_electron_triple=grid_triple,
            )
            record = _current_record(branch.value, level, current)
            record["history_class"] = "off_shell_fixed_coefficient"
            source_rows.append(record)

            if level == 4:
                analytic = _float(current.total_pairing)
                for step in _SOURCE_STEPS:
                    values = []
                    for sign in (1.0, -1.0):
                        displaced = PerturbedVectorPotential(
                            gauge,
                            _variation(),
                            sign * step,
                        )
                        values.append(
                            _float(
                                evaluate_static_nonlinear_wilson_grid_action(
                                    model,
                                    density,
                                    velocity,
                                    displaced,
                                ).electronic_action_value
                            )
                        )
                    finite = (values[0] - values[1]) / (2.0 * step)
                    source_fd_rows.append(
                        {
                            "branch": branch.value,
                            "grid_level": level,
                            "step": step,
                            "analytic_pairing": analytic,
                            "central_finite_difference": finite,
                            "absolute_error": abs(finite - analytic),
                        }
                    )

                stable_action = model.evaluate(density)
                stable_triple = EOMTriple(
                    samples[level].metric,
                    stable_action.lower_mechanical_matrix,
                    samples[level].connection.connection,
                )
                shell_velocity = one_electron_velocity_density(
                    density,
                    stable_triple,
                    model.backend,
                )
                shell_current = evaluate_nonlinear_weak_current_pairing(
                    model,
                    samples[level],
                    density,
                    shell_velocity,
                    _variation(),
                )
                shell_record = _current_record(branch.value, level, shell_current)
                shell_record.update(
                    {
                        "history_class": "accepted_stationary_coefficient_shell",
                        "stationary_density_source": str(_NQ4_ROOT),
                    }
                )
                on_shell_rows.append(shell_record)

                _, _, coefficient_velocity = histories[branch]
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
                    samples[level],
                    coefficients[branch],
                    coefficient_velocity,
                    occupations,
                    gauge_test,
                    gauge_parameter_rate=gauge_rate,
                )
                ward_rows.append(
                    {
                        "branch": branch.value,
                        "source_pairing": _float(ward.source_pairing),
                        "matter_pairing": _float(ward.matter_pairing),
                        "ward_residual": abs(_float(ward.total_ward_residual)),
                        "one_electron_source_pairing": _float(
                            ward.one_electron_source_pairing
                        ),
                        "closure_source_pairing": _float(
                            ward.closure_source_pairing
                        ),
                        "lower_coefficient_residual_relative_norm": (
                            ward.lower_coefficient_residual_relative_norm
                        ),
                    }
                )

                dynamic_density = _normalized_dynamic_density(
                    np.asarray(grid.overlap),
                    9410 + (0 if branch is WilsonStationaryBranch.HARTREE else 1),
                )
                for weight_index, weight in enumerate(
                    (
                        GaussianScalarGaugeVariation(
                            amplitude=1.0,
                            center_au=(0.0, 0.0, 0.0),
                            exponent_au_inverse2=0.2,
                        ),
                        GaussianScalarGaugeVariation(
                            amplitude=1.0,
                            center_au=(0.7, -0.3, 0.2),
                            exponent_au_inverse2=0.5,
                        ),
                        GaussianScalarGaugeVariation(
                            amplitude=1.0,
                            center_au=(-0.4, 0.6, -0.1),
                            exponent_au_inverse2=1.0,
                        ),
                    )
                ):
                    continuity = evaluate_nonlinear_weak_continuity(
                        model,
                        samples[level],
                        dynamic_density,
                        weight,
                        one_electron_triple=grid_triple,
                    )
                    continuity_rows.append(
                        {
                            "branch": branch.value,
                            "weight_index": weight_index,
                            "center_au": list(weight.center_au),
                            "exponent_au_inverse2": weight.exponent_au_inverse2,
                            "weighted_charge": _float(continuity.weighted_charge),
                            "weighted_charge_derivative": _float(
                                continuity.weighted_charge_derivative
                            ),
                            "on_shell_current_pairing": _float(
                                continuity.weak_current_pairing.on_shell_pairing
                            ),
                            "fixed_coordinate_current_pairing": _float(
                                continuity.weak_current_pairing.total_pairing
                            ),
                            "tangential_pairing": _float(
                                continuity.weak_current_pairing.tangential_pairing
                            ),
                            "finite_region_residual": abs(
                                _float(continuity.finite_region_residual)
                            ),
                            "total_charge_rate": _float(
                                continuity.total_charge_rate
                            ),
                            "global_charge_residual": abs(
                                _float(continuity.global_charge_residual)
                            ),
                        }
                    )

                for order in _PATH_ORDERS:
                    value = evaluate_nonlinear_weak_current_pairing(
                        model,
                        samples[level],
                        density,
                        velocity,
                        _variation(order),
                        one_electron_triple=grid_triple,
                    )
                    path_rows.append(
                        {
                            "branch": branch.value,
                            "path_order": order,
                            "fixed_coordinate_total_pairing": _float(
                                value.total_pairing
                            ),
                            "closure_current_pairing": _float(
                                value.closure_pairing
                            ),
                        }
                    )

    main_quadrature = quadratures[4]
    main_grid = grid_actions[4]
    main_triple = EOMTriple(
        main_grid.overlap,
        main_grid.mechanical,
        np.zeros_like(np.asarray(main_grid.overlap)),
    )
    for threshold in _RANK_THRESHOLDS:
        rank_factory = prepare_exact_wilson_stationary_factory(
            main_quadrature,
            auxiliary_basis=_AUXILIARY_BASIS,
            functional=_FUNCTIONAL,
            rank_policy=RIMetricRankPolicy(absolute_threshold=threshold),
        )
        for branch in _BRANCHES:
            density, velocity, _ = histories[branch]
            value = evaluate_nonlinear_weak_current_pairing(
                rank_factory.model(gauge, branch),
                samples[4],
                density,
                velocity,
                _variation(),
                one_electron_triple=main_triple,
            )
            rank_rows.append(
                {
                    "branch": branch.value,
                    "absolute_threshold": threshold,
                    "retained_rank": value.hartree.auxiliary_rank,
                    "hartree_energy_source_direction": _float(
                        value.hartree.source_energy_direction
                    ),
                    "fixed_coordinate_total_pairing": _float(
                        value.total_pairing
                    ),
                }
            )

    np.savez_compressed(output / "observables.npz", **arrays)  # type: ignore[arg-type]
    for name, rows in (
        ("charge.csv", charge_rows),
        ("weak_current.csv", source_rows),
        ("source_finite_difference.csv", source_fd_rows),
        ("path_refinement.csv", path_rows),
        ("on_shell_decomposition.csv", on_shell_rows),
        ("ward.csv", ward_rows),
        ("continuity.csv", continuity_rows),
        ("ri_rank_refinement.csv", rank_rows),
    ):
        _write_csv(output, name, rows)

    result = {
        "schema": "aion.chapter13-nq5-sources",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "equations": [
            "eq:wilson-hartree-ks-charge-density",
            "eq:wilson-hartree-ks-closure-current-pairing",
            "eq:wilson-hartree-ks-off-shell-current-residual-form",
            "eq:wilson-hartree-ks-nonlinear-subspace-current",
            "eq:wilson-hartree-ks-on-shell-current",
            "eq:wilson-hartree-ks-off-shell-ward-identity",
            "eq:wilson-hartree-ks-finite-region-continuity",
            "eq:wilson-hartree-ks-global-charge-conservation",
        ],
        "realization": {
            "system": "H3+",
            "charge": 1,
            "electrons": 2,
            "basis": "cc-pvdz",
            "functional": _FUNCTIONAL,
            "auxiliary_basis": _AUXILIARY_BASIS,
            "field_au": [0.0, 0.0, _FIELD_STRENGTH],
            "gauge_origin_au": list(_GAUGE_ORIGIN),
            "grid_levels": list(_GRID_LEVELS),
            "source_difference_steps": list(_SOURCE_STEPS),
            "path_quadrature_orders": list(_PATH_ORDERS),
            "ri_absolute_thresholds": list(_RANK_THRESHOLDS),
            "precision": "float64_complex128",
            "current_label": "auxiliary Hartree/adiabatic-KS action source current",
            "not_claimed": "interacting physical transverse current",
        },
        "accepted_nq4_source": {
            "root": str(_NQ4_ROOT),
            "result_sha256": nq4_completed["result_sha256"],
            "provenance_sha256": nq4_completed["provenance_sha256"],
            "stationary_states_sha256": _sha256(_NQ4_ROOT / "stationary_states.npz"),
            "reference_sha256": _sha256(_NQ4_ROOT / "h3plus.reference.h5"),
        },
        "charge": charge_rows,
        "off_shell_weak_current": source_rows,
        "source_finite_differences": source_fd_rows,
        "path_refinement": path_rows,
        "on_shell_decomposition": on_shell_rows,
        "off_shell_ward": ward_rows,
        "weak_continuity": continuity_rows,
        "ri_rank_refinement": rank_rows,
    }
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tracked_diff = _git(repo, "diff", "--binary", "HEAD", binary=True)
    assert isinstance(tracked_diff, bytes)
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    assert isinstance(status, str)
    head = _git(repo, "rev-parse", "HEAD")
    assert isinstance(head, str)
    source_paths = (
        repo / "tools/run_chapter13_nq5_sources.py",
        repo / "src/aion/electromagnetism/test_variations.py",
        repo / "src/aion/electronic_structure/time_connection.py",
        repo / "src/aion/electronic_structure/wilson_sources.py",
        repo / "src/aion/electronic_structure/ri_wilson_hartree.py",
        repo / "src/aion/electronic_structure/wilson_lda.py",
        repo / "src/aion/formulations/action.py",
        repo / "src/aion/formulations/exact_one_electron.py",
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
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completed = {
        "schema": "aion.chapter13-nq5-completed",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "result_sha256": _sha256(result_path),
        "provenance_sha256": _sha256(provenance_path),
    }
    (output / "completed.json").write_text(
        json.dumps(completed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "status": "executed_unreviewed"}, indent=2))


if __name__ == "__main__":
    main()
