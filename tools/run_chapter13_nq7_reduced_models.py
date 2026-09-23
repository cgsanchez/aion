#!/usr/bin/env python3
"""Run restartable Chapter 13 NQ7 reduced-model qualification phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aion.config import (
    BackendConfig,
    ElectronicStructureConfig,
    MetadataConfig,
    ReferenceOutputConfig,
    XCFamily,
)
from aion.electromagnetism import (
    AffineMagneticGauge,
    UniformMagneticField,
    UniformMagneticSourceSample,
)
from aion.electronic_structure import (
    AOGridPolicy,
    DependencyVersions,
    ReducedWilsonLevel,
    StationarySCFPolicy,
    WilsonStationaryBranch,
    evaluate_exact_wilson_power,
    load_reference_data,
    prepare_ao_quadrature,
    prepare_exact_wilson_dynamic_spatial_action,
    prepare_exact_wilson_stationary_factory,
    prepare_pyscf_reference,
    prepare_reduced_wilson_factory,
)
from aion.propagation import (
    NonlinearGaussMagnusPolicy,
    propagate_nonlinear_contravariant_density,
)

_CAMPAIGN_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "chapter13_wilson_adiabatic_qualification"
)
_NQ4_ROOT = _CAMPAIGN_ROOT / "nq4_stationary_20260921T232232Z_6197d6828bb3"
_NQ6_ROOT = _CAMPAIGN_ROOT / "nq6_dynamics_20260922T205959Z_a197fbc55b62"
_FUNCTIONAL = "lda,vwn"
_AUXILIARY_BASIS = "weigend"
_GAUGE_ORIGIN = (0.17, -0.31, 0.23)
_FIELDS = (0.0, 0.001, 0.003, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
_DOMAIN_FIELDS = (0.0, 0.03, 0.06, 0.1, 0.2, 0.4, 0.8, 1.2)
_DOMAIN_BASES = ("sto-3g", "cc-pvdz", "aug-cc-pvdz")
_LEVELS = tuple(ReducedWilsonLevel)
_BRANCHES = (
    WilsonStationaryBranch.HARTREE,
    WilsonStationaryBranch.KOHN_SHAM_LDA,
)
_DYNAMIC_CASES = (
    "electric_field_free",
    "electric_static_b",
    "magnetic_induction",
)
_FINAL_TIME = 2.0
_STATIC_FIELD = 0.03
_ELECTRIC_AMPLITUDE = 0.005
_MAGNETIC_AMPLITUDE = 0.005
_SCF_POLICY = StationarySCFPolicy(
    maximum_iterations=160,
    density_tolerance=2.0e-11,
    orbital_tolerance=2.0e-11,
    energy_tolerance_au=2.0e-12,
)
_NONLINEAR_POLICY = NonlinearGaussMagnusPolicy(
    tolerance=1.0e-11,
    maximum_iterations=80,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _relative_difference(value: object, reference: object) -> float:
    candidate = np.asarray(value)
    baseline = np.asarray(reference)
    return float(
        np.linalg.norm(candidate - baseline)
        / max(1.0, float(np.linalg.norm(baseline)))
    )


def _field_tag(value: float) -> str:
    return f"b{value:.3f}".replace(".", "p")


def _checkpoint(output: Path, phase: str, run_id: str) -> tuple[Path, Path]:
    directory = output / "checkpoints" / phase
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{run_id}.json", directory / f"{run_id}.npz"


def _authenticated_parent(root: Path) -> dict[str, Any]:
    completed_path = root / "completed.json"
    provenance_path = root / "provenance.json"
    result_path = root / "result.json"
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    if completed["status"] != "complete":
        raise RuntimeError(f"parent campaign is not complete: {root}")
    if completed["provenance_sha256"] != _sha256(provenance_path):
        raise RuntimeError(f"parent provenance hash mismatch: {root}")
    if completed["result_sha256"] != _sha256(result_path):
        raise RuntimeError(f"parent result hash mismatch: {root}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for relative, expected in provenance["artifacts_sha256"].items():
        path = root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"parent artifact hash mismatch: {path}")
    return {
        "root": str(root),
        "completed_sha256": _sha256(completed_path),
        "provenance_sha256": _sha256(provenance_path),
        "result_sha256": _sha256(result_path),
    }


def _campaign_identity(output: Path, repo: Path) -> dict[str, Any]:
    path = output / "campaign_identity.json"
    head = _git(repo, "rev-parse", "HEAD").strip()
    source_paths = (
        repo / "tools" / "run_chapter13_nq7_reduced_models.py",
        repo / "src" / "aion" / "electronic_structure" / "reduced_wilson.py",
        repo / "src" / "aion" / "electronic_structure" / "wilson_lda.py",
        repo / "src" / "aion" / "electronic_structure" / "wilson_stationary.py",
    )
    current = {
        "schema": "aion.chapter13.nq7.identity.v1",
        "implementation_commit": head,
        "source_sha256": {
            str(item.relative_to(repo)): _sha256(item) for item in source_paths
        },
    }
    if path.is_file():
        stored: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if stored != current:
            raise RuntimeError(
                "campaign implementation identity changed; use a new output directory"
            )
        return stored
    _write_json(path, current)
    return current


def _main_reference() -> Any:
    return load_reference_data(_NQ4_ROOT / "h3plus.reference.h5")


def _factories(reference: Any) -> tuple[Any, Any]:
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(4),
        block_size=1024,
    )
    exact = prepare_exact_wilson_stationary_factory(
        quadrature,
        auxiliary_basis=_AUXILIARY_BASIS,
        functional=_FUNCTIONAL,
    )
    return exact, prepare_reduced_wilson_factory(exact)


def _exact_state_key(field: float, branch: WilsonStationaryBranch, suffix: str) -> str:
    return f"h3plus_{_field_tag(field)}_{branch.value}_{suffix}"


def _density_grid_for_level(action: Any, level: ReducedWilsonLevel) -> np.ndarray:
    result = action.closure.density
    if level in (ReducedWilsonLevel.P0, ReducedWilsonLevel.E1):
        return np.asarray(result.density_zero)
    return np.asarray(result.density_assembled)


def _stationary_phase(output: Path) -> None:
    reference = _main_reference()
    exact_factory, reduced_factory = _factories(reference)
    parent = np.load(_NQ4_ROOT / "stationary_states.npz")
    source_rate = (0.0, 0.0, 0.002)

    for field_value in _FIELDS:
        gauge = AffineMagneticGauge(
            UniformMagneticField((0.0, 0.0, field_value)),
            origin_au=_GAUGE_ORIGIN,
        )
        exact_spatial = prepare_exact_wilson_dynamic_spatial_action(
            exact_factory,
            gauge,
        )
        source = UniformMagneticSourceSample(
            time_au=0.37,
            field=gauge.field,
            magnetic_field_dot_au=source_rate,
            electric_field_origin_au=(0.004, 0.0, 0.0),
            origin_au=_GAUGE_ORIGIN,
        )
        for branch in _BRANCHES:
            exact_density = np.asarray(
                parent[_exact_state_key(field_value, branch, "density")]
            )
            exact_coefficients = np.asarray(
                parent[_exact_state_key(field_value, branch, "coefficients")]
            )
            exact_model = exact_factory.model(gauge, branch)
            exact_action = exact_model.evaluate(exact_density)
            exact_grid_density = np.asarray(
                exact_factory.lda_evaluator.evaluate(exact_density, gauge).density
            )
            exact_dynamic = exact_spatial.sample(source, branch).evaluate(exact_density)
            exact_power = evaluate_exact_wilson_power(exact_dynamic, exact_density)

            solved: dict[str, Any] = {}
            for level in _LEVELS:
                run_id = f"{_field_tag(field_value)}_{branch.value}_{level.value}"
                record_path, array_path = _checkpoint(output, "stationary", run_id)
                if record_path.is_file() and array_path.is_file():
                    continue
                started = perf_counter()
                try:
                    if level is ReducedWilsonLevel.E1 and ReducedWilsonLevel.P0.value in solved:
                        state = solved[ReducedWilsonLevel.P0.value]
                    else:
                        model = reduced_factory.model(gauge, level, branch)
                        state = model.solve(
                            policy=_SCF_POLICY,
                            initial_coefficients=exact_coefficients,
                        )
                        solved[level.value] = state
                    model = reduced_factory.spatial_action(gauge, level).sample(
                        source,
                        branch,
                    )
                    own_action = model.evaluate(state.coefficient_density)
                    matched_action = model.evaluate(exact_density)
                    response = model.source_response(exact_density)
                    power = model.power(exact_density)
                    density_grid = _density_grid_for_level(own_action, level)
                    matched_grid = _density_grid_for_level(matched_action, level)
                    record = {
                        "schema": "aion.chapter13.nq7.stationary.v1",
                        "status": "complete",
                        "run_id": run_id,
                        "field_au": field_value,
                        "branch": branch.value,
                        "level": level.value,
                        "elapsed_seconds": perf_counter() - started,
                        "energy_molecular_total_au": float(
                            own_action.energy_molecular_total_au
                        ),
                        "exact_energy_molecular_total_au": float(
                            exact_action.energy_molecular_total_au
                        ),
                        "energy_error_au": float(
                            own_action.energy_molecular_total_au
                            - exact_action.energy_molecular_total_au
                        ),
                        "mixed_density_relative_error": _relative_difference(
                            state.mixed_density,
                            exact_density @ np.asarray(exact_action.overlap),
                        ),
                        "real_space_density_relative_error": _relative_difference(
                            density_grid,
                            exact_grid_density,
                        ),
                        "matched_lower_relative_error": _relative_difference(
                            matched_action.lower_mechanical_matrix,
                            exact_action.lower_mechanical_matrix,
                        ),
                        "matched_density_relative_error": _relative_difference(
                            matched_grid,
                            exact_grid_density,
                        ),
                        "matched_source_mechanical_rate_au": float(
                            response.mechanical_energy_rate_au
                        ),
                        "exact_source_mechanical_rate_au": float(
                            exact_power.one_electron_matrix_rate_au
                            + exact_power.closure_fixed_density_rate_au
                        ),
                        "source_mechanical_rate_error_au": float(
                            response.mechanical_energy_rate_au
                            - exact_power.one_electron_matrix_rate_au
                            - exact_power.closure_fixed_density_rate_au
                        ),
                        "matched_power_au": float(power.source_power_au),
                        "exact_power_au": float(exact_power.source_power_au),
                        "power_error_au": float(
                            power.source_power_au - exact_power.source_power_au
                        ),
                        "metric_minimum_eigenvalue": (
                            model.spatial.one_electron.metric_minimum_eigenvalue
                        ),
                        "metric_condition_number": (
                            model.spatial.one_electron.metric_condition_number
                        ),
                        "density_zero_minimum": (
                            own_action.closure.density.density_zero_minimum
                        ),
                        "density_assembled_minimum": (
                            own_action.closure.density.density_assembled_minimum
                        ),
                        "orbital_residual": state.orbital_residual,
                        "density_fixed_point_residual": (
                            state.density_fixed_point_residual
                        ),
                        "commutator_residual": state.commutator_residual,
                        "double_counting_residual_au": (
                            state.double_counting_residual_au
                        ),
                        "iterations": len(state.iterations),
                    }
                    np.savez_compressed(
                        array_path,
                        coefficients=np.asarray(state.coefficients),
                        coefficient_density=np.asarray(state.coefficient_density),
                        mixed_density=np.asarray(state.mixed_density),
                        lower_matrix=np.asarray(own_action.lower_mechanical_matrix),
                        density_grid=density_grid,
                        matched_lower_matrix=np.asarray(
                            matched_action.lower_mechanical_matrix
                        ),
                        matched_density_grid=matched_grid,
                    )
                    _write_json(record_path, record)
                except Exception as exc:
                    _write_json(
                        record_path,
                        {
                            "schema": "aion.chapter13.nq7.stationary.v1",
                            "status": "failed",
                            "run_id": run_id,
                            "field_au": field_value,
                            "branch": branch.value,
                            "level": level.value,
                            "elapsed_seconds": perf_counter() - started,
                            "exception_type": type(exc).__name__,
                            "exception": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )


def _domain_reference(output: Path, basis: str) -> Any:
    path = output / "references" / f"h3plus_{basis}.reference.h5"
    if path.is_file():
        return load_reference_data(path)
    base = _main_reference().config
    timestamp = datetime.now(UTC).isoformat()
    config = replace(
        base,
        electronic_structure=ElectronicStructureConfig(
            basis=basis,
            functional=_FUNCTIONAL,
            xc_family=XCFamily.LDA,
            grid_level=3,
            density_fitting=True,
            auxiliary_basis=_AUXILIARY_BASIS,
            scf_energy_tolerance_au=1.0e-12,
            scf_max_iterations=120,
        ),
        backend=BackendConfig(),
        output=ReferenceOutputConfig(path),
        metadata=MetadataConfig(
            label=f"chapter13-nq7-domain-h3plus-{basis}",
            timestamp_utc=timestamp,
            host=platform.node(),
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = prepare_pyscf_reference(config)
    reference.save()
    return reference


def _domain_phase(output: Path) -> None:
    for basis in _DOMAIN_BASES:
        reference = _domain_reference(output, basis)
        quadrature = prepare_ao_quadrature(
            reference,
            BackendConfig(),
            grid_policy=AOGridPolicy.qualification(3),
            block_size=1024,
        )
        exact_factory = prepare_exact_wilson_stationary_factory(
            quadrature,
            auxiliary_basis=_AUXILIARY_BASIS,
            functional=_FUNCTIONAL,
        )
        reduced_factory = prepare_reduced_wilson_factory(exact_factory)
        density = np.asarray(reference.ground_state.density, dtype=np.complex128)
        for field_value in _DOMAIN_FIELDS:
            gauge = AffineMagneticGauge(
                UniformMagneticField((0.0, 0.0, field_value)),
                origin_au=_GAUGE_ORIGIN,
            )
            exact_metric = np.asarray(exact_factory.model(
                gauge,
                WilsonStationaryBranch.HARTREE,
            ).overlap)
            exact_eigenvalues = np.linalg.eigvalsh(exact_metric)
            for level in _LEVELS:
                run_id = f"{basis}_{_field_tag(field_value)}_{level.value}"
                record_path, array_path = _checkpoint(output, "domain", run_id)
                if record_path.is_file() and array_path.is_file():
                    continue
                started = perf_counter()
                try:
                    hartree_model = reduced_factory.model(
                        gauge,
                        level,
                        WilsonStationaryBranch.HARTREE,
                    )
                    hartree = hartree_model.evaluate(density)
                    lda_status = "complete"
                    lda_exception = None
                    try:
                        reduced_factory.model(
                            gauge,
                            level,
                            WilsonStationaryBranch.KOHN_SHAM_LDA,
                        ).evaluate(density)
                    except Exception as exc:
                        lda_status = "failed"
                        lda_exception = f"{type(exc).__name__}: {exc}"
                    one = hartree_model.spatial.one_electron
                    record = {
                        "schema": "aion.chapter13.nq7.domain.v1",
                        "status": "complete",
                        "run_id": run_id,
                        "basis": basis,
                        "field_au": field_value,
                        "level": level.value,
                        "elapsed_seconds": perf_counter() - started,
                        "exact_metric_minimum_eigenvalue": float(
                            exact_eigenvalues[0]
                        ),
                        "exact_metric_condition_number": float(
                            exact_eigenvalues[-1] / exact_eigenvalues[0]
                        ),
                        "metric_minimum_eigenvalue": (
                            one.metric_minimum_eigenvalue
                        ),
                        "metric_condition_number": one.metric_condition_number,
                        "metric_positive": one.metric_positive,
                        "density_zero_minimum": (
                            hartree.closure.density.density_zero_minimum
                        ),
                        "density_assembled_minimum": (
                            hartree.closure.density.density_assembled_minimum
                        ),
                        "electron_count_zero": float(
                            hartree.closure.density.electron_count_zero
                        ),
                        "electron_count_first": float(
                            hartree.closure.density.electron_count_first
                        ),
                        "lda_domain_status": lda_status,
                        "lda_domain_exception": lda_exception,
                    }
                    np.savez_compressed(
                        array_path,
                        metric=np.asarray(one.metric),
                        density_zero=np.asarray(hartree.closure.density.density_zero),
                        density_first=np.asarray(hartree.closure.density.density_first),
                    )
                    _write_json(record_path, record)
                except Exception as exc:
                    _write_json(
                        record_path,
                        {
                            "schema": "aion.chapter13.nq7.domain.v1",
                            "status": "failed",
                            "run_id": run_id,
                            "basis": basis,
                            "field_au": field_value,
                            "level": level.value,
                            "elapsed_seconds": perf_counter() - started,
                            "exception_type": type(exc).__name__,
                            "exception": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )


def _pulse_shape(time_au: float) -> float:
    if time_au <= 0.0 or time_au >= _FINAL_TIME:
        return 0.0
    return math.sin(math.pi * time_au / _FINAL_TIME) ** 2


def _pulse_shape_rate(time_au: float) -> float:
    if time_au <= 0.0 or time_au >= _FINAL_TIME:
        return 0.0
    return (math.pi / _FINAL_TIME) * math.sin(
        2.0 * math.pi * time_au / _FINAL_TIME
    )


def _source(case: str, time_au: float) -> UniformMagneticSourceSample:
    shape = _pulse_shape(time_au)
    if case == "electric_field_free":
        field = 0.0
        field_rate = 0.0
        electric = _ELECTRIC_AMPLITUDE * shape
    elif case == "electric_static_b":
        field = _STATIC_FIELD
        field_rate = 0.0
        electric = _ELECTRIC_AMPLITUDE * shape
    elif case == "magnetic_induction":
        field = _STATIC_FIELD + _MAGNETIC_AMPLITUDE * shape
        field_rate = _MAGNETIC_AMPLITUDE * _pulse_shape_rate(time_au)
        electric = 0.0
    else:
        raise ValueError(case)
    return UniformMagneticSourceSample(
        time_au=time_au,
        field=UniformMagneticField((0.0, 0.0, field)),
        magnetic_field_dot_au=(0.0, 0.0, field_rate),
        electric_field_origin_au=(electric, 0.0, 0.0),
        origin_au=_GAUGE_ORIGIN,
    )


class _ReducedDynamicCache:
    def __init__(self, factory: Any, level: ReducedWilsonLevel, branch: WilsonStationaryBranch):
        self.factory = factory
        self.level = level
        self.branch = branch
        self.samples: dict[tuple[Any, ...], Any] = {}

    def at(self, case: str, time_au: float) -> Any:
        source = _source(case, time_au)
        key = (
            case,
            time_au,
            source.field.magnetic_field_au,
            source.magnetic_field_dot_au,
            source.electric_field_origin_au,
        )
        if key not in self.samples:
            self.samples[key] = self.factory.spatial_action(
                source.gauge,
                self.level,
            ).sample(source, self.branch)
        return self.samples[key]


def _simpson(values: np.ndarray, interval: float) -> float:
    intervals = len(values) - 1
    if intervals <= 0 or intervals % 2:
        raise ValueError("Simpson integration requires a positive even interval count")
    return float(
        (interval / 3.0)
        * (
            values[0]
            + values[-1]
            + 4.0 * np.sum(values[1:-1:2])
            + 2.0 * np.sum(values[2:-1:2])
        )
    )


def _exact_dynamic_key(case: str, branch: WilsonStationaryBranch, suffix: str) -> str:
    run_id = f"{case}_{branch.value}_n32_tol1e-12_timestep"
    return f"{run_id}_{suffix}"


def _dynamic_initial_checkpoint(
    output: Path,
    case: str,
    branch: WilsonStationaryBranch,
    level: ReducedWilsonLevel,
) -> Path:
    field = 0.0 if case == "electric_field_free" else _STATIC_FIELD
    run_id = f"{_field_tag(field)}_{branch.value}_{level.value}"
    record_path, array_path = _checkpoint(output, "stationary", run_id)
    if not record_path.is_file() or not array_path.is_file():
        raise RuntimeError(f"missing stationary prerequisite {run_id}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["status"] != "complete":
        raise RuntimeError(f"failed stationary prerequisite {run_id}")
    return array_path


def _run_dynamic_trajectory(
    output: Path,
    reduced_factory: Any,
    exact_arrays: Any,
    case: str,
    branch: WilsonStationaryBranch,
    level: ReducedWilsonLevel,
    intervals: int,
) -> None:
    run_id = f"{case}_{branch.value}_{level.value}_n{intervals}"
    record_path, array_path = _checkpoint(output, "dynamics", run_id)
    if record_path.is_file() and array_path.is_file():
        return
    started = perf_counter()
    try:
        initial = np.load(
            _dynamic_initial_checkpoint(output, case, branch, level)
        )["coefficient_density"]
        cache = _ReducedDynamicCache(reduced_factory, level, branch)
        step = _FINAL_TIME / intervals
        propagation = propagate_nonlinear_contravariant_density(
            initial,
            initial_time_au=0.0,
            interval_au=step,
            intervals=intervals,
            metric_provider=lambda time: cache.at(case, time).overlap,
            eom_provider=lambda time, density: cache.at(
                case,
                time,
            ).dynamic_evaluation(density).triple,
            backend=reduced_factory.exact_factory.quadrature.backend,
            policy=_NONLINEAR_POLICY,
        )
        times = np.asarray(propagation.times_au)
        energies: list[float] = []
        powers: list[float] = []
        density_zero_minima: list[float] = []
        density_assembled_minima: list[float] = []
        metric_minima: list[float] = []
        metric_conditions: list[float] = []
        for time_au, density in zip(
            propagation.times_au,
            propagation.contravariant_densities,
            strict=True,
        ):
            model = cache.at(case, time_au)
            action = model.evaluate(density)
            power = model.power(density)
            energies.append(float(action.energy_molecular_total_au))
            powers.append(float(power.source_power_au))
            density_zero_minima.append(action.closure.density.density_zero_minimum)
            density_assembled_minima.append(
                action.closure.density.density_assembled_minimum
            )
            metric_minima.append(
                model.spatial.one_electron.metric_minimum_eigenvalue
            )
            metric_conditions.append(
                model.spatial.one_electron.metric_condition_number
            )
        energy_values = np.asarray(energies)
        power_values = np.asarray(powers)
        energy_change = float(energy_values[-1] - energy_values[0])
        integrated_power = _simpson(power_values, step)
        final_mixed = np.asarray(propagation.mixed_densities[-1])
        exact_final = np.asarray(
            exact_arrays[_exact_dynamic_key(case, branch, "final_mixed_density")]
        )
        exact_energies = np.asarray(
            exact_arrays[_exact_dynamic_key(case, branch, "energies_au")]
        )
        exact_powers = np.asarray(
            exact_arrays[_exact_dynamic_key(case, branch, "source_powers_au")]
        )
        record = {
            "schema": "aion.chapter13.nq7.dynamics.v1",
            "status": "complete",
            "run_id": run_id,
            "case": case,
            "branch": branch.value,
            "level": level.value,
            "intervals": intervals,
            "interval_au": step,
            "elapsed_seconds": perf_counter() - started,
            "energy_change_au": energy_change,
            "integrated_source_power_au": integrated_power,
            "power_balance_residual_au": abs(integrated_power - energy_change),
            "final_mixed_density_relative_error": _relative_difference(
                final_mixed,
                exact_final,
            ),
            "energy_trajectory_relative_error": _relative_difference(
                energy_values,
                exact_energies,
            )
            if intervals == 32
            else None,
            "power_trajectory_relative_error": _relative_difference(
                power_values,
                exact_powers,
            )
            if intervals == 32
            else None,
            "metric_minimum_eigenvalue_min": min(metric_minima),
            "metric_condition_number_max": max(metric_conditions),
            "density_zero_minimum": min(density_zero_minima),
            "density_assembled_minimum": min(density_assembled_minima),
            "cross_metric_residual_max": max(
                item.cross_metric_residual for item in propagation.diagnostics
            ),
            "trace_drift_max": max(
                item.trace_drift for item in propagation.diagnostics
            ),
            "occupation_spectrum_drift_max": max(
                item.occupation_spectrum_drift for item in propagation.diagnostics
            ),
            "nonlinear_residual_max": max(
                item.nonlinear_residual for item in propagation.diagnostics
            ),
            "nonlinear_iterations_max": max(
                item.nonlinear_iterations for item in propagation.diagnostics
            ),
            "metric_correction_applied": propagation.metric_correction_applied,
            "density_update": propagation.density_update,
        }
        np.savez_compressed(
            array_path,
            times_au=times,
            final_mixed_density=final_mixed,
            energies_au=energy_values,
            source_powers_au=power_values,
            density_zero_minima=np.asarray(density_zero_minima),
            density_assembled_minima=np.asarray(density_assembled_minima),
            metric_minimum_eigenvalues=np.asarray(metric_minima),
            metric_condition_numbers=np.asarray(metric_conditions),
        )
        _write_json(record_path, record)
    except Exception as exc:
        _write_json(
            record_path,
            {
                "schema": "aion.chapter13.nq7.dynamics.v1",
                "status": "failed",
                "run_id": run_id,
                "case": case,
                "branch": branch.value,
                "level": level.value,
                "intervals": intervals,
                "elapsed_seconds": perf_counter() - started,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


def _dynamics_phase(output: Path) -> None:
    reference = _main_reference()
    _, reduced_factory = _factories(reference)
    exact_arrays = np.load(_NQ6_ROOT / "trajectories.npz")
    for case in _DYNAMIC_CASES:
        for branch in _BRANCHES:
            for level in _LEVELS:
                intervals_to_run = (16, 32) if level.retains_first_magnetic_order else (32,)
                for intervals in intervals_to_run:
                    _run_dynamic_trajectory(
                        output,
                        reduced_factory,
                        exact_arrays,
                        case,
                        branch,
                        level,
                        intervals,
                    )


def _synthesis_phase(output: Path, repo: Path) -> None:
    identity = _campaign_identity(output, repo)
    parents = {
        "nq4": _authenticated_parent(_NQ4_ROOT),
        "nq6": _authenticated_parent(_NQ6_ROOT),
    }
    checkpoint_root = output / "checkpoints"
    records: dict[str, list[dict[str, Any]]] = {}
    for phase in ("stationary", "domain", "dynamics"):
        records[phase] = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((checkpoint_root / phase).glob("*.json"))
        ]
    expected = {
        "stationary": len(_FIELDS) * len(_BRANCHES) * len(_LEVELS),
        "domain": len(_DOMAIN_BASES) * len(_DOMAIN_FIELDS) * len(_LEVELS),
        "dynamics": len(_DYNAMIC_CASES)
        * len(_BRANCHES)
        * (2 + 2 * 2),
    }
    counts = {name: len(value) for name, value in records.items()}
    if counts != expected:
        raise RuntimeError(f"incomplete checkpoint matrix: {counts} != {expected}")
    status_counts = {
        phase: {
            status: sum(item["status"] == status for item in values)
            for status in ("complete", "failed")
        }
        for phase, values in records.items()
    }
    result = {
        "schema": "aion.chapter13.nq7.result.v1",
        "status": "complete_with_visible_failures"
        if any(item["failed"] for item in status_counts.values())
        else "complete",
        "implementation_commit": identity["implementation_commit"],
        "implementation_source_sha256": identity["source_sha256"],
        "accepted_inputs": parents,
        "profile": {
            "system": "equilateral H3+; charge +1; two electrons",
            "main_basis": "cc-pvdz",
            "domain_bases": list(_DOMAIN_BASES),
            "functional": _FUNCTIONAL,
            "auxiliary_basis": _AUXILIARY_BASIS,
            "main_grid_level": 4,
            "domain_grid_level": 3,
            "fields_au": list(_FIELDS),
            "domain_fields_au": list(_DOMAIN_FIELDS),
            "levels": [item.value for item in _LEVELS],
            "branches": [item.value for item in _BRANCHES],
            "dynamic_cases": list(_DYNAMIC_CASES),
            "final_time_au": _FINAL_TIME,
            "dynamic_intervals": [16, 32],
            "scf_policy": asdict(_SCF_POLICY),
            "nonlinear_policy": asdict(_NONLINEAR_POLICY),
        },
        "checkpoint_counts": counts,
        "status_counts": status_counts,
        "records": records,
    }
    result_path = output / "result.json"
    _write_json(result_path, result)
    artifacts = {
        str(path.relative_to(output)): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name not in {"provenance.json", "completed.json"}
    }
    provenance = {
        "schema": "aion.chapter13.nq7.provenance.v1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "repository": str(repo),
        "git_head": _git(repo, "rev-parse", "HEAD").strip(),
        "git_status_porcelain": _git(repo, "status", "--porcelain").splitlines(),
        "environment": DependencyVersions.current().as_mapping(),
        "thread_limits": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "command": " ".join(sys.argv),
        "artifacts_sha256": artifacts,
    }
    provenance_path = output / "provenance.json"
    _write_json(provenance_path, provenance)
    _write_json(
        output / "completed.json",
        {
            "schema": "aion.chapter13.nq7.completed.v1",
            "status": "complete",
            "result_sha256": _sha256(result_path),
            "provenance_sha256": _sha256(provenance_path),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("stationary", "domain", "dynamics", "synthesize"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    _campaign_identity(output, repo)
    if args.phase == "stationary":
        _stationary_phase(output)
    elif args.phase == "domain":
        _domain_phase(output)
    elif args.phase == "dynamics":
        _dynamics_phase(output)
    else:
        _synthesis_phase(output, repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
