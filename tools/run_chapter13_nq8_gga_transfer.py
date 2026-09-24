#!/usr/bin/env python3
"""Run the authenticated Chapter 13 NQ8 pure-GGA transfer campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aion.config import (
    AtomConfig,
    BackendConfig,
    ElectronicStructureConfig,
    MetadataConfig,
    MoleculeConfig,
    ReferenceConfig,
    ReferenceOutputConfig,
    XCFamily,
)
from aion.electromagnetism import (
    GaussianScalarGaugeVariation,
    GaussianVectorPotentialVariation,
    PerturbedVectorPotential,
    UniformMagneticField,
    UniformMagneticSourceSample,
)
from aion.electronic_structure import (
    AOGridPolicy,
    DependencyVersions,
    StationarySCFPolicy,
    WilsonStationaryBranch,
    evaluate_exact_static_wilson_grid_one_electron_action,
    evaluate_exact_wilson_charge,
    evaluate_exact_wilson_power,
    evaluate_nonlinear_weak_continuity,
    evaluate_nonlinear_weak_current_pairing,
    evaluate_static_nonlinear_wilson_grid_action,
    prepare_ao_quadrature,
    prepare_exact_wilson_dynamic_sample,
    prepare_exact_wilson_stationary_factory,
    prepare_pyscf_reference,
    reconstruct_mean_field,
)
from aion.formulations import EOMTriple
from aion.propagation import (
    NonlinearGaussMagnusPolicy,
    propagate_nonlinear_contravariant_density,
)

_FUNCTIONAL = "pbe"
_REALIZATION = "quadrature--Wilson GGA"
_BASIS = "cc-pvdz"
_AUXILIARY_BASIS = "weigend"
_GRID_LEVEL = 4
_BLOCK_SIZE = 1024
_FIELD_AU = 0.03
_FIELD_RATE_AU = 0.001
_ELECTRIC_FIELD_AU = (0.002, -0.001, 0.0005)
_GAUGE_ORIGIN = (0.17, -0.31, 0.23)
_FINAL_TIME_AU = 0.2
_PROPAGATION_INTERVALS = (4, 8)
_SCF_POLICY = StationarySCFPolicy(
    maximum_iterations=160,
    density_tolerance=2.0e-10,
    orbital_tolerance=2.0e-10,
    energy_tolerance_au=2.0e-11,
)
_NONLINEAR_POLICY = NonlinearGaussMagnusPolicy(
    tolerance=1.0e-10,
    maximum_iterations=60,
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
    return completed.stdout


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _relative(left: object, right: object) -> float:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return float(
        np.linalg.norm(left_array - right_array)
        / max(1.0, float(np.linalg.norm(right_array)))
    )


def _float(value: object) -> float:
    return float(np.asarray(value).real)


def _simpson(values: list[float], step: float) -> float:
    intervals = len(values) - 1
    if intervals <= 0 or intervals % 2:
        raise ValueError("composite Simpson integration requires a positive even count")
    return float(
        (step / 3.0)
        * (
            values[0]
            + values[-1]
            + 4.0 * sum(values[1:-1:2])
            + 2.0 * sum(values[2:-1:2])
        )
    )


def _config(system: str, artifact: Path, timestamp: str) -> ReferenceConfig:
    if system == "h3plus":
        atoms = (
            AtomConfig("H", (-0.7, -0.404145188432738, 0.0)),
            AtomConfig("H", (0.7, -0.404145188432738, 0.0)),
            AtomConfig("H", (0.0, 0.808290376865476, 0.0)),
        )
        charge = 1
    elif system == "co":
        atoms = (
            AtomConfig("C", (0.0, 0.0, -1.066)),
            AtomConfig("O", (0.0, 0.0, 1.066)),
        )
        charge = 0
    else:
        raise ValueError(f"unsupported system {system!r}")
    return ReferenceConfig(
        molecule=MoleculeConfig(atoms=atoms, charge=charge, spin=0),
        electronic_structure=ElectronicStructureConfig(
            basis=_BASIS,
            functional=_FUNCTIONAL,
            xc_family=XCFamily.GGA,
            grid_level=_GRID_LEVEL,
            density_fitting=True,
            auxiliary_basis=_AUXILIARY_BASIS,
            scf_energy_tolerance_au=1.0e-12,
            scf_max_iterations=160,
        ),
        backend=BackendConfig(),
        output=ReferenceOutputConfig(artifact),
        metadata=MetadataConfig(
            label=f"chapter13-nq8-{system}-pbe-transfer",
            timestamp_utc=timestamp,
            host=platform.node(),
        ),
    )


def _source(time_au: float) -> UniformMagneticSourceSample:
    return UniformMagneticSourceSample(
        time_au=time_au,
        field=UniformMagneticField(
            (0.0, 0.0, _FIELD_AU + _FIELD_RATE_AU * time_au)
        ),
        magnetic_field_dot_au=(0.0, 0.0, _FIELD_RATE_AU),
        electric_field_origin_au=_ELECTRIC_FIELD_AU,
        origin_au=_GAUGE_ORIGIN,
    )


def _same_grid_pyscf(reference: Any, quadrature: Any, result: Any) -> dict[str, float]:
    from pyscf import dft

    molecule = reconstruct_mean_field(reference, BackendConfig()).mol
    grids = dft.gen_grid.Grids(molecule)
    grids.coords = np.array(quadrature.grid.coordinates_au, copy=True)
    grids.weights = np.array(quadrature.grid.weights_au, copy=True)
    grids.non0tab = grids.make_mask(molecule, grids.coords)
    electron_count, energy, lower = dft.numint.NumInt().nr_rks(
        molecule,
        grids,
        _FUNCTIONAL,
        np.asarray(reference.ground_state.density).real,
    )
    return {
        "electron_count_absolute_residual": abs(
            _float(result.electron_count_grid) - float(electron_count)
        ),
        "energy_absolute_residual_au": abs(_float(result.energy) - float(energy)),
        "lower_relative_residual": _relative(result.lower_xc_matrix, lower),
    }


def _atom_pair_norms(matrix: object, ao_to_atom: np.ndarray) -> dict[str, float]:
    values = np.asarray(matrix)
    atoms = sorted(set(int(value) for value in ao_to_atom))
    result: dict[str, float] = {}
    for left in atoms:
        left_indices = np.flatnonzero(ao_to_atom == left)
        for right in atoms:
            right_indices = np.flatnonzero(ao_to_atom == right)
            block = values[np.ix_(left_indices, right_indices)]
            result[f"atom_{left}_atom_{right}"] = float(np.linalg.norm(block))
    return result


def _static_source_check(model: Any, sample: Any, density: np.ndarray) -> dict[str, Any]:
    grid = evaluate_exact_static_wilson_grid_one_electron_action(
        model.quadrature,
        model.gauge,
    )
    base = EOMTriple(grid.overlap, grid.mechanical, np.zeros_like(grid.overlap))
    rng = np.random.default_rng(1308)
    velocity = 0.03 * (
        rng.normal(size=density.shape) + 1j * rng.normal(size=density.shape)
    )
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
        one_electron_triple=base,
    )
    finite: list[dict[str, float]] = []
    for step in (1.0e-3, 3.0e-4, 1.0e-4):
        values = []
        for sign in (1.0, -1.0):
            perturbed = PerturbedVectorPotential(
                model.gauge,
                variation,
                sign * step,
            )
            action = evaluate_static_nonlinear_wilson_grid_action(
                model,
                density,
                velocity,
                perturbed,
            )
            values.append(_float(action.electronic_action_value))
        derivative = (values[0] - values[1]) / (2.0 * step)
        finite.append(
            {
                "step": step,
                "finite_difference": derivative,
                "absolute_residual": abs(derivative - _float(analytic.total_pairing)),
            }
        )
    assert analytic.exchange_correlation is not None
    return {
        "analytic_total_pairing": _float(analytic.total_pairing),
        "xc_source_energy_direction": _float(
            analytic.exchange_correlation.source_energy_direction
        ),
        "finite_differences": finite,
    }


def _propagate(factory: Any, initial_density: np.ndarray, intervals: int) -> dict[str, Any]:
    backend = factory.quadrature.backend
    cache: dict[float, Any] = {}

    def dynamic(time_au: float) -> Any:
        key = float(time_au)
        if key not in cache:
            cache[key] = prepare_exact_wilson_dynamic_sample(
                factory,
                _source(key),
                WilsonStationaryBranch.KOHN_SHAM_GGA,
            )
        return cache[key]

    step = _FINAL_TIME_AU / intervals
    trajectory = propagate_nonlinear_contravariant_density(
        backend.asarray(initial_density),
        initial_time_au=0.0,
        interval_au=step,
        intervals=intervals,
        metric_provider=lambda time: dynamic(time).one_electron.metric,
        eom_provider=lambda time, value: dynamic(time).evaluate(value).triple,
        backend=backend,
        policy=_NONLINEAR_POLICY,
        hbar=factory.hbar,
    )
    energies: list[float] = []
    powers: list[float] = []
    power_residuals: list[float] = []
    particles: list[float] = []
    for time_au, density in zip(
        trajectory.times_au,
        trajectory.contravariant_densities,
        strict=True,
    ):
        sample = dynamic(time_au)
        evaluation = sample.evaluate(density)
        power = evaluate_exact_wilson_power(evaluation, density)
        charge = evaluate_exact_wilson_charge(sample.model, density)
        energies.append(_float(evaluation.action.energy_molecular_total_au))
        powers.append(_float(power.source_power_au))
        power_residuals.append(abs(_float(power.power_identity_residual_au)))
        particles.append(_float(charge.metric_particle_number))
    work = _simpson(powers, step)
    return {
        "intervals": intervals,
        "step_au": step,
        "times_au": np.asarray(trajectory.times_au),
        "final_density": np.asarray(trajectory.contravariant_densities[-1]),
        "final_mixed_density": np.asarray(trajectory.mixed_densities[-1]),
        "energies_au": np.asarray(energies),
        "source_powers_au": np.asarray(powers),
        "maximum_power_identity_residual_au": max(power_residuals),
        "maximum_particle_number_drift": max(
            abs(value - particles[0]) for value in particles
        ),
        "maximum_hermiticity_residual": max(
            item.contravariant_hermiticity_residual
            for item in trajectory.diagnostics
        ),
        "maximum_nonlinear_residual": max(
            item.nonlinear_residual for item in trajectory.diagnostics
        ),
        "energy_change_au": energies[-1] - energies[0],
        "integrated_source_power_au": work,
        "work_energy_residual_au": energies[-1] - energies[0] - work,
        "metric_correction_applied": trajectory.metric_correction_applied,
    }


def _run_system(output: Path, system: str, timestamp: str) -> dict[str, Any]:
    checkpoint = output / "checkpoints" / f"{system}.json"
    arrays_path = output / "checkpoints" / f"{system}.npz"
    if checkpoint.is_file() and arrays_path.is_file():
        record = json.loads(checkpoint.read_text(encoding="utf-8"))
        if record.get("status") == "complete":
            print(f"reusing completed {system} checkpoint", flush=True)
            return record

    print(f"starting NQ8 transfer system {system}", flush=True)
    started = perf_counter()
    reference_path = output / f"{system}.reference.h5"
    reference = prepare_pyscf_reference(_config(system, reference_path, timestamp))
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(_GRID_LEVEL),
        block_size=_BLOCK_SIZE,
    )
    factory = prepare_exact_wilson_stationary_factory(
        quadrature,
        auxiliary_basis=_AUXILIARY_BASIS,
        functional=_FUNCTIONAL,
    )
    assert factory.gga_evaluator is not None
    zero = UniformMagneticSourceSample(
        time_au=0.0,
        field=UniformMagneticField((0.0, 0.0, 0.0)),
        origin_au=_GAUGE_ORIGIN,
    )
    zero_result = factory.gga_evaluator.evaluate(
        reference.ground_state.density,
        zero.gauge,
    )
    same_grid = _same_grid_pyscf(reference, quadrature, zero_result)
    zero_model = factory.model(zero.gauge, WilsonStationaryBranch.KOHN_SHAM_GGA)
    zero_state = zero_model.solve(policy=_SCF_POLICY)

    static_source = UniformMagneticSourceSample(
        time_au=0.0,
        field=UniformMagneticField((0.0, 0.0, _FIELD_AU)),
        origin_au=_GAUGE_ORIGIN,
    )
    static_dynamic = prepare_exact_wilson_dynamic_sample(
        factory,
        static_source,
        WilsonStationaryBranch.KOHN_SHAM_GGA,
    )
    field_state = static_dynamic.model.solve(
        policy=_SCF_POLICY,
        initial_coefficients=zero_state.coefficients,
    )
    density = np.asarray(field_state.coefficient_density)
    field_action = field_state.action
    assert field_action.exchange_correlation is not None
    source_check = _static_source_check(
        static_dynamic.model,
        static_dynamic.one_electron,
        density,
    )

    dynamic = prepare_exact_wilson_dynamic_sample(
        factory,
        _source(0.0),
        WilsonStationaryBranch.KOHN_SHAM_GGA,
    )
    evaluation = dynamic.evaluate(density)
    power = evaluate_exact_wilson_power(evaluation, density)
    continuity = evaluate_nonlinear_weak_continuity(
        dynamic.model,
        dynamic.one_electron,
        density,
        GaussianScalarGaugeVariation(
            amplitude=0.37,
            center_au=(0.11, -0.17, 0.23),
            exponent_au_inverse2=0.41,
        ),
    )

    propagations = {
        intervals: _propagate(factory, density, intervals)
        for intervals in _PROPAGATION_INTERVALS
    }
    coarse = propagations[_PROPAGATION_INTERVALS[0]]
    fine = propagations[_PROPAGATION_INTERVALS[1]]
    topology = np.asarray(reference.anchor_topology.ao_to_atom)
    record = {
        "schema": "aion.chapter13.nq8.system.v1",
        "status": "complete",
        "system": system,
        "basis": _BASIS,
        "functional": _FUNCTIONAL,
        "functional_family": "GGA",
        "realization": _REALIZATION,
        "auxiliary_basis": _AUXILIARY_BASIS,
        "grid_level": _GRID_LEVEL,
        "grid_points": quadrature.grid.npoints,
        "nao": reference.core_operators.nao,
        "electrons": round(float(np.sum(reference.ground_state.occupations))),
        "reference_fingerprint_sha256": reference.fingerprint_sha256,
        "grid_fingerprint_sha256": quadrature.grid.fingerprint_sha256,
        "same_grid_pyscf": same_grid,
        "zero_field": {
            "energy_molecular_total_au": _float(
                zero_state.action.energy_molecular_total_au
            ),
            "pyscf_reference_energy_au": reference.ground_state.energy_total_au,
            "energy_absolute_residual_au": abs(
                _float(zero_state.action.energy_molecular_total_au)
                - reference.ground_state.energy_total_au
            ),
            "orbital_residual": zero_state.orbital_residual,
            "density_fixed_point_residual": zero_state.density_fixed_point_residual,
            "double_counting_residual_au": zero_state.double_counting_residual_au,
        },
        "static_field": {
            "field_au": _FIELD_AU,
            "energy_molecular_total_au": _float(
                field_state.action.energy_molecular_total_au
            ),
            "orbital_residual": field_state.orbital_residual,
            "density_fixed_point_residual": field_state.density_fixed_point_residual,
            "metric_minimum_eigenvalue": field_state.metric_minimum_eigenvalue,
            "metric_condition_number": field_state.metric_condition_number,
            "density_minimum": field_action.exchange_correlation.density_real_minimum,
            "density_maximum": float(
                np.max(np.asarray(field_action.exchange_correlation.density))
            ),
            "density_gradient_maximum": float(
                np.max(
                    np.linalg.norm(
                        np.asarray(field_action.exchange_correlation.density_gradient),
                        axis=0,
                    )
                )
            ),
            "xc_atom_pair_block_frobenius": _atom_pair_norms(
                field_action.exchange_correlation.lower_xc_matrix,
                topology,
            ),
        },
        "source_derivative": source_check,
        "instantaneous_power": {
            "source_power_au": _float(power.source_power_au),
            "matrix_rate_au": _float(power.matrix_mechanical_energy_rate_au),
            "absolute_residual_au": abs(_float(power.power_identity_residual_au)),
        },
        "instantaneous_continuity": {
            "finite_region_residual": abs(_float(continuity.finite_region_residual)),
            "global_charge_residual": abs(_float(continuity.global_charge_residual)),
        },
        "propagation": {
            str(intervals): {
                key: value
                for key, value in propagation.items()
                if not isinstance(value, np.ndarray)
            }
            for intervals, propagation in propagations.items()
        },
        "propagation_final_mixed_density_relative_difference": _relative(
            fine["final_mixed_density"], coarse["final_mixed_density"]
        ),
        "elapsed_seconds": perf_counter() - started,
    }
    arrays: dict[str, np.ndarray] = {
        "zero_coefficients": np.asarray(zero_state.coefficients),
        "zero_density": np.asarray(zero_state.coefficient_density),
        "field_coefficients": np.asarray(field_state.coefficients),
        "field_density": density,
        "field_lower_matrix": np.asarray(field_action.lower_mechanical_matrix),
        "field_xc_density": np.asarray(field_action.exchange_correlation.density),
        "field_xc_density_gradient": np.asarray(
            field_action.exchange_correlation.density_gradient
        ),
    }
    for intervals, propagation in propagations.items():
        for name in (
            "times_au",
            "final_density",
            "final_mixed_density",
            "energies_au",
            "source_powers_au",
        ):
            arrays[f"n{intervals}_{name}"] = np.asarray(propagation[name])
    np.savez_compressed(arrays_path, **arrays)  # type: ignore[arg-type]
    _write_json(checkpoint, record)
    print(
        f"completed {system} in {record['elapsed_seconds']:.1f} s; "
        f"power residual {record['instantaneous_power']['absolute_residual_au']:.3e}",
        flush=True,
    )
    return record


def _campaign_identity(output: Path, repo: Path) -> dict[str, Any]:
    path = output / "campaign_identity.json"
    source_paths = (
        repo / "tools/run_chapter13_nq8_gga_transfer.py",
        repo / "src/aion/electronic_structure/wilson_gga.py",
        repo / "src/aion/electronic_structure/wilson_stationary.py",
        repo / "src/aion/electronic_structure/wilson_sources.py",
        repo / "src/aion/electronic_structure/wilson_dynamics.py",
        repo / "src/aion/propagation/tensorial.py",
    )
    current = {
        "schema": "aion.chapter13.nq8.identity.v1",
        "implementation_commit": str(_git(repo, "rev-parse", "HEAD")).strip(),
        "source_sha256": {
            str(item.relative_to(repo)): _sha256(item) for item in source_paths
        },
    }
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != current:
            raise RuntimeError(
                "campaign implementation identity changed; use a new output directory"
            )
        return stored
    _write_json(path, current)
    return current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    identity = _campaign_identity(output, repo)
    timestamp = datetime.now(UTC).isoformat()
    records: list[dict[str, Any]] = []
    try:
        for system in ("h3plus", "co"):
            records.append(_run_system(output, system, timestamp))
    except Exception as exc:
        failure = {
            "schema": "aion.chapter13.nq8.failure.v1",
            "status": "failed_visible",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(output / "failure.json", failure)
        raise

    result = {
        "schema": "aion.chapter13.nq8.result.v1",
        "status": "executed_unreviewed",
        "realization": {
            "functional": _FUNCTIONAL,
            "family": "pure GGA",
            "label": _REALIZATION,
            "systems": ["H3+", "CO"],
            "basis": _BASIS,
            "auxiliary_basis": _AUXILIARY_BASIS,
            "grid_level": _GRID_LEVEL,
            "field_au": _FIELD_AU,
            "field_rate_au": _FIELD_RATE_AU,
            "final_time_au": _FINAL_TIME_AU,
            "propagation_intervals": list(_PROPAGATION_INTERVALS),
            "propagator": (
                "self-consistent two-node fourth-order Gauss-Magnus with [2/2] "
                "Pade coefficient links and contravariant-density congruence"
            ),
            "metric_projection_or_correction": False,
        },
        "campaign_identity": identity,
        "records": records,
    }
    result_path = output / "result.json"
    _write_json(result_path, result)
    tracked_diff = _git(repo, "diff", "--binary", "HEAD", binary=True)
    assert isinstance(tracked_diff, bytes)
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    assert isinstance(status, str)
    artifact_paths = tuple(
        sorted(
            path
            for path in output.rglob("*")
            if path.is_file()
            and path.name not in {"provenance.json", "completed.json", "failure.json"}
        )
    )
    provenance = {
        "schema": "aion.chapter13.nq8.provenance.v1",
        "timestamp_utc": timestamp,
        "command": " ".join(
            shlex.quote(value) for value in (sys.executable, *sys.argv)
        ),
        "repository": str(repo),
        "git_head": str(_git(repo, "rev-parse", "HEAD")).strip(),
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
        "policy": {
            "stationary": asdict(_SCF_POLICY),
            "nonlinear": asdict(_NONLINEAR_POLICY),
        },
        "artifacts_sha256": {
            str(path.relative_to(output)): _sha256(path) for path in artifact_paths
        },
    }
    provenance_path = output / "provenance.json"
    _write_json(provenance_path, provenance)
    completed = {
        "schema": "aion.chapter13.nq8.completed.v1",
        "status": "executed_unreviewed",
        "result_sha256": _sha256(result_path),
        "provenance_sha256": _sha256(provenance_path),
    }
    _write_json(output / "completed.json", completed)
    failure = output / "failure.json"
    if failure.exists():
        failure.unlink()
    print(json.dumps({"output": str(output), **completed}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
