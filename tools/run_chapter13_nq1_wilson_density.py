#!/usr/bin/env python3
"""Execute the Chapter 13 NQ1 exact Wilson-density qualification."""

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

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
    dumps_config,
)
from aion.electromagnetism import (
    AffineGaugeDifferenceVariation,
    AffineMagneticGauge,
    MagneticGaugeKind,
    UniformMagneticField,
    affine_gauge_difference_potential,
    build_magnetic_pair_geometry,
)
from aion.electronic_structure import (
    AOGridPolicy,
    DependencyVersions,
    contract_wilson_density_block,
    evaluate_exact_static_magnetic_one_electron_direction,
    evaluate_exact_static_magnetic_one_electron_matrices,
    evaluate_exact_uniform_magnetic_wilson_density,
    evaluate_exact_uniform_magnetic_wilson_density_matter_direction,
    evaluate_exact_uniform_magnetic_wilson_density_source_direction,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
)

_FIELD = (0.013, -0.009, 0.017)
_FIELD_DIRECTION = (-0.4, 0.7, 0.2)
_GAUGE_ORIGIN = (0.17, -0.31, 0.23)
_LANDAU_ORIGIN = (-0.21, 0.37, -0.16)
_GRID_LEVELS = (1, 2, 3, 4, 5)
_MATTER_STEPS = (
    1.0e-1,
    3.0e-2,
    1.0e-2,
    3.0e-3,
    1.0e-3,
    3.0e-4,
    1.0e-4,
    3.0e-5,
    1.0e-5,
    3.0e-6,
    1.0e-6,
)
_SOURCE_STEPS = _MATTER_STEPS


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


def _relative(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / max(1.0, float(np.linalg.norm(reference))))


def _complex_record(value: object) -> dict[str, float]:
    scalar = complex(np.asarray(value).item())
    return {"real": float(scalar.real), "imaginary": float(scalar.imag)}


def _configuration(system: str, artifact: Path, timestamp: str) -> ReferenceConfig:
    if system == "h2":
        atoms = (
            AtomConfig("H", (0.0, 0.0, -0.7)),
            AtomConfig("H", (0.0, 0.0, 0.7)),
        )
    elif system == "lih":
        atoms = (
            AtomConfig("Li", (0.0, 0.0, -1.5)),
            AtomConfig("H", (0.0, 0.0, 1.5)),
        )
    else:
        raise ValueError(system)
    return ReferenceConfig(
        molecule=MoleculeConfig(atoms=atoms, charge=0, spin=0),
        electronic_structure=ElectronicStructureConfig(
            basis="sto-3g",
            functional="lda,vwn",
            xc_family=XCFamily.LDA,
            grid_level=3,
            density_fitting=True,
            auxiliary_basis="weigend",
            scf_energy_tolerance_au=1.0e-12,
            scf_max_iterations=100,
        ),
        backend=BackendConfig(),
        output=ReferenceOutputConfig(artifact),
        metadata=MetadataConfig(
            label=f"chapter13-nq1-{system}-lda-vwn",
            timestamp_utc=timestamp,
            host=platform.node(),
        ),
    )


def _coefficient_density(
    coefficients: np.ndarray,
    occupations: np.ndarray,
) -> np.ndarray:
    return cast(
        np.ndarray,
        np.einsum(
            "mi,i,ni->mn",
            coefficients,
            occupations,
            coefficients.conj(),
            optimize=True,
        ),
    )


def _density_with_line_perturbation(
    quadrature: Any,
    density: np.ndarray,
    gauge: AffineMagneticGauge,
    direction: Any,
    step: float,
) -> np.ndarray:
    backend = quadrature.backend
    xp = backend.namespace
    reference = quadrature.reference
    geometry = build_magnetic_pair_geometry(
        reference.core_operators.nuclei.coordinates_au,
        reference.anchor_topology.ao_to_atom,
        backend,
    )
    result = np.empty((quadrature.grid.npoints,), dtype=np.complex128)
    anchors = geometry.ao_anchor_coordinates_au
    for block in quadrature.blocks():
        base = gauge.anchor_to_point_line_integrals(
            anchors,
            block.coordinates_au,
            backend,
        )
        delta = direction.straight_line_integrals(
            anchors[None, :, :],
            block.coordinates_au[:, None, :],
            backend,
        )
        frame = xp.exp(-1j * (base + step * delta)) * block.values
        block_density = contract_wilson_density_block(frame, density, backend)
        result[block.start : block.stop] = backend.to_host(block_density)
    return result


def _coefficient_frame_residual(
    quadrature: Any,
    density: np.ndarray,
    gauge: AffineMagneticGauge,
    seed: int,
) -> dict[str, float]:
    backend = quadrature.backend
    xp = backend.namespace
    reference = quadrature.reference
    geometry = build_magnetic_pair_geometry(
        reference.core_operators.nuclei.coordinates_au,
        reference.anchor_topology.ao_to_atom,
        backend,
    )
    dimension = density.shape[0]
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    change = np.eye(dimension, dtype=np.complex128) + 0.08 * raw / np.linalg.norm(raw)
    inverse = np.linalg.inv(change)
    transformed_density = inverse @ density @ inverse.conj().T
    original = np.empty((quadrature.grid.npoints,), dtype=np.complex128)
    transformed = np.empty_like(original)
    for block in quadrature.blocks():
        line = gauge.anchor_to_point_line_integrals(
            geometry.ao_anchor_coordinates_au,
            block.coordinates_au,
            backend,
        )
        frame = xp.exp(-1j * line) * block.values
        original[block.start : block.stop] = backend.to_host(
            contract_wilson_density_block(frame, density, backend)
        )
        transformed[block.start : block.stop] = backend.to_host(
            contract_wilson_density_block(
                frame @ backend.asarray(change, dtype=xp.complex128),
                transformed_density,
                backend,
            )
        )
    return {
        "change_condition_number": float(np.linalg.cond(change)),
        "relative_l2_residual": _relative(transformed, original),
        "maximum_absolute_residual": float(np.max(np.abs(transformed - original))),
    }


def _matter_sequences(
    quadrature: Any,
    coefficients: np.ndarray,
    occupations: np.ndarray,
    gauge: AffineMagneticGauge,
    base_density_values: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=coefficients.shape)
    rows: list[dict[str, Any]] = []
    for label, direction in (
        ("real", raw.astype(np.complex128)),
        ("imaginary", 1j * raw),
    ):
        direction = direction / np.linalg.norm(direction)
        analytic = evaluate_exact_uniform_magnetic_wilson_density_matter_direction(
            quadrature,
            coefficients,
            occupations,
            direction,
            gauge,
        )
        analytic_density = quadrature.backend.to_host(analytic.density_direction)
        for step in _MATTER_STEPS:
            plus_coefficients = coefficients + step * direction
            minus_coefficients = coefficients - step * direction
            plus = evaluate_exact_uniform_magnetic_wilson_density(
                quadrature,
                _coefficient_density(plus_coefficients, occupations),
                gauge,
            )
            minus = evaluate_exact_uniform_magnetic_wilson_density(
                quadrature,
                _coefficient_density(minus_coefficients, occupations),
                gauge,
            )
            plus_density = quadrature.backend.to_host(plus.density_direct)
            minus_density = quadrature.backend.to_host(minus.density_direct)
            forward = (plus_density - base_density_values) / step
            central = (plus_density - minus_density) / (2.0 * step)
            rows.append(
                {
                    "kind": "matter",
                    "direction": label,
                    "step": step,
                    "forward_relative_l2_error": _relative(forward, analytic_density),
                    "central_relative_l2_error": _relative(central, analytic_density),
                    "central_maximum_absolute_error": float(
                        np.max(np.abs(central - analytic_density))
                    ),
                    "analytic_imaginary_max_abs": analytic.density_direction_imaginary_max_abs,
                    "analytic_particle_integral": _complex_record(
                        analytic.particle_number_direction_integral
                    ),
                    "analytic_particle_stable_metric": _complex_record(
                        analytic.particle_number_direction_metric_stable
                    ),
                }
            )
    return rows


def _source_sequences(
    quadrature: Any,
    density: np.ndarray,
    gauge: AffineMagneticGauge,
    directions: tuple[tuple[str, Any], ...],
    stable_metric_directions: dict[str, complex],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, direction in directions:
        analytic = evaluate_exact_uniform_magnetic_wilson_density_source_direction(
            quadrature,
            density,
            gauge,
            direction,
        )
        analytic_density = quadrature.backend.to_host(analytic.density_direction)
        for step in _SOURCE_STEPS:
            plus = _density_with_line_perturbation(
                quadrature,
                density,
                gauge,
                direction,
                step,
            )
            minus = _density_with_line_perturbation(
                quadrature,
                density,
                gauge,
                direction,
                -step,
            )
            central = (plus - minus) / (2.0 * step)
            rows.append(
                {
                    "kind": "source_fixed_coefficients",
                    "direction": label,
                    "step": step,
                    "central_relative_l2_error": _relative(central, analytic_density),
                    "central_maximum_absolute_error": float(
                        np.max(np.abs(central - analytic_density))
                    ),
                    "analytic_imaginary_max_abs": analytic.density_direction_imaginary_max_abs,
                    "analytic_particle_integral": _complex_record(
                        analytic.particle_number_direction_integral
                    ),
                    "analytic_particle_grid_metric": _complex_record(
                        analytic.particle_number_direction_metric_grid
                    ),
                    "analytic_particle_stable_metric": _complex_record(
                        stable_metric_directions[label]
                    ),
                }
            )
    return rows


def _system_result(system: str, output: Path, timestamp: str) -> dict[str, Any]:
    config = _configuration(system, output / f"{system}.reference.h5", timestamp)
    (output / f"{system}.reference.toml").write_text(
        dumps_config(config),
        encoding="utf-8",
    )
    reference = prepare_pyscf_reference(config)
    reference.save()
    density = reference.ground_state.density.astype(np.complex128)
    coefficients = reference.ground_state.coefficients.astype(np.complex128)
    occupations = reference.ground_state.occupations
    field = UniformMagneticField(_FIELD)
    symmetric = AffineMagneticGauge(field, origin_au=_GAUGE_ORIGIN)
    landau = AffineMagneticGauge(
        field,
        kind=MagneticGaugeKind.LANDAU,
        origin_au=_LANDAU_ORIGIN,
        landau_axis=(_FIELD[1], -_FIELD[0], 0.0),
    )
    physical_direction = AffineMagneticGauge(
        UniformMagneticField(_FIELD_DIRECTION),
        origin_au=_GAUGE_ORIGIN,
    )
    pure_gauge_direction = AffineGaugeDifferenceVariation(landau, symmetric)
    grid_rows: list[dict[str, Any]] = []
    finest: tuple[Any, Any, Any] | None = None
    for level in _GRID_LEVELS:
        quadrature = prepare_ao_quadrature(
            reference,
            BackendConfig(),
            grid_policy=AOGridPolicy.qualification(level),
            block_size=1024,
        )
        result = evaluate_exact_uniform_magnetic_wilson_density(
            quadrature,
            density,
            symmetric,
        )
        static = evaluate_exact_static_magnetic_one_electron_matrices(
            quadrature,
            (field,),
            direct_gauges=(symmetric,),
        )[0]
        stable_oracle_residual = _relative(
            np.asarray(result.overlap_stable),
            np.asarray(static.lower_exact.overlap),
        )
        grid_rows.append(
            {
                "system": system,
                "level": level,
                "points": quadrature.grid.npoints,
                "grid_fingerprint_sha256": quadrature.grid.fingerprint_sha256,
                "density_direct_factorized_relative": result.density_direct_factorized_residual,
                "overlap_direct_factorized_relative": result.overlap_direct_factorized_residual,
                "stable_overlap_oracle_relative": stable_oracle_residual,
                "density_direct_imaginary_max_abs": result.density_direct_imaginary_max_abs,
                "density_factorized_imaginary_max_abs": (
                    result.density_factorized_imaginary_max_abs
                ),
                "density_direct_real_minimum": result.density_direct_real_minimum,
                "density_factorized_real_minimum": result.density_factorized_real_minimum,
                "particle_integral": _complex_record(result.particle_number_direct_integral),
                "particle_stable_metric": _complex_record(
                    result.particle_number_stable_metric
                ),
                "particle_integral_stable_abs": float(
                    abs(
                        complex(np.asarray(result.particle_number_direct_integral).item())
                        - complex(np.asarray(result.particle_number_stable_metric).item())
                    )
                ),
            }
        )
        finest = (quadrature, result, static)
    assert finest is not None
    quadrature, base_result, static = finest
    base_density_values = np.asarray(base_result.density_direct)

    anchors = reference.core_operators.nuclei.coordinates_au[
        reference.anchor_topology.ao_to_atom
    ]
    gauge_function_at_anchors = np.asarray(
        affine_gauge_difference_potential(
            landau,
            symmetric,
            anchors,
            quadrature.backend,
        )
    )
    coefficient_unitary = np.exp(-1j * gauge_function_at_anchors)
    gauge_density = (
        coefficient_unitary[:, None] * density * coefficient_unitary.conj()[None, :]
    )
    landau_result = evaluate_exact_uniform_magnetic_wilson_density(
        quadrature,
        gauge_density,
        landau,
    )
    invariance = {
        "electromagnetic_gauge": {
            "density_relative_l2_residual": _relative(
                np.asarray(landau_result.density_direct),
                base_density_values,
            ),
            "density_maximum_absolute_residual": float(
                np.max(np.abs(np.asarray(landau_result.density_direct) - base_density_values))
            ),
            "particle_stable_absolute_residual": float(
                abs(
                    complex(np.asarray(landau_result.particle_number_stable_metric).item())
                    - complex(np.asarray(base_result.particle_number_stable_metric).item())
                )
            ),
        },
        "coefficient_frame": _coefficient_frame_residual(
            quadrature,
            density,
            symmetric,
            seed=1701 if system == "h2" else 1702,
        ),
    }

    magnetic_direction = evaluate_exact_static_magnetic_one_electron_direction(
        quadrature,
        static,
        _FIELD_DIRECTION,
        gauge=symmetric,
    )
    physical_stable_metric = complex(
        np.einsum(
            "mn,nm->",
            density,
            np.asarray(magnetic_direction.total.overlap),
            optimize=True,
        )
    )
    phase_direction = -1j * (
        gauge_function_at_anchors[:, None] - gauge_function_at_anchors[None, :]
    )
    pure_gauge_overlap_direction = phase_direction * np.asarray(base_result.overlap_stable)
    pure_gauge_stable_metric = complex(
        np.einsum(
            "mn,nm->",
            density,
            pure_gauge_overlap_direction,
            optimize=True,
        )
    )
    matter_rows = _matter_sequences(
        quadrature,
        coefficients,
        occupations,
        symmetric,
        base_density_values,
        seed=2701 if system == "h2" else 2702,
    )
    source_rows = _source_sequences(
        quadrature,
        density,
        symmetric,
        (
            ("physical_magnetic", physical_direction),
            ("pure_gauge", pure_gauge_direction),
        ),
        {
            "physical_magnetic": physical_stable_metric,
            "pure_gauge": pure_gauge_stable_metric,
        },
    )
    return {
        "system": system,
        "scientific_id": config.scientific_id,
        "reference_fingerprint_sha256": reference.fingerprint_sha256,
        "electron_count": reference.ground_state.electron_count,
        "nao": reference.core_operators.nao,
        "occupations": occupations.tolist(),
        "grid_refinement": grid_rows,
        "invariance": invariance,
        "matter_derivatives": matter_rows,
        "source_derivatives": source_rows,
    }


def _write_tables(output: Path, systems: list[dict[str, Any]]) -> None:
    grid_rows = [row for system in systems for row in system["grid_refinement"]]
    with (output / "grid_refinement.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(grid_rows[0]))
        writer.writeheader()
        writer.writerows(grid_rows)
    derivative_rows: list[dict[str, Any]] = []
    for system in systems:
        for family in ("matter_derivatives", "source_derivatives"):
            for row in system[family]:
                flattened = {
                    "system": system["system"],
                    **{
                        key: value
                        for key, value in row.items()
                        if not isinstance(value, dict)
                    },
                }
                derivative_rows.append(flattened)
    fieldnames = sorted({key for row in derivative_rows for key in row})
    with (output / "derivative_sequences.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(derivative_rows)
    invariance_rows = [
        {
            "system": system["system"],
            "kind": kind,
            **values,
        }
        for system in systems
        for kind, values in system["invariance"].items()
    ]
    invariance_fields = sorted({key for row in invariance_rows for key in row})
    with (output / "invariance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=invariance_fields)
        writer.writeheader()
        writer.writerows(invariance_rows)


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
    systems = [_system_result(system, output, timestamp) for system in ("h2", "lih")]
    result = {
        "schema": "aion.chapter13-nq1-wilson-density",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "equations": [
            "eq:wilson-hartree-ks-density-overlap-form",
            "eq:wilson-hartree-ks-density-orbital-form",
            "eq:wilson-hartree-ks-density-normalization",
            "eq:wilson-hartree-ks-density-matter-rho-variation",
            "eq:wilson-hartree-ks-density-matter-matrix-variation",
            "eq:wilson-hartree-ks-density-source-variation",
            "eq:wilson-hartree-ks-density-source-number-response",
        ],
        "realization": {
            "systems": ["h2", "lih"],
            "basis": "sto-3g",
            "functional": "lda,vwn",
            "reference_grid_level": 3,
            "qualification_grid_levels": list(_GRID_LEVELS),
            "qualification_grid_pruning": "none",
            "field_au": list(_FIELD),
            "field_direction_au": list(_FIELD_DIRECTION),
            "symmetric_gauge_origin_au": list(_GAUGE_ORIGIN),
            "landau_gauge_origin_au": list(_LANDAU_ORIGIN),
            "charge": -1.0,
            "hbar": 1.0,
            "precision": "float64_complex128",
            "coefficient_history_fixed_in_source_derivatives": True,
        },
        "matter_steps": list(_MATTER_STEPS),
        "source_steps": list(_SOURCE_STEPS),
        "systems": systems,
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_tables(output, systems)

    tracked_diff = _git(repo, "diff", "--binary", "HEAD", binary=True)
    assert isinstance(tracked_diff, bytes)
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    assert isinstance(status, str)
    head = _git(repo, "rev-parse", "HEAD")
    assert isinstance(head, str)
    source_paths = (
        repo / "tools/run_chapter13_nq1_wilson_density.py",
        repo / "src/aion/electronic_structure/wilson_density.py",
        repo / "src/aion/electromagnetism/test_variations.py",
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
        "schema": "aion.chapter13-nq1-completed",
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
