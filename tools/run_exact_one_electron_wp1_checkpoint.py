#!/usr/bin/env python3
"""Execute the zero-field H--H overlap/kinetic qualification checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from aion.config import (
    AtomConfig,
    BackendConfig,
    ElectromagneticOrigin,
    OneElectronReferenceConfig,
    canonical_sha256,
)
from aion.electronic_structure import (
    AOGridPolicy,
    evaluate_zero_field_overlap_kinetic,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_FIXTURE = _REPOSITORY / "tests/fixtures/exact_one_electron/hh_sto3g.fixture.json"
_LOCK = _REPOSITORY / "conda-linux-64.lock"
_DEFAULT_OUTPUT_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "exact_one_electron_qualification"
)
_LEVELS = (0, 1, 2, 3, 4)
_BLOCK_SIZE = 512
_RELATIVE_MATRIX_TARGET = 1.0e-5
_HERMITICITY_TARGET = 1.0e-12


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help="campaign root; a new immutable execution directory is created below it",
    )
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _array_record(
    name: str,
    array: np.ndarray,
    *,
    unit: str,
    physical_dimension: str,
    definition: str,
) -> dict[str, object]:
    value = np.asarray(array)
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "unit": unit,
        "physical_dimension": physical_dimension,
        "definition": definition,
        "data_sha256": canonical_sha256(value),
    }


def _fixture_config(fixture: dict[str, Any]) -> OneElectronReferenceConfig:
    config = fixture["config"]
    return OneElectronReferenceConfig(
        atoms=tuple(
            AtomConfig(atom["symbol"], tuple(atom["position_au"]))
            for atom in config["atoms"]
        ),
        basis=config["basis"],
        electromagnetic_origin=ElectromagneticOrigin(
            tuple(config["electromagnetic_origin_au"])
        ),
    )


def _shell_metadata(reference: Any) -> dict[str, object]:
    metadata = reference.basis_metadata
    return {
        "shell_to_atom": metadata.shell_to_atom.tolist(),
        "shell_angular_momenta": metadata.shell_angular_momenta.tolist(),
        "shell_primitive_counts": metadata.shell_primitive_counts.tolist(),
        "shell_contraction_counts": metadata.shell_contraction_counts.tolist(),
        "ao_locations": metadata.ao_locations.tolist(),
        "fingerprint_sha256": metadata.fingerprint_sha256,
    }


def _manifest_without_id(
    *,
    timestamp_utc: str,
    branch: str,
    commit: str,
    dirty: bool,
    fixture: dict[str, Any],
    reference: Any,
    quadrature: Any,
    input_hashes: dict[str, str],
) -> dict[str, object]:
    thread_names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "schema": "aion.exact-one-electron-run-manifest",
        "version": "1.0.0",
        "status": "executed_unreviewed",
        "execution": {
            "timestamp_utc": timestamp_utc,
            "runner": str(Path(__file__).resolve()),
        },
        "code": {
            "repository": str(_REPOSITORY),
            "branch": branch,
            "commit": commit,
            "dirty": dirty,
        },
        "environment": {
            "catalog": "2026.07.1",
            "profile": "gnu",
            "modules": [],
            "conda_prefix": os.environ.get("CONDA_PREFIX", "unreported"),
            "conda_lock_sha256": _file_sha256(_LOCK),
            "dependencies": reference.dependencies.as_mapping(),
            "thread_limits": {
                name: os.environ.get(name, "unset") for name in thread_names
            },
            "hostname": socket.gethostname(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "fixture": {
            "reference_fingerprint_sha256": reference.fingerprint_sha256,
            "config_id": reference.config.scientific_id,
            "atoms_bohr": [
                [atom.symbol, *atom.position_au] for atom in reference.config.atoms
            ],
            "nuclear_charges": reference.core_operators.nuclei.charges.tolist(),
            "coordinate_unit": "bohr",
            "basis": reference.config.basis,
            "spherical": reference.basis_metadata.spherical,
            "ao_labels": list(reference.basis_metadata.ao_labels),
            "shell_metadata": _shell_metadata(reference),
            "ao_to_atom": reference.anchor_topology.ao_to_atom.tolist(),
            "fixture_schema": fixture["schema"],
        },
        "quadrature": {
            "kind": quadrature.grid.kind.value,
            "level": quadrature.grid.level,
            "pruning": quadrature.grid.pruning,
            "npoints": quadrature.grid.npoints,
            "fingerprint_sha256": quadrature.grid.fingerprint_sha256,
            "coordinates_sha256": canonical_sha256(quadrature.grid.coordinates_au),
            "weights_sha256": canonical_sha256(quadrature.grid.weights_au),
            "block_size": quadrature.block_size,
            "backend": quadrature.backend_config.kind.value,
            "evaluator": quadrature.provenance.evaluator,
        },
        "source": {
            "field": "zero",
            "scalar_potential": "zero",
            "vector_potential": "zero",
            "approximation_level": "bare",
            "retained_sectors": ["field_free_overlap", "field_free_kinetic"],
        },
        "conventions": {
            "units": "atomic",
            "q": -1.0,
            "mass": 1.0,
            "hbar": 1.0,
            "matrix_indices": "row_mu_bra_column_nu_ket",
            "endpoint_path": "R_nu_to_R_mu",
            "anchor_to_point_path": "R_mu_to_r",
            "ao_frame": "real_spherical_gaussian",
            "canonical_momentum": "p=-i*hbar*grad",
        },
        "tolerances": {
            "relative_frobenius": _RELATIVE_MATRIX_TARGET,
            "hermiticity_relative_frobenius": _HERMITICITY_TARGET,
        },
        "input_hashes": input_hashes,
    }


def _residual(
    name: str,
    value: float,
    *,
    normalization: str,
    reference: str,
    tolerance: float | None,
) -> dict[str, object]:
    return {
        "name": name,
        "value": value,
        "normalization": normalization,
        "reference": reference,
        "tolerance": tolerance,
    }


def _result_record(
    *,
    manifest_id: str,
    arrays_path: Path,
    overlap: Any,
    kinetic: Any,
) -> dict[str, object]:
    return {
        "schema": "aion.exact-one-electron-matrix-result",
        "version": "1.0.0",
        "status": "executed_unreviewed",
        "manifest_id": manifest_id,
        "matrices": [
            _array_record(
                "overlap/analytic",
                overlap.analytic,
                unit="1",
                physical_dimension="overlap",
                definition="independent PySCF/libcint int1e_ovlp matrix",
            ),
            _array_record(
                "overlap/quadrature",
                overlap.quadrature,
                unit="1",
                physical_dimension="overlap",
                definition="integral of AO bra value times AO ket value",
            ),
            _array_record(
                "kinetic/analytic",
                kinetic.analytic,
                unit="hartree",
                physical_dimension="energy",
                definition="independent PySCF/libcint int1e_kin matrix",
            ),
            _array_record(
                "kinetic/quadrature",
                kinetic.quadrature,
                unit="hartree",
                physical_dimension="energy",
                definition="weak form hbar^2/(2m) integral grad(phi_mu)* dot grad(phi_nu)",
            ),
        ],
        "residuals": [
            _residual(
                "overlap/absolute_frobenius",
                overlap.absolute_frobenius_residual,
                normalization="none",
                reference="analytic_libcint_overlap",
                tolerance=None,
            ),
            _residual(
                "overlap/relative_frobenius",
                overlap.relative_frobenius_residual,
                normalization="max(1, Frobenius norm of analytic libcint matrix)",
                reference="analytic_libcint_overlap",
                tolerance=_RELATIVE_MATRIX_TARGET,
            ),
            _residual(
                "overlap/hermiticity_relative_frobenius",
                overlap.hermiticity_residual,
                normalization="max(1, Frobenius norm of quadrature matrix)",
                reference="quadrature_matrix_adjoint",
                tolerance=_HERMITICITY_TARGET,
            ),
            _residual(
                "kinetic/absolute_frobenius",
                kinetic.absolute_frobenius_residual,
                normalization="none",
                reference="analytic_libcint_kinetic",
                tolerance=None,
            ),
            _residual(
                "kinetic/relative_frobenius",
                kinetic.relative_frobenius_residual,
                normalization="max(1, Frobenius norm of analytic libcint matrix)",
                reference="analytic_libcint_kinetic",
                tolerance=_RELATIVE_MATRIX_TARGET,
            ),
            _residual(
                "kinetic/hermiticity_relative_frobenius",
                kinetic.hermiticity_residual,
                normalization="max(1, Frobenius norm of quadrature matrix)",
                reference="quadrature_matrix_adjoint",
                tolerance=_HERMITICITY_TARGET,
            ),
        ],
        "artifact": {"path": str(arrays_path), "sha256": _file_sha256(arrays_path)},
        "review": {"reviewed": False, "reviewer": None, "decision": "pending"},
    }


def _report(rows: list[dict[str, object]], execution_directory: Path) -> str:
    lines = [
        "# Exact one-electron zero-field checkpoint",
        "",
        "Status: **numerically executed; user review pending**.",
        "",
        "This is the overlap/kinetic subset of WP1. It does not complete or accept G1: ",
        "nuclear attraction, an alternative pruning rule, AO derivative/momentum sign, ",
        "atom-pair floors, and the O--H fixture remain deliberately outside this checkpoint.",
        "No magnetic field was evaluated.",
        "",
        "The declared relative matrix target is `1e-5`, one order below the plan's ",
        "smallest physical reporting threshold (`1e-4`). The Hermiticity target is `1e-12`.",
        "",
        "| grid level | points | overlap rel. Frobenius | kinetic rel. Frobenius | "
        "overlap successive | kinetic successive | max Hermiticity |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {level} | {npoints} | {overlap:.6e} | {kinetic:.6e} | "
            "{overlap_successive} | {kinetic_successive} | {hermiticity:.6e} |".format(
                level=row["level"],
                npoints=row["npoints"],
                overlap=row["overlap"],
                kinetic=row["kinetic"],
                overlap_successive=(
                    "--"
                    if row["overlap_successive"] is None
                    else f"{row['overlap_successive']:.6e}"
                ),
                kinetic_successive=(
                    "--"
                    if row["kinetic_successive"] is None
                    else f"{row['kinetic_successive']:.6e}"
                ),
                hermiticity=row["hermiticity"],
            )
        )
    lines.extend(
        (
            "",
            "Each level directory contains an executed-unreviewed run manifest, a result ",
            "record, and the authenticated raw grid and matrix arrays. The execution root is:",
            "",
            f"`{execution_directory}`",
            "",
            "Passing values in this table are numerical evidence for this declared fixture ",
            "only. They are not a reviewed gate decision.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    arguments = _arguments()
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    reference = prepare_one_electron_ao_reference(_fixture_config(fixture))
    if reference.fingerprint_sha256 != fixture["reference_fingerprint_sha256"]:
        raise RuntimeError("live one-electron reference does not match the frozen fixture")

    branch = _git("branch", "--show-current")
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    started = datetime.now(UTC)
    timestamp = started.strftime("%Y%m%dT%H%M%SZ")
    timestamp_utc = started.isoformat().replace("+00:00", "Z")
    execution_directory = arguments.output_root / f"{timestamp}_{commit[:12]}"
    execution_directory.mkdir(parents=True, exist_ok=False)

    tracked_inputs = (
        _FIXTURE,
        _LOCK,
        _REPOSITORY / "docs/exact_one_electron_qualification_g0.md",
        _REPOSITORY / "docs/schemas/exact_one_electron_run_manifest.schema.json",
        _REPOSITORY / "docs/schemas/exact_one_electron_matrix_result.schema.json",
        _REPOSITORY / "src/aion/electronic_structure/ao_quadrature.py",
        _REPOSITORY / "src/aion/electronic_structure/one_electron.py",
        Path(__file__).resolve(),
    )
    input_hashes = {
        str(path.relative_to(_REPOSITORY)): _file_sha256(path) for path in tracked_inputs
    }
    rows: list[dict[str, object]] = []
    previous_overlap: np.ndarray | None = None
    previous_kinetic: np.ndarray | None = None

    for level in _LEVELS:
        quadrature = prepare_ao_quadrature(
            reference,
            BackendConfig(),
            grid_policy=AOGridPolicy.qualification(level),
            block_size=_BLOCK_SIZE,
        )
        evaluated = evaluate_zero_field_overlap_kinetic(quadrature)
        level_directory = execution_directory / f"grid_level_{level}"
        level_directory.mkdir()
        arrays_path = level_directory / "arrays.npz"
        np.savez(
            arrays_path,
            grid_coordinates_au=quadrature.grid.coordinates_au,
            grid_weights_au=quadrature.grid.weights_au,
            overlap_analytic=evaluated.overlap.analytic,
            overlap_quadrature=evaluated.overlap.quadrature,
            kinetic_analytic=evaluated.kinetic.analytic,
            kinetic_quadrature=evaluated.kinetic.quadrature,
        )
        manifest_core = _manifest_without_id(
            timestamp_utc=timestamp_utc,
            branch=branch,
            commit=commit,
            dirty=dirty,
            fixture=fixture,
            reference=reference,
            quadrature=quadrature,
            input_hashes=input_hashes,
        )
        manifest_id = canonical_sha256(manifest_core)
        manifest = {**manifest_core, "manifest_id": manifest_id}
        result = _result_record(
            manifest_id=manifest_id,
            arrays_path=arrays_path,
            overlap=evaluated.overlap,
            kinetic=evaluated.kinetic,
        )
        _write_json(level_directory / "manifest.json", manifest)
        _write_json(level_directory / "result.json", result)

        overlap_scale = max(1.0, float(np.linalg.norm(evaluated.overlap.analytic)))
        kinetic_scale = max(1.0, float(np.linalg.norm(evaluated.kinetic.analytic)))
        rows.append(
            {
                "level": level,
                "npoints": evaluated.grid_npoints,
                "overlap": evaluated.overlap.relative_frobenius_residual,
                "kinetic": evaluated.kinetic.relative_frobenius_residual,
                "overlap_successive": (
                    None
                    if previous_overlap is None
                    else float(
                        np.linalg.norm(evaluated.overlap.quadrature - previous_overlap)
                        / overlap_scale
                    )
                ),
                "kinetic_successive": (
                    None
                    if previous_kinetic is None
                    else float(
                        np.linalg.norm(evaluated.kinetic.quadrature - previous_kinetic)
                        / kinetic_scale
                    )
                ),
                "hermiticity": max(
                    evaluated.overlap.hermiticity_residual,
                    evaluated.kinetic.hermiticity_residual,
                ),
            }
        )
        previous_overlap = evaluated.overlap.quadrature
        previous_kinetic = evaluated.kinetic.quadrature

    _write_json(
        execution_directory / "execution_index.json",
        {
            "schema": "aion.exact-one-electron-checkpoint-index",
            "version": "1.0.0",
            "status": "executed_unreviewed",
            "levels": rows,
            "review": {"reviewed": False, "reviewer": None, "decision": "pending"},
        },
    )
    (execution_directory / "checkpoint_report.md").write_text(
        _report(rows, execution_directory), encoding="utf-8"
    )
    print(execution_directory)


if __name__ == "__main__":
    main()
