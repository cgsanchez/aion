#!/usr/bin/env python3
"""Run the resumable WP4 first-order hierarchy qualification campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
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
    evaluate_exact_uniform_electric_internal_connections,
    evaluate_magnetic_one_electron_first_derivatives,
    evaluate_uniform_electric_e1_tensor,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_FORMAL_PLAN = Path(
    "/home/cgs/00_WORK/Projection_Full_Formalism/REVIEW/implementation/"
    "exact_one_electron_numerical_qualification_plan.md"
)
_LOCK = _REPOSITORY / "conda-linux-64.lock"
_DEFAULT_OUTPUT_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/exact_one_electron_qualification"
)
_FAMILIES = ("overlap", "kinetic", "nuclear_attraction", "mechanical")
_MODELS = ("p0", "geometric_b1", "full_b1", "complete_first")
_SYSTEMS = ("hh", "oh", "n2", "co")
_BASES = ("sto-3g", "cc-pvdz", "aug-cc-pvdz")
_BLOCK_SIZE = 2048
_ELECTRIC_STEPS = (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_execution_directory", type=Path)
    parser.add_argument("distance_execution_directory", type=Path)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execution-directory", type=Path)
    parser.add_argument("--systems", nargs="+", choices=_SYSTEMS)
    parser.add_argument("--bases", nargs="+", choices=_BASES)
    parser.add_argument("--bond-scales", nargs="+", type=float)
    parser.add_argument("--max-members", type=int)
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return dict(value)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)  # type: ignore[arg-type]
    os.replace(temporary, path)


def _source_records(
    reference_execution: Path, distance_execution: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for execution, audit_name in (
        (reference_execution, "reference_audit"),
        (distance_execution, "distance_audit"),
    ):
        index_path = execution / "execution_index.json"
        audit_path = execution / audit_name / "result.json"
        index = _json(index_path)
        audit = _json(audit_path)
        floors = {
            (row["system"], row["basis"], float(row["bond_scale"])): {
                family: float(
                    row["floors_absolute_frobenius"][family]["refined_working_floor"]
                )
                for family in _FAMILIES
            }
            for row in audit["members"]
        }
        for record in index["completed_members"]:
            value = dict(record)
            scale = float(record["bond_scale"])
            value["source_execution_directory"] = str(execution)
            value["source_execution_index_sha256"] = _sha256(index_path)
            value["source_audit_result_sha256"] = _sha256(audit_path)
            value["refined_floors"] = floors[(record["system"], record["basis"], scale)]
            records.append(value)
    records.sort(key=lambda row: (row["system"], row["basis"], float(row["bond_scale"])))
    keys = {(row["system"], row["basis"], float(row["bond_scale"])) for row in records}
    if len(records) != 36 or len(keys) != 36:
        raise ValueError("WP4 requires the complete unique 36-member WP3 source campaign")
    return records


def _config(source_manifest: dict[str, Any]) -> OneElectronReferenceConfig:
    system = source_manifest["system"]
    return OneElectronReferenceConfig(
        atoms=tuple(
            AtomConfig(atom["symbol"], tuple(atom["position_au"]))
            for atom in system["atoms"]
        ),
        basis=system["basis"],
        electromagnetic_origin=ElectromagneticOrigin((0.0, 0.0, 0.0)),
    )


def _host(value: Any, quadrature: Any) -> np.ndarray:
    return np.asarray(quadrature.backend.to_host(value))


def _derivative_arrays(derivatives: Any, quadrature: Any) -> dict[str, np.ndarray]:
    return {
        "derivative__overlap": _host(derivatives.metric, quadrature),
        "derivative__kinetic_triangle": _host(derivatives.kinetic_triangle, quadrature),
        "derivative__kinetic_anchored_pC": _host(
            derivatives.kinetic_anchored_pC, quadrature
        ),
        "derivative__kinetic_anchored_Cp": _host(
            derivatives.kinetic_anchored_Cp, quadrature
        ),
        "derivative__kinetic_anchored": _host(derivatives.kinetic_anchored, quadrature),
        "derivative__kinetic": _host(derivatives.kinetic, quadrature),
        "derivative__nuclear_attraction": _host(
            derivatives.nuclear_attraction_triangle, quadrature
        ),
        "derivative__mechanical_triangle": _host(
            derivatives.mechanical_triangle, quadrature
        ),
        "derivative__mechanical_anchored": _host(
            derivatives.mechanical_anchored, quadrature
        ),
        "derivative__mechanical": _host(derivatives.mechanical, quadrature),
    }


def _coefficient_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "overlap": arrays["derivative__overlap"],
        "kinetic": arrays["derivative__kinetic"],
        "nuclear_attraction": arrays["derivative__nuclear_attraction"],
        "mechanical": arrays["derivative__mechanical"],
    }


def _adjoint_rows(coefficients: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for family, value in coefficients.items():
        for axis in range(3):
            scale = max(1.0, float(np.linalg.norm(value[axis])))
            rows.append(
                {
                    "family": family,
                    "cartesian_axis": axis,
                    "relative_residual": float(
                        np.linalg.norm(value[axis] - value[axis].conj().T) / scale
                    ),
                }
            )
    return rows


def _magnetic_fd_rows(
    source_manifest: dict[str, Any],
    source_result: dict[str, Any],
    source_arrays: Any,
    coefficients: dict[str, np.ndarray],
    floors: dict[str, float],
) -> list[dict[str, Any]]:
    case_index = {
        case_id: index for index, case_id in enumerate(source_result["case_ids_in_array_order"])
    }
    cases = {
        (case["direction"], int(case["sign"]), float(case["magnitude_au"])): case
        for case in source_result["cases"]
        if int(case["sign"]) != 0
    }
    directions = source_manifest["field_scan"]["directions"]
    initial = tuple(
        float(value) for value in source_manifest["field_scan"]["initial_magnitudes_au"]
    )
    rows = []
    for direction_name, direction_value in directions.items():
        direction = np.asarray(direction_value, dtype=np.float64)
        for magnitude in initial:
            plus = cases[(direction_name, 1, magnitude)]
            minus = cases[(direction_name, -1, magnitude)]
            plus_index = case_index[plus["id"]]
            minus_index = case_index[minus["id"]]
            for family in _FAMILIES:
                exact = np.asarray(source_arrays[f"barred__exact__{family}"])
                finite_difference = (exact[plus_index] - exact[minus_index]) / (
                    2.0 * magnitude
                )
                analytic = np.einsum("x,xmn->mn", direction, coefficients[family])
                error = float(np.linalg.norm(finite_difference - analytic))
                analytic_norm = float(np.linalg.norm(analytic))
                cancellation_floor = floors[family] / magnitude
                rows.append(
                    {
                        "direction": direction_name,
                        "step_au": magnitude,
                        "family": family,
                        "analytic_norm": analytic_norm,
                        "finite_difference_norm": float(np.linalg.norm(finite_difference)),
                        "absolute_error": error,
                        "relative_error": error / max(1.0, analytic_norm),
                        "cancellation_floor": cancellation_floor,
                        "analytic_signal_to_cancellation_floor": (
                            analytic_norm / max(cancellation_floor, np.finfo(np.float64).tiny)
                        ),
                    }
                )
    return rows


def _model_error_arrays(
    source_arrays: Any,
    field_vectors: np.ndarray,
    coefficients: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    output: dict[str, np.ndarray] = {}
    sector_rows: list[dict[str, Any]] = []
    for family in _FAMILIES:
        p0 = np.asarray(source_arrays[f"barred__p0__{family}"])
        exact = np.asarray(source_arrays[f"barred__exact__{family}"])
        full_increment = np.einsum("cx,xmn->cmn", field_vectors, coefficients[family])
        geometric_increment = (
            full_increment if family == "overlap" else np.zeros_like(full_increment)
        )
        models = {
            "p0": p0,
            "geometric_b1": p0 + geometric_increment,
            "full_b1": p0 + full_increment,
            "complete_first": p0 + full_increment,
        }
        for model, values in models.items():
            absolute = np.linalg.norm(exact - values, axis=(1, 2))
            denominator = np.maximum(1.0, np.linalg.norm(exact, axis=(1, 2)))
            output[f"model_error_absolute__{model}__{family}"] = absolute
            output[f"model_error_relative__{model}__{family}"] = absolute / denominator
        sector_rows.append(
            {
                "family": family,
                "geometric_b1_minus_p0_residual": float(
                    np.linalg.norm(models["geometric_b1"] - p0 - geometric_increment)
                ),
                "full_b1_minus_geometric_b1_residual": float(
                    np.linalg.norm(
                        models["full_b1"]
                        - models["geometric_b1"]
                        - (full_increment - geometric_increment)
                    )
                ),
                "complete_first_minus_full_b1_residual": float(
                    np.linalg.norm(models["complete_first"] - models["full_b1"])
                ),
            }
        )
    return output, sector_rows


def _electric_evidence(
    quadrature: Any, e1: Any
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    analytic_dipoles = _host(e1.central_dipoles, quadrature)
    quadrature_dipoles = _host(e1.quadrature_central_dipoles, quadrature)
    analytic_connection = _host(e1.connection_derivatives, quadrature)
    quadrature_connection = _host(e1.quadrature_connection_derivatives, quadrature)
    directions = {
        "x": np.asarray((1.0, 0.0, 0.0)),
        "y": np.asarray((0.0, 1.0, 0.0)),
        "z": np.asarray((0.0, 0.0, 1.0)),
        "oblique": np.asarray((1.0, 2.0, 3.0)) / np.sqrt(14.0),
    }
    fields: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    identities: list[tuple[str, float, int, int]] = []
    for name, direction in directions.items():
        for step in _ELECTRIC_STEPS:
            plus = step * direction
            minus = -step * direction
            plus_index = len(fields)
            fields.append((float(plus[0]), float(plus[1]), float(plus[2])))
            minus_index = len(fields)
            fields.append((float(minus[0]), float(minus[1]), float(minus[2])))
            identities.append((name, step, plus_index, minus_index))
    connections = evaluate_exact_uniform_electric_internal_connections(quadrature, fields)
    host_connections = np.asarray([_host(value, quadrature) for value in connections])
    rows = []
    for name, step, plus_index, minus_index in identities:
        direction = directions[name]
        finite_difference = (host_connections[plus_index] - host_connections[minus_index]) / (
            2.0 * step
        )
        analytic = np.einsum("x,xmn->mn", direction, analytic_connection)
        quadrature_value = np.einsum("x,xmn->mn", direction, quadrature_connection)
        rows.append(
            {
                "direction": name,
                "step_au": step,
                "analytic_norm": float(np.linalg.norm(analytic)),
                "finite_difference_norm": float(np.linalg.norm(finite_difference)),
                "absolute_error_vs_analytic": float(
                    np.linalg.norm(finite_difference - analytic)
                ),
                "absolute_error_vs_quadrature": float(
                    np.linalg.norm(finite_difference - quadrature_value)
                ),
            }
        )
    arrays = {
        "e1__central_dipoles": analytic_dipoles,
        "e1__quadrature_central_dipoles": quadrature_dipoles,
        "e1__connection_derivatives": analytic_connection,
        "e1__quadrature_connection_derivatives": quadrature_connection,
        "e1__direct_fields_au": np.asarray(fields),
        "e1__direct_connections": host_connections,
    }
    evidence = {
        "central_dipole_absolute_residual": float(
            np.linalg.norm(quadrature_dipoles - analytic_dipoles)
        ),
        "central_dipole_relative_residual": float(
            np.linalg.norm(quadrature_dipoles - analytic_dipoles)
            / max(1.0, float(np.linalg.norm(analytic_dipoles)))
        ),
        "connection_adjoint_residual": float(
            np.linalg.norm(
                analytic_connection
                + np.swapaxes(analytic_connection.conj(), 1, 2)
            )
            / max(1.0, float(np.linalg.norm(analytic_connection)))
        ),
        "finite_source_differences": rows,
    }
    return arrays, evidence


def _member(
    record: dict[str, Any], output: Path, environment: dict[str, Any]
) -> dict[str, Any]:
    started = time.perf_counter()
    source = Path(record["member_directory"])
    source_manifest_path = source / "manifest.json"
    source_result_path = source / "result.json"
    source_arrays_path = source / "arrays.npz"
    for path, field in (
        (source_manifest_path, "manifest_sha256"),
        (source_result_path, "result_sha256"),
        (source_arrays_path, "arrays_sha256"),
    ):
        if _sha256(path) != record[field]:
            raise ValueError(f"WP3 source hash mismatch: {path}")
    source_manifest = _json(source_manifest_path)
    source_result = _json(source_result_path)
    config = _config(source_manifest)
    ao_reference = prepare_one_electron_ao_reference(config)
    if ao_reference.fingerprint_sha256 != source_manifest["system"][
        "reference_fingerprint_sha256"
    ]:
        raise ValueError("WP4 reconstructed AO reference does not match WP3")
    quadrature = prepare_ao_quadrature(
        ao_reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(source_manifest["quadrature"]["level"]),
        block_size=_BLOCK_SIZE,
    )
    if quadrature.grid.fingerprint_sha256 != source_manifest["quadrature"][
        "fingerprint_sha256"
    ]:
        raise ValueError("WP4 quadrature grid does not match WP3")

    derivatives = evaluate_magnetic_one_electron_first_derivatives(quadrature)
    arrays = _derivative_arrays(derivatives, quadrature)
    coefficients = _coefficient_arrays(arrays)
    e1 = evaluate_uniform_electric_e1_tensor(quadrature)
    e1_arrays, electric = _electric_evidence(quadrature, e1)
    arrays.update(e1_arrays)
    with np.load(source_arrays_path, allow_pickle=False) as source_arrays:
        field_vectors = np.asarray(source_arrays["field_vectors_au"])
        model_arrays, sector_rows = _model_error_arrays(
            source_arrays, field_vectors, coefficients
        )
        arrays.update(model_arrays)
        magnetic_fd = _magnetic_fd_rows(
            source_manifest,
            source_result,
            source_arrays,
            coefficients,
            record["refined_floors"],
        )
    arrays["field_vectors_au"] = field_vectors
    array_hashes = {name: canonical_sha256(value) for name, value in arrays.items()}

    manifest_core = {
        "schema": "aion.exact-one-electron-wp4-first-order-member-manifest",
        "version": "1.0.0",
        "status": "executed_unreviewed",
        "system": record["system"],
        "basis": record["basis"],
        "bond_scale": float(record["bond_scale"]),
        "reference_fingerprint_sha256": ao_reference.fingerprint_sha256,
        "grid_fingerprint_sha256": quadrature.grid.fingerprint_sha256,
        "source_wp3": {
            "directory": str(source),
            "manifest_sha256": record["manifest_sha256"],
            "result_sha256": record["result_sha256"],
            "arrays_sha256": record["arrays_sha256"],
            "execution_index_sha256": record["source_execution_index_sha256"],
            "audit_result_sha256": record["source_audit_result_sha256"],
        },
        "quadrature": {
            "backend": "cpu",
            "level": source_manifest["quadrature"]["level"],
            "pruning": "none",
            "block_size": _BLOCK_SIZE,
            "npoints": quadrature.grid.npoints,
        },
        "models": list(_MODELS),
        "conventions": {
            "comparison_representation": "endpoint_removed",
            "endpoint_transport": "exact_and_common_to_every_named_model",
            "static_magnetic_sector_reduction": "C1_equals_B1",
            "q": -1.0,
            "mass": 1.0,
            "hbar": 1.0,
        },
        "environment": environment,
        "input_hashes": {
            "formal_plan": _sha256(_FORMAL_PLAN),
            "runner": _sha256(Path(__file__).resolve()),
            "magnetic_evaluator": _sha256(
                _REPOSITORY / "src/aion/electronic_structure/magnetic_matrices.py"
            ),
            "electric_evaluator": _sha256(
                _REPOSITORY / "src/aion/electronic_structure/electric_matrices.py"
            ),
            "magnetic_analysis": _sha256(
                _REPOSITORY / "src/aion/electronic_structure/magnetic_analysis.py"
            ),
            "conda_lock": _sha256(_LOCK),
        },
    }
    manifest_id = canonical_sha256(manifest_core)
    manifest_path = output / "manifest.json"
    arrays_path = output / "arrays.npz"
    result_path = output / "result.json"
    _write_json(manifest_path, {**manifest_core, "manifest_id": manifest_id})
    _write_npz(arrays_path, arrays)
    result = {
        "schema": "aion.exact-one-electron-wp4-first-order-member-result",
        "version": "1.0.0",
        "status": "executed_unreviewed",
        "manifest_id": manifest_id,
        "system": record["system"],
        "basis": record["basis"],
        "bond_scale": float(record["bond_scale"]),
        "refined_floors_absolute_frobenius": record["refined_floors"],
        "magnetic_derivative_finite_differences": magnetic_fd,
        "coefficient_adjoint_checks": _adjoint_rows(coefficients),
        "sector_increment_identities": sector_rows,
        "electric_e1": electric,
        "array_semantic_sha256": array_hashes,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(result_path, result)
    completed = {
        "status": "authenticated_executed_unreviewed",
        "manifest_id": manifest_id,
        "manifest_sha256": _sha256(manifest_path),
        "result_sha256": _sha256(result_path),
        "arrays_sha256": _sha256(arrays_path),
        "array_count": len(arrays),
    }
    _write_json(output / "completed.json", completed)
    return {
        "system": record["system"],
        "basis": record["basis"],
        "bond_scale": float(record["bond_scale"]),
        "member_directory": str(output),
        "manifest_id": manifest_id,
        "manifest_sha256": completed["manifest_sha256"],
        "result_sha256": completed["result_sha256"],
        "arrays_sha256": completed["arrays_sha256"],
        "array_count": completed["array_count"],
        "elapsed_seconds": result["elapsed_seconds"],
    }


def _environment() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "thread_limits": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def _execution_directory(arguments: argparse.Namespace) -> Path:
    if arguments.execution_directory is not None:
        return arguments.execution_directory.resolve()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    commit = _git("rev-parse", "HEAD")[:12]
    return (arguments.output_root / f"wp4_first_order_{timestamp}_{commit}").resolve()


def main() -> None:
    arguments = _arguments()
    reference_execution = arguments.reference_execution_directory.resolve()
    distance_execution = arguments.distance_execution_directory.resolve()
    records = _source_records(reference_execution, distance_execution)
    if arguments.systems is not None:
        records = [row for row in records if row["system"] in arguments.systems]
    if arguments.bases is not None:
        records = [row for row in records if row["basis"] in arguments.bases]
    if arguments.bond_scales is not None:
        requested_scales = {float(value) for value in arguments.bond_scales}
        unknown = requested_scales - {0.8, 1.0, 1.25}
        if unknown:
            raise ValueError(f"unsupported bond scales: {sorted(unknown)}")
        records = [row for row in records if float(row["bond_scale"]) in requested_scales]
    if arguments.max_members is not None:
        if arguments.max_members <= 0:
            raise ValueError("--max-members must be positive")
        records = records[: arguments.max_members]
    execution = _execution_directory(arguments)
    execution.mkdir(parents=True, exist_ok=True)
    lock_stream = (execution / ".campaign.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another WP4 campaign owns {execution}") from error

    environment = _environment()
    plan_core = {
        "schema": "aion.exact-one-electron-wp4-first-order-execution-plan",
        "version": "1.0.0",
        "status": "execution_in_progress",
        "execution_directory": str(execution),
        "reference_execution_directory": str(reference_execution),
        "distance_execution_directory": str(distance_execution),
        "reference_execution_index_sha256": _sha256(
            reference_execution / "execution_index.json"
        ),
        "distance_execution_index_sha256": _sha256(distance_execution / "execution_index.json"),
        "member_count": len(records),
        "environment": environment,
        "code": {
            "repository": str(_REPOSITORY),
            "branch": _git("branch", "--show-current"),
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
    }
    plan_id = canonical_sha256(plan_core)
    _write_json(execution / "execution_plan.json", {**plan_core, "plan_id": plan_id})

    completed_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        scale = float(record["bond_scale"])
        output = execution / record["system"] / record["basis"] / f"bond_scale_{scale:.6f}"
        output.mkdir(parents=True, exist_ok=True)
        completed_path = output / "completed.json"
        label = (
            f"{index}/{len(records)} {record['system']} {record['basis']} {scale}"
        )
        if completed_path.exists():
            completed = _json(completed_path)
            completed_records.append(
                {
                    "system": record["system"],
                    "basis": record["basis"],
                    "bond_scale": scale,
                    "member_directory": str(output),
                    **{
                        name: completed[name]
                        for name in (
                            "manifest_id",
                            "manifest_sha256",
                            "result_sha256",
                            "arrays_sha256",
                            "array_count",
                        )
                    },
                    "elapsed_seconds": _json(output / "result.json")["elapsed_seconds"],
                }
            )
            print(f"skip {label}", flush=True)
            continue
        print(f"start {label}", flush=True)
        completed_records.append(_member(record, output, environment))
        _write_json(
            execution / "execution_index.json",
            {
                "schema": "aion.exact-one-electron-wp4-first-order-execution-index",
                "version": "1.0.0",
                "status": "execution_in_progress",
                "plan_id": plan_id,
                "completed_member_count": len(completed_records),
                "requested_member_count": len(records),
                "completed_members": completed_records,
            },
        )
        print(f"finish {label}", flush=True)

    index = {
        "schema": "aion.exact-one-electron-wp4-first-order-execution-index",
        "version": "1.0.0",
        "status": "authenticated_executed_unreviewed",
        "plan_id": plan_id,
        "execution_directory": str(execution),
        "completed_member_count": len(completed_records),
        "requested_member_count": len(records),
        "total_elapsed_seconds": time.perf_counter() - started,
        "completed_members": completed_records,
    }
    _write_json(execution / "execution_index.json", index)
    print(f"execution={execution}")
    print(f"members={len(completed_records)}")


if __name__ == "__main__":
    main()
