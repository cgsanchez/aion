#!/usr/bin/env python3
"""Audit WP3 pair-scan floors and finite-field gauge covariance."""

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
from scipy.linalg import eigvalsh

from aion.backends import NumPyBackend
from aion.config import (
    AtomConfig,
    BackendConfig,
    ElectromagneticOrigin,
    OneElectronReferenceConfig,
    canonical_sha256,
)
from aion.electromagnetism import (
    AffineMagneticGauge,
    MagneticGaugeKind,
    UniformMagneticField,
    affine_gauge_difference_potential,
)
from aion.electronic_structure import (
    AOGridPolicy,
    evaluate_exact_static_magnetic_one_electron_matrices,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_LOCK = _REPOSITORY / "conda-linux-64.lock"
_FAMILIES = ("overlap", "kinetic", "nuclear_attraction", "mechanical")
_GRID_LEVELS = (4, 5)
_BLOCK_SIZE = 2048
_GAUGE_MAGNITUDE_AU = 1.0
_GAUGE_ORIGINS_AU = ((0.0, 0.0, 0.0), (0.21, -0.17, 0.13))
_GAUGE_CONGRUENCE_TOLERANCE = 1.0e-11
_BARRED_INVARIANCE_TOLERANCE = 1.0e-12


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution_directory", type=Path)
    parser.add_argument("--output-name", default="reference_audit")
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
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)  # type: ignore[arg-type]
    os.replace(temporary, path)


def _config(manifest: dict[str, Any]) -> OneElectronReferenceConfig:
    system = manifest["system"]
    return OneElectronReferenceConfig(
        atoms=tuple(
            AtomConfig(atom["symbol"], tuple(atom["position_au"]))
            for atom in system["atoms"]
        ),
        basis=system["basis"],
        electromagnetic_origin=ElectromagneticOrigin((0.0, 0.0, 0.0)),
    )


def _barred_matrices(result: Any) -> dict[str, np.ndarray]:
    return {
        "overlap": np.asarray(result.overlap.exact),
        "kinetic": np.asarray(result.kinetic.exact),
        "nuclear_attraction": np.asarray(result.nuclear_attraction.exact),
        "mechanical": np.asarray(
            result.kinetic.exact + result.nuclear_attraction.exact
        ),
    }


def _lower_matrices(result: Any) -> dict[str, np.ndarray]:
    return {
        "overlap": np.asarray(result.lower_exact.overlap),
        "kinetic": np.asarray(result.lower_exact.kinetic),
        "nuclear_attraction": np.asarray(result.lower_exact.nuclear_attraction),
        "mechanical": np.asarray(result.lower_exact.mechanical),
    }


def _zero_matrices(
    result: Any,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    analytic = {
        "overlap": np.asarray(result.overlap.zero),
        "kinetic": np.asarray(result.kinetic.zero),
        "nuclear_attraction": np.asarray(result.nuclear_attraction.zero),
    }
    raw = {
        "overlap": np.asarray(result.overlap.quadrature_zero),
        "kinetic": np.asarray(result.kinetic.quadrature_zero),
        "nuclear_attraction": np.asarray(result.nuclear_attraction.quadrature_zero),
    }
    analytic["mechanical"] = analytic["kinetic"] + analytic["nuclear_attraction"]
    raw["mechanical"] = raw["kinetic"] + raw["nuclear_attraction"]
    return analytic, raw


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(1.0, float(np.linalg.norm(right))))


def _spectrum(result: Any) -> np.ndarray:
    return np.asarray(
        eigvalsh(result.lower_exact.mechanical, result.lower_exact.overlap),
        dtype=np.float64,
    )


def _authenticate_source(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    member = Path(record["member_directory"])
    manifest_path = member / "manifest.json"
    result_path = member / "result.json"
    arrays_path = member / "arrays.npz"
    if _sha256(manifest_path) != record["manifest_sha256"]:
        raise ValueError(f"source manifest hash mismatch: {member}")
    if _sha256(result_path) != record["result_sha256"]:
        raise ValueError(f"source result hash mismatch: {member}")
    if _sha256(arrays_path) != record["arrays_sha256"]:
        raise ValueError(f"source arrays hash mismatch: {member}")
    manifest = _json(manifest_path)
    result = _json(result_path)
    if manifest["status"] != "executed_unreviewed":
        raise ValueError(f"source manifest has invalid status: {member}")
    if result["status"] != "executed_unreviewed":
        raise ValueError(f"source result has invalid status: {member}")
    if manifest["manifest_id"] != result["manifest_id"]:
        raise ValueError(f"source manifest identity mismatch: {member}")
    return manifest, result


def _member_audit(
    record: dict[str, Any],
    oblique_direction: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest, source_result = _authenticate_source(record)
    reference = prepare_one_electron_ao_reference(_config(manifest))
    if reference.fingerprint_sha256 != manifest["system"]["reference_fingerprint_sha256"]:
        raise ValueError("reconstructed reference fingerprint differs from source member")

    quadratures = {
        level: prepare_ao_quadrature(
            reference,
            BackendConfig(),
            grid_policy=AOGridPolicy.qualification(level),
            block_size=_BLOCK_SIZE,
        )
        for level in _GRID_LEVELS
    }
    zero_field = UniformMagneticField((0.0, 0.0, 0.0))
    zero_results = {
        level: evaluate_exact_static_magnetic_one_electron_matrices(
            quadrature, (zero_field,), include_direct_oracle=False
        )[0]
        for level, quadrature in quadratures.items()
    }
    analytic, raw4 = _zero_matrices(zero_results[4])
    analytic5, raw5 = _zero_matrices(zero_results[5])
    floors: dict[str, dict[str, float]] = {}
    arrays: dict[str, np.ndarray] = {}
    for family in _FAMILIES:
        if not np.array_equal(analytic[family], analytic5[family]):
            raise ValueError(f"analytic {family} changed with grid level")
        delta_l4_analytic = float(np.linalg.norm(raw4[family] - analytic[family]))
        delta_l5_analytic = float(np.linalg.norm(raw5[family] - analytic[family]))
        delta_l5_l4 = float(np.linalg.norm(raw5[family] - raw4[family]))
        roundoff = 64.0 * np.finfo(np.float64).eps * max(
            1.0, float(np.linalg.norm(analytic[family]))
        )
        working = max(delta_l5_analytic, delta_l5_l4, roundoff)
        stored = float(source_result["numerical_floor_absolute_frobenius"][family])
        reproduction = abs(stored - max(delta_l4_analytic, roundoff))
        floors[family] = {
            "level4_vs_analytic": delta_l4_analytic,
            "level5_vs_analytic": delta_l5_analytic,
            "level5_vs_level4": delta_l5_l4,
            "roundoff_bound": roundoff,
            "refined_working_floor": working,
            "source_level4_floor": stored,
            "source_level4_floor_reproduction_absolute": reproduction,
        }
        arrays[f"zero__analytic__{family}"] = analytic[family]
        arrays[f"zero__level4__{family}"] = raw4[family]
        arrays[f"zero__level5__{family}"] = raw5[family]

    field_vector = _GAUGE_MAGNITUDE_AU * oblique_direction
    gauge_field = UniformMagneticField(tuple(field_vector))
    landau_axis = np.cross(field_vector, np.asarray((0.31, -0.47, 0.79)))
    gauges = (
        AffineMagneticGauge(gauge_field, origin_au=_GAUGE_ORIGINS_AU[0]),
        AffineMagneticGauge(gauge_field, origin_au=_GAUGE_ORIGINS_AU[1]),
        AffineMagneticGauge(
            gauge_field,
            kind=MagneticGaugeKind.LANDAU,
            origin_au=_GAUGE_ORIGINS_AU[0],
            landau_axis=tuple(landau_axis),
        ),
        AffineMagneticGauge(
            gauge_field,
            kind=MagneticGaugeKind.LANDAU,
            origin_au=_GAUGE_ORIGINS_AU[1],
            landau_axis=tuple(landau_axis),
        ),
    )
    gauge_names = (
        "symmetric_origin_a",
        "symmetric_origin_b",
        "landau_origin_a",
        "landau_origin_b",
    )
    gauge_results = evaluate_exact_static_magnetic_one_electron_matrices(
        quadratures[4],
        (gauge_field,) * len(gauges),
        direct_gauges=gauges,
        include_direct_oracle=False,
    )
    baseline = gauge_results[0]
    baseline_lower = _lower_matrices(baseline)
    baseline_barred = _barred_matrices(baseline)
    baseline_spectrum = _spectrum(baseline)
    anchors = reference.core_operators.nuclei.coordinates_au[
        reference.anchor_topology.ao_to_atom
    ]
    backend = NumPyBackend()
    gauge_rows: list[dict[str, Any]] = []
    condition = float(np.linalg.cond(baseline_lower["overlap"]))
    spectrum_scale = max(1.0, float(np.max(np.abs(baseline_spectrum))))
    spectrum_tolerance = max(
        1.0e-11,
        64.0 * np.finfo(np.float64).eps * condition * spectrum_scale,
    )
    for name, gauge, result in zip(
        gauge_names, gauges, gauge_results, strict=True
    ):
        chi = np.asarray(
            affine_gauge_difference_potential(gauge, gauges[0], anchors, backend)
        )
        phase = np.exp(-1j * chi)
        lower = _lower_matrices(result)
        barred = _barred_matrices(result)
        lower_residuals: dict[str, float] = {}
        barred_residuals: dict[str, float] = {}
        for family in _FAMILIES:
            expected = (
                phase[:, None] * baseline_lower[family] * phase[None, :].conj()
            )
            lower_residuals[family] = _relative(lower[family], expected)
            barred_residuals[family] = _relative(
                barred[family], baseline_barred[family]
            )
            arrays[f"gauge__{name}__lower__{family}"] = lower[family]
        spectrum_residual = float(np.max(np.abs(_spectrum(result) - baseline_spectrum)))
        gauge_rows.append(
            {
                "case": name,
                "kind": gauge.kind.value,
                "origin_au": list(gauge.origin_au),
                "landau_axis": (
                    None if gauge.landau_axis is None else list(gauge.landau_axis)
                ),
                "lower_congruence_relative": lower_residuals,
                "barred_invariance_relative": barred_residuals,
                "generalized_spectrum_max_absolute_hartree": spectrum_residual,
                "generalized_spectrum_tolerance_hartree": spectrum_tolerance,
            }
        )
    arrays["gauge__baseline__generalized_spectrum_hartree"] = baseline_spectrum

    maximum_congruence = max(
        value
        for row in gauge_rows
        for value in row["lower_congruence_relative"].values()
    )
    maximum_barred = max(
        value
        for row in gauge_rows
        for value in row["barred_invariance_relative"].values()
    )
    maximum_spectrum = max(
        row["generalized_spectrum_max_absolute_hartree"] for row in gauge_rows
    )
    member_key = (
        f"{manifest['system']['name']}__{manifest['system']['basis']}__"
        f"scale_{float(manifest['system']['bond_scale']):.6f}"
    )
    return (
        {
            "member_key": member_key,
            "source_member_directory": record["member_directory"],
            "source_manifest_id": manifest["manifest_id"],
            "system": manifest["system"]["name"],
            "basis": manifest["system"]["basis"],
            "bond_scale": manifest["system"]["bond_scale"],
            "nao": manifest["system"]["nao"],
            "grids": {
                str(level): {
                    "level": quadrature.grid.level,
                    "pruning": quadrature.grid.pruning,
                    "npoints": quadrature.grid.npoints,
                    "fingerprint_sha256": quadrature.grid.fingerprint_sha256,
                    "coordinates_sha256": canonical_sha256(
                        quadrature.grid.coordinates_au
                    ),
                    "weights_sha256": canonical_sha256(quadrature.grid.weights_au),
                }
                for level, quadrature in quadratures.items()
            },
            "floors_absolute_frobenius": floors,
            "gauge_field_au": list(gauge_field.magnetic_field_au),
            "gauge_field_magnitude_au": gauge_field.magnitude_au,
            "gauge_covariance": gauge_rows,
            "maximum_lower_congruence_relative": maximum_congruence,
            "maximum_barred_invariance_relative": maximum_barred,
            "maximum_generalized_spectrum_residual_hartree": maximum_spectrum,
            "gauge_checks_pass": bool(
                maximum_congruence <= _GAUGE_CONGRUENCE_TOLERANCE
                and maximum_barred <= _BARRED_INVARIANCE_TOLERANCE
                and all(
                    row["generalized_spectrum_max_absolute_hartree"]
                    <= row["generalized_spectrum_tolerance_hartree"]
                    for row in gauge_rows
                )
            ),
        },
        {f"{member_key}__{name}": value for name, value in arrays.items()},
    )


def main() -> None:
    arguments = _arguments()
    execution = arguments.execution_directory.resolve()
    if not arguments.output_name or Path(arguments.output_name).name != arguments.output_name:
        raise SystemExit("output-name must be one nonempty path component")
    output = execution / arguments.output_name
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".audit.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(f"another WP3 reference audit owns {lock_path}") from error
        completed_path = output / "completed.json"
        if completed_path.is_file():
            print(f"skip completed audit {output}", flush=True)
            return

        started_at = datetime.now(UTC)
        started = time.monotonic()
        index_path = execution / "execution_index.json"
        plan_path = execution / "execution_plan.json"
        index = _json(index_path)
        plan = _json(plan_path)
        records = index["completed_members"]
        if not records or index["completed_member_count"] != len(records):
            raise ValueError("pair audit requires a nonempty complete execution index")
        oblique = np.asarray(
            plan["fixture"]["field_directions"]["oblique"], dtype=np.float64
        )
        oblique /= np.linalg.norm(oblique)

        rows: list[dict[str, Any]] = []
        arrays: dict[str, np.ndarray] = {}
        for record in records:
            label = f"{record['system']} {record['basis']}"
            print(f"start audit {label}", flush=True)
            row, member_arrays = _member_audit(record, oblique)
            rows.append(row)
            arrays.update(member_arrays)
            print(f"finish audit {label}", flush=True)

        arrays_path = output / "arrays.npz"
        _write_npz(arrays_path, arrays)
        array_hashes = {name: canonical_sha256(value) for name, value in arrays.items()}
        manifest_core = {
            "schema": "aion.exact-one-electron-wp3-pair-audit-manifest",
            "version": "2.0.0",
            "status": "executed_unreviewed",
            "source_execution_directory": str(execution),
            "source_execution_index_sha256": _sha256(index_path),
            "source_execution_plan_sha256": _sha256(plan_path),
            "code": {
                "repository": str(_REPOSITORY),
                "branch": _git("branch", "--show-current"),
                "commit": _git("rev-parse", "HEAD"),
                "dirty": bool(_git("status", "--porcelain")),
                "audit_runner_sha256": _sha256(Path(__file__).resolve()),
            },
            "environment": {
                "catalog": "2026.07.1",
                "profile": "gnu",
                "modules": [],
                "conda_prefix": os.environ.get("CONDA_PREFIX", "unreported"),
                "conda_lock_sha256": _sha256(_LOCK),
                "hostname": socket.gethostname(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "thread_limits": {
                    name: os.environ.get(name, "unset")
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
            },
            "audit": {
                "grid_levels": list(_GRID_LEVELS),
                "grid_pruning": "none",
                "gauge_field_magnitude_au": _GAUGE_MAGNITUDE_AU,
                "gauge_field_direction": oblique.tolist(),
                "gauge_origins_au": [list(value) for value in _GAUGE_ORIGINS_AU],
                "gauge_representatives": ["symmetric", "landau"],
                "gauge_congruence_relative_tolerance": (
                    _GAUGE_CONGRUENCE_TOLERANCE
                ),
                "barred_invariance_relative_tolerance": (
                    _BARRED_INVARIANCE_TOLERANCE
                ),
                "spectrum_tolerance_rule": (
                    "max(1e-11 Ha, 64 eps cond(S) max(1,max_abs_spectrum))"
                ),
            },
        }
        manifest_id = canonical_sha256(manifest_core)
        manifest_path = output / "manifest.json"
        _write_json(manifest_path, {**manifest_core, "manifest_id": manifest_id})
        result = {
            "schema": "aion.exact-one-electron-wp3-pair-audit-result",
            "version": "2.0.0",
            "status": "executed_unreviewed",
            "manifest_id": manifest_id,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "members": rows,
            "all_gauge_checks_pass": all(row["gauge_checks_pass"] for row in rows),
            "maximum_lower_congruence_relative": max(
                row["maximum_lower_congruence_relative"] for row in rows
            ),
            "maximum_barred_invariance_relative": max(
                row["maximum_barred_invariance_relative"] for row in rows
            ),
            "maximum_generalized_spectrum_residual_hartree": max(
                row["maximum_generalized_spectrum_residual_hartree"] for row in rows
            ),
            "array_semantic_sha256": array_hashes,
            "artifact": {"path": str(arrays_path), "sha256": _sha256(arrays_path)},
            "review": {"reviewed": False, "reviewer": None, "decision": "pending"},
        }
        result_path = output / "result.json"
        _write_json(result_path, result)
        completed = {
            "status": "authenticated_executed_unreviewed",
            "manifest_id": manifest_id,
            "manifest_sha256": _sha256(manifest_path),
            "result_sha256": _sha256(result_path),
            "arrays_sha256": _sha256(arrays_path),
            "member_count": len(rows),
        }
        _write_json(completed_path, completed)
        print(f"audit={output}", flush=True)
        print(f"members={len(rows)}", flush=True)
        print(f"all_gauge_checks_pass={result['all_gauge_checks_pass']}", flush=True)


if __name__ == "__main__":
    main()
