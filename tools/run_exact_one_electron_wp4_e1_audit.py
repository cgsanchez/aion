#!/usr/bin/env python3
"""Refine the WP4 uniform-electric E1 quadrature floor on level-5 grids."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
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
    evaluate_uniform_electric_e1_tensor,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_BLOCK_SIZE = 2048


def _arguments() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution_directory", type=Path)
    return parser.parse_args().execution_directory.resolve()


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


def _config(source_manifest: dict[str, Any]) -> OneElectronReferenceConfig:
    return OneElectronReferenceConfig(
        atoms=tuple(
            AtomConfig(atom["symbol"], tuple(atom["position_au"]))
            for atom in source_manifest["system"]["atoms"]
        ),
        basis=source_manifest["system"]["basis"],
        electromagnetic_origin=ElectromagneticOrigin((0.0, 0.0, 0.0)),
    )


def main() -> None:
    execution = _arguments()
    index_path = execution / "execution_index.json"
    index = _json(index_path)
    if index["status"] != "authenticated_executed_unreviewed":
        raise ValueError("WP4 execution is not complete")
    output = execution / "e1_refinement_audit"
    output.mkdir(parents=True, exist_ok=True)
    members = []
    arrays: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for position, record in enumerate(index["completed_members"], start=1):
        member = Path(record["member_directory"])
        manifest = _json(member / "manifest.json")
        source_manifest = _json(Path(manifest["source_wp3"]["directory"]) / "manifest.json")
        reference = prepare_one_electron_ao_reference(_config(source_manifest))
        quadrature = prepare_ao_quadrature(
            reference,
            BackendConfig(),
            grid_policy=AOGridPolicy.qualification(5),
            block_size=_BLOCK_SIZE,
        )
        e1 = evaluate_uniform_electric_e1_tensor(quadrature)
        analytic = np.asarray(quadrature.backend.to_host(e1.central_dipoles))
        level5 = np.asarray(quadrature.backend.to_host(e1.quadrature_central_dipoles))
        with np.load(member / "arrays.npz", allow_pickle=False) as level4_arrays:
            level4 = np.asarray(level4_arrays["e1__quadrature_central_dipoles"])
            stored_analytic = np.asarray(level4_arrays["e1__central_dipoles"])
        np.testing.assert_array_equal(analytic, stored_analytic)
        level4_error = float(np.linalg.norm(level4 - analytic))
        level5_error = float(np.linalg.norm(level5 - analytic))
        refinement_change = float(np.linalg.norm(level5 - level4))
        roundoff = 64.0 * np.finfo(np.float64).eps * max(
            1.0, float(np.linalg.norm(analytic))
        )
        floor = max(level4_error, level5_error, refinement_change, roundoff)
        scale = max(1.0, float(np.linalg.norm(analytic)))
        key = (
            f"{record['system']}__{record['basis']}__"
            f"scale_{float(record['bond_scale']):.6f}"
        )
        arrays[f"{key}__level5_central_dipoles"] = level5
        members.append(
            {
                "system": record["system"],
                "basis": record["basis"],
                "bond_scale": float(record["bond_scale"]),
                "reference_fingerprint_sha256": reference.fingerprint_sha256,
                "level4_grid_fingerprint_sha256": manifest["grid_fingerprint_sha256"],
                "level5_grid_fingerprint_sha256": quadrature.grid.fingerprint_sha256,
                "level4_absolute_residual": level4_error,
                "level5_absolute_residual": level5_error,
                "level5_vs_level4_absolute": refinement_change,
                "refined_working_floor": floor,
                "level4_relative_residual": level4_error / scale,
                "level5_relative_residual": level5_error / scale,
                "level5_to_level4_residual_ratio": level5_error
                / max(level4_error, np.finfo(np.float64).tiny),
                "level5_residual_to_refined_floor": level5_error / floor,
            }
        )
        print(
            f"finish {position}/{len(index['completed_members'])} "
            f"{record['system']} {record['basis']} {record['bond_scale']}",
            flush=True,
        )

    manifest_core = {
        "schema": "aion.exact-one-electron-wp4-e1-refinement-audit-manifest",
        "version": "1.0.0",
        "status": "executed_unreviewed",
        "source_execution_directory": str(execution),
        "source_execution_index_sha256": _sha256(index_path),
        "grid_levels": [4, 5],
        "block_size": _BLOCK_SIZE,
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "electric_evaluator_sha256": _sha256(
            _REPOSITORY / "src/aion/electronic_structure/electric_matrices.py"
        ),
    }
    manifest_id = canonical_sha256(manifest_core)
    manifest_path = output / "manifest.json"
    result_path = output / "result.json"
    arrays_path = output / "arrays.npz"
    _write_json(manifest_path, {**manifest_core, "manifest_id": manifest_id})
    _write_npz(arrays_path, arrays)
    result = {
        "schema": "aion.exact-one-electron-wp4-e1-refinement-audit-result",
        "version": "1.0.0",
        "status": "executed_unreviewed",
        "manifest_id": manifest_id,
        "members": members,
        "member_count": len(members),
        "maximum_level4_relative_residual": max(
            row["level4_relative_residual"] for row in members
        ),
        "maximum_level5_relative_residual": max(
            row["level5_relative_residual"] for row in members
        ),
        "maximum_level5_to_level4_residual_ratio": max(
            row["level5_to_level4_residual_ratio"] for row in members
        ),
        "maximum_level5_residual_to_refined_floor": max(
            row["level5_residual_to_refined_floor"] for row in members
        ),
        "array_semantic_sha256": {
            name: canonical_sha256(value) for name, value in arrays.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(result_path, result)
    _write_json(
        output / "completed.json",
        {
            "status": "authenticated_executed_unreviewed",
            "manifest_id": manifest_id,
            "manifest_sha256": _sha256(manifest_path),
            "result_sha256": _sha256(result_path),
            "arrays_sha256": _sha256(arrays_path),
        },
    )
    print(f"audit={output}")
    print(f"members={len(members)}")
    print(f"maximum_level5_relative_residual={result['maximum_level5_relative_residual']:.6e}")


if __name__ == "__main__":
    main()
