#!/usr/bin/env python3
"""Refine and decompose the NQ8 CO complete-action source derivative."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from aion.config import BackendConfig
from aion.electromagnetism import (
    GaussianVectorPotentialVariation,
    PerturbedVectorPotential,
    UniformMagneticField,
    UniformMagneticSourceSample,
)
from aion.electronic_structure import (
    AOGridPolicy,
    DependencyVersions,
    WilsonStationaryBranch,
    evaluate_exact_static_wilson_grid_one_electron_action,
    evaluate_nonlinear_weak_current_pairing,
    evaluate_static_nonlinear_wilson_grid_action,
    prepare_ao_quadrature,
    prepare_exact_wilson_dynamic_sample,
    prepare_exact_wilson_stationary_factory,
    prepare_pyscf_reference,
)
from aion.formulations import EOMTriple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_chapter13_nq8_gga_transfer as campaign

_STEPS = (3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5)
_REPEATED_STEPS = frozenset((3.0e-3, 1.0e-3))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _float(value: object) -> float:
    return float(np.asarray(value).real)


def _authenticate(raw: Path) -> dict[str, str]:
    completed_path = raw / "completed.json"
    provenance_path = raw / "provenance.json"
    result_path = raw / "result.json"
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if completed["status"] != "executed_unreviewed":
        raise RuntimeError("NQ8 raw campaign does not have the expected status")
    if completed["result_sha256"] != _sha256(result_path):
        raise RuntimeError("NQ8 result hash mismatch")
    if completed["provenance_sha256"] != _sha256(provenance_path):
        raise RuntimeError("NQ8 provenance hash mismatch")
    for relative, expected in provenance["artifacts_sha256"].items():
        path = raw / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"NQ8 artifact hash mismatch: {path}")
    return {
        "completed_sha256": _sha256(completed_path),
        "provenance_sha256": _sha256(provenance_path),
        "result_sha256": _sha256(result_path),
    }


def _action_components(action: Any) -> dict[str, float]:
    return {
        "one_electron_action": _float(action.one_electron_action.total),
        "negative_hartree_energy": -_float(action.hartree.energy),
        "negative_xc_energy": (
            0.0
            if action.exchange_correlation is None
            else -_float(action.exchange_correlation.energy)
        ),
        "electronic_action": _float(action.electronic_action_value),
    }


def _central_components(
    plus: dict[str, float],
    minus: dict[str, float],
    step: float,
) -> dict[str, float]:
    return {
        name: (plus[name] - minus[name]) / (2.0 * step)
        for name in plus
    }


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    raw = arguments.raw.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    accepted_raw = _authenticate(raw)
    started = perf_counter()
    timestamp = datetime.now(UTC).isoformat()

    reference = prepare_pyscf_reference(
        campaign._config("co", output / "co.probe.reference.h5", timestamp)
    )
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(campaign._GRID_LEVEL),
        block_size=campaign._BLOCK_SIZE,
    )
    factory = prepare_exact_wilson_stationary_factory(
        quadrature,
        auxiliary_basis=campaign._AUXILIARY_BASIS,
        functional=campaign._FUNCTIONAL,
    )
    static_source = UniformMagneticSourceSample(
        time_au=0.0,
        field=UniformMagneticField((0.0, 0.0, campaign._FIELD_AU)),
        origin_au=campaign._GAUGE_ORIGIN,
    )
    dynamic = prepare_exact_wilson_dynamic_sample(
        factory,
        static_source,
        WilsonStationaryBranch.KOHN_SHAM_GGA,
    )
    with np.load(raw / "checkpoints/co.npz", allow_pickle=False) as arrays:
        density = np.asarray(arrays["field_density"])
    grid = evaluate_exact_static_wilson_grid_one_electron_action(
        quadrature,
        dynamic.model.gauge,
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
        dynamic.model,
        dynamic.one_electron,
        density,
        velocity,
        variation,
        one_electron_triple=base,
    )
    assert analytic.exchange_correlation is not None
    analytic_components = {
        "one_electron_action": _float(
            analytic.total_pairing - analytic.closure_pairing
        ),
        "negative_hartree_energy": -_float(
            analytic.hartree.source_energy_direction
        ),
        "negative_xc_energy": -_float(
            analytic.exchange_correlation.source_energy_direction
        ),
        "electronic_action": _float(analytic.total_pairing),
    }

    rows: list[dict[str, Any]] = []
    for step in _STEPS:
        repeats = 3 if step in _REPEATED_STEPS else 1
        derivatives: list[dict[str, float]] = []
        for _ in range(repeats):
            signs: dict[float, dict[str, float]] = {}
            for sign in (1.0, -1.0):
                action = evaluate_static_nonlinear_wilson_grid_action(
                    dynamic.model,
                    density,
                    velocity,
                    PerturbedVectorPotential(
                        dynamic.model.gauge,
                        variation,
                        sign * step,
                    ),
                )
                signs[sign] = _action_components(action)
            derivatives.append(_central_components(signs[1.0], signs[-1.0], step))
        row: dict[str, Any] = {"step": step, "repeats": repeats}
        for component, analytic_value in analytic_components.items():
            values = np.asarray([item[component] for item in derivatives])
            row[component] = {
                "analytic": analytic_value,
                "finite_mean": float(np.mean(values)),
                "finite_minimum": float(np.min(values)),
                "finite_maximum": float(np.max(values)),
                "repeat_range": float(np.ptp(values)),
                "absolute_residual": abs(float(np.mean(values)) - analytic_value),
            }
        rows.append(row)
        print(
            f"step {step:.1e}: total residual "
            f"{row['electronic_action']['absolute_residual']:.3e}",
            flush=True,
        )

    repo = Path(__file__).resolve().parents[1]
    result = {
        "schema": "aion.chapter13.nq8.co-source-refinement.v1",
        "status": "executed_unreviewed",
        "raw_campaign": str(raw),
        "accepted_raw_hashes": accepted_raw,
        "functional": campaign._FUNCTIONAL,
        "realization": campaign._REALIZATION,
        "system": "CO",
        "basis": campaign._BASIS,
        "grid_level": campaign._GRID_LEVEL,
        "thread_limits": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "rows": rows,
        "elapsed_seconds": perf_counter() - started,
    }
    result_path = output / "result.json"
    _write_json(result_path, result)
    provenance = {
        "schema": "aion.chapter13.nq8.co-source-refinement.provenance.v1",
        "timestamp_utc": timestamp,
        "command": " ".join(
            shlex.quote(value) for value in (sys.executable, *sys.argv)
        ),
        "repository": str(repo),
        "git_head": _git(repo, "rev-parse", "HEAD").strip(),
        "git_status_porcelain": _git(
            repo, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines(),
        "host": platform.node(),
        "dependencies": DependencyVersions.current().as_mapping(),
        "result_sha256": _sha256(result_path),
    }
    provenance_path = output / "provenance.json"
    _write_json(provenance_path, provenance)
    completed = {
        "schema": "aion.chapter13.nq8.co-source-refinement.completed.v1",
        "status": "executed_unreviewed",
        "result_sha256": _sha256(result_path),
        "provenance_sha256": _sha256(provenance_path),
    }
    _write_json(output / "completed.json", completed)
    print(json.dumps({"output": str(output), **completed}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
