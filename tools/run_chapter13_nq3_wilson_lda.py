#!/usr/bin/env python3
"""Execute the Chapter 13 NQ3 variational Wilson-LDA campaign."""

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
from dataclasses import dataclass
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
)
from aion.electronic_structure import (
    AOGridPolicy,
    DependencyVersions,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
    prepare_wilson_lda,
    reconstruct_mean_field,
)

_FUNCTIONAL = "lda,vwn"
_FIELD = (0.013, -0.009, 0.017)
_FIELD_DIRECTION = (-0.4, 0.7, 0.2)
_GAUGE_ORIGIN = (0.17, -0.31, 0.23)
_LANDAU_ORIGIN = (-0.21, 0.37, -0.16)
_GRID_LEVELS = (1, 2, 3, 4, 5)
_STEPS = (
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


@dataclass(frozen=True, slots=True)
class _AffineSourceCurve:
    base: Any
    direction: Any
    scale: float

    def straight_line_integrals(
        self,
        starts_au: object,
        ends_au: object,
        backend: Any,
    ) -> Any:
        return self.base.straight_line_integrals(
            starts_au, ends_au, backend
        ) + self.scale * self.direction.straight_line_integrals(
            starts_au, ends_au, backend
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


def _relative(value: object, reference: object) -> float:
    value_array = np.asarray(value)
    reference_array = np.asarray(reference)
    return float(
        np.linalg.norm(value_array - reference_array)
        / max(1.0, float(np.linalg.norm(reference_array)))
    )


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
            functional=_FUNCTIONAL,
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
            label=f"chapter13-nq3-{system}-lda-vwn",
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


def _same_grid_pyscf(reference: Any, quadrature: Any) -> dict[str, Any]:
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
        reference.ground_state.density,
    )
    return {
        "electron_count": float(electron_count),
        "energy": float(energy),
        "lower": np.asarray(lower),
    }


def _grid_rows(system: str, reference: Any) -> list[dict[str, Any]]:
    zero = AffineMagneticGauge(UniformMagneticField((0.0, 0.0, 0.0)))
    density = reference.ground_state.density.astype(np.complex128)
    rows: list[dict[str, Any]] = []
    lower_matrices: list[np.ndarray] = []
    for level in _GRID_LEVELS:
        quadrature = prepare_ao_quadrature(
            reference,
            BackendConfig(),
            grid_policy=AOGridPolicy.qualification(level),
            block_size=1024,
        )
        evaluator = prepare_wilson_lda(quadrature, _FUNCTIONAL)
        result = evaluator.evaluate(density, zero)
        oracle = _same_grid_pyscf(reference, quadrature)
        lower = np.asarray(result.lower_xc_matrix)
        lower_matrices.append(lower)
        rows.append(
            {
                "system": system,
                "functional": _FUNCTIONAL,
                "level": level,
                "points": quadrature.grid.npoints,
                "grid_fingerprint_sha256": quadrature.grid.fingerprint_sha256,
                "energy_au": float(result.energy),
                "pyscf_same_grid_energy_au": oracle["energy"],
                "same_grid_energy_absolute_residual_au": abs(
                    float(result.energy) - oracle["energy"]
                ),
                "same_grid_lower_relative_residual": _relative(lower, oracle["lower"]),
                "same_grid_electron_count_absolute_residual": abs(
                    float(result.electron_count_grid) - oracle["electron_count"]
                ),
                "lower_hermiticity_residual": result.lower_hermiticity_residual,
                "density_imaginary_max_abs": result.density_imaginary_max_abs,
                "density_real_minimum": result.density_real_minimum,
                "energy_to_finest_absolute_residual_au": 0.0,
                "lower_to_finest_relative_residual": 0.0,
            }
        )
    finest_energy = float(rows[-1]["energy_au"])
    finest_lower = lower_matrices[-1]
    for row, lower in zip(rows, lower_matrices, strict=True):
        row["energy_to_finest_absolute_residual_au"] = abs(
            float(row["energy_au"]) - finest_energy
        )
        row["lower_to_finest_relative_residual"] = _relative(lower, finest_lower)
    return rows


def _matter_rows(
    system: str,
    coefficients: np.ndarray,
    occupations: np.ndarray,
    evaluator: Any,
    gauge: Any,
    base: Any,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=coefficients.shape)
    rows: list[dict[str, Any]] = []
    for label, direction in (
        ("real", raw.astype(np.complex128)),
        ("imaginary", 1j * raw),
    ):
        direction /= np.linalg.norm(direction)
        density_direction = np.einsum(
            "mi,i,ni->mn",
            direction,
            occupations,
            coefficients.conj(),
            optimize=True,
        ) + np.einsum(
            "mi,i,ni->mn",
            coefficients,
            occupations,
            direction.conj(),
            optimize=True,
        )
        analytic = float(
            np.einsum(
                "ij,ji->",
                np.asarray(base.lower_xc_matrix),
                density_direction,
                optimize=True,
            ).real
        )
        for step in _STEPS:
            energies: list[float] = []
            for sign in (1.0, -1.0):
                displaced = coefficients + sign * step * direction
                result = evaluator.evaluate(
                    _coefficient_density(displaced, occupations),
                    gauge,
                )
                energies.append(float(result.energy))
            central = (energies[0] - energies[1]) / (2.0 * step)
            rows.append(
                {
                    "system": system,
                    "kind": "matter",
                    "direction": label,
                    "step": step,
                    "analytic_direction_au": analytic,
                    "central_absolute_error_au": abs(central - analytic),
                    "central_relative_error": abs(central - analytic)
                    / max(1.0, abs(analytic)),
                }
            )
    return rows


def _source_rows(
    system: str,
    density: np.ndarray,
    evaluator: Any,
    gauge: Any,
    directions: tuple[tuple[str, Any], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, direction in directions:
        result = evaluator.evaluate(density, gauge, source_direction=direction)
        assert result.source_energy_direction is not None
        assert result.source_density_imaginary_max_abs is not None
        analytic = float(result.source_energy_direction)
        for step in _STEPS:
            plus = evaluator.evaluate(density, _AffineSourceCurve(gauge, direction, step))
            minus = evaluator.evaluate(density, _AffineSourceCurve(gauge, direction, -step))
            central = (float(plus.energy) - float(minus.energy)) / (2.0 * step)
            rows.append(
                {
                    "system": system,
                    "kind": "source_fixed_coefficients",
                    "direction": label,
                    "step": step,
                    "analytic_direction_au": analytic,
                    "central_absolute_error_au": abs(central - analytic),
                    "central_relative_error": abs(central - analytic)
                    / max(1.0, abs(analytic)),
                    "source_density_imaginary_max_abs": (
                        result.source_density_imaginary_max_abs
                    ),
                }
            )
    return rows


def _finite_field_result(system: str, reference: Any) -> dict[str, Any]:
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(5),
        block_size=1024,
    )
    evaluator = prepare_wilson_lda(quadrature, _FUNCTIONAL)
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
    base = evaluator.evaluate(density, symmetric)
    matter = _matter_rows(
        system,
        coefficients,
        occupations,
        evaluator,
        symmetric,
        base,
        seed=4801 if system == "h2" else 4802,
    )
    source = _source_rows(
        system,
        density,
        evaluator,
        symmetric,
        (
            ("physical_magnetic", physical_direction),
            ("pure_gauge", pure_gauge_direction),
        ),
    )

    anchors = reference.core_operators.nuclei.coordinates_au[
        reference.anchor_topology.ao_to_atom
    ]
    gauge_function = np.asarray(
        affine_gauge_difference_potential(
            landau,
            symmetric,
            anchors,
            quadrature.backend,
        )
    )
    unitary = np.exp(-1j * gauge_function)
    landau_density = unitary[:, None] * density * unitary.conj()[None, :]
    landau_result = evaluator.evaluate(landau_density, landau)
    predicted_landau_lower = (
        unitary[:, None]
        * np.asarray(base.lower_xc_matrix)
        * unitary.conj()[None, :]
    )

    rng = np.random.default_rng(5801 if system == "h2" else 5802)
    dimension = density.shape[0]
    raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    change = np.eye(dimension, dtype=np.complex128) + 0.05 * raw / np.linalg.norm(raw)
    inverse = np.linalg.inv(change)
    transformed_density = inverse @ density @ inverse.conj().T
    transformed = evaluator.evaluate(
        transformed_density,
        symmetric,
        coefficient_frame=change,
    )
    predicted_transformed_lower = (
        change.conj().T @ np.asarray(base.lower_xc_matrix) @ change
    )

    return {
        "grid_level": 5,
        "grid_points": quadrature.grid.npoints,
        "provenance": {
            name: getattr(evaluator.provenance, name)
            for name in evaluator.provenance.__dataclass_fields__
        },
        "base": {
            "energy_au": float(base.energy),
            "electron_count_grid": float(base.electron_count_grid),
            "lower_hermiticity_residual": base.lower_hermiticity_residual,
            "density_imaginary_max_abs": base.density_imaginary_max_abs,
            "density_real_minimum": base.density_real_minimum,
        },
        "invariance": {
            "electromagnetic_gauge": {
                "energy_absolute_residual_au": abs(
                    float(landau_result.energy) - float(base.energy)
                ),
                "density_relative_residual": _relative(
                    landau_result.density, base.density
                ),
                "lower_covariance_relative_residual": _relative(
                    landau_result.lower_xc_matrix, predicted_landau_lower
                ),
            },
            "coefficient_frame": {
                "change_condition_number": float(np.linalg.cond(change)),
                "energy_absolute_residual_au": abs(
                    float(transformed.energy) - float(base.energy)
                ),
                "density_relative_residual": _relative(
                    transformed.density, base.density
                ),
                "lower_congruence_relative_residual": _relative(
                    transformed.lower_xc_matrix, predicted_transformed_lower
                ),
            },
        },
        "matter_derivatives": matter,
        "source_derivatives": source,
    }


def _system_result(system: str, output: Path, timestamp: str) -> dict[str, Any]:
    config = _configuration(system, output / f"{system}.reference.h5", timestamp)
    (output / f"{system}.reference.toml").write_text(
        dumps_config(config), encoding="utf-8"
    )
    reference = prepare_pyscf_reference(config)
    reference.save()
    return {
        "system": system,
        "scientific_id": config.scientific_id,
        "reference_fingerprint_sha256": reference.fingerprint_sha256,
        "electron_count": reference.ground_state.electron_count,
        "nao": reference.core_operators.nao,
        "occupations": reference.ground_state.occupations.tolist(),
        "grid_refinement": _grid_rows(system, reference),
        "finite_field": _finite_field_result(system, reference),
    }


def _write_csv(output: Path, name: str, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with (output / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_tables(output: Path, systems: list[dict[str, Any]]) -> None:
    _write_csv(
        output,
        "grid_refinement.csv",
        [row for system in systems for row in system["grid_refinement"]],
    )
    _write_csv(
        output,
        "derivative_sequences.csv",
        [
            row
            for system in systems
            for family in ("matter_derivatives", "source_derivatives")
            for row in system["finite_field"][family]
        ],
    )
    _write_csv(
        output,
        "invariance.csv",
        [
            {"system": system["system"], "kind": kind, **values}
            for system in systems
            for kind, values in system["finite_field"]["invariance"].items()
        ],
    )


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
        "schema": "aion.chapter13-nq3-wilson-lda",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "equations": [
            "eq:wilson-hartree-ks-xc-quadrature",
            "eq:wilson-hartree-ks-xc-lda-matrix",
            "eq:wilson-hartree-ks-xc-matter-differential",
            "eq:wilson-hartree-ks-xc-source-differential",
        ],
        "realization": {
            "systems": ["h2", "lih"],
            "orbital_basis": "sto-3g",
            "functional": _FUNCTIONAL,
            "functional_family": "pure LDA",
            "functional_scan": False,
            "grid_levels": list(_GRID_LEVELS),
            "grid_pruning": "none",
            "field_au": list(_FIELD),
            "field_direction_au": list(_FIELD_DIRECTION),
            "symmetric_gauge_origin_au": list(_GAUGE_ORIGIN),
            "landau_gauge_origin_au": list(_LANDAU_ORIGIN),
            "charge": -1.0,
            "hbar": 1.0,
            "precision": "float64_complex128",
            "coefficient_history_fixed_in_source_derivatives": True,
            "pointwise_functional_engine": "PySCF/libxc CPU",
            "wilson_and_action_contractions": "Aion selected array backend",
        },
        "matter_and_source_steps": list(_STEPS),
        "systems": systems,
    }
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_tables(output, systems)

    tracked_diff = _git(repo, "diff", "--binary", "HEAD", binary=True)
    assert isinstance(tracked_diff, bytes)
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    assert isinstance(status, str)
    head = _git(repo, "rev-parse", "HEAD")
    assert isinstance(head, str)
    source_paths = (
        repo / "tools/run_chapter13_nq3_wilson_lda.py",
        repo / "src/aion/electronic_structure/wilson_lda.py",
        repo / "src/aion/electronic_structure/wilson_density.py",
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
        "schema": "aion.chapter13-nq3-completed",
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
