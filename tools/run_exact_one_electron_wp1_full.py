#!/usr/bin/env python3
"""Execute the complete zero-field exact one-electron WP1 qualification."""

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
    AOPruningKind,
    atom_pair_block_residuals,
    evaluate_zero_field_one_electron,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_FIXTURE_DIRECTORY = _REPOSITORY / "tests/fixtures/exact_one_electron"
_FIXTURES = {
    "hh_sto3g": _FIXTURE_DIRECTORY / "hh_sto3g.fixture.json",
    "oh_sto3g": _FIXTURE_DIRECTORY / "oh_sto3g.fixture.json",
}
_LOCK = _REPOSITORY / "conda-linux-64.lock"
_DEFAULT_OUTPUT_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "exact_one_electron_qualification"
)
_LEVELS = (0, 1, 2, 3, 4, 5)
_PRUNINGS = (AOPruningKind.NONE, AOPruningKind.NWCHEM)
_FAMILIES = ("overlap", "kinetic", "nuclear_attraction", "mechanical")
_ALL_FAMILIES = (*_FAMILIES, "canonical_momentum")
_BLOCK_SIZE = 1024
_RELATIVE_MATRIX_TARGET = 1.0e-5
_ALGEBRAIC_FLOOR_MULTIPLIER = 4.0
_ROUNDOFF_PREFACTOR = 64.0
_SIGN_DISCRIMINATION_MINIMUM = 1.0e3


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


def _config(fixture: dict[str, Any]) -> OneElectronReferenceConfig:
    values = fixture["config"]
    return OneElectronReferenceConfig(
        atoms=tuple(
            AtomConfig(atom["symbol"], tuple(atom["position_au"]))
            for atom in values["atoms"]
        ),
        basis=values["basis"],
        electromagnetic_origin=ElectromagneticOrigin(
            tuple(values["electromagnetic_origin_au"])
        ),
    )


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
    system_name: str,
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
            "system_name": system_name,
            "fixture_schema": fixture["schema"],
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
            "scalar_potential": "all_electron_local_nuclear_attraction",
            "vector_potential": "zero",
            "approximation_level": "bare",
            "retained_sectors": [
                "field_free_overlap",
                "field_free_kinetic",
                "field_free_nuclear_attraction",
                "field_free_mechanical",
                "field_free_canonical_momentum",
            ],
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
            "canonical_momentum": "p=-i*hbar*grad_on_ket",
            "atom_pair_blocks": "all_ordered_bra_atom_ket_atom_blocks",
        },
        "tolerances": {
            "relative_frobenius": _RELATIVE_MATRIX_TARGET,
            "algebraic_floor_multiplier": _ALGEBRAIC_FLOOR_MULTIPLIER,
            "roundoff_prefactor": _ROUNDOFF_PREFACTOR,
            "momentum_sign_discrimination_ratio": _SIGN_DISCRIMINATION_MINIMUM,
        },
        "input_hashes": input_hashes,
    }


def _comparisons(result: Any) -> dict[str, Any]:
    return {name: getattr(result, name) for name in _ALL_FAMILIES}


def _block_arrays(result: Any, reference: Any) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, comparison in _comparisons(result).items():
        blocks = atom_pair_block_residuals(
            comparison,
            reference.anchor_topology.ao_to_atom,
        )
        arrays[f"{name}_pair_indices"] = blocks.pair_indices
        arrays[f"{name}_pair_absolute_frobenius"] = blocks.absolute_frobenius
        arrays[f"{name}_pair_relative_frobenius"] = blocks.relative_frobenius
        arrays[f"{name}_pair_max_absolute_element"] = blocks.max_absolute_element
    return arrays


def _result_record(
    *,
    manifest_id: str,
    arrays_path: Path,
    result: Any,
    pair_arrays: dict[str, np.ndarray],
) -> dict[str, object]:
    matrices: list[dict[str, object]] = []
    residuals: list[dict[str, object]] = []
    definitions = {
        "overlap": ("1", "overlap", "AO value-product integral"),
        "kinetic": (
            "hartree",
            "energy",
            "weak-form hbar^2/(2m) AO-gradient inner product",
        ),
        "nuclear_attraction": (
            "hartree",
            "energy",
            "AO value-product integral of -sum_A Z_A/|r-R_A|",
        ),
        "mechanical": ("hartree", "energy", "kinetic plus nuclear attraction"),
        "canonical_momentum": (
            "atomic_unit_of_momentum",
            "momentum",
            "-i*hbar times ket-AO derivative integral",
        ),
    }
    for name, comparison in _comparisons(result).items():
        unit, dimension, definition = definitions[name]
        matrices.extend(
            (
                _array_record(
                    f"{name}/analytic",
                    comparison.analytic,
                    unit=unit,
                    physical_dimension=dimension,
                    definition=f"independent PySCF/libcint {name}",
                ),
                _array_record(
                    f"{name}/quadrature",
                    comparison.quadrature,
                    unit=unit,
                    physical_dimension=dimension,
                    definition=definition,
                ),
            )
        )
        pair_relative = pair_arrays[f"{name}_pair_relative_frobenius"]
        pair_absolute = pair_arrays[f"{name}_pair_absolute_frobenius"]
        residuals.extend(
            (
                _residual(
                    f"{name}/absolute_frobenius",
                    comparison.absolute_frobenius_residual,
                    normalization="none",
                    reference=f"analytic_libcint_{name}",
                    tolerance=None,
                ),
                _residual(
                    f"{name}/relative_frobenius",
                    comparison.relative_frobenius_residual,
                    normalization="max(1, Frobenius norm of analytic libcint matrix)",
                    reference=f"analytic_libcint_{name}",
                    tolerance=_RELATIVE_MATRIX_TARGET,
                ),
                _residual(
                    f"{name}/hermiticity_relative_frobenius",
                    comparison.hermiticity_residual,
                    normalization="max(1, Frobenius norm of quadrature matrix)",
                    reference="quadrature_matrix_adjoint",
                    tolerance=None,
                ),
                _residual(
                    f"{name}/max_absolute_element",
                    comparison.max_absolute_element_residual,
                    normalization="none",
                    reference=f"analytic_libcint_{name}",
                    tolerance=None,
                ),
                _residual(
                    f"{name}/max_atom_pair_absolute_frobenius",
                    float(np.max(pair_absolute)),
                    normalization="none",
                    reference=f"analytic_libcint_{name}_ordered_atom_pair_blocks",
                    tolerance=None,
                ),
                _residual(
                    f"{name}/max_atom_pair_relative_frobenius",
                    float(np.max(pair_relative)),
                    normalization="max(1, Frobenius norm of analytic atom-pair block)",
                    reference=f"analytic_libcint_{name}_ordered_atom_pair_blocks",
                    tolerance=_RELATIVE_MATRIX_TARGET,
                ),
            )
        )
    for axis, value in enumerate(
        result.canonical_momentum.component_relative_frobenius_residuals
    ):
        residuals.append(
            _residual(
                f"canonical_momentum/component_{'xyz'[axis]}_relative_frobenius",
                float(value),
                normalization="max(1, Frobenius norm of analytic Cartesian component)",
                reference="analytic_libcint_canonical_momentum",
                tolerance=_RELATIVE_MATRIX_TARGET,
            )
        )
    correct = result.canonical_momentum.relative_frobenius_residual
    residuals.extend(
        (
            _residual(
                "canonical_momentum/opposite_sign_relative_frobenius",
                result.opposite_momentum_sign_relative_residual,
                normalization="max(1, Frobenius norm of analytic canonical momentum)",
                reference="negative_analytic_libcint_canonical_momentum",
                tolerance=None,
            ),
            _residual(
                "canonical_momentum/sign_discrimination_ratio",
                result.opposite_momentum_sign_relative_residual
                / max(correct, np.finfo(np.float64).tiny),
                normalization="opposite-sign residual divided by correct-sign residual",
                reference="ket_derivative_sign_test",
                tolerance=None,
            ),
        )
    )
    return {
        "schema": "aion.exact-one-electron-matrix-result",
        "version": "1.0.0",
        "status": "executed_unreviewed",
        "manifest_id": manifest_id,
        "matrices": matrices,
        "residuals": residuals,
        "artifact": {"path": str(arrays_path), "sha256": _file_sha256(arrays_path)},
        "review": {"reviewed": False, "reviewer": None, "decision": "pending"},
    }


def _run_member(
    *,
    execution_directory: Path,
    timestamp_utc: str,
    branch: str,
    commit: str,
    dirty: bool,
    system_name: str,
    fixture: dict[str, Any],
    reference: Any,
    pruning: AOPruningKind,
    level: int,
    input_hashes: dict[str, str],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(level, pruning=pruning),
        block_size=_BLOCK_SIZE,
    )
    result = evaluate_zero_field_one_electron(quadrature)
    comparisons = _comparisons(result)
    pair_arrays = _block_arrays(result, reference)
    member_directory = (
        execution_directory
        / system_name
        / f"pruning_{pruning.value}"
        / f"grid_level_{level}"
    )
    member_directory.mkdir(parents=True)
    arrays_path = member_directory / "arrays.npz"
    arrays: dict[str, np.ndarray] = {
        "grid_coordinates_au": quadrature.grid.coordinates_au,
        "grid_weights_au": quadrature.grid.weights_au,
    }
    for name, comparison in comparisons.items():
        arrays[f"{name}_analytic"] = comparison.analytic
        arrays[f"{name}_quadrature"] = comparison.quadrature
    arrays.update(pair_arrays)
    np.savez(arrays_path, **arrays)

    manifest_core = _manifest_without_id(
        timestamp_utc=timestamp_utc,
        branch=branch,
        commit=commit,
        dirty=dirty,
        system_name=system_name,
        fixture=fixture,
        reference=reference,
        quadrature=quadrature,
        input_hashes=input_hashes,
    )
    manifest_id = canonical_sha256(manifest_core)
    _write_json(
        member_directory / "manifest.json",
        {**manifest_core, "manifest_id": manifest_id},
    )
    _write_json(
        member_directory / "result.json",
        _result_record(
            manifest_id=manifest_id,
            arrays_path=arrays_path,
            result=result,
            pair_arrays=pair_arrays,
        ),
    )
    row: dict[str, object] = {
        "system": system_name,
        "pruning": pruning.value,
        "level": level,
        "npoints": result.grid_npoints,
        "manifest_id": manifest_id,
        "relative_frobenius": {
            name: comparison.relative_frobenius_residual
            for name, comparison in comparisons.items()
        },
        "absolute_frobenius": {
            name: comparison.absolute_frobenius_residual
            for name, comparison in comparisons.items()
        },
        "hermiticity": {
            name: comparison.hermiticity_residual
            for name, comparison in comparisons.items()
        },
        "max_absolute_element": {
            name: comparison.max_absolute_element_residual
            for name, comparison in comparisons.items()
        },
        "max_atom_pair_relative_frobenius": {
            name: float(np.max(pair_arrays[f"{name}_pair_relative_frobenius"]))
            for name in _ALL_FAMILIES
        },
        "opposite_momentum_sign_relative_frobenius": (
            result.opposite_momentum_sign_relative_residual
        ),
        "momentum_sign_discrimination_ratio": (
            result.opposite_momentum_sign_relative_residual
            / max(
                result.canonical_momentum.relative_frobenius_residual,
                np.finfo(np.float64).tiny,
            )
        ),
    }
    quadrature_matrices = {
        name: np.asarray(comparison.quadrature) for name, comparison in comparisons.items()
    }
    return row, quadrature_matrices


def _finalize_sequences(
    rows: list[dict[str, object]],
    matrices: dict[tuple[str, str, int], dict[str, np.ndarray]],
    references: dict[str, Any],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    sequences: list[dict[str, object]] = []
    selected: dict[str, object] = {}
    for system_name, reference in references.items():
        selected[system_name] = {
            "default": {"level": 4, "pruning": "none"},
            "refinement": {"level": 5, "pruning": "none"},
            "alternative_pruning": {"level": 4, "pruning": "nwchem"},
        }
        overlap_condition = float(np.linalg.cond(reference.core_operators.overlap))
        roundoff_bound = float(
            _ROUNDOFF_PREFACTOR
            * np.finfo(np.float64).eps
            * reference.core_operators.nao
            * max(1.0, overlap_condition)
        )
        for pruning in _PRUNINGS:
            matching = sorted(
                (
                    row
                    for row in rows
                    if row["system"] == system_name
                    and row["pruning"] == pruning.value
                ),
                key=lambda row: int(row["level"]),
            )
            previous: dict[str, np.ndarray] | None = None
            for row in matching:
                current = matrices[(system_name, pruning.value, int(row["level"]))]
                successive: dict[str, float | None] = {}
                for name in _ALL_FAMILIES:
                    analytic = getattr(reference.core_operators, name, None)
                    if name == "mechanical":
                        analytic = (
                            reference.core_operators.kinetic
                            + reference.core_operators.nuclear_attraction
                        )
                    scale = max(1.0, float(np.linalg.norm(analytic)))
                    successive[name] = (
                        None
                        if previous is None
                        else float(np.linalg.norm(current[name] - previous[name]) / scale)
                    )
                row["successive_relative_frobenius"] = successive
                previous = current
            finest = matching[-1]
            finest_relative = finest["relative_frobenius"]
            finest_successive = finest["successive_relative_frobenius"]
            finest_hermiticity = finest["hermiticity"]
            coarsest_relative = matching[0]["relative_frobenius"]
            penultimate_successive = matching[-2]["successive_relative_frobenius"]
            floors = {
                name: max(
                    float(finest_relative[name]),
                    float(finest_successive[name]),
                )
                for name in _ALL_FAMILIES
            }
            hermiticity_tolerances = {
                name: max(roundoff_bound, _ALGEBRAIC_FLOOR_MULTIPLIER * floors[name])
                for name in _ALL_FAMILIES
            }
            sequences.append(
                {
                    "system": system_name,
                    "pruning": pruning.value,
                    "levels": list(_LEVELS),
                    "overlap_condition_number": overlap_condition,
                    "roundoff_bound": roundoff_bound,
                    "working_relative_floors": floors,
                    "hermiticity_tolerances": hermiticity_tolerances,
                    "convergence_pass": all(
                        float(finest_relative[name]) < float(coarsest_relative[name])
                        and float(finest_successive[name])
                        < float(penultimate_successive[name])
                        for name in _ALL_FAMILIES
                    ),
                    "final_relative_target_pass": all(
                        float(finest_relative[name]) <= _RELATIVE_MATRIX_TARGET
                        for name in _ALL_FAMILIES
                    ),
                    "floor_target_pass": all(
                        floors[name] <= _RELATIVE_MATRIX_TARGET for name in _ALL_FAMILIES
                    ),
                    "hermiticity_pass": all(
                        float(finest_hermiticity[name]) <= hermiticity_tolerances[name]
                        for name in _ALL_FAMILIES
                    ),
                    "momentum_sign_pass": (
                        float(finest["momentum_sign_discrimination_ratio"])
                        >= _SIGN_DISCRIMINATION_MINIMUM
                    ),
                }
            )
        pruning_comparison: dict[str, float] = {}
        unpruned = matrices[(system_name, AOPruningKind.NONE.value, 4)]
        pruned = matrices[(system_name, AOPruningKind.NWCHEM.value, 4)]
        for name in _ALL_FAMILIES:
            analytic = getattr(reference.core_operators, name, None)
            if name == "mechanical":
                analytic = (
                    reference.core_operators.kinetic
                    + reference.core_operators.nuclear_attraction
                )
            pruning_comparison[name] = float(
                np.linalg.norm(unpruned[name] - pruned[name])
                / max(1.0, float(np.linalg.norm(analytic)))
            )
        selected[system_name]["level4_pruning_comparison"] = pruning_comparison
        selected[system_name]["pruning_comparison_pass"] = all(
            value <= _RELATIVE_MATRIX_TARGET for value in pruning_comparison.values()
        )
        default_row = next(
            row
            for row in rows
            if row["system"] == system_name
            and row["pruning"] == AOPruningKind.NONE.value
            and row["level"] == 4
        )
        selected[system_name]["default_target_pass"] = all(
            float(default_row["relative_frobenius"][name]) <= _RELATIVE_MATRIX_TARGET
            and float(default_row["successive_relative_frobenius"][name])
            <= _RELATIVE_MATRIX_TARGET
            for name in _ALL_FAMILIES
        )
    return sequences, selected


def _report(
    rows: list[dict[str, object]],
    sequences: list[dict[str, object]],
    selected: dict[str, object],
    execution_directory: Path,
) -> str:
    lines = [
        "# Exact one-electron WP1 zero-field qualification",
        "",
        "Status: **numerically executed; G1 user review pending**.",
        "",
        "No finite magnetic field was evaluated. The campaign covers H--H/STO-3G and ",
        "O--H/STO-3G, unpruned and NWChem-pruned grids, and levels 0 through 5.",
        "",
        "The declared normalized target is `1e-5`. Hermiticity is compared with the ",
        "larger of a conditioning-scaled roundoff bound and four times the measured ",
        "matrix-family quadrature floor.",
        "",
    ]
    for system_name in _FIXTURES:
        lines.extend(
            (
                f"## {system_name}",
                "",
                "| pruning | level | points | S | T | Vnuc | K | p |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for row in rows:
            if row["system"] != system_name:
                continue
            values = row["relative_frobenius"]
            lines.append(
                "| {pruning} | {level} | {npoints} | {overlap:.3e} | "
                "{kinetic:.3e} | {nuclear_attraction:.3e} | {mechanical:.3e} | "
                "{canonical_momentum:.3e} |".format(
                    pruning=row["pruning"],
                    level=row["level"],
                    npoints=row["npoints"],
                    **values,
                )
            )
        lines.extend(("", "Working level-5 relative floors:", ""))
        for sequence in sequences:
            if sequence["system"] != system_name:
                continue
            floors = sequence["working_relative_floors"]
            lines.append(
                "- `{pruning}`: S={overlap:.3e}, T={kinetic:.3e}, "
                "Vnuc={nuclear_attraction:.3e}, K={mechanical:.3e}, "
                "p={canonical_momentum:.3e}.".format(
                    pruning=sequence["pruning"],
                    **floors,
                )
            )
        pruning_values = selected[system_name]["level4_pruning_comparison"]
        lines.extend(
            (
                "",
                "Level-4 unpruned versus NWChem-pruned relative differences: "
                "S={overlap:.3e}, T={kinetic:.3e}, Vnuc={nuclear_attraction:.3e}, "
                "K={mechanical:.3e}, p={canonical_momentum:.3e}.".format(
                    **pruning_values
                ),
                "",
            )
        )
    all_sequences_pass = all(
        sequence["final_relative_target_pass"]
        and sequence["floor_target_pass"]
        and sequence["hermiticity_pass"]
        and sequence["momentum_sign_pass"]
        and sequence["convergence_pass"]
        for sequence in sequences
    )
    all_pruning_pass = all(
        bool(value["pruning_comparison_pass"])
        and bool(value["default_target_pass"])
        for value in selected.values()
    )
    lines.extend(
        (
            "## Provisional numerical conclusion",
            "",
            f"All declared sequence checks pass: `{all_sequences_pass}`.  ",
            f"Both level-4 pruning comparisons pass: `{all_pruning_pass}`.",
            "",
            "The proposed default for subsequent static qualification is unpruned level 4; ",
            "unpruned level 5 is its refinement, and NWChem-pruned level 4 is retained as ",
            "the independent pruning check. This ",
            "selection remains provisional until the user accepts G1.",
            "",
            "Raw manifests, results, grids, matrices, and atom-pair residual arrays are at:",
            "",
            f"`{execution_directory}`",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    arguments = _arguments()
    fixtures = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in _FIXTURES.items()
    }
    references = {
        name: prepare_one_electron_ao_reference(_config(fixture))
        for name, fixture in fixtures.items()
    }
    for name, reference in references.items():
        if reference.fingerprint_sha256 != fixtures[name]["reference_fingerprint_sha256"]:
            raise RuntimeError(f"live one-electron reference does not match fixture {name}")

    branch = _git("branch", "--show-current")
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    started = datetime.now(UTC)
    timestamp = started.strftime("%Y%m%dT%H%M%SZ")
    timestamp_utc = started.isoformat().replace("+00:00", "Z")
    execution_directory = arguments.output_root / f"wp1_{timestamp}_{commit[:12]}"
    execution_directory.mkdir(parents=True, exist_ok=False)

    tracked_inputs = (
        *_FIXTURES.values(),
        _LOCK,
        _REPOSITORY / "docs/exact_one_electron_qualification_g0.md",
        _REPOSITORY / "docs/reviews/exact_one_electron_g0_review_20260916.json",
        _REPOSITORY / "docs/schemas/exact_one_electron_run_manifest.schema.json",
        _REPOSITORY / "docs/schemas/exact_one_electron_matrix_result.schema.json",
        _REPOSITORY / "src/aion/electronic_structure/ao_quadrature.py",
        _REPOSITORY / "src/aion/electronic_structure/local_potentials.py",
        _REPOSITORY / "src/aion/electronic_structure/one_electron.py",
        Path(__file__).resolve(),
    )
    input_hashes = {
        str(path.relative_to(_REPOSITORY)): _file_sha256(path) for path in tracked_inputs
    }
    rows: list[dict[str, object]] = []
    matrices: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    for system_name, reference in references.items():
        for pruning in _PRUNINGS:
            for level in _LEVELS:
                row, member_matrices = _run_member(
                    execution_directory=execution_directory,
                    timestamp_utc=timestamp_utc,
                    branch=branch,
                    commit=commit,
                    dirty=dirty,
                    system_name=system_name,
                    fixture=fixtures[system_name],
                    reference=reference,
                    pruning=pruning,
                    level=level,
                    input_hashes=input_hashes,
                )
                rows.append(row)
                matrices[(system_name, pruning.value, level)] = member_matrices

    sequences, selected = _finalize_sequences(rows, matrices, references)
    index = {
        "schema": "aion.exact-one-electron-wp1-index",
        "version": "1.0.0",
        "status": "executed_unreviewed",
        "execution_timestamp_utc": timestamp_utc,
        "declared_tolerances": {
            "relative_frobenius": _RELATIVE_MATRIX_TARGET,
            "algebraic_floor_multiplier": _ALGEBRAIC_FLOOR_MULTIPLIER,
            "roundoff_prefactor": _ROUNDOFF_PREFACTOR,
            "momentum_sign_discrimination_ratio": _SIGN_DISCRIMINATION_MINIMUM,
        },
        "rows": rows,
        "sequences": sequences,
        "selected_grids": selected,
        "review": {"reviewed": False, "reviewer": None, "decision": "pending"},
    }
    _write_json(execution_directory / "execution_index.json", index)
    (execution_directory / "wp1_report.md").write_text(
        _report(rows, sequences, selected, execution_directory),
        encoding="utf-8",
    )
    print(execution_directory)


if __name__ == "__main__":
    main()
