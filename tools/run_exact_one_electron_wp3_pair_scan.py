#!/usr/bin/env python3
"""Run resumable WP3 exact-versus-P0 static pair scans."""

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
from aion.electromagnetism import UniformMagneticField
from aion.electronic_structure import (
    AOGridPolicy,
    angular_channel_changes,
    ao_angular_momenta,
    compare_generalized_spectra,
    diagonal_scaled_element_change,
    evaluate_exact_static_magnetic_one_electron_matrices,
    matrix_partition_changes,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
    static_magnetic_diagnostic_models,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_FIXTURE = _REPOSITORY / "tests/fixtures/exact_one_electron/wp3_pair_campaign.fixture.json"
_FORMAL_PLAN = Path(
    "/home/cgs/00_WORK/Projection_Full_Formalism/REVIEW/implementation/"
    "exact_one_electron_numerical_qualification_plan.md"
)
_G2_REVIEW = _REPOSITORY / "docs/reviews/exact_one_electron_g2_review_20260916.json"
_LOCK = _REPOSITORY / "conda-linux-64.lock"
_DEFAULT_OUTPUT_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/exact_one_electron_qualification"
)
_SYSTEMS = ("hh", "oh", "n2", "co")
_BASES = ("sto-3g", "cc-pvdz", "aug-cc-pvdz")
_FAMILIES = ("overlap", "kinetic", "nuclear_attraction", "mechanical")
_MODELS = ("p0", "form_factor_only", "anchored_vector_only", "exact")
_SECTORS = ("pp", "pC", "Cp", "C2")
_BLOCK_SIZE = 2048


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execution-directory", type=Path)
    parser.add_argument("--systems", nargs="+", choices=_SYSTEMS, default=list(_SYSTEMS))
    parser.add_argument("--bases", nargs="+", choices=_BASES, default=list(_BASES))
    parser.add_argument("--bond-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--grid-level", type=int, default=4)
    parser.add_argument("--max-members", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use two magnitudes, disable adaptive points, and mark output non-qualifying",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)  # type: ignore[arg-type]
    os.replace(temporary, path)


def _config(system: dict[str, Any], basis: str, scale: float) -> OneElectronReferenceConfig:
    distance = float(system["reference_distance_bohr"]) * scale
    symbols = tuple(system["atoms"])
    return OneElectronReferenceConfig(
        atoms=(
            AtomConfig(symbols[0], (0.0, 0.0, -0.5 * distance)),
            AtomConfig(symbols[1], (0.0, 0.0, 0.5 * distance)),
        ),
        basis=basis,
        electromagnetic_origin=ElectromagneticOrigin((0.0, 0.0, 0.0)),
    )


def _field_cases(
    directions: dict[str, np.ndarray],
    magnitudes_by_direction: dict[str, set[float]],
    *,
    include_zero: bool,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if include_zero:
        cases.append(
            {
                "id": "zero",
                "direction": "zero",
                "sign": 0,
                "magnitude_au": 0.0,
                "field": UniformMagneticField((0.0, 0.0, 0.0)),
            }
        )
    for direction_name, direction in directions.items():
        for magnitude in sorted(magnitudes_by_direction[direction_name]):
            for sign, sign_name in ((1, "plus"), (-1, "minus")):
                field_vector = sign * magnitude * direction
                cases.append(
                    {
                        "id": f"{direction_name}__{sign_name}__{magnitude:.12e}",
                        "direction": direction_name,
                        "sign": sign,
                        "magnitude_au": magnitude,
                        "field": UniformMagneticField(tuple(field_vector)),
                    }
                )
    return cases


def _matrix_values(value: Any) -> dict[str, np.ndarray]:
    return {
        "overlap": np.asarray(value.overlap),
        "kinetic": np.asarray(value.kinetic),
        "nuclear_attraction": np.asarray(value.nuclear_attraction),
        "mechanical": np.asarray(value.mechanical),
    }


def _relative(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / max(1.0, float(np.linalg.norm(reference))))


def _numerical_floors(zero: Any) -> dict[str, float]:
    values = {
        "overlap": np.asarray(zero.overlap.quadrature_zero - zero.overlap.zero),
        "kinetic": np.asarray(zero.kinetic.quadrature_zero - zero.kinetic.zero),
        "nuclear_attraction": np.asarray(
            zero.nuclear_attraction.quadrature_zero - zero.nuclear_attraction.zero
        ),
    }
    values["mechanical"] = values["kinetic"] + values["nuclear_attraction"]
    return {
        name: max(
            float(np.linalg.norm(value)),
            64.0 * np.finfo(np.float64).eps * max(1.0, float(np.linalg.norm(value))),
        )
        for name, value in values.items()
    }


def _case_analysis(
    case: dict[str, Any],
    result: Any,
    reference: Any,
    floors: dict[str, float],
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    model_result = static_magnetic_diagnostic_models(result)
    lower = {name: _matrix_values(getattr(model_result.lower, name)) for name in _MODELS}
    barred = {name: _matrix_values(getattr(model_result.barred, name)) for name in _MODELS}
    mapping = reference.anchor_topology.ao_to_atom
    angular = ao_angular_momenta(reference)
    global_relative: dict[str, float] = {}
    diagnostic_relative: dict[str, dict[str, float]] = {
        "form_factor_only": {},
        "anchored_vector_only": {},
    }
    partitions: dict[str, dict[str, Any]] = {}
    angular_records: dict[str, list[dict[str, Any]]] = {}
    diagonal_scaled_max: dict[str, float] = {}
    max_channel_relative: dict[str, float] = {}
    absolute_signal: dict[str, float] = {}
    maximum_channel_absolute: dict[str, float] = {}
    maximum_element_absolute: dict[str, float] = {}
    for family in _FAMILIES:
        p0 = lower["p0"][family]
        exact = lower["exact"][family]
        absolute_signal[family] = float(np.linalg.norm(exact - p0))
        global_relative[family] = _relative(exact, p0)
        for model_name in ("form_factor_only", "anchored_vector_only"):
            diagnostic_relative[model_name][family] = _relative(lower[model_name][family], p0)
        partition = matrix_partition_changes(exact, p0, mapping, floor=floors[family])
        partitions[family] = {
            label: {
                "absolute_frobenius": float(partition.absolute_frobenius[index]),
                "relative_frobenius": float(partition.relative_frobenius[index]),
                "maximum_absolute_element": float(partition.maximum_absolute_element[index]),
            }
            for index, label in enumerate(partition.labels)
        }
        channels = angular_channel_changes(
            exact,
            p0,
            mapping,
            angular,
            floor=floors[family],
        )
        angular_records[family] = [
            {
                "bra_atom": value.bra_atom,
                "ket_atom": value.ket_atom,
                "bra_l": value.bra_angular_momentum,
                "ket_l": value.ket_angular_momentum,
                "classification": value.classification,
                "absolute_frobenius": value.absolute_frobenius,
                "relative_frobenius": value.relative_frobenius,
                "maximum_absolute_element": value.maximum_absolute_element,
                "singular_values": value.singular_values.tolist(),
            }
            for value in channels
        ]
        max_channel_relative[family] = max(value.relative_frobenius for value in channels)
        maximum_channel_absolute[family] = max(value.absolute_frobenius for value in channels)
        maximum_element_absolute[family] = float(np.max(np.abs(exact - p0)))
        diagonal_scaled_max[family] = float(
            np.max(
                diagonal_scaled_element_change(
                    exact,
                    p0,
                    floor=floors[family],
                )
            )
        )
    spectral = compare_generalized_spectra(
        lower["exact"]["mechanical"],
        lower["exact"]["overlap"],
        lower["p0"]["mechanical"],
        lower["p0"]["overlap"],
    )
    same_anchor = mapping[:, None] == mapping[None, :]
    phase_spread = np.asarray(result.phase_spread.rms, dtype=np.float64)
    intersite_phase = phase_spread[~same_anchor]
    p0_metric_eigenvalues = np.linalg.eigvalsh(lower["p0"]["overlap"])
    exact_metric_eigenvalues = np.linalg.eigvalsh(lower["exact"]["overlap"])
    kinetic_scale = max(1.0, float(np.linalg.norm(result.kinetic.zero)))
    sector_norms = {
        name: float(np.linalg.norm(getattr(result.kinetic.exact_sectors, name))) / kinetic_scale
        for name in _SECTORS
    }
    headline = {
        "phase_spread_intersite_max": (
            float(np.max(intersite_phase)) if intersite_phase.size else 0.0
        ),
        "overlap_global": global_relative["overlap"],
        "mechanical_global": global_relative["mechanical"],
        "overlap_max_channel": max_channel_relative["overlap"],
        "mechanical_max_channel": max_channel_relative["mechanical"],
        "overlap_diagonal_scaled": diagonal_scaled_max["overlap"],
        "mechanical_diagonal_scaled": diagonal_scaled_max["mechanical"],
        "spectral_max_abs_hartree": float(np.max(np.abs(spectral.eigenvalue_shifts))),
    }
    spectral_floor = (
        floors["mechanical"]
        + max(1.0, float(np.max(np.abs(spectral.reference_eigenvalues)))) * floors["overlap"]
    ) / max(float(p0_metric_eigenvalues[0]), np.finfo(np.float64).tiny)
    signal_to_floor = {
        "phase_spread_intersite_max": headline["phase_spread_intersite_max"]
        / (64.0 * np.finfo(np.float64).eps),
        "overlap_global": absolute_signal["overlap"] / floors["overlap"],
        "mechanical_global": absolute_signal["mechanical"] / floors["mechanical"],
        "overlap_max_channel": maximum_channel_absolute["overlap"] / floors["overlap"],
        "mechanical_max_channel": maximum_channel_absolute["mechanical"] / floors["mechanical"],
        "overlap_diagonal_scaled": maximum_element_absolute["overlap"] / floors["overlap"],
        "mechanical_diagonal_scaled": maximum_element_absolute["mechanical"] / floors["mechanical"],
        "spectral_max_abs_hartree": headline["spectral_max_abs_hartree"] / spectral_floor,
    }
    headline_above_floor = {name: bool(value >= 4.0) for name, value in signal_to_floor.items()}
    row = {
        "id": case["id"],
        "direction": case["direction"],
        "sign": case["sign"],
        "magnitude_au": case["magnitude_au"],
        "field_au": list(result.field.magnetic_field_au),
        "field_tesla": list(result.field.magnetic_field_tesla),
        "headline": headline,
        "headline_signal_to_numerical_floor": signal_to_floor,
        "headline_above_numerical_floor": headline_above_floor,
        "global_relative": global_relative,
        "diagnostic_relative": diagnostic_relative,
        "partitions": partitions,
        "angular_channels": angular_records,
        "diagonal_scaled_max": diagonal_scaled_max,
        "maximum_angular_channel_relative": max_channel_relative,
        "phase_spread_maximum": float(np.max(phase_spread)),
        "phase_spread_intersite_maximum": headline["phase_spread_intersite_max"],
        "kinetic_sector_relative_norms": sector_norms,
        "p0_metric_minimum_eigenvalue": float(p0_metric_eigenvalues[0]),
        "p0_metric_condition_number": float(p0_metric_eigenvalues[-1] / p0_metric_eigenvalues[0]),
        "exact_metric_minimum_eigenvalue": float(exact_metric_eigenvalues[0]),
        "exact_metric_condition_number": float(
            exact_metric_eigenvalues[-1] / exact_metric_eigenvalues[0]
        ),
        "p0_generalized_eigenvalues_hartree": spectral.reference_eigenvalues.tolist(),
        "exact_generalized_eigenvalues_hartree": spectral.candidate_eigenvalues.tolist(),
        "generalized_eigenvalue_shifts_hartree": spectral.eigenvalue_shifts.tolist(),
        "cumulative_subspace_angles_rad": (spectral.cumulative_subspace_angles_rad.tolist()),
    }
    arrays = {
        f"lower__{model_name}__{family}": values
        for model_name, families in lower.items()
        for family, values in families.items()
    }
    arrays.update(
        {
            f"barred__{model_name}__{family}": values
            for model_name, families in barred.items()
            for family, values in families.items()
        }
    )
    arrays.update(
        {
            f"kinetic_sector__{name}": np.asarray(getattr(result.kinetic.exact_sectors, name))
            for name in _SECTORS
        }
    )
    arrays["phase_spread_rms"] = phase_spread
    arrays["generalized_eigenvalues_p0"] = spectral.reference_eigenvalues
    arrays["generalized_eigenvalues_exact"] = spectral.candidate_eigenvalues
    arrays["cumulative_subspace_angles_rad"] = spectral.cumulative_subspace_angles_rad
    return row, {case["id"]: arrays}


def _thresholds(fixture: dict[str, Any], metric: str) -> tuple[float, ...]:
    key = (
        "spectral_shift_thresholds_hartree"
        if metric == "spectral_max_abs_hartree"
        else "normalized_matrix_thresholds"
    )
    return tuple(float(value) for value in fixture[key])


def _adaptive_magnitudes(
    rows: list[dict[str, Any]],
    directions: dict[str, np.ndarray],
    fixture: dict[str, Any],
) -> dict[str, set[float]]:
    additions: dict[str, set[float]] = {name: set() for name in directions}
    metrics = tuple(rows[0]["headline"]) if rows else ()
    for direction in directions:
        matching = sorted(
            (row for row in rows if row["direction"] == direction and row["sign"] == 1),
            key=lambda row: row["magnitude_au"],
        )
        for metric in metrics:
            for threshold in _thresholds(fixture, metric):
                previous: dict[str, Any] | None = None
                for row in matching:
                    if (
                        row["headline"][metric] >= threshold
                        and row["headline_above_numerical_floor"][metric]
                    ):
                        if previous is not None and (
                            not previous["headline_above_numerical_floor"][metric]
                            or previous["headline"][metric] < threshold
                        ):
                            additions[direction].add(
                                float(np.sqrt(previous["magnitude_au"] * row["magnitude_au"]))
                            )
                        break
                    previous = row
    return additions


def _threshold_crossings(
    rows: list[dict[str, Any]],
    directions: dict[str, np.ndarray],
    fixture: dict[str, Any],
) -> list[dict[str, Any]]:
    crossings: list[dict[str, Any]] = []
    metrics = tuple(rows[0]["headline"]) if rows else ()
    tesla_per_au = float(fixture["magnetic_field_tesla_per_au"])
    for direction in directions:
        matching = sorted(
            (row for row in rows if row["direction"] == direction and row["sign"] == 1),
            key=lambda row: row["magnitude_au"],
        )
        for metric in metrics:
            for threshold in _thresholds(fixture, metric):
                crossing = next(
                    (
                        row
                        for row in matching
                        if row["headline"][metric] >= threshold
                        and row["headline_above_numerical_floor"][metric]
                    ),
                    None,
                )
                crossings.append(
                    {
                        "direction": direction,
                        "metric": metric,
                        "threshold": threshold,
                        "crossed": crossing is not None,
                        "first_field_au": (None if crossing is None else crossing["magnitude_au"]),
                        "first_field_tesla": (
                            None if crossing is None else crossing["magnitude_au"] * tesla_per_au
                        ),
                        "value": (None if crossing is None else crossing["headline"][metric]),
                        "signal_to_numerical_floor": (
                            None
                            if crossing is None
                            else crossing["headline_signal_to_numerical_floor"][metric]
                        ),
                    }
                )
    return crossings


def _parity_rows(
    rows: list[dict[str, Any]],
    case_arrays: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    by_key = {(row["direction"], row["sign"], row["magnitude_au"]): row for row in rows}
    zero_arrays = case_arrays["zero"]
    output: list[dict[str, Any]] = []
    for row in rows:
        if row["sign"] != 1:
            continue
        negative = by_key[(row["direction"], -1, row["magnitude_au"])]
        plus_arrays = case_arrays[row["id"]]
        minus_arrays = case_arrays[negative["id"]]
        families: dict[str, dict[str, float]] = {}
        for family in _FAMILIES:
            plus = plus_arrays[f"barred__exact__{family}"]
            minus = minus_arrays[f"barred__exact__{family}"]
            zero = zero_arrays[f"barred__exact__{family}"]
            scale = max(1.0, float(np.linalg.norm(zero)))
            odd = 0.5 * (plus - minus)
            even = 0.5 * (plus + minus) - zero
            families[family] = {
                "odd_relative_frobenius": float(np.linalg.norm(odd)) / scale,
                "even_relative_frobenius": float(np.linalg.norm(even)) / scale,
            }
        output.append(
            {
                "direction": row["direction"],
                "magnitude_au": row["magnitude_au"],
                "families": families,
            }
        )
    return output


def _stack_arrays(
    cases: list[dict[str, Any]],
    case_arrays: dict[str, dict[str, np.ndarray]],
    quadrature: Any,
) -> dict[str, np.ndarray]:
    ordered_ids = [case["id"] for case in cases]
    names = tuple(case_arrays[ordered_ids[0]])
    arrays = {
        name: np.asarray([case_arrays[case_id][name] for case_id in ordered_ids]) for name in names
    }
    arrays["field_vectors_au"] = np.asarray(
        [case["field"].magnetic_field_au for case in cases], dtype=np.float64
    )
    arrays["field_magnitudes_au"] = np.asarray(
        [case["magnitude_au"] for case in cases], dtype=np.float64
    )
    arrays["field_signs"] = np.asarray([case["sign"] for case in cases], dtype=np.int64)
    arrays["grid_coordinates_au"] = quadrature.grid.coordinates_au
    arrays["grid_weights_au"] = quadrature.grid.weights_au
    return arrays


def _evaluate_cases(
    quadrature: Any,
    cases: list[dict[str, Any]],
    reference: Any,
    floors: dict[str, float] | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]], dict[str, float]]:
    results = evaluate_exact_static_magnetic_one_electron_matrices(
        quadrature,
        tuple(case["field"] for case in cases),
        include_direct_oracle=False,
    )
    if floors is None:
        zero_index = next(index for index, case in enumerate(cases) if case["id"] == "zero")
        floors = _numerical_floors(results[zero_index])
    rows: list[dict[str, Any]] = []
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for case, result in zip(cases, results, strict=True):
        row, case_values = _case_analysis(case, result, reference, floors)
        rows.append(row)
        arrays.update(case_values)
    return rows, arrays, floors


def _run_member(
    execution: Path,
    system_name: str,
    system: dict[str, Any],
    basis: str,
    scale: float,
    grid_level: int,
    fixture: dict[str, Any],
    smoke: bool,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    member = execution / system_name / basis / f"bond_scale_{scale:.6f}"
    completed = member / "completed.json"
    if completed.is_file():
        print(f"skip completed {member}", flush=True)
        saved = json.loads(completed.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError(f"completed member record is not a JSON object: {completed}")
        return dict(saved)
    member.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"start {system_name} {basis} scale={scale:g}", flush=True)
    config = _config(system, basis, scale)
    reference = prepare_one_electron_ao_reference(config)
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(grid_level),
        block_size=_BLOCK_SIZE,
    )
    directions = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in fixture["field_directions"].items()
    }
    initial_values = (
        (1.0e-4, 1.0e-2)
        if smoke
        else tuple(float(value) for value in fixture["initial_positive_field_magnitudes_au"])
    )
    initial_magnitudes = {name: set(initial_values) for name in directions}
    initial_cases = _field_cases(directions, initial_magnitudes, include_zero=True)
    rows, case_arrays, floors = _evaluate_cases(quadrature, initial_cases, reference, None)
    additions = (
        {name: set() for name in directions}
        if smoke
        else _adaptive_magnitudes(rows, directions, fixture)
    )
    adaptive_cases = _field_cases(directions, additions, include_zero=False)
    if adaptive_cases:
        adaptive_rows, adaptive_arrays, _ = _evaluate_cases(
            quadrature, adaptive_cases, reference, floors
        )
        rows.extend(adaptive_rows)
        case_arrays.update(adaptive_arrays)
    cases = initial_cases + adaptive_cases
    ordering = {case["id"]: index for index, case in enumerate(cases)}
    rows.sort(key=lambda row: ordering[row["id"]])
    arrays = _stack_arrays(cases, case_arrays, quadrature)
    arrays_path = member / "arrays.npz"
    _write_npz(arrays_path, arrays)
    threshold_crossings = _threshold_crossings(rows, directions, fixture)
    parity = _parity_rows(rows, case_arrays)
    minimum_exact_metric = min(row["exact_metric_minimum_eigenvalue"] for row in rows)
    minimum_p0_metric = min(row["p0_metric_minimum_eigenvalue"] for row in rows)
    elapsed = time.monotonic() - started
    array_hashes = {name: canonical_sha256(value) for name, value in arrays.items()}
    manifest_core = {
        "schema": "aion.exact-one-electron-wp3-member-manifest",
        "version": "1.0.0",
        "status": "smoke_not_qualification" if smoke else "executed_unreviewed",
        "code": provenance["code"],
        "environment": provenance["environment"],
        "input_hashes": provenance["input_hashes"],
        "system": {
            "name": system_name,
            "atoms": [atom.as_mapping() for atom in config.atoms],
            "basis": basis,
            "basis_role": fixture["basis_roles"][basis],
            "bond_scale": scale,
            "bond_distance_bohr": float(system["reference_distance_bohr"]) * scale,
            "reference_fingerprint_sha256": reference.fingerprint_sha256,
            "config_id": config.scientific_id,
            "nao": reference.core_operators.nao,
            "ao_labels": list(reference.basis_metadata.ao_labels),
            "ao_to_atom": reference.anchor_topology.ao_to_atom.tolist(),
            "ao_angular_momenta": ao_angular_momenta(reference).tolist(),
        },
        "quadrature": {
            "level": quadrature.grid.level,
            "pruning": quadrature.grid.pruning,
            "npoints": quadrature.grid.npoints,
            "fingerprint_sha256": quadrature.grid.fingerprint_sha256,
            "coordinates_sha256": canonical_sha256(quadrature.grid.coordinates_au),
            "weights_sha256": canonical_sha256(quadrature.grid.weights_au),
            "block_size": quadrature.block_size,
            "backend": quadrature.backend_config.kind.value,
        },
        "field_scan": {
            "initial_magnitudes_au": list(initial_values),
            "adaptive_magnitudes_au": {name: sorted(values) for name, values in additions.items()},
            "directions": {name: value.tolist() for name, value in directions.items()},
            "both_signs_executed": True,
            "case_count": len(cases),
        },
        "models": list(_MODELS),
        "conventions": {
            "q": -1.0,
            "mass": 1.0,
            "hbar": 1.0,
            "matrix_indices": "row_mu_bra_column_nu_ket",
            "endpoint_path": "R_nu_to_R_mu",
            "coefficient_subspace_angles": (
                "Euclidean principal angles in common AO coefficient ordering"
            ),
        },
    }
    manifest_id = canonical_sha256(manifest_core)
    _write_json(member / "manifest.json", {**manifest_core, "manifest_id": manifest_id})
    result = {
        "schema": "aion.exact-one-electron-wp3-pair-result",
        "version": "1.0.0",
        "status": "smoke_not_qualification" if smoke else "executed_unreviewed",
        "manifest_id": manifest_id,
        "elapsed_seconds": elapsed,
        "numerical_floor_absolute_frobenius": floors,
        "case_ids_in_array_order": [case["id"] for case in cases],
        "cases": rows,
        "parity": parity,
        "threshold_crossings": threshold_crossings,
        "minimum_exact_metric_eigenvalue": minimum_exact_metric,
        "minimum_p0_metric_eigenvalue": minimum_p0_metric,
        "array_semantic_sha256": array_hashes,
        "artifact": {"path": str(arrays_path), "sha256": _sha256(arrays_path)},
        "review": {"reviewed": False, "reviewer": None, "decision": "pending"},
    }
    _write_json(member / "result.json", result)
    record = {
        "system": system_name,
        "basis": basis,
        "bond_scale": scale,
        "member_directory": str(member),
        "manifest_id": manifest_id,
        "case_count": len(cases),
        "elapsed_seconds": elapsed,
        "minimum_exact_metric_eigenvalue": minimum_exact_metric,
        "minimum_p0_metric_eigenvalue": minimum_p0_metric,
        "arrays_sha256": _sha256(arrays_path),
        "result_sha256": _sha256(member / "result.json"),
        "manifest_sha256": _sha256(member / "manifest.json"),
    }
    _write_json(completed, record)
    print(
        f"finish {system_name} {basis} scale={scale:g} cases={len(cases)} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return record


def _provenance(fixture: dict[str, Any], smoke: bool) -> dict[str, Any]:
    input_paths = {
        "campaign_fixture": _FIXTURE,
        "formal_plan": _FORMAL_PLAN,
        "g2_review": _G2_REVIEW,
        "runner": Path(__file__).resolve(),
        "magnetic_evaluator": _REPOSITORY / "src/aion/electronic_structure/magnetic_matrices.py",
        "magnetic_analysis": _REPOSITORY / "src/aion/electronic_structure/magnetic_analysis.py",
    }
    thread_names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "schema": "aion.exact-one-electron-wp3-execution-plan",
        "version": "1.0.0",
        "status": "smoke_not_qualification" if smoke else "executed_unreviewed",
        "code": {
            "repository": str(_REPOSITORY),
            "branch": _git("branch", "--show-current"),
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "environment": {
            "catalog": "2026.07.1",
            "profile": "gnu",
            "modules": [],
            "conda_prefix": os.environ.get("CONDA_PREFIX", "unreported"),
            "conda_lock_sha256": _sha256(_LOCK),
            "thread_limits": {name: os.environ.get(name, "unset") for name in thread_names},
            "hostname": socket.gethostname(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "fixture": fixture,
        "input_hashes": {name: _sha256(path) for name, path in input_paths.items()},
    }


def _update_index(execution: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(execution.glob("*/*/bond_scale_*/completed.json"))
    ]
    index = {
        "schema": "aion.exact-one-electron-wp3-execution-index",
        "version": "1.0.0",
        "status": provenance["status"],
        "execution_directory": str(execution),
        "completed_members": records,
        "completed_member_count": len(records),
        "total_elapsed_seconds": sum(value["elapsed_seconds"] for value in records),
    }
    _write_json(execution / "execution_index.json", index)
    return index


def main() -> None:
    arguments = _arguments()
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    for scale in arguments.bond_scales:
        if not np.isfinite(scale) or scale <= 0.0:
            raise SystemExit("bond scales must be positive and finite")
    if arguments.grid_level < 0:
        raise SystemExit("grid level must be nonnegative")
    timestamp = datetime.now(UTC)
    commit = _git("rev-parse", "HEAD")
    execution = (
        arguments.execution_directory.resolve()
        if arguments.execution_directory is not None
        else arguments.output_root
        / f"wp3_pair_{timestamp.strftime('%Y%m%dT%H%M%SZ')}_{commit[:12]}"
    )
    execution.mkdir(parents=True, exist_ok=arguments.execution_directory is not None)
    lock_path = execution / ".campaign.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(f"another WP3 pair-scan process owns {lock_path}") from error
        provenance_path = execution / "execution_plan.json"
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        else:
            provenance = _provenance(fixture, arguments.smoke)
            _write_json(provenance_path, provenance)
        members = [
            (system_name, basis, scale)
            for scale in arguments.bond_scales
            for system_name in arguments.systems
            for basis in arguments.bases
        ]
        if arguments.max_members is not None:
            members = members[: arguments.max_members]
        print(f"execution={execution}", flush=True)
        print(f"members={len(members)}", flush=True)
        for system_name, basis, scale in members:
            _run_member(
                execution,
                system_name,
                fixture["systems"][system_name],
                basis,
                scale,
                arguments.grid_level,
                fixture,
                arguments.smoke,
                provenance,
            )
            _update_index(execution, provenance)
        index = _update_index(execution, provenance)
        print(f"completed_members={index['completed_member_count']}", flush=True)
        print(
            f"execution_index_sha256={_sha256(execution / 'execution_index.json')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
