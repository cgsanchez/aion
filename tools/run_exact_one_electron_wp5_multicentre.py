#!/usr/bin/env python3
"""Run the resumable WP5 three-centre loop-flux qualification campaign."""

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
    center_loop_holonomy,
)
from aion.electronic_structure import (
    AOGridPolicy,
    analytic_uniform_magnetic_overlap,
    compare_generalized_spectra,
    evaluate_magnetic_one_electron_first_derivatives,
    evaluate_magnetic_one_electron_matrices,
    metric_spectrum,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
    static_magnetic_first_order_models,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_FORMAL_PLAN = Path(
    "/home/cgs/00_WORK/Projection_Full_Formalism/REVIEW/implementation/"
    "exact_one_electron_numerical_qualification_plan.md"
)
_FIXTURE = (
    _REPOSITORY
    / "tests/fixtures/exact_one_electron/wp5_multicentre.fixture.json"
)
_G3_REVIEW = _REPOSITORY / "docs/reviews/exact_one_electron_g3_review_20260918.json"
_G4_REVIEW = _REPOSITORY / "docs/reviews/exact_one_electron_g4_review_20260918.json"
_LOCK = _REPOSITORY / "conda-linux-64.lock"
_DEFAULT_OUTPUT_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "exact_one_electron_qualification"
)
_MODELS = ("exact", "p0", "geometric_b1", "full_b1")
_APPROXIMATE_MODELS = ("p0", "geometric_b1", "full_b1")
_PAIR_FAMILIES = ("overlap", "kinetic")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_execution_directory", type=Path)
    parser.add_argument("distance_execution_directory", type=Path)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execution-directory", type=Path)
    parser.add_argument("--geometries", nargs="+", choices=("equilateral", "distorted"))
    parser.add_argument("--bases", nargs="+", choices=("sto-3g", "cc-pvdz"))
    parser.add_argument("--smoke", action="store_true")
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
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)  # type: ignore[arg-type]
    os.replace(temporary, path)


def _pair_sources(
    reference_execution: Path,
    distance_execution: Path,
) -> dict[tuple[str, float], dict[str, Any]]:
    sources: dict[tuple[str, float], dict[str, Any]] = {}
    for execution, audit_name in (
        (reference_execution.resolve(), "reference_audit"),
        (distance_execution.resolve(), "distance_audit"),
    ):
        index_path = execution / "execution_index.json"
        audit_path = execution / audit_name / "result.json"
        index = _json(index_path)
        audit = _json(audit_path)
        floors = {
            (row["basis"], float(row["bond_scale"])): {
                family: float(
                    row["floors_absolute_frobenius"][family]["refined_working_floor"]
                )
                for family in _PAIR_FAMILIES
            }
            for row in audit["members"]
            if row["system"] == "hh"
        }
        for record in index["completed_members"]:
            if record["system"] != "hh" or record["basis"] not in ("sto-3g", "cc-pvdz"):
                continue
            scale = float(record["bond_scale"])
            if scale not in (0.8, 1.0, 1.25):
                continue
            key = (record["basis"], scale)
            value = dict(record)
            value["source_execution_index_sha256"] = _sha256(index_path)
            value["source_audit_result_sha256"] = _sha256(audit_path)
            value["refined_floors"] = floors[key]
            sources[key] = value
    expected = {(basis, scale) for basis in ("sto-3g", "cc-pvdz") for scale in (0.8, 1.0, 1.25)}
    if set(sources) != expected:
        raise ValueError("WP5 requires all accepted H-H pair fixtures at scales 0.8, 1.0, and 1.25")
    return sources


def _config(coordinates: np.ndarray, basis: str) -> OneElectronReferenceConfig:
    return OneElectronReferenceConfig(
        atoms=tuple(AtomConfig("H", tuple(row)) for row in coordinates),
        basis=basis,
        electromagnetic_origin=ElectromagneticOrigin((0.0, 0.0, 0.0)),
    )


def _area_vector(coordinates: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(
        np.cross(coordinates, np.roll(coordinates, -1, axis=0)), axis=0
    )


def _positive_fluxes(fixture: dict[str, Any], smoke: bool) -> np.ndarray:
    if smoke:
        return np.asarray((1.0e-3, 0.25, 1.0, 2.0), dtype=np.float64)
    scan = fixture["positive_loop_flux_scan"]
    linear = np.arange(
        float(scan["linear_start"]),
        float(scan["linear_stop"]) + 0.5 * float(scan["linear_step"]),
        float(scan["linear_step"]),
    )
    stress = np.arange(
        float(scan["stress_start"]),
        float(scan["stress_stop"]) + 0.5 * float(scan["stress_step"]),
        float(scan["stress_step"]),
    )
    return np.asarray(
        sorted({*(float(value) for value in scan["logarithmic"]), *linear, *stress}),
        dtype=np.float64,
    )


def _field(flux: float, area_z: float) -> UniformMagneticField:
    return UniformMagneticField((0.0, 0.0, float(flux / area_z)))


def _metric_models(
    reference: Any,
    derivative_z: np.ndarray,
    gauge: AffineMagneticGauge,
) -> dict[str, np.ndarray]:
    analytic = analytic_uniform_magnetic_overlap(reference, gauge)
    field_z = float(gauge.field.magnetic_field_au[2])
    p0_barred = np.asarray(reference.core_operators.overlap, dtype=np.complex128)
    first_barred = p0_barred + field_z * derivative_z
    endpoint = np.asarray(analytic.endpoint_link)
    return {
        "exact": np.asarray(analytic.lower),
        "p0": endpoint * p0_barred,
        "geometric_b1": endpoint * first_barred,
        "full_b1": endpoint * first_barred,
    }


def _loop_checks(coordinates: np.ndarray, field: UniformMagneticField) -> dict[str, float]:
    backend = NumPyBackend()
    gauges = (
        AffineMagneticGauge(field, origin_au=(0.0, 0.0, 0.0)),
        AffineMagneticGauge(field, origin_au=(0.23, -0.19, 0.31)),
        AffineMagneticGauge(
            field,
            kind=MagneticGaugeKind.LANDAU,
            origin_au=(-0.17, 0.29, -0.13),
            landau_axis=(1.0, 0.0, 0.0),
        ),
    )
    forward = [center_loop_holonomy(coordinates, gauge, backend) for gauge in gauges]
    reverse = [center_loop_holonomy(coordinates[::-1], gauge, backend) for gauge in gauges]
    expected = complex(forward[0].expected_flux_phase)
    return {
        "maximum_flux_identity_residual": max(
            abs(complex(value.endpoint_link_product) - complex(value.expected_flux_phase))
            for value in forward
        ),
        "maximum_gauge_residual": max(
            abs(complex(value.endpoint_link_product) - expected) for value in forward
        ),
        "maximum_orientation_reversal_residual": max(
            abs(
                complex(backward.endpoint_link_product)
                - complex(ahead.endpoint_link_product).conjugate()
            )
            for ahead, backward in zip(forward, reverse, strict=True)
        ),
    }


def _metric_scan(
    reference: Any,
    derivative_z: np.ndarray,
    coordinates: np.ndarray,
    positive_fluxes: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    area_z = float(_area_vector(coordinates)[2])
    signed_fluxes = np.concatenate(
        (np.asarray((0.0,)), np.ravel(np.column_stack((positive_fluxes, -positive_fluxes))))
    )
    rows: list[dict[str, Any]] = []
    matrices: dict[str, list[np.ndarray]] = {model: [] for model in _MODELS}
    eigenvalues: dict[str, list[np.ndarray]] = {model: [] for model in _MODELS}
    for flux in signed_fluxes:
        field = _field(float(flux), area_z)
        gauge = AffineMagneticGauge(field)
        models = _metric_models(reference, derivative_z, gauge)
        exact_norm = max(1.0, float(np.linalg.norm(models["exact"])))
        loop = _loop_checks(coordinates, field)
        row: dict[str, Any] = {
            "loop_flux_au": float(flux),
            "field_z_au": float(field.magnetic_field_au[2]),
            **loop,
        }
        for model, matrix in models.items():
            spectrum = metric_spectrum(matrix)
            matrices[model].append(matrix)
            eigenvalues[model].append(spectrum.eigenvalues)
            row[f"{model}_minimum_eigenvalue"] = spectrum.minimum_eigenvalue
            row[f"{model}_condition_number"] = spectrum.condition_number
            row[f"{model}_hermiticity_residual"] = spectrum.hermiticity_residual
            row[f"{model}_positive"] = spectrum.positive
            row[f"{model}_relative_error"] = float(
                np.linalg.norm(matrix - models["exact"]) / exact_norm
            )
        rows.append(row)
    arrays = {
        "metric_scan__loop_flux_au": signed_fluxes,
        "metric_scan__field_z_au": signed_fluxes / area_z,
    }
    for model in _MODELS:
        arrays[f"metric_scan__{model}__matrices"] = np.asarray(matrices[model])
        arrays[f"metric_scan__{model}__eigenvalues"] = np.asarray(eigenvalues[model])
    return rows, arrays


def _select_cases(rows: list[dict[str, Any]], targets: dict[str, Any]) -> list[dict[str, Any]]:
    margin = float(targets["selection_minimum_metric_eigenvalue"])
    positive = [
        row
        for row in rows
        if row["loop_flux_au"] > 0.0
        and all(float(row[f"{model}_minimum_eigenvalue"]) > margin for model in _MODELS)
        and float(row["p0_relative_error"]) > np.finfo(np.float64).tiny
    ]
    threshold = float(targets["physical_threshold"])
    definitions = (
        ("sub_threshold", float(targets["sub_threshold_target"]), lambda value: value < threshold),
        ("near_threshold", float(targets["near_threshold_target"]), lambda value: True),
        (
            "above_threshold",
            float(targets["above_threshold_target"]),
            lambda value: value > threshold,
        ),
    )
    selected = []
    used: set[float] = set()
    for label, target, predicate in definitions:
        candidates = [
            row
            for row in positive
            if predicate(float(row["p0_relative_error"]))
            and float(row["loop_flux_au"]) not in used
        ]
        if not candidates:
            raise ValueError(f"no positive-metric WP5 candidate for {label}")
        choice = min(
            candidates,
            key=lambda row: abs(np.log10(float(row["p0_relative_error"]) / target)),
        )
        used.add(float(choice["loop_flux_au"]))
        selected.append(
            {
                "label": label,
                "target_relative_error": target,
                "loop_flux_au": float(choice["loop_flux_au"]),
                "field_z_au": float(choice["field_z_au"]),
                "p0_relative_error": float(choice["p0_relative_error"]),
                "minimum_model_metric_eigenvalue": min(
                    float(choice[f"{model}_minimum_eigenvalue"]) for model in _MODELS
                ),
            }
        )
    return selected


def _host_models(result: Any) -> dict[str, Any]:
    hierarchy = static_magnetic_first_order_models(result).lower
    return {
        "p0": hierarchy.p0,
        "geometric_b1": hierarchy.geometric_b1,
        "full_b1": hierarchy.full_b1,
        "exact": hierarchy.exact,
    }


def _selected_spectra(
    reference: Any,
    quadrature: Any,
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[tuple[str, int], Any]]:
    cases = [
        (case, sign, UniformMagneticField((0.0, 0.0, sign * case["field_z_au"])))
        for case in selected
        for sign in (1, -1)
    ]
    results = evaluate_magnetic_one_electron_matrices(
        quadrature,
        tuple(field for _case, _sign, field in cases),
        include_direct_oracle=False,
    )
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    indexed: dict[tuple[str, int], Any] = {}
    for (case, sign, field), result in zip(cases, results, strict=True):
        indexed[(case["label"], sign)] = result
        models = _host_models(result)
        exact_metric = analytic_uniform_magnetic_overlap(
            reference, AffineMagneticGauge(field)
        ).lower
        exact_mechanical = np.asarray(models["exact"].mechanical)
        key = f"selected__{case['label']}__{'plus' if sign > 0 else 'minus'}"
        arrays[f"{key}__exact__overlap"] = np.asarray(exact_metric)
        arrays[f"{key}__exact__mechanical"] = exact_mechanical
        row: dict[str, Any] = {
            "label": case["label"],
            "sign": sign,
            "loop_flux_au": sign * case["loop_flux_au"],
            "field_z_au": float(field.magnetic_field_au[2]),
        }
        baseline_gauge = AffineMagneticGauge(field)
        comparison_gauge = AffineMagneticGauge(
            field,
            kind=MagneticGaugeKind.LANDAU,
            origin_au=(0.19, -0.23, 0.31),
            landau_axis=(1.0, 0.0, 0.0),
        )
        baseline_overlap = analytic_uniform_magnetic_overlap(
            reference, baseline_gauge
        )
        comparison_overlap = analytic_uniform_magnetic_overlap(
            reference, comparison_gauge
        )
        anchors = reference.core_operators.nuclei.coordinates_au[
            reference.anchor_topology.ao_to_atom
        ]
        chi = affine_gauge_difference_potential(
            comparison_gauge,
            baseline_gauge,
            anchors,
            NumPyBackend(),
        )
        coefficient_phase = np.exp(-1j * np.asarray(chi))
        expected_comparison = (
            coefficient_phase[:, None]
            * np.asarray(baseline_overlap.lower)
            * coefficient_phase[None, :].conj()
        )
        row["exact_gauge_barred_residual"] = float(
            np.linalg.norm(comparison_overlap.barred - baseline_overlap.barred)
        )
        row["exact_gauge_congruence_residual"] = float(
            np.linalg.norm(comparison_overlap.lower - expected_comparison)
            / max(1.0, float(np.linalg.norm(expected_comparison)))
        )
        row["exact_gauge_spectrum_residual"] = float(
            np.linalg.norm(
                np.linalg.eigvalsh(comparison_overlap.lower)
                - np.linalg.eigvalsh(baseline_overlap.lower)
            )
        )
        for model in _MODELS:
            metric = (
                np.asarray(exact_metric)
                if model == "exact"
                else np.asarray(models[model].overlap)
            )
            mechanical = (
                exact_mechanical
                if model == "exact"
                else np.asarray(models[model].mechanical)
            )
            arrays[f"{key}__{model}__overlap"] = metric
            arrays[f"{key}__{model}__mechanical"] = mechanical
            spectrum = metric_spectrum(metric)
            row[f"{model}_minimum_metric_eigenvalue"] = spectrum.minimum_eigenvalue
            row[f"{model}_metric_condition_number"] = spectrum.condition_number
            if model == "exact":
                row["exact_generalized_eigenvalue_count"] = int(metric.shape[0])
                continue
            comparison = compare_generalized_spectra(
                mechanical,
                metric,
                exact_mechanical,
                exact_metric,
            )
            arrays[f"{key}__{model}__eigenvalue_shifts"] = comparison.eigenvalue_shifts
            arrays[f"{key}__{model}__subspace_angles_rad"] = (
                comparison.cumulative_subspace_angles_rad
            )
            row[f"{model}_maximum_absolute_eigenvalue_shift"] = float(
                np.max(np.abs(comparison.eigenvalue_shifts))
            )
            row[f"{model}_maximum_subspace_angle_rad"] = float(
                np.max(comparison.cumulative_subspace_angles_rad, initial=0.0)
            )
        rows.append(row)
    for case in selected:
        plus = _host_models(indexed[(case["label"], 1)])
        minus = _host_models(indexed[(case["label"], -1)])
        for model in _MODELS:
            for family in ("overlap", "mechanical"):
                plus_value = np.asarray(getattr(plus[model], family))
                minus_value = np.asarray(getattr(minus[model], family))
                arrays[
                    f"reversal_residual__{case['label']}__{model}__{family}"
                ] = np.asarray(
                    np.linalg.norm(minus_value - plus_value.conj())
                    / max(1.0, float(np.linalg.norm(plus_value)))
                )
    return rows, arrays, indexed


def _atom_indices(reference: Any, atoms: tuple[int, int]) -> np.ndarray:
    mapping = np.asarray(reference.anchor_topology.ao_to_atom)
    return np.concatenate(tuple(np.flatnonzero(mapping == atom) for atom in atoms))


def _source_zero_spectra(source: dict[str, Any]) -> dict[str, np.ndarray]:
    directory = Path(source["member_directory"])
    for filename, field in (
        ("manifest.json", "manifest_sha256"),
        ("result.json", "result_sha256"),
        ("arrays.npz", "arrays_sha256"),
    ):
        if _sha256(directory / filename) != source[field]:
            raise ValueError(f"accepted pair fixture changed: {directory / filename}")
    result = _json(directory / "result.json")
    zero = result["case_ids_in_array_order"].index("zero")
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
        return {
            family: np.linalg.eigvalsh(np.asarray(arrays[f"barred__p0__{family}"])[zero])
            for family in _PAIR_FAMILIES
        }


def _pair_restrictions(
    geometry_name: str,
    coordinates: np.ndarray,
    basis: str,
    field: UniformMagneticField,
    h3_reference: Any,
    h3_level4_result: Any,
    fixture: dict[str, Any],
    sources: dict[tuple[str, float], dict[str, Any]],
    smoke: bool,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    levels = fixture["quadrature"]
    level4 = 2 if smoke else int(levels["qualification_level"])
    level5 = 3 if smoke else int(levels["refinement_level"])
    block_size = int(levels["block_size"])
    h3_level5_quadrature = prepare_ao_quadrature(
        h3_reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(level5),
        block_size=block_size,
    )
    h3_level5_result = evaluate_magnetic_one_electron_matrices(
        h3_level5_quadrature, (field,), include_direct_oracle=False
    )[0]
    h3_results = {
        level4: _host_models(h3_level4_result),
        level5: _host_models(h3_level5_result),
    }
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    edges = [tuple(int(value) for value in edge) for edge in fixture["oriented_edges"]]
    scales = fixture["geometries"][geometry_name][
        "edge_bond_scales_in_oriented_order"
    ]
    for edge_index, ((atom_i, atom_j), scale_value) in enumerate(
        zip(edges, scales, strict=True)
    ):
        scale = float(scale_value)
        source = sources[(basis, scale)]
        pair_coordinates = coordinates[[atom_i, atom_j]]
        distance = float(np.linalg.norm(pair_coordinates[1] - pair_coordinates[0]))
        if not np.isclose(distance, 1.4 * scale, atol=2.0e-15, rtol=0.0):
            raise ValueError("WP5 edge does not match its accepted pair distance")
        pair_reference = prepare_one_electron_ao_reference(
            _config(pair_coordinates, basis)
        )
        source_spectra = _source_zero_spectra(source)
        for family in _PAIR_FAMILIES:
            oriented = np.linalg.eigvalsh(
                np.asarray(getattr(pair_reference.core_operators, family))
            )
            residual = float(np.linalg.norm(oriented - source_spectra[family]))
            if residual > 2.0e-11:
                raise ValueError("oriented pair does not reproduce accepted fixture spectrum")
        pair_results: dict[int, dict[str, Any]] = {}
        for level in (level4, level5):
            quadrature = prepare_ao_quadrature(
                pair_reference,
                BackendConfig(),
                grid_policy=AOGridPolicy.qualification(level),
                block_size=block_size,
            )
            value = evaluate_magnetic_one_electron_matrices(
                quadrature, (field,), include_direct_oracle=False
            )[0]
            pair_results[level] = _host_models(value)
        h3_indices = _atom_indices(h3_reference, (atom_i, atom_j))
        for model in _MODELS:
            for family in _PAIR_FAMILIES:
                values: dict[int, tuple[np.ndarray, np.ndarray]] = {}
                for level in (level4, level5):
                    h3_matrix = np.asarray(getattr(h3_results[level][model], family))[
                        np.ix_(h3_indices, h3_indices)
                    ]
                    pair_matrix = np.asarray(
                        getattr(pair_results[level][model], family)
                    )
                    values[level] = (h3_matrix, pair_matrix)
                residual4 = float(np.linalg.norm(values[level4][0] - values[level4][1]))
                residual5 = float(np.linalg.norm(values[level5][0] - values[level5][1]))
                h3_change = float(np.linalg.norm(values[level5][0] - values[level4][0]))
                pair_change = float(np.linalg.norm(values[level5][1] - values[level4][1]))
                source_floor = float(source["refined_floors"][family])
                floor = max(
                    source_floor,
                    h3_change,
                    pair_change,
                    64.0 * np.finfo(np.float64).eps,
                )
                key = f"pair_{edge_index}__{model}__{family}"
                arrays[f"{key}__h3_level4"] = values[level4][0]
                arrays[f"{key}__pair_level4"] = values[level4][1]
                arrays[f"{key}__h3_level5"] = values[level5][0]
                arrays[f"{key}__pair_level5"] = values[level5][1]
                rows.append(
                    {
                        "edge_index": edge_index,
                        "atoms": [atom_i, atom_j],
                        "bond_scale": scale,
                        "distance_au": distance,
                        "model": model,
                        "family": family,
                        "level4_residual": residual4,
                        "level5_residual": residual5,
                        "h3_refinement_change": h3_change,
                        "pair_refinement_change": pair_change,
                        "accepted_pair_floor": source_floor,
                        "refined_working_floor": floor,
                        "level5_residual_to_floor": residual5 / floor,
                        "source_member_directory": source["member_directory"],
                        "source_manifest_id": source["manifest_id"],
                    }
                )
    return rows, arrays


def _member(
    execution: Path,
    geometry_name: str,
    basis: str,
    fixture: dict[str, Any],
    sources: dict[tuple[str, float], dict[str, Any]],
    provenance: dict[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    member = execution / geometry_name / basis
    completed_path = member / "completed.json"
    if completed_path.is_file():
        print(f"skip completed {member}", flush=True)
        return _json(completed_path)
    member.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    coordinates = np.asarray(
        fixture["geometries"][geometry_name]["coordinates_au"], dtype=np.float64
    )
    config = _config(coordinates, basis)
    reference = prepare_one_electron_ao_reference(config)
    levels = fixture["quadrature"]
    qualification_level = 2 if smoke else int(levels["qualification_level"])
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(qualification_level),
        block_size=int(levels["block_size"]),
    )
    derivative = evaluate_magnetic_one_electron_first_derivatives(quadrature)
    derivative_z = np.asarray(derivative.metric[2])
    metric_rows, arrays = _metric_scan(
        reference,
        derivative_z,
        coordinates,
        _positive_fluxes(fixture, smoke),
    )
    selected = _select_cases(metric_rows, fixture["selected_case_targets"])
    spectral_rows, spectral_arrays, indexed = _selected_spectra(
        reference, quadrature, selected
    )
    arrays.update(spectral_arrays)
    near = next(case for case in selected if case["label"] == "near_threshold")
    near_field = UniformMagneticField((0.0, 0.0, near["field_z_au"]))
    pair_rows, pair_arrays = _pair_restrictions(
        geometry_name,
        coordinates,
        basis,
        near_field,
        reference,
        indexed[("near_threshold", 1)],
        fixture,
        sources,
        smoke,
    )
    arrays.update(pair_arrays)
    arrays["coordinates_au"] = coordinates
    arrays["metric_derivative_z"] = derivative_z
    array_hashes = {name: canonical_sha256(value) for name, value in arrays.items()}
    source_records = []
    for scale in sorted(
        set(
            float(value)
            for value in fixture["geometries"][geometry_name][
                "edge_bond_scales_in_oriented_order"
            ]
        )
    ):
        source = sources[(basis, scale)]
        source_records.append(
            {
                "bond_scale": scale,
                "directory": source["member_directory"],
                "manifest_id": source["manifest_id"],
                "manifest_sha256": source["manifest_sha256"],
                "result_sha256": source["result_sha256"],
                "arrays_sha256": source["arrays_sha256"],
                "execution_index_sha256": source["source_execution_index_sha256"],
                "audit_result_sha256": source["source_audit_result_sha256"],
            }
        )
    manifest_core = {
        "schema": "aion.exact-one-electron-wp5-multicentre-member-manifest",
        "version": "1.0.0",
        "status": "smoke_not_qualification" if smoke else "executed_unreviewed",
        "geometry": geometry_name,
        "basis": basis,
        "basis_role": fixture["bases"][basis],
        "system_label": fixture["system_label"],
        "later_linear_dynamics_label": fixture["later_linear_dynamics_label"],
        "coordinates_au": coordinates.tolist(),
        "oriented_edges": fixture["oriented_edges"],
        "reference_fingerprint_sha256": reference.fingerprint_sha256,
        "quadrature": {
            "qualification_level": qualification_level,
            "refinement_level": 3 if smoke else int(levels["refinement_level"]),
            "pruning": "none",
            "block_size": int(levels["block_size"]),
            "qualification_grid_fingerprint_sha256": quadrature.grid.fingerprint_sha256,
        },
        "models": list(_MODELS),
        "metric_regularization": "none",
        "accepted_pair_sources": source_records,
        "provenance": provenance,
    }
    manifest_id = canonical_sha256(manifest_core)
    manifest_path = member / "manifest.json"
    result_path = member / "result.json"
    arrays_path = member / "arrays.npz"
    _write_json(manifest_path, {**manifest_core, "manifest_id": manifest_id})
    _write_npz(arrays_path, arrays)
    result = {
        "schema": "aion.exact-one-electron-wp5-multicentre-member-result",
        "version": "1.0.0",
        "status": "smoke_not_qualification" if smoke else "executed_unreviewed",
        "manifest_id": manifest_id,
        "metric_scan": metric_rows,
        "selected_dynamical_cases": selected,
        "generalized_spectral_comparisons": spectral_rows,
        "pair_restrictions": pair_rows,
        "pair_restriction_scope": (
            "overlap_and_kinetic_only; nuclear attraction includes the third proton "
            "and is not a two-centre restriction"
        ),
        "array_semantic_sha256": array_hashes,
        "elapsed_seconds": time.perf_counter() - started,
        "review": {"reviewed": False, "reviewer": None, "decision": "pending"},
    }
    _write_json(result_path, result)
    completed = {
        "status": "authenticated_smoke_not_qualification"
        if smoke
        else "authenticated_executed_unreviewed",
        "geometry": geometry_name,
        "basis": basis,
        "member_directory": str(member),
        "manifest_id": manifest_id,
        "manifest_sha256": _sha256(manifest_path),
        "result_sha256": _sha256(result_path),
        "arrays_sha256": _sha256(arrays_path),
        "array_count": len(arrays),
        "elapsed_seconds": result["elapsed_seconds"],
    }
    _write_json(completed_path, completed)
    print(
        f"finish {geometry_name} {basis} elapsed={result['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return completed


def _provenance(fixture: dict[str, Any], smoke: bool) -> dict[str, Any]:
    inputs = {
        "fixture": _FIXTURE,
        "formal_plan": _FORMAL_PLAN,
        "g3_review": _G3_REVIEW,
        "g4_review": _G4_REVIEW,
        "runner": Path(__file__).resolve(),
        "magnetic_geometry": _REPOSITORY / "src/aion/electromagnetism/magnetic.py",
        "magnetic_evaluator": _REPOSITORY / "src/aion/electronic_structure/magnetic_matrices.py",
        "magnetic_analysis": _REPOSITORY / "src/aion/electronic_structure/magnetic_analysis.py",
        "overlap_oracle": _REPOSITORY / "src/aion/electronic_structure/magnetic_overlap_oracle.py",
        "conda_lock": _LOCK,
    }
    thread_names = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "schema": "aion.exact-one-electron-wp5-execution-plan",
        "version": "1.0.0",
        "status": "smoke_not_qualification" if smoke else "executed_unreviewed",
        "fixture": fixture,
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
        "input_hashes": {name: _sha256(path) for name, path in inputs.items()},
    }


def _update_index(execution: Path, plan_id: str, expected: int, smoke: bool) -> dict[str, Any]:
    records = [_json(path) for path in sorted(execution.glob("*/*/completed.json"))]
    status = (
        "authenticated_smoke_not_qualification"
        if smoke
        else (
            "authenticated_executed_unreviewed"
            if len(records) == expected
            else "partial_executed_unreviewed"
        )
    )
    index = {
        "schema": "aion.exact-one-electron-wp5-execution-index",
        "version": "1.0.0",
        "status": status,
        "plan_id": plan_id,
        "execution_directory": str(execution),
        "requested_member_count": expected,
        "completed_member_count": len(records),
        "completed_members": records,
        "total_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in records),
    }
    _write_json(execution / "execution_index.json", index)
    return index


def main() -> None:
    arguments = _arguments()
    fixture = _json(_FIXTURE)
    geometries = tuple(arguments.geometries or fixture["geometries"])
    bases = tuple(arguments.bases or fixture["bases"])
    if arguments.smoke:
        geometries = geometries[:1]
        bases = bases[:1]
    sources = _pair_sources(
        arguments.reference_execution_directory,
        arguments.distance_execution_directory,
    )
    timestamp = datetime.now(UTC)
    commit = _git("rev-parse", "HEAD")
    execution = (
        arguments.execution_directory.resolve()
        if arguments.execution_directory is not None
        else arguments.output_root
        / f"wp5_multicentre_{timestamp.strftime('%Y%m%dT%H%M%SZ')}_{commit[:12]}"
    )
    execution.mkdir(parents=True, exist_ok=arguments.execution_directory is not None)
    with (execution / ".campaign.lock").open("a+") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"another WP5 runner owns {execution}") from exc
        provenance = _provenance(fixture, arguments.smoke)
        plan_core = {
            **provenance,
            "requested_geometries": list(geometries),
            "requested_bases": list(bases),
            "requested_member_count": len(geometries) * len(bases),
        }
        plan_id = canonical_sha256(plan_core)
        plan_path = execution / "execution_plan.json"
        if plan_path.is_file():
            existing = _json(plan_path)
            if existing.get("plan_id") != plan_id:
                raise ValueError("existing WP5 execution plan differs from this invocation")
        else:
            _write_json(plan_path, {**plan_core, "plan_id": plan_id})
        expected = len(geometries) * len(bases)
        for geometry in geometries:
            for basis in bases:
                print(f"start {geometry} {basis}", flush=True)
                _member(
                    execution,
                    geometry,
                    basis,
                    fixture,
                    sources,
                    provenance,
                    arguments.smoke,
                )
                _update_index(execution, plan_id, expected, arguments.smoke)
        index = _update_index(execution, plan_id, expected, arguments.smoke)
        print(f"execution={execution}")
        print(f"completed_members={index['completed_member_count']}/{expected}")
        print(f"status={index['status']}")


if __name__ == "__main__":
    main()
