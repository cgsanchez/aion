#!/usr/bin/env python3
"""Execute the exact static one-electron Wilson WP2 qualification campaign."""

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
    analytic_uniform_magnetic_overlap,
    evaluate_magnetic_one_electron_matrices,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
)

_REPOSITORY = Path(__file__).resolve().parents[1]
_FORMAL_PLAN = Path(
    "/home/cgs/00_WORK/Projection_Full_Formalism/REVIEW/implementation/"
    "exact_one_electron_numerical_qualification_plan.md"
)
_FIXTURE_DIRECTORY = _REPOSITORY / "tests/fixtures/exact_one_electron"
_FIXTURE_PATHS = {
    "hh_sto3g": _FIXTURE_DIRECTORY / "hh_sto3g.fixture.json",
    "oh_sto3g": _FIXTURE_DIRECTORY / "oh_sto3g.fixture.json",
}
_G1_REVIEW = _REPOSITORY / "docs/reviews/exact_one_electron_g1_review_20260916.json"
_LOCK = _REPOSITORY / "conda-linux-64.lock"
_DEFAULT_OUTPUT_ROOT = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/"
    "exact_one_electron_qualification"
)
_BLOCK_SIZE = 2048
_FIELD_MAGNITUDE_AU = 0.08
_MATRIX_NAMES = ("overlap", "kinetic", "nuclear_attraction", "mechanical")
_DIRECT_NAMES = ("overlap", "kinetic", "nuclear_attraction")

# Accepted G1 H--H unpruned-level-5 working floors.  The EX-direct/EX-link
# identity must be tighter than these independently measured quadrature floors.
_HH_WP1_FLOORS = {
    "overlap": 1.419e-11,
    "kinetic": 2.849e-10,
    "nuclear_attraction": 2.635e-10,
}
_PAIR_REVERSAL_TOLERANCE = 1.0e-11
_GAUGE_CONGRUENCE_TOLERANCE = 1.0e-12
_SPECTRUM_TOLERANCE = 1.0e-11
_ROTATION_TOLERANCE = 2.0e-7
_ANALYTIC_OVERLAP_TOLERANCE = 1.0e-8


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


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(1.0, float(np.linalg.norm(right))))


def _hermiticity(matrix: np.ndarray) -> float:
    return _relative(matrix, matrix.conj().T)


def _matrix_values(lower: Any) -> dict[str, np.ndarray]:
    return {
        "overlap": np.asarray(lower.overlap),
        "kinetic": np.asarray(lower.kinetic),
        "nuclear_attraction": np.asarray(lower.nuclear_attraction),
        "mechanical": np.asarray(lower.mechanical),
    }


def _spectrum(result: Any) -> np.ndarray:
    return np.asarray(
        eigvalsh(result.lower_exact.mechanical, result.lower_exact.overlap),
        dtype=np.float64,
    )


def _rotation_matrix() -> np.ndarray:
    axis = np.asarray((1.0, -2.0, 0.7), dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = 0.61
    cross = np.asarray(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        )
    )
    return (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
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
    reference: str,
    tolerance: float | None,
    normalization: str = "max(1, Frobenius norm of reference matrix)",
) -> dict[str, object]:
    return {
        "name": name,
        "value": float(value),
        "normalization": normalization,
        "reference": reference,
        "tolerance": tolerance,
    }


def _store(
    arrays: dict[str, np.ndarray],
    records: list[dict[str, object]],
    name: str,
    value: Any,
    *,
    unit: str,
    dimension: str,
    definition: str,
) -> None:
    array = np.asarray(value)
    arrays[name.replace("/", "__")] = array
    records.append(
        _array_record(
            name,
            array,
            unit=unit,
            physical_dimension=dimension,
            definition=definition,
        )
    )


def _field_cases() -> tuple[tuple[str, UniformMagneticField], ...]:
    directions = {
        "parallel": np.asarray((0.0, 0.0, 1.0)),
        "perpendicular": np.asarray((1.0, 0.0, 0.0)),
        "oblique": np.asarray((1.0, 2.0, 3.0)) / np.sqrt(14.0),
    }
    cases: list[tuple[str, UniformMagneticField]] = [
        ("zero", UniformMagneticField((0.0, 0.0, 0.0)))
    ]
    for name, direction in directions.items():
        cases.extend(
            (
                (
                    f"{name}_plus",
                    UniformMagneticField(tuple(_FIELD_MAGNITUDE_AU * direction)),
                ),
                (
                    f"{name}_minus",
                    UniformMagneticField(tuple(-_FIELD_MAGNITUDE_AU * direction)),
                ),
            )
        )
    return tuple(cases)


def _evaluate_hh(
    fixture: dict[str, Any],
) -> tuple[Any, Any, dict[str, np.ndarray], list[dict[str, object]], dict[str, Any]]:
    reference = prepare_one_electron_ao_reference(_config(fixture))
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(4),
        block_size=_BLOCK_SIZE,
    )
    cases = _field_cases()
    results = evaluate_magnetic_one_electron_matrices(
        quadrature, tuple(field for _, field in cases)
    )
    arrays: dict[str, np.ndarray] = {
        "hh_grid_coordinates_au": quadrature.grid.coordinates_au,
        "hh_grid_weights_au": quadrature.grid.weights_au,
    }
    records: list[dict[str, object]] = []
    residuals: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    result_by_name = dict(zip((name for name, _ in cases), results, strict=True))

    for (case_name, field), result in zip(cases, results, strict=True):
        assert result.direct_oracle is not None
        exact = _matrix_values(result.lower_exact)
        exact_grid = _matrix_values(result.lower_exact_grid)
        direct = _matrix_values(result.direct_oracle.lower_grid)
        direct_residuals: dict[str, float] = {}
        hermiticity: dict[str, float] = {}
        for name in _MATRIX_NAMES:
            unit = "1" if name == "overlap" else "hartree"
            dimension = "overlap" if name == "overlap" else "energy"
            _store(
                arrays,
                records,
                f"hh/{case_name}/lower_exact/{name}",
                exact[name],
                unit=unit,
                dimension=dimension,
                definition="endpoint link times corrected exact endpoint-removed amplitude",
            )
            _store(
                arrays,
                records,
                f"hh/{case_name}/barred_exact/{name}",
                (
                    result.overlap.exact
                    if name == "overlap"
                    else result.kinetic.exact
                    if name == "kinetic"
                    else result.nuclear_attraction.exact
                    if name == "nuclear_attraction"
                    else result.kinetic.exact + result.nuclear_attraction.exact
                ),
                unit=unit,
                dimension=dimension,
                definition="exact endpoint-removed Wilson amplitude",
            )
            hermiticity[name] = _hermiticity(exact[name])
            residuals.append(
                _residual(
                    f"hh/{case_name}/pair_reversal/{name}",
                    hermiticity[name],
                    reference="matrix_adjoint",
                    tolerance=_PAIR_REVERSAL_TOLERANCE,
                )
            )
            if name in _DIRECT_NAMES:
                _store(
                    arrays,
                    records,
                    f"hh/{case_name}/lower_exact_grid/{name}",
                    exact_grid[name],
                    unit=unit,
                    dimension=dimension,
                    definition="endpoint-factorized raw-grid exact Wilson matrix",
                )
                _store(
                    arrays,
                    records,
                    f"hh/{case_name}/direct/{name}",
                    direct[name],
                    unit=unit,
                    dimension=dimension,
                    definition="direct Wilson-dressed AO quadrature matrix",
                )
                direct_residuals[name] = _relative(direct[name], exact_grid[name])
                residuals.append(
                    _residual(
                        f"hh/{case_name}/direct_vs_link/{name}",
                        direct_residuals[name],
                        reference="endpoint_factorized_raw_grid_matrix",
                        tolerance=_HH_WP1_FLOORS[name],
                    )
                )

        sectors = result.kinetic.exact_sectors
        for sector_name, value in (
            ("T_pp_F", sectors.pp),
            ("T_pC_F", sectors.pC),
            ("T_Cp_F", sectors.Cp),
            ("T_CC_F", sectors.C2),
            ("T_total_F", sectors.total),
        ):
            _store(
                arrays,
                records,
                f"hh/{case_name}/kinetic_sectors/{sector_name}",
                value,
                unit="hartree",
                dimension="energy",
                definition="corrected exact endpoint-removed kinetic sector",
            )
        sector_closure = _relative(np.asarray(sectors.total), np.asarray(result.kinetic.exact))
        residuals.append(
            _residual(
                f"hh/{case_name}/kinetic_sector_closure",
                sector_closure,
                reference="stored_exact_kinetic_amplitude",
                tolerance=np.finfo(np.float64).eps * 32.0,
            )
        )
        metric_eigenvalues = np.linalg.eigvalsh(exact["overlap"])
        metric_condition = float(np.linalg.cond(exact["overlap"]))
        spectrum = _spectrum(result)
        _store(
            arrays,
            records,
            f"hh/{case_name}/overlap_eigenvalues",
            metric_eigenvalues,
            unit="1",
            dimension="overlap_eigenvalue",
            definition="eigenvalues of exact lower Gram matrix",
        )
        _store(
            arrays,
            records,
            f"hh/{case_name}/generalized_spectrum",
            spectrum,
            unit="hartree",
            dimension="energy",
            definition="generalized eigenvalues of (T+Vnuc, S)",
        )
        analytic_overlap = analytic_uniform_magnetic_overlap(
            reference, result.direct_oracle.gauge
        )
        overlap_oracle_residual = _relative(
            np.asarray(result.lower_exact.overlap), analytic_overlap.lower
        )
        residuals.append(
            _residual(
                f"hh/{case_name}/analytic_overlap_oracle",
                overlap_oracle_residual,
                reference="libcint_AO_pair_Fourier_transform",
                tolerance=_ANALYTIC_OVERLAP_TOLERANCE,
            )
        )
        case_rows.append(
            {
                "case": case_name,
                "field_au": list(field.magnetic_field_au),
                "field_magnitude_au": field.magnitude_au,
                "direct_vs_link": direct_residuals,
                "pair_reversal": hermiticity,
                "kinetic_sector_closure": sector_closure,
                "minimum_overlap_eigenvalue": float(metric_eigenvalues[0]),
                "overlap_condition_number": metric_condition,
                "analytic_overlap_oracle_residual": overlap_oracle_residual,
            }
        )

    zero_result = result_by_name["zero"]
    zero_reduction = {
        "overlap": _relative(
            np.asarray(zero_result.lower_exact.overlap), reference.core_operators.overlap
        ),
        "kinetic": _relative(
            np.asarray(zero_result.lower_exact.kinetic), reference.core_operators.kinetic
        ),
        "nuclear_attraction": _relative(
            np.asarray(zero_result.lower_exact.nuclear_attraction),
            reference.core_operators.nuclear_attraction,
        ),
    }
    for name, value in zero_reduction.items():
        residuals.append(
            _residual(
                f"hh/zero/reduction/{name}",
                value,
                reference=f"WP1_analytic_{name}",
                tolerance=np.finfo(np.float64).eps * 32.0,
            )
        )

    reversal_rows: list[dict[str, object]] = []
    for direction in ("parallel", "perpendicular", "oblique"):
        plus = result_by_name[f"{direction}_plus"]
        minus = result_by_name[f"{direction}_minus"]
        values: dict[str, float] = {}
        for name in _MATRIX_NAMES:
            plus_matrix = _matrix_values(plus.lower_exact)[name]
            minus_matrix = _matrix_values(minus.lower_exact)[name]
            values[name] = _relative(minus_matrix, plus_matrix.conj())
            residuals.append(
                _residual(
                    f"hh/{direction}/field_reversal/{name}",
                    values[name],
                    reference="complex_conjugate_positive_field_matrix",
                    tolerance=_PAIR_REVERSAL_TOLERANCE,
                )
            )
        reversal_rows.append({"direction": direction, "residuals": values})

    gauge_field = result_by_name["oblique_plus"].field
    field_vector = np.asarray(gauge_field.magnetic_field_au)
    landau_axis = np.cross(field_vector, np.asarray((0.31, -0.47, 0.79)))
    origins = ((0.0, 0.0, 0.0), (0.21, -0.17, 0.13))
    gauges = (
        AffineMagneticGauge(gauge_field, origin_au=origins[0]),
        AffineMagneticGauge(gauge_field, origin_au=origins[1]),
        AffineMagneticGauge(
            gauge_field,
            kind=MagneticGaugeKind.LANDAU,
            origin_au=origins[0],
            landau_axis=tuple(landau_axis),
        ),
        AffineMagneticGauge(
            gauge_field,
            kind=MagneticGaugeKind.LANDAU,
            origin_au=origins[1],
            landau_axis=tuple(landau_axis),
        ),
    )
    gauge_results = evaluate_magnetic_one_electron_matrices(
        quadrature,
        (gauge_field,) * len(gauges),
        direct_gauges=gauges,
    )
    gauge_names = ("symmetric_origin_a", "symmetric_origin_b", "landau_origin_a", "landau_origin_b")
    baseline = gauge_results[0]
    baseline_spectrum = _spectrum(baseline)
    anchors = reference.core_operators.nuclei.coordinates_au[
        reference.anchor_topology.ao_to_atom
    ]
    backend = NumPyBackend()
    gauge_rows: list[dict[str, object]] = []
    for gauge_name, gauge, result in zip(gauge_names, gauges, gauge_results, strict=True):
        chi = np.asarray(
            affine_gauge_difference_potential(gauge, gauges[0], anchors, backend)
        )
        phase = np.exp(-1j * chi)
        congruence: dict[str, float] = {}
        barred: dict[str, float] = {}
        for name in _MATRIX_NAMES:
            base_lower = _matrix_values(baseline.lower_exact)[name]
            candidate = _matrix_values(result.lower_exact)[name]
            expected = phase[:, None] * base_lower * phase[None, :].conj()
            congruence[name] = _relative(candidate, expected)
            base_barred = (
                baseline.overlap.exact
                if name == "overlap"
                else baseline.kinetic.exact
                if name == "kinetic"
                else baseline.nuclear_attraction.exact
                if name == "nuclear_attraction"
                else baseline.kinetic.exact + baseline.nuclear_attraction.exact
            )
            candidate_barred = (
                result.overlap.exact
                if name == "overlap"
                else result.kinetic.exact
                if name == "kinetic"
                else result.nuclear_attraction.exact
                if name == "nuclear_attraction"
                else result.kinetic.exact + result.nuclear_attraction.exact
            )
            barred[name] = _relative(np.asarray(candidate_barred), np.asarray(base_barred))
            residuals.extend(
                (
                    _residual(
                        f"hh/gauge/{gauge_name}/lower_congruence/{name}",
                        congruence[name],
                        reference="endpoint_coefficient_congruence",
                        tolerance=_GAUGE_CONGRUENCE_TOLERANCE,
                    ),
                    _residual(
                        f"hh/gauge/{gauge_name}/barred_invariance/{name}",
                        barred[name],
                        reference="symmetric_origin_a_barred_amplitude",
                        tolerance=_GAUGE_CONGRUENCE_TOLERANCE,
                    ),
                )
            )
        spectrum_residual = float(np.max(np.abs(_spectrum(result) - baseline_spectrum)))
        residuals.append(
            _residual(
                f"hh/gauge/{gauge_name}/generalized_spectrum",
                spectrum_residual,
                reference="symmetric_origin_a_generalized_spectrum",
                tolerance=_SPECTRUM_TOLERANCE,
                normalization="maximum absolute eigenvalue difference in hartree",
            )
        )
        gauge_rows.append(
            {
                "case": gauge_name,
                "kind": gauge.kind.value,
                "origin_au": list(gauge.origin_au),
                "landau_axis": list(gauge.landau_axis) if gauge.landau_axis else None,
                "lower_congruence": congruence,
                "barred_invariance": barred,
                "generalized_spectrum_max_absolute_difference_au": spectrum_residual,
            }
        )

    summary = {
        "zero_field_reduction": zero_reduction,
        "field_cases": case_rows,
        "field_reversal": reversal_rows,
        "gauge_covariance": gauge_rows,
    }
    return reference, quadrature, arrays, records, {"residuals": residuals, **summary}


def _evaluate_rotation(
    fixture: dict[str, Any],
) -> tuple[Any, Any, dict[str, np.ndarray], list[dict[str, object]], dict[str, Any]]:
    config = _config(fixture)
    rotation = _rotation_matrix()
    rotated_config = OneElectronReferenceConfig(
        atoms=tuple(
            AtomConfig(
                atom.symbol,
                tuple(rotation @ np.asarray(atom.position_au, dtype=np.float64)),
            )
            for atom in config.atoms
        ),
        basis=config.basis,
        electromagnetic_origin=ElectromagneticOrigin(
            tuple(rotation @ np.asarray(config.electromagnetic_origin.position_au))
        ),
    )
    reference = prepare_one_electron_ao_reference(config)
    rotated_reference = prepare_one_electron_ao_reference(rotated_config)
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(4),
        block_size=_BLOCK_SIZE,
    )
    rotated_quadrature = prepare_ao_quadrature(
        rotated_reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(4),
        block_size=_BLOCK_SIZE,
    )
    field = _FIELD_MAGNITUDE_AU * np.asarray((1.0, 2.0, 3.0)) / np.sqrt(14.0)
    result = evaluate_magnetic_one_electron_matrices(
        quadrature, (UniformMagneticField(tuple(field)),)
    )[0]
    rotated_result = evaluate_magnetic_one_electron_matrices(
        rotated_quadrature,
        (UniformMagneticField(tuple(rotation @ field)),),
    )[0]
    ao_rotation = np.eye(reference.core_operators.nao)
    # The immutable O--H/STO-3G fixture orders O(2px,2py,2pz) at indices 2:5.
    ao_rotation[2:5, 2:5] = rotation
    arrays = {
        "oh_grid_coordinates_au": quadrature.grid.coordinates_au,
        "oh_grid_weights_au": quadrature.grid.weights_au,
        "oh_rotated_grid_coordinates_au": rotated_quadrature.grid.coordinates_au,
        "oh_rotated_grid_weights_au": rotated_quadrature.grid.weights_au,
        "rotation_matrix": rotation,
        "ao_rotation_matrix": ao_rotation,
    }
    records: list[dict[str, object]] = []
    residuals: list[dict[str, object]] = []
    rows: dict[str, dict[str, float]] = {
        "full": {},
        "oxygen_onsite": {},
        "oxygen_p_block": {},
        "oxygen_hydrogen": {},
        "direct_vs_link_original": {},
        "direct_vs_link_rotated": {},
    }
    assert result.direct_oracle is not None
    assert rotated_result.direct_oracle is not None
    original = _matrix_values(result.lower_exact)
    rotated = _matrix_values(rotated_result.lower_exact)
    for name in _MATRIX_NAMES:
        unit = "1" if name == "overlap" else "hartree"
        dimension = "overlap" if name == "overlap" else "energy"
        expected = ao_rotation @ original[name] @ ao_rotation.T
        actual = rotated[name]
        _store(
            arrays,
            records,
            f"oh_rotation/original/{name}",
            original[name],
            unit=unit,
            dimension=dimension,
            definition="original exact lower matrix",
        )
        _store(
            arrays,
            records,
            f"oh_rotation/rotated/{name}",
            actual,
            unit=unit,
            dimension=dimension,
            definition="rigidly rotated exact lower matrix",
        )
        rows["full"][name] = _relative(actual, expected)
        rows["oxygen_onsite"][name] = _relative(actual[:5, :5], expected[:5, :5])
        rows["oxygen_p_block"][name] = _relative(actual[2:5, 2:5], expected[2:5, 2:5])
        rows["oxygen_hydrogen"][name] = _relative(actual[:5, 5:6], expected[:5, 5:6])
        for block_name in ("full", "oxygen_onsite", "oxygen_p_block", "oxygen_hydrogen"):
            residuals.append(
                _residual(
                    f"oh/rotation/{block_name}/{name}",
                    rows[block_name][name],
                    reference="AO_block_rotated_original_matrix",
                    tolerance=_ROTATION_TOLERANCE,
                )
            )
        if name in _DIRECT_NAMES:
            rows["direct_vs_link_original"][name] = _relative(
                _matrix_values(result.direct_oracle.lower_grid)[name],
                _matrix_values(result.lower_exact_grid)[name],
            )
            rows["direct_vs_link_rotated"][name] = _relative(
                _matrix_values(rotated_result.direct_oracle.lower_grid)[name],
                _matrix_values(rotated_result.lower_exact_grid)[name],
            )
            for label in ("direct_vs_link_original", "direct_vs_link_rotated"):
                residuals.append(
                    _residual(
                        f"oh/rotation/{label}/{name}",
                        rows[label][name],
                        reference="endpoint_factorized_raw_grid_matrix",
                        tolerance=_ROTATION_TOLERANCE,
                    )
                )
    spectrum_difference = float(
        np.max(np.abs(_spectrum(rotated_result) - _spectrum(result)))
    )
    residuals.append(
        _residual(
            "oh/rotation/generalized_spectrum",
            spectrum_difference,
            reference="original_generalized_spectrum",
            tolerance=_ROTATION_TOLERANCE,
            normalization="maximum absolute eigenvalue difference in hartree",
        )
    )
    return (
        reference,
        quadrature,
        arrays,
        records,
        {
            "residuals": residuals,
            "rotation_matrix": rotation.tolist(),
            "field_original_au": field.tolist(),
            "field_rotated_au": (rotation @ field).tolist(),
            "block_residuals": rows,
            "generalized_spectrum_max_absolute_difference_au": spectrum_difference,
            "rotated_reference_fingerprint_sha256": rotated_reference.fingerprint_sha256,
            "rotated_grid_fingerprint_sha256": rotated_quadrature.grid.fingerprint_sha256,
        },
    )


def _manifest_without_id(
    *,
    timestamp: str,
    branch: str,
    commit: str,
    dirty: bool,
    hh_reference: Any,
    hh_quadrature: Any,
    oh_reference: Any,
    oh_quadrature: Any,
    fixtures: dict[str, dict[str, Any]],
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
        "execution": {"timestamp_utc": timestamp, "runner": str(Path(__file__).resolve())},
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
            "dependencies": hh_reference.dependencies.as_mapping(),
            "thread_limits": {name: os.environ.get(name, "unset") for name in thread_names},
            "hostname": socket.gethostname(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "fixture": {
            "system_name": "hh_sto3g",
            "reference_fingerprint_sha256": hh_reference.fingerprint_sha256,
            "config_id": hh_reference.config.scientific_id,
            "atoms_bohr": [
                [atom.symbol, *atom.position_au] for atom in hh_reference.config.atoms
            ],
            "nuclear_charges": hh_reference.core_operators.nuclei.charges.tolist(),
            "coordinate_unit": "bohr",
            "basis": hh_reference.config.basis,
            "spherical": True,
            "ao_labels": list(hh_reference.basis_metadata.ao_labels),
            "shell_metadata": {
                "ao_locations": hh_reference.basis_metadata.ao_locations.tolist(),
                "fingerprint_sha256": hh_reference.basis_metadata.fingerprint_sha256,
            },
            "ao_to_atom": hh_reference.anchor_topology.ao_to_atom.tolist(),
            "additional_rotation_fixture": {
                "system_name": "oh_sto3g",
                "reference_fingerprint_sha256": oh_reference.fingerprint_sha256,
                "config_id": oh_reference.config.scientific_id,
                "ao_labels": list(oh_reference.basis_metadata.ao_labels),
            },
            "fixture_schemas": {
                name: fixture["schema"] for name, fixture in fixtures.items()
            },
        },
        "quadrature": {
            "kind": hh_quadrature.grid.kind.value,
            "level": hh_quadrature.grid.level,
            "pruning": hh_quadrature.grid.pruning,
            "npoints": hh_quadrature.grid.npoints,
            "fingerprint_sha256": hh_quadrature.grid.fingerprint_sha256,
            "coordinates_sha256": canonical_sha256(hh_quadrature.grid.coordinates_au),
            "weights_sha256": canonical_sha256(hh_quadrature.grid.weights_au),
            "block_size": hh_quadrature.block_size,
            "backend": hh_quadrature.backend_config.kind.value,
            "evaluator": hh_quadrature.provenance.evaluator,
            "rotation_fixture_grid": {
                "level": oh_quadrature.grid.level,
                "pruning": oh_quadrature.grid.pruning,
                "npoints": oh_quadrature.grid.npoints,
                "fingerprint_sha256": oh_quadrature.grid.fingerprint_sha256,
                "coordinates_sha256": canonical_sha256(oh_quadrature.grid.coordinates_au),
                "weights_sha256": canonical_sha256(oh_quadrature.grid.weights_au),
            },
        },
        "source": {
            "physical_field": "static_uniform_magnetic",
            "field_magnitude_au": _FIELD_MAGNITUDE_AU,
            "directions": ["parallel", "perpendicular", "oblique"],
            "signs": [1, -1],
            "potential_representatives": ["symmetric", "landau"],
            "gauge_origins_au": [[0.0, 0.0, 0.0], [0.21, -0.17, 0.13]],
            "scalar_potential": "all_electron_local_nuclear_attraction",
            "approximation_level": "exact_static_straight_Wilson",
            "routes": ["EX-direct", "EX-link"],
            "kinetic_sectors": ["T_pp_F", "T_pC_F", "T_Cp_F", "T_CC_F"],
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
            "endpoint_coefficient_law": "M_target=D M_reference D_dagger, D_mu=exp(-i chi(R_mu))",
        },
        "tolerances": {
            "pair_reversal_relative": _PAIR_REVERSAL_TOLERANCE,
            "gauge_congruence_relative": _GAUGE_CONGRUENCE_TOLERANCE,
            "spectrum_absolute_hartree": _SPECTRUM_TOLERANCE,
            "rotation_relative": _ROTATION_TOLERANCE,
            "analytic_overlap_relative": _ANALYTIC_OVERLAP_TOLERANCE,
            "direct_link_overlap_relative": _HH_WP1_FLOORS["overlap"],
            "direct_link_kinetic_relative": _HH_WP1_FLOORS["kinetic"],
            "direct_link_nuclear_relative": _HH_WP1_FLOORS["nuclear_attraction"],
        },
        "input_hashes": input_hashes,
    }


def _report(summary: dict[str, Any], execution_directory: Path) -> str:
    lines = [
        "# Exact one-electron WP2 static Wilson qualification",
        "",
        "Status: **numerically executed; G2 user review pending**.",
        "",
        "This campaign uses the occupancy-independent CPU/complex128 reference ",
        "evaluator. It does not promote the exact evaluator into the production ",
        "dynamics path and it makes no G2 acceptance decision.",
        "",
        "## H--H field signs and orientations",
        "",
        "| case | max EX-direct/EX-link | max pair reversal | min eig(S) | "
        "cond(S) | analytic S oracle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["hh"]["field_cases"]:
        lines.append(
            "| {case} | {direct:.3e} | {pair:.3e} | {minimum:.6e} | "
            "{condition:.6e} | {oracle:.3e} |".format(
                case=row["case"],
                direct=max(row["direct_vs_link"].values()),
                pair=max(row["pair_reversal"].values()),
                minimum=row["minimum_overlap_eigenvalue"],
                condition=row["overlap_condition_number"],
                oracle=row["analytic_overlap_oracle_residual"],
            )
        )
    lines.extend(
        (
            "",
            "The direct route is compared with the raw-grid endpoint-factorized route; ",
            "its tolerances are the independently accepted H--H WP1 level-5 floors, ",
            "not a tolerance inferred from this campaign.",
            "",
            "## Field reversal",
            "",
            "| direction | S | T | Vnuc | K |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for row in summary["hh"]["field_reversal"]:
        values = row["residuals"]
        lines.append(
            "| {direction} | {overlap:.3e} | {kinetic:.3e} | "
            "{nuclear_attraction:.3e} | {mechanical:.3e} |".format(
                direction=row["direction"], **values
            )
        )
    lines.extend(
        (
            "",
            "## Gauge representative and origin covariance",
            "",
            "| case | max barred change | max lower congruence | max spectrum shift (Ha) |",
            "|---|---:|---:|---:|",
        )
    )
    for row in summary["hh"]["gauge_covariance"]:
        lines.append(
            "| {case} | {barred:.3e} | {lower:.3e} | {spectrum:.3e} |".format(
                case=row["case"],
                barred=max(row["barred_invariance"].values()),
                lower=max(row["lower_congruence"].values()),
                spectrum=row["generalized_spectrum_max_absolute_difference_au"],
            )
        )
    rotation = summary["rotation"]
    lines.extend(
        (
            "",
            "## O--H rigid rotation with oxygen p mixing",
            "",
            "| comparison | S | T | Vnuc | K |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for label, values in rotation["block_residuals"].items():
        if not all(name in values for name in _MATRIX_NAMES):
            continue
        lines.append(
            "| {label} | {overlap:.3e} | {kinetic:.3e} | "
            "{nuclear_attraction:.3e} | {mechanical:.3e} |".format(
                label=label, **values
            )
        )
    lines.extend(
        (
            "",
            "The rotation acts nontrivially on the oxygen `(2px,2py,2pz)` AO block. ",
            "The maximum generalized-spectrum shift is ",
            f"`{rotation['generalized_spectrum_max_absolute_difference_au']:.3e}` Ha.",
            "",
            "## Gate checks",
            "",
        )
    )
    for name, value in summary["gate_checks"].items():
        lines.append(f"- `{name}`: **{'pass' if value else 'fail'}**")
    lines.extend(
        (
            "",
            "All raw grids, exact lower matrices, direct matrices, endpoint-removed ",
            "matrices, exact kinetic sectors, spectra, and transformation matrices are ",
            "stored in `arrays.npz`. Their semantic array hashes are in `result.json`; ",
            "the file-level hashes and manifest identity are in `execution_index.json`.",
            "",
            f"Execution directory: `{execution_directory}`",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    arguments = _arguments()
    timestamp = datetime.now(UTC).replace(microsecond=0)
    timestamp_text = timestamp.isoformat().replace("+00:00", "Z")
    commit = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    dirty = bool(_git("status", "--porcelain"))
    execution_directory = (
        arguments.output_root
        / f"wp2_{timestamp.strftime('%Y%m%dT%H%M%SZ')}_{commit[:12]}"
    )
    execution_directory.mkdir(parents=True, exist_ok=False)

    fixtures = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in _FIXTURE_PATHS.items()
    }
    hh_reference, hh_quadrature, hh_arrays, hh_records, hh_summary = _evaluate_hh(
        fixtures["hh_sto3g"]
    )
    oh_reference, oh_quadrature, rotation_arrays, rotation_records, rotation_summary = (
        _evaluate_rotation(fixtures["oh_sto3g"])
    )
    arrays = {**hh_arrays, **rotation_arrays}
    arrays_path = execution_directory / "arrays.npz"
    np.savez(arrays_path, **arrays)

    input_paths = {
        "formal_plan": _FORMAL_PLAN,
        "g1_review": _G1_REVIEW,
        "hh_fixture": _FIXTURE_PATHS["hh_sto3g"],
        "oh_fixture": _FIXTURE_PATHS["oh_sto3g"],
        "runner": Path(__file__).resolve(),
        "magnetic_matrices": _REPOSITORY
        / "src/aion/electronic_structure/magnetic_matrices.py",
        "magnetic_geometry": _REPOSITORY / "src/aion/electromagnetism/magnetic.py",
        "overlap_oracle": _REPOSITORY
        / "src/aion/electronic_structure/magnetic_overlap_oracle.py",
        "manifest_schema": _REPOSITORY
        / "docs/schemas/exact_one_electron_run_manifest.schema.json",
        "result_schema": _REPOSITORY
        / "docs/schemas/exact_one_electron_matrix_result.schema.json",
    }
    input_hashes = {name: _file_sha256(path) for name, path in input_paths.items()}
    manifest_core = _manifest_without_id(
        timestamp=timestamp_text,
        branch=branch,
        commit=commit,
        dirty=dirty,
        hh_reference=hh_reference,
        hh_quadrature=hh_quadrature,
        oh_reference=oh_reference,
        oh_quadrature=oh_quadrature,
        fixtures=fixtures,
        input_hashes=input_hashes,
    )
    manifest_id = canonical_sha256(manifest_core)
    manifest_path = execution_directory / "manifest.json"
    _write_json(manifest_path, {**manifest_core, "manifest_id": manifest_id})

    residuals = [*hh_summary.pop("residuals"), *rotation_summary.pop("residuals")]
    failed_residuals = [
        record
        for record in residuals
        if record["tolerance"] is not None and record["value"] > record["tolerance"]
    ]
    minimum_overlap = min(
        row["minimum_overlap_eigenvalue"] for row in hh_summary["field_cases"]
    )
    gate_checks = {
        "zero_field_reduction": all(
            value <= np.finfo(np.float64).eps * 32.0
            for value in hh_summary["zero_field_reduction"].values()
        ),
        "direct_vs_link_below_wp1_floor": all(
            all(value <= _HH_WP1_FLOORS[name] for name, value in row["direct_vs_link"].items())
            for row in hh_summary["field_cases"]
        ),
        "pair_and_field_reversal": not any(
            record["name"].startswith("hh/")
            and ("pair_reversal" in record["name"] or "field_reversal" in record["name"])
            and record["value"] > record["tolerance"]
            for record in residuals
        ),
        "gauge_origin_congruence": all(
            max(row["lower_congruence"].values()) <= _GAUGE_CONGRUENCE_TOLERANCE
            for row in hh_summary["gauge_covariance"]
        ),
        "gauge_origin_spectrum_invariance": all(
            row["generalized_spectrum_max_absolute_difference_au"]
            <= _SPECTRUM_TOLERANCE
            for row in hh_summary["gauge_covariance"]
        ),
        "exact_gram_positive": minimum_overlap > 0.0,
        "rigid_rotation_blocks_and_spectrum": max(
            value
            for values in rotation_summary["block_residuals"].values()
            for value in values.values()
        )
        <= _ROTATION_TOLERANCE
        and rotation_summary["generalized_spectrum_max_absolute_difference_au"]
        <= _ROTATION_TOLERANCE,
        "all_declared_residuals": not failed_residuals,
    }
    summary = {
        "hh": hh_summary,
        "rotation": rotation_summary,
        "gate_checks": gate_checks,
        "failed_residuals": failed_residuals,
    }
    result_path = execution_directory / "result.json"
    _write_json(
        result_path,
        {
            "schema": "aion.exact-one-electron-matrix-result",
            "version": "1.0.0",
            "status": "executed_unreviewed",
            "manifest_id": manifest_id,
            "matrices": [*hh_records, *rotation_records],
            "residuals": residuals,
            "artifact": {"path": str(arrays_path), "sha256": _file_sha256(arrays_path)},
            "review": {"reviewed": False, "reviewer": None, "decision": "pending"},
        },
    )
    report_path = execution_directory / "wp2_report.md"
    report_path.write_text(_report(summary, execution_directory), encoding="utf-8")
    index_path = execution_directory / "execution_index.json"
    _write_json(
        index_path,
        {
            "schema": "aion.exact-one-electron-wp2-execution-index",
            "version": "1.0.0",
            "status": "executed_unreviewed",
            "manifest_id": manifest_id,
            "execution_directory": str(execution_directory),
            "gate_checks": gate_checks,
            "minimum_overlap_eigenvalue": minimum_overlap,
            "artifacts": {
                path.name: _file_sha256(path)
                for path in (manifest_path, result_path, arrays_path, report_path)
            },
        },
    )
    print(execution_directory)
    print(f"manifest_id={manifest_id}")
    print(f"execution_index_sha256={_file_sha256(index_path)}")
    print(f"gate_checks={json.dumps(gate_checks, sort_keys=True)}")
    if failed_residuals or not all(gate_checks.values()):
        raise SystemExit("WP2 campaign completed with failed qualification checks")


if __name__ == "__main__":
    main()
