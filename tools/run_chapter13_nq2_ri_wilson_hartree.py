#!/usr/bin/env python3
"""Execute the Chapter 13 NQ2 variational RI--Wilson Hartree campaign."""

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
from scipy.sparse.linalg import cg

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
    prepare_ri_wilson_hartree,
    reconstruct_mean_field,
)

_FIELD = (0.013, -0.009, 0.017)
_FIELD_DIRECTION = (-0.4, 0.7, 0.2)
_GAUGE_ORIGIN = (0.17, -0.31, 0.23)
_LANDAU_ORIGIN = (-0.21, 0.37, -0.16)
_GRID_LEVELS = (1, 2, 3, 4, 5)
_STEPS = (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5, 3.0e-6, 1.0e-6)
_AUXILIARY_BASES = {
    "h2": (
        "weigend",
        "def2-universal-jkfit",
        "cc-pvdz-jkfit",
        "cc-pvtz-jkfit",
        "cc-pvqz-jkfit",
    ),
    "lih": ("weigend", "def2-universal-jkfit"),
}
_SOLVE_TOLERANCES = (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12)


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
        ) + self.scale * self.direction.straight_line_integrals(starts_au, ends_au, backend)


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
            functional="lda,vwn",
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
            label=f"chapter13-nq2-{system}-lda-vwn",
            timestamp_utc=timestamp,
            host=platform.node(),
        ),
    )


def _metric_inverse(
    metric: np.ndarray,
    *,
    relative_threshold: float = 0.0,
    absolute_threshold: float = 1.0e-7,
    maximum_rank: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    cutoff = max(absolute_threshold, relative_threshold * eigenvalues[-1])
    retained = np.flatnonzero(eigenvalues > cutoff)
    if maximum_rank is not None and retained.size > maximum_rank:
        retained = retained[-maximum_rank:]
    vectors = eigenvectors[:, retained]
    values = eigenvalues[retained]
    inverse = (vectors / values[None, :]) @ vectors.T
    return inverse, values, vectors


def _action_from_three_index(
    density: np.ndarray,
    three_index: np.ndarray,
    inverse: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    moment_complex = np.einsum("ji,Pij->P", density, three_index, optimize=True)
    if np.max(np.abs(moment_complex.imag)) > 2.0e-10:
        raise RuntimeError("stored three-index action produced a complex fitted moment")
    moment = moment_complex.real
    coefficients = inverse @ moment
    lower = np.einsum("P,Pij->ij", coefficients, three_index, optimize=True)
    energy = 0.5 * float(moment @ coefficients)
    return energy, lower, moment, coefficients


def _analytic_field_free_data(
    reference: Any,
    auxiliary_basis: str,
) -> dict[str, Any]:
    from pyscf import df

    molecule = reconstruct_mean_field(reference, BackendConfig()).mol
    auxiliary = df.addons.make_auxmol(molecule, auxiliary_basis)
    metric = np.asarray(auxiliary.intor("int2c2e"), dtype=np.float64)
    three_index = np.asarray(
        df.incore.aux_e2(molecule, auxiliary, intor="int3c2e", aosym="s1")
    ).transpose(2, 0, 1)
    inverse, retained_values, retained_vectors = _metric_inverse(metric)
    density = reference.ground_state.density.astype(np.complex128)
    energy, lower, moment, coefficients = _action_from_three_index(
        density,
        three_index,
        inverse,
    )
    exact_eri = np.asarray(molecule.intor("int2e"))
    exact_lower = np.einsum("ijkl,lk->ij", exact_eri, density, optimize=True)
    exact_energy = 0.5 * float(np.einsum("ij,ji->", exact_lower, density, optimize=True).real)
    return {
        "metric": metric,
        "three_index": three_index,
        "inverse": inverse,
        "retained_values": retained_values,
        "retained_vectors": retained_vectors,
        "energy": energy,
        "lower": lower,
        "moment": moment,
        "coefficients": coefficients,
        "exact_energy": exact_energy,
        "exact_lower": exact_lower,
        "naux": int(metric.shape[0]),
    }


def _auxiliary_rows(system: str, reference: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for auxiliary_basis in _AUXILIARY_BASES[system]:
        data = _analytic_field_free_data(reference, auxiliary_basis)
        row = {
            "system": system,
            "auxiliary_basis": auxiliary_basis,
            "naux": data["naux"],
            "retained_rank": int(data["retained_values"].size),
            "ri_energy_au": data["energy"],
            "exact_four_center_energy_au": data["exact_energy"],
            "energy_absolute_error_au": abs(data["energy"] - data["exact_energy"]),
            "energy_relative_error": abs(data["energy"] - data["exact_energy"])
            / abs(data["exact_energy"]),
            "lower_relative_error": _relative(data["lower"], data["exact_lower"]),
        }
        rows.append(row)
        if auxiliary_basis == "weigend":
            selected = data
    assert selected is not None
    return rows, selected


def _grid_rows(
    system: str,
    reference: Any,
    analytic: dict[str, Any],
) -> list[dict[str, Any]]:
    zero = AffineMagneticGauge(UniformMagneticField((0.0, 0.0, 0.0)))
    density = reference.ground_state.density.astype(np.complex128)
    rows: list[dict[str, Any]] = []
    for level in _GRID_LEVELS:
        quadrature = prepare_ao_quadrature(
            reference,
            BackendConfig(),
            grid_policy=AOGridPolicy.qualification(level),
            block_size=1024,
        )
        evaluator = prepare_ri_wilson_hartree(quadrature, "weigend")
        result = evaluator.evaluate(density, zero)
        rows.append(
            {
                "system": system,
                "level": level,
                "points": quadrature.grid.npoints,
                "grid_fingerprint_sha256": quadrature.grid.fingerprint_sha256,
                "energy_au": float(result.energy),
                "analytic_ri_energy_au": analytic["energy"],
                "energy_absolute_error_au": abs(float(result.energy) - analytic["energy"]),
                "energy_relative_error": abs(float(result.energy) - analytic["energy"])
                / abs(analytic["energy"]),
                "lower_relative_error": _relative(result.lower_coulomb_matrix, analytic["lower"]),
                "three_index_relative_error": _relative(
                    result.three_index, analytic["three_index"]
                ),
                "pair_counting_residual": result.pair_counting_residual,
                "lower_hermiticity_residual": result.lower_hermiticity_residual,
                "moment_imaginary_max_abs": result.moment_imaginary_max_abs,
                "retained_solve_relative_residual": (result.retained_solve_relative_residual),
                "stationary_energy_residual": result.stationary_energy_residual,
            }
        )
    return rows


def _rank_rows(system: str, density: np.ndarray, analytic: dict[str, Any]) -> list[dict[str, Any]]:
    metric = analytic["metric"]
    three_index = analytic["three_index"]
    full_energy = analytic["energy"]
    full_lower = analytic["lower"]
    dimension = int(metric.shape[0])
    ranks = sorted(
        {
            2,
            4,
            8,
            12,
            16,
            20,
            dimension // 2,
            3 * dimension // 4,
            9 * dimension // 10,
            dimension,
        }
    )
    rows: list[dict[str, Any]] = []
    for rank in ranks:
        if rank > dimension:
            continue
        inverse, retained_values, _ = _metric_inverse(metric, maximum_rank=rank)
        energy, lower, moment, coefficients = _action_from_three_index(
            density, three_index, inverse
        )
        projected_residual = np.linalg.norm(
            metric @ coefficients - metric @ inverse @ moment
        ) / max(1.0, np.linalg.norm(metric @ inverse @ moment))
        rows.append(
            {
                "system": system,
                "rank": int(retained_values.size),
                "dimension": dimension,
                "minimum_retained_eigenvalue": float(retained_values[0]),
                "energy_au": energy,
                "full_rank_energy_absolute_error_au": abs(energy - full_energy),
                "full_rank_lower_relative_error": _relative(lower, full_lower),
                "projected_solve_relative_residual": float(projected_residual),
            }
        )
    return rows


def _solve_rows(system: str, analytic: dict[str, Any]) -> list[dict[str, Any]]:
    values = np.asarray(analytic["retained_values"])
    vectors = np.asarray(analytic["retained_vectors"])
    reduced_moment = vectors.T @ np.asarray(analytic["moment"])
    exact_reduced = reduced_moment / values
    exact_energy = 0.5 * float(reduced_moment @ exact_reduced)
    diagonal = np.diag(values)
    rows: list[dict[str, Any]] = []
    for tolerance in _SOLVE_TOLERANCES:
        iterations = 0

        def _count(_: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        solution, info = cg(
            diagonal,
            reduced_moment,
            rtol=tolerance,
            atol=0.0,
            maxiter=1000,
            callback=_count,
        )
        residual = np.linalg.norm(diagonal @ solution - reduced_moment) / max(
            1.0, np.linalg.norm(reduced_moment)
        )
        objective = float(solution @ reduced_moment - 0.5 * solution @ diagonal @ solution)
        rows.append(
            {
                "system": system,
                "requested_relative_tolerance": tolerance,
                "iterations": iterations,
                "info": int(info),
                "achieved_relative_residual": float(residual),
                "stationary_objective_au": objective,
                "exact_stationary_energy_au": exact_energy,
                "energy_absolute_error_au": abs(objective - exact_energy),
                "coefficient_relative_error": _relative(solution, exact_reduced),
            }
        )
    return rows


def _matter_rows(
    system: str,
    coefficients: np.ndarray,
    occupations: np.ndarray,
    evaluator: Any,
    base_result: Any,
    inverse: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=coefficients.shape)
    three_index = np.asarray(base_result.three_index)
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
                np.asarray(base_result.lower_coulomb_matrix),
                density_direction,
                optimize=True,
            ).real
        )
        base_density = np.einsum(
            "mi,i,ni->mn",
            coefficients,
            occupations,
            coefficients.conj(),
            optimize=True,
        )
        base_energy = _action_from_three_index(base_density, three_index, inverse)[0]
        for step in _STEPS:
            energies: list[float] = []
            for sign in (1.0, -1.0):
                displaced = coefficients + sign * step * direction
                density = np.einsum(
                    "mi,i,ni->mn",
                    displaced,
                    occupations,
                    displaced.conj(),
                    optimize=True,
                )
                energies.append(_action_from_three_index(density, three_index, inverse)[0])
            plus = energies[0]
            minus = energies[1]
            forward = (plus - base_energy) / step
            central = (plus - minus) / (2.0 * step)
            rows.append(
                {
                    "system": system,
                    "kind": "matter",
                    "direction": label,
                    "step": step,
                    "analytic_direction_au": analytic,
                    "forward_absolute_error_au": abs(forward - analytic),
                    "central_absolute_error_au": abs(central - analytic),
                    "central_relative_error": abs(central - analytic) / max(1.0, abs(analytic)),
                }
            )
    assert evaluator.provenance.metric_rank == inverse.shape[0]
    return rows


def _source_rows(
    system: str,
    density: np.ndarray,
    evaluator: Any,
    gauge: AffineMagneticGauge,
    directions: tuple[tuple[str, Any], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, direction in directions:
        result = evaluator.evaluate(density, gauge, source_direction=direction)
        assert result.source_energy_direction is not None
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
                    "central_relative_error": abs(central - analytic) / max(1.0, abs(analytic)),
                    "source_moment_imaginary_max_abs": (result.source_moment_imaginary_max_abs),
                }
            )
    return rows


def _finite_field_result(
    system: str,
    reference: Any,
    analytic: dict[str, Any],
) -> dict[str, Any]:
    quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(5),
        block_size=1024,
    )
    evaluator = prepare_ri_wilson_hartree(quadrature, "weigend")
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
    matter_rows = _matter_rows(
        system,
        coefficients,
        occupations,
        evaluator,
        base,
        np.asarray(evaluator.metric_inverse),
        seed=2701 if system == "h2" else 2702,
    )
    source_rows = _source_rows(
        system,
        density,
        evaluator,
        symmetric,
        (
            ("physical_magnetic", physical_direction),
            ("pure_gauge", pure_gauge_direction),
        ),
    )

    anchors = reference.core_operators.nuclei.coordinates_au[reference.anchor_topology.ao_to_atom]
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

    rng = np.random.default_rng(3701 if system == "h2" else 3702)
    dimension = density.shape[0]
    raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    change = np.eye(dimension, dtype=np.complex128) + 0.05 * raw / np.linalg.norm(raw)
    inverse_change = np.linalg.inv(change)
    transformed_density = inverse_change @ density @ inverse_change.conj().T
    transformed_three_index = np.einsum(
        "ia,Pij,jb->Pab",
        change.conj(),
        np.asarray(base.three_index),
        change,
        optimize=True,
    )
    transformed_energy, transformed_lower, transformed_moment, _ = _action_from_three_index(
        transformed_density,
        transformed_three_index,
        np.asarray(evaluator.metric_inverse),
    )
    predicted_transformed_lower = change.conj().T @ np.asarray(base.lower_coulomb_matrix) @ change

    return {
        "grid_level": 5,
        "grid_points": quadrature.grid.npoints,
        "auxiliary_provenance": {
            name: getattr(evaluator.provenance, name)
            for name in evaluator.provenance.__dataclass_fields__
        },
        "base": {
            "energy_au": float(base.energy),
            "pair_counting_residual": base.pair_counting_residual,
            "lower_hermiticity_residual": base.lower_hermiticity_residual,
            "moment_imaginary_max_abs": base.moment_imaginary_max_abs,
            "retained_solve_relative_residual": base.retained_solve_relative_residual,
            "discarded_moment_relative_norm": base.discarded_moment_relative_norm,
            "stationary_energy_residual": base.stationary_energy_residual,
            "self_interaction_included": base.self_interaction_included,
        },
        "invariance": {
            "electromagnetic_gauge": {
                "energy_absolute_residual_au": abs(
                    float(landau_result.energy) - float(base.energy)
                ),
                "moment_relative_residual": _relative(landau_result.moment, base.moment),
            },
            "coefficient_frame": {
                "change_condition_number": float(np.linalg.cond(change)),
                "energy_absolute_residual_au": abs(transformed_energy - float(base.energy)),
                "moment_relative_residual": _relative(transformed_moment, base.moment),
                "lower_congruence_relative_residual": _relative(
                    transformed_lower, predicted_transformed_lower
                ),
            },
        },
        "matter_derivatives": matter_rows,
        "source_derivatives": source_rows,
        "field_free_reference_note": {
            "independent_four_center_energy_au": analytic["exact_energy"],
            "analytic_ri_energy_au": analytic["energy"],
        },
    }


def _system_result(system: str, output: Path, timestamp: str) -> dict[str, Any]:
    config = _configuration(system, output / f"{system}.reference.h5", timestamp)
    (output / f"{system}.reference.toml").write_text(dumps_config(config), encoding="utf-8")
    reference = prepare_pyscf_reference(config)
    reference.save()
    auxiliary_rows, analytic = _auxiliary_rows(system, reference)
    density = reference.ground_state.density.astype(np.complex128)
    return {
        "system": system,
        "scientific_id": config.scientific_id,
        "reference_fingerprint_sha256": reference.fingerprint_sha256,
        "electron_count": reference.ground_state.electron_count,
        "nao": reference.core_operators.nao,
        "occupations": reference.ground_state.occupations.tolist(),
        "grid_refinement": _grid_rows(system, reference, analytic),
        "auxiliary_convergence": auxiliary_rows,
        "rank_convergence": _rank_rows(system, density, analytic),
        "solve_convergence": _solve_rows(system, analytic),
        "finite_field": _finite_field_result(system, reference, analytic),
    }


def _write_csv(output: Path, name: str, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with (output / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_tables(output: Path, systems: list[dict[str, Any]]) -> None:
    for family, filename in (
        ("grid_refinement", "grid_refinement.csv"),
        ("auxiliary_convergence", "auxiliary_convergence.csv"),
        ("rank_convergence", "rank_convergence.csv"),
        ("solve_convergence", "solve_convergence.csv"),
    ):
        _write_csv(
            output,
            filename,
            [row for system in systems for row in system[family]],
        )
    derivative_rows = [
        row
        for system in systems
        for family in ("matter_derivatives", "source_derivatives")
        for row in system["finite_field"][family]
    ]
    _write_csv(output, "derivative_sequences.csv", derivative_rows)
    invariance_rows = [
        {"system": system["system"], "kind": kind, **values}
        for system in systems
        for kind, values in system["finite_field"]["invariance"].items()
    ]
    _write_csv(output, "invariance.csv", invariance_rows)


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
        "schema": "aion.chapter13-nq2-ri-wilson-hartree",
        "schema_version": "1.0.0",
        "status": "executed_unreviewed",
        "equations": [
            "eq:wilson-hartree-ks-ri-metric",
            "eq:wilson-hartree-ks-ri-three-index-data",
            "eq:wilson-hartree-ks-ri-moments-coefficients",
            "eq:wilson-hartree-ks-ri-energy",
            "eq:wilson-hartree-ks-ri-coulomb-matrix",
            "eq:wilson-hartree-ks-ri-matter-differential",
            "eq:wilson-hartree-ks-ri-pair-counting",
            "eq:wilson-hartree-ks-ri-source-differential",
            "eq:wilson-hartree-ks-ri-metric-em-direction-zero",
        ],
        "realization": {
            "systems": ["h2", "lih"],
            "orbital_basis": "sto-3g",
            "selected_auxiliary_basis": "weigend",
            "hartree_realization": "Coulomb-metric RI-Wilson Hartree",
            "rank_policy": {
                "relative_eigenvalue_threshold": 0.0,
                "absolute_eigenvalue_threshold": 1.0e-7,
                "maximum_rank": None,
                "provenance": "PySCF df.incore.LINEAR_DEP_THR accepted at NQ0",
            },
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
            "hartree_self_interaction": "included",
            "scalar_potential_in_mechanical_energy": False,
        },
        "matter_and_source_steps": list(_STEPS),
        "solve_tolerances": list(_SOLVE_TOLERANCES),
        "systems": systems,
    }
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_tables(output, systems)

    tracked_diff = _git(repo, "diff", "--binary", "HEAD", binary=True)
    assert isinstance(tracked_diff, bytes)
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    assert isinstance(status, str)
    head = _git(repo, "rev-parse", "HEAD")
    assert isinstance(head, str)
    source_paths = (
        repo / "tools/run_chapter13_nq2_ri_wilson_hartree.py",
        repo / "src/aion/electronic_structure/ri_wilson_hartree.py",
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
        "schema": "aion.chapter13-nq2-completed",
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
