#!/usr/bin/env python3
"""Synthesize the accepted G3--G7 exact one-electron evidence for WP8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aion.config import canonical_sha256

_CAMPAIGNS = Path(
    "/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/exact_one_electron_qualification"
)
_WP3 = _CAMPAIGNS / "wp3_pair_distance_20260917T002456Z_3fd2a83b77d1/g3_analysis"
_WP4 = _CAMPAIGNS / "wp4_first_order_20260918T191425Z_3fd2a83b77d1/analysis"
_WP5 = _CAMPAIGNS / "wp5_multicentre_20260918T200808Z_3fd2a83b77d1/analysis"
_WP6 = (
    _CAMPAIGNS / "wp6_mixed_magnus_n128_diagnostics_20260919T124656Z_73f6dc1cb760/"
    "endpoint_observables"
)
_WP7 = _CAMPAIGNS / "wp7_observables_cpu_20260919T145221Z_7d603105e1db/analysis"
_REVIEW_DIRECTORY = Path(__file__).resolve().parents[1] / "docs/reviews"
_MODELS = ("p0", "geometric_b1", "full_b1", "complete_first")
_MODEL_LABELS = {
    "p0": "P0",
    "geometric_b1": "gB1",
    "full_b1": "B1",
    "complete_first": "C1",
    "complete_first_order": "C1",
    "e1": "E1",
    "exact": "EX",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wp3-analysis", type=Path, default=_WP3)
    parser.add_argument("--wp4-analysis", type=Path, default=_WP4)
    parser.add_argument("--wp5-analysis", type=Path, default=_WP5)
    parser.add_argument("--wp6-endpoint-analysis", type=Path, default=_WP6)
    parser.add_argument("--wp7-analysis", type=Path, default=_WP7)
    parser.add_argument("--review-directory", type=Path, default=_REVIEW_DIRECTORY)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return dict(value)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _review_paths(directory: Path) -> dict[str, Path]:
    paths = {
        gate: directory / f"exact_one_electron_{gate.lower()}_review_20260919.json"
        for gate in ("G6", "G7")
    }
    paths.update(
        {
            "G3": directory / "exact_one_electron_g3_review_20260918.json",
            "G4": directory / "exact_one_electron_g4_review_20260918.json",
            "G5": directory / "exact_one_electron_g5_review_20260918.json",
        }
    )
    for gate, path in paths.items():
        review = _json(path)
        if review.get("gate") != gate or review.get("decision") != "accepted":
            raise ValueError(f"{gate} is not represented by an accepted review: {path}")
    return paths


def _authenticate_reviewed_summaries(
    reviews: dict[str, Path],
    wp3_summary: Path,
    wp4_summary: Path,
    wp5_summary: Path,
    wp6_endpoint_summary: Path,
    wp7_summary: Path,
) -> None:
    expected = {
        "G3": ("analysis_summary_sha256", wp3_summary),
        "G4": ("analysis_summary_sha256", wp4_summary),
        "G5": ("analysis_summary_sha256", wp5_summary),
        "G6": ("endpoint_observable_summary_sha256", wp6_endpoint_summary),
        "G7": ("analysis_summary_sha256", wp7_summary),
    }
    for gate, (field, path) in expected.items():
        review = _json(reviews[gate])
        if review["evidence"][field] != _sha256(path):
            raise ValueError(f"{gate} accepted-review hash does not authenticate {path}")


def _float_or_none(value: str) -> float | None:
    return None if value == "" else float(value)


def _validity_row(
    key: tuple[str, ...],
    rows: list[dict[str, str]],
    p0_floor: float | None,
    quadrature_floors: dict[tuple[str, str, str, str], float],
) -> dict[str, Any]:
    lowers = [float(row["lower_bracket_au"]) for row in rows]
    uppers = [
        value for row in rows if (value := _float_or_none(row["upper_bracket_au"])) is not None
    ]
    conservative = min(lowers)
    model = key[-3]
    extension = None
    if p0_floor is not None and p0_floor > 0.0:
        extension = conservative / p0_floor
    floors = [
        quadrature_floors[(row["system"], row["basis"], row["bond_scale"], row["family"])]
        for row in rows
    ]
    return {
        "system": key[0] if len(key) == 5 else "all",
        "basis": key[1] if len(key) == 5 else "all",
        "model": model,
        "model_label": _MODEL_LABELS[model],
        "family": key[-2],
        "relative_threshold": float(key[-1]),
        "sample_count": len(rows),
        "bond_scale_min": min(float(row["bond_scale"]) for row in rows),
        "bond_scale_max": max(float(row["bond_scale"]) for row in rows),
        "directions": ";".join(sorted({row["direction"] for row in rows})),
        "conservative_confirmed_below_au": conservative,
        "median_last_below_au": statistics.median(lowers),
        "maximum_last_below_au": max(lowers),
        "minimum_first_above_au": min(uppers) if uppers else None,
        "maximum_first_above_au": max(uppers) if uppers else None,
        "uncrossed_through_1au_count": sum(row["crossed"] == "False" for row in rows),
        "conservative_extension_over_p0": extension,
        "maximum_refined_quadrature_floor": max(floors),
    }


def _validity_tables(
    rows: list[dict[str, str]], remainder_rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quadrature_floors = {
        (row["system"], row["basis"], row["bond_scale"], row["family"]): float(row["refined_floor"])
        for row in remainder_rows
    }
    by_system: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    global_groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_system[
            (
                row["system"],
                row["basis"],
                row["model"],
                row["family"],
                row["relative_threshold"],
            )
        ].append(row)
        global_groups[(row["model"], row["family"], row["relative_threshold"])].append(row)

    by_system_p0 = {
        (key[0], key[1], key[-2], key[-1]): min(float(row["lower_bracket_au"]) for row in values)
        for key, values in by_system.items()
        if key[-3] == "p0"
    }
    global_p0 = {
        (key[-2], key[-1]): min(float(row["lower_bracket_au"]) for row in values)
        for key, values in global_groups.items()
        if key[-3] == "p0"
    }
    detailed = [
        _validity_row(
            key,
            values,
            by_system_p0[(key[0], key[1], key[-2], key[-1])],
            quadrature_floors,
        )
        for key, values in sorted(by_system.items())
    ]
    global_rows = [
        _validity_row(
            key,
            values,
            global_p0[(key[-2], key[-1])],
            quadrature_floors,
        )
        for key, values in sorted(global_groups.items())
    ]
    return detailed, global_rows


def _channel_rows(
    rows: list[dict[str, str]], remainder_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    floor_by_member = {
        (row["system"], row["basis"], row["bond_scale"]): float(row["refined_floor"])
        for row in remainder_rows
        if row["family"] == "mechanical"
    }
    groups: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        if row["status"] != "mechanical_b1_required":
            continue
        location = "onsite" if row["bra_atom"] == row["ket_atom"] else "offsite"
        groups[(location, row["bra_l"], row["ket_l"])].append(
            {
                "improvement": float(row["improvement_factor"]),
                "increment": float(row["mechanical_increment_norm"]),
                "gb1_error": float(row["geometric_b1_error"]),
                "b1_error": float(row["full_b1_error"]),
                "floor": floor_by_member[(row["system"], row["basis"], row["bond_scale"])],
            }
        )
    return [
        {
            "channel": "anchored_vector_mechanical_B1",
            "location": location,
            "bra_l": int(bra_l),
            "ket_l": int(ket_l),
            "resolved_block_count": len(values),
            "minimum_mechanical_increment_norm": min(row["increment"] for row in values),
            "median_mechanical_increment_norm": statistics.median(
                row["increment"] for row in values
            ),
            "maximum_mechanical_increment_norm": max(row["increment"] for row in values),
            "median_gB1_error": statistics.median(row["gb1_error"] for row in values),
            "median_B1_error": statistics.median(row["b1_error"] for row in values),
            "minimum_B1_improvement_factor": min(row["improvement"] for row in values),
            "median_B1_improvement_factor": statistics.median(row["improvement"] for row in values),
            "maximum_B1_improvement_factor": max(row["improvement"] for row in values),
            "maximum_refined_quadrature_floor": max(row["floor"] for row in values),
        }
        for (location, bra_l, ket_l), values in sorted(groups.items())
    ]


def _maximum_pair_changes(
    rows: list[dict[str, str]], geometry: str, basis: str, model: str, family: str
) -> float:
    values = [
        float(row["h3_refinement_change"])
        for row in rows
        if row["geometry"] == geometry
        and row["basis"] == basis
        and row["model"] == model
        and row["family"] == family
    ]
    if not values:
        raise ValueError(f"missing {geometry}/{basis}/{model}/{family} pair refinements")
    return max(values)


def _spectral_rows(
    spectra: list[dict[str, str]], pair_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in spectra:
        if source["sign"] != "1":
            continue
        for model, prefix in (
            ("p0", "p0"),
            ("geometric_b1", "geometric_b1"),
            ("full_b1", "full_b1"),
            ("complete_first", "full_b1"),
        ):
            pair_model = "full_b1" if model == "complete_first" else model
            output.append(
                {
                    "geometry": source["geometry"],
                    "basis": source["basis"],
                    "field_class": source["label"],
                    "loop_flux_au": float(source["loop_flux_au"]),
                    "field_z_au": float(source["field_z_au"]),
                    "model": model,
                    "model_label": _MODEL_LABELS[model],
                    "maximum_absolute_eigenvalue_shift_ha": float(
                        source[f"{prefix}_maximum_absolute_eigenvalue_shift"]
                    ),
                    "maximum_subspace_angle_rad": float(
                        source[f"{prefix}_maximum_subspace_angle_rad"]
                    ),
                    "overlap_quadrature_change": _maximum_pair_changes(
                        pair_rows,
                        source["geometry"],
                        source["basis"],
                        pair_model,
                        "overlap",
                    ),
                    "kinetic_quadrature_change_ha": _maximum_pair_changes(
                        pair_rows,
                        source["geometry"],
                        source["basis"],
                        pair_model,
                        "kinetic",
                    ),
                    "exact_gauge_spectrum_residual_ha": float(
                        source["exact_gauge_spectrum_residual"]
                    ),
                    "static_c1_equals_b1": model == "complete_first",
                }
            )
    return output


def _dynamic_rows(
    rows: list[dict[str, str]],
    pair_rows: list[dict[str, str]],
    e1_quadrature_residual: float,
) -> list[dict[str, Any]]:
    indexed = {(int(row["intervals"]), row["model"]): row for row in rows}
    exact_fine = indexed[(128, "exact")]
    exact_coarse = indexed[(64, "exact")]
    output: list[dict[str, Any]] = []
    quantities = (
        "absorbed_energy_ha",
        "excitation_probability",
        "metric_hilbert_schmidt_distance_from_exact",
    )
    for model in ("p0", "e1", "geometric_b1", "full_b1", "complete_first_order"):
        fine = indexed[(128, model)]
        coarse = indexed[(64, model)]
        row: dict[str, Any] = {
            "geometry": "distorted_h3",
            "basis": "cc-pvdz",
            "field_class": "above_threshold",
            "loop_flux_peak_au": 0.05,
            "model": model,
            "model_label": _MODEL_LABELS[model],
            "fine_intervals": 128,
            "fine_step_au": float(fine["step_au"]),
        }
        pair_model = {
            "p0": "p0",
            "e1": "p0",
            "geometric_b1": "geometric_b1",
            "full_b1": "full_b1",
            "complete_first_order": "full_b1",
        }[model]
        row["overlap_matrix_quadrature_change"] = _maximum_pair_changes(
            pair_rows, "distorted", "cc-pvdz", pair_model, "overlap"
        )
        row["kinetic_matrix_quadrature_change_ha"] = _maximum_pair_changes(
            pair_rows, "distorted", "cc-pvdz", pair_model, "kinetic"
        )
        row["e1_relative_quadrature_residual"] = (
            e1_quadrature_residual if model in ("e1", "complete_first_order") else 0.0
        )
        for quantity in quantities:
            fine_value = float(fine[quantity])
            exact_value = float(exact_fine[quantity])
            coarse_difference = float(coarse[quantity]) - float(exact_coarse[quantity])
            fine_difference = fine_value - exact_value
            stem = quantity.removesuffix("_ha")
            row[f"{stem}_fine"] = fine_value
            row[f"{stem}_exact_fine"] = exact_value
            row[f"{stem}_difference_from_exact"] = fine_difference
            row[f"{stem}_timestep_difference_uncertainty"] = abs(
                fine_difference - coarse_difference
            )
        output.append(row)
    return output


def _hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): _sha256(path) for path in paths}


def _report(summary: dict[str, Any]) -> str:
    measurements = summary["measurements"]
    dynamics = measurements["above_threshold_dynamics"]
    channels = measurements["anchored_vector_channels"]
    e1 = measurements["electric_e1"]
    observables = measurements["action_observables"]
    branch = summary["branch_recommendation"]
    return "\n".join(
        [
            "# WP8 exact one-electron qualification synthesis",
            "",
            "Status: **authenticated derived evidence; G8 user review pending**",
            "",
            "The statements below are restricted to the accepted G3--G7 records listed in ",
            "`manifest.json`. The tables in this directory retain the numerical resolution ",
            "beside each physical comparison.",
            "",
            "## Numerical evidence",
            "",
            "- The pair campaign covers H2, O--H, N2, and CO; STO-3G, cc-pVDZ, and ",
            "  aug-cc-pVDZ; bond scales 0.8, 1.0, and 1.25; and parallel, perpendicular, ",
            "  and oblique uniform magnetic fields through 1 a.u. Exact per-system and ",
            "  conservative all-system validity brackets are in ",
            "  `tables/magnetic_validity_by_system_basis.csv` and ",
            "  `tables/magnetic_validity_global.csv` [G3, G4].",
            "- gB1 extends the overlap/triangle-flux validity envelope but does not extend ",
            "  the kinetic, nuclear-attraction, or mechanical envelope. B1 extends all four ",
            "  families; C1 equals B1 in the static electric-free sector [G4].",
            f"- The mechanical B1 term is resolved as necessary in {channels['total']} ",
            f"  angular/centre blocks, including {channels['onsite']} onsite and ",
            f"  {channels['offsite']} offsite blocks. Its median error-reduction factor is ",
            f"  {channels['median_improvement_factor']:.3g} ",
            "  (`tables/channel_attribution.csv`) [G4].",
            "- The independent E1 coefficient agrees with its exact parent to ",
            f"  {e1['maximum_parent_residual']:.3e}; its finest-grid position-quadrature ",
            f"  relative residual is {e1['maximum_level5_relative_residual']:.3e} [G4].",
            "- Three-centre spectra at sub-, near-, and above-threshold loop flux are in ",
            "  `tables/multicentre_spectral_differences.csv`; every eigenvalue/subspace ",
            "  difference is accompanied by overlap and kinetic quadrature changes [G5].",
            "- In the distorted above-threshold trajectory, C1 changes the absorbed energy ",
            f"  from the exact {dynamics['exact_absorbed_energy_ha']:.6e} Ha to ",
            f"  {dynamics['c1_absorbed_energy_ha']:.6e} Ha. The absolute model difference is ",
            f"  {abs(dynamics['c1_absorbed_energy_difference_ha']):.3e} Ha and the raw n64--n128 ",
            "  difference-of-differences is ",
            f"  {dynamics['c1_absorbed_energy_timestep_uncertainty_ha']:.3e} Ha ",
            "  (`tables/dynamic_endpoint_differences.csv`) [G6].",
            "- Action-derived power, continuity, Ward, and charge identities close at the ",
            "  accepted floors; the finest electric work defect is ",
            f"  {observables['integrated_work_defect_ha']:.3e} Ha and the largest listed ",
            "  finest-grid continuity residual is ",
            f"  {observables['maximum_continuity_residual']:.3e} [G7].",
            "",
            "## Interpretation",
            "",
            "- P0 is a controlled small-field approximation only inside the tabulated ",
            "  field/geometry/basis brackets; no single molecule-independent field cutoff ",
            "  should be inferred from it.",
            "- Triangle-flux corrections control the overlap metric, while anchored-vector ",
            "  corrections control the resolved onsite and offsite mechanical channels. ",
            "  Loop flux makes these omissions visible in multicentre spectra.",
            "- Neither E1 nor B1 alone reproduces the tested simultaneous electric/magnetic ",
            "  dynamics. C1 is the only first-order truncation that tracks the exact ",
            "  endpoint observables in the above-threshold fixture.",
            "",
            "## Extrapolation boundary and unresolved regions",
            "",
            "- No statement is made about self-consistent Hartree/XC response, nonlinear ",
            "  SCEM propagation, moving nuclei, pseudopotentials, periodic systems, or ",
            "  pointwise current-density reconstruction.",
            "- The dynamic evidence is one linear electron on fixed three-proton frameworks, ",
            "  not physical two-electron H3+ and not an adiabatic-TDDFT closure test.",
            "- The reported ranges are interpolation-free scan brackets. Values beyond the ",
            "  last tested field, basis, geometry, or source family are untested rather than ",
            "  failed.",
            "",
            "## Branch recommendation",
            "",
            f"Recommended next branch: **{branch['label']}** (plan option ",
            f"{branch['plan_option']}). The measured implementation gap is now the nonlinear ",
            "self-consistent Hartree/XC closure, whereas the fixed-centre linear exact ",
            "one-electron matrices, transport, observables, and CPU/GPU parity have accepted ",
            "evidence. Nonuniform/higher-curvature work remains valuable but is not required ",
            "to expose the present exact evaluator to Aion's intended TDDFT feedback loop.",
            "",
            "This recommendation is not G8 acceptance. The branch choice becomes reviewed ",
            "evidence only after an explicit user decision.",
            "",
        ]
    )


def main() -> None:
    arguments = _arguments()
    output = arguments.output.resolve()
    tables = output / "tables"
    output.mkdir(parents=True, exist_ok=False)
    tables.mkdir()

    reviews = _review_paths(arguments.review_directory.resolve())
    wp3_summary_path = arguments.wp3_analysis / "summary.json"
    wp4_summary_path = arguments.wp4_analysis / "summary.json"
    wp5_summary_path = arguments.wp5_analysis / "summary.json"
    wp6_summary_path = arguments.wp6_endpoint_analysis / "summary.json"
    wp7_summary_path = arguments.wp7_analysis / "analysis_summary.json"
    _authenticate_reviewed_summaries(
        reviews,
        wp3_summary_path,
        wp4_summary_path,
        wp5_summary_path,
        wp6_summary_path,
        wp7_summary_path,
    )
    wp3_summary = _json(wp3_summary_path)
    wp4_summary = _json(wp4_summary_path)
    wp5_summary = _json(wp5_summary_path)
    wp6_summary = _json(wp6_summary_path)
    wp7_summary = _json(wp7_summary_path)
    validity_source = arguments.wp4_analysis / "tables/model_validity.csv"
    remainder_source = arguments.wp4_analysis / "tables/remainder_fits.csv"
    channels_source = arguments.wp4_analysis / "tables/mechanical_channel_classification.csv"
    spectra_source = arguments.wp5_analysis / "tables/generalized_spectral_comparisons.csv"
    pairs_source = arguments.wp5_analysis / "tables/pair_restrictions.csv"
    dynamics_source = arguments.wp6_endpoint_analysis / "endpoint_observables.csv"

    remainder_rows = _csv(remainder_source)
    pair_rows = _csv(pairs_source)
    detailed_validity, global_validity = _validity_tables(_csv(validity_source), remainder_rows)
    channels = _channel_rows(_csv(channels_source), remainder_rows)
    spectra = _spectral_rows(_csv(spectra_source), pair_rows)
    dynamics = _dynamic_rows(
        _csv(dynamics_source),
        pair_rows,
        float(wp4_summary["checks"]["maximum_e1_level5_relative_residual"]),
    )
    table_values = {
        "magnetic_validity_by_system_basis.csv": detailed_validity,
        "magnetic_validity_global.csv": global_validity,
        "channel_attribution.csv": channels,
        "multicentre_spectral_differences.csv": spectra,
        "dynamic_endpoint_differences.csv": dynamics,
    }
    for name, rows in table_values.items():
        _write_csv(tables / name, rows)

    required_channels = _csv(channels_source)
    improvements = [
        float(row["improvement_factor"])
        for row in required_channels
        if row["status"] == "mechanical_b1_required"
    ]
    c1 = next(row for row in dynamics if row["model"] == "complete_first_order")
    manifest_sources = [
        wp3_summary_path,
        wp4_summary_path,
        validity_source,
        remainder_source,
        channels_source,
        wp5_summary_path,
        spectra_source,
        pairs_source,
        wp6_summary_path,
        dynamics_source,
        wp7_summary_path,
        *reviews.values(),
    ]
    manifest_core = {
        "schema": "aion.exact-one-electron-wp8-synthesis-manifest",
        "version": "1.0.0",
        "status": "derived_executed_unreviewed",
        "analyzer_sha256": _sha256(Path(__file__).resolve()),
        "accepted_review_sha256": {gate: _sha256(path) for gate, path in sorted(reviews.items())},
        "source_sha256": _hashes(manifest_sources),
        "rules": {
            "validity_envelope": (
                "minimum last-below field over every member in the stated grouping"
            ),
            "validity_range": "scan brackets only; no interpolation or extrapolation",
            "dynamic_timestep_uncertainty": (
                "absolute n128-minus-n64 change of each model-minus-exact difference"
            ),
            "spectral_quadrature_companions": (
                "maximum H3 level4-to-level5 overlap and kinetic matrix changes over edges"
            ),
            "dynamic_spatial_quadrature_companions": (
                "same H3 matrix changes plus the G4 E1 level-5 relative residual; these are "
                "matrix-level resolution measures, not propagated observable error bars"
            ),
        },
    }
    manifest_id = canonical_sha256(manifest_core)
    manifest = {**manifest_core, "manifest_id": manifest_id}
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)

    finest_electric = wp7_summary["finest_cpu_electric"]
    finest_magnetic = wp7_summary["finest_cpu_magnetic"]
    summary = {
        "schema": "aion.exact-one-electron-wp8-synthesis-summary",
        "version": "1.0.0",
        "status": "derived_executed_unreviewed",
        "manifest_id": manifest_id,
        "accepted_gate_inputs": sorted(reviews),
        "coverage": {
            "pair_systems": wp4_summary["coverage"]["systems"],
            "pair_bases": wp4_summary["coverage"]["bases"],
            "pair_bond_scales": wp4_summary["coverage"]["bond_scales"],
            "pair_field_range_au": [1.0e-7, 1.0],
            "pair_field_directions": [
                "parallel",
                "perpendicular_x",
                "perpendicular_y",
                "oblique",
            ],
            "multicentre_geometries": ["equilateral_h3", "distorted_h3"],
            "dynamic_scope": "fixed-centre linear one-electron",
        },
        "measurements": {
            "wp3_checks": wp3_summary["checks"],
            "wp5_checks": wp5_summary["checks"],
            "electric_e1": {
                "maximum_parent_residual": wp4_summary["checks"]["maximum_e1_parent_residual"],
                "maximum_level5_relative_residual": wp4_summary["checks"][
                    "maximum_e1_level5_relative_residual"
                ],
            },
            "anchored_vector_channels": {
                "total": len(improvements),
                "onsite": sum(
                    row["status"] == "mechanical_b1_required" and row["bra_atom"] == row["ket_atom"]
                    for row in required_channels
                ),
                "offsite": sum(
                    row["status"] == "mechanical_b1_required" and row["bra_atom"] != row["ket_atom"]
                    for row in required_channels
                ),
                "minimum_improvement_factor": min(improvements),
                "median_improvement_factor": statistics.median(improvements),
                "maximum_improvement_factor": max(improvements),
            },
            "above_threshold_dynamics": {
                "exact_absorbed_energy_ha": c1["absorbed_energy_exact_fine"],
                "c1_absorbed_energy_ha": c1["absorbed_energy_fine"],
                "c1_absorbed_energy_difference_ha": c1["absorbed_energy_difference_from_exact"],
                "c1_absorbed_energy_timestep_uncertainty_ha": c1[
                    "absorbed_energy_timestep_difference_uncertainty"
                ],
                "c1_excitation_probability_difference": c1[
                    "excitation_probability_difference_from_exact"
                ],
                "c1_excitation_probability_timestep_uncertainty": c1[
                    "excitation_probability_timestep_difference_uncertainty"
                ],
                "c1_density_distance": c1["metric_hilbert_schmidt_distance_from_exact_fine"],
                "c1_density_distance_timestep_uncertainty": c1[
                    "metric_hilbert_schmidt_distance_from_exact_timestep_difference_uncertainty"
                ],
                "endpoint_definition": wp6_summary["definitions"],
            },
            "action_observables": {
                "integrated_work_defect_ha": finest_electric["integrated_work_defect"],
                "maximum_instantaneous_power_residual": finest_electric[
                    "maximum_instantaneous_power_residual"
                ],
                "maximum_continuity_residual": max(
                    finest_electric["maximum_continuity_residual"],
                    finest_magnetic["maximum_continuity_residual"],
                ),
                "maximum_ward_residual": max(
                    finest_electric["maximum_ward_residual"],
                    finest_magnetic["maximum_ward_residual"],
                ),
                "maximum_total_charge_error": max(
                    finest_electric["maximum_total_charge_error"],
                    finest_magnetic["maximum_total_charge_error"],
                ),
            },
        },
        "branch_recommendation": {
            "plan_option": 1,
            "identifier": "aion_adiabatic_pure_lda_gga_tddft_closure",
            "label": "Aion adiabatic pure-LDA/GGA TDDFT closure",
            "decision_status": "recommended_pending_user_review",
            "measured_limitation": (
                "self-consistent Hartree/XC source response and nonlinear SCEM propagation "
                "remain outside the accepted fixed-centre linear one-electron scope"
            ),
            "deferred_not_rejected": [
                "Alekto nonorthogonal DFTB2 closure",
                "nonuniform-field and higher-order curvature qualification",
                "alternative equivariant projector and dressing constructions",
                "independent continuum-grid benchmark",
            ],
        },
        "untested_regions": [
            "self-consistent Hartree/XC source response",
            "nonlinear SCEM propagation",
            "moving nuclei",
            "pseudopotentials",
            "periodic systems",
            "general nonuniform-field power balance",
            "pointwise current-density reconstruction",
        ],
        "artifacts": {
            "tables": {name: _sha256(tables / name) for name in table_values},
        },
        "row_counts": {Path(name).stem: len(rows) for name, rows in table_values.items()},
        "review": {"reviewed": False, "reviewer": None, "decision": "pending"},
    }
    summary_path = output / "summary.json"
    report_path = output / "synthesis.md"
    _write_json(summary_path, summary)
    report_path.write_text(_report(summary), encoding="utf-8")
    _write_json(
        output / "completed.json",
        {
            "status": "authenticated_derived_executed_unreviewed",
            "manifest_id": manifest_id,
            "manifest_sha256": _sha256(manifest_path),
            "summary_sha256": _sha256(summary_path),
            "synthesis_sha256": _sha256(report_path),
        },
    )
    print(f"output={output}")
    print(f"manifest_id={manifest_id}")
    print("recommended_branch=1:aion_adiabatic_pure_lda_gga_tddft_closure")
    print("status=authenticated_derived_executed_unreviewed")


if __name__ == "__main__":
    main()
