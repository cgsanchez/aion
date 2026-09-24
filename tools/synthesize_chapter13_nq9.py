"""Authenticate NQ0--NQ8 and build the Chapter 13 NQ9 synthesis package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Claim:
    claim_id: str
    mathematical_object: str
    implementation: str
    tests: str
    gates: str
    numerical_floor: str
    supported_scope: str


CLAIMS = (
    Claim(
        "density",
        r"n_W(r)=P^{ij} chi_i^W(r) chi_j^W(r)^*",
        "src/aion/electronic_structure/wilson_density.py",
        "tests/test_wilson_density.py; tests/test_wilson_density_gpu.py",
        "NQ1,NQ3,NQ8",
        "direct/factorized relative 2.54e-16; grid normalization 2.32e-11",
        "fixed-centre molecular Gaussian bases and declared Wilson paths",
    ),
    Claim(
        "density_directions",
        r"delta_P n_W and fixed-history delta_A n_W",
        "src/aion/electronic_structure/wilson_density.py",
        "tests/test_wilson_density.py; tests/test_wilson_density_gpu.py",
        "NQ1",
        "matter relative 1.68e-12; source absolute 4.92e-11",
        "unrestricted complex coefficient directions and tested affine sources",
    ),
    Claim(
        "ri_hartree",
        r"E_H=1/2 rho^T J^{-1} rho and its two derivatives",
        "src/aion/electronic_structure/ri_wilson_hartree.py",
        "tests/test_ri_wilson_hartree.py; tests/test_ri_wilson_hartree_gpu.py",
        "NQ2",
        "grid energy 1.53e-10 relative; selected auxiliary error 6.90e-5 relative",
        "Coulomb-metric RI with each declared auxiliary basis and rank policy",
    ),
    Claim(
        "lda_action",
        r"E_xc,Q=sum_g w_g f(n_W), V_xc=delta E_xc,Q/delta P",
        "src/aion/electronic_structure/wilson_lda.py",
        "tests/test_wilson_lda.py; tests/test_wilson_lda_gpu.py",
        "NQ3",
        "same-grid energy 1.33e-15 Ha; lower matrix 3.39e-16 relative",
        "restricted spin-unpolarized pure lda,vwn realization",
    ),
    Claim(
        "gga_action",
        r"E_xc,Q=sum_g w_g f(n_W,grad n_W) and its two derivatives",
        "src/aion/electronic_structure/wilson_gga.py",
        "tests/test_wilson_gga.py; tests/test_wilson_gga_gpu.py",
        "NQ8",
        "same-grid energy 5.33e-15 Ha; lower matrix 4.14e-16 relative",
        "restricted spin-unpolarized pure PBE realization",
    ),
    Claim(
        "stationary",
        r"K[P] C=S C epsilon with fixed occupations",
        "src/aion/electronic_structure/wilson_stationary.py",
        "tests/test_wilson_stationary.py; tests/test_wilson_stationary_gpu.py",
        "NQ4,NQ8",
        "orbital residual 3.92e-12 across accepted LDA/PBE fixtures",
        "closed-shell H2, H3+, and CO in the accepted fields and bases",
    ),
    Claim(
        "source_current",
        r"J_alpha=-partial_alpha S_action at fixed coefficient history",
        "src/aion/electronic_structure/wilson_sources.py",
        "tests/test_wilson_sources.py; tests/test_wilson_sources_gpu.py",
        "NQ5,NQ8",
        "LDA complete source 3.64e-10; CO PBE complete source 7.39e-9",
        "weak source pairings; no pointwise exact interacting transverse current",
    ),
    Claim(
        "ward_continuity",
        r"delta_lambda S=0 and d_t int f rho=<j,grad f>",
        "src/aion/electronic_structure/wilson_sources.py",
        "tests/test_wilson_sources.py; tests/test_wilson_gga_integration.py",
        "NQ5,NQ6,NQ8",
        "Ward 6.51e-19; finite-region continuity 2.62e-12",
        "tested weak weights and declared instantaneous or propagated states",
    ),
    Claim(
        "connection_eom",
        r"S Cdot=-(omega+i K/hbar) C; Sdot=omega+omega^dagger",
        "src/aion/electronic_structure/time_connection.py; "
        "src/aion/electronic_structure/wilson_dynamics.py",
        "tests/test_exact_one_electron_wp6.py; tests/test_wilson_dynamics.py",
        "NQ6",
        "connection identity and trajectory residuals at accepted numerical floors",
        "prescribed uniform electric and uniform magnetic-plus-induction sources",
    ),
    Claim(
        "nonlinear_propagation",
        r"P_{n+1}=U P_n U^dagger by self-consistent two-node Gauss--Magnus",
        "src/aion/propagation/tensorial.py",
        "tests/test_tensorial_magnus.py; tests/test_wilson_dynamics.py",
        "NQ6,NQ7,NQ8",
        "global density order >=3.7909; no metric correction or projection",
        "accepted time windows, fields, nonlinear tolerances, and finite bases",
    ),
    Claim(
        "mechanical_power",
        r"U(T)-U(0)=integral <j_action,E> dt",
        "src/aion/electronic_structure/wilson_dynamics.py",
        "tests/test_wilson_dynamics.py; tests/test_wilson_gga_integration.py",
        "NQ6,NQ8",
        "LDA endpoint work defect 4.87e-11 Ha; PBE transfer 4.06e-10 Ha",
        "mechanical energy and current from the same auxiliary adiabatic action",
    ),
    Claim(
        "reduced_actions",
        r"P0, E1, strict C1, and separately named density-resummed C1 actions",
        "src/aion/electronic_structure/reduced_wilson.py",
        "tests/test_reduced_wilson.py; tests/test_reduced_wilson_gpu.py",
        "NQ7",
        "P0 field order 1.0000; strict-C1 field order 1.99996",
        "accepted LDA H3+ comparison domain only",
    ),
    Claim(
        "reduced_domain",
        r"positive reduced metric and declared LDA density domain",
        "src/aion/electronic_structure/reduced_wilson.py",
        "tests/test_reduced_wilson.py",
        "NQ7",
        "48 visible reduced-metric failures; zero hidden corrections",
        "sampled STO-3G/cc-pVDZ/aug-cc-pVDZ stress grid",
    ),
)


ADDITIONAL_NEGATIVE_RESULTS = (
    (
        "NQ2",
        "RI auxiliary approximation",
        "The selected weigend RI Hartree energy differs from the independent "
        "exact four-centre reference by as much as 6.90e-5 relative.",
        "Retain auxiliary-basis error as a separate floor; do not call the RI "
        "action exact Coulomb.",
    ),
    (
        "NQ4",
        "superseded executions and fit",
        "One campaign was incomplete, one complete campaign had wrong "
        "traceability labels, and a four-point derivative-order fit included a "
        "floor-dominated point.",
        "All are preserved; accepted analysis_final_v2 uses the correctly "
        "labelled run and coarse-step fit.",
    ),
    (
        "NQ5",
        "wrong GPU launcher",
        "The first full GPU invocation failed two explicit launch-contract "
        "assertions because CUDA support was absent from that launcher.",
        "The failure is procedural evidence; the corrected tools/gpu-python run passed.",
    ),
    (
        "NQ6",
        "power failure and provisional thresholds",
        "A host power failure interrupted 33/56 trajectories; a preliminary "
        "analysis also used continuity and power limits below measured "
        "quadrature floors.",
        "The campaign was rerun from the beginning and the preliminary failed "
        "analysis remains immutable.",
    ),
    (
        "NQ7",
        "basis-conditioned reduced metric",
        "All tested aug-cc-pVDZ reduced levels fail metric positivity at the "
        "first nonzero sampled field Bz=0.03 au, while the exact Wilson metric "
        "remains positive.",
        "No clipping, regularization, projection, or molecule-independent field "
        "cutoff is accepted.",
    ),
    (
        "NQ7",
        "density resummation",
        "Density-resummed C1 showed no systematic accuracy, stability, or domain "
        "advantage over strict C1.",
        "Retain it as a separately labelled diagnostic action; use strict C1 in "
        "the bounded near-term domain.",
    ),
    (
        "NQ8",
        "CO source finite-difference floor",
        "Fine source steps amplify deterministic subtraction error in separately "
        "evaluated RI-Hartree energies; the best complete residual is 7.39e-9 au.",
        "Use the resolved h=0.003 point and preserve the full refinement sequence.",
    ),
    (
        "NQ8",
        "archival and preliminary-analysis defects",
        "The external shell appended run.log after its internal hash, and the "
        "first analysis tested component composition at the roundoff-dominated "
        "smallest step.",
        "Exclude the non-scientific log from authentication and retain both "
        "analysis versions visibly.",
    ),
)


FIGURES = (
    ("NQ2", "analysis_authoritative/grid_and_auxiliary_convergence.png"),
    ("NQ4", "analysis_final_v2/h3plus_stationary_field_scan.png"),
    ("NQ6", "analysis_final/timestep_convergence.png"),
    ("NQ6", "analysis_final/induction_energy_work.png"),
    ("NQ7", "analysis_final/stationary_field_errors.png"),
    ("NQ7", "analysis_final/domain_metric_boundaries.png"),
    ("NQ7", "analysis_final/dynamic_model_errors.png"),
    ("NQ8", "analysis_final_v2/co_source_refinement.png"),
    ("NQ8", "analysis_final_v2/transfer_diagnostics.png"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _match_hash(paths: list[Path], expected: str, label: str) -> Path:
    for path in paths:
        if path.is_file() and _sha256(path) == expected:
            return path
    raise RuntimeError(f"cannot authenticate {label}: no candidate has {expected}")


def _evidence_candidates(review: dict[str, Any], key: str) -> list[Path]:
    evidence = review["evidence"]
    root = Path(evidence["execution_root"])
    if key == "field_free_result_sha256":
        return [root / "field_free_result.json"]
    if key == "reference_hdf5_sha256":
        return list(root.glob("*.reference.h5"))
    if key == "reference_toml_sha256":
        return [root / "reference.toml"]
    if key == "result_sha256":
        return [root / "result.json"]
    if key == "provenance_sha256":
        return [root / "provenance.json"]
    if key == "completed_sha256":
        return [root / "completed.json"]
    if key == "analysis_summary_sha256":
        accepted = evidence.get("accepted_analysis")
        if accepted is not None:
            return [root / accepted]
        return sorted(root.glob("analysis*/summary.json"))
    if key == "analysis_completed_sha256":
        accepted = Path(evidence["accepted_analysis"])
        return [root / accepted.parent / "completed.json"]
    if key.startswith("refinement_"):
        refinement = Path(evidence["source_refinement_root"])
        name = key.removeprefix("refinement_").removesuffix("_sha256")
        return [refinement / f"{name}.json"]
    if key == "interrupted_record_sha256":
        interrupted = root.parent / "nq6_dynamics_20260922T145942Z_a197fbc55b62"
        return [interrupted / "interrupted.json"]
    if key == "interrupted_log_sha256":
        return [root.parent / "nq6_dynamics_20260922T145942Z_a197fbc55b62.log"]
    raise KeyError(f"unmapped evidence hash: {review['gate']}.{key}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    repo = arguments.repo.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir()
    figures.mkdir()

    review_paths = sorted((repo / "docs" / "reviews").glob("chapter13_nq?_review_*.json"))
    if len(review_paths) != 9:
        raise RuntimeError(f"expected nine NQ0--NQ8 reviews, found {len(review_paths)}")

    accepted_rows: list[dict[str, Any]] = []
    limit_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    authentication_rows: list[dict[str, Any]] = []
    reviews: dict[str, dict[str, Any]] = {}
    for index, review_path in enumerate(review_paths):
        review = _load(review_path)
        gate = f"NQ{index}"
        if review.get("gate") != gate or review.get("decision") != "accepted":
            raise RuntimeError(f"review sequence/decision mismatch: {review_path}")
        reviews[gate] = review
        entrypoint = repo / review["evidence"]["review_entry_point"]
        if not entrypoint.is_file():
            raise FileNotFoundError(entrypoint)
        accepted_rows.append(
            {
                "gate": gate,
                "reviewed_at_utc": review["reviewed_at_utc"],
                "implemented_contract": review["scope"]["implemented_contract"],
                "review_record": str(review_path.relative_to(repo)),
                "review_sha256": _sha256(review_path),
                "entrypoint": str(entrypoint.relative_to(repo)),
                "entrypoint_sha256": _sha256(entrypoint),
                "execution_root": review["evidence"]["execution_root"],
            }
        )
        for name, value in review.get("accepted_measurements", {}).items():
            limit_rows.append({"gate": gate, "measurement": name, "value": value})
        for position, limitation in enumerate(review.get("limitations", []), start=1):
            exclusion_rows.append({"gate": gate, "index": position, "limitation": limitation})
        for key, expected in review["evidence"].items():
            if not key.endswith("_sha256"):
                continue
            matched = _match_hash(_evidence_candidates(review, key), str(expected), f"{gate}.{key}")
            authentication_rows.append(
                {
                    "gate": gate,
                    "field": key,
                    "path": str(matched),
                    "sha256": expected,
                    "authenticated": True,
                }
            )

    claim_rows = [claim.__dict__ for claim in CLAIMS]
    negative_rows = [
        {"gate": gate, "topic": topic, "result": result, "disposition": disposition}
        for gate, topic, result, disposition in ADDITIONAL_NEGATIVE_RESULTS
    ]

    figure_rows: list[dict[str, Any]] = []
    for gate, relative in FIGURES:
        source = Path(reviews[gate]["evidence"]["execution_root"]) / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = figures / f"{gate.lower()}_{source.name}"
        shutil.copyfile(source, destination)
        figure_rows.append(
            {
                "gate": gate,
                "source": str(source),
                "source_sha256": _sha256(source),
                "copy": str(destination.relative_to(output)),
                "copy_sha256": _sha256(destination),
            }
        )

    _write_csv(tables / "accepted_gates.csv", accepted_rows)
    _write_csv(tables / "artifact_authentication.csv", authentication_rows)
    _write_csv(tables / "numerical_limits.csv", limit_rows)
    _write_csv(tables / "equation_code_evidence.csv", claim_rows)
    _write_csv(tables / "limitations.csv", exclusion_rows)
    _write_csv(tables / "negative_results.csv", negative_rows)
    _write_csv(tables / "figure_manifest.csv", figure_rows)

    status = _git(repo, "status", "--porcelain")
    if status:
        raise RuntimeError("repository must be clean before authenticated synthesis")
    result = {
        "schema": "aion.chapter13.nq9-synthesis.v1",
        "status": "executed_unreviewed",
        "accepted_gates": [row["gate"] for row in accepted_rows],
        "authenticated_primary_artifact_count": len(authentication_rows),
        "accepted_measurement_count": len(limit_rows),
        "equation_code_evidence_count": len(claim_rows),
        "limitation_count": len(exclusion_rows),
        "explicit_negative_result_count": len(negative_rows),
        "selected_figure_count": len(figure_rows),
        "supported_domain": {
            "systems": ["H2", "LiH", "equilateral H3+", "CO"],
            "orbital_bases": ["STO-3G", "cc-pVDZ", "aug-cc-pVDZ stress scan"],
            "auxiliary_basis": "weigend in the accepted production fixtures",
            "functionals": ["Hartree only", "restricted pure lda,vwn", "restricted pure PBE"],
            "electromagnetism": (
                "fixed-centre prescribed uniform electric and uniform magnetic "
                "fields, including Maxwell-consistent uniform magnetic induction"
            ),
            "exact_model": "straight-Wilson finite molecular Gaussian subspace",
            "reduced_recommendation": (
                "strict C1 only inside the sampled LDA H3+ domain; exact Wilson "
                "remains the reference"
            ),
            "time_domain": (
                "up to 2.0 au for accepted LDA model comparisons and 0.2 au for PBE transfer"
            ),
        },
        "remaining_exclusions": [
            "moving nuclei and nonadiabatic nuclear connection",
            "spatially nonuniform propagated electromagnetic fields",
            "self-consistent Maxwell backreaction",
            "periodic boundary conditions",
            "pseudopotentials and their gauge/current corrections",
            "spin polarization, hybrids, meta-GGAs, and nonlocal correlation",
            "an exact interacting transverse current beyond the declared "
            "auxiliary adiabatic Kohn--Sham action",
            "long-time stability, spectroscopy, basis convergence, and broader "
            "chemical production claims",
            "GGA reduced P0/E1/C1 actions",
        ],
        "chapter14_handoff": {
            "title": (
                "Numerical realization and qualification of fixed-centre "
                "Wilson-projected adiabatic dynamics"
            ),
            "sections": [
                "Scope, variables, index conventions, and realization tuple",
                "Exact Wilson density and variational Hartree/LDA/GGA actions",
                "Stationary nonlinear states and gauge/frame covariance",
                "Action-derived weak sources, Ward identity, and continuity",
                "Connection-aware nonlinear Gauss--Magnus propagation",
                "Mechanical energy, source power, and work balance",
                "P0, E1, strict-C1, and resummed-C1 validity and failures",
                "Molecular and functional transfer from H2/LiH/H3+ to CO/PBE",
                "Numerical error budget, negative results, and reproducibility",
                "Bounded conclusions and open extensions",
            ],
        },
    }
    _write_json(output / "result.json", result)
    provenance = {
        "schema": "aion.chapter13.nq9-provenance.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repository": str(repo),
        "branch": _git(repo, "branch", "--show-current"),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "review_sha256": {row["gate"]: row["review_sha256"] for row in accepted_rows},
        "table_sha256": {path.name: _sha256(path) for path in sorted(tables.iterdir())},
        "figure_sha256": {path.name: _sha256(path) for path in sorted(figures.iterdir())},
    }
    _write_json(output / "provenance.json", provenance)
    completed = {
        "schema": "aion.chapter13.nq9-completed.v1",
        "status": "executed_unreviewed",
        "result_sha256": _sha256(output / "result.json"),
        "provenance_sha256": _sha256(output / "provenance.json"),
        "artifact_sha256": {
            str(path.relative_to(output)): _sha256(path)
            for path in sorted((*tables.iterdir(), *figures.iterdir()))
        },
    }
    _write_json(output / "completed.json", completed)
    print(json.dumps({"output": str(output), **completed}, indent=2))


if __name__ == "__main__":
    main()
