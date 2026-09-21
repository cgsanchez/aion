#!/usr/bin/env python3
"""Reproduce the Chapter 13 NQ0 field-free H2/LDA reference."""

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
from aion.electronic_structure import AdiabaticPureRKS, DependencyVersions
from aion.electronic_structure.pyscf_rks import prepare_pyscf_reference


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


def _scalar(value: object) -> float:
    return float(np.asarray(value).real)


def _name(value: object) -> str:
    return str(getattr(value, "__name__", type(value).__name__))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact directory: {output}")
    output.mkdir(parents=True)

    repo = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    reference_path = output / "h2_lda_vwn_weigend.reference.h5"
    config = ReferenceConfig(
        molecule=MoleculeConfig(
            atoms=(
                AtomConfig("H", (0.0, 0.0, -0.7)),
                AtomConfig("H", (0.0, 0.0, 0.7)),
            ),
            charge=0,
            spin=0,
        ),
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
        output=ReferenceOutputConfig(reference_path),
        metadata=MetadataConfig(
            label="chapter13-nq0-field-free-h2-lda-vwn",
            timestamp_utc=timestamp,
            host=platform.node(),
        ),
    )
    config_path = output / "reference.toml"
    config_path.write_text(dumps_config(config), encoding="utf-8")

    reference = prepare_pyscf_reference(config)
    reference.save()
    workspace = reference.create_workspace(BackendConfig())
    density = workspace.backend.asarray(reference.ground_state.density, dtype=np.complex128)
    bridge = AdiabaticPureRKS(workspace)
    build = bridge.build(density)
    internal = bridge.energy(build)
    mean_field = workspace.electronic_model
    assert mean_field is not None
    density_real = np.asarray(reference.ground_state.density)
    pyscf_effective = np.asarray(mean_field.get_veff(mean_field.mol, density_real))
    pyscf_hamiltonian = np.asarray(mean_field.get_hcore(mean_field.mol)) + pyscf_effective
    aion_hamiltonian = workspace.backend.to_host(build.hamiltonian)
    rebuilt_density = np.asarray(mean_field.make_rdm1())

    overlap = reference.core_operators.overlap
    electron_count = float(np.einsum("ij,ji->", density_real, overlap, optimize=True).real)
    energy_components = {
        "kinetic_canonical_au": _scalar(internal.energy_kinetic_canonical),
        "electron_nuclear_au": _scalar(internal.energy_electron_nuclear),
        "hartree_au": _scalar(internal.energy_hartree),
        "exchange_correlation_au": _scalar(internal.energy_exchange_correlation),
        "nuclear_repulsion_au": _scalar(internal.energy_nuclear_repulsion),
        "internal_total_au": _scalar(internal.energy_internal_total),
    }
    component_sum = sum(
        energy_components[name]
        for name in (
            "kinetic_canonical_au",
            "electron_nuclear_au",
            "hartree_au",
            "exchange_correlation_au",
            "nuclear_repulsion_au",
        )
    )
    residuals = {
        "electron_count_abs": abs(electron_count - reference.ground_state.electron_count),
        "energy_total_abs_au": abs(
            energy_components["internal_total_au"] - reference.ground_state.energy_total_au
        ),
        "energy_component_sum_abs_au": abs(
            component_sum - energy_components["internal_total_au"]
        ),
        "hamiltonian_relative_frobenius": float(
            np.linalg.norm(aion_hamiltonian - pyscf_hamiltonian)
            / max(1.0, np.linalg.norm(pyscf_hamiltonian))
        ),
        "rebuilt_density_relative_frobenius": float(
            np.linalg.norm(rebuilt_density - density_real)
            / max(1.0, np.linalg.norm(density_real))
        ),
    }
    tolerances = {
        "electron_count_abs": 1.0e-11,
        "energy_total_abs_au": 5.0e-11,
        "energy_component_sum_abs_au": 5.0e-14,
        "hamiltonian_relative_frobenius": 5.0e-13,
        "rebuilt_density_relative_frobenius": 5.0e-13,
    }
    passed = all(residuals[name] <= tolerance for name, tolerance in tolerances.items())

    from pyscf.df import incore
    from pyscf.dft import libxc

    grid = mean_field.grids
    df_model = mean_field.with_df
    result: dict[str, Any] = {
        "schema": "aion.chapter13-nq0-field-free",
        "schema_version": "1.0.0",
        "passed": passed,
        "scientific_id": config.scientific_id,
        "reference_fingerprint_sha256": reference.fingerprint_sha256,
        "core_operator_fingerprint_sha256": reference.core_operators.fingerprint_sha256,
        "grid_fingerprint_sha256": reference.grid.fingerprint_sha256,
        "ground_state": {
            "energy_total_au": reference.ground_state.energy_total_au,
            "electron_count": reference.ground_state.electron_count,
            "overlap_contracted_electron_count": electron_count,
            "occupations": reference.ground_state.occupations.tolist(),
            "orbital_energies_au": reference.ground_state.orbital_energies_au.tolist(),
        },
        "energy_components": energy_components,
        "residuals": residuals,
        "tolerances": tolerances,
        "scalar_potential_contract": {
            "field_free_value": 0.0,
            "included_in_mechanical_energy": False,
            "owner_at_finite_field": "temporal connection",
        },
        "pyscf_realization": {
            "functional": config.electronic_structure.functional,
            "xc_family": config.electronic_structure.xc_family.value,
            "libxc_version": str(libxc.libxc_version()),
            "grid_level": reference.grid.level,
            "grid_points": int(reference.grid.coordinates_au.shape[0]),
            "grid_pruning_recorded": reference.grid.pruning,
            "grid_prune_callable": _name(grid.prune),
            "radial_grid_callable": _name(grid.radi_method),
            "partition_callable": _name(grid.becke_scheme),
            "density_fitting": True,
            "auxiliary_basis": str(df_model.auxbasis),
            "auxiliary_functions": int(df_model.get_naoaux()),
            "pyscf_df_linear_dependence_threshold": float(incore.LINEAR_DEP_THR),
            "nuclear_model": config.electronic_structure.nuclear_model.value,
        },
    }
    result_path = output / "field_free_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    tracked_diff = _git(repo, "diff", "--binary", "HEAD", binary=True)
    assert isinstance(tracked_diff, bytes)
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    assert isinstance(status, str)
    head = _git(repo, "rev-parse", "HEAD")
    assert isinstance(head, str)
    source_paths = (
        repo / "tools/run_chapter13_nq0_field_free.py",
        repo / "src/aion/electronic_structure/adiabatic.py",
        repo / "src/aion/electronic_structure/pyscf_rks.py",
        repo / "environment.yml",
        repo / "conda-linux-64.lock",
    )
    environment = {
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
    }
    provenance = {
        "timestamp_utc": timestamp,
        "command": " ".join(shlex.quote(value) for value in (sys.executable, *sys.argv)),
        "repository": str(repo),
        "git_head": head.strip(),
        "git_status_porcelain": status.splitlines(),
        "git_tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "environment": environment,
        "source_sha256": {str(path): _sha256(path) for path in source_paths},
        "artifacts_sha256": {
            config_path.name: _sha256(config_path),
            reference_path.name: _sha256(reference_path),
            result_path.name: _sha256(result_path),
        },
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "passed": passed, "residuals": residuals}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
