"""Prepare H2 and publish a one-root Casida result through public APIs."""

from __future__ import annotations

import argparse
from pathlib import Path

from aion import prepare_reference
from aion.config import (
    AtomConfig,
    BackendConfig,
    BackendKind,
    ElectronicStructureConfig,
    MoleculeConfig,
    ReferenceConfig,
    ReferenceOutputConfig,
    XCFamily,
)
from aion.spectroscopy import (
    CasidaConfig,
    run_casida,
    save_casida_result,
    select_resonance,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("example-output/h2-casida"))
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_path = output / "reference.h5"
    backend = BackendConfig(
        BackendKind(args.backend),
        device_index=0 if args.backend == "gpu" else None,
    )
    config = ReferenceConfig(
        molecule=MoleculeConfig(
            atoms=(
                AtomConfig("H", (0.0, 0.0, -0.7)),
                AtomConfig("H", (0.0, 0.0, 0.7)),
            )
        ),
        electronic_structure=ElectronicStructureConfig(
            basis="sto-3g",
            functional="pbe",
            xc_family=XCFamily.GGA,
            grid_level=1,
            scf_energy_tolerance_au=1.0e-11,
        ),
        backend=backend,
        output=ReferenceOutputConfig(reference_path),
    )
    reference = prepare_reference(config)
    reference.save()
    result = run_casida(reference, CasidaConfig(nstates=1))
    selection = select_resonance(result, (0.0, 0.0, 1.0), root_index=0)
    result_path = output / "casida.h5"
    save_casida_result(
        result,
        result_path,
        selection=selection,
        reference_artifact=reference_path,
    )
    print(f"reference: {reference_path}")
    print(f"Casida result: {result_path}")
    print(f"lowest excitation: {result.excitation_energy_au[0]:.10f} Ha")


if __name__ == "__main__":
    main()
