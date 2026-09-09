"""Generate strict TOML, run through the CLI, and resume before a kick."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aion import load_reference
from aion.cli import main as aion_cli
from aion.config import (
    AtomConfig,
    BackendConfig,
    BackendKind,
    ElectronicStructureConfig,
    FixedTimeGrid,
    FormulationConfig,
    FormulationKind,
    IntegratorKind,
    KickEventConfig,
    MoleculeConfig,
    ObservableSchedules,
    OutputConfig,
    PropagationConfig,
    ReferenceConfig,
    ReferenceLinkConfig,
    ReferenceOutputConfig,
    SimulationConfig,
    StepSchedule,
    XCFamily,
    ZeroSourceConfig,
    dumps_config,
)
from aion.io.checkpoint import load_checkpoint


def invoke(arguments: list[str]) -> None:
    print("$ aion " + " ".join(arguments))
    return_code = aion_cli(arguments)
    if return_code != 0:
        raise RuntimeError(f"aion CLI exited with status {return_code}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("example-output/checkpoint-resume"))
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    backend = BackendConfig(
        BackendKind(args.backend),
        device_index=0 if args.backend == "gpu" else None,
    )
    reference_path = output / "reference.h5"
    reference_input = output / "reference.toml"
    reference_input.write_text(
        dumps_config(
            ReferenceConfig(
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
        ),
        encoding="utf-8",
    )
    invoke(["prepare", str(reference_input)])
    reference = load_reference(reference_path, backend=backend)
    full_directory = output / "full"
    simulation_input = output / "simulation.toml"
    simulation_input.write_text(
        dumps_config(
            SimulationConfig(
                reference=ReferenceLinkConfig(reference.fingerprint_sha256, reference_path),
                formulation=FormulationConfig(FormulationKind.BARE_LENGTH_GAUGE),
                source=ZeroSourceConfig(),
                propagation=PropagationConfig(
                    FixedTimeGrid(0.0, 0.05, 4),
                    IntegratorKind.FIXED_METRIC_SCEM,
                    density_tolerance=1.0e-11,
                ),
                backend=backend,
                events=(KickEventConfig("kick-z", 2, (0.0, 0.0, 1.0e-3)),),
                output=OutputConfig(
                    full_directory,
                    ObservableSchedules(
                        dipole_current=StepSchedule(every=1),
                        diagnostics=StepSchedule(every=1),
                        checkpoints=StepSchedule(every=2),
                    ),
                ),
            )
        ),
        encoding="utf-8",
    )
    invoke(["run", str(simulation_input)])
    checkpoint = full_directory / "checkpoint_00000002.h5"
    resumed_directory = output / "resumed"
    invoke(["resume", str(checkpoint), "--output", str(resumed_directory)])
    full = load_checkpoint(full_directory / "checkpoint_00000004.h5")
    resumed = load_checkpoint(resumed_directory / "checkpoint_00000004.h5")
    difference = np.linalg.norm(full.density - resumed.density)
    print(f"final uninterrupted/resumed density difference: {difference:.3e}")


if __name__ == "__main__":
    main()
