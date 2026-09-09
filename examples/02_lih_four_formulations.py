"""Run one short LiH pulse through the four comparison formulations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from aion import build_simulation, prepare_reference, run
from aion.config import (
    AtomConfig,
    BackendConfig,
    BackendKind,
    ElectronicStructureConfig,
    FormulationConfig,
    FormulationKind,
    GaugeRepresentation,
    IntegratorKind,
    MoleculeConfig,
    ObservableSchedules,
    OutputConfig,
    PropagationConfig,
    ReferenceConfig,
    ReferenceLinkConfig,
    ReferenceOutputConfig,
    SimulationConfig,
    Sin2VectorPotentialPulseConfig,
    StepSchedule,
    XCFamily,
)
from aion.electromagnetism import pulse_aligned_time_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("example-output/lih-pulse"))
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    backend = BackendConfig(
        BackendKind(args.backend),
        device_index=0 if args.backend == "gpu" else None,
    )
    reference_path = output / "reference.h5"
    reference = prepare_reference(
        ReferenceConfig(
            molecule=MoleculeConfig(
                atoms=(
                    AtomConfig("Li", (0.0, 0.0, -1.5)),
                    AtomConfig("H", (0.0, 0.0, 1.5)),
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
    )
    reference.save()
    pulse = Sin2VectorPotentialPulseConfig(
        peak_electric_field_au=0.01,
        angular_frequency_au=2.0 * np.pi,
        cycles=1,
        polarization=(0.2, -0.3, 1.0),
        carrier_phase_rad=0.37,
        require_zero_impulse=True,
    )
    grid = pulse_aligned_time_grid(pulse, 0.05)
    schedules = ObservableSchedules(
        dipole_current=StepSchedule(every=1),
        energy=StepSchedule(every=5),
        diagnostics=StepSchedule(every=1),
        source=StepSchedule(every=1),
    )
    modes = (
        ("bare-lg", FormulationKind.BARE_LENGTH_GAUGE, GaugeRepresentation.LENGTH),
        ("bare-vg", FormulationKind.BARE_VELOCITY_GAUGE, GaugeRepresentation.VELOCITY),
        ("p0e1-lg", FormulationKind.P0_E1, GaugeRepresentation.LENGTH),
        ("p0e1-vg", FormulationKind.P0_E1, GaugeRepresentation.VELOCITY),
    )
    final_dipoles: dict[str, np.ndarray] = {}
    for label, kind, gauge in modes:
        integrator = (
            IntegratorKind.FIXED_METRIC_SCEM
            if kind in {FormulationKind.BARE_LENGTH_GAUGE, FormulationKind.BARE_VELOCITY_GAUGE}
            else IntegratorKind.CONNECTION_AWARE_SCEM
        )
        simulation = build_simulation(
            SimulationConfig(
                reference=ReferenceLinkConfig(reference.fingerprint_sha256, reference_path),
                formulation=FormulationConfig(kind, gauge),
                source=pulse,
                propagation=PropagationConfig(
                    grid,
                    integrator,
                    density_tolerance=1.0e-11,
                ),
                backend=backend,
                output=OutputConfig(output / label, schedules),
            ),
            reference,
        )
        trajectory = run(simulation)
        definition_id = simulation.calculators.definitions["electronic_dipole"].definition_id
        final_dipoles[label] = trajectory.read_observable(definition_id).values[-1]
        print(f"{label}: {trajectory.path}")
    covariant_difference = np.linalg.norm(final_dipoles["p0e1-lg"] - final_dipoles["p0e1-vg"])
    print(f"P0+E1 final LG/VG dipole difference: {covariant_difference:.3e} a.u.")


if __name__ == "__main__":
    main()
