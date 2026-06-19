#!/usr/bin/env python3
"""Pure length-gauge Crank-Nicolson RT-TDDFT run for benzene."""

from __future__ import annotations

import sys

import numpy as np

from common import DEFAULT_EXCITATIONS_PATH, build_ground_state

from aion import (  # noqa: E402
    GaussianPulse,
    LengthGaugeCNRTTDDFT,
    excitation_by_index,
    load_excitations,
    lowest_active_excitation,
    transition_polarization,
)


def load_driven_excitation():
    if not DEFAULT_EXCITATIONS_PATH.exists():
        print(
            "Missing saved Casida data. Run "
            "python examples/benzene_length_gauge/run_casida.py first.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    metadata, excitations = load_excitations(DEFAULT_EXCITATIONS_PATH)
    selected_root = metadata.get("selected_root")
    if selected_root is None:
        driven = lowest_active_excitation(excitations)
    else:
        driven = excitation_by_index(excitations, int(selected_root))
    return metadata, driven


def main() -> None:
    metadata, driven = load_driven_excitation()
    polarization = transition_polarization(driven)

    print(f"# loaded {DEFAULT_EXCITATIONS_PATH}")
    print(
        "# driving root",
        driven.index,
        f"omega={driven.energy:.10f} Ha",
        f"({driven.energy_ev:.6f} eV)",
        f"osc={driven.oscillator_strength:.6e}",
        "polarization=",
        *(f"{x:.10e}" for x in polarization),
    )
    if metadata:
        print(
            "# reference",
            f"xc={metadata.get('xc', 'unknown')}",
            f"basis={metadata.get('basis', 'unknown')}",
            f"nstates={metadata.get('nstates', 'unknown')}",
        )

    mf = build_ground_state(verbose=3)

    field = GaussianPulse(
        amplitude=1.0e-4,
        omega=driven.energy,
        center=25.0,
        sigma=8.0,
        polarization=polarization,
    )

    rt = LengthGaugeCNRTTDDFT.from_ground_state(mf, field, origin=np.zeros(3))
    coeff0 = rt.initial_coefficients()

    dt = 0.02
    nsteps = 50

    print("# step time N orth idem E0 Ecoup Etot mux muy muz Ex Ey Ez")
    for coeff, rec in rt.propagate(
        coeff0,
        dt=dt,
        nsteps=nsteps,
        record_energy=True,
        energy_stride=5,
    ):
        if rec.step % 5 == 0 or rec.step == nsteps:
            e0 = "nan" if rec.field_free_energy is None else f"{rec.field_free_energy:.12f}"
            etot = "nan" if rec.total_energy is None else f"{rec.total_energy:.12f}"
            print(
                rec.step,
                f"{rec.time:.8f}",
                f"{rec.electron_number:.12f}",
                f"{rec.orthonormality_error:.6e}",
                f"{rec.idempotency_error:.6e}",
                e0,
                f"{rec.field_coupling_energy:.12e}",
                etot,
                *(f"{x:.12e}" for x in rec.dipole),
                *(f"{x:.12e}" for x in rec.field),
            )


if __name__ == "__main__":
    main()
