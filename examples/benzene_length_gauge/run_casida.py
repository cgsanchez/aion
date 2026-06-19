#!/usr/bin/env python3
"""Run and save the expensive benzene LR-TDDFT/Casida calculation."""

from __future__ import annotations

from common import DEFAULT_EXCITATIONS_PATH, build_ground_state

from aion import (  # noqa: E402
    casida_excitations,
    lowest_active_excitation,
    save_excitations,
    transition_polarization,
)


def main() -> None:
    mf = build_ground_state(verbose=3)

    _, excitations = casida_excitations(mf, nstates=5, conv_tol=1.0e-8)
    driven = lowest_active_excitation(excitations)
    polarization = transition_polarization(driven)

    metadata = {
        "molecule": "benzene",
        "basis": "cc-pvdz",
        "xc": "pbe",
        "charge": 0,
        "spin": 0,
        "nstates": 5,
        "singlet": True,
        "selected_root": driven.index,
        "selected_energy": driven.energy,
        "selected_energy_ev": driven.energy_ev,
        "selected_oscillator_strength": driven.oscillator_strength,
        "selected_polarization": polarization.tolist(),
    }
    save_excitations(DEFAULT_EXCITATIONS_PATH, excitations, metadata=metadata)

    print(f"# wrote {DEFAULT_EXCITATIONS_PATH}")
    print("# root omega_Ha omega_eV osc_strength mux muy muz")
    for exc in excitations:
        print(
            exc.index,
            f"{exc.energy:.10f}",
            f"{exc.energy_ev:.6f}",
            f"{exc.oscillator_strength:.6e}",
            *(f"{x:.10e}" for x in exc.transition_dipole),
        )
    print(
        "# selected",
        driven.index,
        f"{driven.energy:.10f}",
        f"{driven.energy_ev:.6f}",
        *(f"{x:.10e}" for x in polarization),
    )


if __name__ == "__main__":
    main()
