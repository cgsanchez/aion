#!/usr/bin/env python3
"""Probe dynamic-source power and continuity under molecular-grid refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from run_chapter13_nq6_dynamics import (
    _AUXILIARY_BASIS,
    _FUNCTIONAL,
    _NQ4_ROOT,
    _DynamicCache,
    _float,
    _initial_key,
)

from aion.config import BackendConfig
from aion.electromagnetism import GaussianScalarGaugeVariation
from aion.electronic_structure import (
    AOGridPolicy,
    WilsonStationaryBranch,
    evaluate_exact_wilson_power,
    evaluate_nonlinear_density_pure_gauge_ward,
    evaluate_nonlinear_weak_continuity,
    load_reference_data,
    prepare_ao_quadrature,
    prepare_exact_wilson_stationary_factory,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+", default=(2, 3, 4))
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    reference = load_reference_data(_NQ4_ROOT / "h3plus.reference.h5")
    states = np.load(_NQ4_ROOT / "stationary_states.npz", allow_pickle=False)
    branch = WilsonStationaryBranch.KOHN_SHAM_LDA
    gauge_test = GaussianScalarGaugeVariation(
        amplitude=0.37,
        center_au=(0.11, -0.17, 0.23),
        exponent_au_inverse2=0.41,
    )
    rows = []
    for level in arguments.levels:
        quadrature = prepare_ao_quadrature(
            reference,
            BackendConfig(),
            grid_policy=AOGridPolicy.qualification(level),
            block_size=2048,
        )
        factory = prepare_exact_wilson_stationary_factory(
            quadrature,
            auxiliary_basis=_AUXILIARY_BASIS,
            functional=_FUNCTIONAL,
        )
        for case in ("electric_static_b", "magnetic_induction"):
            density = np.asarray(
                states[f"{_initial_key(case, branch)}_density"],
                dtype=np.complex128,
            )
            sample = _DynamicCache.create(factory, case).at(0.5, branch)
            evaluation = sample.evaluate(density)
            power = evaluate_exact_wilson_power(evaluation, density)
            continuity = evaluate_nonlinear_weak_continuity(
                sample.model,
                sample.one_electron,
                density,
                gauge_test,
            )
            ward = evaluate_nonlinear_density_pure_gauge_ward(
                sample.model,
                sample.one_electron,
                density,
                power.velocity_density,
                gauge_test,
            )
            rows.append(
                {
                    "grid_level": level,
                    "grid_points": quadrature.grid.npoints,
                    "case": case,
                    "time_au": 0.5,
                    "source_power_au": _float(power.source_power_au),
                    "matrix_rate_au": _float(power.matrix_mechanical_energy_rate_au),
                    "power_residual_au": abs(_float(power.power_identity_residual_au)),
                    "finite_region_continuity_residual": abs(
                        _float(continuity.finite_region_residual)
                    ),
                    "weighted_charge_derivative": _float(
                        continuity.weighted_charge_derivative
                    ),
                    "on_shell_current_pairing": _float(
                        continuity.weak_current_pairing.on_shell_pairing
                    ),
                    "fixed_coordinate_current_pairing": _float(
                        continuity.weak_current_pairing.total_pairing
                    ),
                    "tangential_pairing": _float(
                        continuity.weak_current_pairing.tangential_pairing
                    ),
                    "global_charge_residual": abs(
                        _float(continuity.global_charge_residual)
                    ),
                    "ward_residual": abs(_float(ward.total_ward_residual)),
                    "ward_source_pairing": _float(ward.source_pairing),
                    "ward_matter_pairing": _float(ward.matter_pairing),
                    "metric_compatibility_residual": (
                        sample.one_electron.connection.metric_compatibility_residual
                    ),
                }
            )
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema": "aion.chapter13-nq6-dynamic-grid-probe",
                "schema_version": "1.0.0",
                "status": "diagnostic_executed_unreviewed",
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
