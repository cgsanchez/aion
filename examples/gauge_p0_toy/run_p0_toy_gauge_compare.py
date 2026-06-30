#!/usr/bin/env python3
"""Gauge comparison for the pure Peierls P0 toy model.

This is a small end-to-end regression target for the formal geometry:

- one atom-centered AO per site,
- analytic uniform electric field,
- constant length/velocity interpolation parameter lambda,
- variable-metric SCEM propagation,
- gauge comparison on P0 source observables.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-aion")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/aion-cache")

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aion import (
    AOAnchors,
    LinearOneBodyModel,
    PeierlsGeometry,
    SiteHubbardModel,
    UniformElectricGauge,
    VariableMetricSCEM,
    density_from_coefficients,
    p0_natom_from_rows,
    p0_row_series,
    p0_site_populations,
    record_p0_observables,
    summarize_p0_gauge_errors,
)


DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "results"
GAUGES = {
    "length": 0.0,
    "mixed": 0.5,
    "velocity": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--t-final", type=float, default=80.0)
    parser.add_argument("--field-amplitude", type=float, default=0.035)
    parser.add_argument("--omega", type=float, default=0.35)
    parser.add_argument("--model", choices=["linear", "hubbard"], default="linear")
    parser.add_argument("--hubbard-u", type=float, default=0.2)
    parser.add_argument("--mixing", type=float, default=0.7)
    parser.add_argument("--midpoint-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--density-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def _pair_distances(coords: np.ndarray) -> np.ndarray:
    disp = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(disp, axis=2)


def toy_problem() -> tuple[AOAnchors, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.35, 0.0, 0.0],
            [0.55, 1.05, 0.0],
            [1.85, 0.85, 0.25],
        ],
        dtype=float,
    )
    anchors = AOAnchors(atom_coords=coords, ao_to_atom=np.arange(coords.shape[0]))
    distance = _pair_distances(coords)

    s0 = np.exp(-0.72 * distance**2)
    np.fill_diagonal(s0, 1.0)

    onsite = np.diag([-0.34, -0.08, 0.03, 0.17])
    hopping = -0.42 * np.exp(-0.85 * distance)
    np.fill_diagonal(hopping, 0.0)
    h0 = onsite + hopping

    occupations = np.array([2.0, 2.0])
    return anchors, s0, h0, occupations


def ground_state_coefficients(
    h0: np.ndarray,
    s0: np.ndarray,
    nocc: int,
) -> np.ndarray:
    _, coeff = scipy.linalg.eigh(h0, s0, check_finite=False)
    return coeff[:, :nocc]


def electric_source(
    *,
    lambda_value: float,
    amplitude: float,
    omega: float,
) -> UniformElectricGauge:
    direction = np.array([1.0, 0.35, 0.0], dtype=float)
    direction = direction / np.linalg.norm(direction)

    def field(t: float) -> np.ndarray:
        return amplitude * np.sin(omega * t) * direction

    def field_integral(t: float) -> np.ndarray:
        return amplitude * (1.0 - np.cos(omega * t)) / omega * direction

    return UniformElectricGauge(
        field=field,
        field_integral=field_integral,
        lambda_value=lambda _t: lambda_value,
        lambda_derivative=lambda _t: 0.0,
    )


def run_one_gauge(
    *,
    label: str,
    lambda_value: float,
    anchors: AOAnchors,
    s0: np.ndarray,
    h0: np.ndarray,
    occupations: np.ndarray,
    coeff0: np.ndarray,
    model_name: str,
    reference_populations: np.ndarray,
    hubbard_u: float,
    mixing: float,
    midpoint_tolerance: float,
    density_tolerance: float,
    max_iterations: int,
    dt: float,
    nsteps: int,
    field_amplitude: float,
    omega: float,
) -> list[dict[str, float | str]]:
    geometry = PeierlsGeometry(
        anchors,
        s0,
        electric=electric_source(
            lambda_value=lambda_value,
            amplitude=field_amplitude,
            omega=omega,
        ),
    )
    if model_name == "linear":
        model = LinearOneBodyModel(h0)
    else:
        model = SiteHubbardModel(
            h0,
            hubbard_u=hubbard_u,
            reference_populations=reference_populations,
        )
    rt = VariableMetricSCEM(geometry, model, occupations)

    coeff = coeff0.copy()
    rows = [
        {
            "gauge": label,
            **record_p0_observables(
                step=0,
                time=0.0,
                coeff=coeff,
                geometry=geometry,
                model=model,
                occupations=occupations,
                midpoint_iterations=0,
                hamiltonian_residual=0.0,
            ),
        }
    ]

    for step in range(1, nsteps + 1):
        result = rt.step(
            coeff,
            time=(step - 1) * dt,
            dt=dt,
            midpoint_tolerance=midpoint_tolerance,
            density_tolerance=None if model_name == "linear" else density_tolerance,
            max_iterations=max_iterations,
            mixing=1.0 if model_name == "linear" else mixing,
        )
        coeff = result.coeff_next
        rows.append(
            {
                "gauge": label,
                **record_p0_observables(
                    step=step,
                    time=step * dt,
                    coeff=coeff,
                    geometry=geometry,
                    model=model,
                    occupations=occupations,
                    midpoint_iterations=result.iterations,
                    hamiltonian_residual=result.hamiltonian_residual,
                ),
            }
        )
    return rows


def write_rows(path: Path, rows_by_gauge: dict[str, list[dict[str, float | str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = [row for rows in rows_by_gauge.values() for row in rows]
    fieldnames = list(all_rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def _series(rows: list[dict[str, float | str]], key: str) -> np.ndarray:
    return p0_row_series(rows, key)


def _natom_from_rows(rows: list[dict[str, float | str]]) -> int:
    return p0_natom_from_rows(rows)


def plot_summary(
    path: Path,
    rows_by_gauge: dict[str, list[dict[str, float | str]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = _series(rows_by_gauge["length"], "time_au")
    direction = np.array([1.0, 0.35, 0.0], dtype=float)
    direction = direction / np.linalg.norm(direction)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    ax_dip, ax_energy, ax_pop, ax_error = axes.ravel()

    ref_dip = np.column_stack(
        [_series(rows_by_gauge["length"], f"dipole_{x}") for x in "xyz"]
    )
    ref_energy = _series(rows_by_gauge["length"], "energy")
    natom = _natom_from_rows(rows_by_gauge["length"])

    for label, rows in rows_by_gauge.items():
        dipole = np.column_stack([_series(rows, f"dipole_{x}") for x in "xyz"])
        energy = _series(rows, "energy")
        ax_dip.plot(times, dipole @ direction, label=label)
        ax_energy.plot(times, energy, label=label)
        if label != "length":
            dip_err = np.linalg.norm(dipole - ref_dip, axis=1)
            ene_err = np.abs(energy - ref_energy)
            ax_error.semilogy(times, np.maximum(dip_err, 1.0e-18), label=f"{label} dipole")
            ax_error.semilogy(times, np.maximum(ene_err, 1.0e-18), "--", label=f"{label} energy")

    for atom_index in range(natom):
        ax_pop.plot(
            times,
            _series(rows_by_gauge["length"], f"population_{atom_index}"),
            label=f"site {atom_index}",
        )

    ax_dip.set_title("P0 dipole along field")
    ax_dip.set_xlabel("time / au")
    ax_dip.set_ylabel("electronic dipole / au")
    ax_dip.legend()

    ax_energy.set_title("Material energy")
    ax_energy.set_xlabel("time / au")
    ax_energy.set_ylabel("energy / Ha")
    ax_energy.legend()

    ax_pop.set_title("Length-gauge source populations")
    ax_pop.set_xlabel("time / au")
    ax_pop.set_ylabel("population")
    ax_pop.legend(ncol=2)

    ax_error.set_title("Gauge differences vs length")
    ax_error.set_xlabel("time / au")
    ax_error.set_ylabel("absolute error")
    ax_error.legend(ncol=2)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    nsteps = int(round(args.t_final / args.dt))
    if nsteps <= 0:
        raise ValueError("t-final must be larger than dt")
    t_final = nsteps * args.dt

    anchors, s0, h0, occupations = toy_problem()
    coeff0 = ground_state_coefficients(h0, s0, occupations.size)
    rho0 = density_from_coefficients(coeff0, occupations)
    reference_geometry = PeierlsGeometry(anchors, s0)
    reference_populations = p0_site_populations(rho0, reference_geometry, 0.0)

    rows_by_gauge = {
        label: run_one_gauge(
            label=label,
            lambda_value=lambda_value,
            anchors=anchors,
            s0=s0,
            h0=h0,
            occupations=occupations,
            coeff0=coeff0,
            model_name=args.model,
            reference_populations=reference_populations,
            hubbard_u=args.hubbard_u,
            mixing=args.mixing,
            midpoint_tolerance=args.midpoint_tolerance,
            density_tolerance=args.density_tolerance,
            max_iterations=args.max_iterations,
            dt=args.dt,
            nsteps=nsteps,
            field_amplitude=args.field_amplitude,
            omega=args.omega,
        )
        for label, lambda_value in GAUGES.items()
    }

    csv_path = args.output_dir / "p0_toy_gauge_compare.csv"
    write_rows(csv_path, rows_by_gauge)
    summary = summarize_p0_gauge_errors(rows_by_gauge)

    plot_path = args.output_dir / "p0_toy_gauge_compare.png"
    if not args.no_plot:
        plot_summary(plot_path, rows_by_gauge)

    print(
        "# p0 toy gauge comparison",
        f"model={args.model}",
        f"dt={args.dt}",
        f"t_final={t_final}",
        f"field_amplitude={args.field_amplitude}",
        f"omega={args.omega}",
        f"csv={csv_path}",
        f"plot={plot_path if not args.no_plot else 'disabled'}",
    )
    for label, values in summary.items():
        print(
            label,
            "max_dipole_norm_error={max_dipole_norm_error:.3e}".format(**values),
            "max_energy_abs_error={max_energy_abs_error:.3e}".format(**values),
            "max_population_norm_error={max_population_norm_error:.3e}".format(**values),
            "max_orthonormality_error={max_orthonormality_error:.3e}".format(**values),
            "max_instantaneous_continuity_residual={max_instantaneous_continuity_residual:.3e}".format(**values),
            "max_sampled_continuity_residual={max_continuity_residual:.3e}".format(**values),
            "max_power_residual={max_power_residual:.3e}".format(**values),
            "max_dipole_power_residual={max_dipole_power_residual:.3e}".format(**values),
            "max_abs_current={max_abs_current:.3e}".format(**values),
        )


if __name__ == "__main__":
    main()
