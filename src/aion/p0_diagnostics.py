"""Trajectory diagnostics for pure Peierls P0 dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .gauge import PeierlsGeometry
from .matrix_models import density_from_coefficients
from .observables import (
    coefficient_orthonormality_error,
    electron_count,
    electronic_energy,
    energy_derivative,
    p0_continuity_residual,
    p0_dipole_derivative,
    p0_dipole_moment,
    p0_dipole_power,
    p0_graph_current_vector,
    p0_graph_currents,
    p0_site_charge_derivative,
    p0_site_charges,
    p0_site_populations,
    p0_source_power,
)
from .p0_e1 import (
    p0_e1_dipole_derivative,
    p0_e1_dipole_moment,
    p0_e1_dipole_power,
    p0_e1_intrinsic_dipole_derivative,
    p0_e1_uniform_residual_current,
)


P0Row = dict[str, float | int | str]


def coefficient_derivative(
    coeff: np.ndarray,
    hamiltonian: np.ndarray,
    geometry: PeierlsGeometry,
    time: float,
) -> np.ndarray:
    """Return the instantaneous P0 coefficient derivative."""

    c = np.asarray(coeff, dtype=np.complex128)
    h = np.asarray(hamiltonian, dtype=np.complex128)
    if c.ndim != 2:
        raise ValueError("coeff must be a two-dimensional array")
    if h.shape != (c.shape[0], c.shape[0]):
        raise ValueError("hamiltonian must match the AO dimension of coeff")
    if h.shape != geometry.overlap0.shape:
        raise ValueError(f"hamiltonian must have shape {geometry.overlap0.shape}")

    metric = geometry.metric(time)
    covariant_metric_dot = geometry.covariant_metric_dot(time)
    sigma = geometry.ao_sigma(time)
    return (
        (-1j / geometry.hbar) * np.linalg.solve(metric, h @ c)
        - 0.5 * np.linalg.solve(metric, covariant_metric_dot @ c)
        - sigma[:, None] * c
    )


def density_derivative_from_coefficients(
    coeff: np.ndarray,
    coeff_dot: np.ndarray,
    occupations: np.ndarray,
) -> np.ndarray:
    """Return ``d/dt sum_i f_i |C_i><C_i|`` from coefficient derivatives."""

    c = np.asarray(coeff, dtype=np.complex128)
    c_dot = np.asarray(coeff_dot, dtype=np.complex128)
    occ = np.asarray(occupations, dtype=float)
    if c.ndim != 2:
        raise ValueError("coeff must be a two-dimensional array")
    if c_dot.shape != c.shape:
        raise ValueError("coeff_dot must have the same shape as coeff")
    if occ.shape != (c.shape[1],):
        raise ValueError("occupations must have one entry per occupied column")
    return (c_dot * occ[None, :]) @ c.conj().T + (
        c * occ[None, :]
    ) @ c_dot.conj().T


def electric_field_for_power(geometry: PeierlsGeometry, time: float) -> np.ndarray:
    """Return the uniform electric field used by P0 power diagnostics."""

    if geometry.electric is None:
        return np.zeros(3)
    return geometry.electric.electric_field(time)


def model_energy(
    model: Any,
    density: np.ndarray,
    time: float,
    geometry: PeierlsGeometry,
    hamiltonian: np.ndarray | None = None,
) -> float:
    """Return model energy, falling back to ``Tr rho H`` when needed."""

    if hasattr(model, "energy"):
        return float(model.energy(density, time, geometry))
    if hamiltonian is None:
        hamiltonian = model.hamiltonian(density, time, geometry)
    return electronic_energy(density, hamiltonian)


def model_energy_derivative(
    model: Any,
    density: np.ndarray,
    density_dot: np.ndarray,
    time: float,
    geometry: PeierlsGeometry,
    hamiltonian: np.ndarray | None = None,
    *,
    eps: float = 1.0e-6,
) -> float:
    """Return the material energy derivative along the instantaneous EOM."""

    if eps <= 0.0:
        raise ValueError("eps must be positive")

    if hasattr(model, "energy"):
        plus_density = density + eps * density_dot
        minus_density = density - eps * density_dot
        plus_energy = model.energy(plus_density, time + eps, geometry)
        minus_energy = model.energy(minus_density, time - eps, geometry)
        return float((plus_energy - minus_energy) / (2.0 * eps))

    if hamiltonian is None:
        hamiltonian = model.hamiltonian(density, time, geometry)
    h_dot = (
        model.hamiltonian(density, time + eps, geometry)
        - model.hamiltonian(density, time - eps, geometry)
    ) / (2.0 * eps)
    return energy_derivative(density, density_dot, hamiltonian, hamiltonian_dot=h_dot)


def record_p0_observables(
    *,
    step: int,
    time: float,
    coeff: np.ndarray,
    geometry: PeierlsGeometry,
    model: Any,
    occupations: np.ndarray,
    midpoint_iterations: int = 0,
    hamiltonian_residual: float = 0.0,
    density_residual: float | None = None,
    energy_derivative_eps: float = 1.0e-6,
) -> P0Row:
    """Return one flat row of P0 trajectory diagnostics."""

    rho = density_from_coefficients(coeff, occupations)
    hamiltonian = model.hamiltonian(rho, time, geometry)
    coeff_dot = coefficient_derivative(coeff, hamiltonian, geometry, time)
    rho_dot = density_derivative_from_coefficients(coeff, coeff_dot, occupations)

    populations = p0_site_populations(rho, geometry, time)
    charges = p0_site_charges(rho, geometry, time)
    charge_derivative = p0_site_charge_derivative(rho, rho_dot, geometry, time)
    dipole = p0_dipole_moment(rho, geometry, time)
    metric = geometry.metric(time)
    currents = p0_graph_currents(rho, hamiltonian, geometry, time)
    current_balance = np.sum(currents, axis=1)
    continuity_residual = p0_continuity_residual(charge_derivative, currents)
    p0_current_vector = p0_graph_current_vector(currents, geometry)
    p0_dipole_dot = p0_dipole_derivative(charge_derivative, geometry)
    energy = model_energy(model, rho, time, geometry, hamiltonian)
    energy_dot = model_energy_derivative(
        model,
        rho,
        rho_dot,
        time,
        geometry,
        hamiltonian,
        eps=energy_derivative_eps,
    )
    source_power = p0_source_power(currents, geometry, time)
    dipole_power = p0_dipole_power(
        charge_derivative,
        geometry,
        electric_field_for_power(geometry, time),
    )
    field = electric_field_for_power(geometry, time)
    source_current = p0_current_vector
    source_current_power = float(np.dot(field, source_current))

    row: P0Row = {
        "step": int(step),
        "time_au": float(time),
        "energy": energy,
        "energy_derivative": energy_dot,
        "source_power": source_power,
        "dipole_power": dipole_power,
        "power_residual": energy_dot - source_power,
        "dipole_power_residual": source_power - dipole_power,
        "electron_count": electron_count(rho, metric),
        "dipole_x": float(dipole[0]),
        "dipole_y": float(dipole[1]),
        "dipole_z": float(dipole[2]),
        "source_current_x": float(source_current[0]),
        "source_current_y": float(source_current[1]),
        "source_current_z": float(source_current[2]),
        "source_current_power": source_current_power,
        "source_current_power_residual": energy_dot - source_current_power,
        "source_current_dipole_derivative_residual_norm": float(
            np.linalg.norm(source_current - p0_dipole_dot)
        ),
        "orthonormality_error": coefficient_orthonormality_error(coeff, metric),
        "midpoint_iterations": int(midpoint_iterations),
        "hamiltonian_residual": float(hamiltonian_residual),
        "density_residual": (
            float("nan") if density_residual is None else float(density_residual)
        ),
        "max_abs_current": float(np.max(np.abs(currents))),
        "instantaneous_continuity_residual_norm": float(
            np.linalg.norm(continuity_residual)
        ),
    }
    central_dipoles0 = getattr(model, "central_dipoles0", None)
    if central_dipoles0 is not None:
        e1_dipole = p0_e1_dipole_moment(rho, central_dipoles0, geometry, time)
        e1_dipole_derivative = p0_e1_dipole_derivative(
            charge_derivative=charge_derivative,
            density=rho,
            density_dot=rho_dot,
            central_dipoles0=central_dipoles0,
            geometry=geometry,
            t=time,
        )
        e1_dipole_power = p0_e1_dipole_power(
            charge_derivative=charge_derivative,
            density=rho,
            density_dot=rho_dot,
            central_dipoles0=central_dipoles0,
            geometry=geometry,
            t=time,
            electric_field=field,
        )
        intrinsic_current = p0_e1_intrinsic_dipole_derivative(
            density=rho,
            density_dot=rho_dot,
            central_dipoles0=central_dipoles0,
            geometry=geometry,
            t=time,
        )
        residual_current = p0_e1_uniform_residual_current(
            density=rho,
            density_dot=rho_dot,
            central_dipoles0=central_dipoles0,
            geometry=geometry,
            t=time,
            electric_field=field,
        )
        phase_response = residual_current - intrinsic_current
        base_hamiltonian = hamiltonian - model.e1_hamiltonian(time, geometry)
        split_graph_currents = p0_graph_currents(
            rho,
            hamiltonian,
            geometry,
            time,
            source_hamiltonian=base_hamiltonian,
        )
        split_p0_current = p0_graph_current_vector(
            split_graph_currents,
            geometry,
        )
        source_current = split_p0_current + residual_current
        source_current_power = float(np.dot(field, source_current))
        row.update(
            {
                "e1_dipole_x": float(e1_dipole[0]),
                "e1_dipole_y": float(e1_dipole[1]),
                "e1_dipole_z": float(e1_dipole[2]),
                "e1_dipole_derivative_x": float(e1_dipole_derivative[0]),
                "e1_dipole_derivative_y": float(e1_dipole_derivative[1]),
                "e1_dipole_derivative_z": float(e1_dipole_derivative[2]),
                "p0_action_current_x": float(split_p0_current[0]),
                "p0_action_current_y": float(split_p0_current[1]),
                "p0_action_current_z": float(split_p0_current[2]),
                "e1_polarization_current_x": float(intrinsic_current[0]),
                "e1_polarization_current_y": float(intrinsic_current[1]),
                "e1_polarization_current_z": float(intrinsic_current[2]),
                "e1_phase_response_current_x": float(phase_response[0]),
                "e1_phase_response_current_y": float(phase_response[1]),
                "e1_phase_response_current_z": float(phase_response[2]),
                "e1_residual_current_x": float(residual_current[0]),
                "e1_residual_current_y": float(residual_current[1]),
                "e1_residual_current_z": float(residual_current[2]),
                "source_current_x": float(source_current[0]),
                "source_current_y": float(source_current[1]),
                "source_current_z": float(source_current[2]),
                "source_current_power": source_current_power,
                "source_current_power_residual": energy_dot
                - source_current_power,
                "source_current_dipole_derivative_residual_norm": float(
                    np.linalg.norm(source_current - e1_dipole_derivative)
                ),
                "e1_dipole_power": e1_dipole_power,
                "e1_power_residual": energy_dot - e1_dipole_power,
            }
        )
        if hasattr(model, "coupling_energy"):
            row["e1_coupling_energy"] = float(
                model.coupling_energy(rho, time, geometry)
            )
    for atom_index, population in enumerate(populations):
        row[f"population_{atom_index}"] = float(population)
        row[f"charge_{atom_index}"] = float(charges[atom_index])
        row[f"charge_derivative_{atom_index}"] = float(charge_derivative[atom_index])
        row[f"current_balance_{atom_index}"] = float(current_balance[atom_index])
        row[f"continuity_residual_{atom_index}"] = float(
            continuity_residual[atom_index]
        )
    for row_atom in range(currents.shape[0]):
        for col_atom in range(row_atom + 1, currents.shape[1]):
            row[f"current_{row_atom}_{col_atom}"] = float(currents[row_atom, col_atom])
    return row


def p0_row_series(rows: list[P0Row], key: str) -> np.ndarray:
    """Return one numeric column from flat P0 diagnostic rows."""

    return np.asarray([float(row[key]) for row in rows], dtype=float)


def p0_natom_from_rows(rows: list[P0Row]) -> int:
    """Infer the number of P0 sites from diagnostic row keys."""

    if not rows:
        raise ValueError("rows must be nonempty")
    return sum(1 for key in rows[0] if key.startswith("population_"))


def trajectory_continuity_residual(
    rows: list[P0Row],
) -> np.ndarray:
    """Return central-difference ``dQ/dt + sum_b I_ab`` from recorded rows."""

    if len(rows) < 3:
        return np.empty((0, p0_natom_from_rows(rows)), dtype=float)
    natom = p0_natom_from_rows(rows)
    times = p0_row_series(rows, "time_au")
    charges = np.column_stack(
        [p0_row_series(rows, f"charge_{atom}") for atom in range(natom)]
    )
    balances = np.column_stack(
        [p0_row_series(rows, f"current_balance_{atom}") for atom in range(natom)]
    )
    derivative = (charges[2:] - charges[:-2]) / (times[2:, None] - times[:-2, None])
    return derivative + balances[1:-1]


def summarize_p0_gauge_errors(
    rows_by_gauge: dict[str, list[P0Row]],
    *,
    reference_gauge: str = "length",
) -> dict[str, dict[str, float]]:
    """Return trajectory-level source-observable differences between gauges."""

    if reference_gauge not in rows_by_gauge:
        raise KeyError(f"missing reference gauge {reference_gauge!r}")
    reference = rows_by_gauge[reference_gauge]
    natom = p0_natom_from_rows(reference)
    ref_energy = p0_row_series(reference, "energy")
    ref_dipole = np.column_stack(
        [p0_row_series(reference, f"dipole_{axis}") for axis in "xyz"]
    )
    ref_pop = np.column_stack(
        [p0_row_series(reference, f"population_{atom}") for atom in range(natom)]
    )
    has_source_current = all(
        f"source_current_{axis}" in reference[0] for axis in "xyz"
    )
    ref_source_current = None
    if has_source_current:
        ref_source_current = np.column_stack(
            [p0_row_series(reference, f"source_current_{axis}") for axis in "xyz"]
        )
    has_e1 = all(f"e1_dipole_{axis}" in reference[0] for axis in "xyz")
    ref_e1_dipole = None
    if has_e1:
        ref_e1_dipole = np.column_stack(
            [p0_row_series(reference, f"e1_dipole_{axis}") for axis in "xyz"]
        )

    summary = {}
    for label, rows in rows_by_gauge.items():
        energy = p0_row_series(rows, "energy")
        dipole = np.column_stack(
            [p0_row_series(rows, f"dipole_{axis}") for axis in "xyz"]
        )
        population = np.column_stack(
            [p0_row_series(rows, f"population_{atom}") for atom in range(natom)]
        )
        continuity = trajectory_continuity_residual(rows)
        label_summary = {
            "max_energy_abs_error": float(np.max(np.abs(energy - ref_energy))),
            "max_dipole_norm_error": float(
                np.max(np.linalg.norm(dipole - ref_dipole, axis=1))
            ),
            "max_population_norm_error": float(
                np.max(np.linalg.norm(population - ref_pop, axis=1))
            ),
            "max_orthonormality_error": float(
                np.max(p0_row_series(rows, "orthonormality_error"))
            ),
            "max_continuity_residual": float(
                0.0
                if continuity.size == 0
                else np.max(np.linalg.norm(continuity, axis=1))
            ),
            "max_instantaneous_continuity_residual": float(
                np.max(p0_row_series(rows, "instantaneous_continuity_residual_norm"))
            ),
            "max_power_residual": float(
                np.max(np.abs(p0_row_series(rows, "power_residual")))
            ),
            "max_dipole_power_residual": float(
                np.max(np.abs(p0_row_series(rows, "dipole_power_residual")))
            ),
            "max_abs_current": float(np.max(p0_row_series(rows, "max_abs_current"))),
        }
        if has_source_current and all(
            f"source_current_{axis}" in rows[0] for axis in "xyz"
        ):
            assert ref_source_current is not None
            source_current = np.column_stack(
                [p0_row_series(rows, f"source_current_{axis}") for axis in "xyz"]
            )
            label_summary["max_source_current_norm_error"] = float(
                np.max(np.linalg.norm(source_current - ref_source_current, axis=1))
            )
            label_summary["max_source_current_power_residual"] = float(
                np.max(
                    np.abs(p0_row_series(rows, "source_current_power_residual"))
                )
            )
            label_summary[
                "max_source_current_dipole_derivative_residual"
            ] = float(
                np.max(
                    p0_row_series(
                        rows,
                        "source_current_dipole_derivative_residual_norm",
                    )
                )
            )
        if has_e1 and all(f"e1_dipole_{axis}" in rows[0] for axis in "xyz"):
            assert ref_e1_dipole is not None
            e1_dipole = np.column_stack(
                [p0_row_series(rows, f"e1_dipole_{axis}") for axis in "xyz"]
            )
            label_summary["max_e1_dipole_norm_error"] = float(
                np.max(np.linalg.norm(e1_dipole - ref_e1_dipole, axis=1))
            )
            label_summary["max_e1_power_residual"] = float(
                np.max(np.abs(p0_row_series(rows, "e1_power_residual")))
            )
        summary[label] = label_summary
    return summary
