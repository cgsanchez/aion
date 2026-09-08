"""Reusable pure Peierls P0 SCEM trajectory runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .connection_cayley import ConnectionCayleySCEM
from .gauge import PeierlsGeometry, UniformElectricGauge
from .p0_diagnostics import P0Row, record_p0_observables
from .variable_metric import VariableMetricSCEM


DEFAULT_GAUGE_LAMBDAS = {
    "length": 0.0,
    "mixed": 0.5,
    "velocity": 1.0,
}


@dataclass(frozen=True)
class P0SCEMSettings:
    """Numerical settings for one P0 SCEM trajectory."""

    dt: float = 0.02
    nsteps: int = 10
    midpoint_tolerance: float = 1.0e-9
    density_tolerance: float | None = 1.0e-9
    max_iterations: int = 18
    mixing: float = 0.7
    integrator: str = "lowdin_exponential"

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.nsteps < 0:
            raise ValueError("nsteps must be nonnegative")
        if self.midpoint_tolerance <= 0.0:
            raise ValueError("midpoint_tolerance must be positive")
        if self.density_tolerance is not None and self.density_tolerance <= 0.0:
            raise ValueError("density_tolerance must be positive or None")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not (0.0 < self.mixing <= 1.0):
            raise ValueError("mixing must be in (0, 1]")
        if self.integrator not in {"lowdin_exponential", "connection_cayley"}:
            raise ValueError(
                "integrator must be 'lowdin_exponential' or 'connection_cayley'"
            )


def constant_uniform_electric_gauge(
    field: np.ndarray,
    *,
    lambda_value: float,
) -> UniformElectricGauge:
    """Return a length/velocity interpolation gauge for constant uniform E."""

    field_array = np.asarray(field, dtype=float)
    if field_array.shape != (3,):
        raise ValueError("field must have shape (3,)")
    lam = float(lambda_value)
    return UniformElectricGauge(
        field=lambda _t: field_array,
        field_integral=lambda t: field_array * t,
        lambda_value=lambda _t: lam,
        lambda_derivative=lambda _t: 0.0,
    )


def time_dependent_uniform_electric_gauge(
    *,
    field: Callable[[float], np.ndarray],
    field_integral: Callable[[float], np.ndarray],
    lambda_value: float,
) -> UniformElectricGauge:
    """Return a uniform electric gauge from analytic E(t) and impulse K(t)."""

    lam = float(lambda_value)
    return UniformElectricGauge(
        field=field,
        field_integral=field_integral,
        lambda_value=lambda _t: lam,
        lambda_derivative=lambda _t: 0.0,
    )


def sin2_uniform_electric_gauge(
    *,
    amplitude: float,
    omega: float,
    cycles: float,
    polarization: np.ndarray,
    lambda_value: float,
    t0: float = 0.0,
    phase: float = 0.0,
) -> UniformElectricGauge:
    """Return a finite sin²-envelope uniform electric gauge.

    The field is ``amplitude sin²(pi tau / T) sin(omega tau + phase) e`` for
    ``0 <= tau <= T`` and zero outside.  The impulse is analytic and is held
    constant after the pulse has passed.
    """

    if amplitude < 0.0:
        raise ValueError("amplitude must be nonnegative")
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    if cycles <= 0.0:
        raise ValueError("cycles must be positive")
    pol = np.asarray(polarization, dtype=float)
    norm = np.linalg.norm(pol)
    if norm == 0.0:
        raise ValueError("polarization must be nonzero")
    pol = pol / norm
    duration = cycles * 2.0 * np.pi / omega
    envelope_frequency = 2.0 * np.pi / duration

    def primitive_sine(k: float, tau: float) -> float:
        if abs(k) < 1.0e-14:
            return tau * np.sin(phase)
        return (np.cos(phase) - np.cos(k * tau + phase)) / k

    def scalar_impulse_tau(tau: float) -> float:
        tau_clip = min(max(tau, 0.0), duration)
        base = primitive_sine(omega, tau_clip)
        upper = primitive_sine(omega + envelope_frequency, tau_clip)
        lower = primitive_sine(omega - envelope_frequency, tau_clip)
        return 0.5 * amplitude * (base - 0.5 * upper - 0.5 * lower)

    def field(t: float) -> np.ndarray:
        if t < t0 or t > t0 + duration:
            return np.zeros(3)
        tau = t - t0
        envelope = np.sin(np.pi * tau / duration) ** 2
        carrier = np.sin(omega * tau + phase)
        return amplitude * envelope * carrier * pol

    def field_integral(t: float) -> np.ndarray:
        return scalar_impulse_tau(t - t0) * pol

    return time_dependent_uniform_electric_gauge(
        field=field,
        field_integral=field_integral,
        lambda_value=lambda_value,
    )


def velocity_delta_kick_electric_gauge(
    impulse: np.ndarray,
) -> UniformElectricGauge:
    """Return the post-kick velocity-gauge source for an impulsive E field."""

    impulse_array = np.asarray(impulse, dtype=float)
    if impulse_array.shape != (3,):
        raise ValueError("impulse must have shape (3,)")
    return UniformElectricGauge.velocity(
        field=lambda _t: np.zeros(3),
        field_integral=lambda _t: impulse_array,
    )


def p0_site_gauge_phase(
    source: PeierlsGeometry,
    target: PeierlsGeometry,
    *,
    time: float = 0.0,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return site phases mapping one gauge-related P0 geometry to another."""

    same_anchors = (
        np.allclose(source.anchors.atom_coords, target.anchors.atom_coords)
        and np.array_equal(source.anchors.ao_to_atom, target.anchors.ao_to_atom)
    )
    if not same_anchors:
        raise ValueError("source and target geometries must share anchors")
    if source.charge != target.charge or source.hbar != target.hbar:
        raise ValueError("source and target geometries must share charge and hbar")
    delta = target.site_bond_line_integrals(time) - source.site_bond_line_integrals(
        time
    )
    site_lambda = delta[:, 0]
    residual = delta - (site_lambda[:, None] - site_lambda[None, :])
    if np.linalg.norm(residual) > tolerance:
        raise ValueError("geometries are not related by a site gauge at this time")
    return np.exp((1j * source.charge / source.hbar) * site_lambda)


def transform_p0_coefficients_between_gauges(
    coeff: np.ndarray,
    source: PeierlsGeometry,
    target: PeierlsGeometry,
    *,
    time: float = 0.0,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Transform occupied coefficients between gauge-related P0 geometries."""

    phase = source.anchors.lift_site_vector(
        p0_site_gauge_phase(source, target, time=time, tolerance=tolerance)
    )
    return phase[:, None] * np.asarray(coeff, dtype=np.complex128)


def apply_p0_velocity_delta_kick(
    coeff: np.ndarray,
    field_free_geometry: PeierlsGeometry,
    kicked_geometry: PeierlsGeometry,
    *,
    time: float = 0.0,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Transform field-free coefficients into a post-kick velocity gauge."""

    return transform_p0_coefficients_between_gauges(
        coeff,
        field_free_geometry,
        kicked_geometry,
        time=time,
        tolerance=tolerance,
    )


def run_p0_scem_trajectory(
    *,
    label: str,
    geometry: PeierlsGeometry,
    model,
    occupations: np.ndarray,
    coeff0: np.ndarray,
    settings: P0SCEMSettings,
) -> list[P0Row]:
    """Propagate one P0 trajectory and record diagnostics at every step."""

    integrator_class = (
        ConnectionCayleySCEM
        if settings.integrator == "connection_cayley"
        else VariableMetricSCEM
    )
    rt = integrator_class(geometry, model, occupations)
    coeff = np.asarray(coeff0, dtype=np.complex128).copy()
    rows: list[P0Row] = [
        {
            "gauge": label,
            "integrator": settings.integrator,
            **record_p0_observables(
                step=0,
                time=0.0,
                coeff=coeff,
                geometry=geometry,
                model=model,
                occupations=occupations,
            ),
        }
    ]

    for step in range(1, settings.nsteps + 1):
        result = rt.step(
            coeff,
            time=(step - 1) * settings.dt,
            dt=settings.dt,
            midpoint_tolerance=settings.midpoint_tolerance,
            density_tolerance=settings.density_tolerance,
            max_iterations=settings.max_iterations,
            mixing=settings.mixing,
        )
        coeff = result.coeff_next
        connection_diagnostics = {}
        if hasattr(result, "left_connection_metric_error"):
            connection_diagnostics = {
                "left_connection_metric_error": result.left_connection_metric_error,
                "right_connection_metric_error": result.right_connection_metric_error,
                "maximum_connection_correction_norm": (
                    result.maximum_connection_correction_norm
                ),
            }
        rows.append(
            {
                "gauge": label,
                "integrator": settings.integrator,
                **record_p0_observables(
                    step=step,
                    time=step * settings.dt,
                    coeff=coeff,
                    geometry=geometry,
                    model=model,
                    occupations=occupations,
                    midpoint_iterations=result.iterations,
                    hamiltonian_residual=result.hamiltonian_residual,
                    density_residual=result.density_residual,
                ),
                **connection_diagnostics,
            }
        )
    return rows


def run_p0_electric_gauge_comparison(
    *,
    geometry_factory: Callable[[UniformElectricGauge], PeierlsGeometry],
    electric_factory: Callable[[float], UniformElectricGauge],
    model,
    occupations: np.ndarray,
    coeff0: np.ndarray,
    settings: P0SCEMSettings,
    gauge_lambdas: dict[str, float] | None = None,
) -> dict[str, list[P0Row]]:
    """Run length/mixed/velocity P0 trajectories for an electric protocol."""

    lambdas = DEFAULT_GAUGE_LAMBDAS if gauge_lambdas is None else gauge_lambdas
    return {
        label: run_p0_scem_trajectory(
            label=label,
            geometry=geometry_factory(electric_factory(lambda_value)),
            model=model,
            occupations=occupations,
            coeff0=coeff0,
            settings=settings,
        )
        for label, lambda_value in lambdas.items()
    }


def run_p0_uniform_electric_gauge_comparison(
    *,
    geometry_factory: Callable[[UniformElectricGauge], PeierlsGeometry],
    model,
    occupations: np.ndarray,
    coeff0: np.ndarray,
    field: np.ndarray,
    settings: P0SCEMSettings,
    gauge_lambdas: dict[str, float] | None = None,
) -> dict[str, list[P0Row]]:
    """Run length/mixed/velocity P0 trajectories for one uniform E field."""

    return run_p0_electric_gauge_comparison(
        geometry_factory=geometry_factory,
        electric_factory=lambda lambda_value: constant_uniform_electric_gauge(
            field,
            lambda_value=lambda_value,
        ),
        model=model,
        occupations=occupations,
        coeff0=coeff0,
        settings=settings,
        gauge_lambdas=gauge_lambdas,
    )
