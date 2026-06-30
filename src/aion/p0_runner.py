"""Reusable pure Peierls P0 SCEM trajectory runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

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

    rt = VariableMetricSCEM(geometry, model, occupations)
    coeff = np.asarray(coeff0, dtype=np.complex128).copy()
    rows: list[P0Row] = [
        {
            "gauge": label,
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
        rows.append(
            {
                "gauge": label,
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
            }
        )
    return rows


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

    lambdas = DEFAULT_GAUGE_LAMBDAS if gauge_lambdas is None else gauge_lambdas
    return {
        label: run_p0_scem_trajectory(
            label=label,
            geometry=geometry_factory(
                constant_uniform_electric_gauge(field, lambda_value=lambda_value)
            ),
            model=model,
            occupations=occupations,
            coeff0=coeff0,
            settings=settings,
        )
        for label, lambda_value in lambdas.items()
    }
