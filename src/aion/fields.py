"""Simple electric-field envelopes for real-time tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array3 = np.ndarray


@dataclass(frozen=True)
class ContinuousWave:
    """Linearly polarized constant-amplitude sinusoidal field.

    Parameters are in atomic units.
    """

    amplitude: float
    omega: float
    polarization: Array3
    phase: float = 0.0

    def __post_init__(self) -> None:
        pol = np.asarray(self.polarization, dtype=float)
        norm = np.linalg.norm(pol)
        if norm == 0:
            raise ValueError("polarization vector must be nonzero")
        object.__setattr__(self, "polarization", pol / norm)

    def __call__(self, t: float) -> Array3:
        carrier = np.sin(self.omega * t + self.phase)
        return self.amplitude * carrier * self.polarization


@dataclass(frozen=True)
class GaussianPulse:
    """Linearly polarized Gaussian-envelope sinusoidal field.

    Parameters are in atomic units.
    """

    amplitude: float
    omega: float
    center: float
    sigma: float
    polarization: Array3
    phase: float = 0.0

    def __post_init__(self) -> None:
        pol = np.asarray(self.polarization, dtype=float)
        norm = np.linalg.norm(pol)
        if norm == 0:
            raise ValueError("polarization vector must be nonzero")
        object.__setattr__(self, "polarization", pol / norm)

    def __call__(self, t: float) -> Array3:
        x = (t - self.center) / self.sigma
        env = np.exp(-0.5 * x * x)
        carrier = np.sin(self.omega * (t - self.center) + self.phase)
        return self.amplitude * env * carrier * self.polarization


@dataclass(frozen=True)
class Sin2Pulse:
    """Finite sin^2-envelope sinusoidal field."""

    amplitude: float
    omega: float
    t0: float
    duration: float
    polarization: Array3
    phase: float = 0.0

    @classmethod
    def from_cycles(
        cls,
        *,
        amplitude: float,
        omega: float,
        cycles: float,
        polarization: Array3,
        t0: float = 0.0,
        phase: float = 0.0,
    ) -> "Sin2Pulse":
        if omega <= 0:
            raise ValueError("omega must be positive")
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        duration = cycles * 2.0 * np.pi / omega
        return cls(
            amplitude=amplitude,
            omega=omega,
            t0=t0,
            duration=duration,
            polarization=polarization,
            phase=phase,
        )

    def __post_init__(self) -> None:
        pol = np.asarray(self.polarization, dtype=float)
        norm = np.linalg.norm(pol)
        if norm == 0:
            raise ValueError("polarization vector must be nonzero")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        object.__setattr__(self, "polarization", pol / norm)

    def __call__(self, t: float) -> Array3:
        if t < self.t0 or t > self.t0 + self.duration:
            return np.zeros(3)
        tau = (t - self.t0) / self.duration
        env = np.sin(np.pi * tau) ** 2
        carrier = np.sin(self.omega * (t - self.t0) + self.phase)
        return self.amplitude * env * carrier * self.polarization
