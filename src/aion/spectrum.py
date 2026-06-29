"""Signal-processing helpers for real-time kick spectra."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .linear_response import HARTREE_TO_EV


@dataclass(frozen=True)
class KickSpectrum:
    """Frequency-domain response reconstructed from a delta-kick trajectory."""

    omega_ha: np.ndarray
    energy_ev: np.ndarray
    strength: np.ndarray
    n_time_points: int
    nfft: int
    zero_pad_factor: int


def _uniform_timestep(time_au: np.ndarray) -> float:
    if time_au.ndim != 1 or time_au.size < 2:
        raise ValueError("time_au must be a one-dimensional array with at least two points")
    dt = float(time_au[1] - time_au[0])
    if dt <= 0.0:
        raise ValueError("time_au must be strictly increasing")
    if not np.allclose(np.diff(time_au), dt, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("time_au must be uniformly spaced")
    return dt


def kick_spectrum(
    time_au: Iterable[float],
    dipole_au: Iterable[float],
    *,
    kick_au: float,
    damping_ha: float = 0.0,
    zero_pad_factor: int = 1,
    max_energy_ev: float | None = None,
) -> KickSpectrum:
    """Return a damped linear-response spectrum from a projected dipole trace.

    ``kick_au`` is the integrated electric-field impulse used to create the
    delta kick.  The returned ``strength`` is the conventional arbitrary-unit
    quantity used in the examples, ``omega * abs(Im alpha(omega))``.
    """

    time = np.asarray(time_au, dtype=float)
    dipole = np.asarray(dipole_au, dtype=float)
    if dipole.shape != time.shape:
        raise ValueError("dipole_au must have the same shape as time_au")
    if kick_au == 0.0:
        raise ValueError("kick_au must be nonzero")
    if damping_ha < 0.0:
        raise ValueError("damping_ha must be nonnegative")
    if zero_pad_factor <= 0:
        raise ValueError("zero_pad_factor must be positive")
    if max_energy_ev is not None and max_energy_ev <= 0.0:
        raise ValueError("max_energy_ev must be positive or None")

    dt = _uniform_timestep(time)
    signal = dipole - dipole[0]
    windowed = signal * np.exp(-damping_ha * time)
    nfft = int(zero_pad_factor) * time.size
    alpha = dt * np.fft.fft(windowed, n=nfft) / kick_au
    omega = 2.0 * np.pi * np.fft.fftfreq(nfft, d=dt)
    keep = omega > 0.0
    if max_energy_ev is not None:
        keep &= omega * HARTREE_TO_EV <= max_energy_ev
    omega = omega[keep]
    return KickSpectrum(
        omega_ha=omega,
        energy_ev=omega * HARTREE_TO_EV,
        strength=omega * np.abs(alpha[keep].imag),
        n_time_points=int(time.size),
        nfft=int(nfft),
        zero_pad_factor=int(zero_pad_factor),
    )


def write_kick_spectrum_csv(path: str | Path, spectrum: KickSpectrum) -> None:
    """Write a ``KickSpectrum`` to a CSV file."""

    import csv

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "omega_ha",
                "energy_ev",
                "strength_arb",
                "n_time_points",
                "nfft",
                "zero_pad_factor",
            ]
        )
        for omega, energy, strength in zip(
            spectrum.omega_ha,
            spectrum.energy_ev,
            spectrum.strength,
        ):
            writer.writerow(
                [
                    omega,
                    energy,
                    strength,
                    spectrum.n_time_points,
                    spectrum.nfft,
                    spectrum.zero_pad_factor,
                ]
            )
