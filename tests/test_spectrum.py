from __future__ import annotations

import numpy as np
import pytest

from aion.spectrum import kick_spectrum


def test_kick_spectrum_peak_tracks_signal_frequency():
    dt = 0.05
    time = np.arange(0.0, 600.0 + 0.5 * dt, dt)
    omega0 = 0.2
    signal = 0.01 * np.sin(omega0 * time)

    spectrum = kick_spectrum(
        time,
        signal,
        kick_au=1.0e-3,
        damping_ha=0.002,
        zero_pad_factor=4,
        max_energy_ev=10.0,
    )
    peak = spectrum.omega_ha[np.argmax(spectrum.strength)]

    assert abs(peak - omega0) < 2.0 * np.pi / time[-1]
    assert spectrum.n_time_points == time.size
    assert spectrum.nfft == 4 * time.size
    assert spectrum.zero_pad_factor == 4


def test_kick_spectrum_rejects_nonuniform_time_grid():
    with pytest.raises(ValueError, match="uniformly spaced"):
        kick_spectrum([0.0, 0.1, 0.25], [0.0, 1.0, 0.0], kick_au=1.0)
