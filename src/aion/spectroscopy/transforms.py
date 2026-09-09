"""Explicit positive-frequency transforms and molecular kick responses."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from aion.errors import SpectroscopyError
from aion.io.trajectory import Trajectory, load_trajectory
from aion.spectroscopy.types import (
    BaselineKind,
    FourierTransformResult,
    KickSpectrum,
    KickSpectrumConfig,
    QuadratureKind,
    SpectrumSourceLink,
    TransformConfig,
    WindowKind,
)


def _time_series(
    time_au: object, values: object, name: str
) -> tuple[np.ndarray, np.ndarray, float]:
    time = np.asarray(time_au, dtype=np.float64)
    array = np.asarray(values)
    if time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time)):
        raise SpectroscopyError("transform time grid must contain at least two finite samples")
    if array.ndim < 1 or array.shape[0] != time.size:
        raise SpectroscopyError(f"{name} must use time as its first axis")
    if array.dtype.kind not in "fci" or not np.all(np.isfinite(array)):
        raise SpectroscopyError(f"{name} must contain finite numeric values")
    differences = np.diff(time)
    dt = float(differences[0])
    if dt <= 0.0:
        raise SpectroscopyError("transform times must be strictly increasing")
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(dt), abs(float(time[-1])))
    if not np.allclose(differences, dt, rtol=0.0, atol=tolerance):
        raise SpectroscopyError("positive-frequency FFT requires a uniform time grid")
    return time, np.asarray(array, dtype=np.complex128), dt


def positive_frequency_transform(
    time_au: object,
    values: object,
    config: TransformConfig | None = None,
) -> FourierTransformResult:
    r"""Approximate ``integral f(t) exp(+i omega t) dt`` by trapezoidal FFT.

    The first and last sample receive half weight.  NumPy's inverse FFT supplies
    the declared positive exponent; multiplication by NFFT removes its discrete
    normalization.  Zero padding changes only the sampled frequency grid.
    """

    selected = TransformConfig() if config is None else config
    if not isinstance(selected, TransformConfig):
        raise SpectroscopyError("transform config has the wrong type")
    if selected.window is not WindowKind.RECTANGULAR:
        raise SpectroscopyError(f"unsupported transform window {selected.window.value!r}")
    if selected.quadrature is not QuadratureKind.TRAPEZOIDAL:
        raise SpectroscopyError(f"unsupported time quadrature {selected.quadrature.value!r}")
    time, signal, dt = _time_series(time_au, values, "transform signal")
    tau = time - time[0]
    duration = float(tau[-1])
    nfft = selected.zero_padding_factor * time.size
    damping = np.exp(-selected.damping_energy_au * tau)
    damping_shape = (time.size, *(1 for _ in signal.shape[1:]))
    weighted = signal * damping.reshape(damping_shape)
    summed = np.fft.ifft(weighted, n=nfft, axis=0) * nfft
    omega_all = 2.0 * np.pi * np.fft.fftfreq(nfft, d=dt)
    phase_shape = (nfft, *(1 for _ in signal.shape[1:]))
    endpoint_phase = np.exp(1j * omega_all * duration).reshape(phase_shape)
    transformed = dt * (summed - 0.5 * weighted[0] - 0.5 * weighted[-1] * endpoint_phase)
    keep = omega_all > 0.0
    if selected.maximum_energy_au is not None:
        keep &= omega_all <= selected.maximum_energy_au
    omega = np.asarray(omega_all[keep], dtype=np.float64)
    if omega.size == 0:
        raise SpectroscopyError("transform configuration selects no positive frequencies")
    return FourierTransformResult(
        config=selected,
        omega_au=omega,
        values=np.asarray(transformed[keep], dtype=np.complex128),
        sample_count=time.size,
        nfft=nfft,
        start_time_au=float(time[0]),
        duration_au=duration,
        time_step_au=dt,
        intrinsic_resolution_au=2.0 * np.pi / duration,
        native_grid_spacing_au=2.0 * np.pi / (time.size * dt),
        padded_grid_spacing_au=2.0 * np.pi / (nfft * dt),
        nyquist_energy_au=np.pi / dt,
    )


def compute_kick_spectrum(
    *,
    time_au: object,
    dipole_au: object,
    current_au: object | None,
    impulse_au: object,
    baseline_dipole_au: object,
    config: KickSpectrumConfig,
    source: SpectrumSourceLink,
    event_id: str,
    dipole_definition_id: str,
    current_definition_id: str | None,
) -> KickSpectrum:
    """Compute one explicitly polarized kick-response column from sampled data."""

    if not isinstance(config, KickSpectrumConfig):
        raise SpectroscopyError("kick-spectrum config has the wrong type")
    time, dipole, _dt = _time_series(time_au, dipole_au, "dipole trace")
    if dipole.shape != (time.size, 3):
        raise SpectroscopyError("dipole trace must have shape (ntime, 3)")
    impulse = np.asarray(impulse_au, dtype=np.float64)
    baseline = np.asarray(baseline_dipole_au, dtype=np.float64)
    if impulse.shape != (3,) or baseline.shape != (3,):
        raise SpectroscopyError("kick impulse and dipole baseline must have shape (3,)")
    if not np.all(np.isfinite(impulse)) or not np.all(np.isfinite(baseline)):
        raise SpectroscopyError("kick impulse and dipole baseline must be finite")
    amplitude = float(np.linalg.norm(impulse))
    if amplitude == 0.0:
        raise SpectroscopyError("kick impulse cannot be zero")
    polarization = impulse / amplitude
    delta_dipole = dipole - baseline[None, :]
    dipole_transform = positive_frequency_transform(time, delta_dipole, config.transform)
    alpha_dipole = dipole_transform.values / amplitude
    alpha_parallel = alpha_dipole @ polarization
    absorption = dipole_transform.omega_au * alpha_parallel.imag
    oscillator_density = (2.0 / np.pi) * absorption

    alpha_current: np.ndarray | None = None
    residual: np.ndarray | None = None
    if current_au is None:
        if current_definition_id is not None:
            raise SpectroscopyError("current definition was supplied without a current trace")
    else:
        if current_definition_id is None:
            raise SpectroscopyError("current trace requires its definition identifier")
        current_time, current, _ = _time_series(time, current_au, "current trace")
        if current.shape != dipole.shape or not np.array_equal(current_time, time):
            raise SpectroscopyError("current and dipole traces must use the same Cartesian grid")
        current_transform = positive_frequency_transform(time, current, config.transform)
        if not np.array_equal(current_transform.omega_au, dipole_transform.omega_au):
            raise SpectroscopyError("current and dipole frequency grids differ")
        tau_end = dipole_transform.duration_au
        phase = np.exp(1j * dipole_transform.omega_au * tau_end)[:, None]
        upper = (
            delta_dipole[-1][None, :]
            * np.exp(-config.transform.damping_energy_au * tau_end)
            * phase
        )
        lower = delta_dipole[0][None, :]
        boundary = upper - lower
        denominator = (config.transform.damping_energy_au - 1j * dipole_transform.omega_au)[:, None]
        alpha_current = (current_transform.values - boundary) / denominator / amplitude
        residual = alpha_current - alpha_dipole

    return KickSpectrum(
        config=config,
        source=source,
        event_id=event_id,
        dipole_definition_id=dipole_definition_id,
        current_definition_id=current_definition_id,
        impulse_au=impulse,
        input_polarization=polarization,
        impulse_amplitude_au=amplitude,
        baseline_dipole_au=baseline,
        sample_count=dipole_transform.sample_count,
        nfft=dipole_transform.nfft,
        start_time_au=dipole_transform.start_time_au,
        duration_au=dipole_transform.duration_au,
        time_step_au=dipole_transform.time_step_au,
        intrinsic_resolution_au=dipole_transform.intrinsic_resolution_au,
        native_grid_spacing_au=dipole_transform.native_grid_spacing_au,
        padded_grid_spacing_au=dipole_transform.padded_grid_spacing_au,
        nyquist_energy_au=dipole_transform.nyquist_energy_au,
        omega_au=dipole_transform.omega_au,
        polarizability_dipole_au=alpha_dipole,
        polarizability_current_au=alpha_current,
        polarizability_parallel_au=alpha_parallel,
        absorption_strength_parallel_au=absorption,
        oscillator_strength_density_parallel_au=oscillator_density,
        current_domain_residual_au=residual,
    )


def _unique_definition(trajectory: Trajectory, prefix: str, supplied: str | None) -> str:
    if supplied is not None:
        if supplied not in trajectory.observable_ids or not supplied.startswith(prefix):
            raise SpectroscopyError(f"trajectory lacks requested {prefix} observable {supplied!r}")
        return supplied
    matches = [name for name in trajectory.observable_ids if name.startswith(prefix)]
    if len(matches) != 1:
        raise SpectroscopyError(f"trajectory must contain exactly one {prefix} observable")
    return matches[0]


def kick_spectrum_from_trajectory(
    trajectory: Trajectory | str | Path,
    config: KickSpectrumConfig,
    *,
    event_id: str | None = None,
    dipole_definition_id: str | None = None,
    current_definition_id: str | None = None,
) -> KickSpectrum:
    """Derive a reproducible spectrum from one completed, fully sampled kick run."""

    source_trajectory = (
        load_trajectory(trajectory) if isinstance(trajectory, str | Path) else trajectory
    )
    if not source_trajectory.complete:
        raise SpectroscopyError("kick spectroscopy requires a completed trajectory")
    dipole_id = _unique_definition(source_trajectory, "dipole.electronic.", dipole_definition_id)
    current_id = _unique_definition(source_trajectory, "current.primary.", current_definition_id)
    with h5py.File(source_trajectory.path, "r") as handle:
        source_definition = json.loads(
            bytes(handle["source/definition_json"][()]).decode("utf-8")
        )
        if source_definition.get("kind") != "zero_uniform_source":
            raise SpectroscopyError(
                "linear kick spectroscopy requires a zero continuous electromagnetic source"
            )
        if "records" not in handle["events"]:
            raise SpectroscopyError("kick spectrum requires one stored exact event")
        records = handle["events/records"]
        assert isinstance(records, h5py.Group)
        candidates = [records[name] for name in sorted(records.keys())]
        if len(candidates) != 1:
            raise SpectroscopyError(
                "linear kick spectroscopy requires a trajectory with exactly one event"
            )
        if event_id is None:
            event = candidates[0]
        else:
            selected = [
                item for item in candidates if str(item.attrs.get("event_id", "")) == event_id
            ]
            if len(selected) != 1:
                raise SpectroscopyError(f"trajectory has no unique event {event_id!r}")
            event = selected[0]
        assert isinstance(event, h5py.Group)
        selected_event_id = str(event.attrs["event_id"])
        event_step = int(event.attrs["step"])
        impulse = np.asarray(event["impulse_au"][...], dtype=np.float64)
        pre_path = "pre/observables/" + dipole_id.replace(".", "/") + "/values"
        if pre_path not in event:
            raise SpectroscopyError("event record lacks the selected pre-event dipole")
        event_pre = np.asarray(event[pre_path][...], dtype=np.float64)

    dipole_series = source_trajectory.read_observable(dipole_id)
    current_series = source_trajectory.read_observable(current_id)
    dipole_keep = dipole_series.steps >= event_step
    current_keep = current_series.steps >= event_step
    steps = dipole_series.steps[dipole_keep]
    times = dipole_series.times_au[dipole_keep]
    dipoles = dipole_series.values[dipole_keep]
    if not np.array_equal(steps, np.arange(event_step, source_trajectory.final_step + 1)):
        raise SpectroscopyError("kick spectrum requires dipole sampling at every post-event step")
    if not np.array_equal(current_series.steps[current_keep], steps) or not np.array_equal(
        current_series.times_au[current_keep], times
    ):
        raise SpectroscopyError("primary current must share the complete post-event dipole grid")
    if config.baseline is BaselineKind.EVENT_PRE:
        baseline = event_pre
    elif config.baseline is BaselineKind.FIRST_SAMPLE:
        baseline = np.asarray(dipoles[0], dtype=np.float64)
    elif config.baseline is BaselineKind.NONE:
        baseline = np.zeros(3, dtype=np.float64)
    else:  # pragma: no cover - closed enum
        raise SpectroscopyError(f"unsupported baseline {config.baseline.value!r}")
    return compute_kick_spectrum(
        time_au=times,
        dipole_au=dipoles,
        current_au=current_series.values[current_keep],
        impulse_au=impulse,
        baseline_dipole_au=baseline,
        config=config,
        source=SpectrumSourceLink(
            kind="aion_trajectory",
            path=source_trajectory.path.resolve(),
            sha256=source_trajectory.sha256,
            identifier=source_trajectory.run_id,
            description="completed Aion trajectory",
            metadata_json=json.dumps(
                {
                    "schema": "aion.native-trajectory-source",
                    "version": "1.0.0",
                    "simulation_id": source_trajectory.simulation_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
        event_id=selected_event_id,
        dipole_definition_id=dipole_id,
        current_definition_id=current_id,
    )
