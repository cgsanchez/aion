"""PySCF-backed Casida response with explicit polarized root selection."""

from __future__ import annotations

import numpy as np

from aion.config import BackendConfig
from aion.electronic_structure import PreparedReference, reconstruct_mean_field
from aion.errors import SpectroscopyError
from aion.spectroscopy.types import CasidaConfig, CasidaResult, ResonanceSelection


def run_casida(reference: PreparedReference, config: CasidaConfig) -> CasidaResult:
    """Compute singlet Casida-TDDFT roots from an immutable prepared reference.

    Linear response is reconstructed on the CPU because PySCF's Casida solver
    is the qualified implementation.  This does not rerun the ground-state SCF
    and does not change the preparation or real-time backend recorded by the
    reference.
    """

    if not isinstance(reference, PreparedReference):
        raise TypeError("reference must be PreparedReference")
    if not isinstance(config, CasidaConfig):
        raise TypeError("config must be CasidaConfig")
    try:
        mean_field = reconstruct_mean_field(reference, BackendConfig())
        response = mean_field.CasidaTDDFT()
        response.nstates = config.nstates
        response.conv_tol = config.convergence_tolerance
        response.singlet = config.singlet
        energies, _amplitudes = response.kernel()
        converged = np.asarray(response.converged, dtype=np.bool_)
        transition_dipole = np.asarray(response.transition_dipole())
        oscillator_strength = np.asarray(
            response.oscillator_strength(gauge="length", order=0),
            dtype=np.float64,
        )
    except Exception as exc:
        raise SpectroscopyError("PySCF Casida-TDDFT calculation failed") from exc
    if converged.shape != np.asarray(energies).shape or not np.all(converged):
        failed = np.flatnonzero(~converged).tolist()
        raise SpectroscopyError(f"Casida roots did not all converge; failed indices={failed}")
    if np.iscomplexobj(transition_dipole):
        imaginary = float(np.max(np.abs(transition_dipole.imag), initial=0.0))
        if imaginary > 1.0e-12:
            raise SpectroscopyError(
                f"Casida transition dipoles have unexpected imaginary part {imaginary:.3e}"
            )
        transition_dipole = transition_dipole.real
    transition_dipole = np.asarray(transition_dipole, dtype=np.float64)
    norms = np.linalg.norm(transition_dipole, axis=1)
    directions = np.zeros_like(transition_dipole)
    active = norms > 0.0
    directions[active] = transition_dipole[active] / norms[active, None]
    import pyscf

    return CasidaResult(
        config=config,
        reference_fingerprint_sha256=reference.fingerprint_sha256,
        method="PySCF CasidaTDDFT; adiabatic prepared-reference functional",
        pyscf_version=str(pyscf.__version__),
        excitation_energy_au=np.asarray(energies, dtype=np.float64),
        oscillator_strength=oscillator_strength,
        transition_dipole_au=transition_dipole,
        transition_direction=directions,
        converged=converged,
    )


def select_resonance(
    result: CasidaResult,
    polarization: object,
    *,
    brightness_threshold: float | None = None,
    root_index: int | None = None,
) -> ResonanceSelection:
    r"""Select a root by ``f_e=2 omega |e dot d|^2`` or explicit zero-based index."""

    if not isinstance(result, CasidaResult):
        raise TypeError("result must be CasidaResult")
    vector = np.asarray(polarization, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise SpectroscopyError("resonance polarization must be a finite Cartesian vector")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise SpectroscopyError("resonance polarization cannot be zero")
    unit = vector / norm
    projected = 2.0 * result.excitation_energy_au * np.square(result.transition_dipole_au @ unit)
    if (root_index is None) == (brightness_threshold is None):
        raise SpectroscopyError(
            "choose exactly one resonance policy: brightness_threshold or root_index"
        )
    if root_index is not None:
        if isinstance(root_index, bool) or not isinstance(root_index, int):
            raise SpectroscopyError("selected root index must be an integer")
        if not 0 <= root_index < result.root_count:
            raise SpectroscopyError("selected root index is outside the Casida result")
        selected = root_index
        threshold = None
        user_selected = True
    else:
        assert brightness_threshold is not None
        if (
            isinstance(brightness_threshold, bool)
            or not isinstance(brightness_threshold, int | float)
            or not np.isfinite(brightness_threshold)
            or brightness_threshold < 0.0
        ):
            raise SpectroscopyError("brightness threshold must be finite and nonnegative")
        candidates = np.flatnonzero(projected >= float(brightness_threshold))
        if candidates.size == 0:
            raise SpectroscopyError("no Casida root meets the polarized brightness threshold")
        selected = int(candidates[0])
        threshold = float(brightness_threshold)
        user_selected = False
    if not bool(result.converged[selected]):
        raise SpectroscopyError("the selected Casida root is not converged")
    return ResonanceSelection(
        casida_result_id=result.result_id,
        polarization=unit,
        projected_oscillator_strength=projected,
        selected_root=selected,
        brightness_threshold=threshold,
        user_selected=user_selected,
    )
