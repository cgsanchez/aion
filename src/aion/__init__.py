"""Aion real-time electronic dynamics prototypes."""

from .backends import CPUBackend, CuPyBackend, make_backend
from .cn_tddft import (
    CNPropagationRecord,
    LengthGaugeCNRTTDDFT,
    MidpointConvergenceError,
    SCEMRunSummary,
    SCEMStepResult,
)
from .fields import ContinuousWave, GaussianPulse, Sin2Pulse
from .linear_response import (
    HARTREE_TO_EV,
    Excitation,
    casida_excitations,
    excitation_by_index,
    load_excitations,
    lowest_active_excitation,
    save_excitations,
    transition_polarization,
)
from .rt_tddft import LengthGaugeRTTDDFT, PropagationRecord
from .spectrum import KickSpectrum, kick_spectrum, write_kick_spectrum_csv

__all__ = [
    "CPUBackend",
    "CuPyBackend",
    "Excitation",
    "ContinuousWave",
    "GaussianPulse",
    "HARTREE_TO_EV",
    "KickSpectrum",
    "LengthGaugeCNRTTDDFT",
    "LengthGaugeRTTDDFT",
    "MidpointConvergenceError",
    "CNPropagationRecord",
    "PropagationRecord",
    "SCEMStepResult",
    "SCEMRunSummary",
    "Sin2Pulse",
    "casida_excitations",
    "excitation_by_index",
    "kick_spectrum",
    "load_excitations",
    "lowest_active_excitation",
    "make_backend",
    "save_excitations",
    "transition_polarization",
    "write_kick_spectrum_csv",
]
