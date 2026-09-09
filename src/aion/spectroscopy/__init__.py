"""Reusable, formulation-neutral molecular spectroscopy workflows."""

from aion.spectroscopy.casida import run_casida, select_resonance
from aion.spectroscopy.io import (
    load_casida_result,
    load_kick_spectrum,
    save_casida_result,
    save_kick_spectrum,
)
from aion.spectroscopy.transforms import (
    compute_kick_spectrum,
    kick_spectrum_from_trajectory,
    positive_frequency_transform,
)
from aion.spectroscopy.types import (
    BaselineKind,
    CasidaConfig,
    CasidaResult,
    FourierTransformResult,
    KickSpectrum,
    KickSpectrumConfig,
    QuadratureKind,
    ResonanceSelection,
    SpectrumSourceLink,
    TransformConfig,
    WindowKind,
)

__all__ = [
    "BaselineKind",
    "CasidaConfig",
    "CasidaResult",
    "FourierTransformResult",
    "KickSpectrum",
    "KickSpectrumConfig",
    "QuadratureKind",
    "ResonanceSelection",
    "SpectrumSourceLink",
    "TransformConfig",
    "WindowKind",
    "compute_kick_spectrum",
    "kick_spectrum_from_trajectory",
    "load_casida_result",
    "load_kick_spectrum",
    "positive_frequency_transform",
    "run_casida",
    "save_casida_result",
    "save_kick_spectrum",
    "select_resonance",
]
