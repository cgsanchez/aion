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
from .gauge import (
    AOAnchors,
    PeierlsGeometry,
    UniformElectricGauge,
    UniformMagneticGauge,
)
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
from .matrix_models import (
    LinearOneBodyModel,
    SiteHubbardModel,
    density_from_coefficients,
    site_populations,
)
from .rt_tddft import LengthGaugeRTTDDFT, PropagationRecord
from .spectrum import KickSpectrum, kick_spectrum, write_kick_spectrum_csv
from .variable_metric import (
    VariableMetricConvergenceError,
    VariableMetricSCEM,
    VariableMetricSCEMStepResult,
)

__all__ = [
    "AOAnchors",
    "CPUBackend",
    "CuPyBackend",
    "Excitation",
    "ContinuousWave",
    "GaussianPulse",
    "HARTREE_TO_EV",
    "KickSpectrum",
    "LengthGaugeCNRTTDDFT",
    "LengthGaugeRTTDDFT",
    "LinearOneBodyModel",
    "MidpointConvergenceError",
    "PeierlsGeometry",
    "CNPropagationRecord",
    "PropagationRecord",
    "SCEMStepResult",
    "SCEMRunSummary",
    "SiteHubbardModel",
    "Sin2Pulse",
    "UniformElectricGauge",
    "UniformMagneticGauge",
    "VariableMetricConvergenceError",
    "VariableMetricSCEM",
    "VariableMetricSCEMStepResult",
    "casida_excitations",
    "density_from_coefficients",
    "excitation_by_index",
    "kick_spectrum",
    "load_excitations",
    "lowest_active_excitation",
    "make_backend",
    "save_excitations",
    "site_populations",
    "transition_polarization",
    "write_kick_spectrum_csv",
]
