"""State and self-consistent propagation algorithms."""

from aion.propagation.api import build_initial_orbital_state, build_propagator
from aion.propagation.linalg import (
    cross_metric_residual,
    density_from_orbitals,
    hamiltonian_residual,
    hermitian_cleanup,
    metric_density_residual,
    metric_roundoff_limit,
    orbital_metric_residual,
    pull_lower_matrix,
    rational_map,
    relative_frobenius,
    right_cholesky_metric_link,
)
from aion.propagation.linear import (
    LinearMatrixHistory,
    LinearPropagationHistory,
    LinearStepDiagnostics,
    generalized_spectral_trajectory,
    propagate_linear_matrix_history,
)
from aion.propagation.scem import DampedPicardController, PropagationScratch, SCEMPropagator
from aion.propagation.transports import (
    ConnectionAwareTransport,
    FixedMetricTransport,
    PreparedConnectionAwareTransport,
    PreparedFixedMetricTransport,
    StepSourceSamples,
    TransportApplication,
    TransportTarget,
)
from aion.propagation.types import (
    ElectronicState,
    LinkDiagnostics,
    OrbitalState,
    PropagationStepResult,
    SCEMStepDiagnostics,
)

__all__ = [
    "ConnectionAwareTransport",
    "DampedPicardController",
    "ElectronicState",
    "FixedMetricTransport",
    "LinearMatrixHistory",
    "LinearPropagationHistory",
    "LinearStepDiagnostics",
    "LinkDiagnostics",
    "OrbitalState",
    "PreparedConnectionAwareTransport",
    "PreparedFixedMetricTransport",
    "PropagationScratch",
    "PropagationStepResult",
    "SCEMPropagator",
    "SCEMStepDiagnostics",
    "StepSourceSamples",
    "TransportApplication",
    "TransportTarget",
    "build_initial_orbital_state",
    "build_propagator",
    "cross_metric_residual",
    "density_from_orbitals",
    "generalized_spectral_trajectory",
    "hamiltonian_residual",
    "hermitian_cleanup",
    "metric_density_residual",
    "metric_roundoff_limit",
    "orbital_metric_residual",
    "propagate_linear_matrix_history",
    "pull_lower_matrix",
    "rational_map",
    "relative_frobenius",
    "right_cholesky_metric_link",
]
