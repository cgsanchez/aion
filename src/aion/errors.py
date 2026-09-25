"""Stable exception hierarchy for Aion's public boundaries."""


class AionError(Exception):
    """Base class for errors intentionally exposed by Aion."""


class ConfigurationError(AionError, ValueError):
    """A configuration is malformed, ambiguous, or internally inconsistent."""


class UnknownConfigurationFieldError(ConfigurationError):
    """A configuration contains a field that Aion cannot consume."""


class UnsupportedConfigurationError(ConfigurationError):
    """A valid configuration requests physics outside the validated domain."""


class SchemaError(AionError, ValueError):
    """A persistent artifact does not satisfy its declared schema."""


class IncompatibleSchemaVersionError(SchemaError):
    """An artifact uses a schema major version this reader cannot interpret."""


class IncompleteArtifactError(SchemaError):
    """An operation requiring a completed artifact received an incomplete one."""


class FeatureNotImplementedError(AionError, NotImplementedError):
    """A stable contract exists but its numerical implementation is a later milestone."""


class BackendError(AionError, RuntimeError):
    """An execution backend cannot satisfy its declared numerical contract."""


class DeviceResidencyError(BackendError):
    """An array is not resident on the backend/device where it is required."""


class ReferencePreparationError(AionError, RuntimeError):
    """A static electronic reference could not be prepared or authenticated."""


class WilsonStateError(AionError, RuntimeError):
    """A Wilson stationary state is inconsistent or cannot be authenticated."""


class SourceCompilationError(AionError, ValueError):
    """An electromagnetic source violates its compilation contract."""


class FormulationError(AionError, RuntimeError):
    """A formulation input or instantaneous identity is invalid."""


class PropagationError(AionError, RuntimeError):
    """A propagation state or numerical step violates its declared contract."""


class StateIntegrityError(PropagationError):
    """A dynamic electronic state is non-finite or exceeds a cleanup threshold."""


class MetricConstraintError(PropagationError):
    """A metric factorization or required cross-metric constraint failed."""


class MidpointConvergenceError(PropagationError):
    """The self-consistent midpoint density did not converge."""

    def __init__(
        self,
        message: str,
        *,
        step_index: int,
        iterations: int,
        density_residual: float,
        hamiltonian_residual: float,
    ) -> None:
        super().__init__(message)
        self.step_index = step_index
        self.iterations = iterations
        self.density_residual = density_residual
        self.hamiltonian_residual = hamiltonian_residual


class RunnerError(AionError, RuntimeError):
    """A production run cannot be constructed, advanced, or published safely."""


class RunCancelledError(RunnerError):
    """A run stopped at an accepted boundary after a cancellation request."""


class TrajectoryError(AionError, RuntimeError):
    """A trajectory is incomplete, inconsistent, or cannot be published."""


class CheckpointError(AionError, RuntimeError):
    """A checkpoint cannot authenticate or reconstruct its continuation state."""


class SpectroscopyError(AionError, RuntimeError):
    """A linear-response or real-time spectrum violates its declared contract."""


class MagneticBenchmarkError(AionError, RuntimeError):
    """A static magnetic matrix benchmark violates its declared contract."""
