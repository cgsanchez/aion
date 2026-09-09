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


class SourceCompilationError(AionError, ValueError):
    """An electromagnetic source violates its compilation contract."""


class FormulationError(AionError, RuntimeError):
    """A formulation input or instantaneous identity is invalid."""
