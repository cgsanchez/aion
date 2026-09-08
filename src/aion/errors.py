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
