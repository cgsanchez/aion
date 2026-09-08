"""Independent semantic versions for persistent artifact schemas."""

from __future__ import annotations

import re
from dataclasses import dataclass

from aion.errors import SchemaError

_SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


@dataclass(frozen=True, slots=True, order=True)
class SchemaVersion:
    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        for name in ("major", "minor", "patch"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchemaError(f"schema-version {name} must be a nonnegative integer")

    @classmethod
    def parse(cls, text: str) -> SchemaVersion:
        match = _SEMVER_RE.fullmatch(text)
        if match is None:
            raise SchemaError(f"invalid semantic schema version {text!r}")
        return cls(*(int(value) for value in match.groups()))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def can_read(self, artifact_version: SchemaVersion) -> bool:
        """Whether this reader accepts an artifact version without migration."""

        return self.major == artifact_version.major and artifact_version.minor <= self.minor
