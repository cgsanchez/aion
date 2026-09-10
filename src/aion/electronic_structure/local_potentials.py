"""Pointwise local-potential providers for magnetic scalar hierarchies."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from aion.backends import ArrayBackend
from aion.config import AtomicUnit, NuclearModel, PhysicalDimension, canonical_sha256
from aion.electronic_structure.data import PreparedReference
from aion.errors import ConfigurationError, UnsupportedConfigurationError

_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class LocalPotentialIdentity:
    """Stable semantic identity, units, and provenance for a local potential."""

    name: str
    version: str
    provenance: tuple[tuple[str, str], ...]
    unit: AtomicUnit = AtomicUnit.ENERGY
    physical_dimension: PhysicalDimension = PhysicalDimension.ENERGY
    fingerprint_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _IDENTIFIER.fullmatch(self.name):
            raise ConfigurationError("local-potential name is not a stable identifier")
        if not isinstance(self.version, str) or not self.version:
            raise ConfigurationError("local-potential version must be nonempty")
        normalized: list[tuple[str, str]] = []
        for item in self.provenance:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not all(isinstance(value, str) and value for value in item)
            ):
                raise ConfigurationError("local-potential provenance must contain string pairs")
            normalized.append(item)
        normalized_tuple = tuple(sorted(normalized))
        if len({key for key, _ in normalized_tuple}) != len(normalized_tuple):
            raise ConfigurationError("local-potential provenance keys must be unique")
        if (
            self.unit is not AtomicUnit.ENERGY
            or self.physical_dimension is not PhysicalDimension.ENERGY
        ):
            raise UnsupportedConfigurationError(
                "initial local-potential providers must return pointwise Hartree energies"
            )
        object.__setattr__(self, "provenance", normalized_tuple)
        object.__setattr__(
            self,
            "fingerprint_sha256",
            canonical_sha256(
                {
                    "schema": "aion.local-potential-provider",
                    "version": "1.0.0",
                    "name": self.name,
                    "provider_version": self.version,
                    "provenance": normalized_tuple,
                    "unit": self.unit.value,
                    "physical_dimension": self.physical_dimension.value,
                }
            ),
        )


@runtime_checkable
class BoundLocalPotential(Protocol):
    """A pointwise provider bound to one reference and array backend."""

    @property
    def identity(self) -> LocalPotentialIdentity: ...

    @property
    def zero_matrix_au(self) -> Any: ...

    def values_au(self, coordinates_au: Any) -> Any: ...


@runtime_checkable
class LocalPotentialProvider(Protocol):
    """Provider factory with a stable semantic identity."""

    @property
    def identity(self) -> LocalPotentialIdentity: ...

    def bind(
        self, reference: PreparedReference, backend: ArrayBackend
    ) -> BoundLocalPotential: ...


@dataclass(frozen=True, slots=True)
class NuclearAttractionProvider:
    """All-electron local nuclear attraction from authenticated nuclei."""

    identity: LocalPotentialIdentity = field(
        default_factory=lambda: LocalPotentialIdentity(
            name="nuclear_attraction",
            version="1.0.0",
            provenance=(("model", "all_electron_local_coulomb"),),
        )
    )

    def bind(
        self, reference: PreparedReference, backend: ArrayBackend
    ) -> BoundLocalPotential:
        if (
            reference.config.electronic_structure.nuclear_model
            is not NuclearModel.ALL_ELECTRON_LOCAL
        ):
            raise UnsupportedConfigurationError(
                "nuclear-attraction provider supports only all-electron local nuclei"
            )
        return _BoundNuclearAttraction(
            identity=self.identity,
            backend=backend,
            coordinates_au=backend.asarray(
                reference.core_operators.nuclei.coordinates_au,
                dtype=backend.namespace.float64,
            ),
            charges=backend.asarray(
                reference.core_operators.nuclei.charges,
                dtype=backend.namespace.float64,
            ),
            zero_matrix_au=backend.asarray(
                reference.core_operators.nuclear_attraction,
                dtype=backend.namespace.complex128,
            ),
        )


@dataclass(frozen=True, slots=True)
class ScaledLocalPotentialProvider:
    """Scale another pointwise provider while preserving auditable identity."""

    provider: LocalPotentialProvider
    scale: float
    identity: LocalPotentialIdentity = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.provider, LocalPotentialProvider):
            raise TypeError("provider must satisfy LocalPotentialProvider")
        if isinstance(self.scale, bool):
            raise ConfigurationError("local-potential scale must be finite")
        scale = float(self.scale)
        if not np.isfinite(scale):
            raise ConfigurationError("local-potential scale must be finite")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(
            self,
            "identity",
            LocalPotentialIdentity(
                name=f"{self.provider.identity.name}.scaled",
                version="1.0.0",
                provenance=(
                    ("parent_fingerprint_sha256", self.provider.identity.fingerprint_sha256),
                    ("scale", scale.hex()),
                ),
            ),
        )

    def bind(
        self, reference: PreparedReference, backend: ArrayBackend
    ) -> BoundLocalPotential:
        return _BoundScaledPotential(
            identity=self.identity,
            parent=bind_local_potential(self.provider, reference, backend),
            scale=self.scale,
        )


@dataclass(frozen=True, slots=True)
class _BoundNuclearAttraction:
    identity: LocalPotentialIdentity
    backend: ArrayBackend
    coordinates_au: Any
    charges: Any
    zero_matrix_au: Any

    def __post_init__(self) -> None:
        for name in ("coordinates_au", "charges", "zero_matrix_au"):
            self.backend.assert_resident(getattr(self, name), name=f"nuclear {name}")

    def values_au(self, coordinates_au: Any) -> Any:
        self.backend.assert_resident(coordinates_au, name="local-potential coordinates")
        xp = self.backend.namespace
        distances = xp.linalg.norm(
            coordinates_au[:, None, :] - self.coordinates_au[None, :, :], axis=2
        )
        if self.backend.scalar_to_float(xp.any(distances == 0.0)):
            raise ConfigurationError(
                "quadrature grid contains a nuclear point where attraction is singular"
            )
        result = -xp.sum(self.charges[None, :] / distances, axis=1)
        self.backend.assert_resident(result, name="nuclear-attraction values")
        return result


@dataclass(frozen=True, slots=True)
class _BoundScaledPotential:
    identity: LocalPotentialIdentity
    parent: BoundLocalPotential
    scale: float

    @property
    def zero_matrix_au(self) -> Any:
        return self.scale * self.parent.zero_matrix_au

    def values_au(self, coordinates_au: Any) -> Any:
        return self.scale * self.parent.values_au(coordinates_au)


def bind_local_potential(
    provider: LocalPotentialProvider,
    reference: PreparedReference,
    backend: ArrayBackend,
) -> BoundLocalPotential:
    """Bind and validate a provider without accepting AO-matrix substitutes."""

    if not isinstance(provider, LocalPotentialProvider):
        raise UnsupportedConfigurationError(
            "local potential must implement identity and bind; preassembled AO matrices "
            "and nonlocal operators are unsupported"
        )
    if not isinstance(provider.identity, LocalPotentialIdentity):
        raise ConfigurationError("local-potential provider identity is invalid")
    bound = provider.bind(reference, backend)
    if not isinstance(bound, BoundLocalPotential):
        raise ConfigurationError("bound local potential does not satisfy its protocol")
    if bound.identity != provider.identity:
        raise ConfigurationError("bound local-potential identity changed during binding")
    backend.assert_resident(bound.zero_matrix_au, name="local-potential zero matrix")
    expected_shape = reference.core_operators.overlap.shape
    if bound.zero_matrix_au.shape != expected_shape:
        raise ConfigurationError(
            f"local-potential zero matrix must have shape {expected_shape}"
        )
    return bound
