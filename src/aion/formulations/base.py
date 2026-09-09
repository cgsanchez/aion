"""Shared formulation context and scalar/vector contractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aion.backends import Workspace
from aion.config import GaugeRepresentation
from aion.electronic_structure import (
    AdiabaticPureRKS,
    PreparedReference,
    build_e1_operators,
    expectation,
)
from aion.errors import FormulationError
from aion.formulations.types import AODensity, FormulationSourceSample


@dataclass(frozen=True, slots=True)
class FormulationContext:
    """Immutable physics binding; mutable caches remain owned by ``workspace``."""

    reference: PreparedReference
    workspace: Workspace
    electronic_model: AdiabaticPureRKS
    charge: float = -1.0
    mass: float = 1.0
    hbar: float = 1.0

    @classmethod
    def create(
        cls,
        reference: PreparedReference,
        workspace: Workspace,
        *,
        charge: float = -1.0,
        mass: float = 1.0,
        hbar: float = 1.0,
    ) -> FormulationContext:
        if workspace.caches.get("reference_fingerprint_sha256") != (reference.fingerprint_sha256):
            raise FormulationError("workspace does not belong to the prepared reference")
        if not all(np.isfinite(value) for value in (charge, mass, hbar)):
            raise FormulationError("charge, mass, and hbar must be finite")
        if mass <= 0.0 or hbar <= 0.0:
            raise FormulationError("mass and hbar must be positive")
        return cls(
            reference=reference,
            workspace=workspace,
            electronic_model=AdiabaticPureRKS(workspace),
            charge=float(charge),
            mass=float(mass),
            hbar=float(hbar),
        )

    @property
    def backend(self) -> Any:
        return self.workspace.backend

    @property
    def namespace(self) -> Any:
        return self.backend.namespace

    @property
    def natom(self) -> int:
        return self.reference.anchor_topology.natom

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(pair[0]), int(pair[1])) for pair in self.reference.anchor_topology.pair_indices
        )

    def require_density(self, density: AODensity) -> Any:
        if density.backend is not self.backend:
            raise FormulationError("AO density belongs to a different backend")
        expected = self.workspace.require("operators.overlap").shape
        if density.matrix.shape != expected:
            raise FormulationError(
                f"AO density has shape {density.matrix.shape}; expected {expected}"
            )
        return density.matrix

    def validate_source(self, source: FormulationSourceSample, gauge: GaugeRepresentation) -> None:
        source.assert_resident(self.backend)
        if source.gauge is not gauge:
            raise FormulationError(
                f"source is in {source.gauge.value} gauge; formulation requires {gauge.value}"
            )
        if source.node_scalar_potential.shape != (self.natom,):
            raise FormulationError("source node-scalar-potential shape is inconsistent")
        if source.pair_link.shape != (len(self.pairs),):
            raise FormulationError("source pair topology is inconsistent")

    def relative_position_operators(self, metric: Any | None = None) -> Any:
        xp = self.namespace
        overlap = self.workspace.require("operators.overlap") if metric is None else metric
        origin = xp.asarray(
            self.reference.config.molecule.electromagnetic_origin.position_au,
            dtype=xp.float64,
        )
        return self.workspace.require("operators.position") - origin[:, None, None] * overlap

    def bare_electronic_dipole(self, density: Any) -> Any:
        xp = self.namespace
        return self.charge * xp.real(
            xp.einsum(
                "ij,xji->x",
                density,
                self.relative_position_operators(),
                optimize=True,
            )
        )

    def fixed_nuclear_dipole(self) -> Any:
        xp = self.namespace
        coordinates = self.workspace.require("nuclei.coordinates_au")
        charges = self.workspace.require("nuclei.charges")
        origin = xp.asarray(
            self.reference.config.molecule.electromagnetic_origin.position_au,
            dtype=xp.float64,
        )
        return xp.einsum("a,ax->x", charges, coordinates - origin[None, :], optimize=True)

    def central_dipoles(self) -> Any:
        name = "derived.e1.central_dipoles"
        if name not in self.workspace.arrays:
            bundle = build_e1_operators(
                self.reference.core_operators,
                self.reference.anchor_topology,
                charge=self.charge,
            )
            self.workspace.install_host_array(name, bundle.central_dipoles)
            self.workspace.caches["derived.e1.fingerprint_sha256"] = bundle.fingerprint_sha256
        return self.workspace.require(name)

    def electron_count(self, density: Any, metric: Any) -> Any:
        return expectation(density, metric, self.namespace)

    def zero_scalar(self) -> Any:
        return self.namespace.asarray(0.0, dtype=self.namespace.float64)

    def zero_vector(self) -> Any:
        return self.namespace.zeros((3,), dtype=self.namespace.float64)
