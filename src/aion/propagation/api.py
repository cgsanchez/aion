"""Construction boundary for initial orbital states and WP4 propagators."""

from __future__ import annotations

import numpy as np

from aion.backends import Workspace
from aion.config import (
    FormulationKind,
    GaugeRepresentation,
    IntegratorKind,
    PropagationConfig,
)
from aion.errors import PropagationError
from aion.formulations import Formulation, p0_geometry
from aion.propagation.linalg import metric_roundoff_limit, orbital_metric_residual
from aion.propagation.scem import PropagationScratch, SCEMPropagator
from aion.propagation.transports import ConnectionAwareTransport, FixedMetricTransport
from aion.propagation.types import OrbitalState


def build_initial_orbital_state(formulation: Formulation, workspace: Workspace) -> OrbitalState:
    """Select occupied reference orbitals and dress the covariant initial fiber."""

    if formulation.context.workspace is not workspace:
        raise PropagationError("formulation and workspace do not belong to one simulation")
    reference = formulation.context.reference
    occupied_indices_host = np.flatnonzero(reference.ground_state.occupations > 0.0)
    if occupied_indices_host.size == 0:
        raise PropagationError("reference contains no occupied orbitals")
    backend = workspace.backend
    xp = backend.namespace
    occupied_indices = backend.asarray(occupied_indices_host, dtype=xp.int64)
    coefficients = xp.take(
        workspace.require("ground_state.coefficients"), occupied_indices, axis=1
    ).astype(xp.complex128, copy=False)
    occupations = xp.take(
        workspace.require("ground_state.occupations"), occupied_indices, axis=0
    ).astype(xp.float64, copy=False)
    if formulation.kind in {FormulationKind.P0, FormulationKind.P0_E1}:
        pair_link = workspace.require(f"source.{formulation.gauge.value}_endpoint.pair_link")[0]
        source_node = workspace.require(
            f"source.{formulation.gauge.value}_endpoint.node_scalar_potential"
        )[0]
        geometry = p0_geometry(
            workspace.require("operators.overlap"),
            workspace.require("anchors.ao_to_atom"),
            formulation.context.pairs,
            source_node,
            pair_link,
            workspace.require(f"source.{formulation.gauge.value}_endpoint.pair_link_dot")[0],
            natom=formulation.context.natom,
            charge=formulation.context.charge,
            hbar=formulation.context.hbar,
            backend=backend,
        )
        # Theta is a diagonal congruence for the uniform source.  Recover one
        # representative phase from the physical vector potential so P=Theta*P0.
        vector = workspace.require("source.endpoint.vector_potential_reduced")[0]
        coordinates = workspace.require("nuclei.coordinates_au")
        origin = xp.asarray(
            reference.config.molecule.electromagnetic_origin.position_au,
            dtype=xp.float64,
        )
        atom_phase = xp.exp(
            (1j * formulation.context.charge / formulation.context.hbar)
            * ((coordinates - origin[None, :]) @ vector)
        )
        if formulation.gauge is GaugeRepresentation.VELOCITY:
            coefficients = atom_phase[workspace.require("anchors.ao_to_atom"), None] * coefficients
        residual = orbital_metric_residual(coefficients, geometry.metric, backend)
    else:
        residual = orbital_metric_residual(
            coefficients, workspace.require("operators.overlap"), backend
        )
    limit = metric_roundoff_limit(coefficients.shape[0])
    if residual > limit:
        raise PropagationError(
            f"initial occupied orbitals violate their formulation metric: {residual:.3e}"
        )
    return OrbitalState(coefficients, occupations, backend, 0)


def build_propagator(
    formulation: Formulation,
    config: PropagationConfig,
    workspace: Workspace,
    *,
    diagnostic_disable_metric_correction: bool = False,
) -> SCEMPropagator:
    """Bind the common nonlinear engine to the required transport geometry."""

    if not isinstance(config, PropagationConfig):
        raise TypeError("config must be PropagationConfig")
    if formulation.context.workspace is not workspace:
        raise PropagationError("formulation and workspace do not belong to one simulation")
    fixed_kinds = {
        FormulationKind.BARE_LENGTH_GAUGE,
        FormulationKind.BARE_VELOCITY_GAUGE,
    }
    is_fixed = formulation.kind in fixed_kinds
    expected = (
        IntegratorKind.FIXED_METRIC_SCEM if is_fixed else IntegratorKind.CONNECTION_AWARE_SCEM
    )
    if config.integrator is not expected:
        raise PropagationError(f"{formulation.kind.value} requires {expected.value}")
    if diagnostic_disable_metric_correction and is_fixed:
        raise PropagationError(
            "metric-correction diagnostic mode applies only to connection-aware transport"
        )
    transport = (
        FixedMetricTransport(formulation, config.time_grid.step_au)
        if is_fixed
        else ConnectionAwareTransport(
            formulation,
            config.time_grid.step_au,
            apply_metric_correction=not diagnostic_disable_metric_correction,
        )
    )
    overlap = workspace.require("operators.overlap")
    occupied = int(np.count_nonzero(formulation.context.reference.ground_state.occupations > 0.0))
    scratch = PropagationScratch.create(
        workspace,
        nao=overlap.shape[0],
        norbital=occupied,
    )
    workspace.assert_all_resident()
    return SCEMPropagator(formulation, config, workspace, transport, scratch)
