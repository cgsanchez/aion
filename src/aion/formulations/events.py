"""Formulation-owned exact coefficient maps for impulsive electric events."""

from __future__ import annotations

from typing import Any

import numpy as np

from aion.config import FormulationKind, GaugeRepresentation
from aion.electronic_structure import hermitian_part
from aion.errors import FormulationError
from aion.formulations.base import FormulationContext
from aion.formulations.kernels import p0_geometry
from aion.formulations.types import FormulationSourceSample, InstantaneousEvaluation


def metric_unitary_impulse(
    coefficients: Any,
    metric: Any,
    integrated_hamiltonian: Any,
    *,
    context: FormulationContext,
) -> Any:
    """Apply ``exp[-i S^-1 H_int/hbar]`` in a Cholesky metric frame.

    Cholesky coordinates are used only to evaluate this isolated exact event
    map. This is neither a Löwdin construction nor part of continuous SCEM
    propagation.
    """

    backend = context.backend
    xp = context.namespace
    for name, value in (
        ("event coefficients", coefficients),
        ("event metric", metric),
        ("integrated event Hamiltonian", integrated_hamiltonian),
    ):
        backend.assert_resident(value, name=name)
    if metric.ndim != 2 or metric.shape[0] != metric.shape[1]:
        raise FormulationError("event metric must be square")
    if integrated_hamiltonian.shape != metric.shape or coefficients.shape[0] != metric.shape[0]:
        raise FormulationError("event map dimensions are inconsistent")
    try:
        upper = xp.linalg.cholesky(hermitian_part(metric)).conj().T
        left_solved = xp.linalg.solve(upper.conj().T, hermitian_part(integrated_hamiltonian))
        orthogonal_hamiltonian = xp.linalg.solve(upper.T, left_solved.T).T
        orthogonal_hamiltonian = hermitian_part(orthogonal_hamiltonian)
        eigenvalues, eigenvectors = xp.linalg.eigh(orthogonal_hamiltonian)
        phase = (eigenvectors * xp.exp((-1j / context.hbar) * eigenvalues)[None, :]) @ (
            eigenvectors.conj().T
        )
        return xp.linalg.solve(upper, phase @ (upper @ coefficients))
    except Exception as exc:
        raise FormulationError("exact metric-unitary kick map failed") from exc


def _covariant_dipole_operators(
    context: FormulationContext,
    evaluation: InstantaneousEvaluation,
    *,
    include_e1: bool,
) -> Any:
    xp = context.namespace
    metric = evaluation.triple.metric
    projectors = context.workspace.require("anchors.site_projector_diagonals")
    coordinates = context.workspace.require("nuclei.coordinates_au")
    origin = xp.asarray(
        context.reference.config.molecule.electromagnetic_origin.position_au,
        dtype=xp.float64,
    )
    relative = coordinates - origin[None, :]
    operators = xp.zeros((3, *metric.shape), dtype=xp.complex128)
    for axis in range(3):
        anchored_charge = context.charge * xp.einsum(
            "a,ai->i", relative[:, axis], projectors, optimize=True
        )
        operators[axis] = 0.5 * (
            anchored_charge[:, None] * metric + metric * anchored_charge[None, :]
        )
    if include_e1:
        if evaluation.dressed_central_dipoles is None:
            raise FormulationError("P0+E1 kick requires dressed central dipoles")
        operators = operators + evaluation.dressed_central_dipoles
    return operators


def apply_electric_kick(
    *,
    kind: FormulationKind,
    gauge: GaugeRepresentation,
    gauge_velocity_fraction: float,
    context: FormulationContext,
    coefficients: Any,
    impulse_au: tuple[float, float, float],
    evaluation_before: InstantaneousEvaluation,
    source_before: FormulationSourceSample,
    source_after: FormulationSourceSample,
) -> Any:
    """Return post-event coefficients for one exact electric impulse."""

    context.validate_source(source_before, gauge, gauge_velocity_fraction)
    context.validate_source(source_after, gauge, gauge_velocity_fraction)
    impulse_host = np.asarray(impulse_au, dtype=np.float64)
    if impulse_host.shape != (3,) or not np.all(np.isfinite(impulse_host)):
        raise FormulationError("kick impulse must be a finite Cartesian vector")
    xp = context.namespace
    impulse = context.backend.asarray(impulse_host, dtype=xp.float64)

    if kind is FormulationKind.BARE_VELOCITY_GAUGE:
        jump = source_after.vector_potential_reduced - source_before.vector_potential_reduced
        error = context.backend.scalar_to_float(xp.linalg.norm(jump + impulse))
        scale = max(1.0, float(np.linalg.norm(impulse_host)))
        if error > 64.0 * np.finfo(np.float64).eps * scale:
            raise FormulationError("bare-VG kick source does not satisfy Delta a=-impulse")
        return xp.array(coefficients, dtype=xp.complex128, copy=True)

    if kind is FormulationKind.BARE_LENGTH_GAUGE:
        dipoles = context.charge * context.relative_position_operators(
            evaluation_before.triple.metric
        )
    elif kind in {FormulationKind.P0, FormulationKind.P0_E1}:
        dipoles = _covariant_dipole_operators(
            context,
            evaluation_before,
            include_e1=kind is FormulationKind.P0_E1,
        )
    else:  # pragma: no cover - closed enum
        raise FormulationError(f"unsupported kick formulation {kind!r}")

    integrated = hermitian_part(-xp.einsum("x,xij->ij", impulse, dipoles, optimize=True))
    kicked = metric_unitary_impulse(
        coefficients,
        evaluation_before.triple.metric,
        integrated,
        context=context,
    )
    if kind in {FormulationKind.P0, FormulationKind.P0_E1}:
        delta = source_after.vector_potential_reduced - source_before.vector_potential_reduced
        coordinates = context.workspace.require("nuclei.coordinates_au")
        origin = xp.asarray(
            context.reference.config.molecule.electromagnetic_origin.position_au,
            dtype=xp.float64,
        )
        atom_phase = xp.exp(
            (1j * context.charge / context.hbar) * ((coordinates - origin[None, :]) @ delta)
        )
        kicked = atom_phase[context.workspace.require("anchors.ao_to_atom"), None] * kicked

        post_geometry = p0_geometry(
            context.workspace.require("operators.overlap"),
            context.workspace.require("anchors.ao_to_atom"),
            context.pairs,
            source_after.node_scalar_potential,
            source_after.pair_link,
            source_after.pair_link_dot,
            natom=context.natom,
            charge=context.charge,
            hbar=context.hbar,
            backend=context.backend,
        )
        gram = kicked.conj().T @ post_geometry.metric @ kicked
        identity = xp.eye(gram.shape[0], dtype=xp.complex128)
        residual = context.backend.scalar_to_float(
            xp.linalg.norm(gram - identity) / xp.maximum(xp.asarray(1.0), xp.linalg.norm(identity))
        )
        limit = max(
            1.0e-12,
            8192.0 * np.finfo(np.float64).eps * post_geometry.metric.shape[0],
        )
        if residual > limit:
            raise FormulationError(
                f"covariant kick violates the post-event metric: residual={residual:.3e}"
            )
    return kicked
