"""Stage-B adapter from qualified exact Wilson matrices to Aion's EOM triple."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aion.backends import ArrayBackend
from aion.electromagnetism import build_magnetic_pair_geometry
from aion.electronic_structure.ao_quadrature import AOQuadrature
from aion.electronic_structure.electric_matrices import evaluate_uniform_electric_e1_tensor
from aion.electronic_structure.magnetic_matrices import (
    MagneticOneElectronFirstDerivatives,
    evaluate_magnetic_one_electron_first_derivatives,
)
from aion.electronic_structure.time_connection import ExactWilsonOneElectronSample
from aion.formulations.types import EOMTriple


def exact_wilson_one_electron_triple(
    sample: ExactWilsonOneElectronSample,
) -> EOMTriple:
    """Expose one exact sample as ``(S, K, omega_t)`` without rebuilding it."""

    if not isinstance(sample, ExactWilsonOneElectronSample):
        raise TypeError("sample must be an ExactWilsonOneElectronSample")
    return EOMTriple(
        metric=sample.metric,
        hamiltonian_eom=sample.mechanical,
        connection=sample.connection.connection,
    )


@dataclass(frozen=True, slots=True)
class ExactOneElectronModelContext:
    """Source-independent tensors shared by every first-order model sample."""

    backend: ArrayBackend
    derivatives: MagneticOneElectronFirstDerivatives
    central_dipoles: Any
    ao_anchor_coordinates_au: Any
    pair_midpoints_au: Any
    charge: float
    mass: float
    hbar: float


@dataclass(frozen=True, slots=True)
class OneElectronModelTriple:
    """One action-level EOM triple with its independently assembled metric rate."""

    triple: EOMTriple
    metric_dot: Any


@dataclass(frozen=True, slots=True)
class ExactOneElectronModelTriples:
    """Exact parent and named P0/E1/B1/C1 comparison levels."""

    exact: OneElectronModelTriple
    p0: OneElectronModelTriple
    e1: OneElectronModelTriple
    geometric_b1: OneElectronModelTriple
    full_b1: OneElectronModelTriple
    complete_first_order: OneElectronModelTriple


def prepare_exact_one_electron_model_context(
    quadrature: AOQuadrature,
    *,
    charge: float = -1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    memory_budget_bytes: int | None = None,
) -> ExactOneElectronModelContext:
    """Precompute the analytic E1 and zero-field magnetic response tensors."""

    if not isinstance(quadrature, AOQuadrature):
        raise TypeError("quadrature must be an AOQuadrature")
    derivatives = evaluate_magnetic_one_electron_first_derivatives(
        quadrature,
        charge=charge,
        mass=mass,
        hbar=hbar,
        memory_budget_bytes=memory_budget_bytes,
    )
    electric = evaluate_uniform_electric_e1_tensor(
        quadrature,
        charge=charge,
        hbar=hbar,
    )
    reference = quadrature.reference
    geometry = build_magnetic_pair_geometry(
        reference.core_operators.nuclei.coordinates_au,
        reference.anchor_topology.ao_to_atom,
        quadrature.backend,
    )
    return ExactOneElectronModelContext(
        backend=quadrature.backend,
        derivatives=derivatives,
        central_dipoles=electric.central_dipoles,
        ao_anchor_coordinates_au=geometry.ao_anchor_coordinates_au,
        pair_midpoints_au=geometry.pair_midpoints_au,
        charge=float(charge),
        mass=float(mass),
        hbar=float(hbar),
    )


def exact_one_electron_model_triples(
    sample: ExactWilsonOneElectronSample,
    context: ExactOneElectronModelContext,
) -> ExactOneElectronModelTriples:
    r"""Assemble exact, P0, E1, gB1, B1, and complete-first-order triples.

    Every row is assembled from one action level.  In particular, E1 enters
    the temporal connection only, the B1 metric carries its complete time
    derivative, and C1 contains both electric and magnetic first-order
    partners.
    """

    if not isinstance(sample, ExactWilsonOneElectronSample):
        raise TypeError("sample must be an ExactWilsonOneElectronSample")
    if not isinstance(context, ExactOneElectronModelContext):
        raise TypeError("context must be an ExactOneElectronModelContext")
    if sample.static_result.reference_fingerprint_sha256 != (
        context.derivatives.reference_fingerprint_sha256
    ):
        raise ValueError("sample and model context belong to different AO references")
    if sample.static_result.grid_fingerprint_sha256 != context.derivatives.grid_fingerprint_sha256:
        raise ValueError("sample and model context use different AO grids")
    if (
        sample.static_result.charge != context.charge
        or sample.static_result.mass != context.mass
        or sample.static_result.hbar != context.hbar
    ):
        raise ValueError("sample and model context use different particle conventions")
    backend = context.backend
    xp = backend.namespace
    field = backend.asarray(sample.source.field.magnetic_field_au, dtype=xp.float64)
    field_dot = backend.asarray(sample.source.magnetic_field_dot_au, dtype=xp.float64)
    theta = sample.static_result.endpoint_link
    theta_dot = sample.connection.endpoint_link_dot
    overlap0 = sample.static_result.overlap.zero
    mechanical0 = (
        sample.static_result.kinetic.zero
        + sample.static_result.nuclear_attraction.zero
    )
    overlap_first = xp.einsum(
        "x,xmn->mn", field, context.derivatives.metric, optimize=True
    )
    overlap_first_dot = xp.einsum(
        "x,xmn->mn", field_dot, context.derivatives.metric, optimize=True
    )
    mechanical_first = xp.einsum(
        "x,xmn->mn", field, context.derivatives.mechanical, optimize=True
    )
    metric_p0 = theta * overlap0
    metric_p0_dot = theta_dot * overlap0
    metric_b1 = theta * (overlap0 + overlap_first)
    metric_b1_dot = theta_dot * (overlap0 + overlap_first) + theta * overlap_first_dot
    mechanical_p0 = theta * mechanical0
    mechanical_b1 = theta * (mechanical0 + mechanical_first)

    scalar_at_anchors = sample.source.scalar_potential(
        context.ao_anchor_coordinates_au,
        backend,
    )
    sigma = (1j * context.charge / context.hbar) * scalar_at_anchors
    electric_at_pairs = sample.source.electric_field(context.pair_midpoints_au, backend)
    electric_eta = (-1j / context.hbar) * theta * xp.einsum(
        "mnx,xmn->mn",
        electric_at_pairs,
        context.central_dipoles,
        optimize=True,
    )
    connection_p0 = _compatible_connection(metric_p0, metric_p0_dot, sigma)
    connection_b1 = _compatible_connection(metric_b1, metric_b1_dot, sigma)

    exact = OneElectronModelTriple(
        triple=exact_wilson_one_electron_triple(sample),
        metric_dot=sample.connection.metric_dot,
    )
    p0 = OneElectronModelTriple(
        triple=EOMTriple(metric_p0, mechanical_p0, connection_p0),
        metric_dot=metric_p0_dot,
    )
    e1 = OneElectronModelTriple(
        triple=EOMTriple(metric_p0, mechanical_p0, connection_p0 + electric_eta),
        metric_dot=metric_p0_dot,
    )
    geometric_b1 = OneElectronModelTriple(
        triple=EOMTriple(metric_b1, mechanical_p0, connection_b1),
        metric_dot=metric_b1_dot,
    )
    full_b1 = OneElectronModelTriple(
        triple=EOMTriple(metric_b1, mechanical_b1, connection_b1),
        metric_dot=metric_b1_dot,
    )
    complete = OneElectronModelTriple(
        triple=EOMTriple(metric_b1, mechanical_b1, connection_b1 + electric_eta),
        metric_dot=metric_b1_dot,
    )
    return ExactOneElectronModelTriples(
        exact=exact,
        p0=p0,
        e1=e1,
        geometric_b1=geometric_b1,
        full_b1=full_b1,
        complete_first_order=complete,
    )


def _compatible_connection(metric: Any, metric_dot: Any, sigma: Any) -> Any:
    """Return ``S Sigma + 1/2(D_t S)`` with no internal residual."""

    return metric * sigma[None, :] + 0.5 * (
        metric_dot + sigma[:, None] * metric - metric * sigma[None, :]
    )
