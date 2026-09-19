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
from aion.electronic_structure.time_connection import (
    ExactMagneticFieldSourceDirection,
    ExactWilsonOneElectronSample,
)
from aion.errors import FormulationError
from aion.formulations.action import OneElectronActionMatrixDirection
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
    mechanical0 = sample.static_result.kinetic.zero + sample.static_result.nuclear_attraction.zero
    overlap_first = xp.einsum("x,xmn->mn", field, context.derivatives.metric, optimize=True)
    overlap_first_dot = xp.einsum("x,xmn->mn", field_dot, context.derivatives.metric, optimize=True)
    mechanical_first = xp.einsum("x,xmn->mn", field, context.derivatives.mechanical, optimize=True)
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
    electric_eta = (
        (-1j / context.hbar)
        * theta
        * xp.einsum(
            "mnx,xmn->mn",
            electric_at_pairs,
            context.central_dipoles,
            optimize=True,
        )
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


def exact_site_scalar_action_direction(
    sample: ExactWilsonOneElectronSample,
    site_scalar_direction: object,
    ao_to_site: object,
    backend: ArrayBackend,
) -> OneElectronActionMatrixDirection:
    r"""Return the electronic action direction for independent site scalars.

    The direction contains only the exact anchor-scalar sector

    ``delta omega_mn=(i q/hbar) S_mn delta Phi_site(n)``.

    Internal electric and endpoint-link directions are deliberately held
    fixed so this discrete source component can be qualified separately.
    """

    _validate_sample_backend(sample, backend)
    xp = backend.namespace
    mapping = backend.asarray(ao_to_site, dtype=xp.int64)
    backend.assert_resident(mapping, name="AO-to-site map")
    values = backend.asarray(site_scalar_direction, dtype=xp.float64)
    backend.assert_resident(values, name="site scalar direction")
    if mapping.ndim != 1 or mapping.shape[0] != sample.metric.shape[0]:
        raise FormulationError("AO-to-site map has an incompatible shape")
    if values.ndim != 1 or values.shape[0] == 0:
        raise FormulationError("site scalar direction must be a nonempty vector")
    if _control_bool(xp.any(mapping < 0), backend) or _control_bool(
        xp.any(mapping >= values.shape[0]), backend
    ):
        raise FormulationError("AO-to-site map contains an invalid site index")
    _require_finite(values, backend, "site scalar direction")
    scalar_at_ket = values[mapping]
    connection = (
        (1j * sample.static_result.charge / sample.static_result.hbar)
        * sample.metric
        * scalar_at_ket[None, :]
    )
    zero = xp.zeros_like(sample.metric, dtype=xp.complex128)
    return OneElectronActionMatrixDirection(
        metric=zero,
        mechanical=xp.zeros_like(zero),
        connection=xp.asarray(connection, dtype=xp.complex128),
    )


def exact_endpoint_link_action_direction(
    sample: ExactWilsonOneElectronSample,
    site_link_direction: object,
    ao_to_site: object,
    backend: ArrayBackend,
) -> OneElectronActionMatrixDirection:
    r"""Differentiate only the oriented endpoint link of the exact action.

    ``site_link_direction[a,b]`` is the real line-integral direction from site
    ``b`` to site ``a`` and must therefore be antisymmetric.  The internal
    triangle, anchored-vector, and temporal-electric amplitudes are held
    fixed.  This is the discrete oriented-link source direction requested by
    WP7, not a complete physical magnetic-field variation.
    """

    _validate_sample_backend(sample, backend)
    xp = backend.namespace
    mapping = backend.asarray(ao_to_site, dtype=xp.int64)
    links = backend.asarray(site_link_direction, dtype=xp.float64)
    backend.assert_resident(mapping, name="AO-to-site map")
    backend.assert_resident(links, name="site link direction")
    if mapping.ndim != 1 or mapping.shape[0] != sample.metric.shape[0]:
        raise FormulationError("AO-to-site map has an incompatible shape")
    if links.ndim != 2 or links.shape[0] != links.shape[1] or links.shape[0] == 0:
        raise FormulationError("site link direction must be a nonempty square matrix")
    if _control_bool(xp.any(mapping < 0), backend) or _control_bool(
        xp.any(mapping >= links.shape[0]), backend
    ):
        raise FormulationError("AO-to-site map contains an invalid site index")
    _require_finite(links, backend, "site link direction")
    antisymmetry = xp.linalg.norm(links + links.T)
    scale = xp.maximum(xp.asarray(1.0), xp.linalg.norm(links))
    if backend.scalar_to_float(antisymmetry / scale) > 1.0e-13:
        raise FormulationError("site link direction must be antisymmetric")
    ao_links = links[mapping[:, None], mapping[None, :]]
    phase_direction = (1j * sample.static_result.charge / sample.static_result.hbar) * ao_links
    return OneElectronActionMatrixDirection(
        metric=xp.asarray(phase_direction * sample.metric, dtype=xp.complex128),
        mechanical=xp.asarray(
            phase_direction * sample.mechanical,
            dtype=xp.complex128,
        ),
        connection=xp.asarray(
            phase_direction * sample.connection.connection,
            dtype=xp.complex128,
        ),
    )


def exact_magnetic_field_action_direction(
    response: ExactMagneticFieldSourceDirection,
    backend: ArrayBackend,
) -> OneElectronActionMatrixDirection:
    """Return the complete physical instantaneous-``B`` action direction."""

    _validate_magnetic_response(response, backend)
    xp = backend.namespace
    return OneElectronActionMatrixDirection(
        metric=xp.asarray(response.metric, dtype=xp.complex128),
        mechanical=xp.asarray(response.mechanical, dtype=xp.complex128),
        connection=xp.asarray(response.connection, dtype=xp.complex128),
    )


def exact_internal_magnetic_action_direction(
    response: ExactMagneticFieldSourceDirection,
    backend: ArrayBackend,
) -> OneElectronActionMatrixDirection:
    """Return only triangle and anchored-vector magnetic source response."""

    _validate_magnetic_response(response, backend)
    xp = backend.namespace
    return OneElectronActionMatrixDirection(
        metric=xp.asarray(response.metric_internal, dtype=xp.complex128),
        mechanical=xp.asarray(response.mechanical_internal, dtype=xp.complex128),
        connection=xp.asarray(response.connection_internal, dtype=xp.complex128),
    )


def exact_magnetic_endpoint_action_direction(
    response: ExactMagneticFieldSourceDirection,
    backend: ArrayBackend,
) -> OneElectronActionMatrixDirection:
    """Return only the endpoint-link part of a physical magnetic direction."""

    _validate_magnetic_response(response, backend)
    xp = backend.namespace
    return OneElectronActionMatrixDirection(
        metric=xp.asarray(response.metric_endpoint, dtype=xp.complex128),
        mechanical=xp.asarray(response.mechanical_endpoint, dtype=xp.complex128),
        connection=xp.asarray(response.connection_endpoint, dtype=xp.complex128),
    )


def _compatible_connection(metric: Any, metric_dot: Any, sigma: Any) -> Any:
    """Return ``S Sigma + 1/2(D_t S)`` with no internal residual."""

    return metric * sigma[None, :] + 0.5 * (
        metric_dot + sigma[:, None] * metric - metric * sigma[None, :]
    )


def _validate_sample_backend(
    sample: ExactWilsonOneElectronSample,
    backend: ArrayBackend,
) -> None:
    if not isinstance(sample, ExactWilsonOneElectronSample):
        raise TypeError("sample must be an ExactWilsonOneElectronSample")
    for name, value in (
        ("exact metric", sample.metric),
        ("exact mechanical matrix", sample.mechanical),
        ("exact temporal connection", sample.connection.connection),
    ):
        backend.assert_resident(value, name=name)


def _validate_magnetic_response(
    response: ExactMagneticFieldSourceDirection,
    backend: ArrayBackend,
) -> None:
    if not isinstance(response, ExactMagneticFieldSourceDirection):
        raise TypeError("response must be an ExactMagneticFieldSourceDirection")
    for name, value in (
        ("magnetic metric endpoint response", response.metric_endpoint),
        ("magnetic metric internal response", response.metric_internal),
        ("magnetic mechanical endpoint response", response.mechanical_endpoint),
        (
            "magnetic mechanical triangle response",
            response.mechanical_internal_triangle,
        ),
        (
            "magnetic mechanical anchored response",
            response.mechanical_internal_anchored,
        ),
        ("magnetic connection endpoint response", response.connection_endpoint),
        ("magnetic connection internal response", response.connection_internal),
    ):
        backend.assert_resident(value, name=name)
        _require_finite(value, backend, name)


def _require_finite(value: Any, backend: ArrayBackend, name: str) -> None:
    xp = backend.namespace
    if not _control_bool(xp.all(xp.isfinite(value)), backend):
        raise FormulationError(f"{name} contains non-finite values")


def _control_bool(value: Any, backend: ArrayBackend) -> bool:
    return bool(backend.scalar_to_float(backend.namespace.asarray(value)))
