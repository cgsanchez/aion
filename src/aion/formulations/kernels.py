"""Reusable backend-neutral algebra for formulation physics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aion.backends import ArrayBackend
from aion.electronic_structure import expectation, hermitian_part
from aion.errors import FormulationError


def density_derivative(
    density: Any,
    metric: Any,
    hamiltonian_eom: Any,
    connection: Any,
    *,
    hbar: float,
    backend: ArrayBackend,
) -> Any:
    """Evaluate ``Pdot=A P+P A^dagger`` from the explicit EOM triple."""

    xp = backend.namespace
    generator = xp.linalg.solve(
        metric,
        -connection - (1j / float(hbar)) * hamiltonian_eom,
    )
    return hermitian_part(generator @ density + density @ generator.conj().T)


@dataclass(frozen=True, slots=True)
class MechanicalCurrent:
    paramagnetic: Any
    diamagnetic: Any
    total: Any


def generic_mechanical_current(
    density: Any,
    canonical_momentum: Any,
    vector_potential_operators: Any,
    *,
    charge: float,
    mass: float,
    backend: ArrayBackend,
) -> MechanicalCurrent:
    """Contract local mechanical current for arbitrary AO vector-potential matrices.

    ``vector_potential_operators[alpha]`` is
    ``<phi_mu|a_alpha(r,t)|phi_nu>``.  No idempotency or electron-count
    assumption is made about the finite Hermitian input density.
    """

    for name, value in (
        ("density", density),
        ("canonical_momentum", canonical_momentum),
        ("vector_potential_operators", vector_potential_operators),
    ):
        backend.assert_resident(value, name=name)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise FormulationError("density must be square")
    if canonical_momentum.shape != (3, *density.shape):
        raise FormulationError("canonical momentum must have shape (3, nao, nao)")
    if vector_potential_operators.shape != canonical_momentum.shape:
        raise FormulationError("vector-potential operators must have shape (3, nao, nao)")
    if not np.isfinite(charge) or not np.isfinite(mass) or mass <= 0.0:
        raise FormulationError("charge must be finite and mass must be finite and positive")
    xp = backend.namespace
    momentum_expectation = xp.real(
        xp.einsum("ij,xji->x", density, canonical_momentum, optimize=True)
    )
    vector_expectation = xp.real(
        xp.einsum("ij,xji->x", density, vector_potential_operators, optimize=True)
    )
    paramagnetic = (charge / mass) * momentum_expectation
    diamagnetic = -(charge * charge / mass) * vector_expectation
    return MechanicalCurrent(
        paramagnetic=paramagnetic,
        diamagnetic=diamagnetic,
        total=paramagnetic + diamagnetic,
    )


def uniform_mechanical_current(
    density: Any,
    canonical_momentum: Any,
    overlap: Any,
    vector_potential_reduced: Any,
    *,
    charge: float,
    mass: float,
    backend: ArrayBackend,
) -> MechanicalCurrent:
    """Uniform specialization ``Aop_alpha=a_alpha S``."""

    xp = backend.namespace
    aop = vector_potential_reduced[:, None, None] * overlap[None, :, :]
    return generic_mechanical_current(
        density,
        canonical_momentum,
        xp.asarray(aop, dtype=xp.complex128),
        charge=charge,
        mass=mass,
        backend=backend,
    )


@dataclass(frozen=True, slots=True)
class P0Geometry:
    theta: Any
    theta_dot: Any
    metric: Any
    metric_dot: Any
    sigma_diagonal: Any
    d_site_metric: Any
    connection: Any


def _site_antisymmetric_matrix(
    pair_values: Any,
    pairs: tuple[tuple[int, int], ...],
    natom: int,
    backend: ArrayBackend,
) -> Any:
    xp = backend.namespace
    result = xp.zeros((natom, natom), dtype=xp.float64)
    for pair_index, (a, b) in enumerate(pairs):
        result[a, b] = pair_values[pair_index]
        result[b, a] = -pair_values[pair_index]
    return result


def p0_geometry(
    overlap0: Any,
    ao_to_atom: Any,
    pairs: tuple[tuple[int, int], ...],
    node_scalar_potential: Any,
    pair_link: Any,
    pair_link_dot: Any,
    *,
    natom: int,
    charge: float,
    hbar: float,
    backend: ArrayBackend,
) -> P0Geometry:
    """Build Wilson metric and metric-compatible projected P0 connection."""

    xp = backend.namespace
    site_link = _site_antisymmetric_matrix(pair_link, pairs, natom, backend)
    site_link_dot = _site_antisymmetric_matrix(pair_link_dot, pairs, natom, backend)
    ao_link = site_link[ao_to_atom[:, None], ao_to_atom[None, :]]
    ao_link_dot = site_link_dot[ao_to_atom[:, None], ao_to_atom[None, :]]
    theta = xp.exp((1j * charge / hbar) * ao_link)
    theta_dot = (1j * charge / hbar) * ao_link_dot * theta
    metric = hermitian_part(theta * overlap0)
    metric_dot = hermitian_part(theta_dot * overlap0)
    sigma = (1j * charge / hbar) * node_scalar_potential[ao_to_atom]
    d_site_metric = metric_dot + sigma[:, None] * metric - metric * sigma[None, :]
    connection = metric * sigma[None, :] + 0.5 * d_site_metric
    return P0Geometry(
        theta=theta,
        theta_dot=theta_dot,
        metric=metric,
        metric_dot=metric_dot,
        sigma_diagonal=sigma,
        d_site_metric=d_site_metric,
        connection=connection,
    )


def inverse_dressed_density(density: Any, theta: Any) -> Any:
    return hermitian_part(theta.conj() * density)


def inverse_dressed_density_dot(
    density: Any, density_dot_value: Any, theta: Any, theta_dot: Any
) -> Any:
    return hermitian_part(theta_dot.conj() * density + theta.conj() * density_dot_value)


def central_dipole_matrices(
    overlap0: Any,
    position: Any,
    ao_to_atom: Any,
    atom_coordinates: Any,
    *,
    charge: float,
    backend: ArrayBackend,
) -> Any:
    """Build ``d^alpha=q(r^alpha-R_pair^alpha S0)``."""

    xp = backend.namespace
    ao_coordinates = atom_coordinates[ao_to_atom]
    centers = 0.5 * (ao_coordinates[:, None, :] + ao_coordinates[None, :, :])
    result = xp.empty(position.shape, dtype=xp.complex128)
    for axis in range(3):
        result[axis] = hermitian_part(charge * (position[axis] - centers[:, :, axis] * overlap0))
    return result


@dataclass(frozen=True, slots=True)
class E1Tensors:
    dressed_dipoles: Any
    dressed_dipole_dots: Any
    vector_derivatives: Any
    potential: Any
    connection: Any


def e1_tensors(
    central_dipoles0: Any,
    theta: Any,
    theta_dot: Any,
    ao_to_atom: Any,
    atom_coordinates: Any,
    electric_field: Any,
    *,
    charge: float,
    hbar: float,
    backend: ArrayBackend,
) -> E1Tensors:
    """Dress the E1 tensors and place their variational potential in connection."""

    xp = backend.namespace
    dressed = xp.empty_like(central_dipoles0, dtype=xp.complex128)
    dressed_dot = xp.empty_like(central_dipoles0, dtype=xp.complex128)
    ao_coordinates = atom_coordinates[ao_to_atom]
    displacement = ao_coordinates[:, None, :] - ao_coordinates[None, :, :]
    derivatives = xp.empty((3, 3, *theta.shape), dtype=xp.complex128)
    for alpha in range(3):
        dressed[alpha] = hermitian_part(theta * central_dipoles0[alpha])
        dressed_dot[alpha] = hermitian_part(theta_dot * central_dipoles0[alpha])
        for beta in range(3):
            derivatives[alpha, beta] = hermitian_part(
                (1j * charge / hbar) * displacement[:, :, beta] * theta * central_dipoles0[alpha]
            )
    potential = hermitian_part(-xp.einsum("x,xij->ij", electric_field, dressed, optimize=True))
    connection = (1j / hbar) * potential
    return E1Tensors(
        dressed_dipoles=dressed,
        dressed_dipole_dots=dressed_dot,
        vector_derivatives=derivatives,
        potential=potential,
        connection=connection,
    )


def site_charges(
    density: Any,
    metric: Any,
    projectors: Any,
    *,
    charge: float,
    backend: ArrayBackend,
) -> Any:
    xp = backend.namespace
    values = xp.empty((projectors.shape[0],), dtype=xp.float64)
    for atom in range(projectors.shape[0]):
        diagonal = projectors[atom]
        operator = 0.5 * (diagonal[:, None] * metric + metric * diagonal[None, :])
        values[atom] = charge * expectation(density, operator, xp)
    return values


def site_charge_derivatives(
    density: Any,
    density_dot_value: Any,
    metric: Any,
    metric_dot: Any,
    projectors: Any,
    *,
    charge: float,
    backend: ArrayBackend,
) -> Any:
    xp = backend.namespace
    values = xp.empty((projectors.shape[0],), dtype=xp.float64)
    for atom in range(projectors.shape[0]):
        diagonal = projectors[atom]
        operator = 0.5 * (diagonal[:, None] * metric + metric * diagonal[None, :])
        operator_dot = 0.5 * (diagonal[:, None] * metric_dot + metric_dot * diagonal[None, :])
        values[atom] = charge * (
            expectation(density_dot_value, operator, xp) + expectation(density, operator_dot, xp)
        )
    return values


def p0_pair_currents(
    density: Any,
    metric: Any,
    d_site_metric: Any,
    hamiltonian_dynamic: Any,
    source_hamiltonian: Any,
    ao_to_atom: Any,
    pairs: tuple[tuple[int, int], ...],
    *,
    charge: float,
    hbar: float,
    backend: ArrayBackend,
) -> Any:
    """Evaluate once-oriented variational pair currents ``I_ab``, ``a<b``."""

    xp = backend.namespace
    inverse_metric = xp.linalg.inv(metric)
    k_dynamic = hamiltonian_dynamic - 0.5j * hbar * d_site_metric
    result = xp.empty((len(pairs),), dtype=xp.float64)
    prefactor = charge / hbar
    for pair_index, (a, b) in enumerate(pairs):
        rows_a = ao_to_atom == a
        cols_b = ao_to_atom == b
        rows_b = ao_to_atom == b
        cols_a = ao_to_atom == a
        s_ab = metric * rows_a[:, None] * cols_b[None, :]
        h_ba = source_hamiltonian * rows_b[:, None] * cols_a[None, :]
        overlap_operator = (
            s_ab @ inverse_metric @ k_dynamic + k_dynamic.conj().T @ inverse_metric @ s_ab
        )
        overlap_term = xp.einsum("ij,ji->", density, overlap_operator, optimize=True)
        source_term = xp.einsum("ij,ji->", density, h_ba, optimize=True)
        result[pair_index] = prefactor * xp.imag(overlap_term) + (
            2.0 * prefactor * xp.imag(source_term)
        )
    return result


def p0_graph_current(pair_currents: Any, pair_displacements: Any) -> Any:
    """Return ``sum I_ab (R_b-R_a)`` for stored ``R_a-R_b`` displacements."""

    return -pair_currents @ pair_displacements


def p0_source_power(pair_currents: Any, pair_emf: Any, namespace: Any) -> Any:
    return -namespace.dot(pair_currents, pair_emf)


def p0_continuity_residual(charge_derivatives: Any, incidence: Any, pair_currents: Any) -> Any:
    return charge_derivatives + incidence @ pair_currents


def covariant_ambient_mechanical_current(
    density: Any,
    theta: Any,
    overlap0: Any,
    canonical_momentum0: Any,
    vector_potential_reduced: Any,
    *,
    charge: float,
    mass: float,
    backend: ArrayBackend,
) -> MechanicalCurrent:
    """Ambient projection with explicit canonical/diamagnetic cancellation."""

    xp = backend.namespace
    canonical = theta[None, :, :] * (
        canonical_momentum0
        + charge * vector_potential_reduced[:, None, None] * overlap0[None, :, :]
    )
    metric = theta * overlap0
    aop = vector_potential_reduced[:, None, None] * metric[None, :, :]
    return generic_mechanical_current(
        density,
        xp.asarray(canonical, dtype=xp.complex128),
        xp.asarray(aop, dtype=xp.complex128),
        charge=charge,
        mass=mass,
        backend=backend,
    )
