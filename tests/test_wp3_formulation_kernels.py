from __future__ import annotations

import numpy as np
import pytest

from aion.backends import NumPyBackend
from aion.config import FixedTimeGrid, ObservableSchedules, StepSchedule
from aion.errors import FormulationError
from aion.formulations import (
    AODensity,
    central_dipole_matrices,
    density_derivative,
    e1_tensors,
    generic_mechanical_current,
    inverse_dressed_density,
    p0_continuity_residual,
    p0_geometry,
    p0_pair_currents,
    site_charge_derivatives,
)
from aion.observables import ObservableDependency, build_observable_calculators

pytestmark = pytest.mark.fast


def _hermitian(rng: np.random.Generator, size: int) -> np.ndarray:
    value = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    return 0.5 * (value + value.conj().T)


def test_ao_density_has_explicit_occupation_and_backend_convention() -> None:
    backend = NumPyBackend()
    coefficients = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    occupations = np.asarray([2.0, 0.3])
    density = AODensity.from_coefficients(coefficients, occupations, backend)
    assert np.array_equal(density.matrix, np.diag(occupations))
    assert float(density.hermiticity_residual()) == 0.0
    with pytest.raises(FormulationError, match="not Hermitian"):
        AODensity.from_matrix(np.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=np.complex128), backend)


def test_observable_families_have_independent_schedules_and_declared_dependencies() -> None:
    schedules = ObservableSchedules(
        dipole_current=StepSchedule(every=2),
        energy=StepSchedule(every=5),
        diagnostics=StepSchedule(every=3),
    )
    from aion.config import FormulationKind

    calculators = build_observable_calculators(schedules, FormulationKind.P0_E1, natom=2, npair=1)
    grid = FixedTimeGrid(0.0, 0.1, 10)
    assert calculators.dipole_current.is_scheduled(2, grid)
    assert not calculators.energy.is_scheduled(2, grid)
    assert calculators.energy.is_scheduled(5, grid)
    assert calculators.diagnostics.is_scheduled(3, grid)
    assert ObservableDependency.DENSITY_DOT in calculators.dipole_current.dependencies
    assert ObservableDependency.FIELD_FREE_DFT in calculators.energy.dependencies
    assert (
        calculators.definitions["ambient_projected_mechanical_current"].definition_id
        == "current.ambient_projected_mechanical.p0_e1"
    )
    assert "pair_currents_continuity" in calculators.definitions
    assert "pair_currents_p0_source" in calculators.definitions
    with pytest.raises(TypeError):
        calculators.definitions["unqualified_current"] = calculators.definitions["primary_current"]


def test_generic_mechanical_current_separates_paramagnetic_and_diamagnetic() -> None:
    backend = NumPyBackend()
    density = np.asarray([[0.8, 0.2j], [-0.2j, 0.4]], dtype=np.complex128)
    momentum = np.zeros((3, 2, 2), dtype=np.complex128)
    momentum[2] = np.asarray([[0.0, 1j], [-1j, 0.0]])
    vector_operators = np.zeros_like(momentum)
    vector_operators[0] = np.asarray([[0.5, 0.1], [0.1, 0.25]])
    actual = generic_mechanical_current(
        density,
        momentum,
        vector_operators,
        charge=-1.0,
        mass=2.0,
        backend=backend,
    )
    expected_para = -0.5 * np.einsum("ij,xji->x", density, momentum).real
    expected_dia = -0.5 * np.einsum("ij,xji->x", density, vector_operators).real
    assert np.allclose(actual.paramagnetic, expected_para)
    assert np.allclose(actual.diamagnetic, expected_dia)
    assert np.allclose(actual.total, expected_para + expected_dia)


def test_p0_metric_connection_dressing_and_continuity_identities() -> None:
    backend = NumPyBackend()
    rng = np.random.default_rng(1977)
    ao_to_atom = np.asarray([0, 0, 1, 1], dtype=np.int64)
    pairs = ((0, 1),)
    raw = rng.normal(size=(4, 4))
    overlap = raw @ raw.T + 4.0 * np.eye(4)
    geometry = p0_geometry(
        overlap,
        ao_to_atom,
        pairs,
        np.asarray([0.017, -0.011]),
        np.asarray([0.21]),
        np.asarray([-0.07]),
        natom=2,
        charge=-1.0,
        hbar=1.0,
        backend=backend,
    )
    assert np.linalg.norm(geometry.metric - geometry.metric.conj().T) < 1.0e-13
    assert (
        np.linalg.norm(geometry.connection + geometry.connection.conj().T - geometry.metric_dot)
        < 1.0e-13
    )
    assert np.allclose(np.abs(geometry.theta), 1.0)

    density = _hermitian(rng, 4)
    hamiltonian = _hermitian(rng, 4)
    density_dot = density_derivative(
        density,
        geometry.metric,
        hamiltonian,
        geometry.connection,
        hbar=1.0,
        backend=backend,
    )
    projectors = np.asarray([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    charge_dot = site_charge_derivatives(
        density,
        density_dot,
        geometry.metric,
        geometry.metric_dot,
        projectors,
        charge=-1.0,
        backend=backend,
    )
    pair_current = p0_pair_currents(
        density,
        geometry.metric,
        geometry.d_site_metric,
        hamiltonian,
        hamiltonian,
        ao_to_atom,
        pairs,
        charge=-1.0,
        hbar=1.0,
        backend=backend,
    )
    incidence = np.asarray([[1.0], [-1.0]])
    assert np.linalg.norm(p0_continuity_residual(charge_dot, incidence, pair_current)) < 2.0e-12


def test_inverse_dressing_is_invariant_under_an_analytic_site_gauge_change() -> None:
    backend = NumPyBackend()
    rng = np.random.default_rng(81)
    ao_to_atom = np.asarray([0, 0, 1], dtype=np.int64)
    pairs = ((0, 1),)
    overlap = np.asarray([[1.0, 0.1, 0.2], [0.1, 1.1, -0.04], [0.2, -0.04, 0.9]])
    original = p0_geometry(
        overlap,
        ao_to_atom,
        pairs,
        np.zeros(2),
        np.asarray([0.13]),
        np.zeros(1),
        natom=2,
        charge=-1.0,
        hbar=1.0,
        backend=backend,
    )
    site_lambda = np.asarray([0.31, -0.17])
    changed = p0_geometry(
        overlap,
        ao_to_atom,
        pairs,
        np.zeros(2),
        np.asarray([0.13 + site_lambda[0] - site_lambda[1]]),
        np.zeros(1),
        natom=2,
        charge=-1.0,
        hbar=1.0,
        backend=backend,
    )
    density = _hermitian(rng, 3)
    phase = np.exp(-1j * site_lambda[ao_to_atom])
    transformed_density = phase[:, None] * density * phase.conj()[None, :]
    assert (
        np.linalg.norm(
            inverse_dressed_density(density, original.theta)
            - inverse_dressed_density(transformed_density, changed.theta)
        )
        < 2.0e-13
    )


def test_e1_tensor_vector_derivative_matches_directional_finite_difference() -> None:
    backend = NumPyBackend()
    rng = np.random.default_rng(29)
    ao_to_atom = np.asarray([0, 1, 1], dtype=np.int64)
    coordinates = np.asarray([[0.0, 0.0, -0.8], [0.0, 0.0, 0.9]])
    overlap = np.eye(3)
    position = np.stack([_hermitian(rng, 3).real for _ in range(3)])
    central = central_dipole_matrices(
        overlap,
        position,
        ao_to_atom,
        coordinates,
        charge=-1.0,
        backend=backend,
    )
    vector = np.asarray([0.07, -0.03, 0.11])
    ao_coordinates = coordinates[ao_to_atom]
    displacement = ao_coordinates[:, None, :] - ao_coordinates[None, :, :]
    theta = np.exp(-1j * np.einsum("ijx,x->ij", displacement, vector))
    tensors = e1_tensors(
        central,
        theta,
        np.zeros_like(theta),
        ao_to_atom,
        coordinates,
        np.asarray([0.012, -0.021, 0.009]),
        charge=-1.0,
        hbar=1.0,
        backend=backend,
    )
    direction = np.asarray([0.3, -0.2, 0.7])
    epsilon = 1.0e-6
    phase = np.exp(-1j * epsilon * np.einsum("ijx,x->ij", displacement, direction))

    def dress(value: np.ndarray) -> np.ndarray:
        result = value[None, :, :] * central
        return 0.5 * (result + result.conj().transpose(0, 2, 1))

    finite_difference = (dress(theta * phase) - dress(theta / phase)) / (2.0 * epsilon)
    analytic = np.einsum("b,abij->aij", direction, tensors.vector_derivatives)
    assert np.linalg.norm(finite_difference - analytic) < 2.0e-9
    assert np.linalg.norm(tensors.potential - tensors.potential.conj().T) < 1.0e-13

    density = _hermitian(rng, 3)
    field = np.asarray([0.012, -0.021, 0.009])
    analytic_action_derivative = np.einsum(
        "a,b,abij,ji->", field, direction, tensors.vector_derivatives, density
    ).real

    def action(dressed: np.ndarray) -> float:
        dipole = np.einsum("ij,xji->x", density, dressed).real
        return float(np.dot(field, dipole))

    finite_action_derivative = (action(dress(theta * phase)) - action(dress(theta / phase))) / (
        2.0 * epsilon
    )
    assert analytic_action_derivative == pytest.approx(finite_action_derivative, abs=2.0e-9)


def test_central_dipoles_and_neutral_total_dipole_are_origin_translation_invariant() -> None:
    backend = NumPyBackend()
    ao_to_atom = np.asarray([0, 1], dtype=np.int64)
    coordinates = np.asarray([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    overlap = np.asarray([[1.0, 0.2], [0.2, 1.0]])
    position = np.asarray(
        [
            [[0.0, 0.03], [0.03, 0.0]],
            [[0.0, -0.02], [-0.02, 0.0]],
            [[-0.7, 0.0], [0.0, 0.7]],
        ]
    )
    shift = np.asarray([0.4, -0.3, 1.2])
    original = central_dipole_matrices(
        overlap,
        position,
        ao_to_atom,
        coordinates,
        charge=-1.0,
        backend=backend,
    )
    translated = central_dipole_matrices(
        overlap,
        position + shift[:, None, None] * overlap,
        ao_to_atom,
        coordinates + shift[None, :],
        charge=-1.0,
        backend=backend,
    )
    assert np.allclose(translated, original, atol=2.0e-14)

    density = np.eye(2, dtype=np.complex128)
    electron_count = np.einsum("ij,ji->", density, overlap).real
    nuclear_charges = np.asarray([1.0, 1.0])
    assert electron_count == pytest.approx(nuclear_charges.sum())
    electronic = -np.einsum("ij,xji->x", density, position).real
    nuclear = nuclear_charges @ coordinates
    translated_electronic = -np.einsum(
        "ij,xji->x", density, position + shift[:, None, None] * overlap
    ).real
    translated_nuclear = nuclear_charges @ (coordinates + shift[None, :])
    assert np.allclose(electronic + nuclear, translated_electronic + translated_nuclear)
