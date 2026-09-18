"""Independent analytic Gaussian Fourier oracle for uniform-B overlaps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aion.backends import NumPyBackend
from aion.config import BackendConfig
from aion.electromagnetism import (
    AffineMagneticGauge,
    UniformMagneticField,
    build_magnetic_pair_geometry,
    endpoint_links,
)
from aion.electronic_structure.data import (
    OneElectronAOReference,
    PreparedReference,
    immutable_array,
)
from aion.electronic_structure.one_electron import reconstruct_one_electron_molecule
from aion.electronic_structure.pyscf_rks import reconstruct_mean_field
from aion.errors import ConfigurationError, ReferencePreparationError


@dataclass(frozen=True, slots=True)
class AnalyticMagneticOverlap:
    """Exact finite-field overlap from contracted-Gaussian Fourier integrals."""

    field: UniformMagneticField
    gauge: AffineMagneticGauge
    barred: np.ndarray
    lower: np.ndarray
    endpoint_link: np.ndarray
    atom_pair_wavevectors_au: np.ndarray
    reference_fingerprint_sha256: str
    charge: float
    hbar: float

    def __post_init__(self) -> None:
        if not isinstance(self.field, UniformMagneticField):
            raise TypeError("field must be UniformMagneticField")
        if not isinstance(self.gauge, AffineMagneticGauge) or self.gauge.field != self.field:
            raise ConfigurationError("analytic overlap gauge must represent its field")
        barred = immutable_array(
            self.barred,
            dtype=np.complex128,
            ndim=2,
            name="analytic barred magnetic overlap",
        )
        lower = immutable_array(
            self.lower,
            dtype=np.complex128,
            ndim=2,
            name="analytic lower magnetic overlap",
        )
        endpoint = immutable_array(
            self.endpoint_link,
            dtype=np.complex128,
            ndim=2,
            name="analytic magnetic endpoint link",
        )
        wavevectors = immutable_array(
            self.atom_pair_wavevectors_au,
            dtype=np.float64,
            ndim=3,
            name="analytic magnetic atom-pair wavevectors",
        )
        if barred.shape[0] != barred.shape[1] or lower.shape != barred.shape:
            raise ReferencePreparationError("analytic magnetic overlap matrices are inconsistent")
        if endpoint.shape != barred.shape:
            raise ReferencePreparationError("analytic magnetic endpoint matrix is inconsistent")
        if wavevectors.shape[0] != wavevectors.shape[1] or wavevectors.shape[2] != 3:
            raise ReferencePreparationError("analytic atom-pair wavevectors are inconsistent")
        object.__setattr__(self, "barred", barred)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "endpoint_link", endpoint)
        object.__setattr__(self, "atom_pair_wavevectors_au", wavevectors)


def analytic_uniform_magnetic_overlap(
    reference: PreparedReference | OneElectronAOReference,
    gauge: AffineMagneticGauge,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> AnalyticMagneticOverlap:
    r"""Evaluate the uniform-field overlap without real-space quadrature.

    For row/column anchors ``R_mu`` and ``R_nu``, the barred triangle factor
    is a plane wave with ``k = q (R_mu-R_nu) cross B / (2 hbar)``. PySCF's
    libcint-backed AO-pair Fourier integral supplies its exact contracted-
    Gaussian transform. Only the atom-pair block matching those anchors is
    selected from each Fourier matrix.
    """

    if not isinstance(reference, PreparedReference | OneElectronAOReference):
        raise TypeError("reference must be PreparedReference or OneElectronAOReference")
    if not isinstance(gauge, AffineMagneticGauge):
        raise TypeError("gauge must be AffineMagneticGauge")
    checked_charge = _finite_parameter(charge, "charge")
    checked_hbar = _positive_parameter(hbar, "hbar")
    try:
        from pyscf.gto.ft_ao import ft_aopair
    except ImportError as exc:  # pragma: no cover - managed CPU dependency
        raise ReferencePreparationError(
            "analytic magnetic overlap requires pyscf.gto.ft_ao"
        ) from exc

    molecule = (
        reconstruct_mean_field(reference, BackendConfig()).mol
        if isinstance(reference, PreparedReference)
        else reconstruct_one_electron_molecule(reference)
    )
    atom_coordinates = np.asarray(
        reference.core_operators.nuclei.coordinates_au,
        dtype=np.float64,
    )
    mapping = np.asarray(reference.anchor_topology.ao_to_atom, dtype=np.int64)
    backend = NumPyBackend()
    geometry = build_magnetic_pair_geometry(atom_coordinates, mapping, backend)
    endpoint = endpoint_links(
        gauge,
        geometry,
        backend,
        charge=checked_charge,
        hbar=checked_hbar,
    )
    atom_count = atom_coordinates.shape[0]
    wavevectors = np.zeros((atom_count, atom_count, 3), dtype=np.float64)
    pairs: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    arguments: list[np.ndarray] = []
    field = np.asarray(gauge.field.magnetic_field_au, dtype=np.float64)
    for atom_mu, center_mu in enumerate(atom_coordinates):
        for atom_nu, center_nu in enumerate(atom_coordinates):
            midpoint = 0.5 * (center_mu + center_nu)
            wavevector = (
                checked_charge
                * np.cross(center_mu - center_nu, field)
                / (2.0 * checked_hbar)
            )
            wavevectors[atom_mu, atom_nu] = wavevector
            pairs.append((atom_mu, atom_nu, midpoint, wavevector))
            # ft_aopair evaluates integral AO_mu AO_nu exp(-i G.r) dr.
            arguments.append(-wavevector)

    pair_fourier = np.asarray(
        ft_aopair(molecule, np.asarray(arguments)),
        dtype=np.complex128,
    )
    nao = reference.core_operators.nao
    if pair_fourier.shape != (atom_count * atom_count, nao, nao):
        raise ReferencePreparationError("PySCF AO-pair Fourier result has an invalid shape")
    barred = np.zeros((nao, nao), dtype=np.complex128)
    for index, (atom_mu, atom_nu, midpoint, wavevector) in enumerate(pairs):
        rows = np.flatnonzero(mapping == atom_mu)
        columns = np.flatnonzero(mapping == atom_nu)
        midpoint_phase = np.exp(-1j * np.dot(wavevector, midpoint))
        barred[np.ix_(rows, columns)] = (
            midpoint_phase * pair_fourier[index][np.ix_(rows, columns)]
        )

    return AnalyticMagneticOverlap(
        field=gauge.field,
        gauge=gauge,
        barred=barred,
        lower=endpoint * barred,
        endpoint_link=endpoint,
        atom_pair_wavevectors_au=wavevectors,
        reference_fingerprint_sha256=reference.fingerprint_sha256,
        charge=checked_charge,
        hbar=checked_hbar,
    )


def _finite_parameter(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be finite")
    try:
        checked = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be finite") from exc
    if not np.isfinite(checked):
        raise ConfigurationError(f"{name} must be finite")
    return checked


def _positive_parameter(value: float, name: str) -> float:
    checked = _finite_parameter(value, name)
    if checked <= 0.0:
        raise ConfigurationError(f"{name} must be positive")
    return checked
