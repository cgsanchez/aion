"""Uniform-magnetic-field values and backend-neutral Wilson geometry.

The row AO is the bra and the column AO is the ket. Endpoint links therefore
follow the ket-anchor-to-bra-anchor path. See
``docs/magnetic_matrix_benchmark_contract.md`` for the normative conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from aion.backends import ArrayBackend
from aion.config.units import Vector3, finite_float, vector3
from aion.errors import ConfigurationError

TESLA_PER_ATOMIC_UNIT_MAGNETIC_FIELD = 235_051.757_077


class MagneticGaugeKind(StrEnum):
    """Supported affine representatives of a uniform magnetic field."""

    SYMMETRIC = "symmetric"
    LANDAU = "landau"


@dataclass(frozen=True, slots=True)
class UniformMagneticField:
    """A physical, spatially uniform magnetic field in atomic units."""

    magnetic_field_au: Vector3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "magnetic_field_au",
            vector3(self.magnetic_field_au, "magnetic_field_au"),
        )

    @classmethod
    def from_tesla(cls, magnetic_field_tesla: object) -> UniformMagneticField:
        """Convert an explicit Cartesian tesla input to atomic units."""

        field_tesla = vector3(magnetic_field_tesla, "magnetic_field_tesla")
        return cls(
            (
                field_tesla[0] / TESLA_PER_ATOMIC_UNIT_MAGNETIC_FIELD,
                field_tesla[1] / TESLA_PER_ATOMIC_UNIT_MAGNETIC_FIELD,
                field_tesla[2] / TESLA_PER_ATOMIC_UNIT_MAGNETIC_FIELD,
            )
        )

    @property
    def magnetic_field_tesla(self) -> Vector3:
        """Return the Cartesian field in tesla for recording at a boundary."""

        return (
            self.magnetic_field_au[0] * TESLA_PER_ATOMIC_UNIT_MAGNETIC_FIELD,
            self.magnetic_field_au[1] * TESLA_PER_ATOMIC_UNIT_MAGNETIC_FIELD,
            self.magnetic_field_au[2] * TESLA_PER_ATOMIC_UNIT_MAGNETIC_FIELD,
        )

    @property
    def magnitude_au(self) -> float:
        """Return the Euclidean field magnitude in atomic units."""

        return math.sqrt(sum(component * component for component in self.magnetic_field_au))


@dataclass(frozen=True, slots=True)
class AffineMagneticGauge:
    """An affine vector-potential representative with curl equal to ``field``."""

    field: UniformMagneticField
    kind: MagneticGaugeKind = MagneticGaugeKind.SYMMETRIC
    origin_au: Vector3 = (0.0, 0.0, 0.0)
    landau_axis: Vector3 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.field, UniformMagneticField):
            raise ConfigurationError("field must be a UniformMagneticField")
        try:
            kind = MagneticGaugeKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"unsupported magnetic gauge kind {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "origin_au", vector3(self.origin_au, "origin_au"))
        if self.kind is MagneticGaugeKind.SYMMETRIC:
            if self.landau_axis is not None:
                raise ConfigurationError("landau_axis is valid only for Landau gauge")
            return
        if self.landau_axis is None:
            raise ConfigurationError("Landau gauge requires landau_axis")
        axis = np.asarray(vector3(self.landau_axis, "landau_axis"), dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm == 0.0:
            raise ConfigurationError("landau_axis must be nonzero")
        axis /= norm
        field = np.asarray(self.field.magnetic_field_au, dtype=np.float64)
        field_norm = float(np.linalg.norm(field))
        if field_norm and abs(float(field @ axis)) > 1.0e-12 * field_norm:
            raise ConfigurationError("landau_axis must be perpendicular to magnetic_field_au")
        object.__setattr__(self, "landau_axis", tuple(float(value) for value in axis))

    def affine_matrix(self, backend: ArrayBackend) -> Any:
        """Return ``M`` in ``A(r) = M @ (r-origin)`` on ``backend``."""

        bx, by, bz = self.field.magnetic_field_au
        cross_matrix = backend.asarray(
            ((0.0, -bz, by), (bz, 0.0, -bx), (-by, bx, 0.0)),
            dtype=backend.namespace.float64,
        )
        if self.kind is MagneticGaugeKind.SYMMETRIC:
            return 0.5 * cross_matrix
        assert self.landau_axis is not None
        xp = backend.namespace
        field = backend.asarray(self.field.magnetic_field_au, dtype=xp.float64)
        axis = backend.asarray(self.landau_axis, dtype=xp.float64)
        return xp.outer(xp.cross(field, axis), axis)

    def vector_potential(self, points_au: object, backend: ArrayBackend) -> Any:
        """Evaluate the affine vector potential at Cartesian points."""

        points = _cartesian_array(points_au, backend, name="points_au")
        origin = backend.asarray(self.origin_au, dtype=backend.namespace.float64)
        return (points - origin) @ self.affine_matrix(backend).T

    def straight_line_integrals(
        self,
        starts_au: object,
        ends_au: object,
        backend: ArrayBackend,
    ) -> Any:
        """Integrate ``A.dl`` exactly from each start to each end."""

        starts = _cartesian_array(starts_au, backend, name="starts_au")
        ends = _cartesian_array(ends_au, backend, name="ends_au")
        xp = backend.namespace
        try:
            starts, ends = xp.broadcast_arrays(starts, ends)
        except ValueError as exc:
            raise ConfigurationError("starts_au and ends_au cannot be broadcast") from exc
        displacements = ends - starts
        midpoints = 0.5 * (starts + ends)
        return xp.einsum(
            "...x,...x->...",
            self.vector_potential(midpoints, backend),
            displacements,
            optimize=True,
        )

    def straight_line_integral_gradients(
        self,
        starts_au: object,
        ends_au: object,
        backend: ArrayBackend,
    ) -> Any:
        """Differentiate a straight-line integral with respect to its end."""

        starts = _cartesian_array(starts_au, backend, name="starts_au")
        ends = _cartesian_array(ends_au, backend, name="ends_au")
        xp = backend.namespace
        try:
            starts, ends = xp.broadcast_arrays(starts, ends)
        except ValueError as exc:
            raise ConfigurationError("starts_au and ends_au cannot be broadcast") from exc
        matrix = self.affine_matrix(backend)
        return self.vector_potential(starts, backend) + 0.5 * (
            (ends - starts) @ (matrix + matrix.T).T
        )

    def anchor_to_point_line_integrals(
        self,
        ao_anchor_coordinates_au: object,
        points_au: object,
        backend: ArrayBackend,
    ) -> Any:
        """Return anchor-to-point line integrals with shape ``(npoint, nao)``."""

        anchors = _cartesian_rows(
            ao_anchor_coordinates_au, backend, name="ao_anchor_coordinates_au"
        )
        points = _cartesian_rows(points_au, backend, name="points_au")
        return self.straight_line_integrals(anchors[None, :, :], points[:, None, :], backend)

    def anchor_to_point_line_integral_gradients(
        self,
        ao_anchor_coordinates_au: object,
        points_au: object,
        backend: ArrayBackend,
    ) -> Any:
        """Return endpoint gradients with shape ``(npoint, nao, 3)``."""

        anchors = _cartesian_rows(
            ao_anchor_coordinates_au, backend, name="ao_anchor_coordinates_au"
        )
        points = _cartesian_rows(points_au, backend, name="points_au")
        return self.straight_line_integral_gradients(
            anchors[None, :, :], points[:, None, :], backend
        )


@dataclass(frozen=True, slots=True)
class UniformMagneticSourceSample:
    r"""One Maxwell-consistent sample of a time-dependent uniform field.

    The physical potentials are based on the symmetric representative

    ``A_sym(r,t) = 1/2 B(t) x (r-origin)`` and
    ``Phi_sym(r,t) = -E_origin(t) . (r-origin)``.

    Consequently the electric field contains the affine induction partner
    required by Faraday's law,

    ``E(r,t) = E_origin(t) + 1/2 (r-origin) x Bdot(t)``.

    ``gauge_kind`` may select another affine representative of the same
    magnetic field.  The scalar potential then includes ``-partial_t lambda``
    automatically, so changing representative does not change the physical
    electric field.
    """

    time_au: float
    field: UniformMagneticField
    magnetic_field_dot_au: Vector3 = (0.0, 0.0, 0.0)
    electric_field_origin_au: Vector3 = (0.0, 0.0, 0.0)
    origin_au: Vector3 = (0.0, 0.0, 0.0)
    gauge_kind: MagneticGaugeKind = MagneticGaugeKind.SYMMETRIC
    landau_axis: Vector3 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.field, UniformMagneticField):
            raise ConfigurationError("field must be a UniformMagneticField")
        checked_time = finite_float(self.time_au, "time_au")
        object.__setattr__(self, "time_au", checked_time)
        object.__setattr__(
            self,
            "magnetic_field_dot_au",
            vector3(self.magnetic_field_dot_au, "magnetic_field_dot_au"),
        )
        object.__setattr__(
            self,
            "electric_field_origin_au",
            vector3(self.electric_field_origin_au, "electric_field_origin_au"),
        )
        object.__setattr__(self, "origin_au", vector3(self.origin_au, "origin_au"))
        try:
            kind = MagneticGaugeKind(self.gauge_kind)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"unsupported magnetic gauge kind {self.gauge_kind!r}"
            ) from exc
        object.__setattr__(self, "gauge_kind", kind)
        # Construct both affine gauges now so invalid Landau axes fail at the
        # immutable source boundary.  A fixed Landau direction must work for
        # both B and Bdot because its time derivative is used below.
        _ = self.gauge
        _ = self.gauge_rate

    @property
    def gauge(self) -> AffineMagneticGauge:
        return AffineMagneticGauge(
            self.field,
            kind=self.gauge_kind,
            origin_au=self.origin_au,
            landau_axis=self.landau_axis,
        )

    @property
    def gauge_rate(self) -> AffineMagneticGauge:
        return AffineMagneticGauge(
            UniformMagneticField(self.magnetic_field_dot_au),
            kind=self.gauge_kind,
            origin_au=self.origin_au,
            landau_axis=self.landau_axis,
        )

    def electric_field(self, points_au: object, backend: ArrayBackend) -> Any:
        """Evaluate the physical uniform-plus-induction electric field."""

        points = _cartesian_array(points_au, backend, name="points_au")
        xp = backend.namespace
        origin = backend.asarray(self.origin_au, dtype=xp.float64)
        electric_origin = backend.asarray(self.electric_field_origin_au, dtype=xp.float64)
        field_dot = backend.asarray(self.magnetic_field_dot_au, dtype=xp.float64)
        return electric_origin + 0.5 * xp.cross(points - origin, field_dot)

    def scalar_potential(self, points_au: object, backend: ArrayBackend) -> Any:
        """Evaluate the gauge-resolved scalar potential at Cartesian points."""

        points = _cartesian_array(points_au, backend, name="points_au")
        xp = backend.namespace
        origin = backend.asarray(self.origin_au, dtype=xp.float64)
        relative = points - origin
        electric_origin = backend.asarray(self.electric_field_origin_au, dtype=xp.float64)
        symmetric_rate = AffineMagneticGauge(
            UniformMagneticField(self.magnetic_field_dot_au),
            origin_au=self.origin_au,
        )
        gauge_scalar_rate = affine_gauge_difference_potential(
            self.gauge_rate,
            symmetric_rate,
            points,
            backend,
        )
        return -xp.einsum("...x,x->...", relative, electric_origin, optimize=True) - (
            gauge_scalar_rate
        )

    def radial_electric_line_integrals(
        self,
        starts_au: object,
        ends_au: object,
        backend: ArrayBackend,
    ) -> Any:
        r"""Return ``integral_0^1 (end-start).E(start+s(end-start)) ds``."""

        starts = _cartesian_array(starts_au, backend, name="starts_au")
        ends = _cartesian_array(ends_au, backend, name="ends_au")
        xp = backend.namespace
        try:
            starts, ends = xp.broadcast_arrays(starts, ends)
        except ValueError as exc:
            raise ConfigurationError("starts_au and ends_au cannot be broadcast") from exc
        displacement = ends - starts
        midpoint = 0.5 * (starts + ends)
        return xp.einsum(
            "...x,...x->...",
            displacement,
            self.electric_field(midpoint, backend),
            optimize=True,
        )


@dataclass(frozen=True, slots=True)
class MagneticPairGeometry:
    """Backend-resident AO anchors and row-minus-column pair geometry."""

    ao_anchor_coordinates_au: Any
    pair_midpoints_au: Any
    pair_displacements_au: Any


@dataclass(frozen=True, slots=True)
class TriangleFactors:
    """Exact, linear, and quadratic uniform-field triangle factors."""

    exact: Any
    first: Any
    second: Any


@dataclass(frozen=True, slots=True)
class UniformMagneticFirstDerivatives:
    """Cartesian zero-field derivatives of internal magnetic generators.

    The leading axis is the differentiated magnetic-field component.  The
    triangle-factor derivative has shape ``(3, npoint, nao, nao)`` and the
    anchored-vector derivative has shape ``(3, npoint, nao, 3)``.
    """

    triangle_phase: Any
    triangle_factor: Any
    anchored_vector: Any


@dataclass(frozen=True, slots=True)
class CenterLoopHolonomy:
    """Uniform-field Wilson holonomy for one oriented centre polygon.

    Array-valued members remain resident on the selected backend.  Reversing
    the order of ``vertices`` reverses the oriented area and conjugates both
    phases for real charge and magnetic field.
    """

    oriented_area_vector_au2: Any
    magnetic_flux_au: Any
    endpoint_link_product: Any
    expected_flux_phase: Any


def build_magnetic_pair_geometry(
    atom_coordinates_au: object,
    ao_to_atom: object,
    backend: ArrayBackend,
) -> MagneticPairGeometry:
    """Lift authenticated atom coordinates to AO-pair magnetic geometry."""

    xp = backend.namespace
    atoms = _cartesian_rows(atom_coordinates_au, backend, name="atom_coordinates_au")
    mapping = backend.asarray(ao_to_atom, dtype=xp.int64)
    backend.assert_resident(mapping, name="ao_to_atom")
    if mapping.ndim != 1 or mapping.size == 0:
        raise ConfigurationError("ao_to_atom must be a nonempty one-dimensional array")
    if _control_bool(xp.any(mapping < 0), backend) or _control_bool(
        xp.any(mapping >= atoms.shape[0]), backend
    ):
        raise ConfigurationError("ao_to_atom contains an invalid atom index")
    anchors = atoms[mapping]
    midpoints = 0.5 * (anchors[:, None, :] + anchors[None, :, :])
    displacements = anchors[:, None, :] - anchors[None, :, :]
    return MagneticPairGeometry(anchors, midpoints, displacements)


def endpoint_line_integrals(
    gauge: AffineMagneticGauge,
    pair_geometry: MagneticPairGeometry,
    backend: ArrayBackend,
) -> Any:
    """Return ket-anchor-to-bra-anchor integrals with shape ``(nao, nao)``."""

    anchors = pair_geometry.ao_anchor_coordinates_au
    _require_resident_pair_geometry(pair_geometry, backend)
    return gauge.straight_line_integrals(anchors[None, :, :], anchors[:, None, :], backend)


def endpoint_links(
    gauge: AffineMagneticGauge,
    pair_geometry: MagneticPairGeometry,
    backend: ArrayBackend,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> Any:
    """Return exact ket-to-bra endpoint Wilson links."""

    checked_charge, checked_hbar = _particle_parameters(charge, hbar)
    xp = backend.namespace
    return xp.exp(
        (1j * checked_charge / checked_hbar)
        * endpoint_line_integrals(gauge, pair_geometry, backend)
    )


def center_loop_holonomy(
    vertices_au: object,
    gauge: AffineMagneticGauge,
    backend: ArrayBackend,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> CenterLoopHolonomy:
    """Evaluate endpoint-link holonomy around an oriented centre polygon.

    Vertices are traversed in their supplied order and the final edge closes
    the polygon.  For a uniform magnetic field, the product of straight-path
    endpoint links must equal ``exp(i*q*B.area/hbar)`` independently of the
    affine vector-potential representative.
    """

    checked_charge, checked_hbar = _particle_parameters(charge, hbar)
    vertices = _cartesian_rows(vertices_au, backend, name="vertices_au")
    if vertices.shape[0] < 3:
        raise ConfigurationError("a centre loop requires at least three vertices")
    xp = backend.namespace
    following = xp.roll(vertices, -1, axis=0)
    area_vector = 0.5 * xp.sum(xp.cross(vertices, following), axis=0)
    magnetic_field = backend.asarray(gauge.field.magnetic_field_au, dtype=xp.float64)
    flux = xp.einsum("x,x->", magnetic_field, area_vector, optimize=True)
    line_integrals = gauge.straight_line_integrals(vertices, following, backend)
    endpoint_product = xp.exp((1j * checked_charge / checked_hbar) * xp.sum(line_integrals))
    expected = xp.exp((1j * checked_charge / checked_hbar) * flux)
    return CenterLoopHolonomy(
        oriented_area_vector_au2=area_vector,
        magnetic_flux_au=flux,
        endpoint_link_product=endpoint_product,
        expected_flux_phase=expected,
    )


def triangle_fluxes(
    points_au: object,
    pair_geometry: MagneticPairGeometry,
    field: UniformMagneticField,
    backend: ArrayBackend,
) -> Any:
    """Return oriented triangle fluxes with shape ``(npoint, nao, nao)``."""

    points = _cartesian_rows(points_au, backend, name="points_au")
    _require_resident_pair_geometry(pair_geometry, backend)
    xp = backend.namespace
    magnetic_field = backend.asarray(field.magnetic_field_au, dtype=xp.float64)
    flux_vectors = 0.5 * xp.cross(pair_geometry.pair_displacements_au, magnetic_field)
    point_terms = xp.einsum("px,mnx->pmn", points, flux_vectors, optimize=True)
    midpoint_terms = xp.einsum(
        "mnx,mnx->mn", pair_geometry.pair_midpoints_au, flux_vectors, optimize=True
    )
    return point_terms - midpoint_terms[None, :, :]


def triangle_phases(
    points_au: object,
    pair_geometry: MagneticPairGeometry,
    field: UniformMagneticField,
    backend: ArrayBackend,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> Any:
    """Return the real dimensionless phase ``q*flux/hbar``."""

    checked_charge, checked_hbar = _particle_parameters(charge, hbar)
    return (checked_charge / checked_hbar) * triangle_fluxes(
        points_au, pair_geometry, field, backend
    )


def triangle_factors(
    points_au: object,
    pair_geometry: MagneticPairGeometry,
    field: UniformMagneticField,
    backend: ArrayBackend,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> TriangleFactors:
    """Return exact and first two Taylor factors at bookkeeping amplitude one."""

    xp = backend.namespace
    phase = triangle_phases(points_au, pair_geometry, field, backend, charge=charge, hbar=hbar)
    return TriangleFactors(
        exact=xp.exp(1j * phase),
        first=1j * phase,
        second=-0.5 * phase * phase,
    )


def anchored_vectors(
    points_au: object,
    pair_geometry: MagneticPairGeometry,
    field: UniformMagneticField,
    backend: ArrayBackend,
) -> Any:
    """Return ``0.5*(r-R_mu) x B`` with shape ``(npoint, nao, 3)``."""

    points = _cartesian_rows(points_au, backend, name="points_au")
    _require_resident_pair_geometry(pair_geometry, backend)
    magnetic_field = backend.asarray(field.magnetic_field_au, dtype=backend.namespace.float64)
    return 0.5 * backend.namespace.cross(
        points[:, None, :] - pair_geometry.ao_anchor_coordinates_au[None, :, :],
        magnetic_field,
    )


def uniform_magnetic_first_derivatives(
    points_au: object,
    pair_geometry: MagneticPairGeometry,
    backend: ArrayBackend,
    *,
    charge: float = -1.0,
    hbar: float = 1.0,
) -> UniformMagneticFirstDerivatives:
    """Return analytic Cartesian derivatives at zero magnetic field.

    Uniform-field triangle phases and anchored vectors are exactly linear in
    the magnetic field.  Evaluating their closed forms on Cartesian unit
    directions therefore gives their analytic derivatives without numerical
    differencing.
    """

    xp = backend.namespace
    fields = tuple(
        UniformMagneticField((float(direction[0]), float(direction[1]), float(direction[2])))
        for direction in np.eye(3)
    )
    phases = xp.stack(
        [
            triangle_phases(
                points_au,
                pair_geometry,
                field,
                backend,
                charge=charge,
                hbar=hbar,
            )
            for field in fields
        ]
    )
    anchored = xp.stack(
        [anchored_vectors(points_au, pair_geometry, field, backend) for field in fields]
    )
    return UniformMagneticFirstDerivatives(
        triangle_phase=phases,
        triangle_factor=1j * phases,
        anchored_vector=anchored,
    )


def affine_gauge_difference_potential(
    target: AffineMagneticGauge,
    reference: AffineMagneticGauge,
    points_au: object,
    backend: ArrayBackend,
) -> Any:
    """Return ``chi`` such that ``A_target-A_reference = grad chi``.

    Both representatives must describe exactly the same physical field.
    The additive constant is fixed to zero in the global Cartesian frame.
    """

    if target.field != reference.field:
        raise ConfigurationError("gauge difference requires identical physical fields")
    points = _cartesian_array(points_au, backend, name="points_au")
    xp = backend.namespace
    target_matrix = target.affine_matrix(backend)
    reference_matrix = reference.affine_matrix(backend)
    difference = target_matrix - reference_matrix
    antisymmetric = difference - difference.T
    if backend.scalar_to_float(xp.linalg.norm(antisymmetric)) > 1.0e-12:
        raise ConfigurationError("affine gauge difference is not curl-free")
    target_origin = backend.asarray(target.origin_au, dtype=xp.float64)
    reference_origin = backend.asarray(reference.origin_au, dtype=xp.float64)
    constant = -target_matrix @ target_origin + reference_matrix @ reference_origin
    quadratic = 0.5 * xp.einsum("...x,xy,...y->...", points, difference, points, optimize=True)
    return quadratic + xp.einsum("...x,x->...", points, constant, optimize=True)


def _particle_parameters(charge: float, hbar: float) -> tuple[float, float]:
    checked_charge = finite_float(charge, "charge")
    checked_hbar = finite_float(hbar, "hbar")
    if checked_hbar <= 0.0:
        raise ConfigurationError("hbar must be positive")
    return checked_charge, checked_hbar


def _cartesian_array(value: object, backend: ArrayBackend, *, name: str) -> Any:
    result = backend.asarray(value, dtype=backend.namespace.float64)
    backend.assert_resident(result, name=name)
    if result.ndim < 1 or result.shape[-1] != 3:
        raise ConfigurationError(f"{name} must have final dimension 3")
    if not _control_bool(backend.namespace.all(backend.namespace.isfinite(result)), backend):
        raise ConfigurationError(f"{name} contains non-finite values")
    return result


def _cartesian_rows(value: object, backend: ArrayBackend, *, name: str) -> Any:
    result = _cartesian_array(value, backend, name=name)
    if result.ndim != 2:
        raise ConfigurationError(f"{name} must have shape (n, 3)")
    return result


def _control_bool(value: object, backend: ArrayBackend) -> bool:
    return bool(backend.scalar_to_float(value))


def _require_resident_pair_geometry(
    pair_geometry: MagneticPairGeometry, backend: ArrayBackend
) -> None:
    for name in (
        "ao_anchor_coordinates_au",
        "pair_midpoints_au",
        "pair_displacements_au",
    ):
        backend.assert_resident(getattr(pair_geometry, name), name=name)
