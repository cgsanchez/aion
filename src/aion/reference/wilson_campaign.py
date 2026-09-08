"""Reusable machinery for the H--H/O--H Wilson qualification campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from aion.gauge import AOAnchors, UniformMagneticGauge

from .wilson_grid import WilsonGridResult
from .wilson_measures import (
    MetricNotPositiveDefiniteError,
    atom_pair_block_error,
    compare_spectral_subspaces,
    elementwise_matrix_error,
    estimate_frobenius_floor,
    generalized_hermitian_spectrum,
    phase_spread,
)


FIELD_AU_TO_TESLA = 2.35051757077e5
DEFAULT_GAUGE_ORIGIN = np.array([0.17, -0.31, 0.23])
DEFAULT_NORMALIZED_THRESHOLDS = (1.0e-4, 1.0e-3, 1.0e-2)
DEFAULT_SPECTRAL_THRESHOLDS_HARTREE = (1.0e-4, 1.0e-3, 1.0e-2)
DEFAULT_CLUSTER_GAP_HARTREE = 1.0e-5
MATRIX_NAMES = ("overlap", "kinetic", "potential", "mechanical")


@dataclass(frozen=True)
class PairConfiguration:
    """A fixed two-center framework and AO basis."""

    system: str
    bond_length_bohr: float
    basis: str
    contracted: bool = True

    @property
    def identifier(self) -> str:
        contraction = "contracted" if self.contracted else "uncontracted"
        return "__".join(
            (
                self.system.lower().replace("-", ""),
                f"r{self.bond_length_bohr:.6f}".replace(".", "p"),
                self.basis.lower().replace("-", "_").replace("*", "star"),
                contraction,
            )
        )


@dataclass(frozen=True)
class CampaignCase:
    """One signed field point for a pair configuration."""

    configuration: PairConfiguration
    orientation: str
    field_au: float
    grid_level: int

    @property
    def identifier(self) -> str:
        return "__".join(
            (
                self.configuration.identifier,
                self.orientation,
                _float_label(self.field_au),
                f"grid{self.grid_level}",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["configuration_id"] = self.configuration.identifier
        payload["case_id"] = self.identifier
        payload["field_tesla"] = self.field_au * FIELD_AU_TO_TESLA
        return payload


@dataclass(frozen=True)
class ThresholdBracket:
    """First resolved coarse bracket for one declared physical threshold."""

    threshold: float
    lower_field_au: float
    lower_value: float
    upper_field_au: float
    upper_value: float
    adaptive_field_au: float


def campaign_case_from_dict(payload: dict[str, Any]) -> CampaignCase:
    """Reconstruct a case from a manifest entry."""

    configuration_payload = payload.get("configuration")
    if not isinstance(configuration_payload, dict):
        configuration_payload = {
            key: payload[key]
            for key in ("system", "bond_length_bohr", "basis", "contracted")
        }
    configuration = PairConfiguration(
        system=str(configuration_payload["system"]),
        bond_length_bohr=float(configuration_payload["bond_length_bohr"]),
        basis=str(configuration_payload["basis"]),
        contracted=bool(configuration_payload.get("contracted", True)),
    )
    return CampaignCase(
        configuration=configuration,
        orientation=str(payload["orientation"]),
        field_au=float(payload["field_au"]),
        grid_level=int(payload["grid_level"]),
    )


def accepted_pair_configurations() -> tuple[PairConfiguration, ...]:
    """Return the distinct configurations in the accepted Milestone-4 scan."""

    configurations: list[PairConfiguration] = []
    reference_distances = {"H-H": 1.4, "O-H": 1.8}
    extra_distances = {"H-H": (1.0, 2.0), "O-H": (1.4, 2.4)}
    bases = ("sto-3g", "cc-pvdz", "aug-cc-pvdz")
    for system, reference_distance in reference_distances.items():
        configurations.extend(
            PairConfiguration(system, reference_distance, basis) for basis in bases
        )
        configurations.extend(
            PairConfiguration(system, distance, "sto-3g")
            for distance in extra_distances[system]
        )
        configurations.append(
            PairConfiguration(
                system,
                reference_distance,
                "cc-pvdz",
                contracted=False,
            )
        )
    return tuple(configurations)


def signed_decade_fields() -> tuple[float, ...]:
    """Return zero and matching signed decades from ``1e-7`` through ``1``."""

    positive = np.logspace(-7.0, 0.0, 8)
    return tuple(np.concatenate((-positive[::-1], np.array([0.0]), positive)))


def accepted_scan_cases(*, grid_level: int = 2) -> tuple[CampaignCase, ...]:
    """Return the structured accepted scan without redundant Cartesian axes."""

    reference_distances = {"H-H": 1.4, "O-H": 1.8}
    cases: list[CampaignCase] = []
    for configuration in accepted_pair_configurations():
        is_reference_distance = np.isclose(
            configuration.bond_length_bohr,
            reference_distances[configuration.system],
        )
        if is_reference_distance and configuration.contracted:
            orientations = ("parallel", "perpendicular_1", "perpendicular_2", "oblique")
        else:
            orientations = ("oblique",)
        for orientation in orientations:
            for field in signed_decade_fields():
                cases.append(
                    CampaignCase(
                        configuration=configuration,
                        orientation=orientation,
                        field_au=float(field),
                        grid_level=int(grid_level),
                    )
                )
    return tuple(cases)


def pair_geometry(configuration: PairConfiguration) -> tuple[list[tuple[str, np.ndarray]], int]:
    """Return a displaced, non-axis-aligned two-center geometry and spin."""

    if configuration.system not in {"H-H", "O-H"}:
        raise ValueError("system must be 'H-H' or 'O-H'")
    if not np.isfinite(configuration.bond_length_bohr) or configuration.bond_length_bohr <= 0:
        raise ValueError("bond_length_bohr must be positive and finite")
    first = np.array([0.30, -0.25, 0.40])
    bond_direction = _normalize(np.array([1.30, 0.68, -0.61]))
    second = first + configuration.bond_length_bohr * bond_direction
    symbols = ("H", "H") if configuration.system == "H-H" else ("O", "H")
    spin = 0 if configuration.system == "H-H" else 1
    return [(symbols[0], first), (symbols[1], second)], spin


def build_pair_molecule(configuration: PairConfiguration):
    """Construct the PySCF molecule for a campaign configuration."""

    try:
        from pyscf import gto
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError("PySCF is required for the Wilson pair campaign") from exc

    atoms, spin = pair_geometry(configuration)
    basis: str | dict[str, Any]
    if configuration.contracted:
        basis = configuration.basis
    else:
        basis = {
            symbol: gto.uncontract(gto.basis.load(configuration.basis, symbol))
            for symbol in sorted({symbol for symbol, _ in atoms})
        }
    return gto.M(
        atom=atoms,
        basis=basis,
        unit="Bohr",
        spin=spin,
        verbose=0,
    )


def field_directions(mol: Any) -> dict[str, np.ndarray]:
    """Return parallel, two perpendicular, and generic oblique unit vectors."""

    atom_coords = np.asarray(mol.atom_coords(unit="Bohr"), dtype=float)
    if atom_coords.shape != (2, 3):
        raise ValueError("pair campaign requires exactly two centers")
    parallel = _normalize(atom_coords[1] - atom_coords[0])
    seed = _normalize(np.array([0.37, -0.52, 0.77]))
    perpendicular_1 = _normalize(np.cross(parallel, seed))
    perpendicular_2 = _normalize(np.cross(parallel, perpendicular_1))
    oblique = _normalize(
        0.61 * parallel + 0.52 * perpendicular_1 - 0.33 * perpendicular_2
    )
    return {
        "parallel": parallel,
        "perpendicular_1": perpendicular_1,
        "perpendicular_2": perpendicular_2,
        "oblique": oblique,
    }


def gauge_for_case(case: CampaignCase, mol: Any) -> UniformMagneticGauge:
    """Construct the fixed symmetric gauge used by the accepted scan."""

    directions = field_directions(mol)
    if case.orientation not in directions:
        raise ValueError(f"unknown field orientation {case.orientation!r}")
    return UniformMagneticGauge(
        case.field_au * directions[case.orientation],
        gauge="symmetric",
        origin=DEFAULT_GAUGE_ORIGIN,
    )


def ao_angular_momenta(mol: Any) -> np.ndarray:
    """Return the Gaussian shell angular momentum associated with every AO."""

    locations = np.asarray(mol.ao_loc_nr(), dtype=int)
    angular = np.empty(mol.nao_nr(), dtype=int)
    for shell in range(mol.nbas):
        angular[locations[shell] : locations[shell + 1]] = int(mol.bas_angular(shell))
    return angular


def configuration_metadata(configuration: PairConfiguration, mol: Any) -> dict[str, Any]:
    """Return serializable geometry, anchor, and AO-channel metadata."""

    anchors = AOAnchors.from_mol(mol)
    return {
        **asdict(configuration),
        "configuration_id": configuration.identifier,
        "geometry_bohr": np.asarray(mol.atom_coords(unit="Bohr")).tolist(),
        "atom_symbols": [mol.atom_symbol(index) for index in range(mol.natm)],
        "ao_to_atom": anchors.ao_to_atom.tolist(),
        "ao_angular_momenta": ao_angular_momenta(mol).tolist(),
        "ao_labels": list(mol.ao_labels()),
        "nao": int(mol.nao_nr()),
        "natom": int(mol.natm),
    }


def build_numerical_floor_record(
    coarse: WilsonGridResult,
    fine: WilsonGridResult,
    mol: Any,
) -> dict[str, Any]:
    """Build the declared zero-field/refinement floor policy for one configuration."""

    if np.linalg.norm(coarse.gauge.magnetic_field) != 0.0:
        raise ValueError("coarse floor result must be at zero field")
    if np.linalg.norm(fine.gauge.magnetic_field) != 0.0:
        raise ValueError("fine floor result must be at zero field")
    mapping = AOAnchors.from_mol(mol).ao_to_atom
    matrix_floors: dict[str, Any] = {}
    for name in MATRIX_NAMES:
        coarse_matrix = getattr(coarse.barred, name)
        fine_matrix = getattr(fine.barred, name)
        reference = getattr(fine.bare, name)
        frobenius = estimate_frobenius_floor(
            coarse_matrix,
            fine_matrix,
            analytic_reference=reference,
        )
        element_absolute = max(
            float(np.max(np.abs(fine_matrix - reference))),
            float(np.max(np.abs(fine_matrix - coarse_matrix))),
            _machine_matrix_floor(reference),
        )
        diagonal_scale = np.sqrt(
            np.abs(np.diag(reference))[:, None]
            * np.abs(np.diag(reference))[None, :]
        )
        diagonal_denominator = np.maximum(diagonal_scale, element_absolute)
        diagonal_scaled = max(
            float(np.max(np.abs(fine_matrix - reference) / diagonal_denominator)),
            float(np.max(np.abs(fine_matrix - coarse_matrix) / diagonal_denominator)),
        )
        block_floors: dict[str, Any] = {}
        for atom_a in range(mol.natm):
            for atom_b in range(mol.natm):
                rows = np.flatnonzero(mapping == atom_a)
                columns = np.flatnonzero(mapping == atom_b)
                index = np.ix_(rows, columns)
                reference_block = reference[index]
                absolute = max(
                    float(np.linalg.norm((fine_matrix - reference)[index], ord="fro")),
                    float(np.linalg.norm((fine_matrix - coarse_matrix)[index], ord="fro")),
                    _machine_matrix_floor(reference_block),
                )
                reference_norm = float(np.linalg.norm(reference_block, ord="fro"))
                block_floors[f"{atom_a}-{atom_b}"] = {
                    "absolute_frobenius": absolute,
                    "relative_frobenius": absolute / max(reference_norm, absolute),
                }
        matrix_floors[name] = {
            "element_absolute": element_absolute,
            "diagonal_scaled": diagonal_scaled,
            "frobenius_absolute": max(frobenius.value, _machine_matrix_floor(reference)),
            "refinement_change_frobenius": frobenius.refinement_change,
            "analytic_error_frobenius": frobenius.analytic_error,
            "blocks": block_floors,
        }

    denominator_change = float(
        np.max(
            np.abs(
                fine.phase_spread_moments.denominator
                - coarse.phase_spread_moments.denominator
            )
        )
    )
    denominator_scale = float(np.max(fine.phase_spread_moments.denominator))
    denominator_floor = max(
        denominator_change,
        512.0 * np.finfo(float).eps * max(1.0, denominator_scale),
    )

    spectrum_metric_floor = matrix_floors["overlap"]["frobenius_absolute"]
    coarse_spectrum = _spectrum_pair(coarse, metric_floor=spectrum_metric_floor)
    fine_spectrum = _spectrum_pair(fine, metric_floor=spectrum_metric_floor)
    spectral_floor = max(
        float(np.max(np.abs(fine_spectrum["exact"] - fine_spectrum["p0"]))),
        float(np.max(np.abs(fine_spectrum["exact"] - coarse_spectrum["exact"]))),
        512.0
        * np.finfo(float).eps
        * max(1.0, float(np.max(np.abs(fine_spectrum["p0"])))),
    )
    return {
        "coarse_grid": asdict(coarse.grid),
        "fine_grid": asdict(fine.grid),
        "matrices": matrix_floors,
        "phase_denominator_floor": denominator_floor,
        "phase_denominator_refinement_change": denominator_change,
        "spectral_absolute_hartree": spectral_floor,
        "normalized_thresholds": list(DEFAULT_NORMALIZED_THRESHOLDS),
        "spectral_thresholds_hartree": list(DEFAULT_SPECTRAL_THRESHOLDS_HARTREE),
        "cluster_gap_hartree": DEFAULT_CLUSTER_GAP_HARTREE,
    }


def summarize_case(
    result: WilsonGridResult,
    mol: Any,
    floor_record: dict[str, Any],
) -> dict[str, Any]:
    """Return all scalar measures for one exact-versus-P0 field point."""

    mapping = AOAnchors.from_mol(mol).ao_to_atom
    angular = ao_angular_momenta(mol)
    matrices: dict[str, Any] = {}
    for name in MATRIX_NAMES:
        exact = getattr(result.barred, name)
        reference = getattr(result.bare, name)
        floors = floor_record["matrices"][name]
        element = elementwise_matrix_error(
            exact,
            reference,
            floor=floors["element_absolute"],
        )
        delta = exact - reference
        reference_norm = float(np.linalg.norm(reference, ord="fro"))
        blocks: dict[str, Any] = {}
        for atom_a in range(mol.natm):
            for atom_b in range(mol.natm):
                key = f"{atom_a}-{atom_b}"
                block = atom_pair_block_error(
                    exact,
                    reference,
                    mapping,
                    atom_a,
                    atom_b,
                    floor=floors["blocks"][key]["absolute_frobenius"],
                )
                blocks[key] = {
                    "absolute_frobenius": block.absolute_frobenius,
                    "relative_frobenius": block.relative_frobenius,
                    "difference_singular_values": block.difference_singular_values.tolist(),
                    "floor_absolute_frobenius": floors["blocks"][key]["absolute_frobenius"],
                    "floor_relative_frobenius": floors["blocks"][key]["relative_frobenius"],
                }
        matrices[name] = {
            "element_max_absolute": float(np.max(element.absolute)),
            "element_max_relative_defined": _nanmax_or_none(element.relative),
            "element_max_diagonal_scaled": float(np.max(element.diagonal_scaled)),
            "floor_element_absolute": floors["element_absolute"],
            "floor_diagonal_scaled": floors["diagonal_scaled"],
            "frobenius_absolute": float(np.linalg.norm(delta, ord="fro")),
            "frobenius_relative": float(np.linalg.norm(delta, ord="fro"))
            / max(reference_norm, floors["frobenius_absolute"]),
            "floor_frobenius_absolute": floors["frobenius_absolute"],
            "blocks": blocks,
            "channels": _channel_measures(element, mapping, angular),
        }

    spread = phase_spread(
        result.phase_spread_moments,
        denominator_floor=floor_record["phase_denominator_floor"],
    )
    phase_summary = {
        "maximum": _nanmax_or_none(spread.values),
        "defined_count": int(np.count_nonzero(spread.defined)),
        "undefined_count": int(spread.defined.size - np.count_nonzero(spread.defined)),
        "denominator_floor": spread.denominator_floor,
        "blocks": _phase_block_measures(spread.values, spread.defined, mapping),
    }

    diagnostics: dict[str, Any] = {}
    for diagnostic_name in ("form_factor_only", "anchored_vector_only", "full_exact"):
        calculation = getattr(result.diagnostics, diagnostic_name)
        diagnostic_matrices: dict[str, Any] = {}
        for matrix_name in MATRIX_NAMES:
            value = getattr(calculation.barred, matrix_name)
            reference = getattr(result.bare, matrix_name)
            delta = value - reference
            matrix_floor = floor_record["matrices"][matrix_name]["frobenius_absolute"]
            reference_norm = float(np.linalg.norm(reference, ord="fro"))
            diagnostic_matrices[matrix_name] = {
                "frobenius_absolute": float(np.linalg.norm(delta, ord="fro")),
                "frobenius_relative": float(np.linalg.norm(delta, ord="fro"))
                / max(reference_norm, matrix_floor),
            }
        diagnostics[diagnostic_name] = diagnostic_matrices

    sectors = result.kinetic_sectors
    kinetic_sectors = {
        "pp_minus_t0_frobenius": float(np.linalg.norm(sectors.pp - result.T0, ord="fro")),
        "p_c_frobenius": float(np.linalg.norm(sectors.p_c, ord="fro")),
        "c_p_frobenius": float(np.linalg.norm(sectors.c_p, ord="fro")),
        "c_c_frobenius": float(np.linalg.norm(sectors.c_c, ord="fro")),
    }

    spectrum = _spectral_summary(
        result,
        metric_floor=max(
            floor_record["matrices"]["overlap"]["frobenius_absolute"],
            _machine_matrix_floor(result.S0),
        ),
        spectral_floor=floor_record["spectral_absolute_hartree"],
        cluster_gap=floor_record["cluster_gap_hartree"],
    )
    return {
        "matrices": matrices,
        "phase_spread": phase_summary,
        "diagnostics": diagnostics,
        "kinetic_sectors": kinetic_sectors,
        "spectrum": spectrum,
    }


def result_arrays(result: WilsonGridResult) -> dict[str, np.ndarray]:
    """Return the compact raw matrix payload stored for every campaign case."""

    arrays: dict[str, np.ndarray] = {
        "theta": result.theta,
        "ao_anchors": result.ao_anchors,
        "phase_spread_numerator": result.phase_spread_moments.numerator,
        "phase_spread_denominator": result.phase_spread_moments.denominator,
    }
    for prefix, matrices in (
        ("bare", result.bare),
        ("exact_barred", result.barred),
        ("exact_lower", result.factorized),
        ("p0_barred", result.diagnostics.p0.barred),
        ("p0_lower", result.diagnostics.p0.lower),
        ("form_factor_barred", result.diagnostics.form_factor_only.barred),
        ("anchored_vector_barred", result.diagnostics.anchored_vector_only.barred),
    ):
        for name in MATRIX_NAMES:
            arrays[f"{prefix}_{name}"] = getattr(matrices, name)
    for name, sector in (
        ("pp", result.kinetic_sectors.pp),
        ("p_c", result.kinetic_sectors.p_c),
        ("c_p", result.kinetic_sectors.c_p),
        ("c_c", result.kinetic_sectors.c_c),
    ):
        arrays[f"exact_sector_{name}"] = sector
    anchored = result.diagnostics.anchored_vector_only.kinetic_sectors
    for name, sector in (
        ("pp", anchored.pp),
        ("p_c", anchored.p_c),
        ("c_p", anchored.c_p),
        ("c_c", anchored.c_c),
    ):
        arrays[f"anchored_no_flux_sector_{name}"] = sector
    return arrays


def paired_field_parity(
    positive: dict[str, np.ndarray],
    negative: dict[str, np.ndarray],
    zero: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Return odd/even decomposition and time-reversal residuals for raw cases."""

    report: dict[str, Any] = {}
    for name in MATRIX_NAMES:
        key = f"exact_barred_{name}"
        plus = positive[key]
        minus = negative[key]
        origin = zero[key]
        odd = 0.5 * (plus - minus)
        even = 0.5 * (plus + minus) - origin
        report[name] = {
            "odd_frobenius": float(np.linalg.norm(odd, ord="fro")),
            "even_frobenius": float(np.linalg.norm(even, ord="fro")),
            "time_reversal_residual": float(
                np.linalg.norm(minus - plus.conj(), ord="fro")
            ),
        }
    return report


def threshold_metric_values(summary: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Return declared threshold metrics as ``value, numerical_floor`` pairs."""

    metrics: dict[str, tuple[float, float]] = {}
    phase_maximum = summary["phase_spread"]["maximum"]
    if phase_maximum is not None:
        metrics["phase_spread.maximum"] = (float(phase_maximum), 0.0)
    for name in MATRIX_NAMES:
        matrix = summary["matrices"][name]
        metrics[f"{name}.element_max_diagonal_scaled"] = (
            matrix["element_max_diagonal_scaled"],
            matrix["floor_diagonal_scaled"],
        )
        intersite = matrix["blocks"]["0-1"]
        metrics[f"{name}.intersite_relative_frobenius"] = (
            intersite["relative_frobenius"],
            intersite["floor_relative_frobenius"],
        )
    if summary["spectrum"]["defined"]:
        metrics["spectrum.maximum_absolute_shift_hartree"] = (
            summary["spectrum"]["maximum_absolute_shift_hartree"],
            summary["spectrum"]["floor_absolute_hartree"],
        )
    return metrics


def first_threshold_bracket(
    samples: Iterable[tuple[float, float, float]],
    *,
    threshold: float,
) -> ThresholdBracket | None:
    """Return the first positive-field crossing above floor and threshold."""

    declared = float(threshold)
    if not np.isfinite(declared) or declared <= 0.0:
        raise ValueError("threshold must be positive and finite")
    ordered = sorted(
        (
            (float(field), float(value), float(floor))
            for field, value, floor in samples
            if field >= 0.0
            and np.isfinite(field)
            and np.isfinite(value)
            and np.isfinite(floor)
        ),
        key=lambda item: item[0],
    )
    if not ordered or ordered[0][0] != 0.0:
        raise ValueError("threshold samples must include the zero-field point")
    for index in range(1, len(ordered)):
        upper_field, upper_value, upper_floor = ordered[index]
        if upper_value >= declared and upper_value > upper_floor:
            lower_field, lower_value, _ = ordered[index - 1]
            return ThresholdBracket(
                threshold=declared,
                lower_field_au=lower_field,
                lower_value=lower_value,
                upper_field_au=upper_field,
                upper_value=upper_value,
                adaptive_field_au=adaptive_log_field(
                    lower_field,
                    upper_field,
                    lower_value,
                    upper_value,
                    declared,
                ),
            )
    return None


def adaptive_log_field(
    lower_field: float,
    upper_field: float,
    lower_value: float,
    upper_value: float,
    threshold: float,
) -> float:
    """Interpolate one extra field inside a detected logarithmic bracket."""

    lower_field = float(lower_field)
    upper_field = float(upper_field)
    lower_value = float(lower_value)
    upper_value = float(upper_value)
    threshold = float(threshold)
    if upper_field <= 0.0 or lower_field < 0.0 or lower_field >= upper_field:
        raise ValueError("require 0 <= lower_field < upper_field")
    if lower_field == 0.0:
        return upper_field / np.sqrt(10.0)
    if lower_value <= 0.0 or upper_value <= lower_value or threshold <= 0.0:
        return float(np.sqrt(lower_field * upper_field))
    fraction = (
        (np.log(threshold) - np.log(lower_value))
        / (np.log(upper_value) - np.log(lower_value))
    )
    fraction = float(np.clip(fraction, 0.1, 0.9))
    return float(np.exp(np.log(lower_field) + fraction * np.log(upper_field / lower_field)))


def _spectral_summary(
    result: WilsonGridResult,
    *,
    metric_floor: float,
    spectral_floor: float,
    cluster_gap: float,
) -> dict[str, Any]:
    try:
        exact = generalized_hermitian_spectrum(
            result.diagnostics.full_exact.lower.mechanical,
            result.diagnostics.full_exact.lower.overlap,
            metric_floor=metric_floor,
        )
        p0 = generalized_hermitian_spectrum(
            result.diagnostics.p0.lower.mechanical,
            result.diagnostics.p0.lower.overlap,
            metric_floor=metric_floor,
        )
    except MetricNotPositiveDefiniteError as exc:
        return {
            "defined": False,
            "reason": str(exc),
            "metric_floor": metric_floor,
            "floor_absolute_hartree": spectral_floor,
        }

    shifts = exact.eigenvalues - p0.eigenvalues
    largest_angle = 0.0
    cluster_records: list[dict[str, Any]] = []
    for indices in _joint_eigenvalue_clusters(
        exact.eigenvalues,
        p0.eigenvalues,
        gap=cluster_gap,
    ):
        comparison = compare_spectral_subspaces(
            exact.eigenvectors[:, indices],
            p0.eigenvectors[:, indices],
            result.S0,
            metric_floor=_machine_matrix_floor(result.S0),
        )
        largest_angle = max(largest_angle, comparison.largest_principal_angle)
        cluster_records.append(
            {
                "indices": indices.tolist(),
                "largest_principal_angle": comparison.largest_principal_angle,
                "chordal_distance": comparison.chordal_distance,
            }
        )
    return {
        "defined": True,
        "exact_eigenvalues_hartree": exact.eigenvalues.tolist(),
        "p0_eigenvalues_hartree": p0.eigenvalues.tolist(),
        "eigenvalue_shifts_hartree": shifts.tolist(),
        "maximum_absolute_shift_hartree": float(np.max(np.abs(shifts))),
        "rms_shift_hartree": float(np.sqrt(np.mean(shifts**2))),
        "floor_absolute_hartree": spectral_floor,
        "exact_metric_minimum_eigenvalue": float(exact.metric_eigenvalues[0]),
        "p0_metric_minimum_eigenvalue": float(p0.metric_eigenvalues[0]),
        "exact_metric_condition_number": exact.metric_condition_number,
        "p0_metric_condition_number": p0.metric_condition_number,
        "cluster_gap_hartree": cluster_gap,
        "clusters": cluster_records,
        "maximum_principal_angle": largest_angle,
        "comparison_metric": "bare_S0",
    }


def _spectrum_pair(result: WilsonGridResult, *, metric_floor: float) -> dict[str, np.ndarray]:
    exact = generalized_hermitian_spectrum(
        result.diagnostics.full_exact.lower.mechanical,
        result.diagnostics.full_exact.lower.overlap,
        metric_floor=metric_floor,
    )
    p0 = generalized_hermitian_spectrum(
        result.diagnostics.p0.lower.mechanical,
        result.diagnostics.p0.lower.overlap,
        metric_floor=metric_floor,
    )
    return {"exact": exact.eigenvalues, "p0": p0.eigenvalues}


def _joint_eigenvalue_clusters(
    exact: np.ndarray,
    reference: np.ndarray,
    *,
    gap: float,
) -> tuple[np.ndarray, ...]:
    if exact.shape != reference.shape:
        raise ValueError("exact and reference spectra must have matching shapes")
    clusters: list[np.ndarray] = []
    start = 0
    for index in range(1, exact.size):
        if min(exact[index] - exact[index - 1], reference[index] - reference[index - 1]) > gap:
            clusters.append(np.arange(start, index, dtype=int))
            start = index
    clusters.append(np.arange(start, exact.size, dtype=int))
    return tuple(clusters)


def _channel_measures(element: Any, mapping: np.ndarray, angular: np.ndarray) -> dict[str, Any]:
    row_atoms = mapping[:, None]
    column_atoms = mapping[None, :]
    same_anchor = row_atoms == column_atoms
    diagonal = np.eye(mapping.size, dtype=bool)
    masks: dict[str, np.ndarray] = {
        "onsite_diagonal": diagonal,
        "same_anchor_offdiagonal": same_anchor & ~diagonal,
        "intersite_all": ~same_anchor,
    }
    for location, location_mask in (
        ("same_anchor", same_anchor),
        ("intersite", ~same_anchor),
    ):
        masks[f"{location}_s-s"] = location_mask & (angular[:, None] == 0) & (
            angular[None, :] == 0
        )
        masks[f"{location}_s-p"] = location_mask & (
            ((angular[:, None] == 0) & (angular[None, :] == 1))
            | ((angular[:, None] == 1) & (angular[None, :] == 0))
        )
        masks[f"{location}_p-p"] = location_mask & (angular[:, None] == 1) & (
            angular[None, :] == 1
        )
    return {
        name: {
            "count": int(np.count_nonzero(mask)),
            "maximum_absolute": _masked_max(element.absolute, mask),
            "maximum_diagonal_scaled": _masked_max(element.diagonal_scaled, mask),
        }
        for name, mask in masks.items()
    }


def _phase_block_measures(
    values: np.ndarray,
    defined: np.ndarray,
    mapping: np.ndarray,
) -> dict[str, Any]:
    blocks: dict[str, Any] = {}
    for atom_a in range(int(np.max(mapping)) + 1):
        for atom_b in range(int(np.max(mapping)) + 1):
            mask = (mapping[:, None] == atom_a) & (mapping[None, :] == atom_b) & defined
            blocks[f"{atom_a}-{atom_b}"] = {
                "defined_count": int(np.count_nonzero(mask)),
                "maximum": _masked_max(values, mask),
            }
    return blocks


def _masked_max(values: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    selected = np.asarray(values)[mask]
    finite = selected[np.isfinite(selected)]
    return None if finite.size == 0 else float(np.max(finite))


def _nanmax_or_none(values: np.ndarray) -> float | None:
    finite = np.asarray(values)[np.isfinite(values)]
    return None if finite.size == 0 else float(np.max(finite))


def _machine_matrix_floor(matrix: np.ndarray) -> float:
    array = np.asarray(matrix)
    return float(
        512.0
        * np.finfo(float).eps
        * max(array.shape, default=1)
        * max(1.0, float(np.linalg.norm(array, ord="fro")))
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(array))
    if array.shape != (3,) or norm == 0.0 or not np.isfinite(norm):
        raise ValueError("vector must be a finite nonzero Cartesian vector")
    return array / norm


def _float_label(value: float) -> str:
    if value == 0.0:
        return "b0"
    sign = "p" if value > 0.0 else "m"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = abs(value) / (10.0**exponent)
    return f"b{sign}{mantissa:.8f}e{exponent:+03d}".replace(".", "p")


def unique_configurations(cases: Iterable[CampaignCase]) -> tuple[PairConfiguration, ...]:
    """Return configurations in first-occurrence order."""

    result: list[PairConfiguration] = []
    seen: set[str] = set()
    for case in cases:
        if case.configuration.identifier not in seen:
            result.append(case.configuration)
            seen.add(case.configuration.identifier)
    return tuple(result)
