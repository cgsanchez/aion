from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aion.config import (
    FixedTimeGrid,
    FormulationConfig,
    FormulationKind,
    IntegratorKind,
    KickEventConfig,
    ObservableSchedules,
    OutputConfig,
    PropagationConfig,
    ReferenceLinkConfig,
    SimulationConfig,
    StepSchedule,
    ZeroSourceConfig,
)
from aion.electronic_structure import PreparedReference, prepare_pyscf_reference
from aion.io.trajectory import Trajectory
from aion.spectroscopy import (
    CasidaConfig,
    KickSpectrumConfig,
    TransformConfig,
    kick_spectrum_from_trajectory,
    load_casida_result,
    run_casida,
    save_casida_result,
    select_resonance,
)
from aion.workflows import build_simulation, run
from test_reference_integration import molecular_config

pytestmark = pytest.mark.integration


def _kick_config(reference: PreparedReference, output: Path) -> SimulationConfig:
    quiet = StepSchedule(every=0)
    return SimulationConfig(
        reference=ReferenceLinkConfig(
            reference.fingerprint_sha256,
            reference.config.output.artifact_path,
        ),
        formulation=FormulationConfig(FormulationKind.BARE_LENGTH_GAUGE),
        source=ZeroSourceConfig(),
        propagation=PropagationConfig(
            FixedTimeGrid(start_au=0.0, step_au=0.2, intervals=400),
            IntegratorKind.FIXED_METRIC_SCEM,
            density_tolerance=1.0e-10,
        ),
        events=(KickEventConfig("kick-z", 0, (0.0, 0.0, 1.0e-3)),),
        output=OutputConfig(
            output,
            ObservableSchedules(
                dipole_current=StepSchedule(every=1),
                energy=quiet,
                diagnostics=quiet,
                source=quiet,
                checkpoints=quiet,
            ),
        ),
    )


@pytest.fixture(scope="module")
def molecular_spectroscopy(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, tuple[PreparedReference, Trajectory]]:
    root = tmp_path_factory.mktemp("wp6-spectroscopy")
    outputs: dict[str, tuple[PreparedReference, Trajectory]] = {}
    for name in ("h2", "lih"):
        reference_path = root / f"{name}.reference.h5"
        reference = prepare_pyscf_reference(molecular_config(name, reference_path))
        reference.save()
        trajectory = run(build_simulation(_kick_config(reference, root / name), reference))
        outputs[name] = reference, trajectory
    return outputs


@pytest.mark.parametrize(
    ("name", "expected_root"),
    (("h2", 0), ("lih", 0)),
)
def test_h2_lih_casida_and_kick_spectra_locate_the_same_z_bright_excitation(
    molecular_spectroscopy: dict[str, tuple[PreparedReference, Trajectory]],
    tmp_path: Path,
    name: str,
    expected_root: int,
) -> None:
    reference, trajectory = molecular_spectroscopy[name]
    casida = run_casida(reference, CasidaConfig(nstates=6))
    selection = select_resonance(
        casida,
        [0.0, 0.0, 1.0],
        brightness_threshold=1.0e-5,
    )
    assert selection.selected_root == expected_root
    casida_path = tmp_path / f"{name}.casida.h5"
    save_casida_result(
        casida,
        casida_path,
        selection=selection,
        reference_artifact=reference.config.output.artifact_path,
    )
    loaded, loaded_selection = load_casida_result(casida_path)
    assert loaded.result_id == casida.result_id
    assert loaded_selection is not None
    assert loaded_selection.selected_root == selection.selected_root

    spectrum = kick_spectrum_from_trajectory(
        trajectory,
        KickSpectrumConfig(
            TransformConfig(
                damping_energy_au=0.03,
                zero_padding_factor=8,
                maximum_energy_au=1.4,
            )
        ),
    )
    target = casida.excitation_energy_au[selection.selected_root]
    local = np.abs(spectrum.omega_au - target) <= 0.12
    local_indices = np.flatnonzero(local)
    peak_index = local_indices[np.argmax(spectrum.absorption_strength_parallel_au[local])]
    assert spectrum.absorption_strength_parallel_au[peak_index] > 0.0
    assert spectrum.omega_au[peak_index] == pytest.approx(
        target,
        abs=spectrum.padded_grid_spacing_au,
    )

    # The area of (2/pi) omega Im(alpha_zz) is the polarized oscillator
    # strength.  A finite exponentially damped record retains it to the stated
    # compact-trajectory qualification tolerance.
    band_half_width = 0.13 if name == "lih" else 0.18
    band = np.abs(spectrum.omega_au - target) <= band_half_width
    integrated_strength = np.trapezoid(
        spectrum.oscillator_strength_density_parallel_au[band],
        x=spectrum.omega_au[band],
    )
    expected_strength = selection.projected_oscillator_strength[expected_root]
    assert integrated_strength == pytest.approx(expected_strength, rel=0.22)

    assert spectrum.polarizability_current_au is not None
    assert spectrum.current_domain_residual_au is not None
    residual_ratio = abs(spectrum.current_domain_residual_au[peak_index, 2]) / abs(
        spectrum.polarizability_dipole_au[peak_index, 2]
    )
    assert residual_ratio < 2.0e-3


def test_lih_polarized_selection_distinguishes_longitudinal_and_transverse_roots(
    molecular_spectroscopy: dict[str, tuple[PreparedReference, Trajectory]],
) -> None:
    reference, _trajectory = molecular_spectroscopy["lih"]
    result = run_casida(reference, CasidaConfig(nstates=6))
    longitudinal = select_resonance(
        result,
        [0.0, 0.0, 1.0],
        brightness_threshold=1.0e-5,
    )
    transverse = select_resonance(
        result,
        [1.0, 0.0, 0.0],
        brightness_threshold=1.0e-5,
    )
    assert longitudinal.selected_root == 0
    assert transverse.selected_root in (1, 2)
    assert (
        result.excitation_energy_au[longitudinal.selected_root]
        < result.excitation_energy_au[transverse.selected_root]
    )
