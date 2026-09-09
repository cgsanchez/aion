from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import aion
from aion.cli import main
from aion.config import (
    AtomConfig,
    ElectronicStructureConfig,
    MoleculeConfig,
    ReferenceConfig,
    XCFamily,
    dumps_config,
)

pytestmark = pytest.mark.fast


def test_public_api_is_small_and_versioned() -> None:
    assert aion.__version__ == "0.2.0.dev7"
    assert set(aion.__all__) == {
        "BackendConfig",
        "FormulationConfig",
        "ReferenceConfig",
        "SimulationConfig",
        "__version__",
        "build_simulation",
        "load_config",
        "load_reference",
        "load_trajectory",
        "prepare_reference",
        "resume",
        "run",
    }
    with pytest.raises(TypeError, match="ReferenceConfig"):
        aion.prepare_reference(None)  # type: ignore[arg-type]


def test_import_has_no_pyscf_or_gpu_side_effects() -> None:
    code = (
        "import sys; import aion; import aion.spectroscopy; "
        "assert 'pyscf' not in sys.modules; "
        "assert 'cupy' not in sys.modules; "
        "assert 'gpu4pyscf' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cli_exposes_all_planned_commands() -> None:
    for command in ("prepare", "run", "resume", "inspect", "export"):
        with pytest.raises(SystemExit) as result:
            main([command, "--help"])
        assert result.value.code == 0


def test_cli_validate_only_materializes_configuration_and_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = ReferenceConfig(
        molecule=MoleculeConfig(atoms=(AtomConfig("He", (0.0, 0.0, 0.0)),)),
        electronic_structure=ElectronicStructureConfig(
            basis="cc-pvdz",
            functional="lda,vwn",
            xc_family=XCFamily.LDA,
        ),
    )
    path = tmp_path / "reference.toml"
    path.write_text(dumps_config(config), encoding="utf-8")
    assert main(["prepare", str(path), "--validate-only"]) == 0
    output = capsys.readouterr().out
    assert 'schema = "aion.reference-input"' in output
    assert f"scientific_id = {config.scientific_id}" in output
