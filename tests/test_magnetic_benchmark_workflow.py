from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from aion.config import BackendConfig
from aion.electromagnetism import UniformMagneticField
from aion.electronic_structure import (
    NuclearAttractionProvider,
    estimate_magnetic_block_bytes,
    prepare_pyscf_reference,
)
from aion.errors import ConfigurationError, MagneticBenchmarkError, SchemaError
from aion.io import MAGNETIC_BENCHMARK_SCHEMA, validate_artifact
from aion.io.util import file_sha256
from aion.propagation import SCEMPropagator
from aion.workflows import (
    MagneticBenchmarkConfig,
    MagneticMatrixFamily,
    MagneticValidationPolicy,
    inspect_magnetic_benchmark,
    load_magnetic_benchmark,
    run_magnetic_benchmark,
    save_magnetic_benchmark,
)
from test_reference_integration import molecular_config

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def h2_reference() -> object:
    return prepare_pyscf_reference(molecular_config("h2"))


def _config(**changes: object) -> MagneticBenchmarkConfig:
    values: dict[str, object] = {
        "magnetic_fields": (UniformMagneticField((0.011, -0.007, 0.005)),),
        "backend": BackendConfig(),
        "block_size": 4096,
    }
    values.update(changes)
    return MagneticBenchmarkConfig(**values)  # type: ignore[arg-type]


def test_config_is_normalized_and_semantically_hashed() -> None:
    left = _config(
        families=(
            MagneticMatrixFamily.SPATIAL_CONNECTION,
            MagneticMatrixFamily.ONE_ELECTRON,
            MagneticMatrixFamily.ONE_ELECTRON,
        )
    )
    right = _config()
    assert left.as_mapping() == right.as_mapping()
    assert left.scientific_id == right.scientific_id
    assert left.gauges[0].field == left.magnetic_fields[0]
    with pytest.raises(TypeError, match="PreparedReference"):
        run_magnetic_benchmark(  # type: ignore[arg-type]
            object(),
            right,
        )


def test_workflow_uses_only_the_prepared_reference_and_reports_progress(
    h2_reference: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyscf.dft.rks import RKS

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("magnetic workflow must not run SCF or propagation")

    monkeypatch.setattr(RKS, "kernel", forbidden)
    monkeypatch.setattr(SCEMPropagator, "step", forbidden)
    progress = []
    result = run_magnetic_benchmark(  # type: ignore[arg-type]
        h2_reference,
        _config(),
        progress=progress.append,
    )
    summary = inspect_magnetic_benchmark(result)
    assert summary["matrix_count"] == 61
    assert summary["failed_diagnostics"] == []
    assert result.matrix("0000/kinetic/exact").shape == (2, 2)
    assert not result.matrix("0000/kinetic/exact").flags.writeable
    assert progress[0].stage == "prepare" and progress[0].completed == 0
    assert progress[-1].stage == "finalize" and progress[-1].completed == 1
    stages = {value.stage for value in progress}
    assert {"prepare", "one_electron", "spatial_connection", "finalize"} <= stages


def test_artifact_round_trip_no_overwrite_and_tamper_detection(
    h2_reference: object,
    tmp_path: Path,
) -> None:
    result = run_magnetic_benchmark(h2_reference, _config())  # type: ignore[arg-type]
    path = tmp_path / "magnetic-benchmark.h5"
    checksum = save_magnetic_benchmark(result, path)
    assert checksum == file_sha256(path)
    header = validate_artifact(path, expected_schema=MAGNETIC_BENCHMARK_SCHEMA)
    loaded = load_magnetic_benchmark(path)
    assert header.artifact_id == loaded.result_id == result.result_id
    assert loaded.config.as_mapping() == result.config.as_mapping()
    assert tuple(value.path for value in loaded.matrices) == tuple(
        value.path for value in result.matrices
    )
    for expected, actual in zip(result.matrices, loaded.matrices, strict=True):
        assert actual.unit == expected.unit
        assert actual.physical_dimension == expected.physical_dimension
        assert actual.definition == expected.definition
        np.testing.assert_array_equal(actual.values, expected.values)
    assert loaded.diagnostics == result.diagnostics
    with pytest.raises(SchemaError, match="overwrite"):
        save_magnetic_benchmark(result, path)
    with h5py.File(path, "r+") as handle:
        handle["fields/0000/kinetic/exact"][0, 0] += 1.0e-5
    with pytest.raises(MagneticBenchmarkError, match="numerical content fingerprint"):
        load_magnetic_benchmark(path)


def test_local_provider_family_and_controlled_failure_are_explicit(
    h2_reference: object,
    tmp_path: Path,
) -> None:
    provider = NuclearAttractionProvider()
    repeated_field = UniformMagneticField((0.011, -0.007, 0.005))
    reversed_field = UniformMagneticField((-0.011, 0.007, -0.005))
    local = run_magnetic_benchmark(  # type: ignore[arg-type]
        h2_reference,
        _config(
            magnetic_fields=(repeated_field, reversed_field, repeated_field),
            families=(MagneticMatrixFamily.LOCAL_POTENTIALS,),
            local_potential_identities=(provider.identity,),
        ),
        local_potential_providers=(provider,),
    )
    assert any(value.path.startswith("0000/local/") for value in local.matrices)
    assert any(value.path.startswith("0002/local/") for value in local.matrices)
    reversal = next(
        value for value in local.diagnostics if value.name.startswith("field_reversal/0000_0001/")
    )
    assert reversal.passed is True
    norm = next(
        value
        for value in local.diagnostics
        if value.name.endswith("local/0000_nuclear_attraction/exact/frobenius")
    )
    assert norm.unit == "hartree"
    assert norm.physical_dimension == "energy_operator_norm"
    assert all(value.passed is not False for value in local.diagnostics)

    failure_path = tmp_path / "must-not-exist.h5"
    failing = _config(validation=MagneticValidationPolicy(direct_oracle_tolerance=1.0e-30))
    with pytest.raises(MagneticBenchmarkError, match="validation failed"):
        run_magnetic_benchmark(  # type: ignore[arg-type]
            h2_reference,
            failing,
            output_path=failure_path,
        )
    assert not failure_path.exists()


def test_workflow_memory_ceiling_reaches_every_selected_kernel(
    h2_reference: object,
) -> None:
    block_size = 128
    insufficient = estimate_magnetic_block_bytes(block_size, 2) - 1
    field = UniformMagneticField((0.011, -0.007, 0.005))
    provider = NuclearAttractionProvider()
    cases = (
        ((MagneticMatrixFamily.ONE_ELECTRON,), (), (), "magnetic block"),
        ((MagneticMatrixFamily.SPATIAL_CONNECTION,), (), (), "spatial-connection block"),
        (
            (MagneticMatrixFamily.LOCAL_POTENTIALS,),
            (provider.identity,),
            (provider,),
            "local magnetic block",
        ),
    )
    for families, identities, providers, message in cases:
        config = MagneticBenchmarkConfig(
            (field,),
            families=families,
            local_potential_identities=identities,
            block_size=block_size,
            memory_budget_bytes=insufficient,
        )
        with pytest.raises(ConfigurationError, match=message):
            run_magnetic_benchmark(  # type: ignore[arg-type]
                h2_reference,
                config,
                local_potential_providers=providers,
            )
