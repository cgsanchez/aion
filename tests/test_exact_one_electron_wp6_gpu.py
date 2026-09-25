from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aion.config import (
    AtomConfig,
    BackendConfig,
    BackendKind,
    ElectromagneticOrigin,
    OneElectronReferenceConfig,
)
from aion.electromagnetism import UniformMagneticField, UniformMagneticSourceSample
from aion.electronic_structure import (
    AOGridPolicy,
    evaluate_exact_wilson_one_electron_sample,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
)
from aion.formulations import exact_wilson_one_electron_triple
from aion.propagation import LinearMatrixHistory, propagate_linear_matrix_history

pytestmark = pytest.mark.gpu

_FIXTURE = Path(__file__).parent / "fixtures/exact_one_electron/hh_sto3g.fixture.json"


def _reference() -> object:
    values = json.loads(_FIXTURE.read_text(encoding="utf-8"))["config"]
    return prepare_one_electron_ao_reference(
        OneElectronReferenceConfig(
            atoms=tuple(
                AtomConfig(atom["symbol"], tuple(atom["position_au"])) for atom in values["atoms"]
            ),
            basis=values["basis"],
            electromagnetic_origin=ElectromagneticOrigin(
                tuple(values["electromagnetic_origin_au"])
            ),
        )
    )


def test_wp6_exact_time_triple_and_linear_propagation_cpu_gpu_parity() -> None:
    reference = _reference()
    cpu_quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(),
        grid_policy=AOGridPolicy.qualification(3),
        block_size=1024,
    )
    gpu_quadrature = prepare_ao_quadrature(
        reference,
        BackendConfig(BackendKind.GPU, device_index=0),
        grid_policy=AOGridPolicy.qualification(3),
        block_size=1024,
    )
    source = UniformMagneticSourceSample(
        0.41,
        UniformMagneticField((0.0, 0.0, 0.031)),
        magnetic_field_dot_au=(0.0, 0.0, -0.007),
        electric_field_origin_au=(0.013, -0.009, 0.017),
        origin_au=(0.11, -0.07, 0.05),
    )
    cpu = evaluate_exact_wilson_one_electron_sample(cpu_quadrature, source)
    gpu = evaluate_exact_wilson_one_electron_sample(gpu_quadrature, source)
    gpu_backend = gpu_quadrature.backend
    for cpu_value, gpu_value in (
        (cpu.metric, gpu.metric),
        (cpu.mechanical, gpu.mechanical),
        (cpu.connection.connection, gpu.connection.connection),
        (cpu.connection.metric_dot, gpu.connection.metric_dot),
        (
            cpu.connection.direct_connection_grid,
            gpu.connection.direct_connection_grid,
        ),
    ):
        gpu_backend.assert_resident(gpu_value, name="WP6 GPU matrix")
        np.testing.assert_allclose(
            gpu_backend.to_host(gpu_value),
            cpu_value,
            atol=2.0e-10,
            rtol=2.0e-10,
        )

    static_source = UniformMagneticSourceSample(
        0.0,
        UniformMagneticField((0.0, 0.0, 0.043)),
    )
    cpu_static = evaluate_exact_wilson_one_electron_sample(cpu_quadrature, static_source)
    gpu_static = evaluate_exact_wilson_one_electron_sample(gpu_quadrature, static_source)
    initial_host = np.asarray(((1.0,), (0.31 + 0.17j,)), dtype=np.complex128)
    initial_host /= np.sqrt((initial_host.conj().T @ cpu_static.metric @ initial_host).real.item())
    cpu_triple = exact_wilson_one_electron_triple(cpu_static)
    gpu_triple = exact_wilson_one_electron_triple(gpu_static)
    cpu_history = LinearMatrixHistory(
        endpoint_metrics=(cpu_static.metric,) * 5,
        midpoint_triples=(cpu_triple,) * 4,
        interval_au=0.05,
    )
    gpu_history = LinearMatrixHistory(
        endpoint_metrics=(gpu_static.metric,) * 5,
        midpoint_triples=(gpu_triple,) * 4,
        interval_au=0.05,
    )
    cpu_trajectory = propagate_linear_matrix_history(
        cpu_quadrature.backend.asarray(initial_host),
        cpu_history,
        cpu_quadrature.backend,
    )
    gpu_trajectory = propagate_linear_matrix_history(
        gpu_backend.asarray(initial_host),
        gpu_history,
        gpu_backend,
    )
    for value in gpu_trajectory.coefficients:
        gpu_backend.assert_resident(value, name="WP6 GPU coefficients")
    np.testing.assert_allclose(
        gpu_backend.to_host(gpu_trajectory.coefficients[-1]),
        cpu_trajectory.coefficients[-1],
        atol=2.0e-10,
        rtol=2.0e-10,
    )
