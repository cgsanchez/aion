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
    evaluate_exact_magnetic_field_source_direction,
    evaluate_exact_temporal_source_direction,
    evaluate_exact_wilson_one_electron_sample,
    prepare_ao_quadrature,
    prepare_one_electron_ao_reference,
)
from aion.formulations import (
    OneElectronActionMatrixDirection,
    exact_endpoint_link_action_direction,
    exact_magnetic_field_action_direction,
    exact_pure_gauge_action_direction,
    exact_site_scalar_action_direction,
    exact_wilson_one_electron_triple,
    one_electron_velocity_density,
    restricted_one_electron_action_directional_derivative,
    restricted_one_electron_action_full_directional_derivative,
)

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


def test_wp7_exact_temporal_source_action_derivative_cpu_gpu_parity() -> None:
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
        magnetic_field_dot_au=(0.003, -0.004, 0.007),
        electric_field_origin_au=(0.013, -0.009, 0.017),
        origin_au=(0.11, -0.07, 0.05),
    )
    cpu_sample = evaluate_exact_wilson_one_electron_sample(cpu_quadrature, source)
    gpu_sample = evaluate_exact_wilson_one_electron_sample(gpu_quadrature, source)
    kwargs = {
        "electric_origin_direction_au": (0.17, -0.11, 0.07),
        "magnetic_field_rate_direction_au": (-0.09, 0.05, 0.13),
    }
    cpu_response = evaluate_exact_temporal_source_direction(
        cpu_quadrature,
        cpu_sample,
        **kwargs,
    )
    gpu_response = evaluate_exact_temporal_source_direction(
        gpu_quadrature,
        gpu_sample,
        **kwargs,
    )
    gpu = gpu_quadrature.backend
    for cpu_value, gpu_value in (
        (cpu_response.site_scalar_connection, gpu_response.site_scalar_connection),
        (cpu_response.internal_electric_connection, gpu_response.internal_electric_connection),
        (cpu_response.connection, gpu_response.connection),
        (cpu_response.metric_rate, gpu_response.metric_rate),
    ):
        gpu.assert_resident(gpu_value, name="WP7 GPU source direction")
        np.testing.assert_allclose(
            gpu.to_host(gpu_value),
            cpu_value,
            atol=2.0e-10,
            rtol=2.0e-10,
        )

    magnetic_direction = (0.13, -0.07, 0.09)
    cpu_magnetic = evaluate_exact_magnetic_field_source_direction(
        cpu_quadrature,
        cpu_sample,
        magnetic_direction,
    )
    gpu_magnetic = evaluate_exact_magnetic_field_source_direction(
        gpu_quadrature,
        gpu_sample,
        magnetic_direction,
    )
    for cpu_value, gpu_value in (
        (cpu_magnetic.metric_endpoint, gpu_magnetic.metric_endpoint),
        (cpu_magnetic.metric_internal, gpu_magnetic.metric_internal),
        (cpu_magnetic.mechanical_endpoint, gpu_magnetic.mechanical_endpoint),
        (
            cpu_magnetic.mechanical_internal_triangle,
            gpu_magnetic.mechanical_internal_triangle,
        ),
        (
            cpu_magnetic.mechanical_internal_anchored,
            gpu_magnetic.mechanical_internal_anchored,
        ),
        (cpu_magnetic.connection_endpoint, gpu_magnetic.connection_endpoint),
        (cpu_magnetic.connection_internal, gpu_magnetic.connection_internal),
        (cpu_magnetic.metric, gpu_magnetic.metric),
        (cpu_magnetic.mechanical, gpu_magnetic.mechanical),
        (cpu_magnetic.connection, gpu_magnetic.connection),
    ):
        gpu.assert_resident(gpu_value, name="WP7 GPU magnetic source direction")
        np.testing.assert_allclose(
            gpu.to_host(gpu_value),
            cpu_value,
            atol=2.0e-10,
            rtol=2.0e-10,
        )

    coefficient = np.asarray(((0.71 + 0.19j,), (-0.23 + 0.41j,)))
    coefficient /= np.sqrt((coefficient.conj().T @ cpu_sample.metric @ coefficient).real.item())
    cpu_density = coefficient @ coefficient.conj().T
    gpu_density = gpu.asarray(cpu_density)
    cpu_velocity = one_electron_velocity_density(
        cpu_density,
        exact_wilson_one_electron_triple(cpu_sample),
        cpu_quadrature.backend,
    )
    gpu_velocity = one_electron_velocity_density(
        gpu_density,
        exact_wilson_one_electron_triple(gpu_sample),
        gpu,
    )
    cpu_zero = np.zeros_like(cpu_response.connection)
    gpu_zero = gpu.zeros(gpu_response.connection.shape, dtype=gpu.namespace.complex128)
    cpu_action = restricted_one_electron_action_directional_derivative(
        cpu_density,
        cpu_velocity,
        OneElectronActionMatrixDirection(
            cpu_zero,
            cpu_zero,
            cpu_response.connection,
        ),
        cpu_quadrature.backend,
    )
    gpu_action = restricted_one_electron_action_directional_derivative(
        gpu_density,
        gpu_velocity,
        OneElectronActionMatrixDirection(
            gpu_zero,
            gpu_zero,
            gpu_response.connection,
        ),
        gpu,
    )
    np.testing.assert_allclose(
        gpu.to_host(gpu_action.total),
        cpu_action.total,
        atol=2.0e-10,
        rtol=2.0e-10,
    )

    cpu_magnetic_action = restricted_one_electron_action_directional_derivative(
        cpu_density,
        cpu_velocity,
        exact_magnetic_field_action_direction(
            cpu_magnetic,
            cpu_quadrature.backend,
        ),
        cpu_quadrature.backend,
    )
    gpu_magnetic_action = restricted_one_electron_action_directional_derivative(
        gpu_density,
        gpu_velocity,
        exact_magnetic_field_action_direction(gpu_magnetic, gpu),
        gpu,
    )
    np.testing.assert_allclose(
        gpu.to_host(gpu_magnetic_action.total),
        cpu_magnetic_action.total,
        atol=2.0e-10,
        rtol=2.0e-10,
    )

    mapping = np.asarray(reference.anchor_topology.ao_to_atom)
    gauge_values = np.asarray((0.37, -0.21))
    gauge_rates = np.asarray((-0.16, 0.29))
    cpu_gauge = exact_pure_gauge_action_direction(
        cpu_sample,
        cpu_density,
        cpu_velocity,
        gauge_values,
        gauge_rates,
        mapping,
        cpu_quadrature.backend,
    )
    gpu_gauge = exact_pure_gauge_action_direction(
        gpu_sample,
        gpu_density,
        gpu_velocity,
        gpu.asarray(gauge_values),
        gpu.asarray(gauge_rates),
        gpu.asarray(mapping),
        gpu,
    )
    for cpu_value, gpu_value in (
        (cpu_gauge.matrix.metric, gpu_gauge.matrix.metric),
        (cpu_gauge.matrix.mechanical, gpu_gauge.matrix.mechanical),
        (cpu_gauge.matrix.connection, gpu_gauge.matrix.connection),
        (cpu_gauge.history.density, gpu_gauge.history.density),
        (cpu_gauge.history.velocity_density, gpu_gauge.history.velocity_density),
    ):
        gpu.assert_resident(gpu_value, name="WP7 GPU pure-gauge direction")
        np.testing.assert_allclose(
            gpu.to_host(gpu_value),
            cpu_value,
            atol=2.0e-10,
            rtol=2.0e-10,
        )
    cpu_ward = restricted_one_electron_action_full_directional_derivative(
        cpu_density,
        cpu_velocity,
        exact_wilson_one_electron_triple(cpu_sample),
        cpu_gauge.matrix,
        cpu_gauge.history,
        cpu_quadrature.backend,
    )
    gpu_ward = restricted_one_electron_action_full_directional_derivative(
        gpu_density,
        gpu_velocity,
        exact_wilson_one_electron_triple(gpu_sample),
        gpu_gauge.matrix,
        gpu_gauge.history,
        gpu,
    )
    np.testing.assert_allclose(
        gpu.to_host(gpu_ward.total.total),
        cpu_ward.total.total,
        atol=2.0e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        gpu.to_host(gpu_ward.total.total),
        0.0,
        atol=2.0e-10,
        rtol=0.0,
    )

    scalar_values = np.asarray((0.31, -0.17))
    links = np.asarray(((0.0, 0.29), (-0.29, 0.0)))
    cpu_scalar = exact_site_scalar_action_direction(
        cpu_sample,
        scalar_values,
        mapping,
        cpu_quadrature.backend,
    )
    gpu_scalar = exact_site_scalar_action_direction(
        gpu_sample,
        gpu.asarray(scalar_values),
        gpu.asarray(mapping),
        gpu,
    )
    cpu_link = exact_endpoint_link_action_direction(
        cpu_sample,
        links,
        mapping,
        cpu_quadrature.backend,
    )
    gpu_link = exact_endpoint_link_action_direction(
        gpu_sample,
        gpu.asarray(links),
        gpu.asarray(mapping),
        gpu,
    )
    for cpu_direction, gpu_direction in (
        (cpu_scalar, gpu_scalar),
        (cpu_link, gpu_link),
    ):
        for cpu_value, gpu_value in (
            (cpu_direction.metric, gpu_direction.metric),
            (cpu_direction.mechanical, gpu_direction.mechanical),
            (cpu_direction.connection, gpu_direction.connection),
        ):
            gpu.assert_resident(gpu_value, name="WP7 GPU discrete source direction")
            np.testing.assert_allclose(
                gpu.to_host(gpu_value),
                cpu_value,
                atol=2.0e-10,
                rtol=2.0e-10,
            )
