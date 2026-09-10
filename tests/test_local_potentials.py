from __future__ import annotations

import numpy as np
import pytest

from aion.backends import NumPyBackend
from aion.config import BackendConfig
from aion.electromagnetism import UniformMagneticField
from aion.electronic_structure import (
    LocalPotentialIdentity,
    NuclearAttractionProvider,
    ScaledLocalPotentialProvider,
    bind_local_potential,
    evaluate_local_potential_magnetic_matrices,
    evaluate_magnetic_one_electron_matrices,
    prepare_ao_quadrature,
    prepare_pyscf_reference,
)
from aion.errors import ConfigurationError, UnsupportedConfigurationError
from test_reference_integration import molecular_config

pytestmark = pytest.mark.integration


def test_provider_identity_is_normalized_hashed_and_validated() -> None:
    left = LocalPotentialIdentity(
        "custom_local", "2.1", (("zeta", "last"), ("alpha", "first"))
    )
    right = LocalPotentialIdentity(
        "custom_local", "2.1", (("alpha", "first"), ("zeta", "last"))
    )
    assert left == right
    assert left.fingerprint_sha256 == right.fingerprint_sha256
    assert left.provenance == (("alpha", "first"), ("zeta", "last"))
    with pytest.raises(ConfigurationError):
        LocalPotentialIdentity("not a name", "1", ())
    with pytest.raises(ConfigurationError):
        LocalPotentialIdentity("duplicate", "1", (("key", "a"), ("key", "b")))


def test_nuclear_provider_binds_authenticated_matrix_and_pointwise_values() -> None:
    reference = prepare_pyscf_reference(molecular_config("lih"))
    backend = NumPyBackend()
    bound = bind_local_potential(NuclearAttractionProvider(), reference, backend)
    np.testing.assert_array_equal(
        bound.zero_matrix_au, reference.core_operators.nuclear_attraction
    )
    points = np.array([[0.2, -0.3, 0.7], [-0.4, 0.5, 1.1]])
    distances = np.linalg.norm(
        points[:, None, :] - reference.core_operators.nuclei.coordinates_au[None, :, :],
        axis=2,
    )
    expected = -np.sum(
        reference.core_operators.nuclei.charges[None, :] / distances, axis=1
    )
    np.testing.assert_allclose(bound.values_au(points), expected, atol=0.0, rtol=0.0)
    with pytest.raises(UnsupportedConfigurationError, match="preassembled"):
        bind_local_potential(  # type: ignore[arg-type]
            reference.core_operators.nuclear_attraction, reference, backend
        )


def test_generic_nuclear_hierarchy_matches_MB3_routed_provider() -> None:
    reference = prepare_pyscf_reference(molecular_config("lih"))
    quadrature = prepare_ao_quadrature(reference, BackendConfig(), block_size=521)
    fields = (
        UniformMagneticField((0.013, -0.009, 0.017)),
        UniformMagneticField((-0.013, 0.009, -0.017)),
    )
    generic = evaluate_local_potential_magnetic_matrices(
        quadrature, fields, (NuclearAttractionProvider(),)
    )
    mb3 = evaluate_magnetic_one_electron_matrices(quadrature, fields)
    for local_result, one_electron in zip(generic, mb3, strict=True):
        assert local_result.identity.name == "nuclear_attraction"
        assert local_result.direct_lower_grid is not None
        np.testing.assert_allclose(
            local_result.direct_lower_grid,
            local_result.lower_exact_grid,
            atol=2.0e-11,
            rtol=2.0e-11,
        )
        for name in (
            "zero",
            "quadrature_zero",
            "first_F",
            "second_F2",
            "exact_grid",
            "exact",
        ):
            np.testing.assert_array_equal(
                getattr(local_result.hierarchy, name),
                getattr(one_electron.nuclear_attraction, name),
            )


def test_scaled_provider_scales_every_component_and_preserves_field_reversal() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    quadrature = prepare_ao_quadrature(reference, BackendConfig(), block_size=521)
    base = NuclearAttractionProvider()
    scaled = ScaledLocalPotentialProvider(base, -0.375)
    positive = UniformMagneticField((0.011, -0.007, 0.005))
    negative = UniformMagneticField((-0.011, 0.007, -0.005))
    results = evaluate_local_potential_magnetic_matrices(
        quadrature,
        (positive, negative),
        (base, scaled),
        include_direct_oracle=False,
    )
    base_positive, scaled_positive, base_negative, scaled_negative = results
    assert scaled_positive.identity.fingerprint_sha256 != base_positive.identity.fingerprint_sha256
    for name in (
        "zero",
        "quadrature_zero",
        "first_F",
        "second_F2",
        "exact_grid",
        "exact",
    ):
        base_matrix = getattr(base_positive.hierarchy, name)
        scaled_matrix = getattr(scaled_positive.hierarchy, name)
        np.testing.assert_allclose(scaled_matrix, -0.375 * base_matrix, atol=2.0e-14)
        np.testing.assert_allclose(
            getattr(base_negative.hierarchy, name),
            base_matrix.conj() if name in {"exact", "exact_grid"} else (
                -base_matrix if name == "first_F" else base_matrix
            ),
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            getattr(scaled_negative.hierarchy, name),
            -0.375 * getattr(base_negative.hierarchy, name),
            atol=2.0e-14,
        )


def test_duplicate_provider_identity_is_rejected() -> None:
    reference = prepare_pyscf_reference(molecular_config("h2"))
    quadrature = prepare_ao_quadrature(reference, BackendConfig(), block_size=1024)
    provider = NuclearAttractionProvider()
    with pytest.raises(ConfigurationError, match="unique"):
        evaluate_local_potential_magnetic_matrices(
            quadrature,
            (UniformMagneticField((0.0, 0.0, 0.0)),),
            (provider, provider),
        )
