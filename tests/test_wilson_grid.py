from __future__ import annotations

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import gto

from aion.gauge import UniformMagneticGauge
from aion.reference import evaluate_uniform_magnetic_matrices


_FRAMEWORKS = {
    "H-H": {
        "atom": [
            ("H", (0.25, -0.40, 0.15)),
            ("H", (1.32, 0.51, -0.27)),
        ],
        "spin": 0,
    },
    "O-H": {
        "atom": [
            ("O", (0.30, -0.25, 0.40)),
            ("H", (1.60, 0.43, -0.21)),
        ],
        "spin": 1,
    },
}


@pytest.mark.parametrize("framework", ["H-H", "O-H"])
@pytest.mark.parametrize("gauge_name", ["symmetric", "landau"])
def test_exact_wilson_grid_oracle_runs_for_pair_frameworks(framework, gauge_name):
    specification = _FRAMEWORKS[framework]
    mol = gto.M(
        atom=specification["atom"],
        basis="sto-3g",
        unit="Bohr",
        spin=specification["spin"],
        verbose=0,
    )
    field = np.array([0.09, -0.07, 0.11])
    origin = np.array([0.17, -0.31, 0.23])
    if gauge_name == "symmetric":
        gauge = UniformMagneticGauge(field, gauge="symmetric", origin=origin)
    else:
        gauge = UniformMagneticGauge(
            field,
            gauge="landau",
            origin=origin,
            landau_u=np.array([field[1], -field[0], 0.0]),
        )

    result = evaluate_uniform_magnetic_matrices(
        mol,
        gauge,
        grid_level=0,
        block_size=257,
    )

    shape = (mol.nao_nr(), mol.nao_nr())
    assert result.theta.shape == shape
    assert result.ao_anchors.shape == (mol.nao_nr(), 3)
    assert result.grid.npoints > result.grid.block_size
    assert result.grid.level == 0
    assert result.grid.pruning == "none"

    for matrices in (result.bare, result.direct, result.barred, result.factorized):
        for matrix in (
            matrices.overlap,
            matrices.kinetic,
            matrices.potential,
            matrices.mechanical,
        ):
            assert matrix.shape == shape
            assert np.all(np.isfinite(matrix))
        assert np.linalg.norm(
            matrices.mechanical - matrices.kinetic - matrices.potential
        ) < 1.0e-13

    for name in ("overlap", "kinetic", "potential", "mechanical"):
        direct = getattr(result.direct, name)
        factorized = getattr(result.factorized, name)
        assert not np.shares_memory(direct, factorized)
        assert np.linalg.norm(direct - factorized) < 2.0e-11

    sectors = result.kinetic_sectors
    for sector in (sectors.pp, sectors.p_c, sectors.c_p, sectors.c_c):
        assert sector.shape == shape
        assert np.all(np.isfinite(sector))
    assert np.linalg.norm(sectors.total - result.barred.kinetic) < 2.0e-11


def test_exact_wilson_grid_oracle_respects_explicit_particle_parameters():
    mol = gto.M(
        atom=_FRAMEWORKS["H-H"]["atom"],
        basis="sto-3g",
        unit="Bohr",
        spin=0,
        verbose=0,
    )
    field = np.array([0.04, 0.02, -0.03])
    gauge = UniformMagneticGauge(field, origin=np.array([0.1, -0.2, 0.3]))
    charge = -0.8
    hbar = 1.3
    mass = 1.7

    result = evaluate_uniform_magnetic_matrices(
        mol,
        gauge,
        grid_level=0,
        block_size=257,
        charge=charge,
        hbar=hbar,
        mass=mass,
    )

    assert result.charge == charge
    assert result.hbar == hbar
    assert result.mass == mass
    expected_t0 = (hbar**2 / mass) * mol.intor_symmetric("int1e_kin")
    assert np.linalg.norm(result.T0 - expected_t0) < 1.0e-14
    assert np.linalg.norm(result.direct.kinetic - result.factorized.kinetic) < 2.0e-11
