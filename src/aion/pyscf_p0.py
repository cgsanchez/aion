"""PySCF bridge for pure Peierls P0 gauge geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gauge import AOAnchors, PeierlsGeometry, UniformElectricGauge, UniformMagneticGauge
from .matrix_models import LinearOneBodyModel, density_from_coefficients, hermitian_part


def _as_numpy(array, *, dtype=None) -> np.ndarray:
    if hasattr(array, "get"):
        array = array.get()
    return np.asarray(array, dtype=dtype)


def _occupied_orbitals(mf) -> tuple[np.ndarray, np.ndarray]:
    mo_occ = getattr(mf, "mo_occ", None)
    mo_coeff = getattr(mf, "mo_coeff", None)
    if mo_occ is None or mo_coeff is None:
        raise ValueError("mean-field object must have mo_occ and mo_coeff")
    if isinstance(mo_occ, (tuple, list)) or isinstance(mo_coeff, (tuple, list)):
        raise NotImplementedError(
            "P0 PySCF bridge currently supports restricted references only"
        )
    occ_all = _as_numpy(mo_occ, dtype=float)
    coeff_all = _as_numpy(mo_coeff, dtype=np.complex128)
    occ_mask = occ_all > 0.0
    if not np.any(occ_mask):
        raise ValueError("mean-field reference has no occupied orbitals")
    return coeff_all[:, occ_mask].copy(), occ_all[occ_mask].copy()


def _validate_rks_lda(mf) -> None:
    if not hasattr(mf, "xc") or not hasattr(mf, "_numint"):
        raise TypeError("PyscfP0LdaModel requires a PySCF RKS-like object")
    if isinstance(getattr(mf, "mo_occ", None), (tuple, list)):
        raise NotImplementedError(
            "PyscfP0LdaModel currently supports restricted references only"
        )

    xc_type = mf._numint._xc_type(mf.xc).upper()
    if xc_type != "LDA":
        raise NotImplementedError(
            f"PyscfP0LdaModel currently supports LDA only, got {xc_type}"
        )

    hybrid_coeff = mf._numint.hybrid_coeff(mf.xc, spin=mf.mol.spin)
    if abs(float(hybrid_coeff)) > 1.0e-14:
        raise NotImplementedError("PyscfP0LdaModel does not support hybrid functionals")


@dataclass(frozen=True)
class PyscfP0Reference:
    """Field-free PySCF AO data needed by the P0 geometry layer."""

    mf: object
    anchors: AOAnchors
    overlap0: np.ndarray
    hcore0: np.ndarray
    coeff0: np.ndarray
    occupations: np.ndarray

    @classmethod
    def from_mean_field(
        cls,
        mf,
        *,
        require_converged: bool = True,
    ) -> "PyscfP0Reference":
        if require_converged and not getattr(mf, "converged", False):
            raise ValueError("mean-field object is not converged")
        mol = mf.mol
        coeff0, occupations = _occupied_orbitals(mf)
        overlap0 = _as_numpy(mol.intor("int1e_ovlp"), dtype=np.complex128)
        hcore0 = _as_numpy(mf.get_hcore(), dtype=np.complex128)
        return cls(
            mf=mf,
            anchors=AOAnchors.from_mol(mol),
            overlap0=hermitian_part(overlap0),
            hcore0=hermitian_part(hcore0),
            coeff0=coeff0,
            occupations=occupations,
        )

    @property
    def mol(self):
        return self.mf.mol

    @property
    def nao(self) -> int:
        return int(self.overlap0.shape[0])

    @property
    def nocc(self) -> int:
        return int(self.occupations.size)

    def geometry(
        self,
        *,
        electric: UniformElectricGauge | None = None,
        magnetic: UniformMagneticGauge | None = None,
        charge: float = -1.0,
        hbar: float = 1.0,
    ) -> PeierlsGeometry:
        return PeierlsGeometry(
            self.anchors,
            self.overlap0,
            electric=electric,
            magnetic=magnetic,
            charge=charge,
            hbar=hbar,
        )

    def linear_model(self) -> LinearOneBodyModel:
        return LinearOneBodyModel(self.hcore0)

    def initial_coefficients(self) -> np.ndarray:
        return self.coeff0.copy()

    def initial_density(self) -> np.ndarray:
        return density_from_coefficients(self.coeff0, self.occupations)


@dataclass(frozen=True)
class PyscfP0LdaModel:
    """CPU P0 adiabatic LDA model backed by PySCF builders.

    The dressed density is mapped back to the field-free AO representation,
    PySCF builds the ordinary LDA effective potential there, and the resulting
    one-body matrix is Peierls dressed on output.
    """

    mf: object
    hcore0: np.ndarray
    real_density_for_veff: bool = False

    @classmethod
    def from_reference(
        cls,
        reference: PyscfP0Reference,
        *,
        real_density_for_veff: bool = False,
    ) -> "PyscfP0LdaModel":
        return cls(
            reference.mf,
            reference.hcore0,
            real_density_for_veff=real_density_for_veff,
        )

    def __post_init__(self) -> None:
        _validate_rks_lda(self.mf)
        hcore = _as_numpy(self.hcore0, dtype=np.complex128)
        if hcore.ndim != 2 or hcore.shape[0] != hcore.shape[1]:
            raise ValueError("hcore0 must be a square matrix")
        object.__setattr__(self, "hcore0", hermitian_part(hcore))

    @property
    def mol(self):
        return self.mf.mol

    def dressed_density_for_pyscf(
        self,
        density: np.ndarray,
        t: float,
        geometry: PeierlsGeometry,
    ) -> np.ndarray:
        rho = np.asarray(density, dtype=np.complex128)
        if rho.shape != geometry.overlap0.shape:
            raise ValueError(f"density must have shape {geometry.overlap0.shape}")
        dressed = hermitian_part(geometry.theta(t) * rho)
        if self.real_density_for_veff:
            return np.asarray(dressed.real, dtype=float)
        return dressed

    def veff0(
        self,
        density: np.ndarray,
        t: float,
        geometry: PeierlsGeometry,
    ) -> np.ndarray:
        dm0 = self.dressed_density_for_pyscf(density, t, geometry)
        return _as_numpy(self.mf.get_veff(self.mol, dm0), dtype=np.complex128)

    def hamiltonian(
        self,
        density: np.ndarray,
        t: float,
        geometry: PeierlsGeometry,
    ) -> np.ndarray:
        h0 = self.hcore0 + self.veff0(density, t, geometry)
        return hermitian_part(geometry.dress_matrix(h0, t))

    def energy(
        self,
        density: np.ndarray,
        t: float,
        geometry: PeierlsGeometry,
    ) -> float:
        dm0 = self.dressed_density_for_pyscf(density, t, geometry)
        energy = self.mf.energy_tot(dm=dm0, h1e=self.hcore0)
        return float(np.real(energy))
