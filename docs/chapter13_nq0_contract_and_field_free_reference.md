# Chapter 13 NQ0 contract and field-free reference

Date: 2026-09-21  
Gate state: **accepted by user on 2026-09-21**  
Controlling plan: `/home/cgs/00_WORK/Projection_Full_Formalism/REVIEW/implementation/aion_chapter_13_numerical_qualification_plan.md`  
Agent brief: `/home/cgs/00_WORK/Projection_Full_Formalism/REVIEW/implementation/aion_chapter_13_numerical_qualification_agent_prompt.md`

This is the review entry point for NQ0. The acceptance decision is recorded in
`docs/reviews/chapter13_nq0_review_20260921.json`. This document records the reconciled baseline,
the tensor and action contracts, the concrete numerical realization, the
field-free reference execution, and the bounded implementation started for
NQ1. A software test or an executed campaign does not accept a gate; only the
subsequent user review can change this gate to accepted.

## 1. Baseline reconciliation

The nominal worktree was clean at `9ade645551e28baf7b7ffedde0a83d3ed28a88d7`
on `feature/interpolated-p0-gauge`. The accepted exact one-electron worktree
was clean at `20c7809cc19e3bec226beca27ee5f92c04d6a7c6` on
`feature/exact-one-electron-wp7-action-observables`. Their common ancestor was
`6acb975`; the nominal line had two unique commits and the exact line had 30.

With explicit user authorization, the nonlinear branch
`feature/ch13-wilson-adiabatic-qualification` was created from the accepted
exact commit and the nominal line was merged. The only textual conflict was
the formulation export list; it was resolved by retaining both the exact
one-electron action exports and `resolve_velocity_fraction`. The resulting
baseline is merge commit:

```text
79208d6c6b5bee2a80a4468f49161a6b5e8962da
Merge accepted exact Wilson and gauge interpolation baselines
```

The accepted worktree and its archived numerical evidence were not modified.
The reconciliation check was:

```text
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 \
/home/cgs/01_TOOLS/EasyBuild/conda/bin/conda run \
  -p /home/cgs/01_TOOLS/EasyBuild/conda/envs/aion \
  python -m pytest -q tests/test_config_contracts.py \
  tests/test_exact_one_electron_wp0_wp1.py tests/test_tensorial_magnus.py
```

Result: `22 passed, 4 deselected in 0.88 s`.

The inherited reusable implementation includes exact blocked AO quadrature,
straight-Wilson overlap and one-electron mechanical matrices, the exact
temporal connection, fixed-history one-electron source directions, action
contractions, and mixed-index propagation. The inherited schemas are
`docs/schemas/exact_one_electron_matrix_result.schema.json` and
`docs/schemas/exact_one_electron_run_manifest.schema.json`; the authoritative
accepted evidence index is
`/home/cgs/00_WORK/Projection_Code/aion-magnetic-matrix-benchmark/docs/exact_one_electron_qualification_report.tex`,
with decisions in that worktree's `docs/reviews/` directory.

## 2. Index and action conventions

- `AODensity.matrix` is the contravariant coefficient density
  \(P^{\mu\nu}=\sum_n f_n C^\mu_n C^{\nu *}_n\), not a lower-index
  operator. Its defining Chapter 13 label is
  `eq:wilson-hartree-ks-contravariant-density`.
- The mixed density is \(D^\mu{}_\nu=(PS)^\mu{}_\nu\), as in
  `eq:projected-independent-mixed-density`. It is the object advanced by the
  mixed-index propagator. Conversion back is \(P=DS^{-1}\), implemented as a
  right-side solve rather than an explicit inverse.
- Overlap, connection, and Hamiltonian/mechanical matrices are lower-index
  matrices. The inherited exact data are specified by
  `eq:wilson-hartree-ks-inherited-lower-data`.
- The exact finite-field real-space density is the complete complex Wilson
  pair contraction in `eq:wilson-hartree-ks-density-overlap-form`, equivalently
  the dressed-orbital form in `eq:wilson-hartree-ks-density-orbital-form`.
- The scalar-potential coupling belongs to the temporal connection. It is not
  inserted a second time into the lower mechanical matrix or mechanical
  energy.
- Source derivatives are fixed-coefficient-history derivatives. They are not
  derivatives of a separately reconverged stationary state.

### Stored-array ledger for the NQ0 HDF5 reference

The non-scientific `configuration/*` and `meta/*` datasets store normalized
text, identifiers, and dependency metadata. Every scientific dataset under
`reference/*` has the following single interpretation:

| HDF5 path | Tensor/index type and action-level meaning |
|---|---|
| `reference/anchors/ao_to_atom` | Integer map from each AO coefficient index to its fixed nuclear anchor |
| `reference/anchors/incidence` | Real oriented site--link incidence matrix; topology data, not an electronic operator |
| `reference/anchors/pair_displacements_au` | Real Cartesian anchor displacement for each stored oriented pair |
| `reference/anchors/pair_indices` | Integer endpoint indices for each oriented anchor pair |
| `reference/anchors/site_projector_diagonals` | Real dimensionless AO-index partition weights \(q_{a\mu}\); they are not a lower matrix by themselves and form the lower site-charge operator as \(\tfrac12(q_a S+S q_a)\) |
| `reference/grid/coordinates_au` | Real Cartesian molecular-quadrature sample coordinates |
| `reference/grid/weights_au` | Real scalar quadrature measure associated with those coordinates |
| `reference/ground_state/coefficients` | Contravariant AO expansion coefficients \(C^\mu_n\) of the field-free stationary orbitals |
| `reference/ground_state/density` | Contravariant spin-summed coefficient density \(P^{\mu\nu}=C f C^\dagger\); it is not the mixed density |
| `reference/ground_state/electron_count` | Scalar \(\operatorname{Tr}(PS)\) |
| `reference/ground_state/energy_total_au` | Scalar field-free internal/mechanical DFT energy \(T+V_{\rm eN}+E_{\rm H}+E_{\rm xc}+E_{\rm NN}\) |
| `reference/ground_state/occupations` | Fixed dimensionless orbital occupations \(f_n\), summing to two |
| `reference/ground_state/orbital_energies_au` | Scalar stationary generalized-eigenvalue spectrum; not terms in the mechanical-energy sum |
| `reference/nuclei/charges` | Real fixed nuclear source charges |
| `reference/nuclei/coordinates_au` | Real fixed Cartesian nuclear/anchor coordinates |
| `reference/nuclei/symbols` | Nuclear species labels; metadata, not a numerical tensor |
| `reference/operators/overlap` | Lower-index AO metric \(S_{\mu\nu}\) |
| `reference/operators/kinetic` | Lower-index field-free canonical/mechanical kinetic matrix \(T_{\mu\nu}\) |
| `reference/operators/nuclear_attraction` | Lower-index all-electron local nuclear-attraction matrix \((V_{\rm eN})_{\mu\nu}\) |
| `reference/operators/canonical_momentum` | Three lower-index Cartesian canonical-momentum matrices \((p_\alpha)_{\mu\nu}\) |
| `reference/operators/position` | Three lower-index Cartesian position matrices \((r_\alpha)_{\mu\nu}\) |

The energy fields in `field_free_result.json` are contractions or functional
values of the same declared field-free action. `kinetic_canonical_au` and
`electron_nuclear_au` are \(\operatorname{ReTr}(PT)\) and
\(\operatorname{ReTr}(PV_{\rm eN})\); `hartree_au` and
`exchange_correlation_au` are the PySCF density-fitted Coulomb and Libxc LDA
functional values; `nuclear_repulsion_au` is the fixed-source energy; and
`internal_total_au` is their sum. No scalar-potential energy is stored.

## 3. Equation-to-code-to-test-to-evidence map

All code paths below are absolute. “Absent” means that the Chapter 13
finite-field nonlinear object has not been implemented; a field-free PySCF
analogue is not relabeled as that object.

| Required object | Exact mathematical label | Reusable implementation or explicit gap | Test | Evidence and present state |
|---|---|---|---|---|
| Contravariant coefficient density \(P\) | `eq:wilson-hartree-ks-contravariant-density` | `/home/cgs/00_WORK/Projection_Code/aion/src/aion/formulations/types.py`, `AODensity`; construction keeps complex128 \(C f C^\dagger\) | `/home/cgs/00_WORK/Projection_Code/aion/tests/test_tensorial_magnus.py::test_mixed_and_contravariant_density_round_trip` and Wilson-density tests below | **Implemented and executed**; the misleading historical “lower-index” docstring was corrected |
| Mixed density \(D=PS\) | `eq:projected-independent-mixed-density`; dynamics: `eq:wilson-hartree-ks-covariant-mixed-density-equation`, `eq:wilson-hartree-ks-ordinary-mixed-density-equation` | `/home/cgs/00_WORK/Projection_Code/aion/src/aion/propagation/tensorial.py`, `contravariant_to_mixed_density`, `mixed_to_contravariant_density`, `mixed_eom_generator`, `propagate_experimental_mixed_density` | `/home/cgs/00_WORK/Projection_Code/aion/tests/test_tensorial_magnus.py` | **Implemented and executed** for the accepted linear exact one-electron generator; nonlinear Gauss-node closure is absent |
| Exact Wilson density \(\rho_{\rm W}\) | `eq:wilson-hartree-ks-density-overlap-form`, `eq:wilson-hartree-ks-density-orbital-form` | `/home/cgs/00_WORK/Projection_Code/aion/src/aion/electronic_structure/wilson_density.py`, `evaluate_exact_uniform_magnetic_wilson_density`; blocked direct dressed-AO and independent endpoint/triangle-factorized routes | `/home/cgs/00_WORK/Projection_Code/aion/tests/test_wilson_density.py`; `/home/cgs/00_WORK/Projection_Code/aion/tests/test_wilson_density_gpu.py` | **Implemented and executed as an early NQ1 checkpoint**; NQ1 is not yet accepted |
| Exact Wilson overlap \(S\) | `eq:wilson-hartree-ks-inherited-lower-data` | `/home/cgs/00_WORK/Projection_Code/aion/src/aion/electronic_structure/magnetic_matrices.py`, `evaluate_exact_static_magnetic_one_electron_matrices` | `/home/cgs/00_WORK/Projection_Code/aion/tests/test_exact_one_electron_wp2.py` | **Implemented, executed, and inherited from accepted G2 evidence** |
| Exact temporal connection \(\omega\) | `eq:wilson-hartree-ks-inherited-lower-data` | `/home/cgs/00_WORK/Projection_Code/aion/src/aion/electronic_structure/time_connection.py`, `evaluate_exact_uniform_magnetic_time_connection` | `/home/cgs/00_WORK/Projection_Code/aion/tests/test_exact_one_electron_wp6.py` | **Implemented, executed, and inherited from accepted G6 evidence** |
| One-electron lower mechanical matrix \(K\) | `eq:wilson-hartree-ks-inherited-lower-data` | `/home/cgs/00_WORK/Projection_Code/aion/src/aion/electronic_structure/magnetic_matrices.py`, `evaluate_exact_static_magnetic_one_electron_matrices`; local scalar operators use `/home/cgs/00_WORK/Projection_Code/aion/src/aion/electronic_structure/local_magnetic_matrices.py` | `/home/cgs/00_WORK/Projection_Code/aion/tests/test_exact_one_electron_wp2.py` | **Implemented, executed, and inherited from accepted G2--G7 evidence** |
| Hartree energy \(E_{\rm H}[\rho_{\rm W}]\) | `eq:wilson-hartree-ks-hartree-energy` | Field-free/P0-only bridge: `/home/cgs/00_WORK/Projection_Code/aion/src/aion/electronic_structure/adiabatic.py`, `AdiabaticPureRKS.energy`; exact finite-field RI--Wilson action is **absent** | Field-free reconstruction executed by `/home/cgs/00_WORK/Projection_Code/aion/tools/run_chapter13_nq0_field_free.py` | **Field-free contract reusable and executed; Chapter 13 finite-field implementation absent** |
| Adiabatic XC energy \(E_{\rm xc}[\rho_{\rm W}]\) | `eq:wilson-hartree-ks-xc-functional` | Field-free/P0-only bridge: `/home/cgs/00_WORK/Projection_Code/aion/src/aion/electronic_structure/adiabatic.py`; exact finite-field action-level Wilson-density quadrature and its matrix are **absent** | Same field-free reconstruction | **Field-free LDA contract reusable and executed; Chapter 13 finite-field implementation absent** |
| Local, Hartree, and XC lower matrices | `eq:wilson-hartree-ks-local-potential-matrix`, `eq:wilson-hartree-ks-hartree-matrix`, `eq:wilson-hartree-ks-xc-matrix` | Field-free effective matrix is built by `AdiabaticPureRKS.build`; the finite-field Wilson functional derivatives are **absent** | Field-free Hamiltonian comparison in the NQ0 campaign | **Field-free analogue executed; exact nonlinear Wilson matrices absent** |
| Nonlinear lower Hartree/KS matrix | `eq:wilson-hartree-ks-total-hartree-matrix`, `eq:wilson-hartree-ks-total-ks-matrix` | No reusable exact finite-field nonlinear formulation evaluation exists yet | None | **Specified, not implemented, not executed** |
| Mechanical Hartree/KS energies | `eq:wilson-hartree-ks-hartree-mechanical-energy`, `eq:wilson-hartree-ks-ks-mechanical-energy` | Field-free decomposition exists in `DFTInternalEnergy`; complete finite-field Wilson energies are **absent** | NQ0 component reconstruction | **Field-free analogue executed; exact finite-field object absent** |
| Fixed-history nonlinear source differential | `eq:wilson-hartree-ks-source-differential` | Inherited one-electron directions: `/home/cgs/00_WORK/Projection_Code/aion/src/aion/electronic_structure/time_connection.py` and `/home/cgs/00_WORK/Projection_Code/aion/src/aion/formulations/exact_one_electron.py`; Hartree/XC terms are **absent** | `/home/cgs/00_WORK/Projection_Code/aion/tests/test_exact_one_electron_wp7.py` | **One-electron part implemented and accepted; nonlinear closure contribution specified but not implemented** |
| Numerical realization tuple | `eq:wilson-hartree-ks-realization-tuple` | This document, `reference.toml`, and campaign provenance below | Reproduction command below | **Declared and executed for NQ0** |
| Aion/PySCF density boundary | `eq:wilson-hartree-ks-interface-density-types` | `/home/cgs/00_WORK/Projection_Code/aion/src/aion/electronic_structure/adiabatic.py` explicitly admits only the field-free/P0 real-AO shortcut; `/home/cgs/00_WORK/Projection_Code/aion/src/aion/electronic_structure/wilson_density.py` retains the complete complex finite-field contraction | Field-free and Wilson-density tests | **Audited; no finite-field density is passed through the field-free shortcut** |

## 4. Declared NQ0 realization tuple

The reproducible field-free fixture is:

| Axis | Choice |
|---|---|
| Electromagnetic geometry | Fixed nuclei and AO anchors; straight anchor-to-electronic-position Wilson paths; atomic units; electron charge \(q=-1\), \(\hbar=1\); affine gauge implementation. NQ0 reference has zero external field and origin `(0,0,0)` |
| Molecule | H2, all electron and local nuclei, charge 0, spin 0, restricted closed shell; H at `(0,0,-0.7)` and `(0,0,0.7)` bohr; occupations `(2,0)` |
| Orbital basis | `sto-3g` |
| Hartree realization | PySCF density fitting with `weigend` auxiliary basis for the field-free reference; 22 auxiliary functions; native PySCF Coulomb-metric rank policy with `pyscf.df.incore.LINEAR_DEP_THR = 1e-7`; no additional Aion truncation. The target finite-field RI--Wilson realization remains unimplemented |
| XC realization | Pure LDA `lda,vwn`, Libxc 7.1.2. The target finite-field functional consumes the exact Wilson density; it remains unimplemented |
| Molecular grid | PySCF grid level 3, 19,616 points; `treutler_ahlrichs` radial grid, `original_becke` partition, and `nwchem_prune`; fingerprint `2e48f2721dd7babf0bc98011410696a1963721c057ee923493658817e1208e3b` |
| SCF and precision | Energy tolerance `1e-12` Ha, at most 100 iterations; CPU float64/complex128; eight-thread ceiling |
| Energy convention | \(T_{\rm canonical}+V_{\rm eN}+E_{\rm H}+E_{\rm xc}+E_{\rm NN}\). At finite field the scalar-potential coupling is owned by the temporal connection and is excluded from this mechanical energy |

No undocumented default needed to interpret the result: the output records
the exact PySCF, Libxc, grid, auxiliary, rank-threshold, dependency, module,
thread, source-hash, and artifact-hash data. PySCF's field-free density-fitting
algorithm itself is the declared reference realization, not evidence for the
future finite-field RI--Wilson closure.

## 5. Reproducible field-free execution

Artifact root:

```text
/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq0_field_free_h2_lda_vwn_20260921T180851Z_79208d6
```

Entry points:

- `reference.toml`: complete input contract;
- `h2_lda_vwn_weigend.reference.h5`: transactional PySCF reference;
- `field_free_result.json`: energies, realization, tolerances, and residuals;
- `provenance.json`: exact command, environment, Git state, SHA-256 source
  hashes, and SHA-256 artifact hashes.

Execution command:

```text
module purge
module use /home/cgs/01_TOOLS/EasyBuild/modules/production
module load SciStack/2026.07.1-gnu
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 \
/home/cgs/01_TOOLS/EasyBuild/conda/bin/conda run \
  -p /home/cgs/01_TOOLS/EasyBuild/conda/envs/aion \
  python tools/run_chapter13_nq0_field_free.py \
  --output /home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq0_field_free_h2_lda_vwn_20260921T180851Z_79208d6
```

Recorded result:

| Quantity | Value |
|---|---:|
| Electron count | `1.9999999999999996` |
| Total internal energy | `-1.1212693750656286 Ha` (component reconstruction; SCF reference `-1.1212693750656282 Ha`) |
| Canonical kinetic | `1.2010794986302844 Ha` |
| Electron--nuclear | `-3.706673622301919 Ha` |
| Hartree | `1.3491194978628671 Ha` |
| Exchange--correlation | `-0.6790804635425751 Ha` |
| Nuclear repulsion | `0.7142857142857143 Ha` |

The electron-count, component-sum, and rebuilt-density residuals were exactly
zero in the stored representation. The total-energy residual was
`4.440892098500626e-16 Ha`, and the relative Hamiltonian Frobenius residual
was `7.850462293418876e-17`. Their respective tolerances were `1e-11`,
`5e-11`, `5e-14`, `5e-13`, and `5e-13` for electron count, total energy,
component sum, Hamiltonian, and rebuilt density.

The artifact hashes recorded in `provenance.json` are:

```text
field_free_result.json                 365b5dbef51fe686d8b2206c55aa1461ea8383c4c0d099f001003fc0f1258acd
h2_lda_vwn_weigend.reference.h5       a243f109b78fdbb3d3c783771a7a6776ce0bb669323d60e2c9f463ce76498ae6
reference.toml                        3ef3b2f2e3611d89737d30ee47b518f3fd6ad581fc1c71ca7f166dc4c5f96b21
```

## 6. Early bounded NQ1 implementation

The smallest safe post-contract checkpoint was the exact uniform-magnetic
Wilson density because every Hartree/XC closure must consume that one shared
definition. The reusable evaluator:

- contracts the complete complex contravariant density;
- evaluates algebraically independent direct dressed-AO and
  endpoint-link/triangle-factorized routes;
- preserves raw complex roundoff data and reports imaginary and negativity
  floors rather than clipping or projecting them;
- accumulates both grid normalization and the Wilson overlap contraction;
- blocks all point/pair data and enforces an explicit additional-memory
  estimate; and
- is backend-neutral at complex128 precision.

Focused CPU tests passed (`3 passed in 0.96 s`). The final physical-GPU execution,
with GPU residency asserted and no CPU fallback accepted, passed
(`1 passed in 1.83 s`; the only warning records CuPy as GPU4PySCF's tensor
contraction engine). These executions establish software evidence for
the bounded checkpoint only. A complete NQ1 campaign still needs the
controlling plan's density-domain and frame-invariance coverage before NQ1
can be presented for acceptance.

## 7. Review boundary

The user accepted the following decision on 2026-09-21:

> NQ0 is complete and accepted as an implemented and executed contract with a
> reproducible field-free reference.

The exact finite-field Hartree action, LDA action and lower matrix, nonlinear
formulation evaluation, nonlinear stationary solve, nonlinear source
differential, and nonlinear propagation are intentionally not claimed. The
next gate after an NQ0 decision is NQ1, beginning from the bounded density
evaluator already implemented here.
