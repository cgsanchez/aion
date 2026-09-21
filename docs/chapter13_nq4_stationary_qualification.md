# Chapter 13 NQ4 nonlinear stationary-state qualification

Date: 2026-09-21

Gate state: **accepted by the user**

Accepted prerequisite: `docs/reviews/chapter13_nq3_review_20260921.json`

Acceptance record: `docs/reviews/chapter13_nq4_review_20260921.json`

Controlling plan:
`/home/cgs/00_WORK/Projection_Full_Formalism/REVIEW/implementation/aion_chapter_13_numerical_qualification_plan.md`

This document is the review entry point for NQ4. It describes the reusable
stationary implementation, the independent checks, the authenticated H3+
campaign, and the interpretation boundary of the evidence. The numerical
evidence and stated boundaries were accepted by the user on 2026-09-21; the
separate review record carries that decision and authorizes NQ5.

## 1. Scope and physical states

NQ4 combines the accepted exact Wilson one-electron action, RI--Wilson
Hartree action, and pure-LDA Wilson action into complete instantaneous Hartree
and Kohn--Sham branches. The source in this campaign is a static uniform
magnetic field, so the temporal connection vanishes and the stationary
equation is

```text
K_beta[P] C = S C epsilon,
C^dagger S C = I,
P = C f C^dagger,
```

with one fixed closed-shell occupation `f=(2)`. The two nonlinear branches
are

```text
K_H[P]  = K_1e + J_W[P],
K_KS[P] = K_1e + J_W[P] + V_xc,W[P].
```

Every energy component and lower matrix is evaluated from the same discrete
actions qualified in NQ1--NQ3. The molecular mechanical energies are

```text
U_H  = Tr(P K_1e) + E_H + E_NN,
U_KS = Tr(P K_1e) + E_H + E_xc + E_NN.
```

The runner labels zero-field solutions `field_free_stationary` and nonzero
solutions `finite_field_stationary`. It explicitly records
`adiabatically_prepared` and `sudden_quench` as **not executed**. No propagated
or quenched state is relabeled as a magnetic stationary state.

The physical nonlinear multicentre fixture is H3+, not the inherited
one-electron H3(2+) diagnostic: three protons form an equilateral triangle of
side 1.4 bohr in the xy plane, molecular charge is +1, and the restricted
state has two electrons.

## 2. Reusable implementation

The implementation is in
`src/aion/electronic_structure/wilson_stationary.py`.

### 2.1 Source-independent and source-fixed objects

`ExactWilsonStationaryFactory` prepares the expensive source-independent
objects once:

- the authenticated AO quadrature and backend;
- one `RIWilsonHartreeEvaluator`, including the analytic PySCF auxiliary
  Coulomb metric and reusable auxiliary-potential cache;
- one fixed pure-LDA evaluator; and
- the nuclear repulsion and physical constants.

`factory.model(gauge, branch)` then builds only source-dependent data:

- exact Wilson overlap, kinetic, and nuclear-attraction matrices;
- the source-fixed RI--Wilson three-index tensor; and
- the selected Hartree or LDA Kohn--Sham branch.

`PreparedRIWilsonHartreeAction` in
`src/aion/electronic_structure/ri_wilson_hartree.py` stores a source-fixed
three-index tensor and reuses it for every density evaluation. This is exact
reuse of density-independent action data, not a frozen-density approximation.
`ExactWilsonStationaryModel.for_branch` reuses those geometric actions when
comparing Hartree and LDA branches at the same source.

`prepare_exact_wilson_stationary_model` remains the one-call convenience API;
it is implemented through the reusable factory.

### 2.2 Nonlinear solve

For each density, `ExactWilsonStationaryModel.evaluate` returns one immutable
record containing the complete lower matrix, overlap, all component matrices,
Hartree and XC action records, and kinetic, electron--nuclear, Hartree, XC,
electronic, nuclear-repulsion, and molecular energies.

The reference stationary solver:

1. constructs the contravariant coefficient density `P=C f C^dagger`;
2. evaluates the complete nonlinear lower matrix at that density;
3. solves the Hermitian generalized eigensystem directly with
   `scipy.linalg.eigh(K,S)`;
4. forms the occupied fixed-point density from the lowest orbitals;
5. uses Pulay extrapolation of lower matrices, with damped startup; and
6. repeats until independently evaluated orbital, density fixed-point, and
   energy-change criteria pass.

The solver does **not** build a full-AO Loewdin or Cholesky propagation frame.
The only inverse square root is the small occupied-frame Gram matrix used to
retract a supplied initial occupied frame onto `C^dagger S C=I`. The same
retraction is used in the qualification-only energy-stationarity paths.

The main campaign policy was:

| control | value |
|---|---:|
| maximum iterations | 160 |
| density fixed-point tolerance | `2e-12` |
| generalized orbital tolerance | `2e-12` |
| energy-change tolerance | `2e-13 Ha` |
| startup damping | 0.5 |
| Pulay start | iteration 2 |
| Pulay space | 8 |

The solver is deliberately CPU-only because SciPy supplies the declared
direct generalized eigensystem reference. Complete instantaneous action
evaluation remains backend-neutral and was physically tested on the GPU.

## 3. Independent residuals and algebraic checks

The density residual and orbital residual are intentionally not aliases.
Given an occupied density `P` and its nonlinear matrix `K[P]`, the reported
checks are:

```text
r_orb = ||K C - S C epsilon|| / max(1, ||K C||),
r_den = ||(P_out-P) S|| / max(1, ||P_out S||),
r_com = ||K P S - S P K|| / scale.
```

The final state also records:

- `||C^dagger S C-I||`;
- `Tr(P S)` and its two-electron residual;
- the spectrum of the mixed density `D=P S`;
- the closed-shell polynomial residual `||D^2-2D||`;
- `[epsilon,f]`;
- overlap minimum eigenvalue and condition number; and
- independent orbital-sum double-counting reconstruction.

For Hartree,

```text
U_elec = sum_n f_n epsilon_n - E_H.
```

For Kohn--Sham LDA,

```text
U_elec = sum_n f_n epsilon_n - E_H + E_xc - Tr(P V_xc).
```

The maximum double-counting residual in the full H3+ scan was
`1.3323e-15 Ha`.

## 4. Numerical realization

| axis | declared choice |
|---|---|
| H2 diagnostic | neutral H2, 2 electrons, bond length 1.4 bohr, STO-3G |
| physical target | equilateral H3+, charge +1, 2 electrons, side 1.4 bohr, cc-pVDZ |
| Hartree | RI--Wilson action, `weigend` auxiliary basis, accepted NQ2 rank policy |
| XC | restricted spin-unpolarized pure LDA `lda,vwn`, accepted NQ3 convention |
| main grid | unpruned PySCF level 4, 78,120 points |
| grid refinement | unpruned levels 3, 4, and 5: 45,304, 78,120, 123,904 points |
| magnetic source | static uniform `Bz`, 0 through 0.06 a.u. |
| main gauge | symmetric, origin `(0.17,-0.31,0.23)` bohr |
| independent gauge | Landau, origin `(-0.21,0.37,-0.16)` bohr, x Landau axis |
| precision | CPU float64/complex128, eight-thread ceiling |

The field interval contains nine points:
`0, 0.001, 0.003, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06` a.u.
It was selected after a pilot established a safely positive exact metric at
the upper endpoint and a roughly six-percent maximum triangle-phase spread.

## 5. Field-free recovery

The LDA references are independently converged PySCF density-fitted RKS
states. The Hartree references use a separate PySCF restricted SCF class whose
effective potential is analytic density-fitted J only; no Aion stationary
solver or real-space Wilson tensor is reused as its oracle.

Both comparisons use the authenticated stored PySCF grid to avoid conflating
field-free recovery with the separate unpruned-grid refinement.

| system/branch | energy absolute residual (Ha) | coefficient-density relative residual |
|---|---:|---:|
| H2 Hartree | `3.3070e-10` | `9.6591e-14` |
| H2 LDA KS | `3.3070e-10` | `9.9392e-14` |
| H3+ Hartree | `1.2019e-8` | `7.1281e-8` |
| H3+ LDA KS | `8.2596e-9` | `3.8957e-8` |

The remaining H3+ difference is bounded by the separately measured
real-space RI/grid realization error. It is not an SCF residual: the final
nonlinear residuals are roughly five orders of magnitude smaller.

## 6. H3+ stationary field scan

![H3+ stationary field scan](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq4_stationary_20260921T232232Z_6197d6828bb3/analysis_final_v2/h3plus_stationary_field_scan.png)

Across all 18 exact stationary states:

| diagnostic | maximum or minimum |
|---|---:|
| orbital residual, max | `2.4928e-13` |
| density fixed-point residual, max | `9.0404e-13` |
| commutator residual, max | `3.5257e-13` |
| metric orthonormality residual, max | `4.4409e-16` |
| particle-number residual, max | `8.8818e-16` |
| occupation-spectrum residual, max | `4.7302e-16` |
| closed-shell `D^2-2D` residual, max | `9.2152e-16` |
| exact-overlap minimum eigenvalue, min | `3.4860e-2` |
| Wilson-density imaginary part, max | `1.3010e-18` |
| Wilson-density real minimum, min | `1.2569e-22` |
| direct/factorized density residual, max | `2.4590e-16` |

At `Bz=0.06` a.u., the Hartree and LDA molecular energies have changed by
`1.45696e-3 Ha` and `1.12559e-3 Ha` from their respective zero-field values.

The middle panel records an **exact-minus-P0 one-electron diagnostic** on each
exact stationary density. At the upper endpoint, the maximum triangle-phase
spread is `0.06010`, the exact-minus-P0 one-electron mechanical-matrix change
is `0.02143` in relative Frobenius norm, and the largest contracted
one-electron correction is `2.4904e-3 Ha`. These establish that the interval
reaches non-negligible internal Wilson corrections while the exact metric and
density remain admitted.

This diagnostic is not a self-consistent P0 calculation and is not claimed as
an NQ7 reduced-model result. Full exact/P0/E1/C1 self-consistent comparisons
remain reserved for NQ7.

## 7. Energy stationarity and gauge representatives

![Energy stationarity and gauge checks](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq4_stationary_20260921T232232Z_6197d6828bb3/analysis_final_v2/energy_stationarity_and_gauge.png)

At `Bz=0.03` and `0.06` a.u., both branches were perturbed in independent
real and imaginary unrestricted coefficient directions. Each displaced frame
was retracted to the exact finite-field metric, and central energy derivatives
were evaluated for six step sizes from `1e-2` through `3e-5`. The minimum
three-point coarse convergence order over all eight sequences was `1.99937`,
consistent with the expected second-order central-difference error. The
largest best-step absolute derivative was `6.5281e-9 Ha`; smaller steps enter
the roundoff and quadrature floor rather than continuing monotonically.

Landau-gauge states were solved independently after transforming only the
initial occupied frame. The maximum discrepancies were:

| quantity | maximum residual |
|---|---:|
| molecular energy, absolute | `2.8999e-13 Ha` |
| coefficient-density covariance, relative | `8.5419e-13` |
| complete lower-matrix covariance, relative | `4.2058e-13` |
| pointwise physical Wilson density, relative | `8.6073e-13` |

Thus the result is not merely a matrix congruence applied after a symmetric-
gauge solve: each gauge representative underwent its own nonlinear solve, and
the physical density was compared on the molecular grid.

## 8. Separated numerical errors

![Grid and stationary refinement](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq4_stationary_20260921T232232Z_6197d6828bb3/analysis_final_v2/grid_and_stationary_refinement.png)

Grid refinement was performed independently at `Bz=0`, `0.03`, and `0.06`
for both branches. The maximum level-4 to level-5 energy change was
`6.5291e-9 Ha`. This measured grid error is larger than the converged
stationary-solve energy error and therefore sets the present energy floor.

At `Bz=0.06`, loose, standard, and tight nonlinear policies were restarted
from the field-free occupied frame. The maximum loose-to-tight energy change
was only `1.9806e-13 Ha`; the density and orbital residuals decreased with the
policy. This separates nonlinear-solve error from the dominant grid error.

Auxiliary-basis, retained-rank, and RI solve convergence are inherited from
accepted NQ2 and are not relabeled as new NQ4 evidence. Orbital-basis
convergence is not claimed: cc-pVDZ is the declared NQ4 physical fixture.
Backend error is covered by the physical-GPU test below. Electromagnetic model
error is explicitly deferred to NQ7.

## 9. Backend and software validation

Focused CPU integration tests cover:

- zero- and finite-field convergence of both branches;
- independent orbital and density residuals;
- metric, particle-number, occupation, mixed-density-polynomial, and
  double-counting identities;
- independent field-free PySCF Hartree and LDA recovery;
- independent symmetric/Landau nonlinear solves; and
- real and imaginary retracted energy-stationarity directions.

The physical-GPU test evaluates the complete finite-field LDA action on CPU
and GPU, asserts device residency for one-electron, Hartree, XC, complete
matrix, and energy arrays, and compares all components. It also verifies that
attempting the declared CPU-only stationary eigensolve on the GPU fails
explicitly rather than falling back silently.

Final validation results were:

- Ruff: all checks passed;
- mypy: no issues in 84 source files;
- fast CPU suite: `89 passed, 170 deselected`;
- complete CPU integration suite: `146 passed, 113 deselected`; and
- physical-GPU stationary-action, RI--Hartree, and Wilson-LDA stack:
  `3 passed, 1 warning`.

The expected GPU warning records that CuPy is the GPU4PySCF contraction
engine.

## 10. Evidence and reproduction

Authenticated artifact root:

```text
/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq4_stationary_20260921T232232Z_6197d6828bb3
```

Principal files:

- `result.json`: complete realization, states, residuals, and refinements;
- `stationary_states.npz`: coefficient, density, overlap, and lower-matrix
  arrays for every main H3+ state;
- `h3plus_stationary_scan.csv`: tabular field scan;
- `gauge_checks.csv`, `energy_stationarity.csv`, `grid_refinement.csv`, and
  `stationary_tolerance_refinement.csv`: independent check tables;
- `h2.reference.h5` and `h3plus.reference.h5`: authenticated PySCF inputs;
- `provenance.json` and `completed.json`: source, environment, artifact, and
  completion hashes; and
- `analysis_final_v2/summary.json`: authenticated metrics, proposed thresholds,
  checks, raw hashes, and figure hashes.

Raw hashes authenticated by the analyzer:

| file | SHA-256 |
|---|---|
| `result.json` | `85c5eab8b60afe396c4e605ff3966b3c6153cc792a32f2458a57d9a6785966e3` |
| `provenance.json` | `7a23f0222205c107f64fd984f16cba31eef75d91877323b95e2d1f7c380c0185` |
| `completed.json` | `05436aae6c54838f0fc4d0b0585e352e9ed2c5a56e250c977844c8d01f5b70ce` |
| `analysis_final_v2/summary.json` | `ecb49bdd4f6a594c4e027b72ef56f33b26a749611329656d1b25914c2f07214e` |

Execution command:

```text
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 \
/home/cgs/01_TOOLS/EasyBuild/conda/bin/conda run \
  -p /home/cgs/01_TOOLS/EasyBuild/conda/envs/aion \
  python tools/run_chapter13_nq4_stationary.py \
  --output /home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq4_stationary_20260921T232232Z_6197d6828bb3
```

Analysis command:

```text
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 \
/home/cgs/01_TOOLS/EasyBuild/conda/bin/conda run \
  -p /home/cgs/01_TOOLS/EasyBuild/conda/envs/aion \
  python tools/analyze_chapter13_nq4_stationary.py \
  --raw /home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq4_stationary_20260921T232232Z_6197d6828bb3 \
  --output /home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq4_stationary_20260921T232232Z_6197d6828bb3/analysis_final_v2
```

The earlier directory
`nq4_stationary_20260921T230417Z_6197d6828bb3` is an incomplete, superseded
runner attempt. It stopped before producing `result.json`, `provenance.json`,
or `completed.json` because a complex roundoff particle-number scalar was
cast directly to `float`. It is not evidence and no result from it is used.

The completed `nq4_stationary_20260921T230509Z_6197d6828bb3` package is also
superseded: its numerical data passed, but two traceability strings used
shorthand names rather than actual Chapter 13 equation labels. The final raw
package corrects those identifiers. Its first `analysis_final` directory is a
preserved failed analysis pass: a four-point order fit included one value
already at the finite-difference floor. `analysis_final_v2` uses the same
three genuinely coarse steps as the accepted NQ3 order analysis; it does not
change raw data or any physical residual threshold.

## 11. Equation-to-code-to-test-to-evidence map

| required object | implementation | test | executed evidence |
|---|---|---|---|
| complete Hartree/KS action | `ExactWilsonStationaryModel.evaluate` | `tests/test_wilson_stationary.py` | component energies and lower matrices in `result.json` and raw NPZ |
| reusable source scan | `ExactWilsonStationaryFactory`, `PreparedRIWilsonHartreeAction` | prepared-action reuse tests | nine-field H3+ scan with one shared grid/RI/LDA preparation |
| nonlinear stationary state | `ExactWilsonStationaryModel.solve` | zero/finite-field branch tests | 18 tight H3+ states plus H2 checks |
| independent residuals | `_orbital_residual`, `_mixed_density_residual`, `_commutator_residual` | branch convergence tests | scan table and refinement CSVs |
| metric and occupations | direct generalized eigensolve and final-state audit | invariant assertions | particle, spectrum, polynomial, and orthonormality metrics |
| energy stationarity | exact action on metric-retracted real/imaginary directions | focused direction tests | six-step sequences in `energy_stationarity.csv` |
| double counting | `_build_stationary_state` | branch invariant tests | maximum `1.3323e-15 Ha` |
| gauge-related states | independent source-fixed models and solves | symmetric/Landau test | gauge CSV and physical grid densities |
| field-free recovery | PySCF RKS and independent J-only restricted Hartree | focused recovery test | H2/H3+ recovery block in `result.json` |
| physical GPU parity | backend-neutral `evaluate`, explicit CPU-only `solve` | `tests/test_wilson_stationary_gpu.py` | physical-GPU pytest execution |

## 12. Review boundary

The authenticated analysis passes all proposed thresholds. The evidence
supports the following bounded conclusion:

> For the declared fixed-centre, closed-shell, exact straight-Wilson,
> RI--Hartree and pure-LDA realization, the H2 and physical two-electron H3+
> nonlinear stationary solver recovers the field-free references within the
> separated grid/RI error, produces finite-field stationary states satisfying
> independent orbital and density criteria, is stationary under unrestricted
> real and imaginary coefficient directions, preserves the metric,
> occupations, particle number, and double-counting identities, and gives
> gauge-related physical densities and energies within the measured numerical
> floor over `0 <= Bz <= 0.06` a.u.

This does not accept NQ4 automatically. It does not qualify nonlinear source
derivatives, Ward/continuity identities, nonlinear propagation, reduced
electromagnetic models, orbital-basis convergence, GGA, spin polarization,
moving nuclei, pseudopotentials, or periodic systems. Those remain later gates
or excluded scope.
