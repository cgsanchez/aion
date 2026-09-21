# Chapter 13 NQ1 exact Wilson-density qualification

Date: 2026-09-21

Gate state: **implemented and executed; awaiting user review**

Accepted prerequisite: `docs/reviews/chapter13_nq0_review_20260921.json`

Controlling plan: `/home/cgs/00_WORK/Projection_Full_Formalism/REVIEW/implementation/aion_chapter_13_numerical_qualification_plan.md`

This is the NQ1 review entry point. The gate qualifies the exact finite-field
Wilson density used by every later Hartree and exchange--correlation closure.
It does not qualify any Hartree/XC functional, lower matrix, nonlinear state,
or trajectory.

## 1. Mathematical contract

The implementation follows these Chapter 13 definitions:

- density map: `eq:wilson-hartree-ks-density-overlap-form` and
  `eq:wilson-hartree-ks-density-orbital-form`;
- normalization: `eq:wilson-hartree-ks-density-normalization`;
- unrestricted matter variation:
  `eq:wilson-hartree-ks-density-matter-rho-variation` and
  `eq:wilson-hartree-ks-density-matter-matrix-variation`;
- fixed-coefficient-history source variation:
  `eq:wilson-hartree-ks-density-source-variation`; and
- integrated source response:
  `eq:wilson-hartree-ks-density-source-number-response`.

For dressed frame values \(\chi_i(\mathbf r)\) and contravariant coefficient
density \(P^{ij}\), the one shared contraction is

\[
 n_{\rm W}(\mathbf r)=\sum_{ij}P^{ij}\chi_i(\mathbf r)\chi_j(\mathbf r)^*.
\]

No real-density shortcut, clipping, symmetrization of the output, or
positivity projection is used. The complex result is retained and its
imaginary floor and real minimum are recorded.

At fixed source, an unrestricted coefficient direction \(Z\) gives

\[
 \delta P=ZfC^\dagger+CfZ^\dagger,
 \qquad
 \delta_Zn_{\rm W}=\chi\,\delta P\,\chi^\dagger.
\]

At fixed coefficient history, a real vector-potential direction with
straight-path response \(b_i[\alpha]\) gives

\[
 \delta_\alpha\chi_i={iq\over\hbar}b_i[\alpha]\chi_i,
 \qquad
 \delta_\alpha^C n_{\rm W}
 =\delta\chi\,P\chi^\dagger+\chi P\delta\chi^\dagger.
\]

The production coefficient-space normalization uses the same stable
zero-field correction as the accepted exact one-electron evaluator,

\[
 S_{\rm stable}=S_{\rm grid}+\Theta\left(S_0-S_{0,{\rm grid}}\right),
\]

not the tautological contraction with the raw density quadrature itself.
Therefore \(|\int n_{\rm W}-\operatorname{Tr}(PS_{\rm stable})|\) measures
the unresolved molecular-grid error.

## 2. Reusable implementation

The reusable kernels are in
`/home/cgs/00_WORK/Projection_Code/aion/src/aion/electronic_structure/wilson_density.py`:

- `contract_wilson_density_block` is the complete complex contraction and
  explicit coefficient-frame boundary;
- `evaluate_exact_uniform_magnetic_wilson_density` returns direct and
  endpoint/triangle-factorized densities, raw and stable overlaps, particle
  numbers, and domain diagnostics;
- `evaluate_exact_uniform_magnetic_wilson_density_matter_direction`
  implements the unrestricted real or imaginary coefficient direction; and
- `evaluate_exact_uniform_magnetic_wilson_density_source_direction`
  implements a real vector-potential direction while holding the complete
  coefficient history fixed.

`/home/cgs/00_WORK/Projection_Code/aion/src/aion/electromagnetism/test_variations.py`
adds `AffineGaugeDifferenceVariation`. It represents the curl-free
difference of two affine gauges with the same magnetic field. This is a
fixed-coefficient source direction, not a simultaneous gauge transformation
of the matter coefficients.

All point/pair work remains block-local. No `(npoint, nao, nao)` pair field is
retained. CPU and GPU arrays remain on their selected backend, and the public
API uses complex128.

## 3. Numerical realization

The final execution uses:

| Axis | Choice |
|---|---|
| Systems | H2 at 1.4 bohr and LiH at 3.0 bohr, both neutral restricted closed shell |
| Orbital basis | STO-3G |
| SCF datum | `lda,vwn`, PySCF density fitting with `weigend`, field-free coefficient density |
| Field | oblique \(B=(0.013,-0.009,0.017)\) a.u. |
| Gauge comparison | symmetric origin `(0.17,-0.31,0.23)` versus Landau origin `(-0.21,0.37,-0.16)` with simultaneous coefficient transformation |
| Grid refinement | unpruned PySCF levels 1, 2, 3, 4, and 5 |
| Matter directions | deterministic unrestricted real and imaginary coefficient directions |
| Physical source direction | \(\delta B=(-0.4,0.7,0.2)\) a.u., coefficients fixed |
| Pure-gauge source direction | Landau representative minus symmetric representative at fixed \(B\), coefficients fixed |
| Difference steps | `1e-1, 3e-2, ..., 3e-6, 1e-6` |
| Reference backend | CPU float64/complex128, eight-thread ceiling |

The finite-field coefficient density is deliberately the field-free density.
Consequently \(\operatorname{Tr}(PS(B))\) is not exactly the integer electron
count. NQ1 tests equality of the spatial integral with that coefficient-space
quantity, as required by the theorem; it does not incorrectly force either
value to two or four.

## 4. Raw campaign and refinement decision

The final raw execution is:

```text
/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq1_wilson_density_20260921T184528Z_28b20b8
```

The earlier `nq1_wilson_density_20260921T184329Z_28b20b8` execution is a
preserved pilot. It stopped at grid level 4. That was enough for H2
(`2.19e-11`) but left LiH at `7.47e-9` after non-monotone levels 2--3, so no
tolerance was inferred from it. The final campaign added level 5, reducing
LiH's residual to `9.39e-12` and establishing the resolved sequence.

Normalization residuals were:

| System | level 1 | level 2 | level 3 | level 4 | level 5 |
|---|---:|---:|---:|---:|---:|
| H2 | `1.42e-6` | `1.34e-9` | `3.25e-10` | `2.19e-11` | `2.32e-11` |
| LiH | `1.66e-5` | `1.11e-6` | `9.88e-7` | `7.47e-9` | `9.39e-12` |

The H2 level-4/5 values identify its quadrature floor. LiH requires level 5
for this field, basis, and density.

![Wilson-density normalization refinement](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq1_wilson_density_20260921T184528Z_28b20b8/analysis_final/normalization_refinement.png)

## 5. Results and proposed tolerances

The pilot established the scale of the grid and floating-point floors. The
following tolerances were then applied to the final level-5 execution. All
checks passed.

| Diagnostic | Worst observed | Proposed tolerance |
|---|---:|---:|
| Density imaginary part | `5.20e-18` | `1e-14` |
| Minimum real density | `4.06e-51` | `>= -1e-14` |
| Direct/factorized density residual | `2.54e-16` | `1e-13` |
| Direct/factorized overlap residual | `3.99e-16` | `1e-13` |
| Stable-overlap independent-oracle residual | `5.12e-16` | `1e-13` |
| Finest normalization residual | `2.32e-11` | `5e-10` |
| Electromagnetic-gauge density residual | `2.87e-16` | `1e-12` |
| Gauge-related stable particle-number residual | `4.44e-16` | `1e-12` |
| General complex coefficient-frame residual | `3.70e-16` | `1e-12` |
| Matter-direction best central-difference error | `1.68e-12` | `1e-10` |
| Source-direction best central-difference error | `1.53e-9` | `5e-9` |
| Source-direction best maximum absolute error | `4.92e-11` | `1e-10` |
| Minimum source coarse-regime order | `1.959` | `>= 1.9` |
| Direction imaginary part | `3.33e-16` | `1e-14` |
| Integrated source/stable metric response | `1.07e-15` | `5e-12` |

The coefficient-frame tests use general complex, mildly nonunitary changes
with condition numbers 1.10 (H2) and 1.07 (LiH), together with the exact
contravariant congruence transformation \(P'=A^{-1}PA^{-\dagger}\).

The matter forward differences show the expected linear truncation: reducing
the step from `0.1` to `0.01` reduces the error by a factor of ten in all four
system/direction cases. Because the density is quadratic in the coefficient
array, the central difference cancels that quadratic term exactly and exposes
roundoff directly.

The source central differences show second-order truncation followed by the
roundoff rise. The first-pair orders are 2.000 and 1.992 for H2 physical and
pure-gauge directions, and 1.997 and 1.959 for LiH. The pure-gauge signal
reaches its numerical floor sooner because its direction amplitude is smaller.

![Matter and source derivative sequences](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq1_wilson_density_20260921T184528Z_28b20b8/analysis_final/derivative_sequences.png)

## 6. Independent checks and limitations

The direct dressed-AO density and the endpoint-link/triangle-factorized
density are independent algebraic routes. The stable overlap is additionally
compared against the accepted exact static one-electron evaluator rather than
against itself. Physical source normalization is compared with its accepted
analytic magnetic-matrix direction; pure-gauge normalization is compared
with the exact anchor-phase congruence direction.

Remaining limitations are explicit:

- positivity was exercised on the two positive-semidefinite closed-shell
  ground-state coefficient densities, not arbitrary invalid Hermitian arrays;
- the high-level direct/factorized evaluator is specialized to an affine
  uniform-magnetic base, although its source-direction protocol already
  accepts general straight-line vector-potential variations;
- density gradients required by a future GGA gate are not part of NQ1;
- no Hartree or XC action, matrix, source derivative, stationary state, or
  nonlinear propagation is claimed; and
- GPU parity is supporting implementation evidence, not a replacement for
  the two CPU algebraic routes and analytic matrix oracles.

## 7. Commands, tests, and artifacts

The raw campaign command was:

```text
module purge
module use /home/cgs/01_TOOLS/EasyBuild/modules/production
module load SciStack/2026.07.1-gnu
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 \
/home/cgs/01_TOOLS/EasyBuild/conda/bin/conda run \
  -p /home/cgs/01_TOOLS/EasyBuild/conda/envs/aion \
  python tools/run_chapter13_nq1_wilson_density.py \
  --output /home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq1_wilson_density_20260921T184528Z_28b20b8
```

The verifier and figure command was:

```text
/home/cgs/01_TOOLS/EasyBuild/conda/bin/conda run \
  -p /home/cgs/01_TOOLS/EasyBuild/conda/envs/aion \
  python tools/analyze_chapter13_nq1_wilson_density.py \
  --execution /home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq1_wilson_density_20260921T184528Z_28b20b8 \
  --output /home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq1_wilson_density_20260921T184528Z_28b20b8/analysis_final
```

Validation results:

- Ruff over `src`, `tests`, and `tools`: passed;
- Mypy over all 81 Aion source files: passed;
- default fast suite: `89 passed, 144 deselected`;
- relevant CPU integration suite: `26 passed, 1 deselected`;
- physical GPU density, matter-direction, and source-direction parity:
  `1 passed, 1 warning` (CuPy contraction engine); and
- authenticated analysis: every proposed threshold passed.

Raw artifact entry points are `result.json`, `grid_refinement.csv`,
`derivative_sequences.csv`, `invariance.csv`, `provenance.json`, and
`completed.json`. Principal hashes are:

```text
result.json                 415b1b1a8f4f6c139a227581a5f0289c5e7bfffbaf845e9fae511af94f2c950d
provenance.json             6e1c99ebfcb536b9d67c9c7c4430402bea2eafeede2be9be0ccc71282f0da761
completed.json              e0d594ac7dc5166e10a94c8fc28bb0ebb2390c1d586942416300646b43c0fe0f
analysis_final/summary.json 51576f6cbc651c2c0d66954bea2f3c7286c534f8b9e3a8d07690787e113cf832
```

## 8. Review boundary

Status separation at this boundary is:

- mathematical definitions: **specified**;
- reusable density and direction kernels: **implemented**;
- CPU refinement, derivative, invariance, and physical-GPU checks:
  **executed**;
- NQ1 scientific decision: **awaiting user review**.

The evidence supports accepting NQ1 for the declared H2/LiH, STO-3G,
uniform-magnetic, complex128 realization. If accepted, the next gate is NQ2:
one variational RI--Wilson Hartree energy, its unrestricted lower-matrix
derivative, and its fixed-history source derivative.
