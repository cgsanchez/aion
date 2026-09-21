# Chapter 13 NQ2 variational RI--Wilson Hartree qualification

Date: 2026-09-21

Gate state: **accepted**

Accepted prerequisite: `docs/reviews/chapter13_nq1_review_20260921.json`

Controlling plan: `/home/cgs/00_WORK/Projection_Full_Formalism/REVIEW/implementation/aion_chapter_13_numerical_qualification_plan.md`

This is the NQ2 review entry point. The gate qualifies one Coulomb-metric
RI--Wilson Hartree energy, its unrestricted matter derivative, and its
fixed-coefficient-history electromagnetic derivative as descendants of one
discrete action. It does not qualify exchange--correlation closure,
nonlinear stationary states, or nonlinear propagation.

## 1. One discrete Hartree action

For real gauge-neutral auxiliary functions (B_P), Aion uses the analytic
PySCF Coulomb metric

\[
 M_{PQ}=(P|Q)
\]

and evaluates the Coulomb potential of each auxiliary function on the
declared molecular grid,

\[
 u_P(\mathbf r_g)
 =\int {B_P(\mathbf r')\over|\mathbf r_g-\mathbf r'|}\,d^3r'.
\]

For Wilson-dressed AO values \(\chi_i(\mathbf r_g)\), the same grid and
weights define

\[
 B_{P,ij}^{\mathcal Q}
 =\sum_g w_g u_P(\mathbf r_g)
       \chi_i(\mathbf r_g)^*\chi_j(\mathbf r_g),
 \qquad
 b_P=\sum_{ij}P^{ji}B_{P,ij}^{\mathcal Q}.
\]

The declared retained metric inverse is (M_R^+). The complete stored
action package is

\[
 d=M_R^+b,
 \qquad
 E_{\rm H,RIW}^{\mathcal Q}={1\over2}b^Td,
 \qquad
 J_{ij}^{\mathcal Q}=\sum_Pd_PB_{P,ij}^{\mathcal Q}.
\]

Consequently,

\[
 \operatorname{Tr}(JP)=2E_{\rm H,RIW}^{\mathcal Q},
 \qquad
 \delta_ZE_{\rm H,RIW}^{\mathcal Q}
 =\operatorname{Re}\operatorname{Tr}(J\,\delta_ZP).
\]

No independent matrix builder or energy hint is used. The common-field
Hartree functional includes self-interaction, as specified by Chapter 13.
It is not a self-interaction-corrected Hartree functional.

For a real vector-potential direction \(\alpha\) at fixed coefficient
history,

\[
 \delta_\alpha\chi_i
 ={iq\over\hbar}b_i[\alpha]\chi_i,
\]

which is inserted into the same three-index grid action to form
\(\delta_\alpha B_{P,ij}^{\mathcal Q}\) and
\(\delta_\alpha b_P\). Since the auxiliary functions and Coulomb kernel are
source independent,

\[
 \boxed{\delta_\alpha^CE_{\rm H,RIW}^{\mathcal Q}
 =d^T\delta_\alpha b},
 \qquad \delta_\alpha M=0.
\]

The fitted-coefficient derivative is absent because (d) is stationary for
the same discrete energy. The source tests hold the complete coefficient
array and occupations fixed.

## 2. Reusable implementation

The reusable implementation is
`src/aion/electronic_structure/ri_wilson_hartree.py`:

- `RIMetricRankPolicy` declares the absolute/relative eigenvalue thresholds
  and optional maximum retained rank;
- `prepare_ri_wilson_hartree` prepares the authenticated PySCF auxiliary
  molecule, analytic Coulomb metric, retained inverse, and optional blocked
  host cache of auxiliary Coulomb potentials;
- `RIWilsonHartreeEvaluator.evaluate` builds the complete complex Wilson
  three-index tensor and returns one `RIWilsonHartreeResult` containing the
  energy, lower matrix, fitted moment and coefficients, pair-counting and
  stationarity diagnostics, and an optional fixed-history source
  derivative; and
- `AOQuadrature.pyscf_molecule` is the read-only bridge through which
  analytic integral services consume the authenticated molecule without
  changing atoms, basis, units, or AO order.

PySCF owns the analytic auxiliary data. `int2c2e` supplies (M), and a
cross `int2c2e` against PySCF normalized narrow Gaussian point charges
(exponent (10^{16})) supplies (u_P(\mathbf r_g)). Aion owns the Wilson
phases, complex pair contraction, density contraction, retained metric
inverse, energy, lower matrix, and source direction.

The auxiliary potential cache is host-resident because it is produced by
PySCF/libcint. Each block is explicitly transferred to the selected Aion
array backend. All Wilson pair contractions, action contractions, and
returned numerical arrays remain backend resident. Physical-GPU parity is
tested; this boundary is not represented as a GPU analytic-integral
implementation.

The current realization materializes (B_{P,ij}^{\mathcal Q}) for one
instantaneous action evaluation. It does not retain a space-by-pair
trajectory. A future large-basis density-first/block-reduction layout may
avoid this instantaneous three-index storage, but it must remain a
derivative of the same declared discrete action.

## 3. Declared realization

The authoritative execution uses:

- H2 at (R=1.4\,a_0) and LiH at (R=3.0\,a_0), both all electron,
  restricted closed shell, and fixed centre;
- STO-3G orbital basis and the accepted ground-state coefficient densities;
- `weigend` auxiliary basis: 22 functions for H2 and 60 for LiH;
- the NQ0-accepted PySCF absolute Coulomb-metric threshold
  `df.incore.LINEAR_DEP_THR = 1e-7`, with no maximum rank;
- all 22 H2 and all 60 LiH metric directions retained;
- unpruned PySCF molecular grids at levels 1--5;
- exact affine straight-Wilson phases at
  \(\mathbf B=(0.013,-0.009,0.017)\) a.u.;
- symmetric and Landau gauge representatives, a physical magnetic source
  direction, and a curl-free pure-gauge source direction;
- electron charge (q=-1), \(\hbar=1\), float64/complex128; and
- CPU reference execution with every numerical-library pool capped at eight
  physical cores.

At level 5 the host auxiliary-potential caches are 14,537,600 bytes for H2
and 58,632,960 bytes for LiH. Cache memory is declared and may be bounded or
disabled. The retained metric condition numbers are (8.39\times10^3) and
(7.66\times10^4), respectively.

## 4. Independent references

The field-free three-index reference uses
`pyscf.df.incore.aux_e2`, not Aion's grid-fitted tensor. With the same
analytic metric and retained space, it independently supplies the RI energy
and Coulomb matrix.

The second reference contracts PySCF analytic four-centre `int2e` integrals
with exactly the same AO density. It does not reuse the fitted result. The
selected `weigend` RI energy differs from this exact-in-orbital-basis
Coulomb reference by

- (5.09\times10^{-5}) relative for H2; and
- (6.90\times10^{-5}) relative for LiH.

These are auxiliary-basis approximation errors. They are not grid,
matter-derivative, or source-derivative residuals. Enlarging the auxiliary
basis reduces the energy error: the best tested H2 basis
(`cc-pVQZ-jkfit`, 102 functions) reaches (1.10\times10^{-5}), while the
tested LiH `def2-universal-jkfit` basis reaches (3.02\times10^{-5}).
These named bases are not strictly nested, so the scan is refinement and
sensitivity evidence rather than a mathematical monotonicity claim.

## 5. Convergence and derivative results

The authenticated analysis passes every proposed threshold. Across H2 and
LiH, the finest level-5 grid gives maxima of

| Quantity | Maximum residual |
|---|---:|
| RI energy relative to analytic PySCF RI | (1.53\times10^{-10}) |
| lower Coulomb matrix relative to analytic PySCF RI | (4.23\times10^{-10}) |
| three-index tensor relative to analytic PySCF RI | (5.15\times10^{-10}) |

The rank scan is physically informative: aggressive truncation does not
give a harmless compression for these densities. H2 becomes numerically
full-action accurate only near rank 20 of 22, and LiH retains visible error
until near its full 60-dimensional space. The selected (10^{-7}) absolute
policy retains every direction for both systems.

The production action uses the deterministic retained eigenspace inverse.
A separate qualification-only conjugate-gradient sequence evaluates the
same stationary auxiliary quadratic form at requested tolerances from
(10^{-2}) to (10^{-12}). At the tightest tolerance, the worst achieved
relative residual is (4.10\times10^{-13}) and the recorded stationary
objective error is at the floating-point floor. This sequence qualifies
solve accuracy; it is not a second production Hartree definition.

For the finite-field action:

- pair counting is satisfied to (1.78\times10^{-15}) Ha;
- lower-matrix Hermiticity is satisfied to (6.42\times10^{-17}) relative;
- the retained solve residual is at most (5.04\times10^{-13});
- the stationary energy residual is at most (2.72\times10^{-12}) Ha;
- gauge-related energies agree within (5.20\times10^{-12}) Ha and fitted
  moments within (1.60\times10^{-16}) relative; and
- general complex nonunitary coefficient reframing preserves energy within
  (3.44\times10^{-12}) Ha, fitted moments within (6.08\times10^{-16}),
  and the lower congruence law within (6.82\times10^{-13}) relative.

Real and imaginary unrestricted matter directions show second-order central
finite-difference convergence with minimum coarse-grid order 1.997. Their
worst best-resolved absolute derivative error is (1.52\times10^{-9}) Ha.
The physical source directions show minimum coarse-grid order 1.996. Across
physical and pure-gauge directions, the worst best-resolved absolute source
error is (1.91\times10^{-8}) Ha. The LiH source floor reflects subtractive
cancellation of separately evaluated stationary actions; the full error
sequence is retained rather than hiding the small-step upturn. Source
moments have imaginary contamination below (7.12\times10^{-17}).

The review figures are:

- `analysis_authoritative/grid_and_auxiliary_convergence.png`;
- `analysis_authoritative/rank_and_solve_convergence.png`; and
- `analysis_authoritative/derivative_sequences.png`.

They distinguish grid, auxiliary-space, retained-rank, solve, truncation,
and roundoff regimes.

## 6. Executed validation

Focused CPU integration tests cover the independent analytic three-index
and four-index references, pair counting, Hermiticity, unrestricted real
and imaginary matter directions, physical and pure-gauge source directions,
electromagnetic gauge invariance, general complex coefficient reframing,
rank policy, and cache policy.

The physical-GPU test evaluates the energy, lower matrix, fitted data,
three-index tensor, and source derivative on H2 and verifies CPU/GPU parity
while asserting that every returned action array is GPU resident.

Final validation results are:

- `ruff check src tests tools`: passed;
- `mypy src/aion`: no issues in 82 source files;
- default fast CPU suite: `89 passed, 152 deselected`;
- NQ0/NQ1/NQ2 and inherited exact-one-electron integration set:
  `39 passed, 1 deselected`; and
- physical-GPU RI action/source parity: `1 passed, 1 warning` (CuPy tensor
  contraction engine).

## 7. Evidence and provenance

Authoritative raw root:

`/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq2_ri_wilson_hartree_20260921T200959Z_6a59d5421654`

Principal hashes:

```text
result.json                              643f65cfc029b968a259de8cbd64cac77115af5b9871f0443687aef751fc293d
provenance.json                          862f8f3ba7f2afb41fad670a089f4ecba007865daa11454c92b76b595c63d8ad
completed.json                           34c38621f89b34c02526609ebb05fd3b736ff3c22ae1ab5e01e41b9136708f47
analysis_authoritative/summary.json      31ecee0c93ac15ce36b91fc0f69d9c15641e590a7da17211f25ec7d3a79bfd83
```

The raw artifact contains `result.json`, both authenticated reference HDF5
files and TOML configurations, `grid_refinement.csv`,
`auxiliary_convergence.csv`, `rank_convergence.csv`,
`solve_convergence.csv`, `derivative_sequences.csv`, `invariance.csv`,
`provenance.json`, and `completed.json`.

Two earlier raw roots are deliberately preserved as pilots:

- `nq2_ri_wilson_hartree_20260921T195752Z_6a59d5421654` used a relative
  (10^{-12}) rank safeguard before reconciliation with the accepted NQ0
  absolute PySCF threshold; and
- `nq2_ri_wilson_hartree_20260921T200658Z_6a59d5421654` used the corrected
  threshold but preceded the final public read-only PySCF-molecule bridge.

Neither pilot is the review authority.

## 8. Review boundary

Status separation at this boundary is:

- Chapter 13 RI--Wilson Hartree equations: **specified**;
- reusable action, lower-matrix, and source-direction implementation:
  **implemented**;
- CPU convergence/derivative/invariance campaign and physical-GPU parity:
  **executed**;
- NQ2 scientific decision: **accepted by the user on 2026-09-21**.

The acceptance record is
`docs/reviews/chapter13_nq2_review_20260921.json`. Acceptance is bounded to
the declared H2/LiH, STO-3G, `weigend`, unpruned-grid, uniform-magnetic,
complex128 realization. It authorizes NQ3: one declared Wilson LDA grid
energy, its lower matrix, and its fixed-history source derivative, again
descending from one discrete action.
