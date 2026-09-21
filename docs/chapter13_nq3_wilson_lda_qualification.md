# Chapter 13 NQ3 Wilson-LDA action qualification

Date: 2026-09-21

Gate state: **accepted**

Accepted prerequisite: `docs/reviews/chapter13_nq2_review_20260921.json`

Controlling plan: `/home/cgs/00_WORK/Projection_Full_Formalism/REVIEW/implementation/aion_chapter_13_numerical_qualification_plan.md`

This is the NQ3 review entry point. It qualifies one explicitly fixed,
restricted, unpolarized, pure-LDA quadrature action on the exact Wilson
density. It does not qualify a GGA, a stationary finite-field Kohn--Sham
solution, a nonlinear trajectory, or spectroscopy.

## 1. Mathematical contract

The implementation follows the Chapter 13 definitions
`eq:wilson-hartree-ks-xc-quadrature`,
`eq:wilson-hartree-ks-xc-lda-matrix`,
`eq:wilson-hartree-ks-xc-matter-differential`, and
`eq:wilson-hartree-ks-xc-source-differential`.

For grid nodes and weights \((\mathbf r_g,w_g)\), the Wilson density is

\[
 n_g=\sum_{ij}P^{ij}\chi_i(\mathbf r_g)\chi_j(\mathbf r_g)^*,
\]

and the one declared discrete XC action is

\[
 E_{\mathrm{xc},\mathcal Q}
 =\sum_g w_g n_g\epsilon_{\mathrm{xc}}(n_g).
\]

Writing
\(v_{\mathrm{xc}}(n)=\partial[n\epsilon_{\mathrm{xc}}(n)]/\partial n\),
the lower matrix generated from that same sum is

\[
 V^{\mathrm{xc}}_{ij}
 =\sum_g w_g v_{\mathrm{xc}}(n_g)
       \chi_i(\mathbf r_g)^*\chi_j(\mathbf r_g).
\]

For an unrestricted real or imaginary coefficient direction \(Z\),

\[
 \delta P=ZfC^\dagger+CfZ^\dagger,
 \qquad
 \delta_Z E_{\mathrm{xc},\mathcal Q}
 =\operatorname{Re}\operatorname{Tr}
       (V^{\mathrm{xc}}\delta P).
\]

For a real vector-potential direction at fixed complete coefficient history,

\[
 \delta_\alpha^C E_{\mathrm{xc},\mathcal Q}
 =\sum_g w_g v_{\mathrm{xc}}(n_g)\delta_\alpha^C n_g.
\]

The action, matrix, and source direction use exactly the same grid nodes,
weights, dressed AO values, pointwise density, functional values, and
functional derivatives. There is no separately reconstructed XC matrix or
current formula.

The complete complex Wilson contraction is evaluated before its real part is
passed to the density functional. The implementation rejects an imaginary
component or negative real density outside declared numerical tolerances; it
does not replace the coefficient density by its real part, symmetrize the
pointwise result, clip the density, or project it to positivity.

## 2. Reusable implementation

The reusable implementation is
`src/aion/electronic_structure/wilson_lda.py`:

- `prepare_wilson_lda` prepares an explicitly named pure-LDA action;
- `WilsonLDAEvaluator.evaluate` returns one `WilsonLDAResult` containing the
  energy, lower matrix, pointwise density, energy per particle, density
  derivative, electron-count quadrature, domain diagnostics, and an optional
  fixed-history source density and energy direction;
- `coefficient_frame` applies a fixed square change
  \(\chi'=\chi A\), allowing the same evaluator to test general complex,
  nonunitary coefficient-frame covariance; and
- `WilsonLDAProvenance` records the functional, spin convention, PySCF/libxc
  pointwise engine, libxc version, reference, and grid identities.

Only pure LDA is accepted. Hybrid, exact-exchange, nonlocal-correlation, GGA,
and meta-GGA functionals are rejected at preparation. The present action is
restricted and spin unpolarized.

PySCF/libxc owns the pointwise map
\(n_g\mapsto(\epsilon_{\mathrm{xc},g},v_{\mathrm{xc},g})\) through
`pyscf.dft.numint.NumInt.eval_xc_eff`. Aion owns the Wilson frame, full complex
density and source contractions, quadrature action, and lower matrix. The
pointwise libxc service is presently CPU-only. With a GPU backend, each real
density block is explicitly transferred to the host and the two pointwise
fields are transferred back. All Wilson contractions, grid reductions, lower
matrices, and returned arrays remain on the requested GPU. This is an explicit
backend boundary, not a claim of native GPU libxc execution.

## 3. Declared realization

The authoritative execution uses:

- H2 at \(R=1.4\,a_0\) and LiH at \(R=3.0\,a_0\), both all electron,
  restricted closed shell, and fixed centre;
- the STO-3G orbital basis and accepted field-free coefficient densities;
- exactly one functional identifier, `lda,vwn`, for every calculation;
- unpruned PySCF molecular grids at levels 1--5;
- exact affine straight-Wilson phases at
  \(\mathbf B=(0.013,-0.009,0.017)\) a.u.;
- symmetric and Landau gauge representatives, a physical magnetic source
  direction, and a curl-free pure-gauge source direction;
- general complex nonunitary coefficient-frame changes with condition numbers
  1.057 for H2 and 1.036 for LiH;
- electron charge \(q=-1\), \(\hbar=1\), and float64/complex128; and
- CPU reference execution with every numerical-library pool capped at eight
  physical cores.

Grid refinement and functional selection are distinct axes in the evidence.
Only the grid level changes. No functional comparison or scan is included.

## 4. Independent zero-field reference and grid refinement

At every grid level, an independent PySCF `NumInt.nr_rks` calculation uses the
same stored grid coordinates and weights, the same coefficient density, and
the same `lda,vwn` convention. It does not call the Aion Wilson-LDA evaluator.
Across both systems and all five levels, the worst residuals are:

| Quantity | Maximum residual |
|---|---:|
| XC energy, absolute | `1.33e-15 Ha` |
| lower XC matrix, relative Frobenius | `3.39e-16` |
| integrated electron count, absolute | `4.44e-16` |

These values test implementation agreement on a fixed finite sum. They are
separate from convergence of that sum with the molecular grid. Relative to
level 5, the level-4 results differ by at most `3.27e-9 Ha` in energy and
`1.01e-8` in the lower matrix, both from LiH. H2 is already much better
resolved at level 4: `2.62e-12 Ha` and `1.83e-11`, respectively. The LiH
level-2/3 sequence is mildly nonmonotone, so the full sequence is retained
rather than inferring an order from a selected pair.

![Fixed-functional grid refinement and independent same-grid oracle](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq3_wilson_lda_20260921T211147Z_def189b/analysis_final/grid_refinement_and_pyscf_oracle.png)

## 5. Matter and source derivatives

Real and imaginary unrestricted matter directions show second-order central
finite-difference convergence with a minimum coarse-regime order of 1.998.
The worst best-resolved absolute error is `1.56e-11 Ha`.

The physical magnetic fixed-history source directions have minimum
coarse-regime order 1.996 and worst best-resolved absolute error
`3.30e-12 Ha`. The pure-gauge directions enter their cancellation floor
earlier because their analytic signals are small. Their minimum initial-pair
order is 1.974 and their worst best-resolved error is `4.52e-14 Ha`.
Source-density imaginary contamination never exceeds `3.33e-16`.

The small-step upturns are retained. They are the expected subtractive
cancellation of separately evaluated nonlinear actions and are not hidden by
selecting only the best point.

![Matter and fixed-history source derivative sequences](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq3_wilson_lda_20260921T211147Z_def189b/analysis_final/matter_and_source_derivatives.png)

## 6. Gauge and coefficient-frame covariance

For simultaneous electromagnetic gauge transformation of the source and
coefficient density, the stored double-precision XC energies are identical.
The worst density residual is `2.88e-16`, and the lower matrix satisfies its
gauge covariance law to `1.67e-16` relative.

For the general complex nonunitary change
\(\chi'=\chi A\), \(P'=A^{-1}PA^{-\dagger}\), the stored energies are also
identical. The worst density residual is `4.05e-16`, and the lower matrix
satisfies \(V'=A^\dagger V A\) to `4.05e-16` relative.

![Gauge and coefficient-frame covariance](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq3_wilson_lda_20260921T211147Z_def189b/analysis_final/gauge_and_frame_covariance.png)

Across the field-free and finite-field results, the lower-matrix Hermiticity
residual is at most `6.56e-17`. The largest Wilson-density imaginary residual
is `5.20e-18`; the smallest sampled real density is positive
(`4.06e-51`).

## 7. Executed validation

Focused CPU integration tests cover the independent PySCF same-grid oracle,
lower-matrix derivatives along unrestricted real and imaginary coefficient
directions, physical and pure-gauge fixed-history source directions,
electromagnetic gauge covariance, general complex nonunitary frame
covariance, and rejection of non-LDA functionals.

The physical-GPU test evaluates the energy, lower matrix, pointwise fields,
electron count, and source direction on H2. It asserts device residency for
every returned action array and verifies CPU/GPU parity.

Final validation results are:

- `ruff check src tests tools`: passed;
- `mypy src/aion`: no issues in 83 source files;
- default fast CPU suite: `89 passed, 160 deselected`;
- complete CPU integration suite: `137 passed, 112 deselected`; and
- physical-GPU Wilson-LDA action/source parity: `1 passed, 1 warning` (CuPy
  tensor contraction engine).

## 8. Evidence and provenance

Authoritative raw root:

`/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq3_wilson_lda_20260921T211147Z_def189b`

Principal hashes:

```text
result.json                         5d6ddec7cf34628dc20aa57ef384ece3562a34ee9ac8209f327e672744ed8845
provenance.json                     9a7aa0ea6b402582a25745633fcc11a83e2eca27cd191dae2aa54a0c2b24029e
completed.json                      f7c691d33ab539c23055ef4540ed6e5a1bc542d6ebecaeb3975650255fb91c36
analysis_final/summary.json         92c26c202e78fd9454d1e48782984d13abf045f25f55e3f7817eb956f1b20455
```

The raw artifact contains `result.json`, both authenticated reference HDF5
files and TOML configurations, `grid_refinement.csv`,
`derivative_sequences.csv`, `invariance.csv`, `provenance.json`, and
`completed.json`. `analysis_final` contains the authenticated threshold
summary and three figures.

`analysis_proposed` is deliberately preserved as a superseded first analysis
pass. It authenticated the same immutable raw data and passed the same
thresholds, but its aggregate density-domain metric included only the
zero-field grid rows. `analysis_final` also includes the finite-field base
rows and is the review authority.

## 9. Bounded interpretation and review boundary

This evidence supports one restricted unpolarized `lda,vwn` Wilson action for
the declared molecular, basis, grid, precision, and uniform-magnetic test
domain. It does not establish:

- a GGA density-gradient action;
- native GPU evaluation of libxc pointwise values;
- arbitrary spin polarization, periodic systems, ECPs, or nonlocal ionic
  operators;
- an NQ4 self-consistent finite-field stationary state; or
- NQ5+ nonlinear propagation, power balance, or spectroscopy.

Status separation at this boundary is:

- Chapter 13 Wilson-LDA quadrature equations: **specified**;
- reusable energy, lower-matrix, and source-direction implementation:
  **implemented**;
- CPU refinement/oracle/derivative/covariance campaign and physical-GPU
  parity: **executed**;
- NQ3 scientific decision: **accepted by the user on 2026-09-21**.

The acceptance record is
`docs/reviews/chapter13_nq3_review_20260921.json`. Acceptance is bounded to
the declared restricted unpolarized H2/LiH, STO-3G, `lda,vwn`, unpruned-grid,
uniform-magnetic, complex128 realization and does not extend to NQ4.
