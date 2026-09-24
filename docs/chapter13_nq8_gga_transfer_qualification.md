# Chapter 13 NQ8 pure-GGA and molecular-transfer qualification

Date: 2026-09-24

Gate state: **executed and analyzed; awaiting user review**

Accepted prerequisite: `docs/reviews/chapter13_nq7_review_20260924.json`

Controlling plan:
`/home/cgs/00_WORK/Projection_Full_Formalism/REVIEW/implementation/aion_chapter_13_numerical_qualification_plan.md`

This document is the review entry point for NQ8. It distinguishes the
mathematical GGA action, reusable implementation, executed tests, immutable
raw campaign, follow-up numerical-floor refinement, and proposed scientific
interpretation. It does not record acceptance; a separate review record must
carry the user's decision.

## 1. Mathematical contract

For the exact straight-Wilson density and its ordinary spatial gradient,

```text
n_W(r) = P^{ij} chi_i^W(r) chi_j^W(r)^*,
g_W(r) = grad n_W(r),
```

NQ8 selects the restricted pure-GGA action

```text
E_xc,Q[P,A] = sum_g w_g f(n_W(r_g), g_W(r_g)).
```

The weak lower matrix and a fixed-coefficient-history electromagnetic source
direction are exact derivatives of this one grid sum:

```text
(V_xc)_{ij}
  = sum_g w_g [f_n N_{ij} + f_g . grad N_{ij}],

delta_A E_xc,Q
  = sum_g w_g [f_n delta_A n_W + f_g . delta_A g_W].
```

Here `N_ij=(chi_i^W)^* chi_j^W`. No numerical divergence of `f_g` is
formed. Energy, lower matrix, and source derivative use the same grid nodes,
weights, AO values, AO gradients, screening policy, Wilson phases, and
pointwise libxc data.

For an anchor-to-point straight-line integral `a_i(r)`, the dressed AO and
gradient used in all three descendants are

```text
chi_i^W = exp(i q a_i / hbar) phi_i,

grad chi_i^W
  = exp(i q a_i / hbar)
    [grad phi_i + (i q/hbar) grad(a_i) phi_i].
```

The source derivative differentiates both the phase and `grad(a_i)`. This is
essential: differentiating only the density while omitting the density-
gradient direction would not be the derivative of the declared GGA action.

The realization label is **`quadrature--Wilson GGA`**. None of the accepted
LDA evidence is relabeled as GGA evidence.

## 2. Reusable implementation

`src/aion/electronic_structure/wilson_gga.py` provides:

| API | role |
|---|---|
| `WilsonGGAEvaluator` | blocked CPU/GPU evaluation of the common GGA action and its derivatives |
| `WilsonGGAResult` | energy, weak lower matrix, density, density gradient, pointwise derivatives, and optional source directions |
| `WilsonGGAProvenance` | functional, family, realization, libxc, reference, and grid identity |
| `prepare_wilson_gga` | validated pure-GGA construction boundary |

PySCF supplies the molecular grid, AO values and first derivatives, and
`NumInt.eval_xc_eff` pointwise PBE data. Aion owns the complex Wilson frame,
dressed AO gradients, density and density-gradient contractions, source
directions, and weak matrix assembly. On a physical GPU, only real pointwise
functional inputs and outputs cross the host boundary; Wilson and matrix
contractions remain device resident.

The exact nonlinear model now has separate `KOHN_SHAM_LDA` and
`KOHN_SHAM_GGA` branches. Stationary actions, dynamics, weak currents,
continuity, and power consume the selected XC evaluator through one common
action interface. Reduced P0/E1/C1 GGA is deliberately rejected: deriving and
qualifying a gradient expansion of those reduced actions was not part of
NQ8.

## 3. Implementation-level derivative evidence

The focused tests exercise:

- zero-field PBE energy and weak lower matrix against independent PySCF
  `nr_rks` evaluation on exactly the same grid;
- real and imaginary unrestricted coefficient directions, which vary both
  `n_W` and `grad n_W`;
- affine and localized vector-potential source directions, including
  `delta_A grad n_W`;
- magnetic-gauge representative covariance;
- general complex nonunitary coefficient-frame covariance;
- stationary PBE solution, complete weak source finite difference, weak
  continuity, instantaneous power, and short nonlinear propagation; and
- CPU/physical-GPU action and complete weak-current parity and device
  residency.

These tests are independent of the H3+/CO campaign and remain ordinary
regression tests in the reusable Aion package.

## 4. Numerical realization

The authenticated transfer campaign uses:

| axis | value |
|---|---|
| functional | pure PBE GGA |
| realization | `quadrature--Wilson GGA` |
| systems | equilateral H3+ and closed-shell CO |
| electrons | 2 and 14 |
| AO basis | cc-pVDZ |
| auxiliary basis | weigend |
| grid | unpruned PySCF level 4 |
| grid points | 78,120 (H3+), 106,200 (CO) |
| AO count | 15 (H3+), 28 (CO) |
| static field | `Bz=0.03` a.u. |
| dynamic field | `Bz(t)=0.03+0.001t` a.u. with its induction field |
| uniform electric field | `(0.002,-0.001,0.0005)` a.u. |
| propagation window | 0.2 a.u. |
| timestep comparison | 0.05 and 0.025 a.u. |
| propagator | accepted self-consistent two-node fourth-order Gauss--Magnus / [2/2] Pade link |
| metric correction | none |

CO is the required closed-shell second-row transfer. Its field-state density
reaches `n_max=297.701` and `max|grad n|=4521.23`, compared with `0.335365`
and `0.584613` for H3+. Its PBE lower matrix has nonzero C--O and O--C block
norms `1.35724`, alongside C--C `3.33192` and O--O `4.62969`. The transfer
therefore exercises compact core density, steep gradients, and heteronuclear
localized pair blocks rather than merely repeating the diffuse two-electron
case.

## 5. Same-grid and stationary results

| quantity | H3+ | CO |
|---|---:|---:|
| same-grid XC energy residual (Ha) | `1.11e-16` | `5.33e-15` |
| same-grid XC lower relative residual | `3.11e-16` | `4.14e-16` |
| zero-field molecular-energy difference from PySCF (Ha) | `7.57e-9` | `3.78e-7` |
| zero-field orbital residual | `2.71e-13` | `9.24e-13` |
| finite-field orbital residual | `5.50e-13` | `3.92e-12` |
| finite-field minimum metric eigenvalue | `3.52e-2` | `2.17e-2` |

The same-grid comparison isolates the new GGA action and agrees at
floating-point scale. The larger complete stationary-energy differences
include the already declared real-space/RI Hartree realization and remain
below `1e-6` Ha; they are not attributed to the PBE derivative.

## 6. Source derivative

The H3+ complete localized-source derivative has a best raw central-
difference residual of `2.77e-10` a.u. The initial CO scan used only
`1e-3`, `3e-4`, and `1e-4`; its residuals `0.93--2.30e-7` a.u. did not reveal
a truncation region. A separate authenticated refinement therefore repeats
the accepted NQ5 seven-step range from `3e-2` to `3e-5`, decomposes the
one-electron, Hartree, and PBE pieces, and repeats transition-region samples
to distinguish deterministic truncation from evaluation noise.

The refined CO residuals are:

| step | complete-action residual |
|---:|---:|
| `3e-2` | `9.50e-9` |
| `1e-2` | `1.17e-8` |
| `3e-3` | `7.39e-9` |
| `1e-3` | `9.28e-8` |
| `3e-4` | `2.30e-7` |
| `1e-4` | `1.38e-7` |
| `3e-5` | `2.61e-6` |

All repeated `3e-3` and `1e-3` evaluations are bitwise identical. The useful
window is therefore `3e-3--3e-2`; smaller steps amplify deterministic
subtraction error rather than exposing a stochastic evaluator. At the best
`3e-3` point, the one-electron residual is `1.34e-10`, the RI-Hartree
residual is `7.52e-9`, and the PBE residual is `4.91e-17`. The CO PBE source
direction is symmetry-zero (`4.91e-17` analytically and zero by finite
difference); H3+ independently resolves a nonzero PBE source direction of
`5.12e-6` with a `2.77e-10` complete-action residual. The original fine-step
CO discrepancy is therefore an RI-Hartree energy-difference floor, not a
missing density-gradient source term.

The first analysis draft incorrectly required the finite-difference
component sum to meet a fixed `1e-12` tolerance at every point in the sweep,
including the intentionally roundoff-dominated `3e-5` endpoint. Its maximum
composition residual was `2.37e-10`, consistent with subtracting molecular
energies before division by that very small step. The final decision check
uses the component identity at the same best `3e-3` step as the reported
derivative; its residual is exactly zero. The maximum over the full sweep is
still recorded as a diagnostic rather than hidden.

The final numerical-floor interpretation and plot are taken from
`analysis_final_v2/co_source_refinement.csv` and
`analysis_final_v2/co_source_refinement.png`.

## 7. Continuity, power, and propagation

| quantity | H3+ | CO |
|---|---:|---:|
| finite-region weak-continuity residual | `2.37e-13` | `9.38e-21` |
| global charge-rate residual | `5.34e-19` | `6.43e-19` |
| instantaneous power identity residual (Ha/a.u.) | `3.56e-14` | `2.71e-20` |
| fine-step particle-number drift | `2.66e-15` | `1.24e-14` |
| fine-step Hermiticity residual | `5.55e-16` | `1.23e-15` |
| fine-step nonlinear residual | `1.72e-15` | `2.45e-13` |
| final mixed-density difference, `dt=0.05` vs `0.025` | `1.97e-11` | `2.01e-8` |
| fine-step endpoint work--energy residual (Ha) | `1.05e-13` | `4.06e-10` |

Neither propagation applies a metric projection or correction. CO is more
demanding, as expected from its size, occupied space, core density, and steep
gradient. The two-step comparison is a transfer check, not a new estimate of
the integrator order; NQ6 remains the dedicated timestep/order evidence.

## 8. Authentication disclosure

The raw `result.json`, `provenance.json`, both checkpoint JSON files, and both
NPZ archives authenticate against their recorded SHA-256 values. The raw
campaign exposed one archival defect: `run.log` was hashed inside the Python
process before the outer detached shell appended the final completion text.
Consequently its final hash differs from the provenance entry. The log is not
numerical evidence, and the mismatch is retained visibly rather than repaired
in place. Commit `6bd4215` excludes externally managed logs from future
artifact manifests; every numerical artifact remains strictly authenticated.

## 9. Evidence locations

Raw campaign:

```text
/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/
chapter13_wilson_adiabatic_qualification/
nq8_gga_transfer_20260924T180834Z_57eb8a1b538d
```

CO source refinement:

```text
/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/
chapter13_wilson_adiabatic_qualification/
nq8_co_source_refinement_20260924T205718Z_4de1a243e142/results_v2
```

The immutable accepted-candidate analysis is stored under `analysis_final_v2`
in the raw campaign root. `summary.json` is the machine-readable entry point;
the CSV tables and PNG figures are derived views whose hashes are recorded
there. The earlier `analysis_final` directory is retained as a transparent
record of the superseded worst-sweep-point criterion.

## 10. Bounded proposed conclusion

The completed analysis supports the following bounded statement for user
review:

> The exact straight-Wilson RI-Hartree plus adiabatic pure-PBE action, its
> weak lower matrix, fixed-history source derivative, stationary equation,
> weak continuity identity, mechanical power identity, and short nonlinear
> propagation are implemented as descendants of one declared discrete
> action and numerically transfer from H3+ to closed-shell CO in cc-pVDZ over
> the tested grid, field, and time domain.

This evidence does not qualify hybrids, meta-GGAs, nonlocal correlation,
moving nuclei, pseudopotentials, periodic systems, spatially nonuniform
propagated fields, Maxwell backreaction, reduced-model GGA expansions, or an
exact interacting transverse current. NQ8 remains unaccepted until the user
reviews the final authenticated analysis and records a decision.
