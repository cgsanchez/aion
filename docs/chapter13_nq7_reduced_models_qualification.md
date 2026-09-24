# Chapter 13 NQ7 reduced electromagnetic descendants qualification

Date: 2026-09-24

Gate state: **executed and analyzed; awaiting user review**

Accepted prerequisite: `docs/reviews/chapter13_nq6_review_20260923.json`

Implementation commit: `e11a340c1e0d3049341e85e3dc26b15c7e743a26`

Raw-synthesis commit: `1427fb0`

Evidence-analysis commit: `edeb52e`

Controlling plan:
`/home/cgs/00_WORK/Projection_Full_Formalism/REVIEW/implementation/aion_chapter_13_numerical_qualification_plan.md`

This document is the review entry point for NQ7. It distinguishes the
reduced-action mathematics, reusable implementation, executed calculations,
and the interpretation proposed for review. It does not accept the gate. A
separate review record may be created only after the user reviews and accepts
the evidence.

## 1. Question and conclusion proposed for review

NQ7 asks how the P0, E1, strict-C1, and density-resummed-C1 nonlinear actions
compare with the accepted exact-Wilson Hartree and pure-LDA actions when the
electronic closure and numerical accuracy are matched.

The evidence supports the following bounded conclusions.

1. All four reduced actions recover the exact Wilson action at zero magnetic
   field to the stationary numerical floor, about `3.5e-13` in mixed-density
   relative error.
2. E1 is an exact reduction for the tested zero-magnetic-field, spatially
   uniform electric source. Its finest final-state, energy-trajectory, and
   power-trajectory errors are at most `6.59e-13`, `3.74e-13`, and `1.01e-12`.
3. At nonzero magnetic field, complete first order removes the leading
   state and lower-matrix error. The low-field P0 mixed-density error is first
   order in field (`1.0000` fitted order), while strict C1 is second order
   (`1.99996`). Across the stationary main domain, C1 improves the mixed
   density over P0 by at least a factor of `13.43`.
4. On the two magnetic dynamic cases, C1 improves the final mixed density
   over the better of P0 and E1 by at least a factor of `5.69`. The largest
   C1 final-state error is `1.556e-3`.
5. Those dynamic model differences are resolved from time discretization:
   the smallest C1 model-error to `N=16` versus `N=32` timestep-difference
   ratio is `6.79e3`.
6. C1 does not uniformly reduce every scalar energy or power error. In the
   symmetric stationary fixture, both P0 and C1 energy errors are second
   order, and their coefficients are similar. The power/source derivative of
   an omitted second-order action term is first order and likewise need not
   show the state-error improvement.
7. Strict C1 and density-resummed C1 have the same observed metric and
   density domain and split 16 magnetic dynamic error comparisons evenly,
   eight wins each. The resummation has no systematic accuracy or stability
   advantage in this campaign.
8. Strict C1 is therefore the proposed working action within the tested
   domain. It is the controlled degree-one variational action; the resummed
   model retains undeclared higher closure powers without extending the
   observed domain.
9. This recommendation is basis conditioned. Reduced overlap matrices fail
   positivity much earlier in diffuse bases, while the independently
   recomputed exact Wilson overlap remains positive at every stress point.
   No failed point was clipped, regularized, or silently projected.

The qualified main domain is equilateral H3+ in cc-pVDZ/weigend with
`lda,vwn`, stationary fields `0 <= Bz <= 0.06` a.u., and the declared driven
trajectories of length `2.0` a.u. This is not a transfer claim for other
geometries, bases, fields, functionals, or times.

## 2. Mathematical model hierarchy

All levels retain the exact endpoint Wilson links. They differ in the
endpoint-removed internal finite-spread action retained before matter
differentiation.

Let the real-space density expansion be

```text
n[P,A] = n0[P,A] + n1[P,A] + O(curvature^2),
```

and similarly expand the lower one-electron matrices, overlap, and temporal
connection. The implemented levels are:

| action label | retained one-electron geometry | electric connection | nonlinear closure |
|---|---|---|---|
| P0 | exact endpoint links; internal spread deleted | endpoint/site contribution only | action evaluated with `n0` |
| E1 | same spatial action as P0 | adds the first uniform-electric internal increment | action evaluated with `n0` |
| strict C1 | every degree-zero and degree-one magnetic one-electron, metric, connection, and density term | includes E1 | Hartree and LDA actions truncated consistently through degree one |
| density-resummed C1 | same first-order one-electron and density input as strict C1 | includes E1 | unexpanded Hartree and LDA functionals evaluated on `n0+n1` |

Strict C1 and density-resummed C1 are different actions. The latter is not a
repair or numerical stabilization of the former.

The propagated variable is the contravariant coefficient density

```text
P = C f C^dagger,
```

and the physical mixed tensor is reconstructed as `D=P S`. Every reduced
level supplies one internally consistent triple `(S, omega, K[P])` to the
accepted NQ6 nonlinear Gauss--Magnus propagator. Energies, lower matrices,
fixed-history source responses, and on-shell power are evaluated from that
same selected reduced action. No observable is borrowed from the exact model
or another approximation level.

The complete mechanical energy is

```text
U[P] = Tr(P K_1e) + E_H[n] + E_xc[n] + E_NN,
```

with the pure-LDA term omitted on the Hartree branch. The scalar-potential
coupling remains in the temporal connection. The on-shell source power is
the fixed-history action derivative plus the transport contribution induced
by the selected connection. Consequently each level separately tests

```text
U(T)-U(0) = integral_0^T P_source(t) dt.
```

## 3. Reusable implementation

The implementation is in
`src/aion/electronic_structure/reduced_wilson.py` and is exported through
`aion.electronic_structure`.

| reusable API | role |
|---|---|
| `ReducedWilsonLevel` | unambiguous P0, E1, strict-C1, and density-resummed-C1 labels |
| `PreparedReducedWilsonFactory` | prepares common exact quadrature, RI, LDA, electric, and analytic first-magnetic data once |
| `PreparedReducedWilsonClosure` | evaluates `n0`, `n1`, RI Hartree, strict/resummed LDA, energies, and unrestricted matter derivatives |
| `PreparedReducedWilsonSpatialAction` | caches source-independent spatial data for reuse along dynamic samples |
| `ReducedWilsonModel` | evaluates the chosen action, solves the common stationary problem, constructs the EOM triple, source response, and power |
| `ReducedWilsonOneElectronData` | selected metric, kinetic, nuclear attraction, endpoint links, positivity, and condition diagnostics |
| `ReducedWilsonTimeConnection` | selected connection, metric rate, E1 residual, and metric-compatibility diagnostic |
| `ReducedWilsonActionEvaluation` | component energies and complete lower mechanical matrix from one declared action |
| `ReducedWilsonSourceResponse` | complete fixed-coefficient-history action derivative |
| `ReducedWilsonPowerObservation` | selected action's transport rate and source power |

The implementation reuses the accepted exact-Wilson AO quadrature, exact
endpoint geometry, RI--Wilson Hartree action, Wilson-LDA evaluator, analytic
magnetic one-electron first derivatives, uniform-electric E1 tensor,
stationary solver, and nonlinear contravariant-density propagator. There is
no second ad-hoc formulation stack.

Domain validation is part of the public action boundary. A nonpositive
reduced metric raises `FormulationError`. A reduced density outside the
declared LDA domain also raises visibly. No clipping or spectral repair is
performed.

The reusable unit tests in `tests/test_reduced_wilson.py` cover:

- exact zero-field recovery at every level;
- the exact E1 zero-B uniform-electric connection;
- unrestricted action derivatives for the reduced lower matrix;
- separation of strict and density-resummed actions at second order;
- reuse of common prepared spatial data;
- C1 metric compatibility under time-dependent magnetic field;
- use of the common stationary solver;
- fixed-history source-response finite differences; and
- agreement of action-derived power with the total action derivative on the
  EOM shell.

`tests/test_reduced_wilson_gpu.py` independently exercises the same reduced
actions, source responses, and power on the physical GPU without fallback.

Final verification on the evidence branch produced:

| check | result |
|---|---|
| `ruff check src tests tools` | all checks passed |
| `mypy src` | no issues in 87 source files |
| default fast CPU suite | 95 passed, 201 deselected |
| focused reduced-Wilson and exact-dynamic CPU integration suite | 23 passed |
| focused physical-GPU reduced-action parity | 1 passed, 1 expected CuPy contraction-engine warning |

All CPU invocations used the eight-thread ceiling. The GPU result was run
through `tools/gpu-python`, whose launch contract rejects a missing physical
device or silent CPU fallback.

The restartable campaign driver is
`tools/run_chapter13_nq7_reduced_models.py`. The raw closure verifier is
`tools/synthesize_chapter13_nq7_reduced_models.py`, and the independent
analysis is `tools/analyze_chapter13_nq7_reduced_models.py`.

## 4. Numerical realization

| axis | value |
|---|---|
| system | equilateral H3+, molecular charge +1, two electrons |
| main AO basis | cc-pVDZ |
| domain-stress bases | STO-3G, cc-pVDZ, aug-cc-pVDZ |
| auxiliary basis | weigend |
| branches | RI--Wilson Hartree; RI-Hartree + pure `lda,vwn` |
| main grid | unpruned PySCF level 4 |
| domain grid | unpruned PySCF level 3 |
| gauge | symmetric affine gauge, origin `(0.17,-0.31,0.23)` bohr |
| stationary fields | `0,0.001,0.003,0.01,0.02,0.03,0.04,0.05,0.06` a.u. |
| domain fields | `0,0.03,0.06,0.1,0.2,0.4,0.8,1.2` a.u. |
| source-response direction | `Bdot=(0,0,0.002)` and `E_origin=(0.004,0,0)` a.u. |
| dynamic final time | `T=2.0` a.u. |
| dynamic intervals | `N=32` for every level; additionally `N=16` for both C1 levels |
| nonlinear tolerance | `1e-11`, at most 80 iterations |
| arithmetic | CPU float64/complex128 with an eight-thread ceiling |

The dynamic envelope is

```text
f(t) = sin^2(pi t/T),  0 < t < T,
f(t) = 0,              otherwise.
```

The three dynamic sources are:

1. `electric_field_free`: `B=0`, `E_origin=(0.005 f(t),0,0)`;
2. `electric_static_b`: `Bz=0.03` with the same electric pulse; and
3. `magnetic_induction`: `Bz=0.03+0.005 f(t)`, zero electric field at the
   gauge origin, and the Maxwell-required induction field generated by
   `Bdot(t)`.

Both Hartree and LDA branches are evaluated. The executed matrix is 72
stationary calculations, 96 domain-stress records, and 36 trajectories.

## 5. Authentication and completeness

The original campaign's numerical phases all finished. Its legacy synthesis
step then stopped before writing manifests because it required the immutable
NQ4 and NQ6 raw completion markers to say `complete`. Those markers correctly
remain `executed_unreviewed`; acceptance resides in the later review records.

The separate synthesis tool fixes only that evidence-policy error. It:

- verifies the NQ4 and NQ6 raw manifests and every recorded artifact hash;
- verifies their separate accepted review records and accepted-analysis
  hashes;
- verifies the NQ7 campaign identity and implementation source hashes;
- requires the exact expected run-ID matrix with no duplicates or extras;
- validates every complete NPZ archive and requires every failed record to
  retain its exception and traceback; and
- refuses to overwrite an existing synthesis.

The authenticated matrix is:

| phase | expected records | complete | visible failures | NPZ archives |
|---|---:|---:|---:|---:|
| stationary | 72 | 72 | 0 | 72 |
| domain stress | 96 | 48 | 48 | 48 |
| dynamics | 36 | 36 | 0 | 36 |

Every domain failure is the declared `FormulationError` for a nonpositive
reduced metric. Failed points have no misleading NPZ result. The analyzer
recomputes dynamic endpoint energy changes and Simpson source-work integrals
from the stored arrays.

Raw manifest hashes are:

```text
result.json       8e76e185a8ea65507e3d18fe36406a04fb56dc521db88b3c582297be09ebdab2
provenance.json   5f75702dde377966aa00cf71407a35a821628674b7e2314ae6bf9b9563c5cede
completed.json    5fb933a2fbff5034e828902d9c811337a773ff7bc84ecead16f5079494e918e6
```

The complete analysis was independently repeated, including the exact-metric
stress scan. Both passes produced the identical summary hash:

```text
3e1767e7cedb4caae95695e8fecb25d1b16c3a8b44723f80bf629d80c8b0b2b2
```

## 6. Stationary exact-versus-reduced comparison

![NQ7 stationary field errors](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq7_reduced_models_20260923T134902Z_e11a340c1e0d/analysis_final/stationary_field_errors.png)

Each plotted value is the maximum over the matched Hartree and LDA branches.
The P0 and E1 spatial actions are identical in this static comparison, so
their state, energy, and lower-matrix curves overlap. Strict and
density-resummed C1 also nearly overlap at these fields.

At the largest main-domain field, `Bz=0.06` a.u.:

| level | energy error (Ha) | mixed-density error | matched lower-matrix error | power error (Ha/a.u.) |
|---|---:|---:|---:|---:|
| P0 | `1.491e-3` | `1.711e-2` | `2.572e-2` | `1.033e-4` |
| E1 | `1.491e-3` | `1.711e-2` | `2.572e-2` | `6.432e-5` |
| strict C1 | `1.532e-3` | `1.275e-3` | `1.764e-3` | `1.020e-4` |
| density-resummed C1 | `1.532e-3` | `1.275e-3` | `1.764e-3` | `1.020e-4` |

The low-field fits over `B=0.001,0.003,0.01` give order `1.00002` for the P0
mixed-density error and `1.99996` for strict C1. The strict-C1 energy error
also has order `1.99999`; the absence of a P0/C1 improvement in its
coefficient is therefore not a failure of first-order consistency. The
leading energy remainder is even in field in this symmetric fixture.

The action-derived source and power comparison is also consistent with a
first-order action. Differentiating the omitted `O(B^2)` term with respect to
the magnetic source produces an `O(B)` response error. It is not expected to
inherit the second-order state-error curve.

## 7. Dynamic exact-versus-reduced comparison

![NQ7 dynamic model errors](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq7_reduced_models_20260923T134902Z_e11a340c1e0d/analysis_final/dynamic_model_errors.png)

The zero-B electric case is the decisive E1 reduction test. P0 omits the
internal electric connection and ends with about `4.2e-2` state error. E1 and
both C1 actions reproduce the exact action and trajectory at about
`1e-12` or below.

Finest-step final mixed-density relative errors are:

| source / branch | P0 | E1 | strict C1 | density-resummed C1 |
|---|---:|---:|---:|---:|
| uniform E, B=0 / Hartree | `4.239e-2` | `5.429e-13` | `5.429e-13` | `5.429e-13` |
| uniform E, B=0 / LDA KS | `4.191e-2` | `6.583e-13` | `6.583e-13` | `6.583e-13` |
| uniform E, static B / Hartree | `3.981e-2` | `8.855e-3` | `1.555e-3` | `1.556e-3` |
| uniform E, static B / LDA KS | `3.766e-2` | `8.439e-3` | `9.767e-4` | `9.767e-4` |
| time-dependent B / Hartree | `1.850e-2` | `1.968e-2` | `6.443e-4` | `6.445e-4` |
| time-dependent B / LDA KS | `1.800e-2` | `1.777e-2` | `4.652e-4` | `4.652e-4` |

C1 improves the magnetic final state but not necessarily the energy or power
trajectory norm. For example, in the induction Hartree case the relative
energy-trajectory errors are about `7.89e-4` for P0 and `8.32e-4` for C1.
This is retained as a negative result. A state-accurate first-order action is
not guaranteed to reduce the coefficient of every second-order scalar
remainder on one finite field range.

Each model nevertheless satisfies its own variational power balance. The
largest C1 mismatch between integrated source work and endpoint mechanical
energy is `4.8621e-11` Ha. The largest C1 nonlinear node residual is
`6.93e-15`, the largest cross-metric residual is `5.36e-10`, and no metric
correction or state projection is applied.

## 8. Separation from timestep uncertainty

![NQ7 model versus timestep error](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq7_reduced_models_20260923T134902Z_e11a340c1e0d/analysis_final/model_vs_timestep_error.png)

The grey bar is the final-state difference between the same C1 action at
`N=16` and `N=32`. The purple bar is the `N=32` C1 result relative to the
exact-Wilson `N=32` trajectory. Across all magnetic cases, branches, and both
C1 actions, the model difference is at least `6.79e3` times the timestep
difference. The physical model comparison is therefore not an unresolved
integrator artifact.

The zero-B E1 identity is intentionally not subjected to this ratio: there
the reduced and exact actions are identical, so their matched-`N=32`
trajectory difference cancels the common discretization error and lies below
the separate `N=16` to `N=32` refinement difference.

## 9. Basis-dependent domain failures

![NQ7 reduced metric domain](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq7_reduced_models_20260923T134902Z_e11a340c1e0d/analysis_final/domain_metric_boundaries.png)

The exact overlap eigenvalues were recomputed independently for all 24
basis/field points because the original loop did not retain them at fields
where every reduced level failed. They agree exactly in stored double
precision with every retained raw exact value. The exact Wilson metric stays positive at all stress
points; its smallest eigenvalue is `1.5713e-4` and its largest condition
number is `3.9308e4`, both in aug-cc-pVDZ.

The sampled reduced-domain boundaries are:

| basis | levels | largest passing B (a.u.) | first failing B (a.u.) |
|---|---|---:|---:|
| STO-3G | P0, E1, strict C1, resummed C1 | `0.8` | `1.2` |
| cc-pVDZ | P0, E1 | `0.06` | `0.1` |
| cc-pVDZ | strict C1, resummed C1 | `0.2` | `0.4` |
| aug-cc-pVDZ | all levels | `0.0` | `0.03` |

The diffuse basis is already ill conditioned at zero field, with exact
condition number about `3.10e4`; deleting or truncating internal Wilson
structure then makes the reduced metric indefinite at the first nonzero
sample. C1 substantially extends the sampled cc-pVDZ domain relative to
P0/E1 but does not extend the aug-cc-pVDZ domain.

No LDA-domain evaluation fails where the reduced metric is valid. The only
negative retained pointwise density minimum is `-6.71e-18`, at the numerical
quadrature floor. The observed domain boundary is therefore metric
positivity, not material LDA density failure.

## 10. Strict versus density-resummed C1

The two actions coincide at zero magnetic field and first order by
construction. Their nonzero differences are higher order. In the dynamic
magnetic cases, the final-state strict/resummed relative differences range
from `7.63e-7` to `7.72e-6`, above the timestep differences but far below the
common exact-model errors.

Across final density, energy trajectory, power trajectory, and integrated
power balance for the four magnetic case/branch pairs, strict C1 is smaller
in eight comparisons and density-resummed C1 is smaller in eight. They have
identical sampled domain boundaries. There is therefore no numerical basis
to claim the resummed model is safer.

Strict C1 is proposed as the working approximation because:

- it is the action with a controlled degree-one truncation;
- it produces the expected second-order state and lower-matrix remainder;
- it satisfies its own source-power identity and nonlinear EOM;
- it shares the same tested domain as the resummed action; and
- the retained higher closure powers provide no systematic benefit here.

Density-resummed C1 remains a reusable, separately labeled diagnostic action.
This recommendation may be revisited if a later basis/geometry campaign finds
a reproducible resummation advantage.

## 11. Proposed gate criteria and results

All proposed criteria pass, but acceptance remains a user decision.

| criterion | result | proposed bound |
|---|---:|---:|
| zero-field energy error | `1.94e-13 Ha` | `<=1e-10 Ha` |
| zero-field mixed-density error | `3.49e-13` | `<=1e-10` |
| E1 zero-B final-state error | `6.59e-13` | `<=2e-10` |
| E1 zero-B energy-trajectory error | `3.74e-13` | `<=2e-10` |
| E1 zero-B power-trajectory error | `1.01e-12` | `<=2e-10` |
| strict-C1 low-field state-error order | `1.99996` | `>=1.8` |
| minimum stationary C1/P0 state improvement | `13.43` | `>=10` |
| minimum dynamic C1/best(P0,E1) state improvement | `5.69` | `>=5` |
| maximum C1 stationary energy error | `1.532e-3 Ha` | `<=1.6e-3 Ha` |
| maximum C1 stationary mixed-density error | `1.275e-3` | `<=1.4e-3` |
| maximum C1 matched lower-matrix error | `1.764e-3` | `<=1.9e-3` |
| maximum magnetic C1 final-state error | `1.556e-3` | `<=1.7e-3` |
| minimum model/timestep separation | `6.79e3` | `>=100` |
| maximum C1 power-balance residual | `4.862e-11 Ha` | `<=5e-11 Ha` |
| maximum C1 nonlinear residual | `6.93e-15` | `<=1e-10` |
| exact stress-domain minimum overlap eigenvalue | `1.571e-4` | `>=1e-6` |
| main cc-pVDZ domain failures through B=0.06 | `0` | `0` |
| hidden/clipped failures | `0` | `0` |
| LDA failures where the reduced metric is valid | `0` | `0` |

## 12. Evidence index and reproduction

Campaign root:

```text
/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq7_reduced_models_20260923T134902Z_e11a340c1e0d
```

Important raw files:

- `campaign_identity.json`: immutable implementation commit and source hashes;
- `checkpoints/stationary/`: 72 JSON/NPZ pairs;
- `checkpoints/domain/`: 96 JSON records and 48 valid NPZ archives;
- `checkpoints/dynamics/`: 36 JSON/NPZ pairs;
- `result.json`, `provenance.json`, `completed.json`: authenticated synthesis;
- `continue_campaign.sh`: original restart command.

Analysis entry points:

- `analysis_final/summary.json`: proposed checks, metrics, recommendation, and
  hashes for every table and figure;
- `analysis_final/stationary_records.csv` and
  `stationary_field_errors.csv`;
- `analysis_final/dynamic_finest_records.csv` and
  `dynamic_secondary_comparisons.csv`;
- `analysis_final/domain_records.csv`, `domain_boundaries.csv`, and
  `exact_domain_metrics.csv`;
- `analysis_verification/`: complete independent repeated analysis with the
  same summary hash.

Raw verification:

```text
/home/cgs/01_TOOLS/EasyBuild/conda/bin/conda run \
  -p /home/cgs/01_TOOLS/EasyBuild/conda/envs/aion \
  python tools/synthesize_chapter13_nq7_reduced_models.py \
  --output /home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq7_reduced_models_20260923T134902Z_e11a340c1e0d \
  --verify
```

Analysis reproduction:

```text
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
NUMEXPR_NUM_THREADS=8 MPLCONFIGDIR=/tmp/aion-nq7-mpl \
/home/cgs/01_TOOLS/EasyBuild/conda/bin/conda run \
  -p /home/cgs/01_TOOLS/EasyBuild/conda/envs/aion \
  python tools/analyze_chapter13_nq7_reduced_models.py \
  --raw /home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq7_reduced_models_20260923T134902Z_e11a340c1e0d \
  --output NEW_EMPTY_ANALYSIS_DIRECTORY
```

## 13. Limitations and review boundary

- The physical comparison is one equilateral H3+ geometry, one main AO and
  auxiliary basis pair, one pure LDA functional, one gauge representative,
  one pulse amplitude, and `T=2.0` a.u.
- The basis stress scan diagnoses metric and LDA domains at a fixed reference
  coefficient density; it is not a converged stationary or dynamic campaign
  in every stress basis and field.
- The field grid brackets sampled failure intervals; it does not locate a
  continuous critical field.
- The small negative STO-3G density minimum is at the quadrature/floating-point
  floor and is not interpreted as physical loss of nonnegativity.
- The pure-LDA Kohn--Sham source current is the current of the declared
  auxiliary action, not an exact interacting transverse current.
- Spatially nonuniform fields, moving nuclei, periodic systems, GGA,
  pseudopotentials, Maxwell backreaction, longer trajectories, and chemical
  transfer are not qualified here.
- NQ8 must not begin until this evidence is reviewed and NQ7 is explicitly
  accepted.

The requested review decision is whether to accept the bounded NQ7 claims,
the strict-C1 working-action recommendation, and the explicitly retained
basis-dependent negative result.
