# Chapter 13 NQ6 nonlinear dynamics, energy, and power qualification

Date: 2026-09-23

Gate state: **executed and analyzed; awaiting user review**

Accepted prerequisite: `docs/reviews/chapter13_nq5_review_20260922.json`

Implementation commit: `a197fbc55b62f709830e06faa163b8df252e38c7`

Controlling plan:
`/home/cgs/00_WORK/Projection_Full_Formalism/REVIEW/implementation/aion_chapter_13_numerical_qualification_plan.md`

This document is the review entry point for NQ6. It separates the Chapter 13
mathematical identities, the implemented nonlinear propagator and observable
APIs, the completed numerical tests, and the user's still-pending scientific
decision. Nothing in this document records acceptance or authorizes NQ7.

## 1. Mathematical and physical contract

The finite Wilson frame has an exact time-dependent overlap `S(t)`, temporal
connection `omega(t)`, and complete mechanical lower matrix `K[P,t]`. The
coefficient equation is

```text
S Cdot = -(omega + i K[P,t]/hbar) C,
G[P,t] = S^-1(-omega - i K[P,t]/hbar),
Cdot = G[P,t] C.
```

The connection is not reconstructed from the metric. Both are evaluated from
the prescribed electromagnetic source, and metric compatibility is checked
independently:

```text
Sdot = omega + omega^dagger.
```

The propagated action variable is the contravariant coefficient density

```text
P = C f C^dagger,
```

while the mixed tensor

```text
D = P S
```

is reconstructed to expose particle number, occupation spectrum, and metric
geometry. `P` and `D` are not interchangeable in Wilson contractions.

The two nonlinear actions are the already accepted RI--Wilson Hartree and
restricted pure-LDA Kohn--Sham branches. Their complete mechanical energies
are

```text
U_H  = Tr(P K_1e) + E_H[n_W] + E_NN,
U_KS = Tr(P K_1e) + E_H[n_W] + E_xc[n_W] + E_NN.
```

The scalar-potential coupling remains in the temporal connection and is not
counted again in `U`. Hartree, exchange--correlation, lower matrices, source
derivatives, current, and power descend from the same discrete action.

On the nonlinear coefficient shell, Chapter 13 gives two independent forms
of the instantaneous mechanical power. The matrix route is

```text
Gamma = S^-1 omega,
dU/dt = Tr[P (Kdot_1e - Gamma^dagger L - L Gamma)]
        + (partial_t E_H)_P + (partial_t E_xc)_P,
```

where `L` is the complete nonlinear lower mechanical matrix. The source route
pairs the complete action-derived on-shell current with the physical
uniform-plus-induction electric field:

```text
dU/dt = <j_action, E>.
```

NQ6 checks this instantaneous identity and its integral,

```text
U(T)-U(0) = integral_0^T <j_action(t), E(t)> dt,
```

including Hartree and LDA functional energies. For the Kohn--Sham branch,
`j_action` is the source current of the declared auxiliary adiabatic-LDA
action. It is not claimed to be the exact interacting transverse current.

## 2. Time integration algorithm

The reusable implementation is in
`src/aion/propagation/tensorial.py`. NQ6 uses
`propagate_nonlinear_contravariant_density`, not the earlier experimental
mixed-similarity endpoint update.

For a step of length `h`, the two fourth-order Gauss nodes are

```text
c_minus = 1/2 - sqrt(3)/6,
c_plus  = 1/2 + sqrt(3)/6.
```

At each nonlinear iteration, the exact source and complete nonlinear action
are evaluated at both node times and trial node densities. The two node
generators define partial Magnus links from the left endpoint. The node
densities are updated by congruence,

```text
P(c) = U(c) P_n U(c)^dagger,
```

until both relative node changes satisfy the declared nonlinear tolerance. A
second consistency evaluation is required before the step is accepted. A
failed node solve raises `PropagationError`; it is not silently accepted.

The accepted fourth-order Magnus exponent is

```text
Omega = h/2 (G_minus + G_plus)
        - sqrt(3) h^2/12 [G_minus, G_plus],
```

and `U` is its diagonal `[2/2]` Padé map. The endpoint update is

```text
P_(n+1) = U P_n U^dagger.
```

This makes Hermiticity and positivity of the nonlinear action argument
structural. The exact endpoint metric is then used to reconstruct
`D_(n+1)=P_(n+1) S_(n+1)`. No Löwdin orthogonalization, Cholesky frame,
endpoint metric correction, post-step projection, clipping, or density repair
is applied. The finite-step failure of `U^dagger S_(n+1) U=S_n` remains
visible as the cross-metric residual and therefore as small trace and
occupation defects. The qualification asks whether these defects converge,
not whether a repair can conceal them.

The nonlinear policy is reusable through `NonlinearGaussMagnusPolicy`. The
propagator returns boundary and node contravariant and mixed densities, raw
links, nonlinear iteration counts, nonlinear residuals, cross-metric
residuals, trace drift, and occupation-spectrum drift.

## 3. Dynamic Wilson action and observables

`src/aion/electronic_structure/wilson_dynamics.py` provides the dynamic
composition layer:

| API | role |
|---|---|
| `PreparedExactWilsonDynamicSpatialAction` | reuses source-fixed spatial Wilson, RI, and LDA objects |
| `ExactWilsonDynamicSample` | one exact source sample with `S`, `omega`, mechanical matrices, and nonlinear action |
| `ExactWilsonDynamicEvaluation` | complete action evaluation and EOM triple at one density |
| `evaluate_exact_wilson_power` | independent matrix-rate and action-source power routes |

`src/aion/electronic_structure/time_connection.py` evaluates the exact Wilson
time connection, metric rate, and magnetic-field source derivative. The
dynamic frame-overlap rate uses

```text
Gamma_alpha_dot = S^-1(F_alpha_dot - Sdot Gamma_alpha),
```

rather than treating `Gamma_alpha=S^-1 F_alpha` as time independent. This
term is required in the tangential current variation and was the decisive
connection-aware correction found during NQ6 development.

The trajectory diagnostics reuse the accepted NQ5 APIs in
`src/aion/electronic_structure/wilson_sources.py` for exact Wilson charge,
density-variable Ward identities, and weak finite-region continuity. No
ambient momentum, graph, P0, E1, or bare-gauge observable is substituted.

## 4. Numerical realization

The campaign uses the accepted NQ4 stationary H3+ states and NQ5 source
definitions.

| axis | value |
|---|---|
| system | equilateral H3+, molecular charge +1, two electrons |
| AO basis | cc-pVDZ |
| auxiliary basis | weigend |
| nonlinear branches | RI--Wilson Hartree; RI-Hartree + pure `lda,vwn` |
| real-space grid | unpruned PySCF level 4 |
| gauge | symmetric affine gauge, origin `(0.17,-0.31,0.23)` bohr |
| static field | `Bz=0.03` a.u. where present |
| final time | `T=2.0` a.u. |
| precision | CPU float64/complex128 |
| time steps | `0.5, 0.25, 0.125, 0.0625` a.u. (`4,8,16,32` intervals) |
| timestep-study tolerance | `1e-12` |
| tolerance study | `1e-6,1e-8,1e-10,1e-12` at `dt=0.125` a.u. |

The smooth envelope is

```text
f(t) = sin^2(pi t/T),  0 < t < T,
f(t) = 0,              otherwise.
```

The four physical cases are:

1. `stationary_static_b`: `Bz=0.03`, no electric source;
2. `electric_field_free`: `B=0`, `E_origin=(0.005 f(t),0,0)`;
3. `electric_static_b`: `Bz=0.03` with the same electric pulse; and
4. `magnetic_induction`: `Bz=0.03+0.005 f(t)`, zero electric field at the
   gauge origin, and the Maxwell-required induction field
   `E(r,t)=0.5 (r-origin) x Bdot(t)`.

Both Hartree and LDA branches are run for all cases. The complete matrix is
therefore `4 cases x 2 branches x (4 timestep + 3 additional tolerance)` =
56 trajectories. The finest trajectory in each case and branch evaluates
Ward and continuity diagnostics at nine points. Composite Simpson quadrature
integrates source power and the independent matrix rate.

## 5. Authentication and completeness

The analyzer independently authenticated the completion and provenance
manifests, every artifact listed by the provenance record, and the accepted
NQ4/NQ5 input hashes. It found exactly 56 unique expected runs, 920 trajectory
rows, and 280 expected NPZ arrays, with no missing or extra run. It also
recomputed every endpoint energy change and both Simpson integrals directly
from `trajectories.npz`; the largest difference from the stored summaries is
`1.80e-20` Ha.

The completed campaign contains no `failure.json`. No metric correction was
applied in any trajectory, and every nonlinear step converged within its
declared tolerance.

## 6. Independent timestep convergence

![NQ6 timestep convergence](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq6_dynamics_20260922T205959Z_a197fbc55b62/analysis_final/timestep_convergence.png)

The convergence observable is the relative Frobenius norm of the final mixed
density difference between successive refinements. It is a state comparison,
not a comparison of one method against another. The observed orders are:

| case | Hartree orders | LDA KS orders |
|---|---:|---:|
| electric, `B=0` | `3.791, 3.952` | `3.848, 3.965` |
| electric, static `B` | `3.791, 3.952` | `3.848, 3.965` |
| time-dependent `B` | `3.850, 3.961` | `3.907, 3.978` |

The minimum observed order is `3.7909`, consistent with the designed
fourth-order integrator. At the final refinement, the largest `N=16` to
`N=32` state difference is `6.9659e-8`. The stationary trajectories are
already between `3e-14` and `8e-14` in state difference and are excluded from
an order fit because that would fit floating-point and nonlinear-solver noise.

The geometric behavior is physically informative. Electric driving at fixed
`B` has cross-metric residuals at roundoff because the spatial Wilson frame
and metric do not move. Under time-dependent `B`, the maximum cross-metric
residual decreases from about `1e-5` at `dt=0.5` to `5.3243e-10` at
`dt=0.0625`; the associated occupation drift decreases from about `6e-8` to
`1.3297e-11`. These are uncorrected finite-step defects and visibly converge.

## 7. Independent nonlinear-tolerance refinement

![NQ6 nonlinear tolerance refinement](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq6_dynamics_20260922T205959Z_a197fbc55b62/analysis_final/nonlinear_tolerance_convergence.png)

At fixed `dt=0.125`, the final density from every looser solve differs from
the `1e-12` reference by at most `7.4622e-14`. The maximum energy-change
sensitivity is `4.9427e-13` Ha. The attained node residuals tighten with the
requested tolerance, while the state differences remain at the numerical
floor. This independently shows that timestep truncation, not incomplete
nonlinear convergence, controls the trajectory differences.

## 8. Energy, source power, and integrated work

![NQ6 induction energy and work](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq6_dynamics_20260922T205959Z_a197fbc55b62/analysis_final/induction_energy_work.png)

The figure shows the most geometrically demanding case. The complete
mechanical-energy change and the independently integrated action-derived
source power lie on top of one another. The inset is the instantaneous
difference between the matrix mechanical-energy rate and source-current
power, not a finite-difference derivative of the plotted energy.

At the finest timestep:

| case / branch | `Delta U` (Ha) | integrated source work (Ha) | endpoint residual (Ha) | instantaneous identity max (Ha/a.u.) |
|---|---:|---:|---:|---:|
| stationary / Hartree | `-1.3323e-14` | `0` | `1.3323e-14` | `0` |
| stationary / LDA KS | `2.9754e-14` | `0` | `2.9754e-14` | `0` |
| electric, `B=0` / Hartree | `2.2034104e-5` | `2.2034062e-5` | `4.2054e-11` | `3.2546e-13` |
| electric, `B=0` / LDA KS | `2.1879557e-5` | `2.1879508e-5` | `4.8711e-11` | `3.1106e-14` |
| electric, static `B` / Hartree | `2.2033778e-5` | `2.2033736e-5` | `4.2346e-11` | `2.5354e-13` |
| electric, static `B` / LDA KS | `2.1878369e-5` | `2.1878320e-5` | `4.8366e-11` | `3.1428e-14` |
| induction / Hartree | `1.0930300e-6` | `1.0930417e-6` | `1.1737e-11` | `7.0352e-14` |
| induction / LDA KS | `1.2389324e-6` | `1.2389470e-6` | `1.4603e-11` | `1.5257e-13` |

The largest finest-step work/endpoint mismatch is `4.8711e-11` Ha. The
stationary energy drift is at most `2.9754e-14` Ha. The integrated residuals
decrease systematically with timestep; they are not made small by subtracting
an empirical offset.

## 9. Trajectory identities and invariants

![NQ6 finest trajectory identities](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq6_dynamics_20260922T205959Z_a197fbc55b62/analysis_final/finest_trajectory_identities.png)

The heatmap reports `log10` absolute residuals, so more negative values are
smaller. The largest finest-trajectory values are:

| diagnostic | maximum |
|---|---:|
| cross-metric link residual | `5.3243e-10` |
| metric orthonormality residual | `1.3298e-11` |
| occupation-spectrum drift | `1.3297e-11` |
| mixed-density occupation-polynomial residual | `2.6597e-11` |
| metric particle-number residual | `2.6597e-11` |
| per-step trace drift | `2.4825e-12` |
| contravariant Hermiticity residual | `4.3517e-15` |
| density reconstruction residual | `2.9343e-15` |
| Ward residual | `6.5052e-19` |
| weak finite-region continuity residual | `2.6250e-12` |
| global charge residual | `2.0723e-16` |
| instantaneous power-identity residual | `3.2546e-13` Ha/a.u. |

The real-space grid particle count differs from the stable metric count by at
most `9.4969e-9`, consistent with the independently qualified level-4
spatial-quadrature floor from NQ5. It is kept separate from the much smaller
metric particle-number residual.

The preliminary analysis proposed `2e-12` and `2e-13` limits for weak
continuity and instantaneous power. The measured values were `2.6250e-12`
and `3.2546e-13`, so that analysis correctly reported failure and is retained
as `analysis_preliminary/`. Inspection showed that the values are independent
action-contraction residuals, remain far below the accepted NQ5 spatial
source uncertainty, and accompany fourth-order state and integrated-work
convergence. The final review proposal uses explicit `3e-12` and `5e-13`
bounds. This is a disclosed review criterion change, not a rerun or alteration
of raw data; the user may reject it.

## 10. Interrupted attempt and completed rerun

The first qualification launch was interrupted by a host power failure after
33 of 56 trajectories. The runner writes the synthesized numerical artifacts
only after the full matrix finishes, so that attempt had no recoverable
scientific checkpoint. It remains visible at

```text
/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq6_dynamics_20260922T145942Z_a197fbc55b62
```

Its `interrupted.json` SHA-256 is
`219f49f49cd78ac58597cb446be8107c245f0702bb00f5f75ca30b3c18c148a7`;
the authenticated external log SHA-256 is
`a92a3631e409a8705846c79ca45665f81fee86451b926807b889644f4f21062b`.
It is not used as qualification evidence.

The complete matrix was rerun from the beginning in a new immutable campaign
directory. There was no numerical resume, splicing, or selective replacement.

## 11. Tests and execution record

Implementation validation performed before the qualification campaign:

```text
ruff check src tests tools
All checks passed

mypy src/aion
Success: no issues found in 86 source files

pytest -q
95 passed

pytest -q -m integration
151 passed

tools/gpu-python -m pytest -q -m gpu
4 passed on the physical GPU through the declared launcher
```

The physical-GPU tests cover experimental mixed propagation, nonlinear mixed
and congruence propagation, and the exact dynamic Wilson temporal/source-
response path. The complete nonlinear power composition is covered by CPU
integration tests; the 56-trajectory scientific campaign is a CPU
qualification run under the declared eight-thread ceiling and is not a GPU
performance comparison.

The final analyzer authenticates the raw campaign, recomputes consequential
numbers, checks the complete run matrix, writes auditable CSV reductions, and
generates the four figures. All proposed final checks pass. The analyzer is
`tools/analyze_chapter13_nq6_dynamics.py`.

## 12. Evidence index and reproduction

Authenticated completed root:

```text
/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq6_dynamics_20260922T205959Z_a197fbc55b62
```

Authentication hashes:

| record | SHA-256 |
|---|---|
| `result.json` | `951b7a88e3c5311f3a0da5f4448a59e44458b2c8d95633cab4216cc224a08afb` |
| `provenance.json` | `ae526ed53eb9e11d8c45d5b0ab4d10ff49ddbe1ce6f4357ded344ac3d7a9df19` |
| `completed.json` | `b141be162d2fbee6bfe7c435c5d19be2074840a5a69f713f35c0816f2a3810f4` |
| `analysis_final/summary.json` | `d6da6702165ada1e38f151327d786a4fc5ddc3170ac485d1ae9b7d171b97daac` |

Raw and reduced contents:

| artifact | content |
|---|---|
| `result.json` | realization labels, accepted-input hashes, and all trajectory summaries |
| `provenance.json`, `completed.json` | source, environment, repository, artifact, and completion authentication |
| `summary.csv` | one record for each of 56 trajectories |
| `trajectory.csv` | 920 time-sample diagnostic records |
| `trajectories.npz` | time, final mixed density, energy, source power, and matrix rate arrays |
| `analysis_final/summary.json` | independent checks, metrics, limits, hashes, and interpretation boundaries |
| `analysis_final/timestep_convergence.csv` | state, work, and geometric timestep reductions |
| `analysis_final/nonlinear_tolerance_convergence.csv` | independent nonlinear-tolerance reductions |
| `analysis_final/finest_trajectory_diagnostics.csv` | invariant and identity maxima by case and branch |
| `analysis_preliminary/` | retained first analysis with the disclosed overly tight provisional limits |

Reproduction commands, run in the managed Aion environment with the CPU
thread variables capped at eight:

```text
python tools/run_chapter13_nq6_dynamics.py --output <raw-root> --profile qualification

python tools/analyze_chapter13_nq6_dynamics.py \
  --raw <raw-root> \
  --output <raw-root>/analysis_final \
  --interrupted-root <interrupted-root>
```

The raw provenance records the exact interpreter, dependency versions, Git
head, repository status, tracked-diff hash, source hashes, command, and raw
artifact hashes.

## 13. Status ledger and bounded conclusion

| claim | state after this work |
|---|---|
| Chapter 13 nonlinear density EOM and power identities | specified in the mathematical source |
| reusable exact Wilson dynamic action and power API | implemented and tested |
| connection-aware fourth-order nonlinear congruence propagator | implemented and tested |
| all four required Hartree and LDA physical cases | executed and authenticated |
| timestep and nonlinear-tolerance refinements | executed and independently analyzed |
| metric, occupation, particle-number, and density diagnostics | executed and independently analyzed |
| trajectory Ward, weak continuity, and global charge checks | executed and independently analyzed |
| instantaneous mechanical-rate/source-power identity | executed and independently analyzed |
| complete energy-change/integrated-work equality | executed and independently analyzed |
| NQ6 scientific acceptance | **pending user review** |

The evidence supports the bounded conclusion that the uncorrected
connection-aware fourth-order congruence algorithm converges for the tested
H3+ exact-Wilson Hartree and pure-LDA actions, including electric driving in
zero and static magnetic fields and Maxwell-consistent magnetic induction.
It does not establish basis convergence, stronger-field stability,
long-time behavior, nonuniform magnetic sources, GGA, moving nuclei,
periodicity, interacting transverse currents, or reduced P0/E1/C1 accuracy.
Those claims remain outside NQ6.

If the user accepts this evidence and the disclosed final review criteria,
the next gate is NQ7: matched exact-Wilson comparison against P0, E1, strict
C1, and density-resummed C1 action descendants. Until that decision is
recorded separately, NQ7 is not authorized.
