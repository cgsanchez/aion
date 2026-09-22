# Chapter 13 NQ5 source derivatives and static observables

Date: 2026-09-22

Gate state: **analyzed, awaiting user review**

Accepted prerequisite: `docs/reviews/chapter13_nq4_review_20260921.json`

Controlling plan:
`/home/cgs/00_WORK/Projection_Full_Formalism/REVIEW/implementation/aion_chapter_13_numerical_qualification_plan.md`

This document is the review entry point for NQ5.  It describes the reusable
source-observable implementation, its relation to the Chapter 13 action, the
authenticated H3+ campaign, the numerical error separations, and the limits of
the evidence.  It records an implemented and executed gate; it does not record
acceptance, which remains a user decision.

## 1. Mathematical contract

The two nonlinear branches use the exact fixed-centre Wilson one-electron
action and one closure energy:

```text
Hartree:  S_H  = S_1e - E_H[n_W]
LDA KS:   S_KS = S_1e - E_H[n_W] - E_xc[n_W].
```

The coefficient history and occupations are held fixed during an
electromagnetic source derivative.  The Wilson frame, metric, connection,
one-electron mechanical matrix, and Wilson density are allowed to respond.
Consequently, the closure source term is not zero even though the Hartree
kernel and LDA functional have no explicit electromagnetic dependence: their
restriction to the moving Wilson subspace depends on the source.

The implemented charge is

```text
rho_beta,W(r,t) = q n_W(r,t).
```

The closure contribution to a weak vector-current test `alpha` is

```text
<j_cl,beta,W, alpha> = - integral v_cl,beta[n_W] delta_alpha^C n_W,
```

and the complete fixed-history source pairing is the sum of the one-electron
and closure pairings.  These are
`eq:wilson-hartree-ks-charge-density`,
`eq:wilson-hartree-ks-closure-current-pairing`, and
`eq:wilson-hartree-ks-current-sum` in Chapter 13.

### 1.1 Off-shell and on-shell current

For an arbitrary trial coefficient history, the complete vector-source
pairing has the residual form

```text
<j_beta,W, alpha>
  = <j_min,ind,W, alpha>
    + 2 Re sum_n f_n <delta_alpha chi_A c_n, R_beta,n>.
```

The frame response has a Gauss decomposition into a tangential coefficient
variation and a normal change of the represented subspace.  The instantaneous
fixed-coordinate pairing therefore separates as

```text
J_total = J_tangent + J_minimal + J_normal.
```

On the coefficient shell, the tangential virtual work vanishes and the
physical action-source observable is

```text
J_on-shell = J_minimal + J_normal.
```

This is the implementation of
`eq:wilson-hartree-ks-off-shell-current-residual-form`,
`eq:wilson-hartree-ks-nonlinear-subspace-current`, and
`eq:wilson-hartree-ks-on-shell-current`.  It is important not to identify the
large fixed-coordinate off-shell total with the on-shell current.  A local
matrix integrand can contain a nonzero tangential/total-derivative term even
though its coefficient virtual work vanishes for an allowed compact temporal
variation.

### 1.2 Ward and continuity identities

For the pure-gauge source direction

```text
(varphi, alpha) = (-partial_t lambda, grad lambda),
```

the implementation evaluates the source variation and its simultaneous
coefficient variation independently.  Their sum is the complete off-shell
Ward residual.  The campaign deliberately uses histories with a nonzero
lower coefficient residual, so cancellation is not an on-shell zero test.

On shell, three smooth spatial weights test the fixed-time weak continuity
law

```text
d/dt integral f(r) q n_W(r,t) dr = <j_beta,W(t), grad f>,
```

and the constant-weight limit tests global charge conservation.  NQ5 tests
the continuous-time instantaneous identities.  Time-discrete conservation,
energy, and power belong to NQ6.

For the Kohn--Sham branch, every current reported here is the source current
of the declared auxiliary adiabatic-KS action.  It transports the KS density,
but is not claimed to be the exact interacting transverse current.  The
complete subspace response is reported as a weak source pairing; no smoothed
pointwise complete current is claimed.

## 2. Reusable implementation

### 2.1 Source tests and exact one-electron response

`src/aion/electromagnetism/test_variations.py` adds two reusable source
objects:

- `GaussianScalarGaugeVariation` supplies a smooth scalar test, its analytic
  gradient, and the exact straight-segment endpoint difference needed by a
  pure-gauge Wilson-phase response;
- `PerturbedVectorPotential` represents `A_epsilon=A_0+epsilon alpha` for
  complete-action source finite differences.

`src/aion/electronic_structure/time_connection.py` now exposes
`ExactStaticWilsonGridOneElectronAction` and
`evaluate_exact_static_wilson_grid_one_electron_action`.  They rebuild the
overlap and mechanical one-electron action on the same real-space grid for an
arbitrary static, possibly nonuniform vector potential.  The analytic weak
direction record separately retains frame-overlap, explicit kinetic, and
embedding kinetic responses.  Existing uniform-field optimized evaluators
remain in use for stable stationary states.

`src/aion/formulations/exact_one_electron.py` accepts the general source-test
protocol rather than restricting the weak derivative to one concrete
Gaussian-vector class.

### 2.2 Nonlinear composition layer

The new module `src/aion/electronic_structure/wilson_sources.py` composes the
already qualified one-electron, RI--Wilson Hartree, Wilson-LDA, and action
history derivatives.  Its public records and functions are:

| API | Role |
|---|---|
| `ExactWilsonChargeObservation` / `evaluate_exact_wilson_charge` | exact signed Wilson density, grid integral, and stable metric total |
| `StaticNonlinearWilsonGridAction` / `evaluate_static_nonlinear_wilson_grid_action` | common-grid complete action used as the independent finite-difference parent |
| `NonlinearWeakCurrentPairing` / `evaluate_nonlinear_weak_current_pairing` | one-electron, closure, embedding, tangent, normal, and on-shell weak pairings |
| `NonlinearPureGaugeWardResult` / `evaluate_nonlinear_pure_gauge_ward` | independent source/matter pure-gauge directions and lower-residual diagnostic |
| `NonlinearWeakContinuityResult` / `evaluate_nonlinear_weak_continuity` | instantaneous coefficient-shell weak and global charge balances |

The closure pairing is evaluated from the source derivatives of the same
`RIWilsonHartreeEvaluator` and `WilsonLDAEvaluator` used to build the matter
matrix.  The coefficient-shell velocity is generated from the complete
nonlinear lower matrix.  No ambient momentum current, graph current, P0
current, or reduced-model observable is borrowed.

`src/aion/electronic_structure/__init__.py` and
`src/aion/electromagnetism/__init__.py` export these reusable interfaces.

### 2.3 Backend contract

All new array-valued evaluations use the selected Aion backend and explicitly
assert device residency at existing boundaries.  The GPU test evaluates
charge and every resolved weak-current component on both CPU and physical GPU,
checks GPU residence, and compares the values without a CPU fallback.  PySCF
continues to own the molecule, Gaussian AO basis, nuclei, real-space grid, AO
values, bare integral data, auxiliary basis, and libxc LDA datum.  Aion owns
the anchors, Wilson phases, complex pair contractions, action source
directions, and observable decomposition.

## 3. Numerical realization

The authenticated NQ4 H3+ result supplies the molecule, reference data, and
accepted finite-field stationary orbitals.  The physical system and numerical
choices are:

| axis | value |
|---|---|
| molecule | equilateral H3+, molecular charge +1, two electrons |
| AO basis | cc-pVDZ |
| magnetic source | static uniform `Bz=0.03` a.u. |
| gauge | symmetric affine gauge, origin `(0.17,-0.31,0.23)` bohr |
| nonlinear branches | RI--Wilson Hartree and RI-Hartree + pure LDA `lda,vwn` |
| auxiliary basis | `weigend` |
| grids | unpruned PySCF levels 3, 4, and 5: 45,304, 78,120, and 123,904 points |
| vector source test | Gaussian, amplitude `(0.19,-0.13,0.07)` a.u., exponent `0.41` bohr^-2 |
| source FD steps | `3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5` |
| path orders | 12, 24, 40 |
| RI rank thresholds | `1e-6, 1e-7, 1e-9` |
| precision | CPU float64/complex128; physical-GPU float64/complex128 parity test |

Three state/history classes remain distinct:

1. accepted NQ4 stationary densities are used for charge and the stationary
   coefficient-shell decomposition;
2. deliberately arbitrary coefficient velocities provide the off-shell
   source derivative and Ward test;
3. independently normalized complex densities are placed on the
   instantaneous coefficient shell to test nontrivial weak continuity.

The source finite difference holds each complete coefficient history fixed.
It does not differentiate a re-solved stationary state.

## 4. Charge and grid convergence

The stable metric totals reproduce two electrons.  The maximum particle
number residual over all grids and branches is `5.8098e-12`; the largest
imaginary density component is `7.0473e-19`.

| grid | Hartree `|Q_grid-Q_metric|` | LDA KS `|Q_grid-Q_metric|` |
|---:|---:|---:|
| 3 | `2.1017e-8` | `8.7677e-8` |
| 4 | `7.3563e-9` | `9.4868e-9` |
| 5 | `1.0736e-9` | `1.0926e-9` |

The real-space integral converges monotonically to the stable metric charge.
The level-5 residual is treated as a spatial-quadrature floor, not as a
failure of the metric normalization.

## 5. Complete source derivative

![Complete source finite difference](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq5_sources_20260922T001500Z_23549ab7740a/analysis_final/source_finite_difference.png)

The three coarsest central differences give a minimum fitted order of
`1.99125`, consistent with the expected second-order truncation error.  The
curves then enter a roundoff/grid-action floor.  The best errors are
`7.0936e-11` for Hartree and `3.6414e-10` for LDA KS.  This differentiates the
complete common-grid action, including its history, metric, kinetic,
electron--nuclear, Hartree, and LDA contributions; it is not a comparison of
two analytic formulas sharing the same derivative code.

Independent algebraic residuals are:

| identity | maximum residual |
|---|---:|
| closure pairing equals negative closure-energy source direction | `8.4703e-22` |
| complete pairing equals one-electron plus closure pairing | `5.4142e-18` |
| off-shell Gauss decomposition | `5.8818e-18` |

The closure current is numerically resolved rather than absent: its smallest
level-4 magnitude is `1.1328e-5`.

## 6. Current decomposition

![Weak current Gauss decomposition](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq5_sources_20260922T001500Z_23549ab7740a/analysis_final/current_gauss_decomposition.png)

At grid level 4, the deliberately off-shell histories give:

| component | Hartree | LDA KS |
|---|---:|---:|
| fixed-coordinate total | `5.2506472e-2` | `-1.0179134e-1` |
| tangential | `5.2519263e-2` | `-1.0177879e-1` |
| ambient minimal | `-1.3450058e-5` | `-1.3489956e-5` |
| normal subspace | `6.5891257e-7` | `9.3266677e-7` |
| minimal + normal | `-1.2791146e-5` | `-1.2557289e-5` |

Thus the large off-shell source values are almost entirely tangential.  The
middle panel resolves the much smaller difference and shows that
`J_total-J_tangent` equals `J_minimal+J_normal`.

On the accepted stationary coefficient shell, the tangential pairings are
`-4.9548e-17` and `1.5271e-16`, while the normal responses remain
`6.5893e-7` and `9.3269e-7`.  The on-shell decomposition residual is exactly
zero at the stored precision.  This directly exercises a nonzero normal
subspace response rather than validating only the ambient minimal current.

## 7. Ward identity and continuity

![Ward, continuity, and grid refinement](/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq5_sources_20260922T001500Z_23549ab7740a/analysis_final/ward_continuity_and_refinement.png)

The deliberately off-shell Ward histories have relative lower-residual norms
`1.67447` (Hartree) and `1.51448` (LDA KS).  Their source/matter pairings are
`+0.3496103992/-0.3496103992` and
`+0.3218699727/-0.3218699727`.  The complete Ward residual is zero at stored
precision.  The nonzero source and matter terms demonstrate an off-shell
identity rather than an on-shell `0=0` check.

For three localized Gaussian spatial weights per branch, the largest weak
finite-region continuity residual is `8.8471e-16`.  The maximum global signed
charge-rate residual is `1.2559e-15`.  The tested weighted charge derivatives
are nonzero, from `4.5517e-3` to `7.8386e-2`, so the finite-region check is not
trivial global normalization.

The Ward equality is an exact first-variation identity of the common discrete
action in the declared pure-gauge test space.  The numbers do not by
themselves establish convergence to an arbitrary continuum current test.

## 8. Error-source separation

| source varied alone | comparison | maximum change |
|---|---|---:|
| real-space grid | level 4 to level 5 complete weak source | `5.3766e-10` |
| path quadrature | order 24 to 40 complete weak source | `4.1633e-17` |
| RI rank policy | threshold `1e-6` to `1e-9` complete weak source | `0.0` |

All three RI thresholds retain rank 33.  This campaign therefore establishes
rank stability for this auxiliary basis and fixture; it does not probe an
actual rank transition.  The preliminary `analysis/` directory and final
`analysis_final/` directory have identical summary and figure hashes.  The
former is retained to preserve the immutable analysis history; the latter was
regenerated with the final lint-clean analyzer and is the report target.

The proposed numerical checks all pass.  Their complete machine-readable
values and thresholds are in `analysis_final/summary.json`; its SHA-256 is
`8ec67b01611625c64b7445c88e886d632e07f66417172cffb9f56747e8fb0eb5`.

## 9. Tests and execution record

Focused development checks:

```text
pytest -q -m integration tests/test_exact_one_electron_wp7.py tests/test_wilson_sources.py
12 passed, 1 deselected in 13.33s

pytest -q -m gpu tests/test_wilson_sources_gpu.py tests/test_exact_one_electron_wp7_gpu.py
2 passed, 1 warning in 12.72s
```

Final repository validation:

```text
ruff check src tests tools
All checks passed

mypy src/aion
Success: no issues found in 85 source files

pytest -q
89 passed, 175 deselected in 3.28s

pytest -q -m integration
150 passed, 114 deselected in 57.85s
```

The first attempted full GPU command incorrectly reused the campaign's
eight-thread CPU environment and omitted the required launcher marker:

```text
conda run -p .../envs/aion pytest -q -m gpu
2 failed, 23 passed, 239 deselected, 1 warning
```

Both failures were explicit launch-contract assertions: missing
`AION_GPU_LAUNCHER=1` and host thread limits of eight rather than one.  No
numerical comparison failed.  The run is retained here as procedural negative
evidence.  Re-execution through the repository's declared launcher gave:

```text
tools/gpu-python -m pytest -q -m gpu
25 passed, 239 deselected, 1 warning in 57.75s
```

The warning states that CuPy is the GPU4PySCF tensor-contraction engine.  The
launcher enforces one host thread for GPU tests, exports the physical-GPU
marker, supplies the managed CUDA libraries, and does not permit a silent CPU
fallback.

## 10. Evidence index and reproduction

Authenticated raw root:

```text
/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/chapter13_wilson_adiabatic_qualification/nq5_sources_20260922T001500Z_23549ab7740a
```

Authentication hashes:

| record | SHA-256 |
|---|---|
| `result.json` | `3f9bec5dee959d0b8d9fd85826cbd56f349107649960eb3426503e22529f5b78` |
| `provenance.json` | `14010553a68741b87ce62ec8edbd3c082a0b4633b1324861c0d30bcb3abc341d` |
| `completed.json` | `f72cf23b42db3a56a637d9090edf78c31e012968ae2387d225257c8c68f03ebe` |

Raw contents:

| artifact | content |
|---|---|
| `result.json` | complete structured campaign record and realization labels |
| `provenance.json`, `completed.json` | environment, repository state, source/artifact hashes, completion authentication |
| `charge.csv` | exact Wilson charge and grid/metric totals |
| `weak_current.csv` | off-shell current components at grid levels 3--5 |
| `source_finite_difference.csv` | seven-step complete-action source derivatives |
| `on_shell_decomposition.csv` | accepted-stationary minimal, tangent, and normal decomposition |
| `ward.csv` | deliberately off-shell source/matter Ward pairings |
| `continuity.csv` | three finite-region weights per branch and global charge rates |
| `path_refinement.csv` | independent path-quadrature refinement |
| `ri_rank_refinement.csv` | independent auxiliary-rank-policy refinement |
| `observables.npz` | level-4 signed charge-density arrays |
| `analysis_final/summary.json` | authenticated metrics, proposed thresholds, checks, and figure hashes |

Execution and analysis commands:

```text
python tools/run_chapter13_nq5_sources.py --output <raw-root>
python tools/analyze_chapter13_nq5_sources.py --raw <raw-root> --output <raw-root>/analysis_final
```

Both commands were run in the managed Aion environment.  The campaign used
an eight-thread ceiling for CPU BLAS/OpenMP libraries.  The raw provenance
records the exact interpreter, dependency versions, Git head
`23549ab7740afdc5329cc7f653648fe78885b290`, dirty source list, tracked-diff
hash, source hashes, command, and artifact hashes.

## 11. Status ledger and limits

| claim | state after this work |
|---|---|
| Chapter 13 charge/current/Ward/continuity equations | specified in the mathematical source |
| reusable exact Wilson nonlinear source API | implemented and tested |
| complete fixed-history source finite difference | executed and analyzed |
| off-shell nonzero-residual Ward identity | executed and analyzed |
| stationary minimal plus normal current | executed and analyzed |
| weak finite-region and global continuity | executed and analyzed |
| physical-GPU parity and residency | executed and passed through the declared launcher |
| NQ5 scientific acceptance | awaiting user review |

The bounded evidence covers one static uniform magnetic field, one H3+
geometry, cc-pVDZ, one affine gauge representative, RI--Wilson Hartree, one
pure LDA functional, one physical vector test, one pure-gauge family, and
three continuity weights.  It does not qualify time propagation, energy or
power balance, GGA, basis transfer, nonuniform physical fields, moving
nuclei, current-density functionals, interacting transverse currents,
reduced P0/E1/C1 models, periodic systems, or Maxwell backreaction.

If accepted, the next bounded checkpoint is NQ6: nonlinear time evolution,
invariants, trajectory Ward/continuity diagnostics, and mechanical
energy/source-power balance.  No NQ6 implementation or campaign has been
started here.
