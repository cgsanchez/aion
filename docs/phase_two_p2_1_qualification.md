# Phase Two P2-1 reusable exact-Wilson workflow qualification

Status: **executed; proposed pass; awaiting explicit user review**

Date: 2026-09-25

Branch: `feature/aion-phase-two`

Parent at execution: `cadad8c` (`P2-1D: stream and restart exact Wilson runs`)

## 1. Claim under review

P2-1 removes the Chapter 13 direct campaign driver as the only way to execute
the accepted nonlinear exact-Wilson action. A fresh typed input can now:

1. prepare and reload a source-fixed Wilson stationary state;
2. build an authenticated CPU or GPU exact-action runtime;
3. advance one accepted contravariant-density interval in memory;
4. stream action-owned observables and diagnostics;
5. checkpoint, cancel, resume, inspect, export, and load artifacts; and
6. agree with the retained direct implementation at the measured numerical
   floor.

This is an implementation and numerical-qualification claim. It does not
expand the accepted physics beyond the exact all-electron, fixed-nucleus,
adiabatic pure-LDA/GGA action derived and qualified in Chapter 13.

## 2. Implemented reusable boundary

The public configuration family comprises
`WilsonStationaryConfig` and `WilsonSimulationConfig`. Both have strict TOML
schemas, deterministic resolved serialization, scientific identities, typed
exact/reduced actions, explicit grid/RI/cache realization, analytic affine
source, backend, and output policy. Unknown or inapplicable fields fail.

`prepare_wilson_stationary_state()` builds the selected source-fixed action,
solves the common generalized stationary problem, and returns an immutable
backend-neutral `WilsonStationaryStateData`. Its HDF5 artifact authenticates
the reference, grid, RI auxiliary space, source sample, action, density,
metric, occupied representative, occupation spectrum, energy components, SCF
history, and residuals. Stationary preparation is intentionally CPU-hosted;
the same portable state is accepted by CPU and physical-GPU dynamics.

`BuiltWilsonSimulation` owns a bounded density-independent LRU cache and a
stateful `NonlinearContravariantDensityPropagator`. One `step()` performs the
accepted self-consistent two-node fourth-order Gauss--Magnus update with a
Padé `[2/2]` coefficient link,

```text
P_(n+1) = U_n P_n U_n^dagger.
```

No Löwdin/Cholesky propagation frame, metric projection, post-step cleanup, or
empirical correction is introduced. The historical batch function now loops
over the same one-step engine but remains a separately invoked oracle in the
qualification. Density-dependent Hartree/XC evaluations are never cached.

The streaming runner records source samples, molecular/electronic/fixed-nuclear
dipoles, uniform action source current, metric/grid charge, complete action
energy components, source power, two-node Gauss work, Ward and weak-continuity
identities, and propagation/metric diagnostics on independent schedules.
Energy is not evaluated at every endpoint unless requested; source work is
nevertheless accumulated on every accepted interval. Checkpoints contain the
accepted contravariant density, global boundary, observer schedule state,
initial energy, accumulated work, scientific links, and restart lineage.

The ordinary CLI verbs dispatch by schema:

```text
aion prepare reference.toml
aion prepare stationary.toml
aion run simulation.toml
aion resume checkpoint.h5 --output child-run
aion inspect trajectory.h5
aion export trajectory.h5 csv-directory
```

## 3. Scientific realization

The selected anchor is the accepted Chapter 13 H3+ system:

| item | realization |
| --- | --- |
| molecule | equilateral H3+, charge +1, two electrons |
| geometry | side 1.4 bohr, molecular plane `xy` |
| AO basis | cc-pVDZ |
| action branch | exact Kohn--Sham LDA, `lda,vwn` |
| real-space grid | unpruned PySCF level 4, 78,120 points |
| RI auxiliary basis | `weigend` |
| RI retained-space rule | absolute cutoff `1e-7`, relative cutoff zero |
| gauge | symmetric affine gauge, origin `(0.17,-0.31,0.23)` bohr |
| static field | `Bz=0.03` a.u. |
| propagation | 8 intervals, `dt=0.25` a.u., `T=2.0` a.u. |
| nonlinear solve | tolerance `1e-12`, at most 80 iterations |
| precision | float64/complex128 |

The accepted NQ6 direct-electric pulse is not representable by the first
potential-first reusable source schema and is not silently relabelled. The
qualification therefore uses two complementary comparisons:

- the accepted eight-step NQ6 static-`B` trajectory is reproduced directly;
- a one-cycle compact sin-squared vector-potential pulse with peak electric
  field `0.005` a.u., `omega=pi` a.u., and `x` polarization is run through
  both the retained batch oracle and the integrated runtime.

The second pulse starts and ends with exactly zero reduced vector potential
and has zero impulse. Its stationary input is the same `Bz=0.03` state because
the pulse is exactly zero at `t=0`.

## 4. Execution record and recovery

The first execution root is

```text
/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/aion_phase_two/
  p2_1_qualification_20260925T180828Z_cadad8c
```

It completed fresh reference preparation, tight stationary preparation, the
full static trajectory, controlled cancellation at step 3, and resumed steps
4--8. On entering the separate driven oracle, the new qualification script
passed an option belonging to the stateful propagator to the historical batch
function. The resulting `TypeError` is retained in `failure.json`; none of the
completed static artifacts was altered.

The corrected driver added an authenticated `--reuse-static-root` recovery
mode. It verifies and hashes the failed record, reference, stationary state,
full trajectory, full final checkpoint, interrupted boundary-3 checkpoint,
and resumed final checkpoint before reusing them. The successful continuation
root is

```text
/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/aion_phase_two/
  p2_1_qualification_20260925T183313Z_cadad8c
```

It was launched through the physical-GPU wrapper with one process and at most
eight host threads:

```bash
AION_HOST_THREADS=8 tools/gpu-python \
  tools/run_phase_two_p2_1_qualification.py \
  --output /home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/aion_phase_two/p2_1_qualification_20260925T183313Z_cadad8c \
  --include-gpu \
  --reuse-static-root /home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/aion_phase_two/p2_1_qualification_20260925T180828Z_cadad8c
```

The continuation result and provenance hashes are respectively
`53fd19cf702624eb448bec19717fc4259056787a9b7bb7509bbf6586523d8238`
and
`3ddec9835fce68d78911ad2b344b31838123c849a0570e28bb1ddf3e9e84db68`.
They were independently recomputed from `completed.json` after execution.

## 5. Numerical results

All declared checks passed:

| comparison | measured residual | review limit | result |
| --- | ---: | ---: | --- |
| fresh stationary density vs accepted NQ4 | `9.0885e-13` relative | `2e-10` | pass |
| fresh stationary energy vs accepted NQ4 | `2.4025e-13` Ha | `2e-10` Ha | pass |
| integrated final mixed density vs accepted NQ6 | `1.6292e-12` relative | `3e-10` | pass |
| integrated final energy vs accepted NQ6 | `1.3856e-13` Ha | `3e-10` Ha | pass |
| driven integrated density vs retained batch oracle | exactly `0` | `2e-12` | pass |
| resumed vs uninterrupted final density | bitwise equal | exact | pass |
| resumed vs uninterrupted accumulated work | exactly `0` Ha | `2e-14` Ha | pass |
| physical GPU vs CPU driven final density | `4.5395e-14` relative | `2e-9` | pass |

The portable stationary HDF5 round trip preserved its content fingerprint and
density exactly. The fresh stationary solve converged in six iterations with
orbital residual `2.4727e-13` and density fixed-point residual `1.0402e-12`.
The GPU run explicitly asserted device residency of the initial and stepped
density, every accepted link, endpoint source current, and endpoint dipole.
The only GPU diagnostic was the expected GPU4PySCF notice that CuPy was used
as the contraction engine.

The completed continuation consumed 289.6 s and peaked at approximately
1.10 GB RSS. Its measured propagation stages were:

| stage | elapsed (s) |
| --- | ---: |
| driven retained batch oracle | `185.262` |
| driven integrated CPU runtime | `63.841` |
| driven integrated physical GPU runtime | `21.601` |

These are qualification timings from one machine and are not production
performance guarantees. In particular, the direct oracle deliberately lacks
the integrated runtime's bounded density-independent dynamic cache.

## 6. Software gates

The frozen candidate passed Ruff formatting and lint, strict Mypy over all 96
source files, 90 fast tests, 189 molecular integration tests, and 34
physical-GPU tests. The GPU suite emitted one expected GPU4PySCF notice that
CuPy was the contraction engine. The seven focused Wilson CPU integration
tests also passed before the H3+ campaign.

The required P2-1 behaviors have direct evidence:

| required behavior | evidence |
| --- | --- |
| strict exact/reduced TOML | configuration unit tests and round trips in the H3+ execution |
| algebraic one-step parity | CPU/GPU one-step tests introduced in P2-1B |
| multi-step H3+ parity | accepted NQ6 and driven batch comparisons above |
| stationary serialization | exact fingerprint and array round trip above |
| uninterrupted/resumed parity | H3+ bitwise final density comparison above |
| strict rejection | action/source/backend/link rejection tests |
| bounded memory/cache | duration-independent runner tests and configured LRU bound |
| CPU/GPU parity/residency | H3+ physical-GPU result above plus GPU suite |

## 7. Bounded scope and next gate

P2-1 does not accept pseudopotentials, hybrids, moving nuclei, periodic
boundaries, Maxwell backreaction, adaptive stepping, spatially nonuniform
sources, EELS potentials, reduced-action production dynamics, or a general
time-dependent magnetic envelope. It does not turn the ordinary field-free
Casida solver into a magnetic-field response oracle.

The exact Hartree and pure-GGA action branches exist, but P2-2 is responsible
for molecular transfer to the accepted CO/cc-pVDZ/PBE anchor and then the
fixed-geometry NH3/cc-pVDZ/PBE bridge. No later production application,
basis ladder, long pulse, or ammonia/PNA production choice is made here.

Subject to the final software-suite record, the numerical evidence supports a
**P2-1 pass proposal**. Only an explicit user decision may change the gate from
`executed_unreviewed`/`pending_user_review` to `accepted`.
