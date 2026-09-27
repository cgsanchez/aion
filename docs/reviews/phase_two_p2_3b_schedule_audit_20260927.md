# P2-3B schedule and duplication audit: NH3 exact-Wilson GPU

Status: executed and verified; interpretation proposed, not yet accepted by the user.

Campaign: `/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/aion_phase_two/p2_3_schedule_audit_20260927T164247Z_85b7c8519553`

Driver: `tools/run_phase_two_p2_3_schedule_audit.py` at commit
`85b7c8519553bf411f841f158569de93882aa54e` (clean recorded worktree).
The campaign inherits the accepted P2-2 NH3 reference/stationary state and the
verified P2-3A component profile. It ran on the physical GPU with eight host
threads. The managed Aion Python was
`/home/cgs/01_TOOLS/EasyBuild/conda/envs/aion/bin/python`.

## Verification and scope

- `completed.json` reports `executed_unreviewed`; the service log records all
  six variants and a normal exit. Its only warning is gpu4pyscf selecting CuPy
  for tensor contraction.
- `_authenticate_campaign(campaign_root)` verified all 45 recorded hashes,
  including the result, provenance, identity, six inputs, six trajectories,
  checkpoints, and six per-variant records. Recorded result SHA-256:
  `9300808c48bd5a2bcb83bc3a0eb66a9da29234873cfc90220389b0f46d73c0e6`.
- All variants use the same NH3/cc-pVDZ/PBE initial state, source, nonlinear
  Gauss--Magnus propagator, and `dt=0.025` au. Only output schedules differ.
  Four intervals reach `t=0.100` au, the first half of the accepted
  `0.200`-au one-cycle pulse. The experiment measures workflow cost, **not**
  a new physical accuracy regime.
- The runner forces a complete initial observation regardless of schedules.
  All variants also perform the two mandatory Gauss-node power evaluations per
  interval to integrate source work. Thus `minimal` is not a propagation-only
  benchmark.
- Final contravariant densities and accumulated source work are bitwise equal
  across all six variants. Every common series sample at its recorded step is
  bitwise equal to the `full` variant (26 common series). The `full` variant's
  step-4 checkpoint density and work, and all 130 records across 26 series
  through step 4, are bitwise equal to the accepted eight-step P2-2 NH3 GPU
  trajectory.

## End-to-end measurement

Times are synchronized only at the whole-run boundaries and include normal
streaming and checkpoint I/O. Each variant is one execution, not a statistical
distribution. The initial `full` and final `full_repeat` bound cold/order drift.

| Variant | Observations after initial | Checkpoint files | Run time (s) | Endpoint PBE calls | Gauss-power PBE calls |
|---|---|---:|---:|---:|---:|
| `full` | current, energy, identities | 5 | 186.96 | 50 | 16 |
| `no_diagnostics` | current, energy | 5 | 156.22 | 34 | 16 |
| `energy_only` | energy | 5 | 155.74 | 34 | 16 |
| `minimal` | none | 2 | 94.97 | 10 | 16 |
| `full_sparse_checkpoint` | current, energy, identities | 2 | 181.75 | 50 | 16 |
| `full_repeat` | current, energy, identities | 5 | 181.73 | 50 | 16 |

Every variant also makes exactly 40 propagation PBE evaluations across its
four steps, so scheduling does not change the nonlinear propagation path.
The final `full_repeat` is 5.24 s faster than the initial `full` (2.9% of the
repeat), evidence of run-order/cold-start drift. Against the repeated full
run, removing post-initial identities changes wall time by about 25.5 s;
removing current *output* while retaining energy changes it by only 0.47 s.
The `energy_only` to `minimal` contrast is about 60.8 s, but also changes the
checkpoint schedule. The separate `full_sparse_checkpoint` comparison changes
three checkpoint files (about 128 kB) but differs from `full_repeat` by just
0.02 s. These are directional single-run contrasts, not precise speedup
estimates or an I/O benchmark.

## Duplication audit and optimization implication

The code and measured call counts agree on the following boundaries:

1. `BuiltWilsonSimulation.observe_endpoint()` calls the monolithic
   `evaluate_exact_wilson_endpoint_observation()` for **any** current, energy,
   or identity request. That function always computes power, charge, three
   Cartesian currents, and dipole. Only energy packaging and identities are
   conditional. Accordingly, `energy_only` still makes one complete action
   evaluation and six PBE evaluations at each of its four post-initial
   endpoints, exactly as `no_diagnostics`; their run times are nearly equal.
   The output series names alone do not reveal this, because the forced initial
   observation creates all series even in `minimal`.
2. For an endpoint with identities, PBE is called ten times: three base-action
   evaluations and seven directional evaluations. Without identities it is
   called six times: one base and five directional. The Ward and continuity
   functions each reevaluate `model.evaluate(density)` at the same endpoint
   source/density; the first action was already evaluated by
   `observe_endpoint()`. Reusing that *same-state* action is an exact candidate
   improvement, subject to explicit identity tests.
3. Current/power variation paths repeatedly call the Hartree and PBE
   source-direction evaluators at the same source/density for the electric
   direction, three Cartesian directions, and (when requested) weak tests.
   A local observation workspace or batched variations could share base
   density/frame work. It must never cache nonlinear action data across a
   changed density or source.
4. Gauss-node work contributes 16 directional PBE calls per four-step run,
   even in `minimal`, as required by the work integral. This is a separate
   optimization target from lazy endpoint selection.
5. The dynamic cache reports one spatial miss and 13 sample misses in every
   variant, consistent with the distinct temporal samples. There is no
   evidence here that repeated spatial-action construction dominates.
6. All schedules invoke `_checkpoint()` and flush four times; sparse
   scheduling reduces actual checkpoint files from five to two but did not
   measurably reduce wall time in this experiment. HDF5 scheduling is not the
   first optimization target. GPU-memory-pool usage was the same for all
   variants; the process RSS maxima accumulate across serial runs and are not
   independent per-variant peak-memory measurements.

The measured first P2-3C increment should therefore be **lazy endpoint
observation selection**, preserving the existing definitions and sampling
semantics. Then consider same-source/density action reuse for Ward and
continuity, followed by local source-direction batching. Each increment
requires exact comparison of affected observables, identities, checkpoint/
restart artifacts, and CPU/GPU behavior under the unchanged P2-2 floors.
The present audit does not authorize dropping Gauss-node work or changing the
physical action.

## Entry points

- Campaign `result.json`, `provenance.json`, `completed.json`, and per-variant
  JSON records reside at the campaign root above. Each variant directory
  contains its immutable `trajectory.h5` and checkpoint files.
- The controlling matrix is `docs/phase_two_p2_3_execution_matrix.md`.
- Runner scheduling: `src/aion/workflows/wilson_runner.py`.
- Endpoint observation: `src/aion/observables/wilson.py`.
- Power and weak current: `src/aion/electronic_structure/wilson_dynamics.py`
  and `src/aion/electronic_structure/wilson_sources.py`.
