# P2-3C1: lazy endpoint energy qualification

Status: executed and verified; retention proposed, pending user review. This
does not close or accept the full P2-3 performance gate.

Implementation commit: `4acf67aeab5e029c7eb9f1296da453410a075e8a` on
`feature/aion-phase-two`. The recorded worktree was clean.

Campaign: `/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/aion_phase_two/p2_3_lazy_energy_20260927T173314Z_4acf67aeab5e`

Frozen comparator: the authenticated P2-3B schedule audit at
`/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/aion_phase_two/p2_3_schedule_audit_20260927T164247Z_85b7c8519553`.

## Change and invariants

The runner now selects `BuiltWilsonSimulation.observe_energy()` when energy is
the only scheduled endpoint observation. That method packages the energy from
one fresh exact-Wilson action evaluation. It does not evaluate endpoint
current, power, charge, dipole, Ward, or continuity. The existing complete
endpoint observation remains the path when current or diagnostics are
scheduled; its energy package uses the same helper, preserving the definition.

The initial boundary still receives the existing forced full observation.
The nonlinear Gauss--Magnus step, its action calls, both Gauss-node power
evaluations per interval, source-work integration, checkpoints, and energy
residual validation are unchanged. No physical approximation, tolerance, or
floating-point contraction order was changed.

## Executed evidence

The campaign completed normally on the physical GPU with eight host threads.
`_authenticate_campaign(campaign_root)` verified all 19 recorded hashes,
including its identity, result, provenance, inputs, trajectories, checkpoints,
and per-variant records. Result SHA-256:
`2c6da3ec78eec1d2830225bc99b098f9f2ca94806c747b2f0330350d4071ca87`.
The only service-log warning was gpu4pyscf selecting CuPy tensor contraction.

Both new variants use the accepted NH3/cc-pVDZ/PBE source and the same four
intervals at `dt=0.025` au as P2-3B. The qualification script compared every
numeric HDF5 trajectory dataset with its frozen counterpart, plus density,
source work, initial energy, observer state, and source fingerprint in each
checkpoint. These are exact comparisons, not tolerance-based similarities.

| Variant | Frozen P2-3B run (s) | P2-3C1 run (s) | Endpoint PBE calls, frozen → new | Bitwise-equal numeric datasets | Bitwise-equal checkpoints |
|---|---:|---:|---:|---:|---:|
| full-output control | 186.96 | 187.47 | 50 → 50 | 105 | 5 |
| energy-only | 155.74 | 99.11 | 34 → 14 | 93 | 5 |

For energy-only, the change removes 20 endpoint directional PBE/Hartree
evaluations over four post-initial boundaries. One full initial observation
remains. All 40 propagation PBE evaluations and all 16 mandatory Gauss-power
directional PBE evaluations remain in both variants. The energy-only wall-time
difference is 56.63 s (36.4% of the frozen run); the unchanged full control
differs by 0.51 s (0.3%). The earlier P2-3B full/repeat spread was 5.24 s.
This is strong directional evidence of a material benefit, but single runs
are not a timing distribution or a machine-independent speedup claim.

Explicit instrumented host-to-device calls decreased from 326,402 to 204,870
for energy-only; corresponding logical bytes decreased from 4.976 GB to
2.876 GB. These are instrumented `backend.asarray` transfers, not a hardware
PCIe trace. The full control's counts were unchanged.

Software checks before the campaign: 116 fast tests passed; nine relevant CPU
integration tests passed; three targeted physical-GPU tests passed; Ruff
lint/format, mypy, and `git diff --check` passed. The first physical-GPU test
attempt exposed a test-fixture sampling mismatch, not an implementation
failure; after aligning the baseline energy schedule, all three passed.

## Conclusion and boundary

The lazy energy-only increment is numerically exact for the executed NH3
trajectory and materially reduces its measured endpoint cost. It should be
retained. Other combinations (notably diagnostics-only) still use the
monolithic endpoint path, and this increment does not address repeated
same-source/density work within Ward, continuity, or current directions.
Those remain separate, testable P2-3C candidates. Full P2-3D qualification
has not yet been executed.

Entry points: `result.json`, `provenance.json`, `completed.json`, and the
per-variant JSON files at the campaign root; implementation in
`src/aion/observables/wilson.py`, `src/aion/workflows/wilson.py`, and
`src/aion/workflows/wilson_runner.py`; qualification driver in
`tools/run_phase_two_p2_3_lazy_energy_qualification.py`.
