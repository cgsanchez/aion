# Phase Two P2-0 executable baseline

Status: **analyzed, proposed pass, awaiting user review**  
Date: 25 September 2026  
Branch: `feature/aion-phase-two`  
Accepted parent: `dc6d4e2e0267bf2542074528ca1305e5fdff592b`  
Evidence root: `/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/aion_phase_two/p2_0_baseline_20260925T160229Z_8aa56cbb43c8`

This document is the human entry point for P2-0. It distinguishes inherited
evidence, newly executed numerical evidence, analysis, and user acceptance.
The calculations and synthesis have been executed, but no review record claims
acceptance yet.

## Gate conclusion

The P2-0 analyzer proposes **pass**. The accepted Chapter 13 parent is an
ancestor of the Phase Two branch, the accepted NQ9 package authenticates, all
software-quality and physical-GPU gates pass, the selected ammonia heritage is
hashed without being relabelled as exact-Wilson evidence, and bounded H3+ LDA
and CO PBE exact-Wilson CPU baselines completed with small algebraic and
geometric residuals.

The proposal has two non-blocking inherited or packaging limitations:

1. the old refactor WP7 bounded 6-31G analyzer summary was never produced,
   although ten trajectories and the retained aug-cc-pVTZ evidence exist; and
2. the editable import resolves to this worktree and reports the required
   `0.2.0.dev7`, but the managed environment's distribution metadata still
   says `0.2.0.dev1`. The public source-version contract passed. The managed
   prefix was not mutated during P2-0.

Neither limitation changes which source was executed. They remain visible and
must not be silently reinterpreted as completed evidence.

## Requirement reconciliation

| P2-0 requirement | Evidence | Result |
|---|---|---|
| Branch from accepted `dc6d4e2` | `accepted_parent_is_ancestor=true` in final reconciliation | Pass |
| Authenticate accepted NQ9 | 35 files, manifest SHA-256 `11f956abdf6f4bdabc2324cb6e95fe3ba940d54ee0a5b072deb981aa6445ae84` | Pass |
| Record managed environment and launch boundaries | Final reconciliation, quality provenance, eight physical CPU cores, physical GPU wrapper | Pass |
| Format, lint, type, fast, integration, physical GPU | All six authenticated gates returned zero | Pass |
| Reconcile `0.2.0.dev7`, WP7, and exact Chapter 13 | Source version verified; stale distribution label and incomplete historical WP7 analyzer bounded explicitly | Pass with recorded limitations |
| Authenticate selected ammonia heritage | 16 artifacts, manifest SHA-256 `273de82d853eaaa167885ec5a25f534f00cac0d1605f4d340db2bfcffccd6e75` | Pass |
| Define Phase Two evidence schemas | `docs/phase_two_evidence_contracts.md` | Pass |
| Record H3+ LDA and CO PBE cost baselines | Two completed, hashed exact-Wilson benchmark records | Pass |

## Reproducible lineage and environment

The accepted parent is commit `dc6d4e2`; P2-0 evidence tooling began at
`8aa56cbb43c847b20aea65755731930b661eaa30`. Mechanical locked-Ruff
formatting is isolated in commit `7e5a8aa`. Later commits only repair and make
explicit the P2-0 evidence checks; they do not alter the accepted exact-Wilson
physics in `src/aion`.

The managed environment used Python 3.12.13 with NumPy 2.5.2, SciPy 1.18.0,
h5py 3.16.0, PySCF 2.13.1, CuPy CUDA 12.x 14.1.1, GPU4PySCF CUDA 12.x 1.7.4,
Ruff 0.16.6, mypy 2.3.1, pytest 9.1.1, and pytest-xdist 3.8.0. CPU molecular
operations set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
and `NUMEXPR_NUM_THREADS` to eight. Parallel fast tests used eight workers with
one numerical thread each. GPU tests used `tools/gpu-python` and a physical
device, also with one CPU thread per numerical library.

The quality campaign recorded:

| Gate | Result | Elapsed (s) |
|---|---:|---:|
| Ruff lint | pass | 0.010 |
| Ruff format check | pass | 0.009 |
| mypy | pass | 0.134 |
| fast tests, 73 passed | pass | 2.966 |
| integration tests, 182 passed and 126 deselected | pass | 68.761 |
| physical-GPU tests, 31 passed and 277 deselected | pass | 53.043 |

The GPU run emitted the already expected CuPy contraction warning; it did not
fail a test.

## Heritage boundary

The ammonia manifest authenticates geometry and optimization records, a
field-free Casida reference, resonant-pulse summaries and trajectories, bare
kick summaries, spectra and trajectories, a reduced P0+E1 kick, and CPU/GPU
timing records. These are historical ordinary or reduced-action calculations.
They may guide geometry, source, spectrum, and cost choices, but they are not
evidence for the exact action.

The first exact-Wilson bridge uses the fixed PBE/cc-pVDZ optimized geometry
`G_DZ`. The distinct aug-cc-pVTZ-optimized geometry `G_augTZ` must not be mixed
into a geometry-fixed cc-pVXZ convergence series.

## Exact-Wilson timing baselines

Both records use CPU `float64/complex128`, eight physical cores, grid level 4,
and the `weigend` auxiliary basis. Stationary timings are warm starts from
accepted states and are not from-scratch SCF costs.

| Quantity | H3+ LDA/VWN, cc-pVDZ | CO PBE, cc-pVDZ |
|---|---:|---:|
| AOs | 15 | 28 |
| Grid points | 78,120 | 106,200 |
| Total process time (s) | 50.459 | 815.693 |
| Peak RSS (MiB) | 214.6 | 359.6 |
| Static spatial/sample preparation (s) | 6.446 | 79.798 |
| Warm-start stationary solve (s) | 0.309 | 6.478 |
| One fourth-order nonlinear step (s) | 10.494 | 327.741 |
| Nonlinear iterations | 4 | 3 |
| Nonlinear residual | `3.30e-15` | `1.19e-13` |
| Cross-metric residual | `2.99e-16` | `3.79e-16` |
| Occupation-spectrum drift | `4.49e-16` | `7.24e-16` |
| Power-identity residual (a.u.) | `6.68e-46` | `2.86e-15` |

The H3+ step has interval 0.0625 a.u.; the CO step has interval 0.025 a.u.
CO's recorded stage timings do not include all endpoint diagnostic work: the
815.7 s total also includes construction and evaluation of the endpoint GGA
sample used for the final action and power record. Therefore the total, not
the sum of named stages, is the correct wall-time baseline.

### CO reference reconstruction

The accepted NQ8 CO checkpoint stores occupied coefficients with shape
`(28, 7)`, while the AO density is `(28, 28)`. A freshly repeated PySCF SCF
calculation reproduced the accepted reference energy exactly to the recorded
precision, the electron count to `1.78e-15`, and all expected dimensions. Its
bytewise prepared-reference fingerprint differs because that fingerprint
includes floating-point orbital and density arrays. It is not a portable
scientific-equivalence test for a repeated SCF calculation.

Three failed-visible preflight attempts are retained under `benchmarks/co`,
`benchmarks/co_completed`, and `benchmarks/co_completed_retry`. They document,
respectively, the rejected bytewise fingerprint equality, exact floating-point
electron-count equality, and an incorrect square-coefficient assumption. The
successful `benchmarks/co_completed_final` record replaces none of them.

## Frozen implementation matrix through P2-2

P2-0 freezes scope, not an application campaign:

- P2-1 integrates the already accepted exact Hartree, pure LDA, and pure GGA
  branches into reusable strict configuration, stationary-state, propagation,
  observable, persistence, restart, CLI, and expert-Python APIs. Reduced
  actions remain separately discriminated.
- The accepted fourth-order self-consistent two-node tensorial Gauss--Magnus
  propagation and uncorrected mixed-index geometry are retained unchanged.
- P2-2 first reproduces bounded H3+/cc-pVDZ/LDA and CO/cc-pVDZ/PBE results,
  then transfers to NH3 at fixed `G_DZ`, cc-pVDZ/PBE, initially at `B=0`, and
  finally applies a short smooth electric pulse along the C3 axis.
- `weigend` and grid level 4 are the first exact reference realization. Any
  grid, auxiliary-space, tolerance, source-amplitude, duration, or timestep
  refinement is an explicit linked decision, never an implicit default.
- CPU evidence precedes physical-GPU parity. No long trajectory or production
  molecule is selected by this gate.

## Artifact index and authentication

The final synthesis is `analysis_final/result.json`, with SHA-256
`bc2ec88e4d4ba98141c6542243bcaaf26ff7b9ed55e17615bee466adec425883`.
Its provenance SHA-256 is
`09d85539e98dc385acd6d8d6e37eb4b269642dddeb8b8deddb8e79eb66a747b7`.

Authenticated subordinate results are:

| Artifact | Result SHA-256 | Provenance SHA-256 |
|---|---|---|
| `reconciliation_final` | `ca3cbd645894b553ac4bb705ed7bf2e3234cd22e4366792005e1533b6cd4b2ae` | `f05d3585393ee7699457b76365ca75765066f20abf2c92d7bb5fc8ab325818ce` |
| `quality` | `db447a135d299e7a74761daf3d0ba812269a3f43f915cca46eb7c0c97367abb0` | `3010a446793a9fa2c2678cdb2aac48c8dcf2154974e9b30bd908c23aa995490c` |
| `benchmarks/h3plus` | `917db214308cccf1d3a68bd2f28780079b12271dfa957ae4f287a60aa5d7433d` | `46eb9de91b8d1991629112a1ccc0340d9978b9d11a60b5073ca0206c57588577` |
| `benchmarks/co_completed_final` | `31742143eb13bb945635251f8f7fb9c5c910c82b4de06a9ed212a4fe9898b401` | `5ec742663a8acbe56c5994a4994221eec156691d4e96205411bbb315d5782c66` |

Each completed subcampaign has its own `completed.json`; the analyzer
recomputed every recorded result and provenance hash before synthesis. The
original `reconciliation` and `analysis` artifacts are retained as superseded
evidence because they first exposed the stale distribution label.

## Review decision required

The next permitted action after user acceptance is the P2-1 API and schema
decision record, followed by reusable-runtime implementation. Until a separate
review record authenticates this document and the final synthesis, P2-0 is not
labelled accepted and P2-1 source implementation does not begin.
