# Phase Two P2-3 performance and memory execution matrix

Status: active measurement plan

Date: 2026-09-27

## 1. Objective and invariant boundary

P2-3 measures and reduces the cost of the accepted exact-Wilson runtime. It
does not introduce a physical approximation, select a new scientific
application, or weaken a P2-2 tolerance. Every optimization must preserve the
same discrete action, source sampling, nonlinear solve, mixed-index density
update, observables, and immutable artifact semantics.

The accepted P2-2 CO and NH3 campaigns are the numerical reference. Raw P2-2
artifacts remain immutable. P2-3 uses new campaign roots and authenticates all
inherited inputs before measurement.

## 2. Measurement hierarchy

P2-3 separates three kinds of timing that must not be conflated:

1. **End-to-end wall time** measures the production workflow with its actual
   schedules, transfers, streaming, checkpoints, and synchronization.
2. **Schedule ablations** run the same propagation while changing only which
   already-defined observables or artifacts are requested. They assign cost
   to observations and I/O without changing the propagated state.
3. **Synchronized component microbenchmarks** time one named operation after
   an explicit backend synchronization. They expose expensive components but
   are not added together as a prediction of production wall time because
   synchronization changes overlap and launch amortization.

Peak process RSS, resident GPU memory before and after each stage, cache
statistics, call counts, and transferred byte counts are recorded alongside
time. GPU timing always synchronizes at the measurement boundary. A physical
GPU run may not fall back to CPU.

## 3. P2-3A: unoptimized baseline decomposition

The first target is the accepted fixed-geometry NH3/cc-pVDZ/PBE bridge on the
physical GPU. It is both the first polyatomic transfer case and the case for
which P2-2 already provides authenticated CPU/GPU wall time. The first
campaign reuses the P2-2 reference and stationary state and measures:

- complete runtime construction;
- AO quadrature reconstruction;
- exact spatial one-electron and RI-action preparation;
- temporal connection and source-sample construction, cold and cached;
- one full nonlinear action evaluation;
- RI Hartree contraction;
- PBE grid evaluation, including its explicit GPU--CPU libxc boundary;
- one self-consistent Gauss--Magnus step, its nonlinear action calls, and the
  remaining dense-link/iteration work;
- the two mandatory Gauss-node source-power evaluations;
- endpoint charge/dipole/current/power work;
- Cartesian current components;
- Ward and weak-continuity diagnostics;
- one density host transfer;
- representative HDF5 stream creation, append, flush, finalization, and one
  transactional checkpoint;
- cache hit/miss counts and peak memory.

The existing P2-2 CPU wall times remain the first CPU baseline. A new long CPU
microprofile is launched only after the GPU decomposition identifies a
specific dominant region, so P2-3 does not spend hours measuring code that
will immediately change.

## 4. P2-3B: schedule and duplication audit

Execution status: six-variant physical-GPU audit completed, hashes verified.
The user accepted its priority recommendation for the first P2-3C increment;
the final P2-3 gate remains unreviewed. See the
[P2-3B schedule audit](reviews/phase_two_p2_3b_schedule_audit_20260927.md)
for authenticated measurements, bitwise comparisons, and the proposed P2-3C
priority.

The baseline is used to identify repeated exact work across propagation,
Gauss-node work integration, endpoint observations, identities, and storage.
The audit explicitly checks:

- whether an endpoint request computes unrequested observables;
- whether density-dependent Hartree/PBE action data are recomputed at the
  same state and source;
- whether the three uniform-current directions can share their common base
  density, AO-frame, Hartree, or PBE work without caching a nonlinear action
  across different states;
- whether Ward and continuity diagnostics repeat the same action evaluation;
- whether source samples or spatial tensors are rebuilt despite cache hits;
- whether host/device transfers occur only at the declared CPU-libxc and
  storage boundaries; and
- whether trajectory flushing and checkpoint schedules dominate short runs.

Any proposed reuse must be local to one immutable `(source, density)`
observation and must be invalidated immediately when either changes. P2-3
does not authorize frozen-density, stale-action, or cross-step nonlinear
caching.

## 5. P2-3C: bounded implementation increments

Selected first increment: lazy energy-only endpoint observation, using the
existing action energy without unrequested current, power, charge, and dipole
variations. Implementation and physical-GPU qualification are complete;
retention is proposed pending user review. The
[P2-3C1 qualification](reviews/phase_two_p2_3c1_lazy_energy_qualification_20260927.md)
records the bitwise frozen-trajectory comparison and measured cost. This
increment does not change mandatory Gauss-node work integration.

Optimizations are selected in descending measured cost. Each increment is
independently reviewable and normally belongs to one of these exact classes:

1. lazy observation selection;
2. same-state observation workspace and common-subexpression reuse;
3. batched source variations or contraction reordering;
4. persistent backend-resident arrays and preallocated scratch;
5. density-independent cache or construction reuse;
6. removal of undeclared host/device transfers; or
7. streaming/checkpoint batching that preserves crash and restart semantics.

An increment is retained only if its benchmark improvement exceeds timing
noise and its affected scientific comparisons pass. No threshold may be
changed to make an optimization pass.

## 6. P2-3D: qualification matrix

The final optimized code is compared with the frozen P2-2 implementation on:

- lower mechanical matrices and energy components;
- source directions, current, power, Ward, and continuity quantities;
- one-step and eight-step contravariant densities;
- CO and NH3 endpoint observables;
- restart equivalence and immutable artifact loading;
- CPU/physical-GPU parity and device residency;
- complete fast, integration, type, lint, formatting, and physical-GPU
  software gates; and
- end-to-end time and memory using identical input and output schedules.

Array comparisons use the established P2-2 floors or tighter component-local
roundoff floors. Exact serialized equality is required where an operation is
only rescheduled or buffered and no floating-point contraction order changes.

## 7. Resource and stop rules

- At most one molecular process runs at a time with eight host threads.
- Physical-GPU work uses `tools/gpu-python` with explicit device visibility.
- Long jobs are detached and status-bearing; no automatic polling is used.
- A pilot precedes any new CPU stage estimated above 15 minutes.
- Failed and superseded profiles remain visible in separate campaign roots.
- P2-3 stops for user review if the dominant cost can be reduced only by a
  physical approximation, a changed contraction order outside the accepted
  floor, weakened validation, unbounded memory, or altered restart semantics.

## 8. Gate condition

P2-3 may be proposed for acceptance only when the cost decomposition is
authenticated, retained optimizations show a reproducible benefit, all
affected scientific identities and backend comparisons pass unchanged, and
the remaining bottlenecks and machine-specific limits are stated explicitly.
