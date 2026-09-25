# P2-1 reusable exact-Wilson API and schema decision

Status: implementation decision, accepted by P2-0 dependency order  
Date: 25 September 2026  
Branch: `feature/aion-phase-two`  
Accepted prerequisite: `docs/reviews/phase_two_p2_0_review_20260925.json`

## Decision

P2-1 will integrate the accepted exact-Wilson action through Aion's existing
top-level workflow verbs while giving Wilson stationary states, trajectories,
and checkpoints their own typed schemas. The legacy orbital simulation schema
and all existing `1.x` artifacts remain unchanged and readable.

The public workflow remains:

```text
load configuration
  -> prepare reference
  -> prepare Wilson stationary state
  -> build_simulation
  -> run / checkpoint / resume
  -> load_trajectory
```

`build_simulation()`, `run()`, `resume()`, and `load_trajectory()` dispatch on
typed configuration or artifact identity. They do not ask users to call a
campaign driver. NQ6 and NQ8 remain independent direct numerical oracles.

## Why a separate Wilson schema is required

The existing `aion.simulation-input` and `aion.checkpoint` contracts store
occupied orbitals propagated by the SCEM family. The accepted Wilson algorithm
instead propagates a contravariant AO density by nonlinear Gauss-node
congruences in a time-dependent metric. Reinterpreting `coefficients` as a
density, or adding required density-native state to the existing checkpoint,
would silently change a `1.x` schema.

P2-1 therefore adds:

- `aion.wilson-stationary-input`, version `1.0.0`;
- `aion.wilson-simulation-input`, version `1.0.0`;
- `aion.wilson-stationary-state`, HDF5 version `1.0.0`;
- `aion.wilson-trajectory`, HDF5 version `1.0.0`; and
- `aion.wilson-checkpoint`, HDF5 version `1.0.0`.

Old configuration and artifact identities do not change.

## Action configuration

The `[action]` table is a discriminated union.

### Exact action

`kind = "exact_wilson"` selects the complete straight-Wilson action. Its
explicit branch is one of `hartree`, `kohn_sham_lda`, or `kohn_sham_gga`.
The functional must agree with the prepared reference and branch. Hartree,
pure LDA, and pure GGA are distinct action identities.

### Reduced action

`kind = "reduced_wilson"` has a separate required `level`: `p0`, `e1`,
`strict_c1`, or `density_resummed_c1`, plus the nonlinear closure branch.
Reduced GGA is rejected because it has not been derived or qualified. The
reduced action is never an approximation flag on an exact action.

P2-1 makes exact and reduced configurations strictly parseable and supports
stationary preparation from the already implemented action factories. The
production trajectory gate is exact-first. Reduced production dynamics remain
disabled with a typed unsupported-configuration error until P2-6 qualifies
their integrated path.

## Numerical realization

The `[numerics]` table records every realization choice needed to reconstruct
the discrete action:

- AO grid source (`reference` or explicit qualification grid), level and
  pruning;
- AO block size and optional memory budget;
- a strict maximum number of spatial and temporal dynamic-cache entries;
- auxiliary basis;
- RI metric relative threshold, absolute threshold, and optional maximum
  rank; and
- physical constants, fixed at the qualified atomic-unit values unless a
  later schema explicitly expands the domain.

The first P2-1/P2-2 realization is an unpruned level-4 qualification grid,
`weigend`, RI absolute threshold `1e-7`, relative threshold zero, no maximum
rank, and explicit block size. These values are serialized and hashed rather
than supplied by an undocumented workflow default.

## Analytic affine source

The first reusable Wilson source is a Maxwell-consistent affine provider. It
composes:

- an analytic potential-first uniform electric provider (zero or the existing
  compact `sin^2` vector-potential pulse);
- a constant electric-field offset at the electromagnetic origin;
- a magnetic field specified at an explicit reference time;
- a constant analytic magnetic-field derivative;
- symmetric or Landau affine gauge representative; and
- the electromagnetic origin stored by the reference.

At arbitrary endpoints or Gauss nodes it returns exact `E_origin(t)`, `B(t)`,
and `dB(t)/dt`, and hence the accepted scalar/vector potentials and temporal
connection. A small bounded cache may retain current-step spatial actions; it
must not grow with trajectory length.

The runner depends on a source-provider protocol rather than on pulse details.
This preserves a future extension seam for spatially nonuniform exact actions
without claiming that the present affine implementation already covers EELS,
moving charges, or general non-dipole fields.

## Stationary-state contract

`aion.wilson-stationary-input` contains an authenticated reference link,
action, numerical realization, affine source, source time, SCF policy,
backend, output path, and metadata. Preparation constructs the exact or reduced
source-fixed model and solves it with the accepted common Wilson stationary
algorithm.

The immutable state stores:

- contravariant and mixed densities;
- occupied coefficients and occupations as an optional reproducibility
  representative, not as the propagated state;
- source sample and action/numerical identities;
- metric and energy components;
- stationary, particle-number, spectrum, double-counting, and metric
  residuals; and
- reference, grid, auxiliary-space, source, and state fingerprints.

A Wilson simulation links this artifact by path and SHA-256 content identity.
Construction verifies that reference, action, numerics, source at the initial
time, backend precision, density dimensions, and fingerprints agree.

## Stateful propagation

The accepted batch function is refactored around one stateful
`NonlinearContravariantDensityPropagator`. It owns only:

- one accepted contravariant density and integer boundary index;
- the initial mixed-density occupation spectrum needed for diagnostics;
- fixed step and nonlinear policies; and
- bounded scratch for two Gauss-node densities, generators, and links.

One `step()` performs exactly the accepted self-consistent two-node,
fourth-order Gauss--Magnus algorithm and the `[2/2]` Padé link. The accepted
update remains

```text
P_(n+1) = U P_n U^dagger
```

with no Löwdin or Cholesky propagation frame, metric projection, correction,
or post-step cleanup. The result exposes the two converged node densities and
evaluations, link, endpoint state, and existing diagnostics. The historical
batch function loops over this one-step object so algebraic parity is directly
testable rather than reimplemented.

## Exact runtime and caches

`build_simulation()` constructs one `BuiltWilsonSimulation` containing the
authenticated stationary state, prepared quadrature/factory, analytic source,
bounded dynamic-sample cache, stateful propagator, observable schedules, and
transactional output policy.

For constant `B`, the density-independent spatial action is reused exactly.
For time-dependent `B`, spatial actions are cached only within a declared
small LRU bound. Nonlinear evaluations remain density-dependent and are never
reused for another density.

## Formulation-owned observables

The exact runtime records only definitions belonging to the Wilson action or
explicitly labelled diagnostics:

- metric particle number and occupation-spectrum drift;
- electronic dipole from the dressed real-space charge density, fixed-nuclear
  dipole, and their sum;
- three Cartesian uniform-source-current components from independent unit
  affine electric variations;
- complete on-shell source power for the actual electric/induction source;
- kinetic, electron--nuclear, Hartree, XC, nuclear-repulsion, electronic, and
  molecular-total energies;
- accumulated fourth-order Gauss-node source work;
- matrix-route power and power-identity residual;
- weak continuity and global charge residuals; and
- nonlinear, cross-metric, Hermiticity, trace, occupation, and metric
  diagnostics.

Ambient canonical momentum or bare mechanical current may be added only under
an explicitly diagnostic name. It cannot replace the action source current.
Energy is evaluated only on its configured schedule; it is not computed at
every endpoint by default.

Interval work uses the same two accepted Gauss nodes as the fourth-order
propagator:

```text
Delta W_n = h/2 [P_source(t_-) + P_source(t_+)].
```

This avoids adding a lower-order midpoint quadrature to a fourth-order step.

## Streaming, checkpoint, and restart

Wilson trajectory output is append-only and transactional. Ordinary execution
retains no trajectory-wide density or node history in memory. Optional matrix
snapshots follow an explicit schedule.

Each checkpoint stores the accepted contravariant density, integer step,
accumulated work, initial molecular energy, action/source/state identities,
observer schedule state, and restart lineage. Source providers are analytic
and stateless in P2-1, so no hidden integrator state is required. Resume
rebuilds the exact action, verifies every identity, restores the density and
accumulated work, and creates a child trajectory segment.

Cancellation publishes a final boundary, checkpoint, status, and controlled
failure artifact using the same signal-safe policy as the legacy runner.

## CLI and expert Python API

The CLI keeps the existing verbs:

- `aion prepare reference.toml` prepares a reference;
- `aion prepare stationary.toml` prepares a Wilson stationary state;
- `aion run wilson.toml` builds and runs a Wilson simulation;
- `aion resume checkpoint.h5` dispatches by checkpoint schema; and
- `aion inspect` and `aion export` dispatch by artifact schema.

The Python API adds typed `prepare_wilson_stationary_state`,
`load_wilson_stationary_state`, `BuiltWilsonSimulation`, and Wilson
configuration classes while preserving all existing names.

## Implementation increments

1. **P2-1A -- contracts and sources:** configuration models, strict TOML,
   affine provider, schema declarations, public exports, and rejection tests.
2. **P2-1B -- state and one-step engine:** stationary-state I/O and stateful
   one-step propagator, with batch parity tests on CPU and physical GPU.
3. **P2-1C -- construction and observables:** reusable exact factory/runtime,
   bounded cache, action-owned observables, and direct-driver comparison.
4. **P2-1D -- streaming and restart:** Wilson trajectory/checkpoint writers,
   runner dispatch, cancellation, resume, and bounded-memory tests.
5. **P2-1E -- CLI and qualification:** end-to-end H3+ reproduction, restart
   parity, CPU/GPU parity and residency, documentation, authenticated evidence,
   and user review.

Each increment is committed separately. A direct NQ6/NQ8 driver is never
rewritten to call the integrated runtime; it remains an independent oracle.

## P2-1 pass condition

P2-1 may be proposed for acceptance only when a fresh strict Wilson input can
prepare and reload its stationary state, execute and reload a short exact
trajectory, checkpoint and resume with uninterrupted parity, remain bounded in
memory with duration, and reproduce selected accepted direct results at the
measured floor on CPU and physical GPU. Passing unit tests alone is not gate
acceptance.
