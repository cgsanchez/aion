# Aion Phase Two: reusable exact-Wilson dynamics and qualification

Status: active, mutable execution guide

Date: 25 September 2026

Repository: `/home/cgs/00_WORK/Projection_Code/aion`

Branch: `feature/aion-phase-two`

Accepted parent: `dc6d4e2e0267bf2542074528ca1305e5fdff592b`

## 1. Purpose

Phase Two turns the accepted Chapter 13 exact straight-Wilson Hartree and
adiabatic Kohn--Sham implementation into a reusable Aion simulation path, and
then qualifies that path for longer dynamics and real-time spectroscopy.

The work has four ordered objectives:

1. integrate the exact-Wilson stationary, dynamic, propagation, observable,
   and persistence components behind reusable typed APIs;
2. demonstrate that the integrated path reproduces the accepted Chapter 13
   evidence before expanding its molecular or temporal domain;
3. measure and improve cost and memory without changing the mathematical
   action; and
4. qualify longer trajectories, weak-kick spectra, and selected magnetic
   responses on a bounded molecular set.

This guide replaces the proposed formalism-repository plan named
`aion_phase_two_development_plan.md` as the controlling Aion implementation
guide. That proposal remains useful background at
`/home/cgs/00_WORK/Projection_Full_Formalism/REVIEW/implementation`, but it
assumed that the exact-Wilson action already participated in Aion's ordinary
runner. It does not: the accepted implementation is presently a reusable
kernel and qualification layer called directly by campaign scripts.

The plan is intentionally mutable. A change to workload, thresholds, or low-
level implementation may be recorded at a gate without rewriting accepted
evidence. A change to an action, source definition, current, energy, kick map,
or propagation equation requires explicit mathematical reconciliation and
user review before implementation continues.

## 2. Governing scientific and software authority

Conflicts are resolved in this order:

1. the accepted mathematical definitions in Chapters 12--14 of the formalism
   book;
2. the accepted NQ0--NQ9 Chapter 13 evidence, whose entry point is
   `docs/chapter13_nq9_synthesis.md` and whose final review is
   `docs/reviews/chapter13_nq9_review_20260924.json`;
3. the exact-Wilson implementation and regression tests at the accepted
   parent revision;
4. Aion's reusable configuration, backend, artifact, observation, and
   workflow contracts;
5. this execution guide; and
6. campaign-local scripts, which are evidence generators rather than
   scientific or API authority.

The managed environment is governed by `AGENTS.md`,
`easybuild-project.toml`, and
`/home/cgs/01_TOOLS/EasyBuild/docs/AGENT_USAGE.md`. The project Conda prefix is
`/home/cgs/01_TOOLS/EasyBuild/conda/envs/aion`. Phase Two does not authorize a
dependency update or mutation of the shared toolchain.

## 3. Accepted starting point

The parent revision establishes the following bounded facts.

- The exact straight-Wilson finite molecular action is implemented for fixed
  nuclei, all-electron real-spherical Gaussian AO bases, closed-shell
  spin-summed states, fixed atom-centred anchors, and prescribed uniform
  electromagnetic sources.
- Coulomb-metric RI Hartree, pure LDA, and pure PBE grid actions have accepted
  energy, weak lower-matrix, and fixed-history source derivatives.
- Stationary solutions, weak action currents, Ward and continuity identities,
  nonlinear propagation, mechanical energy, source power, and integrated
  work have accepted evidence.
- The accepted exact action uses the fourth-order, two-node self-consistent
  Gauss--Magnus method with a diagonal Padé `[2/2]` link and contravariant-
  density congruence. It applies no Löwdin frame, endpoint Cholesky correction,
  metric projection, density repair, clipping, or drift subtraction.
- The exact action is the reference model. Strict C1 is a qualified reduced
  working action only in the sampled pure-LDA H3+/cc-pVDZ domain.
- The accepted molecular fixtures are H2, LiH, equilateral H3+, and CO,
  principally in STO-3G and cc-pVDZ. Accepted nonlinear trajectories extend
  only to 2.0 a.u. for LDA and 0.2 a.u. for PBE.

These are accepted results, not general production claims. Phase Two must not
modify their raw roots or retroactively enlarge their interpretation.

## 4. Reconciliation with Aion's current reusable API

### 4.1 Already reusable

The following source components are reusable and tested:

- exact Wilson AO density and source directions;
- exact magnetic one-electron matrices and temporal connection;
- RI--Wilson Hartree and Wilson LDA/PBE evaluators;
- exact stationary factories and solvers;
- exact dynamic samples and mechanical-power evaluation;
- weak current, charge, Ward, and continuity observations;
- exact and reduced action comparisons;
- fourth-order nonlinear contravariant-density propagation; and
- CPU and physical-CuPy kernels for the paths claimed by the accepted gates.

### 4.2 Not yet integrated

The following capabilities do not exist behind the ordinary Aion workflow:

- an exact-Wilson formulation or action variant in the strict configuration
  schema;
- exact stationary-state preparation as a simulation input or immutable
  artifact;
- an exact-Wilson source configuration covering `E(t)`, `B(t)`, and
  `dB(t)/dt` with the required affine potentials;
- a stateful one-step tensorial propagator used by `build_simulation()`;
- streamed exact-Wilson observables and diagnostics;
- exact-Wilson trajectory, checkpoint, restart, cancellation, and failure
  handling;
- a mathematically defined discontinuous kick for the exact action;
- action-derived Cartesian dipole and current streams consumable by the
  spectroscopy API; and
- end-to-end exact-Wilson CPU/GPU workflow parity.

The existing `propagate_nonlinear_contravariant_density()` is appropriate for
bounded qualification histories but stores endpoint contravariant and mixed
densities, both Gauss-node densities, and every link. For 34,000 intervals and
115 AOs, those five complex matrix histories alone require approximately
33.5 GiB. Long trajectories must therefore use a new stateful streaming
boundary; campaign code must not work around this by reducing diagnostics or
silently moving device arrays to the host.

## 5. Scientific terminology

Every result identifies one action and one numerical realization.

- **Exact Wilson** means the complete straight-path Wilson-dressed finite-
  basis action.
- **P0** means the accepted zeroth electromagnetic reduced action.
- **P0+E1** is the user-facing name for the accepted electric first-order
  level. Internal enums may retain `E1`, but documents and figures must not
  suggest an E1 term detached from its P0 parent.
- **Strict C1** means the accepted complete first-order electric and magnetic
  action without a positivity repair.
- **Density-resummed C1** remains a separately labelled diagnostic and may not
  be used to repair strict-C1 metric positivity.
- **Bare length gauge** and **bare velocity gauge** refer only to Aion's
  ordinary fixed-AO finite-basis formulations, not to the exact action.

Gauge representative, physical action, orbital basis, quadrature grid,
auxiliary space, retained-rank policy, and integration algorithm are distinct
parts of the scientific identity.

## 6. Included and excluded scope

### 6.1 Included

- finite fixed-nucleus all-electron molecules;
- closed-shell restricted pure-LDA and pure-GGA Kohn--Sham closures and the
  RI-Hartree branch already implemented;
- fixed occupations and atom-centred anchors;
- prescribed uniform electric sources;
- prescribed static uniform magnetic fields;
- time-dependent uniform magnetic fields with their Maxwell-required affine
  induction electric field;
- exact Wilson as the reference action;
- accepted P0, P0+E1, strict-C1, and density-resummed-C1 reduced actions only
  in their derived domains;
- stationary states, nonlinear real-time propagation, weak-kick spectra,
  action-derived currents, energy, power, and continuity;
- float64/complex128 CPU and physical-CuPy execution; and
- mathematically equivalent performance changes with explicit parity tests.

### 6.2 Excluded

- moving nuclei or anchors, forces, stress, and nuclear currents;
- pseudopotentials, effective core potentials, and nonlocal ionic terms;
- open-shell, spin-polarized, noncollinear, spin--orbit, or relativistic
  electronic structure;
- hybrids, range-separated functionals, meta-GGAs, orbital-dependent
  functionals, or nonlocal correlation;
- periodic systems, Bloch representations, magnetic translations, Ewald
  electrostatics, or a periodic velocity-gauge propagator;
- Maxwell backreaction;
- finite-wave-vector or generally spatially nonuniform propagated fields;
- current-density-functional closure or a claim about the exact interacting
  transverse current;
- reduced GGA P0/P0+E1/C1 actions before their action derivatives are
  derived; and
- screening, clipping, reduced precision, empirical correction, or metric
  repair introduced merely to make a calculation pass.

The Chapter 20 metric-safe holonomy problem remains separate.

## 7. Ammonia heritage evidence

The existing ammonia archive is retained at

```text
/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/
nh3_gauge_basis_validation
```

Its inventory is
`docs/nh3_calculation_inventory_2026-09-14.md` under that campaign. It
contains two optimized geometries, nine Casida datasets, 49 completed
trajectories, conventional and covariant pulse calculations, long bare
length/velocity kick spectra, one long P0+E1 length-gauge kick, and useful
runtime measurements.

Phase Two may use these data as prior information for:

- geometry and orientation;
- basis and auxiliary-basis availability;
- expected bright-state energies and polarization directions;
- conventional finite-basis RT-TDDFT comparisons;
- source and transform design; and
- preliminary cost estimates for the older formulations.

They are not exact-Wilson evidence and may not be copied into a Phase Two
acceptance table as if they were generated by the accepted parent action. The
Phase Two heritage intake will hash the selected original artifacts and state
their historical formulation, code path, and limitations.

The two stored geometries must not be mixed in a basis-convergence claim:

- `G_DZ`, optimized at PBE/cc-pVDZ, was used for the cc-pVXZ ladder and
  fixed-geometry aug-cc-pVDZ calculations;
- `G_augTZ`, optimized at PBE/aug-cc-pVTZ, was used for the aug-cc-pVTZ
  calculations.

The first exact-Wilson bridge will use fixed `G_DZ`, because that maximizes
direct comparison with the existing cc-pVXZ Casida and real-time data. A
later physical ammonia study may choose `G_augTZ`, but must recalculate every
basis result used in a convergence sequence at that same geometry.

## 8. Evidence and review discipline

Every gate distinguishes:

1. mathematical identity;
2. implemented feature;
3. executed numerical test;
4. analyzed evidence; and
5. user acceptance.

Passing tests do not accept a scientific interpretation. Every gate ends with
a Markdown entry point, machine-readable result, provenance and completion
records, retained failures, hashes, and a separate review record.

Every numerical result records at least:

- Aion commit, branch, parent, and worktree state;
- environment lock and relevant Python, PySCF, libxc, NumPy, SciPy, CuPy,
  CUDA, and device identities;
- complete molecule, geometry, charge, occupations, basis, anchors, and
  origin;
- functional, quadrature and pruning, auxiliary basis, rank policy, and
  solver tolerances;
- source potentials and complete analytic time history;
- action and gauge representative;
- timestep, nonlinear policy, Padé order, duration, backend, and restart
  lineage;
- all observable definitions and schedules;
- raw artifacts, generation command, analyzer revision, and SHA-256 hashes;
  and
- visible negative, interrupted, superseded, or failed results.

Campaigns live below

```text
/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/aion_phase_two/
```

and never overwrite NQ0--NQ9 or the historical ammonia roots.

## 9. Target reusable architecture

### 9.1 Action and configuration

The configuration layer will add discriminated exact-Wilson simulation
variants without weakening the existing strict parser. The exact variant must
identify:

- Hartree, pure LDA, or pure GGA closure;
- auxiliary basis and retained-rank policy;
- exact straight-Wilson path rule and anchors;
- stationary-state source and convergence policy;
- fourth-order tensorial Gauss--Magnus integration policy;
- electromagnetic source; and
- observable, checkpoint, and output schedules.

Reduced Wilson actions will use a separate discriminant or explicit level;
they will not be represented as runtime flags on the exact action.

Schema evolution must be explicit. Existing `1.x` artifacts remain readable;
new required state or source fields require an independently versioned schema
or a compatible optional extension, never an in-place reinterpretation.

### 9.2 Source model

The reusable exact source exposes physical `E_origin(t)`, `B(t)`, and
`dB(t)/dt`, together with the affine vector/scalar potentials selected by a
gauge representative. Source values and derivatives are evaluated
analytically at endpoints and Gauss nodes. A source cache is bounded and may
reuse identical spatial fields, but it may not grow once per node over a long
time-dependent-magnetic trajectory.

### 9.3 Stationary state

An exact simulation begins from an authenticated stationary state belonging
to the same action, source, metric, functional, basis, grid, and auxiliary
space. The state artifact stores the contravariant density, occupations,
optional occupied coefficient representative, metric, energy components,
residuals, and fingerprints needed to reconstruct it without rerunning the
solve.

### 9.4 Stateful propagator

The production propagator owns one accepted contravariant density and the
minimal scratch required for a two-node nonlinear step. A successful step
returns:

- the next contravariant density;
- optional mixed density and coefficient representative needed by observers;
- accepted Gauss-node diagnostics;
- the link and cross-metric residual;
- nonlinear iteration count and residual; and
- exact source samples needed for interval-centered observations.

Histories are streamed through observers. Debug retention of nodes or links is
an explicit bounded schedule, not the default. The algorithm remains the
accepted uncorrected fourth-order method.

### 9.5 Observables

The integrated exact formulation owns its observables. At minimum it exposes:

- metric particle number and occupation-spectrum diagnostics;
- electronic and total molecular dipole;
- three Cartesian components of the action-derived uniform source current;
- arbitrary weak-current pairings as expert diagnostics;
- complete mechanical-energy components;
- instantaneous matrix-rate and source-current power routes;
- accumulated source work;
- Ward and weak-continuity residuals; and
- metric, Hermiticity, nonlinear, and cross-metric diagnostics.

Ambient momentum and currents borrowed from bare formulations remain
separately named diagnostics and cannot replace the action current.

### 9.6 Persistence and restart

The ordinary transactional runner will stream exact trajectories and publish
controlled failure artifacts. Checkpoints store the accepted density, time
index, accumulated work, source/event state, observer state, and scientific
identity. Resume reconstructs the simulation and must reproduce an
uninterrupted trajectory at the measured numerical floor.

### 9.7 Spectroscopy

The existing Casida and transform machinery is reused only after exact
trajectory definitions are supported. Field-free Casida is a valid reference
for zero-magnetic-field exact-Wilson dynamics because the spatial Wilson
frame reduces to the ordinary field-free realization. It is not a magnetic-
field Casida oracle.

Dipole- and current-domain transforms retain the finite-record endpoint term.
Intrinsic resolution `2*pi/T`, native frequency spacing, damping, window, and
zero padding remain separately reported.

## 10. Work packages and gates

### P2-0. Reconcile and freeze the executable baseline

Purpose: establish a reproducible parent and a bounded first implementation
matrix.

Required work:

1. create `feature/aion-phase-two` from accepted commit `dc6d4e2`;
2. authenticate the NQ9 review, synthesis, and manifest;
3. record the managed environment and current CPU/GPU launch boundaries;
4. run current format, lint, type, fast, integration, and physical-GPU gates;
5. reconcile `0.2.0.dev7`, the completed refactor WP7 evidence, and the exact
   Chapter 13 branch;
6. create a hashed heritage manifest for selected ammonia geometry, Casida,
   pulse, kick, spectrum, and timing artifacts;
7. define Phase Two result, provenance, completion, failure, and review
   schemas; and
8. record a small baseline timing and peak-memory table for accepted H3+ LDA
   and CO PBE operations.

The first matrix is frozen only through P2-2. Later basis, duration, and
application matrices are versioned decisions informed by profiling; they are
not guessed at P2-0.

Pass condition: the accepted parent is reproduced, selected heritage files
are authenticated without being relabelled, and the first integration matrix
contains no undocumented default.

### P2-1. Integrate the exact action into reusable Aion workflows

Purpose: remove campaign scripts as the only way to run exact nonlinear
dynamics.

Required work:

1. add exact and reduced Wilson configuration types and strict TOML support;
2. add analytic affine electromagnetic source providers and node sampling;
3. add exact stationary-state preparation and immutable state I/O;
4. refactor the accepted contravariant-density algorithm into a stateful
   one-step propagator without changing its equations;
5. connect exact construction to `build_simulation()` and the ordinary runner;
6. add exact formulation-owned observable calculators;
7. add streaming trajectory, checkpoint, failure, cancellation, and restart
   support;
8. expose the new path through the CLI and documented expert Python API; and
9. retain the NQ6/NQ8 direct drivers as independent comparison oracles.

Required tests:

- algebraic one-step parity with the accepted batch propagator;
- multi-step H3+ trajectory parity with NQ6;
- exact stationary-state serialization round trip;
- uninterrupted versus resumed trajectory parity;
- strict rejection of unsupported actions and sources;
- bounded-memory behavior independent of trajectory length; and
- CPU/physical-GPU parity and device-residency checks.

Pass condition: a fresh reusable Aion input can prepare, run, checkpoint,
resume, observe, and load a short exact-Wilson trajectory that agrees with the
accepted direct implementation at its measured floor.

### P2-2. Minimal molecular transfer and cost bridge

Purpose: test the integrated runtime before long propagation or a large basis.

The linked bridge is:

1. H3+/cc-pVDZ/LDA reproduction of selected accepted stationary and dynamic
   measurements;
2. CO/cc-pVDZ/PBE reproduction of a selected accepted short transfer result;
3. NH3 `G_DZ`/cc-pVDZ/PBE, first at `B=0`, using the accepted exact action;
4. a short smooth electric pulse along the NH3 C3 axis; and
5. CPU followed by physical-GPU parity.

NH3 records stationary residuals, density normalization, dipole, Cartesian
action current, energy components, power, work, Ward, weak continuity,
metric diagnostics, time per nonlinear evaluation, and peak memory. Its
zero-field result is compared with the matching ordinary bare-length result
as a reduction and integration check, not as evidence that two different
finite-field actions are interchangeable.

Grid, auxiliary-space, and stationary-tolerance refinements use a linked
rather than Cartesian matrix. The accepted `weigend` auxiliary basis is the
first exact reference; the historical `cc-pvdz-jkfit` result may be added as
an explicit auxiliary-space comparison.

Pass condition: the integrated path reproduces accepted fixtures, NH3
retains every identity at a declared floor, and measured cost determines the
largest safe next basis.

### P2-3. Performance and memory qualification

Purpose: make longer runs possible without adding an unnamed approximation.

Profile separately:

- Wilson AO and AO-pair construction;
- RI Hartree metric and three-centre contractions;
- LDA/PBE grid evaluation;
- exact one-electron spatial samples;
- temporal connection and source derivatives;
- nonlinear Gauss-node iterations;
- dense linear algebra and Padé links;
- observable schedules;
- host/device transfers; and
- HDF5 streaming and checkpoints.

Permitted transformations include blocking, batching, bounded cache reuse,
preallocated scratch, source-sample reuse, persistent device arrays, avoiding
repeated construction, and mathematically identical contraction reordering.

Every optimization is compared with the unchanged CPU reference for energies,
lower matrices, source derivatives, states, current, and power as applicable.
Affected Ward, continuity, work, restart, gauge, frame, and CPU/GPU tests are
rerun.

Pass condition: speed or memory improvements are measured, the scientific
identity remains explicit, and no tolerance or physics test is weakened to
accept the changed path.

### P2-4. Exact kick and long-time propagation

Purpose: establish the event and streaming foundations required by spectra.

Before implementation, derive from the exact action:

- the distributional electric source across the event;
- the finite coefficient or contravariant-density boundary map;
- the metric and connection values on both sides;
- the pre/post action, energy, dipole, and current semantics; and
- gauge and electromagnetic-origin covariance of the event.

The event is qualified by comparison with a sequence of narrowing smooth
pulses at fixed impulse. Agreement must be demonstrated over a regime where
both pulse discretization and finite-width error are resolved.

Long-time cases then include:

1. an undriven stationary state;
2. a weak zero-magnetic-field electric kick or pulse;
3. electric driving in a static uniform magnetic field; and
4. a time-dependent uniform magnetic field with its induction electric field.

At least three timesteps are used for each new order claim. Durations are
chosen from the physical resolution required by P2-5. Restart boundaries,
observable schedules, and checkpoint cadence are refined independently.

Pass condition: the event is action consistent, memory is bounded with
duration, uninterrupted and resumed trajectories agree, expected fourth-
order behavior is visible, and remaining drift is identified and bounded
without repair.

### P2-5. NH3 field-free weak-kick spectroscopy

Purpose: qualify an exact-action spectrum against matching field-free linear
response and historical RT-TDDFT data.

The first spectrum uses NH3 `G_DZ`, PBE, cc-pVDZ, and the C3-axis bright
direction. The historical RI-J Casida root near 6.8366 eV is prior
information, not the new reference; Phase Two creates a fresh authenticated
Casida artifact from the exact prepared reference.

Required refinements:

- at least three kick amplitudes spanning a resolved linear-response window;
- at least three timesteps for the selected propagation comparison;
- at least three effective durations or a justified linked duration study;
- separate grid and auxiliary-space comparisons;
- explicit damping and transform-window variation; and
- a basis extension only after the P2-3 cost gate.

The response is reconstructed independently from the molecular dipole and
the action-derived Cartesian current. The current transform retains its
finite-record endpoint term. Residual DC or persistent current is reported
before any spectral interpretation.

Casida comparisons cover peak position, polarization, and integrated or
fitted strength within the combined duration, damping, grid, basis,
auxiliary, and solver uncertainty. A continuum TRK value is not imposed as an
exact finite-basis identity.

Pass condition: the linear window and resolution are established, dipole and
current routes agree at their measured floor, and Casida differences have a
separated error budget.

### P2-6. Magnetic and reduced-action transfer

Purpose: determine which exact magnetic responses are practical and which
reduced-action conclusions transfer beyond H3+.

The exact magnetic branch first uses the smallest system that resolves the
target effect. NH3 is not automatically selected for a time-dependent-
magnetic production run. Cost, signal, and memory must justify it.

Reduced comparisons are restricted to pure LDA and include exact Wilson, P0,
P0+E1, strict C1, and density-resummed C1. They include one added multicentre
geometry and compact, polarized, and diffuse basis checks chosen after P2-3.

Every scan records exact and reduced metric spectra, conditioning,
stationary residuals, source and power differences, state differences, and
trajectory error. Metric-domain failure, solver failure, and large but
admissible model error are separate outcomes.

Pass condition: the transferred domain and failures are molecule- and basis-
conditioned, and numerical error is separated from electromagnetic model
error.

### P2-7. Select a first physical application

Purpose: propose, but not yet execute, one application using evidence from
P2-2 through P2-6.

Candidate classes include a broader field-free optical spectrum, a magnetic-
field-dependent optical response, and a magnetic-induction energy-uptake
problem. Ammonia may be a candidate, but its role as a qualification fixture
does not select it as the application.

The protocol states molecule, geometry, action, functional, basis, auxiliary
space, source, field range, time/frequency window, observables, refinements,
cost, stopping conditions, and the bounded conclusion that could follow.

Expensive application execution requires a separate user review. No
production choice is made by this plan.

### P2-8. Execute and qualify the selected application

Purpose: obtain one bounded physical result after the method and application
protocol are accepted.

Pilot calculations may bracket signal and cost but do not replace declared
refinements. The final evidence separates numerical uncertainty, orbital-
basis dependence, auxiliary-space error, functional dependence, and
electromagnetic-model error. Exact and reduced actions are compared only
where P2-6 admits the reduction.

Pass condition: numerical evidence and physical interpretation receive
separate reviews, and failed or negative results remain visible.

### P2-9. Synthesis and handoff

Required outputs:

- an authenticated gate, result, limitation, and review table;
- an equation-to-code-to-test-to-evidence map;
- a numerical-limits table containing the controlling floors;
- a performance and memory table tied to exact revisions and inputs;
- a domain table over molecule, basis, grid, auxiliary space, functional,
  source, field, duration, and backend;
- a separate application report, if P2-8 was authorized and completed;
- remaining mathematical, implementation, and application obligations; and
- a branch or release recommendation that does not imply a merge or push.

Pass condition: the synthesis is reconstructible from immutable artifacts and
the user has reviewed its bounded closure statement.

## 11. Initial linked matrix

Only the P2-0 through P2-2 matrix is fixed initially.

| Role | System | Geometry | Basis | Closure | Source | Backend |
|---|---|---|---|---|---|---|
| accepted algebraic/dynamic anchor | H3+ | accepted equilateral | cc-pVDZ | LDA | selected NQ6 source | CPU, then GPU |
| accepted GGA transfer anchor | CO | accepted NQ8 | cc-pVDZ | PBE | selected NQ8 short source | CPU, then GPU |
| polyatomic integration bridge | NH3 | fixed `G_DZ` | cc-pVDZ | PBE | zero field and short axial smooth electric pulse | CPU, then GPU |

No aug-cc-pVTZ exact trajectory is authorized by this table. Polarized,
diffuse, grid, and auxiliary extensions are selected after the first NH3
timing and memory measurements.

## 12. Execution and resource rules

- This host has eight physical CPU cores. Molecular CPU calculations run
  serially with at most eight numerical-library threads.
- Concurrent CPU and GPU processes receive disjoint CPU affinity and thread
  limits; two multi-threaded Python calculations may not oversubscribe the
  host.
- Physical-GPU tests use `tools/gpu-python`, fail on fallback, and request the
  sandbox permission required to see the device whenever a GPU gate is due.
- Long calculations use detached, status-bearing launchers that perform no
  polling. Status is queried only on request.
- A timing pilot and guarded ETA precede every materially longer matrix.
- Peak host and device memory are measured before a basis or duration is
  enlarged.

## 13. Stop conditions

The affected gate stops with the smallest reproducer when:

- mathematical sources disagree on sign, index orientation, charge,
  connection, kick, current, energy, or source derivative;
- the reusable runtime does not reproduce the accepted direct driver;
- a proposed optimization changes a physics quantity beyond its established
  floor;
- a GPU path falls back, transfers an undeclared physics array to the host,
  or loses parity;
- metric positivity is lost;
- a stationary or nonlinear solve misses its declared residual;
- a long trajectory requires projection, clipping, repair, or empirical
  drift removal;
- memory grows with duration contrary to the streaming contract;
- a spectrum changes materially under an unresolved amplitude, timestep,
  duration, grid, auxiliary, damping, or window refinement;
- the requested application leaves the included theory domain; or
- accepted immutable evidence would need to be overwritten.

A stopped gate bounds the result; it does not erase passing evidence from
earlier gates.

## 14. Dependency order

```text
P2-0 baseline and heritage
  -> P2-1 reusable exact runtime
       -> P2-2 H3+/CO reproduction and NH3 bridge
            -> P2-3 performance and memory
                 -> P2-4 exact kick and long-time propagation
                      -> P2-5 NH3 spectroscopy
                 -> P2-6 magnetic and reduced-action transfer
P2-5 + relevant P2-6 evidence
  -> P2-7 application protocol and user decision
       -> P2-8 optional application execution
            -> P2-9 synthesis
```

P2-6 preparation may overlap P2-4 after P2-3 fixes reference comparisons.
Application selection remains gated even if method qualification is already
complete.

## 15. Current execution state and next work

P2-0 is accepted in
`docs/reviews/phase_two_p2_0_review_20260925.json`. P2-1 is accepted in
`docs/reviews/phase_two_p2_1_review_20260925.json` after completing its
implementation, full software gates, authenticated H3+/LDA NQ4/NQ6
reproduction, driven direct-oracle parity, deterministic restart, and
physical-GPU parity/residency campaign. Its review entry point is
`docs/phase_two_p2_1_qualification.md`.

P2-2 is accepted in
`docs/reviews/phase_two_p2_2_review_20260927.json` after completing its bounded
CO/cc-pVDZ/PBE reproduction, fixed `G_DZ` NH3/cc-pVDZ/PBE exact--bare bridge,
CPU/physical-GPU parity, and one predeclared GPU-only timestep halving. Its
review entry point is `docs/phase_two_p2_2_qualification.md`.

The active gate is P2-3 performance and memory qualification. It begins from
the measured exact-runtime cost breakdown and profiles Wilson AO/AO-pair
construction, RI Hartree, PBE grid evaluation, one-electron samples,
temporal/source work, nonlinear propagation, observations, host/device
transfers, and HDF5 separately. No optimization may change the action or
weaken an accepted scientific tolerance.

This update does not select a later production application, expand the P2-2
basis/source matrix, or authorize a long trajectory.

## 16. Change record

### 25 September 2026

- Replaced the proposal's assumption of an already integrated exact runtime
  with an explicit reusable-runtime work package.
- Added stateful streaming, exact stationary artifacts, checkpoint/restart,
  and end-to-end CPU/GPU workflow requirements.
- Added a mathematical and numerical gate for an exact discontinuous kick.
- Converted the full up-front campaign matrix into a bounded initial matrix
  followed by measured, versioned decisions.
- Classified the existing ammonia calculations as authenticated heritage
  rather than exact-Wilson evidence.
- Selected fixed `G_DZ` NH3/cc-pVDZ/PBE as the first polyatomic bridge while
  deferring larger bases until profiling.
- Separated field-free spectroscopy from magnetic and reduced-action transfer.
- Deferred physical-application selection and execution to explicit later
  user gates.
- Completed the P2-0 baseline and recorded its user acceptance.
- Implemented P2-1 through increments A--E and linked its executed,
  H3+/LDA qualification entry point.
- Recorded the user's acceptance of P2-1 and opened bounded P2-2 execution.
- Fixed and implemented the stagewise P2-2 CO/NH3 execution matrix, including
  explicit reference-grid pruning and the continuous-density source-quench
  boundary required by the accepted NQ8 CO initial condition.

### 27 September 2026

- Completed and authenticated the P2-2 CO/NH3 base campaign.
- Corrected a semantic grid-fingerprint comparison after verifying bitwise
  equality of the numerical grid arrays.
- Resolved the sole remaining NH3 current check with the predeclared GPU-only
  timestep halving; its 0.23693 residual ratio identifies the second-order
  bare propagator's temporal error.
- Linked the executed, pending-review P2-2 qualification entry point and
  bounded the next work to P2-3 profiling if the user accepts the gate.
- Recorded the user's acceptance of P2-2 and opened bounded P2-3 performance
  and memory qualification.
