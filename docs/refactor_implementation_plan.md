# Aion 0.2 reusable real-time TDDFT refactor

Status: WP0 and WP1 complete; numerical implementation has not started

Last design review: 2026-09-08

Companion theory note: [theory_and_implementation.tex](theory_and_implementation.tex)

## 1. Purpose

This document is the mutable implementation plan for turning the present Aion
research draft into a reusable real-time TDDFT library. It translates the
agreed physics and numerical contracts into independently reviewable work
packages, tests, milestones, and release gates.

The plan is intentionally not a production-science campaign plan. Molecule
selection beyond the validation fixtures, ammonia/PNA studies, EELS
production work, and periodic velocity-gauge work will be decided separately.
The plan may change as implementation evidence accumulates, provided every
change records its reason and preserves or explicitly revises an agreed
physics contract.

No backward compatibility with the draft API is required.

## 2. Outcomes and boundaries

### 2.1 Required outcomes

The 0.2 design shall provide:

- one typed Python API for preparing a common electronic reference and
  constructing bare length-gauge, bare velocity-gauge, P0, and P0+E1
  simulations;
- formulation-owned metric, equation-of-motion Hamiltonian, connection,
  dipole, primary current, mechanical-current diagnostics, energy ledger, and
  source-work definitions;
- fixed-metric and connection-aware self-consistent exponential-midpoint
  propagators sharing a nonlinear solver;
- potential-first electromagnetic sources, exact discrete kicks, compiled
  endpoint/midpoint source samples, and exact pulse-boundary handling;
- versioned HDF5 references, trajectories, source histories, and restart
  checkpoints;
- reusable Casida and real-time kick-spectrum routines;
- CPU and GPU implementations with no silent GPU-to-CPU fallback;
- validation based primarily on algebraic identities, conservation/Ward
  relations, convergence order, restart equivalence, and CPU/GPU parity.

### 2.2 First-release domain

The validated 0.2 domain is deliberately narrow:

- finite, all-electron molecules;
- fixed nuclei;
- restricted spin-summed RKS;
- local nuclear potentials;
- pure LDA and GGA functionals;
- prescribed uniform electric fields in the dipole approximation;
- fixed time steps;
- float64 and complex128 arithmetic;
- CPU and GPU backends.

Low-level contractions may accept arbitrary shape-compatible Hermitian AO
density matrices and arbitrary occupations, but this does not extend the
validated high-level propagation domain.

### 2.3 Explicit non-goals

The following are not to be approximated or silently accepted:

- pseudopotentials, ECPs, and other nonlocal ionic operators;
- hybrid-functional and exact-exchange current corrections;
- unrestricted, noncollinear, or relativistic spin dynamics;
- moving nuclei and the associated nuclear-velocity connection;
- periodic boundary conditions;
- Maxwell backreaction;
- adaptive or nonuniform time stepping;
- production campaign choices for ammonia, PNA, CO, or EELS.

The interfaces should leave clean extension seams for spatially varying
sources, aloof relativistic projectiles, EELS, and possible future periodic
velocity-gauge work. Those seams must not slow or complicate the uniform-field
hot path.

## 3. Governing contracts

### 3.1 State and AO convention

The canonical AO density is

\[
P=C f C^\dagger,\qquad
N=\operatorname{Re}\operatorname{Tr}(PS),\qquad
\langle O\rangle=\operatorname{Re}\operatorname{Tr}(PO).
\]

The occupation vector or matrix contains the complete spin multiplicity.
There is no hidden factor of two. The public type is AODensity; densities in
other index conventions require distinct types and explicit conversions.

The first production state contains occupied coefficients and occupations.
The state protocol must permit a later density-matrix state without changing
formulation, observable, or storage contracts.

### 3.2 Formulation contract

At each required time and state, every formulation exposes the explicit
triple

\[
(S, H_{\mathrm{eom}},\omega),\qquad
i\hbar(S\dot C+\omega C)=H_{\mathrm{eom}}C .
\]

The formulation also owns:

- its primary dipole and primary/source current;
- source-work rate;
- complete energy components available for that formulation;
- optional ambient-projected mechanical-current diagnostics;
- exact maps for supported discrete events;
- support validation before propagation;
- the operator and source dependencies needed by each observable.

For P0+E1, the E1 variational potential occurs only in the connection,
\(\omega=\omega_{\mathrm{P0}}+(i/\hbar)V_{\mathrm{E1}}\), and never
simultaneously in \(H_{\mathrm{eom}}\).

### 3.3 Source contract

Physical sources and gauge representations are different objects. A compiled
P0 source sample contains node scalar potentials and once-oriented pair
links:

\[
\{\phi_a(t),L_{ab}(t),\dot L_{ab}(t)\},\qquad
L_{ab}=\int_{R_b}^{R_a}\mathbf a\cdot d\boldsymbol\ell ,
\quad a<b.
\]

The pair electromotive potential is

\[
V_{ab}=-\dot L_{ab}-\phi_a+\phi_b .
\]

Positive \(I_{ab}\) is charge flow from node \(a\) to node \(b\);
\(L_{ab}\) and \(V_{ab}\) are oriented from \(b\) to \(a\). These
conventions must be centralized and tested.

Sources are potential-first, analytically evaluable at arbitrary endpoint and
midpoint times, and compiled before a fixed-step run. Multiple providers are
summed before formulation-dependent projection. Temporal envelopes operate on
potentials, including the product-rule contribution to time derivatives.
Finite pulses are ordinary source objects; their use in a gauge comparison,
spectroscopy calculation, or another simulation is a separate workflow choice.
A workflow may request a zero-impulse constraint, but the pulse abstraction
does not impose one.

### 3.4 Units, origins, and time

- Internal numerical APIs use atomic units.
- Every stored dataset declares its unit and physical dimension.
- The reduced vector potential \(\mathbf a=\mathbf A/c\) is named
  vector_potential_reduced.
- Input coordinates are preserved; no implicit recentering is allowed.
- Electronic, nuclear, and total dipoles use one explicit EM origin, defaulting
  to the input coordinate origin.
- Integer step index is authoritative:
  \(t_n=t_0+n\Delta t\), \(n=0,\ldots,N\).
- \(N\) propagation intervals produce \(N+1\) endpoint states.

### 3.5 CPU/GPU contract

The selected backend remains authoritative from reference construction
through formulations, propagation, observables, currents, and energies.
Conversion to host arrays occurs only at the storage/export boundary. A
requested GPU run must fail if the GPU path is unavailable; it must never
fall back to CPU.

Every milestone that adds or changes executable numerical behavior includes
GPU tests using the physical GPU. Because the sandbox hides the device,
agents must request elevated execution immediately when those tests are due.
A sandbox-only CPU result is not a substitute.

## 4. Proposed package architecture

The public package is organized by physics responsibility with one-way
dependencies:

| Package | Responsibility |
| --- | --- |
| aion.config | Immutable typed configuration, strict validation, normalized serialization, semantic hashes |
| aion.backends | Array/backend protocol, CPU and GPU implementations, residency checks, workspaces |
| aion.electronic_structure | PySCF RKS preparation, grids, core/derived operator bundles, reference artifacts |
| aion.electromagnetism | Physical sources, gauge representations, envelopes, compiled samples, kicks |
| aion.formulations | Bare LG, bare VG, P0, P0+E1, formulation protocols and support checks |
| aion.propagation | State protocols, common SCEM nonlinear solve, fixed-metric and connection-aware transport |
| aion.observables | Typed calculators for dipoles, currents, energy ledger, Ward and continuity diagnostics |
| aion.io | HDF5 schemas, transactional writers, checkpoints, restart lineage, CSV export |
| aion.workflows | Reference preparation, simulation builder, run/resume, comparisons, spectra |

Dependencies flow downward from workflows. I/O receives records but does not
enter physics. Observables inspect states and formulations but do not control
propagation. Mutable caches and scratch arrays live only in an explicit,
backend-specific Workspace. References, configurations, sources, operator
bundles, and formulations are immutable.

The small top-level API is planned as:

- prepare_reference
- load_reference
- build_simulation
- run
- resume
- load_trajectory

Expert components remain available from their named subpackages. Incidental
module visibility does not define the public API.

A thin aion command provides prepare, run, resume, inspect, and export. The CLI
uses exactly the typed API and materializes the resolved configuration before
expensive work.

## 5. Static and dynamic object model

### 5.1 PreparedReference

A prepared reference is immutable, versioned, and backend-neutral on disk. It
contains:

- normalized molecular geometry, charge, spin restriction, basis, functional,
  grid, and SCF settings;
- converged ground-state coefficients, occupations, density, and reference
  energy information;
- nuclear positions, charges, repulsion energy, EM origin, and atom anchors;
- the core operator bundle: overlap, kinetic, nuclear attraction, position,
  canonical momentum, and occupations;
- exact DFT quadrature-grid coordinates and weights;
- fingerprints of input, dependencies, numerical construction, and operators.

It does not pickle PySCF objects or GPU runtime objects and does not store AO
values on the grid. Loading reconstructs the live backend/PySCF model, verifies
fingerprints, and constructs device caches.

Derived operator bundles are independently versioned and fingerprinted:

- anchor topology and atomic projectors;
- P0 Wilson/dressing structures;
- P0+E1 central-dipole tensors and derivatives.

### 5.2 Simulation

A simulation binds one prepared reference to:

- one physical source and one derived gauge representation;
- one formulation;
- one state;
- one fixed time grid;
- one propagator and nonlinear-solver configuration;
- typed observable calculators and independent schedules;
- one backend workspace;
- storage, checkpoint, and provenance policies.

Separate simulations may share a prepared reference but never state or
workspace.

## 6. Physics implementation map

### 6.1 Bare length gauge

Use a fixed overlap and

\[
H_{\mathrm{LG}}[P,t]=H_0[P]-q\,\mathbf E(t)\cdot
(\mathbf r-\mathbf O S).
\]

The primary current is the analytic derivative of the formulation dipole.
The ambient mechanical diagnostic is paramagnetic only:

\[
\mathbf J_{\mathrm{para}}=\frac q m\operatorname{ReTr}(P\mathbf p),
\qquad \mathbf J_{\mathrm{dia}}=0.
\]

Finite-basis disagreement between these two current definitions is a
diagnostic, not an internal inconsistency.

### 6.2 Bare velocity gauge

For the validated local all-electron stack,

\[
H_{\mathrm{VG}}[P,t]=H_0[P]
-\frac q m\mathbf a\cdot\mathbf p
+\frac{q^2}{2m}|\mathbf a|^2S .
\]

The primary mechanical current is split and stored:

\[
\mathbf J_{\mathrm{para}}=\frac q m\operatorname{ReTr}(P\mathbf p),
\quad
\mathbf J_{\mathrm{dia}}=-\frac{q^2}{m}N\mathbf a,
\quad
\mathbf J_{\mathrm{mech}}=\mathbf J_{\mathrm{para}}+
\mathbf J_{\mathrm{dia}}.
\]

Momentum uses the documented PySCF derivative-integral sign convention.
ECPs/nonlocal potentials and hybrids are rejected before the run.

### 6.3 P0 and P0+E1

Build Wilson phases, the time-dependent overlap, the projected connection,
inverse-dressed density, P0 Hamiltonian, central dipoles, and E1 derivatives
as independent reusable components. P0 is a complete first-class
formulation; E1 is a composable correction.

The P0+E1 implementation exposes:

- site charges and exact graph pair currents;
- the P0 graph-current vector;
- intrinsic-polarization and phase-response pieces of the E1 residual current;
- the total variational source current;
- covariant dipole and analytic dipole derivative;
- continuity, current-dipole, source-power, and energy-Ward residuals;
- the separately named ambient_projected_mechanical_current.

The ambient mechanical contraction accepts arbitrary precomputed local AO
vector-potential matrices at low level. The first high-level builder supports
only the uniform specialization. In the covariant uniform formulation,
canonical and diamagnetic terms must cancel their explicit reduced-vector-
potential pieces, leaving the dressed field-free momentum contraction.

### 6.4 Energy ledger

Energy is an independently scheduled observable and is not evaluated every
step by default. The ledger uses qualified names and never exposes an
ambiguous total_energy.

Required components include:

- canonical/paramagnetic, linear-vector-potential, diamagnetic, and mechanical
  kinetic terms where applicable;
- electron-nuclear, Hartree, exchange-correlation, and nuclear-repulsion
  terms;
- electronic, fixed-nuclear, and combined scalar EM couplings;
- energy_matter_total and energy_generator_total;
- absorbed energy relative to the initial matter energy;
- independently accumulated source work;
- analytic power/Ward derivatives and residuals.

Work is accumulated on every accepted interval from the converged midpoint
state and source with midpoint quadrature, without another Fock build. It is
persisted according to its own schedule.

## 7. Propagation design

### 7.1 Shared nonlinear SCEM engine

Both production propagators use one self-consistent midpoint iteration. Its
convergence test is the metric-invariant density residual

\[
r_P=
\frac{\|S_{1/2}^{1/2}(P_{\mathrm{out}}-P_{\mathrm{in}})
S_{1/2}^{1/2}\|_F}{\max(N_e,1)} .
\]

Initial defaults are tolerance \(10^{-10}\), at most 50 iterations, and
minimum mixing 0.1. Use residual-monitored damped Picard iteration on the
Hermitian midpoint density: try a full update and reduce mixing when the
residual worsens. Record iteration count, density residual, and an independent
Hamiltonian residual. Failure is hard: never accept a nonconverged step and
never silently change the time step.

### 7.2 Fixed-metric SCEM

The fixed-metric transport uses the direct \(S\)-skew generator and a
rational exponential map. It does not require a time-dependent Löwdin frame.

### 7.3 Connection-aware SCEM

The analytically known diagonal site connection is integrated exactly through
a site-parallel frame. The remaining projected connection and intrinsic
Hamiltonian are pulled into that frame and integrated at the converged
midpoint.

The production rational map is Padé [2/2],

\[
R_{22}(Z)=
(I-Z/2+Z^2/12)^{-1}(I+Z/2+Z^2/12),
\]

with Cayley [1/1] retained as a diagnostic. The whole midpoint method remains
globally second order; increasing only the Padé order does not increase the
global order.

A right-Cholesky cross-metric correction is mandatory:

\[
M=U_{\mathrm{raw}}^\dagger S_{n+1}U_{\mathrm{raw}}
=R_M^\dagger R_M,quad
S_n=R_S^\dagger R_S,quad
Q=R_M^{-1}R_S,quad U=U_{\mathrm{raw}}Q.
\]

Record raw and corrected metric defects and \(\|Q-I\|\). This is a
constraint correction, not evolving-basis Löwdin orthogonalization.

## 8. Storage and provenance

### 8.1 Artifacts

- reference.h5: immutable prepared reference;
- trajectory.h5: canonical versioned trajectory;
- em_source.h5: optional full-resolution/per-provider source history;
- checkpoint_<step>.h5: immutable restart checkpoints;
- status.json: small detached-run monitor.

HDF5 top-level groups are:

- /meta
- /configuration
- /reference
- /time
- /source
- /observables
- /diagnostics
- /events
- /restart

Definition-specific observable paths are mandatory; generic /current,
/energy, and /dipole paths are forbidden. Endpoint and interval-centered
streams each carry their own step and time coordinates. Scalar streams are
uncompressed. Dense arrays and checkpoints are record-chunked with low-level
gzip and shuffle.

### 8.2 Transactional publication

Writers create a .partial artifact, validate and flush it, set an internal
complete flag, and atomically rename it. Completed artifacts are immutable.
Controlled failures may publish trajectory.failed.h5 with complete=false;
hard crashes leave only the partial artifact.

### 8.3 Restart

Restart rebuilds and authenticates the prepared reference, restores state,
occupations, global step, accumulated work, nonlinear continuation data when
needed, and applied event identifiers. It first reconstructs and writes the
shared boundary state before advancing. Child artifacts store parent run ID,
checkpoint hash, and global step offset. Stitching deduplicates only an
authenticated shared boundary.

### 8.4 Identity and reproducibility

The scientific simulation ID is a SHA-256 semantic hash of canonical,
lossless normalized configuration and referenced artifacts. It excludes
paths, timestamps, host names, and output schedules. A unique run ID
identifies each execution attempt. Full software, repository, dependency,
backend, GPU, and hardware provenance is stored. Dirty production campaigns
are rejected by default.

## 9. Spectroscopy

Spectra are postprocessing results, not propagation state. Reusable routines
shall return structured numerical data for:

- Casida roots, oscillator strengths, and transition directions;
- kick-derived polarizability
  \(\alpha_{\alpha\beta}(\omega)=
  \delta\mu_\alpha(\omega)/\kappa_\beta\);
- an independent current-domain frequency check.

The transform convention is

\[
\widetilde f(\omega)=\int_0^T
w(t)f(t)e^{+i\omega t}\,dt .
\]

Store the full complex response and all sign, normalization, time-origin,
quadrature, window, damping, and zero-padding metadata. Rectangular windowing
is the default. Exponential damping \(e^{-\eta t}\) is explicit and
\(\eta\) is stored in energy units. Zero padding interpolates but does not
improve the intrinsic resolution \(2\pi/T\).

For resonant comparisons, select a polarization-bright root using an explicit
threshold or user-selected root and recompute it for every basis.

## 10. Work packages

Effort and compute ranges below are planning estimates, not commitments.
Every work package ends with a focused commit only after its acceptance gate
passes. If evidence invalidates an assumption, stop at that gate, record the
finding, revise the plan, and do not paper over a failed invariant.

### WP0 — Preserve and inventory the research draft

Dependencies: none.

Status: complete on 2026-09-08.

Work:

- record the dirty worktree and current disk inventory;
- create a timestamped .tar.zst of the complete Aion tree, including .git and
  untracked scientific outputs;
- exclude only __pycache__, .pytest_cache, and src/aion.egg-info;
- generate and verify SHA-256;
- create /home/cgs/00_WORK/Projection_Code/CALCULATIONS as a separate Git
  repository;
- move molecule/campaign code, data, plots, reports, and legacy integrator
  studies there using a manifest and checksums;
- retain only library code, compact tests, build/environment files, canonical
  docs, and tiny generic examples in Aion.

Acceptance:

- archive extraction reproduces the pre-refactor tree;
- checksums for every moved item match;
- no symlink-dependent scientific data;
- large artifacts are ignored in CALCULATIONS while inputs, analysis, and
  manifests are tracked;
- no scientific artifact is deleted.

Completion record:

- archive:
  /home/cgs/00_WORK/Projection_Code/aion_pre_refactor_20260908T212444Z.tar.zst;
- archive SHA-256:
  d2d99669321ace86e3496c17eab91a53ecc2cc168ba4abbfc1ea1926863ee24f;
- all 4,641 archived regular files and 201 symlinks were verified by an
  extraction-and-comparison audit before migration;
- 4,185 moved regular files totaling 1,665,696,695 bytes passed the pre/post
  content-hash comparison;
- all 201 moved symlinks remain internal and none points back into Aion;
- the CALCULATIONS repository contains the root move map, source/destination
  checksums, ignored-artifact index, and symlink index;
- Aion now retains the reusable source/tests/tooling and the three canonical
  documentation deliverables; campaign and legacy material has moved;
- branch refactor/0.2 was created after archive validation and before the
  migration.

Risks and recovery:

- Disk pressure: measure before copying and stop if capacity is inadequate.
- Unclassified files: place them in a manifest-listed legacy holding area;
  never guess or delete.
- Recovery is extraction of the immutable archive.

Estimate: 1–2 developer days; I/O time depends on the approximately
gigabyte-scale examples/results tree.

Milestone M0: untouched archive plus auditable clean repository boundaries.

### WP1 — Establish the 0.2 package, configuration, and schema contracts

Dependencies: WP0.

Status: complete on 2026-09-08.

Work:

- rebuild pyproject.toml for version 0.2.0 development;
- declare Python 3.12, explicit h5py runtime dependency, CPU/GPU extras, Ruff,
  mypy, pytest, and xdist policy;
- create package domains and explicit public exports;
- implement strict immutable configuration dataclasses, normalized TOML,
  lossless canonical hashing, units/origin/time-grid types, and support
  validation;
- specify HDF5 schemas, typed observable records, status.json, errors, and
  semantic versioning;
- configure Ruff and strict boundary typing with narrow documented array
  exceptions.

Tests:

- unknown/unused configuration fields fail;
- resolved defaults round-trip and hash identically;
- paths, timestamps, host, and output schedules do not perturb scientific ID;
- physics-changing values do perturb it;
- schema fixtures validate and incompatible major versions fail clearly.

Acceptance:

- package imports with no legacy side effects;
- CLI skeleton and top-level API are documented and typed;
- all fast CPU tests pass on the managed environment;
- no numerical GPU behavior is introduced in this package.

Estimate: 3–5 developer days; negligible scientific compute.

Milestone M1: stable contracts on which numerical work can proceed.

### WP2 — Prepared references, backends, operators, and compiled sources

Dependencies: WP1.

Work:

- implement CPU/GPU backend protocols and residency assertions;
- implement immutable PreparedReference and backend Workspace;
- implement PySCF RKS preparation and local/pure-functional support checks;
- build core and derived operator bundles, momentum sign qualification, exact
  grid persistence, and fingerprints;
- implement reference.h5 transactional I/O and live-object reconstruction;
- implement physical uniform sources, LG/VG gauge derivation, P0 node/link
  samples, envelopes, additive composition, source compilation, pulse
  safeguards, and discrete-event definitions.

Tests:

- H2 and LiH CPU/GPU prepared-reference parity;
- momentum Hermiticity/sign against independent derivative checks;
- reference save/load/rebuild fingerprints;
- field/potential/gauge identities at all compiled endpoints and midpoints;
- finite-pulse endpoint, peak-field, duration, and configured-impulse checks;
- exact event alignment and idempotent event identifiers;
- device-residency tests that detect host transfer.

Acceptance:

- one immutable reference constructs all four formulations without repeating
  SCF;
- the compiled source is identical across backend paths within precision;
- GPU tests run on the physical GPU and pass; no fallback is possible.

Estimate: 5–8 developer days; minutes to a few hours of H2/LiH CPU/GPU test
compute.

Milestone M2: trustworthy common static problem and EM input.

### WP3 — Formulations, currents, dipoles, and energy components

Dependencies: WP2.

Work:

- implement the explicit Formulation triple and bare LG/VG;
- implement P0 Wilson metric, site connection, inverse dressing, PySCF
  Hamiltonian, charges, graph currents, and power;
- implement composable E1 tensors, connection term, covariant dipole, analytic
  derivative, current decomposition, and source current;
- implement generic and uniform mechanical-current calculators with
  paramagnetic/diamagnetic separation;
- implement the complete formulation-specific energy ledger and analytic
  energy derivatives;
- implement typed dependency-declared observable calculators and independent
  schedules.

Tests:

- synthetic Hermiticity, metric compatibility, dressing, pair orientation,
  continuity, gauge transformation, and origin-translation identities;
- directional-derivative checks for the variational P0+E1 source current;
- independent energy-component contractions and Ward derivatives;
- H2 symmetry/zero checks and LiH nonzero E1 checks;
- CPU/GPU parity for density, currents, dipoles, energy components, and
  invariants;
- unsupported ECP/hybrid/spin cases fail before propagation.

Acceptance:

- no observable borrows a definition from another formulation;
- P0+E1 contains V_E1 exactly once in the EOM;
- primary and diagnostic currents have unambiguous stored names;
- all analytic identities pass stated tolerances on CPU and physical GPU.

Estimate: 8–12 developer days; hours of compact CPU/GPU validation.

Milestone M3: static and instantaneous physics is complete and independently
testable.

### WP4 — Shared SCEM and connection-aware transport

Dependencies: WP3.

Work:

- implement orbital-state protocol and shared midpoint-density solver;
- implement damped Picard policy and diagnostics;
- implement common Cayley [1/1] and Padé [2/2] rational maps;
- implement direct fixed-metric SCEM;
- implement analytic site-parallel transport, pulled-frame combined
  generator, and right-Cholesky cross-metric correction;
- preallocate hot-loop workspaces and enforce backend residency;
- retain correction disablement only as an explicit diagnostic mode.

Tests:

- exact synthetic fixed-generator cases;
- algebraic fixed-metric norm preservation;
- evolving-metric compatibility and cross-metric preservation;
- raw versus corrected defects and correction-size scaling;
- formal second-order convergence;
- nonlinear failure and threshold behavior;
- roundoff-only Hermitian cleanup versus hard failure;
- H2 and LiH CPU/GPU trajectory parity and device-residency profiling.

Acceptance:

- fixed and connection-aware transports use one nonlinear engine;
- no evolving-frame Löwdin propagation is present;
- accepted steps satisfy convergence, Hermiticity, and metric constraints;
- measured global order is approximately two in the asymptotic regime;
- GPU tests run on the physical device and show no hot-loop host transfers.

Estimate: 8–12 developer days; several hours to one day of compact convergence
and parity compute.

Milestone M4: the production propagators satisfy their geometric contracts.

### WP5 — Runners, observers, checkpoint/restart, CLI, and monitoring

Dependencies: WP4.

Work:

- implement build_simulation, run, and resume;
- implement independent endpoint/interval observable schedules;
- accumulate midpoint source work every accepted interval;
- implement transactional trajectory/source/checkpoint writers and immutable
  completion;
- implement safe SIGINT/SIGTERM boundary checkpointing;
- implement reconstruction-first restart, event idempotence, lineage, and
  stitching;
- complete prepare/run/resume/inspect/export CLI and status.json;
- implement comparison manifests without coupling independent trajectories in
  one process.

Tests:

- initial/final samples are always present;
- endpoint and midpoint indexing is exact;
- restart equals uninterrupted propagation for state, work, observables, and
  event history;
- deliberate interruption leaves only a valid checkpoint/failed or partial
  artifact;
- completed artifacts cannot be overwritten;
- CPU/GPU restart and output parity;
- detached status transitions are atomic and meaningful.

Acceptance:

- a run can be monitored without polling its process;
- no half-converged state is published;
- every completed trajectory is self-describing and traceable;
- GPU tests run on the physical device.

Estimate: 6–9 developer days; hours of restart/failure-injection CPU/GPU
compute.

Milestone M5: robust reusable execution and persistence.

### WP6 — Spectroscopy and legacy conversion

Dependencies: WP5.

Work:

- implement reusable Casida wrapper and structured output;
- implement kick-response transforms and current-domain cross-checks;
- store all transform conventions and resolution metadata;
- implement resonance selection by polarized brightness;
- create explicit one-way converters for scientifically useful legacy CSV/NPZ
  results into the new schema;
- keep legacy readers out of the core runtime.

Tests:

- analytic sinusoid transform/sign/normalization fixtures;
- damping width, zero-padding, and resolution tests;
- H2/LiH Casida-versus-kick qualitative/quantitative checks;
- converter fixture checksums and metadata completeness;
- CPU/GPU kick propagation parity on the physical GPU.

Acceptance:

- spectra can be reproduced from trajectory plus stored transform
  configuration;
- no smoothing or normalization is hidden;
- converted artifacts are new immutable files linked to parent checksums.

Estimate: 4–7 developer days; several hours of compact kick trajectories.

Milestone M6: reusable linear-response and real-time spectroscopy.

### WP7 — Integrated validation, performance, documentation, and release

Dependencies: WP0–WP6.

Work:

- establish pre-replacement preparation, steady-step, and memory baselines;
- run synthetic, H2, and LiH validation matrices;
- run the agreed NH3 aug-cc-pVTZ release-validation fixture: five driven
  cycles plus five field-free cycles, four formulations, accepted time step,
  with a shorter half-step interval;
- validate gauge relations, continuity, current-dipole identities,
  work-energy Ward identities, timestep convergence, restart parity, and
  CPU/GPU parity;
- revise this plan and the theory note to match the implemented contracts;
- complete API/schema/user documentation and a tiny reproducible example;
- perform clean-environment install and artifact audit.

Testing policy:

- fast unit suite: up to eight xdist workers, one BLAS thread each;
- molecular CPU suite: serial tests with up to eight BLAS threads;
- GPU suite: serial on the physical GPU;
- long production campaigns remain outside Aion;
- flag an unexplained steady-step regression near 10 percent or larger, while
  never trading correctness for the threshold.

Acceptance:

- all release invariants and CPU/GPU tests pass;
- the NH3 fixture supports validation only and makes no premature production
  claim;
- accepted limitations and non-goals are prominent;
- docs, HDF5 schema, CLI, and public API agree;
- no campaign-only code or large result artifact remains in Aion;
- version 0.2.0 is tagged only after all gates pass.

Estimate: 5–8 developer days plus approximately hours to days of validation
compute, to be replaced by measured ranges from the WP0/WP4 baselines.

Milestone M7: releasable Aion 0.2.

## 11. Milestone dependency graph

\[
M0\rightarrow M1\rightarrow M2\rightarrow M3\rightarrow M4
\rightarrow M5\rightarrow M6\rightarrow M7 .
\]

Documentation and profiling are continuous activities, but no downstream
milestone may waive an upstream acceptance gate. Independent preparatory
tasks inside a work package may run in parallel only when they do not obscure
the relevant physics comparison.

## 12. Validation matrix

| Fixture | Purpose | CPU | GPU |
| --- | --- | --- | --- |
| Synthetic matrices | Signs, Hermiticity, metric compatibility, graph orientation, exact transport algebra | Required | Required where backend code exists |
| H2 | Zero field, symmetry, kicks, bare gauges, compact timestep convergence | Required | Required |
| LiH | Nonzero E1, covariant current, origin/gauge transforms, Ward identities, connection transport | Required | Required |
| NH3 aug-cc-pVTZ | Integrated release validation, four formulations, pulse plus free evolution | Required | Required |
| CO and other campaigns | External scientific studies | CALCULATIONS | CALCULATIONS |

Full-array golden trajectories are not the primary oracle. Use small selected
endpoints only where they protect an independent interface. Physics
identities and convergence behavior are authoritative.

## 13. Failure and recovery policy

The implementation must fail early and specifically for unsupported physics,
invalid gauge data, off-grid events, incompatible references, unavailable
requested GPUs, nonlinear nonconvergence, excessive Hermiticity/metric defects,
and schema mismatch.

No recovery path may silently:

- reduce or change the time step;
- loosen tolerances;
- switch formulations;
- switch backend;
- accept a failed midpoint;
- omit a requested observable;
- overwrite a completed artifact.

Explicit recovery experiments receive new run IDs and record the parent and
reason.

## 14. Development and review practice

- Preserve the current dirty draft before restructuring.
- Use one focused commit per accepted work package or coherent sub-gate.
- Run Ruff, mypy, and the appropriate CPU/GPU tests before each numerical
  milestone commit.
- Record numerical tolerances with a physical/numerical rationale.
- Treat source orientation, charge sign, AO-density convention, units, and
  origin as centralized types or constants, not repeated comments.
- Require independent implementation paths for important Ward/current/energy
  checks so tests are not tautologies.
- Measure preparation, steady propagation, Fock builds, host transfers, memory,
  and I/O separately.
- Update this plan when implementation evidence changes ordering, estimates,
  or module placement; record the rationale in a change log.

## 15. Plan change log

| Date | Change | Reason |
| --- | --- | --- |
| 2026-09-08 | Initial plan created from the completed design interview | Establish the agreed 0.2 implementation baseline before any tree mutation |
| 2026-09-08 | Removed special-purpose pulse terminology | Pulses are source objects; their scientific use belongs to a simulation or workflow |
| 2026-09-08 | Completed WP0 archive, migration, and audit | Establish an immutable recovery point and separate reusable code from calculations before refactoring |
| 2026-09-08 | Completed WP1 package, configuration, identity, schema, observable-record, status, API, CLI, and quality contracts | Establish the clean typed 0.2 boundary and remove active access to unreviewed draft numerics before porting physics |

## 16. Present authorization

Creation of this plan, the companion LaTeX note/PDF, and execution of WP0 and
WP1 were authorized and are complete. WP2 through WP7 remain proposed future
actions. This document does not itself authorize numerical implementation, new
calculation campaigns, or release.
