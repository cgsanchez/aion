# Uniform-magnetic-field matrix-element benchmark

Status: approved implementation plan; no numerical implementation started

Branch: `feature/magnetic-matrix-benchmark`

Branch point: Aion commit `6acb975a8e63b0ddbd493430ad2eca9619756c45`

Last updated: 2026-09-10

Companion theory note in the shared working directory:
`/home/cgs/00_WORK/Projection_Code/aion_magnetic_matrix_element_benchmark_note.tex`

## 1. Objective

Implement a reusable, backend-aware static matrix-element laboratory for the
uniform-magnetic-field Wilson hierarchy described in the companion note. The
benchmark shall calculate exact finite-field Wilson matrices and their first-
and second-order internal-field expansions for a fixed zero-field Aion
reference. It shall retain the individual geometric contributions so that the
importance of the triangular form factor and the anchored curvature vectors
can be measured rather than inferred.

The initial scientific targets are

\[
S,\qquad T,\qquad V_{\mathrm{eN}},\qquad
\boldsymbol\omega_{\mathrm{spatial}}.
\]

A generic multiplicative local-potential interface shall also be provided.
Frozen Hartree and exchange--correlation potentials are staged separately so
that explicit Wilson geometry is not confused with self-consistent magnetic
density response.

This is a static benchmark. It must not change the current SCF reference,
real-time formulations, nonlinear solver, propagation algorithm, observables,
or trajectory schema. Evidence from the benchmark may later motivate a
separate dynamical magnetic hierarchy, but that decision is outside this
branch.

## 2. Scope and non-goals

### 2.1 Initial supported domain

- finite, fixed-nucleus, all-electron molecules;
- ordinary real Gaussian AOs anchored at their parent atoms;
- restricted spin-summed zero-field RKS references already accepted by Aion;
- spatially uniform, static magnetic fields;
- straight anchor-to-point and anchor-to-anchor Wilson paths;
- orbital magnetic coupling only;
- atomic-unit internal calculations with `q = -1`, `m = 1`, and `hbar = 1`
  as the public electron default;
- float64/complex128 NumPy and physical-CUDA CuPy execution;
- exact, B1, and B2 internal uniform-field matrices at fixed electronic
  reference.

Here, B2 means second order in the amplitude of a spatially uniform magnetic
field. It is not a magnetic-gradient expansion.

### 2.2 Explicit non-goals

- real-time propagation in a magnetic field;
- magnetic SCF relaxation or current-DFT;
- self-consistent magnetic Hartree/XC response;
- spin Zeeman, spin--orbit, relativistic, unrestricted, or noncollinear terms;
- pseudopotentials, ECPs, or gauge-transformed nonlocal ionic operators;
- moving nuclei or nuclear-velocity connections;
- spatially varying magnetic fields;
- periodic boundary conditions;
- production EELS or aloof-projectile calculations;
- deciding a production dynamical B1/B2 closure.

Unsupported physics must fail explicitly. It must not be approximated by a
silent fallback.

## 3. Conventions and terminology

AO `mu` is anchored at `R_mu`, where the anchor is obtained from the
authenticated `PreparedReference.anchor_topology.ao_to_atom` map. For each AO
pair,

\[
R_{\mu\nu}=\frac{R_\mu+R_\nu}{2},\qquad
\Delta R_{\mu\nu}=R_\mu-R_\nu,\qquad
\xi=r-R_{\mu\nu}.
\]

The internal uniform-field phase and anchored vector are

\[
\phi_{\mu\nu}(r)=\frac{q}{2\hbar}
B\cdot\left(\xi\times\Delta R_{\mu\nu}\right),
\qquad
C_\mu(r)=\frac12(r-R_\mu)\times B.
\]

The endpoint link `Theta` is retained exactly. The primary comparison is made
with the gauge-independent thick-link or barred matrices after removing
`Theta`. Exact means exact in the uniform-field Wilson factors for the
specified finite Gaussian basis, up to the explicitly measured quadrature
error. It does not mean complete-basis, self-consistent finite-field, or exact
many-electron physics.

The code must use `spatial_connection` or `omega_spatial` for the Cartesian
lower-index projected spatial connection. The bare word `connection` is
reserved by the current formulation API for the temporal connection appearing
in the equation of motion. The spatial object has shape `(3, nao, nao)`.

Magnetic-field inputs to the reusable numerical API are named
`magnetic_field_au`. Tesla conversion is explicit and occurs at a typed input
or campaign boundary. Both values are recorded when a campaign begins from a
field specified in tesla.

For the kinetic hierarchy, left and right cross terms remain distinct in the
authoritative result:

\[
T_{pC},\qquad T_{Cp}.
\]

The reported `T_C^(1)` is their sum. The same rule applies to the two
form-factor/cross terms at second order. Keeping the adjoint pair separate is
an important implementation diagnostic.

## 4. Relationship to existing Aion and prior work

The current Aion reference already stores:

- the exact quadrature coordinates and weights used by the zero-field RKS
  reference;
- analytic overlap, kinetic, nuclear-attraction, position, and canonical-
  momentum matrices;
- nuclear coordinates and charges;
- AO-to-atom anchors and atom-pair topology;
- the converged density, orbitals, occupations, and dependency fingerprints.

A live PySCF or GPU4PySCF mean-field object and molecule can be reconstructed
from a `PreparedReference` without rerunning SCF. AO values and first
derivatives are derived blockwise from that reconstructed molecule; they do
not need to be added to the persistent reference artifact.

The checkpoint commit `2befb8c` contains a draft implementation of much of the
static S/T/V problem in the former files `reference/wilson_grid.py` and
`reference/wilson_approximations.py`. Archived evidence under
`CALCULATIONS/legacy/wilson_studies` records successful direct-versus-
factorized, GIAO-derivative, symmetry, grid-convergence, and B1/B2 scaling
tests. That implementation is prior mathematical evidence and a source of
independent test cases. It shall not be cherry-picked wholesale because its
object model, package boundaries, backend policy, and I/O contracts predate
the Aion 0.2 refactor.

## 5. Proposed architecture

Reusable functionality remains in Aion and is organized by physics
responsibility:

```text
src/aion/electromagnetism/magnetic.py
    physical uniform-B values, affine gauge representatives, endpoint links,
    triangle phases, and anchored curvature vectors

src/aion/electronic_structure/ao_quadrature.py
    chunked CPU/GPU AO-value and first-derivative sampling

src/aion/electronic_structure/magnetic_matrices.py
    immutable matrix/result types and exact/B1/B2 contractions

src/aion/io/magnetic_benchmark.py
    versioned transactional HDF5 result artifact

src/aion/workflows/magnetic_benchmark.py
    expert workflow binding a PreparedReference, fields, grid policy, backend,
    requested operators, validation policy, and output
```

The final module names may be adjusted during implementation if the resulting
dependency direction is clearer, but the following rules are fixed:

- magnetic geometry must not depend on propagation or formulations;
- AO sampling belongs to electronic structure, not to a campaign script;
- numerical kernels consume the selected `ArrayBackend` namespace;
- mutable arrays and caches remain workspace-local;
- host transfers occur only at an explicit result/storage boundary;
- the benchmark is an expert subpackage API and does not enlarge the small
  top-level Aion API unless a later review explicitly decides otherwise;
- molecule scans, plotting, and scientific interpretation remain in
  `CALCULATIONS`, not in the reusable Aion tree.

## 6. Numerical design

### 6.1 Blocked evaluation

No implementation may retain arrays of shape `(ngrid, nao, nao)` for the full
grid. The current NH3/aug-cc-pVTZ validation reference has 115 AOs and 43,328
grid points; a single full-grid complex pair array would require roughly
9.2 GB before temporary arrays.

One grid block may contain:

- coordinates and weights;
- AO values `(nblock, nao)`;
- AO gradients `(3, nblock, nao)` or an equivalent documented layout;
- `C_mu` `(nblock, nao, 3)`;
- `phi` or its generator `(nblock, nao, nao)`;
- only the temporary factors needed for one field or one small field batch.

Block size is explicit, validated, included in execution metadata, and tested
for invariance. The implementation should process all requested amplitudes
along a direction while each AO block is available, avoiding repeated AO
evaluation, but it must loop or sub-batch fields when necessary to respect a
declared memory budget.

### 6.2 Stable zero-field correction

The authoritative finite-field construction is

\[
M(B)=M^{\mathrm{analytic}}(0)
     +\left[M^{\mathrm{grid}}(B)-M^{\mathrm{grid}}(0)\right].
\]

For scalar operators, the correction uses `expm1(i*phi)` rather than forming
`exp(i*phi) - 1` separately. Equivalent cancellation-aware expressions are
used for the kinetic and spatial-connection sectors. Both the raw quadrature
zeroth-order matrix and the corrected matrix are retained in validation
metadata so that the numerical floor is visible.

### 6.3 Grid policy

Two grid modes serve different purposes:

1. `reference`: use the exact authenticated grid stored in the
   `PreparedReference`; this measures behavior on Aion's operational DFT grid.
2. `qualification`: build an explicitly unpruned PySCF grid at a declared
   level and repeat selected cases at the next level; this establishes the
   numerical floor of small magnetic corrections.

Scientific acceptance cannot rely only on one pruned DFT grid. Every reported
small correction is compared with the larger of its zero-field analytic
residual and its matching grid-refinement change.

### 6.4 Backend policy

CPU contractions use NumPy and PySCF AO evaluation. GPU contractions use CuPy
and GPU4PySCF AO evaluation, with all block temporaries and accumulators
resident on the selected device. A requested GPU calculation must fail if the
physical GPU path is unavailable; CPU AO evaluation followed by repeated
host-to-device block copies is not the accepted GPU implementation.

Every work package that changes executable numerical behavior requires:

- fast algebraic tests;
- serial CPU integration tests where PySCF is needed;
- physical-GPU execution and CPU/GPU parity tests where the new path applies.

The sandbox hides the GPU, so elevated execution is requested immediately
when those tests are due. CPU and GPU test launchers must set explicit host
thread limits and affinity so that they do not oversubscribe the workstation's
eight physical cores.

## 7. Result contract

The immutable result is keyed by reference fingerprint, benchmark definition,
field direction, field amplitude, grid fingerprint or generated-grid
description, backend, charge, mass, and hbar. Its authoritative components
include at least:

```text
overlap
    zero, first_F, second_F2, exact

kinetic
    zero
    first_F, first_pC, first_Cp
    second_F2, second_F_pC, second_F_Cp, second_C2
    exact

local/<provider-name>
    zero, first_F, second_F2, exact

spatial_connection
    zero
    first_F, first_C
    second_F2, second_FC
    exact
```

Aggregate B1/B2 matrices are deterministic views or validated stored sums of
those components. Every Cartesian spatial-connection matrix is stored
explicitly. Optional direct-gauge oracle results and `Theta` are stored only
when requested; the barred matrices remain the primary scientific objects.

Diagnostics include:

- Frobenius and maximum-element norms;
- onsite, same-anchor, and intersite block norms;
- kinetic `F/C` ratios and separate adjoint-pair residuals;
- Hermiticity or anti-Hermiticity residuals;
- field-reversal residuals;
- direct-versus-factorized residuals;
- zero-field analytic and grid-refinement residuals;
- E1-dipole closure residuals;
- finite-difference coefficient residuals;
- exact-minus-B1 and exact-minus-B2 errors;
- metric eigenvalue extrema and condition numbers;
- CPU/GPU and block-size parity metadata when those comparisons are run.

Pointwise AO values, gradients, `F`, `phi`, and `C` arrays are temporary and
are not stored in the result artifact.

## 8. Work packages

### MB0 — Freeze the mathematical and software contract

**Goal:** turn the companion note and prior evidence into an unambiguous
implementation specification before writing numerical kernels.

**Work:**

- bring a reviewed copy of the LaTeX theory note into branch documentation;
- reconcile its endpoint-link, charge, path-orientation, cross-product, and
  momentum signs with the current P0 conventions;
- define exact array shapes, index order, units, field scaling, and result
  names;
- distinguish the spatial connection from the temporal EOM connection;
- document the direct gauge-space oracle and the endpoint-factorized route as
  independently assembled calculations;
- map each recoverable test or identity from commit `2befb8c` to the new API;
- explicitly document the spinless, all-electron, fixed-reference boundary;
- decide and document the GGA frozen-XC weak-form boundary described in MB5.

**Tests/review:** hand-check `phi`, `C`, `Theta`, and momentum conventions on
one oriented two-center geometry and verify that they reproduce the current
P0 endpoint-link convention at `F = 1`.

**Exit gate:** every sign and unit has one authoritative definition, all
symbols map to code names and shapes, and no unresolved convention is left to
be inferred inside a contraction kernel.

**Dependencies:** none.

### MB1 — Implement magnetic geometry and immutable value types

**Goal:** provide the backend-neutral geometry from which both the exact and
expanded calculations are built.

**Work:**

- add typed uniform magnetic-field and affine magnetic-gauge values;
- implement symmetric and Landau vector-potential representatives for the
  direct oracle;
- implement analytic straight-line anchor-to-anchor and anchor-to-point
  integrals and their spatial derivatives;
- implement AO anchor coordinates, pair midpoints, displacements, endpoint
  links, triangle phases, exact factors, Taylor factors, and anchored vectors;
- add an explicit magnetic-field physical dimension and atomic-unit label to
  configuration/I/O metadata without altering electric-source behavior;
- keep physical `B` separate from its gauge representative;
- support arbitrary finite `q`, positive `m`, and positive `hbar` in low-level
  kernels while making electron atomic units the high-level default.

**Tests:** pair reversal, same-anchor identities, field reversal, symmetric-
versus-Landau gauge relations, gauge-origin endpoint congruence, Taylor
coefficient normalization, rigid rotations, invalid inputs, NumPy/CuPy parity,
and physical-GPU residency/no-fallback behavior.

**Exit gate:** direct and factorized geometry agree analytically for displaced
oblique point/anchor fixtures, and every new physical-GPU test passes.

**Dependencies:** MB0.

### MB2 — Add reusable chunked AO quadrature

**Goal:** expose AO values and first derivatives from a `PreparedReference`
without storing them permanently or rerunning SCF.

**Work:**

- define an AO-block record and iterator owned by electronic structure;
- reconstruct the PySCF molecule through the existing authenticated path;
- evaluate spherical AO values and Cartesian first derivatives using the
  exact supplied grid coordinates;
- provide CPU PySCF and GPU4PySCF implementations with identical layout and
  derivative meaning;
- support the stored reference grid and generated unpruned qualification
  grids;
- validate block size and optional memory budget;
- keep accumulators backend-resident and make transfers explicit;
- record AO-evaluator, dependency, grid, backend, block-size, and thread
  provenance.

**Tests:** reconstruct without SCF, AO and gradient CPU/GPU parity, derivative
sign against the analytic canonical momentum, block concatenation, block-size
invariance, reference-grid fingerprint preservation, generated-grid metadata,
and GPU residency/no-fallback.

**Exit gate:** overlap and canonical momentum reconstructed from blocks
converge to the analytic PySCF matrices at the declared grid floor on H2 and
LiH, with physical-GPU parity passing.

**Dependencies:** MB0 and the existing prepared-reference API. MB1 is needed
only for magnetic block consumers, not for the generic AO iterator.

### MB3 — Implement exact/B1/B2 overlap, kinetic, and nuclear attraction

**Goal:** deliver the core one-electron magnetic hierarchy on the current Aion
reference and backend contracts.

**Work:**

- implement exact barred overlap using the full triangle factor;
- implement overlap zeroth-, first-, and second-order components;
- implement the exact kinetic gradient form;
- retain `pp`, `pC`, `Cp`, and `CC` sectors before aggregation;
- map those sectors into the B1 and B2 `F`, `C`, `F2`, `FC`, and `C2`
  components defined in the theory note;
- implement nuclear-attraction values on each grid block and its exact/B1/B2
  scalar hierarchy;
- use analytic zero-field S/T/V matrices plus cancellation-aware grid
  corrections;
- support multiple amplitudes per direction without reevaluating AO blocks;
- make exact endpoint-dressed lower matrices available as optional views;
- preserve all barred decomposition matrices in immutable result types.

**Tests:** zero-field analytic recovery, sector sums, Hermiticity, field
reversal, onsite and bond-parallel `F = 1` isolation, exact direct-versus-
factorized agreement, independent libcint GIAO first derivatives, finite-
difference first/second coefficients, quadratic B1 remainder, cubic B2
remainder, grid refinement, block-size invariance, CPU/GPU parity, and GPU
residency/no-fallback.

**Exit gate:** H2 and LiH pass every identity above; measured truncation
orders are correct above the cancellation floor; GIAO and direct-oracle
comparisons agree at the declared quadrature tolerance on CPU and GPU.

**Dependencies:** MB1 and MB2.

### MB4 — Implement the projected spatial-connection hierarchy

**Goal:** add the part of the companion note that was absent from the archived
one-electron benchmark.

**Work:**

- assemble the manifestly anti-Hermitian zeroth-order derivative form;
- implement exact triangle-factor and explicit anchored-vector contributions;
- retain first-order `F` and `C` components separately;
- retain second-order `F2` and `FC` components separately;
- provide Cartesian `(3, nao, nao)` exact, coefficient, and truncation views;
- implement the analytic zero-field baseline from the stored momentum matrix;
- implement the E1 central-dipole closure for the first-order `C` term;
- keep this result independent of all temporal formulation objects.

**Tests:** anti-Hermiticity of every exact/truncated Cartesian matrix,
zero-field `omega_spatial = i*p/hbar` under Aion's documented convention,
field reversal and order parity, finite-difference coefficients, E1-dipole
closure, same-anchor/bond-parallel isolation, rigid rotation, block-size
invariance, CPU/GPU parity, and GPU residency/no-fallback.

**Exit gate:** the momentum and E1 closures agree independently at the
quadrature floor, all three Cartesian matrices obey anti-Hermiticity, and
exact-minus-B1/B2 remainders show the expected orders.

**Dependencies:** MB1--MB3. MB3 provides the shared contraction/result
patterns but MB4 must not obtain its answer by differentiating an MB3 output.

### MB5 — Generalize multiplicative local potentials and stage frozen DFT terms

**Goal:** make the scalar hierarchy reusable beyond nuclear attraction while
keeping mathematically different DFT cases explicit.

**Work:**

- define a pointwise local-potential provider protocol with stable identity,
  units, provenance, and support validation;
- express nuclear attraction through that protocol after its dedicated MB3
  tests pass;
- add a frozen Hartree-potential provider derived from the authenticated
  zero-field density, with an independently checked zero-field AO matrix;
- add frozen LDA XC only after the pointwise potential reproduces PySCF's
  zero-field XC matrix within the grid floor;
- investigate GGA separately. PySCF normally assembles GGA XC in a weak form
  involving density and AO gradients. A pointwise scalar provider must not be
  claimed until its Euler--Lagrange potential or an equivalent Wilson-dressed
  weak form has been derived and independently validated;
- record unsupported providers explicitly rather than treating an already
  assembled AO matrix as if it were a pointwise local operator.

**Tests:** provider identity and hashing, zero-field matrix recovery, exact/
expanded scalar hierarchy identities, potential scaling, field reversal,
grid refinement, CPU/GPU parity, and explicit rejection of unqualified GGA or
nonlocal potentials.

**Exit gate:** the generic provider and nuclear case are required for the
initial benchmark. Frozen Hartree and LDA XC may be delivered as a subsequent
milestone. GGA is accepted only with a reviewed derivation and an independent
zero-field/magnetic validation; it is not an initial-release blocker.

**Dependencies:** MB2 and MB3. Frozen DFT providers additionally depend on the
existing adiabatic RKS/PySCF bridge.

### MB6 — Add result artifacts and the reusable workflow API

**Goal:** turn individual kernels into a reproducible benchmark that can be
used without ad-hoc scripts.

**Work:**

- define immutable benchmark configuration and result records;
- accept a prepared reference, selected backend, grid policy, field
  directions/amplitudes, requested matrix families, block/memory policy, and
  validation options;
- evaluate fields deterministically and expose progress at field/block
  boundaries without imposing polling;
- create a versioned HDF5 schema using Aion's transactional no-overwrite
  publication mechanism;
- store matrices, components, units, definitions, diagnostics, source
  reference fingerprint, dependency versions, configuration hash, and
  completion/failure metadata;
- implement strict loader validation and tamper/fingerprint detection;
- add a concise inspection/export path for scalar summaries while keeping raw
  matrix artifacts authoritative;
- keep this as an expert workflow and out of the real-time CLI unless a later
  usability review justifies a dedicated command.

**Tests:** normalized configuration and semantic hashing, HDF5 round trip,
schema/unit validation, transactional completion, no overwrite, controlled
failure, corruption detection, CPU/GPU artifact equivalence within tolerance,
and proof that invoking the workflow does not call SCF or propagation.

**Exit gate:** a previously prepared H2 reference can produce, save, load,
inspect, and validate a complete exact/B1/B2 result using only reusable Aion
APIs on CPU and physical GPU.

**Dependencies:** MB1--MB4; the generic MB5 provider interface, but not every
optional DFT provider.

### MB7 — Overconstrained release qualification

**Goal:** establish that the implementation is the declared mathematics,
independently of any physical conclusion about approximation quality.

**Work:**

- create fast synthetic geometry/contraction tests;
- create small serial PySCF integration tests for H2 and LiH;
- port the useful mathematical cases from the archived Wilson suite into the
  current contracts rather than copying its old object model;
- compare B1 S/T/V derivatives with PySCF/libcint GIAO kernels after removing
  the endpoint-link derivative;
- retain an independent analytic Gaussian Fourier overlap oracle for selected
  finite-field cases;
- run reference-grid and consecutive unpruned-grid calculations;
- measure exact-minus-B1 and exact-minus-B2 slopes over stress fields chosen
  above cancellation noise;
- verify physical-field results separately rather than interpreting stress
  fields as representative;
- run complete CPU and physical-GPU suites, plus ruff and strict mypy;
- audit host/device transfers, memory bounds, thread limits, and reproducible
  environment metadata.

**Acceptance identities:**

- zero-field grid matrices converge to analytic PySCF matrices;
- direct and endpoint-factorized exact matrices agree;
- `M(-B) = M(B)*` for S/T/real local potentials;
- spatial connection obeys its corresponding anti-Hermitian reversal law;
- B1/B2 coefficients agree with central finite differences above the
  subtraction floor;
- E1 dipoles reproduce overlap B1 and spatial-connection `C` B1 closures;
- H2 with `B` parallel to the bond has identically zero triangle phase for
  every atom-anchored AO pair while `C` terms remain;
- B1 and B2 remainders scale respectively as second and third order unless a
  symmetry-forbidden coefficient is explicitly identified;
- results are invariant under chunk size and covariant under rigid rotations;
- CPU/GPU differences remain within a documented FP64/quadrature tolerance;
- requested GPU execution never falls back to CPU.

**Exit gate:** all routine tests pass; all qualification residuals lie below
predeclared numerical floors; any symmetry-suppressed scaling channel is
identified rather than misclassified; both repositories are clean; the
implemented theory documentation matches the code.

**Dependencies:** MB1--MB6.

### MB8 — H2, LiH, and NH3 measurement campaign and decision record

**Goal:** use the qualified API to decide which magnetic hierarchy terms are
numerically important enough to motivate later dynamical work.

**Location:** all campaign drivers, manifests, raw results, plots, and
scientific reports live under a new dedicated campaign in `CALCULATIONS`.
Reusable fixes discovered by the campaign return to the Aion feature branch
with regression tests.

**Work:**

1. H2:
   - bond-parallel and two perpendicular fields;
   - same-anchor and intersite block separation;
   - exact `F = 1` parallel isolation;
   - compact and correlation-consistent basis checks;
   - stored-grid and unpruned-grid refinement.
2. LiH:
   - heteronuclear E1 closure with nonzero central dipoles;
   - parallel, perpendicular, and oblique fields;
   - onsite/intersite and angular-channel decomposition;
   - basis and grid refinement.
3. NH3:
   - a realistic multiorbital reference already used in Aion validation;
   - three independent field orientations tied to molecular axes;
   - memory/performance qualification for aug-cc-pVTZ;
   - selected physical fields and stress fields only after H2/LiH numerical
     floors are understood.

Physical scans should include 1, 10, 30, and 100 T, with an optional extension
to 300 T for comparison with archived evidence. Separate dimensionless or
atomic-unit stress amplitudes are selected to expose B1/B2 asymptotic slopes.
Every plot and table labels these two regimes unambiguously.

**Reported quantities:** all decomposed matrices listed in Section 7;
normalized exact-minus-B1/B2 errors; onsite/intersite block norms; maximum
scaled elements; metric eigenvalues; spatial-connection norms; `T_F/T_C`
ratios; grid floors; and CPU/GPU agreement. Global norms are accompanied by
block or element diagnostics so that symmetry-selected terms cannot be hidden.

**Exit gate:** publish a compact machine-readable evidence file, reproducible
figures, and a written decision record answering:

- when is `F approximately 1` numerically justified?
- is first-order kinetic behavior dominated by `F`, `C`, or their cancellation?
- how rapidly do B1 and B2 converge at physical fields?
- which effects are onsite and which require intersite finite overlap?
- which first-order terms close through existing E1 data?
- is there sufficient evidence to design a dynamical B1/B2 action, and if so,
  which complete closure should be derived next?

No dynamical implementation starts automatically when MB8 completes.

**Dependencies:** MB7.

## 9. Milestones

### Milestone A — Reusable core evaluator

MB0--MB4 complete. A prepared H2/LiH reference produces exact/B1/B2 S/T/V_eN
and spatial-connection matrices through reusable CPU/GPU APIs, without I/O or
campaign assumptions.

### Milestone B — Reproducible benchmark artifact

MB5 generic local-potential support and MB6 complete. Results are
transactional, fingerprinted, loadable, and inspectable.

### Milestone C — Qualified implementation

MB7 complete. Algebraic, PySCF/libcint, direct-Wilson, analytic-overlap,
finite-difference, grid, rotation, CPU, and physical-GPU gates all pass.

### Milestone D — Scientific decision

MB8 complete. H2, LiH, and NH3 evidence supports a documented decision about
the usefulness and domain of the magnetic hierarchy. This milestone does not
imply that a dynamical magnetic formulation has been approved.

## 10. Branch, environment, and integration policy

Development occurs in the dedicated worktree
`/home/cgs/00_WORK/Projection_Code/aion-magnetic-matrix-benchmark`. The active
`/home/cgs/00_WORK/Projection_Code/aion` tree remains on `refactor/0.2` while
the detached Aion 0.2 validation campaign is running.

The feature worktree uses the managed Aion environment without changing its
editable-install target. Tests run from the worktree with its `src` directory
first on `PYTHONPATH`; the worktree-local GPU launcher already enforces this
for GPU commands. No shared toolchain or global Conda state is modified.

The feature branch begins at the current 0.2 release-candidate commit and is
rebased onto the finalized 0.2 tag after the release-validation branch is
complete. Because the benchmark does not modify dynamics, conflicts should be
limited to shared exports, units, documentation, or I/O registries. Merge into
the main development line occurs only after Milestone C. Scientific campaign
results in `CALCULATIONS` are reviewed independently and are not required to
merge the reusable static evaluator unless a release decision explicitly says
otherwise.

Each work package should normally end in one focused commit containing its
implementation, tests, and documentation. If evidence requires changing the
plan, update this document in the same commit and record the reason. The plan
is a mutable engineering guide, not an obstacle to changes justified by
physics or numerical evidence.

## 11. Principal risks and mitigations

- **Sign or path-orientation error:** use independent direct and factorized
  constructions, endpoint-congruence checks, GIAO derivatives, and E1 closure.
- **Quadrature error mistaken for magnetic physics:** use analytic zero-field
  correction, `expm1`, consecutive unpruned grids, and an explicit numerical
  floor.
- **Memory growth as `ngrid*nao^2`:** require blocking, memory-budget tests,
  and no persistent pointwise pair arrays.
- **GPU implementation that is only nominally resident:** use GPU4PySCF AO
  evaluation, residency assertions, transfer audits, and no-fallback tests on
  the physical GPU.
- **Global norms hiding important channels:** store onsite/intersite and
  selected angular-channel diagnostics alongside global norms.
- **Treating diagnostic sector deletion as a physical theory:** label every
  component and truncation precisely and defer dynamical claims to a separate
  action-level derivation.
- **Misrepresenting frozen GGA XC as a simple stored scalar:** require a
  reviewed weak-form or Euler--Lagrange derivation before implementation.
- **Contaminating the ongoing 0.2 validation:** use a separate Git worktree and
  do not repoint the managed editable installation.

## 12. Immediate next action

Begin MB0 by committing a branch-local reviewed theory specification that
incorporates the two-grid qualification policy, the independent direct/GIAO
oracles, the spatial-versus-temporal connection terminology, and the frozen-
GGA boundary. No numerical implementation begins until that specification is
internally consistent with Aion's current charge, momentum, endpoint-link,
and array-index conventions.
