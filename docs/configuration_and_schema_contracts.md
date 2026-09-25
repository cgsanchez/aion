# Aion 0.2 configuration and schema contracts

Status: contracts through reusable execution, observation, checkpoint/restart,
monitoring, and spectroscopy are implemented by WP1--WP7 for Aion
`0.2.0.dev7`. Phase Two P2-1 adds a separate exact/reduced Wilson
configuration and artifact family; its H3+/LDA numerical evidence is an
executed candidate pending explicit user review.

## 1. Authority and strictness

The immutable Python dataclasses in `aion.config` are authoritative. TOML is a
strict reproducibility front end to those same types. Loading a document:

1. rejects unknown fields and fields unused by the selected variant;
2. materializes every stable default;
3. validates the first-release support envelope;
4. retains the exact original TOML text;
5. emits deterministic resolved TOML; and
6. computes the scientific identity from canonical scientific content.

There is no generic options dictionary. Source variants, backends,
formulations, propagators, schedules, and artifact kinds are discriminated
types. A CPU backend with a device index, a zero source with pulse parameters,
an off-grid kick, or a formulation with the wrong integrator fails before
numerical work.

The four configuration schemas are independently identified as:

| Document | `schema` | `schema_version` |
| --- | --- | --- |
| reference preparation | `aion.reference-input` | `1.0.0` |
| real-time simulation | `aion.simulation-input` | `1.0.0` |
| Wilson stationary preparation | `aion.wilson-stationary-input` | `1.0.0` |
| Wilson real-time simulation | `aion.wilson-simulation-input` | `1.0.0` |

Python 3.12 `tomllib` parses user input. `dumps_config` emits complete,
deterministic TOML whose reload produces an equal immutable object and the
same scientific identity. Floating-point scientific hashing uses the exact
hexadecimal representation rather than rounded decimal formatting.

## 2. Reference configuration

A resolved reference document contains:

- `molecule`: fixed atoms in bohr, integer charge, zero spin, and an explicit
  electromagnetic origin in bohr;
- `electronic_structure`: basis, functional, declared LDA/GGA family, grid,
  explicit density-fitting flag and auxiliary basis, SCF tolerance and
  iteration limit, restricted spin, and an all-electron local nuclear model;
- `backend`: CPU/GPU provenance and mandatory float64/complex128 precision;
- `output.artifact_path`; and
- optional human/execution metadata.

Coordinates are preserved exactly; there is no implicit recentering. The
reference scientific identity includes geometry, charge, spin, EM origin,
electronic-structure settings (including density fitting), precision, and
construction schema. It excludes
the output path, label, timestamp, host, preparation backend kind, and GPU
device index. The preparation backend is provenance unless it changes stored
scientific content; the later reference fingerprint authenticates that actual
content.

## 3. Simulation configuration

A resolved simulation document contains:

- an authenticated reference SHA-256 digest plus its relocatable path;
- one of `bare_length_gauge`, `bare_velocity_gauge`, `p0`, or `p0_e1`, with an
  explicit representation (fixed by the bare kind, and selectable from the
  continuous length--velocity gauge family for P0/P0+E1);
- a discriminated physical source definition;
- zero or more uniquely identified exact kick events on integer boundaries;
- an endpoint-inclusive `FixedTimeGrid`;
- SCEM integrator, rational approximation, nonlinear tolerance, iteration
  bound, damping floor, and Hermitian-cleanup threshold;
- explicit CPU/GPU backend and float64/complex128 precision;
- validation policy;
- independent output schedules; and
- optional execution metadata.

Bare formulations require `fixed_metric_scem`; P0 and P0+E1 require
`connection_aware_scem`. The default frozen-exponential approximation is
Padé `[2/2]`; Cayley `[1/1]` is explicit. The fixed time grid treats integer
step as authoritative: `N` intervals always produce `N+1` endpoints.

For P0/P0+E1, `formulation.velocity_fraction` is a constant real number
`lambda` in `[0,1]`. The endpoints are normalized to the existing
representations: `lambda=0` is `gauge="length"`, and `lambda=1` is
`gauge="velocity"`. A strict interior value is represented as
`gauge="mixed"`; for example:

```toml
[formulation]
kind = "p0_e1"
gauge = "mixed"
velocity_fraction = 0.5
```

In Python, `FormulationConfig(FormulationKind.P0_E1,
velocity_fraction=0.5)` infers the mixed representation. Bare formulations
reject an interior fraction. The fraction is part of the simulation's
scientific identity and comparison-manifest provenance. It is constant for a
simulation; a time-dependent gauge-mixing parameter is not implemented.

Sources are potential-first contracts. Zero, compactly supported sin²
vector-potential pulse, authenticated compiled-source, and reproducible
Python-provider records are discriminated inputs. WP2/WP3 implement physical
uniform providers, additive composition, product-rule analytic envelopes,
peak-electric-field pulse normalization, endpoint/midpoint compilation through
the second vector-potential derivative (hence the electric-field derivative),
and LG/VG/P0 node-link gauge derivation. A pulse is simply a source object; its
scientific use belongs to a workflow. Kicks are separate exact events and must
lie on a state boundary.

### 3.1 Exact/reduced Wilson configuration family

`WilsonStationaryConfig` and `WilsonSimulationConfig` are deliberately not
optional tables inside the ordinary simulation schema. They describe a
different nonlinear action, propagated variable, integrator, stationary
state, source domain, and artifact lineage.

A stationary document contains an authenticated reference link, one exact or
reduced action and closure branch, explicit Wilson numerical realization, one
analytic affine electromagnetic source and preparation time, SCF policy, CPU
backend, output artifact, and metadata. The numerical realization fixes:

- reference or explicit qualification AO grid, level, and pruning;
- AO block size and optional memory budget;
- bounded dynamic-cache capacity;
- auxiliary basis and RI metric rank thresholds; and
- optional maximum retained RI rank.

The exact action admits `hartree`, `kohn_sham_lda`, and `kohn_sham_gga`
branches. The reduced action records a named `p0`, `strict_c1`, or
`density_resummed_c1` level, but reusable reduced dynamics is rejected until
P2-6. Stationary preparation is CPU-only; the resulting portable state may be
consumed by CPU or GPU dynamics with the same precision.

The affine source composes a zero or compact potential-first uniform electric
provider, constant electric-field offset, `B` at a reference time, constant
analytic `dB/dt`, electromagnetic origin, and symmetric or Landau gauge. It
is sampled directly at arbitrary Gauss nodes. Arbitrary time-dependent
magnetic envelopes and spatially nonuniform potentials are future provider
variants, not hidden callbacks in this schema.

A Wilson simulation links both reference and stationary-state content hashes,
repeats the action/numerical/source identities, declares its initial-source
policy, fixes an endpoint-inclusive
grid, and requires `nonlinear_gauss_magnus` with Padé `[2/2]`. It stores an
explicit nonlinear tolerance and iteration limit, backend, validation policy,
and the same independent output schedules as the ordinary runner. Construction
rejects a changed reference, grid, RI space, metric, precision, action, or
stationary-state fingerprint before propagation. The default `matched`
initial-source policy also requires the stationary and simulation source
definitions and initial samples to be identical. The explicit
`continuous_density_quench` policy permits an instantaneous change of the
scalar/temporal source while retaining the prepared density only when the
initial spatial Wilson gauge is exactly unchanged. It therefore admits a
finite electric or magnetic-field-rate quench in a fixed spatial frame, but
rejects a magnetic-field, magnetic-gauge, origin, or metric jump. It is not a
discontinuous vector-potential kick or a density transformation.

## 4. Scientific identity

`canonical_sha256` is a type-tagged, length-delimited canonical encoding. It
distinguishes integers from floats, preserves every finite float bit through
`float.hex`, sorts string mapping keys, preserves sequence order, and hashes
NumPy dtype, shape, and canonical little-endian bytes. Non-finite values,
object arrays, and unsupported values fail.

The simulation ID includes the reference fingerprint, formulation, physical
source, events, time grid, numerical algorithm, backend kind/precision, and
validation settings. It excludes:

- reference, compiled-source, and output paths;
- labels, timestamps, and host names;
- GPU device index; and
- every output schedule.

Changing a field, frequency, amplitude, event, timestep, formulation,
integrator, convergence rule, reference digest, or CPU/GPU algorithm therefore
changes the ID. A unique run ID distinguishes
execution attempts sharing one scientific ID.

## 5. HDF5 schemas

Artifact schemas evolve independently of the Python package. Every artifact
has scalar root attributes `schema_name`, semantic `schema_version`,
`artifact_kind`, `complete`, `unit_system=atomic`, `index_base=0`, and
`artifact_id`. Readers accept their own major version and compatible older
minor versions, reject newer unknown minor versions, and report unknown major
versions explicitly. Migration will create a new immutable artifact rather
than edit one in place.

| Artifact | Schema | Required top-level groups |
| --- | --- | --- |
| prepared reference | `aion.reference` `1.0.0` | `meta`, `configuration`, `reference` |
| trajectory | `aion.trajectory` `1.0.0` | `meta`, `configuration`, `reference`, `time`, `source`, `observables`, `diagnostics`, `events`, `restart` |
| Wilson stationary state | `aion.wilson-stationary-state` `1.0.0` | `meta`, `configuration`, `reference`, `source`, `state`, `observables`, `diagnostics` |
| Wilson trajectory | `aion.wilson-trajectory` `1.0.0` | `meta`, `configuration`, `reference`, `stationary_state`, `time`, `source`, `observables`, `diagnostics`, `events`, `restart` |
| source history | `aion.source-history` `2.0.0` | `meta`, `configuration`, `time`, `source` |
| checkpoint | `aion.checkpoint` `1.0.0` | `meta`, `configuration`, `reference`, `time`, `source`, `state`, `events`, `restart` |
| Wilson checkpoint | `aion.wilson-checkpoint` `1.0.0` | `meta`, `configuration`, `reference`, `stationary_state`, `time`, `source`, `state`, `observers`, `events`, `restart` |
| Casida result | `aion.casida` `1.0.0` | `meta`, `configuration`, `reference`, `roots`, `selection` |
| kick spectrum | `aion.kick-spectrum` `1.0.0` | `meta`, `configuration`, `source`, `time`, `frequency`, `response` |

Every dataset must declare `unit` and `physical_dimension`. Direct generic
datasets `/observables/current`, `/observables/dipole`, and
`/observables/energy` are forbidden. Observable definitions instead carry a
namespaced definition ID, originating formulation, equation, physical
dimension, atomic unit, tensor shape, sampling location, and decomposition
labels. Each record carries exact integer step, atomic-unit time, and an
immutable float64 or complex128 value.

`stamp_artifact` creates only a schema skeleton. WP2 adds same-directory
transactional `.partial` construction and no-overwrite atomic publication for
complete prepared references and compiled source histories. Appendable
trajectory streams, resolved compression policy, immutable checkpoints,
controlled failure artifacts, restart-boundary authentication, and lineage
are implemented by WP5. Dense state arrays use gzip plus shuffle; small
scalar/vector streams remain uncompressed.

Casida artifacts store the prepared-reference fingerprint, optional reference
artifact path and checksum, solver/version/configuration (including root
count, convergence tolerance, and maximum iterations), every excitation
energy, length-form oscillator strength, transition dipole and direction,
convergence flag, and optional polarization-resolved resonance selection.
Kick-spectrum artifacts store an exact trajectory or imported-parent checksum,
source metadata, event and observable definition IDs, impulse and baseline,
complete transform configuration, time/frequency resolution metadata, full
complex dipole-domain polarizability, optional independently reconstructed
current-domain polarizability and residual, directional absorption, and
oscillator-strength density. Scientific identities exclude relocatable paths
but include parent checksums and all numerical content.

## 6. `status.json`

The non-authoritative monitoring record has schema `aion.status` version
`1.0.0`. Its strict fields are run and simulation IDs, phase, global and last
accepted step, total steps, latest checkpoint, wall time, ETA, update time,
host, PID, and an optional structured failure. Step ordering, nonnegative
times, SHA-256 syntax, phase names, and exact field sets are validated.
Completed HDF5 artifacts remain authoritative. The runner atomically publishes
status at phase changes and approximately every 30 seconds during propagation.

## 7. API and CLI boundary

The top-level public API exports ordinary and Wilson configuration types and:

```text
prepare_reference  load_reference  build_simulation
run                resume          load_trajectory
prepare_wilson_stationary_state     load_wilson_stationary_state
```

`prepare_reference` and `load_reference` are implemented by WP2, formulation
construction by WP3, the propagation-bound `build_simulation` by WP4, and
`run`, `resume`, and `load_trajectory` by WP5. Preparation
runs one validated RKS calculation; loading authenticates the portable data and
runtime dependency contract without rerunning SCF. Each later simulation gets
a separate backend workspace, which reconstructs PySCF/GPU4PySCF and transfers
the reference exactly once. A built simulation binds the compiled source,
selected gauge/formulation, occupied-orbital state, common SCEM propagator,
backend workspace, event schedule, and typed observable calculators without
starting propagation. Its `step()` method advances one accepted interval in
memory; it performs no event handling or persistent I/O. `run` is the normal
one-process execution boundary. It applies exact events, evaluates independent
schedules, accumulates every accepted interval's source work, publishes status
and checkpoints, and emits either `trajectory.h5` or a controlled
`trajectory.failed.h5`. `resume` authenticates and reconstructs the saved
reference, source, state, work, and event IDs before advancing a child segment.

For a Wilson input, `build_simulation` authenticates the portable stationary
state and constructs a bounded exact-action cache plus the stateful
contravariant-density propagator. `run` streams action-owned dipole, uniform
source current, charge, energy components, source power/work, Ward,
weak-continuity, metric, and nonlinear diagnostics. A checkpoint contains the
accepted density rather than an occupied-orbital gauge representative and
preserves observer schedule state and accumulated Gauss-node work. Resume
reconstructs the same integer boundary and produces a lineage-linked child
trajectory.

The `aion` command exposes `prepare`, `run`, `resume`, `inspect`, and `export`.
`prepare` and `run` accept `--validate-only`, print the fully resolved TOML and
scientific ID, and perform no numerical work in that mode. `prepare` dispatches
between ordinary reference preparation and Wilson stationary preparation by
schema; `run`, `resume`, `inspect`, and `export` likewise dispatch by typed
configuration or artifact schema. Ordinary reference preparation runs RKS and
transactionally publishes the configured reference artifact.
`inspect` validates and reports a completed HDF5 artifact header. `export`
writes one CSV per independently sampled observable plus a provenance manifest.
Exit status 2 denotes configuration/schema failure, 130 denotes graceful
cancellation, and 1 denotes another controlled Aion failure.

Spectroscopy is an explicit named expert API in `aion.spectroscopy`; it is not
silently run by propagation or added to the small top-level API. `run_casida`
reconstructs the qualified CPU PySCF mean field from the prepared reference
without rerunning SCF. `kick_spectrum_from_trajectory` accepts only a completed
trajectory with exactly one kick and dipole/current samples at every
post-event endpoint. `save_casida_result` and `save_kick_spectrum` publish new
immutable HDF5 files and refuse overwrite. Legacy format knowledge remains in
the separate `CALCULATIONS/legacy/conversion` scripts.

## 8. Quality and test tiers

Ruff owns formatting and linting. Mypy is strict over public configuration,
storage, observable, CLI, and orchestration boundaries; array kernels may gain
only narrow documented exceptions. Pytest markers are `fast`, `integration`,
and `gpu`. Fast tests may use eight xdist workers with BLAS/OpenMP capped to one
thread. Molecular CPU integrations run serially with up to eight library
threads. GPU tests run serially on the physical device and may never silently
fall back to CPU.
