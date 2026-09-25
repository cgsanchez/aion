# Aion 0.2 user guide

## Scope

Aion 0.2 propagates fixed-nucleus, finite, all-electron molecules with
closed-shell spin-summed RKS and pure LDA/GGA functionals. The original
runtime supports prescribed uniform electric fields on fixed time grids in
bare length gauge, bare velocity gauge, and the P0 and P0+E1 covariant
models. The Phase Two runtime additionally supports the exact straight-Wilson
action for analytic affine electromagnetic sources, including a uniform
electric field, a uniform magnetic field, and the induction field required by
a linearly time-dependent magnetic field. CPU and NVIDIA GPU backends use
float64/complex128 and never silently fall back between devices.

The release deliberately rejects pseudopotentials and nonlocal ionic
operators, hybrids, moving nuclei, periodic boundary conditions, Maxwell
backreaction, adaptive time steps, and spatially nonuniform sources. Magnetic
sources are accepted only by the exact/reduced Wilson configuration family;
they remain invalid for the original bare/P0/P0+E1 runtime. The source-provider
interface preserves room for later nonuniform sources, but EELS, moving
charges, and general beyond-dipole fields are not claimed by 0.2.

The reusable exact workflow is numerically qualified for the exact
Kohn--Sham LDA action on equilateral H3+/cc-pVDZ with an unpruned level-4
grid and `weigend` auxiliary basis. Exact Hartree and pure-GGA branches are
implemented descendants of the accepted Chapter 13 action, but their
reusable molecular transfer belongs to P2-2; reduced P0/E1 dynamics remains
disabled until P2-6. This distinction is recorded so an available type is not
mistaken for reviewed application evidence.

## Workflow

An ordinary bare/P0/P0+E1 calculation has two immutable stages. First,
`prepare_reference()` (or `aion prepare`) performs the ground-state RKS
calculation and stores its exact
quadrature grid, AO operators, density, orbitals, nuclei, atom anchors, and
fingerprints in `reference.h5`. Second, each `SimulationConfig` binds that
authenticated reference to one source, formulation, fixed grid, backend, and
output policy. `run()` publishes a transactional trajectory and optional
checkpoints; `resume()` reconstructs the scientific state from a checkpoint
and publishes a new lineage-linked trajectory segment.

Use typed objects from `aion.config` in Python. TOML is a strict serialization
of the same types, not a second configuration model. Paths and metadata do not
change scientific identities, while the reference digest, formulation,
source, time grid, numerical tolerances, events, and backend kind do.

Reference preparation may explicitly enable RI density fitting with
`density_fitting=True` and a named `auxiliary_basis`. Both are part of the
scientific identity and reconstruction contract because density fitting
changes the approximated Coulomb functional; Aion never enables it merely as
an unrecorded performance optimization.

`ElectronicStructureConfig.grid_pruning` defaults to `pyscf_default`. Select
`none` explicitly when an unpruned PySCF grid is part of the scientific
realization; that choice enters the reference identity and is restored by
live CPU/GPU reconstruction.

An exact-Wilson calculation adds an immutable stationary-state stage:

1. prepare an ordinary `ReferenceConfig`;
2. prepare a `WilsonStationaryConfig`, which solves the source-fixed exact
   generalized eigenproblem and writes a portable
   `aion.wilson-stationary-state` artifact; and
3. run a `WilsonSimulationConfig` linked by both reference and stationary-state
   content fingerprints.

Stationary preparation is deliberately CPU-hosted. Its artifact contains
backend-neutral complex128 arrays and can be consumed directly by CPU or GPU
dynamics. The runtime authenticates the reference, action, grid, RI auxiliary
space, metric, precision, and stationary-state fingerprints before the first
step. `WilsonSimulationConfig.initial_source_policy` defaults to `matched`,
which also requires the full stationary and simulation source definitions to
agree. `continuous_density_quench` is the narrow expert alternative for
starting a run with a finite electric or magnetic-field-rate quench while
leaving the density continuous: the initial magnetic field, spatial gauge
representative, origin, and metric must remain exactly the same. It cannot be
used for a magnetic-field jump or as a substitute for an exact discontinuous
vector-potential kick.

The runnable workflows in [`examples/`](../examples/README.md) are the
shortest starting points. They cover reference/Casida response, all four
comparison formulations under one pulse, and CLI checkpoint/resume.

## Sources and time grids

`Sin2VectorPotentialPulseConfig` defines the reduced vector potential first;
the electric field and its derivative are analytic derivatives. Its compact
support clamps the vector potential exactly to zero at both ends. Set
`require_zero_impulse=True` when the pulse must return LG and VG to their
field-free source. Always build its grid with `pulse_aligned_time_grid()` so
the support boundary is an exact integer step. Kicks are exact, idempotent
events on state boundaries and have formulation-owned maps.

`AffineElectromagneticSourceConfig` is the exact-Wilson source contract. It
combines zero or the same potential-first compact electric pulse, an optional
constant electric-field offset at the electromagnetic origin, a magnetic
field at an explicit reference time, a constant analytic magnetic-field
derivative, and a symmetric or Landau affine gauge. Sampling at arbitrary
Gauss nodes returns mutually consistent `E_origin(t)`, `B(t)`, and
`dB(t)/dt`; the induction field is not supplied independently. The first
schema intentionally does not encode an arbitrary direct electric-field
envelope.

## Formulations and observables

Bare LG and bare VG use the shared fixed-metric SCEM integrator. P0 and P0+E1
use the connection-aware SCEM integrator, including the analytic projected
connection and cross-metric correction. P0 and P0+E1 expose a continuous
uniform-field gauge family through `velocity_fraction`: zero is length gauge,
one is velocity gauge, and a value strictly between them is a mixed
representation. These are representations of the same covariant model and
should agree to numerical tolerance. For example:

```python
from aion.config import FormulationConfig, FormulationKind

formulation = FormulationConfig(
    FormulationKind.P0_E1,
    velocity_fraction=0.5,
)
```

The parameter is constant throughout a simulation. Aion mixes the primitive
scalar and vector potentials (and their projected node/link data), then builds
the Wilson metric, connection, Hamiltonian, events, and observables from that
single representation. It does not interpolate completed length- and
velocity-gauge Hamiltonians.

Dipoles, currents, energies, and diagnostics are qualified by formulation.
The primary/source current is the current variationally conjugate to the
prescribed source and is the one used in source power. Mechanical current is
also recorded with separate paramagnetic and diamagnetic pieces. For P0+E1,
the ambient projected mechanical current is a comparison diagnostic and is
not substituted for the covariant source current. Energy records separate
canonical/mechanical kinetic pieces, electron-nuclear, Hartree, XC, nuclear
repulsion, scalar-source and E1 couplings, matter energy, generator energy,
analytic power, accumulated source work, and Ward residuals.

Energy is disabled by default except for required initial/final baselines.
Use independent `StepSchedule` values for dipole/current, energy, diagnostics,
source samples, checkpoints, and optional density snapshots. Sparse schedules
reduce I/O and expensive DFT energy evaluations without changing propagation.

The exact runtime propagates the contravariant AO density with the accepted
self-consistent two-node fourth-order Gauss--Magnus method and a Padé `[2/2]`
coefficient link. There is no Löwdin/Cholesky propagation frame and no metric
projection or cleanup. Its primary current, source power, energy components,
dipole, charge, Ward identity, and weak-continuity diagnostic are all derived
from the selected exact action. Gauss-node source power is integrated on every
accepted interval even when endpoint energy output is sparse.

## Spectroscopy

`aion.spectroscopy.run_casida()` reconstructs PySCF Casida-TDDFT from a saved
reference without rerunning SCF. Resonances may be selected explicitly by
zero-based root or as the lowest root above a polarization-projected
brightness threshold. Structured output stores every root and convention.
The requested root count, convergence tolerance, and maximum solver iteration
count are explicit `CasidaConfig` fields persisted with the result.

Kick spectra use a positive-frequency continuous-time trapezoidal transform.
The finite-record baseline, rectangular window, optional exponential damping,
zero padding, intrinsic resolution, native and padded grid spacings, and
Nyquist energy are all explicit. Dipole response is canonical; an independent
current-domain reconstruction retains its finite-record endpoint terms and
stores the complex residual. Zero padding interpolates the displayed grid and
does not improve the intrinsic spectral resolution, approximately
`2*pi/T` for record duration `T`.

## Execution and inspection

The CLI commands are:

```text
aion prepare INPUT.toml [--validate-only]
aion run INPUT.toml [--validate-only]
aion resume CHECKPOINT.h5 [--output DIRECTORY]
aion inspect ARTIFACT.h5
aion export TRAJECTORY.h5 DIRECTORY [--observable DEFINITION_ID ...]
```

`prepare` dispatches by schema: it accepts both `aion.reference-input` and
`aion.wilson-stationary-input`. Likewise `run`, `resume`, `inspect`, and
`export` dispatch between ordinary and exact-Wilson artifacts. A minimal
expert-Python lifecycle is:

```python
from aion import (
    build_simulation,
    load_reference,
    load_wilson_stationary_state,
    prepare_wilson_stationary_state,
    run,
)

reference = load_reference("reference.h5")
state = prepare_wilson_stationary_state(stationary_config, reference)
state.save()
state = load_wilson_stationary_state("stationary.h5")
simulation = build_simulation(simulation_config, reference, stationary_state=state)
trajectory = run(simulation)
```

The two typed configurations may be serialized with `dumps_config`; reloading
the resolved TOML must preserve equality and the scientific ID. The lower
level `BuiltWilsonSimulation.step()` interface advances exactly one accepted
interval in memory and is intended for expert orchestration and tests; `run()`
is the normal streaming, checkpointing, cancellation-aware execution boundary.

Long campaigns should launch one run per process and live outside the Aion
repository. A run directory is immutable: choose a new directory for a new
scientific run or recovery experiment. `status.json` is the lightweight
monitoring surface. Completed HDF5 artifacts can be checked with `aion
inspect`; CSV export is explicit and includes a manifest.

For GPU work, use the project launcher and request a GPU backend in the input:

```bash
tools/gpu-python -m aion run simulation.toml
```

The managed development/test commands and dependency policy are in the root
README. Exact schemas, definition IDs, and normalized TOML fields are in
[`configuration_and_schema_contracts.md`](configuration_and_schema_contracts.md).
The derivation and implementation details are in
[`theory_and_implementation.pdf`](theory_and_implementation.pdf).
