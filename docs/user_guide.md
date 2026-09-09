# Aion 0.2 user guide

## Scope

Aion 0.2 propagates fixed-nucleus, finite, all-electron molecules with
closed-shell spin-summed RKS and pure LDA/GGA functionals. It supports
prescribed uniform electric fields on fixed time grids in bare length gauge,
bare velocity gauge, and the P0 and P0+E1 covariant models. CPU and NVIDIA GPU
backends use float64/complex128 and never silently fall back between devices.

The release deliberately rejects pseudopotentials and nonlocal ionic
operators, hybrids, moving nuclei, periodic boundary conditions, Maxwell
backreaction, adaptive time steps, and magnetic or spatially nonuniform
sources. The covariant interfaces preserve room for later nonuniform sources,
but that physics is not claimed by 0.2.

## Workflow

An Aion calculation has two immutable stages. First, `prepare_reference()` (or
`aion prepare`) performs the ground-state RKS calculation and stores its exact
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

The three runnable workflows in [`examples/`](../examples/README.md) are the
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

## Formulations and observables

Bare LG and bare VG use the shared fixed-metric SCEM integrator. P0 and P0+E1
use the connection-aware SCEM integrator, including the analytic projected
connection and cross-metric correction. P0+E1 is available in length and
velocity representations; those are two representations of the same
covariant model and should agree to numerical tolerance.

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
