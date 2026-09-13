# Aion

Aion is a reusable Python library for gauge-covariant real-time TDDFT in
finite molecular atomic-orbital bases. Version `0.2.0.dev7` is a clean break
from the archived research prototype.

WP1 established the strict package/configuration/storage contracts, WP2 the
common static electronic problem and prescribed EM input, WP3 the complete
instantaneous formulation physics, WP4 the shared propagation layer, WP5
reusable execution and persistence, and WP6 spectroscopy and legacy conversion:

- immutable, strictly validated Python configurations;
- deterministic normalized TOML and lossless scientific identities;
- explicit atomic-unit, electromagnetic-origin, and fixed-time-grid types;
- strict NumPy/CuPy backends, residency assertions, and mutable per-simulation
  workspaces with no GPU fallback;
- validated CPU/GPU PySCF RKS preparation for pure LDA/GGA all-electron
  molecules, with optional scientifically identified RI density fitting;
- portable read-only references containing ground state, exact DFT grid, AO
  operators, nuclei, atomic anchors/topology, and authenticated fingerprints;
- transactional immutable `reference.h5` and compiled-source HDF5 I/O;
- potential-first uniform sources, additive composition, analytic envelopes,
  sin² pulses normalized to peak electric field, exact endpoint/midpoint
  compilation through the analytic field derivative, consistent LG/VG/P0
  node-link data, and exact kick definitions;
- explicit `(S, H_eom, omega)` implementations of bare LG, bare VG, P0, and
  P0+E1 in a continuous covariant gauge family spanning the length and
  velocity endpoints;
- Wilson metrics and projected connections, inverse-dressed pure-DFT matrix
  builds, independently fingerprinted E1 tensors, and the E1 potential placed
  exactly once in the connection;
- formulation-owned dipoles and primary/source currents, separately named
  full-continuity and action-split P0+E1 pair currents, variational E1 current
  pieces, and ambient mechanical diagnostics;
- generic and uniform local mechanical-current contractions with explicit
  paramagnetic/diamagnetic pieces;
- qualified matter/generator energy ledgers, complete bare-VG mechanical
  kinetic terms, source couplings, analytic matter rates, Ward residuals, and
  analytic length-gauge generator rates; and
- dependency-declared observable calculators with independent schedules and
  persistent formulation-qualified definitions;
- occupied-orbital states and one density-self-consistent exponential-midpoint
  engine shared by all four formulations;
- direct nonorthogonal fixed-metric transport and connection-aware covariant
  transport with exact uniform-field site parallel transport, with no evolving
  Löwdin frame;
- selectable Cayley [1/1] and Padé [2/2] rational maps, adaptive damped Picard
  iteration, roundoff-only Hermitian cleanup, and hard nonlinear failure;
- right-Cholesky cross-metric correction with raw/corrected link diagnostics
  and an explicitly diagnostic-only correction-disable mode;
- independently versioned HDF5 schemas, typed observable records, and a strict
  `status.json` record;
- exact formulation-owned kick maps with pre/post event records and
  restart-idempotent potential jumps;
- independently scheduled endpoint observables, interval-centered analytic
  power/Ward streams, and midpoint source-work accumulation without forcing
  complete energy evaluation;
- streaming transactional trajectories, immutable compressed checkpoints,
  controlled failure artifacts, safe boundary cancellation, authenticated
  reconstruction-first restart, and lineage-aware observable stitching;
- explicit CSV export and independent-trajectory comparison manifests;
- a thin operational `prepare`, `run`, `resume`, `inspect`, and `export` CLI;
- a structured PySCF Casida-TDDFT wrapper with length-form oscillator
  strengths, transition dipoles/directions, and explicit polarization-bright
  resonance selection;
- positive-frequency trapezoidal kick transforms with explicit baseline,
  Fourier sign, normalization, rectangular window, exponential damping,
  zero-padding, intrinsic resolution, grid spacing, and Nyquist metadata;
- dipole-domain polarizability as the canonical kick response, independently
  reconstructed current-domain polarizability with finite-record endpoint
  terms, directional absorption, and oscillator-strength density; and
- immutable `aion.casida` and `aion.kick-spectrum` HDF5 artifacts linked to
  exact parent checksums, with one-way CSV/NPZ conversion confined to the
  separate `CALCULATIONS/legacy/conversion` tree.

The in-memory `BuiltSimulation.step()` boundary remains available for expert
use, while ordinary execution should use `run()` so events, work, schedules,
status, and restart state are recorded consistently.

## Validated domain

The first `0.2` implementation targets fixed-nucleus, finite, all-electron
molecules; closed-shell spin-summed RKS; pure LDA/GGA functionals; prescribed
uniform electric fields; fixed timesteps; and float64/complex128 CPU and GPU
execution. Pseudopotentials, hybrids, moving nuclei, periodic systems, Maxwell
backreaction, and adaptive timesteps are rejected rather than approximated.

## Python API

Only names in `aion.__all__` define the stable top-level API. The principal
workflow functions are:

```python
from aion import (
    build_simulation,
    load_reference,
    load_trajectory,
    prepare_reference,
    resume,
    run,
)
```

Expert contracts are available from their named domains:

```text
aion.config
aion.backends
aion.electronic_structure
aion.electromagnetism
aion.formulations
aion.propagation
aion.observables
aion.io
aion.spectroscopy
aion.workflows
```

Start with the [Aion 0.2 user guide](docs/user_guide.md) and the three
[runnable examples](examples/README.md). See
[configuration_and_schema_contracts.md](docs/configuration_and_schema_contracts.md)
for the exact configuration, identity, artifact, observable, and CLI contracts.
The physics and planned implementation are specified in
[theory_and_implementation.pdf](docs/theory_and_implementation.pdf) and
[refactor_implementation_plan.md](docs/refactor_implementation_plan.md).

## Managed development environment

Use the project-owned Python 3.12 prefix without activating Conda:

```bash
/home/cgs/01_TOOLS/EasyBuild/conda/bin/conda run \
  -p /home/cgs/01_TOOLS/EasyBuild/conda/envs/aion \
  python -m pytest -n 8 -m fast
```

Quality gates are:

```bash
/home/cgs/01_TOOLS/EasyBuild/conda/bin/conda run \
  -p /home/cgs/01_TOOLS/EasyBuild/conda/envs/aion \
  ruff check src tests

/home/cgs/01_TOOLS/EasyBuild/conda/bin/conda run \
  -p /home/cgs/01_TOOLS/EasyBuild/conda/envs/aion \
  ruff format --check src tests

/home/cgs/01_TOOLS/EasyBuild/conda/bin/conda run \
  -p /home/cgs/01_TOOLS/EasyBuild/conda/envs/aion \
  mypy
```

Fast tests use at most eight workers. Molecular CPU integrations will run
serially with up to eight numerical-library threads, while physical-GPU tests
will run serially through `tools/gpu-python`.

The molecular and physical-GPU gates are respectively:

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  /home/cgs/01_TOOLS/EasyBuild/conda/bin/conda run \
  -p /home/cgs/01_TOOLS/EasyBuild/conda/envs/aion \
  pytest -q -m integration

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  tools/gpu-python -m pytest -q -m gpu
```
