# Aion

Aion is a reusable Python library for gauge-covariant real-time TDDFT in
finite molecular atomic-orbital bases. Version `0.2.0.dev1` is a clean break
from the archived research prototype.

WP1 establishes the non-numerical contracts on which the new implementation
will be built:

- immutable, strictly validated Python configurations;
- deterministic normalized TOML and lossless scientific identities;
- explicit atomic-unit, electromagnetic-origin, and fixed-time-grid types;
- independently versioned HDF5 artifact schemas and typed observables;
- a strict `status.json` record;
- physics-oriented package domains and a small typed public API;
- a thin `aion` CLI with `prepare`, `run`, `resume`, `inspect`, and `export`.

The numerical reference builder, formulations, propagators, runners, and
spectroscopy workflows are intentionally unavailable at this milestone. Their
public shells fail explicitly instead of importing or executing archived draft
code.

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
aion.workflows
```

See [configuration_and_schema_contracts.md](docs/configuration_and_schema_contracts.md)
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
