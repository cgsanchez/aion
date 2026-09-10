# Static uniform-magnetic benchmark API

The reusable entry point is `aion.workflows.run_magnetic_benchmark`. It takes
an already converged `PreparedReference` and a fully typed
`MagneticBenchmarkConfig`; it does not perform SCF, propagate a state, or
modify the reference. NumPy/PySCF and CuPy/GPU4PySCF execute the same blocked
AO contractions. Device arrays cross to immutable host records only when the
result and its diagnostics are finalized.

```python
from aion.config import BackendConfig, BackendKind
from aion.electromagnetism import UniformMagneticField
from aion.workflows import MagneticBenchmarkConfig, run_magnetic_benchmark

config = MagneticBenchmarkConfig(
    magnetic_fields=(UniformMagneticField.from_tesla((0.0, 0.0, 30.0)),),
    backend=BackendConfig(BackendKind.GPU, device_index=0),
    block_size=2048,
)
result = run_magnetic_benchmark(
    prepared_reference,
    config,
    output_path="h2-bz-30T.magnetic.h5",
)
kinetic_exact = result.matrix("0000/kinetic/exact")
```

The field tuple is ordered and every field has a matching affine gauge. If no
gauges are supplied, symmetric gauges are constructed. `families` selects the
one-electron, spatial-connection, and generic local-potential hierarchies. A
local-potential request must include stable provider identities in the config
and the corresponding providers in the workflow call. Block size, optional
memory ceiling, charge, mass, hbar, direct-oracle selection, grid policy, and
validation thresholds are all part of the normalized configuration and its
scientific hash.

Progress is synchronous and callback-based. The callback receives immutable
`MagneticProgress` values at preparation, AO-block, and finalization
boundaries; the workflow does not start a monitor or polling loop.

The returned `MagneticBenchmarkResult` contains immutable, path-addressable
host matrices and scalar diagnostics. Paths begin with a zero-padded field
index. For example, `0000/overlap/exact`, `0000/kinetic/first_pC`, and
`0000/spatial_connection/first_C` are authoritative stored components.
Definitions and physical-unit metadata accompany every matrix.

Diagnostics include validation residuals, unit-aware Frobenius and maximum-
element norms, same-anchor and intersite norms, and automatic field-reversal
checks whenever the requested field list contains an exact `+B, -B` pair.
`analytic_uniform_magnetic_overlap` is a CPU qualification oracle based on
libcint AO-pair Fourier integrals; it is intentionally separate from the
real-space benchmark workflow.

`save_magnetic_benchmark` publishes the HDF5 artifact transactionally and
refuses to overwrite a completed result. `load_magnetic_benchmark` validates
the schema and recomputes the content identity, detecting changes to matrices,
configuration, metadata, or diagnostics. `inspect_magnetic_benchmark` returns
a compact scalar summary while leaving the matrix artifact authoritative.

Physical GPU execution uses `tools/gpu-python`. Its host numerical-library
thread count defaults to one and can be changed deliberately with
`AION_HOST_THREADS`; a requested GPU backend fails rather than falling back to
CPU AO evaluation.
