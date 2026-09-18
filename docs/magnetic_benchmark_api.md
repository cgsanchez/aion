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

## Exact one-electron pair-analysis API

The qualification-only pair path accepts an occupancy-independent
`OneElectronAOReference`; it never constructs an SCF state. The full
`evaluate_magnetic_one_electron_matrices` entry point returns the exact and
B1/B2 hierarchy. Field scans that need only exact physics use
`evaluate_exact_static_magnetic_one_electron_matrices`, which shares the same
blocked contraction kernel but deliberately omits Taylor-only contractions.
It retains the exact overlap, nuclear attraction, all four exact kinetic
sectors, the three `F=1` anchored-vector sectors, and the stable phase-spread
observable

`epsilon_F = sqrt(integral |e_mu* e_nu| |F-1|^2 / integral |e_mu* e_nu|)`.

`static_magnetic_diagnostic_models` constructs four declared static models:

- P0: endpoint link only;
- form-factor only: exact triangle factor with anchored vectors deleted;
- anchored-vector only: `F=1` with exact anchored-vector sectors;
- full exact: exact triangle factor and anchored vectors.

The middle two are diagnostic sector deletions, not action-level dynamical
formulations. Both endpoint-removed and endpoint-dressed matrices are
returned so gauge-invariant internal changes and generalized spectra can be
examined without conflating them.

The reusable analysis functions resolve matrix changes into
onsite-diagonal, same-anchor off-diagonal, and intersite partitions; enumerate
atom-pair/angular-momentum blocks and their singular values; construct
diagonal-scaled element changes; and compare generalized eigenvalues and
cumulative coefficient-space subspaces. Coefficient subspace angles require
the same AO ordering and gauge representative on both sides.

## Exact time-dependent one-electron API

`UniformMagneticSourceSample` is the immutable source boundary for the WP6
fixed-centre problem. It combines a uniform magnetic field, its analytic time
derivative, a uniform electric field at the declared origin, and either the
symmetric or an affine Landau representative. Its physical electric field
always contains the Maxwell-required affine induction term. Gauge-dependent
scalar-potential terms are generated together with the selected vector
potential rather than supplied independently.

`evaluate_exact_wilson_one_electron_sample` consumes an `AOQuadrature` and one
source sample. It returns the exact metric, mechanical matrix, temporal
connection, independently differentiated metric rate, and ordinary-
derivative matrix `K - 1j*hbar*omega_t`. The production temporal connection
uses exact endpoint factorization plus analytic bare AO moments and a blocked
real-space correction. Independent all-grid direct and factorized values are
retained as numerical oracles.

`prepare_exact_one_electron_model_context` precomputes source-independent
first-order magnetic and uniform-electric tensors. Passing this context and
an exact sample to `exact_one_electron_model_triples` constructs the declared
EX, P0, E1, gB1, B1, and C1 model triples without mixing action levels. The
exact model can also be exposed directly through
`exact_wilson_one_electron_triple`.

`propagate_linear_matrix_history` advances any prequalified sequence of Aion
`EOMTriple` values. Endpoint metrics and midpoint triples remain explicit.
The raw midpoint Padé map and the right-Cholesky cross-metric correction are
both diagnosed; the correction does not replace the supplied connection.
`generalized_spectral_trajectory` supplies an independent fixed-matrix
reference. Persistent run/resume and action-derived observables remain later
workflow stages and are not implied by this linear API.
