# aion

Prototype real-time electronic dynamics code for fixed Gaussian AO bases.

The current useful development target is deliberately narrow:

- closed-shell PySCF RKS references,
- molecular Gaussian basis sets,
- pure length gauge with spatially uniform electric fields,
- occupied-orbital propagation in the AO basis,
- adiabatic local or semi-local DFT, currently tested mainly with PBE,
- strict self-consistent exponential midpoint propagation,
- optional GPU execution through GPU4PySCF and CuPy.

Older leapfrog, generalized Crank-Nicolson, and EP-PC1 paths remain in the tree
for comparison, but new development should use the SCEM path unless there is a
specific reason to compare integrators.

## Core API

The main entry point is:

```python
from aion import LengthGaugeCNRTTDDFT

rt = LengthGaugeCNRTTDDFT.from_ground_state(
    mf,
    field=lambda t: [0.0, 0.0, 0.0],
    backend="cpu",
)
coeff = rt.apply_delta_kick(rt.initial_coefficients(), [0.0, 0.0, 1.0e-3])

for coeff, rec in rt.propagate_scem(coeff, dt=0.05, nsteps=10):
    print(rec.time, rec.dipole)
```

For GPU runs, construct the PySCF object with GPU4PySCF first:

```python
mf = mf.density_fit().to_gpu()
rt = LengthGaugeCNRTTDDFT.from_ground_state(mf, field, backend="gpu")
```

Run GPU examples and tests through the project launcher:

```bash
/home/cgs/00_WORK/Projection_Code/aion/tools/gpu-python -m pytest -q tests/test_gpu_backend.py
```

The launcher sets the local `PYTHONPATH`, CUDA library paths, and cache
directories needed by this workstation.

## Supported Envelope

The validated production-like path is:

- `LengthGaugeCNRTTDDFT.propagate_scem` or `run_scem`,
- closed-shell RKS,
- pure LDA/GGA functionals,
- real AO bases,
- `real_density_for_veff=True`,
- CPU PySCF or GPU4PySCF density-fitted PBE.

Hybrid TDDFT, TDHF, and full complex-density GPU exchange are not supported
yet.  The GPU path intentionally rejects unsupported combinations rather than
silently running a physically different approximation.

## Tests

CPU tests:

```bash
/home/cgs/00_WORK/Projection_Code/miniconda3/bin/python -m pytest -q
```

GPU tests:

```bash
/home/cgs/00_WORK/Projection_Code/aion/tools/gpu-python -m pytest -q tests/test_gpu_backend.py
```

## Examples

Small examples are kept as runnable documentation.  They should not become the
main implementation.

- `examples/gpu_backend/run_h2_gpu_scem.py`: tiny GPU smoke run.
- `examples/gpu_backend/benchmark_scem_cpu_gpu.py`: CPU/GPU timing harness.
- `examples/gauge_p0_toy/run_p0_toy_gauge_compare.py`: pure Peierls P0 toy
  propagation comparing length, mixed, and velocity gauges on source
  observables, graph currents, and continuity diagnostics.  Use
  `--model hubbard` to exercise the nonlinear midpoint solve.
- `examples/small_molecule_references/run_diatomic_kick_spectrum.py`: reference
  diatomic kick spectrum runner with CN, EP-PC1, and SCEM options.

Large exploratory campaigns and generated trajectories are intentionally kept
out of version control.

## Current Numerical Status

See `docs/scem_current_status.md` for the current integrator choice, timestep
guidance, GPU status, and near-term development plan.
