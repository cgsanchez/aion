# GPU Backend Status

This is the first real GPU backend for the SCEM real-time TDDFT path.

The intended execution path is:

1. PySCF or GPU4PySCF provides the converged ground state.
2. `LengthGaugeCNRTTDDFT(..., backend="gpu")` stores the dense propagation
   matrices as CuPy arrays.
3. The SCEM midpoint loop performs dense algebra on the GPU:
   - orthonormal Hamiltonian construction,
   - Hermitian diagonalization,
   - exponential action on occupied coefficients,
   - density construction,
   - residual norms.
4. GPU4PySCF builds `V_eff[P]` for each midpoint density.
5. Per-step records copy only scalar diagnostics and small observables to host.

Run GPU examples and tests through:

```bash
/home/cgs/00_WORK/Projection_Code/aion/tools/gpu-python ...
```

The launcher sets:

- `PYTHONPATH` to the local `aion/src`,
- CuPy cache directories under `/tmp`,
- `LD_LIBRARY_PATH` for the pip-installed CUDA libraries,
- `LD_PRELOAD` for the local cuSOLVER library needed by GPU4PySCF,
- `AION_GPU_LAUNCHER=1`, which enables the GPU pytest file.

The current validated support envelope is deliberately narrow:

- closed-shell RKS,
- fixed real AO basis,
- length gauge,
- SCEM propagation,
- pure LDA/GGA-type functionals,
- `real_density_for_veff=True`,
- GPU4PySCF mean-field objects, usually made with `.density_fit().to_gpu()`.

Hybrid TDDFT, TDHF, and full complex-density GPU `get_veff` are still blocked.
For pure local/semi-local DFT in a real AO basis, the imaginary antisymmetric
part of the Hermitian density does not enter the scalar density/J path, so the
current backend passes the real Hermitian density to GPU4PySCF.

Useful checks:

```bash
/home/cgs/00_WORK/Projection_Code/miniconda3/bin/python -m pytest -q
/home/cgs/00_WORK/Projection_Code/aion/tools/gpu-python -m pytest -q tests/test_gpu_backend.py
```

Tiny smoke example:

```bash
/home/cgs/00_WORK/Projection_Code/aion/tools/gpu-python \
  examples/gpu_backend/run_h2_gpu_scem.py \
  --basis cc-pVDZ --dt 0.05 --nsteps 10
```

## Transfer Audit

The GPU hot path for `run_scem` is intentionally lower-transfer than
`propagate_scem`.

`propagate_scem` remains the convenient trajectory-producing API. It yields a
coefficient array and a full diagnostic record at every step. On the GPU this
means:

- a device coefficient copy at every yielded step,
- scalar transfers for electron number, norm errors, and field coupling,
- a three-component dipole transfer,
- optional energy evaluation if `record_energy=True`.

`run_scem` is the benchmark/production timing API. It uses the same SCEM
midpoint solve but does not yield per-step snapshots. It computes diagnostics
only at the final time. During the loop, the remaining GPU-to-CPU transfers are
the scalar convergence residuals needed to decide whether the nonlinear
midpoint solve is converged.

The main runtime GPU path is therefore:

```text
C, P, H, S, S^(1/2), S^(-1/2)        CuPy device arrays
midpoint diagonalization              CuPy/cuSOLVER
density construction                  CuPy GEMM
V_eff[P]                              GPU4PySCF
SCEM convergence residuals            CuPy reduction, scalar copied to host
final diagnostics                     copied once at the end
```

Known non-hot-path host movement:

- initialization copies some PySCF data through host before placing it on the
  selected backend,
- `record_energy=True` may call PySCF energy machinery and should be treated as
  diagnostic rather than benchmark timing,
- pure local/semi-local DFT passes the real Hermitian density to `get_veff`.

## Current Benchmark

PNA/PBE/cc-pVDZ/grid level 3, `dt=0.05 au`, 10 SCEM steps, one warmup step,
8 PySCF threads:

```bash
/home/cgs/00_WORK/Projection_Code/aion/tools/gpu-python \
  examples/gpu_backend/benchmark_scem_cpu_gpu.py \
  --molecule pna --grid-level 3 --dt 0.05 --nsteps 10 \
  --warmup-steps 1 --backend both --threads 8
```

Result on the RTX 3090:

```text
CPU: 2.565 s/step, 0.801 s/Fock build
GPU: 0.264 s/step, 0.083 s/Fock build
speedup: 9.7x
```

Both runs used 32 total Fock builds and 31 total midpoint iterations. The final
dipole and electron number agreed to numerical precision.
