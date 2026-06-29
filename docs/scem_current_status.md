# SCEM Current Status

This note records the present development baseline for `aion`.  It supersedes
the older CN/EP-PC1-centered implementation notes for new work.

## Chosen Propagator

The working propagator is the strict self-consistent exponential midpoint
method, abbreviated SCEM.

For a fixed AO basis with overlap matrix `S`, occupied coefficient matrix `C`,
and closed-shell occupation matrix `f = 2 I`, the propagated density is

```text
P = C f C^dagger
```

and the equation of motion is

```text
i S dC/dt = H[P,t] C .
```

For a step from `t_n` to `t_n + dt`, SCEM solves a nonlinear midpoint problem:

```text
C_mid = U_S(dt/2; H_mid) C_n
P_mid = C_mid f C_mid^dagger
H_mid = H[P_mid, t_n + dt/2]
```

After convergence, the accepted step is:

```text
C_{n+1} = U_S(dt; H_mid) C_n .
```

`U_S` is evaluated by transforming to an orthonormal representation, diagonalizing
the Hermitian frozen Hamiltonian there, applying the phase factors, and
transforming back.  This avoids forming a non-Hermitian `S^{-1} H` exponential.

The important practical properties are:

- exact preservation of `C^dagger S C` to roundoff for each frozen exponential,
- particle-number and idempotency preservation to roundoff for closed-shell RKS,
- physical midpoint densities generated from occupied orbitals,
- global second-order accuracy when the midpoint fixed point is converged.

The implementation is in `src/aion/cn_tddft.py` under:

- `LengthGaugeCNRTTDDFT.scem_step`,
- `LengthGaugeCNRTTDDFT.propagate_scem`,
- `LengthGaugeCNRTTDDFT.run_scem`.

The class name and filename are historical.  The SCEM methods are the current
path.

## Numerical Guidance

The best verification dataset so far is the resonant five-cycle pulse campaign
on LiH, CO, and N2 using PBE/cc-pVDZ.  The tested nested timesteps were:

```text
0.2, 0.1, 0.05, 0.025, 0.0125 au
```

The campaign supports the following working rule:

- `dt = 0.1 au` is often usable for exploratory molecular runs,
- `dt = 0.05 au` is the current conservative default,
- `dt = 0.025 au` and below are useful for verification/extrapolation, not as a
  default for larger systems.

The small-molecule data show clean behavior for absorbed energy and final
dipole observables.  Some max-over-trajectory observables are less smooth, as
expected for extrema.

## GPU Status

The GPU backend is useful but intentionally narrow.

Validated support:

- closed-shell RKS,
- pure LDA/GGA, mainly PBE,
- real Gaussian AO basis,
- length gauge,
- SCEM,
- GPU4PySCF mean-field object,
- density-fitted `get_veff`,
- real density passed to the pure DFT effective-potential builder.

Known unsupported cases:

- hybrid TDDFT,
- TDHF,
- complex-density exchange,
- arbitrary non-Hermitian density paths in GPU4PySCF.

The PNA/PBE/cc-pVDZ/grid3 benchmark on the RTX 3090 gave about `0.25 s/step`
for SCEM at `dt = 0.05 au`, with roughly three Fock builds per step after the
initial transient.  This was about `9.7x` faster than the tested 8-thread CPU
path.

## What Is Reusable Now

Reusable module pieces:

- dense CPU/GPU backend abstraction in `src/aion/backends.py`,
- SCEM propagation in `LengthGaugeCNRTTDDFT`,
- delta-kick application through `apply_delta_kick`,
- spectrum post-processing through `aion.spectrum.kick_spectrum`,
- Casida helpers in `aion.linear_response`,
- pulse definitions in `aion.fields`.

Reusable examples:

- `examples/gpu_backend/run_h2_gpu_scem.py`,
- `examples/gpu_backend/benchmark_scem_cpu_gpu.py`,
- `examples/small_molecule_references/run_diatomic_kick_spectrum.py`.

Exploratory analysis scripts and large result directories are not part of the
stable module surface.

## Near-Term Plan

The next development steps should be:

1. Keep SCEM as the default real-time propagator.
2. Turn molecule-specific runners into small wrappers around reusable module
   functions.
3. Add a clean trajectory/spectrum workflow for delta kicks:
   ground state, kick, propagate, write trajectory, compute spectrum.
4. Run tensor kick spectra by propagating three polarization directions and
   combining the response.
5. Add velocity-gauge kicks once the length-gauge kick workflow is clean.
6. Only then return to the gauge-invariant dressing hierarchy.

The current code is good enough to build on, but the API should remain narrow
until the GPU complex-density and hybrid/HF questions are settled.
