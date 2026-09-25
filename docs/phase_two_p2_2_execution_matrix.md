# Phase Two P2-2 execution matrix

Status: **implemented, numerical stages not yet executed**

Date: 25 September 2026
Branch: `feature/aion-phase-two`

This record fixes the bounded P2-2 numerical matrix before execution. It does
not select a production application, spectroscopy protocol, or long-time
trajectory.

## Accepted inputs

- H3+/LDA stationary and dynamic reproduction is inherited from accepted P2-1.
- CO uses the authenticated NQ8 PBE/cc-pVDZ/weigend checkpoint and arrays as an
  independent direct oracle.
- NH3 uses the authenticated `G_DZ` geometry and orientation from the Phase
  Two P2-0 heritage manifest. The historical Casida result is retained only as
  context; the P2-2 pulse is not resonant and makes no spectral claim.

## Fixed calculations

| role | system/action | spatial realization | source | trajectory |
|---|---|---|---|---|
| accepted transfer reproduction | CO exact Wilson PBE/cc-pVDZ/weigend | unpruned level-4 qualification grid | NQ8 `Bz(t)=0.03+0.001t` and `E=(0.002,-0.001,0.0005)` a.u. | 8 intervals, `dt=0.025` a.u., `T=0.2` a.u. |
| polyatomic bridge | NH3 exact Wilson PBE/cc-pVDZ/weigend | unpruned level-4 grid, `G_DZ`, `B=0` | one complete potential-first sin2 cycle, C3-axis polarization, peak `E=0.005` a.u., `omega=10*pi` a.u. | 8 intervals, `dt=0.025` a.u., `T=0.2` a.u. |
| reduction/integration check | NH3 ordinary bare length gauge | same reference, unpruned grid, PBE, RI basis, pulse, step and duration | same scalar-potential representative | same trajectory |

Each dynamic path runs on CPU first and then the physical GPU. The NH3 pulse
has zero vector potential and electric field at both endpoints. Its high
carrier frequency makes the complete pulse fit the bounded cost bridge; it is
not intended to model absorption at the historical Casida root.

## Initial-source boundary

The accepted CO trajectory starts from the static-`B` stationary state and
turns on finite electric and magnetic-field-rate terms at `t=0`. P2-2 adds the
explicit `continuous_density_quench` policy for exactly this case. It permits
a continuous density only when the initial spatial Wilson gauge—and therefore
the magnetic field, gauge representative, origin and metric—is unchanged. A
spatial-gauge or magnetic-field jump is rejected. This is distinct from the
future exact discontinuous vector-potential kick.

## Recorded evidence

The stagewise driver is
`tools/run_phase_two_p2_2_bridge.py`. It records stationary residuals, density
normalization, exact dipole and Cartesian action current, resolved energy,
power/work, Ward and weak-continuity identities, metric and nonlinear
diagnostics, timed nonlinear action evaluations, wall time and peak process
memory. Final CPU/GPU states and full streamed trajectories are retained.

The driver stages are:

1. `prepare-co`;
2. `run-co-cpu` and `run-co-gpu`;
3. `prepare-nh3`;
4. `run-nh3-cpu` and `run-nh3-gpu`;
5. `run-nh3-bare-cpu` and `run-nh3-bare-gpu`; and
6. `analyze`.

Every completed stage has a hashed stage record. Failures remain visible and
are never overwritten in place. The analyzer proposes a gate result but does
not accept P2-2; acceptance remains a separate user decision.

`tools/run_phase_two_p2_2_sequence.py` runs this order with one process and an
explicit host-thread cap, invokes physical-GPU stages only through
`tools/gpu-python`, and skips hashed completed stages when resuming an
interrupted campaign. It does not poll a detached run.
