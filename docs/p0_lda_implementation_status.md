# P0/P0+E1 Pure-DFT Implementation Status

This note records the current implementation state of the pure Peierls P0 and
first electric finite-spread P0+E1 layers in `aion`.  It should stay synchronized with
`docs/gauge_invariant_implementation_spec.md` when the design changes.

## Scope

The implemented layer is the atom-anchored P0 restriction of the
gauge-covariant finite-basis dynamics.  Every AO is assigned to one atomic
anchor.  P0 electromagnetic sources are represented by

- site scalar potentials `Phi_a(t)`,
- oriented bond line integrals `Acal_ab(t) = integral_a^b A(r,t).dr`,
- Peierls/Wilson phases `theta_ab = exp(i q Acal_ab / hbar)`.

The code currently targets spatially uniform electric fields and static
uniform magnetic fields.  Electric fields can be represented in length,
velocity, or constant mixed gauges.  Magnetic fields can be represented in both
symmetric and Landau gauges.

The sign convention is

- `q = -1` for electrons by default,
- `hbar = 1` by default,
- Wilson factor `exp(i q integral A.dr / hbar)`.

## PySCF Pure-DFT Model

`PyscfP0DftModel` is the current PySCF-backed adiabatic model.  It supports
restricted closed-shell RKS references and pure no-exact-exchange LDA/GGA
functionals.  `PyscfP0LdaModel` remains as a backward-compatible strict LDA
wrapper for the existing examples and tests.  Hybrids are intentionally
rejected for now.

For a gauge-specific P0 density `rho(t)`, PySCF must see the field-free AO
density

```text
rho0_mu_nu = theta_mu_nu(t)^* rho_mu_nu(t)
```

not `theta rho`.  This inverse dressing is essential: it is what makes pure-DFT
energy and Fock builds gauge-covariant in length, mixed, velocity, symmetric,
and Landau representations.

The model builds

```text
H_P0[rho,t] = theta(t) * (hcore0 + veff_DFT[rho0])
```

and evaluates the PySCF total energy from `rho0`.  This is a P0 Peierls-dressed
pure-DFT model and not a GIAO/London-orbital implementation.

The GGA path is accepted only through PySCF's ordinary restricted DFT builders:
the P0 layer inverse-dresses the density, PySCF evaluates the field-free GGA
energy and `veff`, and the returned lower-index matrix is Peierls dressed.  The
regression tests verify by central finite differences that the returned bare
and P0-dressed PBE Hxc matrices are the functional derivatives of their
corresponding energies along real symmetric and complex Hermitian density
directions.

## P0+E1 Electric Layer

The first electric finite-spread layer is implemented in `src/aion/p0_e1.py`.
The primitive object is the AO-pair central dipole

```text
d^alpha_mu_nu = q ( r^alpha_mu_nu - R^alpha_mu_nu S_mu_nu )
R_mu_nu = 0.5 (R_anchor(mu) + R_anchor(nu))
```

These are the dressable E1 matrices:

```text
d_P^alpha(t) = theta(t) * d^alpha
V_E1(t) = - sum_alpha E_alpha(t) d_P^alpha(t)
```

`P0E1Model` wraps any existing P0 model and adds `V_E1` to its Hamiltonian.
Its `energy()` method returns the internal/base material energy for power
diagnostics; `coupling_energy()` returns `Tr rho V_E1`.  The PySCF helper
`pyscf_central_dipole_matrices(reference)` builds the bare central dipoles from
the analytic `int1e_r` AO position integrals.

For length gauge, the diagnostic pair-center scalar matrix plus `V_E1`
reconstructs the ordinary AO length-gauge matrix:

```text
q Phi(R_mu_nu) S_mu_nu + V_E1_mu_nu = q Phi(r)_mu_nu.
```

The P0+E1 electronic dipole observable is

```text
mu = q sum_a N_a R_a + sum_alpha e_alpha Tr[rho d_P^alpha].
```

## Propagation

The working structure-preserving propagator is `VariableMetricSCEM`, a
strict self-consistent exponential midpoint method for time-dependent P0
metrics.  It propagates occupied AO coefficients and preserves endpoint
orthonormality in the P0 metric.

The reusable runner layer is in `src/aion/p0_runner.py`:

- `P0SCEMSettings`
- `run_p0_scem_trajectory`
- `run_p0_electric_gauge_comparison`
- `run_p0_uniform_electric_gauge_comparison`
- `constant_uniform_electric_gauge`
- `sin2_uniform_electric_gauge`
- `velocity_delta_kick_electric_gauge`
- `apply_p0_velocity_delta_kick`
- `transform_p0_coefficients_between_gauges`

For continuous electric fields the runner requires both `E(t)` and the analytic
impulse `K(t)=int E(t)dt`.  This is necessary because the scalar transport and
Peierls phases are built from the impulse.  The sin² pulse helper has an
analytic impulse, so length/mixed/velocity comparisons do not depend on
numerical quadrature of the source.

Delta kicks are currently represented in the post-kick velocity gauge: after a
kick impulse `K`, the source has zero electric field and constant impulse `K`.
The initial coefficients are transformed from the field-free geometry into the
post-kick Peierls metric before propagation.

## Observables And Diagnostics

The source observables are implemented for P0, with the first P0+E1 dipole
helper available:

- Mulliken/source site populations,
- source charges `Q_a = q N_a`,
- P0 electronic dipole `sum_a Q_a R_a`,
- P0+E1 electronic dipole `q sum_a N_a R_a + Tr rho d_P`,
- P0+E1 electronic dipole derivative and uniform-field `E . dmu/dt` power,
- graph currents,
- instantaneous continuity residual,
- source power,
- dipole power,
- material energy,
- electron count,
- coefficient orthonormality.

The important continuous identities being checked are

```text
dQ_a/dt + sum_b I_ab = 0
dU/dt = - sum_{a<b} I_ab Ecal_ab
       = E . dmu_P0/dt          for spatially uniform electric fields
```

Power residuals use a finite-difference material-energy derivative, so they
are expected to sit near finite-difference noise rather than exact roundoff.

## Gauge Tests In Place

Current tests cover:

- P0 metric and time-connection compatibility,
- exact scalar transport,
- symmetric-origin magnetic gauge equivalence,
- symmetric vs Landau magnetic gauge equivalence,
- SCEM endpoint metric preservation,
- length vs velocity electric gauge covariance,
- P0 graph-current continuity,
- P0 power identities,
- PySCF LDA zero-source agreement with ordinary PySCF Fock/energy,
- PySCF PBE bare and P0-dressed Hxc derivative audits,
- PySCF LDA trajectory gauge covariance for H2,
- PySCF LDA trajectory gauge covariance for non-linear H2O and CH4,
- PySCF LDA static-B symmetric vs Landau gauge covariance for H2O,
- analytic sin² impulse consistency,
- velocity-kick metric transformation.
- P0+E1 central dipole construction from PySCF `int1e_r`,
- P0+E1 length-gauge reconstruction of ordinary AO dipole coupling,
- P0+E1 dressed central-dipole Hermiticity,
- P0+E1 zero-field reduction to the base P0 model,
- P0+E1 total dipole reconstruction of the ordinary AO dipole,
- P0+E1 analytic dipole derivative against finite differences,
- P0+E1 short length/mixed/velocity gauge covariance for H2O,
- P0+E1 uniform-field internal-energy power residuals.

The small-molecule gauge suite includes H2, CO, N2, H2O, and CH4.

## Examples

The main P0/PySCF examples are in `examples/pyscf_p0_bridge`:

- `run_h2_p0_lda_gauge_compare.py`
- `run_small_molecule_p0_lda_gauge_suite.py`
- `run_h2_p0_lda_kick_spectrum.py`
- `run_h2_p0_lda_sin2_pulse.py`
- `run_h2o_p0_lda_static_b_gauge_compare.py`

The examples default to cheap `sto-3g` LDA runs so they are usable as smoke
tests.  The general P0 PySCF model accepts pure LDA/GGA references; the current
examples still use the strict LDA wrapper.

## Current Limitations

This is a coherent P0 pure-DFT implementation, but it is not the full hierarchy.

- P0+E1 currently covers uniform electric fields through central dipoles.  It
  does not yet include higher electric gradients.
- No B1-min or B1-full magnetic hierarchy yet.
- GGA support currently relies on PySCF's field-free pure-GGA builders after
  inverse Peierls density dressing.  Hybrids and custom dressed exchange are not
  included.
- No hybrid support yet; exact exchange needs two Peierls phases on the four
  AO indices.
- No Peierls-dressed SCF solver for magnetic ground states yet.  Static-B
  propagation currently starts from a metric-orthonormalized zero-field occupied
  subspace, which is sufficient for gauge covariance tests but not a magnetic
  ground-state calculation.
- The P0 dipole is the site/source dipole.  It is the correct observable for the
  P0 source hierarchy, but it is not the full AO dipole matrix used in ordinary
  length-gauge TDDFT.  Use `p0_e1_dipole_moment` for the P0+E1 dipole.
- Flat trajectory rows record the total P0+E1 dipole, its analytic derivative,
  and the uniform-field E1 power residual when the model exposes
  `central_dipoles0`.  Local E1 polarization charge/current rows are still not
  implemented because they require an explicit choice of atom partition
  functions `w_a`.
- The GPU backend has not yet been ported to this P0 pure-DFT runner path.

## Next Implementation Layer

The next work inside P0+E1 is to choose and document an atom partition if local
polarization charges/currents are needed.  For the current uniform-field scope,
the practical next step is to run P0 vs P0+E1 spectra in length, velocity, and
mixed gauges.  After that the next formal layer is B1-min.
