# P0 LDA Implementation Status

This note records the current implementation state of the pure Peierls P0 layer
in `aion`.  It should stay synchronized with
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

## PySCF LDA Model

`PyscfP0LdaModel` is the current PySCF-backed adiabatic model.  It supports
restricted closed-shell RKS references and pure LDA functionals.  GGA and
hybrids are intentionally rejected for now.

For a gauge-specific P0 density `rho(t)`, PySCF must see the field-free AO
density

```text
rho0_mu_nu = theta_mu_nu(t)^* rho_mu_nu(t)
```

not `theta rho`.  This inverse dressing is essential: it is what makes LDA
energy and Fock builds gauge-covariant in length, mixed, velocity, symmetric,
and Landau representations.

The model builds

```text
H_P0[rho,t] = theta(t) * (hcore0 + veff_LDA[rho0])
```

and evaluates the PySCF total energy from `rho0`.  This is a P0 Peierls-dressed
LDA model, not yet a P0+E1 model and not a GIAO/London-orbital implementation.

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

The source observables are implemented for P0:

- Mulliken/source site populations,
- source charges `Q_a = q N_a`,
- P0 electronic dipole `sum_a Q_a R_a`,
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
- PySCF LDA trajectory gauge covariance for H2,
- PySCF LDA trajectory gauge covariance for non-linear H2O and CH4,
- PySCF LDA static-B symmetric vs Landau gauge covariance for H2O,
- analytic sin² impulse consistency,
- velocity-kick metric transformation.

The small-molecule gauge suite includes H2, CO, N2, H2O, and CH4.

## Examples

The main P0/PySCF examples are in `examples/pyscf_p0_bridge`:

- `run_h2_p0_lda_gauge_compare.py`
- `run_small_molecule_p0_lda_gauge_suite.py`
- `run_h2_p0_lda_kick_spectrum.py`
- `run_h2_p0_lda_sin2_pulse.py`
- `run_h2o_p0_lda_static_b_gauge_compare.py`

The examples default to cheap `sto-3g` LDA runs so they are usable as smoke
tests.  The basis and functional can be changed from the command line, but only
LDA is currently accepted by the P0 PySCF model.

## Current Limitations

This is a coherent P0 LDA implementation, but it is not the full hierarchy.

- No P0+E1 electric multipole layer yet.
- No B1-min or B1-full magnetic hierarchy yet.
- No GGA support yet; GGA needs dressed density, density gradients, and grid
  handling audited carefully.
- No hybrid support yet; exact exchange needs two Peierls phases on the four
  AO indices.
- No Peierls-dressed SCF solver for magnetic ground states yet.  Static-B
  propagation currently starts from a metric-orthonormalized zero-field occupied
  subspace, which is sufficient for gauge covariance tests but not a magnetic
  ground-state calculation.
- The P0 dipole is the site/source dipole.  It is the correct observable for the
  P0 source hierarchy, but it is not the full AO dipole matrix used in ordinary
  length-gauge TDDFT.
- The GPU backend has not yet been ported to this P0 LDA runner path.

## Next Implementation Layer

The next formal layer should be P0+E1 for LDA.  The first useful target is a
toy/PySCF-compatible electric hierarchy extension with atom-centered first
moments, gauge transformation rules, source observables, and Ward checks.  Only
after P0+E1 is stable should we move to B1-min and then B1-full.
