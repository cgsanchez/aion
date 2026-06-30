# Gauge-Invariant Implementation Spec

This document is the engineering companion to
`Docs/gauge_invariant_implementation_spec.tex`.

Important maintenance rule: the Markdown and LaTeX specifications must be
updated together whenever a design decision changes.  The Markdown file is the
implementation checklist used inside `aion`; the LaTeX file is the formal
handoff note.  They should not be allowed to diverge.

## 1. Scope

The target is the Wilson-projected gauge-invariant finite-basis formalism from
`Docs/gauge_invariant_dirac_frenkel_subbundle.tex`.

The first implementation will stay within:

- atom-centered PySCF Gaussian AO bases,
- AO anchors at the parent atom,
- closed-shell RKS initially,
- pure no-exact-exchange LDA/GGA at P0,
- spatially uniform electric fields,
- static spatially uniform magnetic fields,
- site scalar potentials and bond vector-potential line integrals,
- CPU correctness first, GPU port soon after the CPU path is stable.

We do not plan near-term support for arbitrary spatial field gradients.  The
hierarchy should still leave a path to higher electric and magnetic moments.

## 2. Fixed Decisions

### Basis and graph

Each atom is a graph site.  Every AO basis function is anchored at its parent
atom.  All AOs on the same atom share the same site scalar potential and site
gauge phase.

No localized-MO or Wannier layer is planned.

The atom-pair graph is complete at first.  There is no overlap cutoff in the
gauge geometry.

### Field parametrization

Use site scalar potentials `Phi_a(t)` and oriented bond line integrals
`Acal_ab(t)`, where

```text
Acal_ab = integral from R_b to R_a of A(r,t).ds
Acal_ba = -Acal_ab
```

Endpoint Wilson phases are

```text
Theta_ab = exp(i q Acal_ab / hbar).
```

For an AO pair `(i,j)` anchored on atoms `(a,b)`, use `Theta_ab`.

### Uniform electric field gauge family

For a spatially uniform electric field `E(t)`, define

```text
K(t) = integral from t_ref to t of E(t') dt'
```

and a scalar gauge-mixing parameter `lambda(t)`.

The default gauge family is generated from length gauge by

```text
Lambda(r,t) = -lambda(t) K(t).r.
```

Then

```text
A_E(t)   = -lambda(t) K(t)
Phi_E(r) = -(1-lambda(t)) E(t).r + lambda_dot(t) K(t).r
```

so the site scalar potential is

```text
Phi_a(t) = [-(1-lambda) E + lambda_dot K].R_a.
```

Special cases:

```text
lambda = 0: length gauge
lambda = 1: velocity gauge, if lambda_dot = 0
0 < lambda < 1: interpolating gauge
```

The `lambda_dot K` term is mandatory when `lambda` is time dependent.

### Uniform static magnetic field

Uniform static magnetic fields must support both symmetric and Landau gauges
from the start.  One of the main tests of the formalism is that observables do
not depend on this representational choice.

The symmetric gauge is

```text
A_B(r) = 1/2 B x (r - O_B)
```

where `O_B` is user-configurable.  The default can be the nuclear charge
center.

For a straight bond from `R_b` to `R_a`,

```text
Acal_ab^B = 1/2 B . [(R_b - O_B) x (R_a - O_B)].
```

The Landau gauge family is defined by choosing a unit vector `u` perpendicular
to `B`.  Let

```text
A_B(r) = [(r - O_B).u] (B x u).
```

For a straight bond,

```text
Acal_ab^B =
[(0.5 * (R_a + R_b) - O_B).u] * [(B x u).(R_a - R_b)].
```

Different choices of `O_B`, and different Landau directions `u`, are gauge
choices.  They should change only site phases and must leave gauge-invariant
observables unchanged.  Early magnetic tests should therefore compare at least:

```text
symmetric gauge, origin O_B
symmetric gauge, shifted origin O_B'
Landau gauge with one perpendicular direction u
Landau gauge with another perpendicular direction u'
```

The total bond integral is the sum of the electric-gauge vector-potential
contribution and the magnetic contribution.

For the supported fields, both scalar-potential time integrals and
vector-potential bond line integrals must be evaluated analytically from the
source parametrization.  Numerical quadrature of source phases should not be
part of the trusted P0 implementation.

## 3. Hierarchy

Use the names from the formal note:

```text
P0
P0+E1
B1-min
B1-full
```

### P0

Pure Peierls geometry:

```text
S_ij^P = Theta_ij S_ij^0
eta_t = 0
omega_t = S sigma + 1/2 D_site S
```

For P0 one-electron matrices, initially dress complete field-free PySCF
matrices:

```text
hcore_ij^P = Theta_ij hcore_ij^0.
```

This is the accepted first approximation until `B1-full` repairs missing
magnetic kinetic and orbital finite-size terms.

For local gauge-neutral operators:

```text
O_ij^P = Theta_ij O_ij^0.
```

For Coulomb-type two-electron primitives:

```text
(ij|kl)^P = Theta_ij Theta_kl (ij|kl)^0.
```

This means Hartree and exchange must eventually be built from dressed primitive
ingredients, not by multiplying a completed field-free Fock matrix by one
phase.

### P0+E1

Add the first electric finite-spread residual:

```text
d^alpha_ij = q integral e_i*(r) e_j(r) (r^alpha - R^alpha_ij) dr
R_ij = (R_anchor(i) + R_anchor(j))/2
d_P^alpha_ij(t) = Theta_ij(t) d^alpha_ij
V_E1_ij(t) = - E_alpha(t) d_P^alpha_ij(t)
```

For onsite blocks this is the ordinary AO dipole about the atom.

P0+E1 should be implemented in general length/velocity/interpolating gauges,
not only in length gauge.  This is the first level intended for serious
spectral comparisons:

```text
length gauge P0
velocity gauge P0
length gauge P0+E1
velocity gauge P0+E1
mixed gauge P0/P0+E1
```

The length-gauge P0+E1 matrix should reproduce the usual full molecular dipole
coupling when the P0 endpoint/Mulliken part and intrinsic E1 dipole part are
combined.

Current implementation status: `src/aion/p0_e1.py` implements the uniform-field
central dipoles, Peierls dressing, `V_E1`, a `P0E1Model` wrapper, the P0+E1
dipole observable, and PySCF construction from analytic `int1e_r` integrals.
The P0+E1 polarization charge/current and full E1 power-theorem trajectory
diagnostics are still pending.

### B1-min

Add the first magnetic correction to the metric:

```text
S_ij^B1 = Theta_ij [
    S_ij^0 + (i q / hbar) integral e_i* e_j Phi_ij^B(r) dr
]
```

Then rebuild the time connection from the improved metric:

```text
omega_t^B1 = S^B1 sigma + 1/2 D_site S^B1 + eta_t^B1.
```

At `B1-min`, keep `eta_t^B1 = 0` unless an electric residual is also active.
This level captures magnetic metric and Faraday-consistent connection effects,
but it is not the full orbital magnetic response.

Initially restrict `B1-min` to static uniform magnetic fields.

### B1-full

Add the magnetic spatial connection and matching Hamiltonian data.  This is the
first level that can claim full first-order orbital magnetic finite-size
response.

The required objects include:

- overlap flux moments for `delta S^B1`,
- gradient flux moments,
- magnetic connection moments involving `C_i + C_j`,
- kinetic-energy-like magnetic corrections,
- dressed local-potential matrix elements,
- dressed two-electron primitives if HF/hybrid exchange is used.

For uniform `B`,

```text
C_j(r) = 1/2 (r - R_j) x B
C_i(r) + C_j(r) = (r - R_ij) x B.
```

The plan is to determine whether PySCF/libcint can supply all needed moment and
gradient integrals.  If not, we must either add custom integral code or
reconsider how deeply `aion` can depend on PySCF for `B1-full`.

## 4. Pure-DFT P0 Bridge To PySCF

The first electronic model is a PySCF-backed pure-DFT bridge for restricted
LDA/GGA references.  LDA is still the cheapest smoke-test path, but GGA is
accepted once the returned potential passes finite-difference derivative
audits.

At P0, define an effective bare-AO density matrix for grid and Coulomb
contractions:

```text
rho0_ij = Theta_ij^* rho_ij
```

where `rho` is the gauge-specific P0 density matrix.  This inverse orientation
is required so that `rho0` is the field-free density seen by PySCF.

For LDA:

1. Compute density on the bare AO grid from `rho0`.
2. Evaluate LDA `v_xc[n]` normally.
3. Build the bare AO `vxc_bare` matrix from this scalar potential.
4. Dress the lower-index result:

```text
vxc_ij^P = Theta_ij vxc_bare_ij.
```

For Hartree:

1. Build `J_bare[rho0]` using PySCF Coulomb routines.
2. Dress the output:

```text
J_ij^P = Theta_ij J_bare_ij[rho0].
```

For GGA:

1. Compute density and density gradients from `rho0`.
2. Let PySCF/libxc evaluate the GGA kernel.
3. Dress the resulting lower-index matrix by `Theta_ij`.

Hybrids and HF are excluded from the first implementation because exchange
requires the dressed primitive two-electron grammar.  A final exchange matrix
cannot generally be corrected by one phase.

## 5. Integrator

The current SCEM integrator assumes a fixed metric `S`.  P0 and magnetic levels
have time-dependent Peierls metrics, so we need a connection-aware exponential
midpoint integrator.

The expanded equation is useful for analysis:

```text
i hbar S Cdot =
[H + V_eta - i hbar S sigma - (i hbar / 2) D_site S] C.
```

However, the trusted implementation should preserve endpoint metrics:

```text
C_n^dagger S_n C_n = I
C_{n+1}^dagger S_{n+1} C_{n+1} = I.
```

### Site-parallel frame

Because `sigma` is diagonal in the atom-site basis, introduce a site-parallel
transport `G_sigma(t_a,t_b)` satisfying

```text
dG/dt = -sigma(t) G.
```

For each AO, use the transport of its parent atom.  For the analytic uniform
fields in scope, evaluate

```text
G_sigma(t0,t1) = exp[- integral_t0^t1 sigma(t) dt]
```

from exact source primitives, not midpoint quadrature.  For the electric gauge
family,

```text
integral Phi_a dt =
[-(1-lambda(t)) K(t).R_a]_{t0}^{t1}.
```

Pull endpoint and midpoint lower-index matrices into a common site-parallel
frame, where the site connection is removed and `D_site S` becomes an ordinary
time derivative of the transported metric.

### Metric-sandwich exponential step

In the site-parallel frame, use an endpoint-metric-preserving midpoint step:

```text
C_{n+1} =
S_{n+1}^{-1/2}
exp[-i dt Hbar_mid / hbar]
S_n^{1/2} C_n
```

where

```text
Hbar_mid = S_mid^{-1/2} H_phys_mid S_mid^{-1/2}
H_phys_mid = H_mid + V_eta_mid
```

and all matrices are in the same site-parallel frame.  The midpoint density is
generated by the analogous half step:

```text
C_mid =
S_mid^{-1/2}
exp[-i (dt/2) Hbar_mid / hbar]
S_n^{1/2} C_n.
```

Then iterate the SCEM fixed point:

```text
C_mid -> rho_mid -> H_phys_mid -> C_mid
```

until the midpoint Hamiltonian and midpoint density residuals converge.

This step preserves endpoint metric orthonormality by construction for any
Hermitian `H_phys_mid` in the transported lower-index representation.

An expanded-generator exponential may be implemented as a diagnostic, but it is
not the trusted production integrator.

## 6. Observables And Ward Checks

Observables must come from the same approximate action as the EOM.

P0 source charge:

```text
Q_a = (q/2) Tr[rho {M_a, S^P}]
```

where `M_a` projects AO indices anchored on atom `a`.

P0 graph current should be implemented analytically as the source derivative
with respect to `Acal_ab`.  Finite differences of the action or Hamiltonian may
also be implemented as debugging oracles.

P0+E1 source dipole includes both the endpoint/Mulliken site piece and the
intrinsic overlap dipole:

```text
mu_int = sum_ij rho_ji Theta_ij d_ij.
```

Diagnostics required for every formal-level run:

- electron number,
- metric orthonormality,
- idempotency/projector error,
- site charges,
- graph currents,
- discrete continuity residual,
- source dipole,
- absorbed power,
- internal energy balance.

For uniform electric fields, verify:

```text
dU/dt = E(t).dmu/dt
```

with the source dipole from the same action.

## 7. Testing Milestones

### P0 geometry tests

- `Theta_ba = Theta_ab.conj()`.
- `S^P` is Hermitian.
- Gauge-origin and magnetic-gauge-form changes in uniform `B` act by site
  phases.
- `omega_t + omega_t^dagger = dot S` in the site-parallel frame, or the
  equivalent covariant identity in the original frame.

### P0 integrator tests

- Endpoint metric preservation for prescribed time-dependent `S(t)`.
- Length/velocity/interpolating gauge equivalence for a model matrix system.
- Static uniform `B` ground-state current sanity checks.

### P0 pure-DFT tests

- Zero field reproduces ordinary PySCF LDA.
- For PBE, the bare and P0-dressed Hxc matrices pass finite-difference
  derivative checks against the corresponding energy functional along real and
  complex Hermitian density directions.
- Pure gauge changes leave observables invariant.
- Length/velocity spectra agree at P0 when the same physical field is used.

### P0+E1 tests

- Length-gauge P0+E1 external matrix equals the ordinary AO dipole coupling.
- Dressed central dipoles are Hermitian.
- P0+E1 total dipole reconstructs the ordinary AO dipole at zero field.
- Short H2O trajectories agree in length, mixed, and velocity gauges.
- P0 vs P0+E1 spectra can be compared in length and velocity gauges.
- Source dipole and power theorem agree.

### B1-min tests

- Uniform `B` origin and symmetric/Landau gauge invariance.
- Improved metric remains Hermitian and positive definite for small fields.
- Time connection rebuilt from improved metric passes metric compatibility.

### B1-full tests

- Spatial connection is anti-Hermitian.
- Kinetic corrections reduce to the known zero-field kinetic matrix at `B=0`.
- Uniform `B` results are origin independent.

## 8. Implementation Order

1. Add gauge source objects: sites, bonds, uniform `E`, uniform `B`,
   symmetric/Landau magnetic gauge selection, and gauge mixing `lambda(t)`.
2. Add AO anchor mapping from PySCF molecules.
3. Add P0 geometry: `Theta`, `S^P`, `sigma`, `D_site S`, `omega_t`.
4. Add variable-metric SCEM for an abstract matrix functional.
5. Add P0 pure LDA/GGA bridge to PySCF.
6. Add P0 charges, currents, dipole, and power diagnostics.
7. Test length/velocity/interpolating gauge equivalence.
8. Add E1 AO central dipole moments and `V_E1`. Done for uniform electric
   fields.
9. Add E1 polarization charge/current diagnostics and compare P0 and P0+E1
   spectra.
10. Port P0/P0+E1 pure-DFT path to GPU.
11. Add B1-min uniform static `B`.
12. Specify and then implement B1-full integral requirements.
13. Revisit HF/hybrid exchange with fully dressed primitive two-electron
    contractions.
