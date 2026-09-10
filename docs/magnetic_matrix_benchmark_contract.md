# Uniform-magnetic-field matrix benchmark contract

Status: normative MB0 contract

Theory source: `docs/magnetic_matrix_benchmark_theory.tex`

This document fixes the signs, index orientation, array layout, units, and
independent-oracle boundaries used by the static magnetic matrix benchmark.
Contraction code must implement this contract; it must not select conventions
locally.

## Supported physics boundary

The initial implementation is a fixed-reference, spinless, orbital-magnetic
benchmark for finite, fixed-nucleus, all-electron molecules represented by
real, atom-centred spherical Gaussian AOs. The electronic reference is a
restricted spin-summed zero-field RKS `PreparedReference`. The magnetic field
is static and spatially uniform. Nuclear motion, spin coupling, relativistic
terms, ECPs or pseudopotentials, nonlocal ionic operators, periodic systems,
spatial field gradients, magnetic SCF relaxation, and real-time magnetic
propagation are unsupported and must be rejected explicitly.

## Fundamental conventions

Matrix row `mu` is the bra AO and matrix column `nu` is the ket AO. Their
authenticated atomic anchors are

```text
R_mu = nuclei.coordinates_au[ao_to_atom[mu]]
R_nu = nuclei.coordinates_au[ao_to_atom[nu]]
delta_R[mu,nu,:] = R_mu - R_nu
pair_midpoint[mu,nu,:] = (R_mu + R_nu)/2
xi[p,mu,nu,:] = coordinates[p] - pair_midpoint[mu,nu]
```

The canonical momentum is `p = -i*hbar*grad`. Aion's stored
`canonical_momentum` uses atomic units (`hbar = 1`) and has layout
`(cartesian, mu, nu)`. PySCF's `GTOval_ip_sph` differentiates the ket in the
blocked contraction. Thus

```text
p_grid[x,mu,nu] = -i*hbar * sum_p w[p] ao[p,mu] grad[x,p,nu]
omega_spatial_zero[x,mu,nu] = i*p_grid[x,mu,nu]/hbar
```

The mechanical momentum in a vector potential is `pi = p - q*A`. The
high-level electron default is `q = -1`, `m = 1`, and `hbar = 1`; low-level
geometry and contraction kernels accept any finite `q`, positive finite `m`,
and positive finite `hbar`.

Every Wilson path is oriented from its first argument to its second. The AO
endpoint path runs from the ket anchor to the bra anchor:

```text
L_endpoint[mu,nu] = integral(R_nu -> R_mu) A . dl
Theta[mu,nu] = exp(i*q*L_endpoint[mu,nu]/hbar)
```

This is the same row-minus-column orientation as Aion's P0 link. For a
spatially uniform reduced vector potential `a`, it becomes
`L_endpoint = a . (R_mu - R_nu)`.

For a uniform magnetic field, define

```text
flux[p,mu,nu] = 0.5 * B . (xi[p,mu,nu] x delta_R[mu,nu])
phase[p,mu,nu] = q*flux[p,mu,nu]/hbar
F_exact[p,mu,nu] = exp(i*phase[p,mu,nu])
F_first = i*phase
F_second = -0.5*phase**2
C[p,mu,:] = 0.5*(coordinates[p] - R_mu) x B
```

The oriented flux loop is `R_nu -> r -> R_mu -> R_nu`. Pair reversal changes
the sign of `flux`, `phase`, and the endpoint integral. Same-anchor pairs have
`phase = 0` and `F_exact = 1`, but generally have nonzero `C`.

## Affine magnetic gauges and direct oracle

The physical magnetic field is distinct from a gauge representative. An
affine representative has

```text
A(r) = M @ (r - origin),  curl(A) = B.
```

The symmetric representative is `A = 0.5*B x (r-origin)`. A Landau
representative is `A = (B x u) [u . (r-origin)]`, where `u` is a normalized
vector perpendicular to nonzero `B`. The exact straight-line integral of an
affine potential is its midpoint value dotted with the endpoint displacement.
Its derivative with respect to the final endpoint is

```text
grad_end integral(start -> end) A.dl
    = A(start) + 0.5*(M + M.T)@(end-start).
```

The direct oracle dresses each AO independently:

```text
I_mu(r) = integral(R_mu -> r) A.dl
W_mu(r) = exp(i*q*I_mu(r)/hbar)
chi_mu(r) = W_mu(r)*g_mu(r)
C_mu(r) = grad_r I_mu(r) - A(r)
```

It assembles lower overlap and local-potential matrices from `chi`, and lower
kinetic matrices from `(p-q*A)chi = W*(p+q*C_mu)g`. The direct lower spatial
connection uses the manifestly anti-Hermitian covariant derivative
`D_A = grad - i*q*A/hbar`:

```text
omega_lower = 0.5 * integral[
    chi_mu^* D_A chi_nu - (D_A chi_mu)^* chi_nu
].
```

The independently assembled factorized route removes the exact endpoint
link and contracts `F_exact`, `C_mu`, bare AO values, and bare AO gradients.
For every supported matrix family,

```text
M_direct_lower = Theta * M_factorized_barred
```

where `*` is elementwise multiplication. A gauge or origin change may alter
`Theta` and the direct lower matrix, but it must not alter a barred matrix.
The direct and factorized routes may share AO samples and grid weights, but
must not obtain one result by algebraically redressing or undressing the other.

The release qualification adds a third, overlap-only oracle. For every
atom-pair block it treats the triangle factor as a plane wave with

```text
k_mu_nu = q * (R_mu - R_nu) cross B / (2*hbar)
```

and obtains the contracted-Gaussian AO-pair Fourier transform from PySCF/
libcint. It shares neither the real-space grid nor AO samples with the direct
and factorized contractions. The midpoint phase restores the barred overlap,
and the exact endpoint link restores its lower form. This oracle is exact for
the declared finite Gaussian AO space up to libcint floating-point error.

## Authoritative array layouts and names

Coordinates and AO blocks use:

```text
coordinates_au                    (nblock, 3)       float64
weights_au                        (nblock,)         float64
values                            (nblock, nao)     float64/complex128
gradients                         (3, nblock, nao)  float64/complex128
ao_anchor_coordinates_au          (nao, 3)          float64
pair_midpoints_au                 (nao, nao, 3)     float64
pair_displacements_au             (nao, nao, 3)     float64
triangle_flux_au                  (nblock, nao, nao) float64
triangle_phase                    (nblock, nao, nao) float64
anchored_vectors_au               (nblock, nao, 3)  magnetic vector-potential AU
scalar matrices                   (nao, nao)         complex128
spatial-connection matrices       (3, nao, nao)      complex128/bohr
```

`spatial_connection` and `omega_spatial` refer only to the projected spatial
connection. The unqualified word `connection` remains reserved for the
temporal EOM connection in Aion's propagation/formulation API.

Authoritative matrix component names are:

```text
overlap: zero, first_F, second_F2, exact
kinetic: zero, first_F, first_pC, first_Cp,
         second_F2, second_F_pC, second_F_Cp, second_C2, exact
local/<provider>: zero, first_F, second_F2, exact
spatial_connection: zero, first_F, first_C, second_F2, second_FC, exact
```

The kinetic `pC` component has the bare momentum acting in the bra factor and
the anchored-vector momentum in the ket factor; `Cp` is the converse. Their
adjoint relation is checked before they are summed. The two second-order
form-factor/cross terms are retained separately by the same rule.

`zero`, `first_*`, and `second_*` are Taylor coefficients evaluated for the
supplied physical `magnetic_field_au`; they sum directly at bookkeeping
amplitude one. A directional derivative of order two is twice the stored
second-order coefficient. `B1` is `zero + all first_*`; `B2` is `B1 + all
second_*`. B2 is an amplitude expansion, not a magnetic-gradient expansion.

## Units and stable finite-field construction

All numerical kernels use atomic units. Public field input is named
`magnetic_field_au` and carries physical dimension `magnetic_field` with unit
`atomic_unit_of_magnetic_field`. Optional tesla input is converted explicitly
at a typed workflow/campaign boundary, and both values are recorded. No
electric-source unit or behavior changes as a side effect.

Finite-field matrices use the analytic zero-field baseline:

```text
M_corrected_exact(B) = M_analytic(0)
                     + M_grid_exact(B) - M_grid_zero
```

Scalar `exact-zero` corrections use `expm1(i*phase)`. Kinetic and spatial-
connection corrections retain their explicit field sectors so that the large
zeroth-order grid matrix is never subtracted after aggregation when a more
stable termwise expression is available. Raw quadrature zeroth-order
matrices and their analytic residuals remain diagnostics.

## Closure identities

With Aion's central dipole

```text
d[x,mu,nu] = q * integral xi[x] g_mu^* g_nu,
```

the first-order closures are

```text
overlap.first_F[mu,nu]
  = i/(2*hbar) * B . (d[:,mu,nu] x delta_R[mu,nu])

spatial_connection.first_C[:,mu,nu]
  = i/(2*hbar) * (d[:,mu,nu] x B).
```

Also `omega_spatial.zero = i*canonical_momentum/hbar`. These comparisons use
the stored analytic E1 and momentum data and are independent of the magnetic
quadrature contractions.

For real AOs and real local potentials, exact scalar and kinetic matrices are
Hermitian, `M(-B) = M(B)^*`, first-order coefficients are purely imaginary
and antisymmetric, and second-order coefficients are real and symmetric.
Every exact and truncated spatial-connection Cartesian matrix is
anti-Hermitian and obeys the corresponding field-reversal parity.

## Local-potential boundary

The initial provider protocol accepts a real pointwise multiplicative local
potential with stable identity, units, and provenance. Nuclear attraction is
the required first provider. A frozen Hartree provider and frozen LDA XC are
optional later additions and require independent zero-field AO-matrix
recovery.

GGA XC is explicitly outside the initial provider contract. PySCF assembles a
GGA XC contribution in a weak form involving density and AO gradients. It may
be added only after a reviewed Euler--Lagrange pointwise potential or an
equivalent Wilson-dressed weak-form derivation and an independent zero- and
finite-field validation exist. A preassembled AO matrix is never accepted as
a substitute for pointwise local-potential values. Nonlocal potentials are
rejected.

## Recovered evidence mapped to the new API

The old checkpoint `2befb8c` is evidence, not an implementation dependency.
Its tests map as follows:

| Archived identity | New owner |
|---|---|
| primitive Taylor coefficients and central differences | `electromagnetism.magnetic` MB1 tests |
| endpoint links, triangle flux, pair reversal, affine gauges | `electromagnetism.magnetic` MB1 tests |
| AO/gradient grid reconstruction | `electronic_structure.ao_quadrature` MB2 tests |
| direct/factorized S/T/V agreement | `electronic_structure.magnetic_matrices` MB3 tests |
| zero-field PySCF convergence and positive overlap metric | MB2/MB3 integration tests |
| onsite and bond-parallel `F=1` isolation | MB1/MB3 H2 integration tests |
| independent libcint GIAO derivatives | MB3 qualification tests |
| finite-difference B1/B2 coefficients and remainder slopes | MB3/MB7 tests |
| gauge-independent barred matrices and congruent lower spectra | MB3/MB7 tests |
| analytic Gaussian finite-field overlap oracle | MB7 qualification tests |
| multicentre loop, rotation, and block diagnostics | MB7 qualification tests |
| analytic Gaussian AO-pair Fourier overlap | `analytic_uniform_magnetic_overlap` MB7 tests |

The spatial-connection hierarchy has no archived implementation. It must be
assembled directly in MB4 and checked independently against stored momentum,
E1 dipoles, finite differences, symmetry, and the direct covariant-derivative
oracle.

## Hand-checked oriented fixture

Take `R_nu=(0,0,0)`, `R_mu=(2,0,0)`, `r=(1,3,0)`, `B=(0,0,4)`,
`q=-1`, `hbar=2`, symmetric gauge at the origin. Then

```text
delta_R = (2,0,0)
xi = (0,3,0)
flux = -12
phase = 6
C_mu = (6,2,0)
C_nu = (6,-2,0)
integral(R_nu -> r) A.dl = 0
integral(R_mu -> r) A.dl = 12
integral(R_nu -> R_mu) A.dl = 0
```

Therefore the direct bra/ket phase is
`exp[i*q*(I_nu-I_mu)/hbar] = exp(6i)`, exactly equal to
`Theta*F_exact = exp(6i)`. Moreover
`grad I_mu-A(r)=C_mu` and `grad I_nu-A(r)=C_nu`. This fixes the loop,
cross-product, Wilson, and mechanical-momentum signs in one oblique fixture.
For `B` parallel to `delta_R`, the same definitions give `phase=0` for every
point and hence reproduce the P0 endpoint-link limit with `F=1`.
