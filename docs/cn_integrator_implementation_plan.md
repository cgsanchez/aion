# Crank-Nicolson RT-TDDFT Implementation Plan

This document is a handoff note for implementing a proper real-time TDDFT
integrator in `aion`. It summarizes the current code state, the intended
length-gauge Crank-Nicolson (CN) implementation, diagnostics, benchmark facts,
and the later gauge-covariant extension.

## Current Code State

The package currently lives in:

```text
aion/
  src/aion/
    fields.py
    linear_response.py
    rt_tddft.py
```

The main dynamics class is `LengthGaugeRTTDDFT` in `src/aion/rt_tddft.py`.
It propagates the PySCF AO density matrix directly with a leapfrog update:

```text
rho_{n+1} = rho_{n-1} + 2 dt R[rho_n,t_n]
```

where

```text
R[rho,t] = -i (S^{-1} F[rho,t] rho - rho F[rho,t] S^{-1})
```

in atomic units. The PySCF density convention is used:

```text
rho = sum_n f_n C_n C_n^\dagger
```

For a closed-shell RKS calculation, `f_n = 2` for occupied spatial orbitals.
The conserved electron number is therefore:

```text
N = Tr(rho S)
```

and the closed-shell idempotency condition is:

```text
rho S rho = 2 rho
```

The current length-gauge Hamiltonian is:

```text
F[rho,t] = hcore + v_Hxc[rho] + V_ext(t)
```

with PySCF providing:

```python
veff = mf.get_veff(mol, rho)
```

The external field is spatially uniform, `A = 0`, and is represented by the
full AO dipole matrix:

```text
Phi(r,t) = - E(t) . (r - R0)
d_mu_nu = <chi_mu | r - R0 | chi_nu>
V_ext(t) = q Phi = -q E(t) . d
```

For electrons, `q = -1`, so:

```text
V_ext(t) = E(t) . d
```

In code:

```python
self.rint = mol.intor("int1e_r", comp=3)
self.dipole_position = self.rint - origin[:, None, None] * self.s[None, :, :]
V_ext = -charge * einsum("x,xij->ij", E_t, dipole_position)
```

## Important Gauge-Covariance Distinction

The current implementation is ordinary fixed-basis length-gauge RT-TDDFT. It is
not yet the full gauge-covariant dressed-basis formalism.

In the gauge-covariant formulation, the scalar potential is not merely an
additive Hamiltonian term. It also appears in the analytic time connection:

```text
D_t = partial_t + Sigma_t
```

and the restricted finite-basis EOM has the schematic form:

```text
i hbar S D_t C = K C
```

or, after expanding the covariant derivative,

```text
i hbar S dot C = (K - i hbar S Sigma_t) C
```

depending on conventions.

That scalar-potential connection, often represented in an onsite or block
form, is part of what restores gauge covariance in the dressed finite subspace.
It is not active in the current code.

The first CN implementation should deliberately remain in the simpler setting:

```text
fixed Gaussian AO basis
fixed overlap S
A = 0
spatially uniform E(t)
full AO dipole matrix coupling
no onsite scalar-potential connection
no Wilson/Peierls dressing
```

The scalar-potential connection belongs to the later exact-parallel-transport
stage.

## Why Replace Leapfrog?

Leapfrog preserves `Tr(rho S)` well, but it is not exactly unitary/idempotency
preserving. In the benzene resonant-CW test:

```text
dt = 0.1 au
Emax = 0.5 V/Angstrom
```

the density blew up by about `t = 3 au` and then produced `nan` values. A
timestep scan to `t = 10 au` showed:

```text
dt      final idempotency residual
0.02    6.82e-6
0.01    1.71e-6
0.005   4.27e-7
```

The goal of CN is not only to permit larger timesteps, but to preserve the
occupied metric structure:

```text
C^\dagger S C = I
```

for a frozen Hermitian midpoint Hamiltonian.

## Target CN Equation

Switch from propagating `rho` directly to propagating the occupied orbital
coefficient block `C`.

The fixed-basis length-gauge equation is:

```text
i hbar S dC/dt = H[P,t] C
```

where:

```text
S^\dagger = S > 0
H^\dagger = H
C^\dagger S C = I
P = C f C^\dagger
```

For closed-shell RKS:

```text
f = 2 I_occ
```

The generalized CN step is:

```text
A C_{n+1} = B C_n
```

with:

```text
A = S + i dt H_mid / (2 hbar)
B = S - i dt H_mid / (2 hbar)
```

In atomic units, `hbar = 1`.

For frozen Hermitian `H_mid`, the map

```text
U = A^{-1} B
```

is exactly `S`-unitary:

```text
U^\dagger S U = S
```

so:

```text
C_{n+1}^\dagger S C_{n+1} = C_n^\dagger S C_n
```

up to numerical linear-solve error.

## Proposed New Module

Add:

```text
src/aion/cn_tddft.py
```

with a class such as:

```python
class LengthGaugeCNRTTDDFT:
    ...
```

Do not delete `LengthGaugeRTTDDFT`; keep it as the leapfrog baseline.

Expected public workflow:

```python
rt = LengthGaugeCNRTTDDFT.from_ground_state(mf, field, origin=np.zeros(3))
C0 = rt.initial_coefficients()
for C, rec in rt.propagate(C0, dt=0.02, nsteps=...):
    ...
```

Potential alternatives:

```python
rho0 = rt.initial_density()
```

can still be exposed for diagnostics, but the propagated state should be `C`.

## Initialization From PySCF

For RKS:

```python
mo_occ = np.asarray(mf.mo_occ)
occ_mask = mo_occ > 0
C_occ = np.asarray(mf.mo_coeff[:, occ_mask], dtype=np.complex128)
occ = mo_occ[occ_mask].astype(float)
```

PySCF MOs satisfy:

```text
C_occ^\dagger S C_occ = I
```

The density is:

```python
P = (C_occ * occ[None, :]) @ C_occ.conj().T
```

or equivalently:

```text
P = C f C^\dagger
```

For unrestricted spin cases, this needs a separate design. The first
implementation should target closed-shell RKS only and raise a clear error for
unsupported references.

## Core Methods To Implement

### Basic storage

The CN class should cache:

```python
self.mf
self.mol
self.field
self.charge = -1.0
self.hbar = 1.0
self.s = mol.intor("int1e_ovlp")
self.hcore = mf.get_hcore()
self.rint = mol.intor("int1e_r", comp=3)
self.dipole_position = self.rint - origin[:, None, None] * self.s[None, :, :]
self.mo_occ
self.occ
self.nocc
```

### Density from coefficients

```python
def density_from_coefficients(self, coeff):
    return (coeff * self.occ[None, :]) @ coeff.conj().T
```

### External potential

Same as current code:

```python
def external_potential(self, t):
    e_t = np.asarray(self.field(t), dtype=float)
    return -self.charge * np.einsum("x,xij->ij", e_t, self.dipole_position)
```

### Hamiltonian build

Separate the density-dependent build from the external field:

```python
def hamiltonian_from_density(self, rho, t):
    rho_h = hermitian_part(rho)
    veff = mf.get_veff(mol, rho_h)
    h = hcore + veff + external_potential(t)
    return hermitian_part(h), veff
```

Returning `veff` is useful because energy diagnostics can reuse it instead of
calling `get_veff` again.

### Midpoint density predictor

For the first implementation:

```text
if no previous density:
    P_mid = P_n
else:
    P_mid = 1.5 P_n - 0.5 P_{n-1}
```

Hermitize afterward:

```python
P_mid = hermitian_part(P_mid)
```

This predictor is second-order accurate for smooth trajectories.

### CN solve

```python
def cn_step(self, coeff, h_mid, dt):
    alpha = 0.5j * dt / self.hbar
    A = self.s + alpha * h_mid
    B = self.s - alpha * h_mid
    rhs = B @ coeff
    lu, piv = scipy.linalg.lu_factor(A, check_finite=False)
    coeff_next = scipy.linalg.lu_solve((lu, piv), rhs, check_finite=False)
    return coeff_next
```

The full occupied block is propagated in one solve.

### Optional reorthonormalization

Implement but leave configurable:

```python
def orthonormalize(self, coeff):
    O = coeff.conj().T @ self.s @ coeff
    eig, vec = scipy.linalg.eigh(O)
    O_mhalf = vec @ np.diag(eig**-0.5) @ vec.conj().T
    return coeff @ O_mhalf
```

Policies:

```text
reorthonormalize_every = None
orthonormality_tolerance = maybe 1e-8
```

The method should not rely on reorthonormalization to be stable. It is a
cleanup.

## Propagation Loop

Suggested API:

```python
def propagate(self, coeff0, *, dt, nsteps, t0=0.0):
    ...
```

Loop state:

```python
C_n
P_n
P_{n-1} or None
```

Step:

```text
t_mid = t_n + dt/2
P_mid = predictor(P_n, P_{n-1})
H_mid, veff_mid = hamiltonian_from_density(P_mid, t_mid)
C_{n+1} = cn_step(C_n, H_mid, dt)
optional reorthonormalize
P_{n+1} = density_from_coefficients(C_{n+1})
record diagnostics
P_{n-1} <- P_n
P_n <- P_{n+1}
C_n <- C_{n+1}
```

For strict second-order behavior, the Hamiltonian should be built at the
midpoint external field time:

```text
V_ext(t_n + dt/2)
```

not at `t_n`.

## Optional Predictor-Corrector Upgrade

After the basic linear-predictor CN works, add:

```text
predict P_mid
build H_mid^(0)
CN from C_n -> provisional C_{n+1}^{(0)}
P_mid^(1) = 0.5 (P_n + P_{n+1}^{(0)})
build H_mid^(1)
CN again from original C_n -> final C_{n+1}
```

This costs two Hamiltonian builds per step. Benchmarks show this is the only
cost that matters, so make it optional:

```python
corrector_iterations=0
```

or:

```python
midpoint_mode="linear_predictor" | "predictor_corrector"
```

## Diagnostics

Create a new record dataclass, perhaps:

```python
@dataclass
class CNPropagationRecord:
    step: int
    time: float
    electron_number: float
    orthonormality_error: float
    idempotency_error: float
    field_free_energy: float | None
    field_coupling_energy: float
    total_energy: float | None
    dipole: np.ndarray
    field: np.ndarray
```

Important diagnostics:

```text
orthonormality_error = ||C^\dagger S C - I||
electron_number = Tr(P S)
idempotency_error = ||P S P - f_occ P||
dipole = q Tr(P d)
```

For closed-shell RKS, `f_occ = 2`.

Energy diagnostics:

Energy is expensive if it calls `mf.energy_tot(dm=P)` directly because that
performs another `get_veff`. Prefer a helper that can reuse `veff`:

```python
def field_free_energy_from_veff(self, rho, veff):
    e_elec, _ = self.mf.energy_elec(dm=rho, h1e=self.hcore, vhf=veff)
    return float(e_elec + self.mf.energy_nuc())
```

For GGA DFT, PySCF's `veff` object contains `ecoul` and `exc`, so this should
avoid an extra grid build.

If energy is requested at a step but the available `veff` corresponds to the
midpoint predictor rather than the final `P_{n+1}`, decide explicitly:

1. cheap approximate energy from `P_mid` and `veff_mid`;
2. expensive exact diagnostic energy from `P_{n+1}` with a fresh `get_veff`;
3. no field-free energy unless explicitly requested.

For benchmark/prod runs, option 3 or sparse exact energy is better.

## Tests To Add

Add a `tests/` directory if not present.

### Algebraic fixed-Hamiltonian test

Use a random Hermitian positive definite `S`, random Hermitian `H`, and random
initial occupied block `C` orthonormalized in `S`.

Apply many CN steps with fixed `H`. Assert:

```text
||C^\dagger S C - I|| < ~1e-11
```

This test does not need PySCF.

### Ground-state field-free PySCF test

For a tiny molecule/basis:

```text
H2 or benzene/STO-3G
E(t) = 0
```

Starting from converged KS orbitals, the density should remain stationary:

```text
||P(t) - P(0)|| small
```

Occupied orbitals may rotate by phases; compare density, not coefficients.

### Length-gauge comparison test

Run benzene or a smaller molecule under a weak pulse with:

```text
leapfrog dt = small
CN dt = same
```

Compare dipole time series qualitatively/quantitatively over a short interval.

### Timestep stress test

Repeat the previous failing setup:

```text
benzene PBE/cc-pVDZ
resonant CW field
Emax = 0.5 V/Angstrom
dt = 0.1, 0.05, 0.02
```

CN should keep orthonormality/idempotency much better than leapfrog. Accuracy
still requires convergence checks.

## Benchmark Facts Relevant To Integrator Choice

Benchmarks on acenes PBE/`cc-pvdz` show the TD step cost is dominated by
PySCF's `get_veff`.

Approximate `get_veff` timings:

```text
benzene       nao=114  grids=143560  get_veff ~0.46 s
naphthalene   nao=180  grids=219648  get_veff ~1.05 s
anthracene    nao=246  grids=295736  get_veff ~4.58 s
tetracene     nao=312  grids=371824  get_veff ~7.22 s
```

Dense linear algebra is negligible in comparison:

```text
benzene S-solves/matmuls    ~0.0006 s
tetracene S-solves/matmuls  ~0.0064 s
```

A dense eigendecomposition or exponential at these dimensions is also only
milliseconds.

Therefore:

```text
one-Hamiltonian CN step ~= current leapfrog step cost
one predictor-corrector CN step ~= 2x current step cost
two corrector iterations ~= 3x current step cost
```

The integrator should minimize extra `get_veff` calls.

## Later Gauge-Covariant Extension

After fixed-basis length-gauge CN works, implement the full formalism in two
parts:

```text
exact parallel transport of the analytic connection
+
CN propagation of the residual equation
```

The covariant equation is:

```text
i hbar S(t) D_t psi = K(t) psi
D_t = partial_t + Sigma(t)
```

Define transport:

```text
partial_t P_transport(t,t_n) = -Sigma(t) P_transport(t,t_n)
P_transport(t_n,t_n) = I
psi(t) = P_transport(t,t_n) chi(t)
```

Then:

```text
D_t psi = P_transport dot chi
```

and the residual equation becomes:

```text
i hbar S_tilde(t) dot chi = K_tilde(t) chi
S_tilde = P_transport^\dagger S P_transport
K_tilde = P_transport^\dagger K P_transport
```

If the scalar-potential connection is block diagonal/onsite:

```text
Sigma(t) = (i q / hbar) sum_a phi_a(t) M_a
```

then the transport is cheap and analytic:

```text
P_transport(t_{n+1},t_n)
= sum_a exp[-(i q/hbar) int phi_a(t) dt] M_a
```

The residual CN step is then:

```text
(S_tilde_mid + i dt K_tilde_mid/2) chi_{n+1}
=
(S_tilde_mid - i dt K_tilde_mid/2) chi_n
```

and:

```text
psi_{n+1} = P_transport(t_{n+1},t_n) chi_{n+1}
```

Do not implement this first. The immediate milestone is robust fixed-metric
length-gauge CN.

## Immediate Coding Checklist

Initial implementation status: `LengthGaugeCNRTTDDFT` now exists in
`src/aion/cn_tddft.py`, with an algebraic metric-unitarity test and a benzene
smoke example in `examples/benzene_length_gauge/run_benzene_cn.py`.

1. Create `src/aion/cn_tddft.py`.
2. Add `LengthGaugeCNRTTDDFT` and `CNPropagationRecord`.
3. Implement RKS-only initialization from `mf.mo_coeff` and `mf.mo_occ`.
4. Implement density construction `P = C f C^\dagger`.
5. Reuse existing dipole/external-potential logic.
6. Implement midpoint density predictor.
7. Implement one-build CN step.
8. Add optional occupied reorthonormalization.
9. Add diagnostics, especially `||C^\dagger S C - I||`.
10. Avoid extra `get_veff` calls in energy diagnostics unless explicitly asked.
11. Add small algebraic CN unit test.
12. Add a benzene CN example using saved Casida resonance.
13. Compare CN against leapfrog on the previous timestep scans.

## Initial Verification Results

Commands run after the first implementation:

```bash
python -m compileall src tests
PYTHONPATH=src python -m pytest tests/test_cn_tddft.py -q
python examples/benzene_length_gauge/run_benzene_cn.py
```

The algebraic fixed-Hamiltonian test propagates 100 CN steps with random
Hermitian `S` and `H` and verifies:

```text
||C^\dagger S C - I|| < 1e-11
```

The benzene CN smoke run with the saved Casida resonance and weak Gaussian
field completed 50 steps. Typical final diagnostics were:

```text
N = 42.000000000000
||C^\dagger S C - I|| ~ 7e-14
||P S P - 2P|| ~ 3e-13
```

An additional harsh short check used the previous resonant continuous-wave
field:

```text
Emax = 0.5 V/Angstrom
dt = 0.1 au
t_final = 10 au
```

Unlike leapfrog, which blew up by about `t = 3 au`, the one-build CN
propagator stayed metric-unitary and idempotent through `t = 10 au`:

```text
N = 42.000000000000
||C^\dagger S C - I|| ~ 6e-14
||P S P - 2P|| ~ 3e-13
mu_y(t=10 au) ~ 0.78256
```

This does not prove `dt = 0.1 au` is converged, but it confirms that CN fixes
the immediate structural instability of direct density leapfrog.
