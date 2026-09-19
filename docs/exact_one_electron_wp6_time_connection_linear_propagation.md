# WP6: exact time connection and linear one-electron propagation

Status: implemented, numerically executed, reviewed, and accepted for the
fixed-centre linear one-electron scope.  The initial midpoint checkpoint
failed its model-difference stability criterion; the accepted result uses the
subsequent mixed-index fourth-order Gauss--Magnus experiment and its targeted
64- and 128-interval continuations.  The user accepted gate G6 on
2026-09-19; the immutable decision record is
`docs/reviews/exact_one_electron_g6_review_20260919.json`.

## 1. Scope and boundary

WP6 qualifies the time-dependent geometry of the fixed-centre, linear
one-electron Wilson formulation before any nonlinear electronic closure is
added.  The implementation supplies, at every time sample,

\[
 S(t),\qquad K(t),\qquad \omega_t(t),\qquad
 H_{\partial_t}(t)=K(t)-\mathrm i\hbar\omega_t(t),
\]

and propagates the coefficient equation

\[
 \mathrm i\hbar\bigl(S\dot C+\omega_t C\bigr)=KC.
\]

Here (S) is the dressed finite-subspace metric, (K) is the mechanical
one-electron matrix descended from the spatially covariant action, and
(\omega_t) is the projected temporal connection.  (H_{\partial_t}) is the
matrix that appears if the equation is written with an ordinary coefficient
derivative; it is not independently assembled and must not be interpreted as
the mechanical energy.

This work package does not implement a self-consistent Hartree--XC closure,
action-derived charge/current/power observables, or a physical many-electron
H3 calculation.  The three-centre system contains
one electron on three protons and is therefore H3(^{2+}), used only as a
geometric qualification fixture.

## 2. Maxwell-consistent source sample

For a spatially uniform, time-dependent magnetic field, the symmetric
representative is

\[
 \mathbf A_{\rm sym}(\mathbf r,t)
 =\frac12\mathbf B(t)\times(\mathbf r-\mathbf o),
 \qquad
 \Phi_{\rm sym}(\mathbf r,t)
 =-\mathbf E_{\mathbf o}(t)\cdot(\mathbf r-\mathbf o).
\]

It produces the physical electric field

\[
 \mathbf E(\mathbf r,t)
 =\mathbf E_{\mathbf o}(t)
 +\frac12(\mathbf r-\mathbf o)\times\dot{\mathbf B}(t).
\]

The affine induction term is compulsory: omitting it would violate Faraday's
law whenever (\dot{\mathbf B}\ne0).  `UniformMagneticSourceSample` stores
(\mathbf B), (dot{\mathbf B}), (\mathbf E_{\mathbf o}), the origin, and
the affine gauge representative.  When a Landau representative is requested,
the class adds (-\partial_t\lambda) to the scalar potential automatically.
Thus symmetric and Landau representatives have different potentials but the
same physical electric and magnetic fields.

For an affine electric field the radial line integral is analytic:

\[
 I^E(\mathbf a,\mathbf r)
 =\int_0^1(\mathbf r-\mathbf a)\cdot
 \mathbf E\bigl(\mathbf a+s(\mathbf r-\mathbf a),t\bigr)\,\mathrm ds
 =(\mathbf r-\mathbf a)\cdot
 \mathbf E\left(\frac{\mathbf a+\mathbf r}{2},t\right).
\]

## 3. Exact projected temporal connection

Let AO (\phi_\mu) be anchored at (\mathbf R_\mu), and define its Wilson-
dressed counterpart using the straight anchor-to-electron path.  With the
fixed convention (q=-1), (hbar=1) in the qualification runs, the exact
endpoint-factorized temporal connection is

\[
 (\omega_t)_{\mu\nu}
 =\frac{\mathrm i q}{\hbar}\,\Theta_{\mu\nu}
 \left[
   \Phi(\mathbf R_\nu)\,\bar S_{\mu\nu}
   -\bar I^E_{\mu\nu}
 \right].
\]

(\Theta_{\mu\nu}) is the exact centre-to-centre endpoint link,
(\bar S) is the exact endpoint-removed overlap, and (\bar I^E) is the
endpoint-removed radial-electric matrix.  The production value combines
analytic bare AO overlaps and first moments from PySCF/libcint with a real-
space correction proportional to the exact triangle factor minus one.  This
retains exact zero-field analytic integrals and avoids making the whole answer
inherit the absolute quadrature error.

An independent grid oracle differentiates the ket Wilson factor directly and
adds the pointwise scalar potential before AO contraction.  The direct and
factorized routes are separately coded and compared numerically.

The endpoint-link rate is

\[
 \dot\Theta_{\mu\nu}
 =\frac{\mathrm iq}{\hbar}
 \left(\int_{\mathbf R_\nu}^{\mathbf R_\mu}
 \dot{\mathbf A}\cdot\mathrm d\boldsymbol\ell\right)
 \Theta_{\mu\nu}.
\]

The metric rate is assembled by differentiating both the endpoint link and
the internal triangle factor.  It is not manufactured from the compatibility
identity.  This gives an independent test of

\[
 \boxed{\dot S=\omega_t+\omega_t^\dagger}.
\]

It is also compared with a central finite difference of independently
evaluated exact static metrics at
(mathbf B\pm\epsilon\dot{\mathbf B}).

## 4. Gauge covariance

For two affine representatives related by

\[
 \mathbf A'=\mathbf A+\nabla\lambda,
 \qquad
 \Phi'=\Phi-\partial_t\lambda,
\]

the AO-anchor transformation is diagonal.  Writing it as (G(t)), the
matrices transform as

\[
 S'=GSG^\dagger,\qquad K'=GKG^\dagger,
\]

\[
 \omega_t'=G\omega_tG^\dagger+GS\dot G^\dagger.
\]

The inhomogeneous second term is why a time-dependent basis transformation
cannot be tested by simple matrix congruence alone.  The implementation tests
the complete identity at a matrix sample and compares symmetric- and Landau-
representative trajectories after applying the analytic anchor gauge map.

## 5. Action-level comparison triples

The exact sample is exposed as Aion's existing immutable `EOMTriple` without
rebuilding any matrix.  A shared source-independent context stores the
zero-field magnetic first derivatives and the uniform-electric E1 tensor.
From one source sample the code assembles the following action-consistent
levels:

| Model | Metric | Mechanical matrix | Temporal connection |
|---|---|---|---|
| EX | exact | exact | exact |
| P0 | endpoint link times bare metric | endpoint link times bare mechanical matrix | P0-compatible connection |
| E1 | P0 | P0 | P0 plus the internal electric increment |
| gB1 | full first magnetic metric | P0 | metric-compatible magnetic connection |
| B1 | full first magnetic metric | full first magnetic mechanical matrix | metric-compatible magnetic connection |
| C1 | full first magnetic metric | full first magnetic mechanical matrix | B1 connection plus internal electric increment |

For the first-order magnetic metric,

\[
 S_{B1}=\Theta\left(S_0+B_iS_i^{(1)}\right),
\]

the complete analytic rate is retained:

\[
 \dot S_{B1}
 =\dot\Theta\left(S_0+B_iS_i^{(1)}\right)
 +\Theta\dot B_iS_i^{(1)}.
\]

The P0/B1-compatible connection is constructed so that its Hermitian part is
the declared metric rate.  The internal electric E1 contribution is added
only to the temporal connection.  At zero magnetic field in a uniform
electric field, E1 and the exact projected temporal connection coincide.

These are comparison models descended from declared action levels.  They are
not assembled by borrowing an observable or connection from a different
level.

## 6. Accepted linear propagator

The original checkpoint used one complete EOM triple per midpoint, a diagonal
([2/2]) Padé coefficient step, and a right-Cholesky endpoint correction.
That implementation remains preserved as failed historical evidence; it is
not the algorithm accepted at G6.

The accepted experiment propagates the natural mixed-index density

\[
 D^\mu{}_{\nu}=P^{\mu\lambda}S_{\lambda\nu}=PS.
\]

Metric compatibility reduces its equation of motion to

\[
 \dot D=[G,D],\qquad
 G=S^{-1}\left(-\omega_t-\frac{\mathrm i}{\hbar}K\right).
\]

For a step of length (h), the complete generator is evaluated at the two
Gauss--Legendre nodes (t_-) and (t_+).  The fourth-order Magnus exponent is

\[
 \Omega_4=\frac h2(G_-+G_+)
 -\frac{\sqrt3h^2}{12}[G_-,G_+],
\]

and the transport link is the diagonal ([2/2]) Padé approximation to
(\exp\Omega_4).  The density advances by the tensorial similarity

\[
 D_{n+1}=U_nD_nU_n^{-1}.
\]

This preserves trace, eigenvalues, and mixed-density idempotency algebraically.
The contravariant density is recovered by solving (P=DS^{-1}) without forming
an explicit inverse.

No orthogonalization, endpoint Cholesky factor, or metric correction enters
the accepted experimental step.  Instead, the raw cross-metric residual

\[
 U_n^\dagger S(t_{n+1})U_n-S(t_n)
\]

and the mixed metric-Hermiticity residual are measured against independently
evaluated endpoint metrics.  The implementation remains explicitly named
`Experimental` because G6 qualifies precomputed linear histories, not the
nonlinear self-consistent Gauss-node problem required for adiabatic TDDFT.

The superseded midpoint generator was

\[
 G_{n+1/2}
 =S_{n+1/2}^{-1}
 \left(-\omega_{t,n+1/2}-\frac{\mathrm i}{\hbar}K_{n+1/2}\right).
\]

and is retained only to make the failed checkpoint reproducible.

For a time-independent generalized Hamiltonian, an independent reference
trajectory is obtained by Cholesky reduction of

\[
 Kv=\varepsilon Sv
\]

and direct modal phase evolution.  This tests the complete propagator against
an analytic-in-time answer rather than against another time integrator.

## 7. Numerical qualification design

The immutable WP6 fixture declares all systems, grids, fields, time steps, and
tolerances before execution.

- H2/cc-pVDZ tests the exact matrix identities, static spectral trajectory,
  a combined electric/magnetic pulse, time-step refinement, affine gauge
  covariance, and an in-memory restart boundary.
- The pulse uses 8, 16, and 32 intervals over 4 a.u. in the full
  qualification.
- Equilateral and distorted one-electron three-centre geometries use the
  three sub-threshold, near-threshold, and above-threshold fields accepted at
  G5.  All remain inside every compared model's positive-metric domain.
- The three-centre trajectories use 4 and 8 intervals.  Both geometries are
  evaluated at grid level 4, while the equilateral geometry is repeated at
  grid level 5 to measure quadrature stability.
- The exact time histories contain endpoint and midpoint (S), (K),
  (\omega_t), (dot S), (H_{\partial_t}), (mathbf B),
  (dot{\mathbf B}), and (mathbf E_{\mathbf o}).

The gate requires direct/factorized temporal agreement, analytic/finite-
difference metric-rate agreement, metric compatibility, positive metrics,
physical-norm conservation, expected static and time-dependent convergence,
convergent gauge residuals, and quadrature changes smaller than one quarter
of each nonzero model difference. Post-execution analysis found that the
primary 4/8-interval three-centre histories do not yet establish timestep
stability of the model-minus-exact density differences. An 8/16/32 supplement
was therefore predeclared before its execution and is part of WP6, not WP7.

## 8. Executed evidence

The authenticated qualification directory is

`/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/exact_one_electron_qualification/wp6_linear_qualification_20260918T2055Z_3fd2a83b77d1`

Its raw `execution_plan.json`, `result.json`, `arrays.npz`, and
`completed.json` are immutable source evidence.  The derived `analysis/`
package contains hash-authenticated CSV tables, PNG/PDF figures, a concise
report, and a machine-readable summary.  Exact numerical values and the gate
criterion evaluation are taken from those authenticated artifacts rather
than copied into this implementation note.

The authenticated timestep supplement is

`/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/exact_one_electron_qualification/wp6_h3_timestep_refinement_20260918T212428Z_3fd2a83b77d1`.

It confirms second-order trajectory convergence (minimum measured order
(1.985506)), norm conservation to (3.33\times10^{-15}), metric
compatibility to (2.70\times10^{-17}), and a minimum metric eigenvalue of
(6.54\times10^{-3}).  It does not establish timestep-stable model
differences: the largest 16-to-32 refinement change is (3.338413) times the
32-step model-minus-exact density difference, above the predeclared (0.25)
limit.  Consequently this is executed evidence of a failed numerical
checkpoint and not a G6 pass.

The replacement mixed-density Gauss--Magnus execution is

`/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/exact_one_electron_qualification/wp6_mixed_magnus_20260918T231057Z_38ce83eb2fe9`.

Its targeted 64- and 128-interval continuations are

`/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/exact_one_electron_qualification/wp6_mixed_magnus_n64_20260919T120325Z_57ef2fb6d40e`

and

`/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/exact_one_electron_qualification/wp6_mixed_magnus_n128_diagnostics_20260919T124656Z_73f6dc1cb760`.

In the distorted above-threshold fixture, the measured global orders from the
32/64/128 density changes are 3.978 (EX), 3.707 (P0), 3.716 (E1), 3.986
(gB1), 3.990 (B1), and 3.978 (C1).  The largest change in a model-minus-exact
difference under 64-to-128 refinement is 0.00446 of the final difference,
well below the declared 0.25 limit.  Trace drift remains below
(6.7\times10^{-16}), the idempotency defect below (5.5\times10^{-15}), and
the minimum metric eigenvalue is (6.54\times10^{-3}).  The continuation shows
that the earlier P0/E1 anomaly was a pre-asymptotic timestep effect caused by
their larger and more strongly noncommuting approximate generators, not a
metric singularity or failure of the tensorial equation.

At the field-free endpoint, C1 gives an absorbed energy of
(2.78323\times10^{-5}) Ha versus (2.80540\times10^{-5}) Ha for EX and an
excitation probability of (2.18997\times10^{-5}) versus
(2.19946\times10^{-5}).  The remaining C1--EX separation is stable under
timestep refinement and is therefore a model-truncation difference at the
tested grid and basis, not a propagation error.

## 9. Accepted claim and remaining boundary

The following are implemented features:

- Maxwell-consistent time-dependent uniform magnetic source samples;
- exact direct and endpoint-factorized temporal connections;
- independently differentiated exact metric rates;
- action-consistent EX/P0/E1/gB1/B1/C1 triples;
- mixed-index density propagation with an uncorrected fourth-order two-node
  Gauss--Magnus/[2/2]-Padé link;
- CPU and physical-GPU matrix/propagation paths;
- authenticated raw and derived campaign artifacts.

The user reviewed the targeted convergence and endpoint-observable evidence
and accepted gate G6 on 2026-09-19.  This authorizes WP7 for the same
fixed-centre linear one-electron scope.  It does not claim a nonlinear
self-consistent propagator, a Hartree--XC closure, resolved continuum current
density, nuclear motion, or production-readiness of the experimental API.
