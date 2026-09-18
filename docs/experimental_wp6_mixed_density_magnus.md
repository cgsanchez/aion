# Experimental mixed-density Gauss–Magnus propagation

## Status and scope

This note records an experiment inside WP6. It does not change Aion's selected
production propagator, does not replace the Cholesky-corrected midpoint
baseline, and does not by itself satisfy the WP6 acceptance gate. The public
names carry the `Experimental` qualifier for that reason.

Four distinct claims are kept separate:

1. The tensor equations below are mathematical identities once the AO basis,
   metric, and connection are specified.
2. Aion implements the linear, prequalified-history version of those equations
   in `aion.propagation.tensorial`.
3. The automated tests execute manufactured convergence, invariant, smooth
   frame-transformation, and physical CPU/GPU parity checks.
4. Acceptance for self-consistent exact-Wilson dynamics still requires the
   remaining WP6 physical timestep and model-difference evidence.

## Tensorial equation of motion

The AO coefficient equation is

\[
 i\hbar\left(S\dot C+\omega C\right)=K C,
 \qquad
 G=S^{-1}\left(-\omega-\frac{i}{\hbar}K\right),
 \qquad
 \dot C=G C.
\]

For the contravariant AO density and its natural mixed-index form,

\[
 P^{\mu\nu}=C^\mu{}_i f^i{}_j C^{\dagger j\nu},
 \qquad
 D^\mu{}_\nu=P^{\mu\lambda}S_{\lambda\nu}=PS,
\]

metric compatibility, \(\dot S=\omega+\omega^\dagger\), gives

\[
 i\hbar\left(\dot D+[\Gamma,D]\right)=[H,D],
 \qquad
 \Gamma=S^{-1}\omega,
 \qquad
 H=S^{-1}K,
\]

or simply

\[
 \dot D=[G,D].
\]

Consequently, if \(U\) solves \(\dot U=G U\), then

\[
 C(t_1)=U(t_1,t_0)C(t_0),
 \qquad
 D(t_1)=U(t_1,t_0)D(t_0)U(t_1,t_0)^{-1}.
\]

The similarity update preserves the trace, eigenvalues, and projector identity
\(D^2=D\) algebraically. Physical Hermiticity is the mixed-index relation

\[
 D^\dagger S=SD.
\]

The contravariant density is recovered without forming an explicit inverse by
solving

\[
 P=DS^{-1}.
\]

Thus an AO observable with lower-index matrix \(O\) can be evaluated as either

\[
 \operatorname{Re}\operatorname{Tr}(PO)
 =\operatorname{Re}\operatorname{Tr}(D S^{-1}O).
\]

## Fourth-order link

For a step of length \(h\), Aion evaluates the complete generator at the two
Gauss–Legendre nodes

\[
 t_\pm=t_n+\left(\frac12\pm\frac{\sqrt3}{6}\right)h.
\]

The fourth-order Magnus exponent is

\[
 \Omega_4=\frac h2(G_-+G_+)
 -\frac{\sqrt3 h^2}{12}[G_-,G_+].
\]

The link is the diagonal \([2/2]\) Padé approximation to
\(\exp(\Omega_4)\):

\[
 U_n=R_{22}(\Omega_4)
 =\left(I-\frac{\Omega_4}{2}+\frac{\Omega_4^2}{12}\right)^{-1}
  \left(I+\frac{\Omega_4}{2}+\frac{\Omega_4^2}{12}\right).
\]

The Padé truncation begins at fifth order and therefore does not reduce the
formal fourth-order accuracy of the Magnus step. This statement is also tested
numerically rather than credited only from the formal series.

No orthogonalization, endpoint Cholesky factor, or metric correction enters the
experimental step. Aion instead evaluates the raw geometric defect

\[
 R_{S,n}=U_n^\dagger S(t_{n+1})U_n-S(t_n)
\]

against the independently supplied exact endpoint metrics. It records this
residual together with trace, idempotency, mixed metric-Hermiticity, and
recovered-\(P\) Hermiticity diagnostics.

## What is currently implemented

`ExperimentalGaussMagnusHistory` accepts exact endpoint metrics and complete
`EOMTriple` values at both Gauss nodes. `fourth_order_gauss_magnus_link` builds
the uncorrected link. `propagate_experimental_mixed_density` advances \(D\) by
similarity and records both \(D\) and the recovered \(P\) at every boundary.
All matrix algebra is backend-neutral and remains resident on the selected
NumPy or CuPy backend.

The experiment currently consumes a prequalified linear history. It does not
yet perform the nonlinear self-consistent Gauss-node solve needed by full
adiabatic TDDFT. That coupling must be designed and qualified separately if
the linear experiment is successful.

## Executed numerical checks

The CPU manufactured problem prescribes a noncommuting time-dependent exact
transport \(U(t)\), then constructs

\[
 S(t)=U(t)^{-\dagger}S(0)U(t)^{-1},
 \qquad K(t)=0,
 \qquad \omega(t)=-S(t)\dot U(t)U(t)^{-1}.
\]

This makes the exact coefficient and density solutions known without
integrating another ODE. The tests verify:

- fourth-order global convergence of the mixed density;
- fifth-order decrease of the one-step endpoint-metric defect;
- fourth-order decrease of the accumulated metric-Hermiticity defect;
- roundoff-level trace and idempotency preservation;
- fourth-order convergence between two smoothly related time-dependent
  frames; and
- physical CuPy execution and CPU/GPU parity without host fallback.

These are strong internal checks of the tensor equation and discrete link, but
they are not a substitute for the pending exact-Wilson physical trajectory and
timestep qualification.
