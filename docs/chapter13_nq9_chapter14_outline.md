# Proposed Chapter 14 outline

Date: 2026-09-24

Status: **NQ9 proposal; not yet reviewed or accepted**

Proposed chapter title:

> Numerical realization and qualification of fixed-centre Wilson-projected
> adiabatic dynamics

This outline is derived from the accepted NQ0--NQ8 evidence. It is a writing
handoff, not an authorization to change the formalism book. The chapter should
preserve the distinction between mathematical identities, reusable software,
executed numerical tests, and user-accepted claims.

## 1. Scope, variables, and realization tuple

Purpose:

- define the fixed-centre finite molecular problem actually qualified;
- state the contravariant coefficient density `P`, mixed tensor `D=P S`,
  overlap `S`, temporal connection `omega`, and lower mechanical matrix `K`;
- declare charge, units, Wilson paths, anchors, basis conventions, occupation
  convention, energy convention, and the Aion/PySCF ownership boundary; and
- separate the exact finite-subspace Wilson model from reduced P0/E1/C1
  descendants.

Primary evidence: NQ0 and the accepted exact one-electron report.

Essential warning: the scalar-potential coupling belongs to the temporal
connection and is not added again to the mechanical energy.

## 2. Exact Wilson density and variational nonlinear actions

Purpose:

- introduce the dressed AO frame and exact Wilson density;
- show direct and endpoint/triangle-factorized density evaluation;
- derive unrestricted coefficient and fixed-history source directions;
- define the Coulomb-metric RI Hartree action and its lower matrix;
- define the quadrature LDA and GGA actions, including the GGA density-gradient
  derivative; and
- make explicit that energy, lower matrix, and source derivative descend from
  the same discrete action.

Primary evidence: NQ1--NQ3 and NQ8.

Numerical qualification to report:

- density factorization and gauge/frame covariance at floating-point scale;
- independent four-centre and same-grid PySCF/libxc references;
- separated grid, auxiliary-basis, retained-rank, and solve errors; and
- the deterministic finite-difference floors for LiH Hartree and CO/PBE
  source derivatives.

## 3. Nonlinear stationary states

Purpose:

- present the generalized Hermitian stationary equation;
- describe the direct metric eigensolve and self-consistent Hartree/KS loop;
- define orbital, fixed-point, commutator, occupation, particle-number, and
  stationarity residuals; and
- distinguish field-free PySCF recovery from finite-field Wilson evidence.

Primary evidence: NQ4 and the stationary subset of NQ8.

Bounded result: H2 and equilateral H3+ were qualified for pure LDA, while CO
provides a pure-PBE transfer test. This is not a basis-convergence or broad
chemical benchmark.

## 4. Action-derived sources, Ward identity, and continuity

Purpose:

- define the fixed-coefficient-history source derivative;
- separate minimal, tangential, embedding, and normal-subspace responses;
- explain why the physical on-shell current is not an ambient momentum
  expectation and why no borrowed observable is allowed;
- state the off-shell Ward identity; and
- state the on-shell weak finite-region and global continuity identities.

Primary evidence: NQ5, with transfer checks in NQ8.

Essential limitation: the Kohn--Sham source current is the current of the
declared auxiliary adiabatic action. It is not identified with an exact
interacting transverse current.

## 5. Connection-aware nonlinear time propagation

Purpose:

- derive the coefficient and contravariant-density equations from `S`,
  `omega`, and `K[P,t]`;
- explain metric compatibility as an independently checked identity rather
  than a reconstructed connection;
- give the two-node fourth-order Gauss--Magnus construction and `[2/2]` Padé
  link;
- describe the self-consistent Gauss-node iteration and consistency check;
- explain the congruence update for `P` and reconstruction of `D=P S`; and
- state explicitly that no Löwdin coordinates, Cholesky endpoint correction,
  metric projection, clipping, or density repair is used.

Primary evidence: NQ6.

The chapter should show timestep order, nonlinear-tolerance separation, and
the convergence of visible cross-metric, trace, and occupation defects.

## 6. Mechanical energy, current power, and integrated work

Purpose:

- define the complete Hartree and KS mechanical energies;
- derive the matrix-rate and action-source routes to instantaneous power;
- show why a time-dependent magnetic field must include its induction
  electric field; and
- compare endpoint mechanical-energy change with integrated source work.

Primary evidence: NQ6 and the PBE transfer check in NQ8.

The presentation should keep spatial quadrature, timestep, nonlinear solve,
and action-source derivative errors separate.

## 7. Reduced electromagnetic actions

Purpose:

- define P0, E1, strict C1, and density-resummed C1 as distinct actions;
- state which one-electron geometry, electric connection, and nonlinear
  closure each retains;
- compare stationary states, densities, lower matrices, sources, power, and
  trajectories against the exact Wilson model; and
- separate model error from timestep error.

Primary evidence: NQ7.

Accepted recommendation: strict C1 is the near-term reduced working action
only within the sampled pure-LDA H3+ domain. Density resummation is not a
repair and showed no systematic advantage. The exact Wilson action remains
the reference.

## 8. Domain failures and basis dependence

Purpose:

- report reduced-overlap positivity and conditioning as part of the physical
  approximation domain;
- show the STO-3G, cc-pVDZ, and aug-cc-pVDZ stress results;
- preserve all 48 visible reduced-metric failures; and
- explain why no molecule-independent magnetic-field cutoff is supported.

Primary evidence: NQ7.

Central negative result: every tested aug-cc-pVDZ reduced level failed at the
first nonzero sampled field, `Bz=0.03` a.u., while the exact Wilson metric
remained positive. The sampled grid brackets failures but does not locate a
continuous critical field.

## 9. Functional and molecular transfer

Purpose:

- introduce the separate `quadrature--Wilson GGA` realization label;
- document pure-PBE density-gradient and source-direction derivatives;
- compare same-grid PBE energy and matrix data to PySCF/libxc; and
- transfer stationary, source, continuity, power, and short-propagation tests
  from H3+ to closed-shell CO.

Primary evidence: NQ8.

The chapter must not relabel LDA evidence as GGA evidence and must state that
reduced GGA P0/E1/C1 actions were not derived or qualified.

## 10. Numerical error budget and negative results

Purpose:

- tabulate the limiting grid, RI, rank, finite-difference, stationary,
  nonlinear, and timestep errors for every accepted claim;
- distinguish model discrepancy from numerical uncertainty;
- record incomplete runs, power interruption, wrong-launcher GPU failure,
  superseded analysis criteria, and the NQ8 external-log hash defect; and
- explain why retained failed analyses strengthen rather than weaken the
  audit trail.

Primary evidence: every gate review plus the NQ9 `negative_results.csv` and
`limitations.csv` tables.

## 11. Reproducibility and equation-to-code-to-evidence index

Purpose:

- include the final mapping from mathematical object to reusable code, tests,
  accepted gate, numerical floor, and evidence root;
- describe the managed Aion environment and physical-GPU launch contract;
- publish review-record, entrypoint, raw-artifact, table, and figure hashes;
  and
- give commands for rerunning the NQ9 synthesis verifier and report build.

Primary evidence: NQ9 synthesis package and report.

## 12. Bounded conclusions and open extensions

Supported conclusion:

The exact straight-Wilson finite molecular model, Coulomb-metric RI Hartree
closure, restricted pure-LDA and pure-PBE adiabatic actions, action-derived
weak sources, and connection-aware nonlinear propagation form one numerically
consistent implementation over the accepted fixtures. Strict C1 is a useful
first-order reduced action inside a visibly basis-conditioned domain.

Questions deliberately left open:

- moving nuclei and their nonadiabatic connection;
- spatially nonuniform propagated fields;
- Maxwell backreaction;
- periodic boundary conditions;
- pseudopotential gauge and current terms;
- spin polarization, hybrids, meta-GGAs, and nonlocal correlation;
- exact interacting transverse-current questions requiring TDCDFT;
- long-time stability, spectroscopy, systematic basis convergence, and
  broader chemical production validation; and
- variational GGA reduced actions.

These are possible later programs, not implied conclusions of Chapter 13.
