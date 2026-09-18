# Exact one-electron WP2 static Wilson qualification

Status: **G2 reviewed and accepted by the user on 16 September 2026**
Execution: `wp2_20260916T192437Z_3fd2a83b77d1`
Backend: CPU, complex128, managed Aion environment

## Evidence boundaries

The exact straight-Wilson identities are mathematical input from the accepted
formalism. The evaluator changes described here are reusable implemented
features. The tabulated values are executed numerical evidence. None of those
facts constitutes acceptance of G2; that remains a separate user decision.

The evaluator remains outside the production dynamics formulation path at
this gate. WP2 does not add an exact GPU evaluator and does not exercise a
nonlinear electronic closure.

## Implemented reference API

`evaluate_magnetic_one_electron_matrices` now consumes either the established
SCF-backed `PreparedReference` or the occupancy-independent
`OneElectronAOReference`. The latter builds and authenticates PySCF AO data
without constructing or solving an SCF problem. The evaluator samples each AO
grid block once for all requested fields and constructs two exact routes:

- EX-direct applies the anchor-to-point Wilson factor to each AO and its
  mechanical derivative before quadrature;
- EX-link constructs endpoint-removed amplitudes from the triangle factor and
  anchored magnetic vector, then restores the endpoint link.

The reusable result contains corrected exact and raw-grid lower overlap,
kinetic, and nuclear-attraction matrices. `OneElectronLowerMatrices.mechanical`
returns `T + Vnuc`. The kinetic result exposes the four exact sectors
`T_pp^F`, `T_pC^F`, `T_Cp^F`, and `T_CC^F`. The independent libcint AO-pair
Fourier overlap oracle also accepts the occupancy-independent reference.

The corrected exact endpoint-removed matrices use analytic zero-field PySCF
matrices plus the finite-field real-space correction. EX-direct is compared
with the raw-grid EX-link value so that the identity is not obscured by that
analytic zero-field replacement.

## Numerical campaign

H--H/STO-3G uses the accepted unpruned level-4 grid. The field magnitude is
`0.08` atomic unit and the campaign includes zero field and both signs of
parallel, perpendicular, and oblique directions. The EX-direct/EX-link
tolerances are the independently accepted H--H level-5 WP1 floors, rather than
tolerances estimated from the finite-field results.

| case | max EX-direct/EX-link | max pair reversal | min eig(S) | cond(S) | analytic S oracle |
|---|---:|---:|---:|---:|---:|
| zero | 2.229e-16 | 0.000e+00 | 3.406818e-01 | 4.870581e+00 | 9.269e-17 |
| parallel +B | 3.134e-16 | 0.000e+00 | 3.406818e-01 | 4.870581e+00 | 9.269e-17 |
| parallel -B | 3.134e-16 | 0.000e+00 | 3.406818e-01 | 4.870581e+00 | 9.269e-17 |
| perpendicular +B | 5.699e-16 | 6.777e-19 | 3.414498e-01 | 4.857377e+00 | 4.636e-16 |
| perpendicular -B | 5.699e-16 | 6.777e-19 | 3.414498e-01 | 4.857377e+00 | 4.636e-16 |
| oblique +B | 4.119e-16 | 5.061e-19 | 3.409562e-01 | 4.865856e+00 | 1.854e-16 |
| oblique -B | 4.119e-16 | 5.061e-19 | 3.409562e-01 | 4.865856e+00 | 1.854e-16 |

The largest EX-direct/EX-link residual is `5.70e-16`, more than four orders of
magnitude below the tightest accepted WP1 floor (`1.419e-11` for overlap).
Every exact Gram matrix is positive; the smallest observed eigenvalue is
`0.3406817938651362`.

Field reversal gives zero residual to stored precision for all four lower
matrix families in all three directions. Exact kinetic-sector closure also
passes in every case.

## Gauge representative and origin covariance

The oblique positive field was evaluated in symmetric and Landau gauges at
origins `(0,0,0)` and `(0.21,-0.17,0.13)` bohr. If
`A_target-A_reference = grad chi`, the tested coefficient transformation is

`M_target = D M_reference D^dagger`, with
`D_mu = exp[-i chi(R_mu)]` in electron atomic units.

| representative/origin | max barred change | max lower congruence residual | max spectrum shift (Ha) |
|---|---:|---:|---:|
| symmetric/origin A | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| symmetric/origin B | 0.000e+00 | 1.243e-18 | 2.776e-16 |
| Landau/origin A | 0.000e+00 | 1.686e-16 | 0.000e+00 |
| Landau/origin B | 0.000e+00 | 1.410e-16 | 2.220e-16 |

The generalized spectra are those of `(T+Vnuc, S)`. Barred amplitudes are
identical because they depend on the physical uniform field and anchor
geometry, while the lower matrices carry the endpoint coefficient phases.

## Rigid rotation with angular AO mixing

The O--H/STO-3G fixture supplies an oxygen `(2px,2py,2pz)` block. A general
three-dimensional rotation was applied to the nuclei, electromagnetic origin,
magnetic field, and oxygen p block. This tests actual AO-component mixing,
rather than relying only on a scalar s-orbital spectrum.

| block | S | T | Vnuc | K |
|---|---:|---:|---:|---:|
| full matrix | 4.374e-12 | 2.871e-11 | 8.341e-12 | 3.596e-11 |
| oxygen onsite | 1.093e-16 | 1.110e-11 | 4.536e-17 | 8.797e-12 |
| oxygen p/p | 1.427e-16 | 1.225e-11 | 1.784e-16 | 4.429e-12 |
| oxygen/hydrogen | 8.116e-12 | 5.497e-10 | 7.855e-11 | 2.007e-10 |

The maximum generalized-spectrum change is `2.416e-11` hartree. All block and
spectrum residuals are below the declared `2e-7` propagated rotation
tolerance. The larger inter-centre kinetic residual is consistent with finite
real-space quadrature and remains well below the accepted WP1 target.

## Regressions and artifacts

Five WP2 integration tests cover zero-field reduction, all exact-route and
field identities, gauge/origin covariance, the analytic overlap oracle, rigid
rotation, and reproduction of a hash-authenticated oblique H--H output. The
authenticated regression fixture records semantic SHA-256 hashes of the four
exact lower matrices and four kinetic sectors. The standalone WP2 verifier
recomputes the manifest identity, every recorded file and semantic-array hash,
all tolerance decisions, and the regression-fixture linkage.

Final verification passes Ruff, mypy over all 72 source files, the 83-test
default suite, and 37 sequential CPU integration tests covering WP0--WP2,
AO quadrature, local potentials, static magnetic matrices, and the analytic
overlap oracle. The verifier authenticates 155 arrays and 137 residuals.

The raw execution is
`/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/exact_one_electron_qualification/wp2_20260916T192437Z_3fd2a83b77d1`.
It contains the grids, matrices, kinetic sectors, spectra, transformation
matrices, manifest, residual records, and generated report. Its manifest ID is
`a85f4f0cb78036b8b54e6319cf0c5ce8acc9df2b46c7a6c6bc80f0b447b4c7ae`;
the execution-index SHA-256 is
`d4fac1edec73e21bc7b129763fa66b36a516326fab059ce9802c858b90e20328`.

## G2 decision

The executed evidence satisfies the stated G2 criteria: the two exact routes
agree below the independently measured quadrature floor, gauge-related lower
matrices obey endpoint coefficient congruence, generalized spectra are gauge-
and origin-independent within propagated error, and every exact overlap is a
positive Gram matrix. G2 was accepted after review. The decision is recorded
in `docs/reviews/exact_one_electron_g2_review_20260916.json`, and WP3 is
authorized. The immutable raw result retains its execution-time
`executed_unreviewed` status.
