# Exact one-electron WP1 zero-field qualification

Status: **G1 reviewed and accepted by the user on 16 September 2026**
Execution: `wp1_20260916T190346Z_3fd2a83b77d1`
Finite magnetic-field evidence: none

## Scope and evidence state

This report completes the numerical work requested by WP1 but does not accept
G1. It covers H--H/STO-3G and O--H/STO-3G, explicit unpruned and NWChem-pruned
PySCF grids, and grid levels 0 through 5. Every run uses the occupancy-
independent AO reference and therefore performs no RKS calculation.

The mathematical zero-field identities come from the accepted formalism. The
implemented features are reusable Aion APIs. The values below are executed
numerical evidence tied to raw arrays and manifests. Acceptance remains a user
decision.

## Implemented qualification API

- `AOGridPolicy` now records the pruning choice explicitly; supported WP1
  choices are `none` and `nwchem`.
- `evaluate_zero_field_one_electron` reconstructs overlap, weak-form kinetic
  energy, all-electron local nuclear attraction, total mechanical matrix, and
  canonical momentum in a single blocked AO stream.
- The momentum contraction uses `-i*hbar` times the derivative of the ket AO.
  Both the correct-sign and deliberately reversed-sign residuals are retained.
- `atom_pair_block_residuals` reports every ordered bra-atom/ket-atom block by
  absolute Frobenius norm, normalized Frobenius norm, and maximum absolute
  element. No near-zero elementwise division is used.
- The local nuclear-attraction provider now accepts the occupancy-independent
  one-electron reference while retaining the all-electron/local restriction.

## Numerical results

The declared normalized target is `1e-5`, one order below the smallest planned
physical threshold (`1e-4`). Working floors are the larger of the level-5
analytic residual and the level-4-to-5 change. Hermiticity is accepted against
the larger of a conditioning-scaled roundoff bound and four times that
matrix-family floor.

### Proposed default grid: unpruned level 4

| Fixture | S | T | Vnuc | K | canonical momentum |
|---|---:|---:|---:|---:|---:|
| H--H/STO-3G | 1.510e-11 | 2.821e-10 | 3.069e-10 | 6.050e-10 | 1.890e-10 |
| O--H/STO-3G | 3.213e-9 | 1.188e-8 | 1.762e-9 | 6.664e-9 | 2.127e-8 |

All entries are full-matrix relative Frobenius residuals against independent
PySCF/libcint matrices. The level-4-to-3 changes are also below `1e-5` for all
families and both fixtures.

### Measured level-5 working floors

| Fixture/grid | S | T | Vnuc | K | canonical momentum |
|---|---:|---:|---:|---:|---:|
| H--H, unpruned | 1.419e-11 | 2.849e-10 | 2.635e-10 | 5.416e-10 | 1.695e-10 |
| H--H, NWChem | 1.238e-11 | 2.917e-10 | 2.619e-10 | 5.431e-10 | 1.601e-10 |
| O--H, unpruned | 3.004e-9 | 8.503e-9 | 1.379e-9 | 4.875e-9 | 2.212e-8 |
| O--H, NWChem | 3.004e-9 | 8.503e-9 | 1.379e-9 | 4.876e-9 | 2.213e-8 |

The limiting channel is O--H canonical momentum, with a floor of approximately
`2.21e-8`; this is still about 450 times smaller than the `1e-5` grid target.

At level 4, unpruned-versus-NWChem differences are at most `9.48e-12` for H--H
and `1.41e-10` for O--H. NWChem pruning reduces the level-4 O--H grid from
79,144 to 43,968 points without a material shift, but unpruned level 4 remains
the proposed reference default. Unpruned level 5 is the refinement grid and
NWChem-pruned level 4 is the alternative-pruning check.

## Internal checks

- Every matrix family converges toward its analytic reference in both pruning
  sequences.
- Nuclear attraction is independently converged; it is not inferred from
  overlap or kinetic behavior.
- `K` equals `T + Vnuc` for both analytic and quadrature arrays exactly as
  stored.
- All ordered atom-pair block maxima pass the declared `1e-5` target.
- The finest O--H momentum Hermiticity residual is `1.48e-8`, below the fixed
  four-times-floor tolerance of `8.85e-8`. Scalar-matrix Hermiticity remains at
  roundoff.
- At level 4, the wrong momentum sign is distinguished from the correct sign
  by factors of `5.18e9` for H--H and `9.40e7` for O--H.
- All four pruning/system sequences pass the target, convergence,
  Hermiticity, and sign tests. Both level-4 pruning comparisons pass.

## Verification and artifacts

Ruff and mypy pass. The default suite reports 83 passes. Eight exact-one-
electron integration tests, seven existing AO-quadrature regressions, and five
local-potential regressions pass. The final campaign contains 24 members and
74 files (approximately 24 MiB).

The raw execution is
`/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/exact_one_electron_qualification/wp1_20260916T190346Z_3fd2a83b77d1`.
Its execution-index SHA-256 is
`0cbfca29db9db7049885105bcc2fdb5c321fb51507cd0458cb0a4da27a97f19f`;
the generated report SHA-256 is
`e36760572eede92c4f39dad248857417bf2c9f69c6ada192dee0b8e9c7e97014`.

All 24 manifest identities and artifacts, 240 matrix hashes, grid hashes,
reported residuals, atom-pair maxima, and stored mechanical sums were
independently recomputed after execution.

## Proposed G1 decision

The executed evidence satisfies the declared WP1 criteria. The recommendation
is to accept G1 with unpruned level 4 as the default static qualification grid,
unpruned level 5 as its refinement, and NWChem-pruned level 4 as the independent
pruning check. G1 was accepted after review. The decision is recorded in
`docs/reviews/exact_one_electron_g1_review_20260916.json`, and WP2 is
authorized.
