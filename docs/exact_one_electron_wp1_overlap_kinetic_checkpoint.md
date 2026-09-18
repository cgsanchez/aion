# WP1 overlap/kinetic checkpoint

Status: **reviewed and accepted as the prerequisite checkpoint**
Execution: `20260916T184810Z_3fd2a83b77d1`
Gate effect: G0 accepted and remainder of WP1 authorized; G1 remains open

This is the deliberately limited checkpoint required before adding nuclear
attraction. It uses the immutable H--H/STO-3G fixture, zero electromagnetic
field, real spherical AOs, unpruned PySCF atom-centred grids, and the CPU
backend. It evaluates no magnetic quantity.

## Implementation evidence

- `OneElectronReferenceConfig` and `OneElectronAOReference` provide an
  occupancy-independent AO/integral reference.
- `prepare_one_electron_ao_reference` creates a PySCF `Mole`, libcint core
  matrices, AO anchors, and exact shell metadata without constructing or
  running RKS.
- `AOQuadrature` accepts that reference only with an explicit qualification
  grid; it cannot silently borrow a DFT/SCF grid.
- `evaluate_zero_field_overlap_kinetic` reconstructs overlap from AO values
  and kinetic energy from the weak-form first-derivative contraction.
- The pre-existing magnetic evaluator is unchanged in scope and explicitly
  remains gated to `PreparedReference` until G1 is accepted.

## Executed numerical evidence

The declared relative matrix target is `1e-5`, one order below the accepted
plan's smallest physical reporting threshold (`1e-4`). The Hermiticity target
is `1e-12`.

| Grid level | Points | Overlap relative Frobenius | Kinetic relative Frobenius | Overlap successive | Kinetic successive | Maximum Hermiticity |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,000 | 1.193807e-3 | 6.447939e-4 | -- | -- | 0.000000e0 |
| 1 | 6,600 | 1.011447e-6 | 5.240661e-6 | 1.194347e-3 | 6.469208e-4 | 9.268922e-17 |
| 2 | 15,520 | 4.263067e-9 | 1.108313e-7 | 1.014961e-6 | 5.341134e-6 | 0.000000e0 |
| 3 | 30,200 | 1.598904e-10 | 5.398146e-9 | 4.226334e-9 | 1.054608e-7 | 9.268930e-17 |
| 4 | 52,080 | 1.510281e-11 | 2.820751e-10 | 1.699337e-10 | 5.677581e-9 | 9.268930e-17 |

The analytic residuals decrease monotonically across the full sequence. Both
matrices meet the declared `1e-5` target from level 1 onward. Level 4 is below
that target by roughly six orders of magnitude for overlap and four orders for
kinetic. This is evidence for this fixture and these two matrices only.

The raw execution is stored at
`/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/exact_one_electron_qualification/20260916T184810Z_3fd2a83b77d1`.
All five manifest identities, five NPZ hashes, ten grid-array hashes, and twenty
matrix hashes were recomputed successfully after execution.

## Software verification

- Ruff: passed.
- Mypy: passed for 72 source files.
- Default test suite: 83 passed, 94 deselected.
- New zero-field integration tests: 4 passed.
- Existing AO-quadrature integration regressions: 7 passed.

The new tests also replace `RKS.kernel` with a failing sentinel and demonstrate
that AO-reference preparation and deterministic reconstruction do not invoke
SCF.

## Review boundary

This checkpoint supports reviewing G0 and authorizing the remainder of WP1. It
does **not** complete G1. If the checkpoint is accepted, the next work is
nuclear-attraction quadrature, an alternative pruning rule, AO derivative and
momentum-sign checks, atom-pair numerical floors, and the first O--H fixture.
