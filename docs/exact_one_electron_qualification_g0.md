# Exact one-electron Wilson qualification: G0 record

Status: G0 reviewed and accepted by the user on 16 September 2026
Governing plan: `exact_one_electron_numerical_qualification_plan.md`
Implementation branch observed before edits: `feature/magnetic-matrix-benchmark`
Implementation baseline observed before edits: `3fd2a83b77d1db64451a64e7a3926f185529525b`

This record freezes the authority, conventions, minimal fixture, and record
schemas for WP0. G0 was accepted after review of the authenticated overlap and
kinetic checkpoint. The decision is recorded in
`docs/reviews/exact_one_electron_g0_review_20260916.json`. It makes no
finite-field claim and does not accept G1.

## Authority and evidence states

Mathematical identities are governed by Chapters 3--10 of the mathematical
formalism book. The implementation-facing specialization is
`exact_wilson_matrix_element_qualification.tex`. The execution order and gate
rules are governed by `exact_one_electron_numerical_qualification_plan.md`.

The following states remain distinct in code, reports, and review:

1. **Mathematical identity:** derived in an authoritative theory source.
2. **Implemented:** represented by a named code path and tests.
3. **Numerically executed:** a manifest and raw residuals exist for one
   declared input.
4. **Reviewed and accepted:** the user has reviewed the evidence and the gate
   table names its immutable artifacts.

Existing MB0--MB8 material is candidate evidence. Its earlier milestone names
do not automatically satisfy G0--G8.

## Frozen conventions

| Object | Frozen meaning |
|---|---|
| Units | Atomic units internally. |
| Particle data | Electron default: `q=-1`, `m=1`, `hbar=1`. |
| Matrix indices | Row `mu` is the bra AO; column `nu` is the ket AO. |
| AO frame | Real spherical Gaussian AOs in PySCF ordering and normalization. |
| AO anchor | The nucleus owning the PySCF AO shell. Several AOs may share an anchor. |
| Anchor displacement | `R_mu - R_nu`. |
| Endpoint path | Straight path from ket anchor `R_nu` to bra anchor `R_mu`. |
| Anchor-to-point path | Straight path from `R_mu` to electronic point `r`. |
| Triangle loop | `R_nu -> r -> R_mu -> R_nu`. |
| Canonical momentum | `p=-i*hbar*grad`; PySCF grid derivatives act on the ket. |
| Mechanical momentum | `pi_A=p-q*A`. |
| Full matrices | Complex arrays, including at zero field. |
| Pair reversal | Exchange bra and ket and reverse all oriented geometry. |
| Gauge change | Change the potential representative only; anchors, bare AOs, paths, grid, and nuclei remain fixed. |

These conventions are identical to the existing
`magnetic_matrix_benchmark_contract.md`. The latter remains the detailed
array-layout contract for the reused evaluator.

## Equation-to-evaluator traceability

| Mathematical object | Book authority | Qualification note | Implementation |
|---|---|---|---|
| Fixed AO anchors and bare overlap | `eq:wilson-anchor-position`, `eq:wilson-bare-overlap-matrix` | Scope and meaning of exactness | `build_anchor_topology`, `build_core_operators` |
| Straight anchor-to-point path and open Wilson factor | `eq:wilson-straight-path`, `eq:wilson-line-integral-parametric`, `eq:wilson-open-factor` | Wilson frame equations | `AffineMagneticGauge.anchor_to_point_line_integrals` and the direct route in `evaluate_magnetic_one_electron_matrices` |
| Dressed AO frame | `eq:wilson-dressed-orbital-function` | Wilson frame equations | Direct route in `evaluate_magnetic_one_electron_matrices` |
| Ket-to-bra endpoint link | `eq:thick-link-center-path`, `eq:thick-link-endpoint-link` | Endpoint-link definition | `endpoint_links` |
| Triangle orientation, factor, and flux | `eq:thick-link-triangle-orientation`, `eq:thick-link-triangle-factor-definition`, `eq:thick-link-triangle-flux` | `eq:triangle-factor` | `triangle_fluxes`, `triangle_phases` |
| Exact barred overlap and lower factorization | `eq:thick-link-barred-overlap`, `eq:thick-link-overlap-factorization` | `eq:exact-overlap` | overlap contractions in `evaluate_magnetic_one_electron_matrices` |
| Anchored magnetic vector and dressed momentum | `eq:thick-link-imported-magnetic-vector`, `eq:thick-link-imported-dressed-momentum` | `eq:anchored-C` | `anchored_vectors` and the direct/factorized mechanical derivatives |
| Exact barred kinetic and local potential | `eq:thick-link-barred-kinetic-amplitude`, `eq:thick-link-barred-potential-amplitude` | `eq:exact-kinetic` | kinetic and nuclear-attraction contractions in `evaluate_magnetic_one_electron_matrices` |
| Exact lower mechanical factorization | `eq:thick-link-mechanical-factorization` | `eq:exact-mechanical` | `MagneticOneElectronResult.lower_exact` and `.lower_exact_grid` |
| Uniform-field `C` and triangle flux | `eq:curvature-expansion-uniform-B-example-C`, `eq:curvature-expansion-uniform-B-example-flux` | `eq:uniform-F`, `eq:uniform-C` | `anchored_vectors`, `triangle_fluxes` |
| Internal amplitude grading | `eq:curvature-expansion-amplitude-triangle-factor`, `eq:curvature-expansion-epsilon-overlap`, `eq:curvature-expansion-epsilon-kinetic` | P0 comparison and kinetic sectors | `ScalarMagneticHierarchy`, `KineticMagneticHierarchy` |
| First uniform magnetic response | `eq:first-spread-uniform-triangle-generator`, `eq:first-spread-uniform-anchored-generators`, `eq:first-spread-uniform-overlap-response`, `eq:first-spread-uniform-mechanical-response` | Derivative and truncation tests | first-order sectors in `evaluate_magnetic_one_electron_matrices`; action-level gB1/B1/C1 mapping remains unaccepted |
| Zero-field AO overlap and kinetic quadrature | Zero-field limits `eq:thick-link-zero-magnetic-overlap`, `eq:thick-link-zero-field-mechanical-amplitude` | Priority 1.1 | `evaluate_zero_field_overlap_kinetic` |

The old static hierarchy names arrays by matrix expansion order. The newer
gB1, B1, and C1 names refer to complete action-level models. No old array is
credited as an action-level model merely because both use “B1”.

## Minimal immutable fixture

The G0 fixture is
`tests/fixtures/exact_one_electron/hh_sto3g.fixture.json`:

- nuclei: H at `(0,0,-0.7)` bohr and H at `(0,0,+0.7)` bohr;
- basis: STO-3G, real spherical;
- electromagnetic origin: `(0,0,0)` bohr;
- AO ordering: PySCF labels `0 H 1s` and `1 H 1s`;
- AO-to-anchor map: `[0,1]`;
- analytic matrices: overlap, kinetic, nuclear attraction, position, and
  canonical momentum;
- exact shell layout and fingerprints;
- no occupation, density, functional, SCF state, or propagation state.

`prepare_one_electron_ao_reference` reconstructs this object directly from
PySCF's molecule and libcint interfaces. It never creates an RKS object and
never invokes SCF. Runtime reconstruction re-evaluates and authenticates the
core, anchor, and basis fingerprints.

## Run-manifest and matrix-result schemas

The normative G0 schemas are:

- `docs/schemas/exact_one_electron_run_manifest.schema.json`;
- `docs/schemas/exact_one_electron_matrix_result.schema.json`.

They require the code state, managed environment, fixture identity, exact AO
metadata, quadrature identity, electromagnetic/source convention, numerical
tolerances, immutable input hashes, raw matrix records, residual definitions,
and an explicit review state. `schema_example_not_executed` is the only
allowed status for an illustrative record without numerical evidence.

Raw campaign artifacts will live below
`/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/exact_one_electron_qualification`.
The Aion worktree retains code, compact reports, schemas, and regression
fixtures; the formalism repository retains no raw campaign arrays.

## Existing evidence mapped without acceptance

| New gate | Existing candidate evidence | Missing before acceptance |
|---|---|---|
| G0 | MB0 convention contract, MB6 persistence, new occupancy-independent fixture | User review of this record and schema boundary |
| G1 | MB2 AO quadrature and H2/LiH grid tests | Full matrix-specific grid sequence, alternate pruning, O--H, and nuclear-attraction checkpoint |
| G2 | MB3/MB7 direct, factorized, GIAO, gauge, origin, and rotation tests | Credit only after G1; rerun under new manifest and tolerance rules |
| G3 | MB8 H2/LiH campaign | Required O--H, N2, CO, distance, diffuse, block, and spectral campaign |
| G4 | MB3/MB7 first/second magnetic coefficients | Explicit gB1/B1/C1 action-sector mapping and electric E1 qualification |
| G5 | MB7 multicentre stress fixtures | New three-proton manifests, loop-flux report, and selected positive-metric dynamical cases |
| G6--G7 | No exact-Wilson accepted evidence | Exact time connection, propagation, action derivatives, continuity, and power |

## Immediate numerical checkpoint

The first implemented WP1 slice reconstructs only overlap and kinetic energy
on explicit unpruned PySCF grids. Nuclear attraction is deliberately absent
until this checkpoint is reviewed. The reusable implementation is
`evaluate_zero_field_overlap_kinetic`; it uses the same blocked AO value and
gradient stream that the existing magnetic evaluator consumes.
