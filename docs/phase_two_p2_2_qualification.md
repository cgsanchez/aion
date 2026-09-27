# Phase Two P2-2 molecular-transfer qualification

Status: **executed; proposed pass; awaiting explicit user review**

Date: 2026-09-27

Branch: `feature/aion-phase-two`

Base execution commit: `bb49799250d856dd7c181c1dea6ac22772e4dbd6`

Refinement execution commit: `1bb35525ff9d66014548cbdfcf2f11755b886f89`

## 1. Claim under review

P2-2 tests whether the reusable exact straight-Wilson runtime accepted at P2-1
transfers from the H3+/LDA anchor to molecular pure-GGA calculations before
any long propagation or larger-basis campaign is attempted. The bounded claim
is that:

1. the integrated CO/cc-pVDZ/PBE path reproduces the accepted NQ8 stationary
   and short dynamic result;
2. CPU and physical-GPU CO dynamics agree at the declared floor;
3. fixed-geometry NH3/cc-pVDZ/PBE exact-Wilson dynamics reduce to the matching
   ordinary bare-length calculation at zero magnetic field within the
   declared spatial and temporal floors;
4. all action identities and metric diagnostics remain controlled; and
5. the measured cost is sufficient to select P2-3 profiling targets.

This is a short numerical bridge. The NH3 pulse is not a resonant or
spectroscopic calculation, and this gate makes no absorption claim.

## 2. Scientific realization

The CO calculation uses the accepted NQ8 geometry, cc-pVDZ AO basis, PBE
functional, unpruned level-4 PySCF grid, `weigend` RI auxiliary basis, and the
accepted 0.2-au continuous-density temporal-source quench. It is propagated
for eight intervals of 0.025 au on CPU and physical GPU.

The polyatomic bridge uses fixed `G_DZ` NH3, cc-pVDZ, PBE, the same unpruned
level-4 grid policy and RI auxiliary basis, zero magnetic field, and a compact
one-cycle sin-squared vector-potential pulse polarized along the C3 axis. The
pulse has exactly field-free endpoints and zero net vector-potential offset.
The base calculation uses eight intervals of 0.025 au over 0.2 au. Matching
exact-Wilson and ordinary bare-length trajectories are executed on CPU and
physical GPU.

The exact dynamics use the accepted self-consistent two-node fourth-order
Gauss--Magnus propagator. The ordinary bare-length dynamics use the existing
second-order fixed-metric SCEM propagator. Therefore, an exact--bare
observable comparison at a common finite timestep contains the temporal error
of both propagators and is not an order test of either method by itself.

## 3. Immutable executions

The authenticated base campaign is

```text
/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/aion_phase_two/
  p2_2_bridge_20260926T112431Z_bb49799250d8
```

It completed all preparation, CPU, physical-GPU, and analysis stages. All 49
recorded result, provenance, stage, and subordinate-artifact hashes were
independently recomputed and matched.

The base analyzer proposed failure for two checks. Investigation established
that one was not a scientific discrepancy: the reference and qualification
grids contain bitwise-identical coordinates and weights, both with 131,224
points, but their fingerprints differ because the fingerprint includes the
semantic grid kind (`reference` versus `qualification`). The analyzer was
corrected to compare the numerical grid arrays.

The other base miss was bounded: the largest NH3 exact--bare current residual
was `5.4569198e-5` a.u. against a declared `5e-5` a.u. limit. The residual was
confined to the driven Cartesian component and had the alternating phase
pattern expected from temporal discretization. It exceeded the limit by
9.14%, while density, dipole, energy, identities, and CPU/GPU parity passed.

The predeclared bounded supplement is

```text
/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/aion_phase_two/
  p2_2_current_refinement_20260927T144524Z_1bb35525ff9d
```

It halves the timestep to 0.0125 au and uses 16 intervals with otherwise
identical NH3 inputs. Only the physical-GPU exact and bare trajectories are
rerun; the already-passing CPU/GPU parity is inherited from the authenticated
base campaign. The supplement authenticates the base evidence before use.
Its 11 result, provenance, identity, configuration, trajectory, checkpoint,
and status hashes were independently recomputed and matched.

The original base result remains immutable and visibly records its proposed
failure. The supplement corrects the analyzer semantics and supplies the
discriminating timestep evidence; it does not overwrite the base campaign.

## 4. Numerical results

### 4.1 CO accepted-result reproduction

| comparison | measured residual | limit | result |
| --- | ---: | ---: | --- |
| stationary density vs NQ8 | `1.8626e-11` relative | `2e-8` | pass |
| stationary energy vs NQ8 | `3.7016e-10` Ha | `2e-8` Ha | pass |
| final mixed density vs NQ8 | `1.6436e-11` relative | `1e-7` | pass |
| final energy vs NQ8 | `2.4650e-10` Ha | `1e-7` Ha | pass |
| CPU/GPU final density | `7.7370e-13` relative | `2e-8` | pass |
| CPU/GPU energy series | `4.8797e-10` Ha maximum | `2e-8` Ha | pass |

### 4.2 NH3 reduction and refinement

| comparison | base `dt=0.025` | refined `dt=0.0125` | limit | result |
| --- | ---: | ---: | ---: | --- |
| exact--bare current maximum | `5.4569e-5` | `1.2929e-5` | `5e-5` a.u. | pass after refinement |
| exact--bare dipole maximum | `1.5222e-6` | `7.0769e-7` | `5e-5` a.u. | pass |
| exact--bare energy maximum | `2.8771e-7` | `2.8734e-7` | `2e-6` Ha | pass |
| final density | `8.4698e-7` | `8.4713e-7` | `5e-6` relative | pass |

The current residual refinement ratio is `0.23693`. This is close to the
factor `1/4` expected when the leading discrepancy is the second-order bare
propagator's temporal error. At the common coarse output times, the exact
current changes by only `6.0599e-7` a.u. under timestep halving, whereas the
bare current changes by `4.1072e-5` a.u. Thus the evidence identifies temporal
discretization in the bare comparison trajectory, rather than a current
definition, sign, grid, or exact-action problem, as the source of the narrow
base miss.

The reference and exact qualification grids have bitwise-identical numerical
coordinates and weights. The differing semantic fingerprints remain recorded
as useful provenance and are not treated as numerical inequality.

### 4.3 Exact-action identities at the refined timestep

| diagnostic | maximum residual or drift |
| --- | ---: |
| particle number | `8.8818e-15` |
| Ward identity | `1.0842e-19` |
| finite-region weak continuity | `6.5087e-13` |
| power identity | `3.9431e-15` a.u. |
| work--energy balance | `2.8802e-11` Ha |
| density-shell relative norm | `6.3732e-16` |
| cross-metric residual | `7.1442e-16` |
| occupation-spectrum drift | `1.2882e-15` |
| nonlinear residual | `1.3071e-14` |

Every inherited and refined check in the composite result passes.

## 5. Cost evidence and P2-3 targets

The base exact-action timings were:

| system/backend | elapsed time (s) | action evaluations | action time (s) |
| --- | ---: | ---: | ---: |
| CO CPU | `24348.34` | 80 | `1125.94` |
| CO GPU | `343.32` | 80 | `63.71` |
| NH3 CPU | `16663.05` | 80 | `262.14` |
| NH3 GPU | `368.11` | 80 | `62.84` |

The 16-interval refined NH3 exact-GPU calculation required `671.40` s, of
which `123.67` s was attributed to 158 instrumented action evaluations. The
matching bare-GPU trajectory required `2.74` s. Peak process RSS in the
refinement was approximately 1.16 GB.

The near-linear doubling of the exact-GPU time under timestep halving is
expected. More importantly, most elapsed exact-runtime time is outside the
instrumented nonlinear action evaluation. This makes observation scheduling,
host/device movement, repeated construction, and I/O explicit P2-3 profiling
targets. These machine-specific measurements are not production performance
guarantees.

## 6. Software and execution controls

The analyzer correction and refinement driver passed Ruff formatting and
lint, strict Mypy, 92 fast tests, and 22 physical-GPU tests before execution.
The campaign ran as one detached process through `tools/gpu-python`, with at
most eight host threads. Physical device residency was asserted for both
exact and bare trajectories. The repository was clean at execution commit
`1bb35525ff9d66014548cbdfcf2f11755b886f89`.

## 7. Bounded interpretation and next gate

P2-2 qualifies the reusable exact-Wilson pure-GGA molecular transfer and the
short NH3 zero-magnetic-field integration bridge. It does not qualify a long
trajectory, spectrum, kick, larger basis, reduced P0/E1 action, general
time-dependent magnetic envelope, spatially nonuniform source, EELS
potential, pseudopotential, hybrid functional, moving nucleus, or periodic
system. It does not establish equality of distinct finite-field actions.

The evidence supports a **P2-2 pass proposal**. Only explicit user acceptance
may change the gate from `executed_unreviewed` to `accepted`. If accepted,
P2-3 profiles and optimizes the measured exact-runtime bottlenecks without
changing the action or weakening any scientific tolerance.
