# Chapter 13 NQ9 synthesis and Chapter 14 handoff

Date: 2026-09-24

Gate state: **accepted by the user on 2026-09-24**

Accepted prerequisites:
`docs/reviews/chapter13_nq0_review_20260921.json` through
`docs/reviews/chapter13_nq8_review_20260924.json`.

Acceptance record: `docs/reviews/chapter13_nq9_review_20260924.json`.

This is the review entry point for NQ9. The full human-readable report is
`docs/chapter13_nq9_synthesis_report.tex`; the proposed writing structure is
`docs/chapter13_nq9_chapter14_outline.md`. NQ9 is reconstructive: it
authenticates and synthesizes accepted NQ0--NQ8 evidence without rerunning or
rewriting those campaigns.

## Synthesis result

The independent synthesis at commit `74a493241667` authenticated:

- all nine accepted NQ0--NQ8 review records and their entry documents;
- 41 primary result, provenance, completion, analysis, reference, refinement,
  and interrupted-run artifacts;
- 131 accepted numerical measurements;
- 49 declared limitations;
- 13 equation-to-code-to-test-to-evidence claims;
- eight explicit negative or failed results; and
- nine selected figures copied byte-for-byte from accepted analyses.

No required scientific artifact was missing and no recorded scientific hash
failed. The NQ8 shell-managed `run.log` defect remains explicitly excluded;
all NQ8 numerical artifacts authenticate.

The synthesis result and provenance hashes are:

| artifact | SHA-256 |
|---|---|
| `result.json` | `4eca05f7fbf774a0a47a6d44146929d35941f7fb013acfa6ace973c5080c6671` |
| `provenance.json` | `f17444d63cdbe1974353b596d43827291270c663112dbff92f9baba975e66a83` |

Immutable synthesis root:

```text
/home/cgs/00_WORK/Projection_Code/CALCULATIONS/campaigns/
chapter13_wilson_adiabatic_qualification/
nq9_synthesis_20260924T220919Z_74a493241667
```

## Bounded scientific conclusion

The accepted evidence supports this proposed NQ9 conclusion:

> The exact straight-Wilson finite molecular model, Coulomb-metric RI Hartree
> closure, restricted pure-LDA and pure-PBE adiabatic actions, their
> variational lower matrices and fixed-history weak sources, and
> connection-aware nonlinear propagation form one numerically consistent
> implementation over the accepted H2, LiH, H3+, and CO fixtures. Strict C1
> is a controlled first-order reduced working action inside the sampled
> pure-LDA H3+ domain, whose basis-conditioned failures remain explicit. The
> exact Wilson action remains the reference.

The covered orbital bases are STO-3G and cc-pVDZ. Aug-cc-pVDZ is included
only in the reduced-domain stress scan. Accepted production fixtures use the
weigend auxiliary basis. Accepted dynamics extend to 2.0 a.u. for the LDA
model comparison and 0.2 a.u. for the PBE transfer test.

## Numerical limits that control interpretation

Representative limiting values are:

| claim | accepted limiting evidence |
|---|---:|
| Wilson density direct/factorized relative difference | `2.54e-16` |
| finest density normalization grid residual | `2.32e-11` |
| selected RI auxiliary energy error relative to exact four-centre reference | `6.90e-5` |
| LDA same-grid energy residual | `1.33e-15 Ha` |
| PBE same-grid energy residual | `5.33e-15 Ha` |
| accepted LDA complete source residual | `3.64e-10` |
| best CO/PBE complete source residual | `7.39e-9` |
| minimum nonlinear final-density timestep order | `3.7909` |
| finest LDA endpoint work--energy residual | `4.87e-11 Ha` |
| PBE transfer work--energy residual | `4.06e-10 Ha` |
| strict-C1 low-field state-error order | `1.99996` |
| smallest C1 model/timestep-error ratio | `6.79e3` |

The complete numerical ledger is `tables/numerical_limits.csv` in the
synthesis root.

## Reduced-model recommendation

Strict C1 is the recommended near-term reduced action only within the sampled
pure-LDA H3+ domain. It removes the leading state and lower-matrix error and
is well separated from timestep uncertainty. Density-resummed C1 showed no
systematic advantage and is retained as a separately labelled diagnostic
action.

This recommendation is basis conditioned. Every tested aug-cc-pVDZ reduced
level failed metric positivity at the first nonzero sampled field,
`Bz=0.03` a.u., while the exact Wilson overlap remained positive. No clipping,
regularization, or metric projection is accepted.

## Explicit negative results

The synthesis retains, rather than hides:

- the resolved RI auxiliary-basis approximation error;
- incomplete, incorrectly labelled, and superseded NQ4 artifacts;
- the wrong-launcher NQ5 GPU failure followed by the corrected passing run;
- the NQ6 host-power interruption and the failed preliminary threshold set;
- 48 visible reduced-metric domain failures;
- the lack of systematic advantage from density resummation;
- the deterministic CO RI-Hartree source finite-difference floor; and
- the NQ8 external-log hash defect and superseded preliminary analysis.

The exact descriptions and dispositions are in
`tables/negative_results.csv`; all gate limitations are in
`tables/limitations.csv`.

## Remaining exclusions

NQ9 does not qualify:

- moving nuclei or the associated nonadiabatic connection;
- spatially nonuniform propagated fields;
- Maxwell backreaction;
- periodic systems;
- pseudopotential gauge/current terms;
- spin polarization, hybrids, meta-GGAs, or nonlocal correlation;
- an exact interacting transverse current beyond the declared auxiliary
  adiabatic Kohn--Sham action;
- reduced GGA P0/E1/C1 actions; or
- long-time stability, spectroscopy, systematic basis convergence, or broad
  chemical production use.

## Evidence index

The synthesis root contains:

- `result.json`, `provenance.json`, and `completed.json`;
- `tables/accepted_gates.csv`;
- `tables/artifact_authentication.csv`;
- `tables/numerical_limits.csv`;
- `tables/equation_code_evidence.csv`;
- `tables/limitations.csv`;
- `tables/negative_results.csv`;
- `tables/figure_manifest.csv`; and
- the nine authenticated selected figures.

The full report indexes each accepted gate, module, test family, raw root, and
interpretive boundary. The user accepted the bounded synthesis on 2026-09-24.
The separate review record carries that decision and closes the Chapter 13
qualification plan without altering the immutable execution artifacts.
