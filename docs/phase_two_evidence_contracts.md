# Phase Two evidence contracts

Status: P2-0 implementation contract

## Purpose

Phase Two uses immutable campaign directories below
`CALCULATIONS/campaigns/aion_phase_two`. A completed calculation or synthesis
contains `result.json`, `provenance.json`, and `completed.json`. A failure is
published as `failure.json` without a successful completion record.

The schemas are independently versioned from the Python package. Unknown
major versions are rejected. New optional fields may be added in a compatible
minor version; a field may not be reinterpreted in place.

## Common completion record

Schema: `aion.phase-two.completed`, version `1.0.0`.

Required fields are the schema and version, status, and SHA-256 values for the
result and provenance records. The result and provenance hashes are recomputed
before an artifact is used in a synthesis.

## Common provenance record

Schema: `aion.phase-two.provenance`, version `1.0.0`.

It records the UTC timestamp, repository, Git revision and porcelain state,
Python executable and version, host, thread policy where applicable, and the
hash of every subordinate artifact used by the result. Scientific numerical
records additionally identify dependencies, backend and physical device,
input fingerprints, and execution command.

## P2-0 reconciliation result

Schema: `aion.phase-two.p2-0.reconciliation-result`, version `1.0.0`.

It records the accepted parent relationship, NQ9 verifier output, package and
dependency versions, the inherited WP7 state, and the authenticated ammonia
heritage manifest. Historical evidence is explicitly labelled and cannot be
promoted to exact-Wilson evidence by inclusion in this manifest.

## Ammonia heritage manifest

Schema: `aion.phase-two.p2-0.ammonia-heritage`, version `1.0.0`.

Each record contains role, absolute and campaign-relative paths, byte size,
SHA-256, admissible use, and limitation. The manifest remains external to the
historical campaign and does not alter its files.

## P2-0 quality result

Schema: `aion.phase-two.p2-0.quality-result`, version `1.0.0`.

Each gate records its exact argument vector, exit status, elapsed wall time,
pass state, and authenticated combined stdout/stderr log. CPU and GPU tests
run serially at the gate level. The GPU gate is present only when physical
device access was explicitly requested and obtained.

## P2-0 benchmark result

Schema: `aion.phase-two.p2-0.benchmark-result`, version `1.0.0`.

Each system record identifies the accepted input, action branch, basis, grid,
auxiliary basis, source, stage timings, process peak resident memory, warm-
started stationary residuals, one-step propagation diagnostics, and numerical
backend. A warm-start timing is not represented as a from-scratch stationary
cost.

## Failure record

Schema: `aion.phase-two.failure`, version `1.0.0`.

A failure record contains the failed phase, exception type and message,
traceback, last completed substage, and any artifacts safe to authenticate.
It is retained and never replaced in place. A corrected rerun uses a new
campaign directory.

## Review state

Generated results use `executed_unreviewed`. Analysis may propose a pass or a
bounded failure, but only a separate record under `docs/reviews` can record
the user's decision. A review record authenticates the result and entry
document hashes that the user reviewed.
