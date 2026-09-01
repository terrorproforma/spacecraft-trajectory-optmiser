# CPU/reference matrix campaign

`scripts/cpu/run_supported_matrix.py` expands all family axes in the frozen Paper 1 and Paper 2
matrices. It writes one isolated, schema-validated run directory for every coordinate and resumes
only from an existing `result.json`.

## Execution boundary

- CUDA visibility must be empty or `-1`; GPU fields remain null.
- Every worker has an 8 GiB virtual-address limit.
- The declared per-coordinate timeout is 120 seconds.
- Warm-up and measured replay counts remain the frozen 2 and 7.
- Coordinates are never silently resized. A coordinate outside a bounded owner is retained as
  `timeout`, `unsupported`, or another terminal failure.

The following deterministic preflight boundaries are fixed in source before results are read:

- P1-A exact known-optimum sparse fixture: execute through `N=500`; larger coordinates retain
  timeout records.
- P1-B repeated OSQP/Clarabel HCW solve: execute through `N=500`; larger coordinates retain timeout
  records.
- P1-C nonlinear CPU SCvx: execute through `N=100`; larger coordinates retain timeout records.
- P1-D native corrected quaternion transcription/reference replay: all frozen coordinates.
- P1-E native variational-RK4 transfer/reference replay: execute through `N=2000`; larger
  coordinates retain timeout records.
- P1-F deterministic scenario risk/non-anticipativity reference: all frozen coordinates.
- P2-A/P2-B/P2-C bounded orchestration contracts: all coordinates execute but remain unqualified
  unless a physical parameterized owner supplies the required Lambert/route/certification metrics.
- P2-D/P2-E full-mission formulations remain unsupported because this commit contains component
  seams but no authoritative full multi-spacecraft optimization model.

## Evidence interpretation

`executed` is reserved for a record with complete independently recomputed quality fields. A
component can run successfully and still be `unqualified` when a canonical dual/natural residual,
full nonlinear replay, or requested final-polish owner is absent. This distinction prevents native
reference trajectories, risk aggregation, or orchestration checks from being relabelled as
publication-quality solver results.

`scripts/cpu/finalize_supported_matrix.py` validates all records against
`experiments/schema/cpu_reference_result.schema.json`, checks exact coordinate coverage, renders
only data-bearing diagnostics, compares two byte-identical render trees, and separately hashes:

1. semantic source records with observation timing/resource fields removed; and
2. the actual timing observations, which are not expected to be byte-identical across independent
   executions.

The earlier fail-closed G6 archive remains immutable. Its F01-F08/T01-T06 products are referenced
from the new dashboard rather than overwritten with incomplete CPU records.
