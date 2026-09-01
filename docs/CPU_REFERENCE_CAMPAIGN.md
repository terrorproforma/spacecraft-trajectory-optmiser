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

Every coordinate is launched in a fresh subprocess. The subprocess inherits the memory limit and
receives the complete frozen coordinate without a scale cap. A timeout is emitted only when that
launched process reaches the declared wall limit; its start event and any partial stdout/stderr are
retained. Previous preflight timeout predictions are preserved in the earlier archive but are not
eligible as final execution evidence.

Large P1-A and P1-B sparse matrices are assembled directly in COO/CSC form. Assigning a sparse
identity through a SciPy LIL slice is prohibited because SciPy densifies the right-hand side first
(913 GiB for the largest P1-A control block). P1-C preserves a 40-second physical horizon as `N`
changes instead of holding the step duration fixed.

Clarabel qualification is recomputed in its expanded bound/cone coordinates, including
stationarity, primal and dual cone feasibility, and complementarity. P1-F launches the full
scenario CQP for every requested risk coordinate and independently replays every solved scenario.
The current CQP objective is expected cost; worst-case and CVaR coordinates therefore fail closed
as numerical risk-objective mismatches until a frozen epigraph formulation exists.

## Evidence interpretation

`executed` is reserved for a record with complete independently recomputed quality fields. A
component can run successfully and still be `unqualified` when a canonical dual/natural residual,
full nonlinear replay, or requested final-polish owner is absent. This distinction prevents native
reference trajectories, risk aggregation, or orchestration checks from being relabelled as
publication-quality solver results.

The frozen Paper 2 matrix is a scale/method manifest, not a physical instance specification. It
does not provide target ephemerides or orbital states, epoch values and units, gravitational
parameters, spacecraft/thrust/mass data, service demands, depot states/capacities, transfer
resources, or uncertainty distributions. Consequently:

- P2-A/P2-B/P2-C exercise the implemented Lambert-family, graph, route, pricing, master, and
  certification components but remain `unqualified`; creating full-mission objective, residual,
  and certification records would require inventing missing physical inputs.
- P2-D/P2-E remain `unsupported` at this frozen commit because no parameterized multi-spacecraft
  or robust full-mission formulation binds those missing inputs to the G7 component abstractions.

The implemented G7 headers and native smoke tests establish component support, but they do not
define the absent benchmark instances. These dispositions can only change after a frozen,
versioned physical-instance contract is supplied; synthesizing one during the campaign would
change the benchmark after results.

`scripts/cpu/finalize_supported_matrix.py` validates all records against
`experiments/schema/cpu_reference_result.schema.json`, checks exact coordinate coverage, renders
only data-bearing diagnostics, compares two byte-identical render trees, and separately hashes:

1. semantic source records with observation timing/resource fields removed; and
2. the actual timing observations, which are not expected to be byte-identical across independent
   executions.

The earlier fail-closed G6 archive remains immutable. Its F01-F08/T01-T06 products are referenced
from the new dashboard rather than overwritten with incomplete CPU records.
