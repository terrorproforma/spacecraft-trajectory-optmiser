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
Expected cost uses the native block-arrow quadratic objective. Worst-case and CVaR use one sparse
rotated-SOC quadratic-cost epigraph per scenario; CVaR additionally owns the threshold and
non-negative excess variables. All variants retain exact non-anticipativity rows and run through a
persistent Clarabel SCvx loop. Qualification still requires the independently replayed nonlinear
trajectory—not merely the conic epigraph—to pass.

## Evidence interpretation

`executed` is reserved for a record with complete independently recomputed quality fields. A
component can run successfully and still be `unqualified` when a canonical dual/natural residual,
full nonlinear replay, or requested final-polish owner is absent. This distinction prevents native
reference trajectories, risk aggregation, or orchestration checks from being relabelled as
publication-quality solver results.

The original frozen Paper 2 matrix is a scale/method manifest, not a physical instance
specification. The completed `cpu-actual-c5a4991` archive therefore remains correctly censored.
Follow-on runs use the separately versioned `orbitweaver-earth-leo-v1` contract in
`benchmarks/paper2_instances.json`. It freezes SI units, deterministic circular Earth-orbit target
states, epoch spacing, vehicle/resource limits, uncertainty quantiles, and independent J2-RK4
certification tolerances. Its content hash is carried independently from the matrix hash.

The physical contract removes the instance-data ambiguity; it does not itself qualify component
fixtures. P2 rows remain censored until Lambert/trajectory, route/master, and certification owners
consume that contract and emit their actual residuals and bounds.

Selective delta runs use `--families` and add coordinate, matrix, instance, environment, and driver
hashes. `--preserve-existing` resumes only exact hash matches; a mismatched prior result is renamed
to `result.stale.<hash>.json` before execution rather than silently overwritten.

`scripts/cpu/finalize_supported_matrix.py` validates all records against
`experiments/schema/cpu_reference_result.schema.json`, checks exact coordinate coverage, renders
only data-bearing diagnostics, compares two byte-identical render trees, and separately hashes:

1. semantic source records with observation timing/resource fields removed; and
2. the actual timing observations, which are not expected to be byte-identical across independent
   executions.

The earlier fail-closed G6 archive remains immutable. Its F01-F08/T01-T06 products are referenced
from the new dashboard rather than overwritten with incomplete CPU records.
