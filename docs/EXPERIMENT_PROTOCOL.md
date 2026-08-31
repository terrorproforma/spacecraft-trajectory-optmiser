# SpacePDHCG experiment and evidence protocol

This protocol governs every performance, scaling, memory, energy, and numerical-quality claim in
Paper 1. A solver timing without a committed run manifest is development telemetry, not evidence.

## 1. Evidence units

One run consists of exactly one problem instance, solver mode, requested tolerance, warm-start
mode, hardware configuration, and replicate index. Each run writes a manifest conforming to
`experiments/schema/run_manifest.schema.json` and references any larger artifacts by path and
SHA-256 digest.

The frozen initial sweep is `experiments/configs/paper1_sweep_v0.json`. Changes require a new
suite identifier; results from different suite versions are never pooled silently.

## 2. Repository and solver identity

Every run records:

- the exact SpacePDHCG commit;
- whether the checkout was dirty;
- the exact upstream PDHCG revision;
- QOCO-GPU, CuClarabel, Clarabel, OSQP, compiler, CUDA, driver, MPI, and NCCL versions where
  applicable;
- all solver parameters that differ from committed defaults.

Tracking a moving branch is prohibited for publication runs.

## 3. Hardware identity

The manifest records CPU model, logical cores, host memory, accelerator vendor/model/count,
driver, CUDA runtime, and interconnect. Multi-GPU runs additionally record the logical GPU grid,
rank-to-device mapping, process placement, and collective implementation.

Clock-locking, persistence mode, power limits, MIG configuration, and competing workloads must be
recorded in run notes. Energy results are inadmissible when the measurement source or sampling
interval is unknown.

## 4. Timing boundaries

The following intervals are measured separately:

1. one-time symbolic construction and device allocation;
2. numerical coefficient generation;
3. host-to-device or device-to-device coefficient update;
4. scaling and preconditioner refresh;
5. inner solver execution;
6. nonlinear rollout and path verification;
7. trust-region and forcing decisions;
8. final optional interior-point polish;
9. end-to-end wall time.

Asynchronous GPU work is bracketed by events on the relevant stream and synchronised only at the
declared timing boundary. Host wall time is reported in addition to device event time. H1 uses the
end-to-end denominator, not kernel time.

## 5. Warm-up and replication

Each configuration receives the committed number of unreported warm-ups followed by at least five
reported replicates. First-run JIT compilation, dynamic library loading, CUDA context creation, and
memory-pool growth belong to setup unless the deployment model demonstrably amortises them.

Run order is randomised within hardware-feasible blocks. The randomisation seed is committed in
the suite artifact. Median, interquartile range, minimum, and maximum are reported; a single best
run is never the headline result.

## 6. Matched-quality gate

A speed comparison is valid only when all compared solutions satisfy the suite's declared quality
limits. Required independent checks include:

- canonical primal and dual residuals;
- objective gap against a high-accuracy reference where available;
- nonlinear dynamics defect after independent propagation;
- maximum path and continuous-time certificate violation;
- terminal error;
- non-anticipativity violation for robust scenarios;
- risk-objective recomputation for expected, worst-case, or CVaR modes.

A solver returning a permissive status such as `AlmostSolved` is accepted only after residual and
nonlinear-quality qualification. Failed and timed-out runs remain in the dataset.

## 7. Fixed versus adaptive accuracy

For D, fixed, adaptive, and hybrid modes start from the same reference trajectory and use the same
outer trust-region logic. The adaptive run records the requested and achieved residual at every
outer iteration. A fixed comparator is rerun at the tightest tolerance reached by the adaptive
schedule and at the final quality tolerance.

Work savings are reported using total inner iterations, end-to-end time, and energy at matched
final nonlinear quality. A lower inner-solver time with more rejected outer steps is not a win.

## 8. Persistent versus one-shot solving

Persistent and one-shot modes use identical canonical values and initial iterates. Persistent mode
may retain topology, descriptors, scaling, memory pools, and primal-dual iterates; one-shot mode
must rebuild the public upstream model exactly as documented. The comparison reports the first
solve and steady-state repeated solves separately.

## 9. Multi-GPU protocol

Strong scaling fixes the complete problem while varying GPU count. Weak scaling fixes nodes and
scenarios per GPU. Both report:

- local compute time;
- collective time;
- exposed communication after overlap;
- bytes per collective and aggregate bytes;
- load imbalance;
- peak memory per device;
- numerical agreement with the monolithic CPU oracle.

Generic nonzero-balanced and scenario-aware partitions use the same global CQP and stopping rule.

## 10. Result immutability

Raw manifests and logs are append-only. Data cleaning is implemented as a committed script that
produces a new derived table without modifying source records. Manuscript figures identify their
input suite version and commit. Any excluded run is listed with a machine-readable reason.

## 11. Current hardware boundary

CPU reference runs, schema validation, deterministic C++ tests, and dry-run manifest production
can execute now. CUDA speed, memory, energy, persistence-overhead, and NCCL claims remain blocked
until a compatible NVIDIA runner produces archived manifests. No placeholder number may enter a
results table.
