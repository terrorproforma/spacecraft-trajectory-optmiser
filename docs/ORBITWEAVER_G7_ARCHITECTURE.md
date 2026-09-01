# OrbitWeaver G7 GPU-scale architecture

## Scope and evidence rule

G7 composes the existing OrbitWeaver CPU truth models and public G2/G3 persistent
single-GPU APIs. It does not import the uncommitted G5 implementation. This branch
contains implementation and bounded correctness evidence only. It makes no throughput,
energy, scaling, tractability, optimality, or Paper 2 completion claim.

Evidence labels are intentionally separate:

- `implemented_compiled`: a target configured and compiled;
- `cpu_correctness_tested`: independent host contracts passed;
- `one_gpu_correctness_tested`: an actual CUDA device executed a bounded parity test;
- `physical_multi_gpu_tested`: reserved for later G5-integrated hardware evidence.

Logical-rank mocks test ownership and ordering only. They must never emit the final label.

## Dataflow

1. Deterministic route/arc IDs and frozen seeds enter the scheduler.
2. Lambert requests use fixed per-input result regions. Short/long, zero-revolution,
   positive-revolution branch, no-solution, unsupported, invalid, cancelled and numerical
   statuses remain explicit.
3. Coarse convex jobs are grouped by topology fingerprint, fidelity, node count and
   scenario count.
4. `PersistentArcCallbackBackend` owns the public G3 workspace/SCvx driver lifecycle.
   The G7 scheduler never reconstructs topology and never inspects solver internals.
5. Stable top-K uses `(feasible rank, cost, lower bound, deterministic ID)`.
   Failures remain in the result stream unless a caller explicitly filters them.
6. Warm tokens are carried in arc requests/results between coarse and refined stages.
   Existing `LowThrustWarmStartStore` remains the compatibility authority.
7. Adaptive fidelity changes requested fidelity/node count without changing deterministic
   parent identity. Existing dynamic-discretisation code remains the route-time refinement
   authority.
8. Promising parents expand in parent/scenario order. Expected, worst-case and CVaR
   aggregation validates probabilities, lower bounds and non-anticipative control prefixes.
9. Existing route-master/column-generation code consumes arc costs and lower bounds.
   Its incumbent, restricted-master lower bound, reduced-cost closure and failure semantics
   remain authoritative.
10. Checkpoints store schema version, seed, completed batches, incumbent/bound, sorted arc
    IDs and warm tokens using deterministic encoding and atomic replacement in Python.
11. Independent certification receives an optimizer incumbent but does not trust optimizer
    status. Dynamics, path, terminal, uncertainty and integration checks must all pass.

## Bounded scheduling and ownership

`BoundedArcScheduler`/`BoundedScheduler` enforce:

- maximum submitted arcs (backpressure);
- maximum batch size;
- bytes-per-arc budget;
- maximum workspace bytes;
- stable topology grouping;
- route × arc × scenario × time-node metadata;
- cancellation and backend-exception conversion to retained result records;
- per-group and per-rank/device telemetry.

`SingleDeviceOwnership` is the production ownership policy on this branch.
`LogicalRankOwnership` is a deterministic test mock. G5 will provide an ownership policy
and backend whose rank-local driver/communicator lifecycle is persistent.

## APIs and targets

Native host:

- `spacepdhcg_lambert_family_batch_cpu`: independent fixed-layout CPU parity ABI;
- `spacepdhcg/orbitweaver/g7_orchestration.hpp`: C++ scheduler, backend seam, top-K,
  risk, checkpoint and certification contracts;
- `spacepdhcg_orbitweaver_g7_smoke`;
- `spacepdhcg_orbitweaver_g7_c_api_smoke`.

CUDA:

- `spacepdhcg/cuda/orbitweaver_gpu_c_api.h`;
- `spacepdhcg_orbitweaver_lambert_workspace_*`;
- `spacepdhcg_orbitweaver_lambert_evaluate_async`;
- `orbitweaver_gpu_test`;
- shared target `spacepdhcg_cuda`.

Python:

- `spacepdhcg.orbitweaver`;
- `BoundedScheduler`, ownership policies, top-K, risk, scenario expansion,
  checkpoints, run/result records and frozen matrix loader;
- `spacepdhcg-orbitweaver-g7 validate-config|validate-matrix`.

Schemas:

- `experiments/schema/orbitweaver_g7_config.schema.json`;
- `experiments/schema/orbitweaver_g7_manifest.schema.json`;
- `experiments/schema/orbitweaver_g7_result.schema.json`.

The frozen `benchmarks/paper2_matrix.json` SHA-256 is
`78c4e33e4aabcd85d63ba3f1e03aa2214b3ab207e680bcaaf347516802b2f6a2`.

## Current evidence

At the branch implementation point:

- native Debug + ASan/UBSan + Werror configured and compiled;
- native Release + Werror configured and compiled;
- CUDA 12.8 Debug and Release configured for `sm_120` and compiled;
- CPU G7/C ABI tests passed;
- Python contract tests passed in the isolated worktree environment;
- actual RTX 5090 Debug and Release Lambert CUDA/CPU parity tests passed.

This one-GPU test covers deterministic fixed-layout Lambert evaluation. It does not qualify
the full coarse/refined/robust route path as one-GPU-correctness-tested.

## Deferred validation

Deferred until prerequisite branches and physical hardware evidence exist:

- G4 matched-quality policy evidence and H5/H6 decisions;
- G5 one-rank equivalence, NCCL ownership, collective correctness and sanitizer evidence;
- physical 2/4/8-GPU runs;
- route × scenario strong/weak scaling;
- throughput, energy, memory-crossover and tractability-frontier measurements;
- complete Paper 2 matrix runs and statistical decisions;
- claims of global route optimality outside exact small-instance truth models.

## Exact integration order

1. Merge the stable G4 implementation/evidence branch into its integration branch.
2. Rebase this G7 branch onto that stable G4 merge commit and rerun native/CUDA matrices.
3. Merge the stable G5 public ownership/backend interfaces (not implementation snapshots).
4. Implement a G5 `ArcBatchBackend` adapter using rank-local persistent G3 drivers and
   communicator ownership.
5. Run one-rank G5 versus monolithic G3/CPU truth, including non-anticipativity and risk.
6. Run CUDA sanitizer/race/lifetime checks.
7. Run physical multi-GPU correctness; only then run registered scaling experiments.
8. Freeze G4-G6 evidence and the Paper 2 matrix before any G7 performance campaign.
9. Promote only independently certified route incumbents into Paper 2 result records.

Merge conflicts should resolve in favour of the public G3/G5 ABI versions from their stable
branches; G7 adapts at `PersistentArcCallbackBackend` and the ownership policy, not by
copying solver or communicator internals.
