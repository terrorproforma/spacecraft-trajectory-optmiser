# OrbitWeaver G7 GPU-scale architecture

## Scope and evidence rule

G7 composes the existing OrbitWeaver CPU truth models, the public G3 persistent
single-GPU API, and the integrated public G5 ownership/runtime API. This branch contains
implementation and bounded CPU/logical-rank correctness evidence only. It makes no throughput,
energy, scaling, tractability, optimality, or Paper 2 completion claim.

Roadmap scope update (2026-09-02): `single-gpu-v1` makes the complete one-GPU
coarse/refined/scenario/pricing/master/certification/visualisation flow active. Physical 2/4/8-GPU
ownership, throughput, energy, memory-crossover, and tractability-frontier studies remain
`deferred-not-in-scope`; the G5 adapter is preserved for that later campaign.

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
4. `G3PersistentTrajectoryAdapter` owns one bounded public G3 device-SCvx driver per
   topology/fidelity/rank/device key. `G3ArcBinding` updates fixed device numerical buffers
   in place; G7 never reconstructs topology or inspects solver internals.
5. Stable top-K uses `(feasible rank, cost, lower bound, deterministic ID)`.
   Failures remain in the result stream unless a caller explicitly filters them.
6. Opaque warm tokens are carried between coarse and refined stages. Endpoint, model,
   spacecraft and scenario compatibility is checked before the target G3 driver explicitly
   accepts any interval remeshing. Missing, stale and incompatible tokens remain explicit.
7. Adaptive fidelity changes requested fidelity/node count without changing deterministic
   parent identity. Existing dynamic-discretisation code remains the route-time refinement
   authority.
8. Promising parents expand in parent/scenario order. `G5RankLocalOwnershipAdapter` freezes
   deterministic route/arc/scenario ownership; expected, worst-case and CVaR aggregation
   validates probabilities, lower bounds and non-anticipative control prefixes.
9. Route columns are built from returned arc costs/lower bounds, then independently
   certified. Uncertified combinations cannot become route-master incumbents.
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

`SingleDeviceOwnership` remains the one-device policy. `G5RankLocalOwnershipAdapter` is the
production G5 policy and validates frozen route/arc/scenario metadata before dispatch.
`G5RankLocalArcBackend` verifies local MPI rank/device ownership, wraps rank-local persistent
G3 drivers, synchronizes cancellation/failure status and exposes G5 collective telemetry.
`LogicalRankOwnership` and `LogicalCollective` are CPU test fixtures only.

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
- `spacepdhcg/cuda/orbitweaver_g3_adapter.hpp`, compiled against the public G3 C API.

Distributed:

- `spacepdhcg/distributed/orbitweaver_g5_adapter.hpp`;
- deterministic scenario-aware route/arc/scenario partition metadata;
- rank-local G3 delegation, cancellation/failure propagation, checkpoint compatibility
  through public G5 fingerprints, and collective telemetry.

Python:

- `spacepdhcg.orbitweaver`;
- `G3TrajectoryOracleAdapter`, `G5DistributedAdapter`, bounded schedulers, deterministic
  ownership, top-K, risk, scenario expansion, certified route master,
  checkpoints, run/result records and frozen matrix loader;
- `spacepdhcg-orbitweaver-g7 validate-config|validate-matrix`.

Schemas:

- `experiments/schema/orbitweaver_g7_config.schema.json`;
- `experiments/schema/orbitweaver_g7_manifest.schema.json`;
- `experiments/schema/orbitweaver_g7_checkpoint.schema.json`;
- `experiments/schema/orbitweaver_g7_result.schema.json`.

These are generated from `spacepdhcg.orbitweaver.contracts`; see
`docs/ORBITWEAVER_G7_SCHEMA_CONTRACT.md`.

The frozen `benchmarks/paper2_matrix.json` SHA-256 is
`78c4e33e4aabcd85d63ba3f1e03aa2214b3ab207e680bcaaf347516802b2f6a2`.

## Current evidence

At this adapter branch implementation point:

- native Debug + ASan/UBSan + Werror configured and compiled;
- native Release + Werror configured and compiled;
- CUDA 12.8 and G5 Debug/Release targets configured for `sm_120` and compiled;
- CPU native, Python schema, deterministic fixture and logical-rank adapter tests passed;
- the existing CPU low-thrust G3 reference/oracle smoke tests passed.

No GPU executable was run for this adapter branch. In particular, compilation does not
qualify the full coarse/refined/scenario/pricing/master/certification path as
`one_gpu_correctness_tested`.

## Active and deferred validation

Active for `single-gpu-v1`:

- G4 matched-quality policy evidence and H5/H6 decisions;
- one-GPU coarse/refined/scenario/route replay against independent CPU truth;
- G5 one-rank CUDA/NCCL equivalence and device checkpoint restore;
- one-GPU pricing/restricted-master integration, independent certification, and visualisation.

Deferred to the physical multi-GPU backlog:

- physical 2/4/8-GPU runs;
- route × scenario strong/weak scaling;
- throughput, energy, memory-crossover and tractability-frontier measurements;
- distributed Paper 2 matrix runs and statistical scaling decisions;
- claims of global route optimality outside exact small-instance truth models.

## Exact integration order

1. Merge final native QOCO/P1 changes, retaining the public G3 C API used by
   `G3PersistentTrajectoryAdapter`; resolve only binding construction if the final public
   problem descriptor changes.
2. Merge this branch after G3/QOCO so its adapter and strict `bf9d10` record contracts see
   the final public ABI.
3. Rebuild native/CUDA/G5 Debug/Release/Werror matrices and rerun CPU/logical tests.
4. Run one-GPU coarse/refined/scenario/pricing/master/certification correctness against
   independent CPU replay.
5. Run G5 one-rank equivalence, then CUDA sanitizer/race/lifetime checks.
6. Run physical multi-GPU correctness; only then run preregistered scaling experiments.
7. Freeze G4-G6 evidence and the Paper 2 matrix before any G7 performance campaign.
8. Promote only independently certified route incumbents into Paper 2 result records.

Merge conflicts should resolve in favour of the public G3/G5 ABI versions from their stable
branches; G7 adapts at `PersistentArcCallbackBackend` and the ownership policy, not by
copying solver or communicator internals.
