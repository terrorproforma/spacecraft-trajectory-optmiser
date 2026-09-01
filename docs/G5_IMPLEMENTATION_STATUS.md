# Gate G5 implementation status

**Status:** implementation/build preparation in progress; **G5 is not accepted and must not be
reported as PASS**.

This branch establishes the scenario-aware distributed core from integrated commit
`a33e950e06b0a302815fb079dc95f356c13db5fd`. It does not contain physical 2/4/8-GPU correctness,
strong/weak scaling, performance, energy, or H2/H3/H4 evidence.

## Isolated development state

- Worktree: `/home/angus/worktrees/spacepdhcg-g5`
- Branch: `feat/scenario-aware-multigpu`
- Base: `a33e950e06b0a302815fb079dc95f356c13db5fd`
- Canonical and other worktrees were not edited.
- G4/P1 corrections produced later must be integrated explicitly after this branch is reviewed.

## Toolchain and pinned upstream

Installed toolkit-only distributed prerequisites:

- OpenMPI runtime/development: `4.1.2-2ubuntu1` (`mpirun` 4.1.2)
- NCCL runtime/development: `2.26.2-1+cuda12.8`
- NVIDIA CUDA apt keyring: `1.1-1`
- CUDA toolkit/NVCC: 12.8 / `V12.8.93`
- CMake: 3.31.10 in the worktree-local `.venv-g5`
- Ninja: 1.10.1
- GCC/G++: 11.4.0
- GPU: NVIDIA GeForce RTX 5090
- Windows/WSL GPU driver: 595.97

No `cuda`, `cuda-drivers`, NVIDIA Linux driver, or driver metapackage was installed.

Pinned PDHCG distributed build:

- Repository: `https://github.com/Lhongpei/PDHCG`
- Commit: `167c8b72b4b96d2f94d405b8763e485514192b81`
- Tree: `62b05e6c1bedd385f6c267af3645ae4aae0421b4`
- Build directory: `build/pdhcg-distributed`
- Flags: `RelWithDebInfo`, `PDHCG_COMPILE_DISTRIBUTED=ON`,
  `PDHCG_BUILD_STATIC_LIB=ON`, `PDHCG_BUILD_SHARED_LIB=ON`,
  `PDHCG_BUILD_CLI=ON`, `PDHCG_BUILD_TESTS=ON`,
  `CMAKE_CUDA_ARCHITECTURES=120`, and explicit CUDA 12.8 NVCC
- Result: all 163 compile/link steps completed. `ldd` resolves `libmpi.so.40`,
  `libnccl.so.2`, CUDA 12 libraries, and the expected host runtime libraries.

The upstream 2-rank distributed tests were compiled but not executed on this one-GPU machine.

## Implemented architecture

`cpp/distributed` composes the existing G2 persistent workspace rather than recreating a one-shot
solver:

- `MpiNcclRuntime` duplicates a persistent MPI communicator, derives node-local rank with
  `MPI_COMM_TYPE_SHARED`, maps local rank deterministically to the same-index CUDA device, rejects
  oversubscription/MPS substitution, and creates persistent NCCL, stream, and event resources.
- `DistributedWorkspace` creates one persistent G2 workspace for each whole scenario owned by the
  rank. It forwards values-only updates, primal/primal-dual/full-state warm starts, scaling refresh,
  asynchronous solve/residual calls, diagnostics, cancellation, and rank-local checkpoint/restart.
- `PartitionPlan` uses stable largest-processing-time placement with a deterministic weighted cost
  for Q/A/F nonzeros, cone type/slots, time nodes, replay, risk, and coefficient-update work. The
  generic comparison uses Q/A/F nonzeros only. Both expose predicted loads; measured compute,
  exposed communication, overlap, and work-unit fields are separate.
- `ArrowheadMetadata` freezes shared primal indices, non-anticipativity rows, and worst/CVaR
  threshold/excess ownership. Only these shared quantities and compact global statistics are
  eligible for collectives.
- Local deterministic CSC forward/transpose kernels cover Q/A/F algebra. Local full cone projection
  remains owned by each G2 solver; the G5 library also provides a direct SOC block projection used
  by its independent correctness test.
- Expected risk uses weighted sums, worst risk uses maxima with deterministic host-oracle tie
  handling, and CVaR uses optimization threshold/excess variables plus epigraph residual and
  threshold-dual partials. CVaR is not replaced by host postprocessing.
- NCCL reductions use device pointers directly. MPI is limited to communicator/bootstrap and compact
  status control; there is no implicit host gather in a claimed device reduction.

## Collective contract

Every collective records kind, call count, element count, payload bytes, estimated wire bytes,
frequency, mathematical purpose, collective duration, exposed duration, and overlapped duration.
The defined collective classes are:

- shared-arrowhead sum for non-anticipativity shared primal/gradient terms;
- residual sum for squared primal, dual, and gap components;
- residual max for cone, non-anticipativity, and risk-epigraph violations;
- expected-risk sum for weighted scenario objective/risk contributions;
- worst-risk max for worst-case epigraph loss/violation;
- CVaR sum for weighted excess and threshold-dual contributions;
- rank status max for healthy/cancelled/failed/rank-lost propagation.

The optional overlap path enforces:

1. compute records local-ready;
2. collective stream waits;
3. NCCL is enqueued;
4. collective-complete is recorded;
5. compute waits;
6. independent scenario-local work may overlap.

The transition state machine rejects out-of-order events. Deterministic mode retains a fixed
partition, mapping, collective sequence, float64 payload, and one-stream default.

## Checkpoint and failure semantics

Each rank checkpoint includes schema magic/version, global topology and partition fingerprints,
world size, rank, device, local scenario count, primal/dual/scaling sizes, warm ownership, and an
indexed opaque G2 checkpoint per local scenario. Restore rejects topology mutation, repartitioning,
world-size/rank/device changes, truncation, and local scenario mismatch.

Cancellation is forwarded to every local G2 workspace and aborts the persistent NCCL communicator.
MPI/NCCL errors move the rank to a terminal failed state. Ordinary OpenMPI 4.1.2 is not ULFM:
physical rank loss is detected as communicator failure and is not recoverable in place. Restart
requires a compatible complete rank set and checkpoint.

## Build and static/logical validation completed

Completed without weakening warnings:

- pinned upstream RelWithDebInfo distributed build: 163/163 compile/link steps;
- SpacePDHCG Debug `sm_120`, warnings-as-errors: distributed library and both G5 tests compiled;
- SpacePDHCG Release `sm_120`, warnings-as-errors: distributed library and both G5 tests compiled;
- sanitizer-capable Debug build: host logical test linked with ASan/UBSan and passed with leak
  detection enabled;
- logical 1/2/4/8-rank contracts passed in Debug, Release, and sanitized builds;
- G5 JSON Schema Draft 2020-12 validation and adversarial claim/telemetry tests: 6 passed;
- Ruff check for the G5 schema tests passed;
- idle-GPU final-HEAD Release one-rank MPI/NCCL/CUDA CTest passed, including status telemetry and
  terminal cancellation assertions;
- Debug one-rank Compute Sanitizer memcheck reported zero errors and zero leaked bytes;
- Debug one-rank Compute Sanitizer racecheck reported zero hazards;
- Debug one-rank initcheck without third-party unused-pool reporting and synccheck each reported
  zero errors.

Logical-rank coverage includes deterministic ownership/load, weighted-versus-nonzero partitioning,
shared-arrowhead algebra, non-anticipativity, expected/worst/CVaR reductions, residual reductions,
collective ordering, status/failure/cancellation propagation, rank-loss/world-size checkpoint
incompatibility, topology mutation, warm ownership, and required telemetry fields.

## One-GPU verification boundary

`g5_one_rank_runtime_test` is compiled in Debug, Release, and sanitizer-capable build trees. An
idle-GPU final-HEAD Release run passed on the RTX 5090 with exactly one MPI rank and emitted a
schema-shaped record with `world_size=1`, `device=0`, deterministic one-stream mode, six device
collective classes plus compact MPI status telemetry, zero estimated wire bytes, and
`multi_gpu_scaling_verified=false`. The target
exercises one real MPI rank, one NCCL communicator, CUDA Q/A/F forward/transpose products, SOC
projection, device reductions, stream-event ordering, a real persistent G2 scenario solve,
full-state checkpoint/restore/warm start, scaling refresh, residual calculation, and the frozen G5
implementation-evidence schema.

Re-run it only while the GPU has no other compute process:

```bash
CMAKE_BIN=.venv-g5/bin/cmake \
BUILD_TYPE=Release \
BUILD_DIR=build/g5-release \
RUN_ONE_RANK_GPU=1 \
bash scripts/gpu/run_g5_build_matrix.sh
```

The script checks `nvidia-smi --query-compute-apps=pid` and defers rather than contending with G4.
Memcheck, racecheck, initcheck, and synccheck completed cleanly during idle intervals. Initcheck with
`--track-unused-memory` is not clean: it reports NCCL 2.26.2's own 2 MiB internal communication
pools as mostly unused, with all allocation backtraces inside `libnccl.so.2`; no uninitialized
SpacePDHCG access was reported. The actual two-stream overlap execution remains deferred to physical
multi-GPU validation.

## Exact physical validation still required

Physical 2/4/8-GPU work remains fully deferred. Before any G5 acceptance statement:

1. Use one node with 2, 4, and 8 identical GPUs and a one-rank-per-GPU mapping verified against UUID,
   PCIe/NUMA, and `CUDA_VISIBLE_DEVICES`.
2. Re-run monolithic CPU/single-GPU truth versus distributed expected, worst, and CVaR CQPs,
   including optimization epigraph primal/duals, canonical residuals, nonlinear replay, and
   non-anticipativity.
3. Exercise default one-stream and optional overlap paths under Compute Sanitizer racecheck,
   memcheck, initcheck, and synccheck, then inspect NCCL event ordering in Nsight Systems.
4. Validate cancellation, injected rank failure/communicator failure, incompatible checkpoints,
   topology mutation, restart with an identical rank set, and explicit unrecoverable rank-loss
   behavior.
5. Compare scenario-aware and nonzero-balanced partitions on the identical global CQP/topology,
   recording predicted and measured rank load.
6. Run strong scaling at fixed `(N,S)` and weak scaling at fixed per-GPU scenario/nonzero work.
   Record same-machine one-GPU baselines, local compute, exposed/overlapped communication,
   payload/count/bytes/frequency/purpose, imbalance, memory, failures, and nonlinear quality.
7. Resolve or honestly censor H2/H3/H4 under the frozen decision rules.

Only those physical results can complete Gate G5. This implementation document is not acceptance or
scaling evidence.

## Integration guidance

Keep this branch isolated until G4/P1 corrections are stable. Integrate later fixes by cherry-picking
their reviewed commits onto `feat/scenario-aware-multigpu`, resolve only semantic conflicts in shared
G2/G3 interfaces, then rerun Debug/Release/logical/schema and idle-GPU one-rank correctness. Do not
cherry-pick this branch into a dirty canonical worktree.
