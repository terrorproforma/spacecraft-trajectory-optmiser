# Performance architecture review — 5 September 2026

Reviewed main at `2aecc6594723a263220bfe8537cbce81427705a0`. This is a source review and analysis of committed evidence, not a newly executed GPU benchmark. Recommendations below are proposed work, not implemented speedups. The review is preserved separately from the frozen campaigns.

The core abstractions are useful: fixed symbolic structure, persistent numerical state, a stable C ABI, analytic variational dynamics, explicit solver selection, and independent trajectory verification. The most urgent performance work is inside the implementation of those abstractions.

The central finding is that **device residency has been implemented much more thoroughly than device parallelism**. The persistent first-order solve uses one CUDA block; multiple full-problem operations use one thread. A second, independent issue is poor convergence on difficult subproblems. The mission-search refinement path also has its own CPU assembly/solver lifecycle, so accelerating the CUDA planner alone will not accelerate all GTOC12 work.

## What exists today

| Path | Observed implementation | Performance implication |
|---|---|---|
| Planner | Python JSON interface launches the native executable; default backend is pure QOCO. | Persistence exists within an invocation; repeated API calls still incur process and CUDA setup. |
| Persistent first-order CQP | Fixed buffers and topology; custom device iteration and projected-KKT/CGLS recovery. | Lifecycle foundation is valuable; solver execution is restricted to a single block. |
| Device SCvx | Parallel interval linearisation, device coefficient updates/replay, host trust decisions from compact diagnostics. | Some independent stages are still serial; synchronous instrumentation interrupts submission. |
| QOCO | Persistent solver owner with numeric updates and explicit failure recovery. | Adapter repeatedly reconstructs its host representation and re-downloads topology. |
| OrbitWeaver CUDA screening | Batched Lambert requests with a thread per request. | Genuine request-level parallelism already exists and should be reused where formulations match. |
| GTOC12 refinement | Python/NumPy variational integration, sparse assembly, CPU Clarabel inside each leg solve. | Separate production workload; not automatically routed through the persistent CUDA driver. |

Sources: [planner contract](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/docs/PLANNER.md), [native runner](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/src/spacepdhcg/planner/native_runner.py#L111), [Lambert kernel](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/cpp/cuda/src/orbitweaver_gpu.cu#L244), [GTOC12 pipeline](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/src/spacepdhcg/gtoc12/pipeline.py#L1).

The README and original ARCHITECTURE document retain early-stage descriptions. The September 5 integration notes and actual sources are substantially newer. Physical multi-GPU remains deferred under [single-gpu-v1](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/docs/ACTIVE_SINGLE_GPU_ROADMAP.md); it should not become a prerequisite for these improvements.

## Highest-priority findings

### 1. One large solve cannot use multiple SMs

`kThreads=256`; the main launch is `solve_kernel<<<1, kThreads, ...>>>`, and recovery uses the same one-block launch. CUDA schedules one block on a single streaming multiprocessor. This is a structural limitation for a large individual solve, regardless of the installed GPU's total core count.

The iteration uses block-local barriers and loops striding by `blockDim.x`. Merely increasing the grid size is incorrect: it would duplicate work and lose required global ordering. Introduce a separate multi-block implementation with properly ordered kernel phases, or a deliberately cooperative launch with validated residency constraints.

Source: [thread count](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/cpp/cuda/src/persistent_pdhcg.cu#L42), [solve iteration](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/cpp/cuda/src/persistent_pdhcg.cu#L1069), [launches](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/cpp/cuda/src/persistent_pdhcg.cu#L4037). Scheduling semantics: [NVIDIA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html).

Recommended dispatch:

- Small independent CQPs: compare the existing one-block path, a purpose-built batch with one or more blocks per problem, and a CPU solver.
- Large CQPs: multi-block sparse operators, cone projection kernels, and parallel reductions.
- Similar mission requests: group by dimensions, topology, cone inventory and requested quality; maintain per-problem convergence masks and iteration budgets.

This is a performance hypothesis, not a guaranteed multiplicative speedup. Prior concurrent-lane experiments failed; a new batch implementation must establish its own correctness and throughput.

### 2. Parallelise the serial preamble and reductions first

Several bottlenecks occur even when few solver iterations are needed:

- `coefficient_change_kernel` scans all numerical data using one thread.
- `initialise_control_kernel` executes ten Ruiz equilibration passes and spectral estimation using one thread when refresh is needed.
- Cone blocks are projected serially by thread zero inside the first-order loop.
- `evaluate_report` performs large residual/objective/cone scans on thread zero.
- SCvx numeric updates and metrics launch with one thread.

The preamble's own comment records several seconds of serial refresh work at N=2000. Make row/column equilibration, coefficient-change statistics, cone projections, and residual reductions parallel. Preserve cone-compatible scaling and transactional cancellation: partially refreshed scales must never be used.

Sources: [change scan and preamble](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/cpp/cuda/src/persistent_pdhcg.cu#L425), [serial reporting](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/cpp/cuda/src/persistent_pdhcg.cu#L897), [cone projections](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/cpp/cuda/src/persistent_pdhcg.cu#L1162), [metrics and numeric update](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/cpp/cuda/src/device_scvx.cu#L2245).

A useful early experiment is a complete residual check on an existing warm point before expensive scaling refresh. It may terminate only if that point satisfies the current problem and unchanged requested criteria. This is particularly relevant to already-solved fixtures, not a substitute for testing displaced starts.

### 3. Stop rebuilding the QOCO conversion on repeated solves

After the first solve, `native_qoco_solve` calls `convert` again. That function downloads Q/A/F offsets and indices as well as numerical buffers. Each nonempty download immediately synchronises its stream. It then constructs and sorts triplets into new CSC matrices before checking that the pattern is unchanged.

Compile the canonical-to-QOCO row/cone transform and sparse scatter map once. Retain host topology, staging buffers and output structure. On update, copy/transform only numerical values; enqueue independent transfers before one required completion boundary. A device-native bridge can follow if the pinned QOCO API can support it.

Caching must also lock scalar-row classification, finite-bound masks and cone layout. An equality becoming an inequality changes the converted structure even if the input CSC pattern is unchanged.

Preserve the explicit fresh-solver rebuild after failure until QOCO's persistent internal state can be reset safely. This rebuild addresses observed solver-state contamination and is not gratuitous allocation.

Sources: [synchronous download](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/cpp/cuda/src/native_qoco_adapter.cpp#L135), [conversion](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/cpp/cuda/src/native_qoco_adapter.cpp#L390), [repeated conversion and failure rebuild](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/cpp/cuda/src/native_qoco_adapter.cpp#L977).

### 4. Reduce iteration count as well as iteration cost

The historical identical-CQP diagnostic shows persistent residuals changing from roughly 5.90e-4 at 100,000 iterations to 5.76e-4 at 1,000,000. More iterations were not an effective remedy. The September 5 displaced-start regression still records fixed-tight timeouts with no accepted steps for 3-DoF, 6-DoF and low thrust, while pure IPM has qualified positive regressions.

Sources: [historical convergence ablations](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/docs/G3_GATE_REPORT.md), [current displaced regression](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/results/gpu/current-head-8cb3759-rtx5090/g3/displaced/summary.json).

Recommended experiments:

1. Explicit physical nondimensionalisation, with primal/dual mappings and final residuals checked in the original problem coordinates.
2. Trajectory-block preconditioning, preserving the conic metric; arbitrary component scaling inside an SOC is not interchangeable with Euclidean cone projection.
3. Residual progress per unit wall time to trigger recovery or an explicit IPM handoff.
4. Reuse compatible primal-dual states and scaling across accepted SCvx steps; invalidate incompatible state on topology or model changes.
5. A backend selection table based on family, size, conditioning, warm-start availability and accuracy.

The current code already has Ruiz scaling, warm starts and adaptive policies. The opportunity is improving them and selecting the right method, not reintroducing them under new names. Recovery is currently enabled by an iteration-limit threshold of 350,000, with a 300,000 PDHG prefix; a progress-based policy should be evaluated as a new versioned policy.

The frozen G4 policies and evidence remain intact. A candidate performance policy must use a separate experiment identity and equal final qualification.

### 5. Exploit the trajectory operator instead of scattering generic CSC entries

The custom forward CSC multiply issues an atomic addition for each nonzero. cuSPARSE descriptors are created, but the main solve uses these custom device routines.

For the dynamics constraint
`r[k] = x[k+1] - A[k] x[k] - B[k] u[k] - c[k]`,
assign work by interval/state row and write each output directly. Implement the transpose as a gather from neighbouring interval contributions. Handle boundary, cone, risk and shared-control rows explicitly. The diagonal or small-block Q operator should have a specialised path.

Start with a generic multi-block CSR/CSC baseline, precomputed value permutations and parallel reductions. Compare a structure-specific operator against that baseline before committing to a full matrix-free backend. Keep explicit CSC export for reference solvers and audits; test adjoint consistency and full operator equivalence.

Source: [atomic forward multiply](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/cpp/cuda/src/persistent_pdhcg.cu#L382). [cuSPARSE documentation](https://docs.nvidia.com/cuda/cusparse/index.html) provides a library baseline, not a promise that generic SpMV wins for small trajectory blocks.

### 6. Separate useful diagnostics from submission stalls

SCvx `time_stop` waits for each timer event. Numerical fingerprints hash up to nine buffers, copy to the CPU and synchronise. Replay and metric collection introduce additional completion boundaries.

Record distinct events and read elapsed times after the enclosing work completes. Keep fingerprints on device until the necessary report boundary where practical. Preserve full evidence in audit mode; measure production telemetry with explicitly reported settings. Trust decisions still require either a deliberate host boundary or a device implementation.

CUDA Graphs become useful after the iteration is split into multiple short kernels. Capturing today's long one-block solve does not solve its lack of parallelism. Use bounded graph chunks with device convergence/cancellation state; do not run an uninterruptible graph merely to reduce launch count. [NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

Source: [event and fingerprint synchronisation](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/cpp/cuda/src/device_scvx.cu#L2192).

Nonlinear forward replay has a true dependency across time. Parallelise across trajectories/scenarios and independent multiple-shooting intervals; retain an independent full forward replay for qualification. An affine HCW rollout can use a scan, but that does not justify applying the same operation to a nonlinear rollout without a new algorithm.

### 7. Make GTOC12 refinement persistent on its own terms

`solve_leg` builds its sparse subproblem and invokes `clarabel.DefaultSolver` inside every SCvx iteration. Its variational dynamics are already analytically differentiated and vectorised over intervals; replacing nonexistent finite differences would miss the actual issue.

First freeze the leg transcription's union sparsity pattern, retain explicit numerical zeros, precompute assembly positions and the constant smoothness Hessian, and reuse a Clarabel solver for compatible updates. Benchmark update-compatible settings against cold setup with presolve, because turning off presolve can offset lifecycle savings. [Clarabel update restrictions](https://clarabel.org/stable/user_guide_data_updating/).

Sources: [analytic interval linearisation](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/src/spacepdhcg/gtoc12/low_thrust.py#L240), [fresh Clarabel construction](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/src/spacepdhcg/gtoc12/low_thrust.py#L544), [outer assembly loop](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/src/spacepdhcg/gtoc12/low_thrust.py#L698).

Then connect equivalent batched GPU leg evaluation through the oracle. Preserve GTOC12's cubic control interpolation, mass conventions, free endpoint excess velocities and official verifier semantics: the generic planner is not a drop-in replacement.

Existing search caches and cheap screening should be retained. Cache exact certified requests using endpoints, epochs, mass, model/hold, topology, fidelity, solver policy and tolerance. Nearby requests may supply warm starts, not inherited certificates. Batch screening and refine only competitive candidates; distinguish heuristic ranking from safe bound-based pruning.

The native planner's subprocess interface can become a long-lived worker/API for repeated requests. The G4 transport already demonstrates process persistence, but it does not remove the numerical costs above.

## What the recorded timings actually establish

Committed H1 median SCvx totals, each invocation exercising three repetitions:

| HCW intervals | RTX 5090, source 8cb3759 | H100, source 9e75b47 |
|---:|---:|---:|
| 20 | 0.05537 s | 0.02748 s |
| 100 | 0.26956 s | 0.13026 s |
| 2,000 | 5.97231 s | 3.00843 s |
| 10,000 | 31.00621 s | 15.25801 s |

Sources: [5090 H1](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/results/gpu/current-head-8cb3759-rtx5090/g3/h1/h1_decision.json), [H100 H1](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/results/lambda-h100/v1-reseal-9e75b47-h100/g3/h1/h1_decision.json).

These are residency fixtures with an already-qualified zero trajectory, not a general nonlinear optimisation benchmark or a controlled GPU hardware comparison. Sources and environments differ. The H100 raw N=10000 record reports three inner iterations, zero accepted steps, zero retained change, and approximately 15.258 s total. Its solve timing includes a preamble that is not separately attributed as scaling. This makes serial setup a particularly strong profiling target; it does not establish its exact percentage without a kernel trace.

The H1 compact exporter has concrete semantic defects:

- `inner_iterations` is populated from recovery iterations.
- `accepted_steps` is populated from requested repeats.
- `polish_used` is hard-coded true.
- `allocation_bytes` is relabelled as peak/reserved device memory without establishing it as the total workspace peak.

Consequently compact result-041 reports 0 inner iterations and 3 accepted steps where raw production output reports 3 and 0. Correct the exporter and schema semantics before fitting a cost model. Preserve original artifacts and issue corrected derivatives with explicit provenance.

Sources: [exporter](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/scripts/gpu/run_g3_h1.py#L235), [raw H100 evidence](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/results/lambda-h100/v1-reseal-9e75b47-h100/g3/h1/h1_raw.jsonl), [compact record](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/results/lambda-h100/v1-reseal-9e75b47-h100/g3/h1/compact/result-041.json).

The historical batching report attributes 354.418 s of 356.864 s to CQP work. Its own Amdahl calculation limits eliminating all non-CQP cost to about 1.0069x. It also records failed lane pilots. Treat those as evidence against relying on process parallelism alone, not proof that a one-block kernel saturates all SMs. Ordinary NVML GPU utilisation measures time with any kernel executing, not the fraction of compute capacity used. The failed pilot's actual cause requires profiling. Sources: [batching report](https://github.com/terrorproforma/spacecraft-trajectory-optmiser/blob/2aecc6594723a263220bfe8537cbce81427705a0/docs/G4_BATCHED_EXECUTOR.md), [NVIDIA NVML definition](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html).

## Ordered implementation programme

| Order | Concrete deliverable | Verification and decision |
|---:|---|---|
| 0 | Correct benchmark work/memory semantics; separate preamble, solve, recovery, conversion and replay timing; capture Nsight Systems/Compute on a dedicated supported host. | Reconcile raw and derived counts. Use profiler runs for diagnosis and unprofiled runs for latency. |
| 1 | Parallel change detection, scaling, cone projections and residual reductions; cache QOCO conversion; cache GTOC12 assembly/solver lifecycle. | Compare identical values and independent residuals; prove no repeated symbolic conversion; retain changes only with measured end-to-end benefit. |
| 2 | Generic multi-block solver operators and a trajectory-specialised operator behind the existing ABI. | Ax/A-transpose equivalence, cone parity, cancellation/race checks and qualified solve comparisons on both GPU architectures. |
| 3 | Scaling/preconditioner and progress-based backend experiments. | Same final quality and objective criteria, failure rate included, displaced starts included; no silent backend substitutions. |
| 4 | Purpose-built batched arc solving, request grouping, persistent planner service and bounded graph execution. | Report both single-request latency and qualified arcs/second; check isolated state, deadlines and bounded memory. |
| 5 | Selective mixed precision and advanced structured IPM/partial-condensing experiments. | FP64 residual/replay audit, explicit refinement and recovery, no relaxed final gates; evaluate conditioning and fill-in costs. |
| Later | Multiple GPUs across independent route batches, then coupled scenario decomposition when required. | Separate campaign; measure communication and end-to-end scaling. |

Minimum workload set: nonzero HCW at N=20/100/500/2000/10000, displaced 3-DoF/6-DoF/low-thrust cases at representative sizes, and fixed GTOC12 boundary requests under a fixed mission compute budget. Keep the zero HCW lifecycle test as a diagnostic, not the sole headline.

Report cold startup and warm end-to-end latency separately; median and p95; qualified throughput; rejection/timeout rate; objective and physical residuals; setup/solve/recovery cost; total memory; and candidate counts reaching each fidelity. For mission search, plot verified objective against wall time and report time to a fixed verified objective.

The success metric is **time to a qualified trajectory or verified mission result**. Lower kernel time, higher GPU busy percentage or fewer logged iterations alone are insufficient. There is strong source-level evidence of substantial architectural headroom, but no defensible overall speedup multiplier until these ablations run.
