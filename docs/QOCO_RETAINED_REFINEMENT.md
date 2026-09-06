# Retained GPU refinement and iteration accounting

The optional v116/v117 extensions retain the conditional iterative-refinement
graph across numerical updates to a QOCO workspace. v117 also accumulates accepted
refinement iterations on the GPU. The normal numerical path no longer rebuilds
and instantiates the graph or downloads its iteration count after each linear
solve. The host still dispatches IPM steps, performs cuDSS setup/analysis, calls
the initial linear solve, and consumes stopping metadata. This is progress toward
full GPU control, not a completed GPU-native optimiser or a demonstrated speedup.

## Lifetime and ordering

Apply these postprocessors after the existing v115 retained-factorisation chain:

```sh
python scripts/gpu/prepare_qoco_ir_cache.py --destination /path/to/isolated/source
python scripts/gpu/prepare_qoco_ir_counts.py --destination /path/to/isolated/source
```

The native adapter's queued reduction scope is required. A plain, unscoped QOCO
API call attempts a device synchronization inside capture and fails, including
on the v115 baseline. The reuse probe initially exposed this requirement; its
failed outcomes and unscoped source are preserved alongside the corrected probe.

Each fixed-topology cuDSS workspace retains one graph. Changing its workspace,
RHS or solution pointer, or dimension invalidates the cache after stream
completion. Sparse topology, operator buffers, cuDSS configuration and static
regularization remain workspace invariants. Numerical matrix/RHS values stay in
their existing device buffers. A kernel sets the requested tolerance and maximum
refinement iterations in device memory before every replay, so neither is frozen
at capture time. The existing guarded transformation of two private eight-byte
cuDSS range inputs remains version-specific, not a general host-input conversion.

The graph reads the true, unregularized KKT residual and preserves the original
accept/restore and tolerance logic. A final device kernel adds accepted iterations
to the current IPM step count. An event orders the default-stream consumers after
refinement and accounting; the host does not wait there. The IPM step adds its
count to a device total. Initialization refinement is excluded from that total,
matching QOCO's original API. Final output or verbose logging downloads the two
integer counters. Workspace destruction waits before releasing the graph/buffers.

Diagnostic switches (presence enables each switch):

- `SPACEPDHCG_TEST_QOCO_IR_CACHE_DISABLE`: rebuild the graph for every call.
- `SPACEPDHCG_TEST_QOCO_IR_COUNTS_DISABLE`: restore per-call host counting/waits.
- `SPACEPDHCG_TEST_QOCO_IR_CACHE_TRACE`: print build and replay counts at cleanup.
- `SPACEPDHCG_TEST_QOCO_DEVICE_IR_DISABLE`: use the original host refinement path.

Normal build scripts do not select these experimental postprocessors.

## Accuracy and state reuse evidence

Both frozen versions pass the native controller executable, 51 integration tests,
seven convergence/failure-accounting tests and independently certified PD6 N20
and N500 trajectories at the unchanged 1e-8 reference-objective gate. v117 also
passes all 51 tests with the original stopping/combined-RHS arithmetic audits.

The independent two-variable QP probe changes coefficients, RHS, refinement
tolerance and iteration budget across 16 solves on one workspace. It checks a
closed-form optimum and includes a one-IPM-iteration exit followed by another
solve. All six configurations (v115, v116, v117, host-counting and uncached v117
ablations, and verbose v117) have identical statuses, IPM counts, accumulated
refinement counts and final-step counts. Ten solves exercise nonzero refinement;
zero-budget and loose-tolerance cases return zero counts. Each retained variant
builds one graph for 168 replays; the uncached variant builds 168 graphs.

The original unscoped probe failed before producing numerical results on all
six configurations. Its compile first needed QOCO's QDLDL include directory.
These were probe setup failures, not successful tests.

Full v117 memcheck of the scoped probe still fails at graph launch with CUDA
unknown error, reporting 104 errors and 102 outstanding allocations after abort.
This is consistent with the unresolved conditional-graph instrumentation failure
already reproduced by earlier vendor-free probes. It does **not** prove the
full path is sanitizer-clean. See the raw report; no host-IR ablation is offered
as proof of conditional-path safety.

## Complete-transfer timings

Each experiment uses six balanced triples: two runs per process (warmup plus
measurement), rotating first position evenly. All 36 legs in each experiment
pass independent physics checks and the same 1e-5 kg final-mass gate against
2445.3111007852112 kg. The same v107 GPU-seed/GPU-SCvx core is used throughout.
The local RTX 5090 also drives the display; clocks are not locked.

| Experiment | QOCO variant | Median measured complete attempt |
|---|---|---:|
| v116 | v115 baseline | 563.284 ms |
| v116 | v116 without cache | 607.868 ms |
| v116 | v116 retained graph | 637.236 ms |
| v117 | v116 retained graph | 499.047 ms |
| v117 | v117 with host counting | 510.574 ms |
| v117 | v117 with GPU counting | 520.110 ms |

These measurements show no end-to-end speedup. The structural reduction in host
work is useful for enclosing the numerical solver in GPU control, but does not
justify default promotion or a performance multiplier. IPM dispatch, remaining
barriers, setup/analysis and full SCvx orchestration still need work. No new
mission-fleet score follows from these single-transfer measurements.

The checkpoint embeds exact sources, generated trees, build/runtime hashes,
helpers, failed runs, objective checks and raw paired measurements:
[v117 checkpoint](../artifacts/performance/qoco-ir-counts-v117-checkpoint.json).
The formatted postprocessors reproduce all eight changed v116/v117 prepared
files exactly. Earlier runtime trees and binaries remain frozen.
