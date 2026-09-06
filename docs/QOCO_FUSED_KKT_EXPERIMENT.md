# Fused GPU KKT product experiment (v94)

The optional `--fused-kkt-product` preparation flag replaces the sparse portion
of QOCO's KKT product with one kernel across multiple GPU blocks. It computes
P*x + A'*y + G'*z, A*x and G*x in parallel. It uses retained gather maps and
preserves the original accumulation order within each row. Nesterov–Todd cone
products, iterative-refinement decisions, regularization and tolerances are
unchanged. The flag requires gather, queued operators and deferred transposes.
It is not enabled in existing default build commands.

This is an operator improvement with **mixed complete-solve evidence**, not a
general solver promotion. The new runtime remains experimental, and one strict
coast constraint test failed. That failure is retained; its threshold was not
changed and it was not rerun until passing.

## Evidence

- 72 cases compare the new operator bit-for-bit with the original GPU composition
  and against independently accumulated dense arithmetic. Cases include missing P,
  absent and empty constraint blocks, duplicate entries, rectangular matrices,
  changing inputs and 3/257/4,097 variables.
- All four CUDA sanitizer tools pass the operator test, with zero errors, leaks,
  race hazards or race warnings. This is operator coverage, not full-solver
  cancellation or race coverage.
- A 1,000-product CUDA-event sample measured composed/fused times of 74.25/10.29 us
  at n=3, 75.49/9.58 us at n=257, and 71.78/42.26 us at n=4,097. These include
  host enqueue pacing between event boundaries. They are single operator samples,
  not trajectory throughput measurements.
- The first complete GTOC12 comparison qualified all 18 trajectories, but the
  fused QOCO median was 870 ms versus the earlier baseline's 819 ms. Its five
  measured solves used 1,194 total inner iterations versus 845 previously.
- A subsequent preplanned ten-pair comparison alternated fresh baseline/fused
  processes, each with a recorded warm-up and measured leg. All 40 legs qualified
  at the unchanged physics and fixed mass gate (2445.3111007852112 kg, error
  <=1e-5 kg). Measured medians were 959.55 ms baseline and 805.68 ms fused, a 16.0%
  reduction. Timing and iteration counts remain variable; the earlier regression
  prevents a general speedup claim.
- The GTOC12/CLI selection produced 23 passes and one failure: the fixed-endpoint
  ZOH coast had original equality residual 1.1663e-9 against the test's 1e-9 gate.
  Its normalized conic audit passed. Similar sensitivity was already recorded for
  earlier runtimes. Bitwise operator parity does not establish full-solver
  repeatability or excuse this failure.
- Seven native landing repetitions and the 20/500-interval six-DOF planner cases
  pass their existing certificates. These are additional correctness probes, not
  broad convergence or speed evidence. The legacy conversion contract also passes.
- The CPU tooling/backend tests pass 24/24. Their initial two shell syntax failures
  came from CRLF checkout bytes in files declared `eol=lf`; the scripts were
  normalized without changing their shell contents. Normal CUDA CMake configuration
  excludes the new standalone QOCO test, which needs the isolated QOCO headers.

The normalized prepared source differs from frozen QOCO78 only in the fused
header, its CUDA include, the KKT dispatch and provenance. Re-preparing from the
current script reproduces the v94 prepared source. Core93 and QOCO78 remain frozen.

## Profile and next bottleneck

A qualified baseline leg's CUDA API trace recorded 4,175 `cudaMemcpy` calls
(570 ms API time) and 48,423 kernel-launch API calls (251 ms). It also includes
cold initialization. The trace has no GPU kernel or memory-activity data, so
these totals must not be presented as GPU execution times or all as D2H traffic.
The saved failed GPU-summary attempts and direct SQLite checks document that limit.

The remaining linear-system refinement loop still returns norms to the CPU to
decide whether to solve another correction or restore its best vector. A route
toward GPU-native control is conditional graph execution, but this is not yet
implemented or qualified. NVIDIA documents asynchronous cuDSS solve capture with
allocator restrictions, and CUDA 12.8 conditional bodies exclude allocation
nodes. Any implementation must establish compatible persistent storage and preserve
the existing stop/restore/iteration-limit behavior, including invalid numerical
inputs. See [cuDSS graph support](https://docs.nvidia.com/cuda/cudss/general.html)
and [CUDA 12.8 conditional graphs](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#conditional-graph-nodes).

## Reproduction

The frozen experimental library is
`/home/angus/build-qoco-gpu-fused-kkt-v94/final/libqoco.so` inside WSL, used with
`/home/angus/build-spacepdhcg-gtoc12-v93/final/libspacepdhcg_cuda.so`.
The checkpoint embeds preparation/build/test/profile/benchmark helpers and hashes
both runtimes, sources and all results. Restore helpers to their recorded paths
and use a fresh runtime directory (v95 or later); never overwrite the frozen
baseline or experiment. Serialize GPU work with the shared lock.

- [Checkpoint and reproduction helpers](../artifacts/performance/qoco-fused-kkt-v94-checkpoint.json)
- [Alternating complete-leg comparison](../artifacts/performance/qoco-fused-kkt-v94-paired.json)
- [Operator parity](../artifacts/performance/qoco-fused-kkt-v94-operator.json)
- [Operator sanitizer checks](../artifacts/performance/qoco-fused-kkt-v94-sanitizers.json)
- [Unresolved coast failure](../artifacts/performance/gtoc12-qoco-v94-tests.json)
