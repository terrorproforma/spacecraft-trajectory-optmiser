# Parallel retiming completion — 8 September 2026

The H100 device-driver trace attributed 29.8% of kernel time to path completion
and 10.5% to the controller, including its serial best-candidate copy. Completion
now uses four warps to select the final epoch and independent threads to gather
selected-leg values. Threads also share the best-output copy. The dependent
backpointer chain, forward mass recurrence and price/objective decisions retain
their existing ordered arithmetic.

The final-epoch reduction compares value/index pairs, selects the earliest equal
maximum and ignores NaNs exactly as the old strict comparison did. It does not
reorder a floating-point sum. Full-block barriers separate the reduction,
backtracking, gathers, forward accounting and kernel completion. The controller
also completes its copy with a full-block barrier before leaving the graph body.

## Matched performance

Both libraries use the published device price/mass driver and identical Python
code. Twelve alternating pairs per scope discard the first pair and report the
remaining median. Each library retains its own workspace; every selected schedule
and objective matches exactly. The candidate performs one device-driver call per
retiming, as does the published baseline.

| Scope | GPU | Published | Parallel completion | Time reduction |
|---|---|---:|---:|---:|
| Complete retained-table retiming | H100 | 1.58230 ms | **1.39347 ms** | **11.9%** |
| Complete retained-table retiming | RTX 5090 | 1.94648 ms | **1.69117 ms** | **13.1%** |
| Fresh-table retiming | H100 | 58.63413 ms | 58.24274 ms | 0.7% |
| Fresh-table retiming | RTX 5090 | 215.83560 ms | 215.04227 ms | 0.4% |

Fresh-table runs still build 412,116 Lambert branches; there is no material
overall speedup claim for that scope. These warm measurements are not complete
fleet-search or low-thrust mission-certification throughput.

Separate H100 traces capture 20 complete cached retimings, each with six price
evaluations. Both modes use 20 graph submissions and downloads. Mean kernel times:

| Kernel | Published | Parallel completion | Speedup |
|---|---:|---:|---:|
| Final selection, reconstruction and forward accounting | 55.648 µs | 36.764 µs | 1.51× |
| Price/mass controller and best-output copy | 19.682 µs | 5.875 µs | 3.35× |

Camp and leg selection timings remain approximately 4.31 and 4.12 µs per kernel.
They now account for 36.4% and 34.8% of kernel time; completion accounts for 23.9%
and the controller 3.8%. Instrumented trace times are separate from the benchmark
medians above.

## Validation, rejected variant and replay

**137 tests pass on each GPU.** Six new cases exercise earliest equal maxima
across warp boundaries, across 128-thread strides, at the end of a partial tile,
and with no feasible arrival, in ordinary and graph execution. Existing driver,
forward-accounting and objective tests retain their prior checks and tolerances.

The first parallel variant passed the numerical tests, memcheck, initcheck and
synccheck, but racecheck reported cross-kernel shared-memory hazards inside the
conditional graph: 38 reported hazards on H100 and 36 in the focused local run.
That variant was not published as solver code. Adding explicit full-block
completion barriers cleared the reports locally and on H100. The rejected source,
source hashes and logs are retained under `h100/rejected-v260` for inspection;
the passing reports use the final source. No report suppression or weakened
accuracy check is used.

For the final build, H100 memcheck, initcheck, synccheck and racecheck each pass
**74 cases with zero errors or hazards**. The focused local racecheck also passes
20 driver cases with zero hazards.

The new H100 13-leg replay refines in **7.468 s**, and both checkers accept it.
Independent propagation returns **526.4887063656 kg**, six mined asteroids and no
violations; the official checker returns **526.489 kg**. Maximum discrepancies
are 0.474289 km position, 8.082e-8 km/s velocity and 7.708e-11 kg mass under the
unchanged gates. This single replay does not establish a refinement speedup.

The incumbent fleet remains **12,805.194 weighted kg**. The v267 replay is downloaded separately. A subsequent full search and extension
campaign produces a new 548.255 kg, eight-asteroid mission in 89.55 s; its exact
export is displayed as v269 while the earlier datasets retain their provenance.
[Campaign statistics and loading instructions](../results/lambda/2026-09-08/gpu-native-campaign-v269/README.md).
CPU setup, initial route-mass construction and fleet orchestration remain work
for the wider GPU-native pipeline.

## Evidence and reproduction

Code: `58eefa56`, based on `4099dbbd`.
[Downloaded evidence](../results/lambda/2026-09-08/gpu-parallel-finish-v266/)
contains paired benchmarks, local/H100 tests, four sanitizer logs, both Nsight
traces/CSV exports, runtime/source hashes, the rejected specimen and complete
verified mission. The local and remote release binaries are frozen separately.

With `PYTHONPATH=src`, the pinned catalogue and CUDA runtime configured:

```sh
python results/lambda/2026-09-08/gpu-parallel-finish-v266/benchmark_finish_v259.py \
  build/performance/finish-comparison \
  results/lambda/2026-09-08/gpu-parallel-finish-v266/input-refinements.json \
  results/lambda/2026-09-08/gpu-parallel-finish-v266/input-return.json \
  OLD_LIBRARY NEW_LIBRARY
```

Archived launchers record build, validation, trace and mission commands. Their
absolute paths describe the measured environments and need adapting elsewhere.
