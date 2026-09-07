# Retained CUDA graphs for retiming — 8 September 2026

Fixed-order retiming now records its parallel camp/leg kernels and final path
reconstruction in a CUDA graph owned by the retained workspace. A 13-leg price
evaluation submits one graph instead of 27 individual kernels. The graph reads
current stage policies, masses, price, thrust, exhaust velocity and sweep masks
from device buffers. None of those numerical values is frozen in the graph.

Workspace replacement destroys the old graph; ordinary/graph toggles retain it.
The C API supplies a matched ordinary-launch mode and read-only graph counters.
It preserves all prior evaluation entry points. The numerical operations,
strict tie ordering and independent physics tolerances are unchanged.

## Measured effect

The matched release comparison alternates the previously published v234 library
and Python wrapper with the new graph implementation. Full retiming uses eight
pairs with fresh workspaces, excluding the first pair from the median. Cached
pricing uses 42 alternating pairs across six prices, excluding the first pair.
Every pair chooses exactly the same schedule and objective.

| Workload | GPU | Published v234 | Graph replay | Time reduction |
|---|---|---:|---:|---:|
| One cached price evaluation | H100 | 0.78773 ms | 0.76193 ms | 3.3% |
| One cached price evaluation | RTX 5090 | 1.26604 ms | 1.21600 ms | 4.0% |
| Full return-sweep retiming | H100 | 63.693 ms | 63.618 ms | 0.1% |
| Full return-sweep retiming | RTX 5090 | 223.972 ms | 222.862 ms | 0.5% |

The complete-workload differences are too small to claim a material overall
speedup from this experiment. Graph replay makes repeated price evaluation
modestly cheaper and removes per-stage CPU submissions. It does not address
transfer-table construction or move the price/mass driver onto the GPU.

A second comparison within the new binary toggles graphs off/on and reads native
counters. It verifies one graph build and 42 graph launches across the cached
alternating experiment. Its measurements agree in direction with the release
comparison; the raw files preserve both experiments.

## Where time remains

A separate 100-call boundary measurement warms an unswept, cached retimer and
times both `_dp` and its native C call. Medians exclude the first five calls:

| Boundary | H100 | RTX 5090 |
|---|---:|---:|
| Complete Python `_dp` call | 0.76422 ms | 1.40086 ms |
| Native call, including GPU/copies/synchronization | 0.62072 ms | 1.29055 ms |
| Time outside the native call | 0.14307 ms | 0.10733 ms |

The native boundary accounts for roughly 81% of cached H100 time and 92% locally.
This is not a GPU kernel profile: copies and stream synchronization are included.
These are unswept, instrumented observations, so they should not be substituted
for the paired sweep benchmark above.

The native evaluator still downloads the result, arrivals, departures, delta-v,
sweep inflation and validity through six separate `cudaMemcpyAsync` calls to
host buffers. Consolidating that output is the next concrete experiment, followed
by moving forward mass bookkeeping and price/mass decisions onto the device.
Python preparation exists, but the boundary evidence does not support treating
it as the dominant cached-DP cost.

## Validation and repeat mission

- **91 tests pass on RTX 5090 and H100.** The three new graph cases cover changed
  prices, masses, thrust/exhaust, weights, calibration, model selection, pinned
  visits, ordinary/graph toggles, infeasible tables and workspace replacement.
  CPU-reference tests and return-sweep mutation tests also exercise graph mode.
- H100 **memcheck and initcheck each pass 28 cases with zero errors**.
- The same 13-leg mission is refined again on H100 in **9.873 s**. Both checkers
  pass: official **526.489 kg**, independent **526.4887063656 kg**, six mined
  asteroids and no violations. Maximum replay discrepancies are 0.474767 km
  position, 8.089e-8 km/s velocity and 6.67e-11 kg mass. This is a validation repeat,
  not a refinement speedup measurement or a new score improvement.
- The first remote configuration attempt failed because the copied source lacked
  the Git snapshot required by CMake. The failure log is retained. The corrected
  build, tests and mission all completed successfully.

The 23-ship fleet remains **12,805.194 weighted kg**. The existing viewer retains
the previously validated v235 mission; the v239 replay is separately archived,
and its timing is not substituted into that older viewer dataset.

## Reproduction and provenance

Code commit: `a9fb798e` (based on published `f1e8f2f7`).
The [downloaded evidence](../results/lambda/2026-09-08/gpu-retime-graphs-v238/)
contains source/library hashes, both benchmark scripts, the boundary measurement,
local/H100 test logs, complete remote runners and the independently verified
mission in `h100/mission-v239/output/Result.txt`.

For a same-binary comparison, set the usual pinned catalogue, `PYTHONPATH=src`
and rebuilt `SPACEPDHCG_GTOC12_CUDA_LIBRARY`, then run:

```sh
python results/lambda/2026-09-08/gpu-retime-graphs-v238/benchmark_retime_graph_v236.py \
  build/performance/graph-replay \
  results/lambda/2026-09-08/gpu-sweeps-v234/input-refinements.json \
  results/lambda/2026-09-08/gpu-sweeps-v234/input-return.json
```

The release-comparison script additionally takes the retained old Python wrapper
and old/new library paths. Its archived command is in the H100 mission launcher.
GPU tests require `SPACEPDHCG_GTOC12_GPU_TESTS=1` and exclusive device access.
