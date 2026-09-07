# CUDA collection-tour dynamic programming

Collection-tour Held–Karp search now distributes destination states and epochs
across CUDA blocks. On the matched H100 campaign, this reduces total CLI time
from **90.590 s to 62.526 s** (median of two runs per mode): **1.449× throughput,
30.98% less time**. Search time falls from 50.248 s to 22.745 s. This measures
the complete campaign, including refinement and verification, on one fixed
single-ship search configuration; it is not a universal solver multiplier.

## Implementation and remaining work

`cpp/cuda/src/gtoc12_collect_dp.cu` owns retained tables, policy, DP and backpointer
buffers behind a C API. Kernels process subset cardinalities in dependency order,
including initial uncollected-camp moves. Destination states gather transitions;
pricing uses exact subset mass and the existing thrust-authority equations.
Camping prefix maxima and backtracking execute on device. Each pass returns a
compact 496-byte result. Python does not iterate over DP states or transitions.

The implementation preserves latest-equal camping arrivals, earliest-equal TOFs,
and the CPU predecessor order required by the `candidate > best + 1e-9` rule.
It supports calibrated inflation, phase penalties, banned pairs and certified
return overrides. Ordered epsilon comparisons cannot be replaced with an
unordered maximum without changing decisions. CUDA compilation disables FMA
contraction for this translation unit to retain the reference arithmetic order.

This is not the end of the GPU port. Pair-table ephemeris preparation and packing,
initial mined/subset mass arrays, decisions between mass passes, final Python
result formatting and beam/fleet orchestration still use the host. Both mass
passes use the new GPU operator, but their outer control has not yet moved to
device. The final small terminal selection also has a serial device component.
These are subsequent complete-pipeline targets; no accuracy gate was weakened.

A subsequent [directly timed replay](../results/lambda/2026-09-08/gpu-collect-profile-v288/README.md)
places all 378 native solves plus formatting at 0.173 s, while pair/return table
calls across the campaign take 5.067 s. Table calls occur both inside and outside
collection planning, so inclusive spans must not be added. The 63.938 s profiled
campaign still passes both checkers at 548.255 kg and is excluded from the matched
timing comparison below. This identifies table construction as the next measured
collection target; it is not an attribution of the rest of the campaign.

## Matched full campaigns

All runs use the same H100 binary and mission parameters. The CPU comparison
disables only `GpuLambert.collect_dp_cuda`; other GPU stages remain enabled.

| Run | Collection DP | CLI seconds | Search seconds | Verified kg |
|---|---|---:|---:|---:|
| v280 | CUDA | 62.238 | 22.698 | 548.255 |
| v284 | CPU | 91.663 | 50.300 | 548.255 |
| v285 | CUDA | 62.814 | 22.791 | 548.255 |
| v286 | CPU | 89.516 | 50.195 | 548.255 |

Each produces 106 initial routes, evaluates 2,782,091 collection options and
selects an eight-asteroid mission. CUDA runs record 378 native collection passes
and 44,733,902 screening branches; CPU runs record 44,722,546 branches. Eager
table preparation adds 11,356 requests in the CUDA path. These are stage counters,
not counts of independently certified trajectories, and must not be summed.

Both the official and independent mission checkers pass all four runs with no
independent violations. v285 has maximum position error 0.632368 km, velocity
error 1.106024e-7 km/s and mass error 7.867129e-11 kg. Its unrounded returned mass
is 548.2546201232 kg. The fleet incumbent remains 12,805.194 weighted kg; this
single-ship performance replay is not a replacement fleet or leaderboard gain.

## Warm collection-tour benchmark

These medians exclude the first of six alternating CPU/CUDA pairs and retain
pair tables. Each call includes both mass passes and result construction.

| GPU / asteroids | CPU ms | CUDA ms | Speedup |
|---|---:|---:|---:|
| RTX 5090 / 3 | 1.756 | 1.117 | 1.57× |
| RTX 5090 / 6 | 42.005 | 3.568 | 11.77× |
| RTX 5090 / 9 | 622.175 | 8.261 | 75.31× |
| H100 / 3 | 3.210 | 0.987 | 3.25× |
| H100 / 6 | 74.724 | 2.011 | 37.15× |
| H100 / 9 | 1109.637 | 3.994 | 277.86× |

The nine-asteroid fixture is infeasible. These are proxy-tour evaluations, not
certified missions per second. CPU hosts differ between the two GPU machines.

## Validation and reproduction

Both GPUs pass 69 integrated tests and eight real captured collection inputs
against the uncached CPU reference. The initial 28 native cases pass all four
H100 Compute Sanitizer tools with zero errors/hazards, and RTX 5090 memcheck.
Four additional epsilon-order cases pass locally; the final 32-case native suite
passes on H100 using the unchanged binary. The final test source is archived
separately from the original 69-test snapshot, with its own hash.

[Downloaded evidence](../results/lambda/2026-09-08/gpu-collect-dp-v279/README.md)
contains source snapshots, binary fingerprints, raw test and sanitizer logs,
benchmarks, all four full campaign outputs and exact runner commands. The failed
v281 comparison wrapper is preserved separately: its Python syntax error occurred
before a campaign started and is excluded from performance statistics.

With `PYTHONPATH=src`, `SPACEPDHCG_GTOC12_DATA` pointing to the pinned catalogue,
`SPACEPDHCG_GTOC12_CUDA_LIBRARY` pointing to the built library and
`SPACEPDHCG_GTOC12_GPU_TESTS=1`:

```sh
python -m pytest tests/test_gtoc12_gpu_collect_dp.py -q
python scripts/gpu/replay_gtoc12_collection_fixtures.py \
  results/lambda/2026-09-08/gpu-collect-profile-v272/v272/timing.json \
  build/collection-replay.json
python scripts/gpu/benchmark_gtoc12_collection_dp.py \
  results/lambda/2026-09-08/gpu-collect-profile-v272/v272/timing.json \
  build/collection-benchmark.json
```
