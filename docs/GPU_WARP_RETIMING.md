# Parallel flight-time selection — 8 September 2026

The H100 trace identified the serial flight-time loop as 83.7% of retiming kernel
time. The leg kernel now assigns one full CUDA warp to each arrival epoch.
Its 32 lanes evaluate separate flight-time candidates, then reduce the maximum
objective and smallest winning candidate index. It performs no reordered sum.

This preserves the reference's strict comparison and earliest-TOF tie winner,
including ties across lanes and across multiple 32-candidate tiles. A parallel
prefix cutoff preserves the old early stop at the first shift outside the epoch
grid, even for unsorted caller input. NaN candidates never enter the reduction.
Each warp writes one arrival value and predecessor without atomics. Kernel
arithmetic, trajectory models and final physics gates remain unchanged.

## Matched measurements

The published single-download runtime (`89922bdd`) and new runtime use the same
Python wrapper and recorded return measurement. The full benchmark alternates
eight pairs with fresh retiming workspaces, excluding the first pair from the
median. Cached pricing alternates 42 pairs across six prices, excluding the first.
Every pair selects exactly the same schedule and objective.

| Workload | GPU | Published | Warp selection | Speedup |
|---|---|---:|---:|---:|
| Cached price evaluation | H100 | 0.72567 ms | 0.33182 ms | **2.19×** |
| Cached price evaluation | RTX 5090 | 1.08502 ms | 0.37972 ms | **2.86×** |
| Full return-sweep retiming | H100 | 63.397 ms | 60.943 ms | **1.040×** |
| Full return-sweep retiming | RTX 5090 | 224.537 ms | 219.460 ms | **1.023×** |

Full retiming still builds the 412,116-branch transfer fixture and evaluates six
prices. Its table construction cost limits the complete-workload speedup. These
are warm-runtime measurements, not complete CLI startup or mission-refinement
speedups.

A separate unswept cached-DP experiment varies the timing-grid resolution. It
uses 25 alternating pairs at each resolution, discarding the first pair. Exact
schedule/objective parity holds for all prices and grids:

| Grid spacing | H100 published → new | H100 speedup | RTX 5090 published → new | RTX speedup |
|---|---:|---:|---:|---:|
| 30 days | 0.45904 → 0.26802 ms | 1.71× | 0.66305 → 0.30621 ms | 2.17× |
| 15 days | 0.72923 → 0.31092 ms | 2.35× | 1.14336 → 0.37560 ms | 3.04× |
| 5 days | 2.36207 → 0.51735 ms | **4.57×** | 3.07405 → 0.69927 ms | **4.40×** |

The finer-grid results measure scheduling arithmetic; they are not newly
certified fleet trajectories or score improvements.

## H100 trace

The repeat trace captures 100 warm, unswept price evaluations after graph and
table warm-up. Compared with the prior archived trace:

| Kernel | Previous mean | New mean | New share of kernel time |
|---|---:|---:|---:|
| Leg selection | 36.740 µs | **4.182 µs** | 36.0% |
| Camp selection | 4.280 µs | 4.350 µs | 37.5% |
| Final selection/backtracking | 37.259 µs | 39.958 µs | 26.5% |

Leg selection is about **8.8× faster** in this trace. Total kernel time over the
100 calls falls from 57.05 to 15.09 ms, about 3.8×. Other kernels are unchanged;
their small timing differences are separate trace observations.

The uninstrumented boundary measurement now reports a median 0.31381 ms for the
complete cached `_dp` call: 0.16642 ms inside the native call and 0.14732 ms outside
it. Python preparation is consequently a substantial remaining cost, alongside
camp selection and final selection/backtracking. Moving the price/mass driver
and forward bookkeeping to the device remains necessary for full GPU execution.

## Validation and mission replay

- **99 tests pass on RTX 5090 and H100.** Six new cases cover cross-lane ties,
  multiple flight-time tiles, a partial final block, NaNs, and the early cutoff
  in an unsorted input array, with graph and ordinary execution.
- H100 memcheck, initcheck and synccheck each pass **36 cases with zero errors**.
- The same 13-leg mission is refined again in **9.439 s**. The official checker
  accepts **526.489 kg**, and independent replay reports **526.4887063653 kg**,
  six mined asteroids and no violations. Maximum replay discrepancies are
  0.856497 km position, 8.079e-8 km/s velocity and 8.83e-11 kg mass, under the
  unchanged acceptance rules. This is a validation repeat, not a refinement
  speedup measurement.

The 23-ship fleet stays at **12,805.194 weighted kg**. The existing visualiser
retains its earlier v235 mission samples and timing; the v247 replay is archived
separately.

## Evidence and reproduction

Code: `9ec9afbd`, based on `89922bdd`.
[Downloaded evidence](../results/lambda/2026-09-08/gpu-warp-retime-v246/) contains
the paired benchmarks, grid-scaling experiment, trace, source/library hashes,
tests, sanitizers and full independently verified mission.

For grid scaling, set `PYTHONPATH=src` and the pinned catalogue path, then run:

```sh
python results/lambda/2026-09-08/gpu-warp-retime-v246/benchmark_warp_scaling_v245.py \
  build/performance/warp-scaling.json \
  results/lambda/2026-09-08/gpu-sweeps-v234/input-refinements.json \
  OLD_LIBRARY NEW_LIBRARY
```

The archived launchers include exact full-benchmark, build, validation, trace and
refinement commands. The Python wrapper is unchanged; the native source and new
test are the only code changes in this tranche.
