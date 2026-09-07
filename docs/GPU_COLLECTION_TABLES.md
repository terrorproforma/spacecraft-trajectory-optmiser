# CUDA ephemerides for collection tables

`CollectPairTable` now uses the existing CUDA orbital-element screening API to
build asteroid-pair and Earth-return costs. The CPU ephemeris arrays previously
constructed for every table are eliminated from this CUDA path. The numerical
CUDA library is unchanged; this connects an already verified device operator to
the collection search that was still preparing its inputs on the host.

The original float32 table representation, LRU cache, Earth arrival allowance and
arrival-window gates remain in force. The native path evaluates a rectangular
departure/TOF grid, then masks cells outside the collection window. Its branch
counter therefore includes those evaluated but masked cells. These extra requests
must be reported rather than silently counted as the smaller CPU input set.

This is still not full device residency. Python builds small request grids and
keeps the table cache; final costs return to host, are masked/packed and uploaded
to the collection DP. Removing that round trip, the host mass-pass controller,
and remaining beam/fleet orchestration are subsequent work. Independent mission
verification continues to use the original physics gates.

## Correctness

Both RTX 5090 and H100 pass 71 tests covering collection tables, existing orbital
element screening, native collection DP and the CPU reference. Six table cases
cover different body pairs, Earth return, incomplete batches, unsorted/duplicate
TOFs, the mission-end margin, empty valid windows and LRU cache reuse. They
forbid calls to host ephemeris functions in the new path and compare against the
old host-ephemeris/CUDA-Lambert calculation. Finite masks must match exactly;
float32 costs are compared at rtol/atol 2e-7 to allow final storage rounding.

All eight captured real tours match a separately constructed host-ephemeris table
and uncached CPU DP. The replay utility now constructs that independent table
instead of reusing the candidate's table. Existing route, timing and 1e-6 kg
objective comparisons are unchanged. The nine-asteroid fixture remains infeasible.
Six table cases pass Compute Sanitizer memcheck on both GPUs with zero errors.

## Cold table construction

Six alternating CPU/CUDA pairs build fresh caches; the first pair is discarded.
Both modes use CUDA Lambert screening, with only ephemeris preparation changed.
These are cold tables with a warm CUDA runtime, not complete certified missions.

| GPU / asteroids | CPU ephemerides ms | CUDA ephemerides ms |
|---|---:|---:|
| RTX 5090 / 2 | 40.762 | 34.293 |
| RTX 5090 / 4 | 148.522 | 124.514 |
| RTX 5090 / 8 | 539.340 | 453.982 |
| H100 / 2 | 21.195 | 10.036 |
| H100 / 4 | 78.432 | 36.737 |
| H100 / 8 | 294.617 | 135.447 |

The eight-asteroid build takes 15.8% less time locally and 54.0% less time on H100.
Different CPU hosts contribute to the different relative gains.

## Matched complete campaigns

All four H100 runs use identical CUDA/QOCO binaries and mission parameters. CPU
comparison runs disable only the collection-table ephemeris hook; collection DP
and Lambert screening still run on CUDA in both configurations.

| Run | Table ephemerides | CLI seconds | Search seconds | Verified kg |
|---|---|---:|---:|---:|
| v291 | CUDA | 60.967 | 19.990 | 548.255 |
| v292 | CPU | 63.085 | 22.951 | 548.255 |
| v293 | CUDA | 59.851 | 20.050 | 548.255 |
| v294 | CPU | 62.776 | 22.703 | 548.255 |

With two runs per mode, median CLI time falls **62.930 → 60.409 s (4.0% less)**;
search falls **22.827 → 20.020 s (12.3% less)**. This is a modest sample of one
campaign configuration, not a universal speedup. All four runs pass the official
and independent checkers and produce 106 initial routes. CUDA table runs record
45,188,558 screening branches versus 44,733,902 with host ephemerides, because the
rectangular grid includes masked cells. Neither count is certified missions.

[Downloaded evidence](../results/lambda/2026-09-08/gpu-collect-tables-v290/README.md)
contains complete outputs, raw logs, source/binary hashes, final test formatting
validation, exact commands and paired benchmarks. The unchanged 548.255 kg mission
is a performance replay; the fleet incumbent remains 12,805.194 weighted kg.

## Reproduction

Build the current CUDA library and configure `PYTHONPATH=src`,
`SPACEPDHCG_GTOC12_DATA` with the pinned catalogue,
`SPACEPDHCG_GTOC12_CUDA_LIBRARY` with the library path, and
`SPACEPDHCG_GTOC12_GPU_TESTS=1`. Serialize GPU workloads while benchmarking.

```sh
python -m pytest tests/test_gtoc12_gpu_collect_tables.py -q
python scripts/gpu/replay_gtoc12_collection_fixtures.py \
  results/lambda/2026-09-08/gpu-collect-profile-v272/v272/timing.json \
  build/collection-table-replay.json
python scripts/gpu/benchmark_gtoc12_collection_tables.py \
  results/lambda/2026-09-08/gpu-collect-profile-v272/v272/timing.json \
  build/collection-table-benchmark.json
```
