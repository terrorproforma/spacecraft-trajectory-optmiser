# CUDA collection option pricing and selection

The CUDA screening context now evaluates collection-hop thrust authority,
rocket-equation propellant, ratio-dependent inflation and lost mining yield on
the GPU. It also selects the first feasible Earth-return option in the CPU's
existing sorted order. RouteSearch uses these operators when
`--screening-backend cuda` is selected; the NumPy context remains the reference.

One CUDA thread prices each option. A final device thread folds the costs in
input order using the existing 1e-12 near-tie tolerance and preference for a later
departure. This comparator is not associative: replacing it with an unordered
minimum reduction would change the algorithm. The inexpensive ordered fold is
retained while the divisions, exponentials and feasibility checks run in parallel.
The native device-buffer API supports CUDA Graph capture with retained scratch,
no host copies and no allocations during launch. Its blocking Python bridge
uploads option rows and downloads one 16-byte selection result.

## Measurements

The paired benchmark alternates CPU and CUDA collection selection, with both
modes using the same CUDA Lambert and neighbour operators. It excludes the first
run, then takes the median of three warm runs per mode. It times complete route
searches over the pinned 1,000-asteroid reduced instance, not individual kernels.

| Complete paired search | RTX 5090 | Lambda H100 |
| --- | ---: | ---: |
| CPU collection selection | 3.638 s | 1.825 s |
| CUDA collection selection | 3.575 s | 1.537 s |
| Additional speedup | **1.02×** | **1.19×** |

The local difference is small and is not evidence of a large reliable speedup.
The separate H100 NumPy-versus-CUDA comparison records **12.013 s → 1.536 s
(7.82×)**. Both comparisons retain 27 proxy route candidates and 545,658 Lambert
branches per search. The paired run preserves complete candidate summary
dictionaries exactly. CUDA evaluates **153,799 collection/return option rows per
search**; these are additional selection checks on options already screened by
Lambert, not additional certified trajectories or distinct Lambert solves.

## Correctness and scope

The implementation uses FP64 and disables fused multiply-add for this translation
unit to preserve reference operation order. It keeps the original rocket-equation
`1 - exp(...)` expression and the same physical constants. It does not clip thrust,
change solver tolerances, alter mining rules or replace independent certification.

Tests compare random option sets under flat/ratio-dependent inflation, multiple
masses, penalty scales and time limits. They check stable ties, the exact thrust
authority boundary, empty pools, invalid-input recovery and ABI sizes. Native
tests replay a CUDA Graph with changing queries and verify host/device agreement,
ordering and capacity guards. The targeted local suite passes four tests and the
H100 screening suite passes 30 tests; local/H100 memcheck report zero errors and
H100 racecheck reports zero hazards. Integrated-library evidence is retained
alongside the screening-only H100 library to keep their validation scopes distinct.

Collection option generation, ephemeris packing, collection tour order, dynamic
programming, forward mass replay and fleet selection still contain CPU work. The
GPU prices all options in each submitted list, so invalid options are reported
explicitly even if a first-feasible CPU loop would have stopped earlier. Generated
option lists already exclude failed/nonfinite Lambert solutions. Callers must
serialize each native workspace, including device graph replay and destruction.

The verified fleet score remains **12,805.194 weighted kg**. These measurements
establish search throughput and candidate parity, not a new certified fleet.

## Reproduce

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
export SPACEPDHCG_GTOC12_CUDA_LIBRARY=/absolute/path/to/libspacepdhcg_cuda.so
export SPACEPDHCG_GTOC12_DATA=/absolute/path/to/pinned/gtoc12/data
python scripts/gpu/benchmark_gtoc12_collection.py /new/path/paired.json
python scripts/gpu/benchmark_gtoc12_route_search.py /new/path/routes.json --profile reduced --repeats 2
```

[Evidence](../results/lambda/2026-09-08/gpu-collection-v197) records commands,
source/runtime hashes, every warm timing, candidate summaries and sanitizer
output. Local integrated library: v196. H100 screening library: v197.
The full H100 library was independently rebuilt as v198 and passes **168 GPU
integration tests**, including SCvx, QOCO and the new operators. Its 504 C++/Python
source files match the local source after accounting for line endings. The full
local GTOC12 suite passes **323 tests**. The v198 build commands, runtime hashes,
source manifest and test output are retained under `h100-full-v198/` in the same
evidence directory.
