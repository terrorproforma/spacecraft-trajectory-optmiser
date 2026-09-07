# CUDA neighbour selection

`gtoc12 search --screening-backend cuda --workers 1` now moves RouteSearch's
neighbour filtering and ranking onto CUDA as well as its Lambert screening.
The retained catalogue contains orbital elements, eccentricity/inclination
vectors, orbital bases and mean motion. Each query computes the original band
filter, Kepler positions, phasing cost and positional ranking in parallel.
CUB stable radix sorts preserve asteroid-ID tie order; a device prefix scan
compacts the proxy-first union without downloading the full catalogue.

The filter semantics are unchanged: use every asteroid inside the bands when
there are enough, otherwise use the nearest requested number by band distance.
The final list combines the best requested number by phasing cost with the best
half as many by positional cost, removing duplicates. The host still prepends
the existing co-moving priorities. These are screening proxies; low-thrust
refinement and independent certification remain necessary.

## Measured performance

The paired route benchmark alternates CPU/GPU neighbour selection while retaining
the same CUDA Lambert implementation. It uses the 1,000-asteroid reduced instance,
beam width eight, four maximum deployments and 24 neighbours. One cold run is
excluded, then three warm runs per mode produce the medians below.

| Complete route-search comparison | RTX 5090 | Lambda H100 |
| --- | ---: | ---: |
| CUDA Lambert, CPU neighbours | 4.144 s | 2.712 s |
| CUDA Lambert and neighbours | 3.602 s | 1.821 s |
| Additional speedup | **1.15×** | **1.49×** |

All candidate summary dictionaries match exactly in this paired comparison.
Each search returns 27 proxy routes after 545,658 completed Lambert branches,
373 CUDA hop batches and 77 GPU neighbour queries. The separate NumPy-versus-CUDA
benchmark records **12.326 s → 1.856 s on H100 (6.64×)**, retaining the same route
sequences, dates and mining yields. This is not certified solutions per second.

| Four queries over all 60,000 asteroids | RTX 5090 | Lambda H100 |
| --- | ---: | ---: |
| CPU warm median | 26.277 ms | 44.106 ms |
| GPU warm median | 3.243 ms | 1.831 ms |
| Warm speedup | **8.10×** | **24.08×** |
| CPU first group | 49.630 ms | 86.178 ms |
| GPU first group | 225.285 ms | 314.507 ms |

These four queries use 48 neighbours and five transfer times. All selected IDs
and their order match the CPU. Cold GPU creation/upload is slower; the benefit
requires reuse. Host packing, result transfer and Python dispatch are included.
The CPU reference is the existing NumPy implementation, not a tuned native CPU
solver. Both workloads use the pinned catalogue SHA-256 recorded in the evidence.

## Native interface and correctness

`gtoc12_neighbours_c_api.h` exposes a blocking host bridge and a device-buffer
operator. The latter supports CUDA Graph capture and replay with changing queries,
retained sort/scan scratch and no host copies or allocations during launch.
Its output capacity is the pool size. Workspaces must be serialized, including
graph replay and destruction; the host bridge rejects concurrent use.

The catalogue is an immutable snapshot for a workspace's lifetime. Changing a
pool or transfer-time grid recreates the cache; an earlier search reacquires it
when used again. Mutating catalogue arrays inside a retained context is unsupported.
Queries require positive finite bands, a positive int32 neighbour count and a
finite nonnegative filter scale. Invalid queries or failed ephemerides report an
error rather than silently switching to CPU calculations. Numerical work is FP64;
the neighbour translation unit disables fused multiply-add to preserve reference
operation order. Physics tolerances and mission constraints are unchanged.

The integrated local library passes **319 GTOC12 tests**. **26 targeted H100 tests**
pass. Native tests exercise stable ties, device graph replay, invalid-query
recovery and buffer/pool guards. Local and H100 memcheck report zero errors;
H100 racecheck reports zero hazards. These checks establish the tested operator
and search parity, not global optimality or full-application GPU residency.

## Reproduce and remaining work

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
export SPACEPDHCG_GTOC12_CUDA_LIBRARY=/absolute/path/to/libspacepdhcg_cuda.so
export SPACEPDHCG_GTOC12_DATA=/absolute/path/to/pinned/gtoc12/data
python scripts/gpu/benchmark_gtoc12_neighbours.py /new/path/paired.json
python scripts/gpu/benchmark_gtoc12_neighbour_pool.py /new/path/pool.json
python scripts/gpu/benchmark_gtoc12_route_search.py /new/path/routes.json --profile reduced --repeats 2
```

[Raw evidence](../results/lambda/2026-09-08/gpu-neighbours-v194) includes source and
runtime hashes, test output, timings and sanitizer reports. The integrated local
library is frozen as v193; the H100 screening library is frozen as v194. The latter
contains the screening operators, rather than the complete SCvx library.

Collection scheduling, beam/fleet orchestration, seed/retiming filters and some
packing/setup still run on CPU. This port covers RouteSearch.candidates, not every
element-deviation helper in the application. The fleet score remains
**12,805.194 weighted kg**; this benchmark has not produced a new certified fleet.
