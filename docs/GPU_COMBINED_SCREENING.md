# Combined CUDA transfer screening

With `--screening-backend cuda`, GTOC12 now computes both Lambert directions,
velocity-matching costs, Earth velocity allowances and the cheaper direction in
one CUDA operator. The host packs the request and receives the selected result.
This removes the NumPy cost/selection stage and halves submissions for a batch
that fits the retained workspace. The resident C API also supports CUDA Graph
capture with caller-owned buffers and no host transfers or allocations.

The operator uses FP64 arithmetic and the stable geometry from the previous
[screening checkpoint](GPU_CANDIDATE_SCREENING.md). Equal costs select short-way,
matching the CPU order. Invalid inputs produce infeasible results with infinite
costs and NaN velocities. Allowances must be finite and nonnegative. A failed
workspace recreation can now be retried without leaving a stale scan-grid key.
Native telemetry accounts for the actual input/output record sizes when ordinary
Lambert batches and combined hop batches share a workspace.

These remain impulsive screening estimates. Low-thrust refinement and independent
certification are still required before a candidate contributes to the fleet.
The verified fleet score is unchanged at **12,805.194 weighted kg**.

## Direct performance comparison

The paired benchmark alternates the previous GPU-roots/CPU-costs path with the
combined GPU path, in the same process, with the same retained workspace and
8,192 candidate transfers. It excludes the first call of each path, then takes
the median of six measurements. Ephemeris generation is outside the timed region;
packing, host transfers and result selection are included.

| Warm 8,192-transfer batch | RTX 5090 | Lambda H100 |
| --- | ---: | ---: |
| Previous GPU roots + CPU costs | 13.674 ms | 6.383 ms |
| Combined GPU screening | 10.698 ms | 3.384 ms |
| Additional speedup | **1.28×** | **1.89×** |
| GPU submissions per batch | 2 → 1 | 2 → 1 |

The separate catalogue comparison still preserves the CPU's top 100 candidates
on both GPUs and passes the unchanged 10-metre Kepler closure gate. A native
test covers graph replay with changing inputs, host/device output agreement,
zero-cost ties, invalid inputs and byte-count telemetry. Local and H100 native
memcheck report zero errors; H100 racecheck reports zero hazards. The final
targeted Python suite passes 21 tests on H100, including recreation failure/retry.
The final integrated library passes all **314 local GTOC12 tests**, including
the unchanged physics and archived-score checks.

## Larger route search and corrected counters

The new `--profile reduced` benchmark uses all 1,000 asteroids in the pinned
reduced instance, a beam of eight, up to four deployments and 24 neighbours.
It constructs a fresh RouteSearch for each run. One cold and two warm runs per
backend retain the same workspace; the table uses the median warm time.

| H100 host comparison | CPU NumPy | Combined GPU screening |
| --- | ---: | ---: |
| Route-search time | 12.073 s | **2.710 s** |
| Completed Lambert branch evaluations per run | 545,658 | 545,658 |
| Search expansions | 24 | 24 |
| Proxy route candidates returned | 27 | 27 |

This is **4.45× faster** for this search profile. All 27 candidates retain the
same asteroid sequences, event epochs and mined masses; propellant and final-mass
proxies agree within 1e-6 kg. These are not 27 certified low-thrust solutions.
The GPU submits 373 batches per search, including workspace-sized chunks.

The larger fixture exposed an existing accounting error: manual increments in
RouteSearch omitted work performed by collection and return helpers, reporting
197,566 branches instead of 545,658. RouteSearch now reports the difference in
actual completed operator rows across its full execution context. CPU and CUDA
use the same accounting boundary; the benchmark independently requires the search
total to match the GPU workspace's branch counter. Earlier undercounted reports
are preserved and are not used as the final evaluation totals.

## Remaining work

A separate H100 CPU profile records 3.20 seconds under instrumentation, with
about 0.88 seconds inside the GPU screening bridge (including GPU waits),
0.46 seconds in element-deviation calculation and 0.43 seconds in the phasing
proxy. These cumulative timings overlap and must not be added indiscriminately.
Neighbour filtering/ranking and collection scheduling are substantial remaining
CPU work. Moving their numerical operations onto the GPU is the next target;
the full fleet-search application is not yet GPU-native.

## Reproduce

Use the normal integrated `spacepdhcg_cuda` library and pinned catalogue:

```bash
export PYTHONPATH=src
export SPACEPDHCG_GTOC12_CUDA_LIBRARY=/absolute/path/to/libspacepdhcg_cuda.so
export SPACEPDHCG_GTOC12_DATA=/absolute/path/to/pinned/gtoc12/data
python scripts/gpu/benchmark_gtoc12_combined_hops.py /new/path/paired.json
python scripts/gpu/benchmark_gtoc12_route_search.py /new/path/routes.json --profile reduced --repeats 2
```

The benchmark uses the existing NumPy search as its reference; it is not a
comparison with a tuned native CPU implementation. Raw timings, source/runtime
hashes, sanitizer output, the initial benchmark setup failure and profiling
follow-up are retained in [downloaded evidence](../results/lambda/2026-09-08/gpu-combined-hops-v190).
