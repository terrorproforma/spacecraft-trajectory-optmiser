# CUDA candidate screening for GTOC12

GTOC12 search can now use the native CUDA Lambert operator through
`--screening-backend cuda`. A retained workspace serves successive search batches;
CUDA performs the root scans, bisection and velocity reconstruction. The existing
NumPy implementation remains the default and an independently executable comparator.
An explicitly selected CUDA backend fails on backend errors instead of falling
back to CPU arithmetic. Its current Python search integration requires `--workers 1`.

This is impulsive candidate screening. A candidate still needs low-thrust
refinement and independent physics certification before contributing to a fleet.
No new fleet score is claimed by these benchmarks.

## Changes and accuracy

The existing CUDA kernel was not connected to the GTOC12 screening path. The new
scoped backend connects it to run, cluster-fleet, fleet-master, return-retiming and
joint-itinerary commands. Reports record completed GPU batches and branch requests,
including calls made after the report dictionary is initially constructed.

Zero-revolution scans retain their fixed grid of Stumpff coefficients in device
memory and stop after the first root. A blocking batch bridge avoids the managed
cancellation and telemetry pages that limited the older asynchronous host path on
WSL. The cancellable asynchronous API remains available. A separate device-buffer
operator performs no allocation or host transfer and supports CUDA Graph capture.
Host API entry is guarded so another thread cannot mistake an old completion
event for completion of a blocking call, collect duplicate counters or destroy
its workspace. GPU grid initialization completes before host managed-page writes;
without this barrier, a preliminary WSL build crashed during repeated creation.

The catalogue benchmark exposed a geometry error shared by CPU and GPU paths near
aligned endpoints. Computing the sine from a cross product and the Lambert
geometry coefficient from a vector half-angle identity removes the cancellation.
Both C++ CPU implementations, NumPy and CUDA now use the stable construction.
The first captured case missed its endpoint by about 28 metres. An intermediate
cross-product-only fix failed another case by about 77 metres. Those failed runs
are retained alongside the passing candidate; the 10-metre screening closure gate
was not relaxed. Low-thrust certification tolerances are unchanged.

## Measurements

Evidence: [local and H100 reports](../results/lambda/2026-09-08/gpu-screening-v186).
The catalogue workload contains 8,192 Earth-to-asteroid candidate transfers, with
both directions tested: 16,384 Lambert branch requests per timed batch. Timings
are medians of five warm calls after one cold call, using identical inputs on
each GPU. They include host transfers, velocity-matching costs and branch selection,
but exclude ephemeris generation, catalogue loading and workspace creation.

| Measurement | RTX 5090 | Lambda H100 |
| --- | ---: | ---: |
| CUDA screening batch | 13.395 ms | 6.359 ms |
| Candidate transfers per second | 611,582 | 1,288,273 |
| Existing NumPy screening batch | 148.836 ms | 269.502 ms |
| Relative speedup on the same host | 11.11× | 42.38× |
| Small complete route search, warm speedup | 1.09× | 2.39× |

The CPU comparator is the existing NumPy implementation, not a tuned batched
native CPU implementation. Different CPU performance accounts for part of the
difference between these speedup ratios. They do not measure general low-thrust
solution throughput or the full fleet-search pipeline.

Both GPUs retained the same top 100 catalogue candidates. The largest CUDA
Kepler endpoint error was approximately 6.43 metres, below the unchanged 10-metre
screening gate. Both GPUs passed all 16 targeted Python tests, the existing native
CPU/GPU Lambert comparison, and the new host/async/device-graph and concurrency
test. Local and H100 native memcheck reported zero errors; H100 racecheck reported
zero hazards. The integrated library passed all 309 local GTOC12 tests; four native CPU
Lambert smoke/oracle tests also passed. Their raw reports are retained separately.

The route benchmark starts from 60 asteroids, performs 4,106 Lambert branch
evaluations in 16 GPU batches per search, and yields the same single candidate
on CPU and GPU. Each of six runs constructs a fresh RouteSearch, while the GPU
workspace survives across warm runs. H100 cold search takes 0.371 seconds versus
0.173 seconds on CPU; warm search takes 0.071 versus 0.169 seconds. This bounded
fixture establishes integration parity, not broad mission search quality.

## Reproduce

Build the normal `spacepdhcg_cuda` CMake target. In the configured Linux/WSL runtime:

```bash
export PYTHONPATH=src
export SPACEPDHCG_GTOC12_CUDA_LIBRARY=/absolute/path/to/libspacepdhcg_cuda.so
export SPACEPDHCG_GTOC12_DATA=/absolute/path/to/pinned/gtoc12/data
python scripts/gpu/benchmark_gtoc12_screening.py /new/path/screening.json
python scripts/gpu/benchmark_gtoc12_route_search.py /new/path/routes.json
```

Add `--screening-backend cuda --workers 1` to an existing GTOC12 search invocation.
This selects screening arithmetic independently of the refinement backend.
Ephemeris generation, cost selection, beam/collection search, fleet selection,
input packing and some solver setup still contain CPU work. Completing their GPU
conversion and generating better certified routes remain open work.
