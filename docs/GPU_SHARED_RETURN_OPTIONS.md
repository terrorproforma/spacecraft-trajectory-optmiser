# Shared resident Earth-return options

Collection completion repeatedly asks for the same Earth-return geometry and
epoch/flight-duration grid. The native OrbitWeaver workspace now retains those
immutable option rows and hands callers independently owned handles to shared
device storage. Hits require no Lambert screening, device allocation, device
copy or device synchronisation. Mass, inflation, authority and schedule selection
remain live query inputs in the separate GPU collection operator.

The cache compares exact orbital-element bytes, gravitational parameter,
departure/arrival allowances, interleaved time-grid bytes and sorting mode.
It introduces no rounded epoch keys. A changed numerical query misses the cache.
The existing numerical operator still computes every miss, preserving option
ordering and tie rules. Invalid-time requests retain the existing empty-result
behavior and can also be retained.

The native cache holds at most 256 entries and 64 MiB of retained payload.
Least-recently-used entries release their ownership on eviction, while borrower
handles keep their underlying rows alive. The last owner frees the device
allocation. Cached calls, cached workspace destruction and borrowed option
operations require their owning thread/device. Oversize inputs use the ordinary
uncached computation. Failure to allocate optional retention metadata does not
discard a successfully computed result.

Within the CUDA resident-option path, sorted return queries use this cache by
default. `SPACEPDHCG_TEST_GTOC12_RETAIN_RETURN_OPTIONS=0` selects the original
path for matched measurements. Compact host outputs and unsorted collection-hop
queries keep their established behavior.

Telemetry distinguishes successful cache hits from fresh computations. A hit
increments shared-handle/cached-branch counters without incrementing completed
GPU batches, fresh branch requests or option downloads. RouteSearch's return
evaluation counter uses the actual submitted hop count, including zero on a
hit. This prevents reused numerical results being presented as new solves.

## Paired measurements

| GPU | Route | Fresh return rows | Shared return rows | Search throughput gain |
| --- | --- | ---: | ---: | ---: |
| RTX 5090 | Ship 10 | 13.209 s | 10.123 s | 30.5% |
| RTX 5090 | Ship 21 | 10.959 s | 8.454 s | 29.6% |
| H100 | Ship 10 | 6.954 s | 6.019 s | 15.5% |
| H100 | Ship 21 | 6.090 s | 5.205 s | 17.0% |

These are whole-search medians from three alternating measured samples per
mode after a warm-up, using the same build with only the retention flag changed.
Process startup and catalogue loading are excluded. All 989 candidate records
are byte-identical across modes and repetitions on each GPU, and match the
preceding admission checkpoint. Combined throughput is about 53 candidates/s
on RTX 5090 and 88 on H100. These are surrogate search rates, not certified
solutions per second or inner-solver speedups.

Across both routes, 12,188 of 12,346 return requests reuse rows. Only 158 require
new return screening. This avoids 12,992,408 repeated Lambert branch requests;
the total fresh GPU branch-request count falls from 35,358,344 to 22,365,936.
The audit reconciles these changes against both the GPU telemetry and search
evaluation counters. All option-row builds, including unsorted collection hops,
fall from 13,906 to 1,718. Selection still uses each request's current mass.

Both devices pass 205 tests, 67 tests under each of CUDA memcheck/racecheck/
synccheck, separate controller checks, and 29 focused GPU tests under full leak
checking. There are no reported CUDA errors, hazards or leaks. Coverage includes
exact key changes, independent handle closure, producer replacement, cache
eviction, empty results, wrong-thread rejection and truthful evaluation counts.

The source base is `9681f6ce41cfc11dac927b0965efa879605f4f9a` with seven owned
source/test overlays. v844 builds and validates the runtime, v845 measures it,
v847 checks leaks and v848 profiles the same runtime afterwards. Exact source,
binaries, every paired candidate file and raw profiles are retained in the
[downloaded v846 evidence and audit instructions](../results/lambda/2026-09-09/gpu-shared-return-options-v846/README.md).

## Remaining work and score

The subsequent RTX ship-10 profile attributes 6.48 cumulative seconds of an
11.49-second profiled search to collection-DP scheduling. There are still 1,152
per-tour preparations and 1,731 DP solves, including 2.28 cumulative seconds in
workspace setup/rebinding. These profiling times overlap and are not additional
benchmark costs. Moving that preparation/control onto CUDA and batching route
completions is the next major performance target. Collection ordering and outer
scheduling still involve the host; the complete planner is not yet GPU controlled.

This optimization certifies no new trajectory and changes no fleet score. The
separately verified v633 fleet remains 24 ships / 208 asteroids at 13,526.961241
weighted kg and 14,915.044490 raw kg. Wider saved route candidates still need
native refinement and independent checking against the current fleet budget.
