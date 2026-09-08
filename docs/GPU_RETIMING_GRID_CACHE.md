# Resident retiming transfer-table cache

Retiming variants repeatedly ask for the same immutable Lambert grids. Direct
timers measured 7.258 seconds rebuilding grids across 50 calls, versus only
0.104 seconds in the GPU price/mass driver. A trace contained 821 grid requests
but only 85 distinct combinations of orbital elements, epochs and flight times.

The Lambert workspace now retains completed cost/feasibility tables on the GPU.
The key compares all input bytes, including gravitational parameter and launch/
arrival allowances. Scan configuration is fixed by the owning workspace.
Destroying or recreating that workspace releases its entries. Changed inputs
cannot reuse a stale table. Cache hits copy device-to-device into the retiming
workspace; there are no table downloads or CPU trajectory calculations.

The LRU cache retains at most **256 entries** and **64 MiB of cost/flag payload**.
Host keys, allocation overhead and temporary staging are additional to that
payload bound. Oversize grids are computed without retention. Masses, prices,
authority policies and sweep corrections remain outside the immutable cache and
are read from the current retiming workspace. Caching is enabled by default;
`SPACEPDHCG_TEST_GTOC12_RETIME_TABLE_CACHE=0` selects the same uncached CUDA
arithmetic for controlled comparisons.

## Measurements

Both comparisons use baseline/candidate/candidate/baseline process order, the
same native binary, explicit cache-off/cache-on settings, and complete one-ship
CLI runs. All runs preserve logical search counts and pass both final mission
checkers at 548.254620 weighted kg.

| Hardware | Uncached median | Cached median | Change |
|---|---:|---:|---:|
| RTX 5090 | 43.449518 s | 35.997704 s | 17.15% less time; 1.21× |
| H100 | 42.975965 s | 42.423565 s | 1.29% less time |

There are two samples per mode. The small H100 difference does not establish a
significant end-to-end speedup. These ratios are specific to this fixture;
they are not universal multipliers or independent factors to multiply with other
benchmark ratios.

A subsequent local diagnostic with the final entry cap observed **736 hits,
85 misses, zero evictions and 12,527,082 retained payload bytes**. Native grid
construction fell to **0.823 seconds** and total retiming to **0.981 seconds**.
These are inclusive diagnostic timers, not additional paired benchmarks.
Logical screening-request counters include cache hits; they must not be reported
as newly computed Lambert solutions or fully verified spacecraft missions.

## Validation and evidence

Native tests compare every output byte with the uncached device bridge, on cold
and warm calls and after changes to epochs, flight times, orbital elements,
allowances and gravity. They check invalid inputs, workspace recreation, entry-
count eviction and byte-budget eviction. Assertions remain enabled in release
builds. Memory, synchronization and race checks pass on both GPUs, together with
114 selected Python tests covering resident retiming, price/mass driver graphs,
SCvx, CLI execution and final fleet verification.

The final default-on source also passes an H100 complete campaign in 41.902137
seconds with both final mission checkers passing. This single confirmation is
not a new speed estimate. Physics models, solver tolerances and acceptance
criteria are unchanged. CPU orchestration and independent checking remain.

- [Local profiles, native checks and paired campaigns](../results/lambda/2026-09-08/gpu-grid-cache-local-v401/summary.json)
- [H100 paired experiment](../results/lambda/2026-09-08/gpu-grid-cache-v397/summary.json)
- [H100 additional entry-cap validation](../results/lambda/2026-09-08/gpu-grid-cache-v400/summary.json)
- [H100 final default-on source](../results/lambda/2026-09-08/gpu-grid-cache-v402/summary.json)

Each directory includes checksums, raw archives, source snapshots and commands.
The H100 source is the archived v387 base plus each recorded overlay. Native
runtime hashes distinguish the initial byte-cap experiment, added entry cap and
final default-on build. The independent fleet incumbent remains 12,805.194
weighted kg.
