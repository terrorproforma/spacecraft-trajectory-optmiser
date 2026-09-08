# Fused GPU collection tables

Collection-table construction now writes its final float32 device table directly
from the existing double-precision Lambert results. The old path wrote temporary
double costs and feasibility flags for the entire grid, then launched a separate
rounding kernel. The new path retains reusable epoch/duration axis storage in the
Lambert workspace and removes those temporary full-grid arrays and the temporary
epoch allocation. The output remains owned by each collection table.

This is a memory and construction-time optimisation. Collection tables already
used float32 storage; no physics calculation or acceptance tolerance is reduced.
The cost sum, finite/feasibility checks, end-date condition (`end + 1e-9` days),
and final float32 rounding retain the old operation order. Invalid cells remain
positive infinity. Orbit propagation and Lambert branch selection are unchanged.

The fused path is enabled by default. Set
`SPACEPDHCG_TEST_GTOC12_FUSED_TABLES=0` to reproduce the old path on the same
binary. The native API accepts host axes and a caller-owned device output,
refreshes retained axes for every request, validates workspace device ownership,
and synchronizes its stream before returning. It does not introduce a device
search controller or make the whole planner GPU native.

## Measurements

The isolated replay captures all 1,122 actual collection grids from a complete
one-ship campaign: 4,644,540 cells. It replays both paths with a separate full
warmup for each, followed by baseline/candidate/candidate/baseline passes. Timings
sum actual table-constructor wall time, excluding validation downloads, hashing,
destruction and other mission work. Each pass retains its tables until the end.
Every table has the same SHA-256 hash in every pass, on both GPUs.

| GPU | Previous construction median | Fused construction median | Less construction time |
|---|---:|---:|---:|
| RTX 5090 | 2.8052 s | 2.7281 s | 2.75% |
| H100 | 0.4661 s | 0.4319 s | 7.33% |

Separate complete-campaign comparisons, two observations per mode:

| GPU | Previous CLI median | Fused CLI median | Observation |
|---|---:|---:|---|
| RTX 5090 | 30.6543 s | 28.5748 s | 6.78% less time in this batch |
| H100 | 31.5764 s | 31.7389 s | 0.51% more time; effectively flat |

The isolated improvement is much smaller than the local whole-campaign
difference. The complete-run measurements vary and do not establish an overall
speedup attributable to this change. The H100 result is retained alongside the
local result. All eight campaigns preserve every screening counter and the
complete initial candidate plans exactly. Both mission checkers accept every
campaign at 548.254620 weighted kg, 45,188,558 logical branches and 2,782,091
collection options. Logical branches include table reuse, not unique missions.

The original local profile measured 2.838 seconds in table construction. The
fused profile measured 2.797 seconds, including trace instrumentation, and
captured the exact axes used by the isolated replay. An earlier profile launch
v450 exited at the GPU lock before running any numerical work; that terminal
report is retained, and v451 performed the capture after the benchmark finished.

## Validation and limitations

Tests compare every float32 bit against the old path across growing and shrinking
axis shapes, partial batches, invalid durations/epochs, Earth returns, and
deadlines just inside and outside the existing tolerance. They retain earlier
outputs while the shared axes change and check workspace reuse after invalid
arguments. Existing collection, resident-DP and harvest-window tests also run.
The selected suite contains 26 tests; memory, synchronization and race checks
exercise the actual shared-library path.

The final default-on build passes all 26 tests and all three sanitizer checks on
RTX 5090 and H100. The final H100 campaign completes in 31.610 seconds with both
mission checkers accepting 548.254620 weighted kg. A wider local campaign builds
6,427 resident collection tables and completes in 226.739 seconds, retaining four
ships, 29 mined asteroids and 2,088.668592 weighted kg. Both checkers pass, with
169,753,864 logical branches and 18,250,121 collection options. This wider run is
a validation, not a paired speed comparison. The best retained fleet remains
12,805.194 weighted kg.

There is no new host table-result transfer in construction. Duration storage and
window-pricing buffers still belong to each table. The retained axis scratch
grows with the largest axis pair requested and is included in workspace memory
telemetry. Table creation and search orchestration still make host calls.

The performance comparisons use opt-in v447/v449 native builds. The final build
changes only the default selection of the same fused path. Raw commands, source
snapshots, runtime hashes, complete mission outputs, sanitizer logs and
SHA-256-verified archives accompany the results:

- [Local validation, comparisons, profile and replay](../results/lambda/2026-09-08/gpu-fused-tables-local-v455/summary.json)
- [H100 validation and complete comparisons](../results/lambda/2026-09-08/gpu-fused-tables-v449/summary.json)
- [H100 isolated replay](../results/lambda/2026-09-08/gpu-fused-tables-v453/summary.json)
- [Final default-on H100 validation](../results/lambda/2026-09-08/gpu-fused-tables-v456/summary.json)
