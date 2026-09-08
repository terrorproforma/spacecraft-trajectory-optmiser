# GPU construction of insertion layouts

Insertion screening now uploads shared route/edge inputs once and constructs
every trial layout on CUDA. On the same **12,992-schedule** neighbourhood,
complete warm screening falls from **1.103 s to 59.986 ms on RTX 5090** and
**2.084 s to 55.704 ms on H100**: **18.4× / 37.4× faster** than the host-built
layout path in the same candidate core. This includes source compilation,
transfers, screening and result handling. Throughput is approximately
**217,000 / 233,000 surrogate schedules per second**, not certified trajectories.

## Profile and batch-size search

The v775 Python profile found 3,248 geometry-metadata builds and 116,928
`body_elements` calls in one insertion search. Geometry packing accounted for
most profiled screening time. Profiling adds overhead; its seconds are not the
unprofiled benchmark times below.

The new source has **1,756 shared edges** and uploads **368,832 bytes once**.
Each body's orbital elements are packed once. CUDA generates the deployment/
collection positions, visit metadata, donor indices and edge indices, then
generates the four epoch seeds and performs the existing geometry/mass pipeline.
Only feasible survivors need a host `Visit` list at the refinement boundary.

| Layouts per batch | RTX 5090 median | H100 median |
|---|---:|---:|
| Existing host-built layouts, 256 | 1.103478 s | 2.084305 s |
| GPU-built layouts, 64 | 99.473 ms | 68.988 ms |
| GPU-built layouts, 256 | 78.895 ms | 65.773 ms |
| GPU-built layouts, 1,024 | 63.113 ms | 58.176 ms |
| GPU-built layouts, 4,096 | **59.986 ms** | **55.704 ms** |

The benchmark rotates execution order, uses a fresh learned-itinerary copy for
each call, and records five repetitions after one warm-up. Candidate counts and
result signatures agree in every mode. This particular neighbourhood has zero
feasible insertions; separate warm-cache tests admit feasible rows and check
their mass values and plans. The second archived route has no combined deployment/
collection turnaround and was excluded by the insertion search at this checkpoint;
its no-op timings are not a speed claim. This search-coverage restriction is now
[fixed and tested on both GPUs](GPU_INSERTION_TURNAROUNDS.md).

The default is now 4,096 layouts per batch when the native prepared-input API is
available. `SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_LAYOUTS=0` selects the prior path;
`=1` requires the new API. Older libraries retain the previous default when the
switch is unspecified. Explicit batch limits remain supported.

## Correctness and ownership

Both GPUs pass **124 joint-search regression tests**; seven tests skip absent
historical fixtures. Twenty insertion/layout tests pass under each of CUDA
memcheck, racecheck and synccheck, with no reported errors or hazards.

Tests compare every generated metadata field and selected edge stage against the
original host builder, every enabled epoch exactly, all result rows exactly, and
valid mass/detail rows exactly. They include cold and cached geometry, partial
batches, workspace reuse between old and new operators, and a guard forbidding
host layout enumeration. Native input failures preserve outputs and an existing
prepared source. Input snapshots survive caller mutation; a Python owner rejects
execution after another owner replaces its prepared source.

`spacepdhcg_gtoc12_joint_prepare_insertions_host` owns immutable device inputs
separately from shared evaluation buffers. The prepared evaluation API receives
only a layout range and output pointers. Optional metadata outputs support
independent parity tests. Sparse measured/cache records use shared edge IDs, so
their contents need not be duplicated for every possible route layout.

## Full mission replay

Both official and independent full-fleet checks pass on RTX 5090 and H100 at
unchanged physics tolerances. The score remains **12,843.555696 weighted kg**,
with **14,044.353183 raw kg**, 23 ships and 195 asteroids. Differences of roughly
1e-10 kg across replays are numerical variation, not a score gain.

The replay still performs **15,748 joint evaluations**, **220,079 geometry hops**
and **36 native leg attempts**. All 33 legs of the two accepted routes converge
and have CUDA certificates. Insertion calls fall from 13 to **one**; total joint
calls fall from 193 to **181**. No evaluation or qualification work is removed.

| One complete replay | RTX 5090 | H100 |
|---|---:|---:|
| Route searches including native refinement | 23.734 s | 27.592 s |
| Independent full-fleet audit | 20.510 s | 39.020 s |
| Campaign including checks/export | 46.173 s | 69.823 s |

These complete-campaign figures are single observations, not a repeated
end-to-end speedup study. Refinement and independent auditing dominate them.
The earlier insertion-batch checkpoint took 46.294 / 71.782 seconds.

Shared-edge input compilation, outer route orchestration, feasible-result
sorting and artifact production still involve Python. Full result and epoch
buffers still return to the host. GPU survivor compaction and wider route search
remain work to do; this is not a fully GPU-controlled mission optimiser yet.

## Retrieved H100 result

[Evidence and reproduction material](../results/lambda/2026-09-09/gpu-layouts-v780/)
include both source/runtime snapshots, profiles, all batch-size trials, tests,
physics checks and numerical mission replays. Both archives have 913 hashed
payload members, verified before and after transfer. Scripts retain the original
workspace paths. The visualiser imports exact replay samples and checks the
solution/catalogue hashes and Kepler context against the verified export.

Solution:
`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\gpu-layouts-v780\h100-best\Result.txt`

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process node -ArgumentList @('scripts/serve.mjs','--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-layouts-v780&epoch=69807&preset=oblique&z=1'
```
