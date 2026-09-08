# CUDA insertion generation and batched screening

The insertion search now generates its candidate schedules on CUDA and evaluates
different route layouts together. On the measured 12,992-schedule neighbourhood,
median complete screening time falls from **11.438 to 1.072 seconds on RTX 5090**
and **11.101 to 2.086 seconds on Lambda H100**: **10.67× / 5.32× faster**,
including Python metadata preparation and result reconstruction. Throughput is
approximately **12,121 / 6,227 schedule evaluations per second**. These are
surrogate schedules, not independently certified low-thrust trajectories.

Five measured repetitions follow one warm-up, with alternating old/new call
order and a fresh copy of the same learned itinerary for every call. Each result
has the same candidate count and feasible-option signature. This neighbourhood
has no feasible insertion. The second archived route has no combined deploy/
collect stop and submits zero schedules; its no-op timings are not speed claims.
Separate cached-geometry tests admit nonempty feasible result lists and verify
exact ranking, duplicate/tie order, mass values and reconstructed plans.

## What moved onto CUDA

Each batch holds up to 256 visit layouts and four seeds per layout. A CUDA kernel
generates the midpoint and three camp-borrowing schedules directly from the
original itinerary. It preserves the original arithmetic order, the greater-
than-one-day borrowing gate and candidate order. Preflight, exact-epoch cached
cost lookup, ephemerides, Lambert solves and mass bookkeeping stay on the device.
The evaluator and geometry kernels now index each candidate's layout rather
than assuming every candidate has the same visit order.

`spacepdhcg_gtoc12_joint_insertions_host` retains layout and schedule buffers in
the existing workspace and synchronizes once per batch. A supported active CUDA
joint backend selects this path by default. Explicit
`SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_INSERTIONS=0` retains scalar dispatch for
comparisons; `=1` requires the new native API. Older libraries retain their
existing path when the switch is unspecified. Unsupported overridden numerical
callbacks are rejected before launching candidate work.

Python still enumerates structural layouts, packs metadata and reconstructs/
sorts feasible plans. Full candidate result and epoch buffers still return to
the host. Moving structural generation and survivor compaction onto CUDA is the
next target. This change does not complete GPU control of the whole mission
search.

## Qualification and complete mission replay

Both GPUs pass **116 tests**, with seven historical-fixture tests skipped because
their referenced archives are absent. The insertion suite passes **12 tests**
under each of memcheck, racecheck and synccheck, with zero reported errors or
hazards. The tests compare generated epochs exactly against the original Python
builder, compare each enabled row with the established evaluator, check sparse
cache/layout reuse and malformed-input output preservation, and forbid calls to
the scalar evaluator and seed generator. The initial v768 run's five failures
were test assumptions that a route had a combined deploy/collect stop; its actual
insertion cases passed. Those initial logs are retained.

The v773 mission replay evaluates the same **15,748 joint schedules** and
**220,079 geometry hops** as v766, including the same 36 native leg attempts.
All 33 legs in the two accepted routes converge and have CUDA certificates.
Both official and independent complete-fleet physics checks pass on both GPUs,
at unchanged tolerances. The score remains **12,843.555696 weighted kg** and
**14,044.353183 raw kg**, with 23 ships and 195 asteroids. Approximately 1e-10 kg
differences are numerical variation, not a new score improvement.

| Complete replay measurement | RTX 5090 | H100 |
|---|---:|---:|
| Route searches including native refinement | 24.077 s | 29.621 s |
| Independent complete-fleet audit | 20.420 s | 38.920 s |
| Campaign including checks/export | 46.294 s | 71.782 s |

These are one replay per GPU, not the repeated screening benchmark. Prior v766
campaign times were 59.213 / 82.024 seconds. The replay's joint-call count falls
from **13,172 to 193**, including **13 insertion batches instead of 12,992 scalar
insertion calls**. The remaining 180 calls are other search stages. No work is
removed from the evaluation or certification counts.

## Downloaded evidence and visualiser

[Evidence, frozen source, native binaries and reproduction scripts](../results/lambda/2026-09-09/gpu-insertions-v774/)
include both GPU builds, initial and final tests, all benchmark repetitions and
both complete mission replays. Scripts retain their recorded workspace paths.
Each raw archive has 912 hashed payload members, verified before and after
transfer. Final source excludes mutable pytest cache entries from content
verification; original source manifests remain archived for provenance.

The H100 solution is:
`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\gpu-insertions-v774\h100-best\Result.txt`

Load its replay in the existing visualiser:

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process node -ArgumentList @('scripts/serve.mjs','--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-insertions-v774&epoch=69807&preset=oblique&z=1'
```
