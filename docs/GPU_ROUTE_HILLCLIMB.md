# GPU route hill climb and verified score gain

The new H100 fleet scores **12,843.555695585188 fixed-bonus weighted kg** and
returns **14,044.353182751598 raw kg**. That improves the previous fleet by
**0.585023 weighted kg / 0.602327 raw kg**, a small 0.00456% score gain.
The fleet still has **23 ships and 195 mined asteroids**, with **610.624051 raw kg
per ship**. All deployed miners are collected. The independent ship-count limit
is 23.003431. Both official and independent full-fleet physics checks pass.

The experiment starts from the independently qualified v733 fleet and searches
around its two recently replaced routes, columns 1786 and 2297 (viewer ships 1
and 2). CUDA controls fixed-order epoch search and candidate geometry; native
QOCO SCvx reflies promising changes with CUDA seeds, assembly and certificates.
The other 21 numerical trajectories are retained. Small timing adjustments add
0.164271 and 0.438056 raw kg to the two routes respectively.

## Actual work and timing

| Metric | RTX 5090 | Lambda H100 |
|---|---:|---:|
| Complete campaign, including checks/export | 59.213 s | 82.024 s |
| Two route-search calls, including refinement | 36.066 s | 39.743 s |
| Native leg attempts, including failures | 21.717 s | 26.418 s |
| Independent complete-fleet audit | 21.287 s | 39.011 s |
| Official complete-fleet check | 0.186 s | 0.356 s |
| Weighted score | 12,843.555695585288 kg | 12,843.555695585188 kg |

Native attempt time is contained within route-search time; these rows must not
be added together. Each is one completed campaign per GPU, not a matched
before/after speed benchmark. Import/setup and historical generation of the
retained fleet are outside the campaign timer. The approximately 1e-10 kg score
difference across GPUs is immaterial to this gain; the archived H100 result is
the displayed checkpoint.

Both runs report exactly **15,748 joint evaluations**, **13,172 joint batches**,
**220,079 computed geometry hops**, **154 cached geometry hops**, and **59,114
geometry rejections**. Twelve GPU-controlled epoch searches accept four timing
moves. The Lambert counter records 440,204 branch requests; requests and computed
geometry hops are different counters and are not newly certified trajectories.

There are **36 native leg attempts**: 34 converge and two report infeasible.
All **33 legs in the two accepted routes converge** and pass CUDA certificates.
The other three attempts explore earlier Earth departures/arrivals; none supplies
an accepted Earth-leg change. Their time and failures remain in the report.
Convergence status alone is not a physical-feasibility certificate.

Insertion screening examines 58 and 30 unoccupied neighbour candidates for the
two ships. No insertion seed closes the surrogate constraints, so no asteroid
insertion reaches full refinement. This is a bounded neighbourhood result, not
proof that adding an asteroid is impossible. The run allows 180 seconds and up
to four full-route certifications per ship; both finish before those limits.
The 0.1 kg search acceptance threshold does not change any physics tolerance.

## Verification, evidence and remaining work

Official `ScoreData.txt` masses, recomputed with the pinned bonus coefficients,
match the H100 independent weighted score **exactly**. Bonus-table SHA-256:
`e8a3795e599556ed5b66713ab1fa176de93ef37f93cb2a4a87d561539b1caa21`.
The promoted H100 solution SHA-256 is
`d368babcf0656ab55b9a21bd8ccde4eef22629a37f9bc8f8a1877eb15a3e7efa`.
The independent reports contain no violations; all qualification thresholds
remain unchanged.

The [retrieved evidence](../results/lambda/2026-09-09/gpu-fleet-routes-v767/)
contains both complete campaigns, original inputs, native attempt logs, accepted
route summaries and numerical trajectories, both checker outputs, score audits,
viewer exports, frozen v763 source and both native libraries. Each archive has
857 hashed payload members; hashes, lengths and unique member sets were checked
before and after transfer. The existing v763 core is the tested
[cooperative-tree build](GPU_FLEET_TREE.md); this campaign changes experiment
settings and inputs, not production numerical code. Reproduction scripts retain
their original workspace paths, and the archives contain the corresponding
source, inputs, core and QOCO binaries.

The main remaining search issue is batch size: 13,172 joint calls for 15,748
evaluations averages only 1.20 candidates per call. The insertion loop enumerates
asteroids, visit positions and seed schedules in Python and invokes the GPU
evaluator repeatedly. This is the next concrete target for larger GPU batches
and native candidate generation. The run establishes a real score gain; it
does not establish a fully GPU-controlled mission search or a leaderboard lead.

## Load the downloaded H100 mission

Solution file:
`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\gpu-fleet-routes-v767\h100-best\Result.txt`

The existing WebGL viewer now includes **H100 GPU route search v767**. Its import
checks the solution and catalogue hashes and preserves 11,680 exact replay
samples. The Kepler cross-check covers 73,201 context points. The displayed scene
has been visually checked at physical vertical scale (`z=1`).

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process node -ArgumentList @('scripts/serve.mjs','--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-routes-v767&epoch=69807&preset=oblique&z=1'
```

The score appears under **Compute & optimisation**; the ship bars show raw
collected kilograms. Earlier benchmark and mission datasets remain selectable.
