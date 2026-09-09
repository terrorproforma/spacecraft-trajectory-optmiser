# GPU fleet regeneration — 9 September 2026

The verified fleet improves from **12,843.555696 to 12,992.035662 weighted kg**
while returning **14,271.047228 raw kg**, up **226.694045 kg**. Nine routes are
replaced; 23 ships now mine 200 asteroids. Both the RTX 5090 and Lambda H100
results pass the independent full-fleet propagator and official competition
verifier at unchanged tolerances. The weighted gain is **148.479966 kg (1.156%)**.

| Verified metric | Previous v780 | New v799 H100 |
| --- | ---: | ---: |
| Weighted score | 12,843.555696 kg | 12,992.035662 kg |
| Physical returned material | 14,044.353183 kg | 14,271.047228 kg |
| Raw material per ship | 610.624051 kg | 620.480314 kg |
| Ships / mined asteroids | 23 / 195 | 23 / 200 |
| Continuous ship-count allowance | 23.003431 | 23.928457 |

The score uses the frozen end-of-competition bonus coefficients. It is a local
verified result, not an official leaderboard submission. The display opens the
downloaded H100 trajectory with physical Z scale, both checker results and the
new score. Fourteen incumbent routes are retained.

## What changed in the search

The first all-ship pilot, v792, generated 1,238 surrogate routes. Although some
improved weighted score, every candidate returned less raw material than its
corresponding incumbent. Refining four candidates produced three individually
certified routes. Combining the two best raised the hypothetical score to
12,864.124 kg but reduced raw mass to 13,928.597 kg: the 23-ship fleet exceeded
its allowable 22.544967 ships. Both full-fleet checkers rejected v793. Its
complete rejected trajectory and failure reports are preserved.

The next search, v794, optimises raw mass during candidate generation, then uses
the original bonus weights to shortlist and select the final fleet. It raises
the beam from 16 to 64, neighbours from 48 to 96, and the per-first-asteroid beam
limit from 8 to 64. That last limit matters: every ship has one fixed Earth-leg
seed, so the old diversity limit restricted its effective beam to eight.

Each ship starts from its existing Earth leg. Candidate asteroid sets contain
up to 192 neighbours plus that ship's current asteroids, excluding the other
22 incumbent ships' asteroids. Search keeps at most ten deployments and uses
the existing calibrated collection model. The complete run produces **10,971
surrogate routes and 11,737 beam expansions on each GPU**. All candidate
identities, ordering, epochs and collected masses match between GPUs.

The approximate Lambert, inflation and propellant fields differ slightly:
44,341 floating-point fields differ, with maximum absolute difference
2.30e-11. An initial byte-equality audit therefore failed; the saved-data audit
now requires exact schedules and cargo, allowing less than 1e-9 difference only
in those four proxy fields. This comparison does not alter physics acceptance.

The shortlist retains up to two distinct schedules per ship that gain raw mass
and lose at most 30 weighted kg. **26 routes are refined; 18 certify on each
GPU**. CUDA fleet selection resolves conflicts and the raw-mass ship rule.
The final pool combines 23 incumbent routes, 18 newly certified routes and the
three individually certified pilot routes. Uncertified proxies never enter it.

Increasing the retained CUDA search budgets from 2 million through 2 billion
visits **17,607,866 actual nodes** on the exhaustive final pass. It confirms the
same best fleet within these **44 certified columns**, with matching selected
IDs on both GPUs. This is a finite-pool result, not a global mission optimum.

## Measured work and timings

These are single campaign measurements, not repeated speedup benchmarks.

| Measured scope | RTX 5090 | H100 |
| --- | ---: | ---: |
| Main raw-mass route searches | 515.911 s | 485.964 s |
| Surrogate routes produced per search second | 21.27 | 22.58 |
| Native leg attempts in v795 | 373 | 373 |
| Native call time, including failed attempts | 83.996 s | 112.779 s |
| Native leg attempts per native-call second | 4.44 | 3.31 |
| Refinement phase including fleet selection/checks | 110.881 s | 157.440 s |
| Final exhaustive CUDA fleet pass | 5.391 s | 1.884 s |
| Actual B&B nodes per second on that pass | 3.27 million | 9.35 million |

The 373 attempts include **365 converged** calls on each GPU. RTX records five
failed and three infeasible calls; H100 records two failed and six infeasible.
All eight rejected routes remain excluded. These rates count attempts, not
fresh complete missions or independently certified fleets. The earlier pilot
adds 72 native attempts per GPU, making **445 total native calls** in the two
refinement experiments. Later fleet-master sweeps reuse those certified routes.

Main-search telemetry counts 454,203,618 logical Lambert branch requests plus
46 requests for the fixed Earth-leg screens. Its overlapping counters include
204,833,349 completed element hops, 189,178,441 collection-option checks and
37,792 collection-DP passes. They measure different stages and must not be
added together or described as that many certified trajectory solutions.

The new H100 replay has maximum independent errors of 26.573131 km in position,
4.949671e-6 km/s in velocity and 1.005e-10 kg in mass; both full-fleet checkers
pass. RTX and H100 weighted totals differ by less than 6e-10 kg. The imported
viewer preserves 11,677 exact replay samples and checks 75,006 Kepler context
points against the pinned catalogue.

## Remaining GPU work

This checkpoint tunes the search and exercises the existing native CUDA/QOCO
pipeline. It retains the frozen v788 source/core and the published archived-ZOH
initializer. The first Earth leg is initialised from exact archived controls;
subsequent states are propagated by the native solver. QOCO is the mission
subproblem backend; this campaign does not establish PDHCG convergence.

Python still controls beam expansion, builds inputs and materialises routes.
The H100 main search creates **25,112 collection-DP workspaces**. Completion
packing takes **48.280 s**, while its CUDA kernel takes **0.528 s**. Retaining
DP allocations across changing tours, then moving completion packing and beam
control onto the GPU, are concrete next performance targets. CPU full-fleet
checks and viewer-history generation remain independent audit work. The whole
planner is not yet fully GPU controlled.

## Retrieved result and loading instructions

H100 Result SHA-256:
`97d1f351bf6ad4907ddce887aa47d1fa974f587ab270374f4bb46a5898491d48`

Full result path:
`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\gpu-regeneration-v799\h100-best\Result.txt`

[Open the local visualiser](http://127.0.0.1:4173/?dataset=gtoc12-regeneration-v799&epoch=69807&preset=oblique&z=1).
To start it after a reset, paste into PowerShell:

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process node -ArgumentList @('scripts/serve.mjs','--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-regeneration-v799&epoch=69807&preset=oblique&z=1'
```

[Evidence, raw archives and saved-data auditor](../results/lambda/2026-09-09/gpu-regeneration-v799/)
contain every candidate, native attempt, rejected fleet, accepted trajectory,
frozen source, runtime libraries and final GPU budget sweep. Each GPU archive
contains 1,127 hashed payload files; every file was checked after transfer.
