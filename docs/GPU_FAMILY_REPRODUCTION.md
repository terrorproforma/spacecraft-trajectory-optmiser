# Richer GPU family search and retained score

The family32 GPU pilots reproduced the strong historical first ship on both an
RTX 5090 and Lambda H100. Both final fleets pass the independent and official
GTOC12 checkers. They do not improve the retained fleet.

| Result | Ships | Raw returned kg | Bonus-weighted kg |
| --- | ---: | ---: | ---: |
| Retained v11 | 23 | 14,047.802875 | 12,805.194102 |
| RTX 5090 v588 | 3 | 1,587.268994 | 1,420.909408 |
| H100 v589 | 3 | 1,587.268994 | 1,420.909408 |

The pilot ships return 641.067762, 533.470226 and 412.731006 raw kg. Their event
schedules match across GPUs. The first duplicates a historical route; retained
ship 6 already returns 644.353183 kg from those nine asteroids. The second lost
exactly 60 aggregate mining days (1.642710 kg) relative to its historical route,
while deploying two additional miners for the third ship. Higher-haul retiming
proposals failed Earth-return certification. This is a different search path,
with unchanged physical rules, rather than evidence of a GPU arithmetic loss.

## Why this is not a score improvement

The 23-ship incumbent is only 4.307421 raw kg above the global ship-count rule's
minimum. Keeping those routes unchanged, a new 24th ship must return at least
861.637024 raw kg. The novel pilot routes cannot meet that requirement.

A conservative subset certificate also excludes improvement through replacing
incumbent routes with these three persisted primary routes, at every possible
fleet size. It uses dependency, physical-mass and weighted-objective bounds;
there is no reason to re-run expensive SCvx certification for this closed case.
It does not cover unseen historical alternatives or the pilot's lost variants.

[Detailed audit and reproducible certificate](../results/local/2026-09-09/family-result-audit-v592/README.md)
separate physical haul, weighted score, timing scope and route novelty.

## Work performed

| Work | RTX 5090 | H100 |
| --- | ---: | ---: |
| Complete runner wall time | 934.154 s | 1,016.037 s |
| Native low-thrust solve calls | 468 | 457 |
| Time inside those solve calls | 860.112 s | 907.148 s |
| In-memory master columns | 8 | 8 |
| Verified final ships | 3 | 3 |

H100 telemetry records 104,441,606 logical Lambert branch requests,
8,436,295 collection options, 1,505 collection DP passes and 1,934 retiming DP
calls. These are search work counters, not independent verified mission
solutions. The runs overlapped CPU development work and took different paths
through time-budgeted refinements. They do not establish a hardware speed ratio.

The pilots used frozen base a89ed69f plus the initial weighted cluster-incumbent
fix, stable CUDA/QOCO binaries and unchanged physics tolerances. They searched
one family with one worker and no archive dual prices; the historical campaign
used richer archive pricing. This is not an exact historical campaign replay.
CUDA handles screening, seed generation, discretisation, conic assembly and
SCvx/QOCO loops. Python orchestration and CPU fleet selection remain.

## Preserve useful routes between runs

The pilots produced three primary route columns, four standalone variants and
one three-ship bundle. Only the primary route summaries were saved. The new
artifact writer retains eligible standalone variants under `ship_NN/variants/`
for both cluster and archive-master runs. Reloading keeps the explicit
cooperative primary first, so a heavier standalone alternative cannot silently
remove its supplying role. Manifests distinguish roles and require final fleet
verification. The already-finished pilots' four missing variants cannot be
reconstructed from aggregate scores.

131 focused CPU tests pass, including complete write/discover/recertify round
trips of primary and alternative schedules; Ruff passes.
[Test commands and source hashes](../results/local/2026-09-09/archive-variant-persistence/summary.json)
are retained. [Objective-selection corrections](GTOC12_SCORE_SELECTION.md) were
published separately.

## Display the H100 result

The existing viewer now includes **H100 richer family v589**. It displays all
three ships, 22 mined asteroids and 1,525 exact propagated replay samples.
Its importer verified file hashes and 9,030 context-orbit points; maximum
Kepler disagreement was below 0.000004 km.

Open [the retrieved H100 fleet](http://127.0.0.1:4173/?dataset=gtoc12-family-v589&epoch=69807&preset=oblique&z=1).
To start the viewer if needed:

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
node scripts/serve.mjs --port=4173
# Then open:
# http://127.0.0.1:4173/?dataset=gtoc12-family-v589&epoch=69807&preset=oblique&z=1
```

Raw H100 solution:
`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\family-gpu-v589\output\fleet\Result.txt`

[H100 retrieval hashes](../results/lambda/2026-09-09/family-gpu-v589/retrieval.json)
cover 22 original files; local v588 files are retained under
`results/local/2026-09-09/family-gpu-v588`.
