# v637 unused-family search: no eligible replacement or growth

The bounded RTX 5090 campaign completed successfully and produced **923 distinct
physical route prescriptions** across four previously unused families. None
meets the current fleet's raw-mass rule as a replacement, an additional ship,
two replacements, or two new routes replacing one incumbent. These are proxy
search results; no trajectory was refined or certified and the verified score
remains **13,526.961241 weighted kg / 14,915.044490 raw kg, 24 ships, 208 asteroids**.
The retained fleet is [v633](../fleet-addition-v633/README.md), Result SHA
`1f420928bbef8da91e3a6f8b74d3bf00039c78d4257215c103a484bc55bb22da`.

| Family | Plans | Best raw / weighted kg | Deployments in best | Deepest closed / attempted | Search seconds |
|---|---:|---:|---:|---:|---:|
| 2 | 147 | 522.381930 | 8 | 8 / 10 | 4.220 |
| 3 | 282 | 483.778234 | 7 | 8 / 9 | 10.182 |
| 6 | 122 | 422.176591 | 7 | 7 / 8 | 4.700 |
| 8 | 372 | 452.566735 | 7 | 8 / 10 | 13.039 |

The best raw and best weighted candidate are the same in each family. Every
family is disjoint from the current fleet and from the other tested families.
Within each family its best point dominates all other retained points in both
raw and weighted cargo. Four Pareto representatives therefore cover all
**298,764** cross-family plan pairs. Twenty sorted-removal queries over the
24 single removals and 276 two-ship removals give:

| Composition | Resulting ships | Best possible raw margin, kg |
|---|---:|---:|
| One replacement | 24 | −42.773232 |
| Add one route | 25 | −348.377607 |
| Two routes replace two ships | 24 | −145.853314 |
| Two routes replace one ship | 25 | −435.359126 |

The strongest pair is family 2 index 0 plus family 3 index 2: **1,006.160164 kg**.
The single best route has an unconstrained weighted gain of **19.125590 kg** over
ship 24, but loses **101.656400 raw kg** against that ship, far beyond the fleet's
**5.604591 kg** raw slack. The shared fleet-budget test retained that possibility
for assessment and correctly rejected it. An obsolete per-ship positive-raw
filter did not cause the negative outcome.

The run exercised eight Earth screens: **46,080 paired rows / 92,160 fresh
Lambert branch requests**. It retained all 2,213 returned options, applied the
existing prescreen to 289 eligible options, selected families 2/3/6/8 before
search, and used seven new Earth seeds. The four searches used **43,423,020**
fresh branch requests, making **43,515,180** including screens. They expanded
1,220 parent partials, produced 1,259,796 native valid expansion children and
admitted 1,241 children. Native completion telemetry separately counts 3,633
proxy-cost requests; these internal requests are not distinct mission solutions.
Cached return branches are not added to the fresh-branch count.

The four searches took **32.141645 seconds** in total; the complete worker took
**33.864663 seconds**, and its supervisor **35.889309 seconds**. There were zero
refinements, trajectory certificates or fleet checks. The supervisor exited
normally, reaped its child, retained the lock through cleanup, and recorded an
empty GPU inventory. This new-family workload is not a paired speedup benchmark.

The saved completion failures explain a real limitation of this particular
search, without establishing physical infeasibility. All **121 retained
attempts at deployment depths 9 and 10 failed completion**. There are 321
recorded failures in total. Nonexclusive counts of records containing a reason
are: no collection hop 292, final proxy mass below dry mass plus cargo 184,
no return 150, no DP tour 137, and leg authority 40. These are failures of the
sampled schedules and proxy models. Four failed first-level completions have no
stored reason. Per-child causes of earlier native pruning were not retained.
No search exhausted its time budget.

The evidence supports two narrow conclusions: this unused-family/seed selection
did broaden physical requests, but its completed routes carry too little cargo;
and the deeper admitted prefixes encounter collection/return completion gates.
It cannot establish whether unsaved discarded prefixes would produce useful
routes. A useful next discrimination would retain complete requests and native
readbacks for a bounded set of deeper mass-rejected completions, using the
existing completion-capture hook, before considering fixed-cargo certification.
Widening these same four pools without identifying that loss would not test a
new explanation. No such follow-up was executed here.

The prelaunch README and ready map remain unchanged inside `evidence.zip`.
They describe historical preparation, not additional pending calls. The exact
193-file host and native source identities are linked to the existing
[v846 archive](../../../lambda/2026-09-09/gpu-shared-return-options-v846/README.md);
native library SHA is
`75a9640709216bdb5a381b9e7f895692489ca8d4e6cb8606103ee39c6a3dc15c`.
Nine CPU construction checks and Ruff passed before the one supervised launch.
The package preserves preparation observations, all 70 terminal output files,
the full candidate/failure pools and eight raw screen NPZs. No native binaries
are copied into this package.

`saved-audit.json` contains exact Decimal-65 fleet arithmetic, candidate hashes,
depth/reason histograms and every Pareto/removal query. The stdlib auditor checks
all 923 cargo ledgers and all eight raw screen arrays against saved JSON, and
parses the exact current Result. It performs no orbital propagation or native
calls. `package-map.json` binds included files and existing source dependencies.

From the repository root, use the SHA of this package's `index.json` reported in
the commit/handoff (do not substitute a newly calculated hash as a trust anchor):

```powershell
python results/local/2026-09-09/route-family-v637/verify_package.py --index-sha256 <published-index-sha256> --dependencies --roundtrip
```

Without `--dependencies`, verification checks the sealed local archive and its
binding map. `--roundtrip` additionally replays only saved scalar arithmetic
using the published v633 Result. It does not rerun the search, clustering,
trajectory checkers or historical native tests.
