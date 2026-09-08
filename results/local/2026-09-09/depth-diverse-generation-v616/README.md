# Depth-diverse generation v616 — no eligible replacement

One local RTX 5090 generation completed normally in **18.544582 seconds**, using the fixed
ship 23 Earth leg and its 62-member family. It returned **167 closed proxy candidates**:
24 each at depths 2–7, 21 at depth 8, two at depth 9, and none at depth 10. The search reached
depth 10 and attempted four completions there. All 167 exact plans are distinct; their
chronological deployment/collection orders comprise 92 distinct topologies.

Actual work was **193 expansions, 197 completion attempts, 172 chain-tour requests,
760 CUDA collection-DP passes and 8,556,740 Lambert direction requests**. Every journal
start/finish reconciles with the report; no operation remained active. These are operator
counts, not certified trajectory solutions. There were zero low-thrust solves, refinements,
full-fleet checker invocations or promotions. The retained fleet remains **12,810.135953
weighted kg / 14,051.854894 raw kg across 23 ships**; its earlier certificates were not rerun here.

All candidates use the same Earth leg. The actual frozen default shortlist selects index 0
and discards the other 166. Index 0 predicts 606.981520 raw kg / 557.866680 weighted kg:
its +15.745493 weighted-kg gain cannot offset its 38.758972-kg shortfall against the required
raw fleet floor. The best raw eight-miner candidate carries 609.445585 kg; the best nine-miner
candidate carries 563.449692 kg. Replacing ship 23 requires at least **645.740491 raw kg** and
an improvement over its **542.121187 weighted kg**. No candidate meets both gates. Consequently
this output does not yet justify a depth-diverse refinement comparison or extra refinement budget.

## Why this did not reproduce the incumbent

The family and Earth seed are correct, but the generator does not contain the incumbent's exact
timing choices. **All eight deployment TOFs are off its grid**: for example, the first hop is
286 days, while the closest generation choice is 300 days. Seven of nine collection TOFs are
outside the collection DP grid. The actual retained beam contains the original
30805→40868 prefix at 300 and 360 days, but no prefix beyond its second deployment and no
completed candidate with the original nine-asteroid footprint. The saved trace does not identify
which unrecorded child prune removed a particular off-grid analogue; no such cause is invented.

**Cargo was not shrunk.** Every returned plan carries the full mining-rate cargo implied by its
saved deployment and collection epochs. The completed depth-9 candidates simply have shorter
aggregate mining periods. Among failed completions, all 22 at depth 9 and all four at depth 10
record `mass_below_dry_plus_collected` for both collection DP pricing passes. Their heuristic
alternatives fail at collection-hop or return screening. Full failure strings and epochs survive.

There is also a demonstrated proxy admission problem. Two CPU-only calls to the exact frozen
`RouteSearch._finish` used the incumbent's saved epochs, Lambert values and fixed
654.099932-kg cargo. They bypassed generation solely for diagnosis and emitted no candidate.
Both return `mass_below_dry_plus_collected` at the final mass check in frozen
`search.py:2333–2335`, after the leg authority checks. The separate saved-prefix scalar audit
also passes every deployment reserve test, with about 85.7 kg of reserve margin at depth nine.

The deployment proxy leaves **1465.438985 kg**, versus **1496.802328 kg measured** after the last
deployment, a 31.363342-kg underestimate. Substituting the measured prefix mass still does not
make the current completion proxy accept the known certified schedule. The completion builder
prices return inflation using mass after deployments plus all cargo (`_plan_from_tour`,
`search.py:2085–2087` and `2119–2125`), before accounting for collection burns. That is about
2119.539 kg in the proxy case, versus the recorded **1411.161887-kg** actual return departure
mass. The corresponding current return inflation is **1.462308**, while the retained calibrated
leg stores **1.299799**. This identifies a concrete cost-model difference; it does not establish
the cause of every rejected generated route. The earlier scalar diagnostic evaluated return
inflation at a sequential mass and found a −23.796634-kg proxy final margin; the exact `_finish`
controls use the builder's mass convention and are the authoritative admission checks.

The next bounded milestone is **incumbent schedule and footprint reproduction with stage-by-stage
prune accounting**: test the exact fixed-cargo incumbent as an explicit admission control, retain
every expansion/filter/collection/mass rejection, and distinguish unavailable grid choices from
proxy mass bias before widening generation or changing shortlisting. An archived control must
remain labelled as a control, never passed off as a newly generated alternative. Numerical
physics tolerances and both final fleet checkers must remain unchanged. No additional run was
performed or authorized by this package. Failure here is not proof of physical infeasibility
or exhaustive search of the family.

## Evidence and reconstruction

- `output/` preserves all seven raw files, including the full ordered pool and append-only work
  and completion journals. They were saved before the default shortlist was evaluated.
- `kit/` preserves the exact 220-file preparation manifest, driver, supervisor, 23 passing CPU
  tests, Ruff logs, and the historical 333-export/198-plan audit. The first validation attempt
  had a lint-only error; its passing test log and failure are retained alongside the final pass.
- `source.tar.gz` contains all 190 frozen source files from commit
  `f8b2ac7aeba9e29ea90f5814060bb8cfe5db22d7`; `inputs.tar.gz` contains all six exact inputs.
  Extract them into `kit/source/` and `kit/inputs/` to reconstruct the prepared hashes.
- `kit/profile.json` pins local v702 core SHA256 `65f1335e…` and v683 QOCO SHA256 `5b1b1a04…`.
  QOCO is unused because this run forbids refinement. No native binaries are included. The
  older validation in that profile is inherited provenance, not a new H100 measurement.
- `execution/` records the single supervisor PID310/child387, device, environment, lock check,
  successful exit and 150-second outer timeout with owned-child reaping. It did not time out.
- `audit/report-v2.json`, `diagnosis-v2.json` and `finish-control.json` are authoritative. The
  first audit/diagnosis files are retained under `audit/superseded/`: the corrected version
  reconstructs route order from event epochs instead of JSON map order. Work counts and eligibility
  were unchanged. Audit and diagnostic scripts performed no native loading or GPU calls.
- `archive-audit.json` verifies archive members and reconstructs every prepared hash;
  `sha256.json` indexes every publication file except itself. `.gitattributes` preserves bytes.

CUDA performed Lambert/geometry and collection DP; Python still orchestrated the beam and
constructed inputs. Precision and acceptance gates were unchanged. No speedup comparison,
certified-solutions-per-second result or new leaderboard score is claimed.
