# Bounded asteroid substitutions in the retained fleet — v599

Prepared locally without GPU execution. This experiment changes which asteroids
are mined, preserving the validated trajectory machinery. It does not yet have a
new score or measured GPU runtime.

The immutable starting fleet is v595: 23 ships, 195 collected asteroids,
196 deployed miners, 14,051.854893909 physical kg and 12,810.135953049 fixed-bonus
weighted kg. Its exact Result SHA-256 is
`33701ef2b797f44ef2e8aa50a2dd59cb238df9aab604e7cef25ead6cbdd669e8`.

## Candidate selection and work budget

The CPU preparation attributes each ship's cargo using the independent verifier's
per-asteroid scored masses and the complete pinned bonus table. It assesses the
six weakest weighted contributions, finds nearby unused targets at both original
deployment and collection epochs, and selects three ships with the largest
nearby bonus opportunities. Each ship contributes at most four replaceable bodies
and each body at most 64 substitutes. Ship and slot interleaving gives deterministic
coverage if the soft time limit stops screening early.

| Selected ship | Verified physical kg | Verified weighted kg | Substitution cases |
| --- | ---: | ---: | ---: |
| 2 | 632.251882272 | 517.133478230 | 186 |
| 21 | 623.846680356 | 514.050168154 | 162 |
| 7 | 614.428473648 | 522.635888565 | 148 |

These are geometric opportunities, not proven reachable replacements. The best
single-body bonus ceilings are approximately 21.54, 19.86 and 17.11 weighted kg
for the three ships **if unchanged raw cargo and a feasible trajectory were
available**. They are neither additive fleet gains nor predictions of certification.

| Stage | Bound |
| --- | ---: |
| Distinct paired asteroid replacements | 496 |
| Initial complete-itinerary epoch samples | 1,488 |
| GPU neighbor queries | 24 |
| GPU DP retiming calls | 12 |
| Joint timing polish starts | 4 |
| Total joint candidate rows for this fixture, including polish | At most 4,308 |
| Defensive joint candidate cap | 6,000 |
| Whole-route refinements | At most 4 |
| Search soft budget | 180 s |
| Campaign soft budget | 1,800 s |

The three initial samples retain the original epochs, deploy the substitute ten
days earlier, or collect it ten days later. Complete forward bookkeeping and
changed incident-leg geometry run through the published native joint evaluator.
Different substituted asteroid identities use separate native calls; the three
epoch rows for one identity can share a batch. GPU neighbor ranking is checked
against the prepared pool; any CPU/GPU union mismatch is recorded and that initial
sample is skipped. GPU retiming can still test the prepared geometric choice.

Up to four distinct substituted bodies per ship receive GPU retiming, including
a geometric start if fixed-epoch screening finds no feasible route. Up to four
best feasible plans receive at most two 8-day and two 3-day joint moves. Logical Lambert
requests and DP cells are additional intermediate work, reported separately from
complete-itinerary candidate counts. The soft time limits are checked between
operations; a running solve can finish later. Original per-leg solver limits and
physics tolerances remain unchanged.

## Inventory, objective and acceptance

Each trial replaces one asteroid at its deployment **and** collection visits.
The first deployment and final Earth-return source remain fixed. Mining actions,
miner count, orphan inventory and ship count are preserved. Every other ship's
deployments and collections, plus this ship's original footprint, are excluded
from the new target pool. Merged deployment/collection camps are supported by
construction tests, although none occurs in the selected 496-case fixture.

Only the original certified route is supplied to `joint.learn`. Old measured
costs, Lambert values, inflations and bans are never relabelled to a new body.
Changed incident legs therefore receive new geometry and ordinary unmeasured
surrogate handling. Unchanged exact legs retain the existing mass-tolerance
guard. Retiming uses the published `profile_for_orders` mass guess, including
its prior-mass fallback for a new body; its forward passes correct that guess.

The verified fleet has **8.359440536 raw kg** of slack above the 23-ship threshold
of 14,043.495453373 kg. Candidates must predict a weighted gain greater than
0.5 kg while retaining that fleet raw-mass feasibility. A modest raw loss is
allowed within this budget; requiring raw cargo to rise would unnecessarily
reject valid weighted improvements. The joint search's spare-propellant term is
not reported as mining score and is not sufficient to pass this screen.

At most four distinct substitutions are refined. A route must certify, retain
the exact intended inventory and pass the raw/weighted gate before constructing
a complete fleet. Both the independent and official fleet checkers then must
pass. Only a strictly higher finite independently verified weighted score can
replace the best. Failed, missing or nonfinite scores never promote. Every trial
starts from v595 with one ship replaced; improvements from different trials are
not combined. Each verified promotion has an immutable result directory; a failed
later attempt or viewer export cannot overwrite an earlier result or mislabel its
score. Viewer export failure is recorded separately from an accepted fleet.

## Provenance and CPU validation

`source/` contains 190 files from published commit
`3091c716714c8bdec364d54c5e7357f2b5d85730`: Python implementation, pinned benchmark
metadata, project metadata and the published inflation fit. `source-sha256.json`
is verified before importing this code. Current working-tree GPU geometry and
wrappers are not imported. `inputs/` retains all 23 route summaries, the exact
fleet Result and its existing independent scoring audit, with individual hashes.

Execution requires the previously validated local binaries:

- v596 CUDA core: `86d7fdc952e82f7e5557c1651a1722900d83cdd98d8a709a4b6274fd83af4671`.
- QOCO540: `0cc27a1d8bde1edd74f7ef00e04ec64c2d6c0f8fe526a1d94b04d509a74b3315`.

`test_driver.py` covers the acceptance rules and adverse inventory cases using
CPU execution with native-library loading blocked. `cpu_audit.py` rebuilds the
candidate list deterministically and checks every seed against the actual fleet,
including miner counts, mission bounds, mining stays, exclusion and the work cap.
`cpu-audit.json` records no attempted native-library loads. `validation.json` and
its logs identify the final tested driver/test hashes. These checks validate
construction and bookkeeping; they do not validate the future GPU run's physics.

## Local recipe for the root agent after review

The preparation already exists. Print the precise pinned command and environment
without starting it:

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser /home/angus/worktrees/spacepdhcg-literature-venv/bin/python build/performance/asteroid-substitution-v599/launch.py
```

To start the reviewed job, the root agent can add `--execute`:

```powershell
wsl -d Ubuntu-22.04 --cd /mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser /home/angus/worktrees/spacepdhcg-literature-venv/bin/python build/performance/asteroid-substitution-v599/launch.py --execute
```

The launcher validates `ready-manifest.json`, starts one tracked child, and writes
`launch.json` with its PID and `run.log`. The driver acquires
`/home/angus/.spacepdhcg-gpu.lock` nonblocking before any CUDA work; an occupied
lock causes a recorded failure, not overlap or job termination. It writes to a
fresh `output/` directory. Add `--proxy-only` for screening without refinements.
No GPU job or remote transfer was performed during this preparation.

## Relationship to the existing roadmap

The published milestones call for multi-fidelity arc evaluation, analytical
screening, beam search or column generation, fixed-sequence refinement and final
route certification in the route-layer / multi-destination stages. This pilot
implements a small search neighborhood within that programme: new asteroid
identities produce new route columns, screened cheaply before expensive physical
refinement. It reuses the current verified solver core. It differs from v597,
which retained every asteroid identity and changed only order and timing.

This is progress toward fleet solution quality, not a new solver architecture or
evidence of state-of-the-art fleet quality. It fixes deployment/collection positions,
preserves Earth endpoints, samples only three ships and does not exchange asteroids
between ships or combine successful substitutions. Larger topology search, joint
fleet selection, convergence reliability and broader comparable benchmarks remain
separate roadmap work. A new literature review may inform those next neighborhoods;
it need not interrupt this bounded experiment with the existing solver.
