# GPU asteroid-substitution search — v599

The bounded local RTX 5090 run completed and retained the verified v595 fleet:
**12,810.135953 weighted kg / 14,051.854894 physical kg**, 23 ships. Both freshly
run full-fleet checkers pass. No candidate qualified for physical refinement;
this experiment produced no score improvement.

| Measured work | Result |
| --- | ---: |
| Ships explored | 2, 21, 7 |
| Distinct paired asteroid substitutions | 496 |
| Initial whole-itinerary epoch samples | 1,488 |
| GPU neighbor queries | 24 |
| GPU retiming driver calls | 12 |
| Internal DP / forward calls | 93 / 93 |
| Feasible retimed surrogate plans / polished plans | 2 / 2 |
| Joint candidate rows / batches | 2,778 / 506 |
| Logical Lambert branch requests | 6,430,118 |
| Search stage | 2.044482 s |
| Initial substitution screening | 1.402124 s |
| Baseline independent and official checks | 21.187483 s |
| Campaign timer | 23.409579 s |
| Complete physical route refinements / promotions | 0 / 0 |

These are intermediate search counts, not certified trajectories per second.
Timing starts after initial imports, source/input validation, loading and parsing;
the search stage begins after entering the GPU context. The campaign includes
baseline CPU verification. Logical branch requests can repeat cached geometry.
There is no baseline comparison or whole-campaign speedup claim.

## What failed and what the diagnosis establishes

All 496 prepared cases were screened, with no CPU/GPU neighbor-union mismatch.
Of 1,488 initial epoch samples, 1,487 failed the empirical thrust-authority screen
and one failed the final surrogate mass budget. Ten of twelve retimings failed
their mass budget. The two feasible retimings improved slightly through eight
joint timing moves, but even the better alternative loses **54.476775 weighted
kg and 72.991102 raw kg** relative to its original ship. None meets the weighted
gain and fleet raw-mass gate, so no expensive SCvx refinement was attempted.

A separate **CPU-only** replay checked the exact frozen evaluator, without
changing arithmetic or thresholds or loading native libraries. All three original
certified routes pass the identical proxy settings, reusing all 16/18/17 measured
flight legs. All 1,488 CPU sample outcomes match their recorded GPU reasons.
Every authority failure is on a changed incident leg: 1,421 deployment hops and
66 collection hops. None arises on an unchanged leg losing its measured-mass
allowance. No measured cost was accidentally attached to a replacement endpoint.

The empirical authority limit is 0.55; first-failure ratios range from 0.5524 to
11.899, with median 1.5553. However, **419 lie at or below a unit authority ratio**.
These are conservative surrogate rejections, not physical infeasibility proofs.
No rejected replacement received SCvx/independent physical certification here,
so the experiment cannot establish the screen's false-negative rate.

The next distinct hypothesis is to select and seed replacements by their four
incident-leg windows, bottleneck authority and downstream mass budget, then
retime the least-excess candidates. This directly tests reachability around the
existing chain rather than ranking primarily by nearby orbital elements and
bonus gain. It does not require changing acceptance gates or rerunning this
same sampled neighborhood unchanged.

## Objective and retained evidence

Each trial substitutes an asteroid at both deployment and collection, preserves
the Earth endpoints/miner inventory/ship count, and excludes every other ship's
asteroid footprint. The fleet has 8.359441 raw kg of ship-count slack. A weighted
gain could use that slack; the driver does not incorrectly require raw mass to
increase. The four saved surrogate plans have independently recomputed raw and
weighted cargo, unchanged inventories and no eligible improvement.

The unchanged baseline Result is preserved in the source archive's inputs and
already published in [v595](../orphan-recovery-v595/Result.txt). No new viewer
dataset was added. `report.json` contains the complete run and both checker
summaries; `raw.tar.gz` retains every output file, including all four proxy plans
and the complete per-sample failures. `audit/result-audit.json` reconciles the
work counts and score arithmetic. `audit/authority-replay.json` retains every
CPU replay and inspected first-failure frame.

## Exact source and reproduction

The run uses 190 frozen published files at
`3091c716714c8bdec364d54c5e7357f2b5d85730`, the v596 CUDA core
`86d7fdc952e82f7e5557c1651a1722900d83cdd98d8a709a4b6274fd83af4671`
and QOCO540
`0cc27a1d8bde1edd74f7ef00e04ec64c2d6c0f8fe526a1d94b04d509a74b3315`.
The exact tested driver SHA-256 is
`9e668824fc4542da103777313c23694bae49a628cd5c5f266805cb1f537101a0`.
Joint batch/device selection were enabled; numerical and physics tolerances
were unchanged. Python orchestration and CPU verification remain.

`execution/` contains the exact driver, preparation, 61-test CPU validation,
candidate plan and original 237-file ready manifest. `source.tar.gz` stores the
frozen source and inputs once, with paths rooted at `execution/`; extracting it
reconstructs that complete preparation. Source and archive member hashes were
verified during packaging. The top-level `sha256.json` indexes publication bytes.
The original foreground wrapper is retained under `provenance/`, alongside the
publisher audit. `launch.json` records PID 399, flags and native library paths.

For a deliberate reproduction, extract `source.tar.gz` into this directory,
provide the pinned catalogue/bonus data and recorded compatible native libraries,
then inspect `execution/launch.py` without `--execute`. The saved launcher can
start a reviewed reproduction with `--execute --output /absolute/fresh/output`;
it requires the shared GPU lock and refuses existing launch/output files.
Do not treat a repeated identical run as a new search hypothesis.
