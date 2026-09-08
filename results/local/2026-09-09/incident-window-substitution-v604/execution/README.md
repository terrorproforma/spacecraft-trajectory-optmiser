# v604: complete incident-window asteroid substitutions

Prepared experiment only. No GPU run, speedup, score gain or new trajectory certification is claimed by this kit.

This changes v599's failed search hypothesis: a replacement needs two viable transfer windows at deployment and two at collection, with enough downstream propellant to retain cargo. Matching orbital neighbors and a high bonus coefficient did not establish those properties. All 1,487 v599 authority failures occurred on changed legs; that was a surrogate result, not proof of physical infeasibility.

The search retains v599's exact 496 one-for-one substitutions across ships 2, 21 and 7 of the independently verified v595 fleet. It excludes the complete other-ship footprint, preserves ship/miner/visit counts, fixes Earth leg endpoints and never relabels measured old-body arcs as new-body arcs. The immutable incumbent is 14,051.854893908598 raw kg and 12,810.135953048577 fixed-bonus weighted kg, with 8.359440535672547 raw kg above the 23-ship feasibility threshold.

## Bounded work and controlled comparison

- Each substitution receives the same two-sided 5×5 deployment/collection shift grid: −30, −15, 0, +15, +30 days. Unrelated epochs stay fixed. At most 12,400 complete-itinerary GPU rows are screened; invalid structural windows are omitted before execution and all native rule checks remain unchanged.
- Both ranking policies share that exact screening evidence. The historical arm preserves the twelve actual v599 retiming choices. The experimental arm selects one candidate for each of the same twelve old-asteroid slots using complete incidence and downstream mass estimates.
- Each arm gets twelve retiming calls, with independently initialized equivalent retimers and identical 15-day settings. Arm order alternates by slot. Counts, logical geometry work, native-forward yield and objective-eligible yield are recorded separately. An interrupted or unequal-budget comparison is labeled incomplete.
- At most four feasible candidate plans receive joint epoch polishing (at most 2,820 additional rows); at most four total whole-route SCvx refinements can run. Each trial replaces one ship relative to the original v595 fleet. Trials are not combined.
- The soft screening/retiming budget is 180 seconds; total soft campaign budget is 1,800 seconds. Individual in-flight native operations finish, and overruns are reported.

The shared grid is larger than v599's 1,488 samples. A higher yield from this grid alone cannot be attributed to better ranking. The equal-retiming-budget comparison isolates shortlist policy, while shared fixed-grid yield is reported separately. Total GPU joint-row work remains capped by the CPU construction audit; logical Lambert/DP work is a separate metric.

## Ranking estimates and unchanged acceptance

The native evaluator prepares all active-row leg geometry before its forward check, even when that check stops at the first authority violation. The ranking module reads cached geometry for every changed incident leg. It never substitutes the first failure ratio for the four-leg bottleneck and never launches a hidden CPU geometry fallback for missing rows.

Incident authority ratios use each incumbent leg's reference departure mass, adjusted for changed collected cargo. A separate full-route mass projection applies the frozen exact measured-key/mass-tolerance and inflation formulas to all legs, without truncating at authority failures or trimming mined mass. The ranking bottleneck is the maximum of the four normalized authority ratios and projected propellant divided by available propellant. Weighted mining gain breaks ties. Missing geometry ranks last. Near-threshold estimates remain eligible for retiming; the 10% diagnostic label changes no acceptance threshold.

These are **uncertified ranking estimates**, computed by the Python controller from GPU geometry. They are not native forward passes, scored fleets or physically accepted trajectories. No candidate reaches refinement without an unchanged complete native forward and weighted-objective/raw-fleet-mass gate. SCvx retains the published discretization, tolerances and iteration limits. Promotion requires a strictly greater independently verified fixed-bonus weighted objective and both full-fleet checkers passing. A failed candidate cannot replace the immutable incumbent. Viewer-export failure cannot corrupt a verified promotion checkpoint.

## Exact source and recipe

The source snapshot is published commit `3091c716714c8bdec364d54c5e7357f2b5d85730`, verified through `source-sha256.json`. It uses the validated v596 CUDA core (`86d7fdc952e82f7e5557c1651a1722900d83cdd98d8a709a4b6274fd83af4671`) and QOCO540 (`0cc27a1d8bde1edd74f7ef00e04ec64c2d6c0f8fe526a1d94b04d509a74b3315`). Live production source and concurrently edited geometry code are not used.

`ready-manifest.json` freezes the complete preparation after CPU tests and construction checks. `launch.py` prints its recipe by default; `--execute` is the explicit launch action. Root must first release/inspect the shared GPU slot. The child also takes the nonblocking `/home/angus/.spacepdhcg-gpu.lock` before GPU work, and refuses existing output/launch records.

```powershell
# Print only: no GPU launch.
wsl -d Ubuntu-22.04 -- /home/angus/worktrees/spacepdhcg-literature-venv/bin/python '/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser/build/performance/incident-window-substitution-v604/launch.py'

# Root-controlled launch after reviewing the manifest and GPU process state.
wsl -d Ubuntu-22.04 -- /home/angus/worktrees/spacepdhcg-literature-venv/bin/python '/mnt/c/Users/Angus/Desktop/projects/spacecraft-trajectory-optmiser/build/performance/incident-window-substitution-v604/launch.py' --execute
```

This is the next global-route-quality experiment in the SOTA execution roadmap: it explores new asteroid combinations, instead of measuring only faster evaluation of unchanged routes. It is a bounded neighborhood, not a global optimality claim. It does not yet widen the orbital-neighbor pool, reorder multiple visits, combine substitutions across ships, relax physics gates or measure the surrogate false-negative rate.
