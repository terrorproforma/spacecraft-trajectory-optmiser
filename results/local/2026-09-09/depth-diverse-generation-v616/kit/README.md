# v616: capture a competitive-depth generated pool before changing selection

This is prepared, not executed. It performs one finite candidate-generation run and zero
low-thrust refinements, full-route reruns, or fleet promotions. No production source is edited.

The saved-pool audit found 333 `search.json` exports, including 41 with different depths on the
same Earth leg. They export only top candidates and none saves depth 9 or 10. The complete
v379 pool contains 198 genuinely generated plans of depths 1–8; running the frozen
`refine_candidates` selects indices 0 and 11. Its best 561.670089 raw kg cannot replace even
the lightest incumbent ship without falling 0.730224 kg below the 23-ship raw mass floor.
These are measured selection losses, not evidence that useful certified routes were discarded.
The exact old pool, actual selector, export inventory/hashes and attrition are preserved here.
Archived certification flags were not recomputed during this CPU preparation.

The next missing evidence is a fresh full generated pool. The input is only the exact measured
Earth leg of retained ship 23: Earth departure MJD 64403, arrival at asteroid 30805 after
570 days, measured propellant 419.324385 kg. Its historical `phasing_r1.75` family is reproduced
from the 10,612-object filtered catalogue: 62 members remain after excluding all other ships'
asteroid footprints. The archived nine-miner route is **not** injected as a generated candidate.

Generation uses the unchanged f8b2ac7a source, local v702 native core and frozen catalogue/bonus
tables listed in `profile.json`, `source-sha256.json` and `generation-input.json`. It retains
the numerical authority, mass, mining and mission-window gates. The beam has width 24,
two variants per deployed set, maximum depth 10, and CUDA collection-tour scoring. Because
this run has one Earth seed, its `max_per_first` cap is explicitly 24 instead of the default 8.
Harvest substitution, extra Earth-leg searches, SCvx, retiming and route refinements are disabled.
The flat hop inflation is the frozen cluster-search default; no new fit is trained or implied.

CUDA handles Lambert/geometry, neighbor operators, collection selection and collection DP.
Python still constructs inputs and orchestrates the beam and forward mass checks. A missing
CUDA DP path fails the run; it cannot fall back silently. Precision is unchanged: FP64 native
inputs/arithmetic with the existing FP32 resident collection delta-v tables. All returned
values remain screening estimates. `beam_score_including_propellant_penalty` is distinct from
weighted cargo and raw cargo. The expected full-fleet raw/weighted values are clearly marked
as proxies; neither they nor a generated plan constitute a newly verified fleet score.

Hard limits are one generation, one Earth seed, 62 IDs, depth 10, 193 expansion calls,
217 completion attempts, 192 chain-tour calls and 1,056 native collection-DP passes. These
counts bound high-level operations, not individual CUDA kernels or trajectory solutions.
The stage caps stop before an extra call. The driver checks its 120-second deadline between
operations; the supervisor additionally bounds its owned process group to 150 seconds plus
at most 10 seconds of termination grace, preserving partial evidence without retrying. An
interrupted native call can leave incomplete counts, which the timeout sidecar labels explicitly.
There is one launch marker and an exclusive shared GPU lock. A busy lock or observed compute
process consumes the attempted launch marker; no automatic retry or duplicate output is allowed.

Every completion and failed completion is appended to `generation-events.jsonl`; every successful
completion also survives in `completed-plans.jsonl`. If the generator returns, its entire ordered
pool is saved to `candidate-pool.json` **before** the frozen default selector is evaluated.
`shortlist-attrition.json` reports selected/discarded indices and actual depth contrasts, including
closed candidates that preserve the raw fleet floor and improve the frozen weighted proxy.
Internal beam pruning is still part of generation; “full pool” means all returned completed
candidates, not every uncompleted internal path. An outer timeout may leave only the journals.
The two later refinement arms must use this same immutable pool, baseline, precision, native
core and total full-route budget; fixed cargo and both full-fleet checkers remain mandatory.
Those arms will be prepared only after inspecting this output. No depth-diverse policy is
implemented or enabled by this kit.

`validate_prepare.py` runs CPU tests with native loading forbidden, checks required exported
symbols without loading the library, writes exact logs, and freezes `ready-manifest.json`.
The profile's older v707 validation is inherited provenance, not a new measurement by this kit.
Required storage is a few MB of journals/candidates in addition to the frozen source and baseline;
the finite CUDA/DP tables retain their existing cache caps. Actual Lambert branch requests are
recorded at run time; no new speedup or peak-memory bound is asserted.

From the repository root, inspect the launch command without starting work:

```powershell
wsl -d Ubuntu-22.04 -- /home/angus/worktrees/spacepdhcg-literature-venv/bin/python -B build/performance/depth-diverse-generation-v616/launch.py
```

After root reviews the ready manifest and actual GPU availability, the one authorized launch is:

```powershell
wsl -d Ubuntu-22.04 -- /home/angus/worktrees/spacepdhcg-literature-venv/bin/python -B build/performance/depth-diverse-generation-v616/launch.py --execute --wall-seconds 120
```

Keep that foreground tool session until the supervisor exits. Do not relaunch after an observation
timeout. The incumbent stays at 12,810.135953 weighted kg and 14,051.854894 raw kg unless a future
complete fleet independently passes both checkers and the objective/ship-count gates.
