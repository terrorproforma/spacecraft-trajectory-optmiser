# Prepared experiment: improve the new nine-collect ship 15

Status: prepared and CPU construction-tested; **no GPU run has been launched**.
This searches a new structural neighborhood of the promoted v595 incumbent. It is
an experiment with an uncertain outcome, not an additional score improvement.

The starting 23-ship fleet delivers **14,051.854893909 physical kg** and scores
**12,810.135953049 fixed-bonus kg**, averaging **610.950212779 physical kg/ship**.
Its improved ship 15 delivers 609.691991786 physical kg and has 8.954567419 kg of
propellant remaining after unloading. The preceding version had 4.979312265 kg.
Recovering asteroid 19102 increased the fleet's raw haul by 4.052019165 kg; this
experiment starts from that gain and preserves it unless a better fleet verifies.

The new collection order is:

```text
19102 -> 6757 -> 51831 -> 49856 -> 3299 -> 5162 -> 23153 -> 35473 -> 13077
```

Keep the successful 19102 mining camp first and the existing Earth-return source
13077 last. Mutate the seven interior collectors with pair swaps, segment reversals
and single-collector relocations. Deduplication leaves **61 distinct new orders**.
The nine deployment IDs and deployment order stay fixed, all nine own miners remain
collected, and no other ship shares this footprint. The 18 physical transfer legs
and the 19102 deploy/collect camp are retained. Epochs are reoptimized.

Changing order alone does not increase total mining time at fixed time slots. Its
potential value is cheaper transfers that permit earlier deployment or later
collection after retiming. The remaining propellant margin is evidence of some
physical headroom, not a prediction of useful gain: every new order could be worse
or fail the surrogate mass budget. The campaign cannot establish that a failed
surrogate order is physically impossible.

## Coverage and distinction from other campaigns

`coverage.json` records a read-only scan of 3,738 locally unpacked route summaries,
3,553 certified. Four certified routes share this deployment set; **none uses any
of the 61 proposed collection orders**. The v595 insertion campaign changed where
19102 was collected while retaining the old eight collectors' relative order. The
v229 scans reoptimized epochs at fixed order. This experiment changes the interior
order of the newly certified nine-collect route.

This is not a claim that collection permutations have never been considered:
`collectdp.py` supports Held-Karp collection pricing, which implicitly permits
permutations under its then-current deployment epochs, grid and cost model. The
new work uses the newly flown v595 schedule and its measured leg calibration. The
audit does not enumerate transient DP states or packed-only archives.

The inspected v636/v637/v639 scripts replay the original v595 orphan driver to
compare native joint device selection, including an H100 copy and a baseline retry.
They do not run this interior-order experiment. Their hashes are in `coverage.json`.
The shared GPU lock still prevents this experiment from overlapping another run.

## Bounded recipe

Run with the caller's already validated CUDA core, GPU QOCO, cuDSS library paths and
pinned GTOC12 data. `--repo` selects the frozen Python source tree; `--inputs` selects
the separate immutable fixture directory. Both are needed when moving to H100.
The device winner-selection flag and native libraries remain the caller's choice.
Native joint candidate evaluation must be enabled for the default polishing stage.

```bash
export SPACEPDHCG_TEST_GTOC12_JOINT_BATCH=1
python build/performance/next-score-v597/run.py \
  --repo /path/to/validated/frozen/repo \
  --inputs /path/to/next-score-v597/inputs \
  --output /path/to/new-unique-output-directory \
  --proxy-seconds 120 --retime-fraction 0.7 \
  --joint-seconds 6 --joint-candidates 6 \
  --max-certifications 4 --wall-seconds 1800 \
  --gpu-execution graph
```

1. Freshly verify the full retained 23-ship fleet with the independent and official
   checkers before doing any search.
2. Spend up to 84 of the 120 proxy seconds retiming the new orders on the 15-day
   GPU lattice. Calibrate from the actual flown v595 legs, using the existing
   cluster search/retiming settings and their unchanged margin. Reordered mass
   profiles use the production `profile_for_orders` helper.
3. Polish at most six feasible seeds with native joint evaluation, ranked by actual
   fixed-bonus proxy haul; up to six seconds per seed, within the same 120-second
   total proxy deadline. Infeasible seeds cannot be rescued by the existing joint
   optimizer, so the retimer supplies feasible starting schedules first. Both
   retimed and polished candidates remain eligible; polishing cannot erase a
   better retimed candidate.
4. Re-fly at most four unique whole routes whose proxies improve both physical
   and weighted haul by more than 0.5 kg. This screening threshold is configurable
   down to zero. Native SCvx/QOCO uses the same 40 iterations, 2-day nodes, GPU
   seed/discretization/assembly/outer loop and default physics tolerances as v595.
5. Replace only ship 15 in a separate full-fleet file. Promotion requires both
   full-fleet checkers to pass, a finite weighted score strictly above the current
   verified best, and no decrease in independently verified physical haul. The
   exact deployment and collection footprint is checked again after refinement.
   Later worse candidates cannot overwrite the best, and the input Result hash
   is checked again at completion.

The time limits are checked between operations, including within joint polishing;
an active retiming or native refinement call can finish after its soft deadline.
Overruns and whether all 61 orders were scanned are explicit in `report.json`.
The 120-second proxy timer excludes full-fleet CPU verification and route refinement.
The separate 1,800-second total soft wall limit includes those steps. Actual native
calls and their status/iterations/timing appear in `native-solves.jsonl`; candidate
orders, DP calls, joint evaluations, and certified routes are distinct work counts.
Python orchestration and both independent checkers remain on the CPU.

## Inputs and preparation checks

`inputs/` contains the new full fleet and all 23 route summaries. The ship 15 summary
comes from the promoted v595 attempt 1; the other 22 are unchanged retained inputs.
`inputs/provenance.json` records origins and `inputs/sha256.json` hashes every input.

| Input | SHA-256 |
| --- | --- |
| New v595 `Result.txt` | `33701ef2b797f44ef2e8aa50a2dd59cb238df9aab604e7cef25ead6cbdd669e8` |
| New ship 15 summary | `17bfe449576e6557a2d2cbb48e9c36b45d860e2e3f72e69954de4a0d44e167f7` |

The standard-library `--plan-only` run wrote all 61 cases to
`prepared-orders/orders.json`. Four CPU tests passed in WSL: all order/footprint/
camp/leg-count invariants, exact summary-to-Result event inventory, rejection of the
old eight-collect input, and rejection of an asteroid shared with another ship.
No end-to-end GPU result is claimed for this new driver yet.

```bash
python build/performance/next-score-v597/test_orders.py
python build/performance/next-score-v597/run.py --plan-only \
  --output /path/to/fresh-plan-only-directory
```

Successful output would be `best/Result.txt` and `best/viewer/manifest.json`.
If none qualifies, `report.json` retains the new v595 incumbent and records the
specific failed neighborhood, without calling repeated computation score progress.
