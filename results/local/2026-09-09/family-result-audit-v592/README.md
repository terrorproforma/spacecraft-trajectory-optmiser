# Family 32 result audit — 2026-09-09

The richer GPU pilot reproduced the historical first ship and found a new cooperative
third ship. It did **not** produce a route that improves the retained 23-ship fleet.
This conclusion has an inexpensive, conservative subset certificate; re-flying the
26 persisted routes through SCvx would not change their mining score at fixed epochs.

| Result | Ships | Physical haul, kg | Fixed-bonus score, weighted kg |
| --- | ---: | ---: | ---: |
| Retained v11, independent verification | 23 | 14,047.802874744 | 12,805.194102489 |
| RTX 5090 v588 | 3 | 1,587.268993839 | 1,420.909408490 |
| H100 v589 | 3 | 1,587.268993840 | 1,420.909408491 |

Both new fleets pass the independent and official checkers. Both hosts produced the
same launch, return, deployment and collection schedules. The local report records
932.323 seconds and H100 1,015.286 seconds. These runs overlapped CPU development work
and are score-quality pilots, not controlled hardware timing comparisons.

## Novelty and the lower second-ship haul

| Pilot ship | Raw kg | Weighted kg | Finding |
| --- | ---: | ---: | --- |
| 1 | 641.067762 | 569.053729 | Same schedule as historical `cluster_fleet_v10_control`; retained ship 6 already returns 644.353183 raw / 571.914816 weighted kg from the same nine asteroids. |
| 2 | 533.470226 | 490.111274 | New schedule, same seven collected asteroids as historical 535.112936 raw / 491.608212 weighted kg route; now deploys miners on 49895 and 12043 for ship 3. |
| 3 | 412.731006 | 361.744405 | New collected-asteroid set among the 3,226 locally available historical route summaries; collects those two foreign miners. |

The second ship's seven mining intervals total exactly **60 fewer days**, explaining
the **1.642710472279 kg** decrease through the unchanged 10 kg/year mining law. The
historical campaign ran out of its family time budget after ship 2. The GPU pilot had
time for a return sweep, further retiming, extra deployments, a third ship and orphan
repair. Two proposed higher-haul retimings failed Earth-return certification. Its
orphan on 16387 was later dropped. The records demonstrate a different search path,
not a GPU numerical loss. A later historical joint-itinerary route already returns
555.263518 raw kg on the second ship's seven collected asteroids.

`audit.json` contains matching schedules, closest historical alternatives and every
incumbent conflict. Novelty means absent from those local archives, not proof that no
such route exists elsewhere.

## Why these routes cannot raise the retained score

The ship-count rule is `N <= 2 exp(0.004 × mean physical kg/ship)`. It acts on physical
haul, while ranking acts on bonus-weighted returned mass.

The retained 23-ship fleet is only **4.307421 physical kg** above the minimum total
allowed for 23 ships. A 24-ship fleet needs 14,909.439899 physical kg in total. Keeping
the current 23 routes unchanged therefore requires a new ship returning at least
**861.637024 kg**. Alternatively, existing routes must improve to create that margin.

`subset-certificate.json` proves no gain for the incumbent plus the three persisted
pilot primary routes:

- Pilot ship 1 is dominated in both physical and weighted haul by retained ship 6,
  with the same asteroid footprint and no foreign dependencies.
- Ship 2 alone can be selected; uncollected deployments are allowed. At 23 ships its
  best possible raw total is 14,010.513347 kg, below the required 14,043.495453 kg.
  At 24 ships it also fails. At 22 or fewer ships, the weighted-score upper bound is
  at most 12,264.121730 kg, below the incumbent.
- Ship 3 requires ship 2 at matching deploy epochs. This pair cannot satisfy the raw
  mass rule at 23, 24 or 25 ships, even keeping the heaviest possible incumbent
  subset. At 22 or fewer ships its weighted-score upper bound is at most
  12,103.373315 kg, again below the incumbent.

The certificate uses exact dependency checks and conservative cardinality bounds.
No GPU jobs, optimization solves or trajectory propagation were run. It covers the
23 retained routes plus these primary schedules, not the wider historical archive
or unpersisted pilot variants.

## Persistence issue and bounded future union

The pilot masters received eight columns: three primary ships, four standalone
certified variants and one three-ship bundle. Only the three primary
`route_summary.json` files were written. The current `cmd_cluster_fleet` and
`cmd_fleet_master` artifact loops omit variants. This was reported to the agent
fixing CLI persistence. Those four variants cannot be recovered from their aggregate
master scores alone.

Do not run an expensive archive union for these three primaries: the bound above
already excludes improvement. For a future candidate whose bounds remain open,
use a separate candidate output and only the retained routes plus the novel family:

1. Copy the 23 recovered `incumbent-sources/ship-NN.json` files into a dedicated
   `incumbent-archive/ship_NN/route_summary.json` tree. The existing filenames alone
   are not discoverable by `discover_archives`.
2. Use the following command under the existing GPU lock and validated CUDA/data
   environment (source paths are relative to the repository):

```bash
python -m spacepdhcg gtoc12 fleet-master \
  --run-id candidate_union \
  --output build/performance/candidate-union/output \
  --source build/performance/candidate-union/incumbent-archive \
  --source build/performance/family-gpu-v588/output/clusters/family_0032 \
  --workers 1 --max-ships 25 --node-cap 2000000 --lp-node-limit 20000 \
  --scvx-iterations 40 --node-days 2 \
  --screening-backend cuda --seed-backend cuda \
  --discretisation-backend cuda --assembly-backend cuda \
  --convex-solver qoco --qoco-ruiz-iterations 0 \
  --outer-loop-backend cuda --gpu-execution graph
```

This bounds re-certification to 26 route summaries for these inputs and bounds the
master search nodes; it is not a strict elapsed-time cap. CLI `fleet-master` currently
does not expose a warm incumbent argument, so a union alone is not a no-regression
guarantee. A production runner should map the re-certified retained columns and pass
them through the existing `solve_fleet_master(..., incumbent=...)` API. In every case,
keep v11 intact and promote a candidate only after both independent and official
verification pass and its finite fixed-bonus score exceeds v11's 12,805.194102488575
weighted kg. A failed or lower-scoring candidate leaves v11 selected.

The bonus table is a local, pinned competition input and must not be added to the
published evidence archive. Its SHA-256 is recorded in the certificate.
