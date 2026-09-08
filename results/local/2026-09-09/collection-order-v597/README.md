# Collection-order search around the improved fleet — v597

The complete local GPU experiment retained the verified v595 incumbent:
**12,810.135953 weighted kg / 14,051.854894 raw kg**, 23 ships. All 61 proposed
orders were scanned. None of the evaluated candidates improved both objectives.

The search permutes ship 15's seven interior pickups through swaps, reversals and
relocations, keeping miner 19102 first, collector 13077 last, all nine deployments
and the exact asteroid footprint. It starts from the newly certified v595 route.
The other 22 ships cannot change. Four CPU construction/inventory tests passed.
The archive audit found no exact matching collection order among 3,553 certified
route summaries; earlier dynamic programs did implicitly explore permutations.

| Measured work | Result |
| --- | ---: |
| New collection orders | 61 |
| Surrogate-feasible orders | 9 |
| Joint-polished seeds | 6 |
| Joint candidate evaluations / batches | 49,286 / 286 |
| Logical Lambert branch requests | 34,791,578 |
| Retiming driver calls / internal DP calls | 61 / 458 |
| Search stage wall time | 2.157210 s |
| Baseline independent + official checks | 20.762387 s |
| Campaign timer | 23.108265 s |
| Whole-route refinements | 0 |

The best evaluated alternative still loses 7.638604 raw kg and 8.666599 weighted
kg versus the incumbent under this surrogate. It therefore never enters expensive
refinement. This is a negative result for these sampled starts and neighborhoods,
not proof that all permutations or continuous trajectories are inferior.

The timing includes one fresh complete-fleet verification of the unchanged
baseline. It excludes initial imports, data loading, source hashing and parsing;
the `proxy_seconds` field starts after the CUDA context has been entered. These
counts are intermediate search work, not independently certified solutions/s.

Execution uses published source `2408ca9b`, frozen in `source.tar.gz`, with local
v596 CUDA core `86d7fdc9...`, QOCO, and both joint batch and device selection enabled.
Python orchestration and independent CPU verification remain. Physics and solver
settings are unchanged. The original v595 input hash was checked before and after
execution; both final checkers accept it. No new fleet is promoted.

`report.json` records the complete run. `raw.tar.gz` retains all per-order and
polished plans, and `source.tar.gz` includes frozen code and inputs. The driver is
`run.py`; `preparation.md` records the original bounded plan and reproduction
instructions. `sha256.json` preserves the published evidence hashes.
