# Cooperative CUDA fleet tree search

The same **35,145-node fleet search** now takes **16.29 ms on RTX 5090 /
11.74 ms on Lambda H100**, including the retained Python call and independent
packing checks. That is **16.36× / 18.25× faster** than the preceding seed
implementation. Every compared call returns the same selected routes, score,
bound and work counts. The verified mission remains **12,842.970672 weighted kg
/ 14,043.750856 raw kg**, across 23 ships.

## Full-call measurements

The v757 production benchmark compares the previous v750 and final v763 full
CUDA libraries on 2,492 input columns, of which 2,488 are usable. Seven rotating,
interleaved repetitions compare four modes at each budget. Medians exclude
repetition zero, with CUDA already initialised. All comparisons require exact
selected IDs, objective, bound, exhaustiveness, node count and exchange counts.

| 200,000-node requested budget | RTX 5090 | Lambda H100 |
|---|---:|---:|
| Previous retained call | 266.437 ms | 214.246 ms |
| New retained call | **16.287 ms** | **11.738 ms** |
| Retained-call speedup | **16.36×** | **18.25×** |
| New retained native solve | 11.114 ms | 5.865 ms |
| Previous one-shot call | 289.881 ms | 246.478 ms |
| New one-shot call | 40.545 ms | 43.664 ms |
| One-shot speedup | 7.15× | 5.64× |
| Repeated tree-enabled fleet selections / second | **61.40** | **85.19** |

Each call screens **179,205 logical exchange proposals**, accepts two swaps in
three sweeps and then visits **35,145 actual tree nodes**. The requested 200,000
nodes are partition budgets: invalid prefixes do not redistribute unused work.
These are combinations of existing routes, not newly solved or physics-qualified
trajectories. The measured retained calls process approximately **2.16 million /
2.99 million actual tree nodes per second**, including exchange and Python costs.

With zero tree budget, new retained calls take **7.179 / 8.776 ms**. One-shot
medians change from **30.679 to 32.396 ms locally** and **40.879 to 41.065 ms on
H100**. There is no claimed one-shot improvement for this exchange-only workload;
native one-shot portions are essentially unchanged. The large speedup applies
when tree search runs. First observed workspace creation takes 73.4 / 74.4 ms;
these are individual samples, not setup medians. Retained times exclude setup.

One-shot times include Python serialisation, workspace creation, CUDA selection,
independent packing checks and destruction. The performance claim concerns fleet
packing, not complete mission optimisation or a CPU-equivalent global solver.

## Kernel changes and numerical behaviour

Each prefix partition previously ran on one thread. It scanned all columns at
every node, repeatedly reading selected flags, column records and conflicts.
The replacement uses **one cooperative 128-thread block per partition**, with
the existing 256 partitions at the default eight prefix bits.

Scores, ship counts, chosen flags and conflict counters reside in shared memory.
A sorted depth-first stack stores at most 100 selected columns and their masses.
Each node recomputes selected mass and score in the same canonical order as
before. Threads independently check providers and update distinct conflict
neighbours when the controller pushes or pops a column. Accepted-mask copies are
parallel. The controller preserves the original deterministic traversal, tie
policy and budget assignment.

Threads also calculate partial sums of positive, unblocked future scores.
**Every partial addition and the final bound reduction rounds upward**. The
result remains a conservative pruning bound. This changes the bound's addition
order, so internal bound bits need not match the old serial accumulation on all
possible inputs. Accepted-score arithmetic and the externally reported bound
remain unchanged. All recorded full-pool and randomized comparisons match the
old decisions and counts exactly; that empirical result is not a general proof
of identical traversal for every floating-point input. Physics and packing
tolerances are unchanged.

## Profiling and rejected prototype

| Instrumented tree stage, same workload | RTX 5090 | H100 |
|---|---:|---:|
| Previous serial tree (v751 profile) | 258.897 ms | 197.089 ms |
| Cooperative stack, serial bound (v762 profile) | 99.529 ms | 56.571 ms |
| Final parallel bound (v764 profile) | **7.119 ms** | **2.385 ms** |
| Tree-stage speedup | **36.37×** | **82.65×** |

These standalone fleet libraries insert CUDA events around stages. Three
repetitions cover each budget/round combination; medians exclude repetition zero.
Each profiled call verifies its output against the normal solve. These profiles
guided development; the full production-call table above is the speedup claim.

The first cooperative prototype, v756, passed 55 ordinary tests and memory
checks, but **failed CUDA racecheck** on both GPUs. During backtracking, lane zero
could overwrite control flags before other lanes consumed them. A barrier before
backtracking fixes the race in v761; v763 adds the parallel bound reduction.
The failed source, binary and sanitizer output are preserved. Its v760 timing
profile is retained as rejected-prototype evidence and is excluded from claims.

## Validation and reproducibility

Both GPUs pass **119 tests**, with ten catalogue-dependent tests skipped in the
frozen environment. The fleet subset has 57 cases. New tests traverse and
backtrack from a 100-column branch that violates the mass-dependent ship limit,
checking retention of the feasible 32-ship incumbent under two prefix settings.
Forty-seven cases run under each CUDA memory, race and synchronization checker.
Full-pool sanitizer runs repeat creation, empty selection, exchange selection,
tree search and destruction. Both GPUs report **zero errors, race hazards and
leaked allocations** in the final build.

An additional **192 deterministic randomized comparisons per GPU** vary ship
limits, node budgets, exchange rounds and prefix bits. Inputs include conflicts,
negative weights, bundles, provider cycles, unsupported providers and invalid
incumbents. They compare selections, objective, greedy objective, bound, nodes,
exhaustiveness and exchange counts exactly against the previous implementation.

The final freeze starts from commit
`0876c2f41c807c0e02b8f8cbc9d4bd475f3ab152`, with only the tree kernel and two
parameterized test cases overlaid. The
[retrieved evidence](../results/lambda/2026-09-09/gpu-fleet-tree-v765/)
contains full source and libraries for failed v756 and final v763, plus the
v761 source overlay, complete source manifest, build and validation logs. To
reconstruct v761 source, overlay its two archived files onto the v763 tree and
verify against the v761 manifest. All three instrumented implementations and
their baseline profile are archived with their standalone libraries.

The previous full-core baseline is available in
[`gpu-fleet-seed-v755`](../results/lambda/2026-09-09/gpu-fleet-seed-v755/).
The [summary](../results/lambda/2026-09-09/gpu-fleet-tree-v765/summary.json)
provides exact production medians and stage metrics. Archive hashes, unique
member sets, lengths and member hashes are verified before and after transfer.

From the repository root, replay the downloaded local CUDA build:

```powershell
wsl -- /home/angus/worktrees/spacepdhcg-literature-venv/bin/python results/lambda/2026-09-09/gpu-fleet-tree-v765/reproduce/replay.py results/lambda/2026-09-09/gpu-fleet-tree-v765/local.tar.gz --node-cap 200000 --repeats 5 --output /home/angus/tree-replay.json
```

The output path must not exist. The H100 archive requires its compatible CUDA
runtime. The portable replay checks archive integrity and packing; it does not
perform a fresh complete mission qualification. Experiment scripts retain their
original paths for provenance.

The selected fleet is identical to the independently qualified
[v733 mission and visualiser](GPU_FLEET_EXCHANGES.md), with solution SHA-256
`fba0ee086ae55d6c690f0e5dbaf874834b6f63e2f238e436d90671bdddff3c5f`.
No new mission score or leaderboard placement is claimed. Python input packing,
broader route orchestration and independent mission audits remain outside this
CUDA stage. Improving those paths and generating better routes remain part of
the active GPU-native goal.
