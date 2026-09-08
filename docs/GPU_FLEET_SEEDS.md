# Parallel CUDA fleet seed construction

This page records the seed checkpoint. Subsequent
[cooperative tree search](GPU_FLEET_TREE.md) reduces the same 35,145-node retained
call to 16.29 ms locally / 11.74 ms on H100 with unchanged selected routes.

Repeated fleet selection now takes **7.49 ms on RTX 5090 / 8.86 ms on H100**,
including the Python call and independent packing checks. That is **6.63× /
6.15× faster** than the preceding GPU topology implementation on the same pool.
The selected fleet and verified mission score remain unchanged:
**12,842.970672 weighted kg / 14,043.750856 raw kg**, across 23 ships.

## Measurements

The production benchmark compares the previous v742 and final v750 full CUDA
libraries. Seven rotating, interleaved repetitions cover old/new one-shot and
old/new retained calls at each tree budget. Medians exclude repetition zero.
CUDA is already initialised. Every call must match selected column IDs, score,
bound, exhaustiveness, actual tree nodes and exchange work counts exactly.

| Exchange-only workload | RTX 5090 | Lambda H100 |
|---|---:|---:|
| Previous retained call | 49.619 ms | 54.509 ms |
| New retained call | **7.486 ms** | **8.856 ms** |
| Retained-call speedup | **6.63×** | **6.15×** |
| New retained native solve | 3.318 ms | 3.415 ms |
| Repeated fleet selections / second | **133.59** | **112.91** |
| Previous one-shot call | 73.215 ms | 86.067 ms |
| New one-shot call | 30.921 ms | 41.189 ms |
| One-shot speedup | 2.37× | 2.09× |

Each selection uses **2,492 input columns / 2,488 usable columns**, screens
**179,205 logical packing proposals**, accepts two swaps in three sweeps, and
visits zero tree nodes. Dividing proposals by retained-call time gives about
**23.94 million / 20.23 million logical proposals per second**. Many proposals
exit on cheap rejection tests. These are fleet combinations from existing route
records; they are not newly solved or physics-qualified trajectories.

Retained times exclude one-time workspace creation. Its first observed sample
was 71.9 ms locally / 73.7 ms on H100; these are individual samples, not setup
medians. One-shot times include serialisation, creation, selection, independent
packing checks and destruction.

With a requested 200,000-node budget, every variant actually visits **35,145
nodes**. Retained calls improve from **309.190 to 267.391 ms locally** and
**262.528 to 216.506 ms on H100**: **1.16× / 1.21×**. Tree search now dominates
this workload. The exchange-only ratio does not apply to a complete mission
search or to tree-heavy workloads.

## What changed

CUDA-event profiling located the bottleneck in seed construction. The previous
kernel ran three serial seeds as three threads in one block. The replacement
uses **three blocks of 128 threads**, one block per greedy or incumbent seed.
Greedy column choices remain ordered; block threads mark conflicts in parallel
after each accepted column. Provider checks and victim candidates run across
the selected columns, followed by a deterministic reduction.

Shared memory caches ship counts, greedy order and the selected set. Repair
passes visit at most 100 initial selected columns, skipping removed entries,
instead of repeatedly scanning the whole 2,488-column pool. Mass and value sums
retain the original canonical input order. Conflict-free greedy construction
allows redundant conflict checks to be omitted during removal-only repair.
External incumbent masks still receive full validation, including rejection of
overfull masks. The existing 4,096-input-column and 100-ship limits are unchanged.

| Instrumented seed stage | RTX 5090 | H100 |
|---|---:|---:|
| Previous implementation (v746 profile) | 42.677 ms | 47.857 ms |
| Parallel prototype (v748 profile) | 11.295 ms | 5.057 ms |
| Final shared-memory implementation (v751 profile) | **1.377 ms** | **1.830 ms** |
| Seed-stage speedup | **30.98×** | **26.15×** |

These stage measurements use standalone fleet libraries with inserted CUDA
events: three repetitions per configuration, medians excluding repetition zero.
Every profiled call checks its output against the normal solve. They locate the
bottleneck and guide changes; the production API table above is the speedup
claim. Final profiled exchange acceptance totals about 1.20 / 0.96 ms; tree
search with the larger budget takes about 259 / 197 ms.

## Validation and evidence

Both GPUs pass **117 tests**, with ten catalogue-dependent tests skipped in the
frozen environment. The fleet subset has 55 cases, including new tests for an
overfull incumbent, 100 selected columns and the full 4,096-column capacity.
Forty-five cases also run under each CUDA memory, race and synchronization
checker. Full-pool checks repeat creation, empty selection, exchange selection,
tree search and destruction. Both GPUs report **zero errors, race hazards and
leaked allocations**.

An additional **128 deterministic randomized cases per GPU** compare the old
and final implementations exactly: selections, score, greedy score, bound,
nodes, exhaustiveness and exchange counts. Cases include negative weights,
ties, bundles, unsupported providers, cycles, invalid incumbent masks and
varying ship/node/round limits. This supplements the fixed-pool comparison;
it is not an exhaustive proof over all possible inputs. No numerical or physics
tolerance was relaxed.

The final source freeze starts from `fdf52ae31d259240ffebdcf79ed0965dce8b9298`
and adds only the seed kernel and two tests. That base includes the separately
developed route-completion API; this benchmark does not exercise that API.
Frozen source and binary hashes identify exactly what was measured.

The retrieved [evidence folder](../results/lambda/2026-09-09/gpu-fleet-seed-v755/)
includes both prototype/final source trees and native libraries, all three
instrumented profiles, intermediate/final benchmarks, sanitizer logs, randomized
comparisons, input pool and replay scripts. Each raw archive has 1,717 hashed
payload members. Archive hashes, unique member sets, lengths and member hashes
were checked before and after transfer. The
[summary](../results/lambda/2026-09-09/gpu-fleet-seed-v755/summary.json)
contains exact medians. The previous full-core baseline is retained in
[`gpu-fleet-topology-v745`](../results/lambda/2026-09-09/gpu-fleet-topology-v745/).

## Replay and displayed result

From the repository root, replay the archived local CUDA build:

```powershell
wsl -- /home/angus/worktrees/spacepdhcg-literature-venv/bin/python results/lambda/2026-09-09/gpu-fleet-seed-v755/reproduce/replay.py results/lambda/2026-09-09/gpu-fleet-seed-v755/local.tar.gz --node-cap 0 --repeats 5 --output /home/angus/seed-replay.json
```

The output path must not exist. Use the H100 archive on its compatible CUDA
runtime. The portable replay verifies archived inputs and performs independent
packing checks; it does not run a fresh full mission qualification. The archived
experiment scripts retain their original workspace paths for provenance.

The selected routes are identical to the independently qualified
[v733 mission](GPU_FLEET_EXCHANGES.md), whose solution SHA-256 is
`fba0ee086ae55d6c690f0e5dbaf874834b6f63e2f238e436d90671bdddff3c5f`.
Its existing viewer remains at
[the verified fleet](http://127.0.0.1:4173/?dataset=gtoc12-exchange-v733&epoch=69807&preset=oblique&z=1).
No new mission score or leaderboard placement is claimed by this optimisation.

Fleet numerical setup, seed construction, exchanges and tree search run in
CUDA. Python still serialises route records, orchestrates wider route exploration
and independently checks results. The broader GPU-native goal continues with
those remaining paths, tree-search scaling and better route generation.
