# CUDA fleet scoring and topology construction

This page records the GPU setup checkpoint. Subsequent
[parallel seed construction](GPU_FLEET_SEEDS.md) reduces retained selection to
7.49 ms locally / 8.86 ms on H100 with the same fleet.

The CUDA fleet backend now builds its numerical problem on the GPU. Column
scoring, raw-mass aggregation, foreign-provider eligibility, canonical ordering,
conflict discovery and provider CSR construction join the existing CUDA search.
The existing `--fleet-backend cuda` CLI path uses this implementation, including
one-shot calls. The verified mission remains **12,842.970672 weighted kg**.

## Measured performance

The benchmark uses the same 2,492 input columns and fixed bonus table as the
previous workspace benchmark. CUDA retains 2,488 usable columns. Seven
interleaved repetitions compare old/new one-shot and old/new retained calls;
medians exclude the first repetition with an already initialised CUDA runtime.
Every comparison requires exact equality of selection, objective, bound,
exhaustiveness, tree nodes and exchange counts.

| GPU | Previous one-shot call | GPU-setup one-shot call | Speedup | New retained call |
|---|---:|---:|---:|---:|
| RTX 5090 | 94.930 ms | 72.265 ms | 1.31× | 49.806 ms |
| H100 | 123.978 ms | 86.929 ms | 1.43× | 54.632 ms |

One-shot times include Python serialisation, creation, GPU setup/search, result
checks and destruction. Their native portions are 49.914 ms / 52.909 ms. Retained
calls avoid creating the workspace again; their native solves take 44.298 ms /
48.932 ms, close to the previous retained implementation. This change improves
setup and GPU coverage; it does not claim faster search kernels.

Each exchange workload screens **179,205 packing proposals**, accepts two swaps
in three sweeps and visits zero tree nodes. These are selections from existing
routes, not new trajectory solves. The same requested 200,000-node budget visits
35,145 actual nodes in every variant. Complete one-shot times for that workload
improve from 354.079 to 325.168 ms locally and 343.961 to 296.843 ms on H100.

First workspace creation in the comparison takes 69.791 ms locally / 71.861 ms
on H100. These individual samples include first use of the new setup kernels;
they are not setup medians or a claimed cold-start speedup. Exact reports and
all repetitions are in the [retrieved summary](../results/lambda/2026-09-09/gpu-fleet-topology-v745/summary.json).

## Construction and precision

Python serialises route records: deployment/collection identifiers, epochs,
per-asteroid masses, attached bonus weights, ship counts and certification flags.
CUDA calculates scores with compensated double-precision sums, preserves each
mass record's order and separately rounds multiplication. GPU kernels sort route
keys, test provider availability across blocks, filter eligible columns and rank
them by score and identifier. Sorted-key intersection detects conflicts. CUB
reductions and prefix scans construct deterministic CSR offsets; device kernels
compact neighbours and provider groups in canonical column order.

Eligibility retains the existing single-pass policy: a foreign miner may be
supplied by any certified input column at the matching epoch. After filtering,
only usable columns appear in the actual provider lists. A route whose sole
provider is itself filtered out remains unselectable in the search. Epoch
matching remains 1e-6 days; ship rules and physics tolerances are unchanged.

The first prototype passed small cases but failed the full-pool exact comparison:
its score differed by **1.8189894e-12 kg**. The source of that difference was naive
accumulation versus the reference runtime's compensated sum. The final GPU code
uses Neumaier compensation, as in the
[CPython 3.12 reference implementation](https://raw.githubusercontent.com/python/cpython/v3.12.13/Python/bltinmodule.c).
A cancellation regression checks that weighted terms `1e16 + 1 + 1 - 1e16`
retain the result `2`. Python's independent result check uses `math.fsum`, also
providing accurate checking on Python 3.11. The existing 1e-8 kg packing-check
tolerance was not relaxed; the final full-pool comparison is exact.

C++ validates buffer sizes, owns CUDA allocations and reads integer counts to
size compact buffers. The final permutation and selection return to Python for
independent checking and reporting. Temporary setup arrays are freed after
creation; the retained workspace owns the resulting graph and search buffers.

## Validation and evidence

Both RTX 5090 and H100 pass **115 tests**, with ten catalogue-dependent tests
skipped in the frozen environment. The fleet subset has 53 cases; 43 run under
each CUDA memory, race and synchronisation checker. Full-pool checks also repeat
creation, empty selection, exchange search, tree search and destruction. Both
GPUs report **zero errors, race hazards and leaked allocations**.

New tests compare GPU setup against the legacy CPU-built CSR and native search,
including filtered providers, cycles, ties, negative weights, epoch mismatches,
more foreign requirements than columns, an empty usable pool and cancellation.
A test forbids CPU scoring/topology functions during workspace creation.

[`gpu-fleet-topology-v745`](../results/lambda/2026-09-09/gpu-fleet-topology-v745/)
contains frozen source and libraries for both the first and final prototypes,
the failed strict comparison, final benchmarks, sanitizer logs and pool inputs.
Archive hashes, complete member sets, member lengths and SHA-256 hashes are
verified before and after retrieval. Failed v741 evidence is retained; published
timing claims use successful v743 comparisons of the final v742 code.

This change produces no new mission trajectory or qualification. The unchanged
verified [fleet and visualiser](GPU_FLEET_EXCHANGES.md) remain the score authority,
with solution SHA-256
`fba0ee086ae55d6c690f0e5dbaf874834b6f63e2f238e436d90671bdddff3c5f`.

## Running it

Rebuild the CUDA library before using the updated Python module. It requires the
new `spacepdhcg_gtoc12_fleet_workspace_create_routes_host` symbol and has no CPU
setup fallback. The C header documents the serialized records and outputs. The
legacy CSR C APIs remain available. The route API supports up to 4,096 input
columns, including those later filtered out.

The Python entry points stay the same:

```python
from spacepdhcg.gtoc12.gpu_fleet import CudaFleetWorkspace, solve_fleet_cuda

result = solve_fleet_cuda(columns, weights=weights, incumbent=incumbent, node_cap=0)
with CudaFleetWorkspace(columns, weights=weights) as workspace:
    result = workspace.solve(incumbent=result.selected, node_cap=200_000)
```

From the repository root, replay the archived local build with:

```powershell
wsl -d Ubuntu-22.04 -- /home/angus/worktrees/spacepdhcg-literature-venv/bin/python results/lambda/2026-09-09/gpu-fleet-topology-v745/reproduce/replay.py results/lambda/2026-09-09/gpu-fleet-topology-v745/local.tar.gz --node-cap 0 --repeats 5 --output /home/angus/topology-replay.json
```

The output path must not exist. Use the H100 archive with its compatible runtime.
Replay checks packing and archive integrity; full mission physics is a separate
verification step. The broader application still has Python route orchestration,
input serialisation and independent CPU mission audits. Completing those remaining
paths and improving route generation remain part of the active GPU-native goal.
