# Retained CUDA fleet workspace

This page records the v735 retained-workspace checkpoint. The current backend
also performs [scoring and topology construction on CUDA](GPU_FLEET_TOPOLOGY.md);
the benchmark results below remain tied to their original frozen builds.

Repeated selection from the same 2,492-column pool now takes **50 ms on RTX 5090
and 54 ms on H100**, including Python result materialisation and independent
packing checks. This is **2.03× / 2.32× faster** than the published one-shot
selector on identical work. The mission score remains **12,842.970672 weighted
kg**; this change creates no new trajectory or score improvement.

## What remains on the device

`CudaFleetWorkspace` snapshots the column pool and fixed bonus weights once.
Its C++ workspace retains the CUDA stream, column values, conflict/provider CSR,
GPU-computed column order and all search buffers. Each solve uploads only the
incumbent mask and submits GPU search with the requested ship, node and exchange
limits. All kernels are submitted before the final diagnostic downloads.

At this checkpoint, creation constructs topology and aggregates values in Python.
Subsequent searches reuse that work. The original one-shot entry points remain
available; existing CLI calls still use them. Repeated callers explicitly own a
workspace rather than relying on a hidden global cache. A changed pool, bonus
table or prefix partition requires a new workspace.

The snapshot protects optimisation inputs from later mutation of caller-owned
dictionaries, including bundled columns. Returned packing dictionaries are also
copied. Opaque trajectory artifacts are retained by reference and are not used
to make packing decisions. Python serialises solve/close operations; direct C
callers must serialise workspace access and solve on its creation device.

## Measurements

Seven interleaved repetitions compare the published v727 one-shot path, the new
v735 one-shot path and the retained path. Medians exclude the first repetition;
the CUDA runtime is already initialised. Every comparison checks identical
selected identifiers, objective, bound, search counts, exchange counts and
exhaustiveness. CPU and GPU runs are serialised with the existing GPU lock.

| GPU | Published complete call | Retained complete call | Retained native solve | Repeated-call speedup |
|---|---:|---:|---:|---:|
| RTX 5090 | 101.958 ms | 50.258 ms | 45.210 ms | 2.03× |
| H100 | 125.950 ms | 54.347 ms | 49.349 ms | 2.32× |

Setup takes 124.903 ms / 125.699 ms once with a warm runtime. Including this
measured setup cost, ten searches are 1.62× / 1.88× faster. Cold CUDA startup is
additional. There is no claimed one-shot speedup: the new complete one-shot
medians are 107.415 ms / 130.923 ms. Most of the repeated-call gain comes from
avoiding CPU topology reconstruction, not faster numerical GPU kernels.

Each exchange workload screens **179,205 packing proposals**, accepts two moves
in three sweeps and visits zero tree nodes. Retained throughput is approximately
19.90 / 18.40 fleet selections per second, or 3.96 / 3.63 million proposals per
native second. These are selections from existing routes, not new physically
solved trajectories.

With a requested 200,000-node budget, all variants actually visit **35,145 nodes**
and retain the same fleet. Complete-call medians improve from 354.213 to 302.962
ms locally and 342.761 to 263.195 ms on H100. Unused prefix budgets are still not
redistributed, and the non-exhaustive GPU bound remains weak.

## Validation and evidence

Both GPUs pass **95 tests**, with ten catalogue-dependent tests skipped in the
frozen test environment. The new tests compare repeated searches with exhaustive
small-pool enumeration and one-shot searches, alternate budgets/limits/seeds,
exercise cooperative providers and bundles, reject invalid inputs and check
snapshot isolation and lifetime management.

CUDA memory, race and synchronisation checks pass both a 34-test subset and
repeated full-pool creation/search/destruction, including the tree search. Full
memory checking reports **zero bytes leaked in zero allocations** on both GPUs.
The portable archive replay also reproduces the expected selection locally.

The combined revision, including the separately committed route-completion cost
fix, was subsequently rebuilt on both GPUs: **106 tests pass on each**, with ten
catalogue-dependent skips. Its v739 integration source, build logs and native
library hashes are recorded separately; timing medians above refer to the v735
benchmark binaries.

The local Nsight Systems command completed but its trace contains no CUDA kernel
data. That trace is retained as diagnostic evidence; it does not support a
kernel-by-kernel timing claim.

[Retrieved evidence and exact medians](../results/lambda/2026-09-09/gpu-fleet-workspace-v738/summary.json)
include complete frozen source/native-library archives, all reports, the input
pool and archive member hashes. The unchanged numerical mission remains the
[previously qualified result](GPU_FLEET_EXCHANGES.md), SHA-256
`fba0ee086ae55d6c690f0e5dbaf874834b6f63e2f238e436d90671bdddff3c5f`.
No physics tolerance, route propagation or certification code changed here.

## Using the retained API

```python
from spacepdhcg.gtoc12.gpu_fleet import CudaFleetWorkspace

# columns, weights and incumbent are the existing fleet-master inputs.
with CudaFleetWorkspace(columns, weights=weights) as workspace:
    for budget in (0, 200_000, 2_000_000):
        result = workspace.solve(incumbent=incumbent, node_cap=budget)
        incumbent = result.selected
```

The C ABI exposes `spacepdhcg_gtoc12_fleet_workspace_create_host`,
`spacepdhcg_gtoc12_fleet_workspace_solve_host` and
`spacepdhcg_gtoc12_fleet_workspace_destroy_host`. `setup_seconds` measures Python
packing plus native creation; each result's `native_seconds` measures only its
native solve. Use total wall time including setup when comparing whole jobs.

To replay on the local CUDA-capable WSL environment, from the repository root:

```powershell
wsl -d Ubuntu-22.04 -- /home/angus/worktrees/spacepdhcg-literature-venv/bin/python results/lambda/2026-09-09/gpu-fleet-workspace-v738/reproduce/replay.py results/lambda/2026-09-09/gpu-fleet-workspace-v738/local.tar.gz --node-cap 0 --repeats 5 --output /home/angus/workspace-replay.json
```

The output path must not already exist. Use `h100.tar.gz` with a compatible H100
Python/CUDA environment. Replay validates archive inputs and packing; it does
not rerun mission physics. The verified fleet remains displayed in
`gtoc12-exchange-v733` in the existing web visualiser.

The whole application is not yet GPU native. Python route orchestration and
independent CPU mission audits remain; initial numerical topology construction
has subsequently moved to CUDA as described above.
Further work should target measured kernel bottlenecks, stronger route generation
and broader GPU-controlled search; this fixed pool is already close to the older
CPU relaxation bound and offers limited remaining score headroom.
