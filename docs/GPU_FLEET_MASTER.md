# CUDA fleet selection

The fleet master now has an explicit `cuda` backend. One native call builds greedy
seeds, validates the incumbent, searches independent branches, checks cooperative
miner dependencies and the nonlinear ship-count rule, and selects the winner on
the GPU. There is no CPU optimisation or LP fallback inside that backend.

This is a working GPU implementation, not yet a replacement for the CPU master's
search quality. Python still assembles column topology and fixed-bonus values,
filters unusable columns, and independently checks the returned packing. Wider
route generation/control and the independent full-mission audits remain outside
this GPU stage. The application is not yet fully GPU native.

## Measurements on the archived route pool

The frozen input contains 2,492 columns: 2,469 historical route summaries plus
the 23 ships of the previously verified incumbent. Its SHA-256 is
`3420336f54f067eb4a2c9b17a13f4ede5193048b5f8b22d7c8adb6f2f3efef8b`.
Both devices use the same input, bonus table, incumbent and 200,000-node ceiling.
Five alternating rounds compare native versions; round zero is excluded from
the medians. Each measured call performs exactly 35,145 search nodes and returns
the identical 23 incumbent columns.

| GPU | Initial CUDA native call | Conflict-counter native call | Ratio | Final complete Python call |
|---|---:|---:|---:|---:|
| RTX 5090 | 2.706572 s | 0.300220 s | 9.02x | 0.355882 s |
| Lambda H100 80 GB | 3.597019 s | 0.256157 s | 14.04x | 0.339250 s |

Native timing includes allocations, transfers, all kernels and result return.
The Python-call timing also includes packing, validation and report construction.
These are **fleet-selection stage ratios against the initial CUDA implementation**,
not speedups over the CPU optimiser or over a complete trajectory campaign.
Measured native rates are about 117,064 and 137,201 branch-and-bound nodes/s,
respectively. A node is a partial packing decision, not a new trajectory solution,
Lambert evaluation or independently certified mission.

The initial search launched 256 threads in four blocks. Giving each independent
search its own block alone did not improve the local median (2.749052 s). The
useful change replaces repeated conflict-CSR scans at each node with per-column
exclusion counters, updated when a column is pushed or popped. The final launch
uses one thread per block: it distributes independent searches but leaves warp
lanes idle. Further parallel reduction and scheduling work remains warranted.

The node ceiling is divided among prefix branches. Impossible prefixes terminate
without spending their allocation; unused work is not redistributed. Consequently
200,000 requested nodes produce 35,145 actual nodes on this pool. Reported node
counts are actual work, not the ceiling. With a 20-million-node ceiling, both GPUs
perform 3,515,625 nodes: 15.264862 s locally and 14.416917 s on H100 for the complete
call, without a score improvement. Changing the prefix width changes the search
allocation and amount of work; the faster 10-bit sweep is not an equal-work speedup.

## Score and search-quality limit

The retained result is **12,810.135953 weighted kg / 14,051.854894 raw kg**, from
23 ships. It is exactly the previously checked fleet in
[`gpu-leg-certificate-v712/h100-best/Result.txt`](../results/lambda/2026-09-09/gpu-leg-certificate-v712/h100-best/Result.txt).
No trajectory changed in these packing benchmarks and no new full-mission
propagation was performed. The earlier complete-fleet checks remain the source
of its physics qualification.

The CPU LP-assisted master found a metadata packing worth 12,810.437045 weighted
kg in single runs taking 110.026 s locally and 196.207 s on Lambda. Its bound was
12,844.338856 kg. Ten selected columns lack their numerical solution files in the
local archive; the candidate is **not a newly verified mission score** and is not
displayed as an improvement. In all, 2,437 historical columns have this source-file
limitation. Archived `certified` flags are historical metadata, not fresh audits.

CUDA's non-exhaustive bound is only the sum of positive usable column values
(1,126,320.391048 kg here). It does not prove near-optimality. CPU and GPU searches
therefore differ in result quality, bounds and work; dividing their wall times
would misrepresent a speedup to equivalent optimisation quality. The backend
remains opt-in and the CPU default is retained.

The next search changes should redistribute work from invalid prefixes, improve
ship-count/packing bounds, and explore exchanges around the incumbent. Improving
the score substantially also requires generating better routes; faster enumeration
of this fixed historical pool alone is insufficient.

## Use and implementation boundaries

Add `--fleet-backend cuda` to `gtoc12 run`, `gtoc12 cluster-fleet` or
`gtoc12 fleet-master`. Set `SPACEPDHCG_GTOC12_CUDA_LIBRARY` to a rebuilt native
library and use `--workers 1` on commands exposing that option. Other numerical
backends are selected separately. A missing library/API or failed device result
raises an error rather than running the CPU master. `lp_bound` and `lp_node_limit`
are CPU-backend controls and are not used by CUDA.

Python callers can use:

```python
from spacepdhcg.gtoc12.cooperative import solve_fleet_master

result = solve_fleet_master(
    columns, weights=fixed_bonus_weights, incumbent=verified_columns,
    node_cap=200_000, max_ships=100, backend="cuda",
)
print(result.objective, result.nodes, result.exhaustive, result.native_seconds)
```

The current limits are 4,096 usable columns, 100 ships and 0–10 prefix bits
(default 8 in `solve_fleet_cuda`). Bundle columns consume their member ship count.
Foreign requirements accept alternative deployers only at the matching deployment
epoch. Negative-valued provider columns and cyclic dependencies are supported.
Incumbents are accepted only as complete, feasible selections; an omitted or
invalid incumbent provides no score-retention guarantee.

The C ABI consumes host arrays. Conflict CSR must be symmetric, contain unique
indices and exclude self-edges; `pack_columns` establishes those invariants.
Counts fit in a byte because every selected column consumes at least one of the
100 allowed ships. Topology and temporary device buffers are allocated per call;
retained topology/buffers are a future optimisation. No physics tolerance changes
are part of this work.

## Validation and reproducibility

Both final native builds pass **50 tests**; five unrelated catalogue-dependent
cooperative tests are skipped in the frozen environment. The fleet tests compare
12 random pools with exhaustive subset enumeration and cover shared-miner cycles,
negative providers, bundles, empty inputs, budgets, incumbent retention and
nonfinite data. Memory, race and synchronization sanitizer modes pass both the
13-case correctness subset and the full 2,492-column/35,145-node benchmark on both
GPUs. These checks cover this stage, not every CUDA path in the application.

A reporting follow-up exports unavailable bounds as JSON `null` instead of the
invalid JSON token `Infinity`. The original benchmark reports are preserved;
`report.strict.json` contains derived copies with nonfinite values replaced by
null for browser consumption. The expanded follow-up passes **72 tests on each
GPU**, with ten catalogue-dependent skips. It also repairs a pre-existing CPU
collection-table test double that lacked the resident-table lookup method. That
failure reproduces against v720 before the reporting change; original failure,
baseline reproduction and corrected results are retained in the v725/v726 records.

Final library SHA-256:

* RTX 5090: `810b6ca75be3d52ec1d95ce9760e11be67edf343a757e5467bdc5d535ae43c67`
* H100: `65287f398e1c48ddb69b1e0f48b5dd226427a8862e8bc1b9a0f10c0a3b048cae`

[`gpu-fleet-v724`](../results/lambda/2026-09-09/gpu-fleet-v724/) retains frozen
sources, libraries, build/test/sanitizer logs, the exact pool, benchmark reports,
scripts and member-level hashes in local/H100 archives. `v713` is the initial
implementation; `v718` is the local launch-only experiment; `v720` is the final
counter implementation; `v717`, `v721` and `v723` hold comparisons and full-pool
sanitizers. The v718 first configuration failed because the copied source lacked
Git metadata; its original failure and successful retry are both retained.

Frozen sources start from `a1b02bd20d59e3dccc530eff933f87d20d453534`, overlaid with
the exact fleet files recorded by each source manifest. Concurrent persistent-
solver work was excluded. Every archive member's size/hash and the complete member
set were checked before and after retrieval. Reproduction scripts retain the
original home-directory layouts; the portable `reproduce/replay.py` runs an archived
native variant using the current Python environment's installed dependencies.

The existing visualiser's `gtoc12-fleet-v724` dataset shows the same verified
trajectory samples with the new fleet-selection measurements. The timing displayed
there is for fleet selection, not a new trajectory campaign.

On this Windows workstation, load it with:

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
  Start-Process node -ArgumentList @('scripts/serve.mjs', '--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-fleet-v724&epoch=69807&preset=oblique&z=1'
```
