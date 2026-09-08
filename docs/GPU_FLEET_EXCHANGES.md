# GPU fleet exchanges and the improved verified mission

This page records the fleet-exchange checkpoint. Subsequent
[GPU route hill climbing](GPU_ROUTE_HILLCLIMB.md) increases the verified score
to 12,843.555696 weighted kg.

The new best fleet returns **12,842.970672 weighted kg**, an improvement of
**32.834719 kg** over 12,810.135953 kg with the same pinned fixed bonus table.
Raw returned mass is **14,043.750856 kg**, or **610.597863 kg per ship**, across
23 ships. The optimiser trades 8.104038 raw kg for a larger weighted score.
All 195 deployed miners are collected. The ship-count limit is 23.001022.

Both the official verifier and the independent CPU DOP853 verifier pass the
complete fleet on RTX 5090 and H100. Both replacement routes were freshly solved
with CUDA seed generation, CUDA assembly/discretisation, native QOCO SCvx and
CUDA per-leg certificates. All **33/33 native leg solves converge on each GPU**.
The other 21 trajectories are retained from the previously verified incumbent.
No physics tolerances or mission rules were relaxed.

## Why this search finds a better fleet

The previous bounded depth-first search retained its incumbent even after
3.52 million nodes. The new stage evaluates all single-column additions,
removals and swaps around the best greedy/warm seed. Greedy feasibility repair
can remove a low-mass route without reconsidering additions afterward; the new
search closes that gap and then tests further profitable replacements.

CUDA evaluates each proposal against duplicate deployment/collection conflicts,
all foreign-miner providers, bundle ship counts and the nonlinear ship-count
rule. A deterministic reduction chooses the best improvement and reconstructs
the selected columns. A device flag stops computation when no improving swap
remains. The host queues a fixed maximum number of kernel pairs, without
per-round decisions or downloads. The final exchange statistics are transferred
once; the bounded branch-and-bound stage then uses the improved seed.

The default is 16 exchange rounds, bounded to 0–100 through the Python API.
`exchange_rounds=0` uses the legacy C ABI and disables exchanges. The new
`spacepdhcg_gtoc12_fleet_search_v2_host` extension leaves the original ABI intact.
No CPU optimiser or LP fallback is involved in the CUDA backend. The three new
telemetry fields are `exchange_proposals`, `exchange_moves` and `exchange_rounds`;
`nodes` continues to mean actual branch-and-bound nodes.

## Actual work and qualification

On the same 2,492-column pool, CUDA screens **179,205 exchange proposals** in
three sweeps and accepts two swaps, with **zero branch-and-bound nodes**. Proposal
counts include combinations rejected by cheap feasibility/value checks; they are
not trajectory solutions. A 200,000-node tree budget and a 100-round exchange cap
both retain the same new selection. Exact timing medians and proposal rates are
in [`summary.json`](../results/lambda/2026-09-09/gpu-fleet-exchange-v733/summary.json).

Five alternating repetitions compare exchanges on/off; the first repetition is
excluded from warm medians. These measure an already available column pool and
exclude route generation and physics refinement. The new complete call is around
a tenth of a second locally:

| GPU | Native call median | Complete Python call median | Native proposals screened/s |
|---|---:|---:|---:|
| RTX 5090 | 0.047892 s | 0.099070 s | 3,741,874 |
| H100 | 0.050963 s | 0.127997 s | 3,516,356 |

The previous CPU LP-assisted run took 110.026 s
locally / 196.207 s on Lambda and found a lower metadata objective of 12,810.437045
kg. These are different algorithms and proof/bound quality, not an equal-work
kernel speedup. The GPU non-exhaustive bound remains weak, and no global optimum
or leaderboard placement is claimed.

Columns 1786 and 2297 replace old incumbent ships 3 and 10. One numerical source
was found on Lambda; the other was absent from the checked archives. Both plans
were reconstructed and freshly solved instead of treating old `certified` flags
as fresh qualification. Fleet output is renumbered by selected-column order.

Four complete qualification runs are retained: v729 uses the previously qualified
native refinement runtime; v732 reruns GPU fleet selection and refinement with
the new v727 library. All four pass both complete-fleet checkers at the new score.
The v732 runs take 30.768 s locally and 52.282 s on H100, including parsing,
refinement, full checks and viewer export. Independent CPU propagation takes
20.244 s / 39.028 s of those runs. These single campaign timings are not a measured
whole-campaign speedup.

The final regression set passes **90 tests on each GPU**, with ten catalogue-
dependent skips. Tests cover exhaustive random packing comparisons, greedy
feasibility repair, CPU-enumerated one-swap local optima, negative-value providers,
cooperative cycles, budgets, bundles, unchanged legacy calls and JSON telemetry.
Memory, race and synchronization sanitizer modes pass both the 29-case native
subset and the full 2,492-column exchange workload on both GPUs.

## Evidence, reproduction and display

[`gpu-fleet-exchange-v733`](../results/lambda/2026-09-09/gpu-fleet-exchange-v733/)
contains frozen source/library archives, raw reports, inputs, native solve logs,
both complete mission outputs and independent trajectory exports. Archive member
sets, lengths and SHA-256 hashes are verified before and after transfer. The final
two dependency tests are captured in the v730 record, following the v727 build.

Final fleet-selector/refinement library SHA-256:

* RTX 5090: `53433d81470253325ea794f763922a9f77066b4c56d1285539cd532b1d7e4428`
* H100: `32405cdb7ce3ada705d21e9a648700db8bf270ee5d67e5d384d16019c0f3641f`

The displayed H100 result is
[`h100-best/Result.txt`](../results/lambda/2026-09-09/gpu-fleet-exchange-v733/h100-best/Result.txt),
SHA-256 `fba0ee086ae55d6c690f0e5dbaf874834b6f63e2f238e436d90671bdddff3c5f`.
The dataset is `gtoc12-exchange-v733`. Its samples come from fresh independent
propagation of this exact emitted solution.

```powershell
$viewer = 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
if (-not (Get-NetTCPConnection -LocalPort 4173 -State Listen -ErrorAction SilentlyContinue)) {
  Start-Process node -ArgumentList @('scripts/serve.mjs', '--port=4173') -WorkingDirectory $viewer -WindowStyle Hidden
}
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-exchange-v733&epoch=69807&preset=oblique&z=1'
```

The full application is still not GPU native: Python orchestrates routes, and
independent CPU mission checks dominate elapsed time. Numerical fleet setup has
subsequently moved to [CUDA scoring and topology construction](GPU_FLEET_TOPOLOGY.md).
Further score gains need better route generation and richer exchanges, alongside
work redistribution and stronger GPU bounds. A subsequent
[retained workspace](GPU_FLEET_WORKSPACE.md) removes repeated topology packing
and allocation, with measured gains on both GPUs and no score change.
