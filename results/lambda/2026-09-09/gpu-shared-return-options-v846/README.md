# Shared resident return options: v846 evidence

Native C++ ownership now lets collection scheduling reuse immutable GPU return
rows. Cache hits avoid Lambert screening, device allocation/copy and GPU
synchronisation. Each caller owns a separate handle; eviction or producer
replacement cannot invalidate outstanding borrowers. Exact orbital data,
allowances, time-grid bytes and sorting mode form the cache key.

| GPU | Route | Fresh rows median | Shared rows median | Throughput gain |
| --- | --- | ---: | ---: | ---: |
| RTX 5090 | Ship 10 | 13.209 s | 10.123 s | 30.5% |
| RTX 5090 | Ship 21 | 10.959 s | 8.454 s | 29.6% |
| H100 | Ship 10 | 6.954 s | 6.019 s | 15.5% |
| H100 | Ship 21 | 6.090 s | 5.205 s | 17.0% |

Each mode has one warm-up and three alternating measured samples per GPU using
the same frozen build. `SPACEPDHCG_TEST_GTOC12_RETAIN_RETURN_OPTIONS=0/1` selects
fresh/shared resident returns. CUDA expansion, admission and existing catalogue
and DP reuse are enabled in both modes. Timings cover RouteSearch.run, excluding
startup and catalogue loading. `fresh_seconds`/`reuse_seconds` in raw reports
refer to these two modes.

All **989 surrogate candidates** (517 + 472) are byte-identical across eight runs
per GPU and match the preceding admission checkpoint. Combined search throughput
is about **53 candidates/s on RTX 5090 and 88 on H100**. These rates do not measure
certified trajectories or PDHCG solves.

Across the two probes, **12,188 of 12,346 return queries reuse rows**; only 158
perform fresh return screening. This avoids **12,992,408 repeated branch
requests**, reducing total fresh GPU branch requests from 35,358,344 to 22,365,936.
All resident option builds, including collection-hop options, fall from 13,906
to 1,718. GPU telemetry and RouteSearch's evaluation counter both exclude reused
branch requests; the audit checks their deltas independently.

## Validation and source

* **205 tests per GPU**, all passing.
* **67 tests per GPU under each of memcheck, racecheck and synccheck**, plus
  separate native controller checks; no reported errors or hazards.
* **29 focused GPU tests per device under full leak checking**, zero bytes
  leaked in zero allocations and zero reported errors.
* Frozen base `9681f6ce41cfc11dac927b0965efa879605f4f9a` with seven owned
  source/test overlays, bound to runtime bytes in `source-binding.json`.
* v844 builds and validates the runtime; v845 contains paired runs; v847 checks
  leaks; v848 profiles the same runtime after the preceding work completes.
* Complete runtime source and native binaries, every paired candidate file,
  raw profiles and reports are in the two archives. Each archive has a byte
  manifest; archive receipts pin the downloaded bytes. Top-level extracted
  reports are convenient copies of authoritative archived payloads.

The native cache retains at most 256 entries / 64 MiB of payload. Borrowers may
keep evicted rows alive until their handles close. Cached calls and destruction
of cached workspaces require the creating thread/device. Cache-key changes,
independent close, eviction, producer replacement, empty results and thread
ownership are explicitly tested. No physics or numerical tolerance changes.

## Remaining work and score

The subsequent RTX ship-10 profile attributes 6.48 cumulative seconds of an
11.49-second profiled search to collection-DP scheduling. Its 1,152 per-tour
preparations and 1,731 DP solves remain the main target for retained/batched GPU
control. Profiling times overlap and are not additional benchmark costs.
The complete planner still has host orchestration and preparation.

This optimization adds no trajectory certificate or fleet-score gain. The
separately verified v633 fleet has **24 ships / 208 asteroids, 13,526.961241
weighted kg and 14,915.044490 raw kg**. Wider saved candidates still require
native refinement and independent physics checks against the current fleet
budget. [Fleet provenance](../../../local/2026-09-09/fleet-addition-v633/README.md).

## Load and audit locally

Full directory:
`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\gpu-shared-return-options-v846`

Copy into PowerShell:

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser'
wsl -- python3 results/lambda/2026-09-09/gpu-shared-return-options-v846/audit_saved.py
Get-Content results/lambda/2026-09-09/gpu-shared-return-options-v846/saved-audit.json
```

The portable audit uses Python's standard library and no GPU. It streams each
archive, verifies every payload hash, reconstructs raw profile summaries,
recalculates timing medians and checks exact candidate equivalence and truthful
evaluation counts. Reproduction scripts retain original host paths and need
adaptation for other machines. Scientific reruns also need the original pinned
data, CUDA toolchain and native dependencies.

This performance checkpoint creates no new fleet visualisation dataset. With
the existing viewer running, the separately verified fleet remains available
as [gtoc12-fleet-v633](http://127.0.0.1:4173/?dataset=gtoc12-fleet-v633&epoch=69807&preset=oblique&z=1).
[Implementation and remaining GPU work](../../../../docs/GPU_SHARED_RETURN_OPTIONS.md).
