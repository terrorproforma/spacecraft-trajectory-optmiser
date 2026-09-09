# CUDA beam admission: v841 evidence

Reserve checks, Earth-return feasibility and deploy-beam diversity admission
now run inside the retained CUDA expansion workspace. Immutable return-option
rows remain on the GPU. Only accepted candidates are downloaded for subsequent
collection-tour scoring. This removes a host filtering stage; complete mission
search still has host orchestration, first-level admission and collection scheduling.

| GPU | Route | Host admission median | CUDA admission median | Throughput change |
| --- | --- | ---: | ---: | ---: |
| RTX 5090 | Ship 10 | 12.793 s | 13.074 s | -2.15% |
| RTX 5090 | Ship 21 | 11.221 s | 11.042 s | +1.62% |
| H100 | Ship 10 | 6.998 s | 6.958 s | +0.59% |
| H100 | Ship 21 | 6.317 s | 6.083 s | +3.85% |

Each mode has one warm-up and three alternating measured samples per GPU, using
the same build. `SPACEPDHCG_TEST_GTOC12_GPU_ADMISSION=0/1` selects host/CUDA
admission; CUDA expansion and retained catalogue/DP paths are enabled in both.
Report fields `fresh_seconds` and `reuse_seconds` mean host and CUDA admission.
Timings cover RouteSearch.run, excluding process startup and catalogue loading.
RTX sample ranges overlap: these measurements do not establish a general RTX
whole-search speedup. H100's ship-10 improvement is also small.

The 517 + 472 = **989 surrogate candidate files are byte-identical** across all
eight runs per GPU and match the preceding CUDA-expansion checkpoint. Both
searches together price 1,251,916 valid children. Host candidate construction
falls from **105,755 to 1,092**, and ranked-result downloads fall from
**9,408,696 to 96,096 bytes**: about 99% less. The two admitted counts are 576
and 516. These are search candidates, not certified solutions or PDHCG solves.

## Tests and source provenance

* **190 tests per GPU**, all passing.
* **52 tests per GPU under each of memcheck, racecheck and synccheck**, plus
  separate endpoint-controller checks; zero reported errors/hazards.
* **14 GPU tests per device under full leak checking**, zero bytes leaked in
  zero allocations and zero errors.
* Exact frozen base `a8cd81e520a51390ea8f818d421fc56dcdd669d3` with eight owned
  source/test overlays. `source-binding.json` binds the published changes to
  runtime file bytes; `runtime/source-manifest.json` covers the complete source.
* v839 builds and validates the runtime, v840 contains paired searches, v842
  checks leaks, and v843 profiles the same build after those runs finish.
* The raw archives include exact sources, native binaries, reports, every
  paired candidate file and raw cProfile data. Their per-file manifests and
  archive receipts are checked by the portable audit.

No dynamics or physical tolerances change. This checkpoint produces no new
trajectory certificate and no fleet-score promotion. The separately verified
24-ship fleet reaches **13,526.961241 weighted kg / 14,915.044490 raw kg** through
recovery of an existing route. That composition is outside this benchmark.
[Fleet provenance](../../../local/2026-09-09/fleet-addition-v633/README.md).

## Remaining bottleneck

The post-change RTX ship-10 profile records 7.77 million calls; route completion
takes 12.93 cumulative seconds of a 14.80-second profiled search. It includes
1,152 collection-tour preparations, 1,731 DP solves, 6,443 return-option builds
and 8,552 collection selections. These overlapping cumulative measurements are
not clean timings or independent costs to add together. Retaining and batching
completion state is the next major route toward GPU-controlled execution.

## Retrieve, inspect and audit

Full local directory:
`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-09\gpu-beam-admission-v841`

Copy and paste into PowerShell:

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser'
wsl -- python3 results/lambda/2026-09-09/gpu-beam-admission-v841/audit_saved.py
Get-Content results/lambda/2026-09-09/gpu-beam-admission-v841/saved-audit.json
```

The audit uses Python's standard library, reads each archive in one streaming
pass, checks all payload bytes, reconstructs profile summaries, recalculates
medians and verifies unchanged candidates. It does not run scientific solvers.
The initial read-only extraction pass was stopped to replace repeated gzip
seeks with streaming reads; the downloaded archive bytes and GPU runs were
unchanged. `saved-audit.json` records the completed streaming audit.

Reproduction scripts retain the original host paths and must be adapted for
another machine. Raw archive contents are authoritative; extracted reports
provide convenient inspection. No new trajectory dataset is created by this
performance checkpoint. With the existing viewer server running, the separately
verified fleet is available as [gtoc12-fleet-v633](http://127.0.0.1:4173/?dataset=gtoc12-fleet-v633&epoch=69807&preset=oblique&z=1).

[Implementation and remaining GPU work](../../../../docs/GPU_BEAM_ADMISSION.md).
