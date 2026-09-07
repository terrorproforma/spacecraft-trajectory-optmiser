# H100 development campaign v269 — 8 September 2026

The completed campaign returns **548.2546201231 kg from eight asteroids with one
ship**, accepted by both the official checker (548.255 kg) and independent
propagation. This is **21.765914 kg / 4.13%** above the previous 526.488706 kg
development mission. The separate 23-ship incumbent stays at **12,805.194 weighted
kg**. This route is not automatically compatible with that fleet.

## Work and performance

| Measurement | Recorded result |
|---|---:|
| Complete campaign | 89.553 s |
| Initial beam search | 51.491 s |
| Retiming/extension stage, including its refinement | 10.472 s |
| Catalogue / filtered search pool | 60,000 / 10,612 asteroids |
| Completed transfer branch evaluations | 44,722,546 |
| Branch evaluations / complete campaign second | 499,400 |
| Collection options | 2,782,091 |
| GPU batches | 5,662 |
| Initial candidate routes | 106 |
| Initial refinements / certificates | 3 / 2 |
| Retiming/extension attempts | 2 |
| Final certified candidate columns / selected ships | 3 / 1 |
| Device retiming drivers / DP and forward evaluations | 50 / 293 |
| Host retiming-table uploads | 0 |

These counters measure evaluations, including repeated branches, not millions
of distinct complete missions. The third certified column improves an original
route. Three certificates in 89.553 s is 0.0335/s for this one observation, not a
general sustained solver-throughput claim. The master is exhaustive over its
three input columns only; it does not prove global GTOC12 optimality.

The selected mission has 16 refined transfer arcs and one camp interval. Maximum
independent discrepancies are 0.633488 km position, 1.108251e-7 km/s velocity and
6.185e-11 kg mass, with no violations under unchanged acceptance gates. Search
screening, seed, discretisation, assembly and SCvx outer-loop backends are CUDA;
the conic backend is GPU QOCO. CPU setup, route/fleet orchestration, parsing and
reporting remain. This is not yet a completely GPU-native executable pipeline.

The run uses the final parallel-completion library from v266, published source
`58eefa56`. Its exporter records temporary Lambda Git snapshot `c727c958`; the
source/runtime hashes in `v269/report.json` connect it to the tested code.
`retrieval.json` pins the downloaded archive. The entire output, both checkers,
failed refinement, command, logs and exact viewer export are retained. This
campaign enables retiming/extension, so comparison with v209's 77.276 s run is
not a matched overall speed comparison.

## Display locally

The visualiser includes dataset **GPU campaign v269 (548 kg certified)**, with
507 exact archived replay samples. Import verifies the exporter, solution and
pinned catalogue hashes and 3,010 Kepler context samples. All 38 viewer tests
and all six dataset checks pass. Browser inspection confirms one ship, eight
asteroids, both verifier passes and physical 1x vertical geometry.

Open <http://127.0.0.1:4173/?dataset=gtoc12-v269&epoch=69807&preset=oblique&z=1>.
If the server is stopped, paste this into PowerShell and keep the terminal open:

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-v269&epoch=69807&preset=oblique&z=1'
node scripts/serve.mjs --port=4173
```

Full solution path:
`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-08\gpu-native-campaign-v269\v269\output\fleet\Result.txt`

Use `summary.json` for compact statistics; `v269/output/run_report.json` is the
full authoritative report. `sha256.json` pins each evidence file except itself.
