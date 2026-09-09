# Native collection geometry: local RTX 5090 and Lambda H100 evidence

The paired whole-search gain is **5.6–6.9% on H100** and **1.2–2.8% locally**, with
unchanged candidate files. The local measurements include desktop GPU activity.
Both native builds pass 248 tests, 110 cases under each of three CUDA safety tools,
controller checks, and 43 full leak-check cases.

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `local.tar.gz` | 29,895,357 | `f9c62ef7f91bf8ef3572fd6e5d9624484f4b79891f6a5ddd9dc3cc2997711e30` |
| `h100.tar.gz` | 29,070,239 | `e31555440fd10f2b5bf7fc1c98b2dc973c30752dd91eeab313eae68878368e0d` |

The archives contain frozen source, native library/controller binaries, build and
validation logs, eight benchmark runs per GPU, candidates and telemetry. The H100
archive also retains raw profiles. All owned workers exited before packaging.
The portable audit checks 1,997 archive members, source/binary bindings,
validation logs, candidate hashes against the previous checkpoint, work counts,
paired medians and raw profile summaries. It does not execute archived code,
GPU work, trajectory propagation or an official fleet checker.

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser'
python results/lambda/2026-09-09/gpu-native-collection-geometry-v863/audit_saved.py
```

`saved-audit.json` contains the reconstructed results. `reference.json` pins the
previous published candidate hashes. `source-binding.json` binds the four changed
source/test files to the frozen runtime based on commit
`bd664c279a90e30c914b38aef0b8f9df630be56e`. `index.json` hashes the package files.

The historical scripts in `reproduction/` retain their original host paths.
Rerunning them requires the recorded CUDA/Python environments, upstream source
and GTOC12 data; they are not portable one-command launchers. Presentation and
reproduction text outside the archives uses LF; archive bytes remain unchanged.

[Implementation, measurements and remaining host work](../../../../docs/GPU_NATIVE_COLLECTION_GEOMETRY.md)

The verified score remains **13,526.961241 weighted kg**. This performance
checkpoint does not replace the v633 visualiser dataset.
