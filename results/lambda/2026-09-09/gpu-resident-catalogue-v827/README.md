# Resident CUDA catalogue: measured checkpoint

Both GPUs now keep the same asteroid catalogue allocation through changes to
completion pricing. Across the two saved searches, this reduces catalogue
uploads from 356 to 2 and from 854.4 MB to 4.8 MB. Pricing/grid buffers still
rebuild. This is partial progress toward a GPU-controlled search.

| GPU | Search | Fresh median | Resident median | Throughput change |
| --- | ---: | ---: | ---: | ---: |
| RTX 5090 | Ship 10 | 21.229 s | 20.737 s | +2.37% |
| RTX 5090 | Ship 21 | 19.249 s | 18.810 s | +2.33% |
| H100 | Ship 10 | 17.862 s | 17.846 s | +0.09% |
| H100 | Ship 21 | 17.125 s | 16.938 s | +1.11% |

One warmup per mode, then three measured repeats per mode in alternating order.
Search settings, candidates and arithmetic are unchanged. Both modes retain
the earlier catalogue hash-prefix cache and collection-DP optimization. The
H100 ship-10 result is effectively flat; this small sample does not establish
a general solver speedup. Packing medians improve by 58–69% across these four
cases, while the rest of search remains the dominant cost.

Each two-route run emits 989 surrogate candidates (517 + 472), about 25.0/s
locally and 28.4/s on H100 using the sum of route medians. Each GPU ran eight
such searches, producing 7,912 candidate records including repeated identical
outputs. These are neither distinct new solutions nor certified trajectories.
All candidate files match byte-for-byte across modes and repeats on each GPU,
and match the preceding saved searches. Cross-GPU bytes retain the existing
floating-point differences and are not claimed identical.

The last independently verified fleet remains **13,023.704901 weighted kg /
14,291.006160 raw kg**, 23 ships and 199 asteroids. This checkpoint did not rerun
trajectory refinement or produce a new fleet. The existing dense replay is
available in the [web visualiser](http://127.0.0.1:4173/?dataset=gtoc12-catalogue-v822&epoch=69807&preset=oblique&z=1).

## Saved evidence

- `local.tar.gz` and `h100.tar.gz`: 994 payload files each, plus `FILES.json`.
  Exact frozen source, native binaries, controller, build/test logs, paired
  benchmark scripts, fleet/fit inputs, full candidate outputs and telemetry.
- `local/v826` and `h100/v826`: final test reports and logs. 165 tests per GPU;
  36 tests under each of memcheck, racecheck and synccheck. Controller checks
  also pass under all three tools.
- `local/v828` and `h100/v828`: four focused ownership tests with full CUDA
  leak checking. Zero leaked bytes and zero allocations left outstanding.
- `local/v824/report.json` and `h100/v824/report.json`: raw benchmark results.
- `saved-audit.json`: independently reconstructed medians, upload counters,
  timing ranges, candidate hashes and verification of every payload byte.
- `runtime/source-manifest.json` inside each archive binds final source.
  Base is `6c686cce48763a5ab34687a74d810f5ce693589b` with the four owned completion
  source/test files overlaid. Native compilation is v823; final v826 changes
  only host argument-list syntax and tests, with exact C++ equality checked.
- The initial v823 test failure is preserved. Its new replacement fixture
  retained stale geometry in the scalar reference; CUDA correctly updated
  in both modes. Final tests clear that reference cache and explicitly read
  shared device orbital data after either model owner is destroyed.

The payload includes the exact worker commands. Running those GPU tests requires
the matching GPU/CUDA architecture, the project's Python dependencies, and pinned
GTOC12 input data. The archives are evidence bundles, not Windows executables.

To audit the downloaded files and open the existing fleet replay:

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser'
wsl -- python3 results/lambda/2026-09-09/gpu-resident-catalogue-v827/audit_saved.py
Start-Process 'http://127.0.0.1:4173/?dataset=gtoc12-catalogue-v822&epoch=69807&preset=oblique&z=1'
```

[Implementation and remaining host work](../../../../docs/GPU_RESIDENT_CATALOGUE.md).
