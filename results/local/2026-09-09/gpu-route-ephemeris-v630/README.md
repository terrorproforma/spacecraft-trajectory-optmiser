# CUDA route-boundary ephemerides, v630

The local implementation passes orbital-state parity and the unchanged archived-boundary representation guard. **It has no complete-method speedup at these small fresh-workspace batch sizes.** The native route path now computes its boundary ephemerides on CUDA; its Python boundary bridge still downloads vectors and the solver uploads them. This is not a fully device-controlled mission pipeline or a solver/fleet improvement.

The completed RTX 5090 CUDA 12.8/SM120 evidence contains the fresh source build, 56 passing CPU tests, native API/graph test, zero-error memcheck and independent Python state parity. The three GPU test processes account for 20 native request rows, including expected invalid rows, and 17 Python state requests.

The four exact v629 route prescriptions were compared for five predeclared alternating repeats. The actual counts are 20 CUDA workspace constructions/batches/destructions, **410 CUDA state rows and 720 uncached CPU body calls**. Every state output is retained. All 20 paired comparisons pass position norm <1e-3 km and velocity norm <1e-9 km/s. Their maxima are about **1.764523 mm** and **7.318215e-14 km/s**. All 160 saved first-leg vector restore checks (four vectors × two methods × 20 pairs) pass the existing per-vector `64*eps*max(1,max(abs(generated)))` bound. These are orbital-state/representation checks, not trajectory certificates.

| Route | Five-sample CPU median | Five-sample CUDA median | CPU/CUDA |
| --- | ---: | ---: | ---: |
| Ship 8 rank 0 | 1.390 ms | 1.682 ms | 0.826 |
| Ship 8 rank 1 | 1.575 ms | 1.727 ms | 0.912 |
| Ship 19 rank 6 | 1.662 ms | 1.739 ms | 0.956 |
| Ship 19 rank 7 | 1.422 ms | 1.658 ms | 0.858 |

CUDA timings include exact-key deduplication, construction/element upload, host bridge/calculation/download, result copying and destruction; both methods include measurement counters. The first CUDA sample is 221.074 ms and remains in the five observations. It can include context/library/kernel startup; later samples reuse the process context while still constructing a new workspace. Python imports and full-catalogue setup take 345.216 ms separately. Worker elapsed time through report assembly is 1.263616 s; parent supervision is 1.723236 s. These scopes must not be added together or treated as mission throughput. Files and comparisons are outside individual method timers.

**H100_not_run:** Lambda source/build/test delivery is prepared only. Its read-only host preflight passed, but automatic approval review rejected the exact 1,906,443-byte source upload before any process started. No H100 upload, build, test or performance result is claimed. The rejection and exact proposed destination/payload hash are retained. A future authorized H100 run belongs in a separate artifact; this sealed evidence must not be edited or relabeled.

All frozen source members and inputs are included, with intentionally retained duplicate small source archives where original manifests bind them. The local source archive SHA is `9c63e12b55273b8789c8ac6f8a868a158f76d3145aedb24c0ea43cc084cbeec3`; source tree `737137c4e7e85c762d413e094d69b9df423c409c6ab7581b82ce1f6eee754063`; local manifest `7435b392a5faceff50db4361bab287fb3ce98b1b37aa93d2d5d633c0a9e68a14`. The exact original 60,000-body catalogue remains an external dependency by official SHA `99a42cc30d4498d99b8acf507790ab74f040ff2e202ef6c8e90bbb39b6c46675`; selected raw rows/converted coefficients are included. No compiled binary, credential or cache is packaged. Historical runtime paths are provenance, not commands executed by the auditor.

Run the portable standard-library saved-data audit from any directory:

```text
python -B audit_package.py PATH_TO_THIS_PACKAGE --index-sha256 EXPECTED_INDEX_SHA --output NEW_AUDIT_JSON
```

The auditor verifies top-level/ZIP/nested member hashes and safety, exact 741-member source maps, source/ready/log/report bindings, raw/public CUDA vector bits, all saved CPU/CUDA state comparisons, the unchanged archive representation bound and timing medians. It imports no project/native/numerical package and performs no ephemeris evaluation, propagation, optimization or GPU call. Build/test/measurement scripts are included for source reproduction; executing them is separate work and is not part of saved-data verification. Initial lint findings and the upload refusal are preserved.
