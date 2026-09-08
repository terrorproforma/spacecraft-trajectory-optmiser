# Local GPU joint evaluation validation

The final compatibility wrapper passes **62 tests on the new v596 core** and **54 tests on the older v590 core**, with zero skips. Ruff check and format pass for the final wrapper and compatibility tests.

The unchanged v596 native implementation also passed both standalone tests (full evaluation and winner selection) normally and under memcheck, synccheck and racecheck. Assertions were enabled with `-UNDEBUG`. The 42 original Python tests plus 8 selection tests passed normally and under all three sanitizers, with zero errors or race warnings.

| Ship | Cache | Candidates | Scalar median | Batched median | Speedup | Batched candidates/s |
|---|---|---:|---:|---:|---:|---:|
| ship-01.json | cold | 186 | 84.988 ms | 9.572 ms | 8.88x | 19,431 |
| ship-01.json | warm | 186 | 33.817 ms | 4.285 ms | 7.89x | 43,408 |
| ship-02.json | cold | 156 | 66.888 ms | 8.322 ms | 8.04x | 18,746 |
| ship-02.json | warm | 156 | 24.444 ms | 3.223 ms | 7.58x | 48,395 |

Three ABBA blocks per case, six timings per implementation, 3-day mesh. Every candidate and ordered winner comparison passed. Cold means an empty per-run Lambert cache with CUDA runtime/workspaces already warm; warm uses the same complete geometry cache for both implementations.

Timing includes move construction, Python bookkeeping, cache lookup, geometry, native packing/transfers, joint arithmetic, plan materialization and winner selection. It excludes loading, warmup, the independent parity audit, SCvx refinement and independent trajectory/fleet verification. These are local RTX 5090 search-surrogate results, not complete mission-solve speedups or H100 measurements.

The initial attempt correctly rejected seven skipped tests caused by an omitted historical route fixture. Its exact Git-tracked bytes were added and all requested Python checks were rerun. The first attempt and its successful native checks are retained under `evidence/`; completed parity/sanitizer results are under `evidence-complete-fixture/`; the final compatibility snapshot is under `publication-validation/`.

The native build source and final Python compatibility overlay have separate recorded provenance. `publication-source.tar.gz` contains the exact final test source snapshot, excluding Git pointers, bytecode, data downloads and build outputs. `original-v590/` separately preserves the original native source and earlier test/benchmark evidence used to explain the v595 score experiment.

New core SHA-256: `86d7fdc952e82f7e5557c1651a1722900d83cdd98d8a709a4b6274fd83af4671`.

For the separately certified score experiment, see [GPU orphan recovery](../../../../docs/GPU_ORPHAN_RECOVERY.md). H100 upload/run remains blocked pending user approval; no remote results are claimed here.

Machine-readable index: `publication-summary.json`; per-file hashes: `evidence-files.json`.

The separate CMake test-registration addendum is in `ctest-registration/`: a fresh Release build discovered and passed both joint CTest targets (2 passed, zero skipped, 0.49 s), with `-UNDEBUG` after `-DNDEBUG`. It does not alter the original v596 core build provenance.
