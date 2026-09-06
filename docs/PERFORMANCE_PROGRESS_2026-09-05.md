# Performance implementation and fresh GPU evidence — 5 September 2026

The architecture review `ad2ac42` was fetched and merged into local main with merge
commit `01e3360`. No remote branch or running campaign was changed. The implementation
below starts from `df1858f`; the multi-block GPU redesign remains outstanding.

## Implemented: identical-result Lambert screening

`src/spacepdhcg/gtoc12/lambert.py` now preserves NumPy singleton dimensions instead of
materializing five transfer-by-scan arrays. Stumpff functions are evaluated once on
the shared scan grid, held in a read-only eight-entry LRU cache. Only grid-dependent
values are cached; trajectories, controls and residuals are never cached. The scan
points, root selection, bisection, tolerances, FP64 arithmetic and physics are unchanged.

Seven-repeat measurements on the local WSL CPU, NumPy 2.5.2, one BLAS thread, using
the Lambert API's full 8,192-sample scan:

| Workload | Baseline median | Updated median | Speedup |
|---|---:|---:|---:|
| 1 Lambert transfer | 1.670 ms | 1.354 ms | 1.23x |
| 64 transfers | 41.328 ms | 20.826 ms | 1.98x |
| 256 transfers | 165.816 ms | 76.093 ms | 2.18x |
| 1,024 transfers | 569.422 ms | 294.465 ms | 1.93x |
| Complete 150-day SCvx leg | 136.786 ms | 134.378 ms | 1.02x |

Every Lambert output matched the baseline bit for bit, including feasibility, both
velocities, universal parameter and residual. The complete leg retained five outer
iterations and exactly the same final mass. Its independent DOP853 certificate reports
position error `8.049e-7 km`, velocity error `5.714e-14 km/s`, and final mass
`2445.3111007852112 kg`. The end-to-end change is small and should be treated as timing
noise; this is a screening improvement, not a claimed fleet-search or GPU speedup.

The benchmark loads baseline modules directly from git without switching the checkout:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python \
  scripts/benchmark_gtoc12_hotpaths.py --baseline-ref df1858f \
  --repeats 7 --profile --output artifacts/performance/final.json
```

Raw timings, numerical comparisons, certificate and profile are in
`artifacts/performance/final.json`. Profiling the complete leg still identifies Clarabel
solve/setup and repeated sparse assembly as the main CPU costs. No low-thrust solver
or assembly changes have been applied in this tranche.

The route-screening wrapper uses a coarser existing 256-sample scan. A seven-repeat
run after the regression process finished (`--scan-samples 256`,
`artifacts/performance/screening-256-controlled.json`) measured 1.21x / 1.62x / 1.78x
for 64 / 256 / 1,024 transfers, again with identical outputs. The 1,024-transfer median
fell from 25.043 ms to 14.072 ms. The 1.9–2.2x full-grid result should not be substituted
for the route wrapper's measured speedup. Neither scan resolution was changed.
The earlier overlapping run is retained as `screening-256.json` but is not used for
the final comparison. A separate `bitwise-8192.json` run additionally verified IEEE
bit representations, including signed zeros and NaNs, for the full-grid workload.

## Implemented: H1 telemetry correction

`scripts/gpu/run_g3_h1.py` now uses actual production counters, rejects conflicting
counters and represents missing measurements as null. Legacy production output with
a bare `-inf` diagnostic ratio is handled without deriving counts from requested
repetitions. The native H1 emitter now includes actual outer/inner/accepted/rejected
counts directly. It was compiled with CUDA 12.8 and format warnings treated as errors;
no GPU execution was performed.

Additional corrections:

- Recovery iterations are no longer substituted for primary inner iterations.
- Requested repetitions are no longer substituted for accepted steps.
- Polishing, factorisations and re-solves are unknown unless measured explicitly.
- Partial `allocation_bytes` is no longer labelled peak/reserved device memory.
- Each compact sample reports one measured repetition and zero external warmups.
- Notes explicitly identify scaling work hidden inside legacy solve timing.

`scripts/gpu/reexport_g3_h1.py` generates schema-validated corrected derivatives in a
new directory and records raw/exporter/original-compact hashes. Original artifacts
remain unchanged. All 42 archived H100 H1 samples were re-exported under
`artifacts/performance/h100-h1-corrected/`. Sample 041 now correctly reports
**three inner iterations, zero accepted steps**, and unchanged total time `15.2584705 s`.
Total memory remains unknown. This export fix does not establish the time spent in scaling.

```bash
python scripts/gpu/reexport_g3_h1.py \
  results/lambda-h100/v1-reseal-9e75b47-h100/g3/h1 \
  artifacts/performance/h100-h1-corrected-new
```

## Fresh H100 results reviewed, 12:46:57 UTC / 22:46:57 AEST

Read-only inspection of the running H100 campaign found **108/396 groups complete**,
288 remaining, one running, zero groups invalidated/quarantined. This is a partial
campaign snapshot, not a final comparison or a new benchmark of these code changes.
The latest three completed groups all evaluate adaptive low thrust at N=500,
conditioning label 4. Each contains two warmups and seven measured attempts; all nine
attempts in each group timed out, with no contamination flagged.

The first measured attempt in each group:

| Group suffix | Deadline | Wall time | Inner iterations | Canonical residual | Outer iterations |
|---|---:|---:|---:|---:|---:|
| `2b419f…` | 600 s | 600.076 s | 300,000 | 270.605 | 0 |
| `b3528a…` | 120 s | 120.005 s | 87,900 | 145.213 | 0 |
| `b074b5…` | 120 s | 120.051 s | 88,325 | 152.325 | 0 |

These are different campaign coordinates, so the residuals are not a controlled
120-versus-600-second convergence experiment. All three requested primal warm starts
but recorded actual cold starts. In the first 600-second measured attempt, recorded
primary solve time is approximately 390.993 s and recovery 208.974 s. Its near-zero
dynamics defect does **not** qualify the trajectory: canonical residual is about 270.6
against a requested `1e-6`, and no outer step was accepted.

This reinforces the need to address both per-iteration parallelism and convergence/
backend selection. It does not identify a crossover or establish IPM superiority on
these exact coordinates. Lambda was left running with its frozen policy unchanged.
The raw snapshot and remote file paths are in `artifacts/performance/h100-latest.json`.

A further export issue is visible in these G4 timeout records: `accepted_trajectory_count`
is one although `quality.qualified` is false and `accepted_steps` is zero. The emitter
hard-codes that count (`device_scvx_integration_test.cu`, G4 timing serialization).
Use the explicit qualification/disposition fields for success statistics; this G4 issue
is recorded here and has not been changed in the running frozen campaign.

## Next implementation gates

1. Attribute preamble/scaling, iteration, recovery and conversion costs separately on a
   dedicated GPU; reconcile raw and compact counters before using timings in a cost model.
2. Parallelize the serial change/scaling/reduction stages with transactional cancellation
   and independent residual parity. Measure displaced starts as well as zero fixtures.
3. Cache QOCO conversion and GTOC12 union sparsity/solver updates; measure presolve and
   factorization effects at the same final quality before changing the default.
4. Introduce ordered multi-block operators behind the current ABI, then structured
   trajectory operators and versioned convergence/backend policies. Increasing only
   the current solve launch's block count is incorrect.

## Verification

- All **156 GTOC12 and H1 tests passed**, no skips, in 337.27 s with the existing pinned
  data directory from the WSL release checkout. This includes native C++ Lambert parity,
  Kepler endpoint closure, SCvx certification, search determinism, and the independent
  plus official verifiers on the published 39-, 37- and 36-ship reference solutions.
- The stricter IEEE bit comparison passed separately after tightening the numerical test.
- Ruff and `git diff --check` passed for the changed code.
- CUDA H1 telemetry translation unit compiled with CUDA 12.8; format warnings were
  treated as errors. This was a compile check, not a GPU runtime or performance test.

The statements above describe the initial CPU/telemetry tranche. Subsequent explicit
authorization started local RTX 5090 implementation and measurements, recorded in
[GPU-native optimization progress](GPU_NATIVE_OPTIMIZATION_PROGRESS.md). The Lambda
workloads remain untouched. At the time of this initial report, implementation changes remained in the working tree; only
the requested documentation branch merge was committed. The subsequent GPU work and this initial tranche are included in the September 6 publication.
