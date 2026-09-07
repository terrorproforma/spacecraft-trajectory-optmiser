# GPU solver recovery: 7 September 2026

The Lambda campaign was running the old `1dbcae0` executable. Its downloaded
210-group snapshot contained 1,890 failed attempts: 594 numerical failures and
1,296 timeouts. A completed benchmark group did not mean a successful solve.
This is a solver regression investigation, not a new GTOC12 fleet or leaderboard score.

## Confirmed defects and corrections

QOCO's `safe_div(a,b)` replaced every division with `abs(b) <= 1e-15` by
`DBL_MAX`. This includes valid finite divisions in the cone algebra near the
interior-point boundary. For example, `1e-12 / 1e-18` became `DBL_MAX` instead
of `1e6`. The prepared-source patch now guards exact zero only, preserving
ordinary floating-point division for nonzero denominators. The upstream
zero-denominator behavior is unchanged. Actual GPU tests fail 5/7 cases with
QOCO128 and pass 7/7 with corrected QOCO131.

Separately, changing and restoring the SCvx trust radius or virtual penalty
reapplied the dynamics row-conditioning factors. Repeated updates could turn
a factor of 100 into 1,000,000. Numeric updates now reconstruct the dynamics
rows from retained device variational data before conditioning them. The
regression compares the restored coefficients with the CPU transcription.
This affected the diagnostic preamble; the production driver already refilled
its dynamics each outer iteration, so it does not explain all Lambda failures.

The native driver's internal IPM tolerance now leaves a factor of 100 margin
for the independently computed outer residual. External conic and physics
thresholds are unchanged. Tightening the internal tolerance alone did not fix
the numerical failures.

## Separate recovery profile

`device_scvx_integration_test --g4-recovery MANIFEST POLICY_SHA MATRIX_SHA CAPABILITY_SHA [RUIZ]`
reuses a frozen manifest's physical problem with a separately identified
`gpu-ipm-recovery-v1` profile. It uses native CUDA IPM, cold starts, five GPU
Ruiz passes by default, and minimum one / maximum 100 outer iterations.
The existing convergence and physics checks still decide acceptance.
Optional Ruiz counts are 5, 10, or 20; more passes did not consistently help.

The original benchmark forces its requested outer repetition count as both
minimum and maximum. The recovery profile permits early convergence instead.
Its output explicitly says `official_g4_sample: false`; do not merge these
records into the frozen campaign or relabel them as its pure-IPM baseline.
The frozen hybrid remains a polishing policy requiring PDHCG residual <=1e-6;
it is not a stalled-solve fallback. Recovery can be selected directly.

## Local evidence

RTX 5090, CUDA 12.8, cuDSS 0.8.0.10, corrected QOCO131; six archived physical
coordinates, unchanged conditioning span 4 and physics tolerance 1e-6.
Times cover the native SCvx call and are individual measurements, not medians.

| Intervals | Seed | Seconds | Terminal residual | Outer disposition |
|---:|---:|---:|---:|---|
| 100 | 71 | 0.251 | 2.36e-14 | Converged |
| 100 | 479 | 0.262 | 1.16e-13 | Converged |
| 500 | 617 | 4.262 | 1.15e-13 | Converged |
| 2000 | 389 | 6.659 | 5.08e-13 | Converged |
| 2000 | 101 | 12.641 | 4.98e-13 | Trust region exhausted; physics qualified |
| 2000 | 521 | 7.539 | 3.88e-13 | Converged |

All six pass the independent physics checks. Five meet the outer convergence
test; the remaining case must not be described as converged. All retain raw
statuses, objective values, residuals, and timings. This is not an H100 speedup
measurement, nor proof that the full pipeline is GPU-controlled.

Validation also includes eight direct native conversion/origin configurations,
the producer/input guard tests, and 89 GPU integration tests. The broader suite
passed 330 tests, including the formerly failing coast test. Two injected-error
tests caught an experimental retry hiding failures; that retry was removed.
The final focused suite passes all six recovery, failure-accounting and
persistent-session tests. No failed candidate was promoted by weakening a gate.

After removing the experimental retry, the final native v155 / QOCO131 runtime
passes the complete broader regression: **332 passed in 349.89 seconds**.
This includes the previously failing coast and injected-error cases. The final
raw test output is retained alongside the H100 results as
`local-final-regression.json`.

The prior full-IPM-graph compute-sanitizer limitation remains unresolved. Standalone
audit checks and ordinary numerical tests do not establish a full sanitizer pass.

## Lambda H100 validation

Commit `613e3d3460c0151c3cf0357beb83367b2b1acbaa`, native v155 and corrected
QOCO131 were built for sm90 with CUDA 12.8 and cuDSS 0.8.0.10. The old campaign
was stopped after preserving its SQLite checkpoint and raw logs; no other GPU
process was running when validation began.

| Intervals | Seed | SCvx seconds | Outer disposition |
|---:|---:|---:|---|
| 100 | 71 | 0.230 | Converged |
| 100 | 479 | 0.232 | Converged |
| 500 | 617 | 3.444 | Converged |
| 2000 | 389 | 10.696 | Trust region exhausted; physics qualified |
| 2000 | 101 | 9.022 | Converged |
| 2000 | 521 | 8.193 | Converged |

All six pass the unchanged physics checks with zero CPU solver fallbacks and
one retained QOCO workspace each. Objective differences from the local results
are below 4.72e-9. Outer convergence differs between devices for two seeds, so
these are individual measurements, not a controlled hardware speedup estimate.
Seven arithmetic cases, standalone audit/conversion checks, eight native
conversion configurations and three recovery/session pytest cases pass on H100.
The additional H100 GTOC12 GPU integration suite passes **89 tests in 10.20
seconds**. Its first collection attempt imported Lambda's old editable package;
the corrected invocation bypasses that importer only in the test subprocess
and loads the pinned new checkout. Both logs are retained separately.

The downloaded [summary](../results/lambda/2026-09-07/gpu-recovery-v155/summary.json)
and [raw bundle](../results/lambda/2026-09-07/gpu-recovery-v155/results.tar.gz)
include source/runtime hashes, commands, logs, all six inputs, failed build
attempts and the preserved legacy campaign. Bundle SHA-256:
`ec8d75b63b354f9954ffe0dcb70532b5fd212ae034503af45a1f92bc5769d573`.

The existing web visualiser displays these results under **Lambda solver
recovery — 7 September** in the GTOC12 view. The rendered fleet remains v11.

## Reproduction

Use `scripts/gpu/prepare_qoco_gpu.py` for a fresh patched vendor build; it now
applies the division correction. For an already prepared QOCO128 source copy,
apply `scripts/gpu/prepare_qoco_safe_division.py --destination COPY` once and
rebuild QOCO. Keep the original source/library for comparison.

The arithmetic GPU regression is `cpp/cuda/tests/qoco_safe_division_test.cu`;
compile with the prepared QOCO `include` directory. End-to-end regression is
`tests/test_g4_recovery_gpu.py`, enabled by `SPACEPDHCG_G4_EXECUTOR` and
`SPACEPDHCG_QOCO_LIBRARY`. Its checked-in fixture records the original manifest
and hash provenance. `--g4-dump` extracts the assembled CQP for an independent
CPU diagnostic without introducing a CPU fallback into production.

Evidence bundle: `artifacts/performance/gpu-recovery-v155-checkpoint.json` and
`gpu-recovery-v155-diagnostics.tar.gz` retain the measurements and rejected probes.
