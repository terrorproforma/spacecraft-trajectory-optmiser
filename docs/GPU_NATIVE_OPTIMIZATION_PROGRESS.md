# GPU-native optimization: local implementation and measurements

The active goal is a GPU-native C++/CUDA production numerical pipeline, optimized on
the local RTX 5090 without relaxing physics accuracy. This is an implementation
checkpoint, not a claim that the whole migration is finished.

## Implemented

- Added an occupancy-bounded cooperative CUDA grid for the persistent PDHG solve.
  Device-wide barriers order sparse operations, cone projections, residual evaluation,
  and stop/cancellation decisions. Increasing the old kernel's block count would not
  have provided this synchronization.
- Parallelized the ten cone-preserving Ruiz passes and twenty power iterations, using
  retained reduction buffers. Cancellation preserves the previously committed scales.
- Distributed coefficient-change scans across blocks and reduced their results on the
  device. Identical infinite bounds now contribute zero change, rather than NaN.
- Distributed SCvx numerical coefficient assembly across blocks. Control tracking and
  fuel terms share an owner thread and retain their arithmetic order.
- Removed the CPU download/edit/upload from explicit scaling refresh.
- Reduced numeric fingerprint XORs within each block before one global atomic update.
  The integer fingerprint remains bitwise identical to the independent CPU reference.
- Separated scaling time from iteration/recovery time. Recovery remains a subset of
  the workspace's `solve_seconds`; SCvx separates that subset in its own totals.
- Added an execution-block override for reproducible comparisons. The default uses
  parallel scaling for problems with at least 128 variables, a block-local iteration
  loop below 4,096 variables, and a cooperative loop for larger problems. This retains
  the lower synchronization cost for small operators.

All production arithmetic remains double precision. No fast-math option, tolerance
relaxation, dynamics simplification, or reduced iteration budget was introduced.

## Verified local measurement

Fresh-process HCW fixture: 2,000 intervals, 18,006 variables, 12,012 scalar rows,
8,000 affine rows. Two warmup runs and seven measured runs per configuration,
alternating configuration order. Medians:

| Scope | Saved baseline | Optimized default | Ratio |
|---|---:|---:|---:|
| Complete test command, including process/CUDA startup and checks | 5.164 s | 0.461 s | 11.2x |
| SCvx solve wall time | 4.774 s | 67.204 ms | 71.0x |
| CQP update + scaling + solve | 4.806 s | 1.836 ms | 2,617.8x |

This is an already-feasible zero-trajectory fixture: both versions execute one inner
iteration and one outer iteration, accept zero steps, and report zero canonical and
nonlinear residuals and zero CPU/GPU trajectory difference. It measures the removal
of serial setup costs. These ratios must not be generalized to difficult solves.
Event timings and host wall timings use different clocks and are separate measurements;
their medians are not additive.

The optimized CQP spends approximately 0.082 ms on update, 1.541 ms on scaling and
0.214 ms on iteration/recovery launch work. A local sweep favored 128 cooperative
blocks over 64 or 170 on this fixture. A subsequent 10,000-interval sweep (one warmup,
five measured runs) also favored 128 blocks: CQP medians were 3.160 ms at 128,
3.229 ms at 170, 3.610 ms at 256, and 3.842 ms at 340 blocks. Every sample retained
zero physical residuals and identical work counts. Its full SCvx time remained about
328 ms, showing that the remaining surrounding work dominates this easy fixture.
The raw local artifact is `artifacts/performance/cooperative-hcw-10000-sweep.json`.

Reproduce in WSL with the source tree as the working directory:

```sh
python3 scripts/gpu/benchmark_cooperative.py \
  --executable /home/angus/build-spacepdhcg-gpu-native/cuda-tests/device_scvx_integration_test \
  --baseline-library /home/angus/build-spacepdhcg-gpu-native/baseline/libspacepdhcg_cuda.so \
  --blocks auto,64,128,170 --warmups 2 --repeats 7 \
  --output artifacts/performance/cooperative-hcw-2000-final.json
```

The local JSON artifact contains every sample, executable/library/source SHA-256
values, device/driver identification, and warmup markers. It records the implementation
as measured before its local checkpoint commit on `perf/gpu-native-pipeline`, based on merge commit
`01e336082c2999baa87b6e1449c3a3eafac5f3d6`.

- GPU: RTX 5090, 170 SMs, driver 595.97, CUDA 12.8, architecture `sm_120`.
- Baseline library SHA-256: `e126612a02737e86d57cfb132a27ade586c67b677fefa35e5d552378990681fe`.
- Measured optimized library SHA-256: `fa3a74b50aaa3b124ca68aba94b7d6ba4502fa30b5217009c2f44f2f0a163d39`.
- The GPU also drives the Windows display; clocks were not locked. These are local
  measurements, not H100 measurements. Existing Lambda workloads were left untouched.

## Physics and correctness checks

The native build passes its warnings-as-errors configuration. The following tests
passed locally: cooperative PDHG, persistent equality/QP, persistent SOC, allocation
lifecycle, pointer contracts, stream lifetime, variational dynamics, and time-dilated
dynamics. The cooperative test covers repeated solves, coefficient updates, explicit
refresh, signed structural zero, cancellation, and mixed standard/rotated variable
cones at multiple grid sizes, with independent analytic optima.

The full production integration test passed with eight cooperative blocks across
HCW, powered descent 3DOF, low thrust, and powered descent 6DOF:

- Maximum canonical residual: `9.56640559e-9`.
- Maximum nonlinear residual: `2.92768847e-8`.
- Maximum CPU/GPU trajectory difference: `0`.
- Maximum coefficient difference: `2.7599450502791001e-13`.
- Displaced HCW: three accepted steps, no rejections, 1,050 inner iterations, matching
  the baseline's work count and acceptance decisions.

A real regression was found during this process: the first atomic Ruiz row maximum
treated the unsigned bits of `-0.0` as larger than positive numbers. The nonzero HCW
acceptance test failed although the zero fixture and simple analytic tests passed.
Canonicalizing zero's sign fixed the issue. The displaced HCW scaling factors and
step sizes then matched the serial implementation exactly. A dedicated regression
test now protects that behavior. Earlier exploratory sweep files predate that fix
and are superseded by `cooperative-hcw-2000-final.json`.

The final build passed CUDA Compute Sanitizer `memcheck`, `synccheck`, and
`racecheck`, with zero errors, warnings, or hazards. The full four-family production
test also passed with the automatic execution policy (maximum nonlinear residual
`2.92768852e-8`, zero CPU/GPU trajectory difference).

The fingerprint follow-up, implemented after the main timing table, has its own
isolated benchmark and parity test. Median per-launch kernel time fell from 30.74 us
to 6.01 us for 120,012 entries (5.11x). Across tested sizes from 18,006 to 1,048,576
entries, the isolated ratios were 1.63–5.11x. This is a small whole-solve contribution:
the matched SCvx medians were 69.305 ms before and 68.743 ms after, too close to
promote as a substantial additional end-to-end gain. CPU, legacy GPU, and reduced GPU
hashes match exactly, including signed zeros, infinities, NaNs, and partial blocks.
The fingerprint test passed all three CUDA sanitizer tools with zero errors/hazards.
The four-family production test subsequently passed with independent CPU fingerprint
checks enabled for every family (maximum nonlinear residual `2.92768854e-8`).
Reproduce with `numeric_fingerprint_test --benchmark`; raw logs and a matched JSON
study are in `build/performance/fingerprint-kernel-benchmark.log` and
`artifacts/performance/fingerprint-hcw-2000-comparison.json`.

## SCvx metrics and HCW replay follow-up

The next measured tranche distributes interval/node metrics across up to 128 GPU
blocks. FP64 tree reductions sum objective/merit values and take maxima for all
constraint metrics. Partial results use retained driver storage; no allocation or
host reduction occurs inside the iteration. Terminal sums have one owner.

HCW replay retains the exact ZOH recurrence and its original accumulation order.
Six lanes own the state components, exchange the preceding state through warp
shuffles, and reuse the constant transition/control matrices. Time steps remain
ordered; this is not a parallel prefix approximation. Nonlinear replay is unchanged.

Local RTX 5090 measurements compare against an immutable library from `769b422`:

| Measurement | Before | After | Ratio |
|---|---:|---:|---:|
| HCW SCvx wall time, 2,000 intervals | 67.759 ms | 41.305 ms | 1.64x |
| HCW SCvx wall time, 10,000 intervals | 326.879 ms | 194.086 ms | 1.68x |
| HCW reported replay/metrics phase, 2,000 intervals | 28.886 ms | 2.096 ms | 13.78x |
| Complete HCW command, 2,000 intervals | 474.621 ms | 423.636 ms | 1.12x |
| Isolated HCW replay, 2,000 intervals | 5.385 ms | 0.513 ms | 10.50x |

The isolated metrics pass improved by 175–508x at 2,000 intervals across the four
families, and 882–2,531x at 10,000. These kernel ratios are not whole-mission gains.
Whole-SCvx measurements still use the already-feasible HCW fixture: one inner
iteration, one outer iteration, zero accepted steps, and zero residuals. Each
matched study alternates order with two warmups and seven measured repetitions.
Metrics alone reduced the 2,000-interval SCvx median from 67.097 to 54.769 ms.

Validation includes serial/parallel comparison of all 21 metrics with nonzero
defects and violations, all four models, partial blocks, and virtual control both
enabled and disabled. Maximum metrics match exactly; sum differences must stay
within 3e-12 relative tolerance. Replay matches every state exactly through 10,000
steps for three step sizes, nonzero controls, zero trajectories, and empty horizons.
Both tests passed Compute Sanitizer memcheck, synccheck, and racecheck with zero
errors/hazards. Variational dynamics and allocation/pointer/stream tests also pass.

The final four-family production acceptance test, including CPU fingerprint checks,
passed with maximum canonical residual 9.56640559e-9, nonlinear residual
2.92768849e-8, zero CPU/GPU trajectory difference, and maximum coefficient difference
2.7599450502791e-13. Displaced HCW retained three accepted steps and 1,050 inner
iterations. Recovery counters remain subject to the previously documented bug.

Reproduce kernel tests with `scvx_metrics_test --benchmark` and
`hcw_replay_test --benchmark`. Raw results are in `build/performance/`; matched
studies are `artifacts/performance/metrics-hcw-2000-comparison.json`,
`metrics-replay-hcw-2000-comparison.json`, and
`metrics-replay-hcw-10000-comparison.json`. Source and binary hashes are recorded
by the benchmark runner (the first metrics-only study predates header hashing).

Inspection after these measurements found that the standalone CQP residual API
still launches the old single-block evaluator after the cooperative solve. Its time
is also absent from `residual_seconds`. This is the next concrete target before
attributing the remaining wall time to CPU orchestration.

## Work still required by the active goal

1. Scale the large-trajectory measurements and tune operator ownership, especially
   replacing CSC forward atomic scatter with retained gather/trajectory operators.
2. Parallelize recovery and improve convergence. The powered-descent validation
   fixtures still spend substantial time in recovery; the large zero-HCW setup gain
   does not address that bottleneck.
   Recovery's existing non-cancelled report also substitutes the requested iteration
   limit for completed PDHG iterations; correct that telemetry before using difficult
   recovery runs for a convergence cost model.
3. Parallelize the standalone CQP residual evaluator, repair its timing, and remove
   repeated diagnostic synchronization. SCvx metrics and fingerprints are now
   parallel; nonlinear trajectory replay and device outer decisions remain.
4. Complete device-resident outer-loop decisions and the remaining GTOC12 native GPU
   refinement path. CPU Clarabel reuse is not the production destination for this goal.
5. Remove/cache host-side QOCO conversion as part of a measured GPU-native backend
   strategy. Backend choice must preserve the same independently verified accuracy.
6. Add trajectory batching and graph execution where their measured benefit justifies
   them. Cooperative residency and cancellation must remain correct.

The implementation is not yet a completely GPU-resident end-to-end mission planner.
