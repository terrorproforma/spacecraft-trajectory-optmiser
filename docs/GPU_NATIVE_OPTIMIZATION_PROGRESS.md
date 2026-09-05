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
still launched the old single-block evaluator after the cooperative solve. Its time
was also absent from `residual_seconds`.

## Standalone CQP residual follow-up

The residual API now recomputes the current resident iterate with the cooperative
evaluator, using the retained reduction buffers and the parallel preamble's grid
policy. Occupancy validation includes this kernel. Tiny problems and the explicit
zero-block override retain the legacy evaluator. Both wait and diagnostic polling
collect a dedicated CUDA-event timer; a new solve clears the preceding residual time.

Against the preceding metrics/replay build (`f876abb`), the matched 2,000-interval
HCW SCvx median fell from **41.456 ms to 5.023 ms (8.25x)**. The residual phase now
reports 0.159 ms. The old zero measurement did not mean the residual check was free.
Reported CQP totals are consequently not directly comparable across these builds:
the new total includes a phase that was previously omitted. Complete-command
medians were both about 429 ms in this study; process startup and the surrounding
test harness are much larger than the optimized SCvx operation.

At 10,000 intervals, the corresponding SCvx medians were 195.420 ms and 13.437 ms
(14.54x); complete-command medians were 868.729 ms and 720.622 ms (1.21x).
The measured residual phase was 0.187 ms. This study is recorded in
`artifacts/performance/standalone-residual-hcw-10000-comparison.json`.

Regression tests recompute all reported residual components at both converged and
deliberately infeasible resident points. They compare legacy and cooperative modes,
retain iteration/termination counters, verify allocation stability, and check timing
reset between solves. The extended cooperative suite passed normal execution and
CUDA memcheck, synccheck, and racecheck with zero errors/hazards. The matched record
is `artifacts/performance/standalone-residual-hcw-2000-comparison.json`; validation
logs are in `build/performance/native-checks/standalone-residual-*.log`.

The final four-family production run passed with maximum canonical residual
9.56640559e-9, nonlinear residual 2.92768846e-8, zero CPU/GPU trajectory difference,
and exact CPU/GPU numeric fingerprints. Displaced HCW again accepted three steps
with 1,050 inner iterations. The 6DOF fixture used one recovery attempt in this
run rather than two in the preceding build; its returned trajectory still matched.
No recovery speedup is claimed from this unpaired change in the numerical path.

## Final matched original-library checkpoint

The complete set of GPU changes was also measured directly against the saved
pre-optimization library, using the same executable, two warmups, seven measured
repetitions, and alternating order on the local RTX 5090:

| 2,000-interval HCW measurement | Original | Current |
|---|---:|---:|
| Complete command, including startup and test harness | 5.172190 s | 0.423660 s |
| SCvx wall time | 4.746820 s | 0.005080 s |

This is **12.21x for the complete command** and approximately 934x for the SCvx
operation on this setup-heavy, already-feasible fixture. Every sample retained one
inner iteration, one outer iteration, zero accepted steps, and zero canonical,
nonlinear, and CPU/GPU trajectory residuals. These are not general mission-planning
speedup multipliers. Raw samples and provenance are in
`artifacts/performance/gpu-native-hcw-2000-final.json`.

## Objective and recovery-counter correctness

Both CQP evaluators now form the objective from Qx before adding transpose-dual
contributions to the stationarity gradient. A regression evaluated the same resident
point independently as c'x + x'Qx/2: the old report returned about -0.5 where the
correct value was -0.6666666823. It failed before the fix and passes afterward for
legacy and cooperative modes, including deliberately infeasible points.

Recovery now preserves the PDHG iteration count on every exit and reports completed
projection iterations separately. The rejected-recovery regression previously
substituted its 350,000 budget for 300,000 completed PDHG steps. Production 3DOF now
reports 300,000 PDHG steps plus 50,000 recovery steps, rather than 1,000,000 PDHG
steps. These corrections do not loosen tolerances or skip work.

The recovery suite, analytic objective tests, standard native tests, and all three
CUDA sanitizer tools pass. The four-family production check passes with maximum
canonical residual 9.56640559e-9, nonlinear residual 2.92768853e-8, zero CPU/GPU
trajectory difference, and exact numeric fingerprints. Logs are in
`build/performance/native-checks/*objective*` and `*recovery-count-regression*`.

## Bounded early GPU recovery refinement

Device-clock phase profiling found that over 99.8% of 3DOF recovery time was in
the fixed 50,000-step projected-gradient loop. Feasibility-only probes, extra dual
reconstruction, and a bounded momentum experiment did not improve qualification;
those experiments were discarded.

The retained candidate tries full KKT refinement after 100 projection steps. It
allows four primal corrections, with each dual reconstruction capped at eight CGLS
restarts of at most 256 iterations. Acceptance requires a finite objective and the
complete natural residual below 90% of the original requested tolerance. Failure
restores the saved primal, dual, and projection step before continuing the original
recovery algorithm. Cancellation retains the original transactional rollback.
Primal refinement is shared with the final recovery phase rather than duplicated.
Two trial buffers are allocated once; all numerical work and decisions stay on CUDA.

The separate cached profile API reports device-clock cycles, including failed
certificate work, without additional device transfers or allocations when read.
The original diagnostics ABI remains unchanged. Recovery iteration counts describe
completed projected-gradient steps; refinement cost is included in recovery time
and separately profiled, not disguised as free work.

Matched local RTX 5090 measurements used the same native executable, the
correctness-fixed pre-refinement library, alternating order, one warmup and three
measured repetitions per variant:

| 2-interval 3DOF production fixture | Before | Bounded refinement |
|---|---:|---:|
| SCvx wall time | 17.775993 s | 5.246444 s |
| Recovery time | 13.041146 s | 0.043037 s |
| Complete process | 18.148748 s | 5.628011 s |
| Completed PDHG steps | 300,000 | 300,000 |
| Completed recovery projection steps | 50,000 | 100 |

This is 3.39x for SCvx and 3.22x for the full process on this fixture, with a 303x
reduction in the recovery phase. Every sample met the 1e-8 inner request (maximum
reported residual 1.42323e-10), retained objective 0.494783333, and had zero returned
trajectory residuals and CPU/GPU trajectory difference. This fixture accepts no
outer steps: the inner CQP qualification was checked explicitly, separately from
the returned reference trajectory. The measurements do not establish equivalent
gains on larger problems or complete missions. Raw samples and binary/source hashes:
`artifacts/performance/recovery-bounded-refinement-pd3.json`; reproducible runner:
`scripts/gpu/benchmark_recovery.py`.

Validation passed the recovery, cooperative solver, CW/SOC, allocation, pointer,
and stream suites. The early-success production path passed CUDA memcheck,
synccheck, and racecheck with zero reported errors or hazards. The four-family
production check, including a displaced HCW trajectory with accepted steps and
independent CPU coefficient fingerprints, retained maximum canonical residual
9.56640559e-9, nonlinear residual 2.9276885e-8, zero CPU/GPU trajectory difference,
and maximum coefficient difference 2.75994505e-13. Logs:
`build/performance/native-checks/*bounded-recovery*`.

The equivalent 6DOF comparison was effectively flat: median SCvx time 48.472280 s
before versus 48.483291 s after; recovery 32.682183 s versus 32.648313 s. All measured
samples performed 600,000 PDHG and 100,000 recovery projection steps and still
missed the requested inner tolerance. Returned trajectory qualification and
objective were unchanged. One optimized warmup qualified the inner problem after
one recovery instead of two, showing why its shorter time must not be mixed into
the measured comparison or described as a speedup. The bounded failed probe used
about 0.4% of the last recovery projection phase's clock cycles. Six-DOF inner
convergence remains unresolved. Samples:
`artifacts/performance/recovery-bounded-refinement-pd6.json`.

A separate optimized 20-interval 3DOF production probe did not return metrics before
its 120-second process limit. No qualification or speedup is claimed for that
larger case, and a paired baseline has not been measured. Its empty output log is
`build/performance/native-checks/bounded-recovery-pd3-20.log`. Instrumenting progress
inside that solve is required before further scaling claims.

## Parallel cone projection and recovery reductions

The single-block execution strategy still assigned all cone projections to thread
zero, including medium-size problems selected by the automatic policy. Independent
cones now have separate thread owners in PDHG and recovery. Validated topology
guarantees disjoint cone ranges, and component arithmetic within each cone retains
its original order. Existing block barriers protect producers and consumers.

The six repeated CGLS sum-of-squares calculations in recovery now use FP64 warp
reductions and eight shared warp sums. Vectors with at most 32 elements retain the
serial path. These changes add no allocations, transfers, or host decisions and do
not change tolerances or the final qualification gate.

A new diagnostic mode runs the actual production CQP with an explicit iteration
budget and cooperative deadline cancellation, without running the surrounding
outer loop:

```text
device_scvx_integration_test --production-cqp pd3 20 350000 40
```

Diagnostic completion is not qualification. The output includes termination,
objective, natural residual, PDHG/recovery counts, wall time, and recovery phase
cycles. Before parallelisation, this 20-interval CQP reached 300,000 PDHG steps and
5,400 recovery steps before cancellation at 40 seconds. Afterward it completed
300,000 PDHG and all 50,000 recovery steps in 34.623 seconds. Both returned the same
PDHG residual, 0.00056897551530710189, after recovery rollback. The larger CQP is
still unqualified at its 1e-8 request; these observations establish faster work,
not a qualified larger solve. Logs: `cqp-probe-pd3-20-before.log` and
`cqp-probe-pd3-20-parallel.log` under `build/performance/native-checks/`.

The deadline probe also exposed stale PDHG reports on cancellation between scheduled
residual checks. Both execution strategies now evaluate the returned resident point
and report the actual completed iterations at cancellation. The regression uses a
long check interval and independently evaluates the returned primal's objective,
then recomputes its residual. Before the fix it caught a reported residual of zero
where the returned point's residual was one; the normal regression passes after
the fix. This adds work only on the cancellation exit.

Fixed-work sweeps (one warmup, three measured repetitions, alternating order)
also exposed an overly conservative dispatch cutoff. For the 507-variable,
20-interval 3DOF CQP, 1,000 steps took 63.54 ms in the automatic block-local loop
versus 30.83 ms with two cooperative blocks. Five intervals favoured the legacy
loop (21.00 ms versus 24.72 ms with two blocks); ten intervals favoured two blocks
(34.82 ms versus 28.59 ms); fifty favoured four/eight blocks (159.19 ms legacy,
34.01/32.41 ms). Objectives and residuals agreed within 1e-9 in these fixed-work,
unqualified diagnostic runs. Raw data: `artifacts/performance/medium-dispatch-screen*.json`.

Automatic execution now retains the legacy loop below 256 variables and uses
roughly one cooperative block per 256 variables above that threshold. The prior
dense-large-operator policy is unchanged. This heuristic remains overridable for
different GPUs and sparsity patterns.

Matched qualification measurements on the local RTX 5090:

| Measurement | Before | Current | Change |
|---|---:|---:|---:|
| 2-interval 3DOF SCvx | 5.115960 s | 4.020861 s | 1.27x |
| 2-interval 3DOF recovery | 43.114 ms | 29.679 ms | 1.45x |
| 2-interval 3DOF complete process | 5.588654 s | 4.405835 s | 1.27x |
| Displaced 50-interval HCW SCvx, dispatch change only | 64.503 ms | 38.969 ms | 1.66x |
| Displaced 50-interval HCW complete process | 446.063 ms | 403.257 ms | 1.11x |

The 3DOF comparison used the d36c5c6 library, one warmup and five measured samples;
all samples retained 300,000 PDHG steps, 100 recovery steps, objective 0.494783333,
and inner qualification at 1e-8. The HCW comparison isolated dispatch against
the saved parallel-cones/reductions library: two warmups and seven measured samples,
with a common 12-outer-step protocol, 882 inner iterations and six accepted steps
in every sample. Both comparisons passed unchanged trajectory qualification,
objective-equivalence and CPU/GPU trajectory checks. Raw samples and hashes:
`parallel-recovery-pd3.json` and `medium-dispatch-hcw-50.json` in `artifacts/performance/`.

With the new dispatch, the full 20-interval 3DOF production probe returned in
51.351 seconds, instead of hitting the earlier 120-second process limit. It spent
600,000 PDHG and 100,000 recovery steps, returned the qualified unchanged reference
trajectory, and still missed the inner tolerance (5.58219e-4 versus 1e-8). It
accepted no outer steps. This is not evidence that the larger optimization problem
is solved. Log: `build/performance/native-checks/parallel-medium-production-pd3-20.log`.

The cone and recovery changes passed CUDA memcheck, synccheck and racecheck,
including mixed SOC/rotated-SOC problems and early recovery success. The cancellation
regression passed all three tools as well. Native solver, rollback/cancellation,
allocation, pointer and stream checks pass. Four-family production qualification
with independent CPU coefficient fingerprints passed after the kernel changes;
maximum canonical residual 9.56640559e-9, nonlinear residual 2.92768851e-8,
CPU/GPU trajectory difference zero, coefficient difference 2.75994505e-13.
Logs are under `build/performance/native-checks/` with `parallel`, `cancelled-report`
and `medium-dispatch` prefixes.

## Work still required by the active goal

1. Scale the large-trajectory measurements and tune operator ownership, especially
   replacing CSC forward atomic scatter with retained gather/trajectory operators.
2. Parallelize recovery and improve convergence. The powered-descent validation
   fixtures still spend substantial time in recovery; the large zero-HCW setup gain
   does not address that bottleneck.
   The objective and completed-iteration counters are now corrected for subsequent
   convergence experiments; earlier reports remain unsuitable as work counts.
3. Remove repeated diagnostic synchronization. Standalone CQP residual evaluation,
   SCvx metrics, and fingerprints are now parallel; nonlinear trajectory replay
   and device outer decisions remain.
4. Complete device-resident outer-loop decisions and the remaining GTOC12 native GPU
   refinement path. CPU Clarabel reuse is not the production destination for this goal.
5. Remove/cache host-side QOCO conversion as part of a measured GPU-native backend
   strategy. Backend choice must preserve the same independently verified accuracy.
6. Add trajectory batching and graph execution where their measured benefit justifies
   them. Cooperative residency and cancellation must remain correct.

The implementation is not yet a completely GPU-resident end-to-end mission planner.
