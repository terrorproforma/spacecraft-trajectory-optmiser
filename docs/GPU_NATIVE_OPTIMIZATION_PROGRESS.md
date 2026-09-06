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

## GPU IPM checkpoint: scoped handles and ordered sparse products

The optional local QOCO CUDA backend now supports scoped cuBLAS handles, GPU-built
ordered sparse gathers, a corrected relative stopping scale, deterministic cuDSS
factorization, and checked cuDSS function signatures/runtime versions. A matched
20-interval displaced landing benchmark isolates handle reuse: median SCvx
1.474888 -> 0.623454 seconds (2.37x), complete process 1.929623 -> 1.045398 seconds
(1.85x), with identical 54 inner iterations, two accepted steps, objective and
final residuals in every sample. Qualification remains at 1e-8.

Operator memcheck/initcheck/synccheck/racecheck and the complete landing
memcheck/initcheck/synccheck pass. Full racecheck reports hazards inside cuDSS
0.7.1.6's deterministic factorization; an isolated 0.8.0.10 upgrade instead fails
inside its deterministic forward solve. These findings remain open, and the
backend is not promoted automatically. The pinned upstream and remote campaigns
are untouched. [Implementation, raw evidence and reproduction instructions](GPU_QOCO_LOCAL_BACKEND.md)
describe the measured improvement, rejected experiments and remaining CPU work.

## GPU certificate and dual handback checkpoint

The QOCO adapter's KKT residual certificate and dual transformation now run on
CUDA with retained topology, parallel sparse gathers and deterministic reductions.
The prepared backend supplies resident device solutions; legacy backends use
retained upload buffers. Independent long-double reference tests and all four
CUDA sanitizer tools pass for the new kernels. Real landing solves pass a CPU
oracle, repeatability and full-solve memory/synchronization checks at unchanged
1e-8 accuracy. The known cuDSS race finding remains open.

This migration does not add a measured speedup on the 20-interval landing:
paired medians are 660.848 ms before and 668.247 ms after (54 inner iterations,
two accepted steps throughout). It removes CPU numerical work; it does not yet
remove upstream solution downloads, host warm-start state, conversion, scaling,
KKT assembly, or outer decisions. Native memory exports now include driver and
audit-owned allocations and explicitly exclude opaque QOCO/cuDSS memory. See
[implementation and evidence](GPU_QOCO_LOCAL_BACKEND.md#device-residual-audit-and-dual-mapping).

## Queued GPU operators and device cone reductions

The next optional backend queues vector/sparse kernels within a checked solve
scope, reuses cone scratch buffers, and finishes line-search reductions on CUDA.
It also corrects a residual reduction that ignored partial blocks beyond 1024
and SOC line-search branches that could leave the cone or stall unnecessarily.
Independent feasibility and large-grid regressions fail on the prior code and
pass on the candidate. New kernels pass all four CUDA sanitizers.

Matched local RTX 5090 benchmarks (two warmups/seven measured samples) preserve
the same independent accuracy gates: the 1e-8 landing falls 617.517 → 242.203 ms
(2.55x, 54 → 28 inner iterations); the actual 40-interval 3DOF planner example
falls 638.399 → 426.664 ms (1.50x), and the 20-interval 6DOF planner example falls
2045.458 → 1684.074 ms (1.21x). Both planner examples certify at their unchanged
1e-6 tolerance with matching objectives and independent replay. Their complete
process gains are 1.27x and 1.19x respectively. These are not universal gains.

Full landing memory/initialization/synchronization checks pass, but full racecheck
still reports 30 hazards inside cuDSS's deterministic factorization. An alternative
factorization-mode probe fails the unchanged qualification and is rejected.
The optional candidate is not promoted to the default production dependency.
[Details, regressions and benchmark artifacts](GPU_QOCO_LOCAL_BACKEND.md#queued-operators-device-cone-reductions-and-soc-step-safety)
record the improvements and remaining CPU/dependency work. The full goal remains active.

## Retained topology checkpoint

The adapter now retains sparse topology and checks it exactly on CUDA instead
of repeatedly downloading indices. Numeric downloads share a stream completion,
and an optional QOCO values-only update retains device indices and gather maps.
New kernels and updated operators pass all four sanitizers; real landing solves
pass the independent CPU audit, memory, initialization and synchronization checks.
The previously reported cuDSS factorization race finding remains unresolved.

Paired benchmarks show essentially flat incremental runtime: landing SCvx
260.056 → 255.066 ms, actual 6DOF planner 1662.863 → 1657.908 ms; complete
processes are slightly slower. Accuracy and iteration counts are unchanged.
The change reduces repeated topology transfers, with extra initial upload and
retained memory, and does not establish another speedup. A stricter
nondeterministic factorization probe passed ten solves then failed the eleventh
at the unchanged physics gate and was rejected.

[Evidence and remaining work](GPU_QOCO_LOCAL_BACKEND.md#retained-topology-and-values-only-updates)
separate these results from the earlier measured gains. CPU numerical conversion,
equilibration, KKT updates and outer control still need migration; the full
GPU-native goal remains active.

## CUDA numerical conversion checkpoint

Repeated QOCO numerical conversion now runs on CUDA using compiled, retained
maps. Ordered gathers handle duplicates, bound signs, SOC/rotated-SOC transforms
and right-hand sides; GPU checks reject invalid numerical inputs or changed bound
patterns. The GPU KKT audit consumes the resident converted values. Initial
structure discovery and the QOCO host update interface still use CPU storage;
equilibration, KKT assembly and outer/scalar decisions remain to be migrated.

Independent arithmetic, mixed-cone/duplicate-entry mutation tests, CPU conversion
and KKT oracles pass. New kernels pass all four sanitizers; full landing and
synthetic adapter memory/initialization/synchronization checks pass. The known
cuDSS race remains unresolved. Paired landing SCvx is 258.174 → 249.038 ms and
6DOF planner 1657.412 → 1635.301 ms, with identical iterations and physics results.
These are modest measured changes; initial map setup, transfers and memory grow.
[Full evidence and limitations](GPU_QOCO_LOCAL_BACKEND.md#compiled-cuda-numerical-conversion)
are recorded. The complete GPU-native goal remains active.

## Device numerical updates and GPU Ruiz checkpoint

The optional QOCO backend now consumes converted coefficients directly from GPU
storage and performs initial and repeated requested Ruiz equilibration on CUDA.
The adapter skips repeated converted-array downloads. Tests also exposed and
fixed pinned-backend cost-scaling, transpose-permutation and inserted-diagonal
update bugs. Numerical KKT updates already ran on CUDA; initial KKT assembly,
structure discovery and scalar/outer control still require migration.

The new update kernels pass all four CUDA sanitizers, including racecheck.
Native Ruiz-4 and real landing memory/initialization/synchronization checks pass,
as do independent CPU conversion/KKT oracles and unchanged physics gates. The
existing cuDSS factorization race remains unresolved; the backend stays optional.

Final matched landing SCvx is 259.198 → 259.853 ms and the 6DOF planner is
1675.632 → 1667.908 ms: essentially flat overall. The landing's repeated update
phase improves 1.137 → 0.488 ms and native D2H traffic falls 27.1%; these counters
exclude backend-owned traffic and memory. Iterations and physics results are
unchanged. Mission timings use Ruiz 0; synthetic tests validate nonzero Ruiz.
[Detailed evidence and remaining work](GPU_QOCO_LOCAL_BACKEND.md#device-coefficient-updates-and-ruiz-equilibration)
separate phase improvements from total runtime. The full goal remains active.

## GPU scalar reduction checkpoint

Infinity norms and minimum-absolute-value reductions now return their result
directly from CUDA, removing the intermediate index download. NaN checks reuse
scalar scratch within the solve. Exact-value tests through 1048579 entries and
all four kernel sanitizers pass; real landing/6DOF qualification and iteration
counts remain unchanged. The known cuDSS factorization race is still open.

Paired local SCvx medians improve 245.661 → 227.428 ms for landing and
1662.882 → 1536.696 ms for 6DOF, about 1.08x each. A separate landing API trace
records 847 → 401 stream synchronizations with unchanged kernel launch count.
[Evidence and scope](GPU_QOCO_LOCAL_BACKEND.md#gpu-scalar-extrema-and-retained-nan-checks)
include raw timings, hashes and sanitizer output. Scalar decisions still return
to the CPU; removing that control path remains part of the full active goal.

## Batched stopping metrics checkpoint

Twelve norms and five dot products now retain their results on the GPU and
return one combined six-scalar packet. Existing sparse operators and cuBLAS dot
products remain in use. Metric parity, independent CPU arithmetic and all four
kernel sanitizers pass, together with unchanged landing/6DOF physics gates.

Matched local SCvx medians improve 237.979 → 184.374 ms for landing (1.29x)
and 1540.333 → 1269.197 ms for 6DOF (1.21x), with unchanged 28/179 iterations.
The landing API trace removes 330 synchronous copies and 150 scalar-result
stream waits. [Full evidence and scope](GPU_QOCO_LOCAL_BACKEND.md#batched-gpu-stopping-metrics)
record measured binaries, raw timings and remaining host decisions. The existing
cuDSS race and the full GPU-native goal remain open.

## Shared iteration scalar checkpoint

Objective and mu now join the GPU stopping packet, sharing the quadratic product
and cost dot product instead of recomputing them. The extended path passes
independent arithmetic, all eight metric comparisons, unchanged physics gates
and all four kernel sanitizers. Final binaries were frozen before full
revalidation and matched measurement.

Final SCvx medians improve 186.770 → 175.702 ms for landing and
1240.639 → 1163.785 ms for 6DOF, about 1.06x each. Complete-process landing time
is flat; complete-process 6DOF improves 1.054x. A separate trace shows 120 fewer
scalar waits and 90 fewer launches per landing solve. [Evidence and limits](GPU_QOCO_LOCAL_BACKEND.md#shared-objective-and-complementarity-calculation)
retain superseded trials as well as the final results. Host control and the
cuDSS race remain unfinished; the full goal stays active.

## GPU step control checkpoint

Cone step lengths now feed GPU centering calculations and one multi-block
kernel updating all four iterate vectors. Independent arithmetic, feasibility
bisection, existing scalar/metric comparisons and landing/6DOF physics gates
pass. All four standalone sanitizers pass; full landing memory, initialization
and synchronization checks pass with and without test oracles.

Matched SCvx medians improve 173.834 → 156.348 ms for landing (1.112x) and
1166.371 → 1032.315 ms for 6DOF (1.130x), with unchanged 28/179 iterations.
Startup-inclusive landing is 1.3% slower; startup-inclusive 6DOF improves
1.095x. One 1.300420 s landing sample is retained in the measurements.
A qualified trace removes 112 synchronous copies, 56 stream waits and 308
launches. [Evidence and limits](GPU_QOCO_LOCAL_BACKEND.md#gpu-centering-and-fused-iterate-updates)
include frozen binary hashes and reproduced source. Two scalar metadata returns,
host control and the known cuDSS factorization race remain; the full goal is active.

## Combined RHS experiments and larger qualification

The optional fused RHS consumes device sigma/mu and cached cone offsets.
A second option pins the workspace and queues sigma metadata, removing 28
synchronous copies from the landing API trace. Both versions pass numerical
oracles, unchanged physics gates and standalone sanitizers; metadata event
completion and pinned-storage lifetime are tested explicitly.

Neither experiment establishes a reliable general speedup. In the final trial,
20-interval landing regresses, 20-interval 6DOF SCvx improves about 1%, and a
matched 500-interval run improves SCvx 1.2% while its solver subphase regresses.
Frozen v25 remains the performance baseline; new options are opt-in experiments.
[Full results and limits](GPU_QOCO_LOCAL_BACKEND.md#combined-rhs-experiments-no-general-promotion)
retain both candidates, regression samples and 100/500-interval qualification.

The 500-interval case passes independent conversion/KKT/RHS/metric comparisons
with the same 34 inner iterations. Its baseline QOCO setup median is 344 ms,
larger than the 253 ms solve. Split setup into measured stages next; the whole
GPU-native goal and existing cuDSS race remain open.

## Setup ordering and lifetime checkpoint

Split setup profiling identifies about 232 ms of CPU vendor reordering at 500
intervals. A GPU static-degree ordering prototype passes physics gates but
regresses the large solve substantially, so it remains rejected. The next
ordering design needs explicit trajectory structure and must measure total
factorization/solve cost. Diagnostic fences and retained outliers prevent
treating the profiles as matched speedup measurements.

The v31 ownership correction frees temporary host KKT storage, retains CSR
indices for the vendor matrix lifetime, and creates dense wrappers before
analysis. A focused allocation tracker confirms elimination of 5,315,104 leaked
host bytes per 500-interval setup. Matched timings are effectively flat, with
unchanged physics gates and 28/179/34 iterations; this is a correctness fix,
not a speedup claim. Independent numerical oracles and memory/initialization/
synchronization checks pass. Full racecheck still reports 30 cuDSS factorization
errors, which remain unresolved.

[Measured results, rejected experiment and reproduction details](GPU_QOCO_LOCAL_BACKEND.md#setup-profiling-rejected-gpu-ordering-and-ownership-fixes)
include frozen binaries and full sanitizer output. The complete goal stays active.

## GPU separator-tree checkpoint

CUDA now generates a trajectory permutation and separator sizes from explicit
device index maps. Supplying the tree as well as the permutation avoids the
documented loss of cuDSS factorization parallelism. The native adapter retains
its own GPU metadata; forced-failure reconstruction passes after the original
maps and stream are released.

In matched RTX 5090 runs, 500-interval 6DOF SCvx improves **745.262 → 553.157 ms
(1.347x)**, and the complete process improves **1.223 → 1.048 s (1.167x)**.
All physics gates and tolerances agree, with unchanged 34 inner iterations.
Setup falls from 356.258 to 135.697 ms; solving itself regresses about 16%.
Both small fixtures regress, so no general/default backend promotion is made.

Independent graph/tree tests and all four standalone sanitizers pass. Full
numerical oracles and memory/initialization/synchronization checks pass.
Full vendor racecheck still fails: 38 cuDSS factorization reports versus 30
with the reference ordering. This remains an experimental large-case candidate.
[Results, frozen artifacts and limits](GPU_QOCO_LOCAL_BACKEND.md#gpu-trajectory-ordering-and-separator-trees)
retain intermediate failures and matched measurements. CPU KKT assembly, host
control, further factorization/ordering work and the full GPU-native goal remain open.

## Factorization and runtime hill climb: rejected variants

Eight further frozen local builds (v48–v55) do not earn promotion. Explicit
multiblock factorization on cuDSS 0.7.1 introduces an initialization error in
vendor setup. Standard kernels lose 6DOF qualification. An isolated cuDSS
0.8.0.10 trial reveals a deterministic forward-solve memory error with both
trajectory and vendor ordering; multiblock factorization and superpanels do
not remove the observed 6DOF crash.

The 0.8 standard-kernel candidate initially passes both 6DOF sizes, independent
numerical oracles, seven landing repetitions, and all four full landing
sanitizers. Its matched landing SCvx median improves 194.813 → 151.352 ms
(1.287x versus v42), but its first measured-campaign 6DOF warmup fails:
terminal residual 4.517e-6 exceeds the unchanged 1e-6 certificate tolerance.
The benchmark stops and the candidate is rejected. Initial qualification and
landing-only speed do not establish reliable performance for the whole pipeline.

The existing reference remains unchanged. Benchmark tools now select and hash
each variant's cuDSS runtime explicitly. Optional factorization/superpanel
switches and 0.8 tree enum compatibility support reproducible experiments;
they are not default changes. The next target is numerical conditioning and
SCvx progress sensitivity, alongside the remaining host numerical work.
[Evidence and exact scope](GPU_QOCO_LOCAL_BACKEND.md#factorization-and-cudss-runtime-variants)
retain passing probes, failed repetitions and vendor error reports. The goal is active.

## Per-solve recovery-state isolation

The next investigation finds a concrete state-reuse bug: QOCO retains its best
iterate and progress metric across solves, although SCvx changes the coefficients
and scaling. A stalled solve can restore the previous problem's trajectory.
Patched preparation now invalidates that history and resets iteration counters
at every solve, retaining GPU allocations and explicit primal warm starts. A
regression test changes an equality right-hand side, forces iteration-limit
recovery, and fails on the old library while passing with the patch at zero and
four Ruiz iterations. Pure-QOCO forcing telemetry now uses the same threshold
as its actual acceptance gate.

The patched standard-kernel cuDSS 0.8 candidate passes repeated qualification
and matched campaigns on landing and 20/500-interval 6DOF. Relative to v42, SCvx
medians improve **1.238x / 1.451x / 1.188x** respectively. This does not establish
a uniformly faster backend: the 20-interval mean regresses from 1.173 to 1.252 s,
with a 2.887 s worst measured run. The state reset is retained as a correctness
fix; the runtime configuration remains experimental and no backend default
is changed. The full timing distributions, rejected refinement/regularization/
cold-start hypotheses and explicit sanitizer scopes are in the
[solve-state checkpoint](GPU_QOCO_LOCAL_BACKEND.md#per-solve-recovery-state-isolation).
Host numerical work, runtime variability and the complete GPU-native goal remain open.

## Device solution ownership and host Ruiz consistency

An opt-in QOCO extension now keeps completed solutions and accepted primal warm
starts on the GPU. The native adapter no longer downloads four solution vectors,
copies the primal on the CPU, and uploads it for the next start. Explicit host
export remains available for independent test oracles and legacy callers.
Accepted-start copies are included in telemetry even when no further solve occurs.

Non-unit scaling tests also expose a separate correctness bug: legacy host Ruiz
equilibration uploads scaled matrices but leaves objective and right-hand-side
vectors stale on the GPU. The old library reports solved at x=1.04427 for 2x=2.
Patched preparation now publishes all scaled vectors, giving x=1 to roundoff.
The production device-scaling path already bypasses this host calculation.

The final matched batch observes landing SCvx 188.758 → 186.763 ms and N20 6DOF
1.171 → 1.028 s. These do **not** establish a reliable general speedup: an earlier
batch regresses N20, and both N500 campaigns stop at the unchanged 1e-8 objective
comparison gate despite passing every physics certificate gate. Repeating the
unchanged control also exposes objective spread above that limit. No tolerance
or backend default is changed. Device IO remains an explicit preparation option;
the host Ruiz synchronization fix applies to all patched builds.

Standalone ownership/scaling regressions, independent numerical comparisons and
the documented full/native sanitizer scopes pass. A final landing API trace
shows 564 → 558 synchronous copies with unchanged kernel counts: nine solution/
warm-start copies removed, three setup copies added by the correctness fix.
[Frozen binaries, complete distributions, failures and exact test scopes](GPU_QOCO_LOCAL_BACKEND.md#device-solution-ownership-and-host-ruiz-consistency)
are retained. Conditioning, convergence variability, CPU setup/control and the
rest of the GPU-native goal remain active work.

## GPU KKT CSR assembly

QOCO can now construct its upper-triangular KKT CSR matrix and all five update
maps directly on CUDA (`--gpu-kkt`). This replaces the CPU KKT constructor,
CPU CSC-to-CSR conversion and serial map conversion/upload loops. Per-entry
kernels, prefix scans and stable integer-key sorting preserve the original
matrix entries and their update identities. Temporary arrays share an aligned
GPU allocation; the sort/scan workspace uses a second allocation.

Exact CPU-reference comparisons cover empty blocks/rows, duplicates, pure and
mixed cones, a 257-dimensional cone and 513 cones. Full trajectory qualification,
independent numerical oracles, memory/synchronization checks and forced solver
reconstruction pass at their documented scopes. Both complete benchmark batches
pass the unchanged physics and objective-comparison gates.

This is architectural progress, not a general speedup claim. The latest batch
regresses landing SCvx by about 7%, improves N20 with substantially fewer inner
iterations, and leaves N500 nearly flat. The earlier GPU batch regresses N20.
An isolated 15-sample comparison of pooled versus separate GPU temporaries
observes 163.307 → 146.647 ms landing SCvx, but setup stays approximately 30.03 ms;
that does not establish faster KKT assembly itself. The API trace does confirm
eight removed uploads and eight fewer allocations/frees than the first GPU
implementation. No backend default or accuracy tolerance changes.

[Complete evidence and remaining CPU work](GPU_QOCO_LOCAL_BACKEND.md#gpu-kkt-csr-assembly)
include all distributions, construction parity checks and reconstruction tests.
Initial matrix setup/transposes/regularization, host solver control, production
independent replay and convergence variability remain under the active goal.

## GPU numerical-update setup maps

The QOCO device updater now borrows the matrix-owned GPU transpose ordering
instead of constructing and uploading duplicate CPU inverse maps. CUDA also
builds its P source/diagonal maps and copies existing device cone boundaries.
The qualified landing API trace removes five synchronous uploads and two
allocations, while adding three kernels and a small validation readback.

Exact map and numerical oracles, nine extended update fixtures, full trajectory
qualification and scoped sanitizer/reconstruction tests pass. Full end-to-end
and setup timing remains mixed, including slower landing and N500 setup; N20's
faster solve has a different iteration count. The change remains within the
optional GPU updater and does not establish a general speedup or default
promotion. [Evidence, timings and ownership scope](GPU_QOCO_LOCAL_BACKEND.md#device-numerical-update-setup-maps).

## Scratch-vector arena experiment: promotion rejected

An optional GPU arena now allows 26 post-analysis QOCO scratch vectors to share
one allocation and one device zero-fill. The landing API trace confirms 26
fewer uploads and 25 fewer allocations/frees. Exact initialization, alignment,
aliasing and multi-solver lifetime tests pass, as do the documented numerical
oracles and sanitizer scopes.

The candidate nevertheless fails matched-quality qualification: an N20 objective
differs from the reference by 3.68e-7, above the unchanged 1e-8 gate. A separate
20-run-per-build diagnostic qualifies the unpooled control 20/20 and the arena
19/20. Every physics certificate passes, but the repeated objective failure
prevents promotion. Landing's observed 1.051x SCvx ratio does not establish an
acceptable overall improvement. The arena remains an explicitly selected
experiment; default allocation behavior is unchanged. Numerical sensitivity
and conditioning need investigation before this path can be qualified.
[Full failed-run evidence](GPU_QOCO_LOCAL_BACKEND.md#experimental-gpu-scratch-vector-arena).

## Best-iterate return does not eliminate the outlier

An opt-in QOCO policy now restores the saved best point on qualified inaccurate
exits. Its direct audit checks all four solution vectors in physical units.
However, the longer N20 comparison qualifies the unpooled policy only 19/20,
versus 20/20 for its control and pooled counterpart. Every physics certificate
passes, but the unchanged 1e-8 objective gate rejects the outlier. This policy
has not established a reliable speedup or fixed the numerical sensitivity.

The failed trajectory shows a concrete outer-loop issue: rejected inner solves
shrink the trust radius below the step tolerance, so a step at the trust-region
boundary is mistaken for convergence. Final reporting also loses the accepted
step by measuring the returned trajectory against itself. The next correction
targets that stopping logic. [Evidence and limitations](GPU_QOCO_LOCAL_BACKEND.md#inaccurate-exit-best-iterate-experiment).

## Prevent false convergence after trust-region shrinkage

The outer driver now requires a small accepted step to be inside the trust
region's near-boundary threshold before declaring convergence. Final replay
preserves that step and cannot overwrite cancellation or an iteration limit
with a false convergence result. A forced cancellation of a feasible HCW
trajectory reproduces the old bug and verifies the correction.

On a targeted small-radius N20 input, the old core passes the unchanged objective
gate in 8/20 runs; the guarded core passes 20/20. Another 20 original-input runs
using the previously failing best-return QOCO policy also pass with the guard.
All physics certificates pass. Four regression tests, the documented numerical
oracles and full small-radius memory/leak checking pass.

The original-input matched benchmarks qualify all 54 samples but show mixed
timing: landing 176.179 → 171.186 ms, N20 570.046 → 648.000 ms, N500 458.284 →
465.257 ms. This is a correctness improvement, not a general speedup. The full
GPU-native goal and inner numerical sensitivity remain open.
[Complete evidence](GPU_QOCO_LOCAL_BACKEND.md#scvx-convergence-at-a-small-trust-radius).

## Constraint transposes now have a CUDA construction path

The optional `--gpu-transposes` path reuses resident GPU ordering to construct
both constraint transposes and their update maps. Exact tests cover unsorted
rows, duplicates, empty matrices, signed zeros and independent ownership after
the source is destroyed. Poisoned source host arrays cannot affect the result.
Numerical oracles, scoped sanitizers and reconstruction checks pass.

Large synthetic construction medians improve by 1.446x at 225,000 entries and
1.988x at 900,000, but 9,000-entry construction regresses. All 54 full benchmark
samples qualify at unchanged accuracy, while end-to-end timings regress or stay
nearly flat. The path remains opt-in; no overall speedup or default promotion
is claimed. Eager compatibility downloads and temporary inverse-map allocations
are visible in the API trace and remain to be removed.
[Detailed evidence](GPU_QOCO_LOCAL_BACKEND.md#constraint-transpose-construction-on-cuda).

## Transpose host mirrors materialize only when accessed

The optional GPU transpose path can now defer unused CPU mirrors and skip
uploads of unchanged device-owned values. Legacy CPU access materializes the
cache, and subsequent GPU updates invalidate cached values. Exact ownership
and cache-refresh tests, numerical oracles, scoped sanitizers and recovery
reconstruction checks pass.

At 900,000 synthetic entries, lazy construction takes 3.086 ms versus 7.111 ms
for eager GPU construction and 13.872 ms for the CPU constructor. A qualified
landing API trace confirms 12 fewer synchronous copies and six fewer kernels.
These improvements remain local to construction: all 54 matched trajectory
samples pass unchanged accuracy, but full solve times are flat or slower.
The path remains opt-in, with no overall speedup claim or backend promotion.
Physical transpose storage and inverse-map downloads still remain, along with
CPU setup statistics, host control and production replay work.
[Evidence and exact test scopes](GPU_QOCO_LOCAL_BACKEND.md#defer-unused-transpose-host-mirrors).

The read-only Lambda follow-up at 2026-09-06 02:09 UTC observes H100 utilization
at 100%, 44°C and 1607 MiB used. G4 has 140 completed groups and ordinal 140
running, with 594 numerical dispositions, 666 timeouts and no contaminated
attempts. The three groups since the previous check all timed out; utilization
alone is not useful-solve throughput. GTOC12 v11 remains complete, with 23 ships,
194 asteroids and 14047.8 kg, passing official and independent verification.
It is not proven optimal. Remote campaigns were left untouched.
[Recorded observation](../artifacts/performance/lambda-status-2026-09-06-followup.json).

## GPU solves no longer need physical constraint transposes

The optional deferred-transpose path retains A/G and constructs physical At/Gt
only for explicit compatibility access. GPU numerical updates invalidate these
views instead of maintaining unused copies. Reference counts preserve sources
through deferred chains and reconstruction. The normal landing trace now has
24 fewer GPU allocations/frees, six fewer copies and ten fewer kernels than the
host-lazy version.

All 54 matched trajectory samples pass unchanged accuracy. SCvx medians are
173.335 → 169.569 ms for landing, 1102.284 → 923.818 ms for N20, and 464.399 →
473.573 ms for N500. N20's inner count changes from 206 to 166, while N500 stays
at 34. Timing is still mixed; no general speedup or default promotion is claimed.
Exact cache/lifetime tests, scoped sanitizers and recovery tests cover both
materialized compatibility access and the production deferred path. An old
test's direct access to an absent empty descriptor was corrected, with its
failure evidence retained. The full GPU-native goal remains open.
[Detailed evidence and limitations](GPU_QOCO_LOCAL_BACKEND.md#remove-physical-transposes-from-the-normal-gpu-solve-path).

## CUDA Graphs reduce repeated stopping-calculation launch overhead

The optional metric-graph path replays the existing GPU calculations for
stopping, objective and complementarity. Scope-owned graphs invalidate when
parameters or buffers change and are released before captured storage. The
host scalar download and stopping decisions remain. A landing trace replaces
1700 host kernel-launch calls with graph replay.

Isolated stopping calculations improve 3.771x at 17 variables, 1.996x at 4103
and 1.077x at 100000. All 90 changing-input comparisons match bit for bit.
Lifecycle fixtures and full landing pass all four sanitizer tools; scoped
N500 and recovery checks also pass. All 54 matched trajectory samples qualify
at unchanged accuracy, but complete timing is mixed: landing 162.590 →
152.054 ms, N20 725.136 → 1159.010 ms, N500 577.496 → 434.837 ms. Inner counts
change significantly on N20/N500, and a 656.422 ms landing outlier is retained.
No general speedup or default promotion is claimed.

Capture still introduces cuBLAS asynchronous allocations, and convergence
variability remains unresolved. Reducing those allocations, moving remaining
host decisions and improving conditioning are further work toward the full
GPU-native goal. [Evidence and exact scopes](GPU_QOCO_LOCAL_BACKEND.md#replay-stopping-calculations-with-cuda-graphs).

## GPU equilibration: repair zero norms, reject an unreliable tuning candidate

The native planner now accepts and reports `solver.qoco_ruiz_iterations` (0–100,
default still 0). This selects the existing GPU numerical-update scaler. A
reproducible sweep found that every nonzero setting failed initial setup on the
N20 trajectory: the scaler used `DBL_MAX` for empty row norms. Identity scaling
for these rows and zero objectives prevents overflow without changing their
equations or relaxing tolerances. A regression fixture fails with the old
library and passes with the fix, including an independently known optimum.

After this fix, a small pilot suggested one pass was faster on N20. The longer
alternating comparison rejected it: 2 of 20 measured one-pass solves fail,
while all 22 zero-pass runs (including warmups) qualify. Eight and twelve passes
also fail in the pilot. All 16 N500 pilot solves qualify, but scaling is slower:
zero passes takes 564.648 ms median versus 634.966–653.452 ms for 1/2/4 passes.
The default remains zero; no speedup is claimed for this tranche. The full
GPU-native goal remains open, especially conditioning and solver reliability.
[Evidence and reproducible sweep](GPU_QOCO_LOCAL_BACKEND.md#gpu-ruiz-equilibration-and-zero-norm-constraints).

## Account for failed GPU solves; reject a warm-start shortcut

Core v79 now finalizes elapsed SCvx/CQP timing on early exits and retains the
pure-QOCO component times of failed inner attempts. No GPU synchronization or
numerical operation was added. A fault-injection test completes real GPU work
before forcing first/later failures: both timing cases fail against core72 and
pass against core79. Seven tests pass including convergence and cancellation.

A 72-attempt comparison of primal/cold starts and zero/one GPU Ruiz pass keeps
the default unchanged. On N20, the one-pass variants fail six times, including
two cold-start failures. With scaling off, all samples qualify and primal/cold
medians are 776.836/819.654 ms. All N500 samples qualify; unscaled primal/cold
medians are 426.645/430.309 ms, while scaled variants are slower. Cold starts
are not a general fix for the observed convergence variability. Failed attempts
now report their actual 0.47–0.70 second SCvx cost instead of zero.
[Evidence and limitations](GPU_QOCO_LOCAL_BACKEND.md#failed-solve-timing-and-warm-start-comparison).

## Distribute candidate gathers across GPU blocks

Core v82 replaces all three single-block SCvx candidate gathers with independent
multi-block gathers (256 threads, at most 1,024 blocks). This copies indexed values
without floating-point arithmetic. Sixteen fixtures cover 48 bitwise comparisons,
including repeated indices, offset buffers, guard words and IEEE special values.
All four CUDA sanitizer tools pass the standalone fixture. Full N500 passes
memory, initialization and synchronization checks; seven timing/convergence/
cancellation regressions pass. The host launch observer confirms 28 blocks on
N500 versus one in the same-binary ablation; it does not measure SM occupancy.

Hot-cache GPU-event microbenchmarks show 6.05–9.77x faster gathers for 7,014 state
entries and 103–258x for 1,000,003 entries. These are kernel-only measurements.
The final 88-attempt trajectory comparison qualifies every sample at unchanged
physics gates and fixed objective error <= 1e-8, but complete solve timings remain
mixed: N20 median 696.961 -> 870.593 ms, N500 462.076 -> 450.942 ms. Both tails
regress. An earlier different-binary N20 run had one numerical failure and stopped;
its cause remains unresolved. This tranche does not establish a general solver
speedup or a convergence reliability fix. The full GPU-native goal remains open.

[Detailed scopes and evidence](GPU_QOCO_LOCAL_BACKEND.md#multi-block-scvx-candidate-gathers).

## Downloaded Lambda fleet and partial benchmark evidence

The completed GTOC12 v11 fleet is available in the existing web viewer, copied
under `results/lambda/2026-09-06/visualiser`. Its 23 ships collect 14,047.803 kg;
196 asteroids are visited and 194 collected from. Both verifier reports pass.
The 03:26 UTC read-only download also contains 145 completed G4 groups and a
consistent checkpoint snapshot; G4 remains active and this is not its final report.
Every one of the 1,028 downloaded files passed its recorded SHA-256 check.
[Local launch instructions, source paths and scope](../results/lambda/2026-09-06/README.md).
An isolated sm_90 gather test compiled on Lambda, but its readiness guard refused
GPU execution while G4 was active. No H100 speedup measurement is claimed.

Publication checks: 199 GTOC12, H1 telemetry and planner-schema tests passed;
seven viewer importer/server tests passed. Downloaded evidence and the generated
viewer dataset were also rehashed directly from Git's staging area before
publication. See `artifacts/performance/main-publication-20260906-checks.json`
and `main-publication-20260906-evidence.json`.

Source fingerprint portability: the gather checkpoint records exact test-time
working-file bytes. Git normalizes ordinary source line endings, so those raw
hashes need not match a later Windows checkout. Published Git-blob fingerprints
and the post-checkout download recheck are retained in
`artifacts/performance/main-publication-20260906-source-fingerprints.json`;
the original test-time evidence is unchanged.
