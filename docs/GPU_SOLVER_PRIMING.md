# GPU solver preparation and early graph experiment

The new phase trace identifies host-dispatched solver preparation as a substantial
remaining cost. Early entry into the outer CUDA Graph removes one such dispatch
per transfer, but measurements do not establish a reliable H100 speedup. The early
entry option remains **disabled by default**. Physics and objective qualification
are unchanged, and the best retained fleet remains **12,805.194 weighted kg**.

## What is measured

Set `SPACEPDHCG_TEST_GTOC12_PHASE_TRACE=1` to emit one `SCVX_PHASE` JSON record
per native call on stderr. Unset it to disable instrumentation; presence enables
it, including a value of `0`. Durations cover setup, host priming, graph building,
graph execution, graph closure, final downloads and cleanup. These are host wall
times at existing synchronization boundaries, not GPU kernel times. The trace
adds no CUDA synchronization, events or device downloads. Cleanup includes
workspace destruction. Early error returns retain iteration/status fields of -1.

The complete one-ship profile has 47 native calls and 17 distinct node counts:

| Phase | RTX 5090 v461 | H100 v462 |
|---|---:|---:|
| Native call wall time, including Python wrapper | 10.872 s | 14.658 s |
| Host-dispatched priming | 7.867 s | 9.492 s |
| Outer graph execution | 2.398 s | 4.651 s |
| Outer graph construction | 0.096 s | 0.075 s |
| QOCO setup within priming | 2.269 s | 3.132 s |
| QOCO solves within priming | 5.241 s | 6.025 s |

The last two rows are components of priming and must not be added to it again.
Cumulative native report fields are counted once per call using their maximum;
individual iteration counts are summed. Priming includes 141 outer attempts in
both profiles. It contains substantial GPU computation, so this is not evidence
that all priming time is CPU arithmetic or transferable overhead.

The separate RTX Nsight Systems v457 capture has a CUDA API trace but no kernel
or memory timeline. Its long asynchronous-copy API calls include waiting for
queued work. They cannot be interpreted as pure transfer bandwidth or kernel
durations. Across that capture, 238 graph instantiations took only 0.133 s of API
wall time. Removing graph construction alone cannot explain the measured cost.

## Opt-in readiness change

`SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH=1` enables early entry; unset it or use `0`
for the existing path. A successful synchronous numerical scale-packet update
establishes the updater's `scale_valid` prerequisite. Once a subsequent device
IPM replay has completed, the adapter can enter the outer graph without waiting
for another complete host-dispatched SCvx solve. The readiness marker is cleared
when the QOCO solver is rebuilt. Existing ownership, device, extension, replay and
graph-state checks still apply.

On the measured zero-Ruiz, non-origin path, priming falls from three attempts per
leg to two: **141 to 94** for the one-ship campaign and **675 to 450** for the
225-leg replay. The numerical iteration moves into the GPU graph; it is not
omitted. State-origin mode may already prime in two attempts. The change does not
pool workspaces across legs or remove Python search/fleet orchestration.

## Measured outcomes

Complete campaign comparisons use baseline/candidate/candidate/baseline order,
two observations per mode, on the same binary with only the flag changed:

| GPU | Baseline median | Early graph median | Observation |
|---|---:|---:|---|
| RTX 5090 v467 | 32.108 s | 28.338 s | 11.74% less time in this batch |
| H100 v469 | 31.432 s | 31.929 s | 1.58% more time, overlapping observations |

All eight campaigns preserve the complete initial plans, 45,188,558 logical
branches and 2,782,091 collection options. Both mission checkers accept
548.254620 weighted kg each time. Outer and inner iteration counts vary between
runs, including with the same flag. These observations do not establish a
portable overall speedup.

The fixed 225-leg fixture has SHA-256
`09da720e7eca6481e9504ffa02840b5f35b9c2398ed229098c574c885510c846`.
Baseline precedes candidate in each full replay; these are single passes per
mode, not an interleaved timing experiment:

| GPU | Baseline solve total | Early graph solve total | Certified converged legs |
|---|---:|---:|---:|
| RTX 5090 v470 | 138.650 s | 135.650 s | 205 / 225 in both |
| H100 v471 | 165.432 s | 185.389 s | 205 / 225 in both |

Every converged result passes independent propagation certification. No baseline
converged leg is lost. The maximum certified final-mass difference is
2.10e-7 kg locally and 2.48e-8 kg on H100. Timing includes unsuccessful attempts
but excludes subsequent independent certification and export.

The **12.06% H100 regression is retained**, not averaged away. Case 44 accounts
for 15.162 s of the 19.957 s increase: baseline stops after eight outer attempts
at a stationary point with nonzero defects; candidate runs 43 attempts. Its
history shows tiny merit changes changing accept/reject decisions, followed by
unqualified inner solves and trust contraction. The trajectories already differ
slightly in the first two attempts, before either mode enters the outer graph,
so that divergence cannot be attributed solely to graph-entry timing.

A fresh H100 ABBA diagnostic v473 repeats cases 44, 201 and 98. Case 44 takes
2.954 / 1.862 s in baseline and 2.907 / 3.894 s in candidate, stopping after
14 / 9 and 14 / 18 attempts respectively. The 16.524 s tail does not recur in
these two candidate repeats; variability remains, including a slower candidate
case 98. This does not overturn the negative full replay. Results labelled
`infeasible` in these legacy reports mean remaining virtual control, not a
global infeasibility certificate.

The wider local candidate validation v472 completes in **218.968 s process wall
time** (217.348 s inside the CLI). Both mission checkers retain four ships,
29 mined asteroids and **2,088.668592 weighted kg**, with 169,753,864 logical
branches and 18,250,121 collection options. This is a single validation, not a
paired speed comparison or a new best fleet.

## Validation and reproduction

The SCvx, CLI and final mission-verification suites pass **45 tests on each
GPU**. The graph physics test covers early entry on/off, state origin on/off and
zero/five Ruiz iterations. It retains the independent propagation and fixed
final-mass assertions and forbids host numerical iteration. Current production
source hashes match the validated binaries. The final test edit only wraps a
function signature; its AST matches the validated test snapshot.

The RTX State-controller probe passes memcheck, synccheck and racecheck. This
probe does not exercise the complete cuDSS conditional IPM graph. Earlier v464
and H100 v466 have one incorrect test assertion for origin-mode priming; their
failed logs are retained alongside corrected passing v468/v469. Diagnostic
exporter failures v460/v459 and the pre-launch v463 runner failure are also
retained, with corrected retries. None is relabelled as a passing run.

Run the checked-in test matrix with the project's documented CUDA/QOCO runtime
environment, then select the experimental mode explicitly for comparisons:

```bash
SPACEPDHCG_GTOC12_GPU_TESTS=1 python -m pytest \
  tests/test_gtoc12_gpu_scvx.py tests/test_gtoc12_gpu_cli.py \
  tests/test_gtoc12_run_final_verification.py -q

# Add these to the same otherwise unchanged gtoc12 run command:
export SPACEPDHCG_TEST_GTOC12_PHASE_TRACE=1
export SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH=1
# Restore normal execution after diagnostics:
unset SPACEPDHCG_TEST_GTOC12_PHASE_TRACE SPACEPDHCG_TEST_QOCO_EARLY_OUTER_GRAPH
```

Exact run commands, runtime/source hashes, input fixture, scripts, successful and
failed logs, and outputs are in the verified archives:

- [Local profiles, tests, campaigns and 225-leg replay](../results/lambda/2026-09-08/gpu-solver-priming-local-v472/summary.json)
- [H100 phase profile](../results/lambda/2026-09-08/gpu-solver-phase-v462/summary.json)
- [H100 campaign comparison](../results/lambda/2026-09-08/gpu-early-graph-v469/summary.json)
- [H100 225-leg replay and regression](../results/lambda/2026-09-08/gpu-early-graph-v471/summary.json)
- [H100 focused repeat](../results/lambda/2026-09-08/gpu-early-graph-focus-v473/summary.json)

The next architectural target is repeated QOCO preparation across legs. Reusing
compatible topology and vendor resources must refresh every boundary, numerical
coefficient and captured setting, and reset failed-solve state correctly. These
measurements motivate that work; they do not establish that reuse is implemented.
