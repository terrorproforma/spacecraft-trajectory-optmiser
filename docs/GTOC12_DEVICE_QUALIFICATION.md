# GPU qualification and SCvx consumers

Experimental native core v129 extends the v128 solve/audit stream through the
GTOC12 objective reduction, conic qualification, nonlinear candidate measurement,
SCvx accept/reject decision and accepted-trajectory copy. Previously the host
collected the objective and residual reports and supplied the qualification bit
to SCvx. The new decision reads a retained device report directly.

Select all three switches with QOCO v126 and native core v129:

```sh
export SPACEPDHCG_TEST_QOCO_IPM_GRAPH=1
export SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY=1
export SPACEPDHCG_TEST_GTOC12_DEVICE_QUALIFICATION=1
```

The existing default path remains available for comparison. The optional
`SPACEPDHCG_TEST_GTOC12_DEVICE_QUALIFICATION_TRACE=1` prints submitted device
decisions; `SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY_TRACE=1` identifies prepared replay.
These switches select experimental execution; they do not select the native
SCvx backend by themselves. Use `outer_loop_backend='cuda'` with CUDA assembly
and discretisation and the QOCO convex solver.

## Execution and accuracy

The native cold-solve API accepts a host callback that queues GPU consumers of
borrowed device status and independent audit results. After replay it invokes
that callback before queuing the audit/status report downloads and waiting for
completion. GTOC12 queues its parallel objective reductions and scalar
qualification kernel, then invokes the SCvx consumer on the same stream.

Qualification keeps exactly the prior conditions: raw status 1 or 2; finite
independent primal and dual residuals at most the requested tolerance; finite
primal and dual objectives; and finite global relative objective gap at most
that tolerance. The relative gap is
`abs(primal-dual)/max(1,abs(primal),abs(dual))`. Neither the conic thresholds nor
the separate nonlinear physics certificate is relaxed.

SCvx speculatively measures the candidate on the GPU. Invalid numerical values
are detected by the existing propagation and metric kernels. The decision
kernel reads the device qualification bit and rejects an unqualified candidate
before using those metrics for acceptance. Accepted states/controls are copied
on the same stream. The host receives the completed reports afterwards.

First solves and stale-graph rebuilds still prime synchronously. Successful
priming publishes a small device status packet, then invokes the same consumer.
Failed synchronous priming invokes no consumer; the existing rejection path
handles it. Native warm-start retries retain their previous path. Consumer
launch errors drain outstanding stream work before borrowed contexts expire.

The adapter's existing 64-byte successful-update report limit is unchanged.
The GTOC12 bridge still downloads four objective scalars and now also downloads
the GPU qualification integer for reporting. Those bridge transfers are outside
the adapter counters, as documented by the existing API. This change removes a
host dependency before the numerical decision; it does not eliminate reporting.

## Verification

Enabled, disabled and synchronous-consumer modes each pass the existing 51
integration tests. The broader regression passes all 324 tests in 359.65 s.
A separate captured-graph test runs 210 combinations of raw status,
threshold boundaries and NaN/infinity inputs through the production qualifier
and a downstream GPU consumer. All pass. The actual SCvx controller exercises
its 14 decision cases in both modes; deliberately contradictory host arguments
verify that the device report controls the new mode. Existing final-state and
large reduction checks remain in place.

The qualification and updated controller probes each pass memcheck, initcheck
and synccheck with zero errors; memcheck reports zero leaks. These isolated
checks do not establish full QOCO pipeline sanitizer qualification. The actual
v129 SCvx pipeline memcheck aborts with CUDA error 999 at
`qoco_device_ir.cuh:116` during initial conditional-refinement warm-up, before
the consumer runs. It reports 233 errors and 231 outstanding allocations
(72,214,209 bytes) after the abort. This is the same failure boundary described
for the earlier coast fixture in [the native replay report](QOCO_NATIVE_REPLAY.md).
The full pipeline remains **not sanitizer-qualified**.

The v129 core and its original bundled controller executable were frozen before
the controller test gained its second mode. The updated test was separately
compiled against the same frozen core and run with all three sanitizers. The
checkpoint records both executables and the test sources to distinguish them.

## Complete-transfer measurements

Six balanced triples use the same QOCO v126 runtime and run two complete
transfers per process. All 36 transfers pass independent physics and the
unchanged 1e-5 kg final-mass gate against 2445.3111007852112 kg.

| Core / execution | Median measured complete transfer |
|---|---:|
| v128 native replay | 332.042 ms |
| v129 native replay, host qualification | 314.547 ms |
| v129 native replay, device qualification and SCvx consumer | 312.381 ms |

The device mode submits 132 GPU decisions across its twelve transfers. Replay
traces confirm that prepared solves are used; the additional decisions include
synchronous priming. Warm-ups, slower runs and all traces are retained. Timing
variation is substantial, including measured runs around 720 ms, so these
results do not establish a reliable additional speedup. They show that the
new execution sequence retains the tested accuracy.
All 435 reported conic qualifications across the three variants agree with the
original host-side gate evaluated from their independently reported quantities.

The source, runtime hashes, complete helper scripts, raw outputs and retained
failure are recorded in
[`device-qualification-v129-checkpoint.json`](../artifacts/performance/device-qualification-v129-checkpoint.json).

## Remaining work

The outer SCvx loop still runs on the host and reads its 16-byte command each
attempt. Numerical-update validation and assembly validation still download
flags. Initial topology conversion, setup/analysis and vendor warm-up remain
host work. Reporting still waits, and per-leg workspace reuse, full device
dispatch, batching and fleet search remain unfinished.

This is a solver implementation result, not a new GTOC12 fleet result. The
visualiser continues to display v11 at 12,805.194 weighted kg. Lambda was occupied
by the existing H100 campaign when checked; these measurements use the local
RTX 5090.
