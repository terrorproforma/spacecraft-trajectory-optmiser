# Low-thrust factor stability and stationary-step diagnosis

This checkpoint fixes two distinct causes of intermittent GTOC12 failures.
The physical model, independent propagation certificate, 1e-9 conic audit,
1e-5 kg known-objective check, and requested IPM tolerances are unchanged.

## Identical QPs exposed unstable inner solves

The opt-in native snapshot diagnostic exports retained CSC topology, settings,
original numerical values, and any translated values and state origin. The
standalone `cpp/cuda/tests/qoco_snapshot_replay.cu` then solves those exact
problems without the trajectory loop. Downloads belong only to diagnostics;
normal solves retain their existing GPU path.

With QOCO137 and the previous 1e-13 static factor regularization, repeated
solves of identical QPs returned different iteration counts and residuals.
Independent long-double sparse products confirmed actual original-equation
gap failures; these were not merely different printed metrics. CPU Clarabel
solved each of the eight captured problems as an independent diagnostic oracle.
It is not used as a production fallback.

Changing the internal stopping tolerance alone did not remove the failures.
Increasing the low-thrust static P/A/G factor regularization to 1e-9 passed all
128 local replays in the matched eight-problem experiment, while retaining the
original strict stopping tolerance. The 1e-7 comparison also passed; 1e-5 failed
all 128 attempts. Those rejected settings and results are retained.

Regularization stabilizes the matrix used for factorization. QOCO's existing
iterative refinement evaluates the unregularized KKT equations, and the native
audit independently evaluates the original QP. The change does not replace the
physical problem with a regularized objective. Non-low-thrust settings are
unchanged. Vendor determinism remains disabled; this does not resolve the
separately documented vendor determinism/sanitizer limitations.

## A near-zero merit ratio could destroy an accurate trajectory

The first regularization-only build passed 64/64 local and 32/32 H100 complete
trajectories with five Ruiz passes. Broader testing still failed with scaling
disabled: local regression was 336 passed / 1 failed, and an H100 capture
reproduced a failed trajectory after eight successful ones.

That capture made the second cause explicit. Its third accepted point had merit
0.021875559685678503. A subsequent independently qualified, feasible candidate
had merit 0.021875559685914866 and step 2.4184484914702153e-08. The approximately
2.36e-13 merit increase assigned a negative ratio, so the controller rejected
the point before reaching its 1e-6 objective convergence test. Repeated rejection
collapsed the trust region and eventually produced inner iteration-limit exits.

The GPU controller and CPU reference now accept a stationary candidate when
it is feasible and **both** the predicted and actual merit differences are
within the existing absolute objective tolerance. The normal finer-propagation
confirmation still runs. The ratio is retained in telemetry rather than
rewritten to pretend the merit improved.

The device controller regression includes the captured values and rejects
counterexamples with a material actual increase, material model increase,
excess dynamics defect, excess virtual control, failed conic audit, or invalid
candidate. No trust/budget-exhaustion promotion or hidden solver retry was added.

## Validation

The final native v173/QOCO137 build passes **337 local regression tests** and
**94 H100 integration tests**. Each GPU passes **64/64 complete trajectories**
covering Ruiz 0/5, original/shifted coordinates, and ordinary, deferred, device,
and outer-graph execution. H100 also passes 128/128 identical-QP replays and both
selected nested-SCvx/guarded-solver memory checks. Local native controller,
ownership, deadline and guard checks pass, as do all three recovery/session tests.

The [result summary](../results/lambda/2026-09-07/gpu-stability-v174/summary.json)
links the numerical outcomes to source/runtime hashes and checksum-verified
archives. Earlier failures, including the H100 stationary-step capture, remain
in those archives. Some unscaled inner QPs still fail their accuracy audit and
are rejected; every final trajectory in this reported matrix qualified.
Setup and fleet search still require CPU work, and the previously documented
vendor determinism/local instrumentation limitations are not claimed resolved.

## Diagnostic reproduction

Create an empty directory and set `SPACEPDHCG_QOCO_SNAPSHOT_DIRECTORY` to its
absolute Linux path for an ordinary outer-loop run. The diagnostic synchronizes
and counts its device downloads. Outer graph capture is explicitly unsupported
while snapshotting, so graph iterations cannot silently escape the capture.
Each exclusive `qp-PID-SEQUENCE.txt` file contains round-trip decimal FP64 data.

Build `qoco_snapshot_replay.cu` against the same prepared QOCO headers and library
using the normal CUDA default-stream and architecture options. Invoke it as
`qoco_snapshot_replay snapshot.txt 16` to repeat an identical numerical problem.
`QOCO_REPLAY_REGULARIZATION` and `QOCO_REPLAY_TOLERANCE` are diagnostic-only
overrides; neither changes native production settings. `QP_REPLAY` JSON lines
include all primal, slack, and dual coordinates for an independent audit.

Runtime hashes, captured inputs, raw failed and successful comparisons, exact
build commands, and the independent audit helper are retained with the results.
Do not infer a fleet score or an overall application speedup from this fixture.
