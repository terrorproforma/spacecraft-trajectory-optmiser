# Deferred GTOC12 solver reports

**Unmerged candidate: regression not ruled out.** The broad suite recorded
327 passes and one coast qualification failure. The checkpoint is retained for
review; v136 is not qualified for promotion to main.

Experimental native **v136**, with unchanged QOCO **v128**, separates cold
solver submission from CPU report collection. SCvx can enqueue reference
refresh and its pinned done/error download after the GPU decision and before
collecting the solver report. This removes the intervening host round trip.

Enable `SPACEPDHCG_TEST_GTOC12_DEFERRED_REPORTS=1` together with:

```sh
export SPACEPDHCG_TEST_QOCO_IPM_GRAPH=1
export SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY=1
export SPACEPDHCG_TEST_QOCO_NATIVE_NUMERIC_REPLAY=1
export SPACEPDHCG_TEST_QOCO_DEVICE_VALIDATION=1
export SPACEPDHCG_TEST_GTOC12_DEVICE_ASSEMBLY_VALIDATION=1
export SPACEPDHCG_TEST_GTOC12_DEFERRED_REPORTS=1
```

Deferred reports automatically select device scheduling, qualification and
reference refresh. Initial setup, numeric priming and stale-workspace recovery
use the synchronous path. Verbose and CPU comparison modes are ineligible for
deferred submission. The option is experimental and disabled by default.

## Ownership and numerical gates

`spacepdhcg_gtoc12_qoco_can_enqueue` reports host readiness after a successful
queued numeric replay. `spacepdhcg_gtoc12_qoco_enqueue_controlled` then submits
assembly, conversion, numeric update, IPM, independent audit, objective
qualification and a required GPU consumer. Successful submission performs no
report downloads or waits. It means only that work was submitted.

`spacepdhcg_gtoc12_qoco_finish` collects the original audit and objective report
on the same host thread, device and stream. It applies the same status and
accuracy gates as synchronous completion. The native adapter shares completion
code across both paths. FP64 equations, integration, qualification tolerances,
mass checks and independent nonlinear certification are unchanged.

There is one pending solve per workspace. Inputs and device consumer outputs
must remain alive through completion. Synchronous solve, another enqueue,
accept and warm reset cannot overwrite pending native work. Wrong-thread and
wrong-stream completion are rejected; destruction drains the pending consumer
before freeing its storage. Error paths may wait to drain partial submissions.
External CUDA capture is rejected before bridge assembly is emitted.

The SCvx done/error destination is pinned host memory. Its eight-byte read is
queued before report collection and inspected only after completion. Counters
still distinguish outer command bytes from adapter and opaque QOCO transfers.
This change defers reports; it does not remove their bytes.

## Remaining work

The complete outer loop is still CPU-dispatched. Each attempt eventually calls
finish, and initial setup, failed-solve rebuilds, warm retry, per-leg workspace
reuse and fleet search remain host work. The bridge cannot yet be captured into
an outer CUDA conditional graph. It needs graph-body emission for the nested
IPM conditions, GPU termination/recovery policy and retained per-attempt reports.

The existing strict coast equality variability and full conditional-IPM
memcheck failure remain unresolved. The unchanged fleet's weighted score is
12,805.194102488575 kg; synthetic solver benchmarks do not change that score.

## Measured results and limits

- All 85 GPU integration tests passed. Seven held-stream submissions returned
  before the stream was released, including all six controlled invalid-input
  cases. Pending reuse, wrong thread/stream, double finish, capture rejection
  and destruction with a pending consumer passed.
- The broad suite recorded **327 passed, 1 failed** in 359.54 seconds. The
  failure was `test_known_coast_optimum_and_device_updates[False-False-zoh]`.
  Follow-up balanced fresh-process comparisons qualified 54/54 coast solves on
  published v135 and 53/54 on v136. The rejected v136 solve had primal residual
  1.8087078018572488e-8 against 1e-9. This does not rule out a regression, even
  though earlier versions also have recorded coast variability. No tolerance
  was relaxed and no rejected point was accepted.
- Four native conversion/oracle cases passed (Ruiz 0 and 3), including accepted
  primal/reset and malformed-input contracts. Ten SCvx tests passed without
  queued numeric replay; ten also passed with legacy QOCO v126.
- Full guard initcheck and synccheck both aborted in QOCO initial refinement
  (`qoco_device_ir.cuh:116`, CUDA unknown error), before the deferred cases.
  Their zero-error summaries are **not** sanitizer qualification. The full
  conditional-IPM memcheck limitation also remains unresolved.

All 36 complete synthetic transfer attempts, including warmups, passed their
independent certificate and unchanged mass gate of
2445.3111007852112 +/- 1e-5 kg. All 391 conic reports matched the original host
qualification predicate. The enabled runs performed 94 deferred collections.

| Execution, QOCO v128 throughout | Median complete transfer, warmup excluded |
|---|---:|
| Published v135, device scheduling | 312.269 ms |
| v136, synchronous reports | 391.763 ms |
| v136, deferred reports | 323.040 ms |

No additional speedup is established. All histories and warmups are retained;
solver iteration counts and elapsed times varied. Every run used exactly
`8 * (attempts + 1)` outer command bytes, excluding solver report transfers.

[The v136 checkpoint](../artifacts/performance/native-deferred-v136-checkpoint.json)
contains sources, runtime hashes, successful tests, failed tests and follow-up
evidence. The first diagnostic coast harness omitted `tests` from `PYTHONPATH`;
its failed imports are retained separately from the corrected comparisons.
The initial build deliberately stopped before freezing; the final incremental
build succeeded and froze the library and eight test executables.

The existing local visualiser's **GPU solver progress** panel displays the
candidate, 36/36 transfer qualifications, 327 passes/1 failure and its checksum.
It explicitly identifies v136 as unmerged. Lambda's H100 was still at 100%
with campaign PID 53138 on the read-only recheck; no tests were offloaded.
