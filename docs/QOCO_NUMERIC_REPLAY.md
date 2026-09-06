# Device numerical updates and guarded solver replay

Experimental native core v131 with prepared QOCO v128 connects numerical
equilibration and KKT value updates to solver replay without downloading the
scaling report between them. Previously QOCO downloaded nine doubles and waited
for its numerical update before beginning a solve. The queued path retains that
report on the GPU and supplies its `k` and `kinv` directly to the IPM graph.

Use these switches with the indicated runtimes:

```sh
export SPACEPDHCG_TEST_QOCO_IPM_GRAPH=1
export SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY=1
export SPACEPDHCG_TEST_QOCO_NATIVE_NUMERIC_REPLAY=1
export SPACEPDHCG_TEST_GTOC12_DEVICE_QUALIFICATION=1
```

The last switch selects the preceding
[GPU qualification and SCvx consumers](GTOC12_DEVICE_QUALIFICATION.md). With all
four switches, the stream runs numerical updates, solver replay, independent
audit, objective qualification, candidate measurement and the SCvx decision
before collecting the solve report. The optional
`SPACEPDHCG_TEST_QOCO_NATIVE_NUMERIC_REPLAY_TRACE=1` identifies actual queued
numeric replays. The default path remains available for comparison.

## Ordering, validation and lifetime

The preparer reuses the existing numerical-update kernels and arithmetic. All
16 launch sites and the matrix copies run on the producer stream. KKT CSR value
updates use the retained mappings on that same stream. The next numerical update
reads the previous inverse cost scale from device memory, preserving the prior
objective unscaling order without refreshing a host scalar.

The IPM graph has an outer conditional guard. A numerical update with invalid
scales or a nonzero invalid flag skips the entire IPM body, including
initialisation and factorisation, and publishes a numerical-error completion
with zero iterations. Independent qualification therefore rejects the result.
This avoids sending invalid updated matrices through a factorisation merely
because the host has not read the validation flag yet.

The update result is borrowed from retained workspace storage. Updates, replay
and consumers must use the same stream until their finish APIs complete. Device
and thread ownership and conflicting pending streams are checked. Explicit
materialisation refreshes legacy host scale/range metadata when a caller needs
to return to a host-controlled solver API. A second finish call can materialise
metadata after an earlier wait-only finish. Failed numerical updates require a
rebuild, matching the native adapter's existing recovery policy.

Initial numerical scaling and solver graph preparation remain explicit priming
steps. Zero-Ruiz setup has no numerical scale packet until its first update; that
first update uses the existing synchronous API. Later updates are queued.
Stale/unprepared solver graphs materialise scaling before synchronous priming.
Native warm-start retries retain their existing path. Error exits drain borrowed
work before the caller can reuse inputs.

## Verification and retained failures

The new QOCO path passes 64 queued update/replay/consumer chains with exact
synchronous solution vectors, statuses, iteration counts and objective/residual
reports. A held-stream callback proves that update plus replay submission
returns without waiting. A NaN update skips IPM and leaves the previous solution
vectors unchanged while publishing failure.

Of these 64 deliberately wide-scaling cases, 43 satisfy the independent analytic
coordinate and objective gates (1e-7). The other 21 retain numerical failures;
they are not counted as qualified solves. A separate comparison of the first
17 synchronous cases produces identical results on old QOCO v126 and new v127,
including the difficult transitions from cost scale 100 to 0.01. This identifies
an existing conditioning problem; the queued implementation does not fix it.
The earlier replay probe also passes all 128 exact replay comparisons on v127.

The first queued runtime, QOCO v127, contained an ordering bug: a source regex
missed launch dimensions containing `->`, leaving some kernels on the default
stream. v128 fixes the transformation, asserts the 16-site launch count and
checks every rewritten launch's stream. All six prepared files reproduce
exactly from the preceding v126 source. The failed runtime, sources and probe
outputs are retained.

The first native integration, core v130, failed ten tests because zero-Ruiz
contexts had not primed their numerical scale packet. Core v131 adds the
one-time priming path. Enabled and disabled modes each pass all 51 integration
tests, including the existing 64-byte adapter update-report limit. That limit
excludes opaque QOCO transfers; the eliminated nine-double report belongs to
QOCO, not those adapter counters.

Core v131 also passes the 324-test regression in 369.83 s. Its disabled path
passes all 51 integration tests with the preceding QOCO v126 library too.
Two eight-update coast sequences switch queued updates on and off in a retained
workspace, with zero and three Ruiz passes respectively. All sixteen solves
qualify, and traces confirm actual queued updates in both sequences.

## Sanitizer investigation

A separate probe takes the production device-parameter and conditional-guard
code through 64 cases. Its initial version had incompletely initialised test
storage; the corrected probe explicitly initialises storage on its stream.
Normal execution, initcheck and synccheck then pass. Memcheck still aborts with
CUDA error 999 at result collection, reporting five errors and four outstanding
test allocations (220 bytes). The original test output is retained as well.

An isolated NVIDIA `cuda-sanitizer-13-2` package, version 13.2.87-1 (Compute
Sanitizer 2026.1.1.0), was downloaded and extracted without changing the driver
or system CUDA installation. It reproduces the same minimal-probe memcheck
failure; initcheck and synccheck still pass. The new checker also reproduces the
full SCvx pipeline abort at `qoco_device_ir.cuh:116` during initial refinement,
before queued numerical replay. That abort reports 233 errors and 231
outstanding allocations (72,214,209 bytes).

This narrows the reproduction but does not prove the cause is solely the tool
or driver. The full pipeline remains **not memcheck-qualified**. Both checker
versions, the package hash, minimal reproduction and full failure are retained.

## Complete-transfer measurements

Six balanced triples run two transfers per process. All 36 transfers pass
independent physics and the unchanged 1e-5 kg final-mass gate against
2445.3111007852112 kg.

| Execution | Median measured complete transfer |
|---|---:|
| Core v129 / QOCO v126 | 356.582 ms |
| Core v131 / QOCO v128, synchronous numerical updates | 325.024 ms |
| Core v131 / QOCO v128, queued numerical updates | 308.614 ms |

The new mode executes 90 queued numerical replays across its twelve transfers.
All warm-ups, traces and slower runs are retained. Run-to-run variation remains
substantial, so these medians do not establish a dependable additional speedup.
Compare external complete-transfer time: the native update timer now includes
submission rather than waiting for numerical execution, while replay/audit
timers use GPU events. Internal stage timers are not interchangeable across modes.

## Remaining work

At v131, canonical topology and numerical-conversion validation still downloaded
flags before replay. [Core v132](QOCO_DEVICE_VALIDATION.md) subsequently moves
these checks into the device chain and combines their final report.
GTOC12 assembly validation and the outer SCvx command loop still involve the
host. Initial setup/analysis, vendor warm-up, full device dispatch, native warm
retry policy, per-leg workspace reuse, batching and fleet search remain
unfinished. This is not a fully GPU-controlled mission optimiser.

The visualiser still displays the existing v11 fleet at 12,805.194 weighted kg;
these solver measurements do not constitute a new fleet score.

The implementation, prepared sources, runtime hashes, full helpers, raw results
and failed experiments are retained in
[`native-numeric-v131-checkpoint.json`](../artifacts/performance/native-numeric-v131-checkpoint.json).
