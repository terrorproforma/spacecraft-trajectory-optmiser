# Device validation before guarded QOCO replay

Native core v132 keeps exact sparse-topology checks and canonical coefficient
validation on the GPU until solver replay, independent audit and SCvx consumers
have finished. The preceding path downloaded and waited for each of two flags
before submitting numerical updates. The new path combines those flags on the
producer stream and supplies them to QOCO v128's existing replay guard.

Enable it with the retained v132 native library and prepared QOCO v128:

```sh
export SPACEPDHCG_TEST_QOCO_IPM_GRAPH=1
export SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY=1
export SPACEPDHCG_TEST_QOCO_NATIVE_NUMERIC_REPLAY=1
export SPACEPDHCG_TEST_QOCO_DEVICE_VALIDATION=1
export SPACEPDHCG_TEST_GTOC12_DEVICE_QUALIFICATION=1
```

`SPACEPDHCG_TEST_QOCO_DEVICE_VALIDATION_TRACE=1` records actual guarded calls.
This remains an experimental path; it does not yet remove initial setup,
GTOC12 assembly validation or the host's outer-loop dispatch.

## Ordering and rejection

The exact topology comparison checks all six retained sparse arrays, including
grid-stride tails. Conversion validates bound classifications, finite numerical
values, transformed coefficients and quadratic symmetry. Flags are combined as
bits 1 (bound classification), 2 (nonfinite), 4 (asymmetry), and 8 (topology).

A retained nine-double buffer copies QOCO's numerical scale report and combines
its invalid field with these flags. Valid scale fields remain unchanged. Invalid
data skips the entire IPM graph body and publishes numerical-error status with
zero iterations. The independent qualification consumer rejects that status.
After completion the adapter returns the corresponding canonical API error,
with topology taking precedence over nonfinite values and asymmetry.

Caller-controlled sparse indices are only compared, never used to index the
compiled conversion or audit. Dimensions, buffer views and host cone descriptors
remain checked before submission. Pending validation and numerical work are
drained on error exits before borrowed inputs can be reused. A rejected queued
update requires rebuilding the numerical solver before another attempt because
its matrices may already contain the rejected values.

Initial numerical/graph priming collects validation before using the synchronous
fallback. Diagnostic comparisons, recovery rebuilds and requested warm starts
retain the established path. Numerical-update counters include work actually
submitted even when the guard subsequently rejects it; rejected conversions do
not increment the completed-solve count.

Steady updates download one combined four-byte validation flag at final report
collection. Measured adapter transfers decrease from 64 to 60 bytes per update:
48 audit bytes, eight solver-status bytes and four validation bytes. These
counters exclude opaque QOCO transfers and the GTOC12 bridge's own reports.

## Verification

- Enabled, disabled and legacy-QOCO-v126 configurations each pass 51 integration
  tests, plus the native controller, topology, audit and conversion checks.
- The broader regression passes all 324 tests in 354.19 seconds. Its planner
  checks use the retained v77 planner executable, whose identity is recorded
  separately from the v132 native library.
- A captured topology/conversion/guard/device-consumer chain passes 96 cases:
  every combination of the four canonical flags and a prior scale error, with
  zero, small and large output layouts. It preserves the other eight scale
  fields exactly, performs no internal downloads, and reuses retained storage.
- The isolated conversion executable passes memcheck, initcheck and synccheck
  with zero errors using the isolated Compute Sanitizer 13.2.87 package.
- Native mixed-cone tests pass with zero and three Ruiz passes, both against the
  CPU conversion oracle and through device validation. Ten invalid queued solves
  are confirmed by traces to skip IPM with zero iterations. Restored inputs
  recover with independent primal/dual KKT residuals at most 1e-8.

These isolated sanitizer results do not resolve the existing full-solver
conditional-graph memcheck failure documented in
[the v131 evidence](QOCO_NUMERIC_REPLAY.md#sanitizer-investigation).

Six balanced triples produce 36 complete transfers. All retain independent
physics verification and the unchanged final-mass gate of 1e-5 kg against
2445.3111007852112 kg. All 364 reported conic qualifications agree with the
independent host predicate. Traces record 115 device-validation replays.

| Execution, all using QOCO v128 and queued numerical updates | Median complete transfer |
|---|---:|
| Previous core v131 | 317.136 ms |
| Core v132, host validation | 299.650 ms |
| Core v132, device validation | 311.566 ms |

The new mode removes two intermediate waits; these variable complete-run
measurements do **not** establish an additional speedup. All warm-ups, slower
runs, reports and traces are retained. The fleet and visualiser remain at the
previous independently verified 12,805.194 weighted kg; this is solver evidence,
not a new fleet result.

Sources, frozen runtime hashes, commands and raw evidence are retained in
[`native-validation-v132-checkpoint.json`](../artifacts/performance/native-validation-v132-checkpoint.json).
