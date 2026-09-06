# GTOC12 assembly validation on the GPU

Native core v133 carries the assembly invalid flag through canonical conversion
and into QOCO v128's guarded replay. After initial setup, the enabled path no
longer downloads and waits for this flag before starting conversion. Assembly,
validation, numerical updates, solver replay, independent audit, qualification
and SCvx consumers share the same CUDA stream.

Use the retained native v133 library with prepared QOCO v128 and:

```sh
export SPACEPDHCG_TEST_QOCO_IPM_GRAPH=1
export SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY=1
export SPACEPDHCG_TEST_QOCO_NATIVE_NUMERIC_REPLAY=1
export SPACEPDHCG_TEST_QOCO_DEVICE_VALIDATION=1
export SPACEPDHCG_TEST_GTOC12_DEVICE_ASSEMBLY_VALIDATION=1
export SPACEPDHCG_TEST_GTOC12_DEVICE_QUALIFICATION=1
```

The optional `SPACEPDHCG_TEST_GTOC12_DEVICE_ASSEMBLY_VALIDATION_TRACE=1` records
whether validation was queued and whether the producer rejected its inputs.
This remains experimental. Initial setup, outer-loop dispatch and final report
collection still involve the host.

## Invalid data cannot pass through finite coefficients

Assembly validates trust radii, minimum mass, radius floor, excess velocity,
penalty and smoothing weights, plus the dynamics producer's status. Invalid
parameters can generate finite coefficients, so testing coefficients alone
would weaken validation.

Any nonzero producer flag contributes bit 16 to the existing canonical flags.
The retained numeric-report guard rejects either source of invalid data before
entering the IPM body. The device completion reports numerical failure and zero
iterations; downstream GPU qualification rejects it. Final host reporting keeps
the GTOC12 API's invalid-input code 3 and clears residual/objective availability.
The completed-solve counter does not advance for rejected input.

Initial setup validates before host conversion. If queued replay is disabled,
the scale packet needs priming, or recovery/diagnostic conversion requires host
values, validation is collected before invoking the synchronous path. Borrowed
input reads are drained before the caller can reuse buffers. A guarded rejection
after numerical updates requires a fresh solver on the next attempt.

Successful queued updates still download 60 adapter report bytes. The producer
bit shares the four-byte canonical validation report, replacing the separate
four-byte GTOC12 assembly download and its wait. Priming and fallback transfers
are separate; these counters do not include every QOCO or GTOC12 bridge transfer.

## Verification and retained failures

- The captured topology/conversion/producer/guard/device-consumer test passes
  192 combinations, including negative producer flags, without internal
  downloads or new allocations. It preserves the other numeric-report fields.
  Isolated memcheck, initcheck and synccheck each report zero errors.
- The actual GTOC12 GPU callback test confirms four invalid queued solves reach
  the GPU consumer with status 3, zero iterations and qualification false.
  Eight invalid fallback calls reject before that consumer. Initial invalid
  setup and subsequent qualified recovery also pass.
- A Python integration test checks 18 invalid parameter/state/control cases,
  including finite negative parameters. Every rejected call retains invalid
  input status, unavailable residuals/objectives, and qualified recovery.
- Disabled and legacy-QOCO-v126 integration runs each pass 52 tests. The first
  enabled run passes 51 and fails one existing coast constraint gate; a retained
  follow-up passes all 52. These are separate outcomes, not a replacement of
  the failed run.
- The broader regression passes all 325 tests in 360.50 seconds. Its planner
  checks still use the retained v77 executable, recorded separately from v133.

The initial failure had original equality residual 2.4886523501366312e-9 against
the unchanged 1e-9 gate, despite passing the normalized solver audit. A balanced
diagnostic also fails once with assembly guarding disabled and once enabled.
An additional 18 fresh-process tests of the published v132 baseline fail four
times at the same original-constraint gate, with residuals from 1.641e-9 to
2.741e-9. This establishes an existing convergence variability issue; it does
not resolve it or make those solves qualified. All failures are retained.

NVIDIA documents that cuDSS's default mode permits numerical run-to-run
variation and offers a separate deterministic mode, potentially at a speed
cost. That is a diagnostic candidate, not a proven cause or a change made by
v133. Its compatibility with QOCO's refinement path still needs checking.
[NVIDIA cuDSS general description](https://docs.nvidia.com/cuda/cudss/general.html),
[configuration details](https://docs.nvidia.com/cuda/cudss/types.html).

The full conditional-IPM memcheck failure also remains unresolved. Passing the
isolated validation kernels does not qualify the full solver under memcheck.

## Complete transfers

Six balanced triples produce 36 complete transfers, all passing independent
physics and the unchanged 1e-5 kg mass gate against 2445.3111007852112 kg.
All 484 conic qualifications agree with the independent host predicate, and
traces confirm 145 queued assembly guards.

| Execution, all using QOCO v128 and device canonical validation | Median complete transfer |
|---|---:|
| Published native v132 | 349.171 ms |
| Native v133, host assembly validation | 302.373 ms |
| Native v133, device assembly validation | 355.157 ms |

These highly variable measurements do not establish an additional speedup.
Warm-ups, slower runs, traces, reports and certificates are retained. The
visualiser still displays the existing v11 fleet at 12,805.194 weighted kg;
there is no new fleet or leaderboard submission.

The implementation, runtime hashes, commands and raw evidence are retained in
[`native-assembly-v133-checkpoint.json`](../artifacts/performance/native-assembly-v133-checkpoint.json).
