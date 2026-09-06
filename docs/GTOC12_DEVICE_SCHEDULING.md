# Device scheduling through GTOC12 conic solves

Experimental native **v135** carries the GPU's integration step count through
conic assembly, QOCO orchestration and nonlinear candidate evaluation. The host
outer loop now consumes only an eight-byte done/error packet. Its previous
step-count and reference-refresh downloads are unnecessary in this mode.

Enable `SPACEPDHCG_TEST_GTOC12_DEVICE_SCHEDULING=1`. This also selects the
[v134 device reference refresh](GTOC12_DEVICE_REFRESH.md), even if its separate
option is unset. The other device validation, replay and qualification options
retain their existing meanings. The default path is available for comparison.
QOCO remains **v128**; equations, FP64 arithmetic and acceptance tolerances are
unchanged. There is still a host attempt loop, time limit, setup, solver report
collection and retry handling. This is not yet the full device-controlled loop.

## Interfaces and invalid inputs

`spacepdhcg_gtoc12_conic_launch_controlled_device` accepts a device step count
and optional device enable flag. It captures the complete interval
linearisation and conic assembly without allocations, downloads or host waits.
Null enable means always execute; zero preserves numerical outputs and invalid
status, and any nonzero value enables work.

Invalid step counts or dynamics prevent assembly from reading stale coefficient
buffers. The invalid flag is set and numerical outputs remain unavailable for
acceptance. This includes an invalid first launch, before dynamics coefficients
have ever been initialized. As with the existing device API, return zero only
means work was enqueued; consumers must check the device invalid flag.

`spacepdhcg_gtoc12_qoco_solve_controlled_device_with_consumer` borrows the device
step count through stream completion. It uses the existing assembly-input guard
to reject invalid values before numerical IPM execution. It preserves the
synchronous bridge and GPU consumer contracts. A null step-count pointer returns
invalid-argument status 1; a device value below one returns invalid-input status
3, without qualification. The complete bridge is still not graph-capturable.

The SCvx command layout is private and now places done/error together. No public
report layout changes. Candidate propagation uses the device step count before
the decision kernel can change it for polishing. Reference refresh then consumes
the new count on the same stream. The host downloads neither count nor refresh.

## Verification

- Four first-invalid conic cases run with uninitialized numerical inputs and
  dynamics storage. They set invalid without reading those inputs or changing
  the retained numerical output.
- Forty-eight changing-input conic graph replays retain the original topology,
  objective and Hessian checks. Twenty-eight additional controlled replays
  cover both holds, 3/257 intervals, changing step counts, signed enable values,
  skip, invalid inputs and recovery, with bitwise fixed-step output parity.
- The conic test passes memcheck, initcheck and synccheck with zero errors.
  Existing interval, reference-refresh and SCvx controller tests also pass.
- The actual QOCO callback test exercises ten queued invalid rejections with
  GPU status 3, zero IPM iterations and qualification false; twenty fallback
  rejections occur before the callback. Device-produced step changes, initial
  invalid inputs, null-pointer rejection and qualified recovery pass.
- Eighty-three Python GPU integration tests pass, including both scheduling
  modes, independent CPU dynamics comparisons and certified complete transfers.
  The enabled transfer test enforces eight command bytes per attempt plus eight
  initial bytes, while retaining the original physics and mass gates.
- The broader regression passes all 326 tests in 355.82 seconds, including
  existing planner checks against the separately recorded v77 executable.
  Eight additional SCvx tests pass with device qualification disabled and the
  separate refresh option off, exercising the synchronous qualification path
  while device scheduling selects its required GPU refresh automatically.

All **36 complete transfer runs** from six balanced triples qualify, including
warmups, against independent trajectory replay and the unchanged 1e-5 kg mass
gate around 2445.3111007852112 kg.

| Execution, all with QOCO v128 and device reference refresh | Median complete transfer, warmup excluded |
|---|---:|
| Published native v134 | 327.924 ms |
| Native v135, host step-count handoff | 320.271 ms |
| Native v135, device step-count handoff | 357.430 ms |

The timings remain variable and establish **no additional speedup**. The 12
enabled runs each download exactly `8 * (attempts + 1)` outer command bytes;
the two comparison groups each use `16 * (attempts + 1)`. Solver and bridge
report transfers are separate and are not included in those command counters.
All run histories, warmups, reports and source/runtime hashes are retained in
[the v135 checkpoint](../artifacts/performance/native-scheduling-v135-checkpoint.json).

The pre-existing strict coast equality variability and full conditional-IPM
memcheck failure remain unresolved. Isolated conic sanitizer passes do not
qualify the entire solver under memcheck. The rejected deterministic cuDSS and
Ruiz experiments remain recorded with [v134](QOCO_DETERMINISM_DIAGNOSTICS.md).

## Displaying the results

The existing viewer at <http://127.0.0.1:4173/> now has a **GPU solver progress**
panel below **Compute & optimisation**. Select the GTOC12 fleet and refresh the
page after updating files. It displays the synthetic benchmark separately from
fleet scoring, including qualification, timing, remaining CPU work and the
checkpoint checksum. The small record is generated from the retained results:

`results/lambda/2026-09-06/visualiser/data/gtoc12/solver-progress.json`

The fleet remains v11 at 12,805.194102488575 weighted kg; these are new solver
benchmarks, not a new fleet or leaderboard submission. The fleet-summary label
now calls the verifier's raw mass **Verifier returned mass**, avoiding confusion
with the weighted score displayed under Compute & optimisation.

Remaining work includes capture/emission of the complete outer loop, removal
of intermediate report waits and host retries, setup/workspace reuse, and GPU
fleet search. Lambda's existing H100 campaign remains live and undisturbed.
