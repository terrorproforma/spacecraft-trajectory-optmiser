Prepared production adapter validation
======================================

This is a single-use, root-launched GPU test runner. Preparation ran no GPU
tests. It binds the frozen Python CPU-attempt-b snapshot separately from the
full C++ core and requires the already completed native/42-control parity report.
It does not read the current Git checkout for implementation inputs.

Only the four existing GPU cases in `tests/test_gtoc12_gpu_completion.py` run:
fit, flat, ratio and certified return. Each performs a 259-candidate call and a
retained-workspace three-candidate call: eight native evaluation calls and
1,048 costing cases if all tests complete. There is no native test duplicate,
new Lambert solve, trajectory refinement, mission search or fleet promotion.

`run.py` owns the shared nonblocking GPU lock, verifies the exact RTX 5090 UUID
and empty compute-process list, and launches one tracked process with a
180-second timeout. It consumes a single-use marker before preflight. Failure,
busy state and partial output remain preserved; there is no automatic retry.
Only its own child process group can be terminated on timeout, and the inherited
lock descriptor remains held until that child exits.

`pytest_child.py` calls the frozen tests unchanged. Observation wraps workspace
construction to record each completion C API boundary and save packed inputs
and raw outputs before pytest assertions. It also records scope closure and
telemetry. Both bound native Lambert entry points and public request methods
are guarded against unrelated requests. These observation hooks change neither
cost arithmetic nor gate thresholds. Their serialization adds CPU overhead, so
the observed timings are not benchmark measurements.

Finite values are exported as round-trip JSON numbers; nonfinite values use
explicit `float64` tags. Every completed call saves structured policy, candidate,
deployment, flight and result arrays plus CUDA phase statistics. Failed or
interrupted calls remain distinguishable from completed calls in the child
report. Exact test classifications and tolerances are those in the frozen test.

Preparation attempt a failed before collection because a pytest observer hook
used the wrong argument name. Its source and failure are retained under
`preparation-attempt-a/` (and the original CPU output paths). Only that runner
hook was corrected; adapter/core bytes stayed unchanged. The successful
`cpu-collection-b/` check collected exactly four tests with CUDA visibility empty
and native-library loads blocked. `ready.json` binds the corrected runner,
profile and collection evidence; it is not a record of GPU execution.

Root's reviewed launch command is listed in `ready.json`. Final packaging and
runtime claims wait for a terminal `gpu-output/report.json` and independent
saved-output review.
