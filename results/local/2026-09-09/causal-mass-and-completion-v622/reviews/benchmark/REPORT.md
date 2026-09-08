# Independent completion packing benchmark review, v622

This is a source and saved-evidence review. The reviewer has made no native or GPU calls and has not rerun the benchmark.

The finite design compares the frozen production `GpuCompletion.run` with ordinary host model metadata against compact GPU model metadata. Both arms use the same component library and the real pinned official catalogue. Twenty historical controls are repeated across two models and batch sizes 24, 48, 256 and 1024. The prescribed calls total 80 evaluations, 27,040 candidate evaluations (13,520 per arm), and 160 kernel launches inferred from the frozen source.

The timer includes production packing, source-array hashing and validation, model configuration, synchronous native execution/transfers and plan reconstruction. Ordinary geometry caching is retained normally; no cached geometry is injected. Catalogue/request construction and common workspace creation/close are separate. First calls use fresh workspaces within one process, so they are not independent cold-process measurements. Four subsequent measurements per arm use ABBA/BAAB order. Capture hooks and post-run evidence validation are outside the timed method.

The supervisor binds the source, interpreter, component library and the already-passed compact-g correctness run. It uses a single-use marker, fresh output, the shared nonblocking GPU lock, the pinned local UUID, an active-compute check, a 180-second child deadline, an inherited lock descriptor and bounded termination of only its owned child process group. The frozen inputs have no return sweeps; the production `return_override` returns immediately in that case. There are no fresh Lambert or SCvx jobs in this plan.

One preparation issue was identified and corrected: the initial harness wrote numerical readbacks only after parity assertions, which would discard the failing call's arrays. The final harness retains available output buffers in `finally` immediately after the timed call, before numerical or telemetry assertions. If the method raises, its status explicitly labels the buffers as unvalidated and potentially stale and the native count as unknown. Failed comparator results also save before the assertion. This does not change the measured method, tolerances, source library or work budget.

**GO for one finite supervised run** was sent to the parent after independently rehashing the corrected ready manifest `64d36a429381edeae90f3300520516f5310337f8d154ed292d5c89642143ba9d` (406 indexed files, 6,009,259 bytes) and worker `852d314b75fa78139cf7e2ed0f3b5dbd533949926508e45a606607ea607d5935`. All 18 CPU preparation tests and final lint/format stages passed, including method-exception and downstream-failure retention tests. The old freeze and failed preparation stages remain archived. No reviewer GPU execution was performed.

`review_preparation.py` performs a standard-library-only recheck of all indexed files, the complete source archive and exact compact-g overlays, the saved correctness prerequisite, fixture counts and ordered finite budget. It parses Python source without importing project or native code. It writes `preparation-findings.json`; the ready hash in that result identifies the exact reviewed kit.

The benchmark measures repeated proxy completions per second. It does not generate fresh routes, certify trajectories, improve a fleet score, establish full-mission speed, justify a production default change, or demonstrate SOTA performance. Rejected proxy candidates remain part of the throughput denominator.

## Completed saved-result audit

The parent executed the benchmark once. All 80 methods returned successfully, with 27,040 candidate evaluations and no hidden retry. The independent standard-library audit reads all 80 raw NPZ outputs, reconstructs the source byte hashes, checks call/result logs, and recomputes the timing arithmetic. Classification, failed indices, processing stages, pickup flags and cargo agree exactly with the frozen controls. Each arm's five readbacks are bitwise identical excluding timings. The largest ordinary/compact difference and the largest independent forward-formula discrepancy are both **4.547473508864641e-13**.

The independent formula uses the 20 frozen historical coefficient records. Output-only benchmark NPZs do not contain freshly expanded model metadata; the runtime catalogue/metadata calculation is covered by the prior compact-g correctness run and frozen source identity. This distinction is preserved in `results-findings.json`.

| Batch size | Flat ordinary/compact warm time | Existing fit ordinary/compact warm time |
| ---: | ---: | ---: |
| 24 | 0.6585× | 0.9683× |
| 48 | 0.8955× | 1.2246× |
| 256 | 1.2163× | 1.8768× |
| 1024 | 1.4217× | 2.1589× |

A ratio above one means compact is faster. These are medians of four warm samples per arm. Compact loses on small flat batches and is slightly slower for the fit model at 24 candidates; this supports keeping the mode opt-in. It does not support a universal speedup or default promotion. First-method lazy imports and the first workspace's context startup are separate cold-start effects, not representative warm timings.

`review_results.py` writes `results-findings.json` and `results-readback-index.json`. It compiles only pinned pure helper functions from source, does not import project/native code and does not execute the benchmark. The saved raw benchmark report is `ccd74d20bec17954b72e731bc05341ce93168e80c4876cefd336d5ee8591483a`; launch report is `97a29a527989f7ea30a5cb4fd690722cf8b63cf76e57801c31fd3b4a047606c3`.
