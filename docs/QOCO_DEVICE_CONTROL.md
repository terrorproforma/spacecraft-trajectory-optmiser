# QOCO device decisions and recovery

This experimental preparation moves stopping tolerances, best-iterate selection,
regularization updates, centering/combined-RHS scalars and the NaN-direction
recovery decision into retained CUDA state. It is an explicit extension to the
prepared QOCO backend, not the default build and not a complete GPU-controlled
IPM loop.

## Preparation and ownership

Prepare an isolated copy of the pinned QOCO source with the v109 checkpoint's
`prepare_qoco_gpu.py` flags, including device combined RHS, then run:

```sh
python scripts/gpu/prepare_qoco_device_ir.py --destination /path/to/isolated/source
python scripts/gpu/prepare_qoco_device_control.py --destination /path/to/isolated/source
python scripts/gpu/prepare_qoco_device_direction.py --destination /path/to/isolated/source
```

The postprocessors require exact source anchors and validate them before writing.
The last command is the v110 addition; omit it to reproduce v109. The build
helpers and prepared sources in the checkpoints record the complete build, CUDA
and cuDSS selections. Do not apply these postprocessors to a live upstream tree.

Each QOCO workspace owns its device control allocation. It is reset for each
solve, retained between solves, and freed at cleanup. It never stores pointers
into the shared scalar reduction scratch. Best-vector saves and restores execute
in parallel over the primal, equality dual, slack and cone dual arrays. Original
strict inequalities, best-iterate tie handling and inaccurate-solution rules are
preserved. Threshold arithmetic uses explicit rounded multiply/add operations.

v108 downloads the complete control report at each stopping check. v109 limits
ordinary dispatch to a 16-byte regularization/stop/status packet, materializing
the complete report at completion, restoration, verbosity or explicit audit.
The host still dispatches IPM iterations and cuDSS operations, supplies dynamic
regularization to factorization, and accumulates iteration counts. Setup and
sparse topology/conversion also remain host work. This packet is not a count of
all CPU/GPU traffic.

v110 scans the combined direction for NaNs on the GPU. A per-workspace flag guards
the actual iterate-update kernel: an invalid direction writes alpha zero and
performs no iterate arithmetic. Multiplying a NaN direction by a zero step would
still corrupt the iterate, so the kernel returns before those operations.
Intermediate correction/line-search work can still execute on scratch arrays;
the next predictor-corrector step overwrites that scratch. The scan resets its
flag before every direction. Like the original branch, it detects NaNs rather
than broadening the decision to infinities.

The CPU NaN check remains only in the explicit host-control ablation and optional
step audit. Normal device-control execution does not download this decision.
`SPACEPDHCG_TEST_QOCO_DEVICE_CONTROL_DISABLE=1` selects the host-control ablation;
`SPACEPDHCG_TEST_QOCO_DEVICE_CONTROL_COMPARE=1` compares device decisions with the
original C reference, and the existing combined-RHS and step comparison flags
check their respective arithmetic. Audits are diagnostic work, not timing runs.

## Evidence and limitations

The [v109 checkpoint](../artifacts/performance/qoco-control-v109-checkpoint.json)
embeds v108/v109 prepared sources, local extension sources, build/test helpers,
runtime SHA-256 hashes, raw results and failed experiments. Both versions passed
51 integration tests. v109 also passed the 51-test selection with original C
decision and combined-RHS audits enabled. Native PD6 fixtures at N=20 and N=500
passed independent qualification and the unchanged 1e-8 objective gate.

The control-kernel tests cover 13 stopping/best/regularization cases and four
best-restoration cases, with offset vector allocations and canaries. The initial
v108 test incorrectly demanded bitwise equality of `1e-6 * 10` and the literal
`1e-5`; its corrected assertion allows 1e-20 absolute rounding error. Both that
failure and the corrected executables/results are retained. Corrected v108 and
v109 probes passed all four CUDA sanitizer tools.

Full v108/v109 native transfer and seed tests passed memory/leak, initialization
and synchronization checks with **device iterative refinement disabled**. This
isolates the new control work. A fresh full v109 device-IR memory check failed
with CUDA error 999 in `qoco_ir_solve` and 218 reported errors, followed by leaked
allocations on the abort path. The conditional-graph refinement path remains
unqualified under the sanitizer; it is not described as sanitizer-clean.

Balanced v109 timing used the same v107 GPU-seed/GPU-SCvx core with six triples,
rotating which backend ran first. Every one of the 36 warmup/measured legs passed
the same independent physics and 1e-5 kg objective check. Measured medians:

| QOCO backend | Native transfer attempt |
|---|---:|
| v105 baseline | 535.703 ms |
| v108 device control/full report | 513.570 ms |
| v109 device control/compact dispatch | 567.572 ms |

There is no demonstrated v109 complete-run speedup. The earlier triple benchmark
rotated modulo two and never put the third arm first; it is retained as
exploratory evidence, separately from this corrected schedule. Local display-GPU
timings have substantial slow tails and are not H100 measurements.

The v110 direction test exercises the actual prepared QOCO scan and update
entrypoints with NaNs in different primal/dual slices, multi-block tails, empty
cones, mixed SOC/LP cones and a 262,145-coordinate LP cone block. It verifies
unchanged iterates on rejection, zero alpha, valid-step recovery, input/canary
preservation, NaN-only reference parity, audits and nested reduction scopes.
It passed all four sanitizer tools. The first launch lacked the isolated cuDSS
library path and failed before numerical execution; that failure is preserved
alongside the correctly configured run of the same executable.

v110 passed 51 integration tests normally and again with the C decision/RHS
audits, seven convergence/failure-accounting regressions, both native PD6 fixed
objective fixtures, and 22 transfer/seed tests under each of memcheck, initcheck
and synccheck with host IR. No tolerance or iteration budget was relaxed.

The balanced v110 comparison used six triples and qualified all 36 legs. Medians
were 589.932 ms (v105), 555.361 ms (v109) and 601.990 ms (v110). This removes the
host NaN decision but does not establish a complete-run performance win. The
[v110 checkpoint](../artifacts/performance/qoco-direction-v110-checkpoint.json)
records its exact source, runtime, test and measurement evidence. Native PD6
reports use an older planner executable with dynamically selected QOCO; runtime
hashes identify the measured combination rather than the planner's embedded
source-commit label alone.
