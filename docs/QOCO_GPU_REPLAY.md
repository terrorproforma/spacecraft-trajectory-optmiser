# Device completion and queued solver replay

Subsequent [native core v128](QOCO_NATIVE_REPLAY.md) adopts this API for cold
subproblem solves and queues the independent audit before collecting reports.
The account below records the v126 implementation and its validation boundary.

Experimental v126 adds a nonblocking API for an already prepared QOCO GPU graph.
It queues initialisation, warm-start selection, IPM, terminal recovery/unscaling
and a 64-byte completion packet on the caller's CUDA stream. Subsequent kernels
can consume that packet and the unscaled primal, slack and dual vectors without
downloading solver status or waiting on the CPU.

The native trajectory adapter and SCvx driver **have not adopted this API yet**.
They still use synchronous solver reporting. Connecting the device audit and
SCvx decisions to the new outputs is the next integration step. This extension
does not establish a fully GPU-controlled trajectory pipeline or a new fleet
score. Default builders remain unchanged.

## API and ownership

Apply after the [v125 preparation chain](QOCO_GPU_TERMINAL.md):

```sh
python scripts/gpu/prepare_qoco_ipm_replay.py --destination /path/to/isolated/source
```

Compile CUDA with `--default-stream per-thread`. Prime the fixed-topology solver
with a synchronous `qoco_solve` using `SPACEPDHCG_TEST_QOCO_IPM_GRAPH=1`, with
captured initialisation and terminal handling enabled. The new declarations are
in `qoco_gpu_replay.h` in the prepared CUDA algebra directory.

`qoco_gpu_ipm_replay_device(solver, stream, &output)` queues a replay and returns
device pointers in `QocoGpuOutput`. The completion packet contains ABI version,
status, IPM/refinement counts, restoration flag, residuals, gap, objective and
dynamic regularisation. These are internal solver metrics; independent conic
qualification and nonlinear physics verification remain necessary.

Same-stream consumers can run between repeated solves. The next replay
overwrites the borrowed outputs. Replays remain bound to the owning host thread
and CUDA device. The captured resources survive destruction of the temporary
caller reduction scope.

Call `qoco_gpu_ipm_finish_device` before changing streams, updating coefficients
or using another solver API. It waits for the replay stream, including queued
consumers, without downloading legacy metadata. A subsequent synchronous
`qoco_solve` also completes pending replay work at entry. Host solution status is
invalidated on replay and is not silently refreshed.

Each submission supplies current host settings through device parameters.
Dynamic regularisation changes from the preceding device solve are not
implicitly copied back into host settings. Static-regularisation changes require
re-priming; the API rejects a stale graph. Unprepared graphs, incompatible modes,
another owner thread and a different pending stream are also rejected. The API
does not itself support capture into an outer graph; composing the full SCvx
conditional graph remains separate work.

## Verification

- Cold and warm probes each pass **128 queued solves with GPU consumers** across
  32 coefficient/settings cases and two interleaved workspaces. Every returned
  vector and checked status, residual, objective and iteration count matches the
  synchronous solve exactly. The sequence includes maximum-iteration exits and
  subsequent successful solves.
- A held-stream test confirms submission returns before queued work can finish.
  It runs after the temporary caller scope is destroyed. A watchdog releases the
  hold and fails the test if submission waits; neither cold nor warm tests trigger
  it. Both then successfully consume the result on the GPU.
- Unprepared/null solver, changed static regularisation, pending other stream and
  different owner thread cases return the expected errors with cleared outputs.
- Synchronous graph-enabled and disabled modes each pass the native controller
  and 51 trajectory integration tests. Seven convergence/failure tests pass.
  PD6 N20/N500 pass independent certificates and unchanged 1e-8 objective gates,
  with errors 2.909e-14 and 1.110e-16.

The full replay probe under memcheck aborts during its **first synchronous
priming solve**, before calling the new replay API. CUDA reports error 999 at
the iteration download following the IPM graph launch; the abort leaves 252
allocations and produces 253 errors. The same synchronous interleaved probe
fails at graph completion on both v125 and v126 (251 and 252 errors respectively).
This differs from the earlier fixture's initial-refinement failure boundary.
The full solver and new replay API remain **not sanitizer-qualified**.

The initial test build warned about a renamed `main` without an explicit return;
the shared probe now returns zero. Original test source, binary and output are
retained. A Python import-order lint error was corrected before runtime build.

## Complete-transfer regression

Six balanced pairs run warm-up and measured transfers through the unchanged
v107 native driver. All **24 transfers** pass independent physics and the
unchanged 1e-5 kg final-mass gate against 2445.3111007852112 kg.

| Runtime | Median measured complete transfer |
|---|---:|
| v125 | 346.141 ms |
| v126 | 306.988 ms |

These runs still call the synchronous API, so they cannot measure an end-to-end
benefit from queued replay. Display GPU clocks are unlocked and timings vary
widely; all warm-ups and outliers are retained. No replay speedup is claimed.
Lambda's H100 remains occupied by its existing campaign (81% utilisation,
1,607 MiB during the read-only check); no work was offloaded.

The [v126 checkpoint](../artifacts/performance/qoco-ipm-replay-v126-checkpoint.json)
embeds six sources, prepared source, 10 runtime hashes, 34 helpers and 14 evidence
reports. All seven modified prepared files reproduce exactly, and Ruff passes.
SHA-256: `9e3faceb11be31e80075e1be2a42bd90b2f06632eeeabef4f84914b6c0f1c2d9`.
