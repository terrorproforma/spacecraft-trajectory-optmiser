# GPU outer-loop building blocks

The new prepared-QOCO extension can emit the complete IPM, including nested
iterative refinement and invalid-input guards, into an enclosing CUDA graph.
It also exposes capture of the existing GPU numerical-update kernels. These
are the missing solver building blocks for GPU-controlled SCvx; the GTOC12
SCvx dispatch loop has **not yet been connected** to them.

`scripts/gpu/prepare_qoco_outer_graph.py --destination PREPARED_COPY` applies
the extension after the corrected QOCO131 preparation. Keep the original tree
and library. Preparation uses exact source anchors and records the resulting
hashes in `spacepdhcg-outer-graph.json`.

The ordinary and nested paths share the same IPM node emitter. A kernel resets
the inner loop condition and iteration counter before each solve. CUDA's
default condition values apply at graph launch, so relying on them alone would
skip later solves inside an outer loop. The update guard, initialization,
refinement counters, terminal recovery and completion packet are regenerated
for each invocation. See [NVIDIA's conditional graph requirements](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-graph-nodes).

## APIs and ownership

- `qoco_gpu_ipm_emit_graph` appends a prepared solve to a graph or conditional
  body and returns the node that consumers must depend on. It runs no solve.
- `qoco_gpu_capture_numeric_update` records the prepared coefficient/scaling
  update on an actively capturing stream and returns its borrowed device packet.

Prime the solver and numerical scaling synchronously first. All operands,
solver resources and numeric-update context must remain alive through graph
execution. No other solver API may run while these external graphs exist;
complete their streams and destroy their executables/graphs before changing
settings, invoking a legacy API or cleaning up. Executions sharing the same
workspace must be serialized. Node emission is bound to the original host
thread and CUDA device. Static-setting changes and incompatible dependencies
are rejected. This is an internal experimental interface, not concurrent use
of the same solver workspace.

## Evidence so far

Prepared QOCO133 passes the following on the local RTX 5090 and Lambda H100:

- 128 queued solves retain exact synchronous result and iteration parity.
- 72 nested IPM results retain exact parity, including iteration-limit exits;
  24 invalid updates skip the solver and publish numerical-error status. Later
  valid loop iterations restart correctly.
- 32 changing QPs run coefficient selection, numerical scaling, IPM and result
  consumption under GPU outer control. Every result matches the synchronous
  path exactly and the independently derived analytic optimum within 1e-7.

These are solver correctness probes, not complete trajectory speedup results.
The ordinary GTOC12 integration with native v155 / QOCO133 passes 89 tests on
H100. The first local run passed 88 and failed one physical thrust certificate.
A balanced isolated rerun reproduced that failure on the published QOCO131
baseline (one of four runs); all four QOCO133 reruns passed. These failures are
retained, not discarded as successful benchmark noise.

## Physical thrust acceptance

A relative conic residual can qualify while an individual control exceeds the
absolute GTOC12 certificate's `0.6 N + 1e-9 N` thrust ceiling. Native v157 checks
that same physical ceiling in the GPU candidate reduction before SCvx accepts
the point. It does not clip thrust, modify dynamics, relax the certificate or
change the requested conic tolerance. Infeasible initial references remain
usable; ZOH's unused last control is excluded, matching the certificate export.

The native test covers 24 physical-thrust cases across problem sizes, including
within-tolerance controls, the observed violating magnitude, active and
inactive endpoints, and infeasible initial references. These native cases pass
on both GPUs. Native v157 / QOCO133 passes **332 local regression tests** and
**89 H100 GTOC12 integration tests**. The local standalone GTOC12 integration
also passes 89 tests; these overlap the broader regression and are not an
additional 89 distinct tests.

H100 compute-sanitizer memcheck passes the queued-IPM probe with both QOCO131
and QOCO133, both new nested graph probes, and the full GTOC12 guard test. Every
process exits zero and reports zero errors. The full guard initially exceeded
a 30-second allowance, then completed in 43.6 seconds with a longer allowance;
both records are retained. This narrows the prior Windows/WSL sanitizer problem
but does not establish its cause or resolve that platform's failure.

H100 initcheck with its default API-memory checking trips the held-stream
submission watchdog. A diagnostic copy with a 30-second watchdog still waits
for its release before returning success. A standalone 64-byte device-to-device
`cudaMemcpyAsync` reproduces this without any solver code: ordinary and
memcheck submissions return in 20–27 microseconds, default initcheck waits
3 seconds for the watchdog, and initcheck with `--check-api-memory-access no`
returns in 25 microseconds. These are single diagnostic observations, not
performance benchmarks. This isolates the H100 held-stream failure to the
tool's API-memory instrumentation. NVIDIA documents that this option controls
[cudaMemcpy/cudaMemset checking](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#command-line-options).
Do not interpret the failed run's zero error summary as a passing test, or
claim that disabling API checks provides that missing coverage. Production
code and the original three-second watchdog test are unchanged.
The original full guard passes kernel initcheck with API-memory checking
disabled in 20.8 seconds, exiting zero with zero reported errors.

A fresh preparation from the corrected QOCO131 inputs reproduces all four
QOCO133 extension file hashes. The local native build reused a CMake directory
with an older embedded commit; its actual source file and frozen library
hashes are recorded separately. The H100 native build uses a fresh checkout of
`dcdb812fceda00041db52a2764377e0d1c165414`.

## Remaining integration

The native adapter must capture conversion, original-coordinate auditing and
numeric updates around the solver emitter. The GTOC12 bridge must then append
objective qualification, candidate propagation, SCvx acceptance and reference
refresh to an outer WHILE, with device termination/time limits and retained
per-attempt reports. Initial setup/priming, failed-workspace recovery, reusable
per-leg workspaces and fleet search still need work. The local sanitizer
failure also remains unresolved. No new fleet score or full-trajectory speedup
is claimed by these correctness tests.

## Retrieved evidence and viewer

The checked-in [summary](../results/lambda/2026-09-07/gpu-outer-v157/summary.json)
links source/runtime hashes, test counts, limitations and the checksums of the
local and H100 raw-evidence archives in the same directory. They retain failed
builds, the pre-fix certificate failure, sanitizer timeouts and successful
reruns. `retrieval.json` verifies each downloaded H100 evidence file.

Open the existing viewer at `http://127.0.0.1:4173/`, select **GTOC12 fleet**,
then find **GPU solver validation — latest**. Its validation data is separate
from the rendered v11 fleet. To start the local viewer in PowerShell:

```powershell
Set-Location 'C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser\results\lambda\2026-09-06\visualiser'
npm.cmd run serve
```
