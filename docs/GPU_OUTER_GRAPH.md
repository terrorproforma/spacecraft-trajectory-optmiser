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
inactive endpoints, and infeasible initial references. The corrected local
GTOC12 integration passes 89 tests. H100 validation of this native change is
pending at this checkpoint.

## Remaining integration

The native adapter must capture conversion, original-coordinate auditing and
numeric updates around the solver emitter. The GTOC12 bridge must then append
objective qualification, candidate propagation, SCvx acceptance and reference
refresh to an outer WHILE, with device termination/time limits and retained
per-attempt reports. Initial setup/priming, failed-workspace recovery, reusable
per-leg workspaces and fleet search still need work. Full conditional-IPM
sanitizer failures also remain unresolved. No new fleet score is claimed.
