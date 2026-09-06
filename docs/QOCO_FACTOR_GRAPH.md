# Retained numerical factorisation graph

The optional v115 QOCO extension runs one ordinary cuDSS numerical factorisation
to warm its workspace, captures that phase, and replays the retained CUDA Graph
for subsequent factorisations. The matrix values change on the GPU between IPM
steps; the sparse structure and captured addresses stay fixed for that workspace.
This removes repeated calls to the host cuDSS factorisation entrypoint. The host
still dispatches graph launches and the IPM loop, and iterative refinement still
creates per-solve graphs and downloads its accounting. This is not a complete
GPU-controlled optimiser.

NVIDIA documents graph capture for asynchronous factorisation/solve and requires
compatible allocation handling. Analysis remains synchronous. See
[cuDSS graph support](https://docs.nvidia.com/cuda/cudss/general.html#cuda-graphs-support)
and the [memory-handler contract](https://docs.nvidia.com/cuda/cudss/types.html#cudssdevicememhandler-t).

## Allocation and stream ownership

A memory handler is attached before cuDSS data creation. It retains vendor device
allocations until the solver is destroyed, reusing a free block only on the
stream where it was freed. New allocations are allowed during warm-up but
rejected during capture. Zero-byte requests return a null pointer. CUDA's native
allocation alignment is preserved. No new allocation/free nodes enter the graph.

Capture is restricted to kernels, device-to-device copies, memsets and empty
nodes. Host copies and other node types fail explicitly; this factorisation-only
path does not freeze arbitrary host inputs into constants. Default/solver stream
events order numerical updates, factorisation and its consumers. Cleanup waits
for execution, destroys the graph, and then releases retained storage.

The captured graph is owned by the cuDSS workspace and destroyed with it. Reuse
across changes of sparse structure or solver configuration is not supported.
`SPACEPDHCG_TEST_QOCO_FACTOR_GRAPH_DISABLE=1` selects ordinary cuDSS factorisation
with the same allocator, allowing the graph's effect to be measured separately
from memory retention. Normal build scripts do not select this experimental
postprocessor.

After preparing the v110 device-IR/control/direction tree, run:

```sh
python scripts/gpu/prepare_qoco_factor_graph.py --destination /path/to/isolated/source
```

## Diagnostic development

The earlier v111–v113 probe captures and replays both factorisation and solve
alongside the ordinary trajectory solve. Its result is diagnostic, not a
replacement trajectory solution. v111 failed during analysis; v112 allocation
logging isolated the zero-byte allocation mismatch. v113 corrected that contract.

On real PD6 systems with KKT dimensions 3,026 and 74,066, v113 captured a 25-node
graph and passed all 60 comparisons across 15 successive linear solves per
fixture, with two replays each, at a 1e-12 scaled comparison tolerance. Two
eight-byte host inputs belonged to the solve phase and were snapshotted by the
diagnostic. The actual v115 factorisation-only graph does not use those snapshots.

GTOC12's individual linear solves failed all 60 strict direct-versus-graph replay
comparisons (maximum scaled difference 4.273e-4), although both complete diagnostic transfer runs passed independent
physics/objective qualification. That discrepancy is retained in the evidence;
the diagnostic does not prove universal linear-solve parity. Its first GTOC12
invocation also omitted `SCVX_OUTER` and failed before numerical execution; the
configured rerun is recorded separately.

v114 built the first factorisation-only extension but was not run: source review
identified that the allocator needed to retain the free stream, rather than the
original allocation stream, before reusing a block. v115 contains that correction.
All frozen builds and failed outcomes are preserved. The existing conditional-IR
sanitizer failure is a separate unresolved limitation; checks with host IR cannot
establish that the full conditional path is sanitizer-clean.

## v115 validation and timing

The actual factorisation-only path passed 51 integration tests normally and with
original C decision/combined-RHS audits, seven convergence/failure-accounting
tests, and both independently certified PD6 fixtures at N=20 and N=500. Their
objectives were 0.5129756911916401 and 0.512975691290699, within the same 1e-8
reference gate. The 22 native-transfer/seed tests passed memcheck (zero leaks),
initcheck and synccheck with device IR disabled. This tests the new factor graph
without claiming that the existing conditional-IR path is sanitizer-clean.

Six balanced triples used the same v107 GPU-seed/GPU-SCvx core, rotating which
variant ran first. All 36 warmup/measured transfers qualified against the same
independent physics and 1e-5 kg objective gate. Native transfer attempt medians:

| QOCO variant | Median |
|---|---:|
| v110 baseline | 535.348 ms |
| v115 retained allocator, ordinary factorisation | 516.308 ms |
| v115 retained allocator and factorisation graph | 530.927 ms |

These small, noisy local differences do not demonstrate a consistent graph
speedup. No default backend is promoted on their basis. They are RTX 5090
measurements, not H100 results. The
[checkpoint](../artifacts/performance/qoco-factor-graph-v115-checkpoint.json)
embeds the source, frozen variants, runtime hashes, helpers, raw failures and
measurements. Re-running the final postprocessors reproduced all 12 modified
prepared files exactly after LF normalization.
