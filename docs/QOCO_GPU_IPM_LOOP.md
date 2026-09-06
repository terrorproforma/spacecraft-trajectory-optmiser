# GPU-controlled QOCO iteration loop

The optional v121 extension executes QOCO's complete interior-point iteration
loop in a CUDA conditional graph. Residuals, stopping decisions, Nesterov–Todd
scaling, numerical factorisation, predictor/corrector solves, iterative
refinement and iteration accounting run without a host dispatch between IPM
iterations. It reuses the existing double-precision arithmetic and stopping
rules. This is an experimental implementation, not completion of the full
GPU-native trajectory pipeline.

## Measured complete-transfer performance

On the local RTX 5090, six balanced triples compared the frozen v117 runtime,
v121 with its host IPM loop, and v121 with its GPU loop. Each process ran a warmup
and one measured transfer through the same v107 GPU seed/SCvx core. All **36
transfers** passed independent physics certification and the unchanged **1e-5 kg**
final-mass gate against **2445.3111007852112 kg**.

| Runtime | Median complete measured transfer |
|---|---:|
| v117, device refinement/counts, host IPM dispatch | 550.833 ms |
| v121, host IPM dispatch | 483.960 ms |
| v121, GPU IPM loop | 325.607 ms |

The ratio of medians is **1.69×** against v117 and **1.49×** against the v121
host-loop ablation. These are local fixture results. The GPU also drives the
display, clocks are unlocked, and the raw evidence retains every warmup and
outlier. This experiment establishes neither a universal speedup nor an improved
GTOC12 fleet score.

## Structure and lifetime

Apply `scripts/gpu/prepare_qoco_ipm_graph.py --destination /isolated/source`
after the v117 retained-refinement/counting chain. Compile all prepared CUDA
translation units with **`--default-stream per-thread`**. Set
`SPACEPDHCG_TEST_QOCO_IPM_GRAPH=1` to select the GPU loop. Ordinary build scripts
do not apply this experimental postprocessor. The native adapter's queued
reduction scope is required. Verbose execution, host refinement/counting and
the listed host arithmetic audits are incompatible with this mode and rejected.

The outer WHILE condition is updated on the GPU. Its body computes the residual
and stopping metrics; an IF condition runs the predictor/corrector only when
another iteration is needed. Device kernels preserve the original completed
iteration count, maximum-iteration exit and best-iterate decision. Final
restoration and reporting use the existing routines.

The retained numerical-factorisation graph and a primed cuDSS SOLVE graph become
ordinary child graphs. Refinement WHILE nodes are emitted directly into the
enclosing graph: CUDA does not allow cloning graphs that contain conditional
nodes. Capture is paused and resumed with explicit graph dependencies at these
boundaries. The version-guarded conversion of cuDSS's two private eight-byte
range inputs remains the same as the earlier refinement implementation.
See NVIDIA's [graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html).

cuBLAS dot-product capture initially introduced allocation/free nodes, preventing
conditional-graph instantiation. A retained 32 MiB cuBLAS workspace avoids those
nodes while preserving dot-product arithmetic. It belongs to the queued
reduction scope and is released after stream completion and handle destruction.
The separate metric-graph cache is bypassed in this mode so it cannot switch
the cuBLAS stream and reset the supplied workspace. NVIDIA documents these
capture and workspace behaviours in the [cuBLAS manual](https://docs.nvidia.com/cuda/cublas/).

The whole IPM graph is currently built and destroyed **per solve**. Initial
topology/conversion, cuDSS setup/analysis, solver initialisation, graph construction,
terminal metadata/restoration and native SCvx dispatch still involve the host.
Retaining the whole graph will require extending the lifetimes of its scalar
scratch and cuBLAS workspace and handling changing solver settings explicitly.

## Accuracy and limitations

- Both enabled and disabled v121 pass all 51 trajectory integration tests and
  the native controller executable. The enabled path also passes seven
  convergence/failure-accounting tests.
- Sixteen analytic QP solves on one workspace change matrix values, RHS,
  refinement tolerance and iteration budgets. Host and GPU loops match exactly
  on every printed objective, status, IPM count, refinement count and final-step
  count, including a one-IPM-iteration exit followed by a successful solve.
- PD6 N20 and N500 both pass independent certification and unchanged 1e-8
  reference-objective gates; errors are 9.776e-11 and 9.705e-10 respectively.
- Full memcheck still aborts with CUDA error 999 in the **initial refinement
  graph**, before entering the new IPM graph. It reports 104 errors and 102
  outstanding allocations after abort. This path is **not sanitizer-qualified**.
- v118/v119 failed graph instantiation before execution. v120 failed to compile
  because its reduction-scope header used an unavailable error macro. Both
  failures and their sources are retained.
- The v118 host-loop integration run had one unchanged coast equality gate
  failure (1.804e-9 against 1e-9). Both v121 modes pass that suite; the earlier
  failure is preserved and not treated as proof of universal arithmetic parity.

The [v121 checkpoint](../artifacts/performance/qoco-ipm-graph-v121-checkpoint.json)
embeds current sources, all four prepared variants, frozen binary hashes, build
helpers/logs, failed attempts, full timing/physics reports and Lambda status.
The formatted postprocessor reproduces all nine changed v121 prepared files
exactly. The H100 remains occupied by its existing campaign; no Lambda benchmark
or fleet rerun was performed in this tranche.
