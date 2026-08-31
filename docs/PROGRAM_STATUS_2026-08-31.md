# SpacePDHCG and OrbitWeaver programme status

Status date: **2026-08-31**.

This document distinguishes completed engineering from GPU-dependent experiments. “Implemented”
does not mean “demonstrated faster”; performance claims require the benchmark protocol and real
recorded hardware. The more detailed blocker classification is in
[`PRE_GPU_COMPLETION_AUDIT.md`](PRE_GPU_COMPLETION_AUDIT.md).

# Executive state

The production architecture is now deliberately split as:

```text
Python references / experiments / plots / paper artefacts
                         │
              stable C ABI and parity tests
                         │
C++20 dynamics + transcription + SCvx + CQP + robust routing core
                         │
          future persistent PDHCG CUDA implementation
                         │
            future NCCL/MPI multi-GPU implementation
```

The repeated numerical hot path is intended to be C++/CUDA. Python remains the transparent
correctness oracle, research interface, and experiment layer. Removing Python from those roles
would not accelerate the device-resident solver and would reduce auditability.

# Status legend

- **COMPLETE-REFERENCE** — implemented and covered by CPU/native correctness tests.
- **PREPARED** — interface and host logic exist; the production accelerator implementation remains.
- **GPU-BLOCKED** — implementation or validation intrinsically requires a CUDA device.
- **OPEN-CPU** — useful work remains and does not require a GPU.
- **EXPERIMENT-BLOCKED** — implementation may exist, but the paper result needs hardware runs.

# A — common modelling and benchmark foundation

| Item | Status | Evidence |
|---|---|---|
| Canonical QP/SOCP/CQP representation | COMPLETE-REFERENCE | Python and C++ fixed-pattern structures |
| Stable topology fingerprint | COMPLETE-REFERENCE | matching Python/C++ fingerprint and parity probes |
| HCW rendezvous QP/SOCP | COMPLETE-REFERENCE | repeated updates and independent diagnostics |
| Known-optimum trajectory-banded fixtures | COMPLETE-REFERENCE | deterministic QP/SOCP families |
| Nonlinear 3-DoF powered descent | COMPLETE-REFERENCE | analytic Jacobians, Euler/RK4 propagation and path checks |
| 14-state 6-DoF powered descent | COMPLETE-REFERENCE | quaternion rigid-body dynamics and fixed-pattern CQP |
| Long-horizon two-body low thrust | COMPLETE-REFERENCE | gravity/thrust Jacobians, mass depletion and CQP |
| Selectable higher-order transcription | COMPLETE-REFERENCE | Euler or RK4 discrete-flow linearisation with invariant topology |
| Domain-aware finite differences | COMPLETE-REFERENCE | central interior and valid one-sided boundary derivatives |
| Continuous inter-node certification | COMPLETE-REFERENCE | dense nonlinear propagation and path-violation checks |
| Violation-state CT enforcement inside the CQP | OPEN-CPU | dense checking exists; integral violation states remain |
| Adaptive mesh refinement between episodes | OPEN-CPU | fixed hot-loop topology remains the baseline |

# B — persistent device-resident SCvx

| Item | Status | Notes |
|---|---|---|
| Immutable C++20 CQP ownership | COMPLETE-REFERENCE | owns CSC topology and cone metadata |
| Mutable numerical update buffers | COMPLETE-REFERENCE | topology-preserving updates with validation |
| Persistent workspace state machine | COMPLETE-REFERENCE | update/solve/reset/cancel lifecycle |
| Host/device/unified pointer descriptors | COMPLETE-REFERENCE | API boundary is frozen |
| Scaling reuse/refresh controller | COMPLETE-REFERENCE | change thresholds and reuse budget |
| Primal-dual warm-start lifecycle | COMPLETE-REFERENCE | backend and session contracts |
| Checkpoint/restart | COMPLETE-REFERENCE | deterministic topology-locked state |
| Native persistent outer SCvx driver | COMPLETE-REFERENCE | create once, update thereafter, adaptive solves and trust region |
| Dense C++ ADMM debug backend | COMPLETE-REFERENCE | small QP/SOCP truth solver, not a performance claim |
| Stable C ABI | COMPLETE-REFERENCE | native dynamics/Jacobians/Lambert callable outside C++ |
| Installable native CMake package | COMPLETE-REFERENCE | exported targets and external consumer test |
| Pinned upstream one-shot PDHCG adapter | PREPARED | exact data and cone compatibility exists |
| Real one-shot PDHCG correctness run | GPU-BLOCKED | requires NVIDIA CUDA 12.4+ |
| Concrete persistent PDHCG CUDA workspace | GPU-BLOCKED | must retain preprocessing, scaling, iterates and buffers |
| In-place device coefficient updates | GPU-BLOCKED | implementation and measurement require CUDA |
| Device-side dynamics/Jacobian fill | GPU-BLOCKED | C++ formulae exist; CUDA kernels do not |
| CUDA graph capture and stream overlap | GPU-BLOCKED | meaningful only on hardware |
| DLPack/CUDA-array exchange | PREPARED | stable C ABI exists; accelerator pointer bridge remains |

# D — adaptive inexact and hybrid solving

| Item | Status | Notes |
|---|---|---|
| Repair/progress/refinement/polish forcing | COMPLETE-REFERENCE | Python and C++ implementations |
| Fixed-tolerance comparator | COMPLETE-REFERENCE | experiment baseline |
| Re-solve-before-shrink rule | COMPLETE-REFERENCE | implemented in outer drivers |
| Inexact error ledger | COMPLETE-REFERENCE | accumulated and relative diagnostics |
| Hybrid first-order/IPM plan | COMPLETE-REFERENCE | explicit handoff and final-polish contract |
| Conditional convergence argument | COMPLETE-REFERENCE | assumptions and proof obligations documented |
| Solver-independent incumbent qualification | COMPLETE-REFERENCE | status, residual and feasible-objective checks |
| Native outer-driver policy lifecycle | COMPLETE-REFERENCE | persistent backend contract |
| Full theorem with checked assumptions | OPEN-CPU | mathematical refinement and counterexamples remain |
| PDHCG adaptive-tolerance sweep | EXPERIMENT-BLOCKED | real CUDA PDHCG required |
| QOCO-GPU/CuClarabel comparison | EXPERIMENT-BLOCKED | compatible GPU builds required |
| Empirical crossover map | EXPERIMENT-BLOCKED | matched nonlinear-quality results required |

# C — robust scenario and multi-GPU optimisation

| Item | Status | Notes |
|---|---|---|
| Scenario information histories | COMPLETE-REFERENCE | deterministic scenario tree |
| Shared-prefix non-anticipativity | COMPLETE-REFERENCE | exact sparse rows |
| Block-arrow variable layout | COMPLETE-REFERENCE | local blocks and shared arrowhead |
| Condensed shared-column formulation | COMPLETE-REFERENCE | solver-independent repeated-local certificate |
| Deterministic scenario partition | COMPLETE-REFERENCE | whole-scenario load balancing |
| Communication-volume model | COMPLETE-REFERENCE | ring-allreduce accounting |
| Partition-invariant forward/transpose truth model | COMPLETE-REFERENCE | CPU comparison against monolithic operators |
| 3-DoF robust assembly | COMPLETE-REFERENCE | native and Python paths |
| 6-DoF robust assembly | COMPLETE-REFERENCE | generic 14-state scenario bundle |
| Expected/worst/VaR/CVaR evaluation | COMPLETE-REFERENCE | native and Python aggregation |
| Expected/worst/CVaR CQP augmentation | COMPLETE-REFERENCE | fixed-pattern affine-loss epigraphs |
| Known-incumbent solution qualification | COMPLETE-REFERENCE | catches degenerate IPM false positives |
| Native scenario-local CUDA shards | GPU-BLOCKED | device ownership and kernels required |
| NCCL non-anticipativity reductions | GPU-BLOCKED | multi-GPU node required |
| Communication/computation overlap | GPU-BLOCKED | real topology and profiling required |
| Strong and weak scaling | EXPERIMENT-BLOCKED | 2/4/8-GPU experiments required |

# E — OrbitWeaver multi-destination optimisation

| Item | Status | Notes |
|---|---|---|
| Native fidelity-ladder oracle contract | COMPLETE-REFERENCE | analytical through certified levels |
| Exact request cache and warm-start pipeline | COMPLETE-REFERENCE | solver-independent native implementation |
| Hohmann/phasing screening | COMPLETE-REFERENCE | Python/C++ analytical screen |
| Zero-revolution Lambert solver | COMPLETE-REFERENCE | universal-variable C++ solver |
| Executable Lambert screening oracle | COMPLETE-REFERENCE | ephemerides, matching impulses and mass closure |
| Edelbaum low-thrust screening | COMPLETE-REFERENCE | native radius/inclination estimate |
| Deterministic beam search | COMPLETE-REFERENCE | time/mass-dependent native search |
| Time-expanded moving-target graph | COMPLETE-REFERENCE | scheduled-arc truth graph |
| Exact elementary-route labels | COMPLETE-REFERENCE | small-instance truth model up to 64 targets |
| Optimistic route lower bounds | COMPLETE-REFERENCE | non-elementary dynamic-programming bound |
| Dynamic discretisation discovery | COMPLETE-REFERENCE | route-gap-driven epoch refinement |
| Route-column dominance and pricing | COMPLETE-REFERENCE | reduced-cost candidate filtering |
| Exact multi-spacecraft route master | COMPLETE-REFERENCE | small-instance set-partitioning truth model |
| Full iterative column-generation controller | OPEN-CPU | master/pricing primitives exist; iteration policy remains |
| Coarse convex arc adapter | OPEN-CPU | native CQP backend can be attached to the oracle |
| Refined deterministic SCvx arc adapter | PREPARED | native driver exists; orbital adapter remains |
| Robust SCvx arc adapter | PREPARED | scenario CQP exists; production multi-GPU backend remains |
| Multi-revolution Lambert families | OPEN-CPU | branch enumeration and regression cases remain |
| Final high-fidelity certification adapter | OPEN-CPU | chosen force model and validation policy required |
| Massive route × scenario throughput | EXPERIMENT-BLOCKED | central Paper 2 scaling result needs GPUs |

# Native and reference quality gates

| Gate | Status |
|---|---|
| Full warnings-as-errors native build | COMPLETE-REFERENCE |
| All host-native smoke targets | COMPLETE-REFERENCE |
| ASan/UBSan suite | COMPLETE-REFERENCE |
| macOS native build | COMPLETE-REFERENCE |
| Installed CMake package consumer | COMPLETE-REFERENCE |
| Python 3.11/3.12 lint and reference suite | COMPLETE-REFERENCE pending latest branch run |
| Condensed formulation certificate | COMPLETE-REFERENCE |
| Hardware benchmark provenance schema | COMPLETE-REFERENCE |

# Paper and experiment infrastructure

| Item | Status |
|---|---|
| Paper 1 benchmark protocol and manifests | COMPLETE-REFERENCE |
| Paper 2 experiment manifest | COMPLETE-REFERENCE |
| Failure/OOM/timeout reporting rules | COMPLETE-REFERENCE |
| Paper outlines and notation lock | OPEN-CPU |
| Paper 1 real tables and plots | EXPERIMENT-BLOCKED |
| Paper 2 large-scale tables and plots | EXPERIMENT-BLOCKED |

# Exactly what the absence of GPU runs blocks

No GPU blocks only work or claims that require a CUDA execution environment:

1. importing and executing the pinned upstream PDHCG package;
2. validating one-shot PDHCG against committed CPU references;
3. implementing and debugging persistent ownership against upstream internals;
4. demonstrating allocation-free device updates and residency;
5. comparing PDHCG against GPU interior-point solvers;
6. measuring GPU memory, utilisation, energy and crossover points;
7. implementing and validating NCCL/MPI reductions;
8. measuring strong/weak scaling and communication overlap;
9. validating GPU precision, determinism, streams and graph capture;
10. making any speedup, throughput, memory-scaling or energy claim.

The absence of a GPU does **not** block dynamics, transcriptions, outer-loop logic, risk
formulations, route algorithms, correctness certificates, theory, manifests or paper structure.

# Remaining pre-GPU priorities

In descending leverage:

1. coarse-convex and refined-SCvx OrbitWeaver arc adapters;
2. full iterative column generation with incumbent and bound provenance;
3. multi-revolution Lambert branch families;
4. integral continuous-time violation-state enforcement;
5. stronger variational integration where it beats finite-difference RK4;
6. theorem-level inexact-SCvx assumptions, lemmas and counterexample tests;
7. native wheel packaging and accelerator-pointer exchange design;
8. paper outlines, notation lock and figure-generation schemas;
9. additional adversarial, randomized and property-based tests.

# First GPU-day run order

1. Record the exact hardware/software manifest.
2. Build the pinned upstream PDHCG revision.
3. Run known-optimum banded QP/SOCP fixtures through the one-shot adapter.
4. Compare objective, residuals and cone feasibility against CPU references.
5. Run HCW and deterministic 3-DoF CQP fixtures.
6. Activate persistent workspace ownership and repeated numerical updates.
7. Compare one-shot versus persistent cold and warm solves.
8. Connect the native SCvx driver to the device workspace.
9. Run 6-DoF and low-thrust node-count sweeps.
10. Add scenario sharding and NCCL only after single-GPU correctness closes.
