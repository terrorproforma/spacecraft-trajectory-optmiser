# Roadmap status and dependency ledger

**Status date:** 31 August 2026  
**Scope:** SpacePDHCG Paper 1 (B + D + C), followed by OrbitWeaver Paper 2 (E)

This document is the repository source of truth. A feature is marked complete only when code,
tests, and the relevant evidence are committed. Solver speed, memory scaling, and multi-GPU
claims remain unverified until a compatible NVIDIA runner produces archived result manifests.

## Status vocabulary

- **Complete:** implemented and exercised by committed automated tests.
- **CPU-complete:** mathematical/reference implementation is tested, but the production CUDA
  path or GPU result is still missing.
- **GPU-blocked:** implementation or scientific conclusion requires CUDA hardware, an NVIDIA
  driver, or multiple GPUs.
- **Open:** useful work remains and is not intrinsically blocked by hardware.

## Architecture decision

Python is retained as the transparent research and orchestration layer. C++20 owns the numerical
core, and CUDA will extend that core rather than reimplementing the programme behind a Python
API. The production hot loop is intended to be:

```text
C++/CUDA rollout -> differentiation -> fixed-pattern coefficient update
-> persistent PDHCG solve -> nonlinear verification -> trust-region decision
```

Python remains appropriate for experiment manifests, reference comparisons, plotting,
manuscript generation, and rapid mission-model prototyping. Host-side SciPy canonicalisation and
Python iteration inside the production solve loop are prohibited.

## M0 — repository and numerical contract

| Item | Status | Evidence / remaining work |
|---|---|---|
| Python package, lint, unit tests, CI | Complete | Python 3.11/3.12 workflow |
| Immutable CSC topology and mutable values | Complete | Python CQP contract and C++ `core/cqp.hpp` |
| Solver-independent backend lifecycle | Complete | Python protocol and native `PersistentCQP` interface |
| Exact HCW dynamics | Complete | Python matrix exponential reference and C++ closed-form ZOH model |
| Fixed-pattern CW QP | Complete | Python and native C++ transcription |
| Fixed-pattern CW SOCP | Complete | Python and native C++ SOC transcription |
| Repeated CPU workspace updates | Complete | OSQP/Clarabel references and persistent C++ host PDHG |

**M0 gate:** complete at CPU/reference level.

## M1 — B0/B1 conic bridge and correctness

| Item | Status | Evidence / remaining work |
|---|---|---|
| Trajectory-banded exact-optimum QP/SOCP fixtures | Complete | Python correctness suite |
| Clarabel and OSQP references | Complete | Independent feasibility diagnostics |
| One-shot upstream PDHCG adapter | CPU-complete | API mapping is tested with a fake upstream module |
| Pinned upstream revision | Complete | `third_party/pdhcg.lock.json` |
| Real one-shot PDHCG CUDA solve | GPU-blocked | Manual `gpu-validation` workflow is committed |
| QOCO-GPU adapter and results | GPU-blocked | Backend integration and hardware run remain |
| CuClarabel adapter and results | GPU-blocked | Backend integration and hardware run remain |
| Cross-backend objective/feasibility agreement | GPU-blocked | CPU side passes; accelerator solvers remain |

**M1 gate:** open until real accelerator backends pass the same exact-optimum fixtures.

## M2 — B: persistent device-resident CT-SCvx

| Item | Status | Evidence / remaining work |
|---|---|---|
| Nonlinear 3-DoF powered-descent dynamics | Complete | Python reference and C++ analytic model/Jacobians |
| Fixed-pattern 3-DoF convex subproblem | Complete | Python QP/SOCP transcription |
| 3-DoF outer SCvx loop | Complete | Adaptive forcing, virtual control, trust-region acceptance |
| Independent nonlinear rollout and path checks | Complete | Euler/RK4 references and C++ diagnostics |
| Fixed-grid continuous-time violation certificates | Complete | C++ trapezoidal/Simpson violation-state integration |
| Persistent host first-order workspace | Complete | C++ PDHG value updates, warm starts, SOC/RSOC projection |
| Native 6-DoF rigid-body dynamics | Complete | 14-state C++ quaternion model and variational finite differences |
| 6-DoF fixed-pattern SCvx transcription | Open | Dynamics exist; conic transcription and outer-loop integration remain |
| Persistent upstream PDHCG device ownership | GPU-blocked | Requires concrete CUDA integration with upstream internal state |
| In-place device coefficient kernels | GPU-blocked | Kernel code and device-pointer binding remain |
| Device-side rollout/differentiation/canonicalisation | GPU-blocked | CPU truth models are ready for porting |
| Device-side acceptance and CUDA graph capture | GPU-blocked | Requires completed persistent device workspace |
| Persistence overhead hypothesis H1 | GPU-blocked | Cannot measure without device execution |

**M2 gate:** CPU/reference path is substantially complete; the defining device-resident claim is
not complete.

## M3 — D: adaptive inexact and hybrid solving

| Item | Status | Evidence / remaining work |
|---|---|---|
| Scaled outer residual | Complete | Python and C++ policy cores |
| Adaptive forcing schedule | Complete | Exploration, convergence, and polish phases |
| Re-solve-before-shrink logic | Complete | Solver-independent decision logic |
| Trust-region controller | Complete | Python and C++ state machines |
| Hybrid PDHCG -> QOCO/CuClarabel plan | Complete | Backend selection policy and warm-start contract |
| Fixed-vs-adaptive CPU ablation | Open | Experiment driver and frozen configurations remain |
| Summable/relative-error convergence argument | Open | Mathematical proof obligations are not hardware-blocked |
| H5 inexact-scheduling result | GPU-blocked | Requires matched GPU solver work/quality measurements |
| H6 hybrid-dominance result | GPU-blocked | Requires GPU interior-point backends |

**M3 gate:** algorithmic policy is complete; theory and empirical hypotheses remain open.

## M4 — C: scenario-aware multi-GPU robustness

| Item | Status | Evidence / remaining work |
|---|---|---|
| Scenario tree and information histories | Complete | Python and C++ implementations |
| Common open-loop prefix | Complete | Exact information-node groups |
| Consensus-row block-arrow formulation | Complete | Monolithic correctness oracle |
| Condensed shared-column formulation | Complete | Exact repeated-solution formulation oracle |
| Deterministic scenario partitioning | Complete | C++/Python longest-processing-time partition |
| Communication-volume model | Complete | Shared-arrowhead ring-allreduce accounting |
| Expected-value robust CQP | CPU-complete | Monolithic CPU oracle |
| Worst-case and CVaR risk formulations | Open | Existing robust-risk layer must be reconciled with current repo state and frozen tests |
| Robust 6-DoF SCvx | Open | 6-DoF dynamics exist; scenario transcription remains |
| Scenario-aware CUDA sharding | GPU-blocked | Device-local block storage and kernels remain |
| NCCL collectives and overlap | GPU-blocked | Requires at least two accessible GPUs |
| Generic-vs-scenario-aware comparison | GPU-blocked | H4 requires multi-GPU runs |
| Strong/weak scaling and memory crossover | GPU-blocked | H2-H4 remain empirical questions |

**M4 gate:** decomposition mathematics is ready; actual distributed execution is not.

## M5 — Paper 1 release

| Item | Status | Evidence / remaining work |
|---|---|---|
| Frozen benchmark taxonomy | Open | Problem families exist; final sweep grids need locking |
| Machine-readable run manifest | Open | Schema and hardware capture still need committing |
| CPU reference tables | Open | Can be generated before GPU access |
| GPU result tables and crossover plots | GPU-blocked | No valid accelerator run exists yet |
| Manuscript skeleton and methods sections | Open | Can be drafted now |
| Results and conclusions | GPU-blocked | Must follow evidence, not precede it |
| Tagged reproducibility release | Open | Follows green CPU/GPU gates |

## M6 — E prototype: moving-target trajectory oracle

| Item | Status | Evidence / remaining work |
|---|---|---|
| Backend-independent `evaluate_arcs` contract | Open | Not yet present in the repository |
| Analytical Hohmann/phasing screening | Open | Can be built in C++ now |
| Batched analytical cost matrix | Open | Can be built and CPU-tested now |
| Lambert screening | Open | Not hardware-blocked |
| Coarse convex arc adapter | Open | Depends on stable deterministic CQP API, not GPU results |
| Refined SCvx arc adapter | Open | Depends on credible M2 deterministic solver |
| Robust SCvx arc adapter | Blocked by M4 | Requires scenario solver semantics; GPU scale is later |
| Beam search / column-generation prototype | Open | Can use analytical costs first |
| Fixed-sequence multi-rendezvous refinement | Open | Requires continuous leg optimiser integration |

**M6 gate:** not yet reached in the repository; analytical and API work can start before Paper 1
performance results.

## M7 — E robust multi-destination engine

| Item | Status | Evidence / remaining work |
|---|---|---|
| Multiple spacecraft and moving targets | Open | Paper 2 programme |
| Candidate-route x scenario parallelism | Blocked by M4/M6 | Requires both route layer and robust solver |
| Resource transfer, depots, service windows | Open | Mission-model work is hardware-independent |
| High-fidelity final-route certification | Open | Requires mature trajectory models |
| Paper 2 experiments | Blocked by M6/M4 | No scientific result yet |

## Work that can continue without a GPU

1. Complete the C++ 6-DoF SCvx transcription and derivative validation.
2. Port 3-DoF canonicalisation and outer-loop hot operations from Python to C++.
3. Add pybind11/DLPack bindings around stable C++ types without placing Python in the hot loop.
4. Finish continuous-time path-constraint integration in the convex transcription.
5. Write the inexact-SCvx assumptions, lemmas, and proof obligations.
6. Freeze benchmark configurations and result-manifest schemas.
7. Implement OrbitWeaver's analytical arc API, Lambert screening, and route search.
8. Add mission/resource models and fixed-sequence refinement interfaces.
9. Build deterministic distributed-operator simulations and communication accounting on CPU.
10. Draft Paper 1 methods, software architecture, and experiment protocol sections.

## Work genuinely blocked by missing GPU runs

- Validating upstream PDHCG against real CUDA rather than a fake module.
- Measuring whether persistent allocation and in-place updates reduce device overhead below 5%.
- Comparing PDHCG against QOCO-GPU and CuClarabel at matched nonlinear quality.
- Establishing speed, memory, energy, and accuracy crossover boundaries.
- Executing NCCL collectives and measuring communication overlap.
- Resolving H1-H6 empirically.
- Writing evidence-based Paper 1 results and conclusions.

No speedup, scale, memory, or energy claim should be made before those gates are executed.
