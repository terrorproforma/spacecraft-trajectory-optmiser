# Pre-GPU roadmap and execution gates

Status date: **2026-08-31**.

This document separates work that can be completed on ordinary CPU infrastructure from work
that requires an NVIDIA CUDA runner. A GPU is a validation and measurement dependency, not a
reason to stop architecture, modelling, theory, or reference implementation work.

## Status vocabulary

- **Complete** — implemented, committed, and covered by repository tests.
- **Reference complete** — mathematically executable through CPU/reference backends; native
  performance implementation remains.
- **Prepared** — interface, workflow, or experiment definition exists but has not run on the
  required hardware.
- **GPU-blocked** — correctness or performance cannot be established without CUDA execution.
- **Open, not GPU-blocked** — ordinary implementation or theory work remains and can proceed now.

# B — persistent device-resident SCvx

| Item | Status | GPU dependency |
|---|---|---|
| Canonical fixed-pattern CQP and cone contract | Complete | None |
| HCW QP/SOCP reference family | Complete | None |
| Random trajectory-banded QP/SOCP fixtures | Complete | None |
| Clarabel and OSQP correctness backends | Complete | None |
| One-shot upstream PDHCG adapter | Prepared | Real execution is GPU-blocked |
| C++ owning fixed-pattern CQP core | Complete | None |
| C++ 3-DoF dynamics, Jacobians, rollout, checks | Complete | None |
| C++ persistent workspace lifecycle/state contract | Complete | None |
| C++ powered-descent CQP coefficient filler | Open, not GPU-blocked | None |
| Cross-language CQP fixture/fingerprint parity | Open, not GPU-blocked | None |
| Thin Python/native binding | Open, not GPU-blocked | GPU zero-copy validation later |
| Concrete upstream PDHCG persistent workspace | Prepared design | CUDA build/runtime required |
| Persistent in-place device coefficient update | GPU-blocked | CUDA required |
| Device-side propagation and linearisation | GPU-blocked for validation | CUDA required |
| Device-resident nonlinear acceptance loop | GPU-blocked for validation | CUDA required |
| CUDA graph and stream-overlap study | GPU-blocked | CUDA required |

## B work that can still be completed before a GPU run

1. Port the complete fixed-pattern 3-DoF transcription to C++.
2. Add a backend-independent native SCvx session that reuses one workspace.
3. Add C++/Python binary fixture exchange and topology-fingerprint parity.
4. Add a stable C or pybind/nanobind API around the host core.
5. Port independent nonlinear checks and benchmark record generation to C++.
6. Add continuous-time or higher-order discretisation behind the same topology lifecycle.
7. Add 6-DoF rigid-body dynamics and reference transcription.
8. Add long-horizon low-thrust reference models.

# D — adaptive inexact and hybrid solving

| Item | Status | GPU dependency |
|---|---|---|
| Python adaptive repair/progress/refinement/polish policy | Complete | None |
| C++ adaptive forcing and trust-region policy | Complete | None |
| Re-solve-before-shrink logic | Complete in reference path | None |
| Inexact-error ledger and relative-forcing diagnostics | Complete | None |
| First-order to interior-point handoff contract | Complete | None |
| Conditional convergence assumptions/proof | Open, not GPU-blocked | None |
| Fixed versus adaptive CPU ablation | Open, not GPU-blocked | None |
| PDHCG tolerance/iteration ablation | GPU-blocked | CUDA required |
| QOCO-GPU/CuClarabel final-polish comparison | GPU-blocked | CUDA required |
| Empirical solver crossover surface | GPU-blocked | CUDA required |
| Energy per accepted trajectory | GPU-blocked | GPU telemetry required |

# C — robust scenario and multi-GPU optimisation

| Item | Status | GPU dependency |
|---|---|---|
| Scenario semantics and information histories | Complete | None |
| Exact shared-prefix non-anticipativity | Complete | None |
| Consensus and condensed monolithic CQP oracles | Reference complete | None |
| Expected-value, worst-case, and CVaR references | Reference complete | None |
| Deterministic whole-scenario partitioning | Complete | None |
| C++ scenario tree and block-arrow layout | Complete | None |
| Partitioned CPU forward/transpose operator parity | Reference complete | None |
| C++ block-arrow numerical assembler | Open, not GPU-blocked | None |
| Scenario checkpoint/restart format | Open, not GPU-blocked | None |
| MPI/NCCL process and ownership plan | Prepared | Runtime validation requires GPUs |
| CUDA scenario-local kernels | GPU-blocked for validation | CUDA required |
| NCCL shared-control/risk reductions | GPU-blocked | Multiple GPUs required |
| Communication/computation overlap | GPU-blocked | Multiple GPUs required |
| Strong and weak scaling | GPU-blocked | Multiple GPUs required |
| Multi-GPU failure/restart experiments | GPU-blocked | Multiple GPUs required |

# E — OrbitWeaver multi-destination programme

| Item | Status | Dependency |
|---|---|---|
| Stable trajectory-oracle API | Complete | None |
| Fidelity ladder and cache/warm-start tokens | Complete | None |
| Hohmann/phasing/rocket-equation screening | Complete | None |
| C++ zero-revolution Lambert solver | Complete | None |
| Deterministic beam-search baseline | Complete in Python | None |
| C++ route-state and beam-search core | Open, not GPU-blocked | None |
| Time-dependent transfer graph | Open, not GPU-blocked | None |
| Multi-revolution Lambert families | Open, not GPU-blocked | None |
| Low-thrust analytical screening | Open, not GPU-blocked | None |
| Coarse convex leg oracle | Depends on native B transcription | No GPU needed for reference |
| Refined SCvx leg oracle | Depends on B | Native speed claims need GPU |
| Robust leg oracle | Depends on C | Multi-GPU scale claims need GPUs |
| Column generation and route lower bounds | Open, not GPU-blocked | None |
| Multi-spacecraft assignment/resource coupling | Open, not GPU-blocked | None |
| Final high-fidelity certification | Open, not GPU-blocked | External propagator/model choices |

# What is actually blocked today

The absence of a GPU blocks only claims or implementation steps that require CUDA execution:

1. importing and executing the pinned upstream PDHCG package;
2. validating the native one-shot adapter against CPU references;
3. compiling and exercising the concrete persistent CUDA workspace;
4. proving that updates are allocation-free and remain on-device;
5. profiling kernels, transfers, scaling refreshes, streams, and CUDA graphs;
6. comparing PDHCG against GPU interior-point solvers;
7. running NCCL scenario reductions and multi-GPU scaling;
8. measuring energy, utilisation, and memory crossover points.

It does **not** block spacecraft models, C++ host architecture, fixed-pattern transcription,
cross-language parity, scenario semantics, robust formulations, inexact-solve theory, route
algorithms, Lambert screening, benchmark manifests, or paper structure.

# Critical path to Paper 1

```text
C++ transcription parity
        |
native binding and persistent session
        |
first pinned one-shot GPU correctness run
        |
concrete persistent CUDA workspace
        |
native deterministic SCvx parity
        |
D tolerance/hybrid experiments
        |
C NCCL scenario implementation
        |
strong/weak scaling and crossover matrix
        |
Paper 1 freeze
```

# Critical path to Paper 2

```text
C++ Lambert/time-dependent graph
        |
route lower bounds + beam/column-generation baselines
        |
coarse and refined B-backed leg oracle
        |
robust C-backed leg oracle
        |
multi-spacecraft resources and scheduling
        |
final trajectory certification
        |
Paper 2 freeze
```

# Completion rule

A roadmap box is not complete because an interface exists. It closes only when:

- its implementation is committed;
- deterministic tests pass;
- numerical outputs are independently checked;
- the applicable CPU or GPU environment is recorded;
- any performance claim is end-to-end and reproducible;
- limitations are stated beside the result.
