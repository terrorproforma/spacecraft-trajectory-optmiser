# Pre-GPU completion gate

This document separates work that can be completed on ordinary CPU infrastructure from claims that require a compatible NVIDIA runner.

## Current architectural position

The project now has two deliberately independent paths:

- a transparent Python reference and experiment layer; and
- a C++20 native numerical core intended to become the production C++/CUDA engine.

The native core contains owned sparse CQP structures, exact HCW dynamics, fixed-pattern rendezvous transcription, nonlinear 3-DoF powered-descent dynamics and Jacobians, fixed-pattern powered-descent CQP updates, SCvx forcing/trust policies, scenario partitioning, scenario trees, block-arrow layouts, a stable C ABI, and compiled correctness tests.

## Definition of pre-GPU complete

The programme is pre-GPU complete when all of the following hold on CPU CI.

### Native numerical parity

- [x] Exact HCW state and input matrices in C++.
- [x] HCW semigroup and zero-order-hold composition tests.
- [x] Fixed-pattern CW QP/SOCP generation in C++.
- [x] C ABI parity against the Python HCW oracle.
- [x] Native 3-DoF powered-descent dynamics and analytic Jacobians.
- [x] Finite-difference derivative tests.
- [x] Euler and RK4 rollout implementations.
- [x] Native nonlinear path diagnostics.
- [x] Fixed-pattern 3-DoF powered-descent CQP generation and in-place numerical updates.
- [x] Reference-trajectory interpolation tests for dynamics, cones and trust regions.
- [ ] Native/Python elementwise parity fixtures for every powered-descent CQP array.
- [ ] A native 6-DoF dynamics model and derivative suite.

### Persistent lifecycle

- [x] Immutable structure versus mutable values is explicit.
- [x] Native workspace state and stream semantics are defined.
- [x] Warm-start, solution-staleness and update epochs are represented.
- [x] A stable C ABI exists for incremental native exposure.
- [ ] Concrete host-memory staging objects map owned CQP data to the persistent backend ABI.
- [ ] The complete deterministic SCvx outer loop runs in C++ against a backend interface.
- [ ] A CPU fake/reference backend exercises repeated native update/warm-start/solve transitions.

### Inexact solving

- [x] Adaptive forcing and trust-region policies exist in Python and C++.
- [x] Re-solve-before-shrink is represented.
- [x] Accumulated-error and relative-forcing diagnostics are executable.
- [x] Mathematical assumptions and proof obligations are documented.
- [ ] A backend-independent primal-dual residual and gap recomputation module exists in C++.
- [ ] Fixed/adaptive/hybrid experiment manifests are generated deterministically.
- [ ] The inexact-SCvx convergence argument is completed to paper standard.

### Robust scenarios

- [x] Scenario semantics and probabilities are validated.
- [x] Information histories and recourse splits are represented.
- [x] Native block-arrow variable ownership is deterministic.
- [x] Exact non-anticipativity operators are generated in C++.
- [x] Deterministic whole-scenario load partitioning exists.
- [x] Collective payload accounting exists.
- [ ] Native monolithic robust CQP assembly matches the Python oracle elementwise.
- [ ] Expected-value, worst-case and CVaR epigraph transformations are native.
- [ ] Native scenario checkpoint/warm-start ownership is serialisable.

### Benchmark integrity

- [x] Random trajectory-banded QP/SOCP fixtures exist.
- [x] Known-optimum and independently checked diagnostics exist in Python.
- [x] Compiler-independent native smoke tests are configured.
- [ ] Machine-readable benchmark manifests include source revisions, build flags and hardware.
- [ ] Native benchmark output schema is stable.
- [ ] CPU reference results are regenerated and committed only as small provenance records, not performance marketing.

### OrbitWeaver

- [x] Stable arc-oracle and fidelity concepts exist.
- [x] Analytical routing and caching baseline exists.
- [ ] Lambert screening is implemented and independently verified.
- [ ] Route lower bounds and dominance pruning are formalised.
- [ ] Time-expanded and column-generation reference methods exist.
- [ ] The native continuous-oracle batch contract is frozen.
- [ ] Refined, robust and certified fidelity adapters use the completed trajectory engine.

## GPU-blocked gates

The following cannot be honestly closed without real hardware execution.

### Single-GPU PDHCG

- build the pinned upstream revision with CUDA 12.4 or newer;
- compare real PDHCG primal/dual results against CPU references;
- validate cone conventions and stopping residuals on-device;
- implement and measure persistent device allocations;
- prove no full problem reconstruction or unintended host transfer occurs between SCvx iterations;
- measure cold start, value update, warm start, solve and residual phases separately.

### Solver crossover

- compare persistent PDHCG with QOCO-GPU, CuClarabel and relevant CPU references;
- identify node/scenario/accuracy crossover regions;
- measure memory exhaustion and fill-in regimes;
- profile kernels, sparse products, projections, scaling and communication;
- measure energy only with disclosed hardware and method.

### Multi-GPU

- implement NCCL collectives and MPI/process ownership;
- validate partitioned results against the monolithic oracle;
- measure strong and weak scaling;
- quantify communication/computation overlap;
- test failure, cancellation and checkpoint behaviour across ranks.

## Claims policy

Before the GPU gates close, permitted claims are limited to architecture, mathematical equivalence, CPU correctness, fixed-pattern update behaviour, and benchmark readiness.

The project must not claim:

- GPU acceleration;
- device residency in an executed solve;
- single- or multi-GPU speedup;
- energy reduction;
- a measured crossover against an interior-point GPU solver;
- production or flight readiness.

## Immediate sequence

1. Finish native powered-descent CQP parity against Python.
2. Add a native backend-neutral SCvx driver and fake persistent backend.
3. Port native robust CQP assembly and risk epigraphs.
4. Freeze benchmark manifests and result schemas.
5. Complete inexact-SCvx proof obligations.
6. Build Lambert screening and route lower bounds for OrbitWeaver.
7. Execute the pinned GPU workflow when hardware becomes available.
