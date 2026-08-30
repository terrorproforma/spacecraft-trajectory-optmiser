# SpacePDHCG

**Factorisation-Free Multi-GPU Successive Convexification for Robust Spacecraft Trajectory Optimisation**

SpacePDHCG is a research programme for building a persistent, device-resident trajectory-optimisation engine around PDHCG-CQP. The first paper combines:

- **B — persistent device-resident CT-SCvx**;
- **C — scenario-aware multi-GPU optimisation**; and
- **D — adaptive inexact solving with optional interior-point polishing**.

The resulting continuous trajectory oracle will then support **E — integrated multi-destination spacecraft routing and trajectory optimisation**.

## Research question

Can a persistent, scenario-structured, multi-GPU PDHCG-CQP backend reduce the total time and memory required for large robust spacecraft successive-convexification problems while preserving nonlinear feasibility and final solution quality?

A conditional result is useful: the project will produce a reproducible crossover map showing when first-order multi-GPU conic quadratic optimisation wins, when factorisation-based GPU solvers win, and when a hybrid is best.

## Current milestone

The repository is being built from the bottom up around a permanent correctness spine:

1. fixed-pattern conic quadratic problem contract;
2. Clohessy–Wiltshire rendezvous QP baseline;
3. repeated numerical updates and primal-dual warm starts without symbolic reconstruction;
4. nonlinear 3-DoF powered-descent CT-SCvx;
5. robust 6-DoF scenario optimisation and multi-GPU decomposition;
6. adaptive inexact solves and GPU interior-point polish;
7. multi-destination routing built on the finished trajectory oracle.

See [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the locked programme.

## Package layout

```text
src/spacepdhcg/
  cqp/          fixed sparse-pattern problem representation
  models/       spacecraft dynamics and benchmark models
  backends/     reference and accelerator solver adapters
  benchmarks/   reproducible latency, throughput and accuracy experiments

tests/          correctness and persistence tests
docs/           research scope, architecture and milestone gates
```

## Development status

Research software under active construction. Numerical results are not claimed until they are reproduced by the committed benchmark suite and CI artifacts.
