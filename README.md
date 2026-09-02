# SpacePDHCG

**Factorisation-Free Multi-GPU Successive Convexification for Robust Spacecraft Trajectory Optimisation**

SpacePDHCG is a research programme for building a persistent, device-resident trajectory-optimisation engine around PDHCG-CQP. The first paper combines:

- **B — persistent device-resident CT-SCvx**;
- **D — adaptive inexact solving with optional interior-point polishing**; and
- **C — scenario-aware multi-GPU optimisation**.

The resulting continuous trajectory oracle will then support **E — integrated multi-destination spacecraft routing and trajectory optimisation**.

## Research question

Can a persistent, scenario-structured, multi-GPU PDHCG-CQP backend reduce the total time and memory required for large robust spacecraft successive-convexification problems while preserving nonlinear feasibility and final solution quality?

A conditional result is useful: the project will produce a reproducible crossover map showing when first-order multi-GPU conic quadratic optimisation wins, when factorisation-based GPU solvers win, and when a hybrid is best.

## Current status

- **M0 — repository and numerical contract:** complete at the CPU reference level.
- **M1 — native conic bridge:** active; CPU QP/SOCP references are green and the upstream one-shot adapter is implemented for CUDA integration.

The executable correctness spine contains:

1. immutable CSC structure and mutable CQP values;
2. exact-discrete Clohessy–Wiltshire dynamics;
3. a fixed-workspace OSQP rendezvous QP baseline;
4. native PDHCG-compatible affine cone metadata;
5. a persistent Clarabel SOCP reference with explicit PDHCG cone-coordinate conversion;
6. an optional `PDHCGOneShot` adapter mapping the same canonical problem into upstream `pdhcg.Model`;
7. independent endpoint, dynamics and thrust checks;
8. Python 3.11/3.12 CI and repeat-solve smoke benchmarks.

The public upstream PDHCG API is currently one-shot at the device-workspace level. The main B contribution is therefore a lower-level `PersistentCQP` extension, not a wrapper around `Model.optimize()`.

## Programme ladder

1. fixed-pattern QP/SOCP correctness and cross-solver fixtures;
2. real CUDA execution through the one-shot PDHCG adapter;
3. persistent C++/CUDA ownership, in-place updates and device warm starts;
4. nonlinear 3-DoF powered-descent CT-SCvx;
5. adaptive inexact solves and interior-point polish;
6. robust 6-DoF scenario optimisation and multi-GPU decomposition;
7. Paper 1 crossover study and release;
8. multi-destination routing built on the finished trajectory oracle.

See:

- [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/COMPARATIVE_SOLVER_CAMPAIGN.md`](docs/COMPARATIVE_SOLVER_CAMPAIGN.md)
- [`docs/LITERATURE_TARGETS.md`](docs/LITERATURE_TARGETS.md)
- [`docs/REFERENCE_REPRODUCTION_REPORT.md`](docs/REFERENCE_REPRODUCTION_REPORT.md)
- [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md)
- [`docs/MILESTONES.md`](docs/MILESTONES.md)
- [`docs/PDHCG_INTEGRATION.md`](docs/PDHCG_INTEGRATION.md)
- [`docs/adr/0001-pdhcg-native-canonical-form.md`](docs/adr/0001-pdhcg-native-canonical-form.md)
- [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md)

## Package layout

```text
src/spacepdhcg/
  cqp/          fixed sparse-pattern native CQP representation
  models/       spacecraft dynamics and benchmark models
  backends/     persistent CPU references and accelerator adapters
  benchmarks/   reproducible latency, throughput and accuracy experiments

tests/          algebraic, solver and trajectory feasibility tests
docs/           research scope, decisions, architecture and milestone gates
```

## Commands

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest
spacepdhcg-cw-benchmark --repeats 20 --intervals 40
spacepdhcg-cw-socp-benchmark --repeats 20 --intervals 40
```

The upstream PDHCG package is intentionally optional because it requires a compatible NVIDIA CUDA environment. CPU installation and CI do not import it.

## Development status

Research software under active construction. Numerical results are not claimed until they are reproduced by committed benchmark configurations, independent feasibility checks and CI artifacts.
