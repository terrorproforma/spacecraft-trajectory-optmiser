# Native-core strategy

## Decision

SpacePDHCG will be a **C++/CUDA numerical engine with Python bindings and a Python research control plane**.

Python remains valuable for experiment configuration, reference implementations, ephemeris and data ingestion, plotting, paper analysis, and rapid formulation work. It must not remain in the repeated numerical hot path.

The production execution path is:

```text
Python / CLI / mission application
              |
              v
stable C ABI or thin Python binding
              |
              v
C++ mission and SCvx orchestration
              |
              v
C++/CUDA fixed-pattern transcription and coefficient updates
              |
              v
persistent PDHCG device workspace
              |
              v
CUDA kernels + NCCL collectives
```

## Code that belongs in C++ or CUDA

The following work is latency-, allocation-, or bandwidth-sensitive and is therefore part of the native core:

1. Owned CQP sparse structures, cone metadata, numerical buffers, validation, residuals and objective evaluation.
2. Spacecraft dynamics, propagation, analytic or automatic derivatives, and fixed-pattern transcription.
3. The SCvx outer-loop state machine, forcing policy, trust-region policy, warm-start ownership and acceptance tests.
4. Scenario-tree indexing, block-arrow operators, scenario partitioning and collective communication schedules.
5. PDHCG preprocessing, scaling, iterates, proximal subproblem state, residual evaluation and stopping logic.
6. Batched arc evaluation and the eventual high-volume OrbitWeaver continuous oracle.

## Code that should remain in Python

Python remains the primary layer for:

- experiment manifests and parameter sweeps;
- CPU correctness oracles using Clarabel and OSQP;
- test-case generation and property testing;
- visualisation and result analysis;
- paper tables and plots;
- external astrodynamics and ephemeris integrations;
- exploratory mission models;
- user-facing notebooks and scripting.

Python may call the native engine once per solve or batch. It should not execute per-knot sparse assembly, per-iteration cone operations, or per-scenario communication.

## What is not blocked by the absence of a GPU

The following can be completed and verified on ordinary CPU CI:

- native CQP ownership and structural validation;
- exact HCW propagation and rendezvous transcription;
- nonlinear 3-DoF and 6-DoF dynamics and Jacobian tests;
- backend-independent SCvx policies and lifecycle state machines;
- scenario trees, non-anticipativity layouts and deterministic partitioning;
- monolithic and partitioned operator equivalence tests;
- stable C ABI and optional Python bindings;
- benchmark schemas, manifests and independent residual checking;
- OrbitWeaver request, caching, routing and fidelity interfaces;
- convergence assumptions and proof obligations for inexact SCvx.

## What is genuinely blocked by the absence of a compatible GPU runner

A CUDA 12.4+ NVIDIA runner is required for:

- compiling and executing the pinned upstream PDHCG CUDA implementation;
- validating the real persistent device workspace;
- measuring host-to-device transfers and proving device residency;
- CUDA graph capture and stream-overlap validation;
- QOCO-GPU and CuClarabel comparisons;
- NCCL multi-GPU correctness and scaling;
- kernel profiling, memory-bandwidth analysis and energy measurements;
- any claimed GPU speedup, crossover point or multi-GPU efficiency.

Until those runs exist, the repository must make no GPU performance claim.

## Migration stages

### N1 — CPU-testable native foundation

- dependency-free C++20 CQP structures;
- exact HCW dynamics and fixed-pattern CQP generation;
- native SCvx forcing and trust-region policies;
- deterministic scenario partitioning;
- C ABI and compiled tests.

### N2 — Native deterministic trajectory engine

- port 3-DoF powered descent dynamics, transcription and outer loop;
- compare native numerical values and decisions with the Python oracle;
- make Python a thin experiment wrapper.

### N3 — Persistent PDHCG CUDA backend

- connect native structures to upstream PDHCG internals;
- allocate sparse and iterate buffers once;
- update coefficients in place;
- retain scaling and warm starts across SCvx iterations.

### N4 — Multi-GPU robust engine

- shard whole scenarios;
- implement shared-control reductions with NCCL;
- overlap local sparse products, projections and communication;
- verify against the monolithic CPU oracle.

### N5 — OrbitWeaver native continuous oracle

- batch candidate legs;
- use fidelity- and promise-dependent inner tolerances;
- preserve warm-start tokens across route search;
- expose deterministic, robust and certified arc results through the stable API.

## Performance rule

The criterion is not percentage of source code written in C++. The criterion is whether the repeated numerical work remains native and device-resident. A small Python control plane around a fully native solve is appropriate; a C++ executable that repeatedly reconstructs matrices or transfers data is not.
