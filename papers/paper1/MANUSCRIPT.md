# SpacePDHCG: Device-Resident Multi-GPU Successive Convexification for Robust Spacecraft Trajectory Optimisation

> **Draft status:** methods skeleton. Numerical results, abstract claims, and conclusions remain
> intentionally blank until valid GPU manifests exist.

## Abstract

Large robust spacecraft trajectory problems contain regular parallelism across time nodes,
uncertainty scenarios, and candidate missions, but conventional conic solvers repeatedly assemble
and factorise global KKT systems. We present SpacePDHCG, a fixed-pattern successive-convexification
architecture designed around persistent factorisation-free conic quadratic solving. The system
combines device-resident coefficient updates, adaptive inexact inner solves, scenario-aware
multi-GPU decomposition, and optional interior-point polishing. [Insert only manifest-supported
problem sizes, timings, memory crossovers, and quality results.] Source code, frozen benchmark
configurations, raw run manifests, and figure-generation scripts accompany the paper.

## 1. Introduction

### 1.1 Motivation

- Robust powered descent, rendezvous, and low-thrust design create very large QP/SOCP families.
- Repeated SCvx solves share sparse topology but conventional interfaces rebuild symbolic and
  device state.
- Uncertainty scenarios expose substantial parallelism but generic matrix partitioning can
  communicate more than a scenario-aware decomposition.
- Early SCvx subproblems do not warrant final-solution accuracy.

### 1.2 Research questions

1. At what trajectory/scenario scale does a persistent first-order GPU solver beat GPU
   interior-point methods at matched nonlinear quality?
2. Does retaining quadratic objectives natively reduce work relative to conic epigraph lifting?
3. Can scenario-aware multi-GPU partitioning improve memory capacity and communication overlap?
4. How much end-to-end work does adaptive inner accuracy save?
5. Where does first-order construction plus interior-point polishing dominate either method alone?

### 1.3 Contributions

- **B:** a persistent C++/CUDA SCvx architecture with fixed CSC topology and in-place numerical
  updates around PDHCG-CQP;
- **D:** a residual-driven inner-accuracy schedule, re-solve-before-shrink safeguard, and hybrid
  polish policy;
- **C:** a scenario-tree block-arrow formulation and scenario-aware multi-GPU partition;
- a matched-quality benchmark and reproducibility protocol;
- spacecraft case studies spanning CW rendezvous, 3-DoF and 6-DoF powered descent, and robust
  scenario ensembles.

## 2. Related work

Sections to complete with verified primary-source citations:

- lossless convexification and successive convexification for spacecraft guidance;
- factorisation-free primal-dual methods for optimal control;
- GPU conic quadratic and sparse interior-point solvers;
- robust scenario optimisation, covariance steering, and branch/scenario MPC;
- GPU-accelerated direct collocation and indirect shooting;
- multi-target spacecraft mission design.

## 3. Problem formulation

### 3.1 Nonlinear optimal control

Define state, control, free parameters, dynamics, boundary conditions, path constraints, and
continuous-time certificate states. Distinguish the physical problem from its fixed-grid
transcription.

### 3.2 Convex subproblem

Present the native CQP form

\[
\min_z \frac12 z^TQ_jz+c_j^Tz
\quad\text{s.t.}\quad
\ell_j\le A_jz\le u_j,\qquad F_jz+g_j\in\mathcal K_j,
\]

including virtual control, trust regions, thrust/pointing cones, bounds, and exact penalties.

### 3.3 Fixed-pattern contract

Explain immutable CSC offsets/indices, mutable value buffers, cone metadata, topology fingerprints,
and the rule that a changed sparsity pattern creates a new workspace rather than silently
invalidating persistence.

## 4. SpacePDHCG architecture

### 4.1 C++/CUDA performance core

The production hot loop is:

```text
rollout -> differentiation -> coefficient update -> PDHCG solve
-> nonlinear verification -> trust-region/forcing decision
```

Python is restricted to orchestration, reference solvers, manifests, plots, and manuscript tools.

### 4.2 Persistent workspace

Describe ownership of device sparse structure, values, scaling, preconditioner, iterates, cone
workspaces, streams, events, and optional CUDA graph capture. Separate one-time setup, update,
rescale, solve, and residual timings.

### 4.3 Device-resident nonlinear operations

Document fused or batched kernels for propagation, Jacobians, coefficient insertion, continuous-
time violation states, and acceptance metrics. Specify precision and deterministic-reduction modes.

## 5. Adaptive inexact and hybrid solving

### 5.1 Outer residual and forcing schedule

Present the clipped relative/geometric forcing rule and all thresholds.

### 5.2 Re-solve-before-shrink

Show how the algorithm distinguishes a genuinely poor model step from an inadequately solved
convex subproblem.

### 5.3 Interior-point polish

Define the switch gate, canonical-value identity, primal-dual handoff, and termination conditions.

### 5.4 Convergence statement

Use `docs/INEXACT_SCVX_THEORY.md` as the proof checklist. Do not claim the theorem until all
assumptions and residual mappings are closed.

## 6. Scenario-aware multi-GPU method

### 6.1 Scenario tree and non-anticipativity

Define information histories, shared prefixes, recourse stages, scenario probabilities, and risk
objectives.

### 6.2 Block-arrow operators

Present consensus-row and condensed shared-column forms and prove their equivalence for repeated
solutions.

### 6.3 Logical GPU grid

Describe scenario and time partition axes, local storage, shared arrowhead data, collective
operations, overlap, and deterministic partitioning.

### 6.4 Risk measures

Specify expected value, worst-case epigraph, and CVaR formulations and their independent
recomputation.

## 7. Spacecraft benchmarks

### B0 — CW rendezvous QP

Linear dynamics, box thrust, exact CPU reference.

### B1 — CW rendezvous SOCP

Norm-bounded thrust and native cone coordinates.

### B2 — nonlinear 3-DoF powered descent

Mass depletion, thrust epigraph, tilt, glide slope, virtual control, and nonlinear rollout.

### B3 — robust 3-DoF powered descent

Gravity, thrust-scale, navigation, and mass scenarios with shared controls.

### B4 — 6-DoF powered descent

Quaternion attitude, angular rate, torque, pointing, mass, and landing constraints.

For every family, report variables, scalar rows, cone rows, nonzeros, conditioning indicators, and
requested/achieved tolerances.

## 8. Experimental protocol

Follow `docs/EXPERIMENT_PROTOCOL.md` and
`experiments/configs/paper1_sweep_v0.json`. Explain hardware, solver versions, timing boundaries,
warm-ups, replicates, matched-quality gates, failed-run retention, and statistical summaries.

## 9. Results

No values may be inserted without an archived manifest.

### 9.1 Correctness and residual agreement

[Blocked until real PDHCG/QOCO-GPU/CuClarabel runs.]

### 9.2 Persistence crossover

[Resolve H1.]

### 9.3 Accuracy and hybrid crossover

[Resolve H5–H6.]

### 9.4 Multi-GPU scaling and communication

[Resolve H2 and H4.]

### 9.5 Memory crossover

[Resolve H3.]

### 9.6 Energy

[Report only with calibrated measurement metadata.]

## 10. Discussion

Discuss regimes rather than declaring one universal winner. Expected dimensions include problem
size, cone mix, tolerance, scenario count, warm-start quality, conditioning, memory pressure, and
communication topology.

## 11. Limitations

- fixed-grid results do not by themselves prove continuous-time feasibility;
- SCvx gives local/stationary solutions, not global optimality;
- first-order convergence can be sensitive to scaling and degeneracy;
- CUDA and NCCL portability is narrower than the C++ reference layer;
- hardware-specific crossover boundaries should not be universalised;
- flight qualification and fault tolerance are outside Paper 1.

## 12. Conclusion

Write only after H1–H6 are resolved. The conclusion must report negative or mixed findings as
clearly as positive ones.

## Reproducibility statement

The release will include exact commits, pinned upstream dependencies, C++ and Python tests, frozen
sweep configurations, raw run manifests, derived tables, plotting scripts, hardware metadata, and
commands for every paper figure.
