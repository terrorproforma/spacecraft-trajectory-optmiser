# SpacePDHCG Paper 1 outline

## Working title

**SpacePDHCG: Persistent Factorisation-Free Multi-GPU Successive Convexification for Robust
Spacecraft Trajectory Optimisation**

No numerical performance result is asserted in this outline. Bracketed fields are populated only
from benchmark artifacts satisfying `docs/BENCHMARK_PROTOCOL.md`.

Scope status: the active manuscript build is `single-gpu-v1`. It may populate only the
scope-tagged one-GPU products and H1/H2/H3/H5/H6 claims. The multi-GPU title wording, contribution
C section, robust-scaling result structure, F07/F12/T06 products, and H4 claim below are preserved
as the historical full-paper plan and are deferred—not silently removed or presented as tested.

# Abstract structure

1. Spacecraft SCvx repeatedly solves same-topology conic quadratic subproblems.
2. Conventional pipelines rebuild host sparse objects, transfer data, and/or factorise a global
   KKT system.
3. SpacePDHCG retains topology, values, scaling state, and iterates on device and adds controlled
   inexact outer solves.
4. Scenario-aware decomposition exposes parallelism across trajectories and uncertainty.
5. Report the actual crossover against CPU and GPU interior-point references at matched nonlinear
   feasibility.

# 1. Introduction

- Why trajectory optimisation creates repeated CQP structure.
- Why one-shot GPU solver timing is not end-to-end SCvx timing.
- Why robust guidance creates the scale needed to justify GPUs and multiple GPUs.
- Contributions B, D, and C.
- Explicit non-contributions: no global optimality claim for nonlinear trajectories; no flight
  qualification claim.

# 2. Related work

## 2.1 Spacecraft convexification

- lossless convexification;
- successive convexification and continuous-time variants;
- PIPG/SeCO and generated conic solvers;
- GPU-powered-descent Monte Carlo and time-parallel SCP.

## 2.2 GPU conic and nonlinear optimisation

- PDHG/PDHCG;
- QOCO-GPU and GPU interior-point methods;
- JAX/batched CQP systems;
- fully GPU sparse NLP systems.

## 2.3 Robust and scenario optimal control

- non-anticipativity;
- scenario trees;
- covariance/chance alternatives;
- decomposition and communication structure.

# 3. Mathematical problem

Define nonlinear optimal control:

\[
\min J(x,u,p)
\quad\text{s.t.}\quad
\dot x=f(x,u,p),\quad
h(x,u,p)=0,\quad
g(x,u,p)\leq0.
\]

Define one SCvx subproblem in native quadratic-objective conic form:

\[
\begin{aligned}
\min_z\quad &\tfrac12 z^TQ_jz+c_j^Tz\\
\text{s.t.}\quad &\ell_j\leq A_jz\leq u_j,\\
&F_jz+g_j\in\mathcal K_j,\\
&\ell_j^x\leq z\leq u_j^x.
\end{aligned}
\]

Describe fixed topology and mutable values.

# 4. SpacePDHCG architecture

## 4.1 Owning C++20 CQP contract

- CSC topology fingerprint;
- cone ordering;
- host/device views;
- numerical epochs;
- checkpoint/restart.

## 4.2 Persistent device workspace

- allocation and upload once;
- in-place coefficient update;
- retained scaling and iterates;
- compact residual return;
- CUDA streams and graph capture.

## 4.3 Native spacecraft transcriptions

- HCW;
- 3-DoF powered descent;
- 14-state 6-DoF powered descent;
- long-horizon low thrust;
- Euler and RK4 reference modes;
- independent dense inter-node checks.

# 5. Contribution D: inexact SCvx

## 5.1 Forcing sequence

Define requested CQP residual as a function of nonlinear defect, step size, trust radius, accepted
streak, and model agreement.

## 5.2 Re-solve-before-shrink

Distinguish model failure from an unnecessarily loose inner solution.

## 5.3 Hybrid polishing

PDHCG for construction, interior-point solver for optional terminal polish.

## 5.4 Convergence result

State the conditional theorem from `docs/INEXACT_SCVX_THEORY.md`, then give the final assumptions
verified by the implementation.

# 6. Contribution C: scenario-aware multi-GPU SCvx

Define scenario local variables and information-history controls. Present the block-arrow system
and non-anticipativity constraints.

Explain whole-scenario device ownership and expected communication proportional to shared
controls/risk aggregates rather than all scenario states.

# 7. Experimental protocol

Refer to:

- `benchmarks/paper1_matrix.json`;
- `docs/BENCHMARK_PROTOCOL.md`.

## 7.1 Solvers

Clarabel, OSQP, upstream PDHCG one-shot, SpacePDHCG persistent, QOCO-GPU, CuClarabel, and selected
custom structured baselines.

## 7.2 Hardware

Populate exact CPU/GPU/interconnect/software table from run manifests.

## 7.3 Correctness

- known-optimum CQP fixtures;
- independent nonlinear propagation;
- continuous inter-node checks;
- monolithic versus distributed robust oracle.

# 8. Results

## 8.1 Correctness table

Columns:

- problem;
- dimensions and cone inventory;
- solver;
- objective gap;
- primal/dual residual;
- nonlinear defect;
- path violation.

## 8.2 Persistent update result

Cold setup, one-shot repeated solve, persistent first warm solve, persistent steady-state.

## 8.3 Horizon crossover

HCW, 3-DoF, 6-DoF, and low-thrust plots.

## 8.4 Accuracy policy ablation

Fixed versus adaptive versus adaptive-plus-polish.

## 8.5 Robust scaling

Scenario count, GPU count, communication time, load imbalance, memory, and accepted trajectory
throughput. **Deferred in `single-gpu-v1`; do not populate from logical/one-rank evidence.**

## 8.6 Regime map

Identify where each solver family wins; do not force a universal winner.

# 9. Discussion

- numerical conditioning and scaling reuse;
- tolerance meaning across solvers;
- communication crossover;
- memory limitations;
- onboard versus ground applicability;
- limitations of finite scenario sets and fixed-grid SCvx.

# 10. Conclusion

State only measured findings. Connect the stable continuous oracle to OrbitWeaver without claiming
Paper 2 results.

# Planned figures

1. Architecture and device-residency diagram.
2. Fixed topology versus mutable value diagram.
3. Scenario block-arrow/GPU partition diagram.
4. End-to-end horizon crossover.
5. Peak memory crossover.
6. Adaptive-accuracy ablation.
7. Strong/weak multi-GPU scaling.
8. Timing decomposition.
9. Accuracy–time Pareto surface.
10. Solver regime map.

# Planned appendices

- canonical cone conventions;
- complete problem dimensions;
- inexact-SCvx proof;
- hardware/build manifests;
- solver parameter tables;
- additional failures and OOM results;
- reproducibility commands.
