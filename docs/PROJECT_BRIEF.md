# SpacePDHCG project brief

## Factorisation-Free Multi-GPU Successive Convexification for Robust Spacecraft Trajectory Optimisation

**Version 0.1 — 30 August 2026**

## 1. Locked scope

SpacePDHCG combines three contributions:

- **B — Persistent device-resident CT-SCvx:** nonlinear rollout, differentiation, continuous-time transcription, conic data updates, convex solution, acceptance tests, and warm starts remain on the GPU.
- **C — Scenario-aware multi-GPU optimisation:** uncertainty scenarios are partitioned across GPUs while common controls and non-anticipativity constraints are coordinated with limited collective communication.
- **D — Adaptive inexact/hybrid solving:** early convex subproblems are solved coarsely with PDHCG-CQP; tolerances tighten with outer convergence; final outer iterations may be polished with a GPU interior-point solver.

The primary application is robust powered-descent and rendezvous guidance. Multi-destination optimisation is a later application of the resulting continuous trajectory oracle, not part of the first paper.

## 2. Central research question

Can a persistent, scenario-structured, multi-GPU PDHCG-CQP backend reduce the total time and memory required for large robust spacecraft SCvx problems while preserving nonlinear feasibility and final solution quality?

The project is successful even if the answer is conditional: a scientifically useful result is a reproducible crossover map identifying when first-order multi-GPU CQP beats factorisation-based GPU solvers and when it does not.

## 3. Scenario optimal-control problem

For scenarios `s = 1,...,S`, uncertain parameters `theta_s`, and probabilities `p_s`, solve

\[
\min_{\{x_s,u_s\},w,T}
\sum_{s=1}^{S}p_s
\left[
\Phi_s(x_s(T),T)+
\int_0^T \ell_s(x_s,u_s,t)\,dt
\right]
+\lambda_{\rm risk}\,\mathcal R(\{x_s\})
\]

subject to

\[
\dot x_s=f(x_s,u_s,\theta_s,t),
\qquad x_s(0)=x_{0,s},
\]

path and terminal constraints, and scenario-tree non-anticipativity. If scenarios `r` and `s` share the same information history at node `k`,

\[
u_{r,k}=u_{s,k}.
\]

A common open-loop prefix is the first implementation. Affine feedback and recourse policies are later extensions.

## 4. Convex subproblem at SCvx iteration j

After multiple-shooting discretisation and linearisation,

\[
x_{s,k+1}=
A^j_{s,k}x_{s,k}
+B^j_{s,k}u_{s,k}
+F^j_{s,k}\sigma_k
+d^j_{s,k}
+v_{s,k},
\]

where `v` is virtual control and `sigma` may include time-dilation variables.

Stack all scenario variables and shared decisions into `z`. The convex subproblem is

\[
\begin{aligned}
\min_z\quad &
\frac12 z^\top Q_j z+c_j^\top z
+\lambda_v\mathbf 1^\top t_v
+\lambda_s\mathbf 1^\top t_s\\
\text{s.t.}\quad & A_jz=b_j,\\
& G_jz+h_j\in\mathcal K,\\
& \|D_{j,b}(z_b-\bar z^j_b)\|_2\leq\Delta_{j,b},\\
& N_jz=0.
\end{aligned}
\]

`N_j z = 0` represents non-anticipativity. `K` is a Cartesian product of zero, nonnegative, second-order, and rotated second-order cones in the core spacecraft implementation. Exponential, power, and PSD cones are optional extensions.

Absolute-value penalties are represented by nonnegative epigraph variables. Convex quadratic penalties remain in the native quadratic objective rather than being lifted into an additional cone.

## 5. Continuous-time constraint satisfaction

For path constraints, introduce scenario-specific violation states

\[
\dot y_s=\Lambda(t,x_s,u_s)\geq0,
\]

and enforce over each interval

\[
y_{s,k+1}-y_{s,k}\leq\epsilon_{\rm ct}.
\]

Version 0.1 uses a fixed grid so the conic sparsity pattern remains unchanged across SCvx iterations. Mesh refinement is permitted only between optimisation episodes. Dynamic in-loop remeshing is deferred until after persistent-buffer performance is established.

## 6. Persistent device-resident architecture

### Reference path

OpenSCvx or an equivalent transparent implementation validates transcription, derivatives, objective values, constraints, and reference solutions.

### Performance path

The production path bypasses CVXPY and host-side SciPy assembly:

1. GPU rollout of every scenario and interval.
2. GPU automatic differentiation or custom variational integration.
3. GPU canonicalisation into preallocated CQP buffers.
4. In-place update of values in fixed sparse structures.
5. PDHCG solve with primal-dual warm start.
6. GPU nonlinear rollout and continuous-time feasibility check.
7. GPU calculation of actual/predicted reduction.
8. Device-side accept/reject and trust-region update.

### Required PDHCG extension

Create a `PersistentCQP` API with:

```text
initialize_structure(Q_pattern, A_pattern, cone_layout, partition)
update_values(device_Q, device_A, device_c, device_l, device_u, stream)
set_warm_start(device_x, device_y)
solve_async(tolerance, iteration_limit, stream)
get_device_solution()
get_device_residuals()
```

Data exchange should use direct CUDA pointers, DLPack, or the CUDA array interface. The hot loop must not reconstruct NumPy/SciPy matrices or recanonicalise symbolic expressions.

## 7. Multi-GPU decomposition

Let GPUs form a logical `G_s x G_t` grid:

- `G_s` partitions uncertainty scenarios.
- `G_t` partitions time blocks or variable columns within each scenario group.

The global constraint operator has block-arrow structure:

\[
A=
\begin{bmatrix}
A_1 & & & C_1\\
& A_2 & & C_2\\
& & \ddots & \vdots\\
& & & A_S\; C_S
\end{bmatrix},
\]

where each `A_s` is scenario-local and `C_s` couples local trajectories to shared controls or consensus variables.

Two implementations are compared:

1. **Generic PDHCG partition:** existing two-dimensional nonzero-balanced distribution.
2. **Scenario-aware partition:** scenario blocks remain local; only shared-control and consensus contributions use NCCL collectives.

The hypothesis is that scenario-aware partitioning lowers communication volume when the shared decision dimension is much smaller than the total scenario-local dimension.

## 8. Adaptive inexact solve policy

Define a scaled outer residual

\[
R_j=\max\{r^j_{\rm dyn},r^j_{\rm path},r^j_{\rm term},r^j_{\rm step}\}.
\]

The first-order target is

\[
\epsilon_j^{\rm FO}
=
\min\left\{
\epsilon_{\max},
 c_\epsilon R_j^{1+\alpha},
 \epsilon_0\gamma^j
\right\},
\qquad \alpha>0,\;0<\gamma<1.
\]

The practical solver uses

\[
\epsilon_j=\max\{\epsilon_{\rm floor},\epsilon_j^{\rm FO}\}
\]

until the polish phase. The theoretical variant removes the positive floor or imposes a summable error sequence.

For a candidate `z+`, calculate

\[
\rho_j=
\frac{\Psi(\bar z^j)-\Psi(z^+)}
{m_j(\bar z^j)-m_j(z^+)}.
\]

If the candidate is rejected and the normalised CQP KKT residual is still significant relative to predicted model improvement, re-solve the same CQP at `0.1 epsilon_j` before shrinking the trust region. This separates inner-solver error from model-linearisation error.

## 9. Hybrid polish policy

Use three phases:

1. **Exploration:** PDHCG, typically `1e-3` to `1e-4` scaled tolerance.
2. **Convergence:** adaptively tightened PDHCG tolerance.
3. **Polish:** QOCO-GPU for QP/SOCP problems, or CuClarabel when broader cone support is required.

A provisional switch occurs after two accepted outer iterations when

\[
R_j<R_{\rm switch},
\qquad
\frac{\|z^{j+1}-z^j\|_D}{\max(1,\|z^j\|_D)}<\tau_{\rm switch},
\qquad
\rho_j>\rho_{\rm good}.
\]

If the interior-point solver exceeds memory limits, PDHCG continues to the final requested accuracy.

## 10. Theoretical target

Establish convergence of inexact CT-SCvx when convex-subproblem errors satisfy either

\[
\sum_{j=0}^{\infty}\epsilon_j<\infty
\]

or a relative forcing condition such as

\[
\epsilon_j=O\!\left(\|z^{j+1}-z^j\|^{1+\alpha}\right).
\]

The target conclusion is stationarity of accumulation points for the exact-penalty nonlinear problem, with explicit accounting for inexact primal-dual CQP solutions. This is a research objective, not an assumed theorem.

## 11. Benchmark ladder

### B0 — Solver bridge and canonical form

- Random sparse QP/SOCP instances with trajectory-like band structure.
- Verify objective, primal residual, dual residual, cone feasibility, and warm starts.

### B1 — CW rendezvous

- Deterministic and scenario bundles.
- QP version, then keep-out and thrust SOC constraints.
- Purpose: exact correctness, scale, and partition tests without outer SCvx complications.

### B2 — Nonlinear 3-DoF powered descent

- Mass depletion, thrust bounds, glide-slope and tilt cone.
- Fixed-grid CT-SCvx.
- Purpose: validate persistent outer loop and inexact tolerance controller.

### B3 — Robust 6-DoF powered descent

- Attitude, angular rate, torque, thrust pointing, mass, and free-final-time variants.
- Navigation, thrust, gravity, and initial-state scenarios.
- Purpose: primary large-scale multi-GPU benchmark.

### B4 — Robust orbital rendezvous or low-thrust transfer

- Long horizon and high node count.
- Purpose: demonstrate that conclusions are not landing-specific.

## 12. Solver baselines

- PDHCG-CQP, cold and warm started.
- QOCO-GPU.
- CuClarabel.
- Custom PIPG/SeCO where the formulation permits it.
- CPU QOCO/Clarabel reference.
- Existing OpenSCvx backends for reproducibility.

All timings are end-to-end and include transcription, scaling, transfers, inner solve, nonlinear evaluation, and outer-loop overhead.

## 13. Required measurements

- Wall-clock latency and throughput.
- Number of outer iterations.
- Number of PDHCG outer and inner iterations.
- Sparse matrix-vector products and cone projections.
- Host-device transfer time.
- Peak memory per GPU.
- Inter-GPU bytes and collective time.
- Parallel efficiency.
- Final nonlinear dynamics defect.
- Continuous-time path violation.
- Terminal error.
- Objective gap to a high-accuracy reference.
- Energy per accepted trajectory where hardware counters permit it.

## 14. Falsifiable hypotheses

- **H1 — Persistence:** after initialisation, host-device and canonicalisation overhead is below 5% of total solve time for repeated SCvx iterations.
- **H2 — Scale crossover:** there exists a measurable `N x S` regime in which PDHCG is faster than GPU interior-point baselines at equivalent nonlinear feasibility and objective quality.
- **H3 — Memory crossover:** PDHCG solves robust trajectory instances whose factored KKT systems do not fit in available GPU memory.
- **H4 — Scenario structure:** scenario-aware partitioning outperforms generic nonzero-balanced partitioning on sufficiently large coupled ensembles.
- **H5 — Inexact scheduling:** adaptive tolerances reduce total runtime or total matrix-vector products by at least 2x versus a fixed high-accuracy inner tolerance while preserving final feasibility and objective within predefined limits.
- **H6 — Hybrid dominance:** PDHCG followed by an interior-point polish has lower end-to-end time than pure interior-point SCvx and higher final accuracy than loose-tolerance pure PDHCG.

Failure of a hypothesis is reportable if the crossover boundaries and causes are rigorously characterised.

## 15. Repository architecture

```text
spacepdhcg/
  cpp/
    persistent_cqp.hpp
    persistent_cqp.cu
    device_update_kernels.cu
    scenario_partition.cu
    nccl_collectives.cu
    python_bindings.cpp
  src/spacepdhcg/
    models/
    transcription/
    scvx/
    backends/
    distributed/
    benchmarks/
  tests/
  experiments/
  docs/
```

## 16. First executable milestone

The first milestone is complete when one process can:

1. Allocate a fixed CW-rendezvous CQP.
2. Update numerical coefficients in place for a new initial state and target.
3. Warm-start and solve repeatedly without reconstructing host sparse matrices.
4. Match a high-accuracy reference solution.
5. Produce a timing breakdown proving whether persistence removes setup overhead.

The CPU OSQP implementation in the initial repository is a contract and correctness baseline. It is deliberately shaped like the future persistent GPU API; it is not presented as the final performance implementation.

## 17. Initial publication framing

**Working title:**

*SpacePDHCG: Persistent Multi-GPU Inexact Successive Convexification for Robust Spacecraft Trajectory Optimisation*

**Core claimed contribution, conditional on results:**

A device-resident, factorisation-free, scenario-structured CT-SCvx architecture that scales robust spacecraft optimal-control problems across multiple GPUs, together with an adaptive inner-accuracy policy and high-accuracy hybrid polish.

## 18. Primary references from the programme brief

- Li et al., *GPU-Accelerated Conic Quadratic Programming with Local Linear Convergence under Strict Complementarity*, arXiv:2608.09159, 2026.
- Adams et al., *OpenSCvx: An Open-Source Modular and Extensible Nonlinear Trajectory Planning Package*, arXiv:2608.21631, 2026.
- Zou et al., *Parallel-in-Time Nonlinear Optimal Control via GPU-native Sequential Convex Programming*, arXiv:2603.10711, 2026.
- Chari and Acikmese, *QOCO-GPU: A Quadratic Objective Conic Optimizer with GPU Acceleration*, arXiv:2603.29197, 2026.
- Chen et al., *GPU Acceleration for a Conic Optimization Solver*, arXiv:2412.19027, latest revision 2025.
- Chari and Acikmese, *Spacecraft Rendezvous Guidance via Factorization-Free Sequential Convex Programming Using a First-Order Method*, arXiv:2402.04561, 2024.
