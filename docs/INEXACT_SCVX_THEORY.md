# Inexact successive-convexification theory target

This note defines the theorem the D contribution must prove. It is a proof programme, not a claim
that convergence has already been established.

## 1. Nonlinear problem

Let the discretised spacecraft problem be

\[
\min_z J(z)
\quad\text{subject to}\quad
c(z)=0,\qquad g(z)\le 0,
\]

with an exact-penalty merit function

\[
\Phi_\rho(z)=J(z)+\rho_c\|c(z)\|_1+\rho_g\|[g(z)]_+\|_1.
\]

At accepted iterate \(z_j\), SCvx constructs a convex conic quadratic model over step \(d\):

\[
\begin{aligned}
\min_d\quad & m_j(d) \\
\text{s.t.}\quad & A_j d+b_j=0,\\
& F_jd+h_j\in\mathcal K_j,\\
& \|D_jd\|_2\le \Delta_j,
\end{aligned}
\]

including virtual control and slack variables with an exact-penalty weight. The exact subproblem
minimiser is \(d_j^\star\); the solver returns \(\tilde d_j\).

## 2. Inner-solve error measure

The backend-independent stopping quantity is a scaled conic KKT residual

\[
\eta_j=\max\{r_{p,j},r_{d,j},r_{c,j}\},
\]

where the components are primal feasibility, dual stationarity, and cone/complementarity
residuals in the unscaled canonical coordinates. Solver-native relative residuals may be used only
after their mapping to this definition is documented and tested.

The implementation must support two admissible schedules.

### Summable schedule

\[
\sum_{j=0}^{\infty}\eta_j < \infty.
\]

A practical geometric schedule \(\eta_j\le \eta_0\gamma^j\), \(0<\gamma<1\), satisfies this
condition unless bounded below by a nonzero numerical floor. A fixed floor therefore gives only a
neighbourhood result, not exact stationarity.

### Relative forcing schedule

\[
\eta_j\le
\kappa\min\left\{1,\,R_j^{1+\alpha},\,\Delta_j^{1+\alpha},\,
\|\tilde d_j\|^{1+\alpha}\right\},
\qquad \alpha>0,
\]

where \(R_j\) is the scaled nonlinear outer residual. The committed practical rule is a clipped
version of this condition combined with a geometric safeguard.

## 3. Assumptions to verify

A complete theorem must state and justify at least the following assumptions.

**A1 — smoothness and compactness.** The accepted iterates and trial points remain in a compact
set on which the nonlinear dynamics, objective, and constraint functions are continuously
differentiable with Lipschitz derivatives.

**A2 — first-order model consistency.** At \(d=0\), the convex model matches the nonlinear
functions and first derivatives. Model error is bounded by \(L\|d\|^2\) on the trust region.

**A3 — uniform subproblem regularity.** The quadratic model has uniform strong convexity on the
reduced feasible directions, either naturally or through explicit proximal regularisation.
Cone geometry and scaling do not destroy the error bound used below.

**A4 — feasible convexification.** Virtual control and slack variables make every trust-region
subproblem feasible, and their penalty weights are eventually above the relevant multiplier
thresholds so accepted limit points have zero artificial variables.

**A5 — bounded multipliers and constraint qualification.** A suitable conic constraint
qualification holds near accumulation points and the exact subproblem multiplier sequence is
bounded.

**A6 — truthful acceptance.** Predicted reduction is computed from the same convex model supplied
to the inner solver. Actual reduction and feasibility are evaluated by independent nonlinear
propagation and continuous-time path checks.

**A7 — controlled scaling.** The canonical residual reported after Ruiz or cone-preserving
scaling is converted to an unscaled residual with bounded equivalence constants.

## 4. Required perturbation lemmas

### Lemma L1 — distance to the exact subproblem solution

Under A3–A5, establish a local error bound of the form

\[
\|\tilde d_j-d_j^\star\|\le C_1\eta_j.
\]

For merely convex, non-strongly-convex subproblems, replace point distance with distance to the
solution set and identify the additional regularity required.

### Lemma L2 — model-value perturbation

Use Lipschitz gradients and L1/SOC error bounds to show

\[
|m_j(\tilde d_j)-m_j(d_j^\star)|
\le C_2\eta_j\|d_j^\star\|+C_3\eta_j^2.
\]

This lemma quantifies when an inaccurate solve can change the sign of predicted reduction.

### Lemma L3 — acceptance-ratio stability

For productive exact steps whose predicted reduction is bounded below by
\(c\|d_j^\star\|^2\), show that the difference between exact and inexact agreement ratios tends
to zero under the relative forcing schedule.

### Lemma L4 — rejected-step safeguard

Prove that the re-solve-before-shrink rule cannot loop indefinitely: either the refined inner
residual meets the forcing condition, the backend declares failure, or the trust radius is reduced.
The maximum number of re-solves in the implementation is finite.

### Lemma L5 — vanishing artificial variables

With an exact-penalty weight above the limiting multiplier norm, any stationary accumulation point
of the penalised model has zero virtual control and zero feasibility slack.

## 5. Target convergence statement

A candidate theorem is:

> Under A1–A7, suppose the trust-region acceptance parameters satisfy the standard ordering,
> subproblems are solved with either summable KKT errors or the relative forcing schedule, and the
> exact-penalty weights are eventually sufficiently large. Then every accumulation point of the
> accepted iterate sequence is first-order stationary for the discretised nonlinear trajectory
> problem. If a positive inner residual floor is retained, the stationarity residual is bounded by
> a constant multiple of that floor.

The statement must distinguish stationarity of the fixed discretisation from feasibility of the
continuous-time trajectory. The violation-state certificate supplies the latter only up to its
quadrature and integration assumptions.

## 6. Hybrid polishing

Switching from PDHCG to an interior-point solver is treated as a change of inner algorithm, not a
change of subproblem. The polish solver must receive identical canonical values and a documented
primal-dual mapping. The convergence argument needs only the achieved KKT residual; it must not
assume the polish solver runs.

## 7. Robust scenario extension

For a finite scenario bundle, stack scenario-local residuals with non-anticipativity residuals.
The same theorem applies to the monolithic finite-dimensional CQP if A1–A7 hold uniformly in the
scenario count. Multi-GPU partitioning must be algebraically equivalent to that monolithic
operator. Scaling constants that deteriorate with the number of scenarios must be identified in
the theorem or removed by preconditioning.

## 8. Numerical tests required by the proof

1. Verify unscaled residual reconstruction for every backend.
2. Measure \(\|\tilde d-d^\star\|\) versus achieved KKT residual on small problems.
3. Check model-value perturbation against the L2 bound.
4. Replay fixed and adaptive schedules from identical outer iterates.
5. Record every acceptance decision under progressively tighter inner solves.
6. Demonstrate vanishing virtual control as penalty weights increase.
7. Repeat with scenario bundles to expose any scenario-count dependence.

Until these lemmas and tests are complete, the repository should describe the forcing rule as a
principled, testable policy rather than a proven globally convergent algorithm.
