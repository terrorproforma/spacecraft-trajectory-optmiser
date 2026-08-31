# Inexact SCvx convergence conditions

## Purpose and status

This note fixes the mathematical target for contribution D. It gives a conditional convergence
argument for solving the convex subproblems only approximately. It is not yet a substitute for
a complete paper proof covering every implementation detail, but it is strong enough to define
which residuals must be measured and which forcing rules are admissible.

## Nonlinear problem

Consider

\[
\min_z\; J(z)
\quad\text{subject to}\quad
c(z)=0,\qquad g(z)\le 0,
\]

with a compact level set containing all accepted iterates. At outer iteration \(j\), SCvx builds
a convex model over a trust region \(\|s\|\le \Delta_j\):

\[
\min_s\; m_j(s)
\quad\text{subject to}\quad
\widehat c_j(s)=0,
\qquad \widehat g_j(s)\le 0,
\qquad \|s\|\le \Delta_j.
\]

Virtual control or exact-penalty variables make the convex subproblem feasible. Let
\(s_j^\star\) denote an exact solution and \(\widetilde s_j\) the step returned by the inner
solver.

## Inner residual

The backend-independent inner error is the norm of a scaled KKT residual:

\[
\varepsilon_j
=
\left\|
\begin{bmatrix}
\nabla_s L_j(\widetilde s_j,\widetilde\lambda_j)\\
\widehat c_j(\widetilde s_j)\\
[\widehat g_j(\widetilde s_j)]_+\\
\operatorname{comp}_j
\end{bmatrix}
\right\|_{W_j}.
\]

`W_j` must use the same variable and row scaling used to interpret the solver tolerance. Raw
backend status strings are insufficient. Paper experiments must record both the backend's
reported residual and an independently recomputed canonical residual.

## Assumptions

The intended theorem uses the following assumptions.

**A1 — smooth local model.** `J`, `c`, and `g` are continuously differentiable on an open set
containing the accepted level set, with locally Lipschitz first derivatives.

**A2 — first-order consistency.** At the reference point \(z_j\), the convex model agrees with
the nonlinear problem to first order. Model error over the trust region is bounded by

\[
|\Phi(z_j+s)-\widehat\Phi_j(s)|
\le \kappa_m\|s\|^2,
\]

for the exact-penalty merit function \(\Phi\).

**A3 — bounded scaling.** State, control, row, and cone scalings remain uniformly bounded above
and away from zero on accepted iterates. Residual norms are therefore uniformly equivalent.

**A4 — well-posed convex subproblems.** After the declared regularisation, each convex
subproblem has a unique primal solution and satisfies a uniform local error bound

\[
\|\widetilde s_j-s_j^\star\|
\le \kappa_e\varepsilon_j.
\]

Uniform strong convexity is sufficient but not necessary. A metric-subregular KKT mapping can
replace it.

**A5 — trust-region safeguards.** The radius is bounded, rejected steps do not move the
reference, and the standard ratio test uses predicted and actual merit reduction. The radius is
not expanded after poor agreement.

**A6 — exact-penalty adequacy.** The virtual-control and feasibility penalties eventually exceed
the relevant multiplier bounds, so stationary points of the merit formulation are stationary
for the original constrained problem when virtual control vanishes.

**A7 — inner forcing.** One of the following holds:

1. **summable errors**

   \[
   \sum_{j=0}^{\infty}\varepsilon_j < \infty;
   \]

2. **relative forcing**

   \[
   \varepsilon_j
   \le \eta_j
   \min\{1,\|\widetilde s_j\|,R_j\},
   \qquad \eta_j\to 0,
   \]

   where \(R_j\) is a nonlinear feasibility/stationarity residual;

3. a hybrid rule that is summable outside the local regime and satisfies relative forcing near a
   limit point.

## Conditional proposition

Under A1–A7, suppose infinitely many steps are accepted and the accepted merit sequence is
bounded below. Then:

1. the difference between exact and inexact predicted reduction vanishes;
2. accepted inexact steps retain the trust-region sufficient-decrease property up to a summable
   perturbation;
3. every accumulation point of accepted iterates is first-order stationary for the exact-penalty
   problem;
4. if virtual control converges to zero and A6 holds, every such accumulation point is
   first-order stationary for the original constrained problem.

## Proof skeleton

From A4,

\[
\|\widetilde s_j-s_j^\star\|
\le \kappa_e\varepsilon_j.
\]

Lipschitz continuity of the convex model on the bounded trust region gives

\[
|m_j(\widetilde s_j)-m_j(s_j^\star)|
\le \kappa_p\varepsilon_j.
\]

Thus the inexact predicted reduction differs from the exact one by at most
\(\kappa_p\varepsilon_j\). Model consistency A2 gives an additional
\(O(\|\widetilde s_j\|^2)\) difference between predicted and actual merit change.

For accepted steps, the ratio test therefore yields

\[
\Phi(z_j)-\Phi(z_{j+1})
\ge
\gamma\,\operatorname{pred}_j^\star
-\kappa\varepsilon_j
-O(\|\widetilde s_j\|^2),
\]

for a fixed acceptance constant \(\gamma>0\). Under summable errors, the accumulated negative
perturbation is finite. Since \(\Phi\) is bounded below, the exact predicted reductions cannot
remain bounded away from zero. Under relative forcing, the perturbation becomes lower order
than the step or stationarity measure and the same conclusion follows locally.

Assume an accumulation point is nonstationary. Standard trust-region model adequacy then gives a
uniform descent step and a uniform positive exact predicted reduction in a neighbourhood of that
point. The vanishing inner perturbation preserves a positive fraction of this reduction, which
contradicts convergence of predicted reductions to zero. Hence accumulation points are
stationary for the exact-penalty problem. Exact-penalty adequacy and vanishing virtual control
then transfer stationarity to the original constraints.

## Implementable forcing rules

### Summable schedule

A simple globally safe schedule is

\[
\varepsilon_j
\le
\frac{\varepsilon_0}{(j+1)^{1+\delta}},
\qquad \delta>0.
\]

This is theoretically clean but may oversolve early models.

### Residual-relative schedule

The preferred practical rule is

\[
\varepsilon_j
\le
\min\left\{
\varepsilon_{\max},
\eta_j\max(R_j,R_{\min}),
\eta_j\max(\|s_{j-1}\|,s_{\min})
\right\},
\]

with \(\eta_j\) decreasing only after accepted steps. Repair iterations retain a loose ceiling;
refinement and polish tighten it.

### Agreement-aware correction

If a step is rejected and

\[
\max(r_{\mathrm p},r_{\mathrm d})
> \chi\varepsilon_j,
\]

re-solve the *same* convex model at a tighter tolerance before shrinking the trust region. This
separates model failure from an under-solved subproblem. Only one or two such re-solves should be
allowed per outer iteration.

## Hybrid final polish

A first-order PDHCG solve is sufficient while model error dominates. An interior-point polish is
allowed when:

- the step is accepted;
- the nonlinear residual is below the polish threshold;
- the trust-region step is small;
- the active model is not expected to change materially;
- a compatible high-accuracy backend is available.

The final solver receives both primal and dual starts when supported. The polished point is still
accepted only after nonlinear propagation and independent constraint checks.

## Required experiment columns

Every D experiment must record:

- outer iteration and phase;
- requested and independently achieved inner residual;
- primal and dual residual components;
- inner iteration count;
- exact or high-accuracy reference objective where available;
- predicted and actual merit reduction;
- acceptance ratio and trust radius;
- virtual-control norm;
- nonlinear dynamics, path, and terminal residuals;
- setup, update, solve, residual, and total time;
- whether a re-solve or interior-point polish occurred;
- accumulated inner-error ledger and empirical relative-forcing ratio.

## Remaining theory work

Before publication, this note must be specialised to the exact continuous-time SCvx model,
including state-triggered constraints if used. The final proof must state the KKT mapping and
constraint qualification precisely, show the required error bound for the selected conic
subproblem regularisation, and connect PDHCG's reported residual to the canonical residual used
above.
