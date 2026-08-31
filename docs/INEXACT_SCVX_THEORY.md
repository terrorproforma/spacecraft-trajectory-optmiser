# Inexact SCvx: residual, assumptions, lemmas, theorem, and failure modes

## 1. Scope and epistemic status

This note fixes the mathematical contract used by contribution D. It is a **conditional theorem**
for the implemented trust-region successive-convexification architecture. It does not assert that
every spacecraft problem automatically satisfies strong regularity, exact-penalty adequacy, or
continuous-time consistency. Those conditions are stated explicitly and mapped to quantities that
the implementation and experiments must check.

The result covers:

- convex quadratic/conic subproblems with box, affine, SOC, rotated-SOC, exponential, power, and
  supported PSD blocks;
- virtual controls and exact feasibility penalties;
- scaled canonical KKT residuals;
- adaptive, summable, residual-relative, and hybrid inner forcing;
- trust-region acceptance with re-solve-before-shrink;
- finite scenario bundles with non-anticipativity;
- continuous-time violation-state constraints, provided their quadrature/certification error is
  controlled as stated below.

The result does **not** provide global optimality for the nonlinear trajectory problem.

## 2. Nonlinear merit problem

Let the physical nonlinear problem be

\[
\begin{aligned}
\min_{z\in\mathbb R^n}\quad &J(z)\\
\text{s.t.}\quad &c(z)=0,\\
&g(z)\le 0,\\
&z\in\mathcal Z,
\end{aligned}
\]

where \(\mathcal Z\) contains simple physical bounds. Introduce virtual-control and feasibility
slacks \(v\), and define the exact-penalty merit function

\[
\Phi_\rho(z,v)
=
J(z)
+ho_c\|c(z)-v_c\|_1
+ho_g\|[g(z)-v_g]_+\|_1
+ho_v\|v\|_1.
\]

The implementation may use additional small quadratic regularisers; these are included in the
model below. The physical solution requires \(v\to0\).

At accepted reference \(w_j=(z_j,v_j)\), SCvx forms a convex model on
\(s\in X_j:=\{s:\|D_js\|\le\Delta_j,\ w_j+s\in\mathcal B_j\}\):

\[
\begin{aligned}
\min_s\quad &m_j(s)
=
\tfrac12s^TQ_js+q_j^Ts+r_j\\
\text{s.t.}\quad&G_js+h_j\in C_j,\\
&s\in X_j.
\end{aligned}
\tag{P_j}
\]

Here \(C_j\) is a Cartesian product of zero cones, intervals, nonnegative orthants, second-order
cones, rotated second-order cones, and any other declared convex cones. The sparse pattern of
\(Q_j,G_j,C_j,X_j\) is fixed within one trajectory family; numerical values change with \(j\).

Let \(s_j^\star\) be an exact minimiser and \((\widetilde s_j,\widetilde y_j)\) the primal-dual
pair returned by the inner solver.

## 3. Canonical conic KKT mapping

For a closed convex set \(S\), let \(\Pi_S\) denote Euclidean projection and \(N_S\) its normal
cone. Define

\[
F_j(s):=Q_js+q_j.
\]

The KKT generalized equation is

\[
\begin{cases}
0\in F_j(s)+G_j^Ty+N_{X_j}(s),\\
y\in N_{C_j}(G_js+h_j).
\end{cases}
\tag{KKT_j}
\]

For any positive primal and dual step parameters \(\alpha_j,\beta_j\), define the natural residual

\[
\mathcal R_j(s,y)
=
\begin{bmatrix}
\alpha_j^{-1}\left[s-
\Pi_{X_j}\!\left(s-\alpha_j(F_j(s)+G_j^Ty)\right)\right]\\[1mm]
\beta_j^{-1}\left[(G_js+h_j)-
\Pi_{C_j}\!\left(G_js+h_j+\beta_j y\right)\right]
\end{bmatrix}.
\tag{1}
\]

For the closed convex subproblem, \(\mathcal R_j(s,y)=0\) if and only if \((s,y)\) satisfies
\((KKT_j)\). This residual handles equality, interval, box, and conic complementarity through the
same projection identity; backend status strings are not part of the mathematical definition.

Let \(S_j\) be the block-diagonal variable/row/cone scaling actually used for reporting solver
accuracy. The independently recomputed canonical inner residual is

\[
\varepsilon_j
:=\|S_j\mathcal R_j(\widetilde s_j,\widetilde y_j)\|_2.
\tag{2}
\]

The run manifest must record both (2) and the backend-native residual. A solver is not considered
qualified merely because its status is `optimal`.

## 4. Assumptions

### A1 — compact accepted level set

All accepted iterates lie in a compact set \(\mathcal L\), and
\(0<\Delta_j\le\Delta_{\max}<\infty\). The accepted merit sequence is bounded below.

### A2 — smoothness

\(J,c,g\) are \(C^{1,1}\) on an open neighbourhood of \(\mathcal L\): their first derivatives
are locally Lipschitz. Dynamics flow maps and path functions used in the transcription satisfy the
same property on the certified physical domain.

### A3 — first-order and second-order model consistency

At \(s=0\), the convex model matches the merit value and first derivative. Uniformly for
\(s\in X_j\),

\[
|\Phi_\rho(w_j+s)-m_j(s)|
\le
\kappa_m\|s\|^2+\delta_j^{\mathrm{ct}},
\tag{3}
\]

where \(\delta_j^{\mathrm{ct}}\) is the continuous-time discretisation/certification error.

### A4 — convex feasibility and bounded data

Virtual controls/slacks make every declared \((P_j)\) feasible. Numerical CQP data and cone
parameters remain bounded on \(\mathcal L\), and the trust/physical boxes are nonempty.

### A5 — constraint qualification and KKT existence

Every \((P_j)\) satisfies a Robinson-type constraint qualification at its solution, or an
equivalent product-cone qualification sufficient for existence of bounded multipliers. For purely
polyhedral blocks, the corresponding linear constraint qualification is sufficient.

### A6 — uniform local error bound

There are neighbourhoods \(U_j\) of the KKT solution sets and a uniform \(\kappa_e<\infty\) such
that

\[
\operatorname{dist}((s,y),\mathcal S_j)
\le
\kappa_e\|\mathcal R_j(s,y)\|_2,
\qquad(s,y)\in U_j.
\tag{4}
\]

Sufficient conditions include strong convexity plus a suitable constraint qualification, or strong
metric subregularity of the KKT mapping. This assumption is not automatic for degenerate conic
programmes.

### A7 — uniformly equivalent scaling

There exist constants \(0<\underline\sigma\le\overline\sigma<\infty\) such that

\[
\underline\sigma\|r\|_2
\le\|S_jr\|_2
\le\overline\sigma\|r\|_2
\tag{5}
\]

for every canonical residual vector. Scaling may be reused or refreshed, but may not collapse or
explode silently.

### A8 — exact-model sufficient decrease

Whenever the nonlinear first-order stationarity measure \(\chi(w_j)\) exceeds a fixed positive
number and \(\Delta_j\) is sufficiently small, the exact convex step has Cauchy-type predicted
reduction

\[
\operatorname{pred}_j^\star
:=m_j(0)-m_j(s_j^\star)
\ge
\kappa_c\min\{\chi(w_j)\Delta_j,\Delta_j^2\}.
\tag{6}
\]

### A9 — exact-penalty adequacy

Eventually \(\rho_c,\rho_g,\rho_v\) exceed the relevant local multiplier bounds. Consequently, a
stationary point of \(\Phi_\rho\) with zero virtual control is stationary for the physical
constraints.

### A10 — trust-region safeguards

Rejected steps do not move the reference. The radius is expanded only after accepted steps with
strong agreement and near-boundary step length. If a rejected step has canonical inner residual
materially above its requested tolerance, the same CQP is re-solved before the radius is shrunk.

### A11 — inner forcing

At least one of the following holds:

1. **Summable residuals**
   \[
   \sum_{j=0}^{\infty}\varepsilon_j<\infty.
   \tag{7a}
   \]

2. **Residual-relative forcing**
   \[
   \varepsilon_j
   \le
   \eta_j\min\{1,\|\widetilde s_j\|,\chi(w_j)\},
   \qquad\eta_j\to0.
   \tag{7b}
   \]

3. **Hybrid forcing:** (7a) outside a local neighbourhood and (7b) inside it.

The achieved canonical residual, not only the requested tolerance, must satisfy the rule.

### A12 — continuous-time consistency

The nonlinear dense replay and path checker returns an error bound or estimator
\(\delta_j^{\mathrm{ct}}\) satisfying either

\[
\sum_j\delta_j^{\mathrm{ct}}<\infty
\quad\text{or}\quad
\delta_j^{\mathrm{ct}}=o(\min\{\|\widetilde s_j\|,\chi(w_j)\}).
\tag{8}
\]

For fixed-grid experiments, this is an assumption to be checked by refinement. Violation-state
quadrature does not by itself prove that an unsampled nonlinear path constraint is satisfied.

### A13 — robust scenario regularity

For finite scenario problems, all positive scenario probabilities are bounded away from zero on a
fixed experiment family; non-anticipativity rows use a fixed information history; and the enlarged
scenario CQP satisfies A4–A7. Risk epigraphs are included in the canonical residual.

## 5. Lemmas

### Lemma 1 — residual equivalence

Under A7,

\[
\|\mathcal R_j\|_2
\le\underline\sigma^{-1}\varepsilon_j,
\qquad
\varepsilon_j
\le\overline\sigma\|\mathcal R_j\|_2.
\]

Thus convergence of the reported canonical residual is equivalent to convergence of the unscaled
natural residual.

### Lemma 2 — residual-to-solution distance

Under A5–A7, once the returned point enters \(U_j\), there is a uniform
\(\bar\kappa_e=\kappa_e/\underline\sigma\) such that

\[
\operatorname{dist}((\widetilde s_j,\widetilde y_j),\mathcal S_j)
\le\bar\kappa_e\varepsilon_j.
\tag{9}
\]

Choose the closest exact primal solution \(s_j^\star\); then
\(\|\widetilde s_j-s_j^\star\|\le\bar\kappa_e\varepsilon_j\).

### Lemma 3 — predicted-reduction perturbation

Because the CQP gradient is bounded on the compact trust box, (9) implies

\[
|m_j(\widetilde s_j)-m_j(s_j^\star)|
\le\kappa_p\varepsilon_j.
\tag{10}
\]

Hence

\[
|\widetilde{\operatorname{pred}}_j-\operatorname{pred}_j^\star|
\le\kappa_p\varepsilon_j.
\tag{11}
\]

### Lemma 4 — actual/predicted agreement perturbation

Under A2–A3,

\[
|\operatorname{ared}_j-\widetilde{\operatorname{pred}}_j|
\le
\kappa_m\|\widetilde s_j\|^2
+\kappa_p\varepsilon_j
+\delta_j^{\mathrm{ct}}.
\tag{12}
\]

### Lemma 5 — accepted inexact decrease

For the fixed acceptance threshold \(\gamma\in(0,1)\), every accepted step satisfies

\[
\Phi_\rho(w_j)-\Phi_\rho(w_{j+1})
\ge
\gamma\operatorname{pred}_j^\star
-\kappa_a\varepsilon_j
-\kappa_{ct}\delta_j^{\mathrm{ct}}.
\tag{13}
\]

### Lemma 6 — vanishing exact predicted reduction

Under A1, A10–A12, summing (13) over accepted steps shows

\[
\liminf_{j\in\mathcal A}\operatorname{pred}_j^\star=0,
\tag{14}
\]

where \(\mathcal A\) is the accepted-iteration set. With relative forcing, the perturbations in
(12) are lower order than the stationarity/Cauchy decrease in (6), giving the same conclusion
locally.

### Lemma 7 — exclusion of nonstationary accumulation points

Suppose an accumulation point \(\bar w\) has \(\chi(\bar w)>0\). By continuity, A8 gives a
uniform positive exact predicted reduction for sufficiently small but nonzero trust radius near
\(\bar w\). A11–A12 make the inner/model perturbations a strict lower-order fraction of this
reduction. A10 therefore yields either an accepted uniform decrease or a radius contraction that
eventually restores model agreement. Both alternatives contradict (14) and boundedness below.
Therefore every accumulation point has \(\chi(\bar w)=0\).

### Lemma 8 — transfer to the physical problem

Under A9, if \(v_j\to0\), stationarity of an accumulation point for \(\Phi_\rho\) implies
first-order stationarity for the original nonlinear constraints.

## 6. Main theorem

### Theorem — subsequential first-order convergence of inexact SCvx

Assume A1–A13, infinitely many accepted iterates, and that the returned inner points eventually
enter the local error-bound neighbourhoods in A6. Then:

1. the exact/inexact predicted-reduction difference converges to zero;
2. accepted merit decrease satisfies (13), with a summable or lower-order perturbation;
3. every accumulation point of accepted iterates is first-order stationary for the exact-penalty
   merit problem;
4. if virtual controls vanish and A9 holds, every accumulation point is first-order stationary
   for the physical nonlinear trajectory problem;
5. for a fixed finite scenario tree, the same statements hold for the non-anticipative robust
   problem, including the declared convex risk epigraph.

The theorem is conditional on the error bound, penalty adequacy, and CT consistency. Experiments
must report evidence for those conditions; they may not be hidden in a generic solver status.

## 7. Local corollary

If the physical solution is isolated, the nonlinear KKT mapping is strongly regular, the accepted
iterates enter its neighbourhood, and the forcing satisfies
\(\varepsilon_j=O(\chi(w_j)^{1+\tau})\) for some \(\tau>0\), then the inexact perturbation is
superlinear relative to the first-order stationarity measure. The local rate is therefore governed
by the SCvx model/trust mechanism rather than an inner residual floor. This corollary does not
claim superlinear convergence of the complete algorithm without additional second-order model
conditions.

## 8. Implemented forcing rule and proof mapping

The native `AdaptiveForcingRule` computes

\[
\varepsilon_j^{\rm req}
=
\operatorname{clip}
\left(\eta\,R_j^{p},\varepsilon_{\min},\varepsilon_{\max}\right),
\qquad p>1,
\]

then imposes phase ceilings for repair, progress, refinement, and polish. This is an empirical
relative-forcing rule once the floor is inactive. Therefore:

- the floor **must not remain active at a purported asymptotic solution** unless a final direct
  polish certifies a smaller canonical residual;
- the achieved residual is appended to `InexactErrorLedger`;
- `maximum_relative_forcing()` estimates the constant in (7b);
- a rejected under-solved CQP is re-solved before `TrustRegionController` shrinks the radius;
- the final nonlinear replay and CT checker provide the quantities entering A3 and A12.

A publication run must flag any iteration for which the achieved residual exceeds the requested
forcing threshold.

## 9. Executable counterexamples and adversarial cases

These are not rhetorical caveats; `tests/test_inexact_theory_counterexamples.py` evaluates them.

### C1 — nonvanishing inner error floor

For \(f(x)=\tfrac12x^2\), the exact model step is \(s^\star=-x\). If the inner solver always
returns \(\widetilde s=-x+\epsilon\), then every outer update gives \(x^+=\epsilon\). A fixed
\(\epsilon>0\) prevents stationarity. This is why the polish floor must eventually vanish or be
superseded by a certified final solve.

### C2 — residual without a linear error bound

For \(f(x)=\tfrac14x^4\), the stationarity residual is \(|x|^3\), while distance to the solution
is \(|x|\). No uniform \(\kappa\) satisfies \(|x|\le\kappa|x|^3\) near zero. A6 cannot be
inferred merely from smooth convexity.

### C3 — inadequate exact penalty

Minimise \(-x\) subject to \(x=0\). The \(L_1\) merit is
\(-x+\rho|x|\). For \(\rho<1\), positive movement decreases the merit, so merit stationarity
cannot recover the physical constraint. Penalty adequacy is a mathematical condition, not a tuning
nicety.

### C4 — collapsing residual scaling

Let the unscaled residual be identically one and set \(S_j=1/(j+1)\). The reported scaled residual
converges to zero although the KKT error does not. This violates A7 and motivates recording scaling
extrema.

### C5 — node-only path feasibility is false

On \([0,1]\),

\[
g(t)=4t(1-t)-\tfrac12.
\]

Both endpoints satisfy \(g=-1/2\), while \(g(1/2)=1/2>0\). Knot feasibility alone cannot replace
dense replay, quadrature refinement, or a proven CT bound.

### C6 — shrinking before resolving can create false stagnation

For \(f(x)=\tfrac12(x-1)^2\) at \(x=0\), an under-solved inner problem may return \(s=0\) with a
large KKT residual. Shrinking the trust region interprets solver error as model failure. Re-solving
the same CQP first separates these causes.

### C7 — omitted non-anticipativity solves the wrong problem

Two equally likely scenarios prefer controls \(+1\) and \(-1\). Independent scenario controls
achieve zero expected quadratic loss, while a common pre-observation control has optimum zero and
positive loss. Removing non-anticipativity gives an unattainable policy and invalid robust cost.

## 10. Required experiment fields

Every D experiment must record:

- outer iteration and phase;
- requested and independently achieved canonical residual;
- backend-native primal, dual, complementarity, and gap residuals;
- minimum/maximum primal and row scaling;
- inner matrix-vector products, cone projections, and iteration count;
- exact or high-accuracy reference objective where available;
- predicted and actual merit reduction;
- acceptance ratio, step fraction, and trust radius;
- virtual-control norm and exact-penalty weights;
- nonlinear dynamics, path, terminal, non-anticipativity, and risk-epigraph residuals;
- CT quadrature estimate and dense-replay/refinement discrepancy;
- setup, update, transfer, solve, residual, replay, and total time;
- whether a re-solve or interior-point polish occurred;
- accumulated inner-error sum and empirical relative-forcing ratio.

## 11. Closure checklist for Paper 1

The theorem section is ready for publication only when the paper:

1. states the exact merit and CQP used in each experiment;
2. gives the natural residual (1), scaling convention (2), and all tolerances;
3. identifies which problem families satisfy strong convexity versus only an empirical error bound;
4. reports multiplier/penalty and virtual-control evidence for A9;
5. reports mesh/refinement evidence for A12;
6. shows adversarial tests C1–C7 remain caught by the implementation;
7. separates the conditional theorem from empirically observed convergence;
8. avoids claiming convergence when the inner floor or CT error remains dominant.
