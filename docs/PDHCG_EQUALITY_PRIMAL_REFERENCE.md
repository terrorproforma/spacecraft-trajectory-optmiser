# Equality-feasible primal reference on two frozen captures

The CPU reference removes the original equality-residual obstruction seen in [the retained cold-start comparison](GPU_CORE_RETAINED_COLD_STARTS.md), but **neither of its two results qualifies**. After 10,000 updates per input, original gap, stationarity, block complementarity and primal cone feasibility still fail. This is a mathematical reference on two changed stages of one synthetic fixture, not a production implementation, independent mission sample or GPU throughput result.

The [portable evidence](../results/local/2026-09-09/core-primal-reference-v631/README.md) retains the complete source, exact inputs, rational tests, coefficient checks, all eight original-coordinate checkpoints and unchanged long-double/Decimal65 audits. No native solver or GPU ran.

## Exact objective and equality projection

Both input snapshots contain 453 stored quadratic entries, all numerical zeros. The implementation rejects any nonzero quadratic coefficient and shifted coordinate view. It detects and proves the 525 disjoint original epigraph pairs

    v - t <= 0,  -v - t <= 0,  c_t = lambda = 10000.

The reduced primal retains every state, control and virtual-control variable v. It removes only t and its pair rows, replacing their contribution by lambda*|v|. All remaining scalar inequalities and the original 75 radius-first SOCs are retained.

Let E be the original equality matrix restricted to the retained primal u, C the retained conic operator, and R the virtual-coordinate selector. The fixed dual operator is K=[C;diag(lambda)R]. A normalized dual qhat is clipped to [-1,1], and the original L1 dual is q=lambda*qhat. Therefore its effective original-q step is lambda²*Sigma_qhat.

Each iteration performs the retained scalar/SOC dual projection and normalized L1 dual update, followed by

    w = u - T*(c_ret + C^T*z_new + R^T*q_new)
    mu = (E*T*E^T)^-1*(E*w-b)
    u_new = w - T*E^T*mu
    u_bar = 2*u_new-u.

The original equality dual is y=mu after reversing the equality-row permutation; no extra step-size division is applied. Original readout uses t=|v|, pair multipliers (lambda±q)/2, retained conic duals unchanged, and s=h-Gx. The original L1 pair complementarity is lambda*|v|-q*v. This remains an optimization error during the iteration and is checked in the full original KKT audit.

The initial primal is a **charged projection of zero onto E*u=b** using the same metric and factor. Its extrapolate equals that projected primal; the iterated equality, retained conic and normalized L1 duals start at zero. No predecessor or qualified point is imported. The initial virtual penalty is about 20,548 on each input and is included in the saved initial audit.

## Coefficient checks and numerical scope

Primal and dual diagonals use inward-rounded theta/absolute-coefficient sums with stored theta=0.95, and each SOC shares a conservative common dual step. Exact Fraction checks on the represented coefficients and steps establish the sufficient squared operator-norm bound below 0.9025. The equality operator belongs to the primal proximal constraint, not the dual step-size operator.

The 525 dynamics rows have distinct virtual columns with coefficient -1, while the remaining 17 equality rows constrain distinct boundary/inactive-control coordinates. This proves full row rank directly from these coefficients. The factor has half-bandwidth 19.

Symmetric equality row equilibration uses S=diag(1/sqrt(diag(E*T*E^T))). It changes neither the mathematical projection nor the optimization metric. The estimated condition number falls only about **1.9%**, from 21,245 to 20,850, on either capture. No substantial conditioning improvement is claimed. Virtual primal steps are 9.5e-5; state steps are approximately 0.317–0.475 and control steps 0.19–0.475.

The tiny CPU tests check independent rational projection/multiplier identities, row-scaling equivalence, adjoints, original L1 objective and dual units, four SOC cases and seven malformed/rank rejection cases. The actual-input preflight checks zero Q, pair structure, equality rank, adjoints and the exact metric certificate without projecting or iterating a captured primal.

## Observed original-coordinate results

Exactly one CPU worker completed two fixed cold runs: **20,000 updates and 20,002 projections**, including the two initial projections. Both stopped at their 10,000-update cap. Original long-double and Decimal65 verdicts agree on all eight saved checkpoints.

Original limits are 1e-9 for relative primal/dual/gap/maximum-block measures and 1e-8 for both absolute cone violations. Relative primal below reports the equality and conic-equation residual; it does not include cone membership.

| Final measure | Early | Near-converged |
|---|---:|---:|
| Relative original primal | 2.07644e-16 | 2.03279e-16 |
| Relative original stationarity | 2.44310e-8 | 2.46566e-8 |
| Original objective gap | 0.0914556251 | 0.0815865927 |
| Maximum normalized block complementarity | 9.15807e-4 | 8.36785e-4 |
| Primal cone violation | 7.02638e-7 | 6.49943e-7 |
| Original virtual L1 cost | 0.09395119 | 0.08415928 |
| Full original numerical qualification | Fail | Fail |

The previous dominant equality row 526, initial position-y, now has absolute residual 3.31e-18 / 1.01e-17. The largest equality defect anywhere is below 8.57e-16. Separate unscaled linear-solve and proximal-identity residuals also remain near FP64 roundoff.

The remaining original failures are specific:

- All 525 virtual coordinates remain nonzero. Their pair complementarity sums are 0.09395721 / 0.08416472 and dominate the signed gaps. The full gap identity subtracts aggregate stationarity contributions of approximately 0.00250164 / 0.00257817; retained conic complementarity sums are only about 5.02e-8 / 4.52e-8.
- The largest virtual is original coordinate 1357, interval 74 velocity-x, about 9.16e-8 / 8.37e-8. Its actual pair dual is about -1.00 while lambda is 10,000. No small virtual value is rounded to zero and no sign-based replacement is applied.
- Worst stationarity is original variable 535, the first interval's thrust-magnitude variable Gamma: absolute residual 2.44334e-4 / 2.46591e-4.
- Worst primal SOC is node 56, original G rows 3248–3251, coupling Gamma at variable 759 to thrust variables 756–758. The corresponding scalar Gamma slack at original row 1456 is slightly negative, about -5.55e-9 / -5.00e-9; it is the SOC defect that exceeds the absolute cone threshold.

## Cost and next implication

Complete worker process time was 4.27961 seconds, covering both inputs, Python imports, setup, initial projections, updates, gates and evidence. Per-case complete times were 2.07837 / 1.97712 seconds; update-loop subsets were 1.79283 / 1.67470 seconds. A separate saved-only original audit is outside that worker time. These are single CPU-reference observations, not a matched speed comparison or GPU performance claim.

Equality feasibility is therefore a resolved algebraic subproblem for these two inputs, not a qualified optimizer. The remaining target is simultaneous L1 complementarity and cone/stationarity convergence. A concrete next mathematical test would be a joint equality-and-L1 primal proximal operation with a verifiable finite inner residual and charged work. Simply thresholding the projected primal would generally break its equalities and is not equivalent. That joint operation has not been implemented or tested here. An unchanged cap rerun or broad weight grid is not supported by this result.
