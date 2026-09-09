# Fixed-primal dual correction after PDHCG — v625

[Frozen inputs, returned vectors and independent audits](../results/local/2026-09-09/native-route-and-dual-polish-v625/README.md).

The [v623 warm-transfer experiment](PDHCG_WARM_START_AND_PROJECTION.md) left one
near-converged exact-L1 capture failing only its global objective-gap gate after
10,000 GPU updates. Its normalized dual residual passed, but the remaining
stationarity error still produced a significant objective-gap error. Increasing
the update budget without addressing that behavior was not the next experiment.

This diagnostic keeps the original problem and every primal variable, stored
slack, and retained scalar/SOC dual multiplier unchanged. It adjusts only the
equality multipliers and the 525 L1 subgradients whose virtual variables,
epigraphs and pair slacks are **exactly zero**. Those exact-zero conditions and
the pair structure are checked from the saved original problem. No approximate
cone faces are released.

The auxiliary convex problem minimizes the full original stationarity 2-norm.
It constrains every stationarity coordinate using a fixed denominator based on
the original objective coefficients, preserves each L1 subgradient box, and
requires the original signed absolute gap to be at most 1e-9. With the primal
fixed and the retained cone multipliers fixed, the gap change is `b^T * dy`.
The small existing equality defect is retained; substituting `A*x` for `b` would
change the correction problem.

The implementation uses one CPU Clarabel call on a sparse auxiliary QP, with
2,953 variables and 6,710 constraint rows. Equivalent coefficient-based units
avoid an unnecessarily large Hessian scale. The returned correction is mapped
back into the original multiplier arrays without clamping or further correction.
The unchanged long-double and Decimal65 auditors then recompute every original
numerical acceptance gate from those actual rounded arrays.

| Measurement | Original PDHCG output | After auxiliary correction |
| --- | ---: | ---: |
| Objective gap | 7.453170826e-5 | 9.999999064e-10 |
| Full unscaled stationarity 2-norm squared | 2.518908759e-10 | 5.149771459e-11 |
| Full unscaled stationarity infinity norm | 5.052243363e-6 | 5.068815536e-6 |
| Normalized primal residual | 2.471174842e-10 | unchanged |
| Original primal objective | 0.0218755838757 | unchanged |
| All original numerical KKT gates | fail | pass |
| Native PDHCG termination | iteration limit | unchanged |

The objective is the normalized convex-subproblem objective, not returned kg or
the GTOC12 mission score.

The infinity norm increases slightly; it remains below the fixed conservative
bound of 1.0001e-5. The full 2-norm improves, both subgradient-box checks pass,
and the fixed arrays remain bit-identical. This is more than cancelling a scalar
gap while ignoring stationarity. However, the final gap is only **9.36e-17 below
the 1e-9 limit**. One FP64 result with this margin is not evidence of robust
qualification on other hardware or problems.

The auxiliary solve terminates in 13 iterations. Its measured call takes
11.618 ms, with 4.496 ms of solver setup. The complete CPU worker takes 92.985 ms;
the supervisor takes 316.436 ms. These are separate scopes and must not be added
as independent costs. The original GPU PDHCG solve took 362.248 ms in an earlier
run. This saved-point experiment is not an end-to-end hybrid timing benchmark.

The original PDHCG iteration-limit status is preserved. Both original auditors
therefore report numerical gates passing and native qualification **false**.
The successful result is explicitly a hybrid candidate. It establishes neither
native PDHCG convergence nor nonlinear trajectory feasibility or a fleet gain.

The next numerical experiment is a GPU-resident version of this restricted
correction with adequate numerical margin, followed by the other fixed captures
and a direct comparison against the best qualified GPU baseline. All conversion,
correction, failure and certification time must be counted. If that path fails
to improve complete time to verified accuracy, this one diagnostic is not a
reason to retain it in production. The PDHCG core remains the method under test.

## v626 isolated GPU diagnostic

The next diagnostic implements the equality-constrained least-squares correction
with one GPU QR factorization, one application of its orthogonal factor and two
triangular solves. It targets zero original gap, keeps every primal/slack and
retained dual value fixed, and retains all original KKT thresholds. Ten small
GPU cases produce the intended two numerical passes and eight rejections.

One correction of the same saved original PDHCG iterate passes both independent
long-double and 65-digit original-coordinate audits. Its exact signed gap is
approximately -5.112e-17; the unscaled stationarity norm improves from 1.587e-5
to 7.176e-6. However, the diagnostic **rejects** the candidate because its
separate numerical-quality rule fails. The inherited native termination also
remains `ITERATION_LIMIT`; there is no production qualification or integration.

The quality rejection is understood from saved arithmetic: four isolated
zero-right-hand-side columns receive rounding corrections around 1e-23. The
componentwise quality ratio divides each correction by its own magnitude and
therefore reports one. The largest absolute normal-equation residual across all
columns is about 1.37e-19. That explains the result but does not retrospectively
change the frozen rule or promote the candidate. Exact elimination of those
isolated coordinates or a justified backward-error rule needs a separate test.

Evaluation takes 195.520 ms, with 218.238 ms of workspace creation measured
separately; the complete launched process takes 569.874 ms. This is not a
speedup result. Before another numerical policy is tried, a bounded phase
profile should identify where that time goes. The diagnostic remains separate
from the mission's GPU QOCO backend and earns no fleet score.
