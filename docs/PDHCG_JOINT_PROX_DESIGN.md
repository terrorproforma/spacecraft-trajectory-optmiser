# Joint equality and L1 primal proximal design

The [v631 CPU reference](PDHCG_EQUALITY_PRIMAL_REFERENCE.md) made the original equalities accurate to roundoff on two frozen stages of one synthetic trajectory fixture. Both optimization results still failed the original KKT gate. Their largest gap contribution was virtual-control L1 complementarity: all 525 virtual coordinates remained nonzero while their pair duals were far from the original penalty endpoints.

The proposed next reference puts equality feasibility and the original L1 penalty in the same primal proximal operation. It retains every physical and virtual-control variable, every remaining scalar/SOC constraint, and the original penalty lambda. It applies only to the two exact zero-Q inputs; a nonzero quadratic objective is rejected.

For retained primal u, equality E*u=b and conic operator C, the outer step is

    z_new = project_cones(z + Sigma_C*(C*u_bar-h))
    w = u - T*(c_ret + C^T*z_new)
    u_new = argmin_{E*a=b} .5*||a-w||^2_{T^-1} + sum(lambda*|a_virtual|).

There is no separate L1 dual update or additional old-q term in w. The exact v631 T and Sigma_C are retained. Removing the lambda selector rows from the dual operator preserves its sufficient coefficient-based squared norm bound below 0.9025 for an exact proximal operation; it does not justify increasing any step.

Given an equality multiplier mu, the inner candidate is a diagonal soft threshold:

    a = w-T*E^T*mu
    u_physical = a_physical
    u_virtual = soft(a_virtual, T_virtual*lambda)
    q = clip(a_virtual/T_virtual, -lambda, lambda).

The remaining inner equation is E*u(mu)=b. Its active Hessian is the physical Gram plus diagonal contributions from the active virtual controls. The captured dynamics have distinct virtual columns with coefficient -1, so the active contributions are diagonal after the same trajectory ordering. The full Gram has half-bandwidth 19. A singular or numerically unusable active factor triggers a descent direction using the full Gram E*T*E^T; this is not a ridge or an objective change. Factors and all fallback work must be charged.

Original-coordinate readout uses y=mu, t=|v|, pair multipliers (lambda+q)/2 and (lambda-q)/2, and s=h-G*x. It does not round tiny v values to zero. At an exact inner solution, q is a valid original L1 subgradient; original optimization stationarity still depends on convergence of the outer iteration.

The [published v633 evidence package](../results/local/2026-09-09/core-joint-reference-v633/README.md) contains the sealed v632 derivation, nine independent Fraction fixtures, exact witness-gap bounds, FP64 tests and a deliberately unresolved 32-direction limit case. In its `evidence.zip`, the files are under `build/performance/core-joint-prox-v632/`: `DESIGN.md`, `RESULTS.md`, the tiny sources/results and the eight-file `index.json`. That index's SHA-256 is `27bbd087aedd788f920b434484c56b2a7feec9ce648a45ba038dcfe02bffb854`.

The fixed experimental FP64 stop is 1e-11 for normalized equality, proximal stationarity, L1 Fenchel defect and equality-dual gap debt, with an exact represented q-box check. Each call permits at most 32 directions and 12 Newton halvings per direction. This practical stop is **not** a summably-inexact PDHG convergence guarantee. The separate witness theorem requires exact feasibility or certified roundoff enclosures; the tiny exact bounds are not captured-problem certificates.

The v632 evidence contains no captured proximal calls, optimization runs or GPU work. The subsequent [v633 CPU reference](../results/local/2026-09-09/core-joint-reference-v633/README.md) ran once on the two saved inputs. Both charged initial joint proximal calls passed, but the early case failed its first outer attempt and the near-converged case failed after 73 committed updates. All full/active factors passed; neither failure was an iteration cap. Both stopped at the fallback merit-decrease check after Newton backtracking was exhausted.

All six saved original points fail the unchanged long-double and 65-digit KKT gates. Their virtual L1 cost and pair complementarity are zero, but original cone, stationarity and gap errors remain. Complete worker process time was 0.71946 seconds for this short failed run, including setup and all failed work; it is not a speed improvement over a completed longer run.

Saved-state arithmetic identifies the early blocking inner metric as normalized equality-dual gap debt 1.72283e-10, and the near blocking metric as normalized equality residual 4.06063e-10. Both exceed the fixed 1e-11 budget. The line search computes a trial's residual verdict but does not consult it before Armijo. Rejected trial vectors and decrease terms were not saved, so the evidence does not establish that any discarded trial passed.

The v635 successor preserves all steps and gates: it honors the existing residual stop on a trial, and uses the dual quadratic decrease -F^T*delta + .5*delta^T*H_active*delta on an unchanged signed branch. Delta is the actual represented multiplier change. The full Gram supplies an upper bound across branch changes. The quadratic is evaluated through the original sparse transpose operator, avoiding subtraction of large absolute objectives. This is practical FP64 arithmetic rather than an outward-certified bound.

Nine exact proximal cases, 20 active-Gram identities, three outer-step cases and 59 exact-source sparse-merit comparisons pass, including signed crossings, cancellation, rounding and failure-retention tests. The single prescribed two-capture CPU experiment then completes 10,000 outer updates per input, with no inner failures, backtracking, fallback or factor rejection. Of 15,469 accepted inner directions, 15,460 stop on trial KKT and only nine require a merit product. The two original v633 failures remain preserved.

Both final v635 points pass original primal residual, cone and complementarity gates. Dual stationarity remains about 2.07e-8 and gap about 0.001261, exceeding the unchanged 1e-9 thresholds. All eight saved readouts fail both long-double and 65-digit original-coordinate audits. Full worker-process time is 19.764668 seconds for both captures; diagnostic work is included and this is neither a GPU benchmark nor qualified solver throughput. The result isolates an outer convergence tail after fixing the inner numerical obstruction.

The saved 65-digit decomposition attributes essentially all of the remaining gap to x dotted with the outer stationarity residual; Gamma and thrust variables account for about 98.7%. Equality debt and complementarity are negligible. Final proximal stationarity is around 7e-16, so the evidence does not support tightening the inner tolerance. Both final primal points satisfy the existing eligibility gates for a bounded dual-correction handoff, but the subsequent same-input objective comparison changes that proposed next step.

Both final v635 primal objectives are about **3.631% higher** than the saved qualified QOCO outputs on the identical original inputs. Holding the primal point fixed cannot improve that cost. The v636 dual-correction proposal was therefore canceled before conversion, factorization or native execution; no auxiliary conic solve replaced it. A suspected zero-equality-support obstruction was separately falsified: Gamma has a real mass-equality coefficient and six tiny nonzero entries. The saved analysis preserves that failed hypothesis and the complete objective comparison.

The next bounded intervention changes the primal iteration: combine the working joint prox with reflected Halpern iteration and the existing restart predicate, holding the certified diagonal metrics and inner gates fixed. Only the actual proximal output may supply the exported primal/equality/L1 multipliers; reflected working duals need not be conic. Exact tiny operator, metric-sign and provenance checks precede captured work. This remains an experimental CPU reference, with no demonstrated qualified speed advantage. The present zero-Q reference also does not establish the original PDHCG quadratic-CG advantage; nontrivial nonzero-Q and held-out mission problems remain required.

[Saved objective comparison, canceled correction and primal-iteration design](../results/local/2026-09-09/core-primal-decision-v636/README.md).

The fixed-metric reflected-Halpern experiment has now completed. Three independent exact tiny maps, metric-sign/provenance checks and restart/failure cases pass before the single two-capture CPU run. It performs 10,000 maps per input, eight restarts per input and 20,002 joint calls in total, without an inner failure, fallback or halving. Complete worker-process time is 24.076938 seconds.

All eight saved original-coordinate readouts fail both unchanged independent audits. Final objective values are slightly lower than v635, but remain about 3.517% above the saved qualified QOCO references, and final primal-cone violations regress to approximately 1.0115e-5 and 5.9071e-6. Complementarity also regresses; dual residual and gap still fail. Lower infeasible objectives are not qualified progress. Exact saved provenance confirms that the exported points are the actual proximal outputs, and all eight restart/reference transitions follow the prescribed rule.

Do not port this variant, increase its unchanged iteration budget or tighten the already successful inner solve. Preserve v635 as the better feasibility reference and inspect the remaining coefficient/conditioning problem before selecting another numerical intervention. Any future change still needs the unchanged original gates and a complete cost comparison. [Exact experiment and negative outcome](../results/local/2026-09-09/core-joint-halpern-v636/README.md).

[Source, fixed-input experiment and independent audits](../results/local/2026-09-09/core-joint-reference-v635/README.md).
