This is a coefficient-only design for a future experiment combining exact mass elimination with the existing exact L1 reduction. It contains no solver implementation, GPU experiment, solution-derived parameter, speed prediction or default change. It addresses the invalid step-size reuse identified in `build/performance/mass-structure-review-v621`; it does not establish improved cold convergence.

Diagonal primal-dual preconditioning is an established approach: Pock and Chambolle, *Diagonal preconditioning for first order primal-dual algorithms in convex optimization*, ICCV 2011, pp. 1762–1769, DOI 10.1109/ICCV.2011.6126441. The authors' [institutional publication record](https://tugraz.elsevierpure.com/en/publications/diagonal-preconditioning-for-first-order-primal-dual-algorithms-i/) describes diagonal preconditioners that provide convergence without estimating a spectral step size. The bound and scan specialization below are derived directly for our captured coefficient matrices; no novelty is claimed for the preconditioner.

Let K' be the exact combined constraint operator after removing the positively priced L1 epigraph variables/pair rows and the mass variables/causal equality rows. Keep every virtual control, every retained scalar/SOC constraint, and every affine constant. These two captured Hessians are exactly zero and their original mass objective coefficients are zero. The reduced objective is linear plus separable L1, so its diagonal-metric primal proximal step remains soft thresholding with threshold tau_j times the original L1 coefficient. Constants belong in constraint right-hand sides, not in K' or its adjoint.

For the exact numerical coefficients define row sums a_i = sum_j |K'_ij| and column sums b_j = sum_i |K'_ij|. Set A_i = a_i for scalar/equality rows. Within each SOC block use the same denominator A_i = max of its row sums for every component. With theta = 19/20, choose tau_j = theta/b_j and sigma_i = theta/A_i. A zero denominator uses the positive dummy value 1; constant constraints and all original gates must still be retained. The CPU audit converts each exact rational step to FP64 inward, so the stored positive step is no larger than that ratio. Overflow, underflow to zero and nonfinite data are failures, not reasons to relax the bound.

This is a sufficient norm bound, independent of a power estimate. Write alpha = max_i sigma_i a_i and beta = max_j tau_j b_j. Weighted Cauchy–Schwarz gives, for any vector u,

    sum_i sigma_i (sum_j K'_ij sqrt(tau_j) u_j)^2
      <= sum_i sigma_i a_i sum_j |K'_ij| tau_j u_j^2
      <= alpha sum_j b_j tau_j u_j^2
      <= alpha beta ||u||^2.

Hence ||sqrt(Sigma) K' sqrt(T)||^2 <= alpha beta <= (19/20)^2 = 0.9025 < 1. This is an exact-arithmetic certificate for the represented FP64 coefficients and inward-rounded stored steps. It is not a blanket bound on floating-point errors in iterative products. Tied SOC steps preserve the ordinary Euclidean SOC projection. A fixed positive reciprocal weight T/omega and omega*Sigma preserves the norm product, but no adaptive rule follows from this proof. This design uses direct original-coordinate steps: it must not silently multiply them by the old B/O/D/R factors. Any implementation retaining global normalization must prove the actual applied primal/dual steps equal the stated metric.

The captured mass equations are m_0 = 1 and m_(k+1) - m_k + g_k Gamma_k - nu_m,k = a_k, with g_k > 0. Thus m_j = d_j + sum_(k<j) (nu_m,k - g_k Gamma_k), where d includes every original nonzero a_k. Each node has exactly three retained singleton scalar mass rows with coefficient +1 or -1, and no retained equality or SOC uses mass. Therefore each transformed mass row at node j has exact absolute sum P_j = sum_(k<j) (1 + |g_k|), computed by a prefix sum. The added column sums are 3(K-k) for nu_m,k and 3|g_k|(K-k) for Gamma_k. Add the untouched non-mass coefficient sums. These formulas rely on the detected singleton pattern; they must not be applied to arbitrary mixed mass rows where cancellation could occur.

The operator itself also stays implicit: a prefix computes nu_m - g*Gamma contributions; the transpose first gathers the signed scalar mass-row covectors into nodes and then uses a suffix sum. Constants never enter the transpose. `audit_metric.py` checks the scan-derived sums against every explicitly expanded coefficient using exact fractions. Explicit expansion is a CPU oracle only; materializing the fill is not the proposed production representation.

| Coefficient result | Conditioning | Difficult |
|---|---:|---:|
| Intervals | 210 | 233 |
| Retained variables / constraint rows | 3,580 / 7,600 | 3,974 / 8,432 |
| Non-mass stored numerical coefficients | 19,989 | 22,180 |
| Logical numerical coefficients after mass substitution | 152,919 | 185,746 |
| Tied SOC blocks | 210 | 234 |
| Constant operator rows retained | 3 | 4 |
| Exact stored-step norm-squared upper bound, rounded for display | 0.9025 | 0.9025 |
| Numerical weighted Gram Rayleigh quotient after 80 iterations | 0.9019260 | 0.9021229 |
| Numerical eigenpair residual, L2 | 0.0008457 | 0.0010221 |
| Minimum mass-virtual primal step | 0.00150794 | 0.00135908 |
| Median mass-virtual primal step | 0.00301587 | 0.00270655 |
| Minimum nonconstant mass-row dual step | 0.00451658 | 0.00407367 |
| Median nonconstant mass-row dual step | 0.00903317 | 0.00811251 |
| Minimum L1 threshold | 15.0794 | 13.5908 |
| Median L1 threshold | 9,500 | 9,500 |

The Rayleigh quotients are coefficient-only numerical probes, not certificates or optimization results. The exact rational certificates and full step distributions are in `findings.json`. The large soft thresholds are legal proximal parameters, but their effect on progress still needs measurement; a stability bound alone does not guarantee useful finite-budget convergence.

Reusing the *old, uneliminated* reduced-operator Ruiz metric on the new cumulative operator is invalid. Recomputing its ten passes reproduces both previously recorded D/R hashes exactly. On all mass virtual columns and scalar mass rows those old factors are 1. The three copies of the cumulative mass block give norm-squared lower bound (K+1)(2K+1)/2 before the old eta factor. With the old eta this yields **at least 4,748.19 and 5,913.88**, far above 1; reciprocal B/O or weight factors cancel from this product. This is a rejected hypothetical reuse after mass elimination, not a claim that the current uneliminated implementation violates this particular bound. The new diagonal sums explicitly account for the cumulative coupling and its unequal column participation.

Before an implementation experiment is eligible for GPU comparison, it must satisfy these concrete checks:

1. Match the exact frozen mass and L1 structures, including all affine constants, virtual variables and original objective coefficients. Reject unsupported nonzero Hessians, shifted coordinates, mixed mass rows or ambiguous/duplicate maps until their transformations are separately derived. Check original supplied qualified points before any reduced canonical reconstruction; do not snap tiny nonzero virtual controls to zero.
2. Check signed prefix/suffix products against the explicit CPU operator, an adjoint dot-product oracle, both coordinate maps and all original constraint residuals. Use deterministic finite test vectors that exercise every sign and endpoint; make the constant contribution explicit. Verify positive finite sums, stored steps and soft thresholds, tied SOC entries and an independently recomputed upper norm bound. A production FP64 sum must be conservatively bounded or checked; the exact-fraction CPU certificate does not automatically certify a different GPU reduction's rounding.
3. Recover eliminated equality duals in original coordinates. If the removed block is Lm + Hq = b_E, let w = c_m + C_m^T y_C + G_m^T z for retained original rows and set y_E = -L^(-T) w. For this causal block that is a suffix sum: the initial-mass multiplier is -sum_j w_j and mass-dynamics multiplier k is -sum_(j>k) w_j. Recover these contributions before completing original L1 epigraph multipliers, or use the algebraically identical reduced gradient. Preserve dual objective constants: the general affine substitution changes the objective offset by c_m^T L^(-1)b_E, which is zero in these captures but must be explicit in the contract.
4. Export a complete original x/y/z/slack record and run every unchanged common original-equation KKT gate: normalized stationarity using separate Qx, c, A^T y and G^T z contributions; normalized primal equations; absolute primal/dual cone defects; normalized primal-dual objective gap; and every scalar/SOC block's complementarity. Use the original objective and denominator max(1, |pobj|, |dobj|). Exact causal reconstruction can improve those mass equalities while other feasibility, stationarity or gap remains poor; it must not conceal that result.
5. Isolate the new representation and metric behind an explicit experimental mode. Establish unchanged seeded zero-step acceptance, cancellation and disable/reseed history behavior, then compare a finite cold control/experiment pair per capture at identical original gates and work/deadline budgets. Keep setup time, iteration time and retained bytes separately scoped. No parameter grid, reference-derived weight or automatic promotion is justified by this note.

Reproduce from the repository root with `python build/performance/mass-diagonal-metric-v621/audit_metric.py` using only the Python standard library. It consumes the two canonical snapshots in `build/performance/known-point-replay-v606/inputs`, the pinned reader and mass findings in `build/performance/mass-structure-review-v621`, and the coefficient-only old scaling report in `build/performance/l1-scaling-v615b/findings.json`. It does not open point files. Source/capture/report identities, step hashes, exact rational certificates and probe residuals are recorded in `findings.json`; the mass source bindings resolve to the frozen v618b assembly/discretisation source reviewed previously. Publication should preserve those dependencies or provide a path-only wrapper without changing the numerical audit.
