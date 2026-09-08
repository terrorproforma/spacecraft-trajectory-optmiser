The supplied QOCO points are qualified under the fixed common KKT gate, and their mapped native starts were copied correctly. The persistent solver uses a different stopping metric and starts updating before testing the initial point. Those updates can destroy objective-gap qualification while barely moving the primal. This is a demonstrated stopping/solution-preservation problem; it does not establish the cause of every cold-solve failure.

Evidence: all eight cases in report SHA256 `8c51d621edb648f8fdb355972c4c01add7365fa7c43c9004bf1790cc0f6c452c`, adapter `659f5abd7b41d16dbeac980a6e10d6ca92666314`, core SHA256 `d4b0bea9672bb0cea612b7d4e8db5d448f1b79b9831be5710723276edd128633`. The accompanying standard-library CPU script binds inputs, encoded points, logs, source/core identities and native seed order, then recomputes natural-map components and original gap decompositions. Its Decimal products extend exact FP64 inputs; they do not claim bitwise equivalence to GPU reductions. The separate `audit/` package checks the earlier QOCO-to-encoded source chain, including its documented signed-zero normalization.

At a feasible scalar inequality with nonnegative dual, the native natural residual is `min(slack, dual)`, whereas the common gate checks product complementarity together with feasibility, normalized stationarity and objective gap. These finite-tolerance conditions differ:

| Capture; zero-based G row | Slack | Dual | Product | Native natural residual |
|---|---:|---:|---:|---:|
| Conditioning; 468 | 4.26390e-8 | 3.81865e-8 | 1.62823e-15 | 3.81865e-8 |
| Difficult; 3515 | 2.01798e-6 | 1.40971e-8 | 2.84476e-14 | 1.40971e-8 |

Both exceed the native absolute threshold `1e-9`. The seed SOC natural residuals also exceed it: approximately `1.70525e-8` and `8.73820e-9`. Folding removes the offending scalar rows but turns the corresponding normal condition into projected box stationarity, approximately `3.81865e-8` and `1.40972e-8`; it does not remove the stopping mismatch. The reported native complementarity is **not** a component of the actual stopping maximum, so its difficult-seed value `5.22936e-8` is not an additional rejection cause.

The kernel updates duals first, computes the new dual contribution to the gradient, updates/projects the primal, and only then tests termination. The adapter's pre-step measurement confirms the imported internal vectors are unchanged, but a residual-only API call cannot accept a solve. In exact scalar arithmetic the upper-bound dual update is `z_new=max(0,z+sigma*(Gx-h))`: weakly complementary QOCO multipliers therefore change even without a mapping or floating-point bug. The current expanded implementation additionally subtracts scaled quantities, which can introduce rounding, but this evidence does not isolate rounding as the primary cause.

| Capture / representation / iterations | Final common relative gap | Common KKT passes? |
|---|---:|---|
| Conditioning / generic / 1 | 6.04613e-7 | No: gap |
| Conditioning / generic / 1000 | 4.21818e-7 | No: gap, block complementarity |
| Conditioning / folded / 1 | 1.27414e-7 | No: dual, gap |
| Conditioning / folded / 1000 | 1.84037e-6 | No: gap, block complementarity |
| Difficult / generic / 1 | 2.49764e-10 | **Yes; native termination remains iteration limit** |
| Difficult / generic / 1000 | 1.23070e-7 | No: gap |
| Difficult / folded / 1 | 1.59897e-7 | No: gap |
| Difficult / folded / 1000 | 1.81965e-8 | No: gap |

All eight native solves end at their iteration limit. The largest primal change is only `7.11117e-9`. Conditioning's first generic step changes `c^T x` by `-4.34605e-12`, while `h_nonnegative^T delta_z=-6.04622e-7` dominates its signed-gap change. Its final gap is almost entirely `x^T stationarity=-6.04618e-7`; equality/slack terms are approximately `-1.70e-12` and `6.68e-12`. After 1000 generic steps, difficult's signed-gap change includes `b^T delta_y=-7.74964e-6`, `h_nonnegative^T delta_z=-2.29307e-6`, and `c^T delta_x=-1.25167e-6`. These are conic subproblem objectives, not campaign kilograms or kilograms per ship. Folded dual-change accounting includes export reconstruction because those multipliers are absent from native state.

Source locations in the frozen core:

- `cpp/cuda/src/persistent_pdhcg.cu:992` defines scalar natural projection; line `1102` defines the stopping maximum without complementarity.
- Lines `1226`, `1274`, `1295`, `1305` contain the scalar dual update, explicit primal update, first post-update check and absolute natural-residual acceptance.
- `cpp/cuda/src/cooperative_pdhg.cuh:584`, `715`, `850`, `898`, `919`, `929` implement the corresponding cooperative operations.

The next bounded implementation experiment should isolate an **explicitly named common-KKT stopping policy**, evaluated on the current native iterate before its first update and at normal check points. Start with the generic representation and these unshifted captures. Preserve the existing `1e-9` normalized primal/dual/gap/block-complementarity thresholds, `1e-8` cone threshold, all finite-value checks and independent original-coordinate audit. Continue exporting the absolute natural residual separately; do not silently reinterpret its existing contract. Qualified starts should then return unchanged at zero iterations, and malformed/nonqualified/cancellation fixtures must still reject. Follow with the same cold captures and fixed budgets, with recovery disabled, to measure whether comparable stopping accuracy changes cold qualification. This is a proposed experiment, not a completed fix or a promised speedup. Do not combine it with step tuning, restarts, folding or objective changes.

The core's main update is explicit PDHG; it has no inexact quadratic/conic proximal inner solve. Original QP PDHCG uses CG for an appropriate quadratic proximal system, while PDHCG-CQP uses projected-gradient inner iterations for cone constraints. **Both captures here have P=0**, with an unconstrained or box-constrained primal domain: the linear-objective proximal map already is the explicit step, with applicable box projection. Missing CG, or missing quadratic inner iterations, therefore cannot by themselves explain these controls. Recovery CGLS projects through constraint maps rather than solving a quadratic primal proximal system.

The script also preserves two alternative-assessment results. Existing objective normalization nearly cancels multiplying P/c/offset by `2^-13`: effective original-coordinate primal/dual ratios are `0.97908/1.02136` and `0.98012/1.02028`. At `2^-19` they become `0.42240/2.36744` and `0.43512/2.29820`; improvement is untested. Exact equality augmentation retains `Ax=b` and uses `P_aug=P+rho*A^T*A`, `c_aug=c-rho*A^T*b`, `k_aug=k+(rho/2)*b^T*b`; recover `y_original=y_aug+rho*(Ax-b)`, with z unchanged. In translated coordinates use the translated b/c/offset consistently, then audit the original equations. FP64 Gram formation and cancellation are not exact merely because rho is a power of two.

| Capture | Original full stored P | Full augmented union pattern | Minimum nullity of A^T A |
|---|---:|---:|---:|
| Conditioning | 1,893 | 88,211 | 3,774 |
| Difficult | 2,100 | 97,880 | 4,191 |

Those approximately 46.6-fold pattern expansions retain A's stored structural-zero support and the original P pattern. Using only currently nonzero A coefficients gives smaller full unions of 70,571 and 78,308 entries, with different reuse assumptions. Equality curvature vanishes along feasible tangent directions, and this explicit core would incur a larger Q product and a changed step bound. These are structural and exact-arithmetic observations, not runtime or convergence predictions.
