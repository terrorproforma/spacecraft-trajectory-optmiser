# Exact L1 epigraph reduction: independent review v615

The proposed reduction is mathematically exact for the detected structure. A
dual-first common-PDHG-only diagnostic with an explicitly masked working operator
is a useful bounded ablation. This review does not approve unseen implementation
code or establish convergence, performance or mission/competition results.
No GPU or solver calls were performed.

## Exact detection and equivalence

Require an exactly zero Hessian and a variable t with finite cost lambda>0 whose
only mathematically nonzero constraint uses are these two nonnegative rows:

```text
 v - t <= 0        (positive row)
-v - t <= 0        (negative row)
```

Each row must contain exactly the indicated two nonzero coefficients and zero
RHS. There must be no nonzero equality, SOC or other scalar use of t. Original
primal variables are free in the supported common domain. Reject overlapping or
nested t/v roles and duplicate v targets in this first diagnostic. Structural
zeros may be ignored when proving absence, but no nonzero coefficient, however
tiny, may be discarded. No global sparse-zero removal is implied.

For each feasible original point, t>=|v| and lambda*t>=lambda*|v|. Every feasible
reduced point extends to an original feasible point by t=|v|, with identical
objective. Thus replacing these pairs by lambda*|v| preserves the primal infimum
and optimal feasible points; lambda>0 forces t=|v| at an attained optimum.
This argument is about feasible points, not approximate certificates whose
epigraph inequalities may already have tiny accepted defects.

Independent exact detection finds:

| Capture | Pairs | Logical variables | Equalities | Retained G rows | Total retained K rows |
|---|---:|---:|---:|---:|---:|
| Conditioning | 1470 | 3791 | 1487 | 6324 | 7811 |
| Difficult | 1631 | 4208 | 1648 | 7018 | 8666 |

All costs are exactly 10000. Every map record agrees with the independent
`build/performance/l1-scaling-v615b/findings.json` map. The retained numerical K
entry counts are 21463 and 23815. Difficult has three retained variables after
its epigraph t range, so truncating a presumed trailing epigraph suffix is wrong;
explicit masks/maps are required.

## Diagonal proximal step and scaling

Let g_v be the smooth retained gradient, including retained equality/scalar/SOC
dual contributions but excluding the removed pair. With native diagonal step
d_v=eta*Sx_v (dual-first mode has no Halpern weight), use

```text
u = working_v - d_v*g_v
v_next = sign(u)*max(abs(u)-d_v*lambda, 0)
```

Branch directly on u against +/-d_v*lambda to produce exact zeros. Do not round
other small values to zero. Check finite positive steps/costs, threshold overflow
or underflow-to-zero, and nonfinite centres/outputs. The recorded one-ULP tests
produce v=+/-1.734723475976807e-18 just outside the threshold: these are nonzero
outputs and must receive the corresponding endpoint subgradient.

For x_tilde=B*D*x and objective scaling B*O,
c_tilde=O*c/D and lambda_tilde=O*lambda/D_v. Consequently
Sx_v=O/(B*D_v^2), and the scaled and native soft-threshold formulas agree.
The CPU oracle checks this identity exactly on a power-of-two fixture.

The proposed joint coefficient norm is consistent:

```text
O = 1/(1 + sqrt(sum_active((c_j/D_j)^2)
                 + sum_pairs((lambda/D_v)^2)))
```

Lambda uses the retained target v's D, not eliminated t's D. Both smooth and L1
terms use this same O. D/R and the power operator exclude removed rows/columns;
B uses retained RHS values. Positive finite dummy scales for inactive entries
are safe only when all working norms, products and updates exclude them.

The unscaled original/joint objective norms are about 383406 and 403856, whereas
the retained smooth c norms alone are only 0.02318 and 0.01343. Omitting lambda
from O would change the preconditioning dramatically. Original p.c must remain
available unchanged for readout and the common original-equation audit.

The separately reviewed CPU scaling source uses ten simultaneous Ruiz passes
with SOC block scales tied. Its thresholds are about 0.6070 and 0.5930. Numerical
eigenpair evidence gives eta^2*||K_scaled||^2 about 0.901455 and 0.910035, but this
is not a general proved bound: the conservative one/infinity norm upper bound
still gives values above 1. Preserve the experimental spectral caveat.

## Original dual completion

Original nonnegative multipliers z_plus,z_minus obey

```text
t stationarity: lambda - z_plus - z_minus = 0
v stationarity: g_retained + z_plus - z_minus = 0
slacks:        t-v, t+v
```

At a newly generated canonical point t=|v|, use:

- v>0: z_plus=lambda, z_minus=0.
- v<0: z_plus=0, z_minus=lambda.
- v exactly zero, including signed zero: delta=clip(-g_retained,-lambda,lambda),
  z_plus=(lambda+delta)/2, z_minus=(lambda-delta)/2.

At zero, take g_retained from the actual exported retained dual iterate, not an
earlier working dual. Implement the split without forming potentially overflowing
lambda+delta; the retained oracle forms half terms and a complementary multiplier.
The original audit must check the actual rounded multipliers. If |g_retained|>
lambda, clipping leaves a nonzero stationarity defect; it must not hide it.

At exact zero, this completion is a valid choice in the L1 subdifferential. It
need not equal the particular proximal optimality witness obtained from the
previous working state. At nonzero v, only the true sign endpoint is the exact
subgradient. Choosing a clipped interior completion at nonzero v would be a
different approximate original-QP certificate construction, not this proposed
algorithmic map; it is not recommended for the first diagnostic.

## Qualified seed hazard: measured, not hypothetical

All 1470 conditioning reference v values are nonzero, with magnitudes from
4.3898e-21 to 1.9548e-16. Every pair has two positive multipliers, typically near
5000/5000. Difficult similarly has 1631 nonzero values and interior pairs, with
magnitudes from 1.3447e-17 to 0.00702580. These approximate certificates pass the
same original common gate even though they are not exact reduced subgradients.

The Decimal65 experiments preserve all source points and only transform copies:

| Experiment | Conditioning | Difficult |
|---|---:|---:|
| Objective change from replacing only t by abs(v) | +5.50580e-10 | +2.41213e-7 |
| Relative gap after replacing t only | 5.63185e-10 | 2.80480e-9 |
| Relative stationarity after also forcing strict sign duals | 0.999972 | 1.296493 |

Replacing t alone already makes difficult fail its 1e-9 gap gate. Forcing the
dual endpoints destroys both certificates. Therefore perform the initial common
check on the untouched original x, t, y and z before any canonicalization. If
it passes, return those exact original buffers at zero updates. Cold-generated
soft-threshold outputs may then use the canonical reconstruction above. Never
silently snap reference v values to zero to manufacture this behavior.

## Original audit and state isolation requirements

The working gradient/operator/scaling use the reduced problem. The exported
point and acceptance gate use the full original problem. In particular:

- Reconstruct t and pair duals only for newly generated iterates; keep their
  contributions masked from later working gradients and dual updates.
- Compute the full original objective c_original*x, including lambda*abs(v), and
  the full original stationarity/equality/slack/cone equations. The removed rows
  have zero RHS, so their dual-objective terms are zero, but their stationarity
  and complementarity terms must still be audited.
- Keep the existing original normalization groups and max-block complementarity
  gate. The objective denominator is max(1,abs(original primal objective),
  abs(original dual objective)); do not substitute the smooth-only objective.
- Enabling/changing the map forces transformed scaling refresh. Disabling it
  forces original D/R/B/O/spectral refresh: masked-operator steps are unsafe to
  silently reuse after the omitted original rows are restored. Restore original
  primal history for an explicit mode-off warm-start contract.
- Reject checkpoint/restore while enabled unless the transformed map/scaling
  state is completely bound; the simpler diagnostic contract is rejection.
  Existing common-policy update restrictions must prevent stale maps.

The C API must validate the map independently on GPU before enabling it:
index ranges, unique t/v/row roles, lambda==original c_t>0 finite, scalar upper
row membership with zero RHS and negative-infinite lower bound, exact row
coefficients, absence of every other nonzero t use, globally exact zero Q and
the supported free-primal/common domain. CPU detection is not sufficient proof
for an externally supplied API map. Reject or report failure before any unsafe
index access or cooperative launch. No extra GPU work was performed for this
review; actual source/lifecycle/build inspection remains a later gate.

## Reproduce the independent evidence

From repository root with standard-library Python:

```text
python -B build/performance/l1-prox-review-v615/analyze_l1.py --output build/performance/l1-prox-review-v615/findings.json
```

The script pins both captures and reference files, verifies the exact maps,
audits the original and counterfactual points, checks objective equivalence and
proximal/scaling/zero-completion oracles, and rejects malformed detection and
overflow/underflow fixtures. Its Decimal helper is loaded directly from recorded
source bytes, without inherited Python bytecode. The findings include full map
identities so another implementation can compare every pair.
