# Fixed reciprocal L1 weight: independent CPU review

The proposed finite comparison is unit weight versus one coefficient-only rule,
`omega = O / B`, held fixed within a solve. This isolates the effect of the global
objective/bound normalization on primal/dual balance. It does not select weights
from a known solution, run a pilot, or assert that this rule is an optimal weight.
Either capture can worsen. No solver or GPU call was made for this review.

Run from the repository root:

```powershell
& 'C:/Users/Angus/.local/bin/python3.12.exe' -B build/performance/l1-weight-review-v618/policy_review.py
```

The script reads only the two hash-pinned original coefficient captures. It
independently detects the exact epigraph pairs, reconstructs ten cone-preserving
Ruiz passes and the power20 estimate, and checks 36 rational-arithmetic proximal
identities. It never opens initial-point files, known solutions, the previous
reference-distance analysis, or GPU logs. Its output records all input hashes,
its own hash, dimensions, scale hashes, step ranges and roundoff checks.

## Fixed-weight mathematics

Write the reduced primal objective as
`f(x) = c_s^T x + sum_j lambda_j |v_j|`, with the retained equality, nonnegative,
and SOC constraints represented by `Kx` and their original bounds/offsets. The
removed original rows are exactly `v-t <= 0` and `-v-t <= 0`, with positive
original objective coefficient `lambda` on an otherwise isolated `t` column.

Let positive diagonal Ruiz factors be `D` and `R`, with one common row factor
within each SOC block. Set scaled coordinates `x_tilde = B D x` and
`y_tilde = O R y`. The scaled operator is `K_tilde = R^-1 K D^-1`, and the
scaled smooth and L1 objective coefficients are `O c_s / D` and `O lambda / D_v`.
The single joint objective norm used to obtain `O` remains unchanged.

For a fixed finite positive weight, scaled steps are `tau_tilde=eta/omega` and
`sigma_tilde=eta*omega`. In original coordinates this gives

```
tau_j   = (eta / omega) O / (B D_j^2)
sigma_i = (eta * omega) B / (O R_i^2)
v_next  = soft(v - tau_v * retained_gradient, tau_v * original_lambda)
```

Thus the exact product matrix `Sigma^(1/2) K T^(1/2)` is independent of omega.
For a saddle point and exact proximal operations, the same fixed-metric PDHG
condition `||Sigma^(1/2) K T^(1/2)|| < 1` applies to every positive fixed omega.
The existing power20 estimate is heuristic; weight cancellation neither proves
that estimate is an upper bound nor upgrades a numerical spectral observation
to a general theorem. FP64 product equality is approximate, which the CPU
script measures on every retained numerical matrix entry.

Uniform reciprocal weighting preserves the scalar step within each SOC block,
so the same Euclidean Moreau/SOC projection remains the correct diagonal-metric
proximal operator. It must not change original cone geometry or exported dual
units. After a cold step, completion remains `t=abs(v)` and native pair duals
`z_plus+z_minus=lambda`, `z_plus-z_minus=lambda*sign(v)` for nonzero `v`. At exact
zero, the difference is `clip(-retained_gradient,-lambda,lambda)`. There is no
extra omega factor in those original duals. A tiny nonzero value cannot silently
be snapped to zero to choose an interior subgradient.

The original supplied approximate certificate must be checked before epigraph
completion. Its `x,t,y,z` must remain unchanged on a zero-update accepted solve;
otherwise the near-zero/interior-dual cases already demonstrated in v615 can
lose their original stationarity certificate. The original equality/cone,
stationarity, global gap and per-block complementarity gates remain mandatory
for every exported iterate. Natural residual telemetry still describes the
legacy unweighted original-problem metric, not the weighted update displacement.

## One coefficient-only policy

Choosing `omega=O/B` cancels the two global scale factors in the native steps:
`tau_j=eta/D_j^2`, `sigma_i=eta/R_i^2`. The implementation should compute the
ratio on the GPU once after reduced scaling, retain all original coefficients,
and use the same quotient/multiply operations as a supplied fixed weight. This
is a balance ablation, not removal of the L1 penalty or objective normalization.

| Capture | Pairs | Omega from coefficients | Native unit primal step | Native policy primal step | Native policy dual step |
|---|---:|---:|---:|---:|---:|
| conditioning | 1470 | 0.0001856525042453329 | 0.0000607 | 0.32693 to 0.32700 | 0.32693 to 0.65352 |
| difficult | 1631 | 0.00018024670054936622 | 0.0000593 | 0.32870 to 0.32928 | 0.32870 to 0.65290 |

The unchanged positive penalty of 10000 makes the corresponding policy soft
thresholds about 3270 and 3290. This follows from the larger primal step; it does
not relax the objective. All actual diagonal steps and thresholds are finite
and positive for these captures. The independent D/R fingerprints match v615,
and both global factors reproduce exactly. Represented product/cancellation
relative error is at most `4.44e-16`; each SOC retains identical row steps.

The standard fixed-weight reparameterization and an objective/RHS norm-based
initialization are described in the primary [PDLP paper, Sections 2 and 3.3](https://optimization-online.org/wp-content/uploads/2021/06/8439.pdf).
Our cancellation rule is a separate experimental choice. Applied after the
existing joint global normalization, an analogous coefficient norm ratio is
only 1.01425 or 1.01393 here, close to the unit control. Iteration-displacement
updates in that paper are adaptive heuristics; the fixed-weight argument above
does not prove convergence of an arbitrary adaptive implementation.

## Implementation and finite comparison requirements

- Preserve the compiled unit-weight arithmetic and original initializer. Use a
  separate specialization for nonunit weighting and query its own occupancy.
- Reject invalid API mode/omega before mutation; validate represented omega,
  reciprocal, base steps, every active diagonal step, and every positive L1
  threshold before the first solver update. Skip inactive dummy entries.
- Changing policy restores history from the actual exported primal point,
  refreshes scaling and invalidates old records, while preserving primal/dual
  values. Ordinary reset/reseed retains the configured policy. A fresh L1 enable
  resets to unit. A completed cancellation cannot imply a valid chosen weight
  when initialization was cancelled before choosing one.
- Compare exactly unit versus cancel-global on each of the two original captures
  with the same 128 blocks, 100000-update/30-second limits and original accuracy
  gates; include the two original-seed preservation checks separately. Any seeded
  zero-update acceptance is a lifecycle check, not cold solver performance.
- Retain all failed/cancelled runs and original-coordinate vectors. Evaluate
  equal-accuracy qualification before any speedup, default-promotion or mission
  score claim. Do not select another weight after seeing these results within
  this finite comparison.

This is a mathematical/policy review, not clearance of an unfrozen binary. Final
source, test, occupancy/resource and saved-outcome checks are separate records.
