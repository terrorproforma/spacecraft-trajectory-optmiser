# Cold convergence diagnosis from retained v603/v608/v609 evidence

All eight cold outputs fail the unchanged original-equation gate for substantive
numerical reasons. The decomposition does not identify a capture mapping defect,
false score loss, or a proof of physical infeasibility. The pinned upstream
implementation supplies a concrete missing mechanism to test: restarted Halpern
reflection with primal/dual weight adaptation. It also leaves an unnecessary
sparse-Q inner path active on these mathematically zero-Q captures. Neither
observation proves that an intervention will achieve the requested accuracy.

`analyze_saved_cold.py` reads the immutable published captures and compressed or
plain original logs, records their hashes, and recomputes residuals using Decimal
at 65 digits from the exact represented FP64 values. `saved-cold-findings.json`
retains all eight outcomes, source hashes, matrix scale summaries, top residual
coordinates/rows and a checked signed-gap identity. This review ran no solver.

## What the failed vectors say

For every output the signed gap obeys

`pobj - dobj = x·r_dual - (Ax-b)·y + s·z + (h-Gx-s)·z`.

The final term comes only from exported FP64 slack reconstruction. Its magnitude
is at most about 4.7e-15 here; it cannot explain the failed gaps.

| Capture / run | Signed objective gap | x·stationarity | −(Ax−b)·y | s·z |
|---|---:|---:|---:|---:|
| conditioning / v603 | −1212.55885 | −1224.16369 | 7.75879 | 3.84604 |
| conditioning / upstream v608 | −0.00132932 | −0.00026976 | −0.00079448 | −0.00026508 |
| conditioning / v609 natural | −1017.30299 | −1012.72869 | −3.15554 | −1.41876 |
| conditioning / v609 common | 328.22968 | 328.68284 | −0.33601 | −0.11714 |
| difficult / v603 | 0.95899 | 1.03917 | −0.07087 | −0.00931 |
| difficult / upstream v608 | −1.46727 | −0.41374 | −0.78594 | −0.26760 |
| difficult / v609 natural | −21.06029 | −16.92890 | −2.76912 | −1.36227 |
| difficult / v609 common | 14.76672 | 16.02040 | −0.83638 | −0.41730 |

The v603 conditioning point has primal objective 0.09179 and a nonstationary
dual expression 1212.65064. It is not a valid dual lower bound. Its largest
stationarity defects, about 3.48 on zero-cost variables, come almost entirely
from equality-dual products. Difficult-case defects also involve the scalar/SOC
dual contribution on low-cost variables. Equality and cone violations remain
well outside tolerance. Different iteration counts, execution strategies and
deadline endpoints prevent interpreting these rows as a controlled trajectory
of one solve or an isolated stopping-policy speed comparison.

The global gap also conceals cancellation: upstream conditioning has normalized
global gap 0.00132932 but maximum block complementarity 0.02858688. Its worst
scalar row has slack −5.71728e-6 and multiplier 5000.08432. The independent
per-block gate is necessary and must remain enabled.

## Scaling and representation observations

Both captures have numerical Q=0. Their A rows already have maximum coefficients
near one and G row maxima mostly near one. Nevertheless, individual A entries
range down to about 3e-15–5e-15 and c mixes 10,000 penalties with costs about
0.00044–0.00160 and many zeros. Max-row equilibration is not a test of the active
system's conditioning, and objective normalization does not remove those cost
ratios. The current Ruiz implementation shares each affine SOC block's scale;
there is no observed cone-metric violation in that generic free-variable path.

The separate `scaling-findings.json` analysis reports that the 20-step power
estimate is low by roughly 5–6%, but the measured spectral product is still
below one (about 0.898/0.908). Its qualified-point distance calculation suggests
only a modest initial weight correction, about 1.3, rather than a huge global
balance error. Those are numerical estimates, not formal spectral certificates.
The earlier 2^-13 objective rescaling mostly cancels against existing objective
normalization and is not a justified repeated sweep.

There is also an exact later reformulation candidate: every 10,000-cost
epigraph variable detected by the script has no equality occurrence and appears
only in the two inequalities `v-t<=0` and `-v-t<=0`. With positive cost, eliminating
t yields the objective term `10000*abs(v)` and preserves the feasible optimum in
exact arithmetic. A dedicated diagonal-metric L1 primal prox could remove those
large dual pairs. This is separate from the failed singleton-bound folding and
from the current Halpern experiment. It requires full original t/dual recovery,
original KKT verification and its own reviewed implementation; none was made.

## Exact pinned upstream mechanisms

The reviewed upstream commit is `167c8b72b4b96d2f94d405b8763e485514192b81`.
The source, not a general description of PDHCG, establishes these behaviors:

- `src/utils.cu:324` defaults to ten Ruiz passes, cone-preserving scaling,
  additional Pock–Chambolle alpha=1 scaling, objective/bound scaling, 200-step
  termination checks, reflection coefficient 1, and inner BB tolerances from
  1e-3 down to 1e-9. The replay preserved these defaults.
- `src/pdhg_core_op.cu:933` sets a single base step eta=0.998/estimated sigma_max
  (one if the operator is effectively zero). There is no outer eta line search
  or later eta assignment. Actual primal/dual steps are eta/omega and eta*omega.
  Bound/objective scaling makes initial omega=1.
- `src/pdhg_core_op.cu:707` performs a primal-first proximal PDHG map T, then
  computes both reflected components `2T(z)-z`. `:832` and
  `src/kernels/pdhcg_kernels.cu:283` update the internal working point as
  `z_next=((k+1)/(k+2))*(2T(z)-z)+(1/(k+2))*anchor`. This is anchored Halpern
  iteration, not a running arithmetic average of output iterates.
- `src/utils.cu:291` first restarts at iteration 200, then on fixed-point error
  <=0.2 of its restart value, <=0.8 plus worsening, or an epoch length >=0.36 of
  total iterations. `src/pdhg_core_op.cu:859` moves anchor and working state to
  the proximal PDHG point and resets the epoch counter.
- The restart also adjusts omega from primal/dual displacement norms using a
  guarded log-ratio PI update: gains 0.99/0.01, derivative gain zero, integral
  memory smoothing 0.3. Extreme/nonfinite displacement or residual ratios fall
  back to the saved best weight. This is not the same as merely comparing raw
  primal and dual KKT residuals.
- Residuals at `src/pdhg_core_op.cu:1049`, native acceptance, restart and returned
  vectors all use the proximal point T(z). `create_result_from_state` at :1431
  rescales and copies `pdhg_primal_solution` / `pdhg_dual_solution` at :1462–1469.
  The anchor and reflected working point are not the reported answer.

Our persistent main loop loads fixed primal/dual steps once, advances dual-first
using extrapolated primal state and sets `xbar=2*x_new-x_old`. Its buffers named
`average_primal` and `average_dual` are residual/recovery scratch; they do not
implement Halpern or ergodic averaging. CGLS restart belongs to recovery, not this
outer fixed-point mechanism. Those facts remain true with opt-in common stopping.

## Bounded next experiments and mathematical guards

The clean reference-only ablation removes Q storage only when every captured
coefficient is exactly zero. Pass a null/empty Q through the upstream API while
retaining the original capture, objective and auditor. Do not threshold small
coefficients and do not leave a zero diagonal: the upstream diagonal-Q branch
adds 1e-12 at `src/preconditioner.c:822`. Both real inputs currently retain
1,260/1,398 off-diagonal zero entries, so structural classification
(`src/utils.cu:834`) selects sparse Q and BB. The ablation changes numerical
branch and setup work; speed and convergence effects require measurement.
Use paired controls from the same binary, both known points, both cold inputs,
identical explicit budgets and all original gates. Do not credit zero-step
acceptance as cold optimization throughput.

For a native algorithm experiment, a coherent primal-first map is essential.
With our normal-dual convention, stationarity is `c + K^T*y`, so

`x+ = prox_Tf(x - T*K^T*y)`

`y+ = prox_Sigma_gstar(y + Sigma*K*(2*x+ - x))`.

Its symmetric metric is `M=[[T^-1,-K^T],[-K,Sigma^-1]]`, and the fixed-point
metric contains **minus** `2*delta_y^T*K*delta_x`. Upstream's plus sign uses its
opposite Pi convention. Positive diagonal steps, a uniform scalar metric within
each SOC and `||sqrt(Sigma)*K*sqrt(T)||<1` are required. Changes in omega must be
reciprocal in primal/dual steps, and displacement norms must use the scaled
coordinates, not raw native x/y. Do not wrap the existing xbar history in an
anchor formula and call it this operator.

Keep dedicated anchor, working, reflected and proximal-output state. Check and
export T(z), with explicit iteration-zero seed handling, finite guards and
cancellation precedence. Compare plain Halpern against adaptive-restart Halpern
to test the restart/balance bundle; comparison against the current dual-first
default also changes operator state/order and does not isolate only restart.
Keep generic Q=0/free-variable scope, original common qualification and the
existing default compiled path unchanged. This is a finite diagnostic, with
no forecast of convergence or SOTA performance.

Reproduction, CPU only, from the repository root:

```powershell
& 'C:/Users/Angus/.local/bin/python3.12.exe' -B build/performance/cold-convergence-audit-v610/analyze_saved_cold.py --output build/performance/cold-convergence-audit-v610/saved-cold-findings.json
```
