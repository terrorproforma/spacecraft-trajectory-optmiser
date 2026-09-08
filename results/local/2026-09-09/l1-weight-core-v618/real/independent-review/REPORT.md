# Independent original-equation audit: v618 fixed L1 weight

All six saved vector sets agree with the original-equation Decimal65 audit.
Both supplied original points qualify unchanged at zero updates. All four cold
100000-update cases remain unqualified. The coefficient-only cancellation rule
correctly changes primal/dual balance, but it does not solve the remaining
feasibility problem. No default promotion, equal-accuracy speedup, mission score
increase or GPU-native mission integration is established by this comparison.

Reproduce from the repository root with Python's standard library:

```powershell
& 'C:/Users/Angus/.local/bin/python3.12.exe' -B build/performance/l1-weight-real-review-v618/review_real.py
```

The script directly compiles the pinned Decimal auditor source and uses exact
represented FP64 coefficients/vectors at 65 decimal digits. It verifies original
equalities, cone equations, stationarity, primal/dual cone membership, global
objective gap, and per-block complementarity with the existing gates. It also
checks original point-file hashes and input-to-initial-to-final x/y/z bits,
source/core/report/raw-log identities, all original-coordinate objective terms,
and the coefficient-derived weight/threshold calculation. No GPU, solver,
compiler or parameter search is invoked.

The saved report is SHA
`6454ecc1d507fb8e781f49fe8c8e731e79d77cd567e1914c57c1a43ebe7c068b`.
The exact core is
`1002c69e2418ab8b4ba1cdb7376f2959b8af455ae7ea4815db751508009fceec`,
with uncommitted source-tree identity
`e476bbf3065173ed7b06fca49306537ffe1d47804c281bacd441e5a42104e1a9`.
All source/input/raw hashes and arithmetic decomposition details are retained in
`findings.json`.

## Measured outcomes

These are normalized original residuals, except primal-cone violation, which is
absolute. Required normalized residual/gap/block tolerance remains `1e-9`, and
absolute cone tolerance remains `1e-8`.

| Capture/policy | Primal | Stationarity | Gap | Max block complementarity | Primal cone | GPU solve seconds |
|---|---:|---:|---:|---:|---:|---:|
| conditioning, unit | 1.73919e-4 | 2.24718e-4 | 1.000291 | 4.60723e-15 | 0 | 3.54654 |
| conditioning, O/B | 1.76496e-4 | 7.54268e-8 | 0.051506 | 2.69468e-9 | 1.76100e-6 | 3.75512 |
| difficult, unit | 5.46409e-6 | 1.88836e-5 | 0.00387847 | 7.40658e-4 | 4.24549e-5 | 3.69722 |
| difficult, O/B | 8.73715e-5 | 2.29656e-8 | 0.0565266 | 1.34299e-4 | 7.30277e-5 | 3.77067 |

All four terminate at the iteration limit, with finite diagnostics and no
recovery or deadline cancellation. This is one matched-build, matched-128-block
comparison on an RTX 5090, not a statistical performance distribution. Total
work is six processes, eight solve APIs, two bootstrap updates, and 400000 cold
updates. Seed zero-update checks are separate lifecycle tests.

Both GPU-computed weights equal the independent coefficient-only predictions:
`0.0001856525042453329` and `0.00018024670054936622`. B and O reproduce exactly;
eta and minimum/maximum L1 thresholds agree within `1e-13` relative tolerance.
Original objective coefficients, cone geometry and acceptance gates are intact.

## What remains wrong numerically

Exact L1 completion works. Every cold point has `t=abs(v)`; the original
epigraph-pair complementarity is exactly zero. Nonzero virtual variables use
the correct endpoint duals. Unit difficult has two positive virtual variables;
both cancellation cases have every virtual variable exactly zero. There is no
evidence here of an importer, completion or global scaling mismatch.

Cancellation greatly reduces stationarity residuals, but equality feasibility
persists or worsens. Its dominant equality errors are **mass continuity and
initial mass**. The pinned assembly `gtoc12_conic.cu:153-168` emits seven rows
per interval, with mass at component 6; line 174 emits initial mass. The script
also verifies the actual captured initial-mass row is exactly `x[6]=1`.

For conditioning, initial-mass row 1476 has residual `+0.000624841591`; the first
mass-continuity row 6 has `+0.000624755867`. For difficult, initial-mass row 1637
has `+0.000334817270`, and row 6 has `+0.000334792790`. Similar errors recur at
rows 13, 20, 27, and onward. Difficult equality error is about 16 times the unit
case. Its SOC block 233, starting at original G row 10276, also remains violated.

The low difficult cancellation objective **0.15384065 is infeasible**, not an
improvement over the unit value 85.36945628. Unit pays 85.16955837 in virtual
control penalty; cancellation pays zero while violating mass equations. Its
equality-dual infinity norm is only 7.73625 versus 10000.00092 for unit. The
smaller dual step is consistent with slow accumulation of feasibility-enforcing
multipliers. These saved endpoints support that diagnosis but do not establish
the entire transient history or prove a remedy.

Objective denominators also change. Conditioning unit uses 295.641885 while
cancellation uses 1; difficult unit uses 85.369456 while cancellation uses 1.
Consequently, normalized gap or complementarity changes cannot be interpreted
without their absolute values. Neither unqualified dual objective is a certified
lower bound.

The Decimal signed-gap identity includes all four terms:

```
f - d = x^T stationarity - (Ax-b)^T y + s^T z - (Gx+s-h)^T z.
```

Conditioning cancellation has `-0.05833470 + 0.00682892 - 1.20772e-7`
plus negligible cone-equation roundoff, giving signed gap `-0.05150589`.
Difficult cancellation has `+0.03063479 - 0.08702254 - 0.00013885`, giving
`-0.05652660`. The identity closes within `1e-50` absolute in every record.

The next diagnostic should target this **feasibility/stationarity tradeoff and
mass-row propagation**, using the original gates and a recorded finite work
budget. These results reject promoting a universal O/B cancellation policy.
They do not justify another weight grid, a relaxed physics gate, or selecting a
new intervention before this finite comparison is packaged. Any future
iterate-based metric/restart change needs its own reviewed transition contract;
the fixed-weight convergence argument is not a proof for arbitrary adaptation.
