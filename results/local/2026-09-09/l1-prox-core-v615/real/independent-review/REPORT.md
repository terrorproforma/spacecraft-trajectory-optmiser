# Independent v615 real-capture audit

All six original-equation Decimal65 verdicts match the GPU gate and existing
long-double auditor. Both original supplied certificates qualify at zero steps,
with encoded input, imported point and returned x/y/z bits identical. All four
100,000-step cold endpoints remain unqualified. No source, solver, capture,
acceptance threshold or raw result was changed by this review.

The saved run is local **NVIDIA GeForce RTX 5090**, driver 595.97, with an explicit
128-block grid. Six processes made eight solve API calls: four cold solves,
two zero-step seeded solves and their two separately recorded one-step bootstrap
solves. Total actual updates were 400,002. No deadlines, recovery or retries
occurred. These are not Lambda/H100 timings.

| Capture / mode | Normalized primal | Normalized dual | Relative gap | Max block complementarity / objective scale | Absolute primal cone violation | GPU solve seconds |
|---|---:|---:|---:|---:|---:|---:|
| Conditioning, off | 8.79008e-5 | 3.48341e-4 | 0.9999243 | 6.62898e-4 | 1.55583e-4 | 5.424073 |
| Conditioning, L1 | 1.73919e-4 | 2.24718e-4 | 1.0002905 | 4.61644e-15 | 0 | 3.610990 |
| Difficult, off | 4.15900e-6 | 2.12548e-5 | 0.01122705 | 5.60743e-4 | 7.79867e-6 | 5.549215 |
| Difficult, L1 | 5.46409e-6 | 1.88836e-5 | 0.003878469 | 7.40658e-4 | 4.24549e-5 | 3.656115 |

The fixed original gates are 1e-9 for the four normalized metrics and 1e-8
for absolute primal/dual cone violations. L1 used about 0.666/0.659 of the GPU
solve time for these fixed unqualified workloads. This is not time to the same
verified accuracy, a qualified-solutions speedup, a mission score improvement,
or evidence for promoting the experimental mode to default.

## The reduction behaves as intended

Cold conditioning has exactly 1,470 zero absolute-value variables. Cold difficult
has 1,629 exact zeros and two positive values. Every reconstructed epigraph is
exactly t=abs(v), every nonzero v has the required endpoint pair duals, and all
removed-pair complementarity products are exactly zero. Maximum epigraph-t
stationarity is zero / 9.09e-13. Maximum stationarity at zero v is 9.09e-13 in
both cases. These results support the implemented exact soft prox and original
dual reconstruction. The analogous completion counters for generic off mode
are descriptive comparisons; endpoint completion is not a requirement for its
approximate epigraph iterates.

Independent standard-library FP64 reconstruction of ten Ruiz passes reproduces
both retained D/R byte hashes from the prior CPU analysis. B and joint
smooth-plus-L1 O match exactly. A fresh twenty-pass power calculation and both
threshold ranges agree with GPU telemetry within relative 1e-13. No omitted
L1 cost, incorrect masked row, scaling-origin mix-up or premature completion of
a supplied seed was found. The unchanged spectral estimate remains a heuristic;
the earlier numerical eigenpair evidence is not a formal general norm bound.

## Remaining convergence defects

For conditioning, the removed epigraph pairs and all cone gates are already
within tolerance. Retained stationarity and equality feasibility remain poor:
absolute maxima are 2.247404 and 6.156597e-4. The signed original objective gap
is 295.727783, decomposed independently as

    f - d = x·r_stationarity - (Ax-b)·y + s·z + (h-Gx-s)·z
          = 277.301757 + 18.426026 + 3.10e-12 + 6.61e-28.

The largest stationarity contribution is retained variable 2314, almost wholly
from the equality transpose product. The worst equality is row 1474; rows with
an absolute-value variable also retain defects up to 6.017582e-4. The reported
dual objective is not a certified lower bound at an unqualified point.

For difficult, the relative gap improves by about 2.90-fold, but normalized
primal residual worsens 31%, block complementarity worsens 32%, and absolute
primal cone violation worsens 5.44-fold. Retained stationarity remains 0.188855
absolute. Original SOC block 233, beginning at QOCO inequality row 10276,
contributes cone violation 4.245487e-5 and complementarity -0.06322956. The
signed gap 0.3311028 is the sum -0.0537460 + 0.4490444 - 0.0641957, plus
negligible slack-reconstruction rounding. Thus the smaller global gap does not
mean that all accuracy gates improved.

The projected-reference distance diagnostic still gives unit-weight imbalance
proxies 5.15313e-5 for conditioning and 0.429600 for difficult. At the L1 cold
conditioning endpoint, scaled primal error to that reference is 0.154 of its
zero-start value, while retained-dual error is 746 times its zero-start value.
This is consistent with investigating primal/dual step balance, but comparison
to one approximate reference does not prove causality, uniqueness or a
convergence bound. The reference is not asserted to be an exact reduced
nonsmooth certificate. Smooth-only O reverses the imbalance across the two
captures, so these data do not justify a universal normalization swap.

The next bounded hypothesis should isolate reciprocal primal/dual step balance
inside the existing reduced primal-dual path, with all original gates retained.
Choose any deployable balancing rule without using the known solution as an
oracle. These findings do not prescribe a weight or justify another GPU sweep
without a separate reviewed plan.

## Identity and reproduction

- Real report SHA256: `baf58f2219dd1764a68b9be36171eab66aa54ed8ca1479bbac4e1311d5089e45`.
- Frozen source: `32efbed13eae47d2c8352771e884a2ad5ff5d07e`.
- Core SHA256: `83487574646fe67fce156c9a2055f448341c99f189ac2c66d856bfcf1b280047`.
- Every raw log, all four capture/point inputs, replay/core metadata, owned
  archived source file and composite replay-source digest were checked.
- Decimal arithmetic uses exact represented FP64 values and 65 digits; source
  bytes are compiled directly, without importing inherited Python bytecode.
- Frozen source locations: `persistent_l1.cuh:13` separates the working view;
  `:22` proves the map; `:228` includes both objective terms; `:480` performs
  the retained dual/soft-primal/completion update; `:532` audits original data;
  `:580` preserves the initial original certificate. Host capacity and mode-off
  handling are in `persistent_l1_host.cuh:46` and `:17`.

From the repository root, run Python 3.12 with only its standard library:

```text
python -B build/performance/l1-real-review-v615/review_real.py --output build/performance/l1-real-review-v615/findings.json
```

The script reads the saved raw run, frozen source archive and prior immutable
map/scaling/balance findings. It launches no GPU or solver work. An initial
syntax error during preparation was corrected before this successful CPU-only
analysis; it did not consume a solver call.
