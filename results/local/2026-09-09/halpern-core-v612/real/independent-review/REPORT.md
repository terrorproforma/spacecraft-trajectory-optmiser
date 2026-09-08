# Independent CPU review of real Halpern replay v612

All ten saved original-coordinate vector sets reproduce the recorded acceptance
verdicts in Decimal arithmetic at 65 digits, starting from the exact represented
FP64 inputs. Four supplied qualified starts remain qualified after zero seeded
updates. All six cold starts remain unqualified after 100,000 updates each.
This review performed no CUDA calls and changed no inputs, logs, production
source or acceptance thresholds.

## Identities and work accounting

The reviewed report is
`build/performance/halpern-real-v612/run/report.json`, SHA-256
`700dbe2056e25a650de1a0535922eea38d2c8ce5f488e49762386cfafa16095e`.
The frozen source is `6fbff324b152b75e6c27e3d91b4dc0835c65ab93`,
core library `49d0eef5702a7dee319ef3b7747ded844bec0f39313300405f9d6f30fc61730e`,
and replay binary `872ef9d6fe84e85aee2431e2d92caa1ad43bf2293ef9fc91bd9ee3ba938717f2`.
The script checks all ten raw-log hashes and their report records, the frozen
owned source files, both snapshot hashes, both point hashes, and the runner hash.

The run used 10 process executions and 14 solve API calls: four separate
one-update diagnostic bootstraps, four zero-update seeded solves, and six
100,000-update cold solves. Total actual updates were **600,004**. All cases used
the explicitly requested 128-block grid and the unchanged original-equation
common gate. Every cold solve ended at its iteration limit; no deadline fired,
no recovery ran, and there was no retry or harness failure.

Every seeded x/y/z value matches the encoded reference and final output bit for
bit. The initial native dual vector independently matches equality/upper dual
ordering and the negative radius-last SOC permutation. The pre-step records
correctly identify zero seeded updates and inherited bootstrap counters. Slack
is independently reconstructed and is not included in the bit-preservation claim.

## Cold results

All quantities below are independently recomputed. Gap and block complementarity
use `max(1, abs(primal_objective), abs(dual_objective))`; primal-cone violation is
absolute. The relative gate is 1e-9 and the absolute cone gate is 1e-8.

| Capture | Mode | Relative primal | Relative stationarity | Relative gap | Max block complementarity | Primal cone violation |
|---|---|---:|---:|---:|---:|---:|
| Conditioning | Off | 8.79008e-5 | 3.48341e-4 | 0.999924 | 6.62898e-4 | 1.55583e-4 |
| Conditioning | Plain | 5.58972e-6 | 2.85960e-5 | 1.37016 | 5.33284e-4 | 3.51555e-5 |
| Conditioning | Adaptive | 4.61453e-5 | 3.10541e-10 | 0.0255402 | 0.405898 | 1.57124e-4 |
| Difficult | Off | 4.15900e-6 | 2.12548e-5 | 0.0112271 | 5.60743e-4 | 7.79867e-6 |
| Difficult | Plain | 9.05871e-6 | 2.55681e-5 | 1.02707 | 7.63789e-4 | 3.73829e-5 |
| Difficult | Adaptive | 1.07728e-5 | 5.12258e-7 | 0.0184402 | 1.38720e-3 | 7.68296e-5 |

Adaptive conditioning greatly improves stationarity, which passes its individual
gate. The other gates still fail. Its relative block metric is 612 times the off
value because the objective normalization scale falls from about 1212.65 to 1.
The absolute worst block product actually improves from about 0.803863 to
0.405898; neither is close to qualifying. The worst adaptive scalar block is row
5492: slack 8.11736e-5 times multiplier 5000.370003 gives 0.405898. Signed block
contributions cancel in their global sum, so the smaller total gap cannot replace
the per-block gate.

The independently verified signed-gap decomposition is
`primal - dual = x·stationarity - (Ax-b)·y + s·z + (h-Gx-s)·z`.
For adaptive conditioning its terms are approximately
`4.26657e-6 + 0.00717671 + 0.01835919 + 3.07e-16 = 0.02554017`.
The original-equation primal/dual defects account for the gap. Slack reconstruction
roundoff does not explain it. On difficult adaptive, the corresponding terms are
`-0.00170647 - 0.99657832 - 0.55935796 + 3.82e-17 = -1.55764274`.
All decompositions close to the recorded high-precision tolerance.

Adaptive difficult improves stationarity but worsens relative primal residual,
gap, primal cone violation and block complementarity versus off. Plain mode has
gap above 1 for both captures; its low/negative objective values are infeasible
iterates, not superior feasible solutions. The reported dual objective expression
is not a certified lower bound while stationarity/cone conditions fail.

All Halpern diagnostics report finite states. Plain mode keeps weight 1 and has
no restart. Adaptive conditioning/difficult make 21/22 restarts, with last restart
at 88400/98200, new metric reference at 88401/98201 and 11600/1800 remaining
inner updates. Their final weights are approximately 1.87267e-6 and 0.00768980.
These counters agree with the frozen implementation. A finite observed metric
does not prove the spectral contraction precondition globally.

## Performance interpretation and scope

The six saved GPU solve timings are 5.346/3.474/3.402 seconds for conditioning
off/plain/adaptive and 5.521/3.495/3.487 seconds for difficult. These are single
fixed-work observations with different algorithmic updates and unqualified
endpoints. They do not establish time to equal verified accuracy or a qualified
throughput improvement. Seeded zero-update checks are not cold-start speedups.

The experiment supports keeping the optional implementation for diagnosis; it
does not support changing the default or claiming cold convergence, full mission
integration, improved trajectory physics, a better competition score or SOTA.
This compares existing dual-first PDHG with primal-first reflected Halpern and
the adaptive restart variant; it does not isolate anchoring alone.

## Reproduce

From repository root with standard-library Python:

```text
python -B build/performance/halpern-real-review-v612/review_real.py --output build/performance/halpern-real-review-v612/findings.json
```

`decimal_audit_source.py` is retained verbatim from the earlier independent
Decimal audit (SHA-256 `b8697eedc3d91ed48849725b4fd8e17dc7515c020f5f3d30f1aa151794a5dbc4`).
It is compiled directly from its recorded source bytes; inherited bytecode is
not imported. The compact findings include exact input/log/source identities,
all ten gate results, seed checks, gap decompositions and metric comparisons.
