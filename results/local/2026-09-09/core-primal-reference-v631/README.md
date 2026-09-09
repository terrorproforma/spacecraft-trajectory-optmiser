# Equality-feasible CPU primal reference, v631

**Neither final point qualifies: 0/2.** Equality projection removes the dominant original equality residual observed in the prior cold experiment, but original gap, stationarity, block complementarity and primal cone membership remain above their unchanged limits.

The two exact inputs are changed stages of **one synthetic fixture**, not independent missions. Both contain 453 stored quadratic entries and zero numerical quadratic coefficients. The source rejects any nonzero Q. All 525 virtual controls, the original lambda 10000 L1 penalty, remaining scalar constraints and 75 SOCs are retained in the equivalent split.

One CPU worker ran early then near-converged, 10,000 updates each: 20,000 updates plus two charged initial projections of zero. No known point was imported. Full worker process time was 4.27961s; both native-solver and GPU call counts are zero. This is a mathematical reference, not a GPU throughput or matched-speedup result.

| Decimal65 final measure | Early | Near-converged |
|---|---:|---:|
| Relative original primal equations |2.07644e-16|2.03279e-16|
| Relative original stationarity |2.44310e-8|2.46566e-8|
| Original gap |0.0914556251|0.0815865927|
| Max normalized block complementarity |9.15807e-4|8.36785e-4|
| Primal cone violation |7.02638e-7|6.49943e-7|
| Original virtual L1 cost |0.09395119|0.08415928|

All eight retained checkpoints agree under the unchanged long-double and Decimal65 auditors. The original limits remain 1e-9 for relative equation/stationarity/gap/max-block measures and 1e-8 for both cone violations. No native status is manufactured: sentinel 0 only satisfies the legacy audit schema; the explicit CPU-reference and original numerical statuses remain separate.

Exact Fraction tests validate adjoints, original L1 units and the represented-coefficient metric bound below 0.9025. The equality Gram has half-bandwidth 19 and structurally proven rank. Row equilibration improves its estimated condition by only about 1.9% (21,245→20,850); it is not a convergence cure.

All 525 virtuals remain nonzero. Pair complementarity sums 0.09395721/0.08416472 dominate the signed gaps. Worst stationarity is original Gamma variable 535; worst primal SOC is node 56, original rows 3248–3251, coupling variables 756–759. Original equality row 526 now has defects below 1.1e-17. No small virtual is snapped to zero and no dual is replaced after the solve.

The detailed mathematical contract, timings and remaining-block diagnosis are in the archived standalone [documentation](../../../../docs/PDHCG_EQUALITY_PRIMAL_REFERENCE.md) and kit REPORT.md. The prior [retained cold comparison](../gpu-core-retained-v630/README.md) and prior negative projection results remain distinct experiments; this package does not claim superiority over their solvers.

The archive contains the exact snapshots, frozen source and prior weighted-reference source, rational CPU tests, coefficient preflight, supervisor/worker logs, all original vectors, both unchanged audit sources, saved audit/decomposition findings and the standalone document. Compiled binaries, caches, keys and optimizer/GPU runtime binaries are excluded.

Run the static verifier using Python's standard library:

~~~text
python audit_package.py --package . --extract FRESH_DIRECTORY
~~~

It checks every archive/auxiliary byte, exact source/input/ready/report pins, all eight vector hashes, terminal counts and saved original-gate bindings. Extraction recreates repository-relative paths. It never executes archived code, reconstructs an optimization result or reruns a numerical audit. The explicit prior numeric reports are preserved, not replaced by this static verification.
