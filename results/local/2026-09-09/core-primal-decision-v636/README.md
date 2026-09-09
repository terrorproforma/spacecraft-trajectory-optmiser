# Primal convergence decision — v636

This package records a saved-data comparison, not a new solve. The evidence archive includes all directly inspected inputs, vectors, audit reports and source pins. No archived code runs during verification.

# Decision: retain the evidence; do not run fixed-primal correction

The v635 fixed primal points cost about **3.631% more** than already saved,
qualified QOCO points on exactly the same original inputs. A dual correction
cannot change c^T x. Even if it passed the existing approximate KKT tests, that
would not establish equal objective accuracy or a solver performance advantage.
The proposed two QR evaluations were canceled before import conversion, native
library loading or any factorization. No auxiliary conic solve replaces them.

The saved-data check is `inspect_saved.py`; its completed report is
`saved-support-a.json`, SHA-256
`1ea38dc836b1935a877ffdc43e653901799eb45a70b09d770f30407861d60ec9`.
It binds original snapshots, both v635 final vectors, unchanged v629 source, saved
QOCO readback vectors and their existing audit reports. It uses the unchanged
original snapshot parser, Decimal65 arithmetic and exact Fraction support/face
checks. No optimizer, numerical factorization, new proximal map or GPU was called.

| Saved input | v635 c^T x | Qualified saved QOCO c^T x | v635 excess |
|---|---:|---:|---:|
| early | 0.022669895273641100 | 0.021875586368818126 | 0.000794308904822974 (3.631029%) |
| near-converged | 0.022669874567925268 | 0.021875559685933571 | 0.000794314881991697 (3.631061%) |

The early comparison is v630 QOCO retained trial 1, native status 1. The near
comparison is v629 QOCO's cold exact-input solve, native status 2
(`QOCO_SOLVED_INACCURATE`). Both passed the original long-double and Decimal65
numeric gates and their backend status policy. Their vectors are comparison
evidence only, not inputs to any correction, metric, anchor or algorithm choice.
These are approximate numerical certificates, not proofs of exact optimum.

The suspected zero-equality-support obstruction was also tested and falsified.
Gamma variable 611 couples to mass equality 139 with coefficient
0.00105724176961551324 in both inputs, and has six additional tiny nonzeros.
Those tiny entries are real stored coefficients and were not discarded. Thus
changing equality duals can change its roughly 2.070e-4 residual. All 525 immutable
zero-A/nonselector coordinates are epigraphs with residual exactly zero. The old
restricted correction is not ruled out by this simple support argument.

An initial inspection asserted the expected obstruction instead of reporting
both possible outcomes. That assertion failed before output; its exact source
and explanation remain under `inspection-attempt-a`. The corrected check reports
zero blocked coordinates. This was a failed analytic hypothesis, not a failed
solver run.

All 525 virtual/L1 pairs do satisfy the exact zero-face and strict multiplier
conditions used by v629/v630. That establishes compatibility, not useful dual
correctability. The native v629 fixed infinity test is still
1e-9*(1+||c||inf)=1.0001e-5, alongside strict L2 improvement, absolute gap and all
original KKT tests. None was loosened or replaced by objective normalization.

The stored SOC vectors cannot be labeled exact boundary faces: exact represented
slack classification is 28 inside/47 outside for early and 20 inside/55 outside
for near; duals are 57 inside/18 outside in both. Their tiny distances still pass
the existing cone tolerances. These facts do not authorize freezing a presumed
boundary ray, clipping a multiplier or releasing constraints by tolerance.

The next question is actual primal convergence with the now-working joint prox.
`NEXT_PRIMAL.md` proposes one fixed-metric reflected-Halpern/restart experiment.
It changes primal iterates, keeps the original objective and all gates, and keeps
the qualified same-input objective comparison visible. Implementation and captured
execution remain pending review. Existing production code, v635 evidence and the
v629 diagnostic remain unchanged.

The next-primal design in the archive records the proposal at this decision point. Any later implementation or measured result has a separate evidence package.

Run `python verify.py --index-sha256 <published index hash>` from this directory to check the complete package. This checks preservation and reported provenance; it does not repeat an optimizer or physics certificate.
