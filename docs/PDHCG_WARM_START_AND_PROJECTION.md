# PDHCG warm transfer and equality projection — v623

The new warm-start diagnostic measures actual work between changed SCvx
subproblems. All four GPU executions failed the original accuracy gate after
10,000 updates. A separate CPU equality-projection experiment also failed.
Neither experiment increases the verified fleet score or establishes a solver
speedup. These failures narrow the next convergence work while preserving the
PDHCG-inspired core and the original objectives and constraints.

## Warm transfer through the unchanged GPU core

The old `persistent_snapshot_replay --initial-point` deliberately required the
point to already qualify for its target snapshot. It could test importing a
certificate, but could not test convergence from a useful predecessor iterate.
The additive `--warm-start-source PREDECESSOR` option now validates the supplied
point against that predecessor and transfers its original x/y/z bits unchanged
to a fresh successor workspace. Successor slack is reconstructed from its own
`h-Gx`. Its initial audit may fail; final qualification remains unchanged.
The existing known-qualified-target mode remains strict.

The adapter rejects shifted captures, incompatible sparse topology or cones,
unchanged coefficients, folded bounds, missing/unqualified predecessor points,
nonfinite arithmetic, and incompatible execution options. It requires a common
accuracy policy, positive cooperative block count, one fresh workspace, and no
optional weighting, mass elimination or Halpern. Common PDHG and exact-L1 PDHG
use the same transferred point and qualification rules.

The two fixed pairs come from consecutive accepted **synthetic GTOC12 fixture**
SCvx stages: outer 1→2 and 2→3. Their seeds are saved independent QOCO replays of
the exact predecessor problems, not the actual historical SCvx iterate. Both
seeds qualify for their predecessor and fail their successor. These are not
mission-leg or complete-trajectory convergence measurements. Original
coefficients and objectives are unchanged; the shifted pair uses a view of the
verbatim stored original coefficients with translation metadata removed.

Each successor has 1,886 variables, 542 equalities, 3,324 conic rows, 75 SOCs and
525 exact L1 epigraph pairs. The fixed protocol uses 128 blocks, 10,000 updates,
a 10-second solve deadline, and a separate 35-second process watchdog. Four
primary solves plus four explicitly counted one-update diagnostic bootstraps
perform **40,004 updates through eight solve API calls**. No recovery or new
predecessor solve occurs. A shared inherited GPU lock excludes cooperating jobs.

| Transition | Representation | Solve wall time | Final relative primal | Final relative dual | Final gap | Qualified |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Early | Common PDHG | 0.5394 s | 6.5001e-7 | 2.4855e-6 | 0.222878 | No |
| Early | Exact L1 | 0.3657 s | 1.2147e-6 | 2.2717e-6 | 0.591380 | No |
| Near-converged | Common PDHG | 0.5471 s | 2.6496e-10 | 3.4594e-10 | 8.1330e-5 | No |
| Near-converged | Exact L1 | 0.3622 s | 2.4712e-10 | 5.0517e-10 | 7.4532e-5 | No |

These are single RTX 5090 observations at a fixed unsuccessful update cap,
not time-to-accuracy benchmarks or a cold-to-warm speedup comparison. The
existing core is unchanged: SHA-256
`6b32c2b5c4f87cd8990ee95816b00c0cb5dc863c7225cb244a17993ef344d8b1`.
Only the replay executable and CPU conversion test were rebuilt. Its frozen
manifest separates adapter source identity from the reused core's identity.

An independent 65-digit Decimal audit checks both predecessor certificates,
all four transferred seeds, and all four final vectors. It agrees with every
native common-gate verdict and verifies the exact primal/dual transfer.
Qualification requires relative primal, stationarity, gap and maximum block
complementarity ≤1e-9, absolute primal/dual cone defects ≤1e-8, finite values,
and native optimal termination for a qualified solver result.

The near-converged exact-L1 result fails **only the global objective gap**.
Its gap decomposition is dominated by `x·stationarity` (−7.45557e-5), compared
with approximately +2.3993e-8 from equality residuals and +1.748e-12 from
complementarity. Improving dynamics accuracy alone would miss that failure.
The early transitions worsen their initial gaps; warm transfer alone is not a
convergence remedy.

## CPU test of projection onto every dynamics equality

The two saved cold captures (conditioning and difficult) admit an exact chronological permutation of
`E T Eᵀ` with scalar half-bandwidth 19. All dynamics rows have independent virtual
columns; exact rational elimination establishes independence of the remaining
17 boundary rows. Three initial-velocity auxiliary variables in the difficult
capture remain in the system. The first, overly restrictive singleton-boundary
proof attempt failed and is retained in the evidence.

The tested split removes only exact L1 epigraphs, places `Ex=b` in the primal
proximal operator, and handles virtual L1 penalties through clipped duals.
Original cone constraints, affine offsets and penalty coefficients remain.
An upfront CPU banded Cholesky factor supplies one equality projection per
iteration. This is a SciPy numerical reference, not a GPU implementation.

The fixed coefficient-only diagonal steps use the absolute sums of `[G;R]`,
with equal steps throughout each SOC and theta=0.95. Exact rational checking
bounds the squared scaled operator norm by 0.9025. A small rational oracle
checks the projection, original equality multiplier and L1 dual reconstruction;
SOC projection and clipping branches also pass.

Exactly two reduced zero starts run for 10,000 iterations each, with full
original vectors saved at 0/100/1,000/10,000. Long-double and independent
Decimal65 final audits agree: **zero qualified**. Final gaps are approximately
0.999987 and 0.999679. Equality residuals reach roundoff, but virtual L1 costs
remain approximately 2,164.726 and 1,081.397. This rejects the proposed fixed
metric as a useful convergence improvement at this budget. No CUDA port or
production policy change follows from this result.
These costs are normalized convex-subproblem objective units, not physical
mass or kilograms.

A second, separately bounded CPU test includes the penalty coefficients in the
operator: `[G;diag(lambda)R]`, with the L1 dual expressed in unit-box coordinates.
Its steps follow the same coefficient-only absolute-sum rule; no optimum or
parameter sweep is used. This changes the virtual primal step to 9.5e-5 and
the effective original L1 dual step to 9,500. The exact operator bound still
passes; the equality Gram retains bandwidth 19 and its condition rises to
approximately 20,700. Thirty rational identities check the coordinate change.

This correction reduces the conditioning capture's virtual L1 cost from
approximately 2,164.726 to 0.020065, but **both additional 10,000-iteration CPU
runs still fail**. Final gaps are 0.0184192 and 0.681354; the difficult case also
has a 0.035925 cone violation. The cold CPU work across both policies is four
solves and 40,000 updates; four additional one-step algebra checks are separate.
These measurements support the
penalty-scaling diagnosis, but establish no qualified-throughput improvement.
Both failed policies and their full original vectors are retained.

## Native mission refinement control

A separate bounded campaign uses the published native GPU QOCO/SCvx path and
the current v767 fleet. It permits one unchanged ship-23 control, followed by
at most two positive-gain fixed-cargo requests, with at most 53 native leg
attempts. The preserved settings allow 40 outer iterations plus four polishing
iterations per leg; a shared inherited GPU lock and a 620-second process
watchdog bound execution. Both full-fleet checkers must accept a new candidate
before it can count as a score gain.

Fresh independent CPU and original official checks of the unchanged incumbent
pass, reproducing **12,843.555695585188 weighted kg** and
**14,044.353182751598 raw kg** exactly. The three completion requests all fail
the ordinary final-mass estimate, but the admission queue correctly preserves
them for bounded refinement. The two new requests have forecast weighted gains
of 0.123515 and 0.372079 kg; these are not verified gains.

The cold control fails on its first leg, Earth to asteroid 30805 from MJD 64403
to 64973. Native SCvx returns an iteration limit after 41 outer iterations and
22 accepted steps. Its final normalized dynamics defect is 1.05910e-5 against
the unchanged 5e-9 threshold. The separate CUDA trajectory certificate rejects
the result: propagated terminal position error is **10,193.0196 km**, velocity
error **0.901866 m/s**. Thrust and solar-distance checks pass. The same archived
leg passes the fresh CPU baseline with approximately **0.045809 km** position
error. The failed cold reconstruction does not establish mission infeasibility.

There are 41 conic reports and 3,104 IPM iterations; 11 reports are unqualified.
The final conic report itself qualifies, so the failure cannot be explained by
the final QOCO status alone. The nonlinear outer loop still has unresolved
virtual controls and dynamics defects. Native leg dispatch takes 12.366 seconds;
the complete worker takes 35.390 seconds including baseline checks and output.

The run stops normally after **one native leg attempt, zero certified control
routes and zero fresh-candidate refinements**. Neither positive request is
claimed, no replacement fleet is emitted, and the incumbent score is unchanged.
The next mission prerequisite is reliable native refinement initialized from
the known feasible trajectory, with the same independent physical checks.

Two diagnostic labels in the frozen worker require explicit interpretation.
`ordinary_proxy_survivors=3` counted returned result tuples, not successful
plans: the actual ordinary survivor count is **zero**. Likewise, the legacy
`rk4_vs_dop853_km` field in this CUDA certificate records a propagated-position
discrepancy; this run did not perform a fresh DOP853 comparison for the failed
control. Raw evidence remains unchanged and the independent review records
these corrections. Scheduler `feasible` also does not supersede
`certified=false`.

## Consequence for the roadmap

Keep the numerical core, but judge the next intervention by original objective
gap and independently certified trajectory output. For the near-converged warm
case, aggregate stationarity is now isolated; for the projection split, penalty
handling remains ineffective. Any next metric or correction must have an
explicit mathematical contract and a bounded comparison on these saved inputs.

Mission search and its GPU QOCO refinements continue separately. Their verified
score is recorded in [GTOC12 progress](GTOC12_PROGRESS.md); those gains must not
be attributed to PDHCG convergence. The broader decision rules remain in the
[SOTA execution plan](SOTA_EXECUTION_PLAN_2026-09-09.md).
