# Persistent GPU replay of actual GTOC12 subproblems

The diagnostic importer now accepts the same captured conic problems used by the
QOCO replay. It runs the production persistent C API, exports original-coordinate
primal/dual vectors and subjects them to independent original-equation checks.
This closes a comparison-tool gap; native GTOC12 SCvx backend integration remains
separate work.

The first real comparison exposes a convergence problem. Both captured GTOC12
instances remain unqualified after 100,000 persistent iterations. Their native
solve stages take approximately 5.01 and 5.14 seconds, while scaling takes about
1.7 milliseconds. GPU QOCO qualifies five of six reference repeats. These are
bounded diagnostics with different repeat/setup amortization, not a statistically
qualified speedup or trajectory-throughput comparison.
[Full vectors, residuals, source and timing scopes](../results/local/2026-09-09/persistent-snapshot-v603/README.md).

The failed persistent objective gaps are approximately 1.00 and 0.0112, well
outside the common 1e-9 gate. The independent Python audit agrees with the native
audit. The one failed QOCO output also remains rejected: vendor status alone
cannot override its 3.48e-9 external objective gap.

Both captured Hessians are numerically zero. They measure the conic LP case of
the primal-dual method, rather than a gain from a nonzero quadratic CG subproblem.
This does not establish how the core performs across the broader trajectory
benchmark families.

## Representation and verification

QOCO's upper-triangular quadratic matrix is mirrored once. Equalities and
nonnegative inequalities map to scalar bounds; SOC rows change from radius-first
to the persistent API's radius-last ordering with the corresponding dual sign.
Shifted inputs recover original primal coordinates, objective, multipliers and
slack. The reader rejects inconsistent transforms and unsupported formats before
CUDA. Six analytic GPU runs qualify, covering shifted problems, lifecycle and
retained state. CPU tests cover conversion and malformed inputs.

The independently implemented Python checker parses the same FP64 values as the
native reader and performs long-double sparse products. Its common diagnostic
gate includes per-block complementarity so cancellation cannot create a false
qualification. These checks do not certify nonlinear trajectories or change any
production numerical/physics acceptance threshold.

## Exact bound-projection experiment

The optional `--fold-singleton-bounds` conversion now folds exactly representable
singleton bounds into the native variable projection. It retains nonexact ratios,
intersects repeated bounds and recovers multipliers in the original equations.
The default generic representation remains unchanged. Seven corrected analytic
GPU solves pass, including problems with no SOC rows, duplicate bounds, shifted
coordinates and fixed bounds with retained state. An initial no-SOC storage-view
failure and its adapter-only correction are preserved in the evidence.

The real comparison does **not** support enabling this option. With the same
adapter executable, unchanged production library, 100,000-iteration limit and
accuracy gate, both folded captures remain unqualified. Scalar row counts fall
from 9,911 to 4,638 and 10,992 to 5,144, but primal residuals worsen and objective
gaps remain approximately 1.00 and 1.01. The generic controls reproduce the first
baseline's residuals. Native and independent audits agree for all four outputs.
[Bound-folding source, failed outcomes and complete logs](../results/local/2026-09-09/persistent-singleton-bounds-v605/README.md).

This is a negative convergence result, despite slightly shorter observed native
iteration-budget timings. It does not improve qualified solutions per second.
The known-point diagnostic below now tests the mapped solver at a separately
qualified primal/dual solution. Preserve the numerical core and change one
numerical hypothesis at a time; raw iteration throughput cannot substitute for
passing the original-equation checks.

## Qualified-point replay

The optional `--initial-point PATH` imports a snapshot-bound original or translated
FP64 primal/dual/slack certificate. CPU checks require its original-equation KKT
and coordinate roundtrip to qualify before CUDA. A diagnostic bootstrap establishes
the report epoch, then a full reset and explicit primal/dual warm start install the
point. Residual-only measurement verifies actual internal seed buffers bit for bit;
its counters explicitly exclude bootstrap work. Production solver code and
tolerances are unchanged. Ten GPU-hidden JSON cases and six corrected analytic
GPU solves pass. An earlier JSON serialization failure and a zero-call busy
attempt are retained.

Eight real runs test both representations for one and 1,000 iterations. All eight
imported certificates pass the common normalized KKT gate and all seed mappings
match exactly. Every native solve ends at its iteration limit; only the difficult
generic one-step output retains the common gate. Native and independent audits
agree for every output. [Exact sources, complete vectors, phases and independent audit](../results/local/2026-09-09/persistent-known-point-v606/README.md).

| Capture | Generic normalized gap after 1 / 1,000 steps | Folded normalized gap after 1 / 1,000 steps |
| --- | --- | --- |
| Conditioning | 6.04613e-7 / 4.21818e-7 | 1.27414e-7 / 1.84037e-6 |
| Difficult | 2.49764e-10 / 1.23070e-7 | 1.59897e-7 / 1.81965e-8 |

The gap threshold remains 1e-9. A valid approximate KKT point need not pass the
native absolute natural-residual test: initial residuals are about 3.81865e-8 and
1.40971e-8. For example, the conditioning point has a scalar slack of 4.26390e-8
and multiplier 3.81865e-8. Their product is only 1.62823e-15, while the natural
map measures their minimum and exceeds 1e-9. SOC natural residuals also exceed
the native threshold. These are distinct accuracy predicates, not evidence of
incorrect import or permission to relax final accuracy.

After one generic conditioning step, the primal objective changes by only about
4.35e-12 while the scalar dual objective contribution moves by about 6.05e-7.
That explains its gap failure. It does not represent a measured loss of asteroid
cargo, prove global divergence, or show that more iterations would recover the
certificate. The actual native main loop advances before its first optimality
check. Establish a common, numerically stable GPU qualification measurement and
compare primal-dual balance/restart behavior against the frozen cold baseline.
Merely accepting a supplied answer is not a cold-solve performance improvement.

## Which numerical method is being measured

The production [persistent iteration](../cpp/cuda/src/persistent_pdhcg.cu) and
[cooperative iteration](../cpp/cuda/src/cooperative_pdhg.cuh) use an explicit
quadratic-gradient primal update. Their recovery CGLS routines operate on
constraint maps; they are not the main quadratic proximal solve. Linking the
upstream library does not make this path an execution of the full upstream method.

Upstream PDHCG-CQP solves a conic quadratic proximal subproblem with projected
gradient inner iterations or an applicable direct weighted projection. CG is
associated with the original QP predecessor; it is not a universal replacement
for a cone-constrained inner solve. [Primary algorithm and implementation description](https://arxiv.org/html/2608.09159v1).
For these zero-Hessian captures the linear-objective primal proximal map is already
the explicit step (with the applicable box projection). Missing quadratic inner
iterations therefore do not by themselves explain these two failures. General
nonzero-Hessian comparisons must distinguish the explicit baseline from the full
upstream proximal scheme and include actual quadratic trajectory captures.

Two tempting changes were assessed before consuming another GPU sweep. Scaling
the objective by 2^-13 changes the effective original-coordinate primal/dual
steps only about 2% because this core already balances objective magnitude.
Adding an equality penalty preserves feasible optima in exact arithmetic but
would enlarge the stored full quadratic pattern about 46.6-fold and introduces
FP64 cancellation risks. It would still use an explicit quadratic gradient here.
Neither change has an established speed or convergence benefit; neither is
enabled. The publication retains the CPU balance evidence and numerical assessment.

This implements the first diagnostic step of the
[SOTA execution plan](SOTA_EXECUTION_PLAN_2026-09-09.md). Solver improvements must
eventually deliver qualified complete trajectories; the separate asteroid/fleet
search must produce higher verified mission scores.
