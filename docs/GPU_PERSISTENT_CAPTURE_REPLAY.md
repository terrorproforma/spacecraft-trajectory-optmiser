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
The next diagnostic should test the mapped solver at a separately qualified
primal/dual solution, then investigate scaling, restart and primal-dual balance
against this reproducible cold-start baseline. Preserve the PDHCG core and change
one numerical hypothesis at a time; raw iteration throughput cannot substitute
for passing the original-equation checks.

This implements the first diagnostic step of the
[SOTA execution plan](SOTA_EXECUTION_PLAN_2026-09-09.md). Solver improvements must
eventually deliver qualified complete trajectories; the separate asteroid/fleet
search must produce higher verified mission scores.
