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

## Pinned upstream comparator and optional GPU stopping policy

`upstream_snapshot_replay` now invokes the pinned upstream C API on the same
captured matrices and exports original-coordinate vectors. Both real supplied
known points qualify at iteration zero. Both cold starts remain unqualified after
100,000 iterations. Their zero Hessians retain off-diagonal structural zeros,
which select upstream's sparse-Q inner path; this run did not exercise an explicit
zero-Q fast path. All eight diagnostic outcomes and independent audits are
[retained](../results/local/2026-09-09/upstream-identical-capture-v608/README.md).

The persistent replay now accepts `--common-kkt-stop`. It enables an additive
diagnostic C API that evaluates original-equation primal/dual residuals, objective
gap, cone membership and per-block complementarity on the GPU, before any update
and at subsequent residual checks. It uses compensated FP64 products and retained
row-gather maps, with the existing independent CPU audit still applied to exported
vectors. The fixed thresholds remain 1e-9 for normalized residuals, gap and block
complementarity, and 1e-8 for cone violations. Native natural-residual telemetry is
retained separately.

The option currently supports unshifted generic captures with free primal
variables, an equality prefix, upper-only scalar inequalities and contiguous
standard affine SOCs. Shifted or folded imports are rejected by the replay.
Numerical updates require disabling the policy and revalidating its domain before
reenabling it. This is a diagnostic policy, not yet a native GTOC12 backend or a
certificate of nonlinear trajectory physics. Its convexity assumptions remain
those of the original snapshot audit.

Cancellation takes precedence over acceptance; reset/reseed invalidates old
certificates, and this policy disables the legacy recovery acceptance path. The
new kernels are separate template instantiations. Default single-block and
cooperative register counts remain 148 and 80, respectively; the common-policy
variants use 204 and 96. Preserving those counts does not by itself prove identical
latency. Fourteen focused GPU calls in total across both strategies pass, covering a
non-diagonal quadratic, exact seeds, normalization traps, complementarity,
cancellation, nonfinite data and lifecycle changes.

An explicit `--execution-blocks 0` chooses the single-block iteration and `2`
chooses two cooperative blocks. Omitting the flag retains automatic selection.
The accepted requested count is recorded in replay metadata. Four real
known-point runs across these two strategies stop at zero iterations, preserve
the installed primal/dual bits and pass the unchanged independent gate. Their
four separately reported one-step bootstraps only enable the older residual-only
diagnostic epoch; they are not hidden work or seeded optimisation iterations.
These are correctness results, not a cold-start speedup.

The four matched cold calls force the single-block path in both policies and
remain unqualified at their 60-second deadlines, before the 100,000-iteration
cap. They are not a measurement of automatic multi-block performance. The option
also adds diagnostic workspace: the recorded peaks are about 19 MB versus about
2 MB without it on these inputs. Keep it opt-in while the next convergence and
integration experiments establish usefulness and cost. [Exact sources, fourteen
focused tests, all real vectors and deadline outcomes](../results/local/2026-09-09/persistent-common-kkt-v609/README.md).

## Automatic-grid and exact-linear reference follow-ups

The four v610 cold calls use automatic grid selection and finish all 100,000
iterations. Natural/common stopping takes 5.0354/5.4137 seconds on conditioning
and 5.0787/5.4702 seconds on difficult; every result remains unqualified. The two
policies follow the same iterate trajectory to FP64 rounding differences. The
executable records automatic selection as a null requested block count and does
not export the effective grid size. These measurements supersede neither the
forced-single-block experiment nor its deadline evidence. The first completed
call's parser failure was repaired by reusing its saved output and executing only
the remaining three calls. [All four outcomes and CPU analysis](../results/local/2026-09-09/automatic-cold-v610/README.md).

The upstream comparator now also exposes `--omit-zero-quadratic`. It requires
every quadratic coefficient to be exactly zero before passing a null descriptor,
preserving the original mathematical problem and audit coordinates. The default
still passes the original full symmetric CSC. Five checks with CUDA hidden cover
valid zero-Q inputs, rejection of nonzero-Q inputs and duplicate options.

With this option, two supplied points qualify at zero updates and both cold calls
still fail after 100,000 iterations. Native C API wall time falls to 10.3160 and
8.7967 seconds in these single diagnostic samples, compared with 37.3320 and
34.8176 seconds for v608's general quadratic dispatch. This removes avoidable
reference work; it establishes no qualified cold throughput or mission gain.
Upstream's reported inner counter includes one unconditional increment per outer
update even on its direct linear path: the reported 100,000 is not 100,000 BB
iterations. The original observation failure and bounded continuation preserve
all four actual calls. [Exact-zero dispatch, complete vectors and accounting](../results/local/2026-09-09/upstream-zero-quadratic-v613/README.md).

The independent CPU analysis attributes the failed gaps to actual stationarity,
feasibility and complementarity errors. Numerical spectral estimates put the
current zero-Q stability products at about 0.898 and 0.908, despite modest
underestimation by the twenty-step power method. This is evidence against a
step-size violation on these two captures, not a general spectral certificate.
The restarted Halpern experiment is reported below. Exact elimination of the
detected L1 epigraph pairs remains a separate hypothesis requiring original
primal/dual recovery and equivalence checks.

## Optional Halpern and restart comparison

The replay now accepts `--halpern off|plain|adaptive`. The two experimental modes
require exactly zero quadratic coefficients, the common-KKT policy and an
explicit positive cooperative block count. Each relevant kernel's occupancy
limit is checked before solving; there is no silent fallback. The default remains
off. Existing API struct layouts and the compiled resource counts of the previous
solver, initialization, scaling and recovery kernels are preserved.

The new path forms a primal-first proximal map, reflects it, and blends the
working state with a retained anchor. It exports and checks the actual proximal
point. The adaptive variant additionally restarts and adjusts reciprocal
primal/dual weights on the GPU. This compares algorithm packages against the
existing dual-first default; it does not isolate anchoring from update order.
Checkpoint/restore is explicitly unsupported while enabled. Disabling restores
the default primal history. Cancellation and numerical failure remain authoritative.
See the packaged design for equations, the upstream-derived restart rules, their
documented residual-guard variant and the unproven general spectral precondition.

The fourteen bounded tiny GPU calls pass their expected outcomes, including
scalar/SOC proximal oracles, restart counters, default-mode transitions,
nonfinite input and cancellation. Four supplied real-capture points pass the
common gate at zero iterations with unchanged primal/dual bits, following four
separately counted one-step bootstraps.

Six cold calls use the same v612d binary, 128 blocks, 100,000-iteration cap and
30-second deadline. All reach the iteration cap and remain unqualified:

| Capture | Mode | Native solve seconds | Original normalized gap | Qualified |
| --- | --- | ---: | ---: | --- |
| conditioning | off | 5.345601 | 0.999924306 | No |
| conditioning | plain | 3.474233 | 1.370163221 | No |
| conditioning | adaptive | 3.401526 | 0.025540171 | No |
| difficult | off | 5.520818 | 0.011227050 | No |
| difficult | plain | 3.495427 | 1.027070844 | No |
| difficult | adaptive | 3.486708 | 0.018440177 | No |

Adaptive runs perform 21/22 restarts. The smaller conditioning global gap does
not mean uniform improvement: its normalized worst-block complementarity rises
from about 0.000663 to 0.405898. On difficult, adaptive primal residual and gap
both worsen. These single samples measure unqualified iteration cost, not time
to a qualified solution. Retain both modes as explicit diagnostics; neither earns
default selection, a fleet improvement, or a SOTA claim.
[All source attempts, CPU checks, tiny outcomes and real vectors](../results/local/2026-09-09/halpern-core-v612/README.md).
