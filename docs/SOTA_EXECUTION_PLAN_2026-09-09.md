# SOTA execution plan — 9 September 2026

Keep the PDHCG-inspired numerical core and the GPU-native trajectory pipeline.
The next work must establish its advantage at independently verified accuracy
and improve complete GTOC12 missions. Faster candidate arithmetic and the GPU
integration are useful intermediate results; neither establishes either outcome.

The current decision is to execute the remaining roadmap, preserve PDHCG as the
core under development, and use targeted literature checks to resolve measured
failures. Judge each tranche by complete time to verified accuracy and by verified
mission score. The latest evidence supports broader route construction, reliable
changed-route refinement and removal of measured host overhead. It does not yet
establish a competitive PDHCG mission backend.

## Current checkpoint and next work

The latest locally checked fleet, v628, scores **12,992.407741 weighted kg** and
returns **14,271.485284 raw kg**, or **620.499360 raw kg per ship**, with 23 ships
and 200 asteroids. It combines the H100 v799 fleet with one newly certified local
ship-8 replacement; both complete-fleet checkers accept the exact combined file.
The other 22 ship sections are unchanged. This is a local composition, not an
H100 rerun of the new controller.
[Exact Result and checker evidence](../results/local/2026-09-09/current-fleet-composition-v628/README.md).

The [v799 route-generation campaign](GPU_FLEET_REGENERATION.md) gained 148.479966
weighted kg by generating better routes. The subsequent
[endpoint-merit correction](SCVX_ENDPOINT_MERIT.md) adds 0.372079 weighted kg:
the changed transfer now certifies in four accepted iterations instead of
rejecting every step. This small gain validates a repair needed for further
search. It is not a rate of progress that would close the overall score gap.

The v799 fleet-selection pass exhausts its 44 certified route choices. Increasing
that selection budget alone cannot improve the optimum within that pool. New
high-quality route columns and coupled itinerary changes are therefore the next
score opportunity. Its generator also retains each ship's Earth-leg seed and
excludes other incumbent ships' asteroids; coordinated replacements must expand
those restrictions while preserving fleet feasibility.

| Priority | Concrete deliverable | Decision that advances the goal |
| --- | --- | --- |
| 1. Convert reliability into better missions | Apply the endpoint fix to current-fleet route regeneration; jointly consider deployment/collection timing, changed asteroid sets and compatible cross-ship replacements. Keep fixed-cargo incumbent controls and both full-fleet checks. | Compare verified best score versus total elapsed time from the same incumbent. Count unique candidates, rejected routes and all refinement time. Retain improvements to weighted score while meeting the raw-mass ship rule. |
| 2. Remove the remaining host bottlenecks | Retain collection-DP workspaces across tours, then move completion packing and beam expansion/control into C++/CUDA. H100 v799 creates 25,112 DP workspaces; completion packing takes 48.280 s against 0.528 s in its kernel. | Reproduce route decisions and qualified scores, then reduce complete search time. More kernel evaluations are useful only if they yield more certified competitive routes within the same budget. |
| 3. Make PDHCG qualify reliably | Resolve the isolated GPU correction's numerical-quality rejection, with a mathematically justified treatment of isolated coordinates; profile its factorization/setup costs before integrating it. Retest representative cold and changed warm captures, including nonzero-Q cases. | Every original acceptance gate must pass. The present saved point remains rejected and its native status is iteration-limited. No retrospective acceptance or relaxed physics gates. |
| 4. Demonstrate the core's advantage | Connect a qualified PDHCG path to native GTOC12 SCvx, including retained starts and explicit recovery. Compare pure PDHCG, QOCO and any declared hybrid on identical inputs; then compare complete trajectory systems. | Measure median/p95 total latency, certified legs/s, reliability, objective and memory. Count setup, transfers, failed attempts and correction. Use the existing repeated-run and claim rules below. |
| 5. Spend demonstrated throughput on broader search | Batch independent qualified trajectories, use interval-structured operators where profiling supports them, and expand diverse route pools under equal compute budgets. | Better score-time curves and stable gains on held-out missions, followed by comparison against published fleets rescored with the same bonus table. |

The new GPU dual-QR diagnostic passes the saved point's original KKT numerical
checks but fails its own quality rule, and its evaluation takes 195.520 ms plus
218.238 ms of creation. It is a useful numerical finding, not a qualified solve
or demonstrated speedup. Further isolated experiments need to resolve that
specific obstruction and then reach the comparative suite; repeated tweaks to
one saved point cannot establish SOTA.
[Measured correction and rejection](PDHCG_CONSTRAINED_DUAL_POLISH.md).

The focused reading list already below remains appropriate: upstream conic
PDHCG/PDHCG-II for convergence and stopping, and TheAntipodes for route subsets,
beam search, low-thrust refinement and fleet selection. These sources were
checked again on 9 September. Each proposed change needs a specific missing
capability and an ablation. A new broad survey is lower priority than applying
and testing this existing knowledge.

Historical checkpoints and their frozen settings follow. Their earlier scores
are evidence for those experiments, not the current acceptance baseline.

The [v623 warm-transfer experiment](PDHCG_WARM_START_AND_PROJECTION.md)
tests changed adjacent SCvx inputs with a qualified predecessor iterate.
All four GPU results remain unqualified at the fixed 10,000-update cap; the
near-converged exact-L1 case isolates the remaining global-gap failure.
Separate CPU equality-projection tests on the two cold captures also remain
unqualified. This is new convergence evidence, not an improvement in verified throughput.
The separate native mission control also fails to regenerate a known feasible
departure leg from a cold start, despite a qualified final conic subproblem.
It stops before the two candidate refinements. This makes reliable initialization
from the verified trajectory the next mission gate, with all physical tolerances
retained.

The [v624 native GPU initializer](GPU_VERIFIED_TRAJECTORY_INITIALIZATION.md)
passes that first-leg control: GPU propagation reproduces the archived schedule
within 5.2 mm of the independent CPU reference, and native refinement converges
in one accepted SCvx iteration with a 57 m arrival error. The observed native
call takes 0.416 s. This is successful regeneration of a known feasible trajectory,
not a paired speedup, fleet score gain or PDHCG convergence result. The next
mission checks are the complete archived route with consistent mass carried
between legs and H100 validation, followed by compatible changed-route seeds.

The [v625 constrained dual correction](PDHCG_CONSTRAINED_DUAL_POLISH.md) passes
all original numerical gates for the saved near-converged PDHCG iterate while
keeping its primal variables unchanged. One auxiliary CPU solve takes 11.618 ms;
it is a hybrid diagnostic and preserves PDHCG's iteration-limit status. The final
gap has only 9.36e-17 of margin to its limit. This supports a focused GPU correction
experiment and broader fixed-capture validation, not a convergence or SOTA claim.
The [complete v625 seeded-route regression](GPU_FIXED_ROUTE_INITIALIZATION.md)
passes all 19 native leg solves and both waits in 3.876 s locally. Both full-fleet
checkers accept the replacement at unchanged cargo/score; the full worker takes
25.193 s. A fresh H100 build and the first-leg control pass as well. The final
73-test CPU suite also covers cumulative route telemetry. These reliability
gates are now met for the archived control; compatible changed-route refinement
is the next mission experiment. The complete pipeline and PDHCG backend
integration still have the gaps described below.

This plan began with an audit of published commit
`3091c716714c8bdec364d54c5e7357f2b5d85730`. The frozen v622 mission baseline below
includes the separately published v733 fleet-exchange result at `2ccb93c1`.
Later mission checkpoints are recorded in [GTOC12 progress](GTOC12_PROGRESS.md).
Historical experiments retain their original inputs and scores; each new score
campaign must use the latest independently verified incumbent as its acceptance
baseline. The preregistered claim thresholds and physics gates are unchanged.

## Two outcomes to pursue together

| Outcome | Current evidence | Required evidence |
| --- | --- | --- |
| Competitive trajectory solver | Persistent CUDA operators, multi-block PDHG, native dynamics/assembly and GPU IPM refinement exist. A sustained PDHCG advantage over qualified competitors is not established. | Time to the same qualified solution, independently certified trajectories/second, reliability, objective, and memory versus applicable strong baselines. |
| Best GTOC12 mission | Frozen v733 baseline: 23 ships, 14,043.750856 raw kg, 12,842.970672 fixed-bonus weighted kg, 610.597863 raw kg/ship. That checkpoint gained 32.834719 weighted kg (0.256318%). | Higher independently verified fleet score under a pinned bonus table and common mission rules, with score versus total search time and GPU-hours. |

The [fleet-exchange stage](GPU_FLEET_EXCHANGES.md) makes two profitable swaps
from the same 2,492-column pool. Both complete-fleet verifiers pass on RTX 5090
and H100; all 33 fresh native leg refinements converge on each GPU. It sacrifices
8.104038 raw kg to increase the weighted objective. This is a verified mission
gain from GPU fleet selection and QOCO refinement; it is not evidence that the
PDHCG core has qualified the difficult cold captures below. The historical
ship-23 admission controls remain pinned to the earlier v595/v616 fleet.

The recent GTOC12 trajectory refinements use GPU QOCO. The current
[GTOC12 settings](../src/spacepdhcg/gtoc12/low_thrust.py) accept Clarabel/QOCO as
conic backends and require QOCO for the CUDA outer loop. Our persistent PDHCG
backend is not yet wired into that native GTOC12 path. Those refinements demonstrate
QOCO's capabilities, not a PDHCG speed or convergence advantage. The current
fleet also retains historical ships; it is not a complete fleet regenerated by
a fully GPU-controlled solver. See [the progress evidence](GTOC12_PROGRESS.md).

The destination remains first overall, including published post-competition
solutions. The [current comparison](../README.md#gtoc12-score-versus-the-published-leaderboard)
lists 22,532.672 weighted kg for the competition winner and 24,474.16 for the
strongest post-competition result listed by the ESA portal. Historical scores
used their submission coefficients. Before claiming an overall lead, obtain and
re-evaluate available comparator solutions with the same pinned bonus table and
checker; do not infer it from raw kg/ship or a mixture of scoring tables. These
are finite benchmark targets, not proof of a global optimum.

## What remains from the original roadmap

The [September 5 review](PERFORMANCE_ARCHITECTURE_REVIEW_2026-09-05.md) correctly
identified architectural headroom. Its first tranche is substantially implemented:
parallel preamble/reductions, cooperative multi-block PDHG, retained QOCO conversion
and workspaces, and native GTOC12 SCvx execution. Repeating the old assertion that
every large PDHG solve uses one block would be inaccurate.

| Remaining issue | Source or evidence | Next experiment |
| --- | --- | --- |
| Forward sparse products scatter with atomics | [Cooperative PDHG operator](../cpp/cuda/src/cooperative_pdhg.cuh) | Compare retained row-gather multiplication, then interval-structured operators and compatible preconditioning. |
| Recovery starts after a long fixed prefix and remains one block | [Cooperative stopping policy](../cpp/cuda/src/cooperative_pdhg.cuh), [recovery launch](../cpp/cuda/src/persistent_pdhcg.cu) | Progress-triggered restart/recovery and parallel recovery on difficult captured problems, with all failed attempts timed. |
| Some refinements remain unqualified; convergence can vary | [225-leg replay](GPU_SCALED_WORKSPACE_REUSE.md), [six return replays](../results/local/2026-09-09/return-repeat-v598/) | Isolate conic solve variability, outer acceptance policy and true infeasibility before changing parameters. |
| Batched proxies still lead to individual trajectory solves | [Native refinement](../cpp/cuda/src/gtoc12_scvx.cu), [pipeline](../src/spacepdhcg/gtoc12/pipeline.py) | Reuse compatible outer workspaces, then batch independent refinements; measure certified legs/second. |
| Search and fleet control still include CPU numerical work | [Beam expansion](../src/spacepdhcg/gtoc12/search.py), [epoch moves](../src/spacepdhcg/gtoc12/jointopt.py), [fleet master](../src/spacepdhcg/gtoc12/cooperative.py) | Broaden productive search neighborhoods and move their measured recurring work into retained C++/CUDA execution. |

Generic device-SCvx timer/fingerprint synchronizations still exist. GTOC12 has
its own outer CUDA graph already; improvements to one path must not be described
as missing features of every path. Fully GPU-native completion includes search,
selection and acceptance arithmetic as well as the inner solver. File I/O and
independent CPU certification remain explicit boundaries, and their time remains
visible in the complete result-production time.

## Execution order and decision rules

### 1. Restore a direct test of the PDHCG core

Start with a small diagnostic set of committed, reproducible easy and difficult
trajectory problems. Compare identical assembled problems where interfaces
support them; explicitly expose missing capture/import adapters. Use upstream
PDHCG, our persistent core, GPU QOCO and CPU Clarabel references. Add CuClarabel
and applicable first-order trajectory methods for the subsequent comparative
campaign. Unsupported backends are recorded, not silently replaced.

The initial tooling gap was concrete. [The existing PD3 comparator](../scripts/gpu/diagnose_g3_pd3.py)
audits one canonical dump with CPU Clarabel, upstream PDHCG and GPU QOCO;
`--persistent-output` imports an earlier persistent-solver log rather than
replaying that dump through our core. The native integration harness can emit
the built-in 57-variable, two-interval fixture with `--dump-pd3` and solve that
same generated fixture with `--tight-pd3-100k`, including primal/dual output.
Use this narrow comparison first, with pinned source and verified fixture
identity. Its current backend tolerance settings differ, so use it for numerical
diagnosis first and align requested accuracy and common external qualification
before drawing latency conclusions. General snapshot replay through the
persistent backend, including original-coordinate primal/dual qualification,
is now implemented and measured below. The separate QOCO snapshot executable
remains QOCO-only. The identical-input pinned upstream C API comparison is now
[implemented and measured](../results/local/2026-09-09/upstream-identical-capture-v608/README.md).
Real-capture cold convergence and native GTOC12 backend integration remain gaps.

Do not judge the entire quadratic-solver design from zero-Q cases alone. The
current GTOC12 [default smoothness weight is zero](../src/spacepdhcg/gtoc12/low_thrust.py),
so its conic objective has no numerical quadratic term. The native assembler
supports the existing optional banded control-smoothing Hessian `2*w*D^T*D`;
turning that option on would change the mission objective and is not a remedy
for the failed default cases. Instead, add the existing unchanged HCW and
displaced low-thrust trajectory fixtures from
[device SCvx integration tests](../cpp/cuda/tests/device_scvx_integration_test.cu)
as separate nonzero-Q benchmarks. Record actual inner work and residuals; the
presence of a Hessian alone does not prove useful conjugate-gradient work.
GTOC12's zero-Q problems remain mandatory for mission integration.

Once captured GTOC12 subproblems qualify through the persistent core, add its
explicit backend integration to native GTOC12 SCvx: retained assembly, warm-start
ownership, device acceptance and failure reporting. Compare the resulting complete
path with QOCO on the same inputs. The 225-leg replay currently offers no PDHCG
backend switch; passing it through QOCO cannot stand in for this integration.

Record cold setup, warm update/solve, recovery, verification and total time;
iterations, failures, memory, objective and original-equation residuals accompany
every timing. Keep pure PDHCG, pure IPM and explicit hybrid results separate.
The original plan permits GPU IPM polishing, but a hybrid advantage must show
that PDHCG contributed useful work and that its conversion/polish cost was paid.

Use the fixed 225-leg GTOC12 replay as a separate reliability regression corpus.
Preserve its 205 previously certified legs and classify the other 20 individually;
an unsuccessful call is not proof of physical infeasibility. The reconstructed
v598 return fixture tests repeat variability; it did not reproduce the original
H100 failure and cannot identify that failure's cause by itself.

A bounded first nonlinear diagnostic uses existing corpus indices
`0,68,137,12,44,201`: three previously successful legs, including scaling
regressions, and three unresolved virtual-control cases. Retain the stored
settings, an explicit overall deadline and partial failure evidence. The complete
225-leg replay follows a promising intervention; repeatedly replaying the full
corpus without a changed hypothesis is not the first experiment.

The small diagnostic suite selects the next intervention. It is not a SOTA
benchmark. Subsequent locked experiments retain the existing
[claim rules](../papers/paper1/CLAIMS_AND_DECISION_RULES.md): at least five measured
repeats per deterministic instance, paired confidence intervals, visible failures,
and a sustained scale regime. H2 requires at least 1.20x median PDHCG speedup over
the best qualified GPU IPM; H5 requires at least 15% lower complete SCvx time
under its reliability and accuracy gates; H6 requires at least 10% lower hybrid
time plus its accuracy conditions. Passing these project thresholds alone does
not establish world-best performance. The
[comparative campaign](COMPARATIVE_SOLVER_CAMPAIGN.md) also requires complete
trajectory-system comparisons and equal tuning budgets.

### 2. Improve numerical work where the diagnostic evidence points

The preferred architectural experiment uses the trajectory equations directly:

\[
r_k=x_{k+1}-A_kx_k-B_ku_k-c_k.
\]

Each interval evaluates its own residual; transpose products gather neighboring
contributions. This retains the primal-dual method while testing whether less
indexing and fewer atomics improve time to verified accuracy. Check operator
equivalence, the adjoint identity and original-equation residuals before timing.
Retain a generic sparse path for constraints outside the structured blocks.

Test structure, conditioning, restart policy and inner accuracy separately before
combining them. A block or diagonal preconditioner must preserve the cone metric
and projection mathematics. Objective-preserving scaling has already prevented a
captured numerical failure, but blanket Ruiz changes have also reduced whole-replay
reliability and increased campaign time; [those negative results](GPU_CONDITIONING_DIAGNOSIS.md)
remain part of the decision. Do not repeat an unchanged parameter sweep or relax
physics/objective tolerances to obtain a speedup.

If the profile shows convergence dominates, address that before another kernel
micro-optimization. If many short independent qualified solves dominate, prioritize
retained/batched refinement. If a few large solves dominate, prioritize interval
operators and recovery. Physical multi-GPU partitioning stays deferred under the
[active single-GPU scope](ACTIVE_SINGLE_GPU_ROADMAP.md).

### 3. Broaden mission search in parallel with solver work

The incumbent is the acceptance baseline throughout. The recent fixed-footprint
61-order experiment found no eligible improvement; repeating the same sampled
neighborhood is unlikely to answer why the fleet remains below the leaders.

Asteroid substitution v599 and the subsequent incident-window, fixed-cargo,
return-window and coupled-fuel experiments are complete; their measured outcomes
are recorded below. None improved the incumbent. The next score experiment must
change route construction or improve whole-route fuel prediction; another sweep
of the rejected schedules is not the current priority.

Candidate ranking must use weighted payload and fleet feasibility together.
The v733 fleet has approximately 0.255402 raw kg of ship-count slack; the earlier
v595 fleet had 8.359441 kg. The latest weighted gain consumed most of that margin,
so a further proposal has very little room to reduce raw haul. Every candidate still
needs both complete-fleet checkers and a verified weighted gain before promotion.
Track substitutions proposed, distinct route footprints, screens passed,
refinements attempted/certified, and the resulting fleet-score change.

After this bounded experiment, evaluate broader route construction, bidirectional
deployment/collection search, removal and reinsertion of multiple asteroids, and
cross-ship exchanges. Retain diverse compatible routes for fleet selection.
Audit how much is lost between Lambert estimates, low-thrust refinement and final
fleet selection. Test coupled timing/mass optimization of an entire itinerary
against the current leg-wise refinement where that audit shows an opportunity.
More evaluations of an inaccurate surrogate can amplify the wrong search choices.

Each search ablation starts from the same incumbent with matched compute budgets.
Report verified best score versus elapsed time and GPU-hours, alongside raw
kg/ship, diversity and qualification rate. The first milestone beyond
12,810.135953 weighted kg was achieved by v733 at 12,842.970672 weighted kg.
New campaigns must improve on the latest fully checked incumbent recorded in
[GTOC12 progress](GTOC12_PROGRESS.md), using the same pinned scoring table.
Sustained larger gains and a common-score comparison with the strongest published
missions are required to reach the goal.

Before allocating a larger generation budget, require a positive control: the
generator and screening pipeline must represent and retain the known certified
incumbent route at its actual timings and fixed cargo. Record the first rejection
at each construction, pruning, completion and refinement stage. A proxy rejection
of that control is a search-model discrepancy to diagnose, not evidence that the
certified trajectory is infeasible. Keep genuine low-thrust constraints and the
independent fleet gates unchanged while investigating that discrepancy.

These are separate decision gates. The numerical core must qualify captured
problems before its iteration throughput justifies native mission integration.
The search must reproduce its positive controls before more proxy evaluations
justify a larger search. Only improved independently verified missions justify a
score claim. Preserve the PDHCG-inspired core, but require each representation,
preconditioner and search change to earn its place through these measurements.

## Targeted literature review tied to experiments

| Question | Primary sources | Concrete use |
| --- | --- | --- |
| What useful upstream numerical changes are missing? | [PDHCG-CQP](https://arxiv.org/abs/2608.09159), [PDHCG-II](https://arxiv.org/abs/2602.23967) | Audit adaptive inner accuracy, preconditioning, restart and acceleration against the pinned implementation. PDHCG-II concerns QPs; transferring a rule to cones requires a correctness argument and an ablation. |
| What first-order trajectory systems must we compare with? | [Factorization-free spacecraft SCP](https://arxiv.org/abs/2402.04561), [parallel-in-time GPU SCP](https://arxiv.org/abs/2603.10711) | Add applicable published problems and strong baselines. First-order spacecraft SCP already has prior art; our contribution must be the demonstrated integration, structure or performance advantage. |
| How do strong GTOC12 pipelines convert search into actual haul? | [TheAntipodes GTOC12 methods](https://arxiv.org/html/2411.11279v1) | Compare subset/route construction, reverse collection search and the mismatch between Lambert screening and low-thrust feasibility. Its fifth-place competition result informs experiments; copying it does not establish first place. |

This is a focused update to the existing literature programme, not a restart.
Each proposed method needs a missing capability, a measurable hypothesis and a
bounded experiment. Published authors' speedups are context, not speedups of our
code on our hardware. Preserve [upstream attribution](../README.md#original-sources-and-attribution).

## Execution evidence added on 9 September

The prepared v599 substitution experiment has now completed: all 496 cases were
screened, with 2,778 total joint rows and 12 retimings. No candidate qualified for
physical refinement, so the verified incumbent is unchanged. CPU replay matched
all 1,488 initial GPU outcomes and located authority failures on changed legs.
The follow-up ranks all four incident-leg windows under equal retiming budgets.
[Measured work and negative evidence](../results/local/2026-09-09/asteroid-substitution-v599/README.md).

That controlled v604 follow-up also completed: 12,400 shared initial samples,
12 retimings per arm and 14,940 total joint rows. The new shortlist yielded three
feasible retimed plans versus two for the historical control; neither arm found
an objective-eligible plan. No full refinement was attempted and the verified
fleet score remains unchanged. A slight improvement in this small feasibility
sample is not a performance or score win. Audit proxy false negatives with a
bounded low-thrust truth set before committing to larger searches of the same
neighborhood. [Matched-arm search evidence](../results/local/2026-09-09/incident-window-substitution-v604/README.md).

The persistent snapshot importer now passes analytic GPU checks and directly
replays two actual GTOC12 captures. The generic scalar-inequality representation
does not qualify either within 100,000 iterations; QOCO qualifies five of six
reference repeats under the common external gate. This establishes a concrete
convergence investigation, not a solver advantage. The optional exact
variable-bound folding experiment has also failed to qualify either capture and
worsened their primal residuals; it remains off by default. Seven corrected
analytic solves pass and generic controls reproduce the original baseline.
The known-solution diagnostic is now complete: all eight imported reference
certificates pass the common KKT gate and are installed correctly on the GPU.
The native absolute natural-residual predicate rejects the initial points; after
one or 1,000 steps, seven outputs fail common objective-gap accuracy and none has
native accepted termination. This identifies a stopping/near-solution behavior
investigation rather than an import defect. [Replay evidence and limitations](GPU_PERSISTENT_CAPTURE_REPLAY.md).

The implementation audit also distinguishes the explicit persistent/cooperative
PDHG iteration from upstream PDHCG-CQP's inexact conic quadratic proximal scheme.
Recovery CGLS is not that inner solve. Both current captures have zero Hessians,
so missing quadratic inner iterations cannot explain these particular failures.
Keep the reusable native core, use the identical-input upstream reference
and add nonzero-quadratic trajectory captures before drawing general conclusions
about the method. Audit native versus common KKT stopping, then test one
primal-dual balance or restart intervention at a time. A global 2^-13 objective
rescaling is largely neutralized by existing balance; equality-penalty curvature
adds substantial sparse work without activating an upstream proximal solve.
Neither is a justified default or a measured improvement.

The pinned native upstream reference now replays those exact captures. It accepts
both supplied known-qualified points at iteration zero under its native rule and
the unchanged independent common gate. Both cold starts reach 100,000 iterations
without qualification: common gaps 0.00132932 and 0.0170812, with native one-shot
API wall times 37.3320 and 34.8176 seconds. Thus the evidence supports an explicit
common-KKT check before the persistent iteration, while cold convergence remains
a separate problem in both implementations. Two tiny upstream cold solves report
OPTIMAL but narrowly fail the stricter common gap gate; native status alone cannot
replace external qualification. These eight one-off diagnostics do not establish
a speedup. [Exact inputs, complete vectors and independent audits](../results/local/2026-09-09/upstream-identical-capture-v608/README.md).

An explicit GPU common-KKT stopping policy now fixes the known-point regression:
four actual captured starts across single-block and two-block kernels stop at
zero iterations with unchanged primal/dual bits and pass the independent gate.
Fourteen focused GPU calls cover mathematical and lifecycle failure cases. The
default kernels retain their previous compiled resource footprints. The policy
remains opt-in; four forced-single-block cold comparisons still fail at their
60-second deadlines, so no cold-convergence or general speed claim follows.
[Implementation, memory cost and complete evidence](GPU_PERSISTENT_CAPTURE_REPLAY.md#pinned-upstream-comparator-and-optional-gpu-stopping-policy).

Four subsequent automatic-grid comparisons finish all 100,000 updates in about
5.04–5.47 seconds but remain unqualified; common stopping leaves their trajectories
unchanged to rounding noise. The exact-linear upstream dispatch also remains
unqualified, while reducing its single-sample API times to 10.3160/8.7967 seconds.
The unchanged original equations and gates remain the basis of comparison.
[Automatic-grid and reference follow-ups](GPU_PERSISTENT_CAPTURE_REPLAY.md#automatic-grid-and-exact-linear-reference-follow-ups).

CPU spectral and gap decompositions now narrow the next core intervention.
The two observed failures do not exhibit a violated measured PDHG spectral bound
or a large initial global weight imbalance. They retain substantive equality,
stationarity and block-complementarity errors. The opt-in Halpern and adaptive
restart experiment is now complete. All fourteen tiny GPU calls have their
expected outcomes, and four actual supplied starts stop at zero iterations with
unchanged primal/dual bits. None of six cold comparisons qualifies at 100,000
iterations. Adaptive gaps are 0.0255402/0.0184402 against same-build default
0.999924/0.0112271; plain Halpern gaps exceed one on both captures. Thus this
change does not yet earn default selection or native mission integration.
[Complete source, negative comparisons and accuracy checks](../results/local/2026-09-09/halpern-core-v612/README.md).

The exact L1 proximal representation of the detected 10,000-cost epigraph pairs
has now been implemented and tested as a separate, default-off mode. Nine tiny
GPU calls pass their expected outcomes, and both original supplied certificates
qualify at zero updates with unchanged primal/dual bits. All four same-build cold
comparisons remain unqualified at 100,000 updates. L1 iteration times fall to
3.610990/3.656115 seconds versus 5.424073/5.549215, while its original gaps are
1.000291/0.00387847 versus 0.999924/0.0112271. Primal residuals worsen on both
captures. The logical reduction retains full original allocations for audit and
adds private working storage; it is not a memory reduction. These results earn
neither default selection nor native mission integration.
[Equivalence, independent review and complete GPU evidence](../results/local/2026-09-09/l1-prox-core-v615/README.md).

The fixed reciprocal-weight comparison v618 now tests one coefficient-only
policy, `omega=O/B`, against unit weight with the same L1 representation. It
changes the primal/dual step balance without changing the original equations,
penalties, acceptance gates or theoretical step-product condition. Twenty-two
tiny GPU calls pass their expected outcomes, and both supplied certificates
remain qualified at zero updates with unchanged bits. All four cold calls still
fail at 100,000 updates. Cancellation of global normalization lowers conditioning's
normalized gap from 1.000291 to 0.051506, but slightly worsens its primal residual;
on difficult, the gap rises from 0.00387847 to 0.0565266 and the primal residual
increases about sixteenfold. Neither weight earns default selection. A smaller
stationarity error alone is insufficient while feasibility remains unqualified.
The independent original-equation audit identifies mass continuity and initial
mass among the dominant remaining errors. Keep those feasibility components
visible in the next intervention's stopping and comparison records; the much
lower difficult objective under reciprocal weighting belongs to an infeasible
iterate. It cannot be credited as fuel saved.
[Fixed policy, frozen source and complete outcomes](../results/local/2026-09-09/l1-weight-core-v618/README.md).

Preserve the reusable PDHCG-inspired GPU core. The next numerical intervention
must address the measured feasibility/stationarity and balance behavior; changing
representation or restarting alone has not delivered qualified cold solves.
Interval operators and batching remain on the roadmap, but their speed must
eventually translate into qualified complete trajectories. Use targeted literature
to specify that intervention and its benchmark, rather than restarting a broad
survey or repeating an unchanged iteration budget.

The bounded low-thrust truth set has also completed locally: four routes,
66 native leg solves, 110.374 seconds including verification. Both original
controls pass both fleet checkers. Both proposals certify every preceding leg,
including all changed incident transfers, but fail the unchanged Earth-return
refinement. Their prefixes consume an additional 178.700067 and 95.392567 kg
of fuel, leaving only 13.723155 and 91.543915 kg for return. No feasible proxy
false negative or score gain was demonstrated, and a failed SCvx call still does
not establish physical infeasibility. Preserve whole-route mass checks; shift
the next score experiment toward broader itinerary construction and coupled
timing/mass choices, with a matched-budget baseline, rather than repeatedly
sampling this rejected substitution neighborhood.
[Fixed cargo, full controls, failures and exact work counts](../results/local/2026-09-09/fixed-cargo-truth-v606/README.md).

A final bounded return-window rescue also completed. It reused the certified
ship-7 prefixes and fixed cargo, screened 9,438 legal window pairs, and ran one
fresh original-route control plus four candidate returns. The control passes both
fleet checkers; every candidate remains uncertified. The cheapest proxy estimate
is 213.617946 kg against 91.543915 kg available, and all four low-thrust attempts
consume that allowance while retaining nonzero defects. The run took 86.221170
seconds including verification and produced no score gain. Do not repeat this
return-only grid: the next distinct mission hypothesis must reduce preceding
fuel use or change replacement geometry and coupled itinerary timing. This finite
failed search does not prove physical infeasibility.
[Complete controls, failed arrays and independent accounting](../results/local/2026-09-09/return-window-rescue-v607b/README.md).

The coupled itinerary fuel search v611 also completed: twelve device searches
evaluated 39,852 complete itinerary proxies and 793,152 Lambert direction requests
in 0.450427 seconds of search-wrapper time. The complete run took 34.844880 seconds,
including baseline verification and seventeen native leg refinements. Only the
largest eligible candidate per fuel-price setting was shortlisted; this selected
ship 20's predicted +6.033277 weighted kg. Its first sixteen legs certified, but
the Earth return exhausted its fixed 292.503659 kg propellant allowance and remained
uncertified. Three smaller positive proxy candidates were retained but unrefined;
their feasibility remains unknown. The fleet stays at 12,810.135953 weighted kg.
The evidence supports changing route construction and whole-route fuel modelling,
not extrapolating proxy throughput into certified solutions or claiming every
eligible candidate failed. [Full work counts and retained failures](../results/local/2026-09-09/coupled-fuel-search-v611/README.md).

The route-family audit rules out a global eight-miner cap: search defaults permit
ten deployments and eleven incumbent ships already collect from nine asteroids.
A bounded scan of archived summaries contains 1,374 records marked certified,
independent and closed, representing 1,282 distinct plans; 158 distinct plans
have nine miners and none has ten. These are archived certificates, not fresh
trajectory re-verifications or an exhaustive feasibility result.
`bundles.refine_candidates` retains only one chain per Earth-departure leg before
applying the small refinement limit. Consequently, longer alternatives can be
discarded before low-thrust testing. Once the generator passes incumbent admission
controls and produces an eligible pool, compare that selection with depth-stratified
selection on the same pool and the same total refinement budget. Preserve full collection
and Earth-return costing, fixed cargo during each refinement, and the complete
fleet gates; do not assume adding a miner improves a mission.
[Source and bounded archive evidence](../results/local/2026-09-09/route-family-cap-audit-v611/README.md).

The follow-up generation v616 retained all 167 complete proxy routes from one
certified Earth-departure seed and a 62-asteroid family. It evaluated 8,556,740
Lambert directions in 18.544582 seconds of recorded generation-driver time,
excluding the initial source/input preflight. Two completed
routes collect from nine asteroids; none passes both fleet-improvement screening
conditions. The old selector discards alternatives, but a wider shortlist cannot
rescue this particular pool. No low-thrust refinement or score promotion followed.

The positive-control audit identifies a more immediate gap. All eight asteroid
deployment transfer times in the retained ship-23 incumbent are outside the
generation grid, as are seven of its nine collection transfer times. Its original
asteroid-order prefix appears only through depth two in the completion journal;
the journal does not identify the unrecorded child-pruning decision responsible.
The frozen finishing calculation rejects the actual incumbent schedule at fixed
cargo both with estimated and measured deployment-prefix mass. These two CPU
checks reuse stored Lambert values; they are not fresh trajectory solves. They
demonstrate a discrepancy between the proxy and an already certified route, not
physical infeasibility. All generated candidates retain their full mining-rate
cargo, so cargo shrinkage does not explain this pool's lower haul. Prioritize
incumbent reproduction and measured whole-route proxy error before another broad
generation run. [Complete pool, accounting and positive-control diagnosis](../results/local/2026-09-09/depth-diverse-generation-v616/README.md).

The completion-cost correction v619 fixes two concrete discrepancies: DP's
calibrated hop model was being replaced by the beam model, and generic return
inflation was evaluated before collection burns at an overestimated mass.
Completion now prices each flight once at its actual forward mass and records
the factor it spends; existing certified return-cell overrides and all mission
gates are retained. Forty scalar controls compare old/corrected code, estimated/
measured deployment-prefix masses and flat/existing-fit costs on five historical
routes. Admissions rise from 0/20 to 2/20, both for the estimated prefix of ship 4;
all measured-prefix controls remain rejected. Historical ship 23's flat-model
shortfall falls from 35.56 to 9.59 kg, or 6.39 kg with the existing fit. The 45-test
CPU suite passes. This is a corrected search calculation, not fresh trajectory
certification or a score gain.

An independent scalar diagnostic across all 23 historical routes attributes a
median 22.81 kg of extra predicted return fuel to the old mass argument alone.
Even at measured return mass the generic model overpredicts 20/23 returns, with
a median error of 18.36 kg. Next decompose this remaining proxy error and reproduce
measured-prefix incumbent admission before expanding generation. Port equivalent
completion across a batch with CPU/GPU parity; existing CUDA joint/retiming code
already uses forward mass, but its model set does not reproduce DP's five-feature
hop fit. Do not claim that these differing models are equivalent or relax the
full-route physics checker to make a proxy pass.
[Frozen controls, corrected accounting and reproducible CPU evidence](../results/local/2026-09-09/consistent-completion-v619/README.md).

The next completion tranche now has a full-core CUDA implementation and a
successful RTX 5090 native parity run: 304 fixed candidate evaluations in three
kernel calls, including exact admission and boundary decisions. The production
adapter's four GPU model tests also pass, with eight calls and 1,048 candidates.
This ports the corrected forward cost calculation across independent candidates
while retaining the existing model set. A subsequent 60-call bounded comparison
finds 3.00x/3.39x faster warm complete costing for fitted 24/48-candidate batches,
including host packing and plan readout. Small flat-model batches regress. At
1,024 fitted candidates, average packing takes 37.76 ms while the kernel/event
interval averages 0.101 ms; the median complete method takes 41.21 ms. Prioritize
retained native metadata and request construction over another completion-kernel
optimization. This reuses only 20 historical controls and earns no full-search,
SOTA or certified-throughput claim; the mode remains opt-in. See
[batched route completion](GPU_ROUTE_COMPLETION.md).

The historical residual decomposition also sharpens the mission intervention.
At measured burn masses, return overprediction totals 519.94 kg across 23 routes;
20 returns are overpriced. The existing collection fit improves median per-hop
error while worsening its summed error, so a blanket calibration benefit is not
supported. Test a better whole-route estimator or bounded refinement escalation
for uncertain proxy rejections on separate held-out routes. Preserve the actual
payload and final physics checks. More evaluations of the same biased estimates
do not resolve incumbent admission.

For the numerical core, a coefficient-only review shows that the initial-mass
and mass-dynamics equalities in the two difficult zero-Q captures admit exact
affine elimination with a forward prefix and reverse adjoint. Original mass
equality duals must be reconstructed before original-coordinate qualification.
This preserves the optimization problem, but cumulative mass bounds greatly
increase the reduced operator norm; using the old step size would be unsafe.
An operator-aware metric with a verified norm bound is therefore a prerequisite,
not an optional later optimization. A subsequent coefficient-only audit derives
absolute row/column steps with tied SOC metrics and certifies a squared operator
norm bound of 0.9025 on both captures. The next step is an explicit experimental
implementation with matching prefix/adjoint, dual recovery and original-equation
checks, followed by the same finite cold comparisons. The metric is an established
preconditioning approach, and its stability bound predicts no speed or convergence
gain. Keep unchanged nonzero-Q trajectory benchmarks in
the broader campaign so this mission-specific reduction does not define the
entire solver comparison.

## Reporting after each tranche

### v622 interventions and decisions

The mass-elimination/preconditioning implementation is now additive and off by
default. The independent three-interval GPU oracle, original-seed preservation,
mode transition, cancellation and invalid-input guards pass: seven solve calls
perform five updates. The same-build cold comparison has now completed. All four
100,000-update solves remain unqualified under the unchanged original KKT gate.
Mass elimination takes 7.376578/7.648103 seconds against unit-L1's
3.596469/3.659972 seconds. Conditioning's normalized gap improves from 1.000291 to
0.229510, but difficult's worsens from 0.00387847 to 0.0878914. The original two
supplied certificates still qualify at zero updates with unchanged bits. The
reduction and metric change together; this experiment does not attribute their
effects separately. It earns neither default selection nor mission integration.
The independent 65-digit audit confirms that initial mass is exactly one and
remaining mass-dynamics defects are at most 2.88e-16. The dominant failures have
moved to late position/velocity dynamics, state stationarity and the last thrust
cone. Thus mass bookkeeping has been resolved without resolving the whole conic
problem. On difficult, the absolute objective discrepancy falls while the
normalized gap rises because its denominator changes; neither number is a
qualified objective improvement.

The search intervention is now a bounded fixed-cargo refinement queue, connected
to native completion readback before final-mass rejection discards a request.
Seventeen route-disjoint archival controls, separate from the five development
routes, receive 34 fixed scalar evaluations with their saved DVs and features.
Flat pricing passes eight and rejects nine on final mass; the existing fit passes
three and rejects fourteen. All earlier completion gates pass. These are
historical queue controls, not statistically unseen trajectories or fresh
certifications. The result argues against assuming the existing fit solves the
admission problem or that more evaluations alone will improve mission quality.

The queue retains 48 requests and spends at most four refinement claims, including
two uncertain final-mass rejections. Exact cargo and epochs remain fixed; each
returned route must pass both complete-fleet checkers and improve the verified
weighted objective while satisfying the raw ship-count rule. The source, 76 CPU
integration tests and immutable archival readbacks are prepared. A fresh native
queue execution and any resulting score gain remain to be demonstrated.

Compact native completion requests additionally move geometry and return-cell
selection to retained CUDA data. The final fixture/lifecycle test passes all
seven tests with no skips: 18 valid completion calls, 2,358 candidate evaluations,
and six malformed calls which must leave outputs untouched. The largest saved
mass/result difference is 4.55e-13 kg; gate decisions and cargo match exactly.
An initial unused-metadata mismatch and a later test-fixture construction error
remain archived rather than omitted. The subsequent same-build method comparison
passes 80 calls and 27,040 repeated proxy evaluations from 20 historical controls.
At 48 fitted candidates, warm method time falls from 2.689 to 2.196 ms; at 1,024,
it falls from 40.945 to 18.966 ms (2.159x throughput). Flat 24/48-row cases and
fitted 24-row cases regress. Compact requests therefore remain explicitly selected.
Catalogue/source validation and Python topology packing remain in the measured
cost; the median matching-call packing share is 84.6% for 1,024 fitted rows. The
combined native library also passes the original 262 completion fixtures, seven
compact tests and seven small mass-solver calls. This component work neither
certifies a trajectory nor establishes SOTA.

The next numerical decision needs targeted analysis of the remaining original
equations, not another unchanged 100,000-step run. Prior global-rescaling,
fixed-weight and restart experiments remain relevant negative controls. Review
the interaction between the new metric and objective/constraint scaling before
adding another heuristic. [PDLP's primary paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/a8fbbd3b11424ce032ba813493d95ad7-Paper.pdf)
describes weight updates at restarts; [PDHCG-CQP's implementation section](https://arxiv.org/html/2608.09159v1#S4)
combines cone-aware rescaling, acceleration and adaptive weights. Those are
references for a controlled intervention, not evidence that this application will
converge faster. Retain a broader workload with nonzero-Q trajectory problems and
actual SCvx warm starts alongside these difficult cold regression cases.

The [complete v622 evidence](../results/local/2026-09-09/causal-mass-and-completion-v622/README.md)
retains the failed convergence comparison, admitted/rejected archival controls,
all packing timings and raw readbacks, independent audits and combined-library
integration checks. The mass experiment remains disabled by default, the compact
completion mode remains optional, and no new fleet or score is credited.

Publish the measured outcome, failed cases, exact source/binary/input identities,
and the decision to keep or reject the change. Lead with independently certified
throughput and complete solve time for solver work, and verified fleet-score gain
for search work. Report component timings underneath these outcomes. Archive
negative experiments so the next run changes a hypothesis rather than repeating
the same search. Retrieve and display any new accepted fleet in the existing
visualiser. No new solver benchmark or fleet improvement is claimed by this plan.
