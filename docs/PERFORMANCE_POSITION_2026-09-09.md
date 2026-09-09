# Performance conclusions and the role of PDHCG

The evidence supports keeping the PDHCG-inspired core and changing how we prove
and deploy its advantage. The current obstacle is reliable convergence to the
required accuracy. Separately, better GTOC12 scores require better discrete
mission choices. Neither obstacle is resolved by faster individual kernels.

## Current decision and next experiments

Retain the PDHCG-inspired research core, qualify its numerical method, and
deploy it where complete time to the required accuracy wins. The current
GTOC12 refinement path uses GPU QOCO inside SCvx; search throughput improvements
are not evidence of a PDHCG convergence advantage. Large coupled trajectory or
scenario problems are the strongest prospective fit when factorization cost or
memory dominates. Small individual legs and tight final accuracy need a measured
comparison with a structure-exploiting interior-point solver. Diagonal-Q cases
also require the available closed-form proximal update as a credible baseline.

The next core work separates two questions. The completed saved-data diagnostic
shows that equality projection preserves about 99.853% of the original
objective-gradient energy and almost all remaining stationarity-error energy.
It does not support blaming equality bookkeeping or directly blaming the L1
penalty for small physical steps. The cone projection responds weakly along
the remaining error directions at the inspected point. This narrows the outer
convergence investigation; it supplies no safe larger step size or proof that
a new metric will help. [Full diagnostic and its limitations](../results/local/2026-09-09/core-conditioning-v637/README.md).
Separately, the unchanged displaced HCW N20 problem has now been
assembled and exported: 186 positive quadratic coefficients, 132 equalities and
20 SOC4 blocks. Its native and standard-conic representations preserve the
original objective and coefficients. No optimizer has run on that capture yet;
it supplies an actual matched quadratic correctness input, not a CG speed claim.
[Exact capture](../results/local/2026-09-09/nonzero-q-capture-v637/README.md).

Source inspection also distinguishes algorithm completeness from CUDA execution.
The custom cooperative path computes Q*x and takes one explicit projected
gradient step. The pinned PDHCG-CQP implementation uses an exact diagonal-Q
proximal update, or projected Barzilai-Borwein inner iterations for general Q;
the latter is not conjugate gradient. It also has different outer ordering,
reflection, restarts and weight policies. Its existing upstream snapshot replay
frontend can establish a same-input numerical baseline before those operations
are retained in a new persistent path. The original QP paper's CG algorithm must
be identified and measured separately if making a specifically CG-based claim.
Zero-Q inputs cannot test that distinction, and the new diagonal-Q input does
not require an iterative quadratic inner solve. Count actual work: upstream's
inner counter advances even in a diagonal-Q outer iteration.
[Custom update](../cpp/cuda/src/cooperative_pdhg.cuh),
[upstream replay frontend](../cpp/cuda/tests/upstream_snapshot_replay.cu),
[original QP paper](https://arxiv.org/abs/2405.16160),
[conic extension](https://arxiv.org/abs/2608.09159).
Earlier upstream tests on different conditioning/difficult inputs also failed
their cold starts. The bounded evidence search found no upstream run on these
two exact newer captures or the new HCW input. That matched comparison is useful
new work; upstream is not presumed to solve the observed conic tail.

The new bounded search has now screened eight unused asteroid families and
searched four with new Earth departure geometry. It produced 923 distinct
prescriptions in 33.865 seconds of worker time (32.142 seconds in route searches),
with 43,515,180 fresh Lambert branch requests including the initial screens.
These are surrogate candidates and geometry evaluations, not certified solutions
or a same-workload speed comparison. No candidate or cross-family pair satisfies
the shared raw-mass rule for a profitable replacement or fleet growth.

All 121 attempted completions at depth nine or ten fail; successful closed
routes reach at most eight deployments. No timeout or incumbent-asteroid
conflict explains this. The surfaced completion records include mass shortfall,
missing return/collection choices and authority failures, with overlapping
categories. The native expansion stage admits 1,241 of 1,259,796 valid children,
but discarded-child reasons are not saved. We cannot attribute all missing
routes to a specific gate. Better calibrated whole-route admission, bounded
refinement of uncertain rejections, and coordinated route/fleet changes now
have priority over another unchanged beam-width or unused-family sweep.
[Complete campaign and fleet-selection audit](../results/local/2026-09-09/route-family-v637/README.md).

Joint schedule/control refinement is a separate priority: newly retrieved Lambda
reports show a retimed ship-18 candidate reaching 16 certified flights before
its seventeenth flight, the Earth return, fails. Three later-arrival attempts
also fail, and no full-fleet check runs. Process completion is not trajectory
certification or an infeasibility proof. The verified fleet remains
13,526.961241 weighted kg and 14,915.044490 raw kg, with only 5.604591 kg of
headroom under the 24-ship raw-mass rule. Route ranking must respect that shared
constraint, asteroid conflicts and the actual weighted objective.
[Retrieved reports and exact timing scopes](../results/local/2026-09-09/lambda-run-status-v637/README.md).

For GPU engineering, the existing collection DP is already multi-block, but
its transition kernel launches the full subset/location/time grid for every
cardinality. At ten asteroids that schedules 102,400 coordinates per time
sample across all layers; only 5,119 pass the structural state filters. Exactly
half the dense state storage represents invalid subset/location combinations.
Compact layer scheduling and state storage are therefore concrete candidates
for the next native experiment. Those integer work counts are not speedup
measurements: preserve predecessor order and epsilon tie handling, then test
exact outcomes and complete route-search time. Python call-profile time inside
the native solve includes synchronized GPU work and must not be attributed
entirely to CPU arithmetic.
[Pinned source and integer work counts](../results/local/2026-09-09/collect-dp-work-v637/README.md).

The acceptance targets remain explicit: qualified original-equation solver
results; complete certified trajectories per second including failed attempts;
and the best independently checked weighted fleet score at a fixed compute
budget. A hybrid must include the PDHCG work, handoff and final polish in its
comparison with the complete baseline. A targeted numerical/literature review
supports each identified bottleneck; the current evidence does not justify
another unchanged iteration or beam-width sweep.

The next implementation sequence is:

1. Run the actual pinned upstream method on the two exact zero-Q captures and
   the newly exported HCW capture, beside the existing qualified reference.
   Retain complete primal/dual vectors and measure setup, convergence and final
   checking separately. The HCW case tests diagonal quadratic correctness; a
   legitimate coupled-quadratic physical workload is still needed for a CG claim.
2. Capture a bounded set of the deeper route-completion rejections, including
   original requests and rejection causes, and test uncertain cases against
   actual low-thrust refinement. Use this to calibrate admission before widening
   the same beam again.
3. Optimize the itinerary's event times, controls and mass evolution together
   on the stalled return cases. Require an independent whole-route certificate
   before offering a route to the fleet selector.
4. Add diverse large-neighborhood route moves and coordinated fleet selection
   using asteroid conflicts, raw-mass feasibility and weighted-score opportunity
   prices. Compare verified score reached at the same complete compute budget.
5. In parallel with numerical and search work, test compact collection-DP
   scheduling/storage and retained native preparation using matched output,
   sanitizer and end-to-end timing checks. Promote each change only on measured
   complete-workload benefit; static work counts do not establish that benefit.

## What is established

- The tested PDHCG path has progressed beyond the original one-block solve:
  v630 uses 128 cooperative blocks, parallel setup work and retained storage.
  Its measured solve intervals are roughly 0.374–0.377 seconds for 10,000
  updates, but none of those outputs qualifies. Fast unqualified updates do
  not establish fast trajectory solutions. Legacy single-block paths remain
  in the source. [Exact launch and timing scope](GPU_CORE_RETAINED_COLD_STARTS.md).
- The v630 comparison qualifies **0/4 persistent PDHCG outputs and 1/4 QOCO
  outputs** under the same original-equation gates. The four outputs per backend
  cover two cold trials on each of two captures from one synthetic fixture;
  they are not four independent missions. Retained buffers do not cure the
  PDHCG primal failures. There is no demonstrated qualified speed advantage.
  [Exact inputs, complete costs and independent audits](GPU_CORE_RETAINED_COLD_STARTS.md).
- The new CPU reference enforces the trajectory equalities to approximately
  machine precision, but **0/2 outputs qualify after 10,000 updates each**.
  Its remaining relative gaps are about 0.09146 and 0.08159. Original virtual
  control L1 complementarity dominates the gap; thrust cones and stationarity
  also fail. This isolates a numerical target rather than completing a GPU
  implementation. [Reformulation, arithmetic and measured failures](PDHCG_EQUALITY_PRIMAL_REFERENCE.md).
- The subsequent joint equality/L1 proximal reference removes the saved points'
  virtual-control complementarity error, but still qualifies **0/2 captures**.
  It stops on a numerical merit-decrease check, after zero and 73 committed
  outer updates. Factors succeed; other original cone, stationarity and gap
  errors remain. This is a more specific numerical diagnosis, not a completed
  solver or a throughput improvement. The next bounded intervention tests a
  cancellation-resistant decrease calculation and honors the unchanged inner
  residual stop before Armijo. [Design and recorded failures](PDHCG_JOINT_PROX_DESIGN.md).
- That intervention has now run. The v635 CPU reference completes 10,000 outer
  updates on each capture without an inner failure. Both final points pass the
  original primal, cone and complementarity gates, but dual stationarity
  (about 2.07e-8) and objective gap (about 0.001261) still fail the 1e-9 gates.
  All eight saved points fail both independent original-coordinate audits.
  Complete worker-process time is 19.765 seconds for both cases, including
  diagnosis and export; this is not GPU throughput. The improvement establishes
  a working inner operation and isolates the remaining outer convergence tail.
  [Exact sources, inputs and outcomes](../results/local/2026-09-09/core-joint-reference-v635/README.md).
- The v636 saved-data comparison finds those two final primal objectives about
  **3.631% higher** than qualified QOCO outputs on the identical inputs. A
  fixed-primal dual correction cannot improve the objective, so that proposal
  was canceled before execution. The subsequent bounded numerical intervention changes
  the primal iteration using the working joint prox and fixed-metric reflected
  Halpern/restarts. Original accuracy gates and the same-input cost comparison
  remain visible. [Decision and retained evidence](../results/local/2026-09-09/core-primal-decision-v636/README.md).
- That restart experiment also fails qualification: all eight saved readouts
  fail both independent audits after the bounded two-input run. Each case reaches
  10,000 maps and eight restarts, with no inner failure. Costs are slightly lower
  but cone feasibility and complementarity worsen; final cone violations are
  about 1.01e-5 and 5.91e-6. Objectives remain about 3.517% above the qualified
  QOCO references. Complete CPU worker time is 24.077 seconds for both cases.
  This is not a usable improvement or a CUDA rollout candidate. Conditioning
  diagnosis must precede another core intervention.
  [Complete negative experiment](../results/local/2026-09-09/core-joint-halpern-v636/README.md).
- Host work still matters. Retained catalogue fingerprints improve complete
  route-search throughput by **7.33–7.48% on H100** in the paired two-route
  experiment. Those are surrogate search candidates, not certified trajectories.
  Small fresh CUDA ephemeris batches are slightly slower than CPU preparation
  despite removing the CPU orbital arithmetic. Reuse and complete workload
  size determine whether the transfer is worthwhile.
  [Catalogue measurements](GPU_CATALOGUE_OWNERSHIP.md),
  [ephemeris measurements](GPU_ROUTE_EPHEMERIDES.md).
- The subsequent resident-catalogue comparison cuts upload bytes by 99.44%,
  but improves complete search throughput by only 0.09%/1.11% on H100 and
  2.37%/2.33% locally. H100 produces about 28.4 surrogate candidates/s on those
  two saved routes. Three repeats provide no confidence bound; the 0.09%
  observation is effectively flat. This supports prioritizing the remaining
  search and refinement costs. These are incremental results with the earlier
  hash cache enabled, not additive speedup claims or certified throughput.
  [Downloaded paired results](../results/lambda/2026-09-09/gpu-resident-catalogue-v827/README.md).
- The certified frontier now reaches **13,526.961241 weighted kg**, **14,915.044490
  raw kg**, 24 ships and 208 asteroids. Recovering a complete, compatible archived
  route adds **503.256340 weighted kg (3.86%)** without a new optimizer call.
  Both original full-fleet checkers pass. The original 23-ship Result is an exact
  byte prefix; the additional ship uses the archived row values. This is a
  fleet-selection gain and provides no new PDHCG performance evidence.
  [Exact fleet and fresh checks](../results/local/2026-09-09/fleet-addition-v633/README.md).
- The latest CUDA beam-expansion work demonstrates a larger search-layer gain:
  **2.58x/2.72x on H100 and 1.61x/1.75x locally**, for the same two route
  searches. Combined throughput is about **74.1/41.6 surrogate candidates/s**
  (H100/local). Three alternating measured samples follow warm-up; timings
  exclude process startup and catalogue loading. Topology and ordering match,
  with maximum FP64 difference 9.095e-13 across the two GPUs. The wider saved
  beam yields 1,960 candidates and 509 additional asteroid orders, still
  unrefined. This earns broader search at lower cost, not certified throughput
  or a PDHCG advantage. [Paired evidence](GPU_BEAM_EXPANSION.md).
- The following CUDA admission checkpoint removes about 99% of ranked downloads
  and reduces materialized children from 105,755 to 1,092. Its complete-search
  throughput changes are modest: +0.59%/+3.85% on H100 and -2.15%/+1.62% on RTX
  5090, with overlapping RTX sample ranges. This does not establish a general
  speedup. The saved profile now points to route completion, collection-DP
  preparation and forward scheduling as the next substantial recurring costs.
  All candidate bytes match the preceding expansion checkpoint; score is
  unchanged. [Admission measurements and profile](GPU_BEAM_ADMISSION.md).
- Shared native Earth-return rows subsequently improve the same two narrow
  searches by **15.5–17.0% on H100** and **29.6–30.5% on RTX 5090**. Combined
  throughput reaches about **88/53 surrogate candidates per second**. Three
  alternating measured samples follow warm-up; startup and catalogue loading
  are excluded. Candidate bytes remain unchanged within each GPU. Reuse avoids
  12,992,408 repeated branch requests, which are correctly excluded from fresh
  work counters. Both GPUs pass 205 tests and the recorded sanitizer/leak checks.
  Collection-DP preparation/control remains the main profiled recurring cost.
  This is a search-layer gain and adds no certified trajectory or fleet score.
  [Native ownership, paired records and limits](GPU_SHARED_RETURN_OPTIONS.md).

The saved wider pool has now been compared with the new 24-ship incumbent.
None of its 509 additional asteroid orders provides a positive standalone
weighted replacement. The only positive request in the 1,960-candidate pool
is the same ship-10 prescription already blocked at its return. This is a
saved proxy-data comparison, not a proof that those orders cannot be improved
by continuous optimization or coordinated fleet changes. It does mean that
more width around these two fixed families has not supplied a new profitable
refinement shortlist. The frozen benchmark constructs its beam with every
asteroid weight set to 1.0, applying the original bonus weights only when
sorting completed plans. It also fixes one incumbent Earth seed per ship,
restricts each pool to local neighbors, and supplies no master opportunity
prices. Production already supports bonus weights and opportunity prices;
this historical campaign does not exercise them.
[Saved pool comparison and exact campaign configuration](../results/local/2026-09-09/wide-pool-selection-v635/README.md).

The v636 matched weight experiment now tests that hypothesis on the same frozen
GPU implementation. Both unit controls reproduce the complete old pools exactly.
The first control initially stopped on an in-memory integer-key versus JSON
string-key comparison bug; saved-data equality proved no numerical mismatch,
and the continuation ran only the three unconsumed searches. All four searches
completed within their original budgets with no refinement or certificate.

Bonus-weighted construction yields 851/885 candidates for ships 10/21, with
745/775 new fixed-cargo prescriptions and 355/289 additional deployment orders.
It supplies no positive eligible standalone or paired replacement. Best predicted
weighted cargo falls from 579.157 to 534.734 kg for ship 10 and from 539.208 to
516.524 kg for ship 21. Actual bonus weights change both deploy ranking and
collection scheduling while the heuristic propellant penalties remain fixed;
the final pools do not isolate which rejected prefix causes the loss. This is
negative evidence for this single weight substitution, not a reason to optimize
the competition score using raw mass alone. Preserve both useful objectives in
route generation, broaden Earth seeds/families and use fleet opportunity prices.
[Complete experiment, comparison repair and saved selection audit](../results/local/2026-09-09/weighted-beam-v636/README.md).

The native GTOC12 trajectory refinements currently use **GPU QOCO inside SCvx**.
The custom persistent PDHCG-inspired backend is still awaiting reliable
qualification and explicit integration into that path. Its implementation uses
explicit PDHG updates and does not yet implement the upstream conic quadratic
proximal inner solve. Recovery CGLS is a different operation. These boundaries
must remain visible when attributing performance or mission gains.

The fixed-date v631 GPU return test exposed missing state-mass-shortfall terms
in reference acceptance merit: a dynamically consistent seed was 10.28998 kg
short of required final mass, but all 22 steps were rejected. The v632 fix now
prices all original mass inequalities in reference, candidate and predicted
merits, with focused GPU and sanitizer regressions passing. In the matched
rerun the original control again independently certifies in two iterations.
The candidate accepts two steps, then stalls after 26 total iterations with
dynamics/virtual defects about 0.002782, still far above unchanged gates. No
candidate certificate or fleet improvement results. The two-case worker takes
9.589 seconds; the different iteration paths and single observations do not
establish a speedup. This confirms the acceptance repair while exposing a
remaining trajectory-refinement problem. The subsequent one-call CUDA interval
replay reproduces the reported maximum defect exactly and localizes the sole
component exceeding the original gate: **82.854828 m/s in the first interval's
heliocentric y-velocity**. Thrust in that interval is already at the 0.6 N limit.
Each interval starts from its own saved node, so small remaining interval defects
do not establish a continuous feasible route. This identifies a discontinuity
and does not prove global infeasibility. Two later Earth arrivals, with unchanged
cargo and certified prefix, have now been tested on the local GPU. The +30-day
and +60-day cases remain uncertified after 44 and 32 SCvx updates, with normalized
defects 0.004298 and 0.024890. The worker takes 22.407 seconds, including one CUDA
batch for the two new Earth targets. Neither case reaches an independent flight
certificate or full-fleet check. Simply extending this seed with coast does not
resolve the failure; the next mission work must couple timing and control
initialization across the itinerary. These local failures do not prove global
infeasibility or change the verified score.
[Initialization](GPU_MASS_SCALED_INITIALIZATION.md),
[implemented repair and matched outcome](GPU_MASS_MERIT.md),
[interval replay and exact scope](../results/local/2026-09-09/interval-defect-replay-v633/RESULTS.md),
[two later-arrival outcomes](../results/local/2026-09-09/return-horizon-v634/README.md).

The v636 optimized-control seed experiment also fails: CUDA regenerates the
trajectory from the saved optimized physical thrust and prescribed initial state,
then returns a defect of 0.002781793 after 27 SCvx updates, three accepted.
No certificate or fleet check runs. The nearly identical aggregate defect
rejects this particular initialization hypothesis; it is not an infeasibility
proof or a new localization of the residual.
[Exact seed inputs and complete outcome](../results/local/2026-09-09/optimized-control-seed-v636/README.md).

## Where the method is most promising

PDHCG's attraction is sparse matrix-vector work, vector updates, and manageable
memory use without repeated large KKT factorizations. The original QP algorithm
handles quadratic curvature through approximate conjugate-gradient proximal
solves; the conic extension uses projected-gradient inner solves where needed.
These properties support large convex subproblems and GPU execution.
[Original method](https://arxiv.org/abs/2405.16160),
[conic extension](https://arxiv.org/abs/2608.09159).

Our proposed spacecraft fit is large sparse convex subproblems inside SCvx,
especially long horizons, multiple uncertainty scenarios, and repeated related
problems where structure and iterates can be retained. Batching independent
trajectories and using the interval dynamics directly could improve utilization
and memory traffic. These are hypotheses to test against qualified competitors,
not an established crossover map for this project.

The two recent GTOC12 captures have an exactly zero Hessian. They therefore do
not exercise the original method's quadratic CG advantage. They remain mandatory
zero-Q regressions alongside held-out mission captures, while nontrivial existing
nonzero-quadratic trajectory fixtures
must test the broader numerical design. Adding an artificial smoothing objective
to GTOC12 merely to favor the solver would change the problem and is not the plan.

A source inventory identifies displaced HCW and powered-descent recipes with
positive actual quadratic coefficients. All inspected Hessians are diagonal,
including the synthetic trajectory-banded fixture; free/box quadratic proximal
updates therefore have a closed form. These are useful quadratic correctness
and crossover controls, but do not inherently demonstrate a CG advantage.
No fully bound serialized nonzero-Q physical capture or coupled-Hessian physical
workload was established in this bounded inventory. Export exact matched inputs
from existing recipes before timing them; preserve the CPU Euler versus native
RK4 descent distinction. [Recipes, original gates and source pins](../results/local/2026-09-09/nonzero-q-inventory-v636/REPORT.md).

Small isolated problems, poor conditioning, and the final tight-accuracy phase
can favor an interior-point method. We will measure that crossover rather than
require PDHCG on every subproblem. A hybrid earns its place only when PDHCG's
work plus handoff and polishing costs beat the relevant complete baseline.

## Complementary methods and their responsibilities

| Layer | Proposed or retained method | What it contributes |
|---|---|---|
| Choose asteroid sequences | Diverse beam search and large-neighborhood moves; new Earth seeds, insertion/removal, reordered collections and cross-ship exchanges | Creates new route choices outside the exhausted current pool. |
| Screen many choices | Batched CUDA dynamics/Lambert/low-thrust approximations, calibrated against actual refinements | Spends expensive trajectory solves on promising routes; proxy scores do not establish feasibility. |
| Optimize a continuous trajectory | SCvx with physical propagated initialization, retained iterates, and coupled schedule/mass updates | Handles nonlinear dynamics and refines thrust histories for a selected itinerary. |
| Solve convex subproblems | Qualified PDHCG path, GPU IPM baseline, and an explicitly measured progress-triggered hybrid | Uses each backend where its complete cost and reliability justify it; preserves final gates. |
| Choose the fleet | Weighted set-packing/master selection over certified route columns, combined with pricing/new route generation | Resolves asteroid conflicts and shared raw-mass/ship-count limits; individual kg/ship is not the objective. |
| Accept a result | Independent propagation and both full-fleet checks | Verifies dynamics, mass continuity, thrust, event rules and the actual weighted score. |

Published GTOC12 systems already combine route construction, continuous
low-thrust optimization, timing refinement and fleet selection. TheAntipodes
uses beam search, SCP and genetic selection; OptimiCS describes beam/tabu search
and simulated annealing for route composition. They support this layered design,
not a claim that copying one technique establishes first place.
[TheAntipodes](https://arxiv.org/abs/2411.11279),
[OptimiCS](https://www.sciopen.com/article/10.1007/s42064-024-0223-7).

## Execution order

1. **Repair reliable refinement and the measured numerical obstruction.** The
   mass-merit and inner line-search repairs have passed their focused tests.
   The mission candidate still fails after two later-arrival tests and a fresh
   CUDA propagation of the optimized-control seed. The latter returns a nearly
   unchanged 0.002781793 aggregate defect after 27 updates, with no certificate.
   Seek different timing, cargo or itinerary choices. The joint dynamics/L1 reference now reaches its outer
   cap, with dual stationarity and gap remaining. The saved decomposition
   attributes about 98.7% of that gap to Gamma/thrust stationarity. Both primal
   objectives remain about 3.631% above the saved qualified QOCO references.
   The subsequent fixed-metric restart trial fails and worsens cone feasibility.
   Inspect coefficient/conditioning effects before another numerical intervention. Preserve
   the original objective and gates, and require a qualified reference before
   CUDA integration. Do not repeat unchanged failed experiments.
2. **Establish the backend crossover.** Use fixed nontrivial zero/nonzero-Q
   captures, then held-out missions. Compare upstream, persistent PDHCG, GPU
   QOCO and CPU reference outputs under identical objectives and gates. Include
   cold setup, retained solves, all failed work, handoffs and certification.
   Only a qualified path advances into the native mission backend comparison.
3. **Broaden mission search in parallel.** The existing wider pool supplies no
   new positive standalone replacement, and the matched bonus-weighted variant
   also produces no profitable eligible standalone or paired replacement.
   Preserve the productive unit-weight arm and raw/weighted tradeoffs while
   testing different Earth seeds and master opportunity prices. Generate diverse new route columns,
   jointly modify conflicting ships, optimize deployment/collection timing,
   and explore feasible fleet growth. Preserve profitable weighted-score
   replacements that use the actual fleet raw-mass margin.
4. **Move productive recurring work into retained C++/CUDA.** With deploy
   admission now on CUDA, prioritize collection-tour preparation, scheduling
   and batched route completion, plus native workspace ownership and structured interval operators
   and compatible batches after measuring their complete cost. CPU file I/O
   and independent audit remain explicit and timed boundaries.
5. **Earn the SOTA claim.** Track certified trajectories/second, success rate,
   median/p95 complete latency, memory and best verified score versus elapsed
   time/GPU-hours. Compare published fleets under the same scoring coefficients
   and rules. The existing project speed thresholds are experimental gates,
   not proof of a world-best result.

Literature work is focused on the diagnosed gaps: the conic proximal map,
restart/accuracy policies and preconditioning, plus route construction and
fleet coordination. The [PDHCG-CQP implementation](https://arxiv.org/html/2608.09159v1#S4)
and [PDHCG-II](https://arxiv.org/abs/2602.23967) provide concrete candidates.
QP-specific changes require a correctness argument before use with cones.
Each change needs an ablation and a decision rule; another broad survey is
lower priority than these tests.

The [full execution roadmap](SOTA_EXECUTION_PLAN_2026-09-09.md) retains the exact
historical evidence, unchanged accuracy gates and repeated-benchmark rules.
