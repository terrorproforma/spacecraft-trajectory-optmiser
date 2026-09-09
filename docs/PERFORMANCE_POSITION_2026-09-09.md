# Performance conclusions and the role of PDHCG

The evidence supports keeping the PDHCG-inspired core and changing how we prove
and deploy its advantage. The current obstacle is reliable convergence to the
required accuracy. Separately, better GTOC12 scores require better discrete
mission choices. Neither obstacle is resolved by faster individual kernels.

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
cargo and certified prefix, are the next bounded schedule hypothesis.
[Initialization](GPU_MASS_SCALED_INITIALIZATION.md),
[implemented repair and matched outcome](GPU_MASS_MERIT.md),
[interval replay and exact scope](../results/local/2026-09-09/interval-defect-replay-v633/RESULTS.md).

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
   mass-merit repair and matched control are complete; the candidate still fails.
   The localized departure-interval velocity defect now motivates a bounded
   later-arrival test, preserving cargo and preceding certified flights. For
   the core, the joint dynamics/L1 reference has identified a numerical
   line-search obstruction. Validate the stable decrease calculation on tiny
   exact cases before another fixed capture comparison; CUDA integration
   requires a qualified reference. Preserve failed cases and stop repeating
   unchanged experiments.
2. **Establish the backend crossover.** Use fixed nontrivial zero/nonzero-Q
   captures, then held-out missions. Compare upstream, persistent PDHCG, GPU
   QOCO and CPU reference outputs under identical objectives and gates. Include
   cold setup, retained solves, all failed work, handoffs and certification.
   Only a qualified path advances into the native mission backend comparison.
3. **Broaden mission search in parallel.** Generate diverse new route columns,
   jointly modify conflicting ships, optimize deployment/collection timing,
   and explore feasible fleet growth. Preserve profitable weighted-score
   replacements that use the actual fleet raw-mass margin.
4. **Move productive recurring work into retained C++/CUDA.** Prioritize host
   search/packing, native workspace ownership, structured interval operators
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
