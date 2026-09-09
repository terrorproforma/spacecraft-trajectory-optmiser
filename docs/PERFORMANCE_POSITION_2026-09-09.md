# Performance conclusions and the role of PDHCG

The evidence supports keeping the PDHCG-inspired core and changing how we prove
and deploy its advantage. The current obstacle is reliable convergence to the
required accuracy. Separately, better GTOC12 scores require better discrete
mission choices. Neither obstacle is resolved by faster individual kernels.

## What is established

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
- Host work still matters. Retained catalogue fingerprints improve complete
  route-search throughput by **7.33–7.48% on H100** in the paired two-route
  experiment. Those are surrogate search candidates, not certified trajectories.
  Small fresh CUDA ephemeris batches are slightly slower than CPU preparation
  despite removing the CPU orbital arithmetic. Reuse and complete workload
  size determine whether the transfer is worthwhile.
  [Catalogue measurements](GPU_CATALOGUE_OWNERSHIP.md),
  [ephemeris measurements](GPU_ROUTE_EPHEMERIDES.md).
- The certified frontier remains **13,023.704901 weighted kg**, **14,291.006160
  raw kg**, 23 ships and 199 asteroids. The v630 additional route prescriptions
  produce no complete certified improvement. Its failed optimizer statuses do
  not prove those physical missions infeasible.
  [Fleet baseline](GPU_MASS_BUDGETED_FRONTIER.md),
  [failed prescriptions](GPU_FLEET_BUDGET_ADMISSION.md).

The native GTOC12 trajectory refinements currently use **GPU QOCO inside SCvx**.
The custom persistent PDHCG-inspired backend is still awaiting reliable
qualification and explicit integration into that path. Its implementation uses
explicit PDHG updates and does not yet implement the upstream conic quadratic
proximal inner solve. Recovery CGLS is a different operation. These boundaries
must remain visible when attributing performance or mission gains.

The latest fixed-date GPU return test regenerates the archived control in two
accepted iterations and independently certifies it. At the candidate's changed
starting mass, the GPU-scaled seed is dynamically consistent but finishes
10.28998 kg below the prescribed minimum mass. All 22 subsequent steps are
rejected. The complete two-case worker takes 10.787 seconds, with two native
solves and one fresh certificate; no full-fleet checks or score promotion occur.
Source inspection identifies missing state-mass-shortfall terms in the
reference acceptance merit. That omission is confirmed; whether correcting it
can certify this candidate still needs a new controlled experiment.
[Exact initialization and mission outcome](GPU_MASS_SCALED_INITIALIZATION.md).

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

1. **Repair reliable refinement and the measured numerical obstruction.** Add
   the missing state-mass constraint violations to the reference acceptance
   calculation, test a corrective step and its guards, then rerun the bounded
   original-mass control and changed-mass return with all physical gates fixed. For
   the core, address dynamics together with the original L1 virtual-control
   term; a joint proximal step or compatible structured preconditioner needs
   an independent mathematical reference before CUDA integration. Preserve
   failed cases and stop repeating unchanged experiments.
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
