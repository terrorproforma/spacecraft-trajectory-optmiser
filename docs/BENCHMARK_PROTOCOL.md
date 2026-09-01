# Benchmark protocol

Status date: **2026-08-31**.

This protocol governs all SpacePDHCG and OrbitWeaver performance results. A benchmark result is
not publishable unless its raw manifest, software revisions, hardware description, solver
parameters, correctness checks, and complete timing breakdown are stored together.

## Core rule

> Compare end-to-end accepted trajectories at matched nonlinear quality. Kernel throughput,
> solver status strings, and batched throughput alone are not trajectory-optimisation results.

No GPU or multi-GPU performance claim exists until the pinned workflows have run on recorded
hardware and the returned trajectories pass independent checks.

# Reproducibility record

Every run must record:

- SpacePDHCG repository commit;
- upstream PDHCG repository and exact commit;
- every comparison solver name, version, commit, and build flags;
- operating system and kernel;
- compiler, standard library, CMake, CUDA, driver, cuSPARSE, NCCL, and MPI versions;
- CPU model, socket/core/thread count, NUMA layout, and memory capacity/speed;
- GPU model, count, compute capability, memory capacity, power limit, interconnect, and topology;
- numerical precision and deterministic-mode settings;
- problem family, random seed, horizon, scenario count, cone inventory, and nonzero counts;
- requested tolerance, independently achieved residuals, and iteration limits;
- warm-start state and scaling/preconditioner refresh policy;
- all timings listed below;
- final objective and independently verified nonlinear feasibility;
- energy and memory measurements where available;
- failures, timeouts, out-of-memory events, and excluded samples.

The benchmark harness must refuse to label a result `spacepdhcg-persistent` unless the concrete
persistent CUDA backend is active. CPU references and one-shot upstream solves retain distinct
backend names.

# Timing decomposition

Report at least:

1. topology construction;
2. numerical coefficient generation;
3. workspace creation;
4. numerical update;
5. scaling or preconditioner refresh;
6. host-to-device transfer;
7. solver iterations;
8. residual evaluation;
9. nonlinear propagation and path checks;
10. trust-region and acceptance logic;
11. device-to-host transfer;
12. collective communication;
13. total convex-subproblem time;
14. total outer-SCvx time;
15. time per accepted trajectory;
16. throughput for a declared batch size.

For persistent solves, separate cold start, first warm solve, and steady state. The steady-state
window may begin only after JIT compilation, allocator warm-up, and one complete solver cycle.

# Correctness gates

## Convex CQP gate

For small and medium problems, compare against a high-accuracy Clarabel, OSQP, or other declared
reference. Report:

- canonical primal residual;
- canonical dual residual;
- complementarity/cone residual;
- objective gap;
- primal distance where a known optimum exists;
- solver status and whether it was residual-qualified.

## Nonlinear trajectory gate

Independently recompute:

- dynamics defects with a propagation method at least one order more accurate than the
  transcription update;
- terminal state error;
- path constraints between nodes where applicable;
- thrust, torque, pointing, mass, altitude, glide-slope, angular-rate, and quaternion conditions;
- virtual-control norm;
- actual versus predicted merit reduction.

The independent checker must not reuse the solver's cached residual buffers.

## Gate G4 matched-quality contract

Primary G4 records are qualified only when all of the following are present and pass the frozen
quality tier:

- solver status is `converged` and the convergence criteria are met; `max_iterations` is
  unqualified even when a reported residual is small;
- canonical primal, dual, cone, and gap residuals pass the selected tier;
- the objective is within the frozen absolute-plus-relative practical-equivalence margin;
- an independent higher-order replay passes continuous-time, dynamics, terminal, and
  virtual-control checks without reusing solver residual buffers;
- the complete family-specific path inventory is independently checked;
- requested and actual policy, quality tier, scaling mode, warm-start mode, re-solve rule, and
  frozen policy SHA-256 agree;
- the accepted-trajectory timing boundary and timing sum identities pass; and
- raw artifacts have immutable URIs plus content and internal-index SHA-256 digests.

Local-only historical evidence is retained, but portability is marked absent and the record is
unqualified until an immutable artifact URI is supplied.

## Gate G4 timing boundary

The common accepted-trajectory boundary begins with numerical coefficient generation and includes
workspace/update/scaling/transfers, solve, projected recovery, residual evaluation, hybrid
conversion/setup/polish when applicable, independent replay, and acceptance. CUDA context/JIT
startup is reported separately and explicitly excluded. `cqp_total_seconds` and
`scvx_total_seconds` must carry their component identities and equal the corresponding sums.

## Robust scenario gate

Report:

- non-anticipativity violation;
- scenario-local nonlinear feasibility;
- expected, worst-case, VaR, and CVaR metrics as applicable;
- monolithic versus partitioned operator agreement;
- scenario partition and load imbalance;
- collective payload, frequency, and measured communication time.

## Route gate

For OrbitWeaver, report:

- target and epoch counts;
- route objective and lower bound;
- optimality gap where an exact small-instance result exists;
- arc-oracle fidelity distribution;
- number of candidate arcs generated, screened, refined, cached, and rejected;
- route-search nodes/labels/columns expanded;
- final high-fidelity trajectory feasibility and resource margins.

# Statistical procedure

- Use at least five measured repeats after warm-up for deterministic steady-state latency.
- Use more repeats when the coefficient of variation exceeds 5%.
- Report median, interquartile range, minimum, and maximum; means may be supplementary.
- Randomised problem families use committed seeds and at least 20 instances per reported size.
- Timeouts and out-of-memory events remain in the dataset as censored failures.
- Do not select the best run from a parameter sweep without including tuning cost and the full
  sweep definition.

# Solver comparisons

The initial comparison set is:

- Clarabel CPU correctness reference;
- OSQP CPU QP reference;
- upstream PDHCG one-shot CUDA adapter;
- SpacePDHCG persistent CUDA backend;
- QOCO-GPU when its supported formulation matches;
- CuClarabel when available;
- custom PIPG/trajectory-structured baseline for selected problems.

A solver may be omitted only with a recorded technical reason, such as unsupported cone type,
unavailable build, or memory failure.

# Paper 1 experiment families

## P1-A — known-optimum trajectory-banded CQP

Purpose: isolate solver correctness and scaling without nonlinear outer-loop effects.

Sweep horizon, state/control dimension, QP versus SOCP, conditioning, requested tolerance, and
warm start. These fixtures have a committed exact optimum.

## P1-B — HCW rendezvous

Purpose: spacecraft-specific QP/SOCP baseline and fixed-pattern repeated updates.

Sweep horizon, terminal states, box versus norm-bounded thrust, and update magnitude.

## P1-C — nonlinear 3-DoF powered descent

Purpose: deterministic SCvx lifecycle, adaptive inner accuracy, and end-to-end persistent update
benefit.

Sweep horizon, initial dispersions, tolerance policy, warm start, and final polish.

## P1-D — nonlinear 14-state 6-DoF powered descent

Purpose: flight-relevant cone inventory, rigid-body coupling, quaternion constraints, and larger
CQP blocks.

Sweep horizon, attitude/rate dispersions, thrust/torque limits, and requested accuracy.

## P1-E — long-horizon low thrust

Purpose: expose the matrix-free crossover as horizon and sparse factorisation memory grow.

Sweep horizon from hundreds to tens of thousands of intervals, transfer family, trust radius,
and conditioning.

## P1-F — robust scenario powered descent

Purpose: scenario-axis scaling and scenario-aware multi-GPU partitioning.

Sweep horizon, scenarios, information-sharing prefix, risk metric, GPU count, and uncertainty
magnitude. Include deterministic, expected-value, worst-case, and CVaR evaluations.

# Paper 1 ablations

At minimum:

- one-shot versus persistent workspace;
- host coefficient fill versus native C++ fill versus device fill;
- cold start versus warm start;
- always rescale versus reuse/refresh-if-needed;
- fixed inner tolerance versus adaptive forcing;
- PDHCG only versus PDHCG plus interior-point polish;
- generic sparse partition versus whole-scenario partition;
- communication overlap on versus off;
- single versus double precision where correctness permits;
- CUDA graph capture on versus off.

# Paper 1 primary plots

1. End-to-end time against horizon for each solver.
2. Peak memory against horizon.
3. End-to-end time against horizon × scenario count.
4. Strong and weak multi-GPU scaling.
5. Accuracy–time Pareto surface.
6. Fixed versus adaptive inner accuracy: time, iterations, and final nonlinear quality.
7. Setup/update/solve/residual timing decomposition.
8. Communication versus computation by GPU count.
9. Energy per accepted trajectory.
10. Empirical regime map identifying CPU IPM, GPU IPM, PDHCG, and hybrid winners.

# Paper 2 experiment families

## P2-A — analytical arc screening

Compare Hohmann/phasing, zero-revolution Lambert, multi-revolution Lambert when implemented, and
Edelbaum low-thrust estimates against refined legs.

## P2-B — exact small moving-target routes

Use the native time-expanded graph and elementary label solver to establish truth solutions for
small target/epoch sets. Compare beam search and future column-generation methods.

## P2-C — deterministic multi-destination routes

Sweep target count, epoch resolution, visit count, spacecraft resources, arc fidelity ladder,
and route method.

## P2-D — multi-spacecraft assignment and routing

Sweep spacecraft count, depots, shared resources, service times, and target compatibility.

## P2-E — robust multi-destination missions

Attach scenario sets to promising arcs and routes. Measure the product of candidate-route and
uncertainty parallelism, final risk, and computational effort allocation across fidelity levels.

# Result storage

Raw benchmark output belongs under an external artifact store or GitHub Actions artifact, not in
large binary commits. The repository stores:

- the input manifest;
- a compact machine-readable summary;
- hashes and locations of raw artifacts;
- scripts that reproduce tables and figures;
- the exact code revision.

Generated result directories remain ignored until an explicit release snapshot is approved.
