# Paper 1 figure and table schema

This file freezes the primary visual evidence before GPU results exist. Plot scripts may improve
styling, but may not change inclusion, aggregation, axes, or failure treatment without revising this
schema and documenting the reason.

## 1. General data rules

Every plotted point must trace to one or more valid run manifests and a compact result record
satisfying `experiments/schema/paper1_result.schema.json`.

Primary timing statistic:

```text
point = median of measured repeats after warm-up
band  = [25th percentile, 75th percentile]
```

Also store minimum, maximum, repeat count, coefficient of variation, and censored failures.

Rules:

- never drop timeout, OOM, numerical failure, or quality-gate failure from the dataset;
- failures appear as boundary markers or failure-count annotations;
- no logarithm is applied before aggregation;
- no batch throughput is relabelled single-trajectory latency;
- all solver curves at one coordinate must meet the same nonlinear-quality gate;
- a solver unsupported for the cone inventory is `unsupported`, not `failed`;
- tuning runs are visually and statistically separate from locked evaluation runs;
- medians are not joined across missing or failed coordinates as though a result existed;
- source units are SI; display scales are explicit in labels;
- every figure caption names hardware, precision, warm-start state, and quality threshold.

Canonical filenames are `figNN_<slug>.pdf` for vector publication output and
`figNN_<slug>.png` for review. Source summaries are `figNN_<slug>.json`.

## 2. Figure F01 — architecture and residency

**Filename:** `fig01_architecture_residency`

Diagram, not measured data. Required blocks:

1. nonlinear C++ spacecraft model;
2. variational RK4 coefficient fill;
3. fixed CQP topology and mutable values;
4. persistent PDHCG workspace;
5. adaptive forcing/trust acceptance;
6. scenario shards and NCCL reduction;
7. independent nonlinear replay;
8. optional GPU IPM polish;
9. compact host diagnostics;
10. OrbitWeaver continuous-oracle boundary.

Use distinct arrows for one-time topology upload, per-iteration value updates, retained iterates,
and compact diagnostics. Do not imply implemented CUDA residency until H1 has a qualified run; use
`planned` styling before then.

## 3. Figure F02 — fixed topology, mutable values

**Filename:** `fig02_fixed_topology`

Diagram plus one empirical inset after GPU runs.

- left: CSC offsets/indices and cone metadata allocated once;
- right: Q/A/F values, objective, bounds, and iterates changing by SCvx epoch;
- inset x-axis: update number;
- inset y-axis: bytes uploaded or allocated after creation;
- expected persistent topology allocation line: zero after creation.

Required fields: `topology_bytes`, `numeric_bytes`, `bytes_h2d`, `allocation_count`,
`workspace_epoch`.

## 4. Figure F03 — scenario block-arrow decomposition

**Filename:** `fig03_scenario_partition`

Diagram, with optional measured communication inset.

- scenario-local diagonal blocks;
- shared-control/risk arrowhead;
- whole-scenario GPU assignment;
- local SpMV/cone work;
- collective reduction payload;
- logical \(G_s\times G_t\) grid.

Inset x-axis: scenario count; y-axis: measured collective bytes/time. Show analytic communication
model and measurement separately.

## 5. Figure F04 — end-to-end horizon crossover

**Filename:** `fig04_horizon_crossover`

One panel per deterministic family: P1-B, P1-C, P1-D, P1-E. Separate figures are allowed if the
combined result is illegible, but panel semantics may not change.

- x-axis: intervals \(N\), logarithmic;
- y-axis: median \(T_{\rm SCvx}\) in seconds, logarithmic;
- line/group: solver identifier;
- facet: requested/achieved quality tier;
- marker fill: warm versus cold;
- failure markers: OOM, timeout, failed nonlinear gate.

Do not plot inner `T_solve` as the primary y-axis. Annotate the first sustained crossover where a
candidate is faster for at least three consecutive coordinates and passes quality.

## 6. Figure F05 — peak memory crossover

**Filename:** `fig05_memory_crossover`

- x-axis: intervals \(N\) or \(N\times S\), logarithmic;
- y-axis: measured peak device bytes, logarithmic;
- line: solver;
- panels: deterministic and robust families;
- horizontal line: physical GPU memory;
- OOM points remain at the memory boundary with censored markers.

Report allocator-reserved and actively used bytes separately when available. Primary comparison uses
peak process/device allocation under the locked measurement method.

## 7. Figure F06 — adaptive-accuracy ablation

**Filename:** `fig06_adaptive_accuracy`

Three aligned panels:

1. total \(T_{\rm SCvx}\);
2. total inner work (`matvecs + cone projections`, with components retained);
3. final nonlinear quality (`max(r_dyn,r_path,r_term,r_na,r_risk)`).

- x-axis: policy (`fixed-tight`, `fixed-loose`, `adaptive`, `adaptive+polish`);
- group: problem size/family;
- points: individual instances;
- bar/line summary: median and IQR.

A policy that is faster but fails the quality threshold is not a winner and is visually marked as
unqualified.

## 8. Figure F07 — multi-GPU strong and weak scaling

**Filename:** `fig07_multigpu_scaling`

Two panels:

### Strong scaling

- fixed global \(N,S\);
- x-axis: GPU count \(G\);
- left y-axis: median total time;
- right or companion y-axis: efficiency
  \(\eta_G=T_1/(G T_G)\).

### Weak scaling

- fixed scenarios or nonzeros per GPU;
- x-axis: GPU count;
- y-axis: time and accepted-trajectory throughput.

Show scenario-aware and generic partitioning. Include communication fraction and load imbalance in
the source summary even if omitted from the primary rendering.

## 9. Figure F08 — end-to-end timing decomposition

**Filename:** `fig08_timing_decomposition`

Stacked bars for:

1. topology;
2. coefficient generation;
3. workspace creation;
4. update;
5. scaling refresh;
6. H2D;
7. solve;
8. residual;
9. nonlinear replay;
10. trust/acceptance;
11. D2H;
12. collectives.

Separate cold, first warm, and steady-state bars. Components must sum to reported total within the
manifest timing tolerance; the plotting script rejects inconsistent records.

## 10. Figure F09 — accuracy/time Pareto surface

**Filename:** `fig09_accuracy_time_pareto`

- x-axis: median total time, logarithmic;
- y-axis: achieved canonical residual or final nonlinear residual, logarithmic;
- colour/group: solver/policy;
- marker size: peak memory;
- marker shape: problem family;
- only nondominated qualified points receive a Pareto line;
- unqualified points remain visible but excluded from the frontier.

Do not mix canonical CQP residual and nonlinear trajectory residual in one panel. Use aligned panels.

## 11. Figure F10 — solver regime map

**Filename:** `fig10_solver_regime_map`

A categorical map derived from locked decision rules, not visual judgement.

Suggested coordinates:

- x-axis: problem scale \(N\times S\) or stored nonzeros;
- y-axis: requested achieved-quality tier;
- facets: cone family and hardware count;
- cell: winner among CPU IPM, GPU IPM, persistent PDHCG, hybrid, or no qualified winner;
- hatch: memory-censored comparison;
- annotation: number of qualified repeats.

Winner requires the rule in `CLAIMS_AND_DECISION_RULES.md`. Ties remain ties.

## 12. Figure F11 — variational sensitivity validation

**Filename:** `fig11_variational_validation`

Pre-GPU correctness figure permitted in the main paper or appendix.

- x-axis: randomly generated admissible state/control trial;
- y-axis: maximum absolute or relative difference between analytic variational RK4 and the
  independent finite-difference RK4 reference;
- panels: 3-DoF, 6-DoF, low-thrust;
- reference line: declared tolerance;
- companion metric: CQP coefficient-fill time for analytic versus finite-difference paths on CPU.

For 6-DoF, include maximum quaternion radial sensitivity
\(|\hat q^T\partial q/\partial\xi|\).

## 13. Figure F12 — robust residual anatomy

**Filename:** `fig12_robust_residuals`

- x-axis: outer iteration;
- y-axis: residual, logarithmic;
- lines: dynamics, path, terminal, virtual control, non-anticipativity, risk epigraph, canonical KKT;
- panels: expected, worst-case, CVaR;
- background: accepted/rejected phases and trust radius.

This figure is diagnostic and must not replace aggregate scaling results.

# Tables

## T01 — hardware and software manifest

Columns:

- machine identifier;
- CPU and RAM;
- GPU model/count/memory/power limit;
- interconnect/topology;
- OS/kernel;
- compiler/CMake;
- NVIDIA driver/CUDA/cuSPARSE/NCCL/MPI;
- SpacePDHCG commit;
- upstream solver commits;
- precision/determinism flags.

## T02 — problem dimensions

Columns:

- family/instance;
- \(N,S,n_x,n_u\);
- variables;
- scalar rows;
- affine rows;
- Q/A/F nonzeros;
- cone inventory;
- topology bytes;
- mutable numerical bytes.

## T03 — correctness

Columns:

- family/instance;
- solver;
- status;
- objective;
- objective gap;
- canonical primal/dual/cone/gap residuals;
- nonlinear dynamics/path/terminal residuals;
- virtual control;
- non-anticipativity/risk residuals;
- qualified boolean.

## T04 — persistence

Columns:

- problem/size;
- one-shot setup and total;
- persistent creation;
- first warm total;
- steady-state update/solve/total;
- post-creation topology allocations;
- H2D/D2H bytes;
- persistence overhead fraction.

## T05 — adaptive policy

Columns:

- family/size/policy;
- total time;
- outer iterations;
- accepted/rejected/re-solved steps;
- inner iterations/matvecs/projections;
- final objective and residuals;
- polish used;
- qualified.

## T06 — robust scaling

Columns:

- \(N,S,G\);
- partition method;
- load imbalance;
- collective count/bytes/time;
- local compute time;
- total time;
- throughput;
- peak memory/GPU;
- speedup and efficiency;
- qualified.

## T07 — regime/crossover summary

Columns:

- family and quality tier;
- first compute crossover;
- first memory crossover;
- winner below/above crossover;
- censored range;
- evidence count;
- decision confidence/qualification note.

## T08 — negative and mixed results

Columns:

- hypothesis;
- problem regime;
- observed failure or null result;
- quality status;
- likely mechanism;
- supporting artifact;
- whether the hypothesis is rejected, mixed, or unresolved.

# Machine-readable field requirements

Each compact Paper 1 result record contains:

- identity: schema version, run ID, commit, family, instance, solver, policy;
- dimensions: intervals, scenarios, GPUs, variables, rows, nonzeros, cone inventory;
- quality: objective, gaps, every canonical/nonlinear/robust residual, qualified;
- timing: every component in `docs/BENCHMARK_PROTOCOL.md`;
- work: iterations, matvecs, projections, factorisations, accepted/rejected/re-solved steps;
- memory/energy/communication;
- censoring status and reason;
- aggregation metadata: warm-up count, repeats, statistic, quantiles;
- artifact hashes/locations.

The JSON schema is `experiments/schema/paper1_result.schema.json`.

# Rendering requirements

- vector PDF is the publication source;
- text remains selectable and fonts embedded by the rendering environment, but font files are not
  committed;
- colour is not the sole carrier of meaning;
- log axes contain no zero; censored zero-like values use explicit floor annotations;
- captions state the quality gate and aggregation method;
- plots are generated from compact summaries, never manually edited numerical coordinates;
- every source JSON contains the exact filter query and list of contributing run IDs.
