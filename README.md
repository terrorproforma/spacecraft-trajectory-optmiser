# SpacePDHCG

**GPU-native C++/CUDA spacecraft trajectory optimisation, with independently verified physics.**

SpacePDHCG develops fast trajectory solvers for rendezvous, powered descent and
low-thrust interplanetary transfers, alongside a GTOC12 asteroid-mining fleet
planner. The objective is to keep the numerical pipeline on the GPU: generate a
starting trajectory, propagate dynamics, assemble and solve optimisation
subproblems, and decide which updates to accept without repeatedly moving the
trajectory through CPU memory.

The project includes a factorisation-free PDHCG-inspired backend and a GPU QOCO
interior-point backend. We compare complete solve time at the same verified
accuracy. Fully GPU-controlled execution and scalable multi-GPU trajectory
optimisation remain work in progress.

## What is PDHCG, and why use it for trajectories?

**PDHCG means Primal-Dual Hybrid Conjugate Gradient.** It is an optimisation method:
the primal variables describe a candidate solution, while dual variables enforce
its constraints. The original quadratic-programming method combines primal-dual
updates with approximate conjugate-gradient solves. Its conic extension,
**PDHCG-CQP**, uses projected-gradient inner iterations to handle constraints such
as thrust-vector norm limits. These are related methods; the conic implementation
does not simply use conjugate gradient for every inner solve.
[Original QP paper](https://doi.org/10.1287/ijoc.2024.0983),
[conic extension](https://arxiv.org/abs/2608.09159).

Trajectory optimisation repeatedly asks for a better thrust history while
satisfying dynamics, endpoint, mass and thrust constraints. Successive
convexification (SCvx) turns the nonlinear problem into a sequence of convex
subproblems around the current trajectory. PDHCG-CQP can solve those subproblems;
the dynamics model and independent trajectory replay still determine physical
accuracy.

The speed opportunity comes from how that repeated work is organised:

- **Parallel arithmetic:** matrix-vector products, vector updates and cone
  projections expose GPU parallelism without requiring a direct sparse matrix
  factorisation in the first-order backend.
- **Reuse between solves:** retain sparse structure, buffers and previous
  iterates as SCvx updates numerical coefficients. This avoids rebuilding and
  uploading essentially the same problem at every iteration.
- **Exploit the trajectory structure:** each time interval contributes a small
  dynamics residual, `x[k+1] - A[k] x[k] - B[k] u[k] - c[k]`. Independent interval
  work and neighbouring-state gathers provide a route to less generic sparse
  indexing and fewer atomic updates.
- **Spend iterations where they help:** solve early subproblems inexactly, then
  tighten accuracy or use interior-point polishing when required. Final physics
  and objective acceptance gates stay fixed.

The research contribution pursued here is the persistent GPU trajectory pipeline
around these methods. PDHCG itself comes from the authors credited below.
First-order iterations can be cheap but numerous on poorly conditioned problems;
an interior-point solver can still finish sooner. A faster kernel alone does not
establish a faster, equally accurate trajectory solve.

## Original sources and attribution

Our upstream integration source is **[Lhongpei/PDHCG](https://github.com/Lhongpei/PDHCG)**,
originally pinned to commit
[`167c8b72b4b96d2f94d405b8763e485514192b81`](https://github.com/Lhongpei/PDHCG/tree/167c8b72b4b96d2f94d405b8763e485514192b81).
The [integration contract](docs/PDHCG_INTEGRATION.md) records the mapping and pin.
Upstream PDHCG is Apache-2.0 licensed and credits
[Haihao Lu's cuPDLPx infrastructure](https://github.com/MIT-Lu-Lab/cuPDLPx).

- **Huang, Zhang, Li, Ge, Liu and Ye (2025):**
  [*A Restarted Primal-Dual Hybrid Conjugate Gradient Method for Large-Scale Quadratic Programming*](https://doi.org/10.1287/ijoc.2024.0983),
  INFORMS Journal on Computing. The original PDHCG method.
- **Li, Huang, Liu, Ge and Ye (2026):**
  [*GPU-Accelerated Conic Quadratic Programming with Local Linear Convergence under Strict Complementarity*](https://arxiv.org/abs/2608.09159).
  The PDHCG-CQP extension underlying this project's conic integration.

Upstream solver benchmark results belong to those publications. They are not
measurements of SpacePDHCG's complete spacecraft pipeline.

## Research question

Can a persistent, scenario-structured, multi-GPU PDHCG-CQP backend reduce the total time and memory required for large robust spacecraft successive-convexification problems while preserving nonlinear feasibility and final solution quality?

A conditional result is useful: the project will produce a reproducible crossover map showing when first-order multi-GPU conic quadratic optimisation wins, when factorisation-based GPU solvers win, and when a hybrid is best.

## Current status — 7 September 2026

Native C++/CUDA execution is implemented and tested on the local RTX 5090. The
persistent solver has parallel scaling and reductions, with cooperative
multiple-block execution for larger problems. The experimental GTOC12 native
path also runs Lambert/Kepler seed generation, interval propagation, numerical
assembly, SCvx merit and acceptance decisions, and trust updates on the GPU.
Its v107 regression passed **324 tests**, including independent physics checks.
The experimental v121 QOCO path also runs the complete IPM iteration loop under
GPU control. In six balanced local comparisons, median complete-transfer time
fell from **551 ms to 326 ms (1.69×)** against v117, with all 36 transfers passing
the same independent physics and final-mass gates. This is a named-fixture
measurement, not a fleet-score improvement or universal speedup.

**The complete optimiser is not yet fully GPU-controlled.** Initial sparse
topology/conversion, parts of QOCO setup and control, and native solver dispatch
still involve the host. The QOCO conditional-graph refinement path also has
unresolved sanitizer failures. Python remains available for orchestration,
reference solvers and independent verification. Whole-IPM graphs are currently
built per solve; retained setup and full SCvx orchestration remain unfinished.

See [GPU-native implementation and measured results](docs/GPU_NATIVE_OPTIMIZATION_PROGRESS.md),
[GPU seed and SCvx control](docs/GTOC12_GPU_NATIVE_CONTROL.md),
[GPU IPM loop and v121 measurements](docs/QOCO_GPU_IPM_LOOP.md), and
[QOCO device refinement](docs/QOCO_DEVICE_REFINEMENT.md) for implementation
boundaries, test evidence and limitations. Reported performance improvements
apply to their named fixtures; they are not universal speedup claims.

## GTOC12 score versus the published leaderboard

Our verified `fleet_master_v11` snapshot collects **14,047.8 kg**, using **23 ships**
and collecting from **194 asteroids**. Both the locally run official checker and
the independent verifier accept the final fleet. The unrounded internal total
is 14,047.802874743327 kg.

Compared with the [official leaderboard](https://gtoc12.tsinghua.edu.cn/competition/leaderBoard),
checked on **6 September 2026**, our **bonus-weighted score of 12,805.194 kg**
would slot into **9th place**. The leaderboard uses `sum_i B_i M_i`, not raw
returned mass. The independent report already records this weighted value;
the previous README's 8th-place claim incorrectly compared raw mass with weighted scores.

| Published position | Team / local result | Score (kg) |
|---|---|---:|
| 1 | Jet Propulsion Laboratory | 22,532.672 |
| 2 | BIT-CAS-DFH | 17,727.638 |
| 3 | OptimiCS | 17,081.861 |
| 4 | ESA's Advanced Concepts Team & Friends | 15,727.902 |
| 5 | TheAntipodes | 15,488.896 |
| 6 | NUDT-LIPSAM | 15,160.946 |
| 7 | ∑ TEAM | 14,714.133 |
| 8 | ATQ | 13,105.762 |
| **9th if inserted** | **SpacePDHCG — local verified snapshot** | **12,805.194** |
| 9 | ADL | 12,061.842 |

On this weighted comparison, we are **300.568 kg below ATQ**, **743.352 kg above
ADL**, and at **56.8% of JPL's winning score**. Matching JPL would require another
**9,727.478 weighted kg**. Our score uses the pinned frozen bonus table; historical
competition scores used the coefficients in effect for their submissions.

The physical haul averages **610.8 kg per ship**. The [ESA GTOC portal](https://sophia.estec.esa.int/gtoc_portal/?page_id=1261)
reports **719.8 kg/ship** for JPL's 35-ship competition winner and **742.95 kg/ship**
for Antipodes' 39-ship post-competition fleet (28,975.1 kg raw; 24,474.16 weighted kg).
The latter is the strongest published post-competition result listed there.
Counting each team once still places our result approximately ninth; treating
all three listed post-competition solutions as extra entries places it twelfth.

This is a retrospective comparison with the 2023 competition, not an official
ranked submission or proof of optimality. It measures fleet solution quality;
the recent single-transfer GPU speed tests do not establish a new fleet score
or show that this fleet was produced by the latest GPU-native optimiser.

Evidence: [fleet run report](results/lambda/2026-09-06/fleet_master_v11/run_report.json),
[official checker report](results/lambda/2026-09-06/fleet_master_v11/official_verification.json)
(the final `fleet/Result.txt` row),
[independent verification](results/lambda/2026-09-06/fleet_master_v11/independent_verify.txt),
[recomputed weighted score and per-asteroid contributions](artifacts/performance/gtoc12-v11-weighted-score-20260906.json),
and [web visualiser and loading instructions](results/lambda/2026-09-06/README.md).

## Programme ladder

1. fixed-pattern QP/SOCP correctness and cross-solver fixtures;
2. real CUDA execution through the one-shot PDHCG adapter;
3. persistent C++/CUDA ownership, in-place updates and device warm starts;
4. nonlinear 3-DoF powered-descent CT-SCvx;
5. adaptive inexact solves and interior-point polish;
6. robust 6-DoF scenario optimisation and multi-GPU decomposition;
7. Paper 1 crossover study and release;
8. multi-destination routing built on the finished trajectory oracle.

See:

- [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/COMPARATIVE_SOLVER_CAMPAIGN.md`](docs/COMPARATIVE_SOLVER_CAMPAIGN.md)
- [`docs/LITERATURE_TARGETS.md`](docs/LITERATURE_TARGETS.md)
- [`docs/REFERENCE_REPRODUCTION_REPORT.md`](docs/REFERENCE_REPRODUCTION_REPORT.md)
- [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md)
- [`docs/MILESTONES.md`](docs/MILESTONES.md)
- [`docs/PDHCG_INTEGRATION.md`](docs/PDHCG_INTEGRATION.md)
- [`docs/adr/0001-pdhcg-native-canonical-form.md`](docs/adr/0001-pdhcg-native-canonical-form.md)
- [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md)

## Package layout

```text
src/spacepdhcg/
  cqp/          fixed sparse-pattern native CQP representation
  models/       spacecraft dynamics and benchmark models
  backends/     persistent CPU references and accelerator adapters
  benchmarks/   reproducible latency, throughput and accuracy experiments
  planner/      user-facing planner (schema, CLI/API, CPU reference, viewer export)
  resources.py  locates frozen benchmark/spec assets (override, source checkout, wheel copy)
  _data/        byte-identical mirror of those assets so installed wheels can run every command

cpp/cuda/tools/spacepdhcg_plan.cu   native planner executable on the device SCvx stack
examples/planner/                   one runnable problem document per family
tests/          algebraic, solver and trajectory feasibility tests
docs/           research scope, decisions, architecture and milestone gates
```

## Commands

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest
spacepdhcg-cw-benchmark --repeats 20 --intervals 40
spacepdhcg-cw-socp-benchmark --repeats 20 --intervals 40
```

## Planner

`spacepdhcg plan problem.json --output out/` plans one trajectory (HCW rendezvous, 3-DoF or
6-DoF powered descent, low-thrust transfer) on the validated single-GPU SCvx stack and emits
node/dense histories, per-iteration telemetry, timings, and an independent-replay certificate.
`--backend cpu_reference` runs the clearly labelled Clarabel SCvx reference over the same
native transcription. See [`docs/PLANNER.md`](docs/PLANNER.md) and
[`examples/planner/README.md`](examples/planner/README.md).

The upstream PDHCG package is intentionally optional because it requires a compatible NVIDIA CUDA environment. CPU installation and CI do not import it.

## Frozen assets from an installed wheel

`spacepdhcg literature …`, `spacepdhcg gtoc12 …` and the other tools read their frozen JSON inputs
(literature registry/provenance/pins and profiles, GTOC12 rules/pins/reduced-instance rule, G4
policy/applicability/claim core with hash locks, campaign scopes, paper matrices, the provenance
schema) through `spacepdhcg.resources`, which looks in this order:

1. `SPACEPDHCG_BENCHMARKS_DIR` — an explicit `benchmarks/` directory; when set it is authoritative
   for every `benchmarks/...` asset (a missing file there is an error, never a silent fallback);
2. the source checkout containing the imported module (development trees, editable installs);
3. the copies packaged in the wheel under `spacepdhcg/_data/`.

`src/spacepdhcg/_data` is maintained by `python scripts/sync_packaged_assets.py` (`--check` in CI and
`tests/test_resources.py` prove every copy is byte-identical to the repository original). Large
pinned downloads are never packaged: GTOC12 data goes to `SPACEPDHCG_GTOC12_DATA`, the checkout's
`benchmarks/gtoc12/data`, or `$SPACEPDHCG_CACHE_DIR`/`~/.cache/spacepdhcg/gtoc12` (fetch with
`spacepdhcg gtoc12 fetch`); literature artefacts use `SPACEPDHCG_LITERATURE_CACHE`.

## Development status

Research software under active construction. Numerical results are not claimed until they are reproduced by committed benchmark configurations, independent feasibility checks and CI artifacts.

- [`docs/GTOC12_TRACK.md`](docs/GTOC12_TRACK.md) — GTOC12 asteroid-mining replay (pins, exact verifier, reduced instance, scored routes)

## Published Lambda fleet snapshot (6 September 2026)

The verified 23-ship GTOC12 fleet and partial G4 evidence are in
[`results/lambda/2026-09-06`](results/lambda/2026-09-06/README.md), including a
standalone copy of the existing web viewer and launch instructions. The fleet
collects 14,047.80 kg and passes both verifier reports. The G4 snapshot is partial
(145 completed groups); it is not a completed GPU performance qualification.
Current optimization evidence and remaining GPU-native work are tracked in
[GPU-native optimization progress](docs/GPU_NATIVE_OPTIMIZATION_PROGRESS.md).
