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

The insertion search now also covers routes that deploy at their turnaround and
return to collect that miner later. The widest two-route probe evaluates **499,380
surrogate schedules** on each GPU, including 85,428 previously skipped schedules.
It finds no feasible insertion, so the verified fleet score remains **12,843.556
weighted kg**. [Coverage fix, regression checks and fresh mission replays](docs/GPU_INSERTION_TURNAROUNDS.md).

Insertion route layouts are now constructed on CUDA from shared edge inputs.
The same 12,992-schedule screen takes **60 ms on RTX 5090 / 56 ms on H100**,
another **18.4× / 37.4× improvement** over host layout construction in the same
core. Full mission replays retain the score and pass both physics checkers.
[GPU layout generation, batch-size measurements and latest visualiser replay](docs/GPU_INSERTION_LAYOUTS.md).

CUDA now generates and screens insertion schedules in batches of different route
layouts. The measured 12,992-schedule neighbourhood runs **10.67× faster on RTX
5090 / 5.32× on H100**, at approximately **12,121 / 6,227 schedules per second**.
A complete mission replay reduces joint calls from 13,172 to 193, retaining the
same score and passing both full-fleet physics checks. These schedule evaluations
are search surrogates; native refinement and independent checks still qualify
accepted trajectories. [Measurements, remaining GPU work and latest replay](docs/GPU_INSERTION_BATCHES.md).

The latest verified fleet returns **14,044.353 raw kg / 12,843.556 weighted kg**,
with **610.624 raw kg per ship** across 23 ships. GPU-controlled timing searches
around the two recently replaced routes gained another **0.585 weighted kg**.
Both full-fleet physics checkers pass on RTX 5090 and Lambda H100. The new H100
mission is downloaded and displayed in the existing web visualiser.
[Current score, actual work, evidence and copy-paste loading instructions](docs/GPU_ROUTE_HILLCLIMB.md).

The preceding GPU fleet exchanges replaced
two routes, gaining **32.835 weighted kg** while returning slightly less raw
mass. Both replacement routes were freshly refined on GPU: all 33 native leg
solves converge on both RTX 5090 and Lambda H100, and both complete-fleet physics
checkers pass. [New score, exact work counts, results and visualiser loading
instructions](docs/GPU_FLEET_EXCHANGES.md).

The earlier orphan-recovery search returned 14,051.855 raw kg / 12,810.136 weighted
kg. The performance checkpoints below describe their own historical results.

An earlier 61-order GPU search evaluated 49,286 timing candidates in 2.157 seconds
without improving that incumbent. Six local return-leg diagnostic replays pass
unchanged physics checks but expose varying convergence for identical inputs;
that historical failure is addressed by the conditional retry below. [Current score, why progress
has been slow, remaining GPU work and displayed result](docs/GTOC12_PROGRESS.md).

Joint timing candidates can now be evaluated and ranked in native CUDA batches.
On two real incumbent routes, the measured controller stage runs **7.58–8.88x
faster on RTX 5090**, including candidate construction, geometry and transfers.
The final wrapper passes 62 tests with the current core and 54 with the older
core; the native implementation passes memory, race and synchronization checks.
The feature remains opt-in, with Python still orchestrating the wider search.
[Implementation, exact measurements and GPU/CPU boundaries](docs/GPU_JOINT_CANDIDATES.md).

The subsequent H100 validation measures **10.92–12.81x** for that joint-search
stage and passes 62 tests on each GPU. Eight complete fleet replays with the new
core pass both physics checkers, reproducing **12,810.136 weighted kg**. GPU
winner selection reduces final-result downloads by **99.42%** across the run;
whole-process timing remains too variable to claim a reliable additional speedup.
The retrieved H100 fleet is available in the existing visualiser.
[H100 evidence, repeated campaigns and loading instructions](docs/GPU_JOINT_ITINERARY.md).

An additional opt-in path keeps joint preflight, exact-epoch lookup and Lambert
geometry on the GPU through winner selection. It removes intermediate preflight
downloads and the Python loop over candidate legs. The tested stage is another
**2.13–4.77× faster on RTX 5090** and **3.20–6.40× on H100**; these are component
ratios against the previous batched CUDA wrapper. Both GPUs pass 71 tests and the
resident geometry sanitizer checks. [Scope, measurements and remaining CPU
work](docs/GPU_JOINT_GEOMETRY.md).

CUDA can now generate the ordered timing neighbourhood directly from incumbent
epochs. This adds **7–15% component speedup locally and 34–42% on H100**, with
87 tests per GPU and identical accepted search moves. Eight retained fleets pass
both checkers at the same score. Whole-process comparisons are confounded by
alternative-leg convergence failures; a fixed-input replay now reproduces that
failure, and neither disabling workspace reuse nor five Ruiz iterations fixes
it. [GPU mesh results, diagnosis and visualiser](docs/GPU_JOINT_MESH.md).

An optional CUDA-controlled conditioning retry now fixes that captured return
on both GPUs: six of six complete replays converge and pass independent physics
checks on each. Both GPUs pass five new tests and 117 existing regressions.
Matched 225-leg comparisons preserve all 205 qualified legs while using **7.1%
less solver time locally and 11.7% less on H100** (one paired run per GPU).
All 36 native legs converge in the new full campaigns, retaining the same
**12,810.136 weighted-kg** fleet. The retry adds no attempts or relaxed tolerances;
wider Python search control remains. [Implementation, evidence, timing limits
and downloaded H100 visualiser](docs/GPU_CONDITIONING_RETRY.md).

The whole fixed-order epoch search can now run inside one CUDA conditional
graph: initial evaluation, neighbourhood generation, geometry, winner acceptance,
mesh transitions and deadline checks remain on the GPU. Paired stage timings are
**1.37–1.60x faster locally and 2.29–2.78x on H100**, with exactly matching search
results. Both GPUs pass 111 tests. All four complete fleet replays pass both
physics checkers at the same score; joint host calls fall from **132 to 4**.
This completes GPU control of that search stage; broader route and fleet control
remain. [Measurements, sanitizer limits, evidence and visualiser](docs/GPU_DEVICE_EPOCH_SEARCH.md).

Native CUDA refinement now also uses CUDA DOP853 for per-leg certificates,
with retained buffers and no CPU propagation fallback. Sequential certificate
benchmarks on two emitted routes are **4.88–5.08x faster locally and 19.55–19.58x
on H100**. Four complete campaigns retain the verified fleet score. Independent
CPU fleet audits still dominate total runtime; the overall timing difference is
small. [Profiles, measurements, validation and downloaded results](docs/GPU_LEG_CERTIFICATION.md).

Initial CUDA fleet-packing measurements used the 2,492-column archived pool:
conflict counters make the same 35,145-node search **9.02x faster locally and
14.04x on H100** than the initial CUDA implementation (0.300 s and 0.256 s native
calls). Both GPUs pass 50 tests and full-pool memory, race and synchronization
checks. That checkpoint retained **12,810.136 weighted kg**. Search quality and
bounds then trailed the CPU LP-assisted master; these component ratios are not
whole-campaign or CPU-equivalent speedups. [Implementation, exact work counts,
limitations and retrieved evidence](docs/GPU_FLEET_MASTER.md).

Subsequent GPU exchanges improved the verified score to **12,842.971 weighted
kg**. The retained CUDA workspace now selects the same fleet in **50 ms locally /
54 ms on H100**, **2.03x / 2.32x faster** for repeated fixed-pool calls. Setup is
about 125 ms once with a warm runtime. Both GPUs pass 95 tests and full-pool CUDA
safety checks, including zero leaked allocations. This accelerates packing
existing routes; route generation and full physics validation are separate.
[Retained API, exact timings and retrieved evidence](docs/GPU_FLEET_WORKSPACE.md).

The CUDA backend also builds fleet scores, eligibility, ordering and
conflict/provider topology on the GPU. Existing one-shot CLI calls improve from
**95 to 72 ms locally** and **124 to 87 ms on H100**, with identical full-pool
score, selections and search counts. Both GPUs pass 115 tests; full-pool CUDA
checks find no errors or leaks. Python still serialises inputs and orchestrates
the broader mission search. [GPU setup and measurements](docs/GPU_FLEET_TOPOLOGY.md).

Parallel CUDA seed construction now reduces repeated selection to **7.49 ms
locally / 8.86 ms on H100**, **6.63× / 6.15× faster** than that preceding backend.
This is about **134 / 113 fleet selections per second**, each screening 179,205
logical packing proposals from 2,488 usable columns. Both GPUs pass 117 tests,
128 randomized exact comparisons and CUDA safety checks. The verified score is
unchanged; these are selections from existing routes, not new trajectory solves.
[Bottleneck profile, full-call measurements and retrieved evidence](docs/GPU_FLEET_SEEDS.md).

Cooperative CUDA tree search now reduces the same **35,145-node** retained call
to **16.29 ms locally / 11.74 ms on H100**, **16.36× / 18.25× faster** than the
seed checkpoint. Blocks share conflict state, track selected columns and compute
conservative bounds in parallel. Both GPUs pass 119 tests, 192 randomized exact
comparisons and full-pool CUDA safety checks. This accelerates search through
existing routes; the verified mission score is unchanged.
[Tree implementation, rejected prototype, measurements and replay](docs/GPU_FLEET_TREE.md).

The richer family32 search reproduces the historical **641.068 kg first
ship** on both GPUs. Its verified three-ship fleet returns 1,587.269 raw kg /
1,420.909 weighted kg; it cannot improve the historical 23-ship fleet. Selection
now consistently uses verified weighted scores, and archived standalone route
variants are retained for future fleet searches. At that checkpoint, adding a
24th ship without changing the retained routes required **861.637 raw kg**;
the improved fleet reduces this to **857.585 raw kg**. [Score audit, completed runs and displayed H100
fleet](docs/GPU_FAMILY_REPRODUCTION.md).

CUDA Lambert screening now uses safeguarded interpolation with bisection fallback.
The matched one-ship campaign takes **15.05% less time on RTX 5090** (29.06 →
24.68 s median). H100's median falls **2.78%** (29.81 → 28.98 s), with overlapping
timing ranges. Isolated screening reaches **22.42 million transfers/s on H100**
at 65,536 transfers per batch; these are screening estimates, not certified
low-thrust solutions. Final builds pass 46 tests and three Lambert sanitizer
modes on each GPU. All ten campaign runs pass both mission verifiers, retaining
548.254620 weighted kg. The best fleet remains **12,805.194 weighted kg**.
[Method, measurements, archived results and viewer](docs/GPU_FAST_LAMBERT_ROOTS.md).

An opt-in small-batch kernel now solves short/long Lambert directions in separate
GPU warps. It passes **106 tests per GPU**, exact hop-result comparisons and all
three Lambert sanitizer modes. Complete campaign medians improve 1.57% locally /
7.62% on H100, but ranges overlap, so the default remains unchanged. Larger-batch
layouts that regressed are retained in the evidence archive.
[Parallel-direction measurements and limitations](docs/GPU_PARALLEL_DIRECTIONS.md).

Initial bound classification now also runs on CUDA, downloading one byte per
bound pair instead of four floating-point arrays. First-fresh-solve downloads
fall from **515,324 to 244,499 bytes**. Both GPUs retain all **205 certified
legs**, pass 154 broad tests and 77 final-build tests. Complete campaign medians
improve by 0.64% locally / 2.90% on H100. The local replay is 0.30% slower;
two H100 replays per mode show 2.18% more solver time with overlapping ranges.
These observations do not establish an overall speedup.
[Implementation and reproducible results](docs/GPU_BOUND_CLASSIFICATION.md).

Fresh native trajectory workspaces now initialize numerical values from GPU
buffers, removing their initial CPU conversion round trip. The first campaign
solve downloads **61.10% fewer bytes**, and the 225-leg replay removes **72 host
priming dispatches** while retaining all 205 certified legs on both GPUs. Final
builds pass 65 tests per GPU and the selected H100 memory check. Full campaign
timings remain flat; the local full replay is 2.98% slower and H100 is 0.45% faster.
This is a GPU residency improvement, not an established overall speedup.
[Implementation, measurements and retrieved evidence](docs/GPU_DEVICE_INITIALIZATION.md).

Return and collection options now stay on CUDA through selection and return
pruning. The measured one-ship run eliminates **66.77 MB of option uploads** and
reduces generation-result downloads from **27.18 MB to 9.1 kB**. Complete-process
medians fall **8.99% locally / 2.28% on H100** over two runs per mode, though H100
timing ranges overlap. All 5,930 captured selections retain bitwise results on
both GPUs. Validation also fixed adaptive independent replay crossing changes
in cubic thrust interpolation; no solver or physics tolerance was relaxed.
The reverified incumbent remains **12,805.194 weighted kg**.
[Implementation, verifier diagnosis and reproducible results](docs/GPU_RESIDENT_OPTIONS.md).

An opt-in retained-graph experiment moves the first solve of reused trajectory
workspaces into GPU replay. All 107 validation tests pass on both GPUs. Complete
campaign timings overlap; the local 225-leg replay preserves all 205 certified
legs but is essentially flat in total solver time. The experiment remains
disabled by default. [Measurements and limitations](docs/GPU_RETAINED_REPLAY.md).

Full-workload tests reject enabling two Ruiz scaling passes globally: certified
legs fall from 205/225 to 191 locally and 192 on H100, and complete campaigns take
2.56x / 2.75x as long. An identical saved convex subproblem isolates a scaled
solver accuracy failure. Production settings and acceptance tolerances remain
unchanged. [Diagnosis and reproducible negative results](docs/GPU_CONDITIONING_DIAGNOSIS.md).

An opt-in objective-preserving Ruiz policy now fixes that captured failure in
27 inner iterations and retains all 205 certified legs in the full replay on
both GPUs. All 107 broader tests and three native sanitizer modes pass on each.
It has not established an overall speedup, so zero Ruiz remains the default.
[Implementation, accuracy checks and complete-workload measurements](docs/GPU_OBJECTIVE_PRESERVING_RUIZ.md).

An opt-in extension now reuses compatible objective-preserving scaled workspaces.
The fixed 225-leg replay takes **9.67% less solver time locally / 4.84% less on
H100**, retaining all 205 certified legs while reducing workspace creations from
225 to 72. All 117 tests pass on each GPU. Complete campaign timings remain flat
against production settings; full-solver sanitizer failures also reproduce in
preceding builds and remain unresolved. The extension stays disabled by default.
[Measurements, validation limits and retrieved results](docs/GPU_SCALED_WORKSPACE_REUSE.md).

Compatible trajectory legs now reuse GPU QOCO workspaces, sparse conversion and
vendor graphs. Complete-process one-ship medians fall **5.40% on RTX 5090** and
**5.03% on H100**, with unchanged verified score. The fixed 225-leg replay takes
**8.43% / 8.83% less solver time**, retaining all 205 certified legs. Final builds
pass 54 targeted tests and the workspace-rebind sanitizer probe on both GPUs.
Reuse defaults on for eligible zero-Ruiz graph execution. The fleet incumbent
remains **12,805.194 weighted kg**.
[Implementation, stream-lifetime fix and reproducible evidence](docs/GPU_SOLVER_WORKSPACE_REUSE.md).

New solver phase measurements identify **7.87 s locally / 9.49 s on H100** in
host-dispatched priming across a 47-call campaign. An early CUDA Graph entry
experiment reduces priming dispatches from 141 to 94 and retains all 205
certified legs in the fixed replay, but the H100 replay is **12.06% slower**.
It remains **disabled by default**. The best fleet score is unchanged.
[Phase breakdown, validation and retained negative results](docs/GPU_SOLVER_PRIMING.md).

The CUDA SCvx controller now stops stationary unsuccessful attempts earlier,
while preserving all 205 converged legs in a fixed 225-leg replay on both GPUs.
Replay solve time fell **20.3% on RTX 5090** and **19.5% on H100**. A separate
paired H100 complete-campaign comparison fell from **43.14 to 38.66 seconds**
(10.38% less time), with unchanged search counts, score and both mission checks.
The wider local confirmation retained four ships and 2,088.669 weighted kg in
247.18 seconds. [Stopping rule, accuracy evidence and reproducible results](docs/GPU_STATIONARY_FAILURE.md).

Return-window and collection-hop search now also computes body ephemerides on
CUDA, moving another **1.13 million transfer requests per one-ship run** off CPU
ephemeris calculations. Verified scores and logical search counts are unchanged.
Two runs per mode show overlapping timings, so this ephemeris migration is a GPU-residency
improvement without an established overall speedup.
[Implementation, profiles and accuracy evidence](docs/GPU_SEARCH_EPHEMERIDES.md).

Return and collection search now also sums delta-v, filters invalid options and
orders return candidates on CUDA, preserving exact option values and tie order.
These paths download **66.7% less result data** in the one-ship fixture. A separate
paired H100 comparison reduced complete-run median time from **37.55 to 34.29 s**
(8.69% less time); the RTX 5090 comparison was effectively flat at **30.66 to
30.61 s**. All search counts, scores and final physics checks are retained.
The wider local confirmation returned four ships and 2,088.669 weighted kg in
231.98 seconds. Python still constructs schedule axes and orchestrates the beam
and fleet. [Implementation, limits and reproducible evidence](docs/GPU_COMPACT_SEARCH_OPTIONS.md).

Initial Earth-beam ephemerides, screening, physical proxy scoring and ranking
now also run on CUDA. Isolated beam construction is **1.77x faster on RTX 5090**
and **11.67x faster on H100**, returning the same 582 candidates while reducing
that stage's result downloads from **248.3 MB to 27.9 kB**. Across four complete
runs per mode, combined median runtime falls **2.00% locally** and **7.38% on H100**;
timing ranges overlap because later refinement varies substantially. Both mission
checkers accept every run. A wider confirmation retains four ships and
2,088.669 weighted kg in 221.56 seconds. Pool filtering, route objects and fleet
orchestration remain host work. [Measurements, limits and validation](docs/GPU_EARTH_BEAM.md).

Collection tables now write their final device values directly from double-precision
Lambert results, removing temporary full-grid cost and feasibility arrays and
reusing axis buffers. Replaying 1,122 real grids (4.64 million cells) produces
bitwise-identical tables on both GPUs, with **2.75% less construction time locally**
and **7.33% less on H100**. Complete-run timing varies; the H100 comparison is
effectively flat, so this does not establish an overall speedup. The existing
float32 table storage and all physics gates are unchanged.
[Implementation, measurements and reproducible evidence](docs/GPU_FUSED_COLLECTION_TABLES.md).

A larger H100 campaign generated **32 individual routes** in
**56 min 59 sec**, evaluating **1.417 billion transfer branches** and **137.35
million collection options**. Fleet selection returned **15 ships, 105 mined
asteroids and 7,802.295 weighted kg**, accepted by both final mission checkers.
Bounded CUDA recovery prevents the earlier premature stop after three ships;
Python still orchestrates the fleet. The separate best fleet remains
**12,805.194 weighted kg**. [Implementation, evidence, visualiser and copy/paste loading instructions](docs/GPU_FLEET_RECOVERY.md).

Cooperative Lambert screening now distributes each transfer's bracket scan
across a warp. Complete-run measurements at unchanged accuracy are:

| Hardware / fixture | Previous runtime | Cooperative runtime | Speedup |
|---|---:|---:|---:|
| RTX 5090, one ship (two runs per mode, medians) | 75.39 s | 42.67 s | 1.77× |
| RTX 5090, four ships (one run per mode) | 469.53 s | 287.11 s | 1.64× |
| H100, one ship (two runs per mode, medians) | 53.50 s | 44.00 s | 1.22× |

Each comparison preserves the search counts and passes both final mission
checkers. Candidate throughput on the RTX 5090 is about **1.06 million transfer
branches/s** for the one-ship fixture and **591,000/s** for the four-ship fixture,
using complete-run elapsed time. These are transfer candidates, not fully solved
missions; the runs evaluate 45.19 million and 169.75 million branches respectively.
[Accuracy checks, measurements and reproducible archives](docs/GPU_COOPERATIVE_HOP_SCAN.md).

Retiming now also reuses immutable transfer tables in a bounded GPU cache.
A separate paired one-ship comparison reduced RTX 5090 runtime from **43.45 to
36.00 seconds** (1.21×), with the same verified score. H100 showed only about
1% less time over two samples per mode, which does not establish a significant
speedup there. The cache serves 736 of 821 table requests in the measured local
campaign using about 12 MiB of retained payload. Logical screening counters now
include reused tables; they are not counts of fresh GPU solves.
[Implementation, limits and validation](docs/GPU_RETIMING_GRID_CACHE.md).

Resident harvest-window pricing also removes full collection-table downloads
from the measured campaign. The H100 comparison showed **1.92% longer** median
runtime over two samples per mode, so this is improved GPU residency, not an
established speedup. [Measurements and accuracy checks](docs/GPU_HARVEST_WINDOW.md).

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

The current persistent CUDA implementation uses explicit PDHG updates: it applies
the quadratic gradient in one projected primal update per outer iteration. It does not yet
implement the upstream conic quadratic proximal inner solve. Recovery CGLS solves
constraint systems and is a separate operation. For the two captured GTOC12
problems tested so far, the Hessian is zero, so the absence of a quadratic inner
solve cannot explain their convergence failures. Qualified-point replay now
separates accurate input mapping, the native stopping predicate, and subsequent
objective-gap drift. [Measured diagnostic and implementation boundary](docs/GPU_PERSISTENT_CAPTURE_REPLAY.md).

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

## Current status — 8 September 2026

**Development SOC step correction:** a live QP returned zero step where a
100-digit calculation permits **0.55347**. Compensated GPU coefficients fix
the captured false-zero case. Final RTX/H100 builds pass 14 boundary cases,
four probe sanitizer modes and 38 trajectory/CLI tests each. Four matched H100
campaigns return the same verified **548.255 kg** in median **53.01 s versus
54.10 s**; the final packaged v360 confirmation takes **54.01 s** and is now
downloaded and displayed. Exact-QP qualification failures remain, and the fleet
incumbent is unchanged at **12,805.194 weighted kg**.
[Implementation and limits](docs/GPU_SOC_STEP_ACCURACY.md) ·
[Results and visualiser instructions](results/lambda/2026-09-08/gpu-soc-step-v359/README.md).

**Development refinement fix:** the GPU no longer accepts a NaN correction as an
improvement or overwrites its finite backup. Four matched H100 campaigns reduce
median complete time from **59.18 to 53.08 s (10.31% less time)**, with the same
548.255 kg and both checker passes. The new v342 result is downloaded and displayed.
Exact-QP qualification failures remain; this is a confirmed controller fix and
measured performance improvement, not a completed solver-reliability repair.
[Validation, raw evidence and loading instructions](results/lambda/2026-09-08/gpu-nonfinite-ir-v337/README.md).

**Development GPU graph comparison:** the public CLI now selects native SCvx/QOCO
graph execution with `--gpu-execution graph` (or automatically for CUDA SCvx).
Four matched H100 campaigns reduce median complete time from **60.83 to 58.85 s**
while retaining **548.255 kg** and both mission checker passes. Both GPUs pass
38 CLI/refinement checks. The new result is downloaded and displayed as
**GPU graphs v332 (548 kg certified)**. Inner-solver qualification outliers remain
under investigation; this development candidate has not been merged into main.
[Evidence, residual-arithmetic diagnosis and viewer loading](results/lambda/2026-09-08/gpu-execution-v328/README.md).

**Collection-table ephemerides now run on CUDA:** matched H100 campaigns take
**60.41 s versus 62.93 s** with host table preparation (**4.0% less time**, two
runs per mode). All four preserve the **548.255 kg** mission and pass both
checkers. Cold eight-asteroid table builds take **135 ms versus 295 ms** on H100.
Both GPUs pass 71 tests, eight independent real-tour replays and clean table
memchecks. Host table caching/packing and broader orchestration remain.
[Implementation, scope and downloaded results](docs/GPU_COLLECTION_TABLES.md).

**Collection-tour search now runs across CUDA blocks:** matched H100 campaigns
take **62.53 s versus 90.59 s** with CPU collection DP: **1.45× throughput and
31.0% less runtime**, using two runs per mode. All four runs retain the same
**548.255 kg, eight-asteroid mission**, accepted by both checkers. Exact-mass
pricing, camping prefixes and backtracking run on device; table preparation,
between-pass control and fleet orchestration still include CPU work. This is
not yet a completely GPU-native application. The fleet incumbent remains
**12,805.194 weighted kg**. [Implementation, validation and reproducible results](docs/GPU_COLLECTION_DP_CUDA.md).

**New certified GPU campaign:** the H100 search/refinement/retiming run completed
in **89.55 s**, evaluated **44.72 million transfer branches** and selected a
**548.255 kg, eight-asteroid mission**. Both checkers pass: **4.13% more returned
mass** than the previous development mission. The separate 23-ship incumbent
remains **12,805.194 weighted kg**. The new result is downloaded and available in
the visualiser as **GPU campaign v269 (548 kg certified)**.
[Statistics, evidence and copy-paste loading instructions](results/lambda/2026-09-08/gpu-native-campaign-v269/README.md).

**Parallel retiming completion:** four warps select the final epoch, threads
gather selected-leg values and share the best-candidate copy. Complete cached
retiming takes **11.9% less time on H100 and 13.1% less locally** against the
published device driver; fresh-table timing changes by less than 1%. **137 tests
pass on each GPU**, and all four H100 sanitizers pass 74 cases with zero errors
or hazards after adding explicit block-completion barriers. Both physics checkers
accept the 526.489 kg replay; fleet score remains unchanged.
[Measurements, rejected variant and reproduction](docs/GPU_PARALLEL_RETIMING_COMPLETION.md).

**Device-controlled retiming:** a conditional CUDA graph now performs mass-profile
correction, price bracketing/bisection and best weighted-candidate selection.
Complete retained-table retiming takes **40.0% less time on H100 and 32.3% less
locally** against the host driver with GPU forward accounting. The trace confirms
one submission/download for all six price evaluations. **131 tests pass on each
GPU**, all three H100 sanitizers pass 68 cases with zero errors, and the new
526.489 kg mission passes both physics checkers. Fresh-table runtime changes
little and fleet score is unchanged. CPU setup, route construction and fleet
orchestration remain. [Measurements, accuracy and reproduction](docs/GPU_RETIMING_DRIVER.md).

**GPU forward mass accounting:** mining yield, forward masses, propellant and
mass-budget checks now execute inside the CUDA retiming completion kernel.
The complete retained-table pricing loop takes **23.9% less time on H100 and
8.3% less locally**; fresh-table runtime is essentially unchanged. **111 tests
pass on each GPU**, three H100 sanitizers each pass 48 cases with zero errors,
and both physics checkers accept the new 526.489 kg mission replay. Fleet score
remains unchanged. Price/mass iteration control and fleet orchestration still
need GPU ports. [Measurements, accuracy and reproduction](docs/GPU_FORWARD_RETIMING.md).

**Parallel flight-time selection:** one CUDA warp now evaluates each arrival's
flight-time candidates, preserving exact tie order. Cached pricing is **2.19×
faster on H100 and 2.86× locally**; complete retiming gains **1.040× / 1.023×**.
At five-day grid spacing, cached pricing gains **4.57× H100 / 4.40× RTX 5090**.
The H100 trace measures an **8.8× faster leg-selection kernel**. **99 tests pass
on each GPU**, all three H100 sanitizers report zero errors, and the 526.489 kg
mission passes both physics checkers again. Fleet score remains unchanged.
[Measurements, scaling and remaining work](docs/GPU_WARP_RETIMING.md).

**Single-download retiming output:** consolidating six result transfers into one
reduces cached pricing time by **4.4% on H100 and 15.0% on RTX 5090** against the
published graph runtime. Complete retiming changes by less than 1%. **93 tests
pass on each GPU**, H100 sanitizers report zero errors, and the 526.489 kg mission
passes both physics checkers again. A new H100 CUDA trace attributes **83.7% of
retiming kernel time to leg selection**, making its serial flight-time loop the
next parallelisation target. [Measurements and evidence](docs/GPU_PACKED_RETIMING_OUTPUT.md).

**CUDA graph retiming:** a retained graph replaces 27 per-stage kernel submissions
for a 13-leg price evaluation. Matched comparisons against the published runtime
reduce cached pricing time by **3.3% on H100 and 4.0% on RTX 5090**; complete
retiming changes by less than 1%, so there is no material overall speedup claim.
**91 tests pass on each GPU**, both H100 sanitizers report zero errors, and the
526.489 kg mission passes both checkers again. Native execution, including copies
and synchronization, accounts for most measured cached-DP time and is the next
profiling target. [Measurements and remaining work](docs/GPU_RETIMING_GRAPHS.md).

**Resident return-sweep pricing and 526 kg mission:** nearest measured-return
selection, refusal masks and inflation pricing now run in CUDA, with compact
sweep updates reusing the transfer tables. The measured sweep workload takes
**63.69 ms on H100 versus 72.47 ms with host tables (1.138×)** and **224.42 ms on
RTX 5090 versus 234.61 ms (1.045×)**. The resulting six-asteroid mission passes
both checkers at **526.489 kg**, adding 2.464 kg to the previous 524 kg mission.
**88 tests pass on each GPU**, H100 sanitizers report zero errors, and the new
mission is loaded in the web visualiser. Fleet score remains **12,805.194 weighted
kg**. [Implementation and limits](docs/GPU_RETURN_SWEEPS.md),
[downloaded solution and viewer instructions](results/lambda/2026-09-08/gpu-sweeps-v234/README.md).

**Resident GPU retiming tables:** the common fixed-order path now builds endpoint
states, Lambert requests and transfer-cost tables directly in device memory,
then downloads only the selected schedule and one cost per leg. The full retiming
fixture improves from **71.97 to 63.79 ms on H100 (1.128×)** and **234.06 to
223.83 ms on RTX 5090 (1.046×)**. All selected schedules match the reference;
**84 tests pass on each GPU**, and H100 sanitizers report zero errors. The
524.025 kg mission passes both physics checkers again, with zero host table
uploads. Custom tables, return sweeps and CPU orchestration remain separate work.
[Implementation, measurement scope and evidence](docs/GPU_RESIDENT_RETIMING_TABLES.md).

**Certified route objective fixed:** the retiming certification loop now selects
its preferred route using actual bonus-weighted payload and the configured orphan
credit, consistently with its planner, instead of raw kilograms. Payload reduced
during refinement is accounted for. Seventeen targeted tests pass on both RTX
5090 and H100. All 23 incumbent source archives are now reconciled; a 46-case
cluster retiming scan completes in 14.33 seconds with no improving schedule. A
finer-grid candidate predicts +0.425 weighted kg but fails H100 departure
refinement, so the fleet stays at **12,805.194 weighted kg**.
[Diagnosis, source archives and failed-candidate evidence](docs/GTOC12_CERTIFIED_OBJECTIVE.md).

**GPU schedule dynamic programme:** camp choices, thrust-authority checks,
propellant pricing, arrival choices and path reconstruction now run in CUDA,
with retained transfer tables across price/mass updates. Repeated fixed-order
price evaluations are **15.08× faster on H100** and **4.03× on RTX 5090**.
Including transfer-table construction, the same full retiming fixture gains
**2.00× H100** and **1.10× local** over the previous GPU-endpoint/CPU-DP path.
The chosen schedules and objectives match the reference, and the 524.025 kg
mission passes both physics checkers again. **82 local and 75 H100 tests pass**;
12 DP parity cases pass both H100 sanitizers. The fleet score is unchanged.
[Scope, negative fleet-search results and reproducible evidence](docs/GPU_RETIMING_DP.md).

**Certified retiming improvement:** the latest six-asteroid mission returns
**524.025 kg**, up **10.1%** from 475.975 kg. All 13 legs pass the locally run
official checker and the independent verifier on H100, including a repeat with
the new GPU ephemeris path. This improves one mission; the incumbent 23-ship fleet
remains **12,805.194 weighted kg**. The remaining failed 180-day hop also passes
when given 185 days; CPU and GPU agree on the original residual plateau.
[Downloaded mission, visualiser and copy/paste instructions](results/lambda/2026-09-08/gpu-retiming-v213/README.md).

**GPU retiming tables:** endpoint Kepler propagation and Lambert request assembly
now run in CUDA. The same 412,116-branch retiming fixture takes **0.257 s on RTX
5090 versus 0.659 s previously (2.57×)** and **0.145 s on H100 versus 0.364 s
(2.51×)**, with identical selected schedules and mining yield. These are repeated
retiming measurements, not full mission speedups. **46 local and 30 H100 tests
pass**; the seven new cases also pass H100 memcheck and initcheck with zero errors.
Fleet orchestration and price/mass iteration remain CPU work.
[Implementation, measurement scope and evidence](docs/GPU_RETIMING_ELEMENTS.md).

**Real-mission departure reliability:** a 500-day Earth leg exposed repeatability
failures missed by the earlier coast tests. Identical first subproblems produced
different QOCO outcomes. Increasing low-thrust factor regularization to **1e-8**
passes **32/32 repeated real departures on each GPU**, keeping original-equation,
objective and independent physics checks unchanged. **328 local tests and 173
H100 tests pass**, including the new ordinary/CUDA-Graph regression. The CLI also
stops treating a failed SCvx attempt as a proof that other candidates are infeasible.
See [diagnosis, rejected configurations and evidence](docs/GPU_REAL_DEPARTURE_STABILITY.md).

**Corrected complete H100 run:** **18,484,006 Lambert branches**, **106 route
candidates**, three refinement attempts and **two certified routes in 77.28 s**.
The selected 13-leg, six-asteroid mission returns **475.975 kg** and takes **6.35 s**
to refine. The pre-fix run took 91.52 s and certified one route; these are single-run
observations, not a universal speedup. A later leg in the third candidate fails
at its original timing; the retiming diagnosis above supplies a flyable alternative.
The incumbent fleet score stays
**12,805.194 weighted kg**. Both missions are selectable in the web visualiser:
[downloaded result, full path and loading instructions](results/lambda/2026-09-08/gpu-native-campaign-v209/README.md).

**GPU collection pricing and selection:** thrust-authority checks, inflated
rocket-equation propellant and lost mining yield now run on CUDA for collection
options, with the original ordered near-tie decision rule. The 1,000-asteroid H100
search takes **1.54 s versus 12.01 s on CPU (7.82×)** with matching candidates.
The direct comparison with the previous GPU path adds **1.19× on H100**; the local
gain is only **1.02×**. CUDA prices **153,799 collection/return options per search**.
The integrated libraries pass **323 local GTOC12 tests and 168 H100 GPU tests**;
30 H100 screening tests and sanitizer checks also pass. Fleet score unchanged.
See [measurement scope, limitations and reproduction](docs/GPU_COLLECTION_SELECTION.md).

**Previous GPU neighbour selection checkpoint:** orbital filtering, Kepler positions, phasing costs,
stable ranking and deduplication now run on CUDA for route-search neighbours.
The 1,000-asteroid search takes **1.86 s on H100 versus 12.33 s on CPU (6.64×)**,
retaining the same 27 proxy routes and **545,658 Lambert branches per search**.
Compared with the previous CUDA-Lambert/CPU-neighbour path, this adds **1.15×
local and 1.49× H100**. Four warm queries across all 60,000 asteroids are **8.10×
local and 24.08× H100** faster with identical selected IDs; cold GPU setup is slower.
All **319 local GTOC12 tests and 26 targeted H100 tests pass**. The fleet score
is unchanged; collection/fleet orchestration and seed/retiming work remain on CPU.
See [scope, tests and reproduction](docs/GPU_NEIGHBOUR_SELECTION.md).

**Independent propagation now has a batched CUDA backend:** all 413 legs of the
existing fleet were propagated in about **30 ms on RTX 5090 and 12.4 ms on H100**
(warm batches, including transfers; parsing and mission checks excluded).
Every leg was compared with the CPU verifier; the largest position difference
was below 0.19 m. Use `gtoc12 verify --propagation-backend cuda` with the native
library and pinned data configured. Mission rules and fleet scores are unchanged.
See [accuracy, timing scope and reproduction](docs/GPU_BATCHED_VERIFICATION.md).

**Previous combined GPU screening checkpoint:** velocity-matching costs, Earth allowances and
short/long direction selection now run in the same CUDA operator as the Lambert
solve. A direct comparison with the previous GPU path shows another **1.28× local
and 1.89× H100 speedup**, reaching about **2.42 million candidate transfers/s**
on H100 for the 8,192-transfer batch. A 1,000-asteroid route-search comparison
returns the same 27 proxy candidates in **2.71 s versus 12.07 s** on CPU (**4.45×**).
Actual operator counters now capture collection/return helper work: **545,658
Lambert branches per search**, correcting a prior undercount. These candidates
still require low-thrust certification; the fleet score is unchanged.
See [measurements, accuracy and remaining CPU work](docs/GPU_COMBINED_SCREENING.md).

**Previous GPU candidate screening checkpoint:** GTOC12 search now accepts
`--screening-backend cuda --workers 1`. The retained native Lambert batch screens
8,192 candidate transfers in **13.4 ms on RTX 5090 and 6.36 ms on H100**—about
**612,000 and 1.29 million candidates/s**, including transfers and cost selection.
The same top 100 candidates are retained; a shared near-alignment geometry error
was fixed without relaxing the endpoint check. Small warm route searches improved
1.09× locally and 2.39× on H100; cold GPU startup can be slower. These are impulsive
screening estimates, not certified low-thrust solutions or a new fleet score.
See [accuracy, scope and reproduction](docs/GPU_CANDIDATE_SCREENING.md).

**GPU solver fixes validated:** low-thrust factor stabilization removes the
frozen-QP failures in the reported replay matrix, and SCvx now recognizes
feasible stationary steps instead of collapsing the trust region over negligible
merit changes. Original-equation refinement, independent physics checks, objective
tolerances, and finer propagation remain in place. The final build passes
**337 local regression tests, 94 H100 integration tests, and 64/64 repeated
trajectories on each GPU**, covering scaled/unscaled and ordinary/graph execution.
The full SCvx iteration has an opt-in CUDA graph path; setup and fleet search
still need CPU work. This is validation of the reported cases, not completion
of the entire GPU-native application or a new fleet score. See
[the diagnosis and reproduction](docs/GPU_QP_STABILITY.md) and
[downloaded Lambda results](results/lambda/2026-09-07/gpu-stability-v174/summary.json).

**Previous GPU correctness checkpoint:** native v157 rejects SCvx candidates
above the existing physical thrust ceiling on the GPU before acceptance.
The corrected path passes **332 local regression tests and 89 H100 GTOC12
integration tests**, with unchanged independent physics tolerances. Prepared
QOCO133 can also run changing QPs inside a GPU-controlled outer graph, with
exact synchronous parity in local and H100 probes. At that checkpoint the complete
GTOC12 SCvx loop was not yet connected. H100 memory checking passes
the nested solver probes and full GTOC12 guard test; the earlier WSL sanitizer
failure remains unresolved. See [implementation and validation](docs/GPU_OUTER_GRAPH.md).

**Earlier solver repair:** corrected QOCO's small-denominator division guard and
repeated dynamics conditioning. Native v155 / QOCO v131 now passes all six
archived low-thrust recovery cases' independent physics checks on both the RTX
5090 and Lambda H100. On H100, five converge and one reaches its trust-region
limit with a physics-qualified trajectory; individual SCvx times are
**0.230–10.696 seconds**. No CPU solver fallback was used. These are separately
labelled recovery-profile results, not frozen G4 samples or a new fleet score.
The old failing Lambda campaign has been stopped and archived. See
[diagnosis, fixes and reproducible results](docs/GPU_SOLVER_RECOVERY.md).

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
reference solvers and independent verification. Experimental v123 now retains
whole-IPM graphs and their workspace across solves, with changing settings passed
through device memory; it has not demonstrated an additional speedup over v121.
Experimental v124 also captures repeated initialisation and warm-start selection,
and parallelises the initial cone shift. All 36 measured transfers pass the same
physics and mass gates; its 332 ms median is essentially unchanged from the
334 ms v123 comparison. Experimental v125 moves terminal best-iterate recovery
and unscaling into the graph too; 240 direct terminal cases and all 36 complete
transfers pass, with no additional end-to-end speedup established. Initial vendor
warm-up, setup, host reporting and full SCvx orchestration remain unfinished.
Experimental v126 exposes a nonblocking prepared-solve API with device completion
and unscaled outputs. Cold and warm probes each pass 128 queued solves with GPU
consumers.
Experimental native core v128 now connects that replay API to the independent
GPU audit for cold subproblem solves, removing the CPU wait between them. It
passes 324 regression tests and all 36 complete-transfer accuracy gates while
retaining the 64-byte update-report limit. Final reporting and outer SCvx control
still involve the host; no reliable additional overall speedup is established.
Experimental native core v129 extends the same stream through objective
qualification, nonlinear candidate measurement and the SCvx accept/reject
decision before host report collection. All 36 measured transfers retain the
same physics and mass accuracy; the 312 ms median does not establish a reliable
additional gain given run-to-run variation. Full device dispatch and setup
remain unfinished.
Experimental core v131 with QOCO v128 also queues numerical scaling and KKT
updates before replay, keeping their scaling report on the GPU and rejecting
invalid updates through a device guard. It passes 324 regression tests and all
36 complete-transfer accuracy gates. Its 309 ms median remains a variable local
measurement; host validation/dispatch and the memcheck failure are unresolved.
Experimental core v132 then keeps topology and coefficient validation on the
GPU through guarded replay, removing two intermediate CPU waits. All 36 complete
transfers retain the same accuracy gates; adapter update reports shrink from 64
to 60 bytes. The 312 ms median does not establish another speedup. GTOC12 assembly
validation, setup and outer dispatch remain unfinished.
Experimental core v133 also carries the GTOC12 assembly invalid flag into that
guard, removing its intermediate download/wait after setup. All 36 complete
transfers pass the unchanged physics/mass gates; no additional speedup is
established. Coast diagnostics reproduce an existing strict constraint-gate
failure on both the new and published runtimes; that variability remains open.
Experimental core v134 lets captured interval integration consume GPU step-count
and enable flags, and refreshes the SCvx reference after switching to polishing
without a host branch or an extra command wait. All 36 complete transfers retain
the same physics/mass gates. The 364 ms median does not establish another
speedup; per-attempt host dispatch and initial setup remain unfinished.
Experimental core v135 extends device step counts through conic assembly and
candidate propagation. Outer command downloads shrink to eight bytes per
attempt plus eight initial bytes, with all 36 complete transfers retaining the
same physics/mass gates. The 357 ms median establishes no additional speedup.
The earlier v136 candidate separates GPU submission from report collection,
letting SCvx queue reference refresh before waiting. Its 36 complete transfers
qualified (323 ms median), but the broad suite recorded 327 passes and one
coast qualification failure. It was held back from main pending investigation.
See [deferred reports and v136 evidence](docs/GTOC12_DEFERRED_REPORTS.md).
The subsequent v138 candidate adds GPU reference-centred coordinates
and safe recovery from an invalid first reference. It reconstructs and audits
the original physical trajectory without changing any tolerance. All 24 full
transfers in the v137/v138 comparison qualified (291 ms v138 median), but v138's
broad suite still recorded 331 passes and one strict coast equality failure;
the option-disabled integration also failed that gate. Centring has not
eliminated the issue, and no additional reliable speedup is established. See
[reference-centred coordinates and retained failures](docs/GTOC12_STATE_ORIGIN.md).
With the later division correction and experimental retry removed, the final
v155 build passes **332/332** tests in the broader local regression suite,
including coast accuracy and injected-failure accounting. This does not resolve
the separate full conditional-IPM sanitizer limitation.
The existing visualiser now displays these synthetic solver benchmarks in a
separate GPU solver progress panel; the fleet score remains unchanged.

See [GPU-native implementation and measured results](docs/GPU_NATIVE_OPTIMIZATION_PROGRESS.md),
[GPU seed and SCvx control](docs/GTOC12_GPU_NATIVE_CONTROL.md),
[GPU IPM loop and v121 measurements](docs/QOCO_GPU_IPM_LOOP.md),
[retained IPM graphs and v123 evidence](docs/QOCO_RETAINED_IPM.md),
[GPU initialisation and v124 evidence](docs/QOCO_GPU_INITIALIZATION.md),
[GPU terminal handling and v125 evidence](docs/QOCO_GPU_TERMINAL.md),
[device completion and v126 replay evidence](docs/QOCO_GPU_REPLAY.md),
[native replay and v128 audit integration](docs/QOCO_NATIVE_REPLAY.md),
[GPU qualification and v129 SCvx consumers](docs/GTOC12_DEVICE_QUALIFICATION.md),
[queued numerical updates and v131 evidence](docs/QOCO_NUMERIC_REPLAY.md),
[device validation and v132 evidence](docs/QOCO_DEVICE_VALIDATION.md),
[assembly guard and v133 evidence](docs/GTOC12_DEVICE_ASSEMBLY_GUARD.md),
[device integration control and v134 evidence](docs/GTOC12_DEVICE_REFRESH.md),
[device scheduling and v135 evidence](docs/GTOC12_DEVICE_SCHEDULING.md), and
[QOCO device refinement](docs/QOCO_DEVICE_REFINEMENT.md) for implementation
boundaries, test evidence and limitations. Reported performance improvements
apply to their named fixtures; they are not universal speedup claims.

## GTOC12 score versus the published leaderboard

The comparison below records the earlier v595 snapshot. The current independently
verified result is **12,843.556 weighted kg**; see the
[new mission evidence](docs/GPU_ROUTE_HILLCLIMB.md). This update makes no new
official leaderboard-placement claim.

Our verified `orphan_recovery_v595` snapshot collects **14,051.855 kg**, using **23 ships**
and collecting from **195 asteroids**. Both the locally run official checker and
the independent verifier accept the final fleet. The unrounded internal total
is 14,051.854893908598 kg. It improves the retained v11 fleet by 4.052019165 raw kg
and 4.941850560 weighted kg; the other 22 ships are unchanged.

Compared with the [official leaderboard](https://gtoc12.tsinghua.edu.cn/competition/leaderBoard),
checked on **9 September 2026**, our **bonus-weighted score of 12,810.136 kg**
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
| **9th if inserted** | **SpacePDHCG — local verified snapshot** | **12,810.136** |
| 9 | ADL | 12,061.842 |

On this weighted comparison, we are **295.626 kg below ATQ**, **748.294 kg above
ADL**, and at **56.8% of JPL's winning score**. Matching JPL would require another
**9,722.536 weighted kg**. Our score uses the pinned frozen bonus table; historical
competition scores used the coefficients in effect for their submissions.

The physical haul averages **610.950 kg per ship**. The [ESA GTOC portal](https://sophia.estec.esa.int/gtoc_portal/?page_id=1261)
reports **719.8 kg/ship** for JPL's 35-ship competition winner and **742.95 kg/ship**
for Antipodes' 39-ship post-competition fleet (28,975.1 kg raw; 24,474.16 weighted kg).
The latter is the strongest published post-competition result listed there.
Counting each team once still places our result approximately ninth; treating
all three listed post-competition solutions as extra entries places it twelfth.

This is a retrospective comparison with the 2023 competition, not an official
ranked submission or proof of optimality. It measures fleet solution quality;
single-transfer GPU speed tests are distinct from fleet score. This result
combines 22 retained historical ships with one newly GPU-refined replacement;
the entire fleet has not been regenerated by a fully GPU-controlled application.

Evidence: [new fleet report](results/local/2026-09-09/orphan-recovery-v595/report.json),
[fresh independent and official verification, per-asteroid mass and score](results/local/2026-09-09/orphan-recovery-v595/audit/fresh-verification.json),
[artifact integrity audit](results/local/2026-09-09/orphan-recovery-v595/audit/README.md),
and [web visualiser and loading instructions](docs/GPU_ORPHAN_RECOVERY.md).

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
