# SpacePDHCG and OrbitWeaver programme status

Status date: **2026-08-31**.

Scope update (2026-09-02): the active completion goal is now `single-gpu-v1`, defined by
[`ACTIVE_SINGLE_GPU_ROADMAP.md`](ACTIVE_SINGLE_GPU_ROADMAP.md). This dated status snapshot remains
historical evidence; where it describes physical 2/4/8-GPU work as blocking, that work is now
**DEFERRED-NOT-IN-SCOPE**, with all tooling and acceptance preserved in
[`DEFERRED_MULTI_GPU_BACKLOG.md`](DEFERRED_MULTI_GPU_BACKLOG.md).

Integration note (2026-09-05, second release merge): `main` advanced from 689851b through a chain
of merge commits on `release/single-gpu-v1-merge` (no rebase, no squash; every branch keeps its
history). Landed:

- `integration/single-gpu-v1` 1dbcae0 (merge a93982e): G4 attempt-deadline enforcement inside the
  PDHCG recovery kernel and solve preamble (block-uniform `poll_cancellation`, polled recovery
  phases, pre-loaded kernel modules, `inner_iteration_cap`, re-solve floor capped by the amendment).
- `integration/single-gpu-v2-candidate` 211267d (merge b963259): the five H100-exposed defect fixes
  (pd6_fft host quaternion projection 45b1a1d; hcw/pd3 GPU certification 2bca11d: exact HCW replay,
  relative QOCO residual audit with one cold retry, accepted-solve residual; viewer export
  import-graph discovery 1a4f9b4; deferred-manifest units 41a1d1f; literature H100 report twins
  1f5e034), the preflight self-PID fix 5aabbfc and `feat/viewer-40-ships` 7496c10 (40-colour palette,
  dense rail/legend layouts).
- `feat/gtoc12-joint-itinerary` 8e15b92 (merge d52b3a5): whole-itinerary joint re-optimiser
  (`jointopt`, `jointcampaign`, `joint-itinerary` CLI) and `fleet_master_v7` (21 ships, 177 asteroids,
  12,346.48 kg, proven optimal, official + independent verifiers pass).
- H100 fix c4e2c31 (merge 1c0c32e): fleet-master column DFS recursion limit, resolved as the maximum
  of the WSL (ba9b764) and H100 formulas.
- 0ff4f7c: GPU-deferred manifest/doc record the merged blobs of `persistent_pdhcg.cu` and
  `device_scvx_integration_test.cu` (moved by 1dbcae0).

Conflict resolutions: `viewer_export.py` takes the candidate's `viewer_modules()`/`viewer_scripts()`
discovery over main's static list (68003c2); `native_qoco_adapter.h` keeps both report field sets
(`status_code`/`ruiz_iterations` and `last_status_inaccurate`/`warm_inaccurate_cold_retries`) and
`status_code` is refreshed after the cold retry; `device_scvx.cu` keeps the v1 deadline work and the
2bca11d HCW changes (disjoint regions); `cooperative.py` recursion limit = `max(2n + 200, n + 500)`;
memory files keep both sides' entries chronologically.

Verification of the integrated head (WSL Ubuntu-22.04, RTX 5090 sm_120 shared with a foreign
low-utilisation workload, CUDA 12.8): ruff check + format clean (298 files); full CPU pytest 659
passed / 35 skipped (GPU-gated and offline-artifact tests only) with the CPU-built libqoco; host
RelWithDebInfo `-Werror` build 0 warnings + ctest 50/50; `cpp/native` 8/8; CUDA sm_120 Release
`-Werror` clean rebuild 0 warnings + full CUDA CTest 70/70 (69 + `cancellation_deadline_test`);
planner GPU pytest 9/9; `tests/test_g4_pdhcg_deadline_gpu.py` 13/13 (5 s and 20 s deadlines, N=100
and N=2000, all three policies, claim-core cap); viewer `npm run check` + `npm test` (36 pass, 2
environment skips) with `fleet_master_v7` imported (fleet SHA `e47af8fa…36ec`); manifest/benchmark
tests 14/14; generated-artefact checks (G4 policy header, G7 schemas, literature provenance, packaged
assets) clean; wheel + sdist build and installed-wheel smoke (`spacepdhcg --help`, `literature list`,
`gtoc12 --help`, `gtoc12 reduced-instance --list-ids`, `validate`, `plan --backend cpu_reference`
certified, `python -m spacepdhcg`).

Still off `main` after this merge: the local `feat/gtoc12-asteroid-mining` HEAD (7d2e301+, a v8
harvest-substitution worker is committing there; only c4e2c31 was taken); the H100 G4 claim-core
campaign (running on the Lambda H100; its capability/evidence are not in the tree); the sm_90
(H100) confirmation of the v2 fixes, which so far are verified on sm_120 only (checklist
`~/spacepdhcg/v2-PENDING-H100-GPU-VERIFY.txt` on the H100); `perf/g4-batched-campaign` (failed
protocol-v2 experiment, pushed for provenance).

Integration note (2026-09-05, third release merge): `main` advanced from 8cb3759 through merge
commits on `release/single-gpu-v1-merge` (merge commits only; every branch keeps its history).
Landed:

- `integration/single-gpu-v1` bf4cf0f (merge abd4e81): the H100 evidence-script commit 9e75b47 -
  `scripts/gpu/*` target the local CUDA architecture (sm_90 on the H100, sm_120 here).
- `chore/g2g3-reseal-8cb3759` 06e70b6 (merge 16d5e8e): G2/G3 reseal of main 8cb3759 on the RTX 5090
  (both PASS), compact evidence under `results/gpu/current-head-8cb3759-rtx5090/`.
- `feat/gtoc12-asteroid-mining` 1f6ec50 then b55eb70 (merges a93649d, ace3b25): the ninth GTOC12
  iteration - chain-level objective in the beam, reference-chain prior (`chainprior.py`,
  `benchmarks/gtoc12/chain_prior_v1.json`, `gtoc12 chain-prior`), master-LP duals fed back into the
  family pricing with archive-seeded pricing columns and bound-share duals, joint itinerary inside the
  pricing, the NaN burn-schedule guard, and the `gtoc12 joint-itinerary` CLI import fix
  (`REPOSITORY_ROOT` no longer exported by `gtoc12/data.py`, so the subcommand raised `ImportError`
  on 8cb3759). Results `cluster_fleet_v9`, `joint_itinerary_v3`, `fleet_master_v8` (21 ships,
  177 asteroids, 12,356.30 kg, 588.40 kg average, proven optimal, LP gap 6.4 kg, both verifiers).
  The branch rolled its memory files over (slim live files + `*_2026-09-01_to_2026-09-05.md`).
- H100 GTOC12 line `refs/h100/gtoc12-asteroid-mining` 86a91d3 then 48e5fb7 (merges aaa9657,
  5f23f73): `bundles.family_partitions` (a union of family partitions - several radii x band sets -
  priced in one campaign; `--cluster-radius` list, `--all-family-bands`), and the H100 results
  `cluster_fleet_h100_v1/v2`, `joint_itinerary_h100_v1/v2/v8`, `fleet_master_h100_v1/v2/v3`.
- 1bd78ce: the Windows checkout's 2026-09-05 memory entries (Lambda H100 provisioning, v2
  candidate fixes, viewer palette, release merges, G4 launch, reseal, H100 pipeline finish) folded
  into the live files and the dated snapshots chronologically, nothing dropped.
- 7a30c12 / 5784e64: `.gitignore` rules for root scratch files and the raw Lambda H100 evidence;
  `results/lambda-h100/` compact evidence (608 files, 5.95 MB, `INDEX.json` with sha256 per file and
  the 1171 skipped files listed).

**GTOC12 headline (2026-09-05):** best verified fleet `fleet_master_h100_v2` = `fleet_master_h100_v3`
(byte-identical `fleet/Result.txt`, sha256 `beffeeff…0548`): **22 ships, 187 asteroids,
13,189.60 kg, 599.53 kg average**, master objective 12,203.96 kg, LP bound 12,207.39 kg (gap 3.4 kg;
not proven optimal - node cap), official `GTOC12_Verify` + independent verifier pass on the host and
locally. Best proven-optimal fleet: `fleet_master_v8`, 21 ships / 12,356.30 kg / 588.40 kg average
(LP gap 6.4 kg, LP infeasible at 22). Details and the per-run table: `GTOC12_TRACK.md` section 7.

Conflict resolutions: `gtoc12/cli.py` cluster-fleet imports keep the v9 names
(`load_chain_prior`, `lp_asteroid_prices`) and the `REPOSITORY_ROOT`-free data import; the band
construction is the H100 `cluster_band_partitions()` + `family_partitions(...)` with every v9 setting
kept; `cooperative.py` recursion-limit comment merged (code identical on both sides:
`max(2n + 200, n + 500)`); `GTOC12_TRACK.md` section 7 keeps both sides' rows in order (local v9 rows
then the H100 rows; the H100 joint rows now cite the joint-itinerary section as 6.11, its number
after the branch's renumbering); memory files: the reseal entries inserted chronologically into the
branch's rolled-over live files (line coverage 0 missing on both sides).

Verification of the integrated head 5784e64 (WSL Ubuntu-22.04, RTX 5090 sm_120 shared with a
foreign ~3.5 GB workload, CUDA 12.8; `cpp/` is byte-identical to 8cb3759): ruff check + format clean
(302 files); generated-artefact checks (G4 policy header, G7 schemas, literature provenance 126
records, packaged assets 34) clean; host RelWithDebInfo `-Werror` fresh build 0 warnings + ctest
50/50; `cpp/native` fresh build 0 warnings + 8/8; CUDA sm_120 Release `-Werror` clean rebuild (84
objects) 0 warnings + full CUDA CTest 70/70 in 248 s (incl. `cancellation_deadline_test`); planner
GPU pytest 9/9; full CPU pytest 677 passed / 35 skipped in 595 s (+18 tests over 8cb3759; skips are
the GPU-gated and offline-artifact tests as before); gtoc12 suites + CLI dispatch 147 passed on the
resolved H100 tree; manifest/benchmark/experiment tests 14/14 (deferred-manifest blobs unchanged);
viewer `npm run check` + `npm test` (36 pass, 2 environment skips) with `fleet_master_h100_v2`
regenerated by `gtoc12 export-viewer` and imported (22 ships, 187 asteroids, 13,189.60 kg, fleet SHA
`cbedee96…fd48`, solution SHA = committed manifest, Kepler cross-check max 3.59e-6 km); wheel + sdist
build and installed-wheel smoke in a fresh venv (`spacepdhcg --help`, `literature list`, `gtoc12
--help`, `gtoc12 cluster-fleet --help` with the v9 + H100 flags, `gtoc12 joint-itinerary --help`,
`gtoc12 reduced-instance --list-ids`, `validate`, `plan --backend cpu_reference` certified,
`python -m spacepdhcg`). The 22-minute `test_g4_pdhcg_deadline_gpu.py` matrix was not repeated
(no CUDA/C++ source changed since it passed 13/13 on 8cb3759 and in the 13:50 reseal). 5f23f73 and
the status/memory commit are results/docs only; ruff, the manifest tests and the gtoc12 suites were
re-run on the final tree.

Still off `main` after this merge: `feat/gtoc12-asteroid-mining` beyond b55eb70 (the worker has
continued committing there); the H100 G4 claim-core campaign on 1dbcae0 (running; compact G4
evidence to date is under `results/lambda-h100/g4/`); the sm_90 confirmation of the v2 fixes
(pending an H100 GPU window); `perf/g4-batched-campaign` (provenance only); the raw Lambda H100
artefacts listed as skipped in `results/lambda-h100/INDEX.json` (Windows disk and the host).

This document distinguishes completed engineering from GPU-dependent experiments. “Implemented”
does not mean “demonstrated faster”; performance claims require the benchmark protocol and real
recorded hardware. The more detailed blocker classification is in
[`PRE_GPU_COMPLETION_AUDIT.md`](PRE_GPU_COMPLETION_AUDIT.md).

# Executive state

The production architecture is now deliberately split as:

```text
Python references / experiments / plots / paper artefacts
                         │
              stable C ABI and parity tests
                         │
C++20 dynamics + transcription + SCvx + CQP + robust routing core
                         │
          future persistent PDHCG CUDA implementation
                         │
            future NCCL/MPI multi-GPU implementation
```

The repeated numerical hot path is intended to be C++/CUDA. Python remains the transparent
correctness oracle, research interface, and experiment layer. Removing Python from those roles
would not accelerate the device-resident solver and would reduce auditability.

The current host-native branch now closes the previously highest-priority CPU gaps: automatic
continuous-time violation-state sampling, deterministic and robust low-thrust OrbitWeaver stages,
multi-revolution Lambert families, executable restricted-master column generation, and independent
J2 high-fidelity certification.

# Status legend

- **COMPLETE-REFERENCE** — implemented and covered by CPU/native correctness tests.
- **PREPARED** — interface and host logic exist; the production accelerator implementation remains.
- **GPU-BLOCKED** — implementation or validation intrinsically requires a CUDA device.
- **OPEN-CPU** — useful work remains and does not require a GPU.
- **EXPERIMENT-BLOCKED** — implementation may exist, but the paper result needs hardware runs.
- **DEFERRED-NOT-IN-SCOPE** — preserved do/test-later work that does not block `single-gpu-v1`.

# A — common modelling and benchmark foundation

| Item | Status | Evidence |
|---|---|---|
| Canonical QP/SOCP/CQP representation | COMPLETE-REFERENCE | Python and C++ fixed-pattern structures |
| Stable topology fingerprint | COMPLETE-REFERENCE | matching Python/C++ fingerprint and parity probes |
| HCW rendezvous QP/SOCP | COMPLETE-REFERENCE | repeated updates and independent diagnostics |
| Known-optimum trajectory-banded fixtures | COMPLETE-REFERENCE | deterministic QP/SOCP families |
| Nonlinear 3-DoF powered descent | COMPLETE-REFERENCE | analytic Jacobians, Euler/RK4 propagation and path checks |
| 14-state 6-DoF powered descent | COMPLETE-REFERENCE | quaternion rigid-body dynamics and fixed-pattern CQP |
| Long-horizon two-body low thrust | COMPLETE-REFERENCE | gravity/thrust Jacobians, mass depletion and CQP |
| Selectable higher-order transcription | COMPLETE-REFERENCE | Euler or RK4 discrete-flow linearisation with invariant topology |
| Domain-aware finite differences | COMPLETE-REFERENCE | central interior and valid one-sided boundary derivatives |
| Continuous inter-node certification | COMPLETE-REFERENCE | dense nonlinear propagation and path-violation checks |
| Violation-state CT enforcement inside the CQP | COMPLETE-REFERENCE | immutable affine samples, cumulative states and interval budgets |
| Automatic CT sample/linearisation construction | COMPLETE-REFERENCE | midpoint, trapezoidal, Simpson and four-node Gauss–Lobatto rules |
| Adaptive mesh refinement between episodes | COMPLETE-REFERENCE | route-gap-driven discovery and fixed hot-loop topology |

# B — persistent device-resident SCvx

| Item | Status | Notes |
|---|---|---|
| Immutable C++20 CQP ownership | COMPLETE-REFERENCE | owns CSC topology and cone metadata |
| Mutable numerical update buffers | COMPLETE-REFERENCE | topology-preserving updates with validation |
| Persistent workspace state machine | COMPLETE-REFERENCE | update/solve/reset/cancel lifecycle |
| Host/device/unified pointer descriptors | COMPLETE-REFERENCE | API boundary is frozen |
| Scaling reuse/refresh controller | COMPLETE-REFERENCE | change thresholds and reuse budget |
| Primal-dual warm-start lifecycle | COMPLETE-REFERENCE | backend and session contracts |
| Checkpoint/restart | COMPLETE-REFERENCE | deterministic topology-locked state |
| Native persistent 3-DoF outer SCvx driver | COMPLETE-REFERENCE | create once, update thereafter, adaptive solves and trust region |
| Native persistent low-thrust outer SCvx driver | COMPLETE-REFERENCE | physical initial reference, nonlinear rollout and lifecycle counters |
| Dense C++ ADMM debug backend | COMPLETE-REFERENCE | small QP/SOCP truth solver, not a performance claim |
| Stable C ABI | COMPLETE-REFERENCE | native dynamics/Jacobians/Lambert callable outside C++ |
| Installable native CMake package | COMPLETE-REFERENCE | exported targets and external consumer test |
| Pinned upstream one-shot PDHCG adapter | PREPARED | exact data and cone compatibility exists |
| Real one-shot PDHCG correctness run | GPU-BLOCKED | requires NVIDIA CUDA 12.4+ |
| Concrete persistent PDHCG CUDA workspace | GPU-BLOCKED | must retain preprocessing, scaling, iterates and buffers |
| In-place device coefficient updates | GPU-BLOCKED | implementation and measurement require CUDA |
| Device-side dynamics/Jacobian fill | GPU-BLOCKED | C++ formulae exist; CUDA kernels do not |
| CUDA graph capture and stream overlap | GPU-BLOCKED | meaningful only on hardware |
| DLPack/CUDA-array exchange | PREPARED | stable C ABI exists; accelerator pointer bridge remains |

# D — adaptive inexact and hybrid solving

| Item | Status | Notes |
|---|---|---|
| Repair/progress/refinement/polish forcing | COMPLETE-REFERENCE | Python and C++ implementations |
| Fixed-tolerance comparator | COMPLETE-REFERENCE | experiment baseline |
| Re-solve-before-trust-region-shrink logic | COMPLETE-REFERENCE | implemented in outer drivers |
| Trust-region acceptance and radius updates | COMPLETE-REFERENCE | native deterministic and robust drivers |
| Inexact error ledger | COMPLETE-REFERENCE | accumulated and relative diagnostics |
| Hybrid first-order/IPM plan | COMPLETE-REFERENCE | explicit handoff and final-polish contract |
| Conditional convergence argument | COMPLETE-REFERENCE | assumptions and proof obligations documented |
| Solver-independent incumbent qualification | COMPLETE-REFERENCE | status, residual and feasible-objective checks |
| Native outer-driver policy lifecycle | COMPLETE-REFERENCE | persistent backend contract |
| Full theorem with checked assumptions | OPEN-CPU | mathematical refinement and counterexamples remain |
| PDHCG adaptive-tolerance sweep | EXPERIMENT-BLOCKED | real CUDA PDHCG required |
| QOCO-GPU/CuClarabel comparison | EXPERIMENT-BLOCKED | compatible GPU builds required |
| Empirical crossover map | EXPERIMENT-BLOCKED | matched nonlinear-quality results required |

# C — robust scenario and multi-GPU optimisation

| Item | Status | Notes |
|---|---|---|
| Scenario information histories | COMPLETE-REFERENCE | deterministic scenario tree |
| Shared-prefix non-anticipativity | COMPLETE-REFERENCE | exact sparse rows |
| Block-arrow variable layout | COMPLETE-REFERENCE | local blocks and shared arrowhead |
| Condensed shared-column formulation | COMPLETE-REFERENCE | solver-independent repeated-local certificate |
| Deterministic scenario partition | COMPLETE-REFERENCE | whole-scenario load balancing |
| Communication-volume model | COMPLETE-REFERENCE | ring-allreduce accounting |
| Partition-invariant forward/transpose truth model | COMPLETE-REFERENCE | CPU comparison against monolithic operators |
| 3-DoF robust assembly | COMPLETE-REFERENCE | native and Python paths |
| 6-DoF robust assembly | COMPLETE-REFERENCE | generic 14-state scenario bundle |
| Robust low-thrust SCvx driver | COMPLETE-REFERENCE | common-prefix controls, scenario nonlinear rollouts and affine propellant risk |
| Expected/worst/VaR/CVaR evaluation | COMPLETE-REFERENCE | native and Python aggregation |
| Expected/worst/CVaR CQP augmentation | COMPLETE-REFERENCE | fixed-pattern affine-loss epigraphs |
| Known-incumbent solution qualification | COMPLETE-REFERENCE | catches degenerate IPM false positives |
| Native scenario-local CUDA shards | GPU-BLOCKED | device ownership and kernels required |
| NCCL non-anticipativity reductions | GPU-BLOCKED | multi-GPU node required |
| Communication/computation overlap | GPU-BLOCKED | real topology and profiling required |
| Strong and weak scaling | DEFERRED-NOT-IN-SCOPE | preserved 2/4/8-GPU campaign |

# E — OrbitWeaver multi-destination optimisation

| Item | Status | Notes |
|---|---|---|
| Native fidelity-ladder oracle contract | COMPLETE-REFERENCE | analytical through certified levels |
| Exact request cache and warm-start pipeline | COMPLETE-REFERENCE | solver-independent native implementation |
| Hohmann/phasing screening | COMPLETE-REFERENCE | Python/C++ analytical screen |
| Zero-revolution Lambert solver | COMPLETE-REFERENCE | universal-variable C++ solver |
| Multi-revolution Lambert families | COMPLETE-REFERENCE | short/long directions and lower/higher parameter branches |
| Executable Lambert screening oracle | COMPLETE-REFERENCE | family enumeration, matching impulses and mass closure |
| Edelbaum low-thrust screening | COMPLETE-REFERENCE | native radius/inclination estimate |
| Coarse convex low-thrust arc adapter | COMPLETE-REFERENCE | native CQP solve plus independent nonlinear rollout |
| Refined deterministic SCvx arc adapter | COMPLETE-REFERENCE | persistent native low-thrust driver and warm-reference transfer |
| Robust SCvx arc adapter | COMPLETE-REFERENCE | scenario provider, common-open-loop controls and expected/worst/CVaR certificates |
| Final high-fidelity certification adapter | COMPLETE-REFERENCE | independent J2 RK4 replay, dense path checks and step-doubling error estimate |
| Deterministic beam search | COMPLETE-REFERENCE | time/mass-dependent native search |
| Time-expanded moving-target graph | COMPLETE-REFERENCE | scheduled-arc truth graph |
| Exact elementary-route labels | COMPLETE-REFERENCE | small-instance truth model up to 64 targets |
| Optimistic route lower bounds | COMPLETE-REFERENCE | non-elementary dynamic-programming bound |
| Dynamic discretisation discovery | COMPLETE-REFERENCE | route-gap-driven epoch refinement |
| Route-column dominance and pricing | COMPLETE-REFERENCE | reduced-cost candidate filtering |
| Exact multi-spacecraft route master | COMPLETE-REFERENCE | small-instance set-partitioning truth model |
| Native restricted-master LP | COMPLETE-REFERENCE | dependency-free two-phase simplex and dual recovery |
| Full iterative column-generation controller | COMPLETE-REFERENCE | executable master/pricing loop with incumbent and gap history |
| One-GPU coarse/refined/scenario/pricing/master/certification simulations | ACTIVE | required by `single-gpu-v1` |
| Physical route × scenario throughput/scaling | DEFERRED-NOT-IN-SCOPE | preserved distributed Paper 2 campaign |

# Native and reference quality gates

| Gate | Status |
|---|---|
| Full warnings-as-errors native build | COMPLETE-REFERENCE |
| All host-native smoke targets | COMPLETE-REFERENCE |
| ASan/UBSan suite | COMPLETE-REFERENCE |
| Native-core GCC and Clang build/test | COMPLETE-REFERENCE |
| Native C++/Python parity gate | COMPLETE-REFERENCE |
| macOS native build | COMPLETE-REFERENCE |
| Installed CMake package consumer | COMPLETE-REFERENCE |
| Python 3.11/3.12 lint and reference suite | COMPLETE-REFERENCE |
| Condensed formulation certificate | COMPLETE-REFERENCE |
| Hardware benchmark provenance schema | COMPLETE-REFERENCE |

# Paper and experiment infrastructure

| Item | Status |
|---|---|
| Paper 1 benchmark protocol and manifests | COMPLETE-REFERENCE |
| Paper 2 experiment manifest | COMPLETE-REFERENCE |
| Failure/OOM/timeout reporting rules | COMPLETE-REFERENCE |
| Paper outlines and notation lock | OPEN-CPU |
| Paper 1 scoped one-GPU tables and plots | EXPERIMENT-BLOCKED | complete G4 evidence still required |
| Paper 1 F07/F12/T06 physical products | DEFERRED-NOT-IN-SCOPE | never emitted empty or fabricated |
| Paper 2 one-GPU simulation visualisations | EXPERIMENT-BLOCKED | active G7 completion product |
| Paper 2 physical scaling/energy/crossover plots | DEFERRED-NOT-IN-SCOPE | separate future campaign |

# Exactly what the absence of GPU runs blocks

No GPU blocks only work or claims that require a CUDA execution environment:

1. importing and executing the pinned upstream PDHCG package;
2. validating one-shot PDHCG against committed CPU references;
3. implementing and debugging persistent ownership against upstream internals;
4. demonstrating allocation-free device updates and residency;
5. comparing PDHCG against GPU interior-point solvers;
6. measuring GPU memory, utilisation, energy and crossover points;
7. implementing and validating NCCL/MPI reductions;
8. measuring strong/weak scaling and communication overlap;
9. validating GPU precision, determinism, streams and graph capture;
10. making any speedup, throughput, memory-scaling or energy claim.

The absence of a GPU does **not** block dynamics, transcriptions, outer-loop logic, risk
formulations, route algorithms, correctness certificates, theory, manifests or paper structure.

# Remaining pre-GPU priorities

In descending leverage:

1. stronger variational integration where it beats finite-difference RK4;
2. theorem-level inexact-SCvx assumptions, lemmas and counterexample tests;
3. native wheel packaging and accelerator-pointer exchange design;
4. paper outlines, notation lock and figure-generation schemas;
5. additional adversarial, randomized and property-based tests;
6. reproducible scale-sweep manifests and synthetic memory/work estimates.

# First GPU-day run order

1. Record the exact hardware/software manifest.
2. Build the pinned upstream PDHCG revision.
3. Run known-optimum banded QP/SOCP fixtures through the one-shot adapter.
4. Compare objective, residuals and cone feasibility against CPU references.
5. Run HCW and deterministic 3-DoF CQP fixtures.
6. Activate persistent workspace ownership and repeated numerical updates.
7. Compare one-shot versus persistent cold and warm solves.
8. Connect the native SCvx driver to the device workspace.
9. Run 6-DoF and low-thrust node-count sweeps.
10. Add scenario sharding and NCCL only after single-GPU correctness closes.
