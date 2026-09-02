# Engineering Devlog

## 2026-09-01 18:45 AEST

- Task summary:
  - Implemented the audited CPU/static Gate G4 policy, qualification, decision, coverage, schema,
    timing, and evidence-integrity contracts in an isolated worktree.
- Changes:
  - Added a SHA-256-locked JSON-to-C++ policy generator and selected-quality-tier fixed-tight rule.
  - Added matched-quality, full Cartesian ledger, paired-bootstrap H5/H6, timing identity, and
    portable-artifact validators.
  - Made primary G4 schema records require runtime/outer/hybrid/energy/failure/evidence telemetry
    while preserving legacy 1.0 record validation.
  - Replaced unconditional qualification decisions with preregistered decision functions.
- Validation:
  - Full Ruff check and whitespace check passed.
  - Full Python suite passed: 127 tests.
  - CPU native build and CTest passed: 41 tests.
  - No GPU benchmark or GPU performance workload was run.
- Follow-up notes / risks:
  - The CUDA executable must consume/report the generated runtime request before current output can
    qualify; this is intentionally left as a clean integration point for the CUDA worker.
  - QOCO/hybrid integration must populate permutation, dual disposition, and complete timing fields.

## 2026-09-01 19:55 AEST

- Task summary:
  - Integrated the exact QOCO and audited-harness commits into the canonical CUDA correction branch.
- Changes:
  - Resolved the scratchpad-only cherry-pick conflict by preserving both correction histories.
  - Routed adaptive phase ceilings/limits, trust expansion, identical-CQP re-solve, quality-tier,
    scaling, warm-start, and policy-hash behavior through the generated frozen contract.
  - Added requested/actual CUDA runtime telemetry and RTX trajectory QP/SOCP adapter coverage.
- Validation:
  - Ruff and 142 Python tests passed; three optional QOCO tests skip without an explicit library.
  - With the pinned CUDA library configured, all 18 QOCO adapter tests passed on RTX 5090.
  - Debug and Release CUDA/native suites each passed 52/52 tests.
- Follow-up notes / risks:
  - The integrated P1-C diagnostic still fails forcing at `5.654e-4`; the adapter correction does
    not itself provide nonlinear P1-C/P1-D/P1-E handback.
  - A 15-outer P1-C run exceeded 30 minutes and was terminated. The full matrix remains censored,
    H5/H6 unresolved, G4 failed, and G5 unauthorized.
## 2026-09-01 22:05 AEST

- Task summary:
  - Isolated the displaced P1-C failure block-by-block and added complete recovery/merit telemetry.
- Changes:
  - Corrected actual-merit semantics, raw predicted-reduction sign handling, rejected warm-state
    rollback, converged no-op handling, and re-solve recovery cost accounting.
  - Extended the exact-dump comparator to arbitrary P1-C horizons and optional QOCO-GPU runs.
- Validation:
  - Exact P1-C still fails: recovery primal `1.016e-9`, recovery stationarity `3.853`,
    rolled-back canonical residual `5.654e-4`, zero accepted steps.
  - Identical-CQP QOCO-GPU reaches canonical primal `4.680e-7`, dual `9.090e-10`.
  - Focused Python tests pass 35/35; Release warnings-as-errors build and production outer
    regression pass.
- Follow-up notes / risks:
  - G4 remains failed. No H5/H6 matrix or G5 work was started.

## 2026-09-01 20:55 AEST

- Task summary:
  - Corrected the isolated P1-E displaced-reference qualification path without changing frozen
    policy, tolerances, matrix coordinates, or solver targets.
- Changes:
  - Replaced the invalid P1-E dispersion input with exact trust-radius/transfer-class parsing and
    deterministic reachable two-body low-thrust transfer targets.
  - Replayed from the immutable initial condition and added independent throttle, mass, and
    minimum-radius inventory telemetry.
  - Added all-vector displaced CPU/GPU coefficient and boundary parity, fixed-topology checks,
    adversarial accepted/rejected iterations, terminal reduction, and injected path violations.
- Validation:
  - Adversarial radius raise: 2 accepted steps, step fraction `0.84855828091422303`, scaled terminal
    error `1.7430149824386731e-3 -> 0`.
  - Short GPU production/recovery tests passed; maximum coefficient difference
    `2.7599450502791001e-13`, no topology allocation/copy, no hidden CPU fallback.
  - Full host suite: 41/41; full Python suite: 142 passed, 3 optional QOCO skips.
- Follow-up notes / risks:
  - The committed N=100/dispersion=0.01 P1-E failure is not a valid frozen coordinate. A corrected
    N=100 solve exceeded the short shared-GPU limit and was stopped, so no full qualification or
    performance claim is made.
  - Cherry-picking may require additive conflict resolution in shared CUDA outer-driver/test code
    and these append-only memory files if P1-C lands first.

## 2026-09-01 21:50 AEST

- Task summary:
  - Diagnosed the exact frozen P1-D forcing failure against Clarabel and pinned QOCO-GPU.
- Changes:
  - Removed the contradictory terminal quaternion tangent equation numerically while preserving
    the fixed CSC topology and CPU/CUDA coefficient parity.
  - Added exact P1-D CQP dump, bounded PDHCG iterate dump, active-set conditioning, canonical
    reference comparison, and explicit QOCO handoff qualification diagnostics.
- Validation:
  - Pre-fix Clarabel status was `PrimalInfeasible`; corrected Clarabel status is `Solved`.
  - Pinned QOCO-GPU solved in 20 iterations at canonical primal `1.091e-12` and dual `7.006e-12`.
  - Focused Release/Debug builds, 6-DoF transcription tests, recovery lifecycle/randomized tests,
    P1-D path audit, sanitizer modes, Ruff, and QOCO adapter tests passed.
- Follow-up notes / risks:
  - PDHCG remains unqualified at 300k iterations (`2.818e-2` natural residual), so its primal is
    rejected by the frozen `1e-6` hybrid handoff gate.
  - QOCO evidence is valid only as `pure-gpu-ipm`; fixed-tight PDHCG still has zero qualified
    accepted outer steps. Production nonlinear QOCO handback remains a separate dependency.

## 2026-09-01 22:30 AEST

- What changed:
  - Aligned the isolated QOCO branch with integrated G4 commit `a33e950`, then cherry-picked the
    corrected P1-D lifecycle and terminal-quaternion commits in the requested order.
  - Added native canonical primal/dual transfer, family-complete device nonlinear replay,
    fingerprint/permutation checks, frozen merit/trust acceptance, and transactional handback.
  - Added a strict pre-QOCO PDHCG gate; the known `2.818e-2` predictor remains explicitly
    ineligible and cannot be relabelled hybrid.
  - Added distinct pure/hybrid labels, timing and dual-disposition records plus HCW/P1-C/P1-D/P1-E
    and negative handback fixtures.
- Validation:
  - 156 Python tests passed with pinned CPU QOCO and the host native library.
  - Release host build and 41/41 CTest tests passed.
  - `device_scvx.cu` and its integration fixture compiled with CUDA 12.8 for `sm_120`.
  - Changed Python files pass Ruff lint and format checks; `git diff --check` passes.
- Follow-ups / risks:
  - The dedicated short RTX `device_scvx_qoco_handback_test` is compiled but unrun while the GPU is
    reserved. No performance, energy, or matrix run was performed.

## 2026-09-01 21:15 AEST

- Task summary:
  - Integrated and repaired the complete standalone `cpp/native` implementation/test inventory.
- Changes:
  - Added eight previously omitted implementation files to `spacepdhcg_native_core` and registered
    all seven existing smokes plus a new six-DoF smoke as explicit CTest targets.
  - Added native-core sanitizer controls and GCC/Clang Release, Debug, and ASan/UBSan CI coverage.
  - Fixed the explicit-constructor warning, optional-agreement warning, nodiscard checks, stale test
    namespace assumptions, and robust CVaR/base-layout test mismatch without changing the C ABI.
  - Ignored `build-*/` outputs to prevent generated CTest/compiler files entering source archives.
- Validation:
  - Native inventory: 8/8 in Release, 8/8 in Debug, 8/8 under ASan/UBSan, all with Werror.
  - Existing top-level Release/Debug/ASan suites: 41/41 each; core 5/5; all-native 41/41;
    CMake consumer 1/1; Ruff clean; Python 142 passed with 3 optional QOCO skips; parity 1/1.
  - Editable, native wheel, sdist, wheel consumer, and sdist consumer import/ABI checks passed.
- Follow-up notes / risks:
  - No code was classified obsolete: commit history and dedicated native CI establish this as an
    active parallel native-core API. Conceptual overlap with newer headers is not replacement proof.
  - GPU tests and measured benchmarks were intentionally not run.

## 2026-09-01 22:12 AEST

- Task summary:
  - Prepared Gate G5 in isolated branch `feat/scenario-aware-multigpu`; this is implementation and
    one-rank correctness only, not G5 acceptance.
- Changes:
  - Added persistent rank-local G2 composition, deterministic MPI-rank/CUDA-device mapping,
    MPI/NCCL stream-event runtime, scenario-aware and nonzero partitions, device algebra,
    risk/residual/status reductions, telemetry, checkpoint/restart, and cancellation.
  - Added 1/2/4/8 logical-rank tests, real one-rank MPI/NCCL/CUDA coverage, G5 evidence schema,
    CMake/CTest build targets, manual self-hosted CI, and implementation status documentation.
  - Installed OpenMPI 4.1.2 and NCCL 2.26.2 for CUDA 12.8 only; no driver package was installed.
- Validation:
  - Pinned upstream PDHCG commit `167c8b7`/tree `62b05e6` built all 163 distributed steps.
  - Debug, Release, warnings-as-errors, and sanitizer-capable G5 targets compiled.
  - Logical 1/2/4/8 tests passed; schema/claim-drift tests passed 6/6; Ruff passed.
  - Idle RTX 5090 Release one-rank MPI/NCCL/CUDA CTest passed; memcheck was leak/error clean and
    racecheck reported zero hazards.
- Follow-up notes / risks:
  - A later idle window completed the final-HEAD one-rank rerun, initcheck, and synccheck cleanly.
    Initcheck unused-memory mode separately flags NCCL's own 2 MiB communication pools.
  - Actual overlap execution remains deferred to physical multi-GPU validation.
  - Physical 2/4/8 correctness, rank-failure behavior, strong/weak scaling, and H2/H3/H4 remain
    unverified. G5 must not be reported as PASS.

## 2026-09-01 21:50 AEST - G6 freeze tooling

- Task summary:
  - Implemented Paper 1 G6 aggregation/freeze tooling in the isolated
    feat/paper1-freeze-tooling worktree.
- Changes:
  - Added strict archived-evidence envelopes, immutable hash verification, run indexing, frozen
    F01-F08/T01-T06 source and publication builders, H1-H6 deterministic decision records,
    campaign freeze/refusal, checksums, claim linkage, and clean-clone verification.
  - Added synthetic failure/censoring matrix, schemas, CLI, documentation, and CI.
- Validation:
  - Ruff, 151 Python tests, and the native wheel build passed.
  - Generated a 14-run synthetic-only bundle with 52 deterministic output files and verified a
    byte-identical second build.
  - No GPU benchmark or real scientific campaign was run.
- Follow-up notes / risks:
  - Real freeze remains blocked on complete portable G4/G5 evidence. Existing G4 failures and G5
    authorisation state are not altered by this tooling.

## 2026-09-01 22:24 AEST

- Task summary:
  - Integrated all requested completed G4-G7 implementation commits on isolated branch
    `integration/roadmap-code` from stable base `a33e950`.
- Changes:
  - Resolved shared CUDA outer-driver, QOCO handback, runtime-coordinate, CLI entry-point, and
    append-only history conflicts without weakening classifications, policies, or tests.
  - Preserved P1-E inventory, P1-D path/forcing corrections, P1-C lifecycle accounting, pure/hybrid
    QOCO distinction, G5 interfaces, G6 tooling, and G7 CPU/CUDA seams.
- Validation:
  - Ruff lint passed; full Python suite passed 174 tests with three optional QOCO skips.
  - Pinned CPU QOCO ABI/handback suite passed 29/29 when supplied `libqoco.so`.
  - Top-level native Release, Debug, and ASan/UBSan each passed 43/43; standalone native-core
    Release, Debug, and ASan/UBSan each passed 8/8, all with warnings as errors.
  - CUDA/G5 Release, Debug, and sanitizer-capable configurations built all 77 targets for `sm_120`;
    logical-rank contracts passed in Release and Debug.
  - Pinned QOCO CPU and CUDA/cuDSS builds, native wheel ABI, both packaged CLIs, and the CMake
    consumer passed. No GPU executable, measured benchmark, energy campaign, sanitizer execution,
    or multi-GPU scaling run was launched.
- Follow-up notes / risks:
  - Full-repository Ruff format check reports 43 pre-existing/unintegrated files would be reformatted;
    Ruff lint itself is clean.
  - Add the forthcoming P1-C trust-globalization commit by ordinary cherry-pick, preserve histories
    additively, rerun CPU/static matrices, then serialize deferred GPU validation after contention
    clears.

## 2026-09-01 23:35 AEST

- What changed:
  - Integrated P1-C rejection telemetry and retry globalization as `99abb9e` then `e191a0d`,
    preserving P1-D quaternion corrections and QOCO handback semantics.
  - Added a direct persistent C++ QOCO owner to the CUDA SCvx driver, exact canonical
    QP/SOC/rotated-SOC conversion, canonical primal/dual mapping, independent residuals,
    accepted-primal-only warm starts, dual-discard reporting, and CUDA/cuDSS failure ownership.
  - Added the distinct `pure-gpu-ipm` runtime policy, transactional device nonlinear handback,
    persistent same-pattern updates, complete timing/transfer telemetry, P1-C oracle assertions,
    and unavailable-library/no-fallback coverage.
- Validation:
  - Native P1-C reused one workspace for two solves/one update, accepted both steps, reached
    terminal `1.230e-13`, and produced ratios `0.999931728/0.999998351`; the Python oracle produced
    terminal `2.149e-13` and ratios `0.999896705/0.999998180`.
  - Corrected P1-D preserved complete quaternion/path checks and exact CQP quality; a three-step
    run accepted two then rolled back one rejected candidate. P1-E correctly rejected its adverse
    nonlinear candidate without changing the reference.
  - Release and Debug warnings-as-errors builds, 55 native tests, 157 Python tests, Ruff, and
    memcheck/racecheck/initcheck/synccheck passed.
- Follow-up notes / risks:
  - No performance, energy, or full matrix campaign was run.
  - P1-D and P1-E are not yet nonlinear-qualified by these short correctness runs.

## 2026-09-01 23:00 AEST

- Task summary:
  - Traced P1-C rejection retries, audited weighted trust cones, and wired persistent pure-QOCO
    nonlinear handback through the reference SCvx owner.
- Changes:
  - Added stage/terminal trust-distance and step-fraction telemetry to resident CUDA iterations.
  - Reused and closed one QOCO workspace across rejected trust updates and resolves; retained
    Clarabel compatibility for fixed solve settings.
  - Separated outer feasibility from step convergence/restoration and added zero-step/model-merit
    consistency coverage.
  - Added exact-dump and end-to-end pure-QOCO P1-C lifecycle runners.
- Validation:
  - Radius-1 displaced candidate: step fraction `0.5629445645`, stage/terminal trust distances
    both zero, predicted `0.4545113592`, actual `-18.0675875576`, ratio `-39.7516743866`.
  - Four adaptive PDHCG attempts used radii `1 -> 0.5 -> 0.25 -> 0.125 -> 0.0625`; zero accepted.
  - Pure-QOCO reference lifecycle: converged, two accepted, terminal `2.15e-13`, one workspace,
    one numeric update, two solves; model agreement approached one.
  - Ruff passed; Python `142 passed, 3 optional QOCO skips`; Debug/Release Werror builds passed;
    51 short CTests passed in each build; standalone production outer regression passed; CUDA
    memcheck/racecheck/initcheck/synccheck were clean.
- Follow-up notes / risks:
  - G4 remains FAIL for the resident PDHCG policies and G5 remains unauthorized.
  - A one-attempt G4 invocation is not globalization evidence; branch integration must run a
    multi-attempt policy budget and wire pure QOCO into the canonical C++ RK4 campaign owner.

## 2026-09-01 23:20 AEST

- Task summary:
  - Integrated completed P1-C globalization into the combined G4-G7 branch and made the repository
    Ruff-format clean without running GPU workloads.
- Changes:
  - Preserved additive CUDA metrics and histories while cherry-picking
    `ee2baa5826469a114ffbf4b8d6c2a99416cd2868`.
  - Formatted 45 files mechanically; verified AST identity for all 44 Python files.
  - Corrected the integration-only QOCO handback call to pass its candidate trust radius into the
    P1-C-expanded metric replay signature.
- Validation:
  - Ruff lint/format passed; full Python passed 175 with three optional skips; focused G4-G7 plus
    pinned QOCO CPU ABI/handback passed 85.
  - Native Release/Debug/ASan passed 43/43 each; standalone inventory passed 8/8 each.
  - CUDA/G5 Release, Debug, and sanitizer-capable `sm_120` builds passed; logical-rank tests passed.
  - CPU/CUDA QOCO builds, wheel ABI and both CLIs, and CMake consumer passed.
- Follow-up notes / risks:
  - C++ pure-QOCO RK4 campaign ownership remains in progress; no GPU or measured claim was made.

## 2026-09-02 01:45 AEST

- Task summary:
  - Integrated final native QOCO ownership and ran serialized unified G3-G7 GPU validation.
- Changes:
  - Applied unique source `56fb6d6792bfd73760efd0c7217da7cf950a64ae` as `c89d5ba`; skipped
    rewritten dependencies `99abb9e`/`e191a0d` because their P1-C semantics were already present.
  - Resolved CMake and CUDA-test conflicts additively, preserving native QOCO, OrbitWeaver, complete
    path inventory, unavailable-library classification, persistent reuse, and timing assertions.
  - Moved P1-C assertions after driver destruction so failed qualification repeats still release
    QOCO/cuDSS resources and emit complete diagnostics.
  - Updated G3/G4/G5/G6/G7 status documents without promoting implementation checks to acceptance.
- Validation:
  - Ruff and full Python passed (`175 passed, 3 skipped`); native Release/Debug/ASan passed 43/43
    and standalone inventory passed 8/8 in each build; all three CUDA matrices compiled.
  - P1-C individual native pure-QOCO correctness reproduced two accepts and terminal `1.230e-13`.
    P1-D finished at canonical `1.053e-11`, terminal `9.736e-12`, with 24 accepts/3 rejections.
    P1-E rejected 12/12 and exhausted trust at `1e-4`, retaining terminal `2.989e-5`.
  - QOCO GPU mapping passed 15/15 plus 3/3 canonical-reference tests. Core memcheck, racecheck,
    initcheck, and synccheck each reached zero errors; unused-memory tracking flags only third-party
    cuDSS/cuBLAS reserved buffers.
  - G5 one-rank MPI/NCCL and G7 one-GPU callback/route tests passed. G6 synthetic output was
    byte-reproducible across 52 files and freeze refusal passed.
- Follow-up notes / risks:
  - P1-C measured repeat correctness was 3/7, invalidating primary timing/energy inference.
  - Persistent-PDHCG representative modes were unqualified or timed out; P1-E all four modes timed
    out. The complete 24,883,200-row G4 ledger is not executed, H5/H6 remain unresolved, and G4
    remains failed.
  - No physical G5 2/4/8-GPU evidence, portable immutable archive, Paper 1 freeze, G7 scaling, or
    Paper 2 claim exists.

## 2026-09-02 03:00 AEST

- Task summary:
  - Fixed absolute-feasibility stopping for pinned QOCO and corrected P1-E penalty ownership,
    restoring repeatable P1-C and reachable P1-E globalization without changing G4 gates.
- Changes:
  - Added a declared, hash-base-checked QOCO patch requiring primal and dual residuals to meet
    `abstol`; gap retains QOCO's relative stopping semantics.
  - Added low-thrust-specific KKT regularization/refinement needed to reach the absolute gate.
  - Passed the driver-owned virtual penalty into numeric updates and removed powered-descent
    penalty overrides from the low-thrust fixture.
  - Added a seven-repeat P1-C regression at the actual frozen 0.01 coordinate and a P1-E CQP dump
    mode for independent-solver comparison.
- Validation:
  - P1-C passed 56/56 independent repeats with two accepts each, canonical residual
    `1.67e-13..3.30e-12`, and terminal residual `8.32e-14..1.79e-13`.
  - On the identical frozen P1-E CQP, Clarabel reached primal residual `2.91e-15`; CPU and GPU QOCO
    matched near `3.024e-7` before the fix, proving reachability and ruling out a CUDA-only cause.
  - Corrected P1-E passed 7/7, each with two accepted nonzero steps and terminal residual
    `3.13e-13..3.52e-11`.
- Follow-up notes / risks:
  - Absolute dual stopping removed a separate P1-D premature-success mode (`dres ~= 1.7e-2`);
    the final P1-D repeat set passed 7/7 with 24-25 accepts and 4-9 explicit rejections.
  - Post-fix memcheck, racecheck, initcheck, and synccheck each reported zero errors/hazards.
  - All 15 serialized P1-C/D/E representatives for fixed-tight, fixed-loose, adaptive,
    adaptive+polish, and hybrid reached explicit 120-second timeouts. These are retained negative
    records, not evidence of hybrid eligibility.
  - The complete matrix ledger, H5/H6, corrected measured timing/energy, and final archive sealing
    remain pending.
## 2026-09-02 01:40 AEST

- Task summary:
  - Built physical 2/4/8-GPU preflight, launch, evidence, failure-injection, archive, CI, and runbook
    tooling without executing or claiming multi-GPU validation.
- Changes:
  - Added fail-closed GPU/topology/toolchain/build capture and deterministic PCI/CPU/NUMA/NIC rank
    bindings.
  - Added 4,800-coordinate strong/weak logical plan generation, monolithic references, exact command
    manifests, schema snapshots, failure classifiers, GPU power/memory sampling, and partial logs.
  - Added compiled launch/failure harness modes for rank/MPI/order/cancel/checkpoint/topology/device/
    timeout paths and write-once reproducible archive seals.
- Validation:
  - Focused campaign/tooling tests passed 34/34; full Python passed 171 with three optional QOCO
    skips; full Ruff passed.
  - CPU warnings-as-errors build passed 41/41 CTests.
  - Debug, Release, and sanitizer-capable physical harness targets compiled.
  - All 4,800 logical commands validated against installed OpenMPI 4.1.2, CUDA 12.8, and NCCL
    2.26.2 without launching ranks.
- Follow-up notes / risks:
  - Negative WSL preflight correctly rejected physical execution: only one GPU, 82% free memory, and
    no physical PCIe/NUMA affinity. No new GPU runtime or scaling measurement was run.
  - The harness validates launch/collectives/failures only. P1-F integration and all physical
    2/4/8 correctness, quality, failure, energy, and scaling evidence remain required.
## 2026-09-02 01:40 AEST

- Task summary:
  - Reconciled the authoritative Paper 1 schema with the narrow initial G6 registry.
- Changes:
  - Added deterministic F09-F12 source/PDF/PNG and T07-T08 JSON/CSV/TeX generation.
  - Added paired-repeat regime winners, matched-quality Pareto fronts, variational and robust
    diagnostic completeness checks, full failure retention, claim-product links, schema/CI
    inventory enforcement, and versioned reconciliation documentation.
  - Added synthetic F11/F12 evidence and negative tests for missing diagnostics, manual
    coordinates, failure omission, and unsupported unique winners.
- Validation:
  - Ruff passed; 155 Python tests passed with three optional QOCO tests skipped.
  - The 70-file synthetic-only bundle reproduced byte-for-byte with aggregate SHA-256
    `b7b67930940b54b8a47ff4c061a46d36db10001093d952b2f367b434344b0275`.
  - An isolated wheel installed all rendering dependencies and built all 20 products; synthetic
    freeze was refused with exit code 2.
- Follow-up notes / risks:
  - G4 must supply paired timing repeats and F11 trial evidence; G5 must supply F12 risk-mode
    iteration evidence. No real scientific result or G6 PASS was claimed.

## 2026-09-02 04:25 AEST

- Task summary:
  - Integrated completed G5/G6/G7 roadmap code and implemented a fail-closed, crash-safe scheduler
    for the full frozen G4 ledger.
- Changes:
  - Integrated source commits `f6d4979`, `d24c987`, `7799742`, `34e1fbe`, `bf9d10a`, and
    `887450c`; skipped equivalent G7 schema parent `786c102`.
  - Preserved G4 QOCO/globalization semantics and merged memory histories additively.
  - Added content-addressed coordinate unranking, frozen solver rotation, SQLite checkpoints,
    append-only fsynced journaling, immutable attempt directories, crash recovery, process locking,
    record quarantine, actual process terminal classifications, and energy gap metrics.
- Validation:
  - Scheduler tests passed 3/3; focused integrated G5/G6/G7 tests passed.
  - Scheduler initialization reports exactly 24,883,200 total and remaining rows.
- Follow-up notes / risks:
  - Campaign launch is fail-closed at 0 completed because the current CUDA executable does not apply
    the frozen evaluation-seed or conditioning axes. No row was falsely executed or classified.
  - A production parameter emitter/capability record is required before the durable GPU worker may
    claim its first coordinate.
  - Current-head initcheck exposed 12 uninitialized Lambert result-padding bytes copied from device
    to host. Zeroing the full output transfer region before kernel field assignment fixed the ABI
    padding defect; current OrbitWeaver/G5 one-rank memcheck, racecheck, initcheck, and synccheck are
    clean.

## 2026-09-02 05:15 AEST

- Task summary:
  - Closed the G4 executable-axis gap with deterministic physical instances, equivalent
    conditioning transformations, complete runtime identities, strict capability generation and
    hybrid execution.
- Changes:
  - Added seeded powered-descent and rotationally equivalent low-thrust inputs, exact dynamics-row
    conditioning spans, CPU/GPU coefficient parity and deterministic numeric hashes.
  - Extended the G4 CLI/output with conditioning, evaluation seed, repeat kind/index, solver order,
    coordinate/matrix/capability hashes, complete requested/applied axes and diagnostics.
  - Added hash-mismatch refusal tests, strict frozen-coordinate validation, a stratified axis pilot,
    and a content-addressed capability generator.
  - Implemented hybrid PDHCG-to-QOCO handoff with the frozen `1e-6` qualification threshold and
    explicit ineligible telemetry.
- Validation in progress:
  - Focused scheduler/executor contract tests pass.
  - CUDA Release compiles; P1-C span-0 seeded qualification passes and span-2 conditioning passes
    with `1.78e-15` CPU/GPU coefficient parity.
- Follow-up notes / risks:
  - Final full validation, immutable commit, generated capability, complete pilot and new
    commit-pinned campaign launch remain required before any G4 evidence claim.

## 2026-09-02 09:45 AEST

- Task summary:
  - Changed the active completion goal to versioned one-GPU scope while preserving the original
    full multi-GPU campaign and all G5/OrbitWeaver distributed tooling.
- Changes:
  - Added machine-readable `single-gpu-v1`/`full-multi-gpu-v1` scope records and JSON schema.
  - Made G6 campaign configs, decisions, products, claim linkage and freeze seals scope-aware;
    historical configs remain readable, scoped products carry their ID, H4 is deferred, and
    F07/F12/T06 are explicit exclusions.
  - Kept full-campaign freeze fail-closed without physical P1-F 2/4/8-GPU evidence.
  - Added G7 schema-v2 scope identity and complete one-GPU
    coarse/refined/scenario/pricing-master/certification/visualisation acceptance semantics.
  - Reconciled the active roadmap, programme/G4/G5/G6/G7 status, Paper 1 claims/outline, and Paper 2
    outline without rewriting historical evidence or preregistered thresholds.
- Validation:
  - Full Ruff format/lint and generated-schema checks passed.
  - Full Python suite passed: 250 passed, 3 optional pinned-QOCO tests skipped.
  - Focused scoped G6/G7/schema tests passed; no GPU executable or campaign was run.
- Follow-up notes / risks:
  - `single-gpu-v1` still requires complete portable current-head G4 evidence before G6 freeze.
  - Physical G5 and distributed OrbitWeaver acceptance remain preserved in
    `docs/DEFERRED_MULTI_GPU_BACKLOG.md`.
## 2026-09-02 10:00 AEST

- Task summary:
  - Corrected audited G4 execution contracts and preregistered a separate H5/H6 claim-resolution
    core without running GPU work or reducing the frozen matrix.
- Changes:
  - Added a hash-pinned family×policy×axis applicability contract with exact executable,
    not-applicable, unsupported, QOCO dual-discard, and hybrid-handoff semantics.
  - Corrected physical instance and solver-order identity to include every applicable physical
    class and all order-relevant axes.
  - Changed the campaign scheduler to 2,764,800 persistent groups containing two same-session
    warm-ups and seven measurements, with separate raw attempt records and strict measured
    `paper1_result` validation.
  - Added exact `hybrid_handoff_ineligible`, `not_applicable`, and `unsupported` dispositions to
    schemas, semantic validation, decisions, and product failure retention.
  - Added actual-launch enforcement for timeout/OOM and a publication guard against claim-core
    substitution.
  - Added the pinned 360-group, 3,240-invocation P1-E/P1-C H5/H6 core and a CPU-only schedule
    inspection command.
- Validation:
  - Full relevant Python/schema suite: 101 passed.
  - Focused persistent capability/group suite after final changes: 29 passed.
  - Full repository Ruff check and format check passed.
  - Claim-core planner reported hash
    `40dc217467ffe32e919d4f901943e0200f69e302cf57cd15ccdfa88bfa0c8d0b`,
    360 groups, 720 warm-ups, 2,520 measurements, and 3,240 total invocations.
  - No GPU workload was launched.
- Follow-up notes / risks:
  - The batched native executor must implement `--g4-session` and emit the declared nine-attempt
    protocol before a grouped campaign can run.
  - Initialize a new grouped checkpoint after integration; do not mutate or reuse active
    row-oriented campaign checkpoints.
  - The claim core may resolve only H5/H6 and cannot populate full Paper 1 regime figures/tables.
## 2026-09-02 06:52 AEST

- Task summary:
  - Profiled the active G4 campaign and built the first fail-closed persistent execution layer in
    an isolated worktree.
- Changes:
  - Added a long-lived native CUDA request/result server and thread-safe in-process cancellation.
  - Replaced subprocess power polling with direct NVML, synchronized boundary samples and optional
    CPU affinity.
  - Added content-addressed compressed stdout retention and exact-once locked SQLite migration.
  - Added detailed native timing fields, recovery tests, and a bottleneck/launch-gate document.
- Validation:
  - Focused scheduler/executor tests passed 8/8; Ruff and `git diff --check` passed.
  - CUDA 12.8 RTX 5090 Release and Debug targets built successfully before the final deadline
    patch and were queued for rebuild afterward.
  - Direct NVML test: 41 samples, 0.057374 s maximum gap, valid cadence, no sampler errors.
- Follow-up notes / risks:
  - The current server retains one CUDA context but does not yet retain workspaces across rows.
    Migration remains blocked on workspace batching, full equivalence, sanitizer completion and a
    representative pilot.
  - The active integration campaign was preserved; at recovery it had 6 completed and 1 running
    row out of 24,883,200.

## 2026-09-01 22:05 AEST

- Task summary:
  - Isolated the displaced P1-C failure block-by-block and added complete recovery/merit telemetry.
- Changes:
  - Corrected actual-merit semantics, raw predicted-reduction sign handling, rejected warm-state
    rollback, converged no-op handling, and re-solve recovery cost accounting.
  - Extended the exact-dump comparator to arbitrary P1-C horizons and optional QOCO-GPU runs.
- Validation:
  - Exact P1-C still fails: recovery primal `1.016e-9`, recovery stationarity `3.853`,
    rolled-back canonical residual `5.654e-4`, zero accepted steps.
  - Identical-CQP QOCO-GPU reaches canonical primal `4.680e-7`, dual `9.090e-10`.
  - Focused Python tests pass 35/35; Release warnings-as-errors build and production outer
    regression pass.
- Follow-up notes / risks:
  - G4 remains failed. No H5/H6 matrix or G5 work was started.

## 2026-09-01 20:55 AEST

- Task summary:
  - Corrected the isolated P1-E displaced-reference qualification path without changing frozen
    policy, tolerances, matrix coordinates, or solver targets.
- Changes:
  - Replaced the invalid P1-E dispersion input with exact trust-radius/transfer-class parsing and
    deterministic reachable two-body low-thrust transfer targets.
  - Replayed from the immutable initial condition and added independent throttle, mass, and
    minimum-radius inventory telemetry.
  - Added all-vector displaced CPU/GPU coefficient and boundary parity, fixed-topology checks,
    adversarial accepted/rejected iterations, terminal reduction, and injected path violations.
- Validation:
  - Adversarial radius raise: 2 accepted steps, step fraction `0.84855828091422303`, scaled terminal
    error `1.7430149824386731e-3 -> 0`.
  - Short GPU production/recovery tests passed; maximum coefficient difference
    `2.7599450502791001e-13`, no topology allocation/copy, no hidden CPU fallback.
  - Full host suite: 41/41; full Python suite: 142 passed, 3 optional QOCO skips.
- Follow-up notes / risks:
  - The committed N=100/dispersion=0.01 P1-E failure is not a valid frozen coordinate. A corrected
    N=100 solve exceeded the short shared-GPU limit and was stopped, so no full qualification or
    performance claim is made.
  - Cherry-picking may require additive conflict resolution in shared CUDA outer-driver/test code
    and these append-only memory files if P1-C lands first.

## 2026-09-01 21:50 AEST

- Task summary:
  - Diagnosed the exact frozen P1-D forcing failure against Clarabel and pinned QOCO-GPU.
- Changes:
  - Removed the contradictory terminal quaternion tangent equation numerically while preserving
    the fixed CSC topology and CPU/CUDA coefficient parity.
  - Added exact P1-D CQP dump, bounded PDHCG iterate dump, active-set conditioning, canonical
    reference comparison, and explicit QOCO handoff qualification diagnostics.
- Validation:
  - Pre-fix Clarabel status was `PrimalInfeasible`; corrected Clarabel status is `Solved`.
  - Pinned QOCO-GPU solved in 20 iterations at canonical primal `1.091e-12` and dual `7.006e-12`.
  - Focused Release/Debug builds, 6-DoF transcription tests, recovery lifecycle/randomized tests,
    P1-D path audit, sanitizer modes, Ruff, and QOCO adapter tests passed.
- Follow-up notes / risks:
  - PDHCG remains unqualified at 300k iterations (`2.818e-2` natural residual), so its primal is
    rejected by the frozen `1e-6` hybrid handoff gate.
  - QOCO evidence is valid only as `pure-gpu-ipm`; fixed-tight PDHCG still has zero qualified
    accepted outer steps. Production nonlinear QOCO handback remains a separate dependency.

## 2026-09-01 22:30 AEST

- What changed:
  - Aligned the isolated QOCO branch with integrated G4 commit `a33e950`, then cherry-picked the
    corrected P1-D lifecycle and terminal-quaternion commits in the requested order.
  - Added native canonical primal/dual transfer, family-complete device nonlinear replay,
    fingerprint/permutation checks, frozen merit/trust acceptance, and transactional handback.
  - Added a strict pre-QOCO PDHCG gate; the known `2.818e-2` predictor remains explicitly
    ineligible and cannot be relabelled hybrid.
  - Added distinct pure/hybrid labels, timing and dual-disposition records plus HCW/P1-C/P1-D/P1-E
    and negative handback fixtures.
- Validation:
  - 156 Python tests passed with pinned CPU QOCO and the host native library.
  - Release host build and 41/41 CTest tests passed.
  - `device_scvx.cu` and its integration fixture compiled with CUDA 12.8 for `sm_120`.
  - Changed Python files pass Ruff lint and format checks; `git diff --check` passes.
- Follow-ups / risks:
  - The dedicated short RTX `device_scvx_qoco_handback_test` is compiled but unrun while the GPU is
    reserved. No performance, energy, or matrix run was performed.

## 2026-09-01 21:15 AEST

- Task summary:
  - Integrated and repaired the complete standalone `cpp/native` implementation/test inventory.
- Changes:
  - Added eight previously omitted implementation files to `spacepdhcg_native_core` and registered
    all seven existing smokes plus a new six-DoF smoke as explicit CTest targets.
  - Added native-core sanitizer controls and GCC/Clang Release, Debug, and ASan/UBSan CI coverage.
  - Fixed the explicit-constructor warning, optional-agreement warning, nodiscard checks, stale test
    namespace assumptions, and robust CVaR/base-layout test mismatch without changing the C ABI.
  - Ignored `build-*/` outputs to prevent generated CTest/compiler files entering source archives.
- Validation:
  - Native inventory: 8/8 in Release, 8/8 in Debug, 8/8 under ASan/UBSan, all with Werror.
  - Existing top-level Release/Debug/ASan suites: 41/41 each; core 5/5; all-native 41/41;
    CMake consumer 1/1; Ruff clean; Python 142 passed with 3 optional QOCO skips; parity 1/1.
  - Editable, native wheel, sdist, wheel consumer, and sdist consumer import/ABI checks passed.
- Follow-up notes / risks:
  - No code was classified obsolete: commit history and dedicated native CI establish this as an
    active parallel native-core API. Conceptual overlap with newer headers is not replacement proof.
  - GPU tests and measured benchmarks were intentionally not run.

## 2026-09-01 22:12 AEST

- Task summary:
  - Prepared Gate G5 in isolated branch `feat/scenario-aware-multigpu`; this is implementation and
    one-rank correctness only, not G5 acceptance.
- Changes:
  - Added persistent rank-local G2 composition, deterministic MPI-rank/CUDA-device mapping,
    MPI/NCCL stream-event runtime, scenario-aware and nonzero partitions, device algebra,
    risk/residual/status reductions, telemetry, checkpoint/restart, and cancellation.
  - Added 1/2/4/8 logical-rank tests, real one-rank MPI/NCCL/CUDA coverage, G5 evidence schema,
    CMake/CTest build targets, manual self-hosted CI, and implementation status documentation.
  - Installed OpenMPI 4.1.2 and NCCL 2.26.2 for CUDA 12.8 only; no driver package was installed.
- Validation:
  - Pinned upstream PDHCG commit `167c8b7`/tree `62b05e6` built all 163 distributed steps.
  - Debug, Release, warnings-as-errors, and sanitizer-capable G5 targets compiled.
  - Logical 1/2/4/8 tests passed; schema/claim-drift tests passed 6/6; Ruff passed.
  - Idle RTX 5090 Release one-rank MPI/NCCL/CUDA CTest passed; memcheck was leak/error clean and
    racecheck reported zero hazards.
- Follow-up notes / risks:
  - A later idle window completed the final-HEAD one-rank rerun, initcheck, and synccheck cleanly.
    Initcheck unused-memory mode separately flags NCCL's own 2 MiB communication pools.
  - Actual overlap execution remains deferred to physical multi-GPU validation.
  - Physical 2/4/8 correctness, rank-failure behavior, strong/weak scaling, and H2/H3/H4 remain
    unverified. G5 must not be reported as PASS.

## 2026-09-01 21:50 AEST - G6 freeze tooling

- Task summary:
  - Implemented Paper 1 G6 aggregation/freeze tooling in the isolated
    feat/paper1-freeze-tooling worktree.
- Changes:
  - Added strict archived-evidence envelopes, immutable hash verification, run indexing, frozen
    F01-F08/T01-T06 source and publication builders, H1-H6 deterministic decision records,
    campaign freeze/refusal, checksums, claim linkage, and clean-clone verification.
  - Added synthetic failure/censoring matrix, schemas, CLI, documentation, and CI.
- Validation:
  - Ruff, 151 Python tests, and the native wheel build passed.
  - Generated a 14-run synthetic-only bundle with 52 deterministic output files and verified a
    byte-identical second build.
  - No GPU benchmark or real scientific campaign was run.
- Follow-up notes / risks:
  - Real freeze remains blocked on complete portable G4/G5 evidence. Existing G4 failures and G5
    authorisation state are not altered by this tooling.

## 2026-09-01 22:24 AEST

- Task summary:
  - Integrated all requested completed G4-G7 implementation commits on isolated branch
    `integration/roadmap-code` from stable base `a33e950`.
- Changes:
  - Resolved shared CUDA outer-driver, QOCO handback, runtime-coordinate, CLI entry-point, and
    append-only history conflicts without weakening classifications, policies, or tests.
  - Preserved P1-E inventory, P1-D path/forcing corrections, P1-C lifecycle accounting, pure/hybrid
    QOCO distinction, G5 interfaces, G6 tooling, and G7 CPU/CUDA seams.
- Validation:
  - Ruff lint passed; full Python suite passed 174 tests with three optional QOCO skips.
  - Pinned CPU QOCO ABI/handback suite passed 29/29 when supplied `libqoco.so`.
  - Top-level native Release, Debug, and ASan/UBSan each passed 43/43; standalone native-core
    Release, Debug, and ASan/UBSan each passed 8/8, all with warnings as errors.
  - CUDA/G5 Release, Debug, and sanitizer-capable configurations built all 77 targets for `sm_120`;
    logical-rank contracts passed in Release and Debug.
  - Pinned QOCO CPU and CUDA/cuDSS builds, native wheel ABI, both packaged CLIs, and the CMake
    consumer passed. No GPU executable, measured benchmark, energy campaign, sanitizer execution,
    or multi-GPU scaling run was launched.
- Follow-up notes / risks:
  - Full-repository Ruff format check reports 43 pre-existing/unintegrated files would be reformatted;
    Ruff lint itself is clean.
  - Add the forthcoming P1-C trust-globalization commit by ordinary cherry-pick, preserve histories
    additively, rerun CPU/static matrices, then serialize deferred GPU validation after contention
    clears.

## 2026-09-01 23:00 AEST

- Task summary:
  - Traced P1-C rejection retries, audited weighted trust cones, and wired persistent pure-QOCO
    nonlinear handback through the reference SCvx owner.
- Changes:
  - Added stage/terminal trust-distance and step-fraction telemetry to resident CUDA iterations.
  - Reused and closed one QOCO workspace across rejected trust updates and resolves; retained
    Clarabel compatibility for fixed solve settings.
  - Separated outer feasibility from step convergence/restoration and added zero-step/model-merit
    consistency coverage.
  - Added exact-dump and end-to-end pure-QOCO P1-C lifecycle runners.
- Validation:
  - Radius-1 displaced candidate: step fraction `0.5629445645`, stage/terminal trust distances
    both zero, predicted `0.4545113592`, actual `-18.0675875576`, ratio `-39.7516743866`.
  - Four adaptive PDHCG attempts used radii `1 -> 0.5 -> 0.25 -> 0.125 -> 0.0625`; zero accepted.
  - Pure-QOCO reference lifecycle: converged, two accepted, terminal `2.15e-13`, one workspace,
    one numeric update, two solves; model agreement approached one.
  - Ruff passed; Python `142 passed, 3 optional QOCO skips`; Debug/Release Werror builds passed;
    51 short CTests passed in each build; standalone production outer regression passed; CUDA
    memcheck/racecheck/initcheck/synccheck were clean.
- Follow-up notes / risks:
  - G4 remains FAIL for the resident PDHCG policies and G5 remains unauthorized.
  - A one-attempt G4 invocation is not globalization evidence; branch integration must run a
    multi-attempt policy budget and wire pure QOCO into the canonical C++ RK4 campaign owner.

## 2026-09-01 23:20 AEST

- Task summary:
  - Integrated completed P1-C globalization into the combined G4-G7 branch and made the repository
    Ruff-format clean without running GPU workloads.
- Changes:
  - Preserved additive CUDA metrics and histories while cherry-picking
    `ee2baa5826469a114ffbf4b8d6c2a99416cd2868`.
  - Formatted 45 files mechanically; verified AST identity for all 44 Python files.
  - Corrected the integration-only QOCO handback call to pass its candidate trust radius into the
    P1-C-expanded metric replay signature.
- Validation:
  - Ruff lint/format passed; full Python passed 175 with three optional skips; focused G4-G7 plus
    pinned QOCO CPU ABI/handback passed 85.
  - Native Release/Debug/ASan passed 43/43 each; standalone inventory passed 8/8 each.
  - CUDA/G5 Release, Debug, and sanitizer-capable `sm_120` builds passed; logical-rank tests passed.
  - CPU/CUDA QOCO builds, wheel ABI and both CLIs, and CMake consumer passed.
- Follow-up notes / risks:
  - C++ pure-QOCO RK4 campaign ownership remains in progress; no GPU or measured claim was made.

## 2026-09-02 01:15 AEST

- Task summary:
  - Ran the isolated CPU/reference inventory and component-fixture campaign from frozen commit
    `e95b902d718ceaf05523e469cbe21945013c2f41`.
- Changes:
  - Added a fail-closed CPU campaign finalizer that expands both frozen matrices, archives native
    JUnit and Python logs, validates censored Paper 1 records, builds G6 F01-F08/T01-T06 sources,
    renders provenance-complete diagnostics, verifies hashes twice, and emits a Canvas-ready
    dashboard JSON.
  - Added tests locking the 16,324-coordinate cardinality and chart provenance contract.
- Validation:
  - Ruff lint/format passed for all 140 Python files.
  - Python: 177 passed, three optional pinned-QOCO skips.
  - Top-level host native: 43/43; standalone native core: 8/8.
  - G6: 52 files reproduced byte-for-byte, aggregate
    `41f7818ef8f50d1d010a935ad5b2eef03f10aed6b5610e85e117ed9343d0c0ed`.
  - Diagnostic rendering digest matched twice:
    `ef14819333f90d0f68dde7920f0976b328976bdeb53c394ba8e2d8ff8e44030e`.
- Follow-up notes / risks:
  - All 13,676 Paper 1 matrix coordinates remain explicit `unrun` because this commit has no full
    production emitter for the frozen residual/replay/repeat/resource contract.
  - Paper 2 has 1,640 component-backed `unrun` coordinates and 1,008 unsupported P2-D/P2-E
    full-mission coordinates; all 43 implemented native component fixtures passed.
  - No GPU API, executable, sampler, timing, or energy path was used.

## 2026-09-02 02:05 AEST

- Task summary:
  - Executed and retained a bounded CPU/reference disposition for every frozen Paper 1/Paper 2
    matrix coordinate without GPU use.
- Changes:
  - Added a complete coordinate result schema, checkpointed parallel campaign, native corrected
    6-DoF/low-thrust emitters, independent validators, deterministic diagnostics, and dashboard.
  - Parameterized HCW numerical update magnitude and P1-C initial dispersion/final Clarabel polish.
  - Fixed P1-D to preserve its 10-second physical horizon and replaced a P2 scale cap with explicit
    timeout retention.
- Validation:
  - Exact coverage: 16,324 records, zero schema errors/missing/duplicates.
  - Dispositions: 6,912 executed/qualified, 2,844 unqualified, 5,560 timeouts, 1,008 unsupported.
  - Ruff and format passed for 143 files; Python passed 180 with three optional QOCO skips.
  - Native host passed 45/45; standalone native core passed 8/8.
  - Semantic source SHA-256:
    `54a732bfeba826ffc7b95a5c4bfc5adf9eaf5689f74a694085eec3a02ed352bb`.
- Follow-up notes / risks:
  - Unqualified records prove only the explicitly named component/reference computation; they are
    not complete publication evidence.
  - Remaining CPU gaps are native P1-D/P1-E optimizer residual ownership, full robust P1-F solves,
    and parameterized physical Paper 2 mission optimization.

## 2026-09-02 09:50 AEST

- Task summary:
  - Recovered the interrupted live run and completed/validated all 16,324 exact coordinates.
- Changes:
  - Added complete Clarabel KKT and robust nonlinear replay ownership in prior campaign commits.
  - Reduced checkpoint cadence to 25 and guaranteed a final checkpoint.
  - Preserved completion timestamps during replay rendering and documented the exact Paper 2
    physical-instance contract gap.
- Validation:
  - Final dispositions: 12,148 executed, 384 failed, 259 numerical, 645 actual timeouts, 1,880
    unqualified, and 1,008 unsupported.
  - Matrix coverage/schema passed; 184 Python tests passed with three optional QOCO skips; Ruff
    passed; native CPU tests passed 45/45.
  - Semantic SHA-256:
    `649c7e00106e4d10bd9c960e6de36f837a8dd2dc1ea9118d704dbf63e9756f18`.
  - Two independent render finalizations matched:
    `41c2a80f409898381ce3e2cd76697239f199d2da3eb42a6a781873e5cedaefe1`.
- Follow-up notes / risks:
  - P1-C/D/E/F and Paper 2 physical-route publication evidence remains fail-closed where complete
    solver/formulation or physical benchmark inputs do not exist.
  - GPU timing, energy, and 2/4/8-GPU collective telemetry remain hardware-only.
## 2026-09-02 10:20 AEST

- Task summary:
  - Produced compact, verified trajectory visualisation evidence in an isolated worktree without
    running a GPU workload or manufacturing paths from scalar metrics.
- Changes:
  - Added a header-only C++ state-history emitter for exact-source P1-D/P1-E and G7 Lambert
    requests, including dense nonlinear RK4 replay.
  - Added a Python extractor that verifies archive checksums, recreates P1-B/P1-C solver paths,
    validates source metrics, applies deterministic constraint-aware decimation, and renders
    record-specific XY/XZ/YZ/perspective PNG/PDF previews.
  - Added decimation/classification tests and ignored generated visualisation outputs.
- Validation:
  - Ruff and `git diff --check` passed; focused pytest passed 3/3.
  - All selected CPU archive files matched `checksums.json`; archived campaign validation retained
    zero missing, duplicate, or schema-error records.
  - Five paths passed finite/dimension/endpoint checks. P1-B is qualified; P1-C/P1-D/P1-E remain
    explicitly unqualified; P2 is a CPU replay of the exact request from a passing one-GPU Lambert
    parity test.
  - Two independent output builds produced byte-identical compact datasets and previews.
- Follow-up notes / risks:
  - No real archived P2 coarse/refined/route trajectory state arrays were found.
  - G4/P1-A remains non-trajectory CQP evidence and is excluded from visualisation geometry.

## 2026-09-02 18:22 AEST

- Task summary:
  - Created the isolated `integration/single-gpu-v1` consolidation from `27ef966`.
  - Integrated single-GPU scope, grouped G4 contracts, compatible persistent transport, CPU campaign tooling, trajectory extraction, and the isolated static viewer commit.
- Conflict resolutions:
  - Preserved all memory/devlog histories additively.
  - Combined grouped scheduling with crash-safe terminal migration and imported-ordinal skipping.
  - Excluded failed protocol-v2 lane scheduling while retaining its negative migration-gate report.
  - Combined OrbitWeaver adapter and physical-instance exports.
  - Kept scoped G6/G7 implementations already conflict-resolved in the roadmap base.
- Integration fixes:
  - Normalized merged Python sources with Ruff.
  - Declared `jsonschema>=4.26` as a runtime dependency after fresh-wheel CLI verification exposed the omission.
- Validation:
  - Ruff lint/format, 305 Python tests with native and QOCO, generated G4/G7 schemas, G4 claim-core hash/counts, web deterministic import/check/tests, native Release/Debug/ASan+UBSan inventories, wheel/sdist, fresh CMake consumer, CUDA 12.8 sm_120 Release/Debug/sanitizer builds, selected QOCO/G5/G7 GPU correctness tests, and compute-sanitizer tools passed.
  - G6 synthetic builds were byte-identical and real freeze correctly refused synthetic evidence.
  - No campaign worker was started and no ignored raw campaign evidence was added.
- Follow-up notes / risks:
  - Reseal G0-G3 from the final committed head before any campaign resume.
  - Grouped G4 requires an executor capability implementing `--g4-session`; never reuse the old row checkpoint.
  - Hardened capability generation to reject the currently row-oriented executable until it
    declares the complete `g4-persistent-group-v1` process/workspace contract.
  - Physical 2/4/8-GPU G5 and distributed OrbitWeaver remain explicitly deferred.

## 2026-09-02 19:20 AEST

- Task summary:
  - Implemented native `g4-persistent-group-v1`: one process/context/workspace, two warm-ups, seven
    measurements, deterministic reset/retention, strict partial records, and direct NVML boundaries.
- Changes:
  - Added allocation-free SCvx/QOCO attempt reset, native group parsing/hash refusal, exact terminal
    dispositions, timing/work/resource/source telemetry, crash-safe flush/exit behavior, and one
    scheduler restart after abnormal process exit.
  - Added contract/schema hashes plus a mandatory real CUDA probe to capability generation.
  - Added independent 360-group claim-core checkpoints via `--claim-core`; full grouped scheduling
    remains 2,764,800 sessions and cannot share the historical row checkpoint.
- Validation:
  - Ruff and full Python passed: 309 passed, four optional skips; focused session contracts passed.
  - Native Release/Debug/ASan+UBSan passed 45/45 each; CUDA Release/Debug/sanitizer matrices built
    for CUDA 12.8 `sm_120`.
  - Isolated native session/legacy-row equivalence passed 8/8; QOCO mapping passed 29/29; native
    handback passed 1/1.
  - Session memcheck/initcheck/synccheck reported zero errors; racecheck reported zero hazards,
    errors, and warnings. The briefly overlapped first QOCO run was excluded and rerun serially.
- Follow-ups / risks:
  - Rebuild after the final commit, generate the executable/hash-pinned capability from the clean
    tree, then initialize a new claim-core or full grouped checkpoint.
  - No 3,240-invocation claim core or 2,764,800-session campaign was run; H5/H6 remain unresolved.

## 2026-09-03 02:35 AEST

- Task summary:
  - Turned the comparative-campaign specification into runnable Phase 0-1 targets on
    `feat/literature-targets` (worktree `/home/angus/worktrees/spacepdhcg-literature`, based on
    `b6afb49`); integration and planner worktrees untouched.
- Changes:
  - `715c1db` spec import (user's campaign doc, literature baselines, matrix/protocol/outline
    edits, manifest tests; text verbatim, CRLF normalised).
  - `196b3ba` G7 contracts accept both the sealed and the P2-F-extended Paper 2 matrix digests;
    CPU ledger counts updated (2,684 / 16,360).
  - `4f92133` `src/spacepdhcg/literature/` (provenance, external sources, registry, free-final-time
    SCvx core, P1-C/P1-D/P1-D-MC/P1-E/TOPS/GTOPX/GTOC runners, report, CLI), `spacepdhcg`
    console script, `benchmarks/literature/*` (targets, profiles, provenance store with 126
    records, external pins, frozen TOPS/GTOC selections, seeded Chari samples), schema, six test
    modules.
  - Docs: `docs/LITERATURE_TARGETS.md`, `docs/REFERENCE_REPRODUCTION_REPORT.md` (+ JSON twin and
    `results/literature/*.json`), Track L in `docs/ACTIVE_SINGLE_GPU_ROADMAP.md`, README links,
    campaign-doc status section.
- Validation:
  - Ruff check (whole repository) and Ruff format on all touched files: clean.
  - Full Python suite: 354 passed, 4 skipped, 1 environmental failure
    (`test_native_packaging`, no native library in the fresh venv); the same test passes with
    `SPACEPDHCG_NATIVE_LIBRARY` pointing at an existing build (2 passed). Literature tests: 46.
  - `scripts/literature/build_provenance.py --check`: store up to date; `git diff --check` clean.
  - Reproductions (CPU, RTX 5090 owned by a G4 session all night): Acikmese-Ploen 2007 lossless
    SOCP 400.63 kg vs 399.5 kg (+1.13 kg, 0.28 %); Blackmore 2010 case 1 400.09 vs 399.4 kg;
    repository Euler SCvx 405.65 / 413.43 kg (gap); Szmuk 2018 free-final-time t_f = 3.3901 UT
    with all ten guesses within 0.00054 UT (published: within 0.01 UT); Earth-Mars 603.925 vs
    603.935 kg; Earth-Dionysus not converged (gap); TOPS P4 converged, P3/P1 iteration-limited,
    CR3BP unsupported; GTOPX Cassini1/Rosetta/Messenger-reduced/GTOC1 reproduce the official
    objectives to printed precision (three exactly); GTOC12 official verifier accepts the bundled
    example and both published solutions (338 asteroids / 27,045.3 kg; 356 / 28,975.1 kg);
    GTOC9 examples 1 and 2 validate under the re-implemented rules 4-19; GTOC5 scoring blocked.
  - No native C++/CUDA transcription or kernel was changed, so no Release/Debug Werror, CUDA
    parity, or sanitizer pass was run; no GPU workload was launched.
- Follow-up notes / risks:
  - GPU legs (P1-C pure-QOCO, P1-D-MC persistent batch) blocked; commands recorded in the report.
  - Native `sigma` free-final-time kernels deferred (topology and policy-hash impact).
  - Multi-revolution low-thrust convergence (Dionysus, TOPS P3) needs a better initial guess.
