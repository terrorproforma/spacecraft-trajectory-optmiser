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

## 2026-09-03 00:57 AEST

- Task summary:
  - Ran, repaired, reran, and sealed complete current-head G0-G3 evidence before G4.
- Changes:
  - Fixed H1 parsing so unrelated non-JSON `-inf` diagnostics cannot abort the result owner.
  - Replaced the nominal HCW self-target production fixture with a reachable displaced start and
    required an accepted nonzero outer step.
  - Added compact current-head gate reports while preserving all raw failures in ignored evidence.
- Validation:
  - G0: Ruff; 314 Python tests; 45 top-level and eight standalone tests in RelWithDebInfo, Debug,
    and ASan+UBSan; wheel/sdist plus isolated wheel and CMake consumers.
  - G1: 15 exact-optimum cases, 96 declared box/SOC solves, 16 updates, all tolerances/starts, and
    expected no-device refusal against exact pinned PDHCG.
  - G2: 62 Debug and 62 RelWithDebInfo CTests, QP/SOCP lifecycle, real CuPy/PyTorch/JAX producers,
    and five clean sanitizer records.
  - G3: 62 Debug and 62 Release CTests, displaced HCW and pure-QOCO P1-C/D/E, honest PDHCG
    negatives, H1, sixteen clean sanitizer records, and retained WSL Nsight limitations.
- Follow-ups / risks:
  - G4 is scientifically authorised, but launch is blocked until a new official capability is
    generated for the final clean report descendant. The old b0cd570 capability was not reused.
  - All archives remain local-only; no immutable URI exists.

## 2026-09-03 02:40 AEST

- Task summary:
  - GTOC12 replay track end-to-end on `feat/gtoc12-asteroid-mining` (worktree
    `/home/angus/worktrees/spacepdhcg-gtoc12`, base `96781349`, CPU only).
- Changes:
  - `benchmarks/gtoc12/`: `pins.json` (URLs, sizes, SHA-256 for nine official files),
    `gtoc12_rules.json` (machine-readable rules), `reduced_instance_v1.json` (preregistered rule).
  - `scripts/gtoc12/fetch_gtoc12_data.py`: checksummed fetch into the ignored data directory.
  - `src/spacepdhcg/gtoc12/`: constants, data, ephemeris, solution format, independent verifier,
    official-binary wrapper, Lambert (NumPy port + native ctypes parity), screening, reduced
    instance, beam search, ZOH SCvx low-thrust arcs, G7-adapter pipeline, viewer export, CLI.
  - `spacepdhcg` console script (`spacepdhcg gtoc12 verify|fetch|reduced-instance|run|export-viewer`).
  - Imported the user's `docs/COMPARATIVE_SOLVER_CAMPAIGN.md` and `benchmarks/literature_baselines.json`.
- Validation:
  - Official verifier on archived references: 39/356/28975.1 kg, 37/338/27045.3 kg,
    36/320/26062.6 kg; independent verifier reproduces every per-asteroid mass to 0.0 kg and the
    fixed-bonus weighted scores (24474.15 for the 39-ship file vs published 24474.16).
  - Lambert NumPy vs native kernel: 1e-13 km/s over 300 short/long-way legs.
  - First pipeline route (reduced instance, 2 asteroids): official "Check successfully!"
    195.044 kg; independent 195.04449 kg, max propagation error 0.91 km / 0.15 mm/s.
  - Ruff clean; 24 fast GTOC12 tests pass; full suite launched in background.
- Follow-up notes / risks:
  - Reduced-instance search currently completes mostly one- and two-asteroid chains.
  - Full-catalogue screening and docs/GTOC12_TRACK.md are the next commits.

## 2026-09-03 03:35 AEST

- Task summary:
  - Fixed the route search and emitter, then produced officially verified GTOC12 solutions on the
    preregistered reduced instance and on the full catalogue.
- Changes:
  - `search.py`: phase-drift-aware neighbour proxy, deploy waits, 600-day return/collection
    windows, return-feasibility pruning, per-set diversity, vectorised first level.
  - `pipeline.py`: camp-then-collect visits emit a single collect event at departure.
  - `verifier.py`: 1 uN thrust slack (JPL reference file), 1 uday sample-interval slack.
  - `docs/GTOC12_TRACK.md`, `benchmarks/gtoc12/reference_reproductions.json`, committed compact
    result artifacts under `results/gtoc12/`.
- Validation:
  - Official verifier: reduced-v1-run2 1 ship / 4 asteroids / 253.744 kg (fixed bonus 249.059 kg),
    47 s wall, 8 refined arcs; full-catalogue-run1 1 / 3 / 249.035 kg (fixed bonus 202.995 kg),
    963 s wall; reduced-v1-run1 1 / 2 / 195.044 kg.
  - Independent verifier agrees per asteroid to 1e-10 kg; max propagation error 0.46 km, 8e-5 m/s.
  - Full pytest: 341 passed, 4 skipped, 1 environmental failure (native wheel absent); GTOC12
    verifier tests (all three reference reproductions) pass after the thrust-slack fix.
- Follow-up notes / risks:
  - Scores are ~1/3 of a single archived reference ship; the search, not the refinement, is the
    bottleneck at catalogue scale.

## 2026-09-03 04:50 AEST

- Task summary:
  - Raised the officially verified GTOC12 scores by rebuilding the route search around the
    structure of the archived JPL/Antipodes solutions, adding greedy fleets and bounded memory.
- Changes:
  - `gtoc12/references.py`: decode any solution file into per-ship itineraries and fleet
    statistics (roles from global visit order, TOF/propellant/transfer angle/revolutions,
    cooperative vs self-cleaning, launch spread, element spreads).
  - `gtoc12/proxies.py`: Lambert-free phasing/Edelbaum ΔV pre-ranker (e/i vectors) and a
    mass-consistent thrust-authority test; `scripts/gtoc12/proxy_validation.py` writes the
    proxy-error distributions on 164 certified legs and 1882 reference hops.
  - `gtoc12/search.py`: position-space candidate generation (Δa, Δe-vector, relative
    inclination, phase at departure), element-band pool filter with sparse fallback,
    collect-phase propellant reserve, greedy backward collection order with reverse fallback and
    escalating wait penalty, separate collect-hop TOF grid (90-720 d), per-first-asteroid
    diversity, block-wise Earth-leg screening, wall-clock budget with partial results, failure
    reasons retained, asteroid exclusion for fleets.
  - `gtoc12/fleet.py` + `cli.py --ships/--search-budget-seconds/--pool-*`: greedy fleet
    construction, fleet-rule check, assembled multi-ship file verified as a whole, failed-leg
    skipping across refine candidates, peak-RSS reporting.
  - `docs/GTOC12_TRACK.md`: reference-structure table, proxy validation table, results table,
    rejected variants, limitations and next bottleneck.
- Validation:
  - Official `GTOC12_Verify` "Check successfully!" on every emitted file; independent verifier
    agrees per asteroid to 1e-10 kg.
  - reduced-v1: 314.442 kg (5 asteroids, 10 arcs, 228 s, 0.55 GB) — was 253.744 kg.
  - full catalogue single ship: 548.282 kg (8 asteroids, 16 arcs, 303 s, 0.66 GB) — was 249.035 kg
    at 11.2 GB.
  - 3-ship greedy fleet: 1394.11 kg (548.28 + 442.22 + 403.61; 20 asteroids, 40 arcs, 867 s,
    0.77 GB; fixed-bonus 1318.12 kg; rule 3 ≤ 12.8).
  - Proxies: refined/proxy propellant median 0.96 (p5 0.66, p95 1.08); refined/Lambert ΔV
    median 1.165 (p95 1.49); reference true/Lambert 1.16 (p95 1.41), Spearman 0.90.
  - `pytest`: 349 passed, 4 skipped (native packaging test deselected: no cmake wheel); Ruff clean.
- Follow-up notes / risks:
  - Chains stop at 8 asteroids because the collection tour no longer fits the window
    (`camp_negative`), with 230-430 kg propellant unspent; cluster-first generation and joint
    re-timing are the next levers. Greedy fleets thin the clusters for ships 2-3.
  - `full_catalogue_search2` and `fleet3_full_catalogue` artifacts come from an intermediate
    commit state (documented); `reduced_v1_search3` and `fleet3_full_catalogue_v2` reproduce from
    HEAD.

## 2026-09-03 09:40 AEST

- Task summary:
  - GTOC12 track, second campaign: convert unspent propellant into score (joint re-timing),
    scale the fleet to 10 ships, add clusters, cooperative collection and the fleet master.
    Best verified fleet 1394.1 kg (3 ships) -> 4398.7 kg (10 ships, 62 asteroids).
- Changes:
  - `gtoc12/retiming.py` (new): exact DP over a 15-day lattice re-choosing every epoch of a fixed
    visit order (bonus-weighted mined mass minus propellant price; cached Lambert tables per
    body pair; authority ratio 0.45 for hops), forward-mass re-check, propellant price loop
    (grow until it closes, halve while it keeps closing, bisect twice), chain extension
    (self-cleaning / orphan / foreign-collect insertions), SCvx-in-the-loop certification with
    per-pair bans and calibration. Fixed a DP regression where deploy-only visits of asteroids
    collected later were priced at the orphan credit (0) instead of the full mining rate.
  - `gtoc12/clusters.py` (new): co-moving families (a, e-vector, i-vector, mean longitude) via
    cKDTree, density labelling, precomputed phasing windows, element deviations.
  - `gtoc12/cooperative.py` (new): `MinerPool` (deploy-once / collect-once, orphans), orphan
    credit, `FleetColumn`, exact branch-and-bound `solve_fleet_master` (fixed-bonus objective,
    asteroid uniqueness, deployer-in-fleet for foreign collects, fleet rule, ship cap).
  - `gtoc12/search.py`: `RoutePlan.foreign_deploy_epochs`, per-role leg model
    (inflation, authority ratio), bonus-weighted scoring, cluster prior (off by default),
    seed bonus for uncovered clusters; `pipeline.py`: collected mass from the deployer's epoch;
    `cli.py`: `--retime*`, `--no-cooperative`, pool + master at every checkpoint, timeline.
  - `tests/test_gtoc12_cooperative.py` (new, 15 tests): plan/pool rules, orphan credit, master
    feasibility (each asteroid once, foreign dependency, ship rule), visit orders, cluster
    determinism, phasing windows, re-timer bookkeeping/determinism, orphan-credit invariance,
    cooperative extension variants.
  - `docs/GTOC12_TRACK.md`: section 6.4, results rows `fleet6_retime_v1`, `fleet6_coop_v1`,
    `fleet10_master_v1`, before/after tables, master stats, score-vs-budget, rejected variants,
    limitations and the fleet-rule bottleneck.
- Validation:
  - Official `GTOC12_Verify` "Check successfully!" on the 10-ship file (4398.686 kg, 62
    asteroids, fleet rule 10 <= 11.6) and on every per-ship file; independent verifier agrees
    to 1e-10 kg per asteroid (max position error 23.6 km, within tolerance).
  - `fleet10_master_v1`: 3089 s wall (51 min), peak RSS 0.76 GB; 30 min -> 2378.6 kg (5 ships);
    per-ship before -> after re-timing 3614.9 -> 4398.7 kg (+21.7 %), 9/10 ships certified a
    re-timed variant, 124 refined arcs. Master: 31 columns, 200k nodes (cap), incumbent = greedy.
  - `fleet6_retime_v1` 2744.89 kg (verified); `fleet6_coop_v1` 2641.81 kg (verified, rejected
    variant: orphan credit 0.5 left nine uncollected orphans).
  - Proxy probe on the archived ship-1 plan: 548.3 -> 583.2 kg (DP fix + price loop) vs 564.7
    (previous loop) vs 545.0 (regressed DP).
  - `PYTHONPATH=src .venv/bin/python -m pytest -q --deselect tests/test_native_packaging.py`:
    360 passed, 4 skipped; `ruff check` / `ruff format --check`: clean.
- Follow-up notes / risks:
  - Fleet size is capped by the fleet rule at the current 440 kg average (11 ships); the gap to
    the references is per-ship mass, which needs deployer + collector planned jointly inside one
    co-moving cluster (cooperative columns for the master) and parallel pricing workers.
  - Calibration after a certified attempt can raise hop inflations above 1.2x and make the
    second attempt worse than the first (ship 1: 583 certified, attempt 2 "no proxy
    improvement"); the best certified route is kept, so this only costs time.
  - `results/gtoc12/runs/fleet10_master_v1/fleet/viewer/trajectories.json` (3.6 MB) is not
    committed; regenerate with `python -m spacepdhcg gtoc12 export-viewer` from the committed
    `fleet/Result.txt`.

## 2026-09-03 17:45 AEST

- Task summary:
  - GTOC12 track, third campaign: cooperative cluster pricing in parallel workers, bundle
    columns in the master, archived routes as columns. Best verified fleet 4398.7 kg (10 ships)
    -> 7575.58 kg (15 ships, 109 asteroids visited / 103 mined, average 505.0 kg per ship,
    fleet rule 15 <= 15.08). Target of >= 600 kg/ship not reached (see bottleneck).
- Changes (commits 1ca6a0c, 568980c, 2bfdf27, 79ab08c, f8fa226 + results/docs commit):
  - `gtoc12/bundles.py` (new): per-family pricing (SCvx-certified Earth legs seeding the beam,
    deployer + collector slots sharing a MinerPool, leg bans and beam retries, orphan repair,
    `make_consistent`), `rank_families`, forked-worker `price_clusters`, `bundle_columns`.
  - `gtoc12/archive.py` (new): `discover_archives` / `group_plans` / `recertify_archives` -
    archived `route_summary.json` files of any run are rebuilt into plans, re-flown through SCvx
    in workers and packed into pool-consistent bundles for the master.
  - `gtoc12/pipeline.py`: `plan_from_route_summary` (legacy archives: deploys at first arrival,
    collects at revisit arrival or camp departure by archived mass, foreign collects snapped to
    the deployer); `RefinedRoute.summary` embeds the plan.
  - `gtoc12/cooperative.py`: ship-rule bound (`ship_rule_bound`), two greedy starts + warm
    start from the previous selection, `MinerPool.register_all`.
  - `gtoc12/retiming.py`: `retime_order` fails softly on invalid visit orders.
  - `gtoc12/cli.py`: `cluster-fleet` (checkpoints, verified intermediate fleets, budget marks,
    `--families`, `--node-cap`) and `fleet-master` (archives -> re-certified columns -> master
    -> verified fleet + viewer export).
  - Tests: `tests/test_gtoc12_bundles.py` (bundle columns, master rule/bound/brute-force/warm
    start, Earth-leg certification, injected first level, pricing, orphan repair,
    make_consistent, retimer guard), `tests/test_gtoc12_archive.py` (reconstruction, discovery,
    re-certification). 383 passed, 4 skipped, 1 deselected (pre-existing native packaging).
  - `docs/GTOC12_TRACK.md`: section 6.5, results rows, score-vs-budget, per-ship table,
    cooperative statistics, master convergence, artifact policy, limitations, bottleneck.
- Validation:
  - `fleet_master_v1/fleet/Result.txt`: official `GTOC12_Verify` "Check successfully!" 15 ships
    7575.58 kg; independent verifier agrees (fixed-bonus 7217.51 kg, max position error 56 km).
    Intermediate verified fleets of `cluster_fleet_v1` retained under `fleets/` (30 min 4692.8,
    1 h 5382.7, 2 h 6357.9, 4 h 6932.5 kg); `cluster_fleet_v2_deep` 2878.0 kg,
    `cluster_fleet_v3_repair` 4704.15 kg, all officially verified.
  - 208 archived routes re-certified with 0 failures (deterministic SCvx replay).
  - Memory: campaign main 0.43 GB peak, worker peaks <= 0.60 GB, sampled concurrent total
    1.2-1.5 GB (sum-of-peaks bound 2.8 GB); fleet-master 0.24 GB main.
- Follow-up notes / risks:
  - The rule binds exactly; per-ship mass (Earth-leg cost 371-618 kg, hop cost 75-150 kg) is
    the remaining lever - see docs section 8.
  - Cooperative columns were never selected by the master (collectors 280-330 kg vs
    self-cleaning 450-540 kg in the same family).
  - Master result at 5 M nodes is not a proven optimum for 273 columns.

## 2026-09-03 18:30 AEST

- Scope: GTOC12 per-ship mass levers, part 1 (hop pricing model, continuous Earth-leg optimiser).
- Changes:
  - `gtoc12/screening.py`: `low_thrust_inflation` (ratio model 1.05 + 0.65 r, fitted on 1674
    certified hops; validated: under-pricing of fast hops 18% -> 7% of legs).
  - `gtoc12/retiming.py`: ratio-dependent hop inflation in the DP, forward pass and plan masses;
    `calibrate(..., authority_ratio)` stores residuals; `calibrate_from_route`;
    `Retimer.protect_earth_leg` (Earth-out TOF floor).
  - `gtoc12/search.py`: beam prices hops with the same model (`hop_inflation_for`).
  - `gtoc12/earthleg.py` (new): `EarthLegModel`/`EarthLegBounds`, Lambert `compass_search` and
    `optimise_earth_leg` (coarse start finder), `refine_leg_scvx` (SCvx-in-the-loop compass with
    line search, official launch constraints via bounds, deterministic).
  - `gtoc12/bundles.py`: `certify_earth_legs(continuous=...)` refines every certified grid leg
    (8 SCvx calls), logs grid-vs-refined propellant; launch grid widened to 3 years.
  - Tests: continuous refinement bounds/determinism/no-worse; grid test pinned to `continuous=False`.
- Validation: six archived Earth legs 393-602 kg -> 292-410 kg (mean -104 kg, 8 SCvx / ~20 s
  each); `tests/test_gtoc12_*.py` 73 passed, 1 skipped; Ruff clean.
- Next: phasing-aware families + leg-stats command, collector harvest loop, campaign v4.

## 2026-09-03 21:20 AEST

- Scope: GTOC12 per-ship mass levers, part 2 (phasing-aware families, beam/tour fixes, joint
  collect harvest).
- Changes:
  - `gtoc12/clusters.py`: `ClusterBands.visit_epochs` - phase embedded at the deploy *and* the
    collect epoch (phasing-aware families); `relative_drift_deg_per_year`, `phase_difference_deg`.
  - `gtoc12/legstats.py` (new) + `gtoc12 leg-stats` CLI: per-role leg cost distributions of any
    solution file (ours and the references) through the shared itinerary decoder.
  - `gtoc12/search.py`: beam back to the flat hop inflation (the ratio model stays in the DP);
    `SearchResult.first_level`; collection tour modes `greedy | reverse | forward |
    forward_revisit` (`_complete` keeps the best; `orders_of` epoch-matches collects so a
    revisit tour is decoded correctly).
  - `gtoc12/bundles.py`: window 0 with continuous Earth legs; richer 'no closing chain'
    diagnostics; `_joint_harvest` after the slots (adopt certified joint tours, resolve clashes,
    revert when the bundle collects less); `ClusterBundle.harvest` report.
  - `gtoc12/harvest.py` (new): `joint_collect_orders` (deterministic multi-ship nearest
    neighbour over the pooled miners at collect epochs), `retime_harvest` (DP re-timing with
    foreign deploy epochs, tail dropping), `harvest_report`.
  - `gtoc12/cooperative.py`: `FleetMasterResult.cooperative_columns` in the master summary.
  - Tests: surrogate optimiser constraints/determinism, tour modes (structure, bookkeeping,
    determinism), joint harvest (pool shared once, foreign epochs, DP closure), phasing-aware
    families, inflation model, harvest bookkeeping, master picks cooperative pair.
- Evidence (family 0, 99 members, real SCvx): self-cleaning collect hops re-fly the deploy pairs
  three years later at 2-3x the deploy dV (5441->57635 1.29 vs 3.25 km/s, 23907->16356 2.29 vs
  4.95 km/s); direction does not matter (forward tours: 270/318/365 kg). Pooled nearest-neighbour
  chains at the collect epochs stay at 1.3-2.0 km/s (median 1.6-1.9 vs 2.5-2.7 own-set).
  Proxy replay with the joint harvest: 1258 -> 1502 kg for three ships.
- Next: real-SCvx probe of the joint harvest, then campaign cluster_fleet_v4 + fleet_master_v2.

## 2026-09-03 23:20 AEST

- Scope: GTOC12 per-ship mass levers, part 3 (real-SCvx probe of the joint harvest, re-timing
  consistency fixes, campaign v4 launch).
- Probe `probe_v4_family` (family 0, 3 ships, 19 min, real SCvx): 1014.6 kg / 3 ships (338 kg
  average) - *worse* than the self-cleaning bundle. Root cause was not the harvest idea but the
  bookkeeping around it: ship 1's re-timed variant speculated on 7 orphans (294.9 kg itself),
  ships 2 and 3 collected four of them as foreign (488.7 / 474.3 kg), then the orphan repair
  dropped the three nobody took, which re-timed ship 1's deploy epochs and stranded both
  collectors (`reverted_stranded` -> 328 / 332.6 kg). The joint harvest itself certified two of
  three new tours but was reverted for the same reason ("asteroid 49218 collected against a
  stale deploy epoch").
- Changes (commit eb4a5be):
  - `gtoc12/retiming.py`: `Visit.pinned_arrival` / `build_visits(..., pinned)` /
    `retime_order(..., pinned)`: a deploy another ship collects keeps its exact lattice epoch
    through any re-timing (`_Lattice.exact_index`; off-lattice pins are infeasible, not rounded).
    `_tofs` snaps the TOF grid onto the lattice (the 400 d Earth bound on a 15/30 d lattice made
    the DP price/authority-check a leg at 730 d and fly it at 720 d: forward refused what the DP
    accepted and the mass rounds never converged). `_forward` returns the refused leg's mass so
    the profile correction hits the right entry; corrections carry across price rounds.
  - `gtoc12/bundles.py`: `drop_asteroid(..., pinned)`; orphan repair pins the deploys other
    ships collect and requires fallback variants to reproduce them (epoch, not just membership);
    `_joint_harvest` pins every deploy collected elsewhere (new assignment or kept route).
  - `gtoc12/harvest.py`: `retime_harvest(..., pinned)`.
  - Tests: `test_pinned_deploys_keep_their_epoch_through_drop_and_retiming` (pins reproduced,
    off-lattice pin infeasible, TOF grids on the lattice); orphan-repair monkeypatch updated.
- Validation: full suite 392 passed, 4 skipped (`test_native_library_is_packaged_and_abi_compatible`
  deselected: this worktree has no built native library - environment, not code); Ruff clean.
  Before the fixes a fixture plan re-timed by its own re-timer failed `leg_authority`; after them
  it closes in 2 mass rounds, pinned or not.
- Leg-cost distributions (`gtoc12 leg-stats`, results/gtoc12/leg_stats/before_v4.json), best
  verified fleet (15 ships) vs JPL36 / Antipodes39 / Antipodes37, propellant per leg:
  Earth-out median 484 vs 447 / 461 / 466 kg (TOF 540 vs 523-532 d); deploy hop 100 vs 101 / 96 /
  97 kg (TOF 240 vs 174-183 d); collect hop 110 vs 67 / 66 / 66 kg (TOF 330 vs 181-187 d);
  Earth return 192 vs 206-214 kg; hops <= 75 kg: 21% vs 44-46%; hop propellant per ship equal
  (1448 vs 1448-1464 kg) - the references buy 9-10 asteroids with it, we buy 6.
- Campaign `cluster_fleet_v4` launched 21:25 AEST: 4 workers, 4 ships/family, radius 2.0
  (phasing-aware), continuous Earth legs, `--collector-harvest`, 4 h declared budget
  (`timeout 15300`), log /tmp/cluster_fleet_v4.log.

## 2026-09-04 02:30 AEST

- Scope: GTOC12 per-ship mass levers, part 4 (campaign v4 results, fleet_master_v2, docs,
  commits). Task closed.
- While v4 ran: `MinerPool.register_all` made two-phase (all deploys, then all collects) after
  the campaign's harvest of family 459 certified both new tours and was rejected as "collected
  but never deployed" - the joint harvest produces *mutual* pairs (A collects B's miner and B
  collects A's) that no slot order can register. Collect look-ahead in the beam
  (`SearchSettings.collect_lookahead_weight`, `--collect-lookahead`) built and measured on
  family 247 with proxy pricing: 934 kg (W = 0.5) vs 1023 kg off -> stays off, documented as a
  rejected variant. `HarvestSettings.return_reserve_days` 960 -> 465 after the family-247 replay
  (joint tours ended ~700 d early; 791.8 kg with the fix, still < 884.6 self-cleaning).
  Commit ba93060 (author fell back to the local identity - env vars were not set in that shell;
  the later commits use the track identity).
- `cluster_fleet_v4` finished at 252.5 min (families in flight at the 4 h mark run to
  completion; `timeout 15300` was not hit): 47 families, 119 certified ships, 1322 Earth legs
  flown / 533 certified / 324 continuously re-optimised (477 -> 390 kg median, -102 kg mean), 67
  cooperative collects, 22 orphans, 190 repairs, 226 rejected variants, 38 joint harvests
  attempted / 0 adopted (8: no tour certified, 19: less mass, 11: pool inconsistent - part of
  those are the mutual pairs). Verified intermediate fleets: 30 min 884.6 kg (2 ships; 5 ships
  at 34.1 min), 1 h 4841.7 (11), 2 h 6167.1 (13), 4 h 6926.9 (14); final 6975.69 kg / 14 ships /
  100 deployed, 96 mined, average 498.3 kg. Main 0.46 GB, worker peak 0.81 GB. Leg stats of the
  v4 fleet: Earth-out 404 kg/ship (references 460-474) - lever 1 won; deploy hops 129 kg mean
  (111 before), collect hops 102 kg median (110 before, references 66) - levers 2/3 not won; the
  Earth saving was spent on deploy hops and the average stayed at 498 kg.
- `fleet_master_v2` over all six archives (330 routes re-flown through SCvx in 749 s, 0
  failures; 436 columns; 5 M nodes in 103 s, not exhaustive): **16 ships, 123 asteroids (116
  mined), 8324.27 kg**, official `GTOC12_Verify` "Check successfully!" + independent verifier
  (56 km max propagation error), fixed-bonus 7905.05 kg, average 520.3 kg, rule 16 <= 16.03.
  Sources: v1 8 ships, v4 5 (564/538/524/499/495 kg), fleet10 2, v3 1. One cooperative pair in
  the incumbent (family 0: ship 15 collects 27306 + 30267 deployed by ship 10; 2 foreign
  collects, 0 bundle columns). Best-fleet leg stats: Earth-out 468, deploy hop 110, collect hop
  90 kg median (was 110), 0.23 of hops <= 75 kg (references 0.44-0.46).
- Validation: Ruff clean (check + format), suite 392 passed / 4 skipped / 1 deselected (native
  library not built in this worktree).
- Docs: `docs/GTOC12_TRACK.md` 6.6 (after-campaign leg table, harvest outcome), 7 (v4 and
  fleet_master_v2 rows, budget marks, per-ship table, regenerate commands), 8 (fourth-campaign
  limitation, next bottleneck = the collect hop: price the collect tour exactly in the beam,
  tighter collect-epoch families, exhaustive master).
- Commits: ba93060 (code), 2aabeef (results: v4 reports/bundles/route summaries/intermediate
  fleet.json, fleet_master_v2 report + Result.txt 6.9 MB + fleet.json + viewer manifest, probe,
  leg_stats before/after), docs + memory commit follows.
- Next bottleneck: collect hops (90-102 vs 66 kg, 170-250 kg per ship = the whole 500 -> 740 kg
  gap). The Lambert look-ahead does not predict the DP's cheap collect tours; the beam needs the
  collect tour priced exactly (DP on deploy+collect per surviving partial, or a certified
  collect-pair table at the collect epoch per family) and families re-clustered at radius <= 1.0
  weighted on the collect-epoch phase.

## 2026-09-04 03:20 AEST

- Fifth iteration started (branch `feat/gtoc12-asteroid-mining`, from b4c6b01): attack the
  collect hop. New module `src/spacepdhcg/gtoc12/collectdp.py`: `CollectPairTable` (lazy,
  bounded, float32 Lambert ΔV per ordered pair on a 30-day mission lattice x 10 collect TOFs,
  Earth returns per asteroid) and `plan_collect_tour` (Held-Karp over collected subset x
  location x lattice epoch: free collect order, camp may be left uncollected and revisited,
  camps anywhere, collection on departure with rate x stay and the one-year minimum, objective
  collected - w x propellant; propellant priced as mass x fraction table at the heaviest mass).
  `SearchSettings.collect_dp*`; `RouteSearch._complete` adds the DP tours (w = 1.0 and the
  beam's 0.15) to the heuristic ones and ranks all by `plan_score` (weighted - 0.15 x
  propellant); `_finish` is the shared exact forward mass pass. `ClusterBands.phase_weights` and
  `ClusterBands.collect_window()`; `cluster-fleet --collect-epoch-families --no-collect-dp
  --collect-dp-weight`. Master: `ship_rule_mass_floor`, `_LpModel`, `lp_fleet_bound` (per-N LP
  relaxation with foreign-closure rows), `LpBound.node_bound` (dual bound, kept but weak),
  `lp_branch_and_bound` (fractional-column branching for the sizes whose LP beats the
  incumbent), `FleetMasterResult.lp_bound/lp_relaxations/lp_nodes/proven`. Exporter title now
  derived from the instance (`dataset_title`) and `--run-id`/instance passed through.
- Probe (archived fleet_master_v2 tours, DP vs archived collect tours at equal mass): DP tours
  cut the proxy collect+return propellant on 3 of 6 ships (1398 -> 919, 1411 -> 1302, 1030 ->
  1004 kg) with 10-60 kg less proxy mass; two tours reordered, one used the camp revisit.
  Master probe (312 archived single routes): LP bound 7997.5 kg (0.3 s, 16 LPs), combinatorial
  DFS 7905.0 at 2 M nodes, LP branch and bound 7987.3 kg proven optimal in 39 LPs.
- Tests: `tests/test_gtoc12_collectdp.py` (14: mass-floor rule, LP bound validity vs brute force
  on random column sets, LP B&B exactness, tiny node cap rescued, bundle ships in the LP, DP
  bookkeeping / min stay / single collect / free order, camp revisit, no-tour, off-lattice TOFs,
  pair table determinism + bounded cache + return window, beam completion with DP on/off,
  collect-window families). Family probe `probe_v5_family247` (4 slots) running.

## 2026-09-04 09:10 AEST

- Fifth campaign closed. Probe of family 247 with the DP (4 slots; the first 2-slot probe
  "failed" only because the 12-check Earth-leg limit rejects the same 24 short legs v4 rejected
  and v4 certified its ships in slots 3-4): 884.6 -> 986.0 kg. Campaigns (both 4 h, 4 ships per
  family, radius 2.0, collector harvest, run concurrently on the 16 cores): `cluster_fleet_v5`
  (4 workers, deploy-epoch phasing families + DP) 38 families / 128 ships / 16 verified
  intermediate fleets, marks 30 min 2551.9 kg (5 ships), 1 h 7695.3 (15), 2 h 8365.9 (16), 4 h
  9101.9 (17), final 9101.85 kg / 17 ships / 137 asteroids, DP priced 20 984 times (8391 s),
  won 10 392, failed to close 9825, main 0.45 GB / worker 0.83 GB; `cluster_fleet_v5c` (3
  workers, collect-epoch families) 27 families / 94 ships / 10 fleets, marks 1 h 5569.1 (11),
  2 h 8358.0 (16), 4 h 9111.3 (17), final 9100.89 kg / 17 ships / 140 asteroids, DP 14 673 /
  8305 / 5536. Campaign masters proven optimal by the LP B&B (v5 81 LPs, gap 28 kg; v5c 9 LPs,
  gap 3 kg). Harvest: v5 38 attempts, 2 adopted (459, 280), 27 less mass, 9 no tour; v5c 27, 0.
- `fleet_master_v3` over nine archives: 554 routes re-flown through SCvx in 921 s (8 workers, 0
  failures), 726 columns, DFS 5 M nodes then LP B&B over sizes 18 and 17 (565 LPs, 136 s):
  **18 ships, 147 asteroids (142 mined), 9888.57 kg**, official "Check successfully!",
  independent verifier ok (113 km max propagation error, 9.5e-11 kg mass), fixed-bonus 9329.82
  kg vs LP bound 9334.32 (gap 4.5 kg, proven optimal over the archive; LP infeasible at 19),
  average 549.36 kg, rule 18 <= 18.004. Sources: v5c 8, v5 5, probe_v5 1, v4 2, v1 1, fleet10 1.
  No cooperative column selected.
- Leg stats (`results/gtoc12/leg_stats/after_v5.json`), best fleet before -> after: collect hop
  mean 95.3 -> 89.7, median 90.2 -> 87.1, p75 122.7 -> 115.3, p90 152.5 -> 140.0 kg, share
  <= 75 kg 0.233 -> 0.292 (references 66 kg / 0.44-0.46, TOF 181-187 d vs our 240 d); deploy
  hop median 109.6 -> 98.9 (references 96-101); Earth-out 468 -> 412 (references 460-474);
  Earth return 197 -> 227 (references 208-216). Per-ship collect propellant 691 -> 707 kg
  because ships carry 8.1 instead of 7.7 asteroids.
- Viewer: `export-viewer` regenerated for `fleet_master_v3` (title now "GTOC12 full-catalogue
  OrbitWeaver solution (18 ships, 9888.6 kg)", 8.0 MB); the v2 candidate's importer run with
  Windows node over UNC paths (no Linux node in WSL): 18 ships, 147 asteroids, 9140 replay
  samples, hashes and Kepler cross-check ok; data stays ignored.
- Validation: Ruff clean; suite 406 passed / 4 skipped / 1 failed (`test_native_packaging`,
  native library not built in this worktree - pre-existing). Test
  `test_master_warm_start_never_regresses_when_columns_are_added` updated: a node cap of 0 is
  now rescued by the LP B&B, the cold-start case passes `lp_bound=False`.
- Commits: b1678aa (collectdp, search, clusters, cooperative LP, viewer title, tests), 81d73e4
  (v5/v5c/probe reports, bundles, route summaries, intermediate fleet.json, fleet_master_v3
  report + Result.txt 7.6 MB + fleet.json + viewer manifest, leg stats), docs + memory follow.
- Next bottleneck: collect-hop phase at the harvest epoch (87 vs 66 kg, 240 vs 181 d): tighter
  collect-window families (radius <= 1.0, more ships per family), certified pair costs in the DP
  table, finer Earth-return grid, cheaper Earth-leg pre-screen.

## 2026-09-04 10:15 AEST

- Sixth GTOC12 iteration started (harvest-epoch phase of the collect hop). Code committed as
  b91b334: `hopcalib.py` (certified-hop calibration fit + `gtoc12 hop-calibration`), collect DP
  with fitted per-pair/epoch inflation, 15-day lattice, 30-day 240-720 d return grid, per-move
  mass with a two-pass burn schedule, Earth-leg prescreen (ratio > 0.7 deferred), process-tree
  PSS sampler. Fit residuals (holdout 2925 hops): rms 0.093 (ratio-only 0.111, flat 0.123).
- Probe `probe_v6_family` (radius 1.75 collect-window family of 51, 5 slots): 2720.2 kg / 5
  ships (582.8, 598.4, 484.9, 558.5, 495.7), 41.5 min.
- Campaign `cluster_fleet_v6` launched 10:09 AEST: 4 workers, 5 ships per family, radius 1.75,
  >= 20 members, collector harvest, calibrated DP, 2400 s per family, 4 h budget; GPU busy and
  locked by the G4 campaign (CPU only).

## 2026-09-04 15:40 AEST

- `cluster_fleet_v6` finished (255 min, 4 workers, CPU only): 36 families, 136 certified ships
  (median 478.6 kg; 603.7 / 603.3 / 598.4 / 589.3 / 582.8 / 578.6 best; 9 above 563 kg), verified
  fleet **10 698.0 kg / 19 ships / 159 asteroids (155 mined) / 563.05 kg avg**; budget marks
  60 min 8545.1 kg (16 ships), 120 min 9887.8 (18), 240 min 10 697.1 (19); 19th ship admitted at
  170 min. Memory: main 0.43 GB, worker peak 1.11 GB RSS, process-tree PSS peak 3.04 GB (target
  2 GB missed; transient is native and sits in the collector slots - not localised).
- `fleet_master_v4` over eleven archives: 695 routes re-flown through SCvx in 1020 s (8 workers,
  0 failures), 903 columns, DFS 2 M nodes then LP B&B (635 LPs, 1.0 s): **19 ships, 158
  asteroids (154 mined), 10 700.48 kg**, GTOC12_Verify "Check successfully!", independent ok
  (121 km max propagation error, 9.5e-11 kg), fixed-bonus 10 146.60 vs LP bound 10 159.66 (gap
  13.1 kg, proven optimal over the archive; LP infeasible at 20 ships), rule 19 <= 19.027.
  Sources: v6 10, probe_v6 1, v5 3, v5c 2, v4 2, fleet10/coop 1. No cooperative column.
- Leg stats (`after_v6.json`), best fleet before -> after: collect hop median 87.1 -> 84.4 kg,
  p75 115.3 -> 104.1, p90 140.0 -> 133.9, share <= 75 kg 0.292 -> 0.342 (references 66 kg /
  0.44-0.46), TOF median unchanged 240 d (references 181-187); deploy hop median 98.9 -> 91.3
  (references 96-101); Earth-out 411.8 -> 414.6; Earth return 226.5 -> 279.3 (references
  208-216). Per-ship hop propellant 1492 -> 1431 kg.
- Calibration residuals (holdout 2925 hops): rms 0.093 vs 0.111 ratio-only vs 0.123 flat;
  propellant error median -0.9 kg, p10 -11.5, p90 +5.0, rms 12.2 kg.
- Viewer: `export-viewer` for `fleet_master_v4` (8.5 MB, title "... (19 ships, 10700.5 kg)");
  v2 importer via Windows node: 19 ships, 158 asteroids, 9643 replay samples, hashes ok, Kepler
  cross-check 3.57e-6 km over 59 356 points; data stays ignored.
- Commits: b91b334 (code), 61469d4 (memory interim), 237f5b0 (v6 / probe_v6 / fleet_master_v4
  artifacts, calibration fit, leg stats), docs + memory follow.
- Next bottleneck: Earth return (+65 kg vs references, cheapest ~50 kg per ship -> the 20th
  ship), then the collect-hop phase at the harvest window (84 vs 66 kg, 240 vs 181 d) via
  harvest-window pair costs in the deploy beam; worker memory transient to localise.


## 2026-09-04 17:20 AEST

- Memory transient fixed and localised: the collect DP's unbounded per-mass propellant
  fraction cache (one 20 KB table per Held-Karp expansion) was the +350 MB beam transient; a
  512-entry LRU keeps results bit-identical and the single-slot peak is 329 MB (was 695).
  glibc retention (RSS 670 -> 167 MB on trim) handled by `bound_heap_growth` + `release_heap`
  at every phase mark. `MemoryBudget` 3 x 450 + 250 MB declared; regression test on the
  priced-bundle fixture. Commit 7d87a36.
- Earth-return sweep campaign `return_sweep_v1` (36 best stand-alone archived ships): 13
  improved, +140.5 kg, returns 172-247 kg where improved; v6 family 1 ship 2 +34.9 kg.
- Harvest-window ranking is a negative result on family 54 at every weight (see scratchpad).
- `fleet_master_v5` running over the eleven archives + return_sweep_v1 (3 workers).


## 2026-09-04 17:20 AEST

- Memory transient fixed and localised: the collect DP's unbounded per-mass propellant
  fraction cache (one 20 KB table per Held-Karp expansion) was the +350 MB beam transient; a
  512-entry LRU keeps results bit-identical and the single-slot peak is 329 MB (was 695).
  glibc retention (RSS 670 -> 167 MB on trim) handled by `bound_heap_growth` + `release_heap`
  at every phase mark. `MemoryBudget` 3 x 450 + 250 MB declared; regression test on the
  priced-bundle fixture. Commit 7d87a36.
- Earth-return sweep campaign `return_sweep_v1` (36 best stand-alone archived ships): 13
  improved, +140.5 kg, returns 172-247 kg where improved; v6 family 1 ship 2 +34.9 kg.
- Harvest-window ranking is a negative result on family 54 at every weight (see scratchpad).
- `fleet_master_v5` running over the eleven archives + return_sweep_v1 (3 workers).


## 2026-09-05 00:40 AEST

- cluster_fleet_v7 done (249 min, 3 workers): 25 families, 76 ships, own fleet 9920.47 kg / 18
  ships; marks 60/120/240 min = 5355.8 / 8571.1 / 9922.5 kg; PSS peak 1.19 GB, worker 0.68 GB.
- return_sweep_v2 on v7's 21 best: 13 improved, +225.5 kg.
- fleet_master_v6 (fourteen archives, 779 routes, 1032 columns): 20 ships / 168 asteroids /
  11 515.67 kg / 575.78 avg, proven optimal (gap 5.6), both verifiers ok; Earth return 216.5 kg
  mean. First attempt died on RecursionError in the column DFS (1019 > 1000 frames) after the
  45-min re-certification -> fixed + test (ba9b764), re-run.
- Slot cache release before orphan repair (c495dc0); test updates (930db57); docs + artifacts
  (0c8a533, ded890f). Viewer v2 data re-imported from fleet_master_v6.
- Suite: 421 passed / 4 skipped (+ pre-existing native-packaging failure); Ruff clean.
- Next: collect-hop phase inside the DP (member substitution priced from the pair table); give
  the DP the camp's sweep cells; run retime-returns after every campaign before the master.


## 2026-09-05 04:10 AEST

- Change of direction: reference-methods work (subsets / chain BIP / TOF heuristic, memo)
  dropped and reverted; branch renamed `feat/gtoc12-joint-itinerary`. New lever built:
  whole-itinerary joint re-optimisation (`jointopt.py`, `jointcampaign.py`, `gtoc12
  joint-itinerary`, `tests/test_gtoc12_jointopt.py`) - every epoch of a ship continuous, exact
  mining bookkeeping, calibrated pair-cost surrogate + memoised SCvx-measured legs, pattern
  search on a 45/20/8/3/1 d mesh, full-route SCvx re-certification, monotone certified
  acceptance, one-asteroid insertion. Commit f81e834.
- joint_itinerary_v2 (32 ships, 3 workers, 7 min, 0.12 GB): 32/32 improved, +280.4 kg; the 20
  fleet_master_v6 ships 575.78 -> 586.20 kg average (+208.4; +1.1..+23.3 per ship), 101 SCvx
  certifications / 95 accepted. Redistribution per ship: deploy hops +59 kg (earlier deploys),
  return -12, collect -4, Earth-out 0, margin -> 0. Insertion: 0 of 32 (authority ratio).
- fleet_master_v7 (sixteen archives, 837 routes, 1078 columns, 57 min): **21 ships / 177
  asteroids / 12 346.48 kg / 587.93 avg (rule 21 <= 21.007), proven optimal (gap 5.6 kg),
  both verifiers ok; +830.8 kg over v6.** All 21 columns are joint_itinerary_v2 routes.
- Viewer v2 data re-imported from fleet_master_v7 (21 ships, hashes + Kepler check ok);
  the viewer's check.mjs now trips its 20-colour palette assertion (viewer follow-up).
- Docs: GTOC12_TRACK.md section 6.10, results rows, eighth-iteration narrative + two per-ship
  tables, section 8 entry, next bottleneck (22nd ship needs 599.5 kg average).
- Next: run joint-itinerary after every campaign before the master; new asteroid sets from
  the DP (member substitution, sweep cells) for the 22nd ship - re-timing is saturated.
