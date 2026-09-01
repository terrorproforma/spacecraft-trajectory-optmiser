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
