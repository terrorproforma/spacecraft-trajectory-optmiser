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
