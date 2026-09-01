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
