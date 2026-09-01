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
