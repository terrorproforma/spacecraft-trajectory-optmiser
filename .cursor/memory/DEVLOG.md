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

## 2026-09-01 21:50 AEST

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
