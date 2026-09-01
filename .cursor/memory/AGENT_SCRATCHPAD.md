# Agent Scratchpad

Use this file as persistent, repo-local execution memory.

## File Policy

- Current policy: `COMMITTED`
- Rationale: GPU gate execution lessons and reproducibility guardrails must be shared across workers.

## How To Use

1. Read latest entries before starting meaningful work.
2. Build a short preflight checklist from recurring mistakes and preferences.
3. Re-read before risky operations and broad refactors.
4. Log high-signal learnings immediately during the task.
5. Append one new session entry before handoff.

## Retained Lessons

- `[user]` Work only in `/home/angus/spacecraft-trajectory-optmiser`; preserve existing uncommitted recovery work.
- `[user]` Never weaken frozen tolerances or benchmark definitions, discard negative evidence, use CPU fallback, amend/reset/force, push, or open a PR.
- `[user]` Gate G4 remains unauthorised until every G3 criterion passes.
- `[tool]` Canonical CUDA work runs through WSL Ubuntu-22.04 with CUDA 12.8 and an RTX 5090.
- `[self]` Treat missing production drivers and harnesses as implementation scope, not blockers.
- `[self]` Preserve machine-readable failures and censored benchmark points; report matched end-to-end nonlinear quality.
- `[self]` Never use backslash-sensitive Perl through PowerShell/WSL for line endings; it changed `return`/`pattern` tokens. Use PowerShell `ReadAllText().Replace("`r`n", "`n")`, then rebuild.

## Session Entries

### 2026-09-01 11:42 AEST - Complete G3

#### Task Summary

- Delivered device-only recovery, four production CUDA SCvx drivers, parity, H1, qualification,
  sealed evidence, and the passing G3 report.

#### Mistakes And Fixes

- `[self]` A backslash-sensitive line-ending command changed `return` and `pattern` tokens.
  The warnings-as-errors build detected it; repaired tokens, recorded the PowerShell-only
  normalization rule, and rebuilt all targets.
- `[tool]` Initcheck found recovery/scaling allocations unused on bounded paths. Zero-initialized
  every create-time scratch allocation and reran all four sanitizer tools.
- `[self]` The first summary parser treated racecheck wording and “0 tests failed” incorrectly.
  Preserved the original archive, fixed parser-specific positive markers, and resealed all raw
  evidence with correction provenance.

#### What Worked

- Projected KKT correction after device CGLS reduced 6-DoF stationarity below `1e-6` while
  transactional rollback kept rejected recovery bit-identical to the pre-recovery PDHG iterate.
- A schema-validated six-size H1 sweep with seven repeats and seeded bootstrap retained every point.

#### Guardrails For Next Session

- Re-read the frozen roadmap sections 9-10 and existing native contracts before broad CUDA edits.
- Verify recovery and outer-loop lifecycle under all four Compute Sanitizer tools.
- Treat each sanitizer tool's clean summary text separately; racecheck does not use `ERROR SUMMARY`.

#### Follow-Ups / Risks

- The G3 implementation is committed and G4 is authorised.
- Nsight under WSL previously omitted kernel and GPU-memory records; retain this negative result.

### 2026-09-01 17:30 AEST - Gate G4 qualification

#### Task Summary

- Froze G4 policy, pinned and executed QOCO-GPU, added per-outer forcing/fingerprint/work telemetry,
  and ran nontrivial P1-C/P1-D/P1-E qualification starts.

#### Mistakes And Fixes

- `[self]` Launched tests concurrently with the first power campaign. Preserved that contaminated
  run as negative evidence, excluded it, and reran the campaign in isolation.
- `[tool]` PowerShell expanded Bash loop variables and the login shell omitted CUDA tools from
  `PATH`. Replaced loops with explicit commands and absolute tool paths.
- `[self]` Tightened the legacy G3 production fixture to the G4 floor, causing one regression.
  Restored the G3 `1e-6` fixture while retaining the frozen G4 `1e-8` policy.

#### What Worked

- QOCO-GPU commit `09f0495` built for `sm_120` on CUDA 12.8 with cuDSS 0.7.1.6 and executed a real
  CUDA SOCP smoke solve.
- Device-side numeric fingerprints were identical across rejected under-solved re-solves, and all
  four Compute Sanitizer tools remained clean.

#### Follow-Ups / Risks

- G4 is FAIL/unresolved and G5 is not authorised.
- Nontrivial references expose that the device outer driver does not update every
  reference/trust/penalty coefficient. Fix this before restarting primary H5.
- The pinned QOCO API has primal-only warm start; H6 needs an audited dual-capable adapter or an
  explicit primal-only hybrid hypothesis revision before a future preregistration.
- WSL power sampling showed 1.86-1.94 second gaps despite a 50 ms request; energy is diagnostic.
