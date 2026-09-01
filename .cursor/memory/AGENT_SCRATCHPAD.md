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

- Finish recovery, production CUDA SCvx drivers, parity, H1, qualification, evidence, and gate report.

#### Guardrails For Next Session

- Re-read the frozen roadmap sections 9-10 and existing native contracts before broad CUDA edits.
- Verify recovery and outer-loop lifecycle under all four Compute Sanitizer tools.

#### Follow-Ups / Risks

- Existing projected KKT/CGLS recovery is uncommitted and requires review before extension.
- Nsight under WSL previously omitted kernel and GPU-memory records; retain this negative result.
