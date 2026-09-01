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

### 2026-09-01 18:00 AEST - Displaced-reference root-cause correction

#### Task Summary

- Added fixed-pattern device updates for reference tracking, exact-penalty epigraph costs,
  trust-cone centres/radii, low-thrust radial halfspaces, and 6-DoF quaternion linearisations.
- Added CPU/device coefficient parity plus trust-radius and penalty-mutation checks.
- Corrected outer merit scaling and stale-primal evaluation after an identical-CQP re-solve.

#### Mistakes And Fixes

- `[self]` Initially reused the outer virtual merit weight as the transcription L1 coefficient.
  Separated the fixed transcription weight because these are distinct CPU contracts.
- `[self]` The first metric correction used raw fuel, state, and virtual magnitudes. Matched the
  CPU driver's thrust, trust-scale, terminal, mass, and virtual normalisations instead.
- `[prior implementation]` Re-solve telemetry came from the refined solve while acceptance still
  used the pre-refinement primal. Regathered, replayed, and reevaluated the refined primal.

#### What Worked

- All four production fixtures pass coefficient-by-coefficient CPU/device checks at `5e-12`;
  trust radius and exact-penalty mutations occur in place with no topology mutation.
- Rejected re-solves retain identical numerical fingerprints, while changing trust radii produce
  distinct fingerprints as expected.

#### Follow-Ups / Risks

- The original G4 diagnosis was incomplete: its one-outer-iteration qualification never reached
  a second reference update. After the update fix, P1-C still fails because PDHCG returns a poor
  CQP candidate and eventually exhausts the trust region.
- G3's nominal trajectory parity was materially weaker than displaced-start outer-loop parity;
  the G3 report was amended and the affected displaced-start coverage is no longer claimed.
- G4 remains FAIL and G5 remains unauthorised pending an accurate QOCO-GPU adapter or another
  production solver correction that passes frozen matched-quality criteria.
- The first nominally complete corrective archive was built with the final initial-boundary fix
  still uncommitted. Preserved it as contaminated evidence, committed the fix as `2cbebb3`, and
  reran the frozen three-sample qualification from a clean tree. All three samples remained
  unqualified, so no H5/H6 matrix was started.

### 2026-09-01 18:45 AEST - Audited G4 harness contracts

#### Task Summary

- Implemented locked policy generation, matched-quality qualification, full coverage, H5/H6
  decisions, G4 schema semantics, timing identities, and portable-evidence verification in the
  isolated `exp/g4-harness-parallel` worktree.

#### Mistakes And Fixes

- `[tool]` The WSL system Python lacked pytest/Ruff/CMake. Used ephemeral `uv run --no-project`
  tools and the isolated CPU build directory; no canonical environment was modified.
- `[tool]` PowerShell could not pass its provider-qualified UNC working path to .NET file APIs.
  Used byte-safe Python normalization and then matched each tracked file's committed line-ending
  convention before regenerating the policy hash/header.
- `[self]` The first G4 schema fixture retained legacy residuals above the tight tier. The stricter
  semantic validator exposed it; corrected the fixture instead of weakening the tier.

#### What Worked

- The generated C++ header and SHA-256 lock make JSON/C++ policy drift fatal.
- Adversarial tests cover max-iteration false positives, objective/continuous-time/path omissions,
  policy drift, ledger gaps, timing sums, decision states, and artifact tampering.
- Full CPU validation passed: Ruff, 127 pytest tests, and 41 native CTest smokes.

#### Guardrails For Next Session

- Do not classify current CUDA G4 output as qualified until it reports requested/actual runtime
  policy and consumes the generated configuration.
- Keep QOCO/hybrid conversion telemetry compatible with the primary-G4 schema's permutation,
  dual-disposition, and conversion/setup/polish timing fields.

#### Follow-Ups / Risks

- CUDA executable integration remains for the CUDA worker; this branch intentionally contains no
  CUDA numeric-update edits and made no GPU performance claim.

### 2026-09-01 19:55 AEST - G4 workstream integration

#### What Worked

- Exact QOCO and harness commits cherry-picked; the only conflict was additive scratchpad history.
- Pinned QOCO CUDA/cuDSS passed exact trajectory QP/SOCP, update, warm-primal, residual, and
  explicit dual-discard tests on RTX 5090.
- CUDA now consumes the generated policy hash and frozen numeric policy rather than shadow constants.

#### Mistakes And Fixes

- `[tool]` Initial QOCO execution omitted the package's `nvidia/cu12/lib` runtime directory.
  Added the pinned cuDSS directory and shim to `LD_LIBRARY_PATH`; all 18 tests then passed.
- `[self]` A 15-outer displaced diagnostic was too expensive for interactive qualification and
  exceeded 30 minutes. Terminated it, retained the timeout, and did not count it as matrix evidence.

#### Follow-Ups / Risks

- P1-C still fails the frozen forcing test after an identical-CQP re-solve, so matched quality
  remains the authoritative blocker. Do not execute or claim the full matrix.
- The Python QOCO adapter has a valid primal-only handoff, but the nonlinear G4 family owners still
  need an end-to-end handback before H6 can be decided.


### 2026-09-01 22:12 AEST - Gate G5 distributed core preparation

#### Task Summary

- Built the pinned upstream MPI/NCCL target and implemented isolated G5 ownership, partition,
  algebra, telemetry, checkpoint, lifecycle, build, CI, schema, and logical/one-rank tests.

#### Mistakes And Fixes

- `[tool]` Linux-style paths passed to the file patch tool landed under `C:\home`; moved the three
  files into the WSL worktree, deleted the accidental copies, and used UNC paths for all later edits.
- `[self]` The first distributed G2 composition solved on a stream different from workspace creation,
  producing pointer-contract status 4. Copied each exchange and bound creation to the persistent
  rank compute stream.
- `[self]` Requested residuals while the local solve was still in flight, producing busy status 5.
  Enforced solve-wait-residual-wait ordering in one-rank coverage.
- `[tool]` Compute Sanitizer initcheck accepts `--track-unused-memory` without `yes`; the malformed
  invocation was discarded and not counted as validation.

#### User Preferences Learned Or Reinforced

- `[user]` G5 implementation/build work is explicitly authorised on this isolated branch while G4
  continues, but no G5 PASS or 2/4/8-GPU scaling claim is authorised.
- `[user]` Never install a Linux NVIDIA driver, use fake ranks/MPS, or run GPU work while another
  compute process owns the RTX 5090.

#### What Worked

- Deterministic logical 1/2/4/8-rank tests exposed ownership, risk, checkpoint, failure, and schema
  contracts without pretending logical ranks are physical GPUs.
- GPU-process checks found short idle windows for a clean Release one-rank CTest plus memcheck and
  racecheck while preserving G4 priority.

#### Guardrails For Next Session

- Use UNC paths for WSL file edits and keep Bash scripts LF-only.
- Bind every rank-local G2 workspace at creation to the persistent rank compute stream.
- Wait for solve completion before requesting independent residuals.
- Require same-machine 1/2/4/8 physical evidence before any scaling efficiency or G5 acceptance.

#### Follow-Ups / Risks

- Initcheck, synccheck, actual overlap-stream execution, and every physical 2/4/8-GPU test remain
  deferred; ordinary OpenMPI rank loss is fatal rather than ULFM-recoverable.
- Later G4/P1 fixes must be cherry-picked and the full G5 build/one-rank matrix rerun.
