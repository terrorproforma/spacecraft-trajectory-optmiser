# Archived Scratchpad Summary

- Active window: 2026-09-01 22:18 AEST to 2026-09-04 23:20 AEST (rolled over 2026-09-05 during the Lambda H100 migration).
- Workstreams covered: benchmark/comparative-campaign design (Paper 1/2 matrices, GTOPX boundary,
  GTOC replay track); the `integration/single-gpu-v1` current-head G0-G3 reseal on the RTX 5090
  (`results/gpu/current-head-b0cd570`, b6afb49); the G4 H5/H6 claim-core campaign chain
  26def2b -> a08f5e2 -> 857f99a -> ccd5596 -> 46bc895 -> 4db5047 (amendments v1.1/v1.2, contamination
  control, executor defects, replay-validation defect, pause at 13:15Z on 2026-09-04); the GTOC12
  asteroid-mining track (campaigns v1-v7, fleet_master_v5 20 ships 11521 kg); the v2 candidate
  consolidation and the WebGL GTOC12 fleet viewer.
- Most important lessons preserved here: PowerShell -> WSL quoting (always script files, LF endings);
  never edit a worktree with a running seal; executor deadline/cancel paths must be exercised with a
  short deadline before a long campaign; capability regeneration after every commit; pause campaigns at
  the journal boundary; verify executor fixes on a failing coordinate; QOCO defaults must be read, not
  assumed; CRLF from the Write tool into WSL paths; `AwaitShell` without an id does not sleep.
- Provenance note: the original live file was deleted by a faulty rollover command on 2026-09-05
  (`[IO.File]::ReadAllText` with a relative path resolved against the process cwd, then `Remove-Item`
  ran anyway). The body below was reconstructed the same minute from a complete verbatim read of the
  file taken earlier in that session; wording is faithful, line wrapping may differ slightly.
- Superseded by the slim `AGENT_SCRATCHPAD.md` created at rollover.

# Agent Scratchpad

Use this file as persistent, repo-local execution memory.

## File Policy

- Current policy: `COMMITTED`
- Rationale: benchmark and research-workflow lessons should be shared across sessions.

## Retained Lessons

- [user] Use the learning-scratchpad and devlog loops for every meaningful task.
- [user] When implementation is requested, make the change and explain it; do not stop at a proposal.

## Session Entries

### 2026-09-01 22:18 AEST - Spacecraft trajectory benchmark research

#### Task Summary

- Investigated standard spacecraft trajectory-optimization benchmarks and assessed their fit for SpacePDHCG.

#### Mistakes And Fixes

- [tool] The repository had no `.cursor/memory` files.
- Fix: initialized the committed scratchpad and devlog.
- Preventive rule: read both memory files at the start of each meaningful task.

#### User Preferences

- [user] Research questions should lead to concrete project recommendations.

#### What Worked

- Cross-checked the project benchmark ladder against ESA GTOPX/TOPS, canonical powered-descent papers, OpenSCvx, SCPToolbox, and published GPU Monte Carlo work.

#### What Did Not Work

- TOPS public documentation exposes named cases but does not make every case's physical description easy to inspect without loading its JSON data.

#### Guardrails For Next Session

- Distinguish global mission-design benchmarks from sparse optimal-control/transcription benchmarks before selecting comparisons.
- Compare end-to-end accepted trajectories at matched independently verified nonlinear quality.

#### Follow-Ups / Risks

- Pin exact literature parameter sets, source revisions, objective conventions, and reference outputs before adding benchmark fixtures.

### 2026-09-01 22:24 AEST - Comparative evaluation objective

#### Task Summary

- Refined the benchmark objective from selecting recognizable trajectories to proving whether SpacePDHCG is worthwhile against competing approaches.

#### Mistakes And Fixes

- [self] The initial research answer emphasized trajectory selection more than the required comparison architecture.
- Detection: the user clarified that accuracy, speed, and wider performance evidence are the primary goal.
- Fix: separate component-level solver comparisons from complete end-to-end software comparisons.
- Preventive rule: benchmark both the claimed technical contribution and the user's actual software-level decision.

#### User Preferences

- [user] The decisive outcome is objective evidence of usefulness relative to other approaches, including accuracy, speed, and other material performance measures.

#### What Worked

- The existing preregistered quality gates, crossover hypotheses, censored failures, and confidence-interval rules already support a defensible comparative study.

#### What Did Not Work

- The current `paper1_matrix.json` primarily names conic solver backends and does not yet define a complete-system comparison against OpenSCvx, SCPToolbox/SCvxGEN, or a direct-NLP implementation.

#### Guardrails For Next Session

- Never infer usefulness from kernel or inner-solver timing alone.
- Treat independently replayed nonlinear feasibility and objective quality as prerequisites for timing comparisons.

#### Follow-Ups / Risks

- Add a distinct system-level baseline matrix before collecting final GPU results.

### 2026-09-01 22:33 AEST - Comparative campaign and GTOC replay

#### Task Summary

- Wrote and encoded the comparative solver campaign, literature evidence policy, GTOPX boundary, and historical GTOC replay track.

#### Mistakes And Fixes

- [self] The premise that published GTOPX timing could replace local competitor runs was not technically valid for the SpacePDHCG claim.
- Detection: GTOPX is a low-dimensional black-box global-search suite, while SpacePDHCG targets repeated sparse local CQP solves; published hardware and quality gates also differ.
- Fix: use published definitions/objectives as references, require common-hardware reruns for speed, and keep GTOPX as a secondary global-search track.
- Preventive rule: classify external results by problem class and admissible comparison field before adding them to a winner map.
- [tool] The active PowerShell version rejected `&&`, and the active Python environment did not contain Ruff.
- Preventive rule: run independent PowerShell commands separately and do not claim lint validation when Ruff is unavailable.

#### User Preferences

- [user] Include recognized external mission challenges, especially historical GTOC editions, to demonstrate real software utility.

#### What Worked

- Added machine-readable source provenance and explicit reduced/full GTOC5, GTOC9, and GTOC12 progression with official validation.
- Targeted manifest tests passed after both Paper 1 and Paper 2 expansions.

#### What Did Not Work

- GTOC leaderboard results cannot establish speed because they include weeks of team compute, human intervention, and often unpublished tooling.

#### Guardrails For Next Session

- Validate official example solutions before defining reduced GTOC cases.
- Freeze reduced-case selection from metadata before observing SpacePDHCG or OrbitWeaver scores.
- Track offline transfer-database generation and human intervention separately from online solve time.

#### Follow-Ups / Risks

- External repository revisions and archive checksums remain intentionally unpinned until the artifacts are downloaded.
- GTOC11 and GTOC13 remain deferred until their additional scheduling, mixed-propulsion, ballistic-flyby, and solar-sail models exist.

### 2026-09-02 23:30 AEST - Current-head G0-G3 reseal slowness investigation (WSL)

#### Task Summary

- Investigated why the `integration/single-gpu-v1` current-head reseal (evidence root `results/gpu/current-head-b0cd570/`, WSL worktree `/home/angus/worktrees/spacepdhcg-single-gpu-integration`) was taking hours, without touching in-flight evidence.

#### Mistakes And Fixes

- [tool] `wsl.exe ... bash -lc '...|...'` from PowerShell splits on `|`; scripts written with the Write tool carry CRLF and break bash (`$'\r': command not found`).
- Fix: write the probe script to `%TEMP%`, then run `wsl.exe -d Ubuntu-22.04 -u angus -- bash -c "tr -d '\r' < /mnt/c/.../x.sh > /tmp/x.sh; bash /tmp/x.sh > /mnt/c/.../out.txt 2>&1"` and read the output file.
- [self] The WSL worktree's `.cursor/memory/*` files are tracked, and the in-flight `g3/run.sh` asserts `git status --porcelain` is empty at the end and records `git rev-parse HEAD` in `manifest.txt`. Editing or committing anything in that worktree during the run would fail or corrupt the seal.
- Fix: keep all notes in this Windows checkout; never edit, add, or commit in a worktree that has a running seal script.

#### What Worked

- Per-step wall-clock reconstruction from log mtimes (each `run_log` step writes one `<name>.log`, so consecutive mtimes give exact durations) identified the dominant sink without instrumenting anything.
- `pstree -p -a <run.sh pid>` plus `nvidia-smi --query-gpu` confirmed the active step owned the GPU load (no hang).

#### Guardrails For Next Session

- Distinguish "slow by design" (declared sanitizer cases, honest-negative timeouts) from hangs using log growth + owning-PID CPU + GPU utilisation before killing anything.
- Any change to a sealed test binary or evidence script forces a full preflight+G0-G3 reseal (~105 min here); batch such fixes and dry-run the changed G3 step against an existing build before committing.
- Reading a running bash script's file while bash executes it is fine; writing to it is not.

#### Follow-Ups / Risks

- `compute-sanitizer --tool racecheck recovery_test --sanitizer` costs ~54 min per G3 pass (59-60% of G3; measured 54m09s and 54m33s) because `check_cancellation_and_destruction` runs a 350,000-iteration, 1e-20-tolerance solve that only terminates on the iteration limit under sanitizer instrumentation.
- Proposal committed on side branch `proposal/g3-sanitizer-recovery-cap` (9fafee8, worktree `/home/angus/worktrees/spacepdhcg-reseal-proposal`): sanitizer-mode budget 20,000, native budget unchanged; compile-verified only (GPU was owned by the seal). Adopt only when another reseal cycle is needed anyway.
- `nsys stats` without `--force-export=true` reuses a stale `.sqlite` left by an earlier attempt in the same evidence dir; the reseal worker had to re-export (`corrected_nsys_export=true`). Add `--force-export=true` to future G3 scripts.
- A foreign GPU-idle watcher (`watch_gpu_idle_6h.sh`, Reality-Simulator overseer) signals a relaunch once the GPU is idle 8 min; the GPU went idle at 00:53 after G3 PASS, so unrelated GPU work may start before G4 unless the worker claims the device first.
- Outcome: all gates PASS on b6afb49 (G0 23:07, G1 23:10, G2 23:20, G3 00:53:10 AEST 3 Sep); `seals/` still empty, G4 not launched.

### 2026-09-03 01:05 AEST - G4 H5/H6 claim-core campaign launch (WSL)

#### Task Summary

- Verified the sealed head, added GPU contamination control to the campaign scheduler, generated the official capability, piloted, and launched the 360-group claim core from `/home/angus/worktrees/spacepdhcg-single-gpu-integration`.

#### Mistakes And Fixes

- [tool] PowerShell strips inner double quotes from `wsl -e bash -lc '...'` arguments and splits unquoted `|`; `python -c "..."` and `grep -E "a|b"` silently break.
- Fix: write every non-trivial command as a script under `%TEMP%\g4\`, run `wsl -e bash -lc 'tr -d \\r < /mnt/c/.../x.sh > /tmp/x.sh && bash /tmp/x.sh'`.
- [tool] The Write tool sometimes writes CRLF to `\\wsl.localhost\...` paths (one of two files got CRLF). Always `grep -c $'\r'` and `sed -i 's/\r$//'` before ruff/pytest.
- [tool] `AwaitShell` without a shell id returned after ~15 s of wall time while claiming 180-240 s; use `sleep N` inside the WSL command for real waits.
- [self] First contamination rule treated every `/dev/dxg` holder as foreign compute; the pilot showed sibling jobs with `CUDA_VISIBLE_DEVICES=` (empty) opening `/dev/dxg` without any GPU work.
- Fix: classify holders by their `/proc/<pid>/environ` CUDA visibility and record them separately (`wsl_cuda_disabled_holders`).
- [env] WSL2 `nvidia-smi --query-compute-apps` is always empty; Windows `nvidia-smi.exe pmon -c 1` (reachable from WSL via interop) shows host compute-only (`C`) contexts with SM/mem %.

#### What Worked

- Verifying the G3 release binary by `cmake --build` in the sealed build tree (`ninja: no work to do`) plus hash equality with the G3 manifest and `git diff --stat b6afb49 HEAD -- cpp src ...` empty.
- Capability regeneration is cheap (~10 s incl. real CUDA probe); do it after every commit and before `init`.
- `CampaignStore` pins `source_commit`; never commit on the campaign branch/worktree while a checkpoint is live unless you intend to `init` a new checkpoint and `migrate` terminal rows.

- [self] The pilot's first group never emitted an attempt in 91 min: `spacepdhcg_cuda_workspace_wait` held the workspace mutex across `cudaEventSynchronize`, so the deadline watchdog's `workspace_cancel` blocked behind it and the kernel ran its full 1e6-iteration budget.
- Detection: `/proc/<pid>/task/*/wchan` showed the main thread spinning (`0`), one thread in `futex_do_wait`, and a 30 s-deadline `--g4-session` repro emitted nothing for minutes; no gdb was available, so thread wchan + code reading carried the diagnosis.
- Fix: release the mutex during the blocking wait; map a cancelled inner solve to `SCVX_CANCELLED` (launched `timeout`, not `numerical`); a cancelled attempt forces a cold next boundary (warm reset on a cancelled workspace failed instantly as `numerical`).
- Preventive rule: any deadline/cancel path must be exercised end-to-end with a short deadline (`SPACEPDHCG_G4_ATTEMPT_DEADLINE_SECONDS=20`) before a long campaign; the capability probe's 1-outer-iteration session never reaches the deadline.
- [tool] `pkill -f` on the parent python left the CUDA executor orphaned for 55 min, hogging the GPU and blocking the real worker behind the contamination guard; kill the executor PID explicitly and re-check `/dev/dxg` holders.
- [self] `device_scvx_integration_test` links `libspacepdhcg_cuda.so` dynamically, so the executable hash did not change when the solver library changed; the capability now pins runtime libraries via `ldd`.

#### Guardrails For Next Session

- Never call `device_scvx_integration_test --help`: unknown flags run the default GPU integration test.
- The GPU is shared with Reality-Simulator/weldsim agents, the planner worker, and Windows-side jobs; expect `waiting_for_foreign_gpu_processes` events and contaminated re-runs.
- Fixed-tight P1-E N=100 runs ≈1 ms per PDHCG iteration, so attempts that do not converge reach the frozen 600 s deadline and a group costs ≈91 min; the claim core is a multi-day campaign at that rate. Report measured throughput, never predict timeouts.
- The claim-core checkpoint pins `source_commit=26def2b`; restart, status and finish only from the detached worktree `/home/angus/worktrees/spacepdhcg-g4-claim-core-26def2b` (scripts in `build-integration-report/g4-claim-core-26def2b-{worker,status,observer,finish}.sh`).
- A deadline that fires while the workspace is not solving is invisible to the device: every cancel path needs a driver-side check before each inner solve and a watchdog that keeps re-asserting; a single-shot watchdog is not enough.
- With all attempts at the frozen 600 s deadline, a group needs ≈5405 s; the executor group deadline (5460 s) is authoritative and the outer boundary must sit strictly beyond it (now +300 s) or the executor's explicit unlaunched records are lost to an outer kill.
- `AwaitShell` without a shell id does not sleep; poll with `sleep` inside WSL commands (≤299 s per `sleep`, chained).

### 2026-09-03 03:40 AEST - GTOC12 replay track built in WSL (pointer entry)

- Full session notes live in the tracked `.cursor/memory/*` of branch `feat/gtoc12-asteroid-mining`
  (WSL worktree `/home/angus/worktrees/spacepdhcg-gtoc12`, HEAD `accc5df`).
- [tool] Write tool -> `\\wsl.localhost` paths produces CRLF; normalise in WSL before tests/commits.
  `setsid nohup` background jobs from `wsl.exe` are unreliable; run long jobs in the foreground
  with `timeout`.
- [model] Emit zero-order-hold thrust arcs for GTOC12: cubic-interpolated bang-bang profiles make the
  official RKF78 verifier disagree by thousands of km; ZOH arcs agree to < 1 km.
- [rule] Each asteroid may carry at most two rendezvous events; camping never emits an event.

### 2026-09-03 06:20 AEST - Single-GPU v2 candidate consolidation (pointer entry)

- Full session notes live in the tracked `.cursor/memory/*` of `integration/single-gpu-v2-candidate`
  (WSL worktree `/home/angus/worktrees/spacepdhcg-single-gpu-v2`, HEAD `2c8c651`, base `63271d5`).
- [self] Quoted branch HEADs go stale while campaigns run (`9678134` in the brief vs live `63271d5`);
  base candidates on the live tip so promotion stays `--ff-only`, and say so in the report.
- [self] `python -S -m pytest` needs the venv `site-packages` on `PYTHONPATH` (the sealed G0 recipe
  does this); `PYTHONPATH=src` alone makes pytest/build "missing".
- [self] The cuDSS/CUDA `libqoco.so` aborts the whole pytest process (rc=1, no summary) under
  `CUDA_VISIBLE_DEVICES=''`; CPU gates must use a builtin-algebra build of pinned QOCO 09f0495.
- [tool] A PowerShell session whose cwd is a `\\wsl.localhost` UNC path returns no output afterwards;
  always pass `working_directory` or `Set-Location C:\...` first. Windows node cannot serve the viewer
  from WSL paths; the viewer's `npm test` is Windows-specific, `check.mjs` runs on Linux node 20
  (`/home/angus/.local/node/bin`).
- [rule] Never `ctest` a CUDA tree while the G4 campaign owns the device (even the
  `spacepdhcg_plan_capabilities` test); configure+build only.

### 2026-09-03 11:40 AEST - G4 claim core relaunched under amendment single-gpu-v1.1

#### Task Summary

- Paused the 26def2b worker (0 groups complete), implemented run-and-flag contamination
  (Decision A) and the preregistered amendment `single-gpu-v1.1` (Decision B: deterministic-replay
  timeouts, 120 s / 200k censoring, 10 % `censoring_sensitivity` twins at 600 s / 1M with the
  acceptance rule), relaunched as `g4-claim-core-a08f5e2` from the detached pin worktree.

#### Mistakes And Fixes

- [self] `SPACEPDHCG_SOURCE_COMMIT` is baked at CMake *configure* time; a rebuilt-only tree
  emitted `identity.repository_commit=b6afb49` under a campaign pinned at the live head, which the
  decision step rejects. Fix: executor reports `compiled_source_commit`; capability generator and
  scheduler refuse a mismatch. Preventive rule: after every commit on the campaign branch run
  configure -> build -> capability before initialising or restarting a checkpoint.
- [self] `decide_h6` stored `math.inf` for missing residuals and the decision file is written
  with `allow_nan=False`; the finish step would have crashed on the first failed/contaminated
  coordinate. Fix: null + explicit `is not None` gates. Preventive rule: run `--preview` on a
  one-group throwaway checkpoint before launching a multi-day campaign.
- [tool] Passing a multi-line `python3 -c "..."` through `wsl --% -e bash -lc` fails in PowerShell
  parsing; write a `.py` helper, strip CR, run it.

#### Guardrails For Next Session

- Campaign pin is now `a08f5e2` (`/home/angus/worktrees/spacepdhcg-g4-claim-core-a08f5e2`);
  scripts `build-integration-report/g4-claim-core-a08f5e2-{worker,status,observer,finish}.sh`.
  The old 26def2b checkpoint/worktree/scripts are retained read-only.
- Groups run pure-gpu-ipm -> adaptive -> hybrid -> fixed-tight; each 600 s twin runs right after
  its core group. Reordering was permitted because the claim core never bound execution order to
  the `solver_order` rotation (recorded as an identity axis only).
- A foreign Windows `python.exe` (PID 40636, ~96 % SM) has been on the GPU throughout; expect
  every measured attempt to be flagged `contaminated` until it stops. Contaminated attempts keep
  disposition/quality but never enter timing/energy statistics; rows report the n actually used.
- Pure-gpu-ipm P1-E N=100 fails `numerical` in ~0.5 s with zero QOCO workspace creations (executor
  defect candidate, pre-existing; reproduced without amendment env vars).

### 2026-09-03 14:40 AEST - pure-gpu-ipm defect triage (three root causes, two relaunches)

#### Task Summary

- Triaged the "every pure-gpu-ipm attempt numerical in 0.5 s, zero QOCO workspaces" symptom on
  `g4-claim-core-a08f5e2`. Environment/library wiring was fine. Root causes: (1) pure-QOCO warm
  boundary asked the PDHCG workspace for a FULL_RETAINED warm start it never had (INVALID_STATE
  recorded as `numerical`); (2) failure branch dropped QOCO counters; genuine QOCO `numerical
  error` at conditioning 4.0; (3) QOCO carries `kkt_dynamic_reg` escalation and its best-iterate
  tracker across solves on one solver, so re-solves after a failure were 1-iteration re-failures.
- Fixes `857f99a` (warm boundary, `executor_defect` disposition, IPM capability probe + library
  pin, `invalidate` ledger action, GPU regression test) and `ccd5596` (settings restore + fresh
  solver after a failed solve). Campaign chain a08f5e2 (26 groups invalidated) -> 857f99a (9
  invalidated) -> ccd5596.

#### Mistakes And Fixes

- [self] Trusted the first-fix probe (converging coordinate) as proof of independence; the failing
  coordinate exposed root cause 3 only in the live campaign. Preventive rule: verify an executor
  fix on a *failing* coordinate too (verbose QOCO log, per-solve data hash), not only on the
  capability probe coordinate.
- [self] The capability generator and `invalidate` need a clean tree at the pinned commit: run
  them from the detached pin worktree, never from the integration worktree with uncommitted docs.
- [tool] `pkill -f <pattern>` inside `wsl bash -lc "..."` matches its own command line and kills
  the shell (exit 15). Use a script file or `pgrep` first.
- [tool] `set -o pipefail` + `bash script | grep | cut` hides the failing script's stderr; keep
  the `tee` of raw output when a stage can fail silently.

#### Guardrails For Next Session

- Pin is `ccd5596` (`/home/angus/worktrees/spacepdhcg-g4-claim-core-ccd5596`); scripts
  `build-integration-report/g4-claim-core-ccd5596-{worker,status,observer,finish}.sh`. Old
  checkpoints a08f5e2/857f99a are invalidated evidence, never resumed (`claim()` skips rows).
- `qoco_workspace_creations >= 1` is the invariant (not `== 1`): the adapter rebuilds the QOCO
  solver after any failed solve.
- Open H6 caveats (need a preregistered amendment, not a silent change): IPM baseline runs QOCO
  without Ruiz equilibration (5/10 iterations NaN'd in diagnosis); a single QOCO solve is not
  interruptible, so IPM attempts overrunning the 120 s deadline are recorded by QOCO's outcome
  (`numerical`), not `timeout`; the failing IPM is non-deterministic run-to-run on cuDSS
  (13-199 iterations on identical data), so deterministic replay never triggers for it.
- Failing IPM groups now cost ~9 x (100-200 s) at N=100 under the foreign load; the executor's
  group deadline (1140 s) records the tail as unlaunched.

### 2026-09-03 17:10 AEST - Amendment single-gpu-v1.2 (IPM equilibration, wall-clock timeouts)

#### Task Summary

- Preregistered and applied amendment `single-gpu-v1.2` (frozen 2026-09-03T06:45:00Z, SHA-256
  `673e0670...73e1b9`) on `integration/single-gpu-v1`: rule A IPM baseline equilibration recorded
  per attempt (QOCO native default = `ruiz_iters 0` at the pinned commit; `scaling_mode:
  not_applicable_ipm_native` for pure-gpu-ipm), rule B wall past deadline is `timeout` for every
  backend, rule C N=2000 hard bound unchanged. Commits `aca6500` (executor/adapter) and `46bc895`
  (amendment, contracts, scheduler, decision, docs, tests). Campaign chain ccd5596 (4 IPM groups
  -> diagnostic stratum `ipm_no_equilibration_v1_1`) -> `g4-claim-core-46bc895`, capability
  `827ce9e8...51bce3`.

#### Mistakes And Fixes

- [self] Assumed with the user that QOCO's default is "Ruiz on"; at the pinned commit
  `set_default_settings` has `ruiz_iters = 0`. Preventive rule: read the vendored solver's default
  settings before writing a fairness rule that names them.
- [self] The pause script TERMed the worker the instant the last `--g4-session` exited, before the
  worker wrote the group result: ccd5596 ordinal 4 (a 600 s sensitivity twin) is `running`/lost.
  Preventive rule: pause by waiting for the *journal* `completed` event (or a `group_finished`
  line), not for the executor process to disappear.
- [self] Rebuilding the executor while a worker is alive would swap the binary under the next group
  launch; pause first, build second (done this time, worth remembering).
- [tool] Files written from the Windows side into the WSL worktree arrive with CRLF; `sed -i
  's/\r$//'` before running or committing them. PowerShell also eats `$var` and heredocs inside
  `wsl --% -e bash -lc "..."`: put multi-line Python/bash into files.
- [tool] `pyt2.sh` (`.venv-current-head`) lacks matplotlib and the native wheel, so the full suite
  shows 12 environmental failures there; run the full suite with `fullpy.sh` (tool venv) and read
  the focused G4 suites from `pyt2.sh`.

#### What Worked

- Separating "Ruiz is broken in this QOCO CUDA build" (two defects: `safe_div(1,0)=DBL_MAX` on
  empty G/A rows, `scale_arrayf` without host fallback) from "Ruiz does not help this problem
  class" with a scratch-patched library, then selecting the shipped default with the evidence in
  the amendment JSON.
- Generating the v1.2 JSON from the frozen v1.1 JSON (inherited sections copied byte-for-byte) so
  the amended schedule hash is provably unchanged.
- Schema reuse via `if/then/else` on `amendment_id`: one amendment schema validates both documents.

#### Guardrails For Next Session

- Pin is `46bc895` (`/home/angus/worktrees/spacepdhcg-g4-claim-core-46bc895`); scripts
  `build-integration-report/g4-claim-core-46bc895-{worker,status,observer,finish}.sh`; worker PID
  1911827; capability `/home/angus/g4-executor-capability-46bc895.json`.
- `label-stratum` moves rows to state `diagnostic` (disposition = stratum name); `migrate` refuses
  cross-amendment sources; the decision refuses a v1.2 checkpoint without the
  `ipm_no_equilibration_v1_1` citation in `diagnostic_strata` metadata.
- Under rule B, failing IPM attempts at conditioning 4.0 still take 50-310 s each (uninterruptible);
  they are now recorded `timeout` (wall > 120 s) with the solver's `numerical` attached. First five
  amended groups: 25 timeout / 10 numerical / 10 unrun / 0 qualified, 100 % contaminated, ~20.6 min
  per group; projection ~2.5-4.5 days total from 2026-09-03 07:05Z.
- Helper scripts for this campaign: `/home/angus/newattempts.py <campaign>` (per-attempt echo view),
  `/home/angus/report5.sh` (group summary), `stage7.sh` (the full hygiene/relaunch sequence used).

### 2026-09-04 01:40 AEST - GTOC12 fleet visualisation in the WebGL viewer (WSL + Windows, CPU only)

#### Task Summary

- Regenerated and verified the fleet_master_v1 viewer export, extended `web/trajectory-viewer` with
  a dataset selector and a Sun-centred GTOC12 fleet view (Keplerian Earth/asteroid orbits, per-ship
  arcs, deploy/collect markers, mission timeline, picking), captured browser screenshots and a
  matplotlib fallback, committed on `integration/single-gpu-v2-candidate`.

#### Mistakes And Fixes

- [self] UA `[hidden]` lost to author `display: grid` on `.controls-panel`/`.dataset-panel`, so
  archive panels stayed visible in fleet mode. Fix: `[hidden] { display: none !important; }`.
  Preventive rule: every toggled panel with a `display:` class rule needs the global override.
- [self] Switching renderers on one WebGL2 context left enabled vertex attribs pointing at deleted
  buffers ("no buffer is bound to enabled attribute" warnings). Fix: `dispose()` disables all attrib
  arrays and unbinds before deleting buffers/programs.
- [self] `focusShip` at 2.3 r with a 45° vertical FOV cropped bounding-sphere extremes off-canvas and
  the hover test hit nothing. Fix: distance 3 r; the test asserts the projected marker is inside the
  canvas before hovering.
- [tool] Playwright reports `requestfailed net::ERR_ABORTED` for a `HEAD` fetch on Chromium although
  `response.ok` is true; probe optional datasets with GET (1.7 KB manifest).
- [tool] Playwright actions have no default timeout, so a failing actionability check hangs
  forever, and an uncaught assertion leaves the browser open (node never exits). Always
  `page.setDefaultTimeout(...)` and close the browser in `finally`.
- [tool] The server CSP `style-src 'self'` forbids inline `style=` attributes but not CSSOM; per-ship
  colours live in `.ship-colour-N` classes (check.mjs asserts parity with `SHIP_COLOURS`), tooltips
  and event labels are positioned through `element.style`.
- [tool] Linux node 20 prints TAP (`ok N - …`, `# pass N`), not the `✔` reporter; grep accordingly.
  `tests/server.test.mjs` used `pathname.slice(1)` (Windows-only) → `fileURLToPath`.

#### What Worked

- Regenerating the export into a fresh ignored directory (`results/gtoc12/viewer-exports/…`) kept
  the GTOC12 worktree's tracked `fleet/viewer/manifest.json` untouched while its campaign ran; the
  only difference from the committed export is the commit string.
- Attaching pinned Keplerian elements at import and cross-checking the JS propagation against the
  exporter's 41k context points (3.5e-6 km) gives exact Earth/asteroid positions at any epoch
  without shipping 6 MB of sampled orbits.
- Event markers are the transcription nodes (exact archived body states), so deploy/collect
  positions need no ephemeris lookup and the legend can honestly say "no interpolation".

#### Guardrails For Next Session

- The v2 candidate is authoritative for the viewer; sync the Windows mirror with `tr -d '\r'` and
  compare with `diff -rq --strip-trailing-cr` before committing anywhere.
- `web/trajectory-viewer/data/gtoc12/` is ignored; regenerate with `npm run import-gtoc12 -- --export …
  --catalogue … [--solution … --fleet …]` (README); `check.mjs` validates it only when present.
- Browser check needs `PLAYWRIGHT_PATH=C:/Users/Angus/AppData/Local/Temp/ptd-browser/node_modules/playwright`
  (Playwright 1.62.1, chromium-1234) and `npm run serve` on 4173; WSL has the browsers but no module.

#### Follow-Ups / Risks

- `cluster_fleet_v4` was still running at commit time (best 6975.69 kg / 14 ships < 7575.58);
  export + import any fleet that beats fleet_master_v1.
- The exporter title constant says "reduced-instance" for full-catalogue fleets; fix on the gtoc12
  branch when it is next touched.

### 2026-09-04 13:05 AEST - G4 claim core: replay-validation defect triage and 4db5047 relaunch (WSL)

#### Task Summary

- Read-mostly triage of `g4-claim-core-46bc895` (5 quarantined, 54 completed at 12:10 AEST): four
  quarantines were a scheduler/contract defect, one was the preregistered rule C hard bound. Fixed
  (`4db5047`), paused at a group boundary, relaunched as `g4-claim-core-4db5047` with the 55
  completed groups migrated and the quarantined rows left behind for re-run.

#### Mistakes And Fixes

- [self] When amendment v1.2 was added (46bc895) the replay branch of `validate_attempt_record`
  kept `policy_amendment == "single-gpu-v1.1"`; `validate_amendment_records` was updated but the
  shared per-record contract was not, so every v1.2 replay group was quarantined and no test built
  a replay record under v1.2. Preventive rule: when an amendment supersedes another, grep every
  literal `AMENDMENT_ID` comparison in the contract, scheduler and decider, and add a test that
  runs a full replay group through `validate_group_success`-level checks under the new amendment.
- [self] The first gate-report tally I wrote (34/31/19) did not match my own table (31/31/19) and
  the "median" of four values was the upper-middle element; re-derive every number from the script
  output before committing prose, and quote small samples as lists rather than medians.
- [self] Assumed `migrate` only carries `completed` rows; it also carries `quarantined` rows, so
  "invalidate + init + migrate" alone would never have re-run the quarantined groups. Read the
  ledger action's SQL before describing the re-run path. Fix: `migrate --skip-quarantined`.
- [tool] `wsl -e bash -lc '... grep "a\|b" ...'` and `git log --format="%h %an"` lose their inner
  quotes/backslashes in PowerShell (silently wrong grep, `%h` printed literally). Every command
  with quotes, `\|`, `%` or `$` goes into a script under `%TEMP%\g4triage\` and runs via
  `tr -d \\r < /mnt/c/... > /tmp/x.sh && bash /tmp/x.sh`. Plain `wsl -e bash -lc 'cd ... && tail'`
  is fine.
- [tool] `AwaitShell` with a `shell_id` and a `pattern` does block correctly (used for the 25 min
  boundary wait); it is only the id-less form that returns early.

#### What Worked

- Re-validating the quarantined groups' retained `stdout.jsonl` through the patched
  `validate_group_success` with the real `coordinate.json` proved the fix on the actual evidence
  (4/4 "all raw attempts and measured Paper 1 results validated") before any relaunch.
- Pause-at-boundary by polling the *journal* terminal event for the running ordinal, then TERM,
  then kill `--g4-session`/`--g4-server`/observer by pattern: ordinal 59 was committed as valid,
  ordinal 60 was claimed but never launched, GPU idle ≈ 70 s end to end.
- Stage script pattern (pin worktree -> reconfigure/rebuild 22 s -> selftest -> capability with
  `--check` -> mkscripts -> init citing the stratum -> migrate -> setsid nohup launch) reused from
  `stage7.sh` as `stage8.sh`; `compiled_source_commit` verified before the capability.

#### Guardrails For Next Session

- Pin is `4db5047` (`/home/angus/worktrees/spacepdhcg-g4-claim-core-4db5047`); scripts
  `build-integration-report/g4-claim-core-4db5047-{worker,status,observer,finish}.sh`; worker PID
  3778145, server 3778162, observer 3778194; capability `/home/angus/g4-executor-capability-4db5047.json`
  (`5c849945...c80b8`). Old checkpoints 46bc895/ccd5596/... are retained read-only.
- Branch `integration/single-gpu-v1` is at `addac2b` (docs) > `d1865a6` (docs) > `4db5047` (fix);
  the worker runs from the detached pin, so docs commits on the branch are safe.
- Deterministic replay is trivially eligible for IPM groups whose attempts never complete an outer
  iteration (trace = zeros + `inner_iterations 200`); it is literal to the amendment and counts as
  censoring only, but say so whenever H6 is read. Changing it needs a preregistered amendment.
- Rule C hard-bound groups (`N=2000` IPM, 460-530 s per uninterruptible QOCO solve) cost 2886 s
  and end as error records with `raw_attempts: []`; the partial records stay in
  `stdout.restart-0.jsonl`/`stdout.jsonl`. Not the "gen-0 silence": records are emitted normally.
- IPM `N=2000` runs at 4-9 % GPU utilisation (cuDSS/host bound); a clean GPU does not shorten it.
- Foreign GPU load returned at 02:51Z (Windows `python.exe` 49548, 95 % SM) after a clean window
  01:36Z-02:51Z; WSL weldsim pytest processes hold `/dev/dxg`. Expect contaminated attempts again.
- No PDHCG-policy group has run in this campaign yet; every ETA for adaptive/hybrid/fixed-tight is
  an assumption (replay path ≈ 7 min per core group, ≈ 31 min per twin) until ordinal 66+ lands.

#### Follow-Ups / Risks

- Re-runs of 56 and 59-class groups will likely hit the hard bound again (48 min each, by design).
- Update `docs/G4_GATE_REPORT.md` once the first adaptive groups calibrate the PDHCG per-group wall.

### 2026-09-04 16:10 AEST - GTOC12 3D scene raised to a GPU-rendered look (v2 candidate 3373988, Windows ff994cf)

#### Task Summary

- Rebuilt the fleet renderer around instanced lit spheres, true tube meshes on the archived samples,
  distance fog, a procedural sky and cursor-anchored wheel dolly; canvas is now 72vh full-bleed and
  opens on the 30-degree oblique preset at 6x (labelled). fleet_master_v4 (19 ships, 158 asteroids,
  10700.48 kg). Screenshots + 10-frame sequence + GIF captured; checks/tests green on both OSes.

#### Mistakes And Fixes

- [tool] `wsl -e bash -lc '... tr -d "\r" ...'` from PowerShell: PowerShell re-quotes the argument so
  bash received `tr -d \"\r\"` and `tr` deleted every `r`, `"` and `\` from the synced files (camera.js
  shrank by 420 bytes and lost `dollyTowards`; md5 "matched" only because the check reused the same
  broken pipe). Fix: write the sync/commit steps to a `.tmp_*.sh` file, `tr -d '\r'` it into /tmp and
  run `bash /tmp/x.sh`. Preventive rule: never pass quoted escape sequences through `wsl -e bash -c`;
  always verify a sync with an independent `grep -c <new symbol>` on the destination.
- [tool] `wsl -e bash -c` (non-login) resolves `npm`/`node` to the Windows binaries (`Cannot find
  module 'C:\Windows\app.js'`), and `bash -lc` has an empty PATH here. Use
  `export PATH=/home/angus/.local/node/bin:/usr/local/bin:/usr/bin:/bin` explicitly (Linux node 20).
- [self] Instanced attributes leave `vertexAttribDivisor = 1` on their locations; a later program
  reusing that location would read one value per instance. `attribute()`/`constantAttribute()`/
  `dispose()` now reset the divisor to 0.
- [self] Float32Array normals fail `1e-9` unit-length assertions; tube tests use `1e-6`.
- [self] After the wheel-dolly test the preset click kept the shifted target (presets keep the
  target by design), so later screenshots were off-centre. The check now clicks Reset view and
  asserts the Sun-centred framing before continuing.
- [self] Hovering the ship marker at its return epoch picks the coincident Earth-return event (event
  bonus 4 > ship 3); the assertion accepts `ship` or `event` for that ship.

#### What Worked

- One instanced draw (`bodyInstances()` -> 9 floats per body, `bufferSubData` each frame) replaced
  ~180 per-body draws, and per-instance emissive/radius made mined/hovered/pulsing states free.
- Tube radius and sphere radii scale with camera distance (`radii()`), so real 3D geometry keeps a
  constant on-screen size like a screen-space line while still being lit and depth-tested.
- Fog density `0.25 / camera.distance` gives ~6% at the target and ~40% at 3x behind it at every zoom.
- `window.viewerDebug.glInfo` (context attributes, depth test, instance count, tube sides) lets the
  browser check assert the rendering contract instead of only pixels.

#### Guardrails For Next Session

- Screenshots for the user live in `web/trajectory-viewer/test-artifacts/gtoc12-3d-*.png` (+ `gtoc12-3d-preview.gif`);
  regenerate with `npm run serve` + `PLAYWRIGHT_PATH=... node scripts/browser-check.cjs` then
  `python scripts/build_gif.py`.
- Ruff lives in `/home/angus/worktrees/spacepdhcg-g6-tooling/.venv/bin/ruff` (v2 has no venv).
- Dev server on http://127.0.0.1:4173/ is left running (Windows); open `?dataset=gtoc12`.

#### Follow-Ups / Risks

- Whole-fleet end-of-mission view is inherently dense (158 orbit ribbons + 19 tubes); mid-mission,
  edge-on and follow views are the legible ones. A "hide orbits" toggle would help if asked.
- Tube frames are built in unexaggerated space; at 20x the cross-section is slightly skewed (thin
  tubes, not visible in practice).
### 2026-09-04 23:20 AEST - G4 claim core 4db5047 paused (RTX 5090 freed, resumable)

#### Task Summary

- Cleanly paused `g4-claim-core-4db5047` at 13:15:04Z: worker 3778145 -> session executor 241196 ->
  server 3778162 -> observer 3778194, all exited on TERM within 8 s. 72 completed / 1 quarantined
  (65) / 324 remaining preserved; ordinal 73 (first adaptive group, censoring_sensitivity twin)
  marked `interrupted`, coordinate left `running` so `claim()` re-runs it first. Manifest:
  `g4-claim-core-4db5047/pause-2026-09-04T131504Z-manifest.json` (+ `-worker-environ.txt`).

#### Mistakes And Fixes

- [self] The "72 completed rows readable" check compared `result.json.attempt_id` with the ledger
  attempt id and flagged 55 rows; those are the 46bc895 imports (`import-*` attempt ids carry the
  source campaign's attempt id in `result.json`). Preventive rule: for migrated rows compare
  `group_id` + `disposition` + `run_directory` name, not `attempt_id`.
- [self] Waiting for the boundary was not an option: ordinal 73's generation 1 had 44 min left and
  would have ended `invalid_evidence` anyway (gen 0 hit the 5760 s boundary after 5 x ~19.4 min
  attempts at the 1M inner-iteration cap). Check `stdout.restart-N.jsonl` + executor etime against
  `9*deadline+60+300` before deciding to wait.
- [tool] `WslRun` PowerShell function does not persist across Shell calls; dot-source
  `%TEMP%\g4pause\WslRun.ps1` every call. `[IO.File]::WriteAllText` with LF avoids the CRLF/`tr`
  problem entirely (plain `bash /mnt/c/...sh`).

#### What Worked

- Kill order worker -> session -> server -> observer with PID/cmdline verification first; the
  python worker has no SIGTERM handler so the DB row stays `running` and no partial record is
  committed; `flock -n <lock> true` proves the shared GPU lock and `gpu-worker.lock` are free.
- Marking the attempt `interrupted` with the scheduler's own vocabulary (state/disposition/reason)
  and appending an `interrupted` journal event keeps `decide` (`state IN ('quarantined',
  'interrupted', 'invalidated')`) and `claim()` (`AND state='running'` no-op) consistent.

#### Guardrails For Next Session

- Restart (append, never `>` on worker.log): `cd .../build-integration-report && setsid nohup
  ./g4-claim-core-4db5047-worker.sh >> g4-claim-core-4db5047/worker.log 2>&1 < /dev/null & disown;
  setsid nohup ./g4-claim-core-4db5047-observer.sh > /dev/null 2>&1 < /dev/null & disown`.
- `/home/angus/.spacepdhcg-gpu.lock` removed (payload saved in the manifest); the worker recreates it.
- PDHCG attempts ignore the wall deadline until the inner-iteration cap: 1M cap = ~19.4 min per
  attempt (adaptive, P1-E N=100 tight). A plain re-run of 73 will be quarantined again; claim_core
  stratum (200k cap, 1440 s boundary) is projected to overrun too. Triage before resuming.
- GTOC12 `fleet_master_v6` (PIDs 261346/261347/261379-261381, CPU) and a stale ccd5596 observer
  (1669905) were left running on purpose.

#### Follow-Ups / Risks

- Scratchpad is ~570 lines: roll over (archive-first) at the next quiet point.
