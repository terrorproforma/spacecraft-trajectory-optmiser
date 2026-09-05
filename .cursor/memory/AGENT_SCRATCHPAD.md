# Agent Scratchpad

Use this file as persistent, repo-local execution memory. Detailed history: see
`AGENT_SCRATCHPAD_2026-09-01_to_2026-09-05.md` (rolled over 2026-09-05) and the Windows checkout's
`AGENT_SCRATCHPAD_2026-09-01_to_2026-09-04.md`.

## File Policy

- Current policy: `COMMITTED`
- Rationale: campaign execution lessons and reproducibility guardrails are shared across workers.

## How To Use

1. Read this file before meaningful work; build a preflight checklist from the guardrails.
2. Re-read before risky operations (campaign launches, merges, wide refactors).
3. Log high-signal learnings immediately with `[self]` / `[user]` / `[tool]` tags.
4. Append one session entry before handoff.

## Durable User Preferences

- [user] Use the learning-scratchpad and devlog loops for every meaningful task; implement, do not
  just propose. Explain the code that was written.
- [user] GTOC12 rules of engagement: CPU-first, 3 workers, `nice 19`, process-tree PSS < 2 GB, the
  RTX 5090 carries a foreign workload and is never used; every emitted solution passes
  GTOC12_Verify and the independent verifier; merge commits only, no push/amend/force/reset, no
  git config edits; export GIT_AUTHOR_*/GIT_COMMITTER_* in every shell.
- [user] Report measured numbers only (paired A/B on the same partition, before/after splits,
  LP gap, whether the next ship is reached), and name the next bottleneck.

## Regression-Prevention Guardrails

- PowerShell -> WSL: every non-trivial command goes into a script file; the Write tool emits CRLF
  even on `\\wsl.localhost` paths, so run scripts through `/tmp/gtoc12_scripts/run.sh <name> [args]`
  (strips CR, sets PYTHONPATH=src and the git author, cd's to the worktree) and normalise repo
  files with `/tmp/gtoc12_scripts/eol.sh` before ruff/commit. Never `wsl -- bash -c "..."` with
  quotes, `|`, `$`, `<` inside.
- `.venv` has no installed `spacepdhcg`: `PYTHONPATH=src` for every python/pytest call.
- Never edit `src/` while a campaign runs (forked workers may lazily import); stop first.
- Size campaign wrappers' `timeout` with headroom: `timeout 15600` (260 min) killed
  `cluster_fleet_v9` at 246 logged minutes + the last family's tail, before the final master and
  `fleet/` were written (v8 had fitted in 249 min). Use 18000 for a 4 h campaign.
- Check `ps` for a run id before launching; one script per run (the generic runner drops args).
- The Held-Karp DP's mass model is deliberately pessimistic (collected set mined to the window
  end on every move): judge closure of a tour by the exact forward pass (`_plan_from_tour` ->
  `_finish`), never by `mass - DP propellant >= dry`.
- A method that updates state on the object it scores (`_chain_score` writes `chain_burn`)
  changes what a second call computes; tests must fetch the cached artefact first.
- Merging `main` into a results branch: rename colliding `results/gtoc12/runs/<id>` *before* the
  merge (`git mv` + rewrite run_id/artifact paths), keep both; both sides' docs sections are
  kept and renumbered; preview conflicts with `git merge-tree <base> HEAD main`.
- A campaign's families are disjoint: per-asteroid master duals only bite with the archive's
  columns in the LP (`--dual-archive`), and a near-integral LP puts the rent on the column bounds
  (`--dual-bound-share`), not on the asteroid rows - check `priced_members` per family before
  expecting dual feedback to steer anything.

## Active Risks / Workstreams

- GTOC12: `fleet_master_v8` = 21 ships / 12 356.30 kg / 588.40 avg (proven optimal over 1296
  columns, LP(22) infeasible). Ship 22 needs 599.5 kg average. Levers that moved the fleet:
  joint itinerary (+208 kg v2, +411 kg v3 over v8/v9 ships); levers that did not: chain-level
  beam objective (paired median 0.0), reference prior at 0.5, LP duals (nothing to price).
- Next: score phase alignment of consecutive deploys at the projected harvest epoch in the beam;
  Earth-leg optimiser trading propellant for arrival time; joint itinerary over every archived
  stand-alone ship (~3 h) before the next master; `Result.txt` -> column ingester for the H100
  fleets copied to the Windows repo.
- Unlocalised: the NaN pass-1 burn schedule that crashed `cluster_fleet_v9` family 10 (guarded).

## Windows-Checkout Notes Carried On 2026-09-05

Verbatim blocks from the Windows checkout's live file (entries other workers wrote there on 2026-09-05; carried by the third release merge, grouped under their original headings).

### File Policy (Windows checkout)

- Rationale: benchmark and research-workflow lessons should be shared across sessions.
- Rolled over 2026-09-05 (archive-first). Detailed history 2026-09-01..2026-09-04 lives in
  `.cursor/memory/AGENT_SCRATCHPAD_2026-09-01_to_2026-09-04.md` (reconstructed from a verbatim read
  after a faulty rollover deleted the live file; see the provenance note there).

### Regression-prevention guardrails (Windows checkout)

- Seal order is summarize -> validate -> seal -> verify, and nothing may write into the evidence
  tree while `archive_run.py` indexes it (redirect `root-index.log` outside the tree).
- Native-Linux `nvidia-smi` lists compute apps (WSL never does): any "foreign process" preflight
  must exclude `os.getpid()`, and every GPU gate on Lambda will see its own PID.
- Recursive Python searches over archive-wide pools (fleet-master: one frame per column) must set
  `sys.setrecursionlimit` for the search; CPython's 1000 is hit at ~1000 columns.
- `scp -r` of hundreds of small files from Lambda takes ~12 min for 29 MB; tar on the remote first
  next time, and drop candidate `fleets/*/Result.txt` (6-8 MB each) before staging.
- Lambda `apt` OpenMPI conflicts with the preinstalled stack; reuse the existing `mpicc`/`mpirun`
  and set `MPI_HOME` instead of installing `libopenmpi-dev`.
- "Fast-forward of X" claims must be checked with `git merge-base --is-ancestor X Y`: `git log X..Y`
  showing one commit also holds when Y sits on X's parent. Never move a branch non-ff; check the
  target commit out on its own branch and report.
- The G4 executor bakes `SPACEPDHCG_SOURCE_COMMIT` at CMake configure time: after any commit,
  configure into a *fresh* build dir (a rebuild alone keeps the old commit and the capability
  generator refuses it).
- The G4 contamination monitor's WSL channels (`/dev/dxg` holders, Windows `nvidia-smi.exe pmon`)
  are inert on native Linux; point `--host-nvidia-smi` at a wrapper that runs `nvidia-smi pmon -c 1`
  and drops rows belonging to the worker's own process tree (`~/s/host-pmon-linux.sh`), otherwise
  the worker either flags itself or detects nothing. `SHARED_GPU_LOCK_FILE` is hard-coded to
  `/home/angus/...`; create that directory for the advisory lock rather than patching the source.
- `run_g4_campaign.py` writes its JSON events to **stderr**; status scripts must grep the stderr log.
- A `--g4-session` manifest must carry exactly nine attempts; to exercise one 600 s attempt cheaply,
  shrink `SPACEPDHCG_G4_GROUP_DEADLINE_SECONDS` (the executor clamps and records the rest `unrun`).
- Header hooks found by ADL (`project_rk4_variational`, `has_quaternion_projection`) silently
  disappear when a TU does not include the model's variational header: every header that dispatches
  on such a trait must `#include` the provider and `static_assert` the trait is visible. A CPU
  smoke test that includes *only* the dispatching header is the regression guard.
- Device/host "parity" tests must compare like for like: a replay kernel that RK4-integrates a
  family whose host adapter uses the exact matrix-exponential ZOH (HCW) fails at ~1e-6 forever.
- Certificates read the residual of the *accepted* solve; reporting the last attempted (rejected
  polish) candidate fails a converged plan. QOCO's raw residuals are absolute; the audit is relative.
- Warm-started QOCO returns `SOLVED_INACCURATE` after one iteration on some polish solves; a single
  cold re-solve fixes it (keep `dual_discarded` describing the *requested* mode, not the retry).
- `examples/planner/*.json` are user-unit documents (pd3/pd6 in degrees); only the Python CLI
  normalises them. Native executables take `spacepdhcg validate` output. Manifest commands that
  call an executable on a raw example are refused by `tests/test_gpu_deferred_manifest.py`.
- Static copy lists for the viewer bundle rot; discover the ES module graph from `app.js` imports.
- Under `bash -c '... &'` from `wsl -e`, the whole list is backgrounded and killed when wsl exits;
  use `nohup setsid ... & disown` and `sed` the CRLF *before* the launch line, not inside it.
- Windows `nvidia-smi.exe` lists every DWM process as a "compute app"; the WSL `nvidia-smi
  --query-compute-apps` is the one that shows real CUDA contexts (with WSL PIDs, `[Not Found]` name).
- [user] 2026-09-05 03:46 AEST: the Lambda H100 GPU is reserved exclusively for the G4 claim-core
  campaign; no GPU process of any kind on the H100 (no CTest/pytest-GPU/sanitizer/examples). The
  H100 clone may be used CPU-only at <= 4 cores, `nice 10` (a 22-worker GTOC12 campaign shares the
  host). GPU verification goes to the WSL RTX 5090 (sm_120) after checking `nvidia-smi` (WSL and
  Windows `nvidia-smi.exe --query-compute-apps`) for foreign `python.exe` compute.

### Active workstreams / risks (as of 2026-09-05) (Windows checkout)

- Lambda H100 (`ubuntu@192.222.55.229`, key `traj-key.pem`, git-ignored): clones at
  `/home/ubuntu/spacepdhcg/{v1,v2,gtoc12}`; env in `~/spacepdhcg/env.sh`; logs in `~/logs`; helper
  scripts in `~/s`. **G4 claim core is RUNNING on the H100** since 2026-09-04T18:55Z: v1 checked out
  at 1dbcae0 on branch `g4/h100-1dbcae0` (1dbcae0 is a sibling of 9e75b47, so
  `integration/single-gpu-v1` there still points at 9e75b47), campaign
  `~/spacepdhcg/v1/build-integration-report/g4-claim-core-1dbcae0-h100`, capability
  `~/g4/g4-executor-capability-1dbcae0-h100.json` (`0b4c8c38?`), worker on cores 0-3, logs
  `~/logs/g4-h100/` (events in `worker.err`), `~/g4/STATUS.txt` has monitor/pause/restart/finish
  commands (`~/s/g4-status.sh`, `~/s/g4_progress.sh`, `~/s/g4-worker.sh`, `~/s/g4-finish.sh`).
  Another agent runs GTOC12 `cluster_fleet_h100_v2` (29 procs, moved to cores 4-25) on the same host.
  Evidence copied to `results/lambda-h100/` (ignored). See the 2026-09-05 H100 session entries below.
- H100-side fix commits are in WSL as fetch-only refs `refs/h100/single-gpu-v1` (9e75b47),
  `refs/h100/single-gpu-v2-candidate` (5aabbfc), `refs/h100/gtoc12-asteroid-mining` (c4e2c31);
  v2 and gtoc12 WSL branches have since moved (1/1 and 1/4 ahead/behind), so they need a merge,
  not a fast-forward. Bundles under `/home/angus/bundles/from-h100/`.
- v2 candidate defects exposed by the first real-GPU run (`results/lambda-h100/v2-deferred-3373988/
  triage.md`) are FIXED on `integration/single-gpu-v2-candidate` 1f5e034 (WSL worktree
  `/home/angus/worktrees/spacepdhcg-single-gpu-v2`; H100 clone `~/spacepdhcg/v2` fast-forwarded to
  the same SHA, CPU-only). All GPU verification ran on the RTX 5090 (sm_120); the H100 (sm_90)
  re-run is PENDING a GPU window - checklist in `~/spacepdhcg/v2-PENDING-H100-GPU-VERIFY.txt` on
  the H100. See the 2026-09-05 05:30 session entry.
- Palette collision RESOLVED 2026-09-05 05:35: `feat/viewer-40-ships` 7496c10 merged into
  `integration/single-gpu-v2-candidate` as 211267d (dynamic `viewer_modules()` + new
  `viewer_scripts()` discovery, 40-colour palette, check.mjs reads the size from gtoc12.js and
  regenerates from the spec). Bundle `/home/angus/bundles/single-gpu-v2-viewer40-211267d.bundle`
  (sha 2dc159da...d792; copy at `C:\Users\Angus\Desktop\projects\`). Windows repo has the refs
  (`integration/single-gpu-v2-candidate` 211267d, `feat/viewer-40-ships` 7496c10), no checkout.
- SUPERSEDED 2026-09-05 06:40 (merge landed, see the 06:30 session entry): the merge-to-main
  prediction above was accurate - `viewer_export.py` took the candidate's discovery,
  `native_qoco_adapter.h` kept both field sets (+ `status_code` refreshed after the cold retry),
  the blob table was refreshed in 0ff4f7c. `main` = origin/main = 8cb3759. The Windows checkout is
  still `feat/webgl-trajectory-viewer` d88eb51 (20-colour palette); `..\viewer-live` serves 211267d.

## Session Entries

### 2026-09-05 06:50 AEST ? orchestrator: roadmap dashboard truncated and restored

#### Mistakes And Fixes

- `[self]` Used PowerShell `Get-Content -Raw` + `Set-Content` to bulk-edit the live canvas
  `~/.cursor/projects/<ws>/canvases/spacepdhcg-roadmap-status.canvas.tsx` while the canvas host held it
  open. `Set-Content` failed with "Stream was not readable" AFTER truncating the file to 0 bytes; the
  subsequent `StrReplace` edits all reported "string not found". Recovered from the newest copy in
  `%APPDATA%\Cursor\User\globalStorage\anysphere.cursor-retrieval\checkpoints\*\files\*` (4 Sep 17:00,
  found with `rg -l 'Adaptive / IPM / hybrid study'`) and re-applied the day's state by hand.

#### Guardrails

- Never edit a `.canvas.tsx` (or any file the IDE holds open) with shell redirection / `Set-Content`;
  use the editor tools (`StrReplace` with `replace_all` for bulk renames, or `Write`). Check
  `(Get-Item f).Length` immediately after any shell write that errors.
- Canvas files are not in git: the only backups are Cursor retrieval checkpoints. Before a large
  dashboard rewrite, copy the current file to `C:\Users\Angus\AppData\Local\Temp\` first.

### 2026-09-05 08:40 AEST - H100 GTOC12 v2 watch (in progress; lessons captured as they occur)

- `[tool]` `powershell -NoProfile -File x.ps1 -Files a.sh, b.sh, c.py` binds only the first element:
  `-File` passes plain strings, so a `[string[]]` parameter never sees the comma list. Loop over the
  files inside the current session (or dot-source the helper) instead of a `-File` invocation.
- `[tool]` The Write tool produces CRLF for files under `C:\Users\Angus\h100work\s\`; every `rput`
  of a Write-created script must be preceded by the LF rewrite (`bash -n` on the host reports
  `$'do\r'` otherwise). `grep -c $'\r' <file>` on the host is the cheap post-upload check.
- `[self]` `finalize_v2.sh` ran the v2 viewer importer with `cd ~/spacepdhcg/v2/web/trajectory-viewer`
  (writes `data/gtoc12` inside the clone the user declared read-only). Changed to
  `node <v2>/scripts/import-gtoc12.mjs ... --output ~/stage/viewer-import/<run>` - the importer
  supports `--output`; read a script's argument parser before deciding a clone must be written to.
- Family completions arrive in ~110-min batches (6600 s/family, 22 workers): "no new family for two
  polls" is normal; use the cluster log mtime/size and `run_report.json` mtime as the stall signal.
- `[self]` Three times in one session an inline `rsh "... $(pgrep ...)"` was parsed by PowerShell
  (`pgrep` not recognised) despite the standing rule. Zero-tolerance version of the rule: *every*
  remote command line goes through `wlf` + `rrun`, even one-liners; `rsh` is for `cat`/`tail` only.
- `[self]` Killed a local `ssh` process by age ("older than 2 min") to free a hung launch session;
  the older PID (started 05:44 AEST) could have been another agent's live session. Rule: identify
  the ssh PID of the *own* launch (`Get-Process ssh` before and after) and kill only that one; the
  remote `setsid nohup` job survives either way (G4 procs verified alive afterwards).
- `[tool]` `AwaitShell` with no `shell_id` slept 57-102 min when asked for 30-35 min twice; treat
  the wall clock from the poll output as the schedule, not the requested sleep.
- `[tool]` `setsid nohup ... &` inside a `rrun` script still keeps the ssh client open when the script
  continues with `sleep`/`cat` after the launch (the harness backgrounds the Shell); end launch
  scripts right after the `&` line and do the status read in a second `rrun`.

### 2026-09-05 09:20 AEST - H100 G4 claim core launched on the deadline fix 1dbcae0

#### Task Summary

- Bundle `single-gpu-v1-1dbcae0.bundle` sha256 `5e4de5e3?` verified in WSL and on the H100. 1dbcae0
  is **not** a descendant of 9e75b47 (both sit on addac2b), so `git merge --ff-only` refused; checked
  1dbcae0 out as branch `g4/h100-1dbcae0` in `~/spacepdhcg/v1` and left `integration/single-gpu-v1`
  at 9e75b47 (no non-ff moves). Commit carries `cpp/cuda/tests/cancellation_deadline_test.cu` and
  `tests/test_g4_pdhcg_deadline_gpu.py`.
- Fresh sm_90 configure+build (`build-g4-cuda-{release,debug}`, ~20 s each on the 8480+): CTest
  63/63 Release (190 s) and 63/63 Debug (196 s), `cancellation_deadline_test` 49.7 s / 48.8 s;
  `test_g4_pdhcg_deadline_gpu.py` 13/13 in 1319 s; ordinal-73 twin repro (600 s / 1M cap, group
  deadline clamped to 1260 s): warm-up/0 600.054 s, warm-up/1 600.029 s at inner_iterations 300000,
  measured/0 cancelled at 59.66 s, six `unrun`. Evidence `~/g4/fix-verification-1dbcae0/`.
- Capability `0b4c8c38a6ba34b45cdf1ee5ae72869da272df5d866801009930f2b235a6f7f5` (executable
  `3703d52c?`, libqoco `5f778efb?` = reseal library, IPM probe 9/9 workspace creations, 4 s); fresh
  checkpoint 0/396 under amendment v1.2 with the ccd5596 `ipm_no_equilibration_v1_1` stratum cited
  by metadata only.
- Launched 2026-09-04T18:55:16Z: worker pid 53138 on cores 0-3, observer 53137, `--g4-server 600`
  persistent process + one `--g4-session` per group. GPU exclusive (no compute apps before launch);
  29 GTOC12 `cluster_fleet_h100_v2` processes (another agent) moved from 0-25 to 4-25.
- First groups: 66 IPM groups all `numerical` x9 (same class as the RTX 5090 ccd5596 stratum) at
  ~20 s each; ordinal 66 (first PDHCG core group) 1080.73 s = 9 x 120 s timeouts, cancel latency
  +12..+49 ms; ordinal 73 (first twin) 5400.67 s < 5460 s group deadline, 9 x 600 s timeouts, cancel
  +8..+51 ms, every attempt at inner_iterations 300000. 0 contaminated attempts in 75 groups.
- Projection (deadline-bounded upper bound, every remaining PDHCG attempt timing out): 134 h ->
  ~2026-09-10T14:00Z; per class 1080.6 s core / 5400.7 s twin measured, hybrid and fixed-tight not
  yet sampled.
- Evidence home: `results/lambda-h100/g4/` (capability, checkpoint snapshot, fix verification,
  ordinals 0/5/66/67/73 records, logs, scripts); `~/g4/STATUS.txt` has monitor/pause/restart/finish.

#### Mistakes And Fixes

- `[self]` Took the user's "fast-forward of 9e75b47" at face value; the fetch+ff step failed and the
  ancestry check showed a sibling. Rule recorded (verify with `merge-base --is-ancestor`).
- `[tool]` `run_g4_campaign.py` logs JSON events to stderr; the first status script grepped the
  empty stdout log. Fixed to `worker.err`.
- `[tool]` Reading the ccd5596 stratum checkpoint read-only still created `-wal/-shm` files;
  removed them.

#### What Worked

- Exercising a 600 s attempt in 21 min by shrinking the group deadline (executor clamps; rest `unrun`).
- `host-pmon-linux.sh`: native `nvidia-smi pmon` with the worker's own process tree filtered gives
  the v1.2 run-and-flag monitor a working channel on Linux (`host_active` empty, no self-flagging).
- Progress analyser (`g4_progress.sh`) that derives remaining classes from the amended schedule,
  not from the lazily materialised `coordinates` table.

#### Follow-Ups / Risks

- Merge `refs/h100/single-gpu-v1` (9e75b47, evidence-script arch parametrisation) into
  `integration/single-gpu-v1` in WSL; the H100 clone's integration branch is behind 1dbcae0.
- Watch the first hybrid-pdhcg-ipm (ordinal 198) and fixed-tight (264) groups; the projection
  tightens once they are sampled. Run `g4-finish.sh --preview` at ~50 % for an early decision read.
- The GTOC12 v2 campaign on cores 4-25 keeps host load ~27; the worker's cores 0-3 are exclusive.

### 2026-09-05 11:30-13:55 AEST - G2/G3 reseal of main 8cb3759 on the WSL RTX 5090 (PASS/PASS)

- Outcome: G2 PASS, G3 PASS on main 8cb3759 (tree 6d27f25), commit `06e70b62c2c8e708a9221c7508e21b58e8d5da37`
  on `chore/g2g3-reseal-8cb3759` (docs section + compact evidence force-added; archives local-only).
  Root index sha256 `443a8caf16e09699c67f499d59078261cfb94b5408c59e07c0e03dd83cd4e4a2`; g2 archive
  `095f33dc...da45d`, g3 archive `609e0acb...39de6`. Wall 6795 s (G2 775 s, G3 5786 s of which the
  recovery racecheck 56m31s). Key numbers: tight canonical max 9.69295039e-7 (pd6 now 2.83e-8 vs
  1.15e-8 at b6afb49 - library change, well inside 1e-6); displaced HCW 3 steps; pure-QOCO 2/24/2;
  fixed-tight 3/3 honest negatives; production canonical 9.57e-9 / nonlinear 2.93e-8 / trajectory 0;
  H1 supported from 20; 5 + 16 sanitizer logs clean. One 180 s foreign-GPU wait before G2 (WSL
  weldsim CUDA process, another agent), recorded in `preflight/orchestrator-gpu-waits.log`.
- Where: fresh worktree `/home/angus/worktrees/spacepdhcg-reseal-8cb3759` on branch
  `chore/g2g3-reseal-8cb3759` cut at 8cb3759 (spacepdhcg-main verified clean at 8cb3759 on `main`
  and left untouched so the branch commit does not move the main worktree). Evidence tree
  `results/gpu/current-head-8cb3759-rtx5090/{preflight,g2,g3,seals}`; helper scripts + step logs
  `/home/angus/reseal8cb/` (`run_all.sh`, `status.sh`, `peek.sh`, `logs/RESULT`). Windows copies of
  every script: `C:\Users\Angus\AppData\Local\Temp\reseal8cb\`.
- Procedure = the sealed b6afb49/9e75b47 per-gate `run.sh` templates (from
  `spacepdhcg-single-gpu-integration/results/gpu/current-head-b0cd570/`) with commit/tree/branch
  substituted, G0/G1 dropped from the seals tooling, plus `gpu_guard` (pre-step foreign-GPU wait,
  logged to `foreign-gpu-waits.log`) and absolute nice-10 `-j8` builds (`nice -n $((10-$(nice)))`).
  No cherry-pick of 9e75b47 was needed: main's scripts hard-code `CMAKE_CUDA_ARCHITECTURES=120`
  and `hardware_id local-rtx-5090`, both correct for this GPU.
- `[tool]` WSL `nvidia-smi --query-compute-apps` lists WSL CUDA contexts as `<pid>, [Not Found],
  [N/A]`; `ps -o cmd -p <pid>` names them. A foreign weldsim `demo_everything_on.py --device cuda:0`
  (another agent, `/home/angus/wt/d24integ/.venv`) held the GPU at launch; the orchestrator waited
  180 s (recorded) before G2. Windows `nvidia-smi.exe` showed no `python.exe` at any check.
- `[tool]` `wsl -- bash -lc 'a && b | head && c'` from PowerShell silently returned nothing (rc 1)
  when a middle command such as `grep` matched nothing; probes must be script files that write a log
  (`set -u`, no `-e`) and the log is read back through `\\wsl.localhost`.
- `[tool]` uv venvs have no `pip`; toolchain capture needs an `importlib.metadata` fallback.
- `[self]` `wrun` helper takes `-Args` as one string; passing a second positional arg fails.
- `[self]` First seals pass recorded only the per-step guard waits; the orchestrator's 180 s
  gate-level wait lived outside the tree. Fixed by copying the orchestrator wait log into
  `preflight/` and re-running summarize -> validate -> seal -> verify (raw gate evidence untouched;
  first-pass hashes a5a15c2b/fe30ceb8/16370272 retained in `preflight/orchestrator-first-pass.log`).
  Rule: any wait/guard record the summary cites must be inside the evidence tree before sealing.
- `[self]` `run.sh` overwrites `status.txt` at completion, dropping `started_utc`; keep both stamps
  (or read the orchestrator log) if a summary wants durations.
- `[tool]` The 9fafee8 `--sanitizer` 20k cancellation cap did not shorten the `recovery_test`
  racecheck (56.5 min here, 54 min b6afb49, 61 min H100): budget G3 at ~95 min and do not mistake
  the long racecheck for a hang while GPU util is 100 %.
- What worked: template `run.sh` reused verbatim + two additive wrappers (`gpu_guard`, absolute
  nice-10 builds via `nice -n $((10-$(nice)))`); docs numbers rendered from the sealed summaries
  (`make_docs_section.py`); compact evidence force-added with `git add -f` under ignored `results/`
  (precedent: tracked `results/gpu/g2|g3/*` seals, whose `commands.txt` also carries `%q ` trailing
  spaces - do not "fix" sealed evidence for `git diff --check`).
- Cleanup left in place on purpose: worktree `spacepdhcg-reseal-8cb3759` with build dirs (~950 MB),
  `_upstream/{pdhcg,qoco-current-head}`, `.venv-current-head`; `/home/angus/reseal8cb/` scripts+logs.

### 2026-09-05 13:50 AEST - G2/G3 reseal of main 8cb3759 on the WSL RTX 5090

#### Task Summary

- G2 and G3 PASS on main 8cb3759 (sm_120), evidence `results/gpu/current-head-8cb3759-rtx5090/`
  (root index sha256 `443a8caf...4e4a2`, g2 archive `095f33dc...da45d`, g3 archive
  `609e0acb...39de6`), committed compactly on `chore/g2g3-reseal-8cb3759` (worktree
  `/home/angus/worktrees/spacepdhcg-reseal-8cb3759`; helper scripts + step logs `/home/angus/reseal8cb/`).
  Wall 6795 s; G3 racecheck of `recovery_test --sanitizer` alone 56.5 min.

#### Mistakes And Fixes

- `[self]` The first seals pass recorded per-step waits only; the orchestrator's 180 s gate-level
  wait lived outside the tree. Re-ran summarize -> validate -> seal -> verify after copying the
  orchestrator wait log into `preflight/` (raw gate evidence untouched, first-pass hashes retained).
  Rule: every wait/guard log that the summary cites must live inside the evidence tree before sealing.
- `[self]` The runner's final `status.txt` overwrites the `started_utc` line; summaries that want
  start/end stamps must read the orchestrator log or keep both lines in `status.txt`.
- `[tool]` `wsl -- bash -lc 'a && b | head && c'` from PowerShell returned nothing (rc 1) whenever a
  middle `grep` matched nothing; probes are script files that write a log read back over
  `\\wsl.localhost`.

#### What Worked

- Reusing the sealed per-gate `run.sh` templates verbatim (commit/tree/branch substituted) plus two
  additive wrappers: `gpu_guard` (WSL `nvidia-smi --query-compute-apps` + Windows `nvidia-smi.exe`
  filtered for python/torch/cuda names, 30 s poll, every check logged) and
  `nice -n $((10-$(nice)))` for builds so `-j8` builds land at absolute nice 10 under a nice-5 runner.
- Generating the docs section from the sealed summaries (`make_docs_section.py`) instead of typing
  numbers; force-adding only compact files (`git add -f`) under the ignored `results/` tree, matching
  the pattern of the earlier tracked `results/gpu/g2|g3` seals.
- A fresh worktree on the chore branch at the same commit keeps `spacepdhcg-main` on `main` and lets
  the evidence commit land without moving any shared worktree.

#### Guardrails For Next Session

- WSL `nvidia-smi --query-compute-apps` shows WSL CUDA contexts as `<pid>, [Not Found], [N/A]`;
  identify them with `ps -o cmd -p <pid>` (a weldsim demo from another agent held the GPU for 3 min).
- The recovery racecheck is ~55-60 min on every host regardless of the 9fafee8 `--sanitizer` cap;
  budget G3 at ~95 min and do not treat a long racecheck as a hang while GPU util stays 100 %.

### 2026-09-05 (ninth iteration: chain-aware beam, reference prior, LP duals, joint itinerary in the pricing)

#### Task Summary

- Step 0: renamed this branch's 20-ship `fleet_master_v7` to `fleet_master_v7_v8archives`
  (610c18d), merged `main` 8cb3759 (5eeb7da; joint-itinerary code, 21-ship `fleet_master_v7`,
  recursion fix; docs §6.10 ours / §6.11 main's), suite 125 -> 133 passed.
- v9 code (9325252, 1f6ec50, + fixes): `chainprior.py` + `gtoc12 chain-prior` +
  `benchmarks/gtoc12/chain_prior_v1.json`; `RouteSearch._select` shortlist -> `_chain_score`
  (exact DP tour -> exact plan -> `plan_score`), `chain_burn` inheritance, tour cache;
  `lp_asteroid_prices` (+ `bound_share`), `archive.pricing_columns`, dispatch-time prices in
  `price_clusters`; joint itinerary per self-cleaning slot in `price_cluster`; NaN burn guard.
  Tests: brute-force DP exactness, chain-score arithmetic and prices, prior monotonicity and
  reproducibility, dual prices and dispatch snapshots, bound-share conservation, beam
  determinism + price steering (data-backed).
- Runs: probe family 7 (622.6 kg, same chain as v8 + 6.2 joint), diagnostics (48 vs 144
  candidates identical), `cluster_fleet_v9` (20 families / 60 ships / 18-ship 9960.3 kg
  incumbent; killed by the wrapper timeout), `joint_itinerary_v3` (+411.5 kg, 35/40),
  `fleet_master_v8` (21 / 12 356.30 / 588.40, +9.8 over v7, both verifiers ok).

#### Mistakes And Fixes

- [self] First chain scorer judged closure by the DP's pessimistic mass model: `chain-scored 0`
  at depth 9 on family 7 while `_complete` closed those chains. Fixed by scoring the exact plan
  (`_plan_from_tour` -> `_finish`), after which 9 of 24 depth-9 chains scored; the campaign was
  stopped and relaunched (attempt 1 lost 5 min).
- [self] Launched the campaign with campaign-only duals before realising disjoint families make
  them inert (attempt 2, 37 min lost); relaunched with `--dual-archive` over the seventeen
  archives. Even so only 22-24 asteroids carried a dual and one lay in a priced family.
- [self] `timeout 15600` on the wrapper was too tight for a run whose families take longer than
  v8's (joint step + DP scoring): the final master/fleet were never written. Incumbent used.
- [tool] `gtoc12 joint-itinerary` on `main` imports `REPOSITORY_ROOT` from `.data`, which does
  not exist after the release merge (ImportError on the first ji3 launch); fixed to
  `resources.repository_root()`.
- [tool] `git add` of renamed result files needs `-f` (results/ is ignored, files force-added).

#### What Worked

- Measuring the DP's cost profile before designing the shortlist (probe_dp_time.py); the
  brute-force exactness test caught nothing but pins the DP semantics for the scorer.
- `pricing_columns` over 915 archived routes reproduces `fleet_master_v7`'s LP bound in 0.5 s
  without re-certification - the LP can be studied offline.
- The joint-itinerary post-pass as a cheap archive-wide lever (7.8 min for 40 ships, +411 kg).

#### What Failed Or Was Inefficient

- The chain-level objective is neutral (paired median 0.0 kg, 7 up / 6 down) at 78 s per beam;
  the prior at 0.5 did not change winners; the duals had nothing to price. Two relaunches.

#### Guardrails For Next Session

- See "Regression-Prevention Guardrails" (timeout headroom, disjoint families vs duals, exact
  closure). Do not spend another campaign on per-pair or per-chain *propellant* scoring: the
  remaining gap is phase (|Δλ| at harvest 2.7 deg in the references) and Earth-leg arrival time.

### 2026-09-05 16:40 AEST - Third release merge into main (v9 gtoc12, H100 v2/v3, reseal, Windows memory + evidence)

#### Task Summary

- Merged onto `release/single-gpu-v1-merge` from 8cb3759 (WSL release worktree, identity via
  `GIT_*` env, merge commits only): `integration/single-gpu-v1` bf4cf0f (abd4e81),
  `chore/g2g3-reseal-8cb3759` 06e70b6 (16d5e8e), `feat/gtoc12-asteroid-mining` 1f6ec50 (a93649d)
  then b55eb70 (ace3b25, after the user's update), `refs/h100/gtoc12-asteroid-mining` 86a91d3
  (aaa9657) then 48e5fb7 (5f23f73, after the second update); Windows memory fold 1bd78ce,
  `.gitignore` 7a30c12, `results/lambda-h100` compact evidence 5784e64, status/memory (this commit).
- Headline: `fleet_master_h100_v2` = `v3` 22 ships / 13,189.60 kg / 599.53 avg (LP gap 3.4, not
  proven), best proven-optimal `fleet_master_v8` 21 / 12,356.30 / 588.40. Helper scripts + logs
  `/home/angus/integ3/`.

#### Mistakes And Fixes

- `[self]` Started the H100 merge before the user's b55eb70 update arrived; aborted the uncommitted
  merge (`git merge --abort`, no history touched), merged b55eb70 first, redid the same two code
  resolutions. Rule: before starting merge N+1, re-read the source branch tips - a worker may have
  moved them - and keep every resolution as a reproducible edit (StrReplace on the marker block).
- `[tool]` `git merge-tree <base> HEAD <tip>` reported 0 conflict hunks for the memory rollover and
  the docs table, yet `git merge` conflicted on both: the pre-scan is a hint, not a gate. Always run
  the real merge with `--no-commit` and read `git diff --name-only --diff-filter=U`.
- `[self]` First fold of the Windows memory entries re-sorted the snapshot's existing sections (the
  old files were appended, not sorted) and produced a 1000-line reordering diff. Fixed by stable
  insertion (new section after the last existing section with key <= its key; undated headings sort
  as end of day); the snapshot diffs became insert-only. Rule: memory merges must be insert-only -
  check `git diff -- <file> | grep -c '^-[^-]'` is 0.
- `[tool]` The Windows `DEVLOG.md` held 42 cp1252 bytes (0x85 `…`, 0x96 `–`, 0x97 `—`, 0xD7 `×`)
  inside UTF-8 text: `read_text(encoding="utf-8")` fails. Decode with `surrogateescape` and map the
  stray bytes through cp1252; compare lines ASCII-folded so mojibake variants of known lines are not
  "new content".
- `[tool]` `git -C /mnt/c/... status` from WSL showed ~250 modified tracked files that Windows git
  did not (autocrlf=true lives in the Windows system gitconfig). The Windows checkout is inspected
  with Windows git; WSL only compares CR-stripped file contents.
- `[tool]` The committed `fleet/viewer/` directories hold only `manifest.json`; the viewer import
  needs `trajectories.json`, regenerated with `spacepdhcg gtoc12 export-viewer <Result.txt> --output
  <dir> --run-id <run>` (10 MB, kept under `build-rel-verification/`, ignored). The stale
  `data/gtoc12/fleet.json` from the previous pass made `npm run check` pass on the wrong fleet
  first - check the validated run id in the output, not just the rc.
- `[tool]` `npm ci` fails in `web/trajectory-viewer` (no lockfile / no dependencies); `check` and
  `test` need no install.

#### What Worked

- One generic `merge_one.sh` (pre-scan, `--no-ff --no-commit`, commit only when no unmerged paths)
  + `commit_merge.sh` asserting `git merge-base --is-ancestor <tip> HEAD` after each merge.
- Line-coverage checks after every semantic resolution (docs table, memory rollover, Windows fold):
  `sort -u` both parents and the result, `comm -23`, and explain every missing line.
- Verification split into two parallel streams (host build + ctest + CPU pytest; CUDA build + CTest
  + planner GPU) with the wheel and viewer in between: full matrix in ~12 min wall on 16 cores.
- `results/lambda-h100/INDEX.json` (kept files with sha256, skipped files with sizes, the policy)
  makes the "what was committed vs left out" question answerable from the tree.

#### Guardrails For Next Session

- The v9 rollover convention: live memory files stay slim; anything dated at or before the
  snapshot's last timestamped entry goes into `*_2026-09-01_to_2026-09-05.md`. The live
  scratchpad is 400+ lines again after the Windows fold - roll it over at the next quiet point.
- sdist is 31.7 MB because `web/trajectory-viewer` (28 MB incl. the ignored `data/gtoc12` import)
  and `tests/__pycache__` are packed; not touched here - fix the sdist include list separately.
- `feat/gtoc12-asteroid-mining` moved past b55eb70 (bc7ef8e) while this merge ran; only b55eb70 is
  on main.
