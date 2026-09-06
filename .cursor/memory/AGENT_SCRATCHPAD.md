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

- GTOC12: `fleet_master_v9` (local, 25 archives) = 22 ships / 13 188.61 kg / 599.48 avg, proven
  optimal, rule binding at 22 <= 22.0007; `fleet_master_h100_v2/v3` = 22 / 13 189.60 / 599.53;
  **`fleet_master_v10` (H100, 36 archives) = 23 ships / 14 044.80 kg / 610.65 avg** (rule
  binding at 23 <= 23.005; ship 24 needs 621.2 kg average, +865 kg). Levers that moved the
  fleet: breadth (all 35 families priced) + joint itinerary (+208 v2, +411 v3, +316 v4, +664 v10);
  neutral: chain-level beam objective, reference prior, LP duals, harvest-phase prior, Earth-out
  leg stage (+494 kg over 131 ships but nothing on the 22-ship master).
- Next: plane-aware families (relative inclination / node gap of consecutive pairs: ours 2.4-2.7
  deg / 34 deg vs the references' 1.85 / 20); single-leg SCvx sweep of archived Earth legs
  across the launch window; deploy-phase time weight (deploy hops 240 d vs 183).
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

### Active Risks / Workstreams (Windows checkout, 2026-09-05 21:40)

Verbatim bullets of the Windows checkout's live `## Active Risks / Workstreams` section
(written 2026-09-05 21:40 by the tenth-iteration worker, carried by the fourth release merge;
the bullets already present above are not repeated).

- GTOC12 (2026-09-05 21:40, tenth iteration): **`fleet_master_v10` = 23 ships / 14 044.80 kg /
  610.65 avg** (H100, 36 archives, LP gap 6.3, proven optimal, both verifiers); local
  `fleet_master_v9` (25 archives) 22 / 13 188.61 / 599.48. Branch `feat/gtoc12-asteroid-mining`
  at dfdeca8f (WSL worktree clean; H100 clone c2730b1 clean). Ship 24 needs 621.2 kg average.
  Levers that moved the fleet: breadth (all 35 families) + joint itinerary (+208 v2, +411 v3,
  +316 v4, +664 v10); neutral: chain-level beam objective, reference prior, LP duals,
  harvest-phase prior, Earth-out leg stage (+494 kg over 131 ships, nothing on the master).

- Next: plane-aware families (relative inclination / node gap of consecutive collect pairs: ours
  2.5 deg / 34 deg vs the references' 1.85 / 20); single-leg SCvx sweep of archived Earth legs
  across the launch window (405 vs 430 kg legs exist); deploy-phase time weight (240 d vs 183).

- [self] Session lessons (long form in the branch scratchpad): detached launch scripts must end
  right after the `&` (`disown`, no trailing `sleep`/`cat`); `mkdir -p` every out-of-tree target
  before writing; compute a reference statistic on *our* archives before building a term from
  it; the H100's per-core speed is ~0.6x the WSL box, so pair arms on the same host.

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
  recursion fix; docs �6.10 ours / �6.11 main's), suite 125 -> 133 passed.
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
  remaining gap is phase (|??| at harvest 2.7 deg in the references) and Earth-leg arrival time.

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
- `[tool]` The Windows `DEVLOG.md` held 42 cp1252 bytes (0x85 `?`, 0x96 `?`, 0x97 `?`, 0xD7 `�`)
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
- `[self]` **Data loss on the Windows checkout, recovered.** Cleaning the dirty tree before
  `git checkout main` ran `git checkout -- .` and then `git clean -f`. The first command reverted
  the locally edited `.gitignore` to d88eb51's version, which lacked `_upstream/`, `*.pem` and
  `traj-key*`; the second then deleted the (previously ignored) `_upstream/` upstream checkouts and
  the Lambda key `traj-key.pem`, which the backup step had not copied because they were ignored
  when it ran. Recovered: `traj-key.pem` restored from `Downloads\traj-key.pem` (the original
  download, sha256 `479e5275?69ef`); `_upstream/{pdhcg,qoco,qoco-g4}` re-cloned at the pinned
  commits 167c8b72 (lock tree 62b05e6c matches), 89706f60 and 09f04959 (tree c85fe82f matches),
  mirroring the WSL checkouts. Rules: (1) never run `git clean` after reverting `.gitignore` -
  revert `.gitignore` last, or remove untracked files by explicit list from the pre-revert status;
  (2) before any clean, list `git status --ignored --porcelain` too and treat `!!` entries as
  precious; (3) no `-d` and no wildcard clean on a checkout that holds keys or vendored trees.

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
### 2026-09-05 (tenth iteration: Earth-out leg stage, harvest-phase prior, archive-wide joint, H100 paired arms) - in progress

#### Captured during the work

- [self] `setsid nohup ... &` at the end of a script run through `wsl -- bash run.sh` died with the
  wsl session (empty log, empty output dir, no process) - the first `joint_itinerary_v4` launch
  never ran. The recorded rule (`disown` + a `sleep` after the launch line) applies to *every*
  detached launch, not only to `wsl -- bash -c`.
- [self] `scp file host:~/dir/` with a missing remote directory fails with "Is a directory"
  under OpenSSH's SFTP mode; the standing rule "mkdir -p the remote target before rput" was
  skipped once more. Now `rput.sh` targets an explicit remote file name after `mkdir -p`.
- [self] A rigid 60-day shift of a whole deploy chain breaks a hop's authority ratio (hop 6 of
  the family-7 ship: Lambert 2.69 km/s at the shifted epochs, ratio 0.94 > 0.55): hops are
  phase-sensitive, so Earth-leg seeds must shift a *prefix* of the chain and let one hop absorb
  the shift (`earth_leg_seed(prefix=)`); the stage evaluates every prefix per candidate leg.
- [self] The joint evaluator's "proportional ore scaling" cannot close a propellant deficit:
  the closure rule `final >= dry + ore` cancels the ore, so scaling recovers only the ore's
  own propellant (~3 %). An unmeasured shorter Earth leg at the pair-calibrated 1.03x Lambert
  never closes on a propellant-bound ship; the seeds are ranked at the certified leg's
  *measured* inflation (0.83-0.89) instead (`_screen_earth_out`) and SCvx decides.
- Finding (measured, not assumed): our archived chains are already phase-aligned at harvest -
  `cluster_fleet_v9` |??| at collect departure median 2.5 deg / p75 4.0 deg, `fleet_master_v8`
  2.4 / 4.1, references 2.7 / 4.8 - so the brief's "nothing scores phase" diagnosis is not
  what separates the 210-day hops from the references' 181-day ones. The harvest-phase prio
  is calibrated and wired but bites on ~15 % of hops; measure it, do not expect it to move the
  fleet. The 210 vs 181 d and 85 vs 66 kg gap needs a different explanation (?a / ?i of the
  pairs, deploy-hop TOF 255 vs 183 d).
- Finding: a certified Earth leg is rarely the cheapest of its launch window - the 555-day leg
  of the 622.6 kg ship measured 405 kg against 430 kg at the certified 585 d. Earth legs are
  worth sweeping with SCvx around the certified point even when no chain shift is wanted.
- Finding: the Earth-leg exchange rate is tiny on a propellant-bound ship - 30 d earlier is
  +0.7 kg of ore for 8 miners while the leg's propellant delta must come out of margin; the
  stage yields ~+1 kg/ship where margin exists and nothing elsewhere.
- [tool] `pgrep -c -f fleet-master` inside a remote `bash -c` returns 1 when nothing runs
  (matches itself): read the ps listing, not the count, before deciding a host is idle.
- [self] `tar czf /home/angus/stage/x.tgz` failed because `/home/angus/stage` did not exist -
  the third "create the target directory first" miss in one session; every script that writes
  outside the worktree now starts with `mkdir -p`.
- [tool] `rput.sh` ran `bash -n` on a `.py` file; helpers that syntax-check must dispatch on
  the extension (`py_compile` for Python).
- [self] A `setsid nohup ... &` launch script followed by `sleep`/`cat` kept the ssh client
  attached for minutes (the harness backgrounded it); killed only the wsl process that was
  mine (PID checked), the remote job survived. Launch scripts must end right after the `&`.
- [tool] The H100 host's per-core speed is ~0.6x the WSL box for this workload: at the v9
  2400 s family budget its families reach ship slot 2-3 where the WSL box reached 3-4, so a
  paired comparison against a WSL campaign needs the same *hardware*, not the same budget; the
  two arms on the same host (`cluster_fleet_v10` vs `_control`) are the valid pair.

#### Task Summary

- Merge of the H100 v2 line (bc7ef8e), Earth-out leg stage (9ce3162), harvest-phase prio
  (f8e870c), campaign report script (8e2b6bf..ec23f01), results `joint_itinerary_v4/v5`
  (bfaee6e), `fleet_master_v9` (bfaa429): 22 ships / 13188.61 kg / 599.48 avg, proven
  optimal, both verifiers ok - the same 22-ship level as the H100 v2 master. H100 paired arms
  neutral (median 0.0 kg over 35 families); `fleet_master_v10` over 36 archives (H100, 22
  workers): **23 ships / 14 044.80 kg / 610.65 avg, LP gap 6.3, proven optimal, both verifiers
  ok** - ship 23 reached through breadth (35 families x 2 arms) + `joint_itinerary_v10
  --earth-leg` (+663.6 kg over 51 fresh chains), not through the two new per-ship terms.

#### What Worked

- Measuring first: `scripts/gtoc12_campaign_report.py` over the existing archives showed in
  minutes that the phase hypothesis was already satisfied and pointed at the orbital planes.
- Single-leg SCvx as the Earth-leg oracle (8 s) with the surrogate only ranking seeds; every
  accepted itinerary still goes through the whole-route certification and both verifiers.
- Two arms on one host at once (11 + 11 workers on cores 4-25): a paired A/B in 3 h with the
  same wall budget, the same families and the same hardware.
- Pipeline scripts with a status file per stage (`pipeline_a.status`, `gtoc12-v10-RESULT`) and
  polling only; nothing was babysat.

#### What Failed Or Was Inefficient

- Both new levers are neutral on the fleet: +2.3 kg median per ship from the Earth-leg stage,
  the phase prior inert on 76-85 % of our hops. One relaunch (joint_v4, dead detached launch).

#### Guardrails For Next Session

- Before building a term from a reference statistic, compute the same statistic on our own
  archives first (`gtoc12_campaign_report.py --run <ours> --fleet-ships <master>`).
- Ship 23 needs 610.6 kg average: the archives hold 24-25 chains >= 600 and one >= 650; the
  next lever must produce *new* chains (plane-aware families), not re-time old ones.

### 2026-09-05 22:10 AEST - Fourth release merge into main (gtoc12 v10: fleet_master_v10 23 ships)

#### Task Summary

- Merged `feat/gtoc12-asteroid-mining` dfdeca8f onto `release/single-gpu-v1-merge` from 2aecc65
  (fd7ef6d, criss-cross bases 48e5fb7 + b55eb70, four one-hunk conflicts), folded the Windows
  checkout's 21:40 memory notes (ed71737), extended `results/lambda-h100` with the v10 compact
  evidence + regenerated INDEX.json (f4028b3), verified, wrote the status note (this commit).
- Headline: **`fleet_master_v10` 23 ships / 196 asteroids / 14,044.80 kg / 610.65 avg, LP gap 6.3,
  proven optimal, both verifiers** (H100, 36 archives). Helper scripts + logs `/home/angus/integ4/`.

#### Mistakes And Fixes

- `[self]` First run of a Write-tool script through a fresh `run.sh` executed with CRLF intact (the
  bootstrap `sed` went through `wsl -- bash -c '...'` and lost its quoting): `cd "$R"` failed on
  `$'...\r'` and the read-only inspection commands ran in the *Windows checkout* (the `wsl` cwd).
  Nothing was written, but the rule is now mechanical: bootstrap a new helper dir's `run.sh` through
  an existing LF-clean runner (`bash /home/angus/integ3/run.sh /home/angus/integ4/run.sh <script>`),
  make `run.sh` `cd` into the helper dir before exec, and write every `cd "$X"` as `cd "$X" || exit 1`.
- `[tool]` `wsl -d ... -- <cmd>` from PowerShell re-splits the command line in the Linux shell:
  parentheses, `|`, `$` and quotes are lost (`grep -n -E "^(<<<<<<<|...)"` -> syntax error). Use the
  Grep/Read tools on `\\wsl.localhost\...` paths for ad-hoc inspection, scripts for everything else.
- `[self]` `git merge-tree <base> HEAD <tip>` picked one of two merge bases and reported 0 conflict
  hunks; the real merge (recursive over both bases) conflicted in four files. Same lesson as the third
  pass: the pre-scan is a hint; go straight to `git merge --no-ff --no-commit`.
- `[self]` Commit message of f4028b3 says "637 kept" where INDEX.json has 636 (608 + 28); recorded in
  the DEVLOG since amending is forbidden. Compute the number in the script and paste it, do not
  pre-write counts into message files.

#### What Worked

- `resolve_conflicts.py` with a per-file mode (`both` = HEAD side then incoming side, `theirs`,
  `ours`) + the `sort -u` / `comm -23` coverage check per parent: every missing line explained.
- `win_mem_fold.py`: section/bullet-level insert-only fold with an in-order survival assertion and an
  ASCII-folded "every Windows line present" check - a 30-line replacement for the third pass's tool.
- `regen_index.py` walks `git ls-files` rather than the copy list, so "every tracked file has a hash"
  is asserted, not assumed (tracked == kept, 0 hashed-but-untracked).
- Reusing the third pass's host build for the CPU pytest after proving `git diff 2aecc65 -- cpp/` is
  empty: the whole matrix (gtoc12 8 min, full pytest 10.5 min, viewer + wheel 1 min) in ~11 min wall.

#### Guardrails For Next Session

- The Windows checkout is not "clean on main" while a worker writes memory notes there: check
  `git status --porcelain` with *Windows* git before planning the final `git pull --ff-only`, fold the
  notes into main first, and never touch that tree with `checkout -- .` / `clean`.
- The live scratchpad is ~600 lines after two folds - roll it over at the next quiet point (archive-first).

### 2026-09-05 22:20 AEST - Fourth release merge landed (pushed refs, Windows checkout, live viewer)

- Pushed as fast-forwards from PowerShell after a bundle round-trip: `main` and
  `release/single-gpu-v1-merge` 2aecc65 -> 05d972f, `feat/gtoc12-asteroid-mining` b55eb70 -> dfdeca8f,
  `h100/gtoc12-asteroid-mining` 48e5fb7 -> c2730b1; `ls-remote` confirmed from both sides.
- `[user]` The Windows checkout was described as clean, but another worker had written memory notes into
  it. Resolution that respects "never `checkout -- .` / `clean`": fold the notes into main first
  (ed71737), copy the files to `%LOCALAPPDATA%\Temp\spacepdhcg-tmp\win-checkout-backup-<ts>\`, then
  `git stash push -- <exact files>` (recoverable, stash@{0}) and `git pull --ff-only`. Report the
  deviation. Rule: re-run Windows `git status --porcelain` immediately before the pull; a stash of the
  named files is the only working-tree touch allowed, and it is never popped over folded content.
- `[tool]` Windows autocrlf turns the force-added `results/lambda-h100` files into CRLF copies on
  checkout (592 "differing" files, 0 after CR stripping); compare evidence trees CR-stripped, never
  byte-wise, before deciding anything changed.
- `[tool]` The viewer-live importer accepts absolute Windows paths for every input; staging the WSL
  export/catalogue/Result.txt/fleet.json under `%LOCALAPPDATA%\Temp\viewer23\inputs\` avoids both UNC
  quirks and the Windows checkout's CRLF copy of `Result.txt` (whose hash would not match the export).
- Live viewer: `data/gtoc12.bak-h100v3-22ships` holds the previous 22-ship dataset; `data/gtoc12` is
  `fleet_master_v10` (23 ships, fleet sha `b9b3b6ba?aa62`); `browser-check.cjs` rc 0 with 23 distinct
  colours; screenshots in `C:\Users\Angus\AppData\Local\Temp\viewer23\`.

### 2026-09-05 22:35 AEST - PAUSED (user paused local development; v10 integration is COMPLETE)

- State at pause: the fourth release merge (GTOC12 v10) is finished and landed. No merge is in progress
  anywhere; no process of this session is running (all pytest/npm/node/build jobs exited before 22:11;
  the pre-existing viewer server on :4173, PID 47428, was never touched). Nothing on the Lambda instance
  was touched.
- Release worktree `/home/angus/worktrees/spacepdhcg-release`: branch `release/single-gpu-v1-merge` at
  `df1858f3`, `git status --porcelain` empty ("ahead 46 of origin/main" there is only the WSL clone's
  stale remote-tracking ref; WSL never fetches origin).
- `main`: GitHub `refs/heads/main` = `df1858f3` (= `release/single-gpu-v1-merge`); WSL
  `/home/angus/worktrees/spacepdhcg-main` = `df1858f3` (clean); Windows checkout = `df1858f3` on `main`,
  clean before this note (this PAUSED entry is now the only local change). Also pushed:
  `feat/gtoc12-asteroid-mining` = dfdeca8f (the tip merged), `h100/gtoc12-asteroid-mining` = c2730b1.
- Remaining to finish the v10 integration: nothing. Merged tip dfdeca8f; every verification step ran
  (ruff, full CPU pytest 683/35 skipped, gtoc12 + CLI dispatch 153, manifest 14/14, viewer check/test
  36/2, wheel + sdist + smoke); docs headline updated; live viewer serves `fleet_master_v10` (23 ships).
- If resuming integration work later (v11): merge `feat/gtoc12-asteroid-mining` at its tip then
  (`811ebe42` = plane-aligned family partition at pause time; worktree `spacepdhcg-gtoc12` had
  uncommitted files - do not touch it) into the release worktree with
  `bash /home/angus/integ4/run.sh /home/angus/integ4/merge_one.sh feat/gtoc12-asteroid-mining <sha> <msg>`;
  expect memory-file and `GTOC12_TRACK.md` conflicts (resolve with `resolve_conflicts.py`, both sides);
  rerun `60_cpu_static.sh`, `30_gtoc12_tests.sh`, `67_viewer_then_wheel.sh`; if `cpp/` changes, rebuild
  host/native/CUDA as in `/home/angus/integ3/61_cuda_build.sh` / `62_cpu_static.sh`.
- Temp files left in place: `/home/angus/integ4/` (scripts + logs), `/home/angus/bundles/release-merge-4-05d972f1.bundle`
  and `release-merge-4b-df1858f3.bundle` (copies on `C:\Users\Angus\Desktop\projects\`),
  `%LOCALAPPDATA%\Temp\spacepdhcg-tmp\win-checkout-backup-20260905-220632\` (the stashed memory notes,
  also in `git stash` stash@{0} of the Windows checkout - content is on main as ed71737, do not pop),
  `%LOCALAPPDATA%\Temp\viewer23\` (import inputs + browser-check artefacts), Windows refs `refs/integ4/*`,
  `build-rel-verification/{viewer-export-fleet_master_v10,artifacts-f4028b32}` and
  `build-rel-wheel-consumer` under the release worktree (all ignored), viewer-live
  `data/gtoc12.bak-h100v3-22ships`.

### 2026-09-05 22:40 AEST (12:40Z) - PAUSED: GTOC12 eleventh iteration (v11) - local stopped, H100 v11 campaign LEFT RUNNING

#### State at pause

- Local WSL worktree `/home/angus/worktrees/spacepdhcg-gtoc12`, branch `feat/gtoc12-asteroid-mining`,
  clean at `f20ede03` (no side branch needed; every edit passed ruff + the gtoc12 tests). Commits this
  iteration: `811ebe42` plane-aligned family partition (`planepairs.py`, `benchmarks/gtoc12/plane_pairs_v1.json`,
  `ClusterBands.plane_aligned` / `node_deg`, `--plane-families/--plane-only`, `gtoc12 plane-pairs`,
  `gtoc12 family-census`, `tests/test_gtoc12_planepairs.py`); `e70863e7` jointopt `earth_leg_sweep`
  (`--earth-leg-sweep`, `--earth-leg-sweep-days`; test in `tests/test_gtoc12_jointopt.py`); `f20ede03`
  memory notes. Not pushed (rule: no push from this line).
- No local process of this session is running (`pgrep -af "spacepdhcg gtoc12|pytest"` empty). The local
  sweep run `joint_itinerary_v11_sweep` (fleet_master_v10's 23 ships + 40 standalone chains, 600 s/ship,
  `--earth-leg --earth-leg-sweep`) was killed after 63 records; its partial `ships.jsonl` is on disk at
  `results/gtoc12/runs/joint_itinerary_v11_sweep/` (git-ignored, no `run_report.json`). At 12 ships:
  10 had a measurable sweep leg, 5 cheaper than the certified leg (median 13.8 kg, max 42.5), 4 no-shift
  legs accepted, +12.1 kg ore total (max +7.0/ship).
- Family census measured (committed in 811ebe42 as `results/gtoc12/leg_stats/plane_census_v11.json` and
  `plane_census_v11_widened.json`): as-specified plane thresholds (i-vector band 1.5-2.5 deg, radius
  1.6/1.75, node band 0/20 deg) give NO family of >= 18 members except (3.0 deg, r1.75, no node) = 22
  families / 19 new; the explicit node feature empties every cell. One documented widening (node off,
  3.0 deg, radius 2.0, phase 10 deg) = 74 families, 73 new vs the 167-family union (1 near-duplicate),
  nearest-pair mutual inclination median 1.80 deg / p75 2.69, node gap 21.6 deg, vs references
  1.85 / 3.45 / 20.3 and the union 2.45 / 3.61 / 28.8. On the references' 1014 collect pairs a degree of
  mutual inclination costs 6.6 kg of hop propellant, a degree of node gap 0.28 kg.
- Complement: 56 labels of the 167-family union never priced by `cluster_fleet_h100_v2`
  (H100 `~/s/complement56.txt`).

#### H100 (ssh -i /tmp/traj-key.pem ubuntu@192.222.55.229; first `cp .../traj-key.pem /tmp/ && chmod 600 /tmp/traj-key.pem`)

- Running unattended since 2026-09-05T12:06:59Z: `~/s/gtoc12_v11_campaign.sh` (setsid nohup, taskset 4-25,
  nice 5, `CUDA_VISIBLE_DEVICES=""`; verified at pause: 26 gtoc12 processes all on cores 4-25, none on the
  GPU, the G4 worker untouched). Clone `~/spacepdhcg/gtoc12` at `811ebe4`. Stages: arms
  `cluster_fleet_v11_plane` (74 plane families, 12 workers) || `cluster_fleet_v11_complement` (56 families,
  10 workers), 3840 s/family, campaign budget 23400 s, wrapper timeout 30000 s (arms end by ~20:30Z) ->
  `joint_itinerary_v11 --earth-leg` (22 workers) -> `fleet_master_v11` (v10's 36 sources + 3 new archives,
  LP bound) -> GTOC12_Verify + independent verifier + stats + report. Expect `stage=done` ~22:30Z +-1 h.
  At pause (12:40Z) 0/74 and 0/56 families finished (first ones due ~13:10Z).
- Status commands: `cat ~/logs/gtoc12-v11-RESULT` (RUNNING stage=arms|joint-v11|fleet-master|verify;
  `status=PASS stage=done`); `tail -3 ~/logs/gtoc12_v11_campaign.sh.log`;
  `ls ~/spacepdhcg/gtoc12/results/gtoc12/runs/cluster_fleet_v11_plane/clusters | wc -l` (and
  `..._complement/clusters`); `cat ~/gtoc12-v11/STATUS.txt` (full check/finalize/restart/stop sheet).
- Finalize (only when RESULT says stage=done): `taskset -c 4-25 nice -n 5 bash ~/s/finalize_v11.sh` ->
  commits compact v11 artefacts on the clone, viewer export if v11 beats v10, writes
  `~/bundles/from-h100/gtoc12-h100-v11-<sha>.bundle` (base `^811ebe4`), `~/stage/gtoc12-h100-v11-compact.tgz`
  + `-SHA256` + `-HEAD`.
- Restart if the orchestrator died: during the arms, wait until `pgrep -af cluster-fleet` is empty (arms
  checkpoint `fleets/` + `run_report.json` themselves), then
  `setsid nohup taskset -c 4-25 bash ~/s/resume_v11.sh > ~/logs/resume_v11.sh.log 2>&1 < /dev/null &`
  (post-arms stages only; refuses while any gtoc12 process runs; delete a half-written
  `results/gtoc12/runs/joint_itinerary_v11` or `fleet_master_v11` first). Stop:
  `pkill -TERM -f "bash /home/ubuntu/s/gtoc12_v11_campaign.sh"; pkill -TERM -f "spacepdhcg gtoc12"`
  (never `run_g4_campaign` / `g4-*`).

#### Resume steps (local)

1. `scp` the bundle + tarball + SHA256 home (`/home/angus/bundles/from-h100/`), verify the hash, then in
   the WSL worktree `git fetch <bundle> feat/gtoc12-asteroid-mining:refs/h100/gtoc12-asteroid-mining &&
   git merge --no-ff refs/h100/gtoc12-asteroid-mining`; unpack compact copies under
   `results/lambda-h100/gtoc12/`; run both verifiers locally on `fleet_master_v11/Result.txt`.
2. Optionally finish the local Earth-leg sweep (`joint-itinerary --run-id joint_itinerary_v11_sweep`
   over the remaining ships, 3 workers nice 19) and feed `fleet_master_v11`'s archives + the sweep
   archive into a `fleet_master_v11b` if the sweep freed any ore.
3. Report per the v11 brief: census (above), chain-mass distribution >= 600/620/650/700 of the two arms
   (`chain_stats` in the campaign report), pair geometry before/after/references (`--pair-geometry` in
   the campaign report vs `plane_pairs_v1.json`), fleet score/ships/avg/LP gap, ship 24 (>= 621.2 avg)
   yes/no, next bottleneck; docs `GTOC12_TRACK.md` �6.14 / �7 / �8, scratchpad + devlog; regenerate the
   viewer export if v11 beats 14,044.80 kg; commit results+docs.
- Helper scripts of this session (WSL): `/home/angus/v11/run.sh <script>` (all commands go through it -
  PowerShell breaks quoted `wsl -- bash -c`), `bg.sh` (detached), `stop_local.sh`, `h100_pause.sh`.

### 2026-09-05 22:50 AEST — performance review, initial implementation and live evidence audit

- [user] Resumed local code review/optimisation from the supplied Cursor transcript; asked to merge the performance-review branch and review new GPU results. Existing H100 campaigns were inspected read-only and left running.
- [self] Fetched origin main + review/performance-architecture-2026-09-05; review-only ad2ac42 merged locally as 01e3360. No push. Pre-existing scratchpad edits preserved.
- [tool] Cached/broadcast Lambert scan at unchanged FP64/tolerances: final seven-repeat speedups 1.23x (one), 1.98x (64), 2.18x (256), 1.93x (1024). All outputs bitwise equal; complete synthetic 150-day SCvx leg unchanged final mass/certificate, no meaningful end-to-end speedup claimed.
- [self] Fixed H1 compact work/memory/repeat semantics and added native H1 actual counters. Re-exported 42 archived H100 samples as new derivatives with hashes in artifacts/performance/h100-h1-corrected; frozen originals unchanged. CUDA telemetry TU compiled with nvcc 12.8, -Wformat/-Werror=format; no GPU execution.
- [tool] H100 at 12:46:57Z: 108/396 complete, 288 remaining, one running. Latest three adaptive N=500 low-thrust groups all 9/9 timeout, including 600s sensitivity group. Actual cold starts despite requested primal; canonical residuals 145–271, no outer progress. Snapshot artifacts/performance/h100-latest.json.
- [self] Full report docs/PERFORMANCE_PROGRESS_2026-09-05.md. GPU solver architecture, GTOC12 assembly persistence, QOCO conversion and G4 frozen policy are unchanged; no GPU speedup claim. New G4 telemetry concern: accepted_trajectory_count=1 on unqualified timeout records (schema currently requires positive count); use qualification/disposition, review contract separately.
- [tool] Final verification: full pinned-data GTOC12 + H1 suite 156 passed, zero skips, 337.27s; includes both verifiers on published 39/37/36-ship reference fleets. Stricter IEEE-bit comparison rerun: 3 passed. Ruff and diff whitespace checks passed.
- [tool] Actual route wrapper uses 256 scan samples, not API-default 8192. After tests stopped, controlled seven-repeat 256-grid batch speedups were 1.21x/1.62x/1.78x for 64/256/1024 transfers (25.043 ms -> 14.072 ms for 1024); standalone leg remains about 1x. Do not use the full-grid ~2x result as a fleet-search speedup. See artifacts/performance/screening-256-controlled.json and bitwise-8192.json.

## GPU-native active goal — local checkpoint, 2026-09-05
- [user] Explicitly requested a fully GPU-native C++/CUDA pipeline, local RTX 5090 testing, aggressive hill climbing, and an active goal retaining physics accuracy. This supersedes the previous CPU-only/no-local-GPU scope. Goal remains ACTIVE; no token budget. Do not mark it complete while CPU numerical hot paths and serial recovery/outer-loop work remain. Lambda campaigns remain untouched.
- [self] Working branch perf/gpu-native-pipeline. Added occupancy-bounded cooperative PDHG/scaling kernels, grid reductions/cone projections, multi-block coefficient scans, parallel SCvx numeric assembly, device-only explicit scaling refresh, and separate scaling/solve timers. Default parallel scaling at >=128 variables, multi-block iteration at >=4096. Legacy execution remains a comparison path.
- [tool] Fixed and tested a real signed-zero bug in atomic Ruiz row maxima: -0.0 bit patterns suppressed positive row norms. Nonzero HCW acceptance caught it. After canonicalizing zero, serial/cooperative HCW scales and steps match exactly, and 3 accepted / 1050 inner iterations match baseline.
- [tool] Corrected 2k HCW study: 2 warmups + 7 measured per configuration. Full command median 5.164s -> 0.461s (11.2x); SCvx wall 4.774s -> 67.204ms (71x); CQP 4.806s -> 1.836ms. ZERO-trajectory setup fixture, NOT a difficult-solve/global speedup. Every run has 1 actual inner iteration, 0 accepted steps and zero residuals. artifacts/performance/cooperative-hcw-2000-final.json holds raw samples and hashes.
- [tool] 10k HCW sweep (1 warmup + 5 samples): CQP 3.160ms at 128 blocks, 3.229ms at 170, 3.610ms at 256, 3.842ms at 340. SCvx ~328ms, dominated by remaining surrounding work. No baseline ratio claimed for 10k.
- [self] Follow-up numeric fingerprint block XOR reduction: bitwise-identical CPU/old/new GPU hashes, isolated kernel 1.63-5.11x faster; small/no substantial full-solve gain. Standalone numeric_fingerprint_test --benchmark plus all three sanitizers passed. Header cpp/cuda/internal/numeric_fingerprint.cuh.
- [tool] Native tests pass: cooperative (mixed SOC/RSOC, signed zero, cancellation, refresh, updates), persistent CW/SOC, allocation lifecycle, pointer contracts, stream lifetime, variational and time-dilated dynamics. Full four-family production test passes forced 8 blocks and automatic mode; latest with independent CPU fingerprint checks has maximum canonical 9.56640559e-9, nonlinear 2.92768854e-8, trajectory difference 0. memcheck/synccheck/racecheck all zero errors/hazards on cooperative and fingerprint tests.
- [tool] Build /home/angus/build-spacepdhcg-gpu-native via WSL, nvcc12.8 sm120. Cmake path /home/angus/spacecraft-trajectory-optmiser/.venv/bin/cmake. Baseline immutable library in build.../baseline; before-hash library in build.../before-hash. Local GPU jobs use flock /home/angus/.spacepdhcg-gpu.lock. No GPU jobs left running at this checkpoint.
- [self] Remaining priorities: parallel SCvx replay/metrics and profiling; CSR/structured forward gather instead of CSC atomics; recovery and convergence; GPU GTOC12 refinement; host QOCO conversion/device updates; device outer decisions and batching/graphs. Existing recovery still reports requested iteration_limit instead of actual PDHG count on non-cancelled completion, and canonical objective currently includes transposed dual gradient: inspect/fix before convergence cost-model tuning. New progress report docs/GPU_NATIVE_OPTIMIZATION_PROGRESS.md. Preserve earlier CPU Lambert/telemetry edits and pre-existing scratchpad changes.


### 2026-09-06 local GPU hill climb continuation
- Committed SCvx metrics/replay tranche as f876abb on perf/gpu-native-pipeline. Metrics across up to 128 blocks with retained FP64 reductions; exact HCW recurrence across six warp lanes, matrices reused. New internal headers scvx_metrics.cuh/hcw_replay.cuh and direct serial-oracle tests.
- Metrics-only H1 N=2000 SCvx 67.097 -> 54.769 ms. Combined metrics/replay 67.759 -> 41.305 ms; N=10000 326.879 -> 194.086 ms. Isolated HCW replay 10.5x; all computed states exactly match serial through 10k steps. Metrics maxima exact and sums within 3e-12 for four models with nonzero violations.
- Both new tests passed memcheck/synccheck/racecheck, plus full four-family production acceptance and CPU fingerprints. No tolerance changes. Raw artifacts in artifacts/performance/metrics*-comparison.json, standalone kernel logs and native-checks in build/performance.
- Found next dominant cost: standalone CQP residual API still used legacy one-block report after cooperative solve, and residual_seconds always zero. Added cooperative_residual_kernel, occupancy validation, retained scratch, residual CUDA timer, wait/poll collection, reset per solve. Extended cooperative_pdhg_test compares all residual fields at optimal and deliberately displaced infeasible resident points, preserves counters/allocation count. Normal and all three sanitizers passed. Full production passed max canonical 9.56640559e-9, nonlinear 2.92768846e-8, CPU/GPU trajectory 0, fingerprints exact. PD6 used one recovery vs earlier two; do not claim recovery speedup from that unpaired path change.
- Residual-only matched H1 N=2000 SCvx 41.456 -> 5.023 ms (8.25x), process ~429ms both (startup/harness dominates). N=10000 195.420 -> 13.437 ms (14.54x), process 868.729 -> 720.622 ms. Residual phase 0.159/0.187ms now reported, so CQP total accounting differs from old build. Final matched original-library benchmark is in progress; result gpu-native-hcw-2000-final.json.
- Immutable WSL libraries: baseline/ original e126..., before-hash/, before-metrics/ (769b422), before-replay/ (metrics only), before-residual/ (f876abb), all under /home/angus/build-spacepdhcg-gpu-native. Current build same directory. GPU lock /home/angus/.spacepdhcg-gpu.lock; do not touch Lambda campaigns.
- Next required work: recovery counters currently replace real PDHG count with iteration_limit at three evaluate_report calls (use saved pdhg_iterations), and report fixed 50000 recovery steps instead of completed_recovery_iterations. CQP objective in BOTH evaluate_report and grid_evaluate_report wrongly includes transpose-dual terms in 0.5*x*gradient; fix using Qx before dual addition, with analytic objective test. Then convergence/recovery parallelization, gather operators, GPU outer decisions, GTOC12 native GPU refinement, QOCO conversion, batching/graphs. Goal remains active and NOT complete.

- Residual tranche committed as 67b371d. Final matched original baseline study completed: process 5.172190 -> 0.423660 s (12.21x), SCvx 4.746820 -> 0.005080 s (~934x), already-feasible N=2000 HCW only, unchanged one inner/outer and zero residuals. Artifact gpu-native-hcw-2000-final.json. All local GPU jobs finished and lock released. GPU code clean; only previously preserved CPU/telemetry/memory tranche remains uncommitted. Next goal continuation should start with recovery telemetry and analytic objective correctness, then actual recovery/convergence optimization.


### 2026-09-06 recovery hill-climb checkpoint
- Commits c2028b4 (correct CQP objective and completed recovery counts) and d36c5c6 (bounded early GPU KKT refinement plus cached phase profiling) on perf/gpu-native-pipeline. No push. Preserve older CPU/Lambert/exporter and user memory edits.
- Objective now uses Qx before transpose-dual additions in BOTH evaluators. Regression failed before fix (-0.5 vs correct -0.6666666823) and passes afterward. Recovery retains actual PDHG count (300000) separately from actual completed PGD count rather than requested 1M/350k budget. Native/sanitizer/4-family tests pass; see docs/GPU_NATIVE_OPTIMIZATION_PROGRESS.md.
- Profile found >99.8% PD3 recovery in fixed 50000 projection loop. Discarded feasible-only, dual-only, periodic-certificate and momentum experiments: no qualification gain. Retained early full KKT refinement after 100 PGD steps: at most 4 primal corrections, each CGLS reconstruction 8 restarts of min(256,2*coefficients). Full finite objective/natural-residual gate at 0.9*requested tolerance. Failed probe restores primal, dual, and PGD step and continues original algorithm. Cancellation restores original PDHG state. Shared recovery_refine_primal helper, two retained trial buffers, cached public recovery_profile API (no extra read D2H/alloc).
- Matched RTX5090 PD3 N=2, alternating order, 1 warmup + 3 samples each: SCvx 17.775993 -> 5.246444 s (3.39x), recovery 13.041146 -> 0.043037 s (303x), process 18.148748 -> 5.628011 s (3.22x). Actual PDHG same300k, PGD50000->100. All inner residuals <=1.42323e-10 for1e-8 request; same objective .494783333; returned physics/trajectory differences0. No outer steps accepted; inner qualification checked explicitly. Raw artifacts/performance/recovery-bounded-refinement-pd3.json. Runner scripts/gpu/benchmark_recovery.py.
- Matched PD6 N=2 flat: SCvx48.472280->48.483291 s; recovery32.682183->32.648313 s, all measured600kPDHG+100kPGD. Measured inner results still UNQUALIFIED (~1.734e-4 vs1e-8), returned initial trajectory qualified/objective same. Optimized warmup happened to qualify after one recovery; DO NOT claim its shorter time as speedup. Last failed probe ~0.4% projection cycles (~67ms); previous32-refinement probe was more expensive, now bounded4. Raw recovery-bounded-refinement-pd6.json.
- Validation current d36c5c6: recovery_test, cooperative_pdhg_test, persistent CW/SOC, allocation, pointer and stream tests pass. Focused production PD3 early-success recovery_kernel passed memcheck/synccheck/racecheck (kernel filter kns=recovery_kernel; tools still slow PDHG, memcheck95s/racecheck78s). Four-family production with independent CPU fingerprints passes canonical9.56640559e-9, nonlinear2.9276885e-8, CPU/GPUtrajectory0, coefficient2.75994505e-13. Logs build/performance/native-checks/*bounded-recovery*. Runner build/performance/validate_bounded_recovery.py completed.
- Larger optimized --production-outer pd3 20 probe did NOT return output before120s process limit. Session11099 finished, no process remains; empty log bounded-recovery-pd3-20.log. No baseline at this size yet, no qualification/speedup claim. Next scaling work must instrument progress inside this solve rather than repeat opaque long timeouts. Try limited-iteration/direct CQP/progress controls to isolate PDHG vs recovery; do not assume N2 fast path scales.
- Current library/build /home/angus/build-spacepdhcg-gpu-native, nvcc12.8 sm120; d36c5c6 immutable .so copied to before-recovery-progress/. Correctness-fixed pre-refinement baseline before-recovery-fastpath/ (c2028b4); pre-correctness before-recovery-correctness/ (67b371d) has known bugs, do NOT use for matched quality counts/objectives. Other older snapshots retained. GPU lock /home/angus/.spacepdhcg-gpu.lock free at checkpoint, no GPU jobs active. Native source committed; old CPU/telemetry/memory changes remain untouched and uncommitted.
- Active goal remains incomplete, not blocked. Remaining: 300k-PDHG prelude (~5s even fastPD3), 6DOF convergence and largerPD3 scaling, serial recovery CGLS/cone reductions, gather/structured operators, device outer decisions, GTOC12 GPU-native refinement/search, GPU-resident QOCO conversion/backend, batching/graphs. device_scvx.cu still host forcing/acceptance loops around1527/2176/2648/2871. native_qoco_adapter.cpp convert() downloads arrays and host sparse assembly (~420..640), host residual evaluation~660..780, dynamic qoco API setup~872+ and updates~970+. Not fully GPU-native mission planner yet. Remote Lambda campaigns untouched.


### 2026-09-06 parallel recovery / medium dispatch checkpoint
- Committed 571d924 on perf/gpu-native-pipeline; no push. User goal remains ACTIVE/incomplete, previous goal turn made concrete progress. No local GPU process remains; lock verified free. Preserve unrelated CPU/Lambert/exporter/memory edits. Remote Lambda campaigns untouched.
- Six block-local cone projection call sites now distribute disjoint cones across threads; arithmetic order within a cone unchanged. Six CGLS squared norms in recovery_project_image/recovery_reconstruct_dual now use FP64 warp sums (serial <=32 elements), no extra alloc/copies. Existing barriers protect scratch. Kernels in persistent_pdhcg.cu.
- Added --production-cqp FAMILY INTERVALS ITERATIONS DEADLINE_SECONDS to device_scvx_integration_test (hcw/pd3/pd6/low-thrust): assembles actual production CQP, runs single solve with cooperative deadline cancellation, prints phase/work/residual metrics. Return0 means diagnostic completion, NOT qualification. With pre-change d36 library, pd3N20/350000/40 reached300kPDHG+5400PGD before40s cancellation. After parallel cones/norms, completed300k+50000 in34.623s, recovery17.008s, still unqualified; rollback natural residual exactly .00056897551530710189. Old projection loop was~1.6ms/step, new~0.34ms/step, but do not claim precise matched full-solve ratio from censored baseline.
- Found/fixed another telemetry bug: PDHG cancellation returned stale report from previous scheduled residual check. Both legacy/cooperative kernels now keep iteration loop index live and evaluate final resident point at cancellation, iterations=iteration-1. Sparse-check regression independently checks CPU objective of returned x and fresh GPU residual; old failed reported0 vs true1. New normal and all3 CUDA sanitizers pass, including zero-delay and legacy/cooperative cancellation. Logs cancelled-report-before/after and cancelled-report-{memcheck,synccheck,racecheck}.
- Fixed-work dispatch sweep, PD3 1000iterations,1warm+3 measured alternating: N5(var132) legacy21.00ms vs2blocks24.72ms; N10(var257) legacy34.82 vs2blocks28.59; N20(var507) legacy63.54 vs2blocks30.83; N50(var1257) legacy159.19 vs4blocks34.01/8blocks32.41ms. All objectives/residuals agree<1e-9 at fixed work (NOT qualified). Artifacts medium-dispatch-screen.json (N20), medium-dispatch-screen-{5,10,50}.json. Script build/performance/profile_medium_dispatch.py accepts optionalintervals.
- Automatic cooperative cutoff now256variables (was4096). Below256 keep legacy loop; parallel preamble stillstarts128. Above useceil(vars/256) blocks, with unchanged dense>=4096 target128 rule; occupancycap unchanged. Explicit overrides preserved. This exploits multiple SMs for medium problems where measured faster.
- Added focused displacedHCW support: --production-outer hcw 50 uses common12-outer-step protocol (minimum andmaximum12); 3steps insufficientlargerfixture,12passes. Same882inner and6accepted inALL matchedsamples. Compared solelydispatch against saved before-medium-dispatch library:2warm+7samplesalternating medianSCvx64.50266->38.969153ms (1.66x), process446.063->403.257ms (1.11x). Returnedcanonical~8.68e-9, terminal~1.3e-7, CPU/GPUtrajectory0, objective~1.33e-9. Artifact medium-dispatch-hcw-50.json. benchmark_recovery.py nowaccepts hcw.
- Matched PD3N2 against d36c5c6 (before-recovery-progress .so),1warm+5samples: SCvx5.115960->4.020861s (1.27x), recovery43.114->29.679ms (1.45x), process5.588654->4.405835s. Same300kPDHG+100PGD, objective.494783333, inner1e-8qualificationallpassed. Artifact parallel-recovery-pd3.json.
- New full --production-outer pd3 20 now returns in51.3512s vs prior120s timeout. It returns qualified unchanged reference with0accepted, objective.494483333, actual600kPDHG+100kPGD and STILL misses inner tolerance (5.58219448e-4 vs1e-8). Larger nonlinearoptimization is NOT solved; do not confuse fast failure/returning initial reference with convergence. Log parallel-medium-production-pd3-20.log.
- Validation: parallel-recovery/norm early-success and mixedcone kernels passed all3sanitizers; cancellationfix passedall3 aswell; nativecooperative,CW/SOC,recovery rollback/cancel/randomproperties,allocation,pointer,stream pass. Afterfinaldispatch policy, fresh cooperative test and full4-family with CPUfingerprints pass: canonical9.56640559e-9, nonlinear2.9276885e-8, trajectory0, coeff2.75994505e-13. NewdisplacedHCW50 passes. Logs prefixesparallel-recovery,parallel-cones,cancelled-report,medium-dispatch; runners validate_parallel_recovery.py and validate_cancelled_report.py completed. PD6sometimes1vs2recoveryattempts/qualifiedvsunqualified nearboundary; no robust same-work PD6speedup claim fromsmokes.
- Build /home/angus/build-spacepdhcg-gpu-native, current frozen library snapshot 571d924/libspacepdhcg_cuda.so. Other snapshots: parallel-cones-only (before normchanges), before-medium-dispatch (cones+norms+cancellationfix, old4096cutoff), before-recovery-progress=d36c5c6 (earlyrefinement), before-recovery-fastpath=c2028b4, baseline/original etc. Never rebuild or run anotherGPUtest while existingprocesslive. GPUlock /home/angus/.spacepdhcg-gpu.lock.
- NEXT priority: actual convergence of medium PD3/PD6, then remove remaining CPU numerical hot paths / whole GPU-native GTOC12. Read-only audit found existing GPU QOCO sources at /home/angus/spacecraft-trajectory-optmiser/_upstream/qoco-g4 HEAD09f049597deef2a7ead15b3da19a9456ff7d4e53; sharedlibrary /home/angus/spacecraft-trajectory-optmiser/build/qoco-g4/libqoco.so. It is CUDA backend percache and exports CUDAfunctions. ldd shows only libc/libm BECAUSE CUDA libraries deliberately dlopened; do NOT concludeCPUbackendfromlddalone. CUDAbackendcode has cuda_linalg.cu, cudss_backend.cu, cone.cu. cuDSSsymlink build/qoco-cudss-lib/libcudss.so -> /home/angus/spacecraft-trajectory-optmiser/.venv/lib/python3.12/site-packages/nvidia/cu12/lib/libcudss.so.0. Need LD_LIBRARY_PATH incl qoco-cudss-lib, /usr/local/cuda-12.8/lib64 and possibly nvidia/cu12/lib. No QOCO GPU test was launched this turn, only source/library inspection. Verify ABI and true GPU execution before benchmark; libraryloadpath may need CUDA JITsupportsm120. Do not modify pinned/sharedupstream orotherworktrees: copy/fork locally forports.
- Possible next concrete action: benchmark existing GPU-IPM on real medium cases using native --p1c-qoco-repeatability / --g4-sample afterreadingargument/setupcontracts, comparequality then port adapter conversion/dualhandback/verification toresidentCUDA. native_qoco_adapter.cpp currently downloads/rebuilds conversion and computes residuals onCPU; QOCOitself host qoco_api loopcontrolsGPUvector/cone/KKT operations and has scalar synchronizations; DO NOT label fullyGPUresidentbeforeauditing/removing numericalhostloops. C API native SettingsAbi/SolutionAbi are handwritten, checkpinnedheadersbeforeusinglibrary. Alternative follow-on GPUoperatorwork: retained GPU-built CSR gather maps for A/F/Q replacing atomic CSC scatter (also deterministic accumulation), plus fuse cooperative residual reductions tocutgridbarriers. Full GPU outerdecisions, GTOC12 refinement/search, batching/graphs remain outstanding.


### 2026-09-06 GPU QOCO checkpoint: 0b3c130
- ACTIVE GOAL remains incomplete. This turn made concrete local progress; NOT blocked. User wants production numerics GPU-native C++/CUDA, unchanged physics, local RTX5090 hill climbing. No subagents (not authorized), no pushes, no remote Lambda activity. Preserve unrelated Lambert/exporter/CPU/memory changes.
- Committed 0b3c130: optional QOCO scoped cuBLAS handles in native_qoco_adapter.cpp; source-preparation script and GPU sparse gather patch; stopping-test bug fix; checked cuDSS ABI upgrade support; focused operator tests; benchmark/evidence/docs. Production backend selection unchanged, candidate NOT auto-promoted because vendor sanitizer issue remains.
- Native core build /home/angus/build-spacepdhcg-gpu-native includes adapter extension and optional CLI repeat count (--p1c-qoco-repeatability [positive count], default7). Optional paired dlsym begin/end callbacks; RAII scope closes after normal solve and cold retry. Original libraries with neither callback still supported. Each scope owns thread/device-specific cuBLAS handle; no TLS destructor or handle retained past solve. Three upstream reduction sites used to create/destroy handle every call. Full QOCO backend remains host controlled/numerical; DO NOT say100%GPUresident.
- Source prep scripts/gpu/prepare_qoco_gpu.py requires clean PIN09f049597deef2a7ead15b3da19a9456ff7d4e53 at /home/angus/spacecraft-trajectory-optmiser/_upstream/qoco-g4. Copies tracked files to NEW isolated tree, excludes benchmark submodule. CMake writes qoco_config.h INSOURCE; NEVER configure shared pinned source. --gather --correct-stopping --deterministic --checked-cudss-abi used for final candidate. --original-handles for control, --unmodified for completely original source. All builds FP64/int32/CUDA12.8/sm120/Release. Pinned checkout verified clean after work.
- qoco_gather.cuh builds retained CSR row maps onGPU with CUB stable radix sort; row offsets binarysearch sortedkeys, entrymap reads originalCSCnumericvalues. Forward/symmetric rowgather eliminates atomics; symmetric gathers mirrored offdiagonals fromsameCSCcolumn; transpose now256threads/block insteadone. Refreshmaps on sync_matrix_to_device, retainedsortscratch avoids reallocation. QOCOMatrix extension opaque andallCUDAtranslationunitsrebuilt. Tests cover rectangular, empty, null matrix, duplicates, symmetric diagonal/offdiagonal, values/topology refresh, bitwiseproductrepeats, nestedscopes, unscopedreductionsaftercleanup. Updatedtestcompiles -Wall -Wextra -Werror standalone, upstream include+lib/qdldl/include (notinmainCMake, docscommand).
- Found actual stopping bug upstreamsrc/qoco_utils.c check_stopping: cinf used norm(Dinv*x), should norm(Dinv*c). Regression injects feasible x=1e9 withstationarity.5 andmustNOTaccept; original accepted, correctedrejects. --stopping-only test failsoriginal1, passcorrected inclmemcheck. Original GPU IPM wasflaky: somefreshP1C processes rejectfirststep(inner~2e-8) thenmiss terminal1e-8 in2outerbudget. Savedfailures in qoco-scoped-handles-pd3.json and qoco-gather-scopes-correct-stopping-pd3.json. Correctingstoppingalone+gathersNOTenough; cuDSS deterministicmode givesstablefull54inner (17+37) and2acceptedforfixture.
- MATCHED MEASUREMENT artifacts/performance/qoco-deterministic-scoped-handles-pd3.json:2warm+7measured pervariant alternatingfreshprocesses. Control/optimized BOTHgathers+correctstopping+deterministiccuDSS0.7.1.6; onlyhandle reuse differs. Medians SCvx1.474888457 ->.623454312s(2.36567x), QOCOsolve1.436293976 ->.58302318s(2.46353x), process1.929623256 ->1.045398076s(1.84583x). ALL54inner/2accepted, objective.49448537334898213, canonical8.5196031142981985e-12, terminal5.735494440495259e-12. Quality1e-8unchanged. P1-C20intervaldisplacedlanding, DIFFERENTfromprior--production-outerpd3referencefixture; NOcrossfixture51s/.6speedclaim. trajectory_difference476meansdisplacementfromreference, NOTCPU/GPUparity.
- API profiles SAME54inner/2accepted: cudaMalloc5752->1135, cudaFree7059->903, cudaDeviceSynchronize11965->5809, kernel launches12263both. /usr/local/bin/nsys2024.6.2 onlyCUDAAPItraceworks onRTX5090; GPUkernel/memtimelineNOTcaptured. Profiles /home/angus/build-spacepdhcg-gpu-native/qoco-{control,scoped}-deterministic-profile.{nsys-rep,sqlite}. CSVbuild/performance/qoco-{control,scoped}-deterministic-api.csv. benchmark/provenance+summary artifacts/performance/qoco-gpu-checkpoint.json. Helperbuild/performance/profile_qoco_scopes.py finished.
- FinalQOCOcandidate /home/angus/build-qoco-gpu-gather-v9/build/libqoco.so, source sibling. Uses0.7.1.6andcheckedABI. Final7repeatandnative memcheckpasssame54/objective/residualasv4. Operatorall4sanitizers(memcheck/initcheck/synccheck/racecheck), stoppingmemcheckpass, noleaks. Nativeunavailablecontractpass. 6DOF--qoco-handbackcontractpass BUT candidateREJECTED, no6DOFconvergenceclaim (wholemodealsoexecutesfixedtightPDHG~45s andreturnsunqualified; qoco_workspace_creations0inthatouterrecord).
- IMPORTANT OPEN VENDOR ISSUE: Full landingracecheckcuDSS0.7.1.6deterministicfactorize_dtmn_ker reports57hazardsunderCUDA12.8 (runfinishesqualified54butexit91,95.7sinstrumented). CurrentCUDA13.2sanitizer also reportshazards infullrun (returned1withoutcleanfinalsummary). Firstfactorizationkernel-onlyfilteredcheck13.2passes, NOTproofwholekernel/fullrunpasses. DO NOT suppress or callfullbackend racecheckclean. Newoperators themselvespassracecheck. Logs qoco-v4-landing-racecheck.log, qoco-v4-full-racecheck-13.2.log, qoco-v4-cudss-racecheck-13.2.log. Initialinitcheck/memcheck/syncchecknativepassedall. NeedresolvevendororusealternativeGPUfactorizationbeforeproductionpromotion.
- Isolatedcurrent sanitizer downloadedofficialaptcuda-sanitizer-13-2=13.2.87-1, dpkg-deb extractedONLY(noinstalledtoolkitmodified) /home/angus/tools/compute-sanitizer-13.2/extracted/usr/local/cuda-13.2/compute-sanitizer/compute-sanitizer version2026.1.1.0. PackageSHA8899b0126b75be7ff0df6b11b685312b180c5928f567c67f1e4962d4df6b4aad. Only/usr/local/cuda12.8compilerinstalled.
- IsolatedcuDSS0.8.0.10 pip--target /home/angus/tools/cudss-0.8.0.10, lib nvidia/cu12/lib/libcudss.so.0 withnewlibcudss.so symlink, headers nvidia/cu12/include. REJECTED upgrade: oldhardcodedfunctiontablecompiledagainstnewheadersbutCSRsignaturechanged(missingoffsetType/newdatatype) ->INVALID_VALUE. Added --checked-cudss-abi: derive12funcpointertypesdecltypeofficialheaders, conditional>=800 newtypes+extraoffsettype, runtimeGetPropertymajor/minormatchguardBEFOREcalls. Newversionthenhitactualinvalidsharedread incudss::fwd_dtmn_ker, memcheck13.2reports3errors andnativefails; superpanelsenabledalsofails. v7andv8librariesareREJECTEDprobes, NOTfinalcandidate. runtimeguard0.8header/0.7libandreversebothrejectgracefullybeforemismatchedcalls (nativeAPIexit3insubprocess, toolreported1inotherinvocations); testassertlogmessage+nonzeronotassumeexactexit1. Neednotredo failed0.8asdrop-in; NVIDIA0.8migrationguideexplainsAPIchanges.
- FrozenQOCOdirectories: /home/angus/build-qoco-gpu-control originalrebuilt; build-qoco-gpu-native scopedhandlesonly; build-qoco-gpu-gather-v2 gathers+scopesnostopfix; v3+stopfixnondeterministic; v4+deterministic (MEASUREDoptimized), build-qoco-gpu-deterministic-control gathers+stopfix+deterministicoriginalhandles (MEASUREDcontrol); build-qoco-gpu-correct-control stopfixonlystillflaky; v5rejected0.8oldABI; v6partialpreparationfailed(no build); v7checkedAPI0.8failsGPU; v8+superpanelsalsofails; v9 FINALcheckedABI0.7otherwisev4. Preserveallrawmeasurements/failures. No modelkernels/ABIprecisionchangedtoobtain speedup.
- Buildcommands/docs docs/GPU_QOCO_LOCAL_BACKEND.md; docs/GPU_NATIVE_OPTIMIZATION_PROGRESS.md appendedcheckpoint. NativecoreC++ rebuiltbeforebenchmarks. Ruffpassesnew2Pythonscripts, Windowgitdiff--checkpasses. A WSLgitdiff--check onWindowsworkingtree was mistakenlyrunandemittedmassiveCRLFfalsepositives; terminatedonlythatread-onlygitprocess. UseWINDOWSgitforWindowscheckout diffs/status; WSLgit okayforrevparse/pureWSLpin. Do notmass-normalize unrelatedfiles.
- Validationhelpers build/performance/validate_qoco_extensions.py stoppedatEXPECTEDOPENnativevendor racecheck91beforehandback; handbacksubsequentlyrunseparatelypassed. validate_qoco_final.py completedpositivechecks, thenstoppedbecauseharnessincorrectlyexpectedmismatchexit1vsactual3; loghascorrectexplicitmismatch, standaloneunavailablethenpassedseparately. This isnotanunresolvedloaderfailure. Summaryrecordbuild/performance/record_qoco_checkpoint.py validates logmarkers/0errors, recordsopenfindingsandhashes. Do notrerunexpensivevendorfullracecheckswithoutnewreason. Currenttesthelper maystillhaveexpected1 formismatch; fixifyoureuse.
- NEXT: resolve/sidestepcuDSSdeterministicvendorhazards whilekeepingfullquality; actualmedium/largePD3/PD6convergence; remove remainingCPU QOCOconversion/scaling/KKT assembly/dualhandback/residual loops. Profilingstillshows5809device syncs/2188blockingcopies: scopeddefaultstreamvector kernel ordering coulddeferperopsync(withcheckedscopecompletion), butNOTimplementedorvalidatedyet. NativeouterdevicecontrolandGPU GTOC12refinement/search/batchingremain. Couldportretainedgathers tocorePDHG too. Goalnot100%GPUresident, notcomplete.

## Local GPU audit checkpoint ea25de0 (2026-09-06)
- Active goal remains ACTIVE and incomplete. No subagents authorized. Branch perf/gpu-native-pipeline HEAD ea25de0, no push. Preserve unrelated dirty files/remote Lambda campaigns. Actual WSL distro is Ubuntu-22.04 (not Ubuntu). Windows git for Windows checkout.
- Added cpp/cuda/internal/native_qoco_gpu.h and src/native_qoco_gpu.cu: fixed CSC->CSR gathers built on CUDA/CUB once; retained numeric buffers; GPU KKT primal/dual/cone/complementarity/objective normalization; parallel row/variable/cone tasks and deterministic reductions; direct GPU canonical dual mapping. Six audit doubles (48 bytes) downloaded. Nonfinite inputs/intermediates reject. All double precision, same normalization/tolerances. Values-only updates allocate nothing. Tracks audit-owned allocations/current/peak and transfers.
- Native adapter creates audit workspace + immutable linear dual transform, checks row mapping equality on updates, uses GPU audit/map in production. CPU residual/map retained ONLY behind SPACEPDHCG_TEST_QOCO_GPU_AUDIT_COMPARE=1. Optional backend qoco_gpu_get_solution returns borrowed completed unscaled device x/y/z, adapter D2D primal. Older backend without export uses retained host-solution staging uploads but same GPU audit. Backend itself STILL downloads solution; adapter STILL stores host primal for accepted warm state. CPU conversion/scaling/KKT/upstream scalar reductions/outer decisions remain. Do NOT call pipeline fully GPU resident.
- New cpp/cuda/patches/qoco_device_solution.cuh automatically included in modified prepare_qoco_gpu.py output (unmodified control preserved). Final backend /home/angus/build-qoco-gpu-gather-v10/build/libqoco.so and source sibling: gather + corrected stopping + deterministic + checked cuDSS ABI + new device-view export, CUDA12.8 FP64 sm120 int32 cuDSS0.7.1.6. v9 is legacy-no-view compatibility control. Preserve all v4-v10 experiments. Pinned upstream untouched.
- CUDA build /home/angus/build-spacepdhcg-gpu-native current core includes ea25de0. Frozen core snapshot /home/angus/build-spacepdhcg-gpu-native/ea25de0/libspacepdhcg_cuda.so. Baseline snapshot0b3c130 before GPU audit preserved. Full ninja -j3 succeeded (qoco-audit-full-build.log). Fixed prior CMake regression: automatic *_test.cu glob now excludes standalone qoco_gpu_operators_test which needs separate upstream includes/library.
- New qoco_gpu_audit_test compares independent dense long-double reference, empty equality/cones, duplicate entries, n1/7/17/513, SOC4/8/3/5/32/257, numeric updates, exact repeated outputs, general dual transforms, nonblocking stream, NaN x/Inf P rejection, and no allocations during updates/runs. Normal and all4 CUDA12.8 sanitizers PASS, memcheck leak0; logs build/performance/native-checks/qoco-audit-{memcheck,initcheck,synccheck,racecheck}.log.
- Real landing with CPU audit+map oracle: --p1c-qoco-repeatability 7 passed ALL54inner (17+37),2accepted, objective.49448537334898213, canonical8.5196031142981856e-12, terminal5.7354944404952589e-12,1e-8qualification. Full landing memcheck+synccheck (oracle enabled) PASS, leaks0. Legacy v9 backend fallback also passedoracle. Existing --qoco-unavailable passes. Existing --qoco-handback passes but is synthetic handback after unqualified PD6 PDHG, does NOT exercise new audit/convergedPD6. It rejects candidate, no PD6 convergence claim. Logs qoco-audit-oracle-repeat7.{jsonl,err}, qoco-audit-landing-{memcheck,synccheck}.log, qoco-audit-legacy-view.{jsonl,err}, qoco-audit-pd6-handback.{jsonl,err}.
- Matched freshprocess benchmark SAMEv10QOCObackend, baseline core0b3c130 vs current,2warm+7measuredalternating: SCvx median.660848182 -> .668247346s (1.1% SLOWER); process1.068991482 ->1.112926476s (4.1% SLOWER). All54inner,2accepted,samephysics. No additional speedup established; change removes CPU numerical work, large audit scaling unmeasured. Artifact artifacts/performance/qoco-gpu-audit-pd3.json; benchmark_qoco.py supports --baseline-core/--optimized-core now, hashes both and prepends LD_LIBRARY_PATH. Prior2.37x scoped-handle improvement remains separate validated result.
- Memory telemetry: driver allocation bytes now include audit peak; G4 export peak_device_bytes sums workspace peak+driver/audit peak and declares device_memory_scope=native_owned_peak_upper_bound_excludes_qoco_cudss. This is upper bound for known allocations NOT total GPUmemory; opaque upstream excluded. Corrected landing sample506880bytes vs old187660partialworkspace only. Benchmark predates final exporter fix; do NOT use its oldmemoryfield as wholepipeline memory. Later qoco-audit-accounted.jsonl validates corrected scope/count withsamephysics. Adapter copy counters exclude opaqueQOCOinternalcopies. Audit residualtime included pureQoco and delta added hybrid.
- Evidence summary artifacts/performance/qoco-gpu-audit-checkpoint.json includes SHA256, validations, corrected sample, remaining work. Generated by ignored build/performance/record_qoco_audit.py which assertslogmarkers/qualification. docs/GPU_QOCO_LOCAL_BACKEND.md and GPU_NATIVE_OPTIMIZATION_PROGRESS.md updated. Ruff2scripts and gitdiffcheck pass. No live GPU jobs/builds at checkpoint.
- OPEN vendor race issue unchanged: full cuDSS0.7.1.6 deterministicfactorization racecheck hazards previously reproduced12.8+13.2. Newaudit kernelsall4clean does NOT clear fullbackend. cuDSS0.8upgrade remains rejected invalidsharedread. Optional GPUIPM not auto-promoted.
- NEXT: optimize/remove CPU conversion/scaling/KKT and upstream per-vector device synchronizations (5809device syncs in priorprofile), hostsolution/warmstate transfers; validate largePD3/PD6 actual convergence and audit scaling; consider alternative factorization to resolve cuDSSrace; deviceouterdecisions/GPU GTOC12search+refinement/batching still outstanding. Reuse GPUlock /home/angus/.spacepdhcg-gpu.lock; no simultaneous GPUmeasurements; do not rebuild a live .so. Runtime LD_LIBRARY_PATH=/home/angus/spacecraft-trajectory-optmiser/build/qoco-cudss-lib:/usr/local/cuda-12.8/lib64.

## Queued operators / device cone reductions checkpoint c890e2e (2026-09-06)
- Previous goal turn ea25de0 was PROGRESS; this turn also PROGRESS (implementation, regressions, measurements). Full goal stays ACTIVE. Branch perf/gpu-native-pipeline HEAD c890e2e, no push, no subagents. Existing unrelated dirty work preserved. No changes to CUDA core .so this turn; current core and saved ea25de0 snapshot remain identical. Optional QOCO source copies/backends changed only in isolated /home/angus/build-qoco-gpu-* trees.
- prepare_qoco_gpu.py new --queued-operators requires --gather and scoped handles. Replaces6vector/transpose deviceSync sites +gather product completion with helper that skips device-wide waits INSIDE solve scope. Same defaultstream ordering; CPUscalarreads,setup/refresh,cuDSS sync remain. Outermost end checks cudaStreamSynchronize(nullptr) and freesresources; unscoped calls keep synchronous semantics. New dependent chain unit tests with inplaceew/axpy/scale/copy/forward+transpose andeventquery beforedownload verify nestedscopes andcompletion.
- --device-cone-reductions enables new qoco_device_cone_reductions.cuh and scope-owned growable GPUscalar workspace. LPstep staysondevice, SOCstep usesdeviceLPstep, reductions+stepsafeguard/factorGPU, only1scalarD2H perlinesearch. cone_residualscratchretained acrosscalls; allfreedatoutermostscope. Removeshostblockreductions and per-call allocation. Standaloneunscoped calls temporaryscratch preservecompatibility.
- Found/fixed real cone residual BUG: original cone_residual_stage2 capped1024threads and read onlyonepartial/thread, omitted blocks>=1024. New gridstridereductioncoversall. Independentcase262145LPentries,lastx=-7 =>oldmisses violation, newresidual7. beforelog qoco-v11-cones-before.log expectedFAIL.
- Found/fixed SOC line search BUGS: original ifabs(a)<1e-14 returnsalpha ignoringlinear bconstraint; c==0 branch checkedonlya leading infeasible stepsorunnecessarystalls. New a==0 handlesb<0 limit-c/b; c==0 handlesb<0 or(b==0,a<0)->0, elsea<0 limit-b/a. Existingstablequadraticrootformulaotherwise. x=(1,0,0),dx=(-1,1,0) requiresmax.5(old1). Six independentlongdoublefeasibilitybisection cases: exactlinear, boundarya>0negativeb, boundarynegativea, feasibleboundarylinear, nearlineara~1e-15, boundarybpositivea<0. Oldv13 fails--cone-boundaries-only. Finalallpass. This correction changes actual landing convergence54->28inner, NOTjusttiming.
- Final backend /home/angus/build-qoco-gpu-cones-v14/{source,build/libqoco.so}, all flags --gather --correct-stopping --deterministic --checked-cudss-abi --queued-operators --device-cone-reductions, FP64int32sm120CUDA12.8cuDSS0.7.1.6. v10 baselineGPUauditdeviceview/noqueues/conereductions; v11 build-qoco-gpu-queued-v11 queueonly; v12 build-qoco-gpu-cones-v12 FAILEDcompilemissingcone_residualforwarddecl +badmanualprototypelocation (do notuse); v13 cones+scratch+largeresidualfix BEFORE SOCboundarycorrection; v14 finalcorrection. v15 samev14butNOdeterministicflag REJECTED: firstlanding failed2accepted/1e-8gate, only1accepted, terminal5.4844079940608025e-8,48inner. Planned15repeatsstoppedfirstfailure; doNOTclaim15tested. Log qoco-v15-repeat15.{jsonl,err}.
- Matchedbenchmarks all2warm+7measuredalternatingfreshprocesses: queueonlyv10->v11SCvx.623111270->.495659729s1.257x (same54inner); cones/scratchv11->v13.536114361->.486666269s1.102x(same54inner). Combinedv10->FINALv14 20interval1e-8landing: .617516826->.242203437s2.5496x, process1.012821374->.653366790s1.550x, QOCOsolve.574331233->.197070385s2.914x; 54->28inner(17+11candidate),2acceptedall. Candidateobjective.49448537365291756 vscontrol.49448537334898213 (<1e-8difference), canonical2.618510687647072e-11,terminal7.034629823099436e-13. SevenextraCPUaudit-oraclerepeatsallqualified. Artifacts qoco-queued-operators-pd3.json, qoco-device-cone-reductions-pd3.json, qoco-queued-device-cones-pd3.json.
- Actual native planner examples also now MEASURED (notjustsynthetichandback). NativeC++ expectsCANONICALUNITS. Initialdirectexamples/planner/powered_descent_6dof.json failedvalidationmaximumtilt30radians; normalizing via PYTHONPATH=src /home/angus/worktrees/spacepdhcg-literature-venv/bin/python -m spacepdhcg.planner.cli validate FILE writes canonicaldoc. This isunitnormalizationoutsidebenchmark, NOTsolverCPUnumerics. Inputs build/performance/planner-{pd3,pd6}-canonical.json; solverexe /home/angus/build-spacepdhcg-gpu-native/cuda-tools/spacepdhcg_plan. Native solve only C++/CUDA plus independent host replay, QOCOremainingCPUloopsstillpresent.
- New scripts/gpu/benchmark_planner.py accepts --executable --problem CANONICAL.json --baseline LIBQOCO --optimized LIBQOCO --output JSON;2warm7default, GPUlock, newfullresults.samplesdir, SHA256, allcertificategates+continuoustime+objectivewithin1e-8, nativeprocess timing includesindependent replay, noCPUfallback. Doesnotrequireequaliterations. Rawfulloutputsretainedunder artifacts/performance/qoco-queued-planner-pd{3,6}.samples (ignored). Summaryreports+representativebaseline/optimizedmeasuredsampleJSONs committed.
- Actual40interval3DOF150msoftlanding: v10->v14SCvx.638398554->.426663518s1.496x,process.952517775->.750726067s1.269x,44->41inner,2acceptedall,1e-6physicsgateallcertified. Actual20interval6DOFbraking/hover: SCvx2.045458191->1.684073953s1.215x,process2.452120331->2.059881728s1.190x,162->179inner,5->6outerattempts,2acceptedboth,1e-6gateallcertified. Candidate6DOFobjective.5129756912893927 vscontrol.5129756912891978,canonical1.04707785e-11,terminal8.15236767e-15,CPU/devicecoefficient5.29e-17. No universal speedupclaim; completeobjective andphysics comparisons inreport.
- Finalv14cone unit all4sanitizers PASS (logs qoco-v14-cones-final-{memcheck,initcheck,synccheck,racecheck}.log), zero leaks. MixedLP/SOC counts8193/1027, SOCsizes3/5/33,empty,262145largeLP,all6boundarydirections,scoped/unscopedrepeated. V11operatorchains all4pass; finalv14operator+stoppingmemcheckpass. Fullv14landing memcheckzero leaks/initcheck/synccheckpass(CPUauditflag1). Fullracecheck REPRODUCESOPENcuDSSfactorize_dtmn_kerissue:30hazardsdisplayed(errors30,warnings0),28iterpointstillqualified; toolnonzero. Logqoco-v14-landing-racecheck.log. It is NOTfullysanitizerqualified andNOTpromoted. v15nondeterministicmodefailedphysics, notacceptableworkaround.
- New APIprofile baselinev10vsfinalv14: deviceSync5809->65,cudaMalloc1183->381,cudaFree951->301,kernelLaunch12287->6073,cudaMemcpy2188->981. ALSO54->28iterations, so countscombineoperatorandconvergenceeffects. Nsight2024.6.2 API ONLY,noRTX5090GPUtimeline. /home/angus/build-spacepdhcg-gpu-native/qoco-{baseline,queued-cones}-queued-profile.{nsys-rep,sqlite}; CSV build/performance/qoco-{baseline,queued-cones}-queued-api.csv; committedqoco-queued-api-profile.json. Helperprofile_qoco_queued.py initiallyincorrectlyasserted54candidate;fixedrecordactualcounts, reusedexistingreports(noGPUreruns). Nsystats needed--force-export=true becauseSQLiteolderthaninputmtime; helperregeneratesderivedSQLiteonly.
- Summary artifacts/performance/qoco-queued-checkpoint.json validatesall5pairedbenchmarks,representativeplannerresultshashes,positive/negativeunitlogs,fullvendorfindingsandfailedv15, recordsfinalsourceprovenance/runtime/corebackendSHA. Generated ignored build/performance/record_qoco_queued.py. DocsGPU_QOCO_LOCAL_BACKEND.md andGPU_NATIVE_OPTIMIZATION_PROGRESS.md updated. Ruffprepare+newbenchmark andgitdiffcheckpassed. Commitc890e2e18files. No liveGPUtests atcheckpoint; core ea25de0 unchanged.
- NEXT: needresolve/sidestepcuDSSvendorhazards with independentlyqualifiedGPUfactorization (simplynondeterministicfails;0.8upgradepreviouslyinvalidread). RemoveCPUconversion/equilibration/KKTsetup+upstreamscalar/outerdecisions+warmstatehostroundtrips. Actualmedium/large nonlinear scaling and offnominal6DOFstillneedcoverage; current6DOFexamplealignedbrakingnotrotationalmaneuver. GPU GTOC12search/refinement/batching/graphs remain. TheuserwholeGPUgoalnotcomplete. UseGPUlock /home/angus/.spacepdhcg-gpu.lock, preserve all measured binaries, WSLUbuntu-22.04, runtimeLD_LIBRARY_PATH=/home/angus/spacecraft-trajectory-optmiser/build/qoco-cudss-lib:/usr/local/cuda-12.8/lib64.

## Local retained topology checkpoint d35fe7e (2026-09-06)
- ACTIVE goal, incomplete, PROGRESS this turn. No subagents authorized, no push; remote Lambda campaigns and shared pinned QOCO untouched. Branch perf/gpu-native-pipeline HEAD d35fe7e. Preserve existing unrelated dirty files. Actual WSL distro Ubuntu-22.04, CUDA12.8, local5090; Windows git for checkout (WSL git diff creates CRLF noise).
- Native adapter caches six host sparse index arrays once. QocoGpuTopology retains device copies; subsequent exact parallel integer compare downloads one mismatch int, detects changed data with same pointers/fingerprint. Numeric D2H calls now queue in one DownloadBatch with one stream sync and RAII destination lifetime protection. This STILL rebuilds CPU numeric rows/CSC/cone maps every update: do not call GPU conversion complete. Adds native cache memory/transfer accounting to audit stats.
- prepare_qoco_gpu.py --values-only-updates patches qoco_update_matrix_data P/A/G to upload values only, preserving device indices and gather maps. Full public sync_matrix_to_device still handles real topology changes. Tests use new symbol for values-only then full topology mutations against long-double dense arithmetic. Final optional backend /home/angus/build-qoco-gpu-values-v16/{source,build/libqoco.so}, built with v14 flags plus values-only. It remains opt-in, not default production promotion. Reprepared /home/angus/build-qoco-gpu-values-repro/source hashes exactly match measured prepared source.
- Added topology tests: empty arrays, nonblocking stream, counts 0/1/257/513/4097/131073, in-place last-entry mutation, changed pointer/equal data, wrong dimensions, no validation allocations, one 4-byte D2H flag per call. All4 sanitizers clean. v16 operators all4 sanitizers clean; full landing memcheck(no leaks)/initcheck/synccheck clean with CPU-audit oracle. Seven repeated CPU-oracle landing solves all same28inner/2accepted/1e-8qual. Known cuDSS0.7.1.6 deterministic vendor race remains OPEN from v14 (30 displayed); not re-run this turn.
- Matched core ea25de0+backendv14 vs newcore+v16, 2warm7measured alternating fresh processes: landing SCvx .260055554 -> .255065742s (~1.02x), process .644094460 -> .652263540s (~.987x); actual20intervalPD6 SCvx1.662863467 ->1.657907972s(~1.003x), process2.027960160 ->2.046949575s(~.991x),179inner both. Same physics/accepted steps. NO clear incremental runtime improvement; record as topology data movement reduction, not speedup. Native landing D2H137504->118172 bytes, H2D116552->135888 (initial cache upload), native peak506880->526220. Opaque QOCO/cuDSS not included. CPU-oracle D2H larger131676; production118172 is correct.
- New benchmark_planner.py --baseline-core / --optimized-core supports paired core changes, SHA-256 hashes and LD_LIBRARY_PATH; readelf confirms executable RUNPATH overridden by loader. Evidence committed artifacts/performance/qoco-cached-topology-{checkpoint,pd3,planner-pd6,planner-pd6-baseline-sample,planner-pd6-optimized-sample}.json. Full samples ignored .samples directory. Docs GPU_QOCO_LOCAL_BACKEND.md and GPU_NATIVE_OPTIMIZATION_PROGRESS.md updated. Ruff and git diff check pass.
- Rejected v17 /home/angus/build-qoco-gpu-margin-v17: v16 without deterministic cuDSS, temporarily multiply accurate abstol/reltol by .1 inside qoco_solve wrapper (restore caller settings). Ten samples qualified, sample11 failed, stopped before requested15. Failed first inner natural2.338862291e-7, final terminal5.484484238e-8 >1e-8,53inner/1accepted. NOqualifiedrepeat15claim. Experimental --stopping-margin flag REMOVED after failed probe. Prepared source+binary preserved, wrapper/provenance/all11samples+failediterations recorded checkpoint. Earlier v15 withoutmargin failedfirst; nondeterministic mode not valid workaround. Do not retry same factorization casually or loosen gates.
- Frozen native core/planner/tests at /home/angus/build-spacepdhcg-gpu-native/d35fe7e. Current build cuda/libspacepdhcg_cuda.so same measured hash890c36659bbfd4f05c46416b17fa10b1fa17ebce11c2023727b7644ee4c7c1e8. v16backendhash5ed6531d1988c0104ff72f548ce2aa1bfe83b2b05b571f08be77f825a1762533. Full source/build code unchanged after measured checks except private header comment. GPU jobs done; sessions8799/85907/77038 closed. Helpers build/performance/check_qoco_cache.py,record_qoco_cache.py,freeze_qoco_cache.py (freeze target exists; don'trerun) are ignored. Raw logs build/performance/native-checks/qoco-topology-*,qoco-v16-*,qoco-v17-repeat15.*.
- NEXT: compile immutable numeric conversion maps once and evaluate values/bounds/SOC transforms on CUDA, preserve checks for changed bound classifications/cone layouts/symmetry, independent CPU oracle. Then GPU equilibration/KKT assembly and outer/warm decisions. Current convert in native_qoco_adapter.cpp (~460-800) downloads9numericarrays and rebuilds triplets/rows/unorderedmap/make_csc; cached6indices alone did not improve runtime. Direct preparedQOCOdevice update API likely needed to avoid roundtrip/reupload; do not pretendCPU setup/scaling has moved. Major remaining vendorfactorization, nonlinear largePD3/rotationalPD6, trajectorystructureoperators/batching/graphs, nativeGPU GTOC12. Goal stays active, not complete or blocked.

## Compiled CUDA conversion checkpoint 4556941 (2026-09-06)
- ACTIVE goal remains incomplete; previous/current turn PROGRESS. HEAD4556941 perf/gpu-native-pipeline, no push, unrelated dirty files and remote campaigns preserved; no subagents authorized. d35fe7e remains baseline snapshot. GPU checks/benchmarks all complete, no live job left.
- native_qoco_adapter.cpp now compiles fixed numerical mappings once (compile_conversion) from retained CSC topology and Formulation row maps. Maps cover P/A/G/c/b/h, duplicate sparse entry order, scalar/box signs/equalities, affine and variable ordinary SOC and rotated SOC transforms. Initial CPU convert discovers structure, compiles maps then refresh_conversion evaluates CUDA before setup. Every repeated update calls refresh_conversion: no CPU triplet/row construction, sort, quadratic symmetry scan or coefficient arithmetic. CPU converter retained as SPACEPDHCG_TEST_QOCO_GPU_CONVERSION_COMPARE=1 oracle (8*epsilon scaled per entry + pattern/maps exact); absent in production.
- QocoGpuConversion in native_qoco_gpu.{h,cu}: plan offsets/terms/boundtypes/symmetry pairs retained, parallel ordered gathers with explicit __dmul_rn/__dadd_rn, input finite/bound-kind checks, symmetric quadratic numerical checks. Nine canonical input pointers on declared stream, six packed output groups resident; outputs + validation int D2H because QOCO still host API. qoco_gpu_conversion_values exposes retained device output. New qoco_gpu_audit_update_device copies packed values D2D into audit and avoids H2D reupload on updates; adapter accounts D2D copies. Audit/conversion/topology allocations/transfers included native upper-bound telemetry. Initial compilation timing included in conversion_seconds; numeric-update phase includes GPUrefresh/upstreamupdate.
- Exact cached sparse validation extracted validate_cached_topology used both CPU/GPU conversion paths. Cone kind/start/vector_dimension compared against retained host metadata; bounds checked on GPU (free/upper/lower/both/equality), NaNs/infinities outsidebounds or transformed nonfinite reject, quadratic symmetry same1e-12 relative as old CPU dictionary (last duplicate per coordinate). Invalid conversion does not mutate accepted Formulation/QOCO/audit values; nextvalidupdate recovers. Failure numerical now reported NUMERICAL, unsupported symmetry UNSUPPORTED, changed layout TOPOLOGY_MISMATCH.
- New qoco_gpu_conversion_test.cu: outputs0/7/131075, input131073 grid-stride tail, long-double independent expressions, fixed floating sum cancellation order, resident data equality, changing values at same bound type, changed finite/equality kinds, asymmetry, unused NaN/nonfinite transformed output, restored workspace, nullinput rejection, no allocationdelta. All4 sanitizers pass. New native_qoco_conversion_test.cu: optional QOCO env needed (otherwise explicit SKIP), n8 mixedscalar/box/affine+variable SOC/RSOC with duplicate CSC entries and byte-offset views; CPU conversion+KKT oracles; initial/update/restored solves succeed, rejects new bounds/cone metadata/in-place indices/asymmetricQ/NaNF with correct flags. Native synthetic memcheck(no leaks),initcheck,synccheck pass. No full vendor race rerun; known cuDSS0.7.1.6 deterministic30hazards remainsopen.
- Real landing7repeat passes BOTH CPU oracles same28inner2accepted tol1e-8 objective.49448537365291756 canonical2.618510687647072e-11 terminal7.034629823099436e-13. Actual20intervalPD6 planner also passesbothoracles. Full landing memcheck(no leaks),initcheck,synccheck pass. New synthetictest ran AFTERminorfinalreportD2D/failure classification updates; initialfull7landing/oracle/sanitizers were before those reporting-only changes (GPUcodeunchanged). Finalpairedbenchmarks andsyntheticsanitizers afterallcodechanges.
- Matched same backendv16 with baselinecore d35fe7e vsnewcore,2warm7samples alternating: landing SCvx .258173814 ->.249037551s(1.036686x), process.657813228->.637775330s(1.031418x); PD6SCvx1.657411652->1.635300755s(1.013521x),process2.024929175->2.002447668s(1.011227x). Same28/179inner,2accepted,independent gates/objectives. Modestscopedgains notuniversal. Landing initialconversion .771344->1.644193ms; numericupdate1.479357->1.097336ms. InitialmapuploadincreasesnativeH2D135888->286728bytes,D2H118172->150840,nativepeak526220->758744. ExcludesopaqueQOCO/cuDSS; donotclaimwholetransferreduction. CPUoracles adddownloadsnotincludedpairedtimings.
- Committed artifacts/performance/qoco-device-conversion-{checkpoint,pd3,planner-pd6,planner-pd6-baseline-sample,planner-pd6-optimized-sample}.json. Docs GPU_QOCO_LOCAL_BACKEND.md#compiled-cuda-numerical-conversion and GPU_NATIVE_OPTIMIZATION_PROGRESS.md updated. Checkpointhashes binaries/libs and10sanitizerlogs. Full samples ignored adjacent.samples. Ignoredhelpers build/performance/check_qoco_conversion.py (allGPUtests incl7repeat/PD6oracle),check_native_conversion.py,record_qoco_conversion.py,freeze_qoco_conversion.py. Freezealreadycreated4556941; don'trerunwithoutnewname. Logs native-checks/qoco-conversion-*.log, PD6oraclefullJSON build/performance/qoco-conversion-pd6-oracle.json. Ruffnotapplicable noPythonproductionedits; gitdiffchecks/builds pass.
- FROZEN core/planner/device_scvx_integration_test/qoco_gpu_conversion_test/native_qoco_conversion_test at /home/angus/build-spacepdhcg-gpu-native/4556941. Currentbuild cuda/libspacepdhcg_cuda.so same finalmeasuredbinary, backend unchanged /home/angus/build-qoco-gpu-values-v16/build/libqoco.so. Checkpointrecord SHA authoritative. BuildWSLCMake /home/angus/spacecraft-trajectory-optmiser/.venv/bin/cmake, CUDA12.8, distroUbuntu-22.04. ReconfigCMake auto-glob tests writes pinned copied source underbuild, notsharedQOCO; okay. GPUlockfree allsessions97702/7392/64815/90056/33281/20788terminalclosed.
- NEXT realgoal: remove remaining CPU QOCO numeric update/equilibration/KKT assembly and converted-values D2H. Investigate isolated pinnedQOCO qoco_update_matrix_data/qoco_update_vector_data (switch CPUmode and rebuild/scalecoeffs) and cudss KKT update pipeline. Compile conversion residentoutput sixsegments ready for optional direct-deviceupdateAPI; avoidCPUroundtrip. Still initialCPUstructurediscovery, upstreamCPUscalars, hostacceptedwarmstate/outercontrol, vendor factorization issue, largenonlinearPD3/rotationalPD6, trajectorystructureoperators/batching/graphs/nativeGPU GTOC12. Do notmistake thischeckpointforfullGPUcompletion. Do not retryrejectedcuDSSmodes/versionsorloosenqualification withoutnewreason. Goal staysactive, notblocked.


### GPU device updates checkpoint 038695b (2026-09-06)

- Goal active; no GPU jobs remain. Local branch perf/gpu-native-pipeline, no push.
- Optional backend v20 at /home/angus/build-qoco-gpu-updates-v20/build-fixed/libqoco.so,
  source reproduced at /home/angus/build-qoco-gpu-updates-repro/source. Flags: gather,
  correct-stopping, deterministic, checked-cudss-abi, queued-operators,
  device-cone-reductions, values-only-updates, device-numeric-updates.
- Native core + planner + conversion/landing tests frozen at
  /home/angus/build-spacepdhcg-gpu-native/038695b. Core SHA256
  f7c127c20becf2f9c4ec3d757e700c646adbf6dbb487339c76e35415ccac4df8;
  v20 QOCO SHA256 98be90fe7d249b3eb6a8b0da43cf8f145c5c2634921c177e491d756dcfc29e38.
- Device context owns numerical updates; legacy host shadows stale. Initial native
  setup temporarily uses Ruiz0, then requested Ruiz on GPU. Repeat P/A/G/c/b/h
  stay device resident; GPU scaling/transpose/regularization; existing GPU KKT
  coefficient update reused. Backend downloads 72 scalar bytes per update,
  no new allocations. Native conversion downloads status only except initial,
  rebuild, verbose, or test oracle paths. CPU structure/initial KKT/scalar/outer
  control remain. Earlier docs implying CPU numerical KKT updates were incorrect.
- Fixed pinned optional host scale_arrayf no-op, inverse transpose permutation
  direction and added P diagonal update mapping. v18/v19 failed prototypes retained.
- Final 2warm/7measured paired benchmarks: landing SCvx .259198246 -> .259852690s,
  process .653672541 -> .644672720; PD6 SCvx1.675632290 ->1.667908129,
  process2.064489571 ->2.047172846. Overall essentially flat, unchanged28/179inner,
  2accepted/physics. Update phase1.137233ms ->.488178ms. Native D2H150840->110000B,
  H2D286728/peak758744 unchanged; backend/cuDSS traffic/memory excluded.
- Mission benchmarks use Ruiz0. Nonzero Ruiz validated in synthetic direct/native
  solver tests. Seven standalone cases incl n1031/Ruiz4/zero P/absent constraints/
  missing diagonal; all4sanitizers clean. Native Ruiz4 + landing mem/init/sync clean,
  zero leaks. 7landing and actualPD6 pass CPU conversion+KKT oracles. Existing cuDSS
  factorization race unresolved/not rerun; optional backend not promoted.
- Evidence artifacts/performance/qoco-device-updates*.json + docs/GPU_QOCO_LOCAL_BACKEND.md.
  Standalone executable /home/angus/build-qoco-gpu-updates-v20/qoco_gpu_numeric_update_test.
  Helpers build/performance/{check_qoco_device_updates,record_qoco_device_updates}.py.
- Use Ubuntu-22.04, CUDA12.8, cuDSS0.7.1.6, flock /home/angus/.spacepdhcg-gpu.lock.
  Do not touch shared pinned QOCO or remote campaigns. Never rebuild live binaries.
  Next: profile factorization/iteration costs; remaining CPU control/setup,
  conditioning beyond synthetic cases, large structured trajectories/batching,
  GPU GTOC12. Full goal not complete.


### GPU scalar reduction checkpoint 1c2e76c (2026-09-06)

- Previous/current turn PROGRESS. Goal active; all GPU sessions terminal. No push.
- New optional backend /home/angus/build-qoco-gpu-scalars-v21/build/libqoco.so,
  all v20 flags plus --device-scalar-reductions. Prepared hashes reproduced at
  /home/angus/build-qoco-gpu-scalars-repro/source. Pinned/shared source untouched.
- Core unchanged: use frozen /home/angus/build-spacepdhcg-gpu-native/038695b
  core/planner/native/landing test binaries. Scalar standalone executable at
  /home/angus/build-qoco-gpu-scalars-v21/qoco_gpu_scalar_test. Hashes verified in
  artifacts/performance/qoco-device-scalars-checkpoint.json.
- inf_norm/min_abs_val GPU result now value-only (previously cuBLAS index download
  plus value download). check_nan uses retained scalar pool, no per-call flagalloc/
  hostzero upload. New header cpp/cuda/patches/qoco_device_scalar.cuh; small arrays
  <=4096 one block; larger inputs multi-block grid-stride partial then final,
  bounded256blocks. Extrema finite exact, NaNs explicitly propagate, all FP64.
  Existing synchronous scalar decisions/defaultstream/scope lifecycle unchanged.
- New test exact finite/NaN/inf/subnormal/zero/tails up to1048579, byteoffset,
  empty, nested/unscoped lifecycle. All4san clean inclrace/zero leaks. NativeRuiz4,
  7landing repeats and PD6 all pass CPU conversion+KKT oracles; landing mem/init/
  sync clean. Existing cuDSS race remains open/notrerun, optional notpromoted.
- Matched2warm7measured v20->v21 same038core:
  landingSCvx .245660622 ->.227428327s (1.080x); process .660829355 ->.605240207;
  PD6SCvx1.662881627 ->1.536695984s (1.082x), process2.027659650 ->1.905294178.
  Same28/179inner,2accepted,objectives/physics. Retained slowbaselinewarmup raw.
- Separate qualified landing API trace: streamSync847->401, memcpyAsync1232->786,
  memcpy953->925, malloc405->379/free325->299; kernelLaunch6065unchanged.
  Nsight2024.6 APIonly, noRTX5090GPUtimeline; don't interpret API durations as
  exclusive GPU timings. Reports /home/angus/build-spacepdhcg-gpu-native/
  qoco-v{20,21}-api-profile.nsys-rep and .sqlite; CSVs build/performance.
- Helpers build/performance/{build_qoco_scalars,check_qoco_scalars,record_qoco_scalars,
  profile_qoco_v20,profile_qoco_v21}.py; logs native-checks/qoco-v21-*.
- Next: check_stopping in prepared source/src/qoco_utils.c has many separately
  returned norms (b,s,c,h,Aty,Gtz,Px,Ax,Gx,primal,dual) and dot products; batch
  device calculations/decisions to cut further scalar synchronizations. Preserve
  exact independent physics acceptance. Initial KKT/structureCPU, outer control,
  large trajectories/conditioning/batching/GTOC12 and vendorissue stillunfinished.
- Preserve unrelated dirty files. Use Ubuntu-22.04 / CUDA12.8 / cuDSS0.7.1.6;
  flock /home/angus/.spacepdhcg-gpu.lock. Never rebuild live libraries/executables.


### Batched stopping checkpoint dd47990 (2026-09-06)

- Current/previous turn PROGRESS. Goal ACTIVE. All GPU/build sessions terminal.
  Commit dd47990 on perf/gpu-native-pipeline, no push. Preserve unrelated changes.
- Candidate /home/angus/build-qoco-gpu-stopping-v22/build/libqoco.so; all v21 flags
  plus --batched-stopping. Baseline v21 /home/angus/build-qoco-gpu-scalars-v21/build.
  Core unchanged: frozen /home/angus/build-spacepdhcg-gpu-native/038695b binaries.
  Candidate SHA256885fedc418b98770740f3542b5b89668be279fd4b7823f87a3c9036cb7e85e09.
  Prepared source reproduced /home/angus/build-qoco-gpu-stopping-repro-final/source.
- New cpp/cuda/patches/qoco_batched_stopping.cuh queues12norms/5Ddot results onGPU,
  1GPUcombine and 1sixdouble (48B) D2Hpacket. Uses retainedscalarpool279doubles,
  defaultstream ordering; cuBLAS pointermodeDevice onlyinsidefunction, restored
  beforehostread/endnested scope. Sameexisting sparseoperators/cuBLAS dot.
  NewpointermodeGet/Set functions dynamically loaded/checked inbackend table.
- check_stopping CPU policy/bestiterate/stallsettings unchanged; replacesmetrics
  arithmeticwithGPU. Original metrics retained exportedC as
  qoco_reference_stopping_metrics; devicefunction qoco_gpu_stopping_metrics.
  SPACEPDHCG_TEST_QOCO_BATCHED_STOPPING_COMPARE=1 checks all6metrics eachstopcheck
  tolerance2e-12 relativefloor1. Flagrequires device-scalar-reductions,
  queued-operators,correct-stopping. No tolerance relaxations.
- Standalone /home/angus/build-qoco-gpu-stopping-v22/qoco_gpu_stopping_test covers
  independentlongdouble CPUformulas+oldmetrics; n17/4103,emptyconstraints,zeroP,
  nontrivialscales,3iteratechanges,nested/unscoped,restorehostdotmode. All4sanpass
  inclrace/zero leaks. NativeRuiz4,7landing,actualPD6 pass stoppingcompare +
  CPUconversion/KKToracles; landingmem/init/syncpass. CuDSSrace stillopen/notrerun.
- Matched2warm7measuredv21->v22:
  landingSCvx .237978969 ->.184373653s (1.2907x), process .724607417 ->.661227233;
  PD6SCvx1.540333017 ->1.269196548s (1.2136x),process1.918976981 ->1.619480325.
  Same28/179inner,2accepted/objectives/independentphysics. Rawvariabilityretained.
- SeparatequalifiedlandingAPItrace baselinev21reuseidenticalbinary: memcpy925->595,
  memcpyAsync786->636,streamSync401->251,kernelLaunch6065->6095 (30newcombine),
  malloc379/free299unchanged. Nsight2024.6APIonly noRTX5090kerneltimeline.
  /home/angus/build-spacepdhcg-gpu-native/qoco-v22-api-profile.nsys-rep/.sqlite.
- Evidence artifacts/performance/qoco-batched-stopping*.json + docs/GPU_QOCO_LOCAL_BACKEND.md.
  Helpers build/performance/{build_qoco_stopping,check_qoco_stopping,
  record_qoco_stopping,profile_qoco_v22}.py. Testbuildmanualnvcc std17 sm120 with
  v22/source/include and source/lib/qdldl/include, -L v22/build -lqoco -ldl.
  Aftertests removedunusedCvariable: resultingbackendbyte-identical, sourcehashes
  updatedverified; initialbuildlogwarningisresolved. Do NOT rerun tidyhelper.
- Next CPUiterationscalarwork: objective+mu, centering+line search,
  iterative refinement control, bestiterate/stop policy. InitialCPUstructure/KKT,
  outercontrol/warmstate, cuDSSrace, large trajectories/conditioning/batching,
  GPU GTOC12 all stillunfinished. Goalactive.
- Use Ubuntu-22.04, CUDA12.8,cuDSS0.7.1.6; flock /home/angus/.spacepdhcg-gpu.lock.
  Never rebuild live .so/testexe. SharedpinnedQOCO and remote campaigns untouched.


### Shared iteration scalars checkpoint 7ab9a8d (2026-09-06)

- Previous/current turn PROGRESS. Goal ACTIVE, all GPU/build sessions terminal.
  Localbranch perf/gpu-native-pipeline, HEAD7ab9a8d, no push. Unrelateddirty preserved.
- FINAL candidate immutable /home/angus/build-qoco-gpu-iteration-v24/final/libqoco.so
  SHA256def9941450775abccfa99f310d28a70f9e9e51b2c44359a07dc00f10dd1db797.
  Finaltest /same/final/qoco_gpu_stopping_test, pass --iteration to require8metricAPI.
  Coreunchanged: /home/angus/build-spacepdhcg-gpu-native/038695b allfrozenbinaries.
  Baselinev22 /home/angus/build-qoco-gpu-stopping-v22/build/libqoco.so.
- Allv22flags plus --batched-iteration-scalars (requiresbatched-stopping). Objective
  andmujoin64Bpacket, samecuBLASDdot, reuseP*x beforecorrection andc*x; preserve
  separateregularizationcorrection, safe_divguard, mu0fornoinequalities. Public
  sixmetricAPIretained, new qoco_gpu_iteration_metrics outputs8. Header templated.
  qoco_solve removes separateobjective/mu calls; check_stopping appliesreturned
  obj/mu beforebestiterate/policy. AllotherCPUcontrolunchanged.
- Enhancedindependenttestnormal/zeroP/noineq/n17+4103/nontrivialscales guards
  objective vsscaledstoppingPproduct, mu vsscaledgap, zeroobjective-scale contract,
  bothinterfaces,nestedscopes/restorehostmode. All4sanpass/0leaks. NativeRuiz4,
  7landing,actualPD6 pass all8metriccomparison+CPUconversion/KKT/physicsoracles.
  Fulllandingmem/init/syncpass. CuDSSraceopen/notrerun; optionalbackendnotpromoted.
- FINALmatched2warm7measuredv22->v24:
  landingSCvx .186769964 ->.175701629s (1.063x),process .575536424 ->.575911532flat;
  landingQOCOSolve .132608308 ->.136713226 slightlyslower (don'tclaimallphaseswin).
  PD6SCvx1.240639367 ->1.163785273s (1.066x),process1.613479252 ->1.530127625
  (1.054x). Same28/179inner,2accepted,obj/physics. Timingvariabilityretained.
- FinalqualifiedlandingAPItrace:streamSync251->131,asyncCopies636->516,
  kernelLaunch6095->6005,syncCopies595same,malloc379/free299same. Nsight2024.6
  APIonly/noRTX5090kerneltimeline. /home/angus/build-spacepdhcg-gpu-native/
  qoco-v24-final-api-profile.nsys-rep/.sqlite, priorv22baselineidenticalbinary.
- v23prototypepassedunit butfailednativeoracle: missingkkt.h declarations made
  C assumeintegerreturns forcompute_objective/mu. Fixedinclude; v24 builtwith
  -Werror=implicit-function-declaration. v23source/lib/logretained unqualified.
- IMPORTANT: originalv24 buildpath was rebuilt duringcommentcleanup, hashchanged
  evenafterrestoringsource. Original9867... binary NOTretained. Earlier
  qoco-batched-iteration-{pd3,planner-pd6}.json annotatedsuperseded; use *final*
  artifacts. FinalcandidateCOPIED toimmutablefinal/ thenALLtestsandbenchrerun.
  Do notreruntidy/restore/freezehelpers. Never rebuild oroverwritefinal/ binaries.
  Preparedhashesreproduce /home/angus/build-qoco-gpu-iteration-repro/source.
- Evidence artifacts/performance/qoco-batched-iteration-checkpoint.json plus
  *final*.json. Helpers build/performance/{check_qoco_iteration_frozen,
  record_qoco_iteration_final,profile_qoco_v24_final}.py. Logs qoco-v24-final-*.
  build_qoco_iteration_final.py buildsmutablebuild/ only, don'tconfusewithfinal/.
- Next: centering,line search,IRscalarcontrol,bestiterate/stopGPUdecisions. Full
  CPUinitialstructure/KKT,outercontrol/warmstate,large trajectories/conditioning/
  batching,GPU GTOC12 andvendorissue remainunfinished. Preservefullgoal.
- Ubuntu-22.04,CUDA12.8,cuDSS0.7.1.6, flock /home/angus/.spacepdhcg-gpu.lock.
  SharedpinnedQOCO,remotecampaigns untouched. No subagents authorized.


### GPU step control checkpoint 09f2572 (2026-09-06)

- PROGRESS, full goal ACTIVE. Local branch perf/gpu-native-pipeline, HEAD09f2572,
  no push. All GPU/build jobs terminal; unrelated dirty files preserved.
- New frozen candidate /home/angus/build-qoco-gpu-step-control-v25/final/libqoco.so
  SHA256 a48278eaaab05add8b3098e9817fdad52bed2328dc88b2317783203e62fa443a.
  Same final/ qoco_gpu_step_control_test SHA60f8ed69cadf4a3ee2a910fadffee2d6236ae7553ad4ec1532d883d48922ea48.
  Baseline /home/angus/build-qoco-gpu-iteration-v24/final/libqoco.so SHAdef994...;
  unchanged core/executables /home/angus/build-spacepdhcg-gpu-native/038695b.
  NEVER rebuild/overwrite frozen files; helpers check_qoco_steps.py freezes once.
- All v24 preparation flags plus --device-step-control. Requiresbatched-stopping
  and prerequisites. Two new CUDA headers: qoco_device_step_cones.cuh queues
  existing LP/SOC boundary reductions; qoco_device_step_control.cuh computes
  centering vectors, two cuBLAS device dots, sigma clipping/cube, and fuses
  finalx/y/s/z updates into one gridstride kernel with device alpha. Stillreturns
  one hostsigma+onehostalpha perinner; combinedRHS/stop/bestpolicy stillhost.
  qoco_min macro previously reevaluated one linesearch; newpath queues eachonce.
  Retains NaNdirectionearlyexit, nearzeroalpha cutoff, FP64 order, emptyNaNsigma.
- Testmode SPACEPDHCG_TEST_QOCO_DEVICE_STEPS_COMPARE=1 compares oldsigma/alpha.
  Newstandalonetest independentlongdoublearithmetic+feasibilitybisection, LP/SOC/
  mixed/empty, boundary/nearlylinear/tinyalpha,262145LP/1027SOC,offsetguards,
  scope/unscoped/nest. All4sanitizerspass; nativeRuiz4,7landing,PD6passstep+8metric+
  CPUconversion/KKTphysicsoracles. Fulllandingmem/init/syncpass bothwithoracles
  and productionwithoutoracles. cuDSSfactorizationrace remainsopen/notrerun.
  Optionalbackendnotpromoted; nofullracecheckcleanclaim.
- Matched2warm7measuredv24->v25 frozenbinaries:
  landingSCvx .173833972 -> .156347721s (1.112x), QOCOSolve .128148830 ->.110147996;
  landingprocess .581577767 -> .588902853 (1.3%slower). Optimizedrepeat2
  took1.300420s; retainedinthe7samples, causenotestablished, nooutlierdiscard.
  PD6SCvx1.166371462 ->1.032314632 (1.130x), QOCOSolve1.050969959 ->.934502965;
  process1.522863519 ->1.390631905 (1.095x). Same28/179inner,2accepted,obj/physics.
- QualifiedlandingAPI counts syncCopies595->483,asyncCopies516->460,
  streamSync131->75,launch6005->5697,malloc379/free299same. ProfileAPIonly
  Nsight2024.6 noRTX5090timeline. /home/angus/build-spacepdhcg-gpu-native/
  qoco-v25-api-profile.nsys-rep/.sqlite; v24finalbaselineprofile reused.
- Evidence artifacts/performance/qoco-device-steps-{checkpoint,pd3,planner-pd6,
  pd6-oracle,planner-pd6-baseline-sample,planner-pd6-optimized-sample}.json.
  Source reproduced /home/angus/build-qoco-gpu-step-control-repro/source.
  BuildCflags-Werror=implicit-function-declaration; CUDA12.8 sm120 cuDSS0.7.1.6.
  Helpersbuild/performance/{build_qoco_steps,check_qoco_steps,
  check_qoco_steps_production,measure_qoco_steps,record_qoco_steps}.py.
  Logsbuild/performance/native-checks/qoco-v25-*. Sourcepreparationandruffpass.
- Next: hostsigma/alphaconsumers, iterative refinement, bestiterate/stop decisions;
  initialCPUstructure/KKT,outercontrol/warmstate,largertrajectoriesconditioning/
  batching,GPU GTOC12,vendorissue remain. Do notmarkwholegoalcomplete/blocked.
  No subagents authorized. SharedpinnedQOCO and Lambda/H100 untouched.
  WSLUbuntu-22.04; flock /home/angus/.spacepdhcg-gpu.lock forGPU serialization.


### Combined RHS experiments checkpoint 851841d (2026-09-06)

- Previous goal turn PROGRESS: two GPU RHS experiments, independent tests,
  matched measurements, 100/500 interval qualification and a changed next target.
  Goal ACTIVE, not complete/blocked. Branch perf/gpu-native-pipeline HEAD851841d,
  no push. All GPU/build/test/probe jobs terminal. Unrelated dirty work preserved.
- IMPORTANT: neither experiment is a reliable general speedup. Keep v25 as the
  PERFORMANCE BASELINE: /home/angus/build-qoco-gpu-step-control-v25/final/libqoco.so
  SHA a48278eaaab05add8b3098e9817fdad52bed2328dc88b2317783203e62fa443a.
  Same frozen core/executables /home/angus/build-spacepdhcg-gpu-native/038695b.
  No production backend promotion; new options are explicit opt-in experiments.
- v26 frozen /home/angus/build-qoco-gpu-combined-rhs-v26/final/libqoco.so
  SHA811e7bd2b198cda88e78caa97f468388651bb875adf9612f5f0c2527e726c7f1.
  v27 frozen /home/angus/build-qoco-gpu-combined-rhs-v27/final/libqoco.so
  SHAe99226f73b2f1c1d9f9fb988cc2a0fd24b6e7a4b1e245cb05d4b99c6ccb89e6c.
  Each final/ also holds its qoco_gpu_step_control_test. Use --combined to require
  the new API. v27 test SHA436342acbfaf3a6c7cf9ee3420d45a6a5d5b106b506398dae8b2b4e604d85536.
  NEVER rebuild or overwrite frozen files. Source copies and original v26
  reproduction retained even though repository header later evolved for v27.
- Flags: all v25 flags plus --device-combined-rhs (requires step control).
  v27 additionally --queued-centering-metadata. New headers
  qoco_device_combined_cones.cuh and qoco_device_combined_rhs.cuh. Refactored
  centering implementation allows device scalars to feed combined RHS directly.
  Fuses two Jordan products, identity shift, negate and correction into one
  grid-stride cone kernel, cached SOC offsets remove quadratic add_e prefix scan.
  Same FP64 product/subtraction order. Mu recomputed from already available z.s/m;
  s/z remain unchanged between metric calculation and combined RHS.
- v26 downloads sigma after RHS chain; v27 pins the QOCOWorkspace allocation
  (same layout, matched cudaHostAlloc/cudaFreeHost) and queues sigma D2H. Production
  has no host sigma consumer before next synchronous result or scope completion.
  Free helper explicitly fences default stream. Old standalone centering API
  remains synchronous. Alpha/stop/best/IR host decisions still remain.
- Test environment SPACEPDHCG_TEST_QOCO_COMBINED_RHS_COMPARE=1 compares full Ds
  and RHS against original C construction. Existing step/eight-metric/CPU
  conversion/KKT oracles also pass. Extended standalone test uses independent
  long-double Jordan products/inverse with identity NT, LP/SOC/mixed/empty,
  262145 LP/1027 SOC, boundaries, canaries, input preservation. v27 tests pinned
  allocation and sigma completion at stream event, oracle on/off, nested scopes.
  Native Ruiz4, seven landing and 20-interval PD6 pass. Both candidates all four
  standalone sanitizers and landing mem/init/sync pass; production no-oracle
  landing mem/init/sync also pass. All memchecks report zero leaks. Existing
  cuDSS factorization race remains OPEN/not rerun, no full race-clean claim.
- v26 matched vs v25: landing SCvx .151187255 -> .165627312 s (regression),
  process .560818278 -> .584534285. PD6 SCvx1.043636990 ->1.033875746 (1.009x),
  QOCO solve .944271953 ->.907025868 (1.041x). Landing .463112s outlier retained.
  v27 matched vs v25: landingSCvx .148963942 ->.157818108 (regression), process
  .537324639 ->.582599999; QOCOsolve .110555108 ->.110039977 essentially flat.
  PD6SCvx1.042608951 ->1.031575469 (1.011x), process1.400294339 ->1.421922312
  (slower); QOCOsolve .932651578 ->.929488744 essentially flat. Same28/179inner,
  objectives/2accepted/physics. Separate v26/v27 batches do not isolate causality.
- v27 landing API vs v25: syncCopies483->455, async460->488, launches5697->5585,
  streamSync75->76 (workspace destruction), hostAlloc/free4->5, device malloc379/
  free299 same. Nsight2024.6 API only, no RTX5090 GPU timeline. All profiles
  /home/angus/build-spacepdhcg-gpu-native/qoco-v26-api-profile and qoco-v27...
- NEW larger fixtures: artifacts/performance/qoco-rhs-scaling-input-{100,500}.json
  copy canonical PD6 changing intervals only, same 4s duration and 1e-6 tolerance.
  Single probes both v25/v27 qualify with identical objectives: N100 inner103,
  2accepted/1rejected; N500 inner34,2accepted. Not speedup benchmarks.
  Matched N500: 2warm+7measured/variant, SCvx .721922862 ->.713040918 (1.012x),
  process1.227711921 ->1.220916158 (1.006x), BUT QOCOsolve .252728495 ->.258415270
  (regression). All18certified, sameobjective/34inner/2accepted. Full N500 passes
  conversion/KKT/RHS/step/eight-metric oracles. No universal speedup/promotion.
- IMPORTANT NEXT TARGET: N500 baseline median QOCO setup .343772081s > solve
  .252728495s. Conversion .051605096s. Topology .096730470s, coefficient .054196038s,
  replay .056464352s; these are existing named timings, not disjoint GPU stages.
  Split QOCO setup into measured host/GPU stages before attributing bottlenecks.
  Likely inspect initial structure/KKT assembly, allocations/vendor analysis;
  do not assume scalar IPM calls still dominate. Larger N2000 could be useful
  after safe profiling, but not yet run. N500 has24514vars,22028scalar/18522affine
  rows. Do not interpret cross-size times monotonically (iterations differ).
- Evidence qoco-device-combined-rhs-checkpoint.json and qoco-queued-combined-rhs-
  checkpoint.json; latter includes large500 stages and raw report links. Reports,
  representative results and scaling probes committed. Helpers in build/performance:
  build_qoco_rhs.py, build_qoco_rhs_queued.py; check_qoco_rhs.py and *_queued.py
  freeze once; measure_qoco_rhs.py, *_queued.py, *_large.py; record_qoco_rhs.py,
  *_queued.py then record_qoco_rhs_large.py (latter appends larger results; rerun
  both if regenerating). probe_qoco_rhs_scaling.py. Logs qoco-v26-* and qoco-v27-*.
  Repro sources build-qoco-gpu-combined-rhs-repro/source (v26 before changes),
  build-qoco-gpu-combined-rhs-queued-repro/source (v27 current code).
- All C builds -Werror=implicit-function-declaration, CUDA12.8 sm120,
  cuDSS0.7.1.6. Ubuntu-22.04, flock /home/angus/.spacepdhcg-gpu.lock. No subagents
  authorized. Shared pinned QOCO and Lambda/H100 campaigns untouched.
- Full goal still includes initial CPU structure/KKT, outer control/warmstate,
  host stop/best/IR/alpha, larger structured operators/conditioning/batching,
  GPU GTOC12, vendor issue. Do not mark complete or blocked.

## GPU-native goal checkpoint 29daf14 — setup profiling and ownership (2026-09-06)

Goal remains ACTIVE and incomplete. No running local GPU/build jobs remain. Do not touch shared pinned QOCO or Lambda/H100 campaigns. No subagent authorization. Branch perf/gpu-native-pipeline, committed only this checkpoint; unrelated dirty files remain.

Retain frozen core /home/angus/build-spacepdhcg-gpu-native/038695b, SHA f7c127c20becf2f9c4ec3d757e700c646adbf6dbb487339c76e35415ccac4df8. NEVER rebuild/overwrite frozen artifacts. v25 remains historical performance comparison: /home/angus/build-qoco-gpu-step-control-v25/final/libqoco.so SHA a48278eaaab05add8b3098e9817fdad52bed2328dc88b2317783203e62fa443a. v31 retained opt-in correctness candidate: /home/angus/build-qoco-gpu-lifetimes-v31/final/libqoco.so SHA 4a8a0d15e83bcdd68fc0910718a30602417f61ec4cf2e315a7f06a0cb2b358a7. All v25 flags plus --setup-lifetimes, NO profile or GPU ordering. Prepared source hashes reproduce; docs and committed artifacts hold exact flags/hashes.

v31 fixes temporary host KKT leak, retains CSR indices through vendor matrix lifetime, creates dense wrappers before analysis. Focused Linux interposer proves 559188 bytes leaked by seven landing setups and 5315104 by one N500 setup in v25; v31 frees all tracked bytes. Scope only construct_kkt four allocations, not whole process heap. Added tracker cpp/cuda/tests/qoco_host_kkt_tracker.cpp. Extra retained CSR memory excluded by native-owned telemetry.

Native Ruiz4, seven landing repeats, N20/N500 PD6 CPU/GPU conversion/KKT/step/eight metric oracles pass. All four standalone step sanitizers pass. Full landing mem/init/sync pass with and without test oracles. Full production racecheck RERUN terminal99, SAME 30 cuDSS factorization errors; unresolved, do not suppress or claim full raceclean. Committed full log qoco-setup-lifetimes-checkpoint.json. Matched v25/v31 2warm+7 alternating measured: landing SCvx148.338/150.244ms; N20PD6 1064.851/1061.243ms; N500 731.741/730.562ms, same28/179/34 inner; no general speedup. Retain correctness fix, not speed claim.

Setup --profile-setup requires checked ABI; env SPACEPDHCG_TEST_QOCO_SETUP_PROFILE=1 enables fenced diagnostics, NOT speedup timing. v30 /home/angus/build-qoco-gpu-setup-profile-v30/final/libqoco.so split analysis: N500 median CPU reordering231.925ms, symbolic43.553ms, handles47.201ms, hostassemble2.087ms, conversion3.640ms. Three probes eachN20/N500 allcertified. One large symbolic428ms/workspace782ms outlier retained unknowncause. v28 old profile labels symbolic_analysis=combinedANALYSIS; v30 separates. v28/v30 artifacts committed.

Rejected GPU static-degree permutation v29 /home/angus/build-qoco-gpu-ordering-v29/final/libqoco.so SHA73beb0537ca4900201752705fa727ac15072857001bb78522ac48bd17dbf1871. New opt-in --gpu-degree-ordering uses CUB degree/index stable sort + device CUDSS_DATA_USER_PERM; NOT AMD. Unit0/1/13/4099/65539 with repeats/guards/bijection/disconnected/duplicate tests + all4unit sanitizers clean; native/landing andN20/N500 probesqualified. N500solve2.416s vs default~.25s; SCvx2.894s, no matchedspeeduptrial. Landing36vs28inner. Fillunmeasured. Historical v29 profile combinedanalysis + createCSR includesGPUperm; oldpreparedsource preserved in qoco-gpu-ordering-checkpoint. NEVER promote simply because orderingonGPU.

Next substantial task: trajectory separator-aware GPU ordering with explicit stage metadata, no inferring family from n; measure reorder AND factor/solve costs. PD6 layout n=49*N+14, states14*(N+1), controls7*N, virtual +/-14*N, native adapter cpp/cuda/src/native_qoco_adapter.cpp. Or tackle CPU KKT assembly/caching/host control, but reorder ~232ms dominant setup. Whole CPU-free goal, vendor race, structured large solves and GPU GTOC12 remain unfinished.

Ignored reusable helpers build/performance/{build_qoco_lifetimes,check_qoco_lifetimes,finish_qoco_lifetimes,repro_qoco_lifetimes,record_qoco_setup}.py. DO NOT rerun freezehelpers: final directories exist. finish helper already ran ALL checks+matched3benchmarks andN500oracles. Reprohelper existingdest cannot rerun; choosefresh. WSL Ubuntu-22.04, serializeGPU flock /home/angus/.spacepdhcg-gpu.lock. No active jobs. Commit29daf14 no push.

## GPU tree checkpoint 44db7cb — 2026-09-06

Goal ACTIVE, incomplete. This turn made concrete progress (GPU ordering/tree + matched large-case gain). No local processes remain running. Preserve Lambda/H100 campaigns and shared upstreams. No agents authorized. Branch perf/gpu-native-pipeline; commit44db7cb, no push. Unrelated dirty files preserved.

Latest experimental candidate: /home/angus/build-qoco-gpu-trajectory-v42/final/libqoco.so SHA8c96980e396412aaeda225d71578feaa7bf906ebff34c004e369d6b8f794e203. Flags ALL v25 + --setup-lifetimes --trajectory-ordering --trajectory-tree. NO profiling. Native core /home/angus/build-spacepdhcg-trajectory-v41/final/libspacepdhcg_cuda.so SHAba045295670b421e049fa1fc21d399279fbfa523d71b2062d3f07b9620823d14. Both frozen before final checks/benchmarks. NEVER rebuild/overwrite these snapshots. Reference QOCO v31 SHA4a8a0d15e83bcdd68fc0910718a30602417f61ec4cf2e315a7f06a0cb2b358a7; matched SAME core41 forboth. Old integration/planner executables /home/angus/build-spacepdhcg-gpu-native/038695b stillused with LD_LIBRARY_PATH override. Standalone tree test qoco42/final/qoco_trajectory_ordering_test SHA84eb4b4cc38e1e34dc6ba5c5fc0cc3452ed8b18c4e15663ae4e2bbc63dacecdf.

Matched 2warm+7alternating measured, allphysics/objectivegates pass: landingSCvx154.919->180.697ms, inner28->36 (regression); PD6N20 1038.735->1159.789ms, inner179->176 (regression); PD6N500 745.262->553.157ms=1.347x, same34inner. N500process1.223355->1.048186s=1.167x. N500Qoco setup356.258->135.697ms, solve261.967->303.429ms (solve subphase regresses ~16%). Keep experimental LARGE-casecandidate, no general/defaultpromotion. Wholegoalnot100%GPUyet: hostKKTassembly, solver/outercontrol,otherfamilies,physicsqualifiedselection etc remain.

Critical finding: USER_PERM alone loses cuDSS parallelization metadata. NVIDIA https://docs.nvidia.com/cuda/archive/13.1.1/cudss/advanced_features.html#user-provided-elimination-tree-data-saving-reordering documents exact contract. Supply CUDSS_DATA_USER_ELIMINATION_TREE + CUDSS_CONFIG_ND_NLEVELS. Tree sizes bottom-up level order contiguous groups, 2^k-1 nodes, rootlast. Implemented GPU graph labels from actual device index maps, propagate cone labels BEFORE epigraph inference, ancestor partition guard, local auxiliaries moved to leaf descendant, contiguous radixsortgroups and GPUatomicintegercounts. All graph ordering data CUDA, hostdims only. Current k=floor(log2(N+1)) capped8 and MIN2; direct installedcuDSS0.7.1 probe rejectsND_NLEVELS=1 (default10); ordinary positives2..16accepted/Getmatches. Earlierdocs0.6 poweroftwo incompatible; useverifiedruntime. Permutation is new->old as A[P,Q].

Source cpp/cuda/patches/qoco_trajectory_ordering.cuh, optional new adapter hook qoco_gpu_set_trajectory. Native adapter now owns one GPUcopy of state/control/virtualmaps,3D2Dcopies telemetry; clears hook aftersetupanddropsoriginalstream dependency. Invaliddeviceindicesduplicates/outofrange rejected. Synthetic no-trajectory tests stillgenericvendor; actualPD6/landing logs confirmorderingused. All validtestedgraphs rootpromotions0; guard codeconservative, independenttreeedgetest covers actualreturnedpartition. GraphunitincludesN1,2,7,24,65,10000 =>70001vertices, shuffledmaps, longrangeedges, independentCPUeliminationfront, permutation/guards/repeats, malformedmetadata/concurrentduplicates, nondefaultstreams andCPUtreecheckeveryedge. All4unit sanitizers pass.

Latest owned-metadata test /home/angus/build-spacepdhcg-trajectory-v41/ownership-v47/native_qoco_conversion_test SHA57f06673bc91b70cca25fc25d62d0d9d0c1966d8bba85f92230343490b66707e. Test-only proxy /home/angus/qoco-recreate-probe-v45/proxy.so SHA0014f8c9a9298a223dbd77176f571d9fde9bb297e186bba2ddb3c62eba7f8ad8. Source cpp/cuda/tests/qoco_recreate_probe.cpp forwardsrealQoco via DT_NEEDED, onlyinjects nextsolve rawstatus3and0iterations; no productionfault hook. Run native... 4 trajectory with SPACEPDHCG_QOCO_LIBRARY=proxy, LD_LIBRARY_PATH core41/final:qoco42/final:cuDSS:CUDA. Test frees originalmaps, destroysoriginalstream, createsnewstream, injectsfailureandprovesworkspacecreationsincreasedwithsameKKTaccuracy. Allmem/init/syncpass noleaks. Normal tests/timings actualQoco42, neverproxy.

Full numerical: nativeRuiz4,7landingrepeats,PD6N20/N500 conversion/KKT/eightmetric/device-step comparisons pass. Fullproductionlanding mem/init/syncpass. Fullracecheck FAILS99 with38reports (all cudss::factorize_dtmn_ker) vsreference30. DO NOT say unchanged30 or fullyraceclean; countsarenotseverity. No reportsuppressed. Needresolvevendor issue before fullproductionqualification.

Profiling --profile-setup now also reports LU_NNZ afteranalysis. FLOPS query absentinstalled0.7.1 (only compiled>=800); recorded -1/status4/bytes0 meansUNSUPPORTED. Baselineprofilev37NNZ1089492 vspermonlyv38 976632 vsfinaltreeprofilev46 976603 atN500. Finalprofile46 frozenprivate /home/angus/build-qoco-gpu-trajectory-tree-profile-v46/final/libqoco.so, records qoco-gpu-tree-factor-profile.json. Onefenced N500diagnostic:createCSR stageincludesGPUordering/tree/submission3.046ms, vendorreorder10.951ms,symbolic49.514ms. Profilingisnotmatchedspeeduptiming.

Intermediate frozen versions: v32 failedboundedchainfront (epigraphs mislabeledroot); v33 fixedconepropagation butslowN500solve1.382s; v34 eliminateslocalvarsfirst, N500solve1.837s despite~10%fewerfactorentries; v39 samepermutation withduplicateCASwritefix +largertests; v40 adds tree butmin1breakstinysynthetic; v42 min2fixed. core32 wasborrowedmaps, core41 owns. v35profile failed compileFLOPS absent; v37/38 fixedversiongate. Native test prototypes v41/v43/v44 assumed infeasiblefixtureforcesnumericalfailure, but QOCO canreturn raw2SOLVED_INACCURATE/API0; don'tusebackendstatusasphysicsqualification. Finalproxyforcesexplicitfailure cleanly; independentcertificate remainsrequired.

Committed artifacts qoco-gpu-tree-{checkpoint,reproduction,pd3,pd6,pd6-500,20-oracle,500-oracle,factor-profile,input-20}.json plus representatives; qoco-trajectory-ordering-history.json archivesoldsource/probechecks; qoco-trajectory-factor-profile.json andolderqualifiedprobeoutputscommitted. AllpreparedQoco42hashesreproduce. docsGPU_QOCO_LOCAL_BACKEND.md andGPU_NATIVE_OPTIMIZATION_PROGRESS.md describecase-specificgainandlimits. Originalraw *.samples directories remainignored local (notcommittedduplicates).

Next useful work: tune tree depth/within-leaf ordering and cuDSS factorization algorithm to recover remaining16%solve regression while stayingGPU; addressdeterministicfactorizationrace (alternatevendoralgorithm mayhelp; do not disable accuracygates or suppressreports). Current max8/floorlog2N needexplicitconfigtoexplore; smallerfixturesregress andbackendselectionnotchanged. Freshnewbuildfoldersversions48+; no rebuildfrozen. WSLUbuntu-22.04; GPUflock/home/angus/.spacepdhcg-gpu.lock. PrivateCUDSSconfigprobe /home/angus/probe_cudss_levels with /home/angus/probe-cudss-runtime/libcudss.so.0 symlink onlyprivate (upstreamunchanged).

Ignoredhelpers build/performance build_qoco_trajectory_v42.py; validate_qoco_tree_final.py; measure_qoco_tree_final.py; repro_qoco_tree_final.py; record_qoco_tree_final.py; build_qoco_tree_profile_v46.py; profile_qoco_tree_v46.py; check_owned_stream_v47.py. ALL completed; freezehelpers mustnotrerun with existingdest. Mutable core41 buildlibrary mightdifferafterNVCCtestrelink; ALWAYS use frozen final/ba045... . No active sessions.

## GPU factor/runtime checkpoint c51fbbd (goal remains active)
- Eight frozen QOCO v48-v55 variants tested locally; NONE promoted. See committed artifacts/performance/qoco-factor-runtime-checkpoint.json for exact libraries/hashes, prepared backend snapshots, full checks, runtime paths and CMake caches.
- v48/v49 (cuDSS0.7 multiblock) fail landing initcheck in vendor preprocess_block_mapping; v50 standard kernels fail 6DOF certificate.
- Private /home/angus/cudss-isolated-0.8.0.10 installation only; shared runtime/upstreams/H100 unchanged. v51/v52 deterministic0.8 custom tree fail 6DOF illegal memory access; v53 vendor ordering fails landing with same fwd_dtmn_ker invalid shared read. v55 superpanels also fails 6DOF. Do not blindly repeat.
- v54 standard0.8 passes initial 20/500 probes, seven landing repeats, all numerical oracles and ALL FOUR FULL LANDING sanitizers (racecheck 0). BUT matched 6DOF first optimized warmup fails trust-region exhaustion: terminal 4.517e-6 > 1e-6, 1988 inner iterations. Candidate REJECTED. Planned 6DOF sanitizer helper was NOT run. No v54 N500 matched campaign. Landing-only 194.813->151.352ms (1.287x) is a rejected-candidate result.
- benchmark_qoco.py and benchmark_planner.py now support --baseline-cudss/--optimized-cudss with separate runtime directories and hashes. New prepare opt-in --multiblock-factorization and --superpanels; 0.8 tree enum guarded.
- Current reference stays frozen v42 + corev41, prior experimental N500 1.347x SCvx and known vendor race limitation. No default changed. Goal is NOT complete or blocked. No live GPU jobs.
- Next useful direction: conditioning/SCvx progress sensitivity and removing host numerical work. Standard factorization roundoff can alter SCvx convergence radically despite tiny final canonical residual. Failed result with full iterations: artifacts/performance/qoco-cudss08-v54-rejected-pd6.json. Do not relax physics gates to accept it.
- Current task changes committed c51fbbd; unrelated user changes remain unstaged, including these existing dirty memory files. Frozen artifacts must never be rebuilt/overwritten.

## c1ae043: GPU solve-state isolation (goal ACTIVE; progress, no blocker)
- Found real bug: pinned qoco_solve does not reset best_valid/best_metric/best_iter between CQP solves. restore_best_iterate can return previous problem vectors/status. Macro failed v54 repeated previous trajectory; direct regression changes x=1 equality to x=4 and old lib returns x~=1 with SOLVED_INACCURATE.
- scripts/gpu/prepare_qoco_gpu.py now ALWAYS resets per-solve best validity/counters/status in patched builds (unmodified early-return control preserved). --reset-solve-state accepted explicitly but redundant. GPU buffers and explicit primal warm start retained. New cpp/cuda/tests/qoco_solve_state_test.cu passes with 0/4 Ruiz, old v54 fails precisely stale-best assertion. New forcing_satisfied uses pure-QOCO exact acceptance tolerance (old factor5 mislabeled4 records; new0 mismatches,17 above-tolerance records covered).
- Frozen QOCO /home/angus/build-qoco-gpu-reset-v58/final/libqoco.so SHA4bd7c6408b4a90a00050992c0bfd9fce1616d66be1e60b9a99572b455cdebb99; standard cuDSS0.8.0.10 at /home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib. Frozen core /home/angus/build-spacepdhcg-reset-v58/final/libspacepdhcg_cuda.so SHA38beedb31dd7294583caace754ed0ff4334f362b83a35530abbde758b25cfb0c. Do not overwrite/rebuild frozen artifacts.
- v58 QOCO differs from v54 in only src/qoco_api.c. Native core only numerical-code difference from41 is forcing telemetry; all original numeric settings restored after failed trials. Source reproduces without explicit reset flag, unmodified control unchanged. See artifacts/performance/qoco-reset-v58-reproduction.json.
- Initial: native4, landing7, PD6N20 nine consecutive runs, N500 pass. Full independent numerical oracles at20/500 and landing7 pass. Matched 2warm+7alternating measured pervariant all qualify and objective within1e-8. Baseline v42/core41 vs v58/core58/runtime0.8 (combined comparison, NOT isolated reset speedup).
- SCvx medians: landing203.147->164.095ms1.238x; PD6N20 1169.896->806.350ms1.451x; N500549.632->462.553ms1.188x. Process1.101x/1.309x/1.058x. IMPORTANT N20 mean regresses1.173->1.252s and max1.204->2.887s. N500mean.555->.526s butmax.593->.689s. Keep runtime config EXPERIMENTAL/no backend default change; state reset is default correctness fix for new prepared builds.
- Full landing ALL4 sanitizers pass. Full PD6N20 ALL4 including racecheck pass. N500 mem/init/sync pass; racecheck500 NOT RUN. Standalone state regression ALL4 at Ruiz0 and4 pass. Rawresults and exactcommands in committed qoco-reset-v58-checkpoint.json. All GPU jobs terminal; no live test handles.
- Rejected this turn: core56 tighterIR10/1e-10 fails third PD6 repeat; core57 smallerAregularization1e-13 fails first. Cold starts only fail fourth. All reverted (native_qoco_adapter.cpp is unchanged fromHEAD c51). Old v54 PD6 mem/init/sync pass, racecheck timed out240s (not a pass).
- v59 deterministic0.8 + state reset still illegal memory onPD6; reject. Frozen /home/angus/build-qoco-gpu-reset-v59/final/libqoco.so SHA96c48feb6236a3df174977b9206141d5d09b0cb9febe84d983504e44ef44634e. No speedup claim.
- Helpers build/performance/*reset_v58*.py and *reset_v59*.py stored in checkpoint. Freeze/check helpers create final directories once; do NOT rerun them. test_reset_v58.py recompiles external unit, final frozen unit is separate. Repro source destinations already exist; choose fresh destinations.
- Next high-leverage goal work: remove host KKT assembly / CSC-to-CSR setup and host solution exports/warm-start copies; investigate remaining convergence variance, broader physics families before runtime default promotion. Whole GPU-native objective remains unfinished/ACTIVE. No subagents authorized. H100/Lambda/shared pinned upstreams untouched. Unrelated user files and dirty memories remain unstaged.


## GPU device IO and host Ruiz checkpoint — 30705df

Goal remains ACTIVE; this turn made concrete GPU IO/correctness progress, not
whole-pipeline completion. Branch perf/gpu-native-pipeline, commit 30705df.
Do not spawn subagents. Preserve unrelated dirty files and Lambda/shared runtimes.
No local GPU job remains active. Serialize future runs using the existing flock.

Final frozen pair (never overwrite/rebuild):
- QOCO /home/angus/build-qoco-gpu-device-io-v62/final/libqoco.so,
  SHA256 2445292ccef52ede26d7a5a370cb3f5f99f9b95e6dca644b24eb1cd2a2f4c58a.
- Core /home/angus/build-spacepdhcg-device-io-v63/final/libspacepdhcg_cuda.so,
  SHA256 b4bfba99813663f26e10d244381a2a7491d724a0cc7287b129fce8d79c3393c6.
- Same isolated cuDSS 0.8.0.10 standard kernels; no backend default promotion.

Implemented optional --device-io QOCO preparation: complete symbol negotiation,
device output without mandatory x/s/y/z host exports, accepted unscaled primal
saved D2D into existing x0. Cold/rejected solves preserve accepted cache. Legacy
host x0 overwrite invalidates GPU cache. No new GPU allocation/global cache.
Adapter skips host primal arrays except explicit CPU oracle; host export and
legacy libraries work. Accept/reset return errors. Final core v63 immediately
reports accepted D2D copies even without a subsequent solve; direct test checks it.

New correctness regression: legacy host Ruiz uploads matrices/scales but not
scaled c/b/h. Old v58 reports x=1.0442737824 for 2x=2, then x=0.6484197773 instead
of .5 after A update. Preparation now ALWAYS fixes vector sync in patched builds;
--unmodified stays untouched. Final scalar results agree to roundoff. Production
GPU scaling already bypassed this CPU calculation.

Validation: QOCO v62/core v61 full landing and N20 all four sanitizers pass;
N500 mem/init/sync pass (N500 race NOT run). All 16 standalone device-IO/Ruiz
sanitizers pass. Core v63 only changes telemetry and its test: native conversion
at FOUR RUIZ PASSES (argument 4 is NOT four repetitions), all four native
sanitizers, legacy compatibility, landing7, N20/N500 qualification + all oracles
pass. v62 also has nine plain N20 repeats. Explicit scope in checkpoint/docs.

Final measured pair v62/core63 versus v58/core58, same runtime: landing SCvx
188.758→186.763ms (1.011x), process .974x; N20 1.171099→1.027652s (1.140x), process
1.033x. Earlier v61 N20 batch regresses .796724→.864510s (.922x). Strong iteration
and latency variance persists. BOTH N500 campaigns STOP at unchanged 1e-8
objective comparison: v61 delta5.5537667e-8; final delta2.3926362e-8. Every physics
certificate gate passes; these are repeatability/equal-objective failures, not
trajectory feasibility failures. No general speedup/default promotion claimed.
Ten-repeat diagnostic also reproduces 8.6953563e-8 spread on UNCHANGED v58 control.
Do not erase failed campaigns or loosen the objective gate.

Final Nsight CUDA API counts landing: memcpy564→558, async422→424, stream sync83→87,
launch7046 unchanged, malloc398 unchanged. Device IO removes9 synchronous copies;
Ruiz fix adds3 setup copies; two accepted D2D copies and4 explicit waits added.
Discarded LD_PRELOAD trace could not see statically linked CUDA; not evidence.
Initial non-unit unit-test assertion caused exit leaks because the old Ruiz bug
fired before device mode; dedicated regression cleans up and confirms bug/fix.

Committed authoritative evidence: artifacts/performance/qoco-device-io-v62-checkpoint.json
(includes full failures, test scopes, frozen hashes, helper sources, old v61 source
override), qoco-device-io-v62-reproduction.json, v61/v62 pd3/pd6/pd6-500 campaigns.
Docs: GPU_QOCO_LOCAL_BACKEND.md and GPU_NATIVE_OPTIMIZATION_PROGRESS.md sections
Device solution ownership and host Ruiz consistency. Raw .samples remain ignored.
Builders/check/record helpers under build/performance/*device_io*.py. v60 was only
a failed source preparation; v61/v62/63 finals are frozen. For new work use v64+.

Next targets remain numerical conditioning/convergence variability and eliminating
CPU KKT setup/host control/production CPU independent replay, then broader physics
families. No numerical tolerance/refinement/regularization settings changed here.

### GPU KKT checkpoint f45761c
- Optional --gpu-kkt constructs KKT CSR and five maps from device CSC with multi-block kernels/CUB; exact CPU oracle and isolated direct test cover seven cases including duplicates and irregular/large cones. Pooled temporary allocations reduced 10 to 2; eight legacy uploads removed.
- Frozen QOCO /home/angus/build-qoco-gpu-gpu-kkt-v67/final/libqoco.so SHA256 67c8981047fdbe756a5b361f112ed56a6dc43b215416245f8b06115a29c860dd; unchanged core v63. Do not rebuild/overwrite v64-v67. Next private build v68+.
- Final full benchmark batches pass unchanged physics/objective gates. Mixed overall timing: landing 0.932x, N20 1.520x with fewer inner iterations, N500 0.990x. Direct pooling 15-sample landing 1.114x SCvx but setup flat. No broad speedup or default promotion.
- Seven direct cases, full landing/N20 and forced reconstruction pass all four sanitizer tools. N500 mem/init/sync pass; N500 racecheck not run. Independent KKT/conversion/audit/device-I/O oracles pass. No live GPU jobs remain.
- Source/test hashes, frozen binaries, reproducible helper sources, failures and full distributions committed in artifacts/performance/qoco-gpu-kkt-v67-checkpoint.json and linked reports. Docs have exact scopes. Existing dirty user files untouched/uncommitted.
- Goal remains active. CPU initial matrix creation/transposes/regularization, native setup, solver/SCvx control, production replay, convergence variability and broader families remain. Candidate next work: profile remaining setup and conditioning without loosening tolerances. Current GPU KKT rejects legacy --profile-setup. Shared runtimes/pinned upstreams/remote campaigns untouched.

### GPU update maps v68 / Lambda check (ee30543, 08c14c8)
- PROGRESS; goal ACTIVE/incomplete. Updated qoco_device_update.cuh removes CPU inverse-map/source/diagonal/cone scans: borrows stable A/G gather entries, CUDA builds P source/diagonal maps, copies existing device cone boundaries. Added-P indices still supplied by host regularization; a validation readback/sync remains. Context destroyed before solver cleanup/reconstruction (verified).
- Frozen candidate /home/angus/build-qoco-gpu-gpu-kkt-v68/final/libqoco.so SHA256 828f6b9249909134c10475dba3111b110bf03872b398c3ab98efc8db5a0c3ece, unchanged core63/runtime0.8. Baseline frozen v67. Do not overwrite v68; next build v69+.
- Three matched campaigns 2 warmups+7 measured each pass unchanged physics/objective gates. SCvx landing0.981x, N20 1.621x (iterations204 to112), N5001.001x. Setup landing37.426 to41.241ms, N20 80.001 to78.492ms, N500114.961 to132.524ms. Mixed, no overall speed claim/default promotion.
- Landing Nsight API counts: synchronous copies550 to545; allocations400 to398; free308 to307 (new empty temporary still cudaFree(nullptr)); launches7068 to7071; async copies424 to425; streams sync88 to89. Potential next cheap cleanup: guard Buffer's null free and redundant host Ruiz-vector repair now also done globally; benchmark before retaining. Larger remaining work CPU initial setup/transposes/regularization, control, conditioning.
- All numerical oracles landing/N20/N500 pass. Original8 and extended9 numeric-update fixtures (including1031 offdiagonal P with inserted diagonals) pass normal+all4sanitizers. Full landing all4; N500mem/init/sync; reconstructionnormal+all4. No N500race or N20fullsanitize this checkpoint. Extended test frozen /home/angus/build-qoco-gpu-gpu-kkt-v68/extended-test/qoco_gpu_numeric_update_test. All local jobs terminal, session93004exit0, no GPU jobs outstanding.
- Complete reproducible source/test/binary/evidence committed in qoco-update-maps-v68-checkpoint.json plus full distributions/reproduction; ee30543. User requested Lambda check: READ-ONLY SSH completed, snapshot08c14c8 artifacts/performance/lambda-status-2026-09-06.json. Remote source/runtime/jobs unchanged.
- Lambda at2026-09-06T01:02Z: source1dbcae0, G4 worker53138 live;137completed groups, nextordinal137 running.594 numerical+639timeouts,0contaminated. GPU100%,43C,1607MiB. SQLite total138 counts STARTED coordinates, not planned total396. Latestordinal136 completed5400.415s within5460s deadline. Old server53183 plus session observed. No stop/restart/new remote job.
- GTOC12v11 CPU campaign DONE at2026-09-05T21:19:54Z, official+independent fleet pass;23ships194asteroids14047.8kg,~3kg overv10. Not provenoptimal,weightedobjectivegap29.204kg (not collected-mass gap). Some archived columns fail official checks; finalselectedfleet passes. No finalization/merge/download of full result run requested/performed. Existing unrelated dirty user files preserved.

### GPU scratch arena v69 experiment (1735306): NOT QUALIFIED
- PROGRESS, goal ACTIVE/incomplete. Added explicit --vector-arena experiment in preparer + qoco_vector_arena.cuh, 26 post-analysis scratch vectors in one aligned GPU allocation, one device memset. Host mirrors calloc; individual vector frees honor arena_owned; solver frees arena after all vectors/vendor resources. No globals/TLS or cross-solver cache. Default without flag unchanged. Flag help explicitly says not qualified.
- Frozen /home/angus/build-qoco-gpu-gpu-kkt-v69/final/libqoco.so SHA256 353b4cc498f5e385dfc4364e98284e53d13fa614d5720c8f2cd7ec9d5254cb81. Core63, isolated cuDSS0.8 standard, controlv68. Preserve v69; next build v70+.
- Exact arena initialization/alignment/disjointness/tag tests cover4shapes (17m0,1031m257,33m1,65m3); three solvers coexist, middle teardown/reallocate preserves survivors. Arena + nine numeric-update tests + full landing ALL4sanitizers pass; fullN500 mem/init/sync passes, noN500race/noN20fullsanitize. Forced reconstructionnormal+all4 passes. Initial numericaloracles landing/N20/N500 pass. Reproductionexact; ruffpass.
- Qualified landing profile confirms allocations398->373, frees307->282, copies545->519, asyncmemsets565->566, streamsync89->90, kernellaunch7071unchanged. Landing batch2warmup+7measuredallpass,36inner throughout: SCvx169.559->161.358ms(1.051x), process1.026x, setup32.268->32.441ms(flat).
- IMPORTANT N20 matched campaign FAILED candidate measured repeat2 after6total samples. Physics certificate allpass, but objective0.51297605935915824 vsreference0.51297569119164033: delta3.681675179e-7 >unchanged1e-8.471inner iterations. No matchedN500campaignran. Do not erase/relabel/relax gate.
- Followup alternating20runs EACH: v68control20/20qualified, objective spread1.012112283e-9; v69arena19/20qualified, spread3.674969706e-7. All40physicscertificatespass. This reproduces candidate-specific qualification failure in this batch; do not dismiss as proven pre-existing variability. Cause unresolved despite clean ownership/sanitizers. Arena is experimental, NO productionpromotion/general speedclaim.
- Full scopes/helper sources/hashes committed in qoco-vector-arena-v69-checkpoint.json; complete landing,failedN20,pairedrepeatability,reproduction artifacts committed. Compact diagnostic retains allcertificates/counters/summary/hash references; rawfulloriginal retained ignored build/performance/vector-arena-v69-repeatability-full.json plus perrunfiles. Source/test/header matches compiledpreparedsource; preparer later help-onlystring changed, currenthashreportupdated.
- ALL local jobs terminal: build42585,check64456,finish17150,diagnose88525; no GPUjob remains. finishbenchmarkreturned1 (knownobjectivefailure), allsanitizer/recovery/profile stagescompleted0. No remote Lambda activity this turn. Unrelated userfiles preserved.
- Next priority: investigate numerical sensitivity/conditioning before promoting arena or chasing smaller setup changes. Candidate pooling mathematically identical zeros/layoutbounds but changedallocation/syncpattern; cause could still involve arithmetic scheduling or hidden synchronization, not established. Do not random-retry until passing and claimfixed. Existing cuDSS0.8deterministic kernels have known memoryerrors; do not blindlyretry. Initial CPUstats/input/scaling/transposes/regularization remain, plus hostSCvxcontrol/GTOC12pipeline. Whole GPUgoal unfinished.

### QOCO best-return probe and SCvx convergence correction (873bfc8, 22c3eb0)
- Goal ACTIVE and incomplete. No subagents; no Lambda mutation or recheck in this continuation. Preserve unrelated dirty files and all frozen runtimes. All local jobs are terminal; no GPU job remains.
- 873bfc8 adds opt-in --restore-inaccurate-best and test-only trace/policy/return-audit proxies. Fingerprints match across 24 initial v68/v69 solves, but iterations/results differ; no root numerical cause established. Eight repeats of 8 proxy variants all pass; tighter IR costs more. Direct v68/v70/v71 repeatability: 20/20,19/20,20/20 objective gates. v70 sample18 again misses fixed .51297569119164033 reference by3.68167585e-7 >1e-8. All physics pass. Policy and arena remain experimental; no matched timing campaign after this failure.
- Frozen QOCO v70 /home/angus/build-qoco-gpu-gpu-kkt-v70/final/libqoco.so SHA cadbcdb60e5fee1cde7a7313e1382602338cd257193e317fe0ac8faabd3acf0e (best return, no arena). v71 same path version71 SHA c9e4c2aaddfc568ad2994d18da9a344017c41a262ad6676173ef67f347a97d5f (best return+arena). Exact prepared-source reproduction. Both: state reuse Ruiz0/4, numerical unit/native and landing/N20/N500 oracles pass. Both state4 memory/sync and N20 memory/init/sync pass; v71 N500 memory pass. v70 full N20 racecheck TIMEOUT600s, not a pass; no orphan remains. v71 racecheck not attempted. 11 completed sanitizer checks plus one timeout, recorded in qoco-best-return-v71-checkpoint.json.
- Failure trace exposed outer stopping bug: 6 inner rejections shrink radius to .015625, then boundary step .9999992 of radius satisfies absolute .02 step tolerance. Final replay compares retained trajectory to itself, erases step and can overwrite iteration-limit/cancelled with converged.
- 22c3eb0 fixes device_scvx.cu: accepted step must satisfy existing step tolerance AND be inside existing near_boundary_fraction * radius_before to establish convergence. Retain that bool with accepted point; preserve actual accepted step through final replay; remove final unconditional convergence promotion. No tolerance or backend default changes. Correctness fix, not a general speedup.
- Frozen core72 /home/angus/build-spacepdhcg-boundary-v72/final/libspacepdhcg_cuda.so SHA 993fd2c0b0765b3696507aea086952177d0c686367d0b0c02f1f9707c9e2e241. Uses QOCO68 and isolated cuDSS0.8 standard for qualification. Core63 and other runtimes unchanged. Next new core/build version73+; do not overwrite72.
- Four native pytest regressions pass (tests/test_scvx_convergence_gpu.py): small radius requires interior step, two budget-limit variants, feasible zero-HCW pre-solve cancellation. Old core reproduces false certification after cancellation with zero outer iterations; new core returns cancelled and uncertified. Small-radius full memory/leak check0errors/0leaks. Native conversion4, landing7,N20,N500 with numerical oracles pass. No new GPU kernels and no new-core full race campaign.
- Targeted 60-run repeatability: small-radius input oldcore63/QOCO68 qualifies8/20; guardedcore72/QOCO68 qualifies20/20; original input guarded72/QOCO70 qualifies20/20. All60physics pass. Objective maxdelta3.7012e-7 /7.7509e-10 /8.5263e-10. Boundary medians .730661 -> .874078 seconds (old fails quality). Do not claim unresolved inner-solve variability eliminated.
- Original-input full benchmarks2warmup+7measured each,54samples allqualitypass: landing176.179->171.186ms (36inner); N20 570.046->648.000ms (102->112inner); N500458.284->465.257ms (34->36inner). Mixed timing, no general speedup. Exact evidence in scvx-boundary-v72-{checkpoint,checks,cancellation,repeatability,pd3,pd6,pd6-500}.json. Checkpoint embeds helper sources/frozen hashes. Ruff and git diff checks pass. No automatic push.
- Next work: return to GPU-native hot paths (initial matrix/transposes/regularization/statistics, host solve/SCvx control) and conditioning; production replay/GTOC12/batching remain. Prefer core72 guarded convergence for new experiments. QOCO68 unpooled remains control; v69/v70/v71 not promoted. Never loosen physics or1e-8 objective gates or retry until lucky pass. cuDSS deterministic kernels have known memory errors; do not blindly retry.

### GPU transpose construction v73 (a8b58d1)
- Goal ACTIVE/incomplete. Verified progress: --gpu-transposes replaces both qoco_setup CPU create_transposed_matrix calls and inverse-map loops with CUDA construction using source->gather stable order. New qoco_gpu_transpose.cuh owns all result arrays; no pointers borrowed across source lifetime. Host CSC mirrors and inverse maps are still eagerly downloaded for legacy compatibility. No backend default changed; no overall speedup claim.
- Frozen /home/angus/build-qoco-gpu-gpu-kkt-v73/final/libqoco.so SHA620d3f8c7387c7a344a087c12293d82cbf9588ab14c242a1d8a153322fcf3a51. Built standard v68 flags plus --gpu-transposes (no arena/no best-return). All qualification uses guarded core72 and cuDSS0.8 standard. Preserve73; next candidate74+.
- Exact source reproduction passed. Eight matrix fixtures test null/empty/zero rows/zero columns, unsorted/duplicate entries, signedzero, empty columns, up to1031x1537 and11853entries. Poison source HOST p/i/x; build two GPU results; destroy source+sibling; verify surviving host/device arrays and inverse map bitwise; double transpose after original source destruction verifies own gather. Direct linked symbol prevents silent baseline fallback. All4sanitizers pass these fixtures.
- Existing nine numeric-update cases, native conversion4, landing7, fullN20,N500 with independent numerical oracles pass. Full landing memory/leak; N500 memory/init/sync pass. Forced recovery/reconstruction after caller indices+stream destroyed passes normally+all4. No new fullN20sanitizer/N500race. All jobs terminal (build56980/check terminal/measure69535/sanitize49110/recovery61042 etc). No Lambda actions. Ruff/diff checks pass.
- Full matched benchmarks2warmup+7measured each: all54qualitypass. SCvx landing157.373->171.571ms, N20 599.820->1059.803ms (104->193medianinner), N500479.141->486.956ms (34->34medianinner). N20 slowdown remains unresolved numerical sensitivity; do NOT promote as faster. Complete setup medians34.099->32.967ms /79.721->78.261ms /133.406->116.303ms are observations, not proof of isolated phase savings.
- Isolated SAME-library/source construction benchmark has2warmup+21measured/strategy/shape, excludes result destruction and includes completion/mirrors. 9kentries:CPU .213468ms vsGPU .325258ms(.656x);225k:2.921234 vs2.019762ms(1.446x);900k:15.303359 vs7.697880ms(1.988x). CPU strategy means CPU transpose+normalGPUmatrix constructor, not CPU-only matrix copying. Large phase speedup is real measured evidence, not total solver multiplier.
- Qualified landing nsys confirms copies545->547, async425->427, malloc398->400/free307->309, kernels7071->7073, streamsync89unchanged. Six old H2D matrix-array uploads replaced by8D2H mirror/map downloads;2D2Doffsetcopies+2entrykernels;2temporary inverse allocations. Eager host mirrors/temp maps are next optimization target, especially small matrices. Investigate lazy host materialization through get_csc_matrix and legacy update paths carefully; raw AtoAt/GtoGt host maps need compatible behavior. CPUinitialstats/Pregularization and hostcontrol still remain.
- Committed artifacts qoco-gpu-transpose-v73-{checkpoint,microbenchmark}.json, qoco-gpu-kkt-v73-{pd3,pd6,pd6-500,reproduction}.json. Checkpoint embeds helper sources, all checks/profiles/frozen hashes. Helpers under ignored build/performance/*gpu_transposes_v73.py and benchmark_transpose_v73.py. Exact prepared source remains /home/angus/build-qoco-gpu-gpu-kkt-v73/source. Unrelated working files preserved; memoryappend not staged.

### GPU-native goal checkpoint: v74 lazy host mirrors (e1e770a)
- Goal remains ACTIVE/incomplete. No subagents. Preserve frozen binaries; next candidate v75+. Unrelated user edits remain unstaged. No push.
- Optional --lazy-transpose-mirrors requires --gpu-transposes. Host At/Gt arrays allocated/downloaded on explicit CPU accessor; numerical updates invalidate cached values, topology retained. Skip redundant uploads while device owns values. Independent GPU ownership preserved. Physical At/Gt arrays, gathers, inverse temp and host map download still remain.
- Frozen QOCO /home/angus/build-qoco-gpu-gpu-kkt-v74/final/libqoco.so SHA256 6205b2d2b0ff4e2710634e4a96dbfc0b690b3de55ab726fd1d0134d7747e2df2. Core72 SHA993fd2c0b0765b3696507aea086952177d0c686367d0b0c02f1f9707c9e2e241. cuDSS0.8 standard. No arena/best-return/default promotion.
- 54 matched trajectory samples qualify unchanged physics/objective: v73->v74 landing155.585->156.167ms(36->36inner), N20 720.336->726.110ms(129->126), N500465.788->478.443ms(34->36). No general speedup.
- Synthetic900k constructor CPU13.871560ms, eagerGPU7.110569ms, lazyGPU3.086033ms;225k2.760474/1.952241/1.502419ms. Lazy defers host mirrors; phase only, not end-to-end.
- Qualified landing trace: memcpy547->535, async427same, launches7073->7067, device sync65->63, stream sync89same, malloc400/free309same.
- Checks:16transposefixtures+9numeric-updatecases all4sanitizers; native/landing/N20/N500 numerical oracles; full landingmemory/leak and N500memory/init/sync. Forced caller-index/stream-release recovery normal+all4 passed. No fullN20sanitizer or N500racecheck. Exact source reproduction and Ruff pass.
- All evidence/helpers embedded artifacts/performance/qoco-lazy-mirrors-v74-checkpoint.json. Docs appended. No active test processes at checkpoint.
- Fresh read-only Lambda check02:09UTC: H100100%,44C,1607MiB; G4worker53138/server53183,140groupscomplete ordinal140running;594numerical666timeouts0contamination. Three new groups since priorcheck alltimeouts. GTOC12v11stilldone23ships194asteroids14047.8kg bothverifierspass,weightedgap29.204kg notoptimal. Campaign/source unchanged. Evidence lambda-status-2026-09-06-followup.json.
- Next architecture lead (unimplemented): actual GPU solver arithmetic/KKT uses sourceA/G directly; At/Gt consumers only legacyhostRuiz/qoco_update, test-onlyKKT/maporacle, and GPUupdater maintenance. Consider solver-owned deferred physical transpose, materialize only legacy/debug access. Need safe source lifetime and update invalidation; standalone eager/lazy constructors retain independent ownership. Do not silently break private-struct debug probes/hostmap compatibility. Broader goal still includes CPUsetupstats/Pregularization, hostsolve/SCvxcontrol, replay,GTOC12,batching and conditioning.

### GPU-native goal checkpoint: v75 deferred physical transposes (b081931)
- Goal ACTIVE/incomplete. This turn made verified progress; no blocked state. No subagents or Lambda actions. Unrelated user dirty files preserved. No push. No live tests left. Next runtime version76+; never overwrite frozen versions.
- Added optional --deferred-transposes (requires lazy-transpose-mirrors+gpu-kkt). SourceA/G already serve GPU matvecs/Ruiz/KKT. CompatibilityAt/Gt now retain reference-counted sources and dimensions; no physicalGPU/CPU arrays, gathers or map values until explicit accessor/matvec. Device updater skips transpose_values and invalidates optional caches; later access refreshes values, retains topology. Host inverse-map capacity still allocated, borrowedpointer lifetime ownedQOCOProblemData. Standalone eager/lazy constructors still independent; deferredview is internal, not arbitrary auto-updating view.
- Frozen QOCO75 /home/angus/build-qoco-gpu-gpu-kkt-v75/final/libqoco.so SHA1f4208e21bf0441607a79d0cc5c919cb466cc18083eb393749cfb29b4aaa0dc4. Core72 unchanged (SHA993fd2c0b0765b3696507aea086952177d0c686367d0b0c02f1f9707c9e2e241), cuDSS0.8 standard. No arena/bestreturn/defaultpromotion.
- 54matchedbenchmark samples v74->75 allqualify unchangedphysics/objective: landing173.335->169.569ms(36->36inner), N20 1102.284->923.818ms(206->166), N500464.399->473.573ms(34->34). Setup32.447->33.914/83.625->91.056/127.149->110.651ms. No general speedup; N20iterationvariability remains.
- Qualified landingtrace removes24allocations/frees(400->376/309->285),4sync copies535->531,2asynccopies427->425,10kernels7067->7057,2devicesync63->61. Streamsync89unchanged. This confirms unusedphysicaltransposes gone from normalproduction path.
- 25transpose/lifetimecases=8eager+8hostlazy+8devicedeferred+1deferredchain; all4sanitizers. Numerical9cases, native/landing/N20/N500 oracles pass. Full landingmemory/leak, N500memory/init/sync run WITHOUT materializingoracles. Recovery forcedreconstruction aftercallerindex/streamdestruction passesnormal+all4 bothwithoutandwithKKToracle(10checks). No fullN20sanitizer orN500racecheck.
- Initial numeric sanitizer exited11,0GPUerrors: outdated unit test dereferencedCPUreference emptyAt/Gt absentd_csc_host. All-oracle initialcheck had hidden it bymaterializingKKT. UBSan confirmed testline103nullaccess. Fixed testnowrequestsget_csc_matrix(pair.second) priorprivateinspection. Newfrozenbinary qoco_gpu_numeric_update_access_test passeswith/withoutKKToracle+all4sanitizers. Original qoco_gpu_numeric_update_test remainsfrozen and is unsafewithoutKKToracle; doNOT useforfuturetests. Allfuturecompilefromcorrectedsource. HostASan diagnostic failedCUDAalloc andisNOTvalidationpass. Failure+diagnostics retainedcheckpoint.
- Provenancefilelist had omitted3transposeheaders; fixedpreparemanifest. Completefreshsource reproduction v75-complete-reproduction compares ALL3 plusotherpreparedfiles againstuntouchedfrozenruntime source; passes. Earlierpartialreproduction retainedasevidence; no runtime rebuildneeded(provenance-onlychange). Ruff+diffcheckpass.
- Evidence/helpers: artifacts/performance/qoco-deferred-transposes-v75-checkpoint.json includesallrawchecks/trace/failure/debug/helpertext. Benchmarks qoco-gpu-kkt-v75-{pd3,pd6,pd6-500}.json. Completepreparedreproduction qoco-gpu-kkt-v75-complete-reproduction.json. Docsupdated.
- Nextmajorlead read-only: qoco_batched_stopping.cuh/qoco_gpu_metrics emits manyfixedkernels+cuBLASdots everyinneriter, only8scalarscopiedend. CUDAgraph replay this repeated numerical phase could cut launch overhead substantially before movingouterhostdecisions. Need dedicated capturestream; existingops alluselegacydefaultstream, cannotcaptureitdirectly. Add scopedoperatorstream for participating kernels/gather/norm/cuBLAS, captureonnewstream then replaygraphondefaultstream tokeepordering. Preserve allmath/operators/cuBLASsum and scalarreturn exactly; warmupcuBLAS beforecapture, checkcapturecompatibility ratherthanguess. cuda_funcs maylackcublasSetStream loading(needsinspection). No graphcodeimplemented.
- Graphownership constraints: qoco_reduction_scope.cuh threadlocalhandle/scratch scopedtoenclosing solve (outerdepth1; metricsbegin=>2). Sharedscalarworkspace maygrow/free betweenmetrics and otherops; graphcache mustinvalidateonbufferchange and destroybeforeworkspace/handlefree. Paramkey mustinclude solver/work pointers,dimensions,buffers,k/kinv/reg (hostscalar kernelparams vary acrossscales/solves). Unscoped singlemetrics calls(depth1) shouldnotcachebeyondcall. Hoststop decisions/check_stopping remain outsidegraphinitially; fullGPUcontrol andinitialsetupstats/Pregularization/replay/GTOC12 stillrequired. Do not equate graphphase withfullgoal.

### GPU-native goal checkpoint: v76 metric CUDA Graphs (bf525e1)
- Goal ACTIVE/incomplete; verified progress. No subagents, Lambda actions, push, or unrelated staging. No live processes left. Next candidate77+. Frozen versions remainuntouched.
- Added --metric-graphs requiresbatched-iteration-scalars+gather. Captures existing stopping/objective/mu GPUops; eight-scalar D2H andhostdecisionsremainoutside. Captureon2dedicatednonblockingstreams, replayondefaultstream; cuBLASpointermode/stream restored. Scope-owned two caches(6metric/8iterationvariants), firstcallnormalprimeslibrary; subsequentcapture/replay. Key48pointerarray(solver/work/data/scratch/handle/vector/matrix/gather),3dims,3scalars(k/kinv/reg). Scratchgrowthreleasesgraphs beforefree; outerend sync,releasesgraphs,scratch,cublas. Unscopedno retainedgraph. Testdisable SPACEPDHCG_TEST_QOCO_METRIC_GRAPH_DISABLE=1 givesuncapturedGPUcalculations.
- Prepared patch streams only participating ew_product/scale/axpy/SpMtv/gather/norm/finish; SpMtv zeroingnowcudaMemsetAsync onselectedstream. Added checkedcublasGet/SetStream functionptrloader. Changesmanifestincludesgraphheaderandallpriortransposeheaders. Completefreshsource reproductionexact; Ruff/diffchecks pass.
- Frozen /home/angus/build-qoco-gpu-gpu-kkt-v76/final/libqoco.so SHA e77a1ea2cee34b693eff031b05967c6ece56c2d3c531d686eb26925cec4bd1ff. Core72unchanged, cuDSS0.8standard. Baseline75SHA1f4208e21bf0441607a79d0cc5c919cb466cc18083eb393749cfb29b4aaa0dc4. Alltestexecutablesinfinal.
- 54fullmatchedsamples qualify: landing162.590->152.054ms (36->36inner), N20 725.136->1159.010ms(134->254), N500577.496->434.837ms(59->35). Setup29.257->30.748/72.310->71.291/102.348->100.142ms. Landingcandidate656.422msoutlier retained. No generalspeedup/defaultpromotion; iterationvariabilityunresolved.
- Isolatedsame-librarymetrics benchmark: 3primingcallsseparatelytimed,50timedcalls/sample,2warmups+7measured/strategy/size. ScalarD2Hretained; fixture+scopecreation/destructionexcluded. n17uncaptured448.926us ->graph119.059us=3.771x;4103 537.663->269.414us=1.996x;1000003551.140->3298.539us=1.077x. Graph/uncaptured90changingxinputs(30eachsize) all8outputsBITWISEIDENTICAL. Not universalbitwiseproof.
- Qualifiedlandingtrace: hostkernellaunches7057->5357(-1700),36graphreplays2captures; streamcreate/destroy2->4. Copies531/async425,malloc376/free285,streamsync89unchanged. cudaMallocAsync16/cudaFreeAsync16 addedfromcuBLAScapture; capturesusememorynodespotentialdevice-launchlimitation. DoesNOTreduceactualGPUarithmetic1700kernels, onlyhostlaunchcalls.
- Validation11sanitizerchecks:4graphmetricfixtures(n17/4103,absentconstraints,zeroP) all4tools; fulllandingall4; N500mem/init/sync. Testschangevalues/inputpointer/scratchcapacity/k/kinv includingk0,verifygraph/streamlive0afterouterend,nestedscopes,unscopedcalls,ordinarydotaftermode/streamrestore. Numeric9cases/native/landing/N20/N500oraclespass. Recoveryforcedreconstructioncallerindices/streamreleased passesnormal+all4 bothKKToracleoff/on(10). No fullN20sanitizer/N500racecheck claimed. No failures. Paritycompile originallywarned unusedbenchmarkmainstubreturn; explicitreturn0added afterwards (unusedstub, no GPUcodechange), finalsourceclean forfuturebuild.
- Evidence artifacts/performance/qoco-metric-graphs-v76-checkpoint.json embedsallhelpers/checks/traces+microbenchmark+parity. Separate qoco-metric-graphs-v76-{microbenchmark,parity}.json; qoco-gpu-kkt-v76-{pd3,pd6,pd6-500,reproduction}.json. Helpers build/check/measure/sanitize/recovery/reproduce/profile/benchmark/parity/record_metric_graphs_v76.py underbuild/performance(ignoredbutembedded). checkhelpercompilesqoco_gpu_stopping_test andnumericupdate fromcurrentcorrectedsource, nooldv75emptydescriptorbug.
- Nextperformancelead: eliminatecuBLASasyncallocnodes duringcapture usingcache-owned cuBLASworkspace (checkedcublasSetWorkspace afterSetStream, destroysafelybeforehandle/scope ends). NVIDIAcuBLAS12.8docs graphsection saysuserworkspace avoidsgraphmemorynodes; SetStream resetsworkspace config, needrestore/clear workspacebeforefree. Thismayreducecoldcapturecost (trace16allocs one8msallocation), retainmath/bitwiseparity. Fullbestnextscope also convergence/conditioning toaddresslargeN20variance. qoco_ruiz_iterations CAPIexists nativeadapterinitialsetupforces0 thenGPUupdatesrequestedcount; plannerdoesnotexposefieldyet (rgcpp/tools nohits). Donotretunetolerances/objectivegate tohidefailure. Hostsolve/SCvxdecisions/CPUsetupstats/Pregularization/replay/GTOC12/batchingremain. AllGPU-nativeobjectiveintact.
- OfficialCUDA12.8 docschecked: https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cuda-graphs-support and https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__STREAM.html (defaultstreamcannotcapture, device-dotoutputs supported, cublasgraphmemorynodes). Linksindocs/checkpoint.

## GPU Ruiz experiment checkpoint — 9130d68
- Goal remains ACTIVE and incomplete. No push; no Lambda reads/mutations this tranche. Preserve unrelated dirty work. Next runtime version79+; do not overwrite frozen planner77/core72/QOCO78.
- Planner77 exposes optional solver.qoco_ruiz_iterations integer0..100 (default0), forwards native option and exports requested/applied (null without workspace). Native C++ option smoke and 43 Python schema tests pass.
- QOCO78 fixes empty/tiny equilibration denominators (existing <=1e-15 threshold) to identity instead of DBL_MAX. Inverse-scale reciprocals unchanged. Ten zero-norm/zero-objective cases with 30 updates and independently known optimum reproduce failure on QOCO76 and pass78. Existing nine numerical-update parity cases/native/landing/N20/N500 oracles pass.
- Frozen planner: /home/angus/build-spacepdhcg-ruiz-v77/final/spacepdhcg_plan SHA aa1e8a1e6e21c0c0e43c47a6f06d33f2979af327f674297fb5b2d5875497a390. Links unchanged core72. QOCO: /home/angus/build-qoco-gpu-gpu-kkt-v78/final/libqoco.so SHA bddec9453b32242e901919443a8df11cbfb3878218035a836930d0858ee0cb30. cuDSS0.8 unchanged. All final binaries read-only.
- Baseline QOCO76 N20 pilot24 attempts: all20 nonzero-pass setup attempts fail, zero-pass4/4 qualify. Corrected78 N20 pilot24: 0/1/2/4 each4/4 qualify,8 fails3/4,12 fails2/4. N500 pilot0/1/2/4 all16 qualify but slower with scaling: medians564.648/653.452/634.966/651.418ms.
- Longer N20 alternating0/1: 2 warmups+20 measured each. Zero qualifies22/22, median728.445ms, p95nearest-rank1867.149ms,max2285.965ms, median146.5inner. One-pass fails2/20 measured (repeat9,20) and qualifies20/22 total. REJECT general one-pass/default promotion. Corrected78 total84 attempts77qualify7fail. No general speedup claimed.
- All15 sanitizer checks pass: all4tools zero fixtures, numerical-update fixtures and full N20 one-pass; memory/init/sync full N500 one-pass. Instrumented trajectories also fixed objective <=1e-8. No N500racecheck. Complete prepared-source reproduction/Ruff/diff checks pass. No owned GPU jobs remain running.
- Evidence artifacts/performance/planner-ruiz-v78-checkpoint.json with source/runtime hashes, checks, helper_sources and v76 checkpoint helper-source inheritance. Full sweeps and prepared reproduction committed. Helpers build/performance/{build_planner_ruiz_v77,build/check/probe/sanitize/reproduce/record_ruiz_zero_v78}.py retained and embedded. New reusable runner scripts/gpu/benchmark_planner_ruiz.py.
- Next useful work: improve unreliable convergence/regularization with unchanged gates; investigate failed-path timing (failed SCvx result fields can remain0 despite nonzero iterations, external process_seconds retained); optional cuBLAS graph-owned workspace to eliminate16capture async alloc/free nodes remains unimplemented. GPU Ruiz timing currently included setup/update, legacy scaling_seconds does not isolate it. Host setup/decisions, CPU independent replay and GTOC12/batching still outstanding for fullGPU goal.

## Failure timing and warm-start checkpoint — ca1923f
- Goal ACTIVE/incomplete. Previous turn progress. No push or Lambda access/mutation. All owned build/GPU sessions terminal. Preserve unrelated dirty work. Next runtime version80+.
- Core79 /home/angus/build-spacepdhcg-timing-v79/final/libspacepdhcg_cuda.so SHA8155b5f837a22874fc1bfd00d02af0b7048d64ac0da7f3451db34827c7037fa6, read-only. Planner77/QOCO78/cuDSS0.8 unchanged.
- device_scvx.cu now uses host ResultTiming scope destructor for SCvx/CQP totals on all exits after timed scope starts. Failed pure-QOCO generic update/solve/residual times assign cumulative adapter values. No numerical/GPU synchronization changes. Broader failure allocation/transfer telemetry, hybrid accounting, failed-QOCO-setup component times remain unrepaired.
- New Linux interposer scvx_qoco_failure_timing_probe.cpp completes actual GPU work then injects failure on first/second native QOCO call. tests/test_scvx_failure_timing_gpu.py checks status staysfailed, no certificate, cumulative times and inner counts. Baselinecore72 has2expectedfailures,1pass. Core79 has7passes including4existing convergence/cancellation tests. Initial bad test assumption (first valid inner solution alwaysaccepted) failed once; corrected to count completed work independently. Initial+validatedlogs retained. No new sanitizer campaign for host-only accounting.
- benchmark_planner_ruiz.py accepts --warm-starts primal none; variantscomposedwith--passes; settingskeys count:mode whenmultiple modes. Default singlemodekey remainscount; generated filenamesnowincludewarmmode. All72experimentalruns havepositive correcttiming incl6naturalnumericalfailures0.47-0.70s.
- 2warm+7measured each: N20 0:primal9/9qualmedian776.836ms156inner;0:none9/9qual819.654ms176inner;1:primal5/9qual;1:none7/9qual. N500 all36qual:0:primal426.645ms34inner;0:none430.309ms38inner;1:primal637.819ms71inner;1:none542.291ms54inner. Cold starts do NOT cure one-passfailures orimproveunscaledmedian materially. No defaultchange/general speedupclaim.
- Artifacts core-timing-v79-checkpoint.json, planner-warm-start-v79-{20,500}.json committed with hashes/logs/helper_sources. build/performance/{build,check,validate,record}_core_timing_v79.py embedded. Ruff/diff pass. Goalnotcomplete.
- Useful nextwork: actual kernel/control architecture beyond parameter sweeps. Remaining gather_scvx_candidate_kernel in device_scvx.cu has per-thread gridloop absent blockIdx and3 launches<<<1,256>>>; independent gather can be distributed after proper parity/measurements, no cross-thread synchronization. Nonlinear replay remains largely serial (PD6 warp-specialized path exists), cannot parallelize intervals naively. Metric CUDA Graph cuBLAS user workspace proposal stillunimplemented (v76 trace16asyncalloc/free capture nodes). Conditioning/linear-solve convergence remainslarger latencyissue; simpleRuizandcoldstartknobs ruledout asgeneral fix.

## Lambda download and viewer handoff — 2026-09-06 03:28 UTC
User requested offloading tests if Lambda free, then downloading results and displaying them in the existing web visualiser with full paths/copy-paste instructions. Fresh H100 check busy at 100% with G4 worker 53138/server53183/session active. Left campaign untouched. CPU-only gather benchmark successfully compiled sm90 in /home/ubuntu/spacepdhcg-offload/gather-v82-20260906; guard refused GPU tests (75), source/build evidence artifacts/performance/lambda-gather-v82-staged.json. No new H100 measurements.
Downloaded 1,028 files, verified each SHA256, results/lambda/2026-09-06. Completed GTOC12 v11 fleet and partial G4 (145 completed, one active;594 numerical711timeout). Archive f368123d0abfdb6d26b2bc820d518edc4e78a8b244147891422aede7d2a1278d. Original fleet verifier passes;23ships196visited194collected14047.802874743327kg. Existing viewer copied into results/lambda/2026-09-06/visualiser; import and check pass; dataset SHA0ebd0dfaf0b483418e8ebd261eac2671528cac283429216777ecda79a55deb93. Displayed in Codex browser tab1, marked deliverable; active WebGL2 RTX5090, epoch69807, manually set Z exaggeration1 (query parameter was ignored). URL http://127.0.0.1:4178/?dataset=gtoc12&epoch=69807&preset=oblique. Read README.md in snapshot for full paths/instructions.
Viewer node server is the only owned running local process, exec session41800 (intentional user-facing service); no local GPU solve running. Hidden background launcher command was rejected by policy; used ordinary foreground exec node server successfully, no launcher file created. First snapshot archive failed before data due Python3.10 tarfile stream compresslevel unsupported; preserved failed empty first archive, retried without option successfully. Snapshot local only; no source/dependency/runtime remote mutations except new isolated gather staging folder. Parent goal remains active. Core82 gather code/docs/checkpoint tranche still awaiting final checkpoint/commit from pre-steering work; all tests completed as prior note. Next fresh runtime83+. Do not overwrite frozen82 binaries. No subagents or push.

## September 6 publication
User explicitly requested committing all outstanding changes, merging to main and pushing to GitHub after the viewer was running. This supersedes the earlier instruction to leave the Lambert/H1 dirty files unstaged. Reviewed changes: Lambert broadcast/cache parity optimization; truthful H1 telemetry/re-export and fixture; multi-block gather and tests; all prior GPU branch commits; performance docs and complete downloaded Lambda snapshot with existing viewer copy. Remote main had no divergence (0 behind/36 ahead before this commit). Running Lambda campaign remains untouched. Results byte-preservation attributes added so Git CRLF conversion cannot invalidate download/viewer hashes. Full GPU-native goal remains active; GTOC12 CPU refinement, convergence robustness and host decisions are unfinished. No new GTOC12 GPU migration implementation was started before this publication request.

Publication validation: 199 GTOC12/H1/planner-schema tests passed in 352.83 s, no skips; Ruff passed. Seven viewer importer/server tests passed; imported fleet check passed. All 1,028 raw downloaded files plus generated viewer dataset rehashed from the Git index successfully. Gather sources match the frozen v82 test checkpoint. git diff --cached --check passes; byte-exact raw evidence is exempt from whitespace rewriting, normal viewer source retains standard whitespace checks. Ready for the requested main merge/push; inspect Git refs for the actual final publication state.

## GTOC12 v85 checkpoint
See latest DEVLOG entry and artifacts/performance/gtoc12-discretisation-v85-checkpoint.json. Goal remains ACTIVE. Final247 tests and sanitizer checks pass; measured representative full leg14.2% faster, not complete GPU-native refiner. CPU conic assembly/Clarabel next. Additional free-v-infinity case fails qualification on both CPU/GPU and remains unresolved. Runtime85 frozen (next86+), all GPU jobs terminal. No Lambda access; existing viewer remains on4178. Branch perf/gtoc12-native-refinement originated from834420d3; publication under user's existing main/push authorization is in progress, inspect refs before continuing.

## GTOC12 conic v87 checkpoint
Goal ACTIVE. Native fixed-topology GPU conic assembly is implemented and opt-in; default unchanged. Full regression284 passed, native all-four and host memory/init/sync sanitizers pass. Coefficient path66.92x/64.12x faster atN2001 versusGPUdynamics+CPUassembly, but representative full leg23%slower (151.595vs123.214ms) dueClarabeliterationpath. All33timedlegsqualify. v86thrustfailure rejected; v87structuralmass-row zero removal restoresqualification. Five additionalpairsqualify; free-vinf casefailsboth. SeeDEVLOG/docs/GTOC12_GPU_ASSEMBLY.md/checkpointJSON. Frozen87(next88+), allGPUjobs terminal, noLambdaaccess. NextdirectGPUQOCOconnection mustexposetolerance1e-9 andexplicitlygateauditresiduals. Publicationinprogress onperf/gtoc12-native-assembly from405ad6fb; inspectrefs.

## GTOC12 GPU QOCO v93 checkpoint
Goal ACTIVE. Native retained conic->QOCO connection implemented as explicit experimental backend with unchanged external1e-9 residual/globalobjectivegap gates and independentphysics. Legacy default/API unchanged. ZOH redundant finalGamma/SOC removal retains original feasible set. v88-92 failed variants retained; v91coast numerical sensitivity unresolved. Final93 matched18/18 ZOHlegs qualify fixedmass1e-5kg, but GPUQOCOmedian818.922ms vsCUDAassemblyClarabel145.656ms (5.62x slower), so no speed/default promotion. Final regression297passed and hostlifecycle4memcheckpass; finalnative/hostsanitizers/checkpointrecorded separately. Timings/counters cumulative, usefinalsnapshot/deltas. NewGPUgapguard needed because localresidualaudit couldmissobjectiveerror. Nextconiciterationcost/convergence:solve-region3.764s of4.047s aggregatewall. CPUsetup/outer/seed/verification andbatchingremain. Frozen88-93(next94+); noLambdaaccess, existingverifiedviewer4178unchanged. DEVLOG/docs/GTOC12_GPU_QOCO.md and checkpoint holdfullsources/reproduction/results. Userauthorizedmainmerge/push; inspectGitrefsforactualpublication.

## Fused KKT v94 checkpoint
GoalACTIVE; priorprogresspublishedmain dce23ed8. Newoptional --fused-kkt-product makes sparseKKToneGPUmultiblockkernel, originalroworderbitwise72cases, all4operatorsanitizerspass. Defaultbuildersunchanged. Isolatedoperator1.7-7.9xfaster; fulltimingsmixed(initial870vs819ms; subsequentall40qualifiedpaired959.555->805.680ms16%less). Strictcoastteststillfails1.1663e-9>1e-9;23otherfocusedtests pass; no retry-to-pass, NOgeneralpromotion. Native7landing/N20/N500certificates+fixedobjective1e-8 pass. CPU24tests passafter2shellCRLFnormalizationfixes;CMake/Ruff/reproductionpass. Runtime94frozenab19af76fcc7ba9b9bf5d563ea8f56450e5b7db516bf97d7426d0ba269a98349 withcore93;next95+. Alljobs terminal,noLambdaaccess. docs/QOCO_FUSED_KKT_EXPERIMENT.md/checkpointJSONfullsources/evidence. NextGPUconditionalIRcontrolrequirescapture-compatiblepersistentcuDSSmemory+streampropagation;CUDAconditionalbodiesnoallocationnodes. APItraceonly, GPUactivitydataabsent. KeepgoalwholeGPUsetup/outer/batching/physics intact. CheckactualGitrefsforlocalexperimentalcheckpointpublicationstatus.


## Conditional cuDSS / reset checkpoint, 2026-09-06
Goal ACTIVE/incomplete. Current experiment diagnostic-only, no production IR integration or speed claim. QOCO95 late-stream capture misses solve kernels;96 early stream captures but conditional host-copy restriction;97bitwise/98scaled GTOC12 comparisons fail.99 GPU snapshots of2eight-byte range inputs permit conditional WHILE controls:120checks over15real systems(allpass), landing70/70strict1e-12scaled replayparity;GTOC12 all35fail,also10/10ordinaryrepeatcomparisonsfail. All3 original-algorithm fullruns qualify;N20/N500fixedobjectiveerrors2.85e-10/9.40e-10<=1e-8. Snapshot/lifetimeassumption privatevendor range; no productionpromotion.
NativeN20memcheckfailsCUDA999(239outstandingallocsafterabort),racecheckexit11,synccheckCUDA999;initcheckpasses. Minimalvendor-free100/101 reproducememfailurewith/withoutcopyornestedchild.102presyncpartialinterruptedbycomputerreset, executableabsentafterreset(donotclaimhash). Fresh103normalboth+6race/init/syncpass, bothmemchecks999persist. Frozen95-99/100/101/103 retained,next104+. AllownedGPUprocessesterminal. Fullsources/logs/helpers/artifactsin qoco-conditional-solve-v99-checkpoint.json and docs/QOCO_CONDITIONAL_SOLVE_EXPERIMENT.md. NeedactualGPUresidual/backup/stop/refinementintegration, factorbuffergraphlifetimeproof, cancellation and accuracy/convergence/perf; fullGPUgoalremains.
UserrequestedLambdaaftercomputerreset: temporary/tmp/traj-key.pemmissing;restoredowner-onlycopyfromoriginalwithoutprintingsecret. FreshreadonlySSH08:56UTC H100100%,1607MiB,44C;worker53138/server53183/session2619902live.159groupscompleted+1running(ordinal159),latestgroupall9timeouts. Source1dbcae0unchanged;GTOC12v11stillpassesbothverifiers23ships194collected14047.802874743327kg. NoLambdaGPUoffloadwhilebusy,nojob/sourcechanges. Freshstatus+summaryartifactsafter-reset retained.

Viewer after reset: restarted node serve.mjs --port4178 in results/lambda/2026-09-06/visualiser; execsession7869 intentionallyleftlive. HTTP200+scripts/check.mjs pass, fleetSHA0ebd0dfa unchanged. Browseropen queued, nofreshvisual-renderclaim. CheckactualGitref for diagnosticcheckpointcommit; no productionpromotion.

## GPU-native seed and outer control, v104–v107

Explicit full GPU-native goal recreated via create_goal (get_goal returned null). ACTIVE/incomplete; no subagents. User rejected stopping at partial GPU solver. New verified work: optional actual QOCO device IR104/105; native GPU SCvx arithmetic/control106; parallel GPU Lambert/Kepler seed107. Default CPU path unchanged. See docs/GTOC12_GPU_NATIVE_CONTROL.md and QOCO_DEVICE_REFINEMENT.md plus artifacts/performance/gtoc12-native-v107-checkpoint.json, embedding all runtime hashes, prepared104/105 sources,106 snapshot,107 current sources, helpers/results/failures.

Core107 frozen /home/angus/build-spacepdhcg-gtoc12-v107/final/libspacepdhcg_cuda.so. Prior106 retained with source-snapshot directory. QOCO105 frozen /home/angus/build-qoco-gpu-device-ir-v105/final/libqoco.so. Next fresh runtime108+; never overwrite frozen files. Default builder unchanged. Initial107 test build failed include lookup; fixed relative include, rebuilt before freezing; failed log retained. All final source matches frozen107.

CUDA outer selection requires CUDA dynamics+assembly+QOCO; --outer-loop-backend cuda, seed auto follows outer. Optional --seed-backend numpy is explicit CPU-seed ablation. Native API takes null seed pointers for GPU seed, otherwise supplied host seed. GPU metrics/merit, defects, candidate acceptance, trust updates, polishing, final status, and accepted trajectory copies; host reads16-byte dispatch perstep+initial/polish, existing QOCO still synchronous host dispatch/audit. Trajectory upload0 for GPU seed; download11*nodes*8 only once, history final. Counters exclude setup, reports and QOCO internal transfers. GPU scan66blocks/8193points per2branches; two scalar bisections, GPU cost selection and parallelpernode safeguarded Kepler. NumPy-independent seed parity20cases (normal/long/collinear/hyperbolic/dense x4freeends) pass5e-12scaled. Full new path test forbids CPU seed and Python numerical iteration entrypoints and independently certifies same2445.3111007852112kg target1e-5kg.

106 controller14branches+10finalstates+8multi-block/NaN-tail reductions passall4sanitizers. Full106 hostIRablation memory0leaks/init/sync pass;304regressionpassed354s.107 seed all4sanitizers; full107 transfer+zerotime+seed22tests passmemory0leaks/init/sync withhostIRablation;39focused;final324regressionpassed354.89s withoutskips. Full deviceIR105conditionalgraphsanitizers remain UNRESOLVED; hostIRablation safety is scoped, not whole105qualification. Need mid-solve cancellation, broader full transfer/hold reliability and setup path migration.

105longpaired COMPLETED before reset:32freshprocesses64legs allqualify,16measured/arm.94median612.088mean711.578ms vs105median545.038mean682.943ms (~11%median,4%mean).1056modifiedpreparedfiles reproducedexactLF fromfrozen94+currentpostprocessor.104failedfullsanitizers andhostrace180stimeout retained.10531focused+SCvxtests andPD6N20/N500fixedobjective checks preserved. No general speedup.

106 sixpaired/24legsallqualify: CPUseedboth, Pythonouter568.335median555.503meanms vsGPUouter529.730median624.251mean (slowtail1.169s).107sixpaired/24legsallqualify: PythonouterCPUseed541.044median535.413mean vsGPUouterGPUseed589.183median666.029mean (slowtail1.216s). No overall optimization win; migration with unchangedaccuracy. Timing files have prefix qoco-gtoc12-scvx-v106/v107-paired.json duehelpername, not gtoc12-scvx alone.

Lambda readonly11:50:20UTC stillH100100%,1607MiB39C;server53183/session2800998. Temp/tmp/traj-key.pem gone afterWSLreset; restoredO_EXCL0600fromoriginalrepo key withoutprinting;failed+success checksretained. No remote mutation/offload. Viewer3filesdirty fromotherwork(app.js,index.html,styles.css), preserveunstaged. No vieweractions thisturn. LocalGPUtests allterminal; currentgoal remainsfullpipeline, nextQOCOhostnumerics/setup/convergence/batching. InspectGitref forcheckpointcommit; no mainpromotion orpushthisturn.

Read-only next-QOCO lead: actualprepared src/qoco_api.c still check_stopping(host8metrics, best/state/dynamicreg/status); kkt.c predictor_corrector factorscuDSS, affineIR, Ds, GPUcentering returnswork.sigma, CPU sigma*mu inconstruct_kkt_comb_rhs, check_nan hostbranch, GPUtake_step returnswork.a. Device-combined-RHS header exists but is NOT selected105; read priorDEVLOG v26/27 failures/ablation before reenabling. Need persistent device scalar state for stopping, sigma*mu, dynamicreg and check_nan, plus captured/preallocated cuDSS factor/solve lifetime before full GPU loop. Do not equate removing Python outer with100%GPU. Full project also initial topology/conversion, mission search/batching, replay remains.

## 2026-09-06 README and leaderboard refresh

- [user] Asked for our leaderboard standing and a README explanation of PDHCG, the trajectory speed opportunity, and original-source attribution.
- [tool] Read the live rendered official GTOC12 leaderboard (https://gtoc12.tsinghua.edu.cn/competition/leaderBoard). JPL 22532.672; seventh Sigma TEAM 14714.133; eighth ATQ 13105.762. Local fleet_master_v11 final fleet row passes official checker at 14047.8 kg, 23 ships, 194 collected asteroids. Hypothetical inserted position eighth, not an official submission. Gap to seventh 666.333 kg; above ATQ 942.038 kg; 62.3441% of winner, requiring 60.4000% improvement to match it.
- [self] Updated README intro, current implementation boundaries, PDHCG versus PDHCG-CQP explanation, persistent SCvx speed rationale, upstream Lhongpei/PDHCG pinned revision, 2025 QP paper and 2026 conic paper attribution. Checked every local README link (23) and score arithmetic; git diff --check clean. README opened in Codex (queued).
- [self] GPU-native goal remains ACTIVE/incomplete. Parallel prior work v108/v109 remains uncommitted in qoco_device_control.cuh, prepare_qoco_device_control.py, and qoco_device_control_probe.cu; frozen libraries at /home/angus/build-qoco-gpu-device-ir-v108/final and v109/final. Do not overwrite. No subagents. Three unrelated visualiser files remain dirty and unstaged.
- [tool] Corrected balanced v109 benchmark finished (session22004 terminal), six triples/36 legs all qualified; medians baseline105 535.703 ms, control108 513.570 ms, control109 567.572 ms. Artifact qoco-qoco-control-v109-balanced-paired.json; no v109 timing win. Earlier unbalanced triple run is exploratory and must be retained separately.
- [tool] v109 host-IR ablation sanitizers completed (session79492 terminal): all 22 tests pass under memcheck/initcheck/synccheck, zero errors/leaks. Fresh actual device-IR full transfer memcheck returns99, CUDA999 in qoco_ir_solve cudaMemcpyAsync, 218 errors and abort-path leaks; full raw artifact qoco-control-v109-full-device-ir-memcheck.json retained. No claim of sanitizer-clean conditional IR. No GPU subprocess remains live from these checks.
- [self] Next: finish source/runtime/evidence checkpoint for108/109 (include initial probe assertion failure, corrected probe_v2 all-four sanitizer passes, 51 normal plus51 audited integration passes, native PD6 fixed objective gates, all paired runs and fresh failed IR memcheck), then migration of remaining host NaN branch/dispatch/setup. README commit is separate documentation work; solver sources still need their checkpoint. No main merge or push in this documentation step.

## 2026-09-06 — device control v108–v110 and publication repair

- [self] Previous goal turn was progress: README attribution/leaderboard published with core107 checkpoints at main5789977e. Goal remains ACTIVE and incomplete; no subagents. Preserve the three unrelated visualiser edits.
- [self] v108/v109 now preserved in artifacts/performance/qoco-control-v109-checkpoint.json (6,179,333 bytes): 17 evidence files,75 helpers/raw outputs,10 runtime hashes, both prepared source variants and initial probe failure. Current109 extension sources remain byte-identical after LF normalization. See docs/QOCO_DEVICE_CONTROL.md.
- [self] v110 adds separate prepare_qoco_device_direction.py and qoco_device_direction.cuh after the109 postprocessor. Persistent State gets an invalid_direction flag; parallel NaN scan resets it each combined direction; real update_iterates returns before arithmetic and writes alpha0 if flagged. Original NaN-only semantics, host ablation and explicit audit remain. Scratch correction/line search can still run on rejected directions; no accepted iterate mutation. Host IPM/cuDSS dispatch/setup remain, so no full-GPU claim.
- [tool] Frozen110 /home/angus/build-qoco-gpu-device-ir-v110/final/libqoco.so and qoco_device_direction_test. Never overwrite, next111+. Actual scan/step tests: NaN across primal/dual slices/tails, mixed SOC/LP,262145LP,emptycones,valid-after-invalid,Inf parity,guards,audit,nested scopes all pass normal + all4 CUDA sanitizers. First launch failed missingLD_LIBRARY_PATH before numerical execution; same frozen executable passed configured run, both artifacts retained.
- [tool]110 integration51normal +51audited pass;22 native/seed tests pass each mem/init/sync with deviceIR disabled,0errors/leaks.7 convergence/failure regressions pass. PD6N20/N500 independently certified objectives0.5129756912894413/0.5129756922612151, unchanged1e-8gate. Balanced6triples36legsallqualify; medians105589.932ms,109555.361ms,110601.990ms. No timing win. Fresh actual109deviceIRmemcheck stillCUDA999/218errors, remains unresolved; don't repeat or report110hostIR checks as fullconditional qualification.
- [self]110 checkpoint artifacts/performance/qoco-direction-v110-checkpoint.json (2,170,385bytes) includes sources/prepared10files/runtimehashes34helpers10evidence/fixedobjectivechecks. Separate qoco-direction-v110-reproduction.json proves current postprocessors reproduce10modified frozen files exactly. GPU processes all terminal; no test/benchmark running at checkpoint.
- [tool] Lambda read-only v110 first failed because/tmp/traj-key.pem disappeared; owner-only O_EXCL temp key restored from repo original without printing. Second read succeeds: H100100%,1607MiB,41C, existing campaign still active. No offload or remote mutations. Both status artifacts retained in110checkpoint.
- [self] Fixed two pre-existing CI problems in separate main publication worktree C:/Users/Angus/Desktop/projects/spacecraft-main-publish-20260906: cpp/all missing c_api.cpp in ABI smoke executables; wheel workflow matrix reference moved from workflow-level to job-level concurrency. All48native tests pass normal+ASan/UBSan; artifact ci-repair-20260906.json. Repairs83ac28d2 merged/pushed main86b0f93b. GitHub cpp-all subsequentlyPASS; wheel Linux/macOSPAss, Windows exposed missing<string> include. Added explicit<string> fixed_cqp.hpp in535630e3 onperfbranch; native c_api.cpp strictsyntaxpasses; latestpublication pending. CI/full-gate86 still running at lastcheck, don't claimallgreen.
- [self] Next substantial boundary: qoco_ir_solve still downloads IR State and synchronizes solver stream twice per IPM iteration, then destroys per-solve graphs. Investigate retained device IR accounting + event-ordered default-stream consumers and graph lifetime, then device-controlled IPM/cuDSS factor/SOLVE dispatch; remaining setup/mission batching/replay scope stays intact. No default backend promotion based on these mixed timings.

## 2026-09-06 — retained factorisation graph v111–v115

- [self] Previous goal turn was progress (110 GPU decisions published39cb492d). Goal remains ACTIVE/incomplete. No subagents; preserve the three unrelated viewer changes. No GPU process remains running from this tranche.
- [tool] NVIDIA cuDSS docs confirm asynchronous factorisation/solve can be captured with compatible allocator; analysis is synchronous. Memory-handler callbacks must be stream-ordered,256-byte aligned,return0success; data owns allocations until destruction. These are real boundaries, not a claim that setup/IPM is finished.
- [self] Diagnostic111/112 custom allocator returned a non-null pointer for size0 and failed analysis;112logging identified it.113zero-size-null fix enabled captured factor+solve replay. PD6Kn3026/74066:25nodes,13kernels4copies8memsets,two8bytehost solve-range snapshots;60/60strict1e-12scaledreplayspass across15successive solves eachfixture. GTOC12 all60strictcomparisonsFAIL,maxscaled4.273e-4;both full diagnostic legs stillqualify. FirstGTOC12helperfailedmissingSCVX_OUTER; configuredrerunseparate. Don't claimuniversal solveparity or production use of diagnostic snapshots.
- [self] Actual115extension retains only the numerical-factorisation graph: warm vendor factoronce, captureonce, then graphlaunch instead of repeated host cuDSSfactorcalls. Device-only nodeguard rejects host copies/unsupportednodes. Per-workspacepool retains allocations and reuses blocks only on the free stream;allocmissduringcapturefails.114builtbutnotrun,115correctsfree-streamownership. Currentdiagnosticheaderalsofixedafter113;checkpointretainsbefore-fixsourceandallfrozen111-113preparedheaders.
- [tool] Frozen111–115 /home/angus/build-qoco-gpu-device-ir-v{version}/final/libqoco.so. Next116+;neveroverwrite.115 passes51integrationnormal+51audited,7convergence/failuretests,PD6N20/N500independentcerts +fixed1e-8objectivegates(errors9.776e-11,9.705e-10),22native/seedtests underhostIRmem/init/sync,zeroerrors/leaks. ConditionalIRsanitizerfailure remains unresolved; no fullsanitizercleanclaim.
- [tool]115sixbalancedtriples36legsallqualify. Medians110535.348ms,pool-only115516.308ms,factorgraph115530.927ms. No convincing graphspeedup. Sourcepostprocessors reproduce12modifiedpreparedfiles exactly. Checkpoint artifacts/performance/qoco-factor-graph-v115-checkpoint.json embeds5current extension sources,allfivepreparedvariants,10runtimehashes,47helpers,12evidencefiles,rawfailures,objectives andtimings. Sourcehashesverifiedbeforecommit. docs/QOCO_FACTOR_GRAPH.md coversboundaries.
- [self] CIWindows wheel after<string>fix exposed strncpy C4996. Replaced withbounded memcpy intoalreadyzeroedpathname array, preservingterminator/padding forgeneratednames. Strictcompile+both C ABI smoke executables pass.35b21262+ci-portabilityevidence merged/pushed maina4f5cb58. WheelnowPASSall3OS,cpp-allPASS,paper1/corePASS;ci/fullgate stillrunninglastread. No Lambdaoffload;lastverifiedH100busy100%v110status.
- [self] Next substantial work: qoco_ir_solve still rebuilds two IR graphs per IPM iteration, downloads acceptedIRcount and synchronizesstream. With retainedfactorworkspace, investigate caching IRgraphs by operandaddress/shape/tolerances, GPUaccounting and eventorderedconsumers, then capture fullIPM/SCvxcontrol. Initial sparseanalysis/topology/setup andmissionbatching/replay remain in scope. Graphfactorphase is a step towardGPUcontrol, notgoalcompletion. CheckGitrefsforcurrentpublication.


## 2026-09-06 — retained refinement v116/v117 and corrected scoring

- [user] Continue toward fully GPU-native C++/CUDA and climb toward the strongest overall published fleet. Benchmark target is Antipodes 24,474.16 weighted kg; do not confuse faster transfer solves with improved fleet score.
- [self] SCORE CORRECTION supersedes earlier eighth-place/62.3% claims: v11 raw mass14,047.802874743327kg, 23ships/610.774kg each; independent fixed-bonus score12,805.194102488575kg. Recomputed194asteroid contributions against pinned e8a3795e... bonus table,109discounted. Approx ninth against historical competition scores; best-per-team postcomp ninth, all3 postcomp solutions as extra entries twelfth. Corrected README/viewer metadata/fleet.py documentation, committed9f9f6037, merged/pushed main10d7901f. No official submission.
- [self] v116 caches one conditional IR graph per fixed-topology cuDSS workspace, tolerance/max passed via device parameter kernel. v117 uses device step/total counters and completion events, removes per-linear-solve D2H+CPU wait. Terminal/verbose counts preserve exclusion of initialization IR. Files scripts/gpu/prepare_qoco_ir_cache.py, prepare_qoco_ir_counts.py and cpp/cuda/patches/qoco_ir_cache.cuh, qoco_ir_device_counts.cuh. Apply after115 factor graph; normal builders unchanged.
- [tool] Frozen /home/angus/build-qoco-gpu-device-ir-v116 and v117 /final/libqoco.so; never overwrite. Next runtime118+. Both native controller+51integration+7convergence/failure+PD6N20/N500 fixed objective gates pass;11751audited pass. Independent analyticQP probe16solves across6variants has exact IPM/IR/step count and status parity with115,10nonzeroIRcases, matrix/RHS/tolerance/budget changes inclzero and1IPMtermination/recovery. Cache1build168replays vsuncached168builds. Firstunscopedprobe failscapture onallversions; corrected explicit queued scope matchesnativeadapter. Firstcompile missingQDLDLinclude also retained.
- [tool]116balanced36legs allqualify: medians115563.284ms/uncached116607.868/cached116637.236.117balanced36legs allqualify:116499.047/hostcount117510.574/GPUcount117520.110ms. NO measured speedup/default promotion. Actual117scopedQPmemcheck stillCUDA999 atgraphlaunch,104errors/102outstandingallocationsafterabort; earlierminimalvendor-freeconditionalfailure remains unresolved. No fullsanitizercleanclaim. Eightgeneratedfiles reproducedexactfrom115+formattedpostprocessors.
- [tool] Lambda116readonlyrestoredkeycheck: existingH100100%,1607MiB,42C campaign; nooffload/mutation/newpaidinstance. AlllocalGPUchecks terminal. ViewerNode93742 serves4173; user currently viewing it. Latest docs/evidence in docs/QOCO_RETAINED_REFINEMENT.md and artifacts/performance/qoco-ir-counts-v117-checkpoint.json.
- [self] Fullgoal ACTIVE/incomplete. Next: GPU IPM/SCvx dispatch, remaining barriers, initialsetup/reordering and replay/batching, then broader fullmission search under true fixed-bonus score/physics gates. Latestuserauthorization overrides historical scratchpadCPU-only/no-push restrictions.

## 2026-09-07 — complete GPU IPM iteration loop v118–v121

- [self] Goal remains ACTIVE/incomplete. v121 optional whole-IPM conditional graph now runs residual/stop, NT scaling, factorisation, predictor/corrector, nested IR and counts without host IPM dispatch. Apply prepare_qoco_ipm_graph.py after117 and compile CUDA with --default-stream per-thread; SPACEPDHCG_TEST_QOCO_IPM_GRAPH=1. Normal builders unchanged. Whole graph is rebuilt each solve; initial setup/analysis/initialisation, terminal recovery/reporting and SCvx dispatch remain host work.
- [self] Captured factor/SOLVE child graphs cannot clone conditional IR, so IR WHILE nodes are emitted into the parent. v118/v119 instantiation failed because cuBLAS dot capture inserted allocation/free nodes. Retained 32MiB cuBLAS workspace and bypassing separate metric stream switches fixed it. v120 compilation failed from unavailable CUDA_CHECK in reduction-scope header;121 explicit error handling fixes it. All failed sources/logs/DOT are retained.
- [tool] Frozen121 /home/angus/build-qoco-gpu-device-ir-v121/final/libqoco.so;118/119 also frozen,120 failed build tree retained. Next122+, never overwrite. v121 enabled AND disabled modes each pass native controller and51integration tests. Enabled7convergence/failure tests PASS; PD6N20/N500 independently certified with fixed1e-8 objective errors9.776e-11/9.705e-10. Analytic QP16cases exactly match all printed objectives/status/IPM/IR/step counts between host/GPU, including one-IPM max exit and subsequent recovery. All9postprocessed files reproduced exactly after Ruff formatting; Ruff PASS.
- [tool] Balanced six triples36complete GTOC12 legs allqualified under unchanged physics and1e-5kg finalmass2445.3111007852112 gate. Medians117550.833ms,host121483.960ms,GPU121325.607ms =>1.69x vs117,1.49x same-runtime host ablation. Display5090 clocks unlocked; no universal/fleet speedup claim. v118disabled suite had one equality gate failure1.804e-9>1e-9; retained despite121passing. v121 actualmemcheck stillabortCUDA999 in INITIAL IR before newloop,104errors/102outstandingallocations afterabort; NOTsanitizerqualified.
- [tool] Checkpoint artifacts/performance/qoco-ipm-graph-v121-checkpoint.json,2,797,217bytes SHA133be15c285ce6f2aa19953a36080a2c50a393cabdd0c4fa7035f2af1f4b7045 embeds4sources,4preparedversions,9runtimes,49helpers,14evidencefiles inclfailures. docs/QOCO_GPU_IPM_LOOP.md+README describe real boundaries. Lambda121readonlycheck on20260907 H100100%,1607MiB,41C existingcampaign; nooffload/mutation/newinstance. All localGPUtests terminal; viewer4173 remains existing v11 fleet, not new121fleet output.
- [self] Next: retain whole-IPM graph with workspace/scalar/cuBLAS lifetime and changing-settings guards, GPU SCvx orchestration/setup and batching; continue independent full physics/weighted-score gates. Fleet score remains12,805.1941weightedkg, historicalapprox9th; benchmark24,474.16. Current tranche is progress, not full GPU-native completion. Publish using existing main worktree/user commit-merge-push authorization.

## 2026-09-07 — retained complete IPM graph v122/v123

- [self] Previous goal turn made progress:121 GPU IPM loop committed04ca244a, merged/pushed main6c802eb3 with 1.69x local fixture ratio vs117. This turn retains the whole graph/resources; goal remains ACTIVE/incomplete, no agents used.
- [self] Optional prepare_qoco_ipm_cache.py after121 adds per-cuDSS-workspace IPM graph/cache,32MiB BLAS workspace+handle/scalar scratch,64-byte device parameters for k/kinv,tolerances,IPM/IR budgets. Borrow/restore queued-scope TLS slots only during IPM, preserving caller initialisation/reporting scope. Fixed topology/pointers owned byQOCO; staticP/A/G regularisation/control/workspace changes invalidate graph. Device/thread-bound use rejects migration/reentry. Cleanup destroysgraph+resources beforevendorcuDSSteardown.123 allocatesBLASworkspaceatresourcecreation;122 allocateditduringprepare. Defaultbuildersunchanged.
- [tool] Frozen122/123 /home/angus/build-qoco-gpu-device-ir-v{122,123}/final/libqoco.so; next124+, neveroverwrite. Both51integration,7convergence/failure,nativecontroller,PD6N20/N500 independentcert+unchanged1e-8objectivegatesPASS;123disabled51PASS.123objectiveerrors4.141e-14/9.715e-10. All9changedprepared123filesreproduceexactfrom121+formattedpostprocessor;RuffPASS. NoGPUtestprocessstillrunning.
- [tool]122originalQP16cases1build16replays exactparity. Newinterleaved2workspace32caseprobe testsnestedscopes,coefficient/RHS,tolerance,IR/IPMbudgetchanges,staticG invalidation,maxexit/recovery.121graph/host123/uncached123/cached123 exactallprintedobjective/status/IPM/IR/stepcountparity. Cached2build16replaysperworkspace vsuncached16builds. Initialunscaledprobehadk=1despitecostchanges;addedQOCO_IPM_PROBE_RUIZ=1 reference3Ruizsteps variant,32distinctkvalues exactfourwayparity withsamecachecounts. HostRuizdiagnosticisnotGPUsetupproof.
- [tool] Isolatedresourceprobe2retainedworkspacesacross8callerscopelifetimes+nestedscopes+actualcuBLASdot+scratchrestoration passesnormal+memcheck0errors/0leakedallocations. Actualfull123memcheckstillabortCUDA999 initialIRbeforeenteringnewresources,104errors/102outstandingallocsafterabort. FullsolverNOTsanitizerqualified.
- [tool]122balanced36legsallqualifybutpartiallyoverlappedCPUbuild123;retainasphysics evidence,excludeperformanceconclusion.123balanced36legsallqualify same1e-5kg mass2445.3111007852112gate; medians121332.363ms/uncached123387.147/cached123340.241. NO additionalendtoendspeedupvs121. DisplayGPUclocksunlocked,outliers+warmupsretained. Retention is structural step for fullGPUorchestration,notcompletedgoal/defaultpromotion.
- [tool] Checkpoint artifacts/performance/qoco-ipm-cache-v123-checkpoint.json 4,710,473bytes SHAe65d8844d0f5bf7c5a46c379b3e1574c6ff5db9287fe4502d62d9c7389700573 embeds6sources,2preparedvariants,12runtimes,67helpers,18evidence. docs/QOCO_RETAINED_IPM.md+README accurate. Lambda123firstSSHcheckfailedmissingtemporarykey;restoredsecurely,separatereadonlyretryH100100%,1607MiB,41C. Nooffload/mutation/newinstance. Existingv11viewerfleetunchanged;score12,805.1941weightedkg,notnew123GPUfleet.
- [self] Next substantial target: GPUinitialisation/control and fullSCvx orchestration. Currentqoco_solve stillinitializes outsidegraph; nativeadaptercreatesephemeralqueuedhandle eachsolve; terminalrestore/reportalsohost. RetainedIPMresources currentlyenteredAFTERinitialisation andreturnedbeforecopy_solution. Consider extendingresource lifetime overinitialisation then buildingcomposableGPUinitialisation+IPM+SCvx graph, while preserving independentphysics/objectives and dynamicparameters. Wholeconditionalgraphs cannotbeclonedintoparent; mustemitconditionals intoenclosinggraph. Initialtopology/analysis andmissionbatch/search remainunfinished. Keepfullgoalactive.

## 2026-09-07 — captured initialisation and parallel initial cones v124

- [self] Previous retained-IPM tranche committed98c60087, merged/pushed main595aeedc. Goal ACTIVE/incomplete; no agents used. v124 captures initial control, initial linear solve, cone shifts, current warm-start IF and the IPM loop after one-time vendor warm-up. dynamic_reg/warm_start travel in device parameters. Initialisation mode invalidates cache. Default builders unchanged; postprocess after123 with prepare_qoco_ipm_initialization.py and PTDS. INIT_DISABLE ablation preserves host init; HOST_INITIAL_CONE diagnostic incompatible with captured init.
- [self] Finite cone shift now uses existing parallel residual reduction and parallel entry update, no host residual gate/serial scan. Nonfinite exceptional fallback preserves legacy comparisons on device. Scratch reserves combined l+nsoc partials. Pure-SOC NT identity now clears compact tails even when l=0; baseline123 direct invariant fails,124 passes.
- [tool] Frozen /home/angus/build-qoco-gpu-device-ir-v124/final/libqoco.so; next125+, never overwrite. Enabled AND disabled nativecontroller+51 integration PASS,7 convergence PASS,PD6N20/N500 certificates and fixed1e-8objectivegates PASS(errors5.399e-10/9.705e-10).32scaled warm/cold/dynamicreg interleaved cases exact printed parity across123/host124/hostinit124/init124;32distinctk,2build16replays perworkspace.84cone cases inclnonfinite,1029partials,independentfiniteformula,full returned-bit memcmp against original allPASS normal+captured. Earlier fingerprint probe memcheck/initcheck/synccheck0errors,memcheck0leaks; identity memcheck0errors0leaks.
- [tool] NewpureSOC analytic12update probe passes existing1e-8objective/feasibility limits across123/hostinit124/init124; coordinatebound derivedfromstrongconvexity+coneviolation. Notbitexact. Initial1e-7coordinateexpectationfailedwarmupdate(all3); tight1e-13stoppingalsofailsstatus(all3). Final QOCO_PURE_SOC_STRICT preservesfailedtightdiagnostic andalllogs; normalpassdoesnotsupersedestrictfailure. Full124memcheckstillCUDA999one-timeinitialIRwarmupbeforecombinedgraph,110errors/108outstandingallocsafterabort; NOTfullsanitizerqualified.
- [tool] Sixbalancedtriples36completelegsallqualifiedunchangedphysics+1e-5kgmass2445.3111007852112gate. Medians123333.859ms/hostinit124363.031ms/init124332.389ms. NOadditionalendtoendgain. NoCPUbuildoverlap,displayGPUclocksunlocked,warmups/outliersretained. All12preparedfilesreproduceexact;RuffPASS. AlllocalGPUtestscompleted. Lambda124read-onlyH100100%,1607MiB,41C124.04Wexistingcampaign; nooffload/mutation. Existingv11viewer/scoreunchanged12,805.1941weightedkg.
- [tool] Checkpoint artifacts/performance/qoco-ipm-initialization-v124-checkpoint.json 2,578,102bytes SHA63908cb4aeec3c3c3e81b0447500f018dd0940e63d99cdf9d6229d7608408912 embeds7sources,preparedtree,16runtimehashes,47helpers,18evidence inclfailures. Docs/QOCO_GPU_INITIALIZATION.md andREADME recordbounds. Source/testchangesreadyforauthorizedcommit/mainmerge/push.
- [self] Next: remaining terminalrestore/unscale/report andSCvxhostdispatch; initialvendorwarmup/setup/analysis; per-solve ephemeralqueuedresourcecreation and per-legsession recreation. FullSCvxgraphmustemitnestedconditionalsdirectlyintoparent, cannotcloneconditionalgraphs. Consider batch/workspace reuse acrosslegs toremove repeatedsetup. Fleetsearch remains unfinished; no newfleetclaim/no goalcompletion.

## 2026-09-07 — captured terminal recovery/unscaling v125

- [self] Previous goal turn PROGRESS:124 committed6bd76abb,merged/pushed maind155545b. Goal ACTIVE/incomplete. This turn adds prepare_qoco_ipm_terminal.py after124 and qoco_ipm_terminal.cuh: after IPM WHILE, GPU reads status, restores best valid iterate on numerical/max exits and upgrades inaccurate status using same<=1metric rule; then parallel unscale x/y/s/z withcurrentdevicekinv andexplicitsequentialroundedproducts. Hostqoco_api skipsdoubleterminalwork viaIPMreturn2. Terminalmodeinvalidatescache;presence SPACEPDHCG_TEST_QOCO_IPM_TERMINAL_DISABLE isablation. Defaultbuildersunchanged.
- [tool] Frozen /home/angus/build-qoco-gpu-device-ir-v125/final/libqoco.so. Next126+,neveroverwrite. FirstbuildPASS. Enabled+disabled eachnativecontroller+51integrationPASS;7convergence/failurePASS. PD6N20/N500independentcert+fixed1e-8objectivegatesPASS(error9.776e-11/1.110e-16). All6preparedfilesreproduceexact;RuffPASS. Noagentused.
- [tool]32caseinterleaved2workspaceQPwarm/scaling/dynamic/staticreg/budget/maxexit/recovery:124graph/125host/125hostterminal/125terminal exactallprintedmetadataANDallprimal/slack/dualvectors,plusanalyticvectorchecks.32distinctcostscales,2build16replaysperworkspace. ExtraTERMINAL_TOGGLE run alternatesmodeeveryupdate;exactvectors/metadata,12516build16replaysperworkspace. Existingwarmonlyprobecompiledpre-vectortestchange; currenttoggleprobehasalltestextensions.
- [tool]Directpreparedterminalkernel240graphreplays:5statuses,2bestvalidstates,3metricthresholdcases,2kinv,4layoutswithzero/multiblocksections. ExactindependentState/vectorbits+canariesPASS. memcheck/initcheck/synccheck0errors,memcheck0leaks. Initialprobecompileambiguousnestedinitializerlistfailed;correctedexplicitvector<vector<int>>,failure/sourcepreserved. PureSOC12normalupdatespass3variants1e-8objective/feasibility;notbitexact. Priorstrictprecisionfailureunresolved/notrerun. Full125memcheckstillCUDA999initialconditionalIRwarmupbeforecombinedgraph,110errors/108outstandingallocs;NOTfullsanitizerqualified.
- [tool]Balanced6triples36completelegsallqualifiedunchangedphysics+1e-5kgmass2445.3111007852112. Medians124321.820ms/hostterminal125402.392ms/terminal125315.516ms. Widevariation,NOadditionalendtoendgainclaim;externalwalltimeused. InternalQOCOtimer nowincludescapturedterminalworkwhereoldtimerstoppedbeforeterminal;differentboundariesdocumented. AllGPUtestprocessescompleted. Lambda125initialSSHmissingkeyfailed;securekeyrestore+separatereadonlyretryH100100%,1607MiB,41C123.99Wexistingcampaign. Nooffload/mutation/newinstance. Fleetv11/visualiserunchanged12,805.1941weightedkg;no newscore.
- [tool] Checkpoint artifacts/performance/qoco-ipm-terminal-v125-checkpoint.json 2,626,956bytes SHA9950610d8dd22a087b0e38d7609d773b0fb51eb16b6e190c89e1b72619101ad2 embeds4sources,preparedtree,12runtimehashes,41helpers,15evidenceinclfailures. docs/QOCO_GPU_TERMINAL.md+READMErecordbounds. Readyforauthorizedcommit/mainmerge/push.
- [self] Next substantial boundary: qoco_gpu_ipm_loop still synchronouslydownloadsiterationcount+controlmetadata; qoco_api copy_solution/devicecompletion andIRcounts host; native adapter solve_with_reduction_scope creates/destroysqueuedhandleeachcall; gtoc12_qoco_solve_device qualifies/statusreports onhost; gtoc12_scvx hostloopreads16bytecommandperattempt. Need device completion/report API and outergraph emission rather thanconditionalgraphcloning. Initialtopology/setup/analysis/vendorwarmup andper-legworkspacecreation remain. FullSCvxGPUdispatch andmissionbatch/search areunfinished. Keepfullgoalactive.

## 2026-09-07 — device completion and asynchronous replay v126

- [self] Previous goal turn PROGRESS:125 committed8d541144 merged/pushed maine8c95da2. GoalACTIVE/incomplete. New prepare_qoco_ipm_replay.py after125 + qoco_gpu_replay.h/qoco_ipm_completion.cuh/qoco_ipm_replay.cuh. Adds64byteGPUcompletionatendcapturedterminalgraph. qoco_gpu_ipm_replay_device queues currentparameters+iterationreset+graph oncallerstream and returns borroweddevicepacket/x/y/s/z withoutreadback/wait. Hostsolstatusinvalidated. Preparedinit+terminalrequired;fixedtopology/staticreg/thread/deviceguards. Samestreamreplays+consumers canchain; differentpendingstreamrejected; finish_device waitsstreamincludingconsumers. Updates/otherAPI requirefinish; normalqoco_solve finishespendingatentry. Currenthostdynamicreg suppliedeachcall, notimplicitlyupdatedfromdevice. Replayitselfrejectsoutercapture; defaultbuildersunchanged.
- [tool] Frozen /home/angus/build-qoco-gpu-device-ir-v126/final/libqoco.so; next127+,neveroverwrite. FirstbuildPASSafterRuffimportorderfix. All7modifiedpreparedfilesexactreproduction;RuffPASS. EnabledANDdisablednativecontroller+51integrationPASS;7convergence/failurePASS. PD6N20/N500independentcertificate+fixed1e-8objectivegatesPASSerrors2.909e-14/1.110e-16. Noagentsused.
- [tool] Newreplayprobe128coldand128warmqueuedsolves(32coefficientcases2workspaces4replays),GPUconsumer snapshots betweenreplays exactsynchronousvectors/metadata/counts inclmaxexit/recovery. Held-streamcallback+watchdog provesAPIreturnswhilequeuedworkheld, AFTERcallerreductionscopedestroyed. Stream/static/null/unprepared/threadguardsPASS. Warmreferenceusesfixedsavedx0+resetinitialdynamicregthenasyncsameparameters. Initialprobecompilewarnedrenamedmainmissingreturn;sharedprobeexplicitreturn0fix; oldbinary/source/logpreserved. Held/warmtestexecutablesdistinctfrozen.
- [tool] Actualreplay-probe memcheckFAILS atFIRST SYNCHRONOUS priminggraphcompletion,cudaMemcpy_ptds999 qoco_ipm_graph.cuh:254 beforeasyncAPIcalls,253errors252outstandingallocs67,705,774bytesafterabort. ImportantdifferentboundaryfromearlierfixtureinitialIR. Sameoldinterleavedprobeunder125/126memcheckbothfailgraphcompletion:125251errors250allocs,126252errors251allocs. No fullsolver/asyncsanitizerqualification; failureevidencepreserved.
- [tool]6balancedpairs24completeGTOC12legsallqualifiedunchangedphysics+1e-5kgmass2445.3111007852112. Median125346.141ms/126306.988ms,widevariationdisplayGPUunlocked. Nativev107driver STILLUSES SYNCHRONOUS API, so timingsareonlyregressionchecks,NOTasyncspeedupevidence. AllGPUtestscomplete. Lambda126read-onlyH10081%,1607MiB,41C124.06Wexistingcampaignactive;nooffload/mutation/newinstance. Existingv11fleet/visualiserunchanged12,805.1941weightedkg.
- [tool] Checkpoint artifacts/performance/qoco-ipm-replay-v126-checkpoint.json 2,121,178bytesSHA9e3faceb11be31e80075e1be2a42bd90b2f06632eeeabef4f84914b6c0f1c2d9 embeds6sources,preparedtree,10runtimehashes,34helpers,14evidence. docs/QOCO_GPU_REPLAY.md+READMEdocumentcontractandlimits. Useraskedstatusduringwork;answeredbrieflyandcontinuedauthorizedpublication.
- [self] NEXT integrate nativeadapter/deviceaudit/SCvxwithnewdevicepacket. cpp/cuda/src/gtoc12_qoco.cu solve_device readsassemblyinvalid onhost, callsnative_qoco_update_solve, downloadsobjective4scalars andqualifiesonhost. native_qoco_adapter.cpp solvepath~1668 readsQOCOhoststatus,handleswarmcoldretry,runGPUauditwhichstilldownloadsresult; see native_qoco_gpu.cu qoco_gpu_audit_run~358. gtoc12_scvx.cu stillhostforloop/16byteCommanddownloads. NewAPIisavailablebutNOTadoptedthere. Needdeviceauditasyncresult and statusqualifieddecision/kernelconsumers,thencomposableoutergraphemission(conditionalgraphscannotbecloned),initialsetup/vendorwarmup+per-legsessionreuse,batching/missionsearch. Fullgoalstillunfinished;no newfleetresult.

## 2026-09-07 — native replay plus independent audit core127/core128

- [self] Previous goal turn PROGRESS:QOCO126 committed7037a55f,merged/pushed mainb4396e23. GoalACTIVE/incomplete. Newnative_qoco_gpu run_device exposes sameaudit/dualmapping withoutD2H/wait; oldrun wrapperqueuesexplicitdownloadthenwait. Graphcapturetested. Optional SPACEPDHCG_TEST_QOCO_NATIVE_REPLAY=1 nowconnects nativecold-solve adapter toQOCO126replay thenindependentaudit+primalcopyonsamestream beforeonefinalreportwait. First/rebuilt/stalegraph uses synchronousGPUpriming. Nativewarm retainsoldsynchronous warm-inaccurate coldretry to preservehost-mutateddynamicreg semantics. Defaultmodeunchanged;noagents.
- [self] Core127 downloadedfull64bytecompletion+48audit+flags,caused8transferbudgetfailures(120bytes>64),physicschecksPASS. Core128 addsretained8byteQocoReplayStatus, validatesABI1onGPU, downloadsstatus/iterationsonlyplus48auditandflags;unchanged64byteupdategatePASS. No testsweakened. Hostsolonlystatus/iters materialisedfornativehandling/accept action2. Retained3CUDAevents partitionreplay/audittiming; oldpathhosttiming,compareexternalwalltime. PendingRAIIdrainsqueuedworkbeforestackcompletionpacketexpiresonfailure. Source127savedbuild/performance/native-replay-v127-source-record.json;frozen127neveroverwrite.
- [tool] Frozen /home/angus/build-spacepdhcg-gtoc12-v{127,128}/final/{libspacepdhcg_cuda.so,gtoc12_scvx_test,qoco_gpu_audit_test}. QOCO126unchanged. Nextcore129+,runtimeQOCO127+ ifseparate;avoidversionconfusion.128enablednativecontroller+audit+51integrationPASS;full324regressionPASS359.78s inclconvergence/failure.128disabledaudit/controllerPASSbutfirstcoastqocostatus2unqualifiedthen1failed50pass(beforeasyncpath).27followupcoastsolves across107/128disabled/128enabledallqualify,butdoNOTeraseunresolvedqualificationvariability.127disabled51PASS;127enabled8budgetfail43PASSretained.
- [tool] PD6N20/N500core128+QOCO126 independentlycertified,unchanged1e-8objectivegateerrors5.399e-10/9.705e-10. N20tracesconfirmnativereplay. Directauditdense/graph/reuse/nonfinite/mappingPASS;memcheck+initcheck0errors,memcheck0leaks. Separate64capturedABI/status/iterationcasesunknownABI->-1,retainedalloc/nointernaldownloadPASSnormal+memcheck/initcheck/synccheck0errors0leaks.
- [tool] FullnativecoastmemcheckstillCUDA999 initialIRwarmup qoco_device_ir.cuh:116beforeasyncadaptercalls,208errors206outstandingallocations60,731,703bytesafterabort. NOTfullpipelinequalified. Failedartifactnative-replay-v128-full-memcheck.jsonpreserved. Full324passeddoesnotresolveablationqualificationfailureorsanitizerfailure.
- [tool]6balancedtriples36completeGTOC12legsallqualify unchangedphysics+1e-5kgmass2445.3111007852112. Core107337.209ms/128host570.374ms/128replay324.569msmedians;widevariationdisplayGPUunlocked,NOreliableoverallgainclaim. Traceenabledallenvs;onlyreplayprints148actualcallsacross6processes(14to47each),costincluded. Allwarmups/outliersretained. AllGPUtestscompleted. Lambda128read-onlyH100100%,1607MiB,41C124.20Wexistingcampaign;nooffload/mutation/newinstance. Fleetv11viewerunchanged12,805.1941weightedkg,no newfleet.
- [tool] Checkpoint artifacts/performance/native-replay-v128-checkpoint.json 3,028,113bytesSHA27c8f7fbae4817f9bd86839c08d251155cac5199a916fe719b3a66335b1efdac embeds5sources,11runtimehashes,41helpers,12evidenceinclfailures. docs/QOCO_NATIVE_REPLAY.md+README+historical126doclink updated. Sourcesinclude native_qoco_gpu.h/.cu, native_qoco_adapter.cpp, audit_test.cu, replay_status_test.cu. Readyforauthorizedcommit/mainmerge/push.
- [self] NEXT: final8+48bytestatus/auditreportstillCPUandGTOC12objective4scalars/qualificationhost. Needdevicequalificationreport(consumesQocoCompletion+QocoAuditResult+independentobjective) andnativeSCvxdecisionkernelintegration toremovefinalper-solvewait. Nativeadapterstillhostnumericconversionvalidation/settings+initialsetup. gtoc12_scvx.cu hostforloopreads16byteCommandaftereachattempt, qoco_conic invalidflagdownload. Nativewarmretry needsGPUdecision beforefullpipelinepromotion. Fullconditionaloutergraphemissionnotclone; initialsetup/vendorwarmup/per-legworkspace reuse, batch/missionsearchremain. Keepfullgoalactive.


## 2026-09-07 — native v129 GPU qualification and SCvx consumers
- Previous status turn was a verified wait: live Lambda campaign PID53138 and H100100% were checked. Current goal turn PROGRESS: added native cold-solve consumer API and GTOC12 objective/qualification/SCvx consumers before replay report collection. No agents. Goal remains ACTIVE and incomplete.
- Frozen /home/angus/build-spacepdhcg-gtoc12-v129/final (core, original controller test, audit test); QOCO126 unchanged. SPACEPDHCG_TEST_GTOC12_DEVICE_QUALIFICATION=1 plus NATIVE_REPLAY=1 and IPM_GRAPH=1 queues independent objective+qualification, nonlinear measurement, decide and accept on same stream before native report downloads/final wait. Initial/stale priming stays synchronous; successful priming publishes device status and invokes consumer. Failed priming invokes none. Native warm retry unchanged. Default remains unchanged.
- GPU qualifier retains exact status1/2, finite residual/objective/gap and requested-tolerance gate. SCvx decide reads retained device report; candidate measurement is speculative but acceptance is gated. Original 64-byte native adapter update limit unchanged. Bridge adds4byte qualification report to existing32byte objective readback (outside adapter counters). Host report waits, validation flags, outer16byte command loop, initial setup/analysis/warmup, per-leg reuse/batching/fleet search remain.
- Enabled51PASS, disabled51PASS, synchronous-consumer51PASS; broader324PASS359.65s. 210 production qualifier boundary/nonfinite/status graph-consumer cases pass normal/memcheck/initcheck/synccheck with zero errors/leaks. Updated actual controller runs14cases in EACH of2modes with contradictory host inputs to prove device precedence; final states/reductions preserved. Updated standalone build/performance/gtoc12_scvx_v129 passes all3sanitizers. Frozen bundled controller executable predates the second test mode (explicitly distinguished in docs/checkpoint). Public header C11 syntax compile PASS.
- Six balanced triples,36complete transfers all independent physics + unchanged1e-5kg massgate against2445.3111007852112PASS. Median128replay332.042ms/129host314.547ms/129device312.381ms. Wide timing variation including720ms; NO reliable additional gain established. 132 device decisions on12newmode transfers; everyprocess actualnative replay traces. All435reported qualifications acrossvariants match originalhostgate. Fullwarmups/outliers/tracesretained.
- Actual v129 full SCvx memcheck abortsCUDA999 qoco_device_ir.cuh:116initialIRwarmup BEFOREconsumer,233errors231outstandingallocations72,214,209bytesafterabort. NOTfullpipelinequalified. Same boundary as128coast, differentfixtureallocations. Retain rawfailure,doNOT claim cleanfullsanitizer.
- Lambda129read-only H100100%,1607MiB, campaignPID53138; nooffload/mutation/newinstance. Fleetv11viewerunchanged12,805.1941weightedkg; no newfleet or officialsubmission.
- NEXT: remove host numerical-update/assembly validation and per-attempt reporting dependencies, then GPU-controlled outer SCvx dispatch. Nativewarmretry device policy, initialvendorwarmup and setup, per-leg reuse/batching and fleet search remain. Resolve fullconditionalIRsanitizer failure and qualification variability before defaultpromotion. Keepimmutable129; nextcore130+.
- [tool] v129 checkpoint artifacts/performance/device-qualification-v129-checkpoint.json:1,899,806bytes SHAfe24bf1233a4b3fd9deb69cf5ec7fdac490b7a629368a0f612a6d307e69b6b18. Embeds source/runtime identity, helper scripts, raw success/failure evidence and v128 predecessor hash. Ready for authorized commit/main merge/push.


## 2026-09-07 — native v131 with queued QOCO128 numerical replay
- Previous goal turn PROGRESS: core129 committed25fca62e and merged/pushed main66fcc62e. Current turn PROGRESS: new scripts/gpu/prepare_qoco_numeric_replay.py after126; cpp/cuda/tests/qoco_numeric_replay_probe.cu; native_qoco_adapter.cpp optional queuednumeric integration. No agents. Goal ACTIVE/incomplete.
- Frozen QOCO /home/angus/build-qoco-gpu-device-ir-v{127,128}/final/libqoco.so; nextQOCO129+. Frozen native /home/angus/build-spacepdhcg-gtoc12-v{130,131}/final/{libspacepdhcg_cuda.so,gtoc12_scvx_test,qoco_gpu_audit_test}; nextcore132+. Do not confuse QOCO128 with older core128. Helpers build_numeric_replay_v127/v128.py; v128initialpreparer countassert15failed beforewrites, resume_numeric_replay_v128.py verifiedunfrozenbase thenfixed16sitesandbuilt. All6generatedfiles exactlyreproduce from126.
- QOCO new Context retains device scale result9, prior kinv read fromdevice for next update. queued update executes16kernel launches+matrixcopies+explicitKKTcsrupdate onsameproducerstream without9doubleD2H/globalwait. Ownerthread/device/pendingstream guards; scale_valid requires initialsyncnumericupdate. qoco_gpu_update_numeric_device(ctx,packed,stream,&result), qoco_gpu_finish_numeric_update(ctx,materialize0/1), qoco_gpu_ipm_replay_updated_device(solver,stream,result,&out). Host metadata stale untilmaterialize; laterfinish1worksafterfinish0. OldnumericAPI materializespending/stalefirst. Destructiondrains. Errornumericresultrequiresrebuild.
- QOCO IPMparameters update_invalid plusGPUsetterreadsdevice k/kinv andinvalidreport. NewouterconditionalIF skips entireinitialization/IPMfactorbody forinvalid update andwrites numerical-errorcompletion status3iterations0infmetrics;validbodyemitnestedinitialization/WHILE/IF/IR thencompletion. No conditionalgraphcloning. Normalreplay/defaultstillavailable. Prepared defaultbuildersunchanged.
- QOCO127 queueBUG:regex [^>] missed w->launchargs soseveralkernelsstayeddefaultstream;wrongk/outputs. Retainedfailedruntime/preparer/probes.128regexnon-greedy<<<...>>>plusassert16allproducerlaunchesfix. Direct64queued update/replay/consumer chains EXACTvectors,status,iterations,IR,obj,residuals vs synchronous; heldstreamwatchdog provesupdate+replaysubmitdoesnotwait; NaNguard preservespriorvectors/skipsIPM.43analyticcoordinate/objective1e-7qualified;21retainednumericalfailurecases after100->.01scaletransitions. First17syncresults126and127exactincludingfailures:existingconditioningissue, NOT64qualified. Oldregularreplay127128queuedcasesPASS.
- Native130 failed10/51tests becausezeroRuizsetupdoesnotprimecontextscalepacket.131oncefallsbacksyncnumericwhenqueueAPIreturns2unprepared;subsequentupdatesqueue. SPACEPDHCG_TEST_QOCO_NATIVE_NUMERIC_REPLAY=1 plusNATIVE_REPLAY1/IPM_GRAPH1;optionalGTOC12_DEVICE_QUALIFICATION1chainsallthroughSCvx. ColdNONEonly;warmexistingpath. Cache/stale fallbackmaterializesnumericbeforeoldsyncsolve; errorRAIIdrains. Newtriooptionaldlsym;featureenabledrequiresAPIs;defaultQOCO126compatretained. NATIVE_NUMERIC_REPLAY_TRACE printsactualcalls. Adapter64byteupdatebudgetunchanged; removed72byteQOCOreportisopaque/excludedfromadaptercounters. Internalhostupdate_secondsnow submissiontime,compareexternalcompletewall.
- Native131enabled51PASS,disabled51PASS,legacyQOCO126disabled51PASS; full324PASS369.83s. Two8updatecoast sequences togglequeue0/1 withRuiz0and3:16allqualified,9actualqueuedcalls. Sixbalancedtriples36fulltransfersALLphysics+unchanged1e-5kg2445.3111007852112massgatePASS. Medianscore129/QOCO126356.582ms,131/QOCO128sync325.024ms,131/QOCO128queue308.614ms;90actualnumericreplays. All396conicqualificationsmatcholdhostgate. Widevariance/no reliableadditionaloverallgainclaim,allwarmups/outliers/tracesretained.
- Minimalconditionalguard extractedproductioncode64cases normalPASS. Initialtestuninitialisedstoragevariantretained; initializedstreamstorage normal/initcheck/synccheckPASS,memcheckCUDA999atD2H5errors4allocations220bytesafterabort. Isolatedofficialpackage cuda-sanitizer-13-2=13.2.87-1 extracted /home/angus/tools/compute-sanitizer-13.2.87/extract/usr/local/cuda-13.2/bin/compute-sanitizer (2026.1.1.0);no system/driverchanges. Newcheckersame minimalmemfailure,init/syncPASS. Full131SCvxmemchecksameCUDA999 qoco_device_ir.cuh:116 initialIRbeforequeuednumeric,233errors231outstandingallocations72,214,209bytes. NOTmemcheckqualified;minimalreproducer narrows butdoesNOTprove solelytool/driverbug. Artifactsretainold/newcheckers+packageSHA+allfailures.
- Lambda freshread-onlyH100100%,1607MiBcampaignPID53138. Nooffload/disruption/newinstance. Fleetv11viewerunchanged12,805.194weightedkg; no newfleet/officialsubmission. LocalGPUsessionsallterminal now. GPUlock /home/angus/.spacepdhcg-gpu.lock used forallGPUtests.
- Checkpoint artifacts/performance/native-numeric-v131-checkpoint.json2,271,360bytes SHA56632fe585adad4a51b30504a79960edaf0747dfc80586843037ebea05b86c41 embeds3sources,12preparedfiles,runtimehashes,65helpers,26evidenceinclfailures,previous129hash. docs/QOCO_NUMERIC_REPLAY.md+READMEupdated;readyforauthorizedcommit/mainmerge/push.
- NEXT: remove remainingCPUvalidationdependencies: native refresh_conversion callsvalidate_cached_topology thenqoco_gpu_conversion_run(eachD2Hflag+wait); GTOCqoco_solve_device firstdownloadsassemblyinvalidflag+wait. Needretaineddevicevalidationflagsandguardedconsumers beforeonefinalreportcollection withoutlettingmalformedtopologyindexinvalidstorage. ThenhostouterSCvx16bytecommand+perattemptreporting, warmretrypolicy, initialsetup/analysis/vendorwarmup, perlegreuse/batching/missionsearch. CurrentqueuednumericAPIrejectsexternalcapture,IPMcontainsnestedconditionals;fulloutergraphmustemitnotclone. Maintainphysicsgatesandretainedfailures,goalnotcomplete.

## 2026-09-07 03:28 AEST - Core v132: queued canonical validation

- [self] PROGRESS: native topology comparison and canonical conversion validation now expose capture-compatible device flags. Guard copies QOCO128 nine-double numeric report into retained adapter storage, ORs canonical errors into its invalid field, and skips IPM on invalid inputs. No arithmetic/tolerance changes. Feature SPACEPDHCG_TEST_QOCO_DEVICE_VALIDATION=1 requires NATIVE_NUMERIC_REPLAY=1 and NATIVE_REPLAY=1; IPM_GRAPH=1 plus GTOC12_DEVICE_QUALIFICATION=1 chains through SCvx. Default/oracle/warm/rebuild paths preserved.
- [self] Exact flags: bound classification=1, nonfinite=2, asymmetric Q=4, topology=8. Host dimensions/views/cones checked before submission; caller sparse indices only compared, never consumed as compiled conversion indices. Guarded invalid update may mutate numeric buffers, so rebuild next attempt. Failed update drains borrowed work. Initial scale/graph priming collects validation before synchronous fallback. Final combined flag reduces adapter update downloads64->60bytes; opaque QOCO and GTOC bridge transfers excluded.
- [tool] Frozen native /home/angus/build-spacepdhcg-gtoc12-v132/final (library, controller, audit, conversion, native conversion executables); prepared QOCO remains128. Enabled51/disabled51/legacyQOCO12651 allPASS. Broad324PASS354.19s; older v77 planner runtime identity retained. Captured topology/conversion/guard/device consumer96combined-error casesPASS; isolated memcheck/initcheck/synccheck0errors. Native oracle/device validation Ruiz0/3 allPASS; traces confirm10invalid queued solves status3/zeroIPM and recovered KKT<=1e-8. Full conditional-IPM memcheck issue unchanged, not full sanitizer-qualified.
- [tool] Six balanced triples36complete transfers ALL independent physics and unchanged1e-5kg mass gate against2445.3111007852112PASS. All364 conic qualification reports match host predicate;115actual device validations. Medians core131317.136ms,132host299.650ms,132device311.566ms. No reliable extra speedup; all warmups/outliers retained. Adapter update byte deltas64 for old/disabled and60 for enabled throughout.
- [tool] Lambda H100100%,1607MiB, live campaignPID53138 on fresh read-only recheck. Temporary WSL SSH key missing first probe; safely restored from existing local key0600, failure and successful recheck retained. No offload/disruption/newinstance. Fleet/visualiser unchanged v11 weighted12805.194102488575kg.
- [self] Checkpoint artifacts/performance/native-validation-v132-checkpoint.json913790bytes SHA82915f42af9af327caf72557a21ee3a0430b54ef49856b813f0482d3909850d7 includes5sources, frozen runtime hashes, helpers/raw evidence, previous131hash. docs/QOCO_DEVICE_VALIDATION.md and README updated. All local test handles terminal. Ready for previously authorized commit/main merge/push.
- [self] NEXT: GTOC12 assembly flag still downloads/waits in gtoc12_qoco.cu solve_device_with_consumer before conversion. Must propagate producer-invalid into GPU replay guard: invalid trust/mass/radius/vinf/smoothness or dynamics output can produce finite assembled values, so canonical finite checks alone cannot replace it. First setup must still validate before host conversion until setup moved. Keep assembly return-code3 and qualification rejection; handle pending flag and fallback lifetimes. Then remove outer16byte command/report host dispatch, native warm retry, setup/analysis/vendor warmup, per-leg reuse/batching and mission search. QOCO queued API rejects external capture; nested conditional bodies must be emitted rather than cloned. Goal incomplete, retain physics gates.

## 2026-09-07 03:49 AEST - Native v133 assembly guard and coast variability evidence

- [self] PROGRESS: GTOC12 assembly invalid status now feeds the native replay guard as bit16 via same-stream qoco_gpu_conversion_include_producer. Any nonzero producer flag rejects (including negative); finite bad trust/mass/radius/vinf/weight values cannot slip through canonical finite checks. Native report internal-only producer_invalid/producer_validation_queued; public GTOC12 C ABI unchanged. Invalid producer takes precedence, returns GTOC12 API3, zeroiterations/unavailable residuals/objectives, no completed-solve increment. GPU consumer seesstatus3/unqualified; numeric-buffer mutation requires next solver rebuild. First setup and recovery/diagnostic/disabled queued paths still collect flag before synchronous use; borrow lifetime drained.
- [self] Feature SPACEPDHCG_TEST_GTOC12_DEVICE_ASSEMBLY_VALIDATION=1 plus QOCO_DEVICE_VALIDATION=1/NATIVE_NUMERIC_REPLAY=1/NATIVE_REPLAY=1/IPM_GRAPH=1 and optional GTOC12_DEVICE_QUALIFICATION=1. Trace GTOC12_ASSEMBLY_VALIDATION queued=X invalid=X. Successful adapter updates remain60bytes:producer shares existing4byteflag, removes separate GTOC assembly4byteD2H+wait. Fallback report counters include extra producerflag transfer. New C++ API with_input_guard; existing with_consumer forwards nullptr producer. Host metadata and compiled-map safety preserved.
- [tool] Frozen core /home/angus/build-spacepdhcg-gtoc12-v133/final (6files including new gtoc12_qoco_guard_test); prepared QOCO remains128. New captured topology/conversion/producer/guard/consumer192casesPASS, isolated memcheck/initcheck/synccheck0errors. Native old mixed-cone oracle/device tests Ruiz0/3PASS. Actual GTOC devicecallback confirms4queued invalid rejections (status3,iterations0,qualified0),8fallback invalids invoke no consumer, first invalidsetup and qualified recoveryPASS. Python18invalid parameter/state/control casesPASS withallqueued rejection traces. Ruff format/checkPASS.
- [tool] Initial enabled integration1failed51passed: [False-False-zoh] original equality abs2.4886523501366312e-9 exceeds1e-9 despite normalizedauditpassing. Retainfailure. Fourbalancedtriples targetedcoast:onehost133failure/onedevice133failure,4baseline132pass. Additional18fresh baseline132trialsFAIL4 atsameoriginalabsoluteconstraintgate1.641e-9..2.741e-9. Existing convergence variability CONFIRMED on published132; NOTfixed and failed cases NOTqualified. Old128disabledintegrationalsohadcoastfailure. Followupenabled52PASS,disabled52PASS,legacyQOCO12652PASS. Full325PASS360.50s. All outcomes retained; don't claim every run passes or all coast results qualify.
- [tool] Six balancedtriples36complete transfers ALL independentphysics+unchanged1e-5kg finalmass gate2445.3111007852112PASS. Medians132349.171ms/133host302.373ms/133device355.157ms.145queuedassemblyguards,484matchingconicqualifications. Allupdates60adapterbytes; no reliable extra end-to-end speedup, allwarmups/outliersretained. Full conditional-IPM memcheck stillunresolved; isolatedsanpasses notfullqualification.
- [tool] Lambda read-only17:45UTC H100100%,1607MiB livePID53138 existingcampaign; nooffload/disruption/instance. Localhttp127.0.0.1:4173returns200, visualiserstillv11weighted12805.194102488575kg, no newfleet. AlllocalGPU/testhandlesnowterminal.
- [self] Checkpoint artifacts/performance/native-assembly-v133-checkpoint.json1258074bytes SHA149874ec071c35268ff430149c52862289c186afcc6a138666bf695bd4fe46ec embeds8sources, frozenruntimes, allhelpers/rawresults/failures, previous132hash. docs/GTOC12_DEVICE_ASSEMBLY_GUARD.md+README. Readyforauthorizedcommit/mainmerge/push.
- [self] NEXT: Investigate strict coast gate variability without relaxing it. cuDSS0.8 localheader has CUDSS_CONFIG_DETERMINISTIC_MODE (disabled default); NVIDIA docs https://docs.nvidia.com/cuda/cudss/general.html and types.html say different potentiallyslowerkernels offerbitwiserepro, unsupportedvendorIR/hybrid/multiGPU. Current QOCO128 cudss_backend.cu onlysetsUSE_SUPERPANELS0 plus ordering, no deterministicsetting. QOCO SpMv/USpMv ALREADY use stable GPUgather (old atomic kernels present but unused); don't repeat that redesign. QOCO own deviceIR may coexist with deterministic vendor mode if vendorIRdisabled, requires testing not assumed. Diagnostic option could distinguish cuDSSvariability; need inspect default vendorIR_N_STEPS beforechange. Then full outerSCvxGPUdispatch/report collection, warmretry, setup/analysis/vendorwarmup, perlegreuse/batching/fleetsearch. Current QOCO replay API rejects externalcapture, nestedconditionals need emission notcloning. No goalcompletion claim.

## 2026-09-07 04:28 AEST — Native v134 GPU integration control and reference refresh

- PROGRESS after the previous status turn's verified Lambda wait. New capture-compatible spacepdhcg_gtoc12_discretisation_launch_controlled_device reads device substeps and optional enable. Nonzero enables; zero preserves every output and invalid flag. Enabled substeps<1 marks invalid and preserves numerical outputs; consumers must check invalid. Same stream/device/lifetime/non-alias contract, no allocation/download/wait. Original fixed-step API remains. FP64 equations and all tolerances unchanged.
- SPACEPDHCG_TEST_GTOC12_DEVICE_REFRESH=1 queues controlled propagation, gated metrics reductions and gated set_reference after acceptance, before the regular command read. Final kernel clears refresh only after consumers. Removes host refresh branch and extra command read/wait. Still host attempt loop/time limit, per-solve reports, conic substep argument, warm retry, setup/analysis, per-leg reuse and fleet search. NOT fully GPU native; goal stays ACTIVE. No agents.
- Frozen native /home/angus/build-spacepdhcg-gtoc12-v134/final (library +6 tests); QOCO remains128. Next native135+, next QOCO131+. First build failed only new test initializer-list type deduction; corrected followup build succeeds, both logs retained. Sources frozen before validation. 24 changing-input and28 controlled graph cases pass both holds, analytic mass and affine closure; outputs bitwise match fixed API.12 captured refresh cases pass skip/invalid/enable/reference-state parity. Both tests memcheck/initcheck/synccheck zero errors; original controller/reductions pass.82 integration PASS;325 broad PASS363.61s (planner remainsv77). Full conditional-IPM memcheck remains unresolved.
- Six balanced triples36 full transfers ALL independent certificate and unchanged mass2445.3111007852112±1e-5kg PASS;425 conic qualification reports match host predicate. Medians133350.887ms/134host323.730ms/134device364.085ms; no extra speedup claim. Enabled12legs exactly16*(attempts+1) command bytes each. Extra refresh reads total1339/134host12/134device0, iteration histories differ. Solver/bridge reports excluded. AllGPUhandles terminal.
- Retained diagnostics from preceding work: QOCO129 determinism6/6freshprocesses illegal GPU access; baseline128/default12918/18coast each. Disabling IPM/factor/deviceIR/metric graphs did not rescue tested configs. Memcheck all-off invalid shared read in cuDSS fwd_dtmn_ker, not proven complete rootcause. Superpanels QOCO130 deterministic2configs fail; nondeterministic2/3strictpass. Standalone probe13010/12pass, det symmetricN64/N256 fail; expandedprobe131ALL8pass including previously failing symmetric default,2memcheckpass. Build/execution sensitive, no universal failure or verified workaround. Diagnostics NOT production/default. Formatted preparer reproduces frozen130source exactly. Source/probe/helper/raw evidence published in checkpoint.
- Ruiz coast90updates:0=18/18,1=16/18,3=18/18,5=17/18,10=17/18. Four bad cases external-audit rejection/unavailable vectors; earlier strict absolute failures remain. Full48transfers:Ruiz0=12/12;3/5/10each0/12. No promotion. Original coast harness mislabels Ruiz field deterministic (actualvendoroff); raw retained +correctly labelled derivative. Four-way full-run ordering cyclic not fullybalanced; negativequalification evidence only. No fixed accuracy variability claim.
- Lambda fresh18:21UTC H100100%,1607MiB livecampaignPID53138; nooffload/interruption/newinstance. Viewer remains v11 weighted12805.194102488575kg, no newfleet. Latest docs GTOC12_DEVICE_REFRESH.md, QOCO_DETERMINISM_DIAGNOSTICS.md+README.
- Checkpoint artifacts/performance/native-refresh-v134-checkpoint.json4098762bytes SHA51afb0affb6676425760b2caf6847a0658fca1901172ffed2aa3e24cb7db4381 embeds7sources, runtime hashes, helpers, all new validation/negative diagnostic evidence, previous133hash. Ready for authorized commit/main merge/push.
- NEXT move controlled substeps/enable through conic assembly and QOCO bridge; capture/retain refresh work then full outer GPU loop emission. QOCO queued API rejects external capture, conditional bodies must be emitted not cloned. Keep accuracy gates, unresolved strict coast variability and full-IPM sanitizer failure visible. Do not spend indefinitely on determinism/scaling: neither produced an improvement. User still wants full pipeline +fleet search/GPU and a stronger verified score.

## 2026-09-07 04:52 AEST — Native v135 device scheduling and live solver results

- PROGRESS: previous v134 published source9ed791df/main29c104cf. New controlled conic API reads GPU substeps/optional enable through integration+assembly; capture-compatible/no allocations/downloads/waits. Zero enable preserves packed outputs and invalid status, nonzero enables; bad step count/dynamics stops assembly before stale/uninitialized coefficient reads. First-ever invalid scheduling is safe. New QOCO controlled-step consumer bridge reuses existing invalid-input guard, synchronous/reporting contract and code3 rejection. Null pointer code1. Bridge itself still not graph-capturable. QOCO128 unchanged, FP64 equations/tolerances unchanged.
- SPACEPDHCG_TEST_GTOC12_DEVICE_SCHEDULING=1 propagates device count through assembly AND candidate measurement and automatically enables GPU reference refresh. Private Command now done/error/substeps/refresh (no public report-layout change); CPU loop reads first8 bytes only, never stepcount/refresh. Other replay/qualification flags remain separate. Firstmeasurement follows initialize on same stream; candidate measurement precedes decide's polish-count update; refresh follows. Full host loop/time limit, intermediate reports, warmretry, setup/workspace reuse and fleetsearch still unfinished. Goal ACTIVE, no agents.
- Frozen /home/angus/build-spacepdhcg-gtoc12-v135/final (library+8tests); next native136+, QOCO131+ ifneeded. Buildfirstpass. Conic48 changing-input +28controlled graph cases bothholds3/257intervals bitwise parity;4firstinvalid cases use uninitialized numerical inputs and preserve outputs. Conicmemcheck/initcheck/synccheck all0errors. Existinginterval24+28,refresh12,controller/reductionsPASS. ActualQOCOguard10queuedcallbacks status3/zeroIPM/unqualified,20fallbackno-callback,device-produced8/16steps, initialinvalid+null+recoveryPASS.83GPUintegrationPASS;326broaderPASS355.82s (plannerstillv77).8additionalSCvxPASS withdevicequalification0,refresh0,scheduling1 proving synchronous qualification fallback and automaticGPUrefresh.
- Sixbalancedtriples36complete transfersALLindependentcertificate and unchangedmass2445.3111007852112±1e-5kgPASS;434conic reports match hostqualificationpredicate. Median134327.924ms/135host320.271ms/135device357.430ms, noextraspeedupclaim. All12enabledruns commandbytes8*(attempts+1);all24comparisons16*(attempts+1). Solver/bridge reports excluded. Existingstrictcoastvariability andfullconditionalIPMmemcheckfailure remain unresolved. AlllocalGPUtesthandles nowterminal.
- Viewer at127.0.0.1:4173 hasnew GPU solver progress panel below Compute & optimisation. Metadata data/gtoc12/solver-progress.json generatedfromcheckpoint; syntheticfixture explicitlyseparatefromunchangedv11fleet12805.194weightedkg. Fixed misleading Final official score raw-mass label to Verifier returned mass. No fleetdata/scorechanged. Firstnpmcheck rejectedexternaldocslink underexistingoffline rule; changedlinktolocalbenchmarkJSON. npmcheck+38testsPASS. CUA reloadedexistingtab(provider cd30cc5b-35b2-48fc-9ffd-89c3d4e86cf0, browser1/tab2), clickedGPUheadingandvisuallycheckedreadablepanel:36/36,357.4ms,8bytes,326PASS,remainingCPUwork,knownaccuracylimits,checksum2a59a80597f88528. Weightedscore12805.194 andrawmass14047.8 correctlyseparated. Markedtabdeliverable. NoSKILL/agents used;CUAruntime docs read.
- Lambda18:43UTC H100100%,1607MiB livecampaignPID53138; nooffload/interruption/newinstance. Main goalstillactive. DocsGTOC12_DEVICE_SCHEDULING.md+README. User's previously requested results nowlocallydisplayed; fullpath results/lambda/2026-09-06/visualiser/data/gtoc12/solver-progress.json.
- Checkpoint artifacts/performance/native-scheduling-v135-checkpoint.json1105365bytesSHA2a59a80597f885285f968e60ab28e273aeb9147b61e9d3fadb6750406b99bca7 embeds11sourcesincludingviewer, frozenruntimes, helpers, allrawnewchecks/transfers,Lambda,previous134hash. Viewer metadata generated AFTERcheckpoint toavoidcyclicchecksum. Readyforauthorizedcommit/mainmerge/push.
- NEXT: retain/capture reference-refresh launch sequence (currently5hostlaunches/attempt, GPU-gated), then assemble+numericupdate+IPM+audit+qualification+SCvx into GPU-controlled outer loop. QOCO queuedAPI rejects externalcapture; nestedconditionalgraphs require bodyemission, notcloning. Native bridge stillwaits/reports/CPUfallback; need asynchronous graph ownership plus deferred per-attempt report storage and eventualwarmretryonGPU. Setup/perlegreuse/fleetsearch remain. Do not claimfullGPUorimprovedfleet. Determinism/scalingdiagnostics from134failed, notdefaults; avoidrepeatingwithoutnewreason.
