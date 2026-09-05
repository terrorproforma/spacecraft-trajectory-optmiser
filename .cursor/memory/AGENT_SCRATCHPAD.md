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

## Session Entries

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
  `cluster_fleet_v9` |Δλ| at collect departure median 2.5 deg / p75 4.0 deg, `fleet_master_v8`
  2.4 / 4.1, references 2.7 / 4.8 - so the brief's "nothing scores phase" diagnosis is not
  what separates the 210-day hops from the references' 181-day ones. The harvest-phase prio
  is calibrated and wired but bites on ~15 % of hops; measure it, do not expect it to move the
  fleet. The 210 vs 181 d and 85 vs 66 kg gap needs a different explanation (Δa / Δi of the
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
