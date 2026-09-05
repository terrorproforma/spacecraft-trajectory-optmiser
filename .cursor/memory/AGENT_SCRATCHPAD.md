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

## Session Entries

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
