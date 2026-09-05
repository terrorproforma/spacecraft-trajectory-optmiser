# DEVLOG

Rolled over on 2026-09-05: the detailed 2026-09-01..2026-09-05 history moved to
`DEVLOG_2026-09-01_to_2026-09-05.md` (both branches' entries, ours first, then the joint-itinerary
branch's merged from `main`). Carried forward: GTOC12 best verified fleet and the open levers
(see the scratchpad's Active Risks).

## 2026-09-05 13:50 AEST - G2/G3 reseal of main 8cb3759 on the WSL RTX 5090 (sm_120)

- Task summary:
  - Resealed Gate G2 (persistent device-resident PDHCG workspace) and Gate G3 (device-resident
    deterministic SCvx) on `main` `8cb3759b29ea8c7d843322a940a7ebcabfd9ff21` (tree
    `6d27f2552d882b4418d16e4342e6854a436a952d`) because the shared CUDA library changed after the
    b6afb49 (RTX 5090) and 9e75b47 (H100) seals through 1dbcae0 and 2bca11d. Both gates **PASS**.
  - Ran from a fresh worktree `/home/angus/worktrees/spacepdhcg-reseal-8cb3759` on branch
    `chore/g2g3-reseal-8cb3759` cut at 8cb3759 (`spacepdhcg-main` verified clean at 8cb3759 and left on
    `main`). No source change; the main evidence scripts already target `CMAKE_CUDA_ARCHITECTURES=120`
    and `hardware_id local-rtx-5090`, so the 9e75b47 script commit was not needed.
- Changes (this commit only; the sealed source is unchanged):
  - `docs/CURRENT_HEAD_G0_G3_REPORT.md` new section, `docs/G3_GATE_REPORT.md` pointer.
  - Compact evidence force-added under the ignored `results/gpu/current-head-8cb3759-rtx5090/`:
    root `evidence-index.json` + `.sha256`, `current-head-summary.json`, per-gate `summary.json`,
    `status.txt`, `commands.txt`, `manifest.txt`, `foreign-gpu-waits.log`, `evidence-index.json`, the
    runner scripts (`preflight/*.sh`, `g2/run.sh`, `g3/run.sh`, `g3/run_displaced_regressions.py`,
    `seals/*`), displaced/H1 decisions, `seals/archives.json` and the archive `.sha256` sidecars. The
    `.tar.gz` archives, raw logs and the nsys report stay local-only.
- Validation (all under the sealed template's tests/tolerances/timeouts; evidence at nice 5, builds at
  nice 10 `-j8`; wall 6795 s from 01:49:05Z to 03:42:20Z, seals stage re-run 03:47Z to include the
  orchestrator wait record):
  - G2 (775 s): CUDA Debug + RelWithDebInfo CTest 70/70 each; ten-update QP worst CPU error
    3.23909889e-7, pinned one-shot 3.90241894e-7, natural residual 3.09112063e-7; SOCP cone distance
    and natural residual 6.45576925e-11; CuPy/PyTorch/JAX DLPack max solution error 7.1133e-8;
    post-create allocation delta 0; 4 warm modes, checkpoint/restore, streams, cancellation,
    destruction, 5/5 error paths; 5 sanitizer logs clean (memcheck x2 with 0 bytes leaked, racecheck,
    initcheck, synccheck).
  - G3 (5786 s): Release + Debug CTest 70/70 each; tight canonical residuals HCW 9.69295039e-7,
    P1-C 1.42322019e-10, P1-E 4.58086731e-7, P1-D 2.82913893e-8 (max 9.69e-7 <= 1e-6); displaced HCW
    3 accepted steps, retained change 0.118457409, terminal 2.92768846e-8; pure-QOCO displaced
    warmups accepted 2/24/2 steps (P1-C/P1-D/P1-E) with canonical residuals 8.52e-12 / 7.73e-12 /
    4.28e-12 and terminal residuals <= 5.03e-11; fixed-tight PDHCG representatives 3/3 honest
    negatives (150 s timeouts, 0 accepted); production max canonical 9.56640559e-9, nonlinear
    2.92768846e-8, CPU/GPU trajectory difference 0, coefficient difference 2.76e-13; device
    variational max differences HCW 2.842e-14, pd3 8.327e-17, low thrust 4.139e-13, pd6 1.110e-16;
    topology allocation/copy deltas 0, no hidden CPU fallback; no-device control failed as expected
    (exit 2); H1 `supported` from 20 intervals (6 sizes x 7 repeats, omega bootstrap [0, 0]; SCvx
    medians 0.055 s @20 ... 31.0 s @10000); 16/16 sanitizer logs clean (recovery racecheck 56m31s,
    the dominant cost, as in both prior seals); Nsight under WSL again exposes no kernel/memory
    records (retained limitation, no residency claim).
  - Seals: `seals/g2-8cb3759b29ea.tar.gz` sha256
    `095f33dc83328290ea1533d0bc9531b17a316f004f0c7f8b5cd0057471fda45d`;
    `seals/g3-8cb3759b29ea.tar.gz` sha256
    `609e0acbed65d7c4449148677cbd69b2703ba23ea277a53a7034da742c439de6`; root `evidence-index.json`
    (178 artifacts) sha256 `443a8caf16e09699c67f499d59078261cfb94b5408c59e07c0e03dd83cd4e4a2`;
    `verify_seals.py` PASS; schema/scope pytest PASS. First-pass hashes before the seals re-run
    (a5a15c2b..., fe30ceb8..., 16370272...) are retained in `preflight/orchestrator-first-pass.log`.
  - Foreign GPU: a WSL weldsim `demo_everything_on.py --device cuda:0` (another agent) held the GPU at
    launch; the orchestrator waited 180 s (01:49:32Z-01:52:58Z) before G2 and recorded it. All 48
    per-step guards inside G2/G3 found the GPU clear; Windows `nvidia-smi.exe` never showed a
    `python.exe` compute workload.
- Follow-up notes / risks:
  - The `--sanitizer` 20k cancellation cap (9fafee8) did not shorten the recovery racecheck (56 min
    here vs 54 min b6afb49 / 61 min H100); the racecheck cost sits elsewhere in `recovery_test`.
  - `g3/summary.json` `timing.started_utc` is null because the runner's final `status.txt` carries only
    the completion stamp; start/end stamps are in the orchestrator log copied to `preflight/`.
  - G0/G1 were not re-run (out of scope); the H100 G4 claim core continues on 1dbcae0 untouched.

## 2026-09-05 16:10 AEST - ninth GTOC12 iteration (chain-aware beam, reference prior, LP duals, joint itinerary in the pricing)

- Integration: `results/gtoc12/runs/fleet_master_v7` (this branch's 20-ship master over the v8
  archives) renamed `fleet_master_v7_v8archives` with its run_id/artifact paths and docs rows
  (610c18d); `main` 8cb3759 merged (5eeb7da) - jointopt/jointcampaign, `gtoc12 joint-itinerary`,
  the 21-ship `fleet_master_v7`, the merged recursion limit `max(2n+200, n+500)`; conflicts in
  the two memory files and docs resolved by keeping both sides (main's §6.10 renumbered §6.11).
  Ruff clean; gtoc12 suites 125 passed (later 133 with the new tests).
- Code (9325252, 1f6ec50, and the fixes committed with the results):
  `src/spacepdhcg/gtoc12/chainprior.py` (reference extraction + `ChainPrior.penalty`),
  `search.py` (`chain_tour_scoring`, shortlist re-scoring by the exact DP tour through
  `_plan_from_tour`, `chain_burn` inheritance, tour cache, `asteroid_prices` in `_select` and
  `plan_score`, NaN burn guard), `collectdp.py` (non-finite `burn_per_hop` -> two-pass),
  `cooperative.py` (`usable_columns`, `lp_asteroid_prices` with `bound_share`, `_LpModel` row
  maps), `archive.py` (`pricing_columns`), `bundles.py` (settings, prices/prior into the beam,
  joint itinerary per self-cleaning slot, dispatch-time prices in `price_clusters`), `cli.py`
  (`chain-prior` command; cluster-fleet flags `--chain-tour-scoring --chain-tour-candidates
  --chain-prior --chain-prior-weight --no-lp-duals --dual-price-weight --dual-archive
  --dual-target-size --dual-bound-share --joint-itinerary --joint-budget-seconds`; joint-itinerary
  import fix). Tests: `tests/test_gtoc12_chain.py` (11 tests), `test_gtoc12_bundles.py` cache set.
- Data: `benchmarks/gtoc12/chain_prior_v1.json` (112 reference ships; collect hop median 66.3 /
  p75 82.9 kg, deploy hop median 97.9 / p25 75.7, collect share 0.42, |Δλ| at collect 2.7 deg).
- Runs: probe family 7 slot 1 622.6 kg (same chain as v8; +6.2 joint, 880 s); beam diagnostics
  (48 = 144 candidates; depth-9 closure fixed); `cluster_fleet_v9` 20 families / 60 ships,
  incumbent 9960.33 kg / 18 ships / 553.4 avg, PSS peak 0.92 GB, killed by `timeout 15600` at
  260 min before the final master (v8: 10 697.6 / 19); paired best ship per family 7 up / 6 down /
  6 equal, median 0.0 kg; collect hop 97.5 -> 90.6 kg median over all campaign ships, deploy
  716 -> 729 per ship, collected mean 465 -> 456; chain scorer 9562 tours / 4935 s / 582
  re-rankings; inline joint 11 slots +146.9 kg; duals 22-24 priced asteroids, one in a family.
  `joint_itinerary_v3` (best 40 v8/v9 ships): 35 improved, +411.5 kg, 466 s.
  `fleet_master_v8` (nineteen archives, 1006 routes re-flown, 0 failures, 1296 columns): 21 ships
  / 177 asteroids / 12 356.30 kg / 588.40 avg, LP bound 11 448.02 (gap 6.4), proven optimal,
  LP(22) infeasible; official "Check successfully!" + independent ok (mass error 1e-10 kg);
  +9.8 kg over `fleet_master_v7`, 3 ships swapped, 16 `v2` + 5 `v3` joint columns.
  Leg table `results/gtoc12/leg_stats/after_v9.json`: fleet deploy 838.5 kg/ship (refs 837-851),
  collect hop 84.3 kg / 210 d (refs 66 / 181-187), Earth-out 407 (refs 460-474), return 204.
- Validation: ruff format/check clean; `tests/test_gtoc12_*.py` 133 passed (8 min 43 s); both
  verifiers on `fleet_master_v8/fleet/Result.txt`.
- Follow-ups: phase-alignment term in the chain score; Earth-leg arrival-time trade; joint
  itinerary over all 992 archived stand-alone ships before the next master; `Result.txt` ->
  column ingester for `results/lambda-h100/gtoc12`; localise the NaN pass-1 burn (family 10).
