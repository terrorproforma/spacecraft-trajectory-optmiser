# DEVLOG

Rolled over on 2026-09-05: the detailed 2026-09-01..2026-09-05 history moved to
`DEVLOG_2026-09-01_to_2026-09-05.md` (both branches' entries, ours first, then the joint-itinerary
branch's merged from `main`). Carried forward: GTOC12 best verified fleet and the open levers
(see the scratchpad's Active Risks).

## 2026-09-05 08:10 AEST - Lambda H100: GTOC12 v2 breadth campaign launched (running unattended)

- Source sync: WSL `feat/gtoc12-asteroid-mining` 7d2e301 and `feat/gtoc12-joint-itinerary`
  f81e834/8e15b92 bundled (`/home/angus/bundles/to-h100/`) and merged into the H100 clone
  `~/spacepdhcg/gtoc12` (c4e2c31 -> 950fea7 -> fd59ad9 -> e9c9cd8 -> 282be45). One conflict
  (`cooperative.py` recursion limit, ba9b764 vs c4e2c31) resolved as `max(2n+200, n+500)`; both
  regression tests kept (union merge). The joint-itinerary and recursion changes were disjoint.
- New code e9c9cd8 (`bundles.family_partitions`, `cli.cluster_band_partitions`): cluster-fleet
  prices the union of several family partitions (`--cluster-radius 1.75,1.6 --all-family-bands`),
  labels offset per partition (`FAMILY_LABEL_STRIDE`), duplicate member sets dropped, cheapest
  first; run report lists the partitions; budget marks gain 480 min. Tests: partition union
  (monkeypatched ranker), CLI parsing; ruff clean; full gtoc12 suite 126 passed / 1 deselected
  (memory-budget test) on the merged tree. Family census on the 10612-asteroid pool at >= 20
  members: collect_r1.75 47, phasing_r1.75 56, collect_r1.6 29, phasing_r1.6 35 = 167 unique
  (ranking 203 s).
- Campaign `~/s/gtoc12_v2_campaign.sh` started 2026-09-04T18:03Z (CPU only, nice 5, GPU left to
  the G4 probe): cluster_fleet_h100_v2 (22 workers, 8 h, 6600 s/family, 5 ships, beam 32,
  refine-top 3, calibrated DP, harvest substitution, return-sweep cells) || joint_itinerary_h100_v1
  (4 workers, every archived chain >= 450 kg, fm7 ships first) -> joint_itinerary_h100_v2 (new
  chains, 22 workers) -> fleet_master_h100_v2 (fm7's 16 sources + cluster_fleet_v8 (copied from
  WSL, 59 routes) + cluster_fleet_h100_v1/v2 + joint_itinerary_h100_v1/v2, LP bound) -> official
  GTOC12_Verify over every Result.txt + `gtoc12 verify --official` + leg stats + chain stats.
  Extra: joint_itinerary_h100_v8 over cluster_fleet_v8 (35 ships, 31 improved, +369 kg) on the
  spare cores, not yet in a master.
- Measured: 2 h - 22 families, 71 ships, 10701.0 kg / 19 ships; 4 h - 44 families, 146 ships,
  11522.0 kg / 20 ships / 167 asteroids / 576.1 kg avg (above fleet_master_h100_v1's 11517.6
  from a single campaign); chains 38 >= 550, 4 >= 600, 1 >= 650 (652.6 kg). Joint v1 done in
  3.5 h: 339 ships, 294 improved, +4665 kg (mean +13.8, max +67.2), 0 insertions, chains >= 600
  kg 9 -> 10. Union of all archives now 569 chains, 15 >= 600 kg.
- Partial evidence copied to `results/lambda-h100/gtoc12/{cluster_fleet_h100_v2.partial-4h,
  joint_itinerary_h100_v1,joint_itinerary_h100_v8}`; helper scripts in `results/lambda-h100/scripts/`.
- Follow-ups: when `~/logs/gtoc12-v2-RESULT` shows `stage=done`, run `bash ~/s/finalize_v2.sh`
  on the host (docs, commits, bundle, tarball) and `C:\Users\Angus\h100work\pull_v2.ps1` locally;
  `bash ~/s/resume_v2.sh` if the orchestrator dies after cluster-fleet. Consider a
  fleet_master_h100_v3 that adds joint_itinerary_h100_v8.

## 2026-09-05 09:20 AEST - H100 G4 claim core launched on the executor deadline fix (1dbcae0)

- Fix bundle `/home/angus/bundles/single-gpu-v1-1dbcae0.bundle` (sha256 `5e4de5e3defaa538�`)
  verified in WSL and after scp on the H100. 1dbcae0 sits on addac2b beside 9e75b47 (the H100
  evidence-script commit), so it is not a fast-forward of the H100 `integration/single-gpu-v1`;
  checked out as branch `g4/h100-1dbcae0` in `~/spacepdhcg/v1` (clean), integration branch left
  at 9e75b47. Commit contains `cancellation_deadline_test.cu` and `test_g4_pdhcg_deadline_gpu.py`.
- Fix verification on the H100 (`results/lambda-h100/g4/fix-verification-1dbcae0/`): fresh sm_90
  configure + build, Release CTest 63/63 (190 s), Debug CTest 63/63 (196 s),
  `cancellation_deadline_test` 49.7 s / 48.8 s; GPU deadline matrix 13/13 in 1319 s (20 s cases
  180.7-181.6 s per 9-attempt session, N=2000 5 s cases replayed after three identical timeouts);
  ordinal-73 twin reproduction (600 s / 1,000,000 cap): warm-up/0 600.054 s, warm-up/1 600.029 s,
  both at inner_iterations = 300000 (recovery-phase cancel), measured/0 cancelled at 59.66 s under a
  group deadline clamped to 1260 s, remaining attempts `unrun`.
- Capability `~/g4/g4-executor-capability-1dbcae0-h100.json` sha256
  `0b4c8c38a6ba34b45cdf1ee5ae72869da272df5d866801009930f2b235a6f7f5` (compiled_source_commit
  1dbcae0, executable `3703d52c�`, libqoco `5f778efb�` = the 9e75b47 G1-G3 reseal library, QOCO
  cuda-algebra 09f0495 + absolute-KKT patch, IPM probe 9/9 QOCO workspace creations, status codes
  1,2,2,�). Fresh checkpoint `~/spacepdhcg/v1/build-integration-report/g4-claim-core-1dbcae0-h100`
  under amendment single-gpu-v1.2, 396 groups, no RTX 5090 rows; ccd5596 stratum
  `ipm_no_equilibration_v1_1` cited by metadata. `hardware.txt` records H100 80GB HBM3 / sm_90 /
  580.105.08 / CUDA 12.8.
- Launch 2026-09-04T18:55:16Z (04:55 AEST): GPU exclusive (no compute apps), worker pid 53138
  pinned to cores 0-3, observer 53137, logs `~/logs/g4-h100/`. 29 GTOC12 `cluster_fleet_h100_v2`
  processes had affinity 0-25 and were moved to 4-25 (`taskset -a -cp`). Contamination monitor
  run-and-flag with a native-Linux host channel (`~/s/host-pmon-linux.sh`: `nvidia-smi pmon` minus
  the worker's own tree); advisory lock dir `/home/angus` created for the hard-coded lock path.
- First-group evidence: 66 pure-gpu-ipm groups `numerical` x9 at 10-33 s each (same disposition
  class as the RTX 5090 stratum). Ordinal 66 (adaptive P1-E N=100, 120 s): 1080.73 s, 9 timeouts,
  cancel latency +0.012..+0.049 s, ~549k inner iterations per attempt. Ordinal 73 (censoring twin,
  600 s / 1M): 5400.67 s within the 5460 s group deadline, 9 timeouts, cancel +0.008..+0.051 s,
  inner_iterations 300000 on every attempt. 75 groups done, 0 contaminated attempts, no restarts.
- Projection at H100 speed: 1080.6 s per 120 s core group, 5400.7 s per twin; if every remaining
  PDHCG attempt times out (upper bound) 134 h remain -> ~2026-09-10T14:00Z; hybrid-pdhcg-ipm and
  fixed-tight classes are not yet sampled. Left running.
- Home copy: `results/lambda-h100/g4/` (87 files: capability, checkpoint snapshot, fix
  verification, ordinals 0/5/66/67/73, logs, scripts, STATUS.txt). Monitor:
  `ssh � 'bash ~/s/g4-status.sh'`, `'bash ~/s/g4_progress.sh'`, `'cat ~/g4/STATUS.txt'`.

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

## 2026-09-05 13:55 AEST - G2/G3 reseal of main 8cb3759 on the WSL RTX 5090 (sm_120): PASS/PASS

- Task summary:
  - Resealed G2 (persistent PDHCG workspace) and G3 (device-resident deterministic SCvx) on `main`
    `8cb3759b29ea8c7d843322a940a7ebcabfd9ff21` (tree `6d27f2552d88...`) because the shared CUDA library
    changed after the b6afb49 (5090) / 9e75b47 (H100) seals via 1dbcae0 and 2bca11d. Both gates PASS.
  - Worktree `/home/angus/worktrees/spacepdhcg-reseal-8cb3759`, branch `chore/g2g3-reseal-8cb3759`
    cut at 8cb3759 (`spacepdhcg-main` verified clean at 8cb3759 and left on `main`). No cherry-pick of
    9e75b47 was needed: main's scripts already hard-code `CMAKE_CUDA_ARCHITECTURES=120` /
    `hardware_id local-rtx-5090`. Sealed SOURCE = main 8cb3759 (evidence recorded before any commit).
- Changes (WSL commit `06e70b62c2c8e708a9221c7508e21b58e8d5da37`, author SpacePDHCG-Integration via
  env; no push/amend/reset; this Windows checkout only carries these memory notes):
  - `docs/CURRENT_HEAD_G0_G3_REPORT.md` new section (rendered from the sealed summaries),
    `docs/G3_GATE_REPORT.md` pointer, `.cursor/memory/*` entries, and the compact evidence force-added
    under ignored `results/gpu/current-head-8cb3759-rtx5090/` (root + per-gate `evidence-index.json`,
    `current-head-summary.json`, per-gate `summary.json`/`status.txt`/`commands.txt`/`manifest.txt`/
    `foreign-gpu-waits.log`, runner scripts, displaced + H1 decisions, `seals/archives.json`, archive
    `.sha256` sidecars). Archives, raw logs and the nsys report stay local-only.
- Validation (sealed b6afb49/9e75b47 per-gate procedure unchanged; evidence at nice 5, builds at
  absolute nice 10 `-j8`; foreign-GPU guard before every GPU step; wall 6795 s 01:49:05Z-03:42:20Z,
  seals stage re-run 03:47Z to include the orchestrator's wait record):
  - Preflight: pinned PDHCG 167c8b7 / tree 62b05e6c, QOCO 09f0495 + absolute-KKT patch built for
    sm_120 with cuDSS 0.7.1.6 (24 s).
  - G2 (775 s): CTest 70/70 Debug + 70/70 RelWithDebInfo; QP worst CPU error 3.23909889e-7, pinned
    one-shot 3.90241894e-7, natural residual 3.09112063e-7; SOCP 6.45576925e-11; DLPack
    CuPy/PyTorch/JAX max error 7.11e-8; post-create allocation delta 0; 4 warm modes,
    checkpoint/restore, streams, cancellation, destruction, 5/5 error paths; 5 sanitizer logs clean.
  - G3 (5786 s): CTest 70/70 Release + 70/70 Debug; tight canonical HCW 9.69295039e-7, P1-C
    1.42322019e-10, P1-E 4.58086731e-7, P1-D 2.82913893e-8 (max 9.69e-7); displaced HCW 3 accepted
    steps, retained change 0.118457409, terminal 2.92768846e-8; pure-QOCO displaced 2/24/2 steps
    (canonical 8.52e-12 / 7.73e-12 / 4.28e-12, terminal <= 5.03e-11); fixed-tight 3/3 honest
    negatives (150 s timeouts); production canonical 9.56640559e-9, nonlinear 2.92768846e-8, CPU/GPU
    trajectory difference 0, coefficient difference 2.76e-13; device variational HCW 2.842e-14 / pd3
    8.327e-17 / low thrust 4.139e-13 / pd6 1.110e-16; topology deltas 0, no fallback; no-device
    control exit 2 as expected; H1 supported from 20 intervals (6 sizes x 7 repeats, omega [0, 0];
    SCvx medians 0.055 s @20 ... 31.0 s @10000); 16/16 sanitizer logs clean (recovery racecheck
    56m31s); Nsight under WSL exposes no kernel/memory records (retained limitation).
  - Seals: g2 `095f33dc83328290ea1533d0bc9531b17a316f004f0c7f8b5cd0057471fda45d`, g3
    `609e0acbed65d7c4449148677cbd69b2703ba23ea277a53a7034da742c439de6`, root `evidence-index.json`
    (178 artifacts) `443a8caf16e09699c67f499d59078261cfb94b5408c59e07c0e03dd83cd4e4a2`;
    `verify_seals.py` PASS; schema/scope pytest PASS; sidecars re-checked with `sha256sum -c`.
  - Foreign GPU: WSL weldsim `demo_everything_on.py --device cuda:0` (another agent, pid 1794260)
    held the GPU at launch; the orchestrator waited 180 s before G2 and recorded it; all 48 in-gate
    guards clear; Windows `nvidia-smi.exe` never showed a `python.exe` compute workload.
- Follow-up notes / risks:
  - The `--sanitizer` cancellation cap (9fafee8) did not shorten the recovery racecheck; the cost sits
    elsewhere in `recovery_test`.
  - `g3/summary.json` `timing.started_utc` is null (runner's final `status.txt` drops it); start/end
    stamps are in `preflight/orchestrator-first-pass.log`.
  - Worktree, build dirs (~950 MB), `_upstream/`, `.venv-current-head` and `/home/angus/reseal8cb/`
    left in place; branch not pushed.

## 2026-09-05 15:20 AEST - Lambda H100: GTOC12 v2 pipeline watched to completion, finalised, pulled home (v3 master running)

- Watch (`~/s/v2_status.sh` + `~/s/v2_poll.sh`, local log `C:\Users\Angus\h100work\v2_watch.log`, 9 polls
  22:41Z-04:41Z): families 44 -> 49 -> 61 -> 85 -> 89 -> 98 -> 111 (final, of 167); incumbent 11522.0 / 20
  ships (236 min) -> 11525.1 / 20 (315 min) -> 12348.7 / 21 / 588.0 avg (403 min) -> 12348.9 / 21 / 182
  asteroids (530 min, final). No crash, no stall, no OOM; one family (`family_200010`, collect_r1.6, 27
  members) died inside its worker with `ValueError('cannot convert float NaN to integer')` at 0 s and was
  recorded as `stopped: crashed` (0 ships) - the campaign continued. Cluster-fleet exit 0 at 03:15:35Z
  (552 min wall, 386 ships, 62 >= 550 / 7 >= 600 / 1 >= 650 kg, PSS peak 5.07 GB); joint v2 231 ships /
  219 improved / +3211.6 kg / >= 600 kg 6 -> 13 in 20 min on 22 workers; master 03:35-04:35Z (1911
  recert tasks 2411 s, master 1135 s, 2480 columns); verify PASS, `stage=done` 04:41:29Z.
- **fleet_master_h100_v2: 13189.60 kg, 22 ships, 187 asteroids (184 mined), 599.53 kg average** -
  first fleet at the 22-ship threshold (599.5 kg; independent verifier `ship_limit` 22.0047). Master
  objective 12203.96 kg, LP bound 12207.39, gap 3.43 kg (2 M node cap + 20000 LP nodes, not proven).
  Official `GTOC12_Verify` "Check successfully!" on the fleet; 2492/2826 emitted Result.txt pass (all
  24/24 cluster fleets, 294/294 joint v1, 219/219 joint v2; per-ship cooperative diagnostics fail
  Error803 by construction). Chain-mass union over the 21 sources: 1207 chains, 161 >= 550, 21 >= 600,
  1 >= 650 (652.6), top10 652.6/649.7/634.4/632.3/628.5/625.4/624.0/622.6/620.8/620.1.
- Finalise (`taskset -c 4-25 nice -n 5 bash ~/s/finalize_v2.sh`, author via env): commits `810f041`
  (H100 v1 artefacts) and `86a91d3` (v2 campaign + docs section 7, 6 rows / 8 paragraph lines); viewer
  export + v2 importer (22 ships, 187 asteroids, 11179 replay samples, fleet.json sha a3c20148...c4cf,
  Kepler 3.59e-6 / 3.07e-7 km) written to `~/stage/viewer-import/` so the read-only v2 clone is not
  touched (finalize_v2.sh edited: `node <v2>/scripts/import-gtoc12.mjs ... --output`). Bundle
  `~/bundles/from-h100/gtoc12-h100-v2-86a91d3.bundle` (12.8 MB, sha256 d5cea153...a1a1), tarball
  `~/stage/gtoc12-h100-v2-compact.tgz` (7.2 MB, sha256 b0da1b5d...d3f4).
- Pull (`pull_v2.ps1`): SHA-256 of both files match the host (`verify_pull.ps1`); tarball unpacked into
  `results/lambda-h100/gtoc12/` (+ `logs/`, `bundles/`, `viewer/fleet_master_h100_v2/`); WSL
  `refs/h100/gtoc12-asteroid-mining` 282be45 -> 86a91d3 (6 ahead / 77 behind the moving WSL branch,
  merge later). Local official verifier (WSL venv) on the pulled `fleet/Result.txt`: ok, 22 ships, 184
  mined, 13189.6 kg, agreement with the independent verifier exact (Result.txt sha256 beffeeff...0548).
- v3 launched 05:09:43Z (`~/s/master_v3.sh`, `taskset -c 4-25`, 22 workers): 29 sources = v2's 21 +
  joint_itinerary_h100_v8, fleet3_full_catalogue{,_v2}, fleet6_coop_v1, fleet6_retime_v1,
  full_catalogue_search2, reduced_v1_search2/3 (probe `probe_omitted.py`: 1965 archives, 817 groups).
  Excluded on purpose: fleet_master_h100_v1 + its attempt1 (800 re-certified duplicates each, 0 new sets,
  0 better masses) and fleet_master_v1-v7 (no ship_NN archives).
- Local helper scripts: `C:\Users\Angus\h100work\{poll_v2,pull_v3,verify_pull,lf}.ps1`,
  `C:\Users\Angus\h100work\s\{v2_poll,master_v3,finalize_v3,probe_omitted,launch_*}.sh`,
  `{verify_runs2,docs_v3,probe_omitted}.py` (all mirrored to `~/s` on the host, LF, bash -n / py_compile clean).

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

## 2026-09-05 16:40 AEST - Third release merge into main: gtoc12 v9, H100 v2/v3 results, G2/G3 reseal, Windows memory + Lambda evidence

- Task summary:
  - Integrated every completed line since main 8cb3759 onto `release/single-gpu-v1-merge` (WSL
    worktree `/home/angus/worktrees/spacepdhcg-release`, author/committer SpacePDHCG-Integration via
    env; merge commits only, no rebase/squash/amend/force), folded the Windows checkout's memory and
    Lambda H100 evidence, verified the head, fast-forwarded `main`, bundled to Windows and pushed from
    there. Helper scripts and logs: `/home/angus/integ3/`.
- Merge chain (first parent on top of 8cb3759; parents in brackets):
  - `abd4e81` merge `integration/single-gpu-v1` bf4cf0f [8cb3759, bf4cf0f] - evidence scripts target
    the local CUDA architecture (H100 9e75b47). No conflicts.
  - `16d5e8e` merge `chore/g2g3-reseal-8cb3759` 06e70b6 [abd4e81, 06e70b6] - G2/G3 reseal evidence
    (PASS/PASS, `results/gpu/current-head-8cb3759-rtx5090/`). No conflicts.
  - `a93649d` merge `feat/gtoc12-asteroid-mining` 1f6ec50 [16d5e8e, 1f6ec50] - chain-level objective,
    chain prior, LP duals, archive-seeded pricing. Conflicts: memory files (both sides appended) ->
    both kept chronologically.
  - `ace3b25` merge `feat/gtoc12-asteroid-mining` b55eb70 [a93649d, b55eb70] - `--dual-bound-share`,
    NaN burn guard, joint-itinerary CLI import fix, `cluster_fleet_v9` / `joint_itinerary_v3` /
    `fleet_master_v8`, memory rollover. Conflicts: `AGENT_SCRATCHPAD.md`, `DEVLOG.md` (branch rolled
    over vs our appended reseal entries) -> branch's slim live files with the 13:50 reseal entry
    inserted before the 16:10 ninth-iteration entry; snapshots from the branch; coverage 0/0 missing.
  - `aaa9657` merge `refs/h100/gtoc12-asteroid-mining` 86a91d3 [ace3b25, 86a91d3] -
    `family_partitions`, H100 v1/v2 results (`fleet_master_h100_v2` 22 ships). Conflicts:
    `gtoc12/cli.py` imports (v9 names + `REPOSITORY_ROOT`-free import; `ClusterBands` now inside
    `cluster_band_partitions()`), `cooperative.py` comment (code identical), `GTOC12_TRACK.md` §7
    rows (both kept; H100 joint rows cite §6.11). 147 gtoc12 + cli-dispatch tests on the resolved tree.
  - `1bd78ce` Windows memory fold (`win_mem_merge.py`: cp1252 repair, ASCII-folded novelty, preamble
    blocks carried under "Windows-Checkout Notes", sections by boundary into live/snapshot, insert-only).
  - `7a30c12` `.gitignore` (`.tmp_*`, raw lambda-h100 parts); `5784e64` `results/lambda-h100` compact
    evidence 608 files / 5.95 MB (+ `INDEX.json`); 1171 files / 94.7 MB left out (bundles, run trees
    already on main under `results/gtoc12/runs/*_h100_*`, viewer datasets, seal tarballs, CSV replays,
    `stdout.jsonl`, sqlite, patch, files >= 200 KB).
  - `5f23f73` merge `refs/h100/gtoc12-asteroid-mining` 48e5fb7 [5784e64, 48e5fb7] -
    `fleet_master_h100_v3` (byte-identical fleet to v2), `joint_itinerary_h100_v8`; docs auto-merged,
    §6.10 -> §6.11 in the new row fixed in this commit.
  - status/memory commit (this entry): `docs/PROGRAM_STATUS_2026-08-31.md` third integration note +
    GTOC12 headline.
- Windows checkout: spec edits (README, BENCHMARK_PROTOCOL, matrices, OUTLINEs, test_benchmark_manifests,
  literature_baselines, COMPARATIVE_SOLVER_CAMPAIGN) - every line added there is already on main
  (only Ruff re-wraps differ) -> nothing committed; `.gitignore` edits already on main; scratch
  `.tmp_*` (6 files) moved to `%LOCALAPPDATA%\Temp\spacepdhcg-tmp\`. Landing: bundle
  `release-merge-3-e259809.bundle` (14.4 MB, sha256 `454adbcc…854c` on both sides) fetched by
  Windows git; pushed as fast-forwards `main` 8cb3759 -> e259809, `release/single-gpu-v1-merge`
  8cb3759 -> e259809, `integration/single-gpu-v1` 1dbcae0 -> bf4cf0f, `feat/gtoc12-asteroid-mining`
  4dd4fdb -> b55eb70, new `chore/g2g3-reseal-8cb3759` 06e70b6 and `h100/gtoc12-asteroid-mining`
  48e5fb7. Windows checkout: dirty files backed up to
  `%LOCALAPPDATA%\Temp\spacepdhcg-tmp\win-checkout-backup-20260905-164309\`, tree restored,
  `git checkout main` -> e259809 clean. Incident: the pre-checkout `git checkout -- .` reverted the
  local `.gitignore` edits before `git clean -f`, which deleted the ignored `_upstream/` checkouts
  and `traj-key.pem`; both restored (key from `Downloads`, upstreams re-cloned at the pinned
  commits, trees verified). See the scratchpad entry.
- Validation (head 5784e64; `cpp/` byte-identical to 8cb3759; RTX 5090 with a foreign 3.5 GB
  workload left alone; logs `/home/angus/integ3/logs/`): ruff check/format clean (302 files);
  generated-artefact checks clean; host RelWithDebInfo -Werror fresh 0 warnings, ctest 50/50;
  cpp/native 8/8; CUDA sm_120 Release -Werror clean rebuild 0 warnings, CUDA CTest 70/70 (248 s);
  planner GPU pytest 9/9; full CPU pytest 677 passed / 35 skipped (595 s; +18 tests); manifest tests
  14/14 (no blob refresh needed); viewer `npm run check` + `npm test` 36/2 skips with
  `fleet_master_h100_v2` regenerated via `gtoc12 export-viewer` (22 ships, 187 asteroids,
  13,189.60 kg, fleet SHA `cbedee96…fd48`); wheel (`c3cc5186…`) + sdist + consumer-venv smoke all rc 0.
  Not repeated: `test_g4_pdhcg_deadline_gpu.py` (13/13 on the same CUDA sources at 06:30 and 13:50).
- Off main: `feat/gtoc12-asteroid-mining` past b55eb70 (bc7ef8e+); H100 G4 claim core (running);
  sm_90 confirmation of the v2 fixes; `perf/g4-batched-campaign`; raw Lambda artefacts per INDEX.json.
## 2026-09-05 21:30 AEST - tenth GTOC12 iteration (Earth-out leg stage, harvest-phase prior, archive-wide joint, H100 paired arms)

- Integration: `refs/h100/gtoc12-asteroid-mining` 86a91d3 (H100 v2 line: `family_partitions`,
  `--cluster-radius a,b`, `--all-family-bands`, H100 v1/v2 artefacts) merged as bc7ef8e; three
  one-hunk conflicts (cli.py imports: chainprior + lp_asteroid_prices kept, REPOSITORY_ROOT
  dropped for `resources.repository_root()`, ClusterBands import moved into
  `cluster_band_partitions`; cooperative.py comment; docs §7 rows both kept). All CLI flags
  from both sides present; gtoc12 suite 138 passed.
- Code: 9ce3162 Earth-out leg stage (`jointopt.JointSettings.earth_leg*`,
  `JointItinerary.earth_leg_candidates/earth_leg_seed/first_collect`, `free_earth_leg`,
  `_screen_earth_out`, `earth_out_inflation`, `certify_leg` hook, `JointResult.earth_leg`;
  `jointcampaign` settings + totals; CLI `joint-itinerary --earth-leg --earth-leg-shifts
  --earth-leg-certifications`, `cluster-fleet --joint-earth-leg`, `ClusterPricingSettings.
  joint_earth_leg`); f8e870c harvest-phase prior (`harvestphase.py`, `gtoc12 harvest-phase`,
  `benchmarks/gtoc12/harvest_phase_v1.json`, `CollectDPSettings.harvest_phase/phase_weight`,
  `CollectPairTable.phase_deg/phase_penalty`, DP move cost + `CollectTour.hop_phase_deg/
  phase_penalty_kg`, `SearchSettings.harvest_phase_path/weight`, `_chain_score` penalty,
  `cluster-fleet --harvest-phase --harvest-phase-weight`); 8e2b6bf/7c6c68a/ec23f01
  `scripts/gtoc12_campaign_report.py` (roles, TOFs, |Δλ|, relative inclination / node gap /
  Δa of the collect pairs, chain-mass counts, paired families, joint totals, masters).
  Tests: `tests/test_gtoc12_jointopt.py` +2 (seed bookkeeping vs forward replay; monotone
  certified-only stage acceptance with trusting/refusing/dearer leg certifiers, determinism),
  `tests/test_gtoc12_harvestphase.py` 4 (penalty arithmetic, DP ranking flip on a synthetic
  pair, chain-score penalty, extraction reproducibility vs the committed document). Ruff clean.
- Data (`harvest_phase_v1.json`, 1014 reference collect hops): |Δλ| median 2.67 deg, p75 4.81,
  p90 7.32; 2.46 kg/deg, 2.62 d/deg over harvest hops (|Δλ| <= 30 deg, TOF <= 400 d);
  exchange 0.246 kg/day (9 asteroids); 99.8 % of reference hops depart within 15 deg.
- Runs (local, 3 workers nice 19): `joint_itinerary_v4` 562 stand-alone sets (19 local + 4 H100
  archives), 99 improved, +316.0 kg, 18 min; `joint_itinerary_v5` (`--earth-leg`, top 300):
  131 improved, +494.0 kg, 296 stages / 720 legs flown / 534 measured / 126 accepted shifts
  (87 x 30 d, 33 x 60, 5 x 90, 1 x 150), Earth-out TOF 600 -> 570 d median at 416 -> 433 kg,
  30 min; `fleet_master_v9` (25 archives, 2235 routes re-flown, 2849 columns): **22 ships /
  187 asteroids / 13188.61 kg / 599.48 avg, LP gap 0.95 kg, proven optimal, rule 22 <= 22.0007**,
  official "Check successfully!" + independent ok, 124 min, main 1.30 GB. 1.0 kg below
  `fleet_master_h100_v2` (whose `joint_itinerary_h100_v8` source is not archived locally).
- Runs (Lambda H100, cores 4-25, nice 5, CPU only; G4 untouched on 0-3 + GPU): bundle
  `gtoc12-v10-f8e870c.bundle` (sha 907f20b1...) merged into the host clone as 735aa1a (no
  conflicts); paired arms `cluster_fleet_v10` (v9 line + `--harvest-phase --joint-earth-leg`)
  and `cluster_fleet_v10_control` (v9 line), 11 + 11 workers, 35 families each, 172/173 min,
  61/63 ships, incumbents 11516.2 / 11520.3 kg (20 ships), PSS peak 2.23/2.24 GB; per family
  the two arms are neutral (identical best ship in most families; family 0 +42.3 kg, family 2
  -7.6, the rest within +-2.5); with 2400 s per family the H100 cores reach ship slot 2-3 where
  the WSL box reached 3-4. `joint_itinerary_v10` (`--earth-leg`, 22 workers): 47 of 51
  improved, +663.6 kg, 398 s; union of the arms 70 chains / 7 >= 600 kg / best 644.4.
  `fleet_master_v10` (36 sources, 2449 routes re-flown in 3265 s, 3142 columns, master 385 s):
  **23 ships / 196 asteroids / 14044.80 kg / 610.65 kg avg (threshold 610.6; rule 23 <= 23.005),
  LP gap 6.3 kg, proven optimal, official "Check successfully!" + independent ok** - the 23rd ship
  comes from six new arm/joint_v10 chains (644.4, 635.7, 624.0, 617.1, 614.4, 610.0) plus the
  joint/Earth-leg passes' 654.1 / 623.8 / 622.2. H100 commit c2730b1 (bundle
  `gtoc12-h100-v10-c2730b1.bundle` sha 56a104f5..., tarball sha b89333d6...) fetched as
  `refs/h100/gtoc12-asteroid-mining` and merged; compact copies under the Windows repo's
  `results/lambda-h100/gtoc12/{cluster_fleet_v10,cluster_fleet_v10_control,joint_itinerary_v10,
  fleet_master_v10,leg_stats}`.
- Measurement that should have preceded the design: our chains are already phase-aligned at
  harvest (|Δλ| at collect departure median 2.4-2.7 deg, p75 4.0-4.7 in the v8/v9 fleets vs the
  references' 2.7 / 4.8); the collect pairs differ in the orbital plane instead (relative
  inclination median 2.4-2.7 deg vs 1.85, node gap 34-35 deg vs 20, same Δa 0.013 AU, same
  drift alignment 37-40 %); our hops fly 2.64 km/s where the references' 69 kg imply ~2.0.
- Validation: ruff check/format clean; `tests/test_gtoc12_*.py` 138 passed on the merge, +6 new
  tests passing (jointopt 9, harvestphase 4, chain/collectdp 33); H100 quick suite 13 passed;
  both verifiers on `fleet_master_v9/fleet/Result.txt` (and on `fleet_master_v10`'s, see §7).
- Follow-ups: weight the relative inclination / node gap of consecutive pairs in the family
  bands and the beam's expansion order; sweep archived Earth legs with single-leg SCvx across the
  launch window (405 vs 430 kg legs exist); a deploy-phase time weight (deploy hops 240 d vs 183);
  the H100 host's per-core speed is ~0.6x the WSL box - size per-family budgets accordingly.

## 2026-09-05 21:40 AEST - tenth GTOC12 iteration: ship 23 reached (fleet_master_v10 14044.80 kg / 23 ships / 610.65 avg)

- Where: WSL worktree `/home/angus/worktrees/spacepdhcg-gtoc12`, branch `feat/gtoc12-asteroid-mining`
  b55eb70 -> dfdeca8f (14 commits, clean; the long-form devlog/scratchpad entries are committed there in
  `.cursor/memory/`). Helper scripts + logs `/home/angus/v10/` (`run.sh` strips CR and sets the git identity
  env; `rput.sh`/`rrun.sh` for the H100). H100 clone `~/spacepdhcg/gtoc12` at c2730b1 (clean); bundle
  `/home/angus/bundles/from-h100/gtoc12-h100-v10-c2730b1.bundle` fetched as `refs/h100/gtoc12-asteroid-mining`
  and merged; compact copies in this repo's ignored `results/lambda-h100/gtoc12/{cluster_fleet_v10,
  cluster_fleet_v10_control,joint_itinerary_v10,fleet_master_v10,leg_stats,logs,bundles}`.
- Code: merge of the H100 v2 line 86a91d3 (both mechanisms, every CLI flag from both sides, both docs
  blocks); Earth-out leg stage in the joint itinerary (`jointopt` `earth_leg`, single-leg SCvx oracle,
  prefix seeds, monotone acceptance; `joint-itinerary --earth-leg`, `cluster-fleet --joint-earth-leg`);
  harvest-phase prior (`harvestphase.py`, `benchmarks/gtoc12/harvest_phase_v1.json`, DP move penalty +
  chain score; `cluster-fleet --harvest-phase`); `scripts/gtoc12_campaign_report.py`. Tests +6 (144 pass).
- Runs: `joint_itinerary_v4` (562 sets, +316 kg), `joint_itinerary_v5` (`--earth-leg`, 300 ships, +494 kg,
  126 accepted shifts), `fleet_master_v9` (local, 25 archives): 22 / 13188.61 / 599.48, proven optimal;
  H100 paired arms `cluster_fleet_v10` vs `_control` (35 families each, median delta 0.0 kg: the phase
  prior and the Earth-out stage are neutral), `joint_itinerary_v10` (+663.6 kg over 51 fresh chains),
  **`fleet_master_v10` (36 archives): 23 ships / 196 asteroids / 14044.80 kg / 610.65 avg, LP gap 6.3 kg,
  proven optimal, official GTOC12_Verify + independent verifier ok** (re-verified locally on the merged
  tree, Result.txt sha256 3c052997...1b3e). Ship 24 needs 621.2 kg average.
- Findings: our chains were already phase-aligned at harvest (|Δλ| 2.4-2.7 deg vs the references' 2.7);
  the collect pairs differ in the orbital plane (relative inclination 2.5 vs 1.85 deg, node gap 34 vs
  20 deg); a certified Earth leg is rarely the cheapest of its launch window (405 vs 430 kg); the H100's
  per-core speed is ~0.6x the WSL box (1 ship slot less per 2400 s family).
- Not done / follow-ups: plane-aware families; single-leg SCvx Earth-leg sweeps; deploy-phase time
  weight. The G4 job on the H100 (cores 0-3 + GPU) was never touched; no GPU process of ours anywhere.

## 2026-09-05 22:10 AEST - Fourth release merge into main: gtoc12 v10 (fleet_master_v10, 23 ships), Windows memory notes, Lambda v10 evidence

- Task summary:
  - Integrated `feat/gtoc12-asteroid-mining` dfdeca8f (the branch tip at merge time; no v11 commit had
    landed on it) onto `release/single-gpu-v1-merge` from 2aecc65 (WSL worktree
    `/home/angus/worktrees/spacepdhcg-release`, author/committer SpacePDHCG-Integration via env; merge
    commits only, no rebase/squash/amend/force/reset; the gtoc12 worktree was only read). Helper scripts
    and logs: `/home/angus/integ4/` (`run.sh` strips CR and cd's to the helper dir first, `merge_one.sh`,
    `commit_merge.sh`, `resolve_conflicts.py`, `win_mem_fold.py`, `regen_index.py`, `60_*`/`65_*`/`66_*`).
- Merge chain (first parent on top of 2aecc65; parents in brackets):
  - `fd7ef6d` merge `feat/gtoc12-asteroid-mining` dfdeca8f [2aecc65, dfdeca8f] - criss-cross merge
    (bases 48e5fb7 and b55eb70). Conflicts, one hunk each: `AGENT_SCRATCHPAD.md` + `DEVLOG.md` (main's
    16:40 entry vs the branch's tenth-iteration entry) -> both kept in order; `GTOC12_TRACK.md` §7
    (H100 v3 row vs the seven new rows) -> rows appended after the v3 row, H100 joint rows keep §6.11;
    `cooperative.py` recursion-limit comment -> branch wording (code identical). Coverage: 0 branch
    lines missing anywhere; 8 main scratchpad lines missing = the superseded `fleet_master_v8`
    Active-Risks bullet; 4 main comment lines in cooperative.py by choice.
  - `ed71737` Windows memory fold (`win_mem_fold.py`): the Windows checkout's uncommitted 21:40
    tenth-iteration notes - DEVLOG section inserted after the branch's 21:30 entry, three novel
    Active-Risks bullets carried verbatim under a dated subsection of "Windows-Checkout Notes";
    insert-only (0 removed), 0 Windows lines missing after ASCII-folded comparison.
  - `f4028b3` `results/lambda-h100` +28 compact files (234 KB: `fleet_master_v10` / `cluster_fleet_v10`
    chain stats, `fleet_master_h100_v3` stats + verify, `leg_stats/{after_v10,v10_report}.json`, 18 small
    logs) under the 5784e64 policy; `INDEX.json` regenerated by `regen_index.py` over every tracked file
    (636 kept with sha256 - the commit message says 637, a typo - 1569 skipped with sizes; 0 previously
    indexed hashes changed; the 592 byte-differing Windows copies differ by CRLF only).
  - docs/status/memory commit (this entry): `docs/PROGRAM_STATUS_2026-08-31.md` fourth integration note
    + GTOC12 headline (the 16:40 headline marked superseded); README quotes no headline; `GTOC12_TRACK.md`
    §6.13/§7/§8 came with the branch.
- Validation (head f4028b3; `cpp/` and `scripts/gpu/` byte-identical to 2aecc65 -> host/native/CUDA
  builds and the GPU matrix of the third merge stand, not repeated; RTX 5090 foreign ~3.6 GB workload
  left alone; logs `/home/angus/integ4/logs/`): ruff check/format clean (305 files); generated-artefact
  checks clean (provenance 126 records, 34 packaged assets); full CPU pytest 683 passed / 35 skipped in
  631 s (+6 tests: jointopt 2, harvestphase 4); gtoc12 suites + CLI dispatch 153 passed (478 s) on
  fd7ef6d; manifest tests 14/14; viewer `npm run check` + `npm test` 36 pass / 2 skips with
  `fleet_master_v10` regenerated via `gtoc12 export-viewer` and imported (23 ships, 196 asteroids, 11,681
  samples, 14,044.8 kg, fleet SHA `b9b3b6ba…aa62`, solution SHA `3c052997…1b3e` = committed manifest,
  Kepler max 3.59e-6 km; palette 40 colours, synthetic 21/39/40 pass, 41 refused); wheel
  (`13f984de…`) + sdist (31.8 MB) + consumer-venv smoke rc 0 (`gtoc12 --help` lists chain-prior /
  harvest-phase / joint-itinerary / export-viewer; `joint-itinerary --help` has `--earth-leg
  --earth-leg-shifts --earth-leg-certifications`; `cluster-fleet --help` has `--harvest-phase
  --harvest-phase-weight --joint-earth-leg --chain-prior --dual-bound-share --all-family-bands`).
- Off main: `feat/gtoc12-asteroid-mining` beyond dfdeca8f (v11 worker, if started); H100 G4 claim core
  (running); sm_90 confirmation of the v2 fixes; `perf/g4-batched-campaign`; raw Lambda artefacts per
  INDEX.json; sdist include list (web/ + `tests/__pycache__`).

## 2026-09-05 22:20 AEST - Fourth release merge landed: pushed refs, Windows checkout on 05d972f, live viewer on fleet_master_v10 (23 ships)

- Landing: `main` fast-forwarded in `/home/angus/worktrees/spacepdhcg-main` 2aecc65 -> 05d972f (16 commits, first
  parents fd7ef6d / ed71737 / f4028b3 / 05d972f); bundle `release-merge-4-05d972f1.bundle` (6.9 MB, sha256
  `f10130a9…6a9a` identical in `/home/angus/bundles/` and on the Windows desktop) fetched by Windows git into
  `refs/integ4/*` (left in place, refs only), fast-forwardness of every target checked with
  `merge-base --is-ancestor`, then pushed from PowerShell: `main` 2aecc65 -> 05d972f,
  `release/single-gpu-v1-merge` 2aecc65 -> 05d972f, `feat/gtoc12-asteroid-mining` b55eb70 -> dfdeca8f (the tip
  merged; the branch had not moved, its worktree carried 5 uncommitted v11 files at bundle time),
  `h100/gtoc12-asteroid-mining` 48e5fb7 -> c2730b1. `git ls-remote` from Windows and from WSL both show the four
  refs at those SHAs.
- Windows checkout (`C:\Users\Angus\Desktop\projects\spacecraft-trajectory-optmiser`): it was *not* clean - the
  tenth-iteration worker had written the 21:40 notes into `.cursor/memory/{AGENT_SCRATCHPAD,DEVLOG}.md`
  (+42/-8 lines, unchanged since the fold). Those two files were copied to
  `%LOCALAPPDATA%\Temp\spacepdhcg-tmp\win-checkout-backup-20260905-220632\.cursor\memory\` and then saved
  with `git stash push -- <the two files>` (stash@{0}, message names the fold commit ed71737 and the backup);
  no `checkout -- .`, no `clean`. `git pull --ff-only origin main` -> 05d972f, `git status --porcelain` empty,
  `_upstream/` and `traj-key.pem` present. The stash is not popped: its content is on main (ed71737).
- Live viewer (`C:\Users\Angus\Desktop\projects\viewer-live`, node pid 47428 on :4173, `Cache-Control:
  no-store`): `data/gtoc12` (22 ships, `fleet_master_h100_v3_fleet`, fleet sha `f7c53c1e…aa6e`) copied to
  `data/gtoc12.bak-h100v3-22ships`; inputs staged in `%LOCALAPPDATA%\Temp\viewer23\inputs\` from the WSL
  release worktree (export `trajectories.json` 10,557,185 B + `manifest.json`, catalogue sha `99a42cc3…c46675`,
  `Result.txt` sha `3c052997…1b3e`, `fleet.json`); `node scripts/import-gtoc12.mjs ... --output data/gtoc12`:
  23 ships, 196 asteroids, 11,681 exact replay samples, 14,044.8 kg, fleet sha `b9b3b6ba…aa62`, Kepler max
  3.59e-6 km; `node scripts/check.mjs` rc 0 (palette 40 colours; validated `fleet_master_v10`);
  `Invoke-WebRequest http://127.0.0.1:4173/data/gtoc12/fleet.json` -> 200, 2,162,208 bytes, 23 ships,
  run_id `fleet_master_v10`; `browser-check.cjs` (Playwright from `reducers-vc-stuff\website-v2`, Chromium
  151.0.7922.34, WebGL2) rc 0, 0 errors, 23 distinct colours, dense layout + no-overflow at 1440x900 and
  1920x1080, followed ship 23; artefacts in `C:\Users\Angus\AppData\Local\Temp\viewer23\`, opening view
  `C:\Users\Angus\AppData\Local\Temp\viewer23\gtoc12-3d-oblique-fleet.png`.
- Off main: `feat/gtoc12-asteroid-mining` beyond dfdeca8f (v11 worker, uncommitted at bundle time); H100 G4
  claim core (running); sm_90 confirmation of the v2 fixes; `perf/g4-batched-campaign`; raw Lambda artefacts
  per INDEX.json.

### 2026-09-05 — performance optimisation and fresh H100 review

Merged architecture review ad2ac42 locally with 01e3360. Implemented read-only cached Lambert scan functions and preserved broadcast dimensions: 1.9–2.2x faster 64–1024 transfer batches in seven-repeat runs, all outputs bitwise identical. Fixed H1 exported counters/unknown measurements/single-sample aggregation; updated native emitter and created immutable corrected derivatives for 42 H100 records. Added numerical regression tests and baseline-ref benchmark. Native telemetry TU compiled, no GPU jobs started. H100 read-only snapshot at 12:46:57Z showed 108/396 complete; last three adaptive low-thrust N=500 groups all timed out. See docs/PERFORMANCE_PROGRESS_2026-09-05.md and artifacts/performance/. Existing remote campaigns and pre-existing Windows scratchpad content preserved; no push.
Validation complete: all 156 GTOC12/H1 tests passed with pinned data and official verifier (337.27s); bit-pattern checks passed. Final 256-grid wrapper benchmarks after tests: 1.21x/1.62x/1.78x at 64/256/1024 transfers. No claimed complete-leg speedup. Report updated with both scan resolutions and remaining GPU implementation gates.

### 2026-09-05 — first GPU-native implementation checkpoint
User explicitly authorized local GPU hill climbing and created an active GPU-native C++/CUDA goal. Implemented cooperative scaling/PDHG and parallel coefficient assembly/scans, retained buffers, honest phase timing, and a bitwise-preserving fingerprint reduction. The signed-zero row-maximum regression was found by displaced HCW acceptance and fixed with a permanent test. Full four-family physics qualification, native contracts, and CUDA memory/synchronization/race checks pass. Corrected 2k zero-HCW benchmark: full command 5.164s -> 0.461s, SCvx 4.774s -> 67.204ms, CQP 4.806s -> 1.836ms; do not generalize these setup-heavy ratios to hard solves. Larger-grid sweep and fingerprint microbenchmark recorded. Goal remains active with recovery, serial outer work, CPU numerical paths, and batching outstanding. See docs/GPU_NATIVE_OPTIMIZATION_PROGRESS.md. Remote workloads untouched; no push.

2026-09-06: Committed f876abb (parallel SCvx metrics and exact HCW warp replay) and 67b371d (cooperative standalone CQP residuals plus honest residual timing). Local RTX5090 final matched complete HCW command 5.172s -> 0.424s, 12.21x; SCvx 4.747s -> 5.08ms on already-feasible 2000-interval fixture. All four physics families pass unchanged qualification; new CUDA tests clean under memcheck/synccheck/racecheck. Goal remains active for recovery/convergence and full GPU-native mission pipeline. See docs/GPU_NATIVE_OPTIMIZATION_PROGRESS.md.

2026-09-06: c2028b4 fixes canonical objective and actual recovery work counts; d36c5c6 adds bounded early GPU KKT refinement and cached phase profiles. Matched local PD3 N2 SCvx17.776->5.246s (3.39x), recovery13.041s->43ms, unchanged objective/inner accuracy. Matched PD6 remains~48.48s with unresolved inner convergence; optimized N20PD3 did not return before120s, no larger-case claim. Native contracts, 3 CUDA sanitizer tools on early recovery and four-family physics/fingerprints pass. Goal active for larger-scale convergence and whole GPU pipeline. No local GPU jobs active, remote work untouched, no push. See docs/GPU_NATIVE_OPTIMIZATION_PROGRESS.md.

2026-09-06: 571d924 parallelizes block-local cone projections and recovery CGLS norms, fixes stale cancelled PDHG reports, and moves medium (>=256 variable) solves to measured cooperative dispatch. Matched PD3N2 SCvx5.116->4.021s (1.27x), displacedHCW50 dispatch-only64.503->38.969ms (1.66x), same882inner/6accepted. FullPD3N20 now returns51.35s but still missesinner1e-8; no convergenceclaim. All3CUDA sanitizers andfour-family physics/fingerprints pass. Goal active forconvergence andfullGPU-nativepipeline. Snapshotbuild/571d924, GPUlockfree; remotejobs untouched; nopush. See docs/GPU_NATIVE_OPTIMIZATION_PROGRESS.md.

2026-09-06: 0b3c130 optionalGPUQOCOhandle reuse+orderedgathers+correctstoppingscale+checkedcuDSSABI. MatchedP1C20same54inner/2accepted:SCvx1.474888->.623454s2.37x,process1.85x;allquality1e-8unchanged. Operatorall4sanitizers/no leaks, native3pass; fullcuDSS0.7deterministicracecheckreportsvendorhazards,0.8upgradehitsinvalidsharedreadandisrejected. Finalisolatedcandidatev9cuDSS0.7,NOTautopromoted. Goalactive/incomplete; docs/GPU_QOCO_LOCAL_BACKEND.md and artifacts/performance/qoco-gpu-checkpoint.json. Preserveunrelatedworkandremotecampaigns. No push.

- 2026-09-06 ea25de0: moved native QOCO KKT audit and dual mapping to retained CUDA gathers/reductions, optional resident-device solution export and legacy compatibility; independent long-double and real landing CPU-oracle checks pass, all4 new-kernel sanitizers and full landing memory/sync checks pass. Full CUDA build repaired. Matched small landing shows no new speedup (660.848->668.247ms, same54inner/2accepted/1e-8physics); recorded honestly. Corrected native memory counter scope, documented remaining CPU paths and open vendor race finding. Goal remains active, fullGPUplanner incomplete. Evidence qoco-gpu-audit-{pd3,checkpoint}.json.

- 2026-09-06 c890e2e: queued default-stream QOCO operators, retained GPU cone scratch and GPU final line-search reductions; corrected lost residual blocks beyond1024 and SOC linear/boundary step bugs with independent regressions. RTX5090 matched landing2.55x(54->28inner), real40interval3DOFplanner1.50x,20interval6DOF1.21x; allsamephysicsgates/replay/objectiveaccuracy. Newkernelsall4sanitizersclean, fullmemory/init/syncpass; fullracecheckstill30cuDSSfactorizationhazards. Nondeterministicalternativev15failedqualificationandrejected. Finaloptionalbackendv14, coreea25de0unchanged, noautopromotion. Newbenchmark_planner.py+fullrepresentativeresults/evidencecommitted; goalactiveCPUhotpaths/largecases/GTOC12/batching/vendorissueremain.

- 2026-09-06 d35fe7e: native retained sparse topology with exact GPU comparison and batched numeric downloads; opt-in QOCO values-only updates preserve GPU indices/gather maps. All4 new-kernel/operator sanitizers and full landing memory/init/sync plus7CPU-oracle repeats pass unchanged physics. Paired landing/6DOF runtime essentially flat, no added speedup claimed; reduced repeated D2H with initial cache upload/memory cost. Nondeterministic tighter-tolerance v17 passed10 then failed11th and was rejected, experimental flag removed. Evidence qoco-cached-topology-checkpoint.json and paired full results committed; frozen core d35fe7e/backendv16. FullGPUgoal active: CPU numerical conversion/scaling/KKT/outer work and known cuDSS race remain. No push/remote changes.

- 2026-09-06 4556941: compiled repeated QOCO numerical conversion to CUDA (ordered sparse/constraint/SOC+RSOC maps, finite/bound/symmetry checks), resident output feeds GPU KKT audit D2D. Mixed-cone/duplicate-entry mutation contracts and independent CPU conversion+audit oracles pass; newkernelsall4sanitizers, full landing/synthetic memory-init-syncpass. Paired landing258.174->249.038ms,PD61657.412->1635.301ms withsamephysics/iterations, modest gains; initialmapcompilation/transfer/memorycosts documented. Fullgoalstillactive: initialCPUstructure, QOCOhostupdates/equilibration/KKT/scalardecisions/outer/warmstate/vendorissueremain. Evidence qoco-device-conversion-checkpoint.json; snapshot4556941,backendv16unchanged,remotejobsuntouched,nopush.


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

## 2026-09-06 — multi-block gather and Lambda viewer publication
All three SCvx gathers distribute independent writes across blocks, with 48 bitwise fixture comparisons, scoped CUDA sanitizer checks and same-binary trajectory ablation. Kernel cost improves substantially at large sizes; full-solve timing remains mixed and an earlier numerical failure is retained, not declared fixed. See scvx-gather-v82-checkpoint and GPU_QOCO_LOCAL_BACKEND for scopes. Lambda v11 fleet and completed G4 snapshot downloaded and checksummed; existing viewer rendered 23 ships and 14047.80 kg locally. User requested merging all changes/results into main and pushing after display. Main publication includes earlier Lambert and H1 changes plus tests, without interrupting remote work. Full GPU-native objective remains incomplete.

Publication validation on September 6: 199 physics/telemetry/planner-schema tests pass (352.83 s), seven viewer importer/server tests pass, imported fleet validation passes, Ruff and staged diff checks pass. Download and indexed Git objects preserve all 1,028 source checksums plus the viewer dataset. Raw evidence is preserved without line-ending conversion.

## GTOC12 native interval dynamics, runtime v85
- Goal ACTIVE/incomplete. Native FP64 CUDA interval RK4/variational path supports both holds and preserves GTOC12 norm-of-interpolated-thrust mass flow plus Gamma sensitivity surrogate. Explicit Python/CLI selection; NumPy remains default. No silent fallback. CUDA CLI requires one worker.
- Device API retains buffers, accepts external stream/device inputs, graph-captures without allocation or host synchronization. Python bridge still transfers coefficients to CPU sparse assembly/Clarabel. Independent verification and CPU outer decisions remain.
- Matched RTX 5090 representative full-leg median 137.914 -> 118.281 ms (1.166x, 14.2% less time), all 22 samples independently qualified at fixed mass error <=1e-5 kg. N2001 linearisation 48.46x ZOH / 76.90x Lagrange including bridge transfers. Not fleet throughput evidence.
- Five additional pairs qualify. Both solvers report infeasible and fail qualification on the free-endpoint-velocity pair; physical infeasibility is not established. Failure retained, unresolved.
- Final regression 247 passed without skips; native graph API all four CUDA sanitizer tools pass; host complete-leg/reuse memory/init/sync pass; lifecycle memcheck zero errors/leaks. No full-leg racecheck or mid-kernel cancellation claim. Final 11 CLI tests and Ruff pass after test-only syntax cleanup.
- Frozen core /home/angus/build-spacepdhcg-gtoc12-v85/final/libspacepdhcg_cuda.so SHA256 007a7eca4a25fc55b735aab94937ed52b79e32bfc0009c40d9875681cbd8e04f. Native test SHA256 c93cf2369158eb1ab35d916c371db5799f0d1cff249c8d21c64fafde320df189. Next fresh runtime86+, never overwrite85. All owned build/test/benchmark jobs terminal.
- Evidence artifacts/performance/gtoc12-discretisation-v85-checkpoint.json embeds source hashes (raw and normalized LF), frozen runtime hashes, reproduction helpers, fixtures, build/profile logs and evidence hashes. docs/GTOC12_GPU_REFINEMENT.md gives usage and scopes.
- Profile points to CPU sparse assembly and Clarabel as next costs: fixed native conic layout must include the union of structurally possible Psi entries before GPU assembly/direct GPU solver integration. Existing conic tolerance1e-9 and physics gates stay fixed. True GPU batching remains pending.
- No Lambda access in this tranche; existing campaign untouched. Prior downloaded fleet remains served at http://127.0.0.1:4178/?dataset=gtoc12&epoch=69807&preset=oblique (HTTP200 checked). User authorized publication of all changes/results to main; this tranche is ready to commit/merge/push. Check actual Git refs for publication completion.

## GTOC12 native conic assembly, runtime v87
- Goal ACTIVE/incomplete. Previous turn published v85 CUDA interval dynamics at405ad6fb. This turn adds a fixed C++ conic topology and multi-block FP64 numerical assembly directly consuming device interval outputs. No coefficient round-trip between these kernels. Device outputs persist for the upcoming GPU conic solver connection. Python bridge still downloads A/b/q/P for CPU Clarabel; seed, outer decisions and independent verification remain CPU.
- Explicit --assembly-backend cuda requires --discretisation-backend cuda and workers1; NumPy assembly remains default. Both holds, all endpoint choices, constraints/cones/fuel/smoothness terms are represented. No physics/conic tolerance changes or clipping.
- v86 full mass-row pattern: matrix parity passed, but one complete ZOH leg failed thrust qualification at0.6000002424951428N after28iterations. Rejected and recorded. Zero-removal and structural-only probes isolate sparse representation sensitivity. v87 omits only identically zero mass-row Phi/Psi terms; all potentially nonzero entries are retained with fixed topology. No general convergence reliability fix claimed.
- v87 focused78 tests pass; full combined284 tests pass without skips in377.33s. Native48 changing-input graph replays pass all four sanitizer tools. Host5 complete-leg/recovery/exception tests pass memcheck/leak,init,sync; existing dynamics lifecycle memcheck3 tests passes. No full-leg racecheck or new mid-kernel cancellation. Ruff/diff checks pass.
- Matched2warmup+9 measured rotating3backend sample sets: coefficient N2001 native1.615msZOH/2.572msLagrange versus GPU dynamics+CPU assembly108.062/164.920ms (66.92x/64.12x faster phases). Representative full-leg medians140.045msCPU,123.214msGPU dynamics,151.595msGPU dynamics+assembly:23.0% full-leg regression versus prior path. All33 qualify with final mass<=1e-5kg from fixed2445.3111007852112kg. Do not promote as generally faster full-leg default. Profile130/146ms inClarabel wrapper+solve; native pattern7outeriters vsprior5.
- Five additional grid/hold pairs qualify with fixed mass agreement. Free-v-infinity case remains solver-reported infeasible and unqualified on both CPU/GPU; physical infeasibility unproven, failure retained.
- Frozen runtime /home/angus/build-spacepdhcg-gtoc12-v87/final/libspacepdhcg_cuda.so SHA256 e0ef981d1aa2ac8214223f707a7a69e9ff977f8754dcb0dfeb3f4c0146c8dabc. Native conic test SHA256016095fc844f0cd65bb5f3b92d586dd1c06f3a102c030917c5ccb6885d50029e. Do not overwrite86/87; next88+. All owned build/test/benchmark sessions terminal. No Lambda access; leave its campaign untouched.
- Evidence artifacts/performance/gtoc12-conic-v87-checkpoint.json embeds raw+normalizedLF source hashes, runtime hashes, all reproduction helpers and measured script, fixture sources, logs and raw report hashes. Benchmark script later only explicit-bound closure variables and wrapped a string for lint; exact measured version retained. Numerical Python source hashes match benchmark. docs/GTOC12_GPU_ASSEMBLY.md records use and limitations.
- Next: direct canonical GPU conversion (split combined A into scalar rows and F=-A_soc, offset=b_soc; equality lower=upper=b, inequality lower=-inf upper=b; full symmetric Q from upper P; free variable bounds), then existing native_qoco adapter. Its create currently hardcodes1e-8 tolerance, while GTOC12 requires1e-9. Expose requested tolerance and gate audited residuals explicitly: adapter success only checks residual finiteness. Preserve objective/physics gates. CPU outer decisions and batching still pending.
- Branch perf/gtoc12-native-assembly from405ad6fb. Publishing validated explicit path and all evidence under user's main/push authorization; inspect actual Git refs for completion.

## Experimental GTOC12 GPU QOCO connection, runtime v93
- Goal ACTIVE/incomplete. Added direct native CUDA interval/conic/canonical conversion/GPU QOCO connection and retained device primal. Python/CLI explicit opt-in requires both CUDA backends and patched device extensions; no CPU conic fallback. Defaults and legacy adapter tolerance/API unchanged. Host initial topology/conversion, QOCO control/scalars, SCvx outer loop/seed/independent verification remain CPU.
- New wrapper compiles CSC conversion once, mirrors Q, permutes SOC [t,vector] to canonical [vector,t], and removes exactly redundant final ZOH Gamma bounds/thrust cone (final controls already equality-fixed). Reference-dependent trust rows retained; original constraints checked independently.
- External tolerance1e-9 unchanged; internal IPM drives100x tighter. GPU audit requires rawstatus1/2, finite relative primal/dual residuals and independently reduced global objective gap<=requested tolerance. Global objective from unscaled original matrices and mapped primal/dual; host output only on qualified result. Reports include objective/gap and true inner iterations. Adapter timings/counters are cumulative snapshots, not per-attempt values; do not sum snapshots. Invalid audit fields are JSON null.
- Rejected experiments preserved: v88 residual-only gate could accept objective errors; v89 gap gate revealed poor convergence; v90 tighter iterative refinement all6fulllegs fail and reverted; v91 reduced ZOH constraint formulation one of6transfer cases qualifies, coast failure once/pass rerun retained; v92 reduced formulation plus tighter internal tolerance one of6qualifies, other5failphysics. No general reliability claim. v93 same normal numerics as92, plus invalid residual reset and tiny-positive tolerance underflow guard.
- Final matched150dayZOH:18/18fullleg attempts converge, pass independentphysics and fixedmass2445.3111007852112kg within1e-5kg. One warmup+5 measured per backend rotatingorder. Medians NumPy147.461ms, CUDAassembly+Clarabel145.656ms, GPUQOCO818.922ms:5.62x slower. GPUQOCO measuredmassmaxerror2.12e-9kg, outer8/9/9/13/6;11of45subproblemsrejected. Keep experimental, not default promotion. Final cumulative QOCO solve-region3.764s of4.047s wall over5measuredlegs; setup0.171s/update0.016s/audit0.007s. Conic convergence/iteration cost is next bottleneck.
- Full combined regression297passed without skips in377.89s; four Python lifecycle memcheck tests pass with0leaks/errors. Final native/host sanitizer results and runtime/source/evidence hashes are in artifacts/performance/gtoc12-qoco-v93-checkpoint.json. v92 native all4tools and legacyconversion oracle pass; final93 results recorded separately. No full-leg racecheck/new mid-solve cancellation claim.
- Runtime /home/angus/build-spacepdhcg-gtoc12-v93/final/libspacepdhcg_cuda.so; unchanged QOCO78 /home/angus/build-qoco-gpu-gpu-kkt-v78/final/libqoco.so. CUDA12.8/cuDSS0.8, RTX5090 driver595.97. All versions88-93 frozen; next94+. Exact intermediate conversion snapshots and reproduction helpers embedded in final checkpoint. No source edits after final tests other than documentation/memory.
- No Lambda access/jobs in this tranche. Downloaded23shipfleet viewer HTTP200 and scripts/check.mjs pass; dataset hash0ebd0dfaf0b483418e8ebd261eac2671528cac283429216777ecda79a55deb93. Existing viewer remains on4178. Browser panel open request queued (not evidence of visible tab). User's publication authorization persists; branch perf/gtoc12-gpu-qoco from43b69c9c, inspect actual refs for commit/merge/push completion.
- Final v93 native all-four sanitizer tools and legacy conversion oracle pass; four Python bridge lifecycle tests also pass memory/init/sync. All owned GPU jobs are terminal. Core93 SHA256950f12cbdd950de526aa83a3e581f49ccc6b7d6a28fcc0f624c338bb78de937f; native test SHA256d6aa5858af040efa674e16ee01304603a28b9f87a6b21bd18f548ae870f1245b. Checkpoint embeds28 evidence records plus helpers and88-93runtime hashes. Ruff passes. Publication now proceeds under explicit user authorization.

## Fused GPU KKT experiment v94
- Previous goal turn was progress: experimental GTOC12 GPU solver published on main dce23ed8, with measured full-solve regression. Goal stays ACTIVE/incomplete. This turn profiles its QOCO solve region and implements opt-in --fused-kkt-product in prepare_qoco_gpu.py, requiring gather/queued/deferred-transpose flags. One multiblock kernel replaces sparse P*x+A^T*y+G^T*z/A*x/G*x composition, with original within-row accumulation order and __dadd_rn joins. NT products, refinement, regularization/tolerances unchanged; not enabled in default builders.
- 72 operator fixtures compare bitwise to old GPU composition plus independent dense arithmetic; n3/257/4097, missingP/emptyorabsentA/G/duplicates/changinginputs. All4sanitizers pass zeroerrors/leaks/hazards/warnings. Operator CUDAevent1000sample composed/fused74.245/10.291us(n3),75.492/9.576us(n257),71.784/42.256us(n4097). Includeshostenqueuepacing, singlephasesamplesnotthroughputclaim.
- Initial18GTOC12fulllegsallqualify, but fusedQOCOmedian870ms vsprior819ms; fivefusedmeasured1194inner vs845baseline, outer11/11/25/10/9. Follow-up PREPLANNED10alternatingpair/freshprocesses, each1warmup+1measured:all40legsqualify fixedmass2445.3111007852112kg +/-1e-5 andindependentphysics; medians959.555msbaseline/805.680msfused (16.0%less). MixedsequencesmeanNOgeneralspeedpromotion.
- FocusedGTOC12/CLI23pass1FAIL: fixedbothZOHcoast originalequality1.166299e-9>1e-9 despite normalizedauditpass. Not rerun untilpassing orloosened. Earlier93/91showrelatednumericalsensitivity. Fusedoperatorbitwiseparitydoesnotprovefullsolverreliability. Keepoptinexperiment, notgeneralpromotion.
- Legacyconversionoracle,7nativelandingrepetitions, N20/N500sixDOFcertificatespass. Fixedobjective comparisonwithv76baseline sample0:4.66e-15/4.487e-10differences<=1e-8. SourcecommitinsidefrozenplannerresultidentifiesitsolderCLI; checkpoint separatelyhashesactualplanner/core93/QOCO94runtimechain. Nofullsolverrace/cancellationcoverageclaim.
- CPUtooling/backend24tests pass after normalizing two preexisting CRLF shellfiles alreadydeclared eol=lf; initial2failuresretained, shellsemanticsunchanged. CMakeconfigurepassesandexcludesstandaloneQOCOtestfromnormalnativeGLOB. Ruff/diffchecks pass. Freshpreparedsource matchesfrozen94normalized; relative78differencesonlynewheader,kktdispatch,cuda_linalginclude,provenance.
- CUDAAPItrace qualifiedbaselineleg:4175cudaMemcpy calls570ms,48423kernel-launchAPIcalls251ms; includescoldinit. OriginalGPUstats failedSQLiteagecheck; directSQLite reports SKIPPED,noGPUkernel/memorydata. DoNOTdescribeasGPUtimeprofileorallD2Hcopies. Tracepaths/hashesanderrorsretained.
- FrozenQOCO94 /home/angus/build-qoco-gpu-fused-kkt-v94/final/libqoco.so SHAab19af76fcc7ba9b9bf5d563ea8f56450e5b7db516bf97d7426d0ba269a98349, core93 unchangedSHA950f12cbdd950de526aa83a3e581f49ccc6b7d6a28fcc0f624c338bb78de937f, baselineQOCO78 unchanged. Next95+, neveroverwrite94. Operatorandbenchmarkexecutablesalsofrozen. Allownedjobs terminal, noLambdaaccess, viewer4178intentionallyleftup.
- docs/QOCO_FUSED_KKT_EXPERIMENT.md andartifacts/performance/qoco-fused-kkt-v94-checkpoint.json hold17evidencerecords,helpers,source/runtimehashes,reproduction,failedtest. Nextsubstantialtarget: moveIRnorm/stop/bestrestore controltoGPU. NVIDIAcuDSSdocs sayfactor/solvegraphcaptureworkswithcompatibleallocator; CUDA12.8conditionalbodiesexcludeallocnodes. Needprovepersistentmemory/capturebeforeconditionalIRloop. Existingstreamglobalops/hostcallbacksneedexplicitstreampropagation; currentfusedkerneldefaultstreamnotgraphcapture-ready. Source: https://docs.nvidia.com/cuda/cudss/general.html andCUDA12.8conditionalgraphguide. FullGPUouter/setup/batching/convergencegoalincomplete. Preserveexperimentalbranch; no default promotion fromthisevidence.


## Conditional cuDSS / reset checkpoint, 2026-09-06
Goal ACTIVE/incomplete. Current experiment diagnostic-only, no production IR integration or speed claim. QOCO95 late-stream capture misses solve kernels;96 early stream captures but conditional host-copy restriction;97bitwise/98scaled GTOC12 comparisons fail.99 GPU snapshots of2eight-byte range inputs permit conditional WHILE controls:120checks over15real systems(allpass), landing70/70strict1e-12scaled replayparity;GTOC12 all35fail,also10/10ordinaryrepeatcomparisonsfail. All3 original-algorithm fullruns qualify;N20/N500fixedobjectiveerrors2.85e-10/9.40e-10<=1e-8. Snapshot/lifetimeassumption privatevendor range; no productionpromotion.
NativeN20memcheckfailsCUDA999(239outstandingallocsafterabort),racecheckexit11,synccheckCUDA999;initcheckpasses. Minimalvendor-free100/101 reproducememfailurewith/withoutcopyornestedchild.102presyncpartialinterruptedbycomputerreset, executableabsentafterreset(donotclaimhash). Fresh103normalboth+6race/init/syncpass, bothmemchecks999persist. Frozen95-99/100/101/103 retained,next104+. AllownedGPUprocessesterminal. Fullsources/logs/helpers/artifactsin qoco-conditional-solve-v99-checkpoint.json and docs/QOCO_CONDITIONAL_SOLVE_EXPERIMENT.md. NeedactualGPUresidual/backup/stop/refinementintegration, factorbuffergraphlifetimeproof, cancellation and accuracy/convergence/perf; fullGPUgoalremains.
UserrequestedLambdaaftercomputerreset: temporary/tmp/traj-key.pemmissing;restoredowner-onlycopyfromoriginalwithoutprintingsecret. FreshreadonlySSH08:56UTC H100100%,1607MiB,44C;worker53138/server53183/session2619902live.159groupscompleted+1running(ordinal159),latestgroupall9timeouts. Source1dbcae0unchanged;GTOC12v11stillpassesbothverifiers23ships194collected14047.802874743327kg. NoLambdaGPUoffloadwhilebusy,nojob/sourcechanges. Freshstatus+summaryartifactsafter-reset retained.

Viewer after reset: restarted node serve.mjs --port4178 in results/lambda/2026-09-06/visualiser; execsession7869 intentionallyleftlive. HTTP200+scripts/check.mjs pass, fleetSHA0ebd0dfa unchanged. Browseropen queued, nofreshvisual-renderclaim. CheckactualGitref for diagnosticcheckpointcommit; no productionpromotion.

## 2026-09-06 — README and live leaderboard

Refreshed README with current CUDA implementation boundaries, an explanation of PDHCG and the distinct conic inner method, credited Lhongpei/PDHCG and its QP/conic papers, and compared the verified 14047.8 kg fleet with the official published leaderboard: hypothetical eighth, 666.333 kg below seventh, 62.3% of JPL. All 23 local README links resolve and decimal score arithmetic is checked. No new fleet score or full-GPU completion claimed. See scratchpad for terminal v109 benchmark/sanitizer state; ongoing solver edits are separate and uncommitted.

## 2026-09-06 — QOCO device stopping and NaN recovery

Implemented optional retained GPU stopping/best-vector/scalar control (108/109) and GPU NaN-direction recovery (110). Same tolerances, objective gates, and original NaN-only branch semantics. All direct scan/guard sanitizers pass; 51 integration tests pass normally and with audits; seven convergence/failure cases and both native PD6 objective fixtures pass. Full native host-IR ablations pass mem/init/sync; conditional IR still has unresolved CUDA999 sanitizer failure. Balanced110 medians105589.932/109555.361/110601.990ms: no speed claim. Checkpoints preserve all sources, runtimes, helpers, failures, and measurements. Lambda remains busy at100%; no offload. CI linkage/concurrency repairs published86b0f93b and local48native normal/sanitized checks pass; subsequent Windows missing<string> include fixed535630e3. Full GPU-native goal remains active: next host IR-accounting synchronization and IPM/cuDSS dispatch/setup.

## 2026-09-06 — numerical factorisation graph

Optional115 factorisation graph replaces repeated host cuDSS factor calls after one warm-up/capture using retained stream-ordered allocations. All51normal/audited integration tests,7convergence/failuretests,PD6fixedobjectives,and22hostIRmem/init/synctests pass. Balanced110/pool115/graph115 medians535.348/516.308/530.927ms;no graphspeedclaim. Diagnostic111–113 failures/parity differences and unrun114 stream-ownership correction retained in115checkpoint. GTOC12diagnosticstrictlinearparityfails60/60 despitequalifiedcompletelegs;production115factor-only gatespass. GoalstillneedsIRgraph/accountingandIPM/SCvxdispatch/setup/replay. WindowsC4996fixedwithboundedcopy,ABIchecksPASS,maina4f5cb58wheelallOSPASS.
