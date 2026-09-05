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
