# DEVLOG

Rolled over on 2026-09-05: the detailed 2026-09-01..2026-09-05 history moved to
`DEVLOG_2026-09-01_to_2026-09-05.md` (both branches' entries, ours first, then the joint-itinerary
branch's merged from `main`). Carried forward: GTOC12 best verified fleet and the open levers
(see the scratchpad's Active Risks).

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
