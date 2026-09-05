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
