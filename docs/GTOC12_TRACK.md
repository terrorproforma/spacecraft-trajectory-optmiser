# GTOC12 "Sustainable Asteroid Mining" replay track

Layer E / P2-F of the comparative campaign (`docs/COMPARATIVE_SOLVER_CAMPAIGN.md`). This track
replays the 12th Global Trajectory Optimisation Competition (Tsinghua University, June–July 2023)
with the OrbitWeaver route-and-trajectory stack. Everything here is CPU-only: the single RTX 5090
was owned by the G4 measured campaign for the whole session and was never touched.

Branch `feat/gtoc12-asteroid-mining` (worktree `/home/angus/worktrees/spacepdhcg-gtoc12`, base
`96781349` of `integration/single-gpu-v1`).

## 1. Sources and pins

All official material is pinned by URL, byte size and SHA-256 in
[`benchmarks/gtoc12/pins.json`](../benchmarks/gtoc12/pins.json) and fetched by
`spacepdhcg gtoc12 fetch` (or `python scripts/gtoc12/fetch_gtoc12_data.py`, a thin wrapper around
`spacepdhcg.gtoc12.fetch`) into the data directory: `$SPACEPDHCG_GTOC12_DATA`, else the ignored
`benchmarks/gtoc12/data/` of a source checkout (or of `$SPACEPDHCG_BENCHMARKS_DIR`), else
`<cache>/gtoc12` below `$SPACEPDHCG_CACHE_DIR` / `~/.cache/spacepdhcg` for an installed wheel. No
multi-megabyte dataset is committed or packaged; the small `pins.json`, `gtoc12_rules.json`,
`reduced_instance_v1.json` and `reference_reproductions.json` are mirrored into the wheel by
`spacepdhcg.resources` so every `spacepdhcg gtoc12` command runs from an installed package.

| File | Role | SHA-256 (prefix) | Source |
| --- | --- | --- | --- |
| `GTOC12_Problem.pdf` | problem statement (19 June 2023) | `fcdc2dad…` | ESA portal + Tsinghua API |
| `GTOC12_Submission_Format.pdf` | solution file format | `cb6ebcda…` | ESA portal + Tsinghua API |
| `GTOC12_Asteroids_Data.txt` | 60,000 asteroid elements at 64328 MJD | `99a42cc3…` | ESA portal + Tsinghua API |
| `bonus_coefficients.txt` | frozen end-of-competition bonus table | `e8a3795e…` | Tsinghua API only |
| `GTOC12_Verification_Program.zip` | official verifier (Linux/macOS/Windows) + example `Result.txt` | `50657b23…` | ESA portal + Tsinghua API |
| `GTOC12_JPL_merged_solution_36sc.txt` | JPL post-competition 36-ship solution | `7ab630de…` | Tsinghua API only |
| `39_mass_optimal.txt` | TheAntipodes 39-ship solution | `5aff46f8…` | ESA portal + Tsinghua API |
| `37_mass_optimal_self_cleaning.txt` | TheAntipodes 37-ship self-cleaning solution | `c7dbb8a6…` | ESA portal + Tsinghua API |
| `ConferenceHandbook_GTOC12_Workshop.pdf` | workshop handbook (optional) | `dc7db8e8…` | Tsinghua API |

Every Tsinghua copy was byte-identical to the ESA copy and to the literature worker's cache
(`/home/angus/worktrees/spacepdhcg-literature-cache/raw/gtoc12/`; that branch had no committed pins
or fetch scripts to reuse at the time). The Tsinghua site is a JavaScript SPA; the files are served
by `/prod-api/common/downloadProblemFile?fileName=<name>` (`/prod-api/common/getCompetitionFileList`
lists them). The UI gates `bonus_coefficients.txt` behind a login (`/common/downloadCoef` returns
401), but the problem-file endpoint serves it unauthenticated; the pin records this.

Licence: the organisers publish these files for competition/research use without an explicit
licence text; they are pinned, not redistributed.

## 2. Rules encoding

`src/spacepdhcg/gtoc12/constants.py` transcribes every constant and rule; `rules_payload()` is
mirrored by [`benchmarks/gtoc12/gtoc12_rules.json`](../benchmarks/gtoc12/gtoc12_rules.json) and a
test asserts equality. Key items: μ = 1.32712440018e11 km³/s², AU = 1.49597870691e8 km, window
64328–69807 MJD, Isp = 4000 s, T_max = 0.6 N, m_dry = 500 kg, m_0 ≤ 3000 kg, 40 kg miners (≤ 20),
k = 10 kg/yr, ≥ 1 yr stay, v∞ ≤ 6 km/s at Earth launch/unload, r ≥ 0.3 AU, tolerances 1000 km /
1 m/s / 0.001 kg, fleet rule N ≤ min(100, 2 exp(0.004 M̄)), Venus/Earth/Mars elements and
minimum pericentre radii.

Bonus: B_i = (1 + 2(1 + 0.05 M̃_i)^−0.1)/3. The archived table's first row (B = 0.859074317330498
at M̃ = 194.9805 kg) reproduces the formula to 1e-15. **Dynamic** competition scores used the live
table at submission time; the **fixed post-competition** score uses the archived table. The verifier
reports both the unweighted total (what the offline verifier prints) and the fixed-bonus score, and
can recompute a dynamic score from a supplied already-mined table.

## 3. Verifier

The official Linux binary is wrapped (`spacepdhcg.gtoc12.official`, scratch-directory execution,
stdout and `ScoreData.txt` parsed). Its unstripped symbols show RKF78 integration
(`RK::RKF78`) and Lagrange thrust interpolation (`LagInterp`); its diagnostic strings give the
complete rule catalogue (`Error001`–`Error901`, `ErrorA00`–`ErrorA23`), which the independent
verifier (`spacepdhcg.gtoc12.verifier`) encodes one-for-one: event pairing, epoch ordering and
window, launch/rendezvous/flyby state matching, v∞ bounds, GA turn-angle bound, miner/collect mass
jumps, one-year stay, ≤ 2 visits, unload-at-Earth accounting, fleet rule, thrust bound, sample
spacing, and propagation between events (exact Kepler coasts, DOP853 burns with cubic-Lagrange
thrust interpolation of the daily samples).

Acceptance (official binary vs independent verifier):

| Solution | Official ships / asteroids / mass | Independent | Per-asteroid max diff | Fixed-bonus score |
| --- | --- | --- | --- | --- |
| Organisers' example `Result.txt` | 1 / 0 / 0 kg | 1 / 0 / 0 kg | — | 0 |
| Antipodes `39_mass_optimal.txt` | 39 / 356 / 28975.1 kg | 39 / 356 / 28975.140269 kg | 0.0 kg | 24474.15 kg (published 24474.16) |
| Antipodes `37_mass_optimal_self_cleaning.txt` | 37 / 338 / 27045.3 kg | 37 / 338 / 27045.268330 kg | 0.0 kg | 23219.98 kg (published 22847.92 used the live table) |
| JPL `GTOC12_JPL_merged_solution_36sc.txt` | 36 / 320 / 26062.6 kg | 36 / 320 / 26062.646065 kg | 0.0 kg | (published 21904.51 with end-of-competition table) |

Independent propagation errors on the reference files reach 873 km / 0.11 m/s (their generators'
own models), all inside tolerance. CLI: `spacepdhcg gtoc12 verify <solution> [--official]`; Python:
`spacepdhcg.gtoc12.verify_solution_file`.

## 4. Physics and arc model

- Ephemerides: the official element→state formulas (`ephemeris.py`); Earth at 64328 MJD matches
  the organisers' example launch line to 3e-8 km; universal-variable Kepler propagation (bracketed
  Newton) closes to millimetres over 1000 days.
- Lambert: `lambert.py` is a vectorised NumPy port of the repository kernel
  (`cpp/include/spacepdhcg/orbitweaver/lambert.hpp`); `NativeLambert` compiles `cpp/src/c_api.cpp`
  and calls `spacepdhcg_lambert_zero_revolution` / `spacepdhcg_lambert_family_batch_cpu` via ctypes.
  Parity: 1e-13 km/s over 300 short- and long-way heliocentric legs (test).
- Low-thrust arcs: `low_thrust.py` solves each leg by SCvx in nondimensional units (AU, TU,
  m/m₀, T/T_max) with zero-order-hold control on 2-day nodes, batched RK4 variational
  linearisation, an L1 exact penalty on virtual control, the lossless `|T| ≤ Γ ≤ T_max` cone,
  a linearised `r ≥ 0.3 AU` half-space, box trust regions with the standard reduction-ratio rule,
  and Clarabel for the convex subproblem. Earth ends carry a free v∞ ≤ 6 km/s. The nonlinear model
  uses `|T(t)|` for the mass flow exactly as the verifier does.
- Certification: every leg is re-propagated with the verifier model (DOP853 + Lagrange) from the
  emitted samples; the pipeline's `IndependentCertifier` accepts a leg only within half the
  official tolerances. Legs in the scored runs certify to ≤ 0.46 km and ≤ 8e-5 m/s.
- Emission: each constant-thrust segment becomes its own burn arc (daily samples repeating the
  constant vector, as in the JPL file). Cubic interpolation of a constant is exact, so the official
  RKF78 and our DOP853 agree; emitting cubic-interpolated bang-bang profiles instead produced a
  3143 km official mismatch from |T(t)| kinks (retained as a lesson in `.cursor/memory`).

## 5. Reduced instance (preregistered)

[`benchmarks/gtoc12/reduced_instance_v1.json`](../benchmarks/gtoc12/reduced_instance_v1.json),
committed before any search ran. Rule SHA-256 `718dd7e76f8f09295ae53de58b56626c5d8eb42fa397a27ab190b6511b39bd25`.

- Eligibility from catalogue metadata only: i ≤ 6°, e ≤ 0.2, 2.5 AU ≤ a ≤ 3.2 AU (9803 asteroids).
- Rank key: SHA-256 of `gtoc12-reduced-v1:<id>` as a big-endian integer; keep the 1000 smallest.
- Selection SHA-256 `e2bbbca1ca31afdcb8272fbecb54c932884b343f394a39ec91e5cbc5da5d7781`
  (first IDs 23987, 50057, 26689, 55077, 57073, …).
- One ship; full official window; official dynamics, rules, verifier and scoring unchanged.

## 6. Route search and pipeline

### 6.1 What the archived references do (`references.py`)

`spacepdhcg.gtoc12.references.decode_file` decodes any solution file into per-ship itineraries
(launch, deploy/collect roles from the global visit order per asteroid, per-leg TOF, propellant,
transfer angle, revolutions, cooperative vs self-cleaning collections) and fleet statistics.
Decoding the three archived files (`results/gtoc12/references/*.itinerary.json`) gives:

| Statistic (median unless noted) | Antipodes 37 (self-cleaning) | Antipodes 39 | JPL 36 |
| --- | --- | --- | --- |
| asteroids per ship | 9 (8–10) | 9 | 9 (7–11) |
| collected per ship | 726 kg (703–781) | ~743 kg | 718 kg (476–873) |
| launch v∞ / launch epochs | 6.0 km/s saturated / first 9 months | same | same |
| Earth → A₁ | 532 d, 466 kg, 0.14 rev | — | 530 d, 447 kg |
| hop TOF / propellant / transfer angle | 183 d / 78 kg / 40° | — | 184 d / 80 kg / 38° |
| hop revolutions | 0 (p95 0.004) | — | 0 |
| deploy phase / collect phase | 2120 d / 1910 d | — | 2030 d / 1980 d |
| stay per miner | 3110 d (803–4450) | — | 2860 d |
| Earth return | 486 d, 206 kg | — | 473 d, 211 kg |
| final mass | 500–501 kg (all margin spent) | — | 500 kg |
| per-ship a / i spread | 0.035 AU / 3.8° | — | 0.069 AU / 4.0° |
| cooperative collections | 0 of 338 | — | 279 of 320 |

Per hop (1882 samples, `scripts/gtoc12/proxy_validation.py`): |Δa| ≤ 0.041 AU, |Δe| ≤ 0.045,
|Δi| ≤ 3.0° and — decisive — the **target is within ±3.3° of the ship's heliocentric phase at
departure** (p95). Hops are sub-revolution drifts between co-located asteroids on nearly identical
orbits, flown at 37 % (p95 79 %) of full-thrust authority. True low-thrust ΔV is 1.16× the
zero-revolution Lambert ΔV (p90 1.34, p95 1.41; Spearman 0.90). The 338 Antipodes asteroids
occupy a ∈ [2.27, 2.85] AU, e ≤ 0.18, i ≤ 6.1°: 5,701 catalogue asteroids fall in that box.

Consequences encoded in the search: (i) candidate targets are ranked in *position space at the
departure epoch* (`positional_candidates`: (Δa, Δe-vector, relative inclination, phase) scaled by
the p95 bands) and by a Lambert-free phasing/Edelbaum ΔV (`proxies.phasing_edelbaum_proxy`), and
only that union gets Lambert evaluations; (ii) the candidate pool is filtered to orbits that stay
collectable years later (1.5× the p95 bands on Δa, eccentricity-vector and inclination-vector
differences, with a nearest-neighbour fallback for sparse pools); (iii) hop inflation 1.2 and
duty 0.8 come from the measured ratios; (iv) chains keep a collect-phase propellant reserve
(0.9× the deploy-hop propellant + 250 kg return) so the beam does not fill with uncollectable deploy
phases; (v) the full-catalogue pool is the reference a/e/i box (a ∈ [2.2, 3.0] AU, e ≤ 0.15,
i ≤ 8°: 10,612 asteroids), which bounds memory.

### 6.2 Search

`search.py`: deterministic beam search over self-cleaning routes
`Earth → A₁ → … → A_k (deploy) → camp at A_k → collection tour → Earth`. Deploy hops expand
forwards from a launch grid (Earth legs 300–900 d, hops 60–480 d, waits 0–120 d) over the
position-space candidates above; the collection tour is scheduled backwards from the window end
(hops 90–720 d, per-hop wait windows) with the order chosen greedily by proxy cost (strict reverse
as fallback, and an escalating wait penalty when the tour does not fit), the first-deployed
asteroid collected last and the camp asteroid collected first. Earth returns may arrive in the
last 600 days. Costs are Lambert rendezvous ΔV (6 km/s Earth allowance credited) inflated ×1.6
(Earth legs) and ×1.2 (hops) against a 0.8-duty thrust authority. Beams cap variants per deployed
set and per first asteroid, drop chains below the dry mass + reserve, and prune first asteroids
without a feasible return. The Earth-leg grid is screened in 1,500-asteroid blocks (memory
bounded: 0.65 GB peak at catalogue scale vs 11.2 GB before), a wall-clock budget stops expansion
while retaining completed plans, and every failed chain is kept with its reason
(`no_collect_hop`, `camp_negative`, …). Ties break on asteroid ID; no randomness.

`fleet.py`: greedy fleets — ship *k* searches with ships 1..k−1's asteroids excluded, its best
certified route is kept, the routes are assembled into one file (ship IDs 1..N) under the rule
N ≤ 2 exp(0.004 M̄), and the fleet file is verified as a whole. In `cmd_run`, a plan containing a
leg SCvx already proved infeasible is skipped ("retain failed chains").

`pipeline.py`: each planned leg becomes an `ArcRequest`; `G3TrajectoryOracleAdapter` owns one
`Gtoc12ScvxDriver` per topology group; `BoundedScheduler` orders the work; the certified legs form
a `RouteDefinition` column and pass `solve_certified_route_master`. Collected masses start at the
rule maximum and are scaled down if the final-mass rule (`m_f ≥ 500 kg + carried`) would fail.
The route is emitted as an official file, scored by both verifiers, and exported for the viewer.

CLI: `spacepdhcg gtoc12 run --run-id <id> --output <dir> [--beam-width 32 --max-deploys 10
--neighbours 64 --refine-top 3 --ships 3 --search-budget-seconds 1800 --stop-at-first-certified
--search-only --full-catalogue --pool-a-min 2.2 --pool-a-max 3.0 --pool-e-max 0.15 --pool-i-max 8
--budget-seconds 7200 --retime-attempts 4 --retime-budget-seconds 900 --no-retime
--no-bonus-weights --no-cooperative]`.

### 6.4 Joint re-timing, chain extension, clusters, cooperative collection, fleet master

`retiming.py` — after a chain certifies, its visit order is fixed and *every* epoch is re-chosen
jointly by an exact dynamic programme over a 15-day lattice (launch, each arrival, each camp,
each departure, Earth return). Stage values are the bonus-weighted mined mass (a deploy at `t`
contributes `−w k t`, a collect at `t` contributes `+w k t`, so the objective is exactly
`Σ w_i k (t_collect − t_deploy)`) minus a propellant price × rocket-equation propellant of each
leg; legs come from cached Lambert tables per body pair (departure lattice × TOF grid: hops
60–720 d, Earth legs 400–900 d) and are admissible under the role's authority ratio (hops 0.45
here — every certified hop had ≤ 0.49 and half of those at ≥ 0.49 failed SCvx). The DP runs on a
per-leg mass profile; a forward pass then rebuilds the plan with true masses, and the price is
raised geometrically until the mass budget closes (or lowered geometrically while it keeps
closing), then bisected twice towards the last failing price so the margin is *spent* rather
than left over. (`fleet6_retime_v1` ran with a price loop that stopped after one halving;
letting it keep halving lifts ship 1's proxy re-timing from 564.7 to 583.2 kg, which is what
`fleet10_master_v1` uses.) Chain extension inserts one more asteroid
(deployed last, collected first) from the position-space candidates and re-times each variant;
the SCvx-in-the-loop driver (`improve_and_certify`) re-flies the improved plan, *bans* the body
pair of any leg that fails (authority ratio ≤ 0.9 × the failing ratio in later attempts) and
*calibrates* the propellant inflation of every pair that certifies (SCvx ΔV / Lambert ΔV × 1.03),
then tries again from the certified plan. Only certified routes are returned; the previously
certified route is kept otherwise.

Effect on the 6-ship fleet (`fleet6_retime_v1`, before → after, all officially verified):
ship 1 548.3 → 592.6 kg (8 → 8 asteroids, final mass 693 → 511 kg: the margin became
faster hops); ship 2 372.4 → 425.5 (6 → 7, 1030 → 572 kg); ship 3 419.2 (unchanged: the
re-timed variant did not certify, E→16459 banned at ratio 0.49); ship 4 341.3 → 446.8 (5 → 6,
1047 → 502 kg); ship 5 333.4 → 465.3 (6 → 7, 1004 → 523 kg); ship 6 329.1 → 395.5 (5 → 6,
1270 → 616 kg). Fleet 2343.6 → 2744.9 kg (+17 %); every re-timed ship ends within 11–116 kg of
the 500 kg floor, i.e. the "unspent propellant" of §7 is gone, and the binding constraint is
again the 15-year window (extensions fail with `mass_below_dry_plus_collected` once the ship is
at the floor, or `leg_authority` for the heavy collect hops).

Effect on the 10-ship fleet (`fleet10_master_v1`, full price loop; before → after, asteroids,
refined final mass excluding collected ore): ship 1 548.3 → 583.2 kg (8 → 8, 693 → 574 kg);
ship 2 372.4 → 490.3 (6 → 6, 1030 → 584); ship 3 419.2 (unchanged, E→16459 ban again); ship 4
341.3 → 433.7 (5 → 6, 1047 → 541); ship 5 333.4 → 452.2 (6 → 6, 1004 → 635); ship 6 329.1 →
446.0 (5 → 6, 1270 → 553); ship 7 314.3 → 448.9 (5 → 6, 968 → 620); ship 8 310.7 → 396.7 (5 →
6, 1091 → 616); ship 9 299.2 → 354.8 (5 → 5, 1232 → 638); ship 10 346.9 → 373.7 (6 → 7, 709 →
571). Greedy fleet 3614.9 → 4398.7 kg (**+21.7 %**); 9 of 10 ships certified a re-timed
variant, in 2–4 SCvx attempts (36–132 s per ship, 14–20 legs re-flown per attempt). Ships 2, 5
and 7 gained > 30 % from re-timing alone (no new asteroid): their beam-search chains had 430–560
kg of propellant left that the DP turned into 2–3 years more mining per miner.

`clusters.py` — co-moving families: features `(a, e cos ϖ, e sin ϖ, i cos Ω, i sin Ω, λ)` scaled
by the reference p95 bands (0.04 AU, 0.06, 4.5°, 8° in mean longitude), a `cKDTree` ball count
gives each asteroid's co-moving density, greedy density-ordered labelling gives clusters, and
`phasing_window` returns the first window in which two members are within a phase band. The
prior in the beam (`cluster_min_density`, `cluster_bonus_kg`, co-moving-first expansion) is
implemented but **off by default**: on the full-catalogue pool it lost the 544–548 kg
8-asteroid chain (446 kg at best; with the 0.85 Earth-leg ratio it reached 549 kg proxy but
those Earth legs failed SCvx). The same element bands are used after the beam instead — for
insertion candidates, for ranking another ship's orphans against a collect tour, and for
seeding the next ship's first level.

`cooperative.py` — miners as shared resources. `RoutePlan` carries `foreign_deploy_epochs`
(miners this ship collects but another ship deployed) and may leave *orphans* (deployed, not
collected); `refine_route`, `emit_solution`, the re-timer's DP (a foreign collect is admissible
only ≥ 1 year after the *deployer's* epoch) and the independent verifier all handle the split.
`MinerPool` is the fleet-level registry (each asteroid deployed once, collected once, a collect
must match the registered deploy epoch); ship *k*'s extension may insert pool orphans into its
collect tour (first, or beside the tour asteroid they co-move with best) and, when
`orphan_credit > 0`, may deploy-and-leave a miner valued at `credit × k (T_end − 400 d − t)`.
The next ship's first level earns `seed_bonus_kg` (120 kg) for Earth targets co-moving with an
orphan ("pricing seeded from uncovered clusters"). `solve_fleet_master` is the G7-style master:
columns are certified itineraries (every certified candidate and every certified re-timed
variant of every ship), the objective is the fixed-bonus score `Σ B_i M_i`, constraints are
deploy-once / collect-once per asteroid, the deployer of every foreign collect must be in the
fleet with the same epoch, `N ≤ min(100, 2 exp(0.004 M̄))`, and a ship cap; solved exactly by
depth-first branch and bound with a suffix-sum bound and a node cap (deterministic, order
invariant). The greedy fleet is one of its feasible subsets, so the master never scores lower.

What cooperation did on the full catalogue (`fleet6_coop_v1`, credit 0.5): ships 1, 2, 4 and 6
took 1–3 orphan insertions each (9 orphans, e.g. ship 1 → 10 asteroids: 8 collected + 2 left),
because a deploy-only hop on a light ship is cheap and the credit (≈ 45 kg per orphan) beat the
10–36 kg of own collection it displaced. **Nobody collected them**: the next ship's beam still
started in a different cluster (the 120 kg seed did not outweigh the Earth-leg economics) and
every cross-cluster foreign collect was DP-infeasible, so the fleet scored 2641.8 kg versus
2744.9 kg self-cleaning. On a co-moving pair the mechanism itself works: ship 1's un-re-timed
route (548 kg) collecting two orphans deployed 660–720 days after launch by a neighbouring ship
re-times to 705 kg (10 asteroids) at proxy level. The credit therefore defaults to 0 (foreign
collects are still attempted when orphans exist), and the honest conclusion is that cooperative
collection needs the *deployer and collector to be planned together* in one cluster (the JPL
pattern: 279/320 collections cooperative, ships sharing a/e/i bands), which is the master's
next pricing problem, not a greedy side effect.

Master convergence: `fleet6_coop_v1` 20 columns, 9,177 nodes, exhaustive, optimum = greedy
incumbent (2641.8 kg). `fleet10_master_v1` 31 columns (3 candidates + up to 2 certified re-timed
variants per ship), 200,009 nodes (node cap), suffix-sum upper bound 12,256 kg (loose: it adds
every ship's best column ignoring asteroid conflicts), incumbent = greedy 4398.7 kg, 21 columns
rejected as dominated by the incumbent's column of the same ship (`fleet/master.json`). With
one launch slot per ship and no cooperative columns the master is an audit, not a lever; it
becomes one when several ships price against the same cluster.

### 6.5 Cooperative cluster pricing and bundle columns (`bundles.py`, third campaign)

The third campaign attacks per-ship mass directly: the master's pricing problem is solved *per
co-moving family* (deployer + collectors planned together), and the master accepts the result as
one multi-ship column.

**Earth legs.** The gate that kept every earlier ship in the a 2.23–2.43 AU region was our own
Lambert authority ratio (0.5), not physics: the three archived Antipodes Earth legs (a 2.77 AU,
509–587 d, Lambert ratio 0.80–0.92) all certify in our ZOH SCvx in 2–5 s at the reference
propellant (503/428/445 kg), and reference hops that "fail" at 3000 kg certify at the mass they
actually fly (1300–2300 kg). `certify_earth_legs` therefore screens the family from the launch
grid with a permissive limit (0.9× inflation, ratio 0.95), ranks the distinct `(target, launch)`
legs, and *flies the best ones in SCvx* before the beam sees them (≤ 12 checks, ≤ 4 certified
legs per ship slot, cached per family). `RouteSearch(first_level=...)` then seeds the beam from
those legs: each certified leg unlocks the Lambert grid for its target within ±200 d of its
launch and TOF, priced with the target's *measured*/Lambert ΔV ratio, while the exact certified
legs enter at their measured propellant. (Seeding with the four exact legs alone starved the beam
— one arrival epoch each — and three of the first four families closed no chain.)

**Return pruning.** The beam pruned first asteroids whose Earth return did not fit at a mass
guess of "post-deploy mass + cargo" (~3000 kg). The ship actually returns at dry mass + cargo +
return propellant (~1300–1500 kg), so ratio-0.35 returns looked like 0.7 and every a-2.77 family
was unreachable. The guess is now `min(post-deploy + cargo, dry + cargo + 200 kg)`.

**Ships in a family.** `price_cluster` builds up to three itineraries one after another inside
the family (beam → SCvx → joint re-timing/extension with the shared `MinerPool`); ships 1–2 may
leave miners (orphan credit 1.0 kg/kg) and later ships insert them as foreign collects, the last
ship never leaves any. Legs SCvx refuses are *banned* for every later slot of the family
(`RouteSearch.banned_pairs` for hops, `banned_earth` for grid Earth legs, filled from the refined
route's failure records), refinement flies one chain per distinct Earth leg with exactly-certified
legs first, and the beam is re-run (≤ 2 times) when everything flown was refused and something new
got banned. **Orphan repair** then guarantees an orphan-free bundle: each leftover miner is offered
to every ship as a foreign collect (re-timed, re-certified); failing that, the deployer drops the
visit (re-timed, re-certified) *or* reverts to its best clean certified variant, whichever collects
more (a re-timed chain that speculated on orphans can fall below the plain chain once they are
dropped: 455 vs 465 kg on the first family). Families are priced cheapest-first
(`rank_families`: mean propellant of the 5 cheapest Lambert Earth legs + 4× the nearest-hop
proxy; 61 s for the 132 families of the box) — largest-first spent the first 20 minutes on
eccentric 49-member families with 1.3 km/s hops and 10 km/s returns.

**Parallel pricing.** `price_clusters` runs `price_cluster` in forked worker processes (the
catalogue and settings are inherited copy-on-write), submitting families as workers free up so
the declared budget stops new families promptly; each worker returns a picklable `ClusterBundle`
(routes, variants, reject log, repairs, cooperative statistics, peak RSS), and a worker exception
becomes a "crashed" bundle rather than the end of the campaign. Peak RSS per worker 250–430 MB,
main process ~0.45 GB (bound: 2 GB total with 4 workers).

**Bundle master.** `FleetColumn.from_bundle` aggregates a family's ships into one column
(deploys/collects/masses of all members; a foreign collect satisfied inside the bundle is not a
foreign requirement of the column) that counts `len(members)` ships towards the fleet rule.
`solve_fleet_master` takes bundle and single-ship columns alike: `greedy_fleet` first (value
order and value-per-ship order, then drop the lowest value-per-ship columns until
`N ≤ 2 exp(0.004 M̄)` and the foreign closure hold — the fixed-point iteration on the mean mass),
then the exact branch and bound from the best of those incumbents and the caller's previous
selection (warm start; columns are only ever added, so it stays feasible). The bound at every node
is the **ship-rule bound** (`ship_rule_bound`): adding `k` ships collects at most the `k` largest
remaining per-ship masses, so if `N + k` already breaks `2 exp(0.004 M̄)` at that mass no
`k`-ship completion is feasible, and the value gained is at most the `k` largest remaining
per-ship values. With the plain suffix-sum bound the master hit its 200k-node cap at 45 columns
and fell back to the greedy — which is not monotone in the column set, so the campaign's fleet
went from 11 ships / 4839.7 kg to 9 ships / 4089 kg when a family was *added*; with the ship-rule
bound the same 64 columns solve exhaustively in 34k nodes (0.2 s). `cluster-fleet` verifies the
incumbent fleet (independent + official verifier) whenever the master changes it, writes every
verified intermediate fleet to `fleets/`, and records score at the 30 min / 1 h / 2 h / 4 h marks.

**Archived routes as columns (`archive.py`, `gtoc12 fleet-master`).** Every run archives its
emitted ships as `ship_NN/**/route_summary.json`. `fleet-master` rediscovers those archives
across runs (grouped by ship-directory parent: a family of a cluster run or a fleet run;
`fleets/` and `viewer/` are skipped), rebuilds each plan — recent archives embed it, older ones
only carry the flown legs and collected masses, from which `plan_from_route_summary` recovers
deploys (first arrival), collects (revisit arrival or camp departure, whichever reproduces the
archived mass) and foreign collects (a first-visit collect with no own deploy, snapped to the
deployer's epoch inside the group) — then **re-flies every leg through SCvx** in forked workers
before it may become a column. Failed primaries and collectors left without their deployer are
dropped and logged; groups keep their cooperative structure as bundle columns. Reconstruction was
checked exactly against the ten `fleet10_master_v1` plans (camps included) and every priced
family. This is how the fleet10 ships and the cooperative bundles are selected *together*.

### 6.6 Per-ship mass levers (fourth campaign: `earthleg.py`, `harvest.py`, `legstats.py`)

The fleet rule `N ≤ 2 exp(0.004 M̄)` binds at 15 ships (average 505 kg), so the fourth campaign
attacked the per-ship leg economy. `gtoc12 leg-stats` decodes any solution file (ours and the
references) through the shared itinerary decoder and reports per-role propellant/TOF
distributions; the measurement that framed the work (`results/gtoc12/leg_stats/before_v4.json`,
propellant per leg, `fleet_master_v1` vs JPL 36 / Antipodes 39 / Antipodes 37):

| Leg role | Statistic | ours (15 ships) | JPL 36 | Antipodes 39 | Antipodes 37 |
| --- | --- | --- | --- | --- | --- |
| Earth-out | median / mean / p90 kg | 484 / 479 / 601 | 447 / 473 / 561 | 461 / 472 / 537 | 466 / 474 / 536 |
| Earth-out | TOF median d | 540 | 530 | 523 | 532 |
| deploy hop | median / mean / p90 kg | 100 / 111 / 172 | 101 / 107 / 162 | 96 / 103 / 160 | 97 / 105 / 165 |
| deploy hop | TOF median d | 240 | 182 | 174 | 183 |
| **collect hop** | median / mean / p90 kg | **110 / 109 / 165** | **67 / 69 / 101** | **66 / 67 / 103** | **66 / 66 / 102** |
| collect hop | TOF median d | 330 | 187 | 181 | 182 |
| Earth return | median / mean / p90 kg | 192 / 205 / 277 | 211 / 216 / 252 | 214 / 214 / 246 | 206 / 208 / 249 |
| hops ≤ 75 kg | fraction | 0.21 | 0.44 | 0.46 | 0.46 |
| per ship | Earth-out kg / hops kg | 479 / 1448 | 460 / 1464 | 472 / 1448 | 474 / 1453 |

After the campaign (`results/gtoc12/leg_stats/after_v4.json`, same decoder; `cluster_fleet_v4`
is the 14-ship fleet the campaign itself emitted, `fleet_master_v2` the 16-ship best fleet that
mixes v4 ships with earlier archives):

| Leg role | Statistic | `cluster_fleet_v4` (14 ships, all continuous Earth legs) | **`fleet_master_v2`** (16 ships) | references (JPL 36 / Ant. 39 / Ant. 37) |
| --- | --- | --- | --- | --- |
| Earth-out | median / mean / p90 kg | **399 / 404 / 475** | 471 / 468 / 612 | 447–466 / 472–474 / 536–561 |
| Earth-out | TOF median d | 600 | 545 | 523–532 |
| deploy hop | median / mean / p90 kg | 122 / 129 / 198 | 110 / 114 / 168 | 96–101 / 103–107 / 160–165 |
| deploy hop | TOF median d | 255 | 240 | 174–183 |
| collect hop | median / mean / p90 kg | 102 / 107 / 173 | **90 / 95 / 152** | 66–67 / 66–69 / 101–103 |
| collect hop | TOF median d | 300 | 292 | 181–187 |
| Earth return | median / mean / p90 kg | 194 / 202 / 257 | 192 / 197 / 267 | 206–214 / 208–216 / 246–252 |
| hops ≤ 75 kg | fraction | 0.20 | 0.23 | 0.44–0.46 |
| per ship | Earth-out kg / hops kg | **404** / 1525 | 468 / 1454 | 460–474 / 1448–1464 |

The continuous Earth-leg optimiser (§ below) put the campaign's Earth legs **70 kg per ship below
the references** (404 vs 460–474 kg mean; the 324 optimised grid legs went from a 477 kg median to
390 kg, saving 102 kg on average, p10/p90 20/218 kg, with the flown legs at 397 kg median). The
saving did not become collected mass: the v4 ships spent it on deploy hops (129 vs 111 kg mean)
and the collect hops stayed at 102 kg, so the per-ship average of the v4 fleet is 498 kg against
505 kg before. The collect hops of the best fleet improved from 110 to 90 kg median because the
master now picks v4 ships whose collect tours the re-timer made cheaper, not because the phase
drift was removed (still 0.23 of hops under 75 kg vs 0.44–0.46).

Earth legs and deploy hops were already at the reference cost before the campaign; the whole gap is in the **collect
hops** (110 vs 66 kg, 7 per ship ≈ 300 kg) — the hop propellant per ship is *equal* (1448 vs
1448–1464 kg) but the references buy 9–10 asteroids with it and we buy 6–7. Self-cleaning tours
re-fly the deploy pairs three years later, when the family's relative phase drift (a few degrees
per year) has made the same pairs cost 2–3× (family 0, measured by SCvx: 5441→57635 1.29 km/s
out, 3.25 km/s back; 23907→16356 2.29 vs 4.95 km/s), in either direction (forward tours were
built and measured: 270/318/365 kg, no better than reverse). Nearest-neighbour chains over the
*pooled* miners of a family at the collect epochs stay at 1.3–2.0 km/s — the reference collect
economy — which is why the references' ships both deploy and collect across a shared cluster.

What was built (all deterministic, CPU):

- **Continuous Earth-leg optimisation (`earthleg.py`).** A Lambert-surrogate compass search was
  built first and rejected: on 181 certified Earth legs the measured/Lambert ratio scatters
  0.86–1.51 at equal authority ratio, so the surrogate steers towards shorter TOFs SCvx cannot
  fly. `refine_leg_scvx` runs the compass search (launch epoch × TOF, official launch window and
  v∞ ≤ 6 km/s as bounds, line search, time weighted at the mining rate 0.22 kg/day) with the SCvx
  arc as the objective: 8 SCvx calls (~20 s) per certified grid leg, −104 kg per leg on six
  archived legs (393–602 → 292–410 kg), 409/368/374/385/397/374 kg in the family-0 probe.
  `Retimer.protect_earth_leg` keeps the DP's Earth-out TOF at or above the optimised one (its
  Lambert table barely depends on TOF and would shorten the leg again). The launch grid spans the
  first three years of the window (`launch_window_days`), as the references do.
- **Ratio-dependent low-thrust inflation (`screening.low_thrust_inflation`).** Fitted on 1674
  certified hops: measured/Lambert = 1.05 + 0.65 r (r = Lambert ΔV over full-thrust authority),
  p60–p75 of every ratio bin. The flat 1.2× under-priced fast hops by 9.5 % and over-priced slow
  ones by 11 %; the model is used in the DP/forward pass (per-pair residual calibration after each
  SCvx pass). The beam keeps the flat factor — with the model it closed fewer and shorter chains
  (6 asteroids / 401 kg vs 7 / 440 kg) because it priced its fast deploy hops out of the budget.
- **Phasing-aware families (`ClusterBands.visit_epochs`).** The family feature vector embeds the
  phase at the deploy *and* the collect epoch, so members are co-located at every visit epoch,
  not only at t0 (`relative_drift_deg_per_year`, `phase_difference_deg` for diagnostics). At radius
  2.0 the fraction of member pairs under 75 kg at the collect epoch rises to 98 % in the probed
  family (static families: 21 % of our flown hops).
- **Collection tour modes** (`greedy | reverse | forward | forward_revisit` in the beam,
  `orders_of` epoch-matches collects so a repositioning hop is decoded as such) and the
  **joint harvest (`harvest.py`)**: after a bundle's ships are certified, a deterministic
  multi-ship nearest-neighbour construction re-plans the collect tours over the pooled miners
  (each miner collected once, camps first, Lambert ΔV at the running collect epoch + 0.05 kg/day),
  the per-ship DP re-times each new order with the foreign deploy epochs, SCvx certifies, ships
  whose tour does not certify keep their route and the clashes are re-timed once more; the bundle
  adopts the result only if it collects at least as much and forms a consistent pool.
- **Pinned deploy epochs (`Visit.pinned_arrival`).** The family-0 probe of the harvest collected
  *less* (1014.6 kg / 3 ships): ship 1's re-timed variant speculated on 7 orphans, ships 2–3
  collected four of them as foreign (488.7 / 474.3 kg), the orphan repair dropped the other three
  — which re-timed ship 1's remaining deploy epochs and stranded both collectors back to their
  clean variants (328 / 333 kg). Any re-timing of a deployer now pins the deploys another ship
  collects to their exact lattice epoch (`drop_asteroid`, `retime_harvest`; off-lattice pins are
  infeasible, not rounded), and a fallback variant must reproduce them.
- **Re-timing consistency.** Two latent DP/forward disagreements surfaced under the pins: the TOF
  grid was not on the lattice (a 400 d Earth bound on a 15/30 d lattice made the DP price and
  authority-check a leg at 730 d and fly it at 720 d, so the forward pass refused legs the DP had
  accepted), and the refused leg's own mass was missing from the profile correction, so the mass
  rounds corrected the wrong entry and never converged. `_tofs` snaps the grid, `_forward`
  returns the refused mass, and corrections carry across price rounds: a fixture plan that failed
  `leg_authority` when re-timed by its own re-timer now closes in two mass rounds.
- `MinerPool.register_all` registers a bundle in two phases (all deploys, then all collects):
  the joint harvest produces *mutual* pairs (ship 1 collects one of ship 2's miners and ship 2 one
  of ship 1's), which no deployer-before-collector slot order can register — the campaign's
  harvest of family 459 certified both new tours and was rejected as "collected but never
  deployed" for that reason alone.
- **Collect look-ahead in the beam** (`SearchSettings.collect_lookahead_weight`, off by default,
  `cluster-fleet --collect-lookahead W`): each deploy pair is also priced at
  `departure + 3 years` (cheapest collect TOF) and the beam score charges `W ×` that propellant;
  pairs that cannot be re-flown are not deploy candidates. Measured on family 247 (proxy
  pricing, 2 ships): `W = 0.5` collected 934 kg against 1023 kg with the look-ahead off, with the
  same collect-hop ΔV median (2.81 km/s) — the Lambert cost of the pair three years ahead does
  not predict which tour the DP re-timer will make cheap, so the option stays off.
- `FleetMasterResult.cooperative_columns` reports how many selected columns are cooperative
  (foreign collects / multi-ship bundles); `cluster-fleet --collector-harvest
  --earth-leg-refinements N --static-families`.

What the campaign measured about the harvest itself (`cluster_fleet_v4`, 4 h, 4 ships per
family, 47 families priced): the joint nearest-neighbour tours were attempted in 38 families and
**adopted in none** — in 8 no re-timed tour certified at all (ships stop with `no_reachable_miner`
after 3–6 collects), in 19 the certified tours collected *less* than the self-cleaning routes (every measured
before → after pair lost mass: 884.6 → 669.0, 1416 → 1168, 1739 → 1368, 903 → 641, 1666 →
1113, 1799 → 1388, 812 → 690 kg …; family 247 recovers to 791.8 kg in the offline replay with
the corrected 465 d return reserve, still below 884.6) and in 11 because the pool was
inconsistent ("collected but never deployed": the campaign ran with the single-pass
`register_all`, which rejects the mutual pairs the harvest produces — family 459 was verified to
be one — so part of those 11 are false rejections the two-phase registration now accepts). The
replay shows why: in a 13–20-member family the pooled alternatives
at the collect epochs are not cheaper than the ship's own pairs (the joint tour of family 247
still flies 5.1–7.1 km/s hops), so cooperation cannot buy back what the phase drift costs; the
references' 66 kg collect hops come from *tighter* clusters (Antipodes' 37-ship solution is
self-cleaning and has the same 66 kg collect hops as the cooperative JPL one), not from
cooperation per se. The lever is therefore family tightness at the collect epoch (Δa spread), not
the collector assignment — see §8.

### 6.7 Exact collect-tour pricing, collect-epoch families, master LP bound (fifth campaign: `collectdp.py`)

The fourth campaign left the collect hop as the whole gap (90–102 kg median against 66 kg in the
references). The Lambert look-ahead at a guessed collect epoch had failed to predict which deploy
chains re-fly cheaply, so the beam now prices the collect phase *exactly* for its own leg model:

- **Pair-cost table (`CollectPairTable`).** Per ordered asteroid pair, the zero-revolution
  Lambert ΔV on an absolute 30-day mission lattice × the collect TOF grid (90–600 d), computed
  lazily in one batched call per pair (about 0.12 s, 25 k solves/s) as a `float32 (n_t, n_tof)`
  block (7 kB), kept in a bounded LRU cache shared by every partial of a search
  (`collect_dp_cache_pairs`, 20 000 pairs ≈ 140 MB; the campaigns' searches used 364–782 pairs);
  the same for the Earth return of each asteroid with the 6 km/s arrival allowance and the
  end-of-window margin. ΔV becomes propellant through the certified-hop inflation model of
  `screening` (`hop_inflation`, ratio-dependent slope, authority ratio), evaluated once at the
  heaviest mass the ship can reach so a lighter ship is never priced optimistically.
- **Collect DP (`plan_collect_tour`).** A Held-Karp dynamic programme over
  `(collected subset, location, lattice epoch)` for the `k ≤ 10` deployed asteroids of a partial:
  every collect order is allowed (the deploy order is not imposed), the ship may leave its camp
  uncollected and come back (the "revisit" move), may camp anywhere (`maximum.accumulate` over
  arrival epochs), and every collect epoch is chosen with the official bookkeeping (rate × stay,
  one-year minimum, collection on departure) against the propellant of the hops it moves; the
  objective is `Σ collected − w × propellant`. Complexity `O(k² 2^k n_t n_tof)` vector operations:
  192–448 DP states and < 0.3 s per partial at `k = 7–8`; the table build dominates.
- **Beam integration.** `RouteSearch._complete` adds the DP tours at two weights (`w = 1.0` and
  the beam's 0.15) to the heuristic tours (greedy/reverse/forward) and ranks all by
  `plan_score = weighted mass − 0.15 × propellant`; the chosen order is re-priced by the exact
  forward mass pass and later by the 15-day re-timer with SCvx in the loop, as before.
  Telemetry per search: `collect_dp.priced/won/failed/seconds`, `collect_table_pairs`.
- **Collect-epoch families (`ClusterBands.collect_window`, `--collect-epoch-families`).** The
  phase embedding gets one `(cos, sin)` pair per epoch with weights: the deploy epoch (year 3) at
  0.5 and the harvest epochs (years 8.5, 11, 13.5) at 1.0, so neighbours are asteroids that stay
  co-located while the tour is flown rather than when the miners are dropped.
- **Master LP bound (`cooperative.py`).** For every fleet size `N` a sparse LP relaxation
  (asteroid packing ≤ 1, foreign-closure rows, `Σ ships = N`, and the fleet rule as the mass
  floor `Σ collected ≥ N ln(N/2) / ρ` — `ship_rule_mass_floor`) gives `lp_fleet_bound`; the
  combinatorial branch and bound keeps its ship-rule bound and warm start, and
  `lp_branch_and_bound` then branches on fractional columns for the sizes whose LP beats the
  incumbent. `FleetMasterResult` reports `lp_bound_kg`, `lp_gap_kg`, `lp_nodes`,
  `proven_optimal`; `fleet-master --lp-node-limit` (default 20 000). On the 312 archived
  single-ship routes the LP bound is 7997.5 kg (16 LPs, 0.3 s) where the DFS at 2 M nodes had
  7905.0 and a bound of 111 125 kg; the LP branch and bound finds 7987.3 kg and proves it in 39
  LPs. The pure dual node bound (`LpBound.node_bound`, root duals + reduced costs) is kept but
  did not prune the DFS on these degenerate packing duals.

What the probe and the campaigns measured. Family 247 with the v4 configuration (4 slots): 884.6
→ 986.0 kg over the same two ships (+11.5 %). `cluster_fleet_v5` (v4 configuration + DP, 4
workers, 38 families, 128 certified ships) and `cluster_fleet_v5c` (the same with collect-epoch
families, 3 workers, 27 families, 94 ships) both reached **17 ships / 9101.9 and 9111.3 kg
verified** on their own — above the 16-ship / 8324 kg fleet that had needed six archives — with
12 and 10 ships at or above 535 kg (the v4 campaign had 3). The DP was priced 20 984 / 14 673
times (8391 / 5786 s in the workers), won the completion for 50 % / 57 % of the chains it
closed and failed to close 47 % / 38 % (the heuristic tours kept those). In the best fleet the
collect hop moved from 95.3 kg mean / 90.2 median / p90 152.5 to 89.7 / 87.1 / 140.0 (share
≤ 75 kg 0.233 → 0.292; references 66 kg, 0.44–0.46): the DP removed the expensive tail, not the
median, and ships now carry one more asteroid (8.1 vs 7.7 per ship) so the per-ship collect
propellant is unchanged (707 vs 691 kg). Deploy hops fell to the reference level (109.5 mean /
98.9 median vs 114.1 / 109.6; references 103–107 / 96–101). The Earth return got dearer (227 vs
197 kg mean; references 208–216): the DP ends its tours late and the return TOF grid is coarse.
The joint harvest again adopted almost nothing (v5: 38 attempts, 2 adopted, 27 collected less,
9 no certified tour; v5c: 27 attempts, 0 adopted); the two collect-epoch campaigns' families
differ in membership, and the v5c ships that entered the best fleet (8 of 18) are the
self-cleaning ones.

### 6.8 Calibrated DP hop costs, two-pass mass schedule, finer lattices (sixth campaign: `hopcalib.py`)

The fifth campaign left the collect hop at 87 kg / 240 d median. The sixth attacks the *pricing*
of the collect DP — what it believes a hop costs and how heavy the ship is when it flies it — and
the lattices the DP and the Earth return are searched on:

- **Per-pair calibrated hop cost (`hopcalib.py`).** Every certified asteroid→asteroid hop of the
  archive (`route_summary.json`, SCvx propellant and endpoints) is re-screened with the
  zero-revolution Lambert ΔV at its certified epochs, and the ratio `refined ÷ Lambert` is fitted
  by least squares on `[1, authority ratio, TOF/yr, |Δa|/0.1 AU, |Δλ|/π]` (`design_matrix`,
  `fit_inflation`); the constant is then shifted so that the chosen residual quantile
  (`--quantile 0.65`) is zero, i.e. 65 % of certified hops are priced at or above their true cost
  (conservative for feasibility, without the flat 1.2 tail). Fit on 3285 hops of the pre-v5
  archives, held out on the 2925 hops of `v5`/`v5c`/`probe_v5_family247`: coefficients
  `[0.917, 0.840 ratio, 0.035 TOF, −0.007 Δa, 0.387 Δλ]` — the authority ratio and the phase
  difference carry the signal, Δa nothing. Holdout residual (model − measured ratio): rms 0.093,
  p10 −0.116, median −0.013, p90 +0.055; propellant error median −0.9 kg, p10 −11.5, p90 +5.0,
  rms 12.2 kg. Baselines on the same holdout: the flat 1.2 rms 0.123 (p90 +0.109, p95 +0.187),
  the ratio-only slope model of `screening` rms 0.111 (median −0.067: it over-prices almost
  every hop). The fit is `results/gtoc12/hop_inflation_fit.json` (`gtoc12 hop-calibration
  --train … --holdout … --quantile 0.65 --output …`) and enters the DP pair table
  (`CollectDPSettings.inflation_fit`, `--collect-dp-inflation-fit`), where `pair_geometry` gives
  Δa and the wrapped Δλ per lattice epoch; the beam's forward mass pass prices the DP's legs with
  the same fit (`RouteSearch._dp_hop_inflation`) so the two never disagree.
- **Two-pass mass schedule.** The v5 DP priced every move at the heaviest mass the ship can reach
  (camp mass + every miner mined to the window end); certified tours priced by the DP's own table
  came out 100–300 kg dearer than the DP's chosen tour *because their Earth return was infeasible
  for it* (7.4 km/s at 1120 kg is ratio 0.36; at the heavy mass 0.60, over the 0.5 limit).
  `plan_collect_tour` now prices the move out of the `h`-th collected asteroid at
  `camp mass + mined(S) − burn × (h − 1)`; pass 1 runs with `burn = 0`, pass 2 with the pass-1
  tour's mean hop propellant (`burn_per_hop`, reported in `diagnostics`), and a tour that closes
  for no burn at all is retried once at a nominal 6 % of the camp mass. The forward mass pass
  still judges the result exactly.
- **Finer lattices.** DP step 15 d (`--collect-dp-step-days`), collect TOFs 60–600 d (11 values),
  Earth-return TOFs 240–720 d in 30 d steps (was 300–900 in 60); the pair cache is bounded at
  6000 pairs (`collect_dp_cache_pairs`, 14 kB each on the 15-day lattice).
- **Earth-leg prescreen (`--earth-prescreen-ratio 0.7`).** Measured on the archive first: 81 % of
  the Earth legs that certified fly below Lambert authority ratio 0.6, 32 % of the 0.6–0.7 legs
  certify, 9 % of 0.7–0.8, 5 % above; a hard skip at 0.7 would have discarded 28 % of the legs
  that certified (p95 of certified legs is 0.83 with the 6 km/s launch credit). Legs above the
  ratio are therefore *deferred* behind every cheaper pair in the SCvx queue, not dropped.
- **Tighter collect-window families.** Radius ≤ 1.0 on the four-epoch collect-window bands yields
  0–1 families on the 10 612-asteroid pool (the scaled distance grows with the extra phase
  features); the campaign runs radius 1.75 with ≥ 20 members (47 families, median 26, max 54) and
  5 ships per family.
- **Memory accounting.** `cluster-fleet` samples the process tree's proportional set size
  (`/proc/<pid>/smaps_rollup` Pss of the main process and its workers, `_MemorySampler`) and
  reports `memory_total_pss_peak_mb`, the number the "< 2 GB total" budget is judged on; RSS sums
  over forked workers double-count the shared catalogue and libraries.

Probe of one radius-1.75 family with the full v6 configuration (5 slots): 5 ships, 582.8 / 598.4
/ 484.9 / 558.5 / 495.7 kg in 41 min — two above the 563 kg a 19th ship needed, where the best
archived single ship was 564.0 kg.

### 6.9 Worker memory, SCvx Earth-return sweep, harvest-window ranking (seventh iteration: `memory.py`, `returnsweep.py`, `returncampaign.py`)

- **Memory transient localised and fixed (`memory.py`).** The 3.04 GB process-tree PSS of four
  v6 workers had two causes, both measured on one slot of the 26-member family 54 with per-phase
  RSS marks (`PhaseMemory`, recorded in every `bundle.json` as `memory_phases`; `ru_maxrss` only
  grows, so the phase during which it grew hosted the peak): (i) *retained-but-free glibc heap* —
  RSS 670 MB at the end of the beam, 167 MB after `malloc_trim(0)`; glibc's dynamic mmap
  threshold moves the freed multi-MB DP tables onto the brk heap where they stay resident behind
  later allocations. Every worker now pins the threshold (`bound_heap_growth`, `mallopt`) and
  hands freed pages back at each phase mark (`release_heap`). (ii) A *live* +350 MB beam
  transient, all of it the collect DP (`collect_dp_stats["peak_growth_mb"]` attributes the
  high-water mark to the DP calls): `plan_collect_tour` cached the propellant fraction table per
  `(pair, move mass)`, and the move mass is the mined total of the collected subset, so every
  Held-Karp expansion produced a distinct 20 kB `float64` table — `2^k × k²` of them. The cache
  is a 512-entry LRU (`fraction_cache_entries`); the tables are recomputed otherwise and the
  route is bit-identical (664.1 kg on family 54, same legs). The pair-geometry cache is a 2048
  LRU (was an 80 000-entry dict) and the DP back-pointers `int32`. Single-slot peak 695 → 329 MB,
  the remainder being the Earth-leg SCvx phase (+227 MB, Clarabel). The declared budget is
  `MemoryBudget(slot_peak_mb=450, workers=3, parent_mb=250)` = 1.6 GB < 2 GB and the test
  `test_single_slot_pricing_stays_inside_the_declared_memory_budget` asserts a priced slot stays
  inside it (plus the trim marks). The return campaign's workers peaked at 221 MB RSS, the master's
  at 463 MB.
- **Earth return: TOF/ratio model, SCvx sweep, strict re-timing.** The certified archive shows
  the return's SCvx-over-Lambert inflation is strongly TOF dependent (1.38× at 420 d, 0.98× at
  540 d) while the DP priced every return at a flat 1.6; `screening.return_inflation_model`
  (piecewise-linear in TOF with an authority-ratio correction, fitted on
  `hopcalib.certified_returns`) replaces it in the collect DP and the beam
  (`CollectDPSettings.return_tof_model`, `SearchSettings.earth_return_tof_model`). For archived
  ships the model is bypassed altogether: `returnsweep.sweep_return` flies a lattice of
  departures (6 steps back to 6 forward of the archived return, 15 d) × TOFs (420–645 d, 45 d)
  through SCvx and the verifier-model rollout at the certified return mass, and
  `Retimer.set_return_sweep` prices the DP's return grid from the measurement — *strictly*: only
  swept certified cells and their neighbours (`RETURN_SWEEP_REACH` = 2 steps) are feasible, and a
  cell SCvx refuses on the re-flight is retired (`refuse_return`) before the next attempt (the
  first experiment lost its attempt to exactly that: the sweep certified (69203, 450 d) at
  215 kg, the re-flight refused it at the same mass, attempt 2 certified a neighbour). The
  re-timer's Lagrangian propellant price (bisected until the mass budget closes) is the
  fleet-rule exchange rate the DP's `w = 1` lacks: propellant is a constraint, collected mass the
  objective. `retime_return` / `returncampaign.run_return_campaign` / `gtoc12 retime-returns`
  apply this to every stand-alone certified archived ship (best first, deduplicated by asteroid
  set) and archive the improved routes for `fleet-master`.
- **Harvest-window deploy ranking (negative result).** `CollectPairTable.harvest_window_cost`
  is the window minimum (both directions, TOF ≤ 240 d, departures in years 11.4–13.9) of the
  calibrated pair cost; with `--collect-lookahead W` the beam charges it to each deploy pair
  (`harvest_window_costs`; an unreachable pair is charged 250 kg, not pruned — the pruning
  variant lost 124 kg at every weight). On family 54 the single-ship result is 664.1 kg at
  `W = 0`, 591.4 at 0.2, 524.4 at 0.5, 540.0 at 1.0: the share of collect hops ≤ 75 kg rises
  (0.12 → 0.38) but the deploy chain loses mass or an asteroid every time. The default stays
  `W = 0`; the collect-hop phase has to be attacked in the DP's window, not at deploy time.

### 6.10 Harvest substitution, sweep cells in the collect DP, external archives (eighth iteration)

- **Harvest substitution (`RouteSearch._substitution_pass`).** A ship only collects what it
  deployed, so "substitute a neighbouring miner inside `plan_collect_tour`" is a change to the
  deploy chain, not to the DP alone. It is implemented as a local search after the beam has
  completed its chains: the `substitution_hops` dearest collect hops (inflated Lambert ΔV) of the
  top `substitution_top` feasible plans name their endpoints (never the certified Earth-leg
  target); for each endpoint the substitutes are the screened deploy-hop neighbours of the chain's
  predecessor at the same departure, ranked by *(summed harvest-window pair cost of the substitute
  against the endpoint's tour partners − what the tour pays for the endpoint's hops now) + (deploy
  propellant into and out of the substitute at the chain's exact masses − the chain's own two
  hops)*; those predicted within `substitution_slack_kg` (60 kg) of paying are re-flown
  (`_rebuild_chain`: Earth leg verbatim, same camp waits, cheapest feasible TOF into/out of the
  substitute, every other hop re-priced at its shifted departure, one miner per stop, banned pairs
  and double visits refused) and re-toured through the same heuristics + Held-Karp DP + exact
  forward mass pass as any beam plan. A swap is kept only if the plan score rises; each accepted
  step is also kept as a fall-back candidate for SCvx. Exactness of the mass/time bookkeeping is
  therefore the beam's own (the test replays every accepted plan's legs and asserts the ship mass
  chain and the collected set to 1e-9 / no repeated asteroid). Telemetry per ship:
  `search.substitution` (`endpoints`, `candidates`, `tried`, `improved`, `gain_kg`, `seconds`,
  `best_predicted_kg`) and the rejected tours' reasons.
- **What the probe measured (family 7, 9 miners, 611.9 kg seed).** Ranking on the harvest side
  alone re-flew every substitute chain at +180 to +260 kg of deploy propellant and none closed a
  tour; with the deploy delta in the ranking the best prediction was +71.5 kg (no substitute is
  predicted to pay), and forcing ten re-tours at a 200 kg slack (deploy deltas +155 to +233 kg)
  ended every one in `mass_below_dry_plus_collected`. The beam already places its miners where
  the deploy hop is cheap; the dear collect hop is the price of that cheap deploy hop, and a
  substitute pays it back on the deploy side. The pass stays on by default (≈10 s per beam at the
  60 kg slack) so the campaign measures it across families; see §7 for the v8 telemetry.
- **Sweep cells in the collect DP (`CollectPairTable.set_return_sweep`).** The seventh
  iteration's SCvx return sweep only re-timed archived ships; new tours were still priced by the
  return TOF model (233 kg mean in v7, the sweep buying the rest afterwards). The family pricing
  now sweeps the camp as soon as the beam's route is certified (`bundles.sweep_route_return`:
  `return_sweep_back_steps`/`forward_steps` around the route's return departure ×
  `return_sweep_tofs` snapped to the re-timer lattice, nearest cells first inside
  `return_sweep_budget_seconds`, one shared cache per family) and hands the cells to *both* the
  re-timer and the slot's DP table. `return_override` mirrors the re-timer's strict rule on the
  table's (epoch × 30-day return TOF) grid: a swept cell lands on its nearest node with the
  inflation measured against the cell's own Lambert ΔV, only certified cells and their
  `RETURN_SWEEP_REACH` neighbours are feasible, and the authority-ratio prune is bypassed for a
  certified cell (SCvx already flew it). `_solve_collect_dp`'s terminal step and
  `_plan_from_tour` read the override, so a tour can end early and return long when the
  certified cells say so. Tests: `test_collect_dp_prices_the_return_strictly_from_certified_sweep_cells`
  (the DP picks the single certified cell and refuses everything else),
  `test_collect_table_prices_a_swept_return_from_the_certified_cells`,
  `test_sweep_return_flies_the_nearest_cells_first_and_stops_at_the_budget`, and the
  `price_cluster` wiring test.
- **External archives.** `fleet-master --source <dir>` accepts any archive directory (for example
  a `cluster_fleet_h100_v1` produced on another machine from a bundle of this branch);
  `discover_archives` groups columns by ship parent, so family labels overlapping ours do not
  collide (`test_discover_archives_groups_by_ship_parent_and_orders_variants`).

### 6.11 Whole-itinerary joint re-optimisation (eighth iteration, joint-itinerary branch: `jointopt.py`, `jointcampaign.py`, `gtoc12 joint-itinerary`)

Until now a certified ship was improved one lever at a time: the lattice DP re-timed the visit
order on a 15-day grid with Lambert leg costs, the return sweep measured one leg, SCvx certified
the result afterwards. `jointopt.optimise_ship` treats **every epoch of the ship's timeline as
one continuous decision vector** - launch, each visit's arrival and departure (hence every leg's
TOF, every miner's deploy and collect epoch and dwell), the Earth return - and optimises them
jointly against exact bookkeeping:

- **Objective and constraints (`JointItinerary.evaluate`).** Bonus-weighted collected mass with
  the mining-rate bookkeeping evaluated exactly (`10 kg/yr × (collect − deploy)`, deploy at the
  arrival of the deploy visit, collect at the departure of the collect visit) plus
  `margin_price` (0.05) kg per kg of spare final-mass margin - the exchange rate at which freed
  propellant is worth keeping for the insertion step. Hard constraints: the 2035–2050 window
  (`end_margin_days`), one-year minimum stay, non-negative dwell bounded by the re-timer's camp
  limits, the per-role TOF envelopes (Earth legs 400–900 d with the certified Earth-out TOF as a
  floor, hops 60–720 d), the authority-ratio limit per role including the re-timer's bans, no
  double deploy or collect, and the final mass `≥ dry + collected`; a schedule whose propellant
  does not close loads proportionally less ore (as `refine_route` sizes collected masses to
  fit), so the surrogate degrades smoothly instead of failing.
- **Leg costs.** Legs SCvx has flown at exactly these epochs (memoised by `(pair, departure,
  arrival)` and reused while the departure mass is within 60 kg) are priced with the *measured*
  verifier-model ΔV `v_e ln(m_before / m_after)`; every other leg uses the calibrated pair-cost
  surrogate: zero-revolution Lambert at the continuous epochs × the ratio-dependent hop model /
  TOF-ratio return model × the pair's SCvx-calibrated residual (`Retimer.leg_inflation`,
  `calibrate_from_route`). At the warm start every leg is measured, so the surrogate reproduces
  the archived route bit-exactly (`baseline_error_kg = 0` on all 32 ships) and the search
  starts from the true incumbent value.
- **Outer optimiser.** Deterministic steepest-ascent pattern search on a shrinking mesh
  (45, 20, 8, 3, 1 days): moves are single epochs (an arrival = a deploy epoch and the hop's
  TOF, a departure = a collect epoch or dwell), whole visits (dwell kept), the launch and the
  return, *phase shifts* (everything up to visit k's arrival, everything from visit k's
  departure) and the whole itinerary. Each mesh level runs to a local optimum; Lambert legs are
  memoised so a move re-prices only the legs it touches (≈4000 evaluations, 1100 Lambert
  solves, 2.6 s for a 16-leg ship).
- **SCvx in the loop, monotone acceptance.** Whenever the surrogate finds ≥ 0.5 kg more ore the
  whole itinerary is re-flown with the existing arc refiner (`refine_route`, 8–10 s per ship on
  one core), every leg is certified by the verifier-model rollout, and the route replaces the
  incumbent only if it certifies with more (weighted) collected mass. Each certification - accepted
  or not - memoises its measured legs and calibrates the pairs; a leg SCvx refuses bans its pair's
  authority ratio (`ban_factor`), and the level is re-searched from the incumbent. Because moved
  legs are priced at the calibrated residual × 1.03 margin, the finer meshes (≤ 8 d) rarely
  propose gains once the spare margin is spent: 1–5 certifications per ship, all but a handful
  accepted, 12–70 s per ship.
- **Insertion of one more asteroid.** After the schedule converges the co-moving neighbourhood
  of the camp (`returnsweep.neighbourhood`, radius widened to 2.5 bands) is enumerated over
  every (deploy slot, collect slot) with seeds that split the gap or lend camp dwell to the two
  new hops; feasible seeds are pattern-searched and the best certified. **Negative result on the
  v6 fleet**: 16–47 neighbours per ship, 0 feasible seeds - every candidate hop exceeds the
  authority ratio (6–15 km/s Lambert to the chain's members), i.e. the families the beam
  exhausted have no unused co-moving member left; the joint schedule frees 0–9 kg of margin,
  far from a 40 kg miner plus two hops.
- **Campaign (`jointcampaign.run_joint_campaign`).** The ships of a fleet report (matched to their
  archives by asteroid set, all archived variants considered, orphan-leaving stand-alone ships
  allowed) first, then the best remaining stand-alone archives; other fleet ships' asteroids are
  never inserted. Improved routes are archived as columns for `fleet-master`.

Tests (`tests/test_gtoc12_jointopt.py`): warm-start bookkeeping exactness (collected, spare,
per-asteroid mining rule, mass identity, bit-identical re-evaluation), rule violations (window,
dwell, stay, TOF envelope, double deploy/collect), pattern-search determinism and monotonicity,
certified-only acceptance (proxy refiner: every accepted certification is a strict gain and the
sequence is monotone; a refuser bans the pair and leaves the incumbent; a dearer refiner never
gets a lighter route accepted), insertion structure (exactly one new asteroid, one deploy and one
collect visit, sorted best first), fleet-first task selection with de-duplication.

### 6.12 Chain-level objective in the beam, reference-chain prior, LP-dual feedback, joint itinerary in the pricing (ninth iteration: `chainprior.py`, `search.RouteSearch._chain_score`, `cooperative.lp_asteroid_prices`, `archive.pricing_columns`)

The eighth iteration's diagnosis was that the collect hop (ours 85–87 kg / 210–225 d median
against the references' 66–67 kg / 181–187 d) is a *deploy-chain* property: the references pay
~70 kg more per ship on deploy hops (837–851 kg against our 768–773) so that consecutive miners are
also cheap harvest pairs, and neither a per-pair deploy-time surcharge (§6.9) nor a post-beam
substitution (§6.10) can move a chain the beam has already closed one pair at a time. The ninth
iteration therefore scores partial chains *by their eventual tour*:

- **Chain-level objective (`SearchSettings.chain_tour_scoring`, `RouteSearch._select` →
  `_chain_score`).** From `chain_tour_min_deploys` (3) deploys on, the `chain_tour_candidates`
  (48) best partials of a beam level by the heuristic score — after the same reserve, per-set and
  per-first-asteroid caps and the return-feasibility prune the heuristic selection applies — are
  re-scored by their actual collect tour: one Held-Karp pass of `plan_collect_tour` over the chain
  so far (the slot's calibrated pair table, sweep-cell returns, propellant weight 1.0) with the
  *parent chain's measured burn schedule* as the mass model (`_Partial.chain_burn`, the mean
  collect-hop propellant of the parent's scored tour; the nominal `chain_tour_burn_fraction` ×
  mass at the first scored level), and the tour is turned into a plan by the same exact forward
  mass pass every completed plan goes through (`_plan_from_tour` → `_finish`). The chain's score
  is then exactly the `plan_score` its completion would get — bonus-weighted collected mass at the
  tour's epochs − `propellant_weight` × everything spent from launch to Earth arrival − the
  master's asteroid prices — minus `chain_prior_weight` × the prior penalty below; a chain with no
  tour, or whose plan does not close on the true mass profile, keeps its heuristic score minus
  10 000 kg so it ranks below every closing chain but can still fill an otherwise empty beam. The
  beam is the best `beam_width` of the re-scored shortlist, so a costlier deploy hop is taken
  whenever it puts the miner on a cheaper collect loop *at the exchange rate the DP measures*.
  Everything is deterministic (fixed shortlist size, stable sorts, ties on epoch and asteroid
  IDs) and tours are cached per (deployed set with epochs, camp, camp epoch, mass, burn). Cost:
  the DP is 3 ms at k = 3 and 0.65 s at k = 8 on a warm pair table (1.7 s at k = 9, two passes) but
  a *cold* pair table costs ~0.19 s per ordered pair (366 lattice epochs × 11 TOFs of Lambert) and
  0.3 s per return table, so adding an asteroid to a k-chain costs 1.3–1.9 s the first time; the
  shortlist is what keeps a beam at +140–180 s (48 candidates × 8 levels; ~8 500 children per
  level, so the shortlist is the top 0.6 %), and `_complete`'s own DP gets the warmed pairs back
  (family 7: beam 392 s with scoring against 610 s without on a loaded machine). Judging closure
  on the DP's own mass model instead of the exact pass was the first version's mistake: the DP
  prices every move at the collected set mined to the window end and rejected every 9-asteroid
  chain of family 7 that `_complete` closes (`chain-scored 0` at level 9; 9 of 24 after the fix).
  Telemetry per beam: `search.chain_tour` (`scored`, `no_tour`, `not_closing`, `cache_hits`,
  `seconds`, `reranked` = selected partials the tour ranking moved into the beam, `levels`).
- **Reference-chain prior (`chainprior.py`, `gtoc12 chain-prior`,
  `benchmarks/gtoc12/chain_prior_v1.json`).** The three archived reference files (JPL 36,
  Antipodes 37 and 39 ships) are decoded with the shared itinerary decoder and, per ship, the
  propellant split by role, every hop's propellant and TOF, every hop's geometry (mean-longitude
  difference of the pair at departure and arrival, semi-major-axis gap) and the loop's compactness
  (a-spread of the deployed set, largest phase gap between consecutive collect stops) are
  recorded; the document stores the quantile tables, the per-ship records, the source files'
  SHA-256 and the commit, and a second extraction on the same pins is bit-identical (test).
  Measured on 112 ships / 1 014 collect hops / 903 deploy hops: collect hop median 66.3 kg (p75
  82.9, p90 102.1), TOF median 183 d; deploy hop median 97.9 kg (p25 75.7); per ship collect
  609 ± (p10 473, p90 724) against deploy 845 (p10 746, p90 942), collect share of the hop
  propellant 0.42 (p90 0.50); |Δλ| at collect departure median 2.7° (p75 4.8°), Δa median
  0.01 AU; loop a-spread median 0.05 AU, largest phase gap 7.8°. `ChainPrior.penalty` is two
  per-hop terms so partial chains of any depth compare: the projected collect propellant above
  `collect hops × p75` (dear harvest), plus — when the collect hops run above the reference median
  — the deploy propellant *below* `deploy hops × p25` (the "cheap deploy, dear harvest" signature
  of §6.10: the references buy their cheap harvest with dearer deploy hops and a chain that has not
  paid that is charged the shortfall). It is non-decreasing in the collect propellant,
  non-increasing in the deploy propellant and zero on the reference manifold (tests); the beam
  subtracts `chain_prior_weight` (0.5) × penalty from the chain score — at 0.5 a chain whose eight
  collect hops run 20 kg above the reference p75 loses 80 kg of score, about one asteroid's mining,
  so it ranks below an equal-depth reference-like chain but not below a chain one asteroid
  shorter. Every threshold is a named quantile of the JSON; the weight is the one free parameter.
- **LP duals into the family pricing (`cooperative.lp_asteroid_prices`,
  `archive.pricing_columns`, `RouteSearch(asteroid_prices=)`).** After every master solve the
  campaign re-solves the master's LP relaxation (§6.7's `_LpModel`) at the requested fleet size
  when feasible (`--dual-target-size 22`: which asteroids stand between the archive and a 22nd
  ship), else at N* + 1, else at the largest feasible size, and reads the duals of each asteroid's
  deploy and collect rows as its *price* (kg of master objective the fleet already pays for it);
  `price_clusters` evaluates the price provider when a family is *dispatched*, so every family that
  starts after a master solve prices around its duals (dual feedback monotonicity test: the
  dispatch-time snapshot per family). The beam subtracts the prices of the asteroids a chain claims
  from its heuristic score, its chain score and `plan_score` (the column's reduced cost;
  `dual_price_weight` 1.0), never from the reported mass, so a family emits the ship the master
  would take. Because one campaign's families are disjoint, prices only bite when the LP also
  holds the *archive's* columns: `--dual-archive <run>` adds every earlier run's certified routes
  as pricing-only columns (`pricing_columns`: asteroid sets and archived collected masses from the
  `route_summary.json`, no re-certification, never assembled; 915 columns from the seventeen
  archives in 0.5 s). Over those the LP reproduces `fleet_master_v7`'s bound (11 396.8 kg at N = 21,
  LP(22) infeasible) with ship-count dual μ = 1 316 kg, mass-floor dual ν = 1.40 (the rule is
  binding: a column enters the basis when 2.4 m_c − Σ prices > 1 316, i.e. a conflict-free ship of
  ≥ 548 kg) and 22 priced asteroids (max 131 kg, median 35): a price is a *conflict* price, not a
  value (a single family's disjoint ships price nothing — test).
- **Joint itinerary inside the pricing (`ClusterPricingSettings.joint_itinerary`).** Every
  emitted stand-alone ship is handed to §6.11's `optimise_ship` right after `improve_and_certify`
  (same calibrated re-timer, SCvx in the loop, insertion off, `joint_budget_seconds` 150 inside
  the family budget) and the certified result replaces the slot's route when `plan_value` rises;
  collectors (foreign miners) are skipped as in §6.11. Probe on family 7 slot 1: 616.4 → 622.6 kg in
  17 s (2 certifications), the same chain v8 found.
- **Pipeline (`cluster_fleet_v9`).** beam with chain-tour scoring + prior + prices → collect DP
  with sweep cells → harvest substitution → SCvx → return sweep → re-timing → joint itinerary →
  archive-wide master (`fleet_master_v8`) with the LP bound. Same partition, seeds, workers and
  budgets as v7/v8 (paired A/B).

### 6.3 Proxy validation (`results/gtoc12/proxy_validation.json`)

| Data set | Quantity | p5 | p25 | median | p75 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| our certified legs (164, all runs) | refined ÷ proxy propellant | 0.66 | 0.90 | 0.96 | 1.02 | 1.08 |
| our certified legs (164) | refined − proxy propellant (kg) | −112 | −8.5 | −3.6 | +3.1 | +26 |
| our certified legs (164) | refined ÷ Lambert ΔV | 0.98 | 1.08 | 1.165 | 1.23 | 1.49 |
| our deploy hops (71) / collect hops (64) | refined ÷ Lambert ΔV, median (p95) | 1.12 (1.30) / 1.19 (1.29) | | | | |
| our Earth-out (16) / return (13) | refined ÷ Lambert ΔV, median (p95) | 1.27 (1.74) / 0.98 (1.04) | | | | |
| reference hops (1882) | true ÷ Lambert ΔV | 1.03 | 1.10 | 1.16 | 1.23 | 1.41 |
| reference hops (1882) | true ÷ phasing/Edelbaum ΔV | 1.02 | 1.34 | 1.76 | 2.29 | 3.68 |
| reference hops (1882) | true ΔV ÷ full-thrust authority | 0.10 | 0.25 | 0.37 | 0.52 | 0.79 |

Spearman rank correlation with the true ΔV: Lambert 0.90, phasing/Edelbaum 0.63 (0.47 with scalar
Δe/Δi). The Lambert-free proxy is therefore only a pre-ranker; the zero-revolution Lambert ΔV with
1.2× inflation is the screening cost (hops land at 1.12–1.19× with a 1.30× tail, so ~half the
hops cost a few kg more than planned and the mass reserve absorbs it). Earth-out legs on the
catalogue pool reach 1.74× Lambert, which is why the 1.6× Earth factor stays: lowering it to 1.3
(runs `*_search3`) admitted 450–500-day Earth legs SCvx could not fly. Multi-revolution Lambert was
not needed: reference hops have zero revolutions (p95 0.004).

## 7. Results (all CPU, 16-core WSL2; single process up to `fleet10_master_v1`, 4–6 worker processes for the third campaign)

| Run | Instance | Ships | Asteroids | Collected mass (official) | Fixed-bonus score | Refined arcs | Search wall | Refine wall | Total wall | Hardware |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `reduced-v1-run1` (beam 24, ≤4 deploys, before phasing fixes) | gtoc12-reduced-v1 | 1 | 2 (36777, 37351) | **195.044 kg** | 195.044 kg (both B = 1) | 4 | 18.5 s | 7.0 s | 37 s | CPU |
| `reduced-v1-run2` (beam 24, ≤4 deploys, phasing-aware) | gtoc12-reduced-v1 | 1 | 4 (1265, 21191, 27292, 40808) | **253.744 kg** | 249.059 kg | 8 | 24.5 s | 6.0 s | 47 s | CPU |
| `full-catalogue-run1` (beam 8, ≤3 deploys, 39.1 M Lambert screens) | full catalogue (60,000) | 1 | 3 (20194, 23644, 15033) | **249.035 kg** | 202.995 kg | 6 | 956 s | 7.1 s | 963 s (11.2 GB peak RSS) | CPU |
| `reduced_v1_search3` (search v2: beam 48, ≤8 deploys, 64 neighbours) | gtoc12-reduced-v1 | 1 | 5 (37351, 36777, 44316, 6249, 1128) | **314.442 kg** | 314.442 kg | 10 | 194 s | 34 s (3 candidates) | 228 s, 0.55 GB | CPU |
| `full_catalogue_search2` (search v2, beam 32, ≤10 deploys, 64 neighbours, pool 10,612) | full catalogue | 1 | 8 (8846, 27861, 37385, 49900, 8123, 1122, 12992, 57949) | **548.282 kg** | 548.282 kg | 16 | 261 s | 42 s (2 candidates) | 303 s, 0.66 GB | CPU |
| `fleet3_full_catalogue` (search v2 pre-final, 3 ships greedy, beam 24, ≤10, 48 nb) | full catalogue | 2 (ship 3 uncertified) | 13 | **965.804 kg** | 893.263 kg | 26 | 719 s (3 searches) | 32 s | 751 s, 0.75 GB | CPU |
| `fleet3_full_catalogue_v2` (final code, 3 ships greedy, beam 32, ≤11, 64 nb, 1800 s budget/ship) | full catalogue | 3 | 20 | **1394.11 kg** (548.28 + 442.22 + 403.61) | 1318.117 kg | 40 | 814 s (3 searches) | 53 s (5 candidates) | 867 s, 0.77 GB | CPU |
| `fleet6_retime_v1` (6 ships greedy + joint re-timing/extension with SCvx in the loop, beam 24, ≤10, 48 nb) | full catalogue | 6 | 40 | **2744.89 kg** (592.6 + 425.5 + 419.2 + 446.8 + 465.3 + 395.5) | 2744.89 kg (all B = 1: the bonus-weighted beam avoids asteroids other teams mined) | 80 | 6 × ~120–180 s | ~25 min incl. re-timing | 1938 s (32 min), 0.78 GB | CPU |
| `fleet6_coop_v1` (as above + cooperative pool, orphan credit 0.5, master over 20 columns) | full catalogue | 6 | 45 deployed / 36 collected | **2641.81 kg** | 2641.81 kg | 85 | 6 × ~120–180 s | ~28 min | 2080 s, 0.76 GB | CPU |
| `fleet10_master_v1` (10 ships greedy + re-timing with the full price loop, cooperative pool with credit 0, master over 31 columns; 2 h budget) | full catalogue | 10 | 62 | **4398.69 kg** (583.2 + 490.3 + 419.2 + 433.7 + 452.2 + 446.0 + 448.9 + 396.7 + 354.8 + 373.7) | 4398.69 kg (all B = 1) | 124 | 10 × ~130–220 s | ~40 min incl. re-timing | 3089 s (51 min), 0.76 GB | CPU |
| `cluster_fleet_v1` (third campaign: 108 co-moving families priced cheapest-first in 4 workers, deployer + collectors per family, orphan repair, bundle master; 4 h budget) | full catalogue | 14 | 99 | **6932.48 kg** | 6407.87 kg | 202 | 108 families × 166–1700 s (4 in parallel) | in the family pricing | 14921 s (249 min); main 0.43 GB, worker peak 0.60 GB | CPU |
| `cluster_fleet_v2_deep` / `cluster_fleet_v3_repair` (re-pricing of the 6 richest and the 7 lost families, 4–5 ships per family, 5–6 re-timing attempts) | full catalogue | 7 / 11 | 41 / 81 | 2878.00 kg / 4704.15 kg | 2603.22 / 4315.35 kg | 85 / 158 | 6 / 7 families | in the family pricing | 2742 s / 2166 s; 0.44 GB main | CPU |
| `fleet_master_v1` (master over every archived certified route of the four runs above: 208 routes re-flown through SCvx, 273 columns, 5 M nodes) | full catalogue | 15 | 109 (103 mined, 6 miners left) | **7575.58 kg** (per-ship table below) | 7217.51 kg | 217 | — | 405 s re-certification (6 workers) + 67 s master | 488 s, 0.24 GB main | CPU |
| `cluster_fleet_v4` (fourth campaign: phasing-aware families, continuous SCvx Earth legs, 4 ships per family, joint harvest, 4 workers; 4 h budget) | full catalogue | 14 | 100 (96 mined) | **6975.69 kg** | 6737.08 kg | 201 | 47 families × 523–2400 s (4 in parallel) | in the family pricing | 15151 s (252.5 min); main 0.46 GB, worker peak 0.81 GB | CPU |
| `fleet_master_v2` (master over every archived certified route of all six runs: 330 routes re-flown through SCvx, 436 columns, 5 M nodes) | full catalogue | 16 | 123 (116 mined, 7 miners left) | **8324.27 kg** (per-ship table below) | 7905.05 kg | 246 | — | 749 s re-certification (6 workers) + 103 s master | 872 s, 0.29 GB main | CPU |
| `probe_v5_family247` (family 247 with the v4 configuration + exact collect DP, 4 slots) | full catalogue | 2 | 14 | 986.04 kg (was 884.6 in v4) | 986.04 kg | 29 | 1 family, 679 s | in the family pricing | 826 s, 0.46 GB | CPU |
| `cluster_fleet_v5` (fifth campaign: v4 configuration + exact collect-tour DP in the beam, 4 workers, 4 ships per family; 4 h budget) | full catalogue | 17 | 137 | **9101.85 kg** | 8631.32 kg | 280 | 38 families × 470–2478 s (4 in parallel) | in the family pricing | 15584 s (260 min); main 0.45 GB, worker peak 0.83 GB | CPU |
| `cluster_fleet_v5c` (as `v5` with collect-epoch families, 3 workers; 4 h budget) | full catalogue | 17 | 140 | **9100.89 kg** (best intermediate fleet 9111.29 kg) | 8544.10 kg | 288 | 27 families × 608–2280 s (3 in parallel) | in the family pricing | 15348 s (256 min); main 0.44 GB, worker peak 0.77 GB | CPU |
| **`fleet_master_v3`** (master over every archived certified route of all nine runs: 554 routes re-flown through SCvx, 726 columns, 5 M nodes + LP branch and bound, **proven optimal**) | full catalogue | **18** | **147** (142 mined, 5 miners left) | **9888.57 kg** (per-ship table below) | **9329.82 kg** (LP bound 9334.32, gap 4.5 kg) | 298 | — | 921 s re-certification (8 workers) + 136 s master | 1075 s, 0.39 GB main | CPU |
| `probe_v6_family` (one radius-1.75 collect-window family with the sixth-campaign configuration, 5 slots, 1 worker) | full catalogue | 5 | 41 | 2720.19 kg (582.8 / 598.4 / 484.9 / 558.5 / 495.7) | 2720.19 kg | 86 | 1 family, 2496 s | in the family pricing | 2497 s, worker 0.95 GB | CPU |
| `cluster_fleet_v6` (sixth campaign: calibrated DP hop costs, two-pass mass schedule, 15-day DP lattice, 30-day return grid, Earth-leg prescreen, radius-1.75 collect-window families, 5 ships per family, 4 workers; 4 h budget) | full catalogue | **19** | 159 (155 mined) | **10698.0 kg** (marks: 60 min 8545.1 / 16 ships, 120 min 9887.8 / 18, 240 min 10697.1 / 19) | 9785.03 kg (LP bound 9796.82, closed by branching, exhaustive) | 328 | 36 families × 818–2630 s (4 in parallel) | in the family pricing | 15305 s (255 min); main 0.43 GB, worker peak 1.11 GB, **process-tree PSS peak 3.04 GB** | CPU |
| **`fleet_master_v4`** (master over every archived certified route of all eleven runs: 695 routes re-flown through SCvx, 903 columns, 2 M nodes + LP branch and bound, **proven optimal**) | full catalogue | **19** | **158** (154 mined, 4 miners left) | **10700.48 kg** (per-ship table below) | **10146.60 kg** (LP bound 10159.66, gap 13.1 kg) | 324 | — | 1020 s re-certification (8 workers) + 68 s master | 1108 s, 0.45 GB main | CPU |
| `return_sweep_v1` (SCvx Earth-return sweep + strict re-timing of the 36 best stand-alone archived ships, 3 workers, `nice 19`) | full catalogue | 13 improved of 36 | — | +140.5 kg (improved returns 172–247 kg; best ship 633.3 kg) | — | — | 36 ships × 99–590 s (3 in parallel) | in the re-timing | 2990 s (50 min); main 0.12 GB, worker peak 0.22 GB | CPU |
| **`fleet_master_v5`** (master over the eleven runs + `return_sweep_v1`: 690 routes re-flown through SCvx, 918 columns, 2 M nodes + LP branch and bound, **proven optimal**) | full catalogue | **20** | **168** (165 mined, 3 miners left) | **11521.42 kg** (576.07 kg average; per-ship table below) | **10573.17 kg** (LP bound 10589.13, gap 16.0 kg; LP infeasible at 21) | 344 | — | 2461 s re-certification (3 workers) + 71 s master | 2552 s, 0.46 GB main | CPU |
| `cluster_fleet_v7` (seventh campaign: v6 configuration + return TOF model in the DP, bounded DP caches, radius-1.6 / ≥ 18-member families (new partition), 5 ships per family, **3 workers, `nice 19`**; 4 h budget) | full catalogue | 18 | 147 | **9920.47 kg** (551.3 kg average; marks: 60 min 5355.8 / 10 ships, 120 min 8571.1 / 16, 240 min 9922.5 / 18) | 9173.66 kg (LP bound 9186.18, gap 12.5 kg, exhaustive) | — | 25 families × 797–3046 s (3 in parallel), 76 ships | in the family pricing | 14950 s (249 min); main 0.43 GB, worker peak 0.68 GB, **process-tree PSS peak 1.19 GB** | CPU |
| `return_sweep_v2` (return sweep + re-timing of the 21 best `cluster_fleet_v7` ships, 3 workers) | full catalogue | 13 improved of 21 | — | +225.5 kg (improved returns 159–274 kg; family 7 ship 3 528.3 → 602.9, ship 1 607.8 → 616.4) | — | — | 21 ships (3 in parallel) | in the re-timing | 2181 s (36 min); worker peak 0.21 GB | CPU |
| **`fleet_master_v6`** (master over all fourteen runs: 779 routes re-flown through SCvx, 1032 columns, 2 M nodes + LP branch and bound, **proven optimal**) | full catalogue | **20** | **168** (164 mined) | **11515.67 kg** (575.78 kg average; 5.8 kg below `v5` in raw mass, 166 kg above it in the score the master optimises) | **10739.27 kg** (LP bound 10744.85, gap 5.6 kg) | 343 | — | 2660 s re-certification (3 workers) + 69 s master | 2749 s, 0.49 GB main | CPU |
| `cluster_fleet_v8` (eighth campaign: v7 configuration + harvest substitution after the beam + SCvx return-sweep cells in the DP and the re-timer; same radius-1.6 / ≥ 18 partition as v7, **3 workers, `nice 19`**, 4 h budget, on a machine shared with two other agents' runs) | full catalogue | 19 | 161 | **10697.60 kg** (563.0 kg average; marks: 60 min 5155.8 / 10 ships, 120 min 8453.2 / 16, 240 min 10707.2 / 19; first fleet at 40 min) | 9502.24 kg (LP bound 9514.75, gap 12.5 kg, exhaustive) | 864 (all 59 ships) | 20 families × 1096–3026 s (3 in parallel), 59 ships | in the family pricing | 14938 s (249 min); main 0.43 GB, worker peak 0.52 GB, **process-tree PSS peak 0.91 GB** | CPU |
| **`fleet_master_v7_v8archives`** (this branch's eighth-iteration master over all fifteen v8-era runs: 854 routes re-flown through SCvx, 1109 columns, 2 M nodes + LP branch and bound; renamed from `fleet_master_v7` when `main` — which carries the joint-itinerary branch's 21-ship `fleet_master_v7` — was merged) | full catalogue | **20** | **168** (164 mined) | **11515.67 kg** (575.78 kg average — the same twenty ships as `v6`: no v8 column enters, see §7 text) | **10739.27 kg** (LP bound 10744.90, gap 5.6 kg) | 343 | — | 2767 s re-certification (3 workers) + 70 s master | 2855 s, 0.52 GB main | CPU |

| `joint_itinerary_v2` (eighth iteration, §6.11: whole-itinerary joint re-optimisation of the 20 `fleet_master_v6` ships + 12 best stand-alone archives, SCvx re-certification in the loop, 3 workers, `nice 19`) | full catalogue | 32 improved of 32 | — (0 inserted) | +280.4 kg (fleet ships +208.4: 575.78 → 586.20 kg average; +1.1 to +23.3 per ship) | — | — | 32 ships × 13–70 s (3 in parallel) | in the joint search (101 certifications, 95 accepted) | 435 s (7 min); main 0.12 GB, worker peak 0.12 GB | CPU |
| **`fleet_master_v7`** (master over all sixteen runs incl. `joint_itinerary_v1/v2`: 837 routes re-flown through SCvx, 1078 columns, 2 M nodes + LP branch and bound, **proven optimal**) | full catalogue | **21** | **177** (173 mined) | **12346.48 kg** (587.93 kg average vs the 21-ship threshold 587.8; rule 21 ≤ 21.007; all 21 columns are `joint_itinerary_v2` routes) | **11391.12 kg** (LP bound 11396.76, gap 5.6 kg) | 362 | — | 3237 s re-certification (3 workers) + 139 s master | 3398 s, 0.52 GB main | CPU |
| `cluster_fleet_v9` (ninth campaign: v8 configuration + chain-tour scoring (48 candidates per level), reference-chain prior (0.5), archive-seeded LP duals (915 pricing columns, priced at N = 21: 22–24 asteroids), joint itinerary on self-cleaning slots; same radius-1.6 / ≥ 18 partition and seeds as v7/v8, **3 workers, `nice 19`**, 4 h budget; the 260-min wrapper timeout killed the run in the last family's tail, so the last verified incumbent stands) | full catalogue | 18 | 155 | **9960.33 kg** (553.4 kg average; marks: 60 min 5197.9 / 10 ships, 120 min 8343.8 / 16, 240 min 9960.3 / 18) | 9015.34 kg (LP bound 9017.0, exhaustive; the killed final master would have repeated it) | — | 19 families × 1619–2636 s (3 in parallel), 60 ships, family 10 lost to a NaN burn schedule in the substitution pass (guarded since) | in the family pricing | 14750 s (246 min logged); main 0.44 GB, worker peak 0.53 GB, **process-tree PSS peak 0.92 GB** | CPU |
| `joint_itinerary_v3` (whole-itinerary joint re-optimisation of the 40 best stand-alone `cluster_fleet_v8` / `cluster_fleet_v9` ships, no insertion, 3 workers, `nice 19`) | full catalogue | 35 improved of 40 | — | +411.5 kg (v8 22 of 24, +241.0; v9 13 of 16, +170.5; best 622.6, 619.2, 601.2, 600.6) | — | — | 40 ships × 12–60 s (3 in parallel) | in the joint search | 466 s (7.8 min); worker peak 0.11 GB | CPU |
| **`fleet_master_v8`** (master over all nineteen runs incl. `cluster_fleet_v9` and `joint_itinerary_v3`: 1006 routes re-flown through SCvx, 0 failures, 1296 columns, 2 M nodes + LP branch and bound (7481 LPs), **proven optimal**) | full catalogue | **21** | **177** | **12356.30 kg** (588.40 kg average vs the 21-ship threshold 587.8; rule 21 ≤ 21.046; +9.8 kg over `fleet_master_v7`: three ships swapped, all 21 columns are joint-itinerary routes — 16 `v2`, 5 `v3`) | **11441.59 kg** (LP bound 11448.02, gap 6.4 kg; LP infeasible at 22) | — | — | 3271 s re-certification (3 workers) + 186 s master | 3477 s, 0.59 GB main | CPU |
| `cluster_fleet_h100_v1` (Lambda H100 host, 26-core Xeon 8480+, v6 recipe, 16 workers on cores 10-25, 6 h budget) | full catalogue | **19** | **158** | **10699.50 kg** (563.13 kg average) | 10012.61 kg (LP bound 10021.18, gap 8.6 kg; exhaustive) | — | 47 families × 1937–3257 s (16 in parallel) | in the family pricing | 8876 s (148 min); process-tree PSS peak 2.71 GB | CPU (H100 host) |
| **`fleet_master_h100_v1`** (master over 13 archives incl. `cluster_fleet_h100_v1`, 1055 columns, 16 workers) | full catalogue | **20** | **165** | **11517.60 kg** (575.88 kg average) | 10786.55 kg (LP bound 10798.44, gap 11.9 kg) | — | 1454 s re-certification + 126 s master | — | 1616 s | CPU (H100 host) |
| `cluster_fleet_h100_v2` (ninth campaign on the H100 host: union of 4 family partitions (radii 1.75/1.6 × collect-window + phasing bands, ≥ 20 members: collect_r1.75 47, phasing_r1.75 56, collect_r1.6 29, phasing_r1.6 35), 5 ships per family, beam 32, refine-top 3, 6600 s per family, harvest substitution + return sweep cells, 22 workers `nice 5`, 8 h budget; marks: 30 min —; 60 min —; 120 min 9104.6 / 17 ships; 240 min 11522.0 / 20 ships; 480 min 12348.7 / 21 ships) | full catalogue | **21** | **182** | **12348.90 kg** (588.04 kg average) | 11181.47 kg (LP bound 11242.54, gap 61.1 kg) | — | 111 families × 0–7614 s (22 in parallel), 386 ships (7 ≥ 600 kg, 1 ≥ 650, 0 ≥ 700, best 652.6) | in the family pricing | 33137 s (9.2 h); process-tree PSS peak 5.07 GB | CPU (H100 host) |
| `joint_itinerary_h100_v1` (§6.10 joint re-optimisation of every archived chain ≥ 450 kg of the 16 local archives + `cluster_fleet_h100_v1`, `fleet_master_v7` ships first, 4 workers alongside the campaign) | full catalogue | 294 improved of 339 | 0 inserted | +4665.0 kg (chains ≥ 600 kg 9 → 10, ≥ 650 0 → 0; best +67.2) | — | — | 339 ships × 1–419 s | in the joint search | 11033 s; worker peak 0.13 GB | CPU (H100 host) |
| `joint_itinerary_h100_v2` (§6.10 joint re-optimisation of every `cluster_fleet_h100_v2` chain ≥ 450 kg, 22 workers) | full catalogue | 219 improved of 231 | 0 inserted | +3211.6 kg (chains ≥ 600 kg 6 → 13, ≥ 650 0 → 0; best +71.7) | — | — | 231 ships × 1–236 s | in the joint search | 1217 s; worker peak 0.12 GB | CPU (H100 host) |
| **`fleet_master_h100_v2`** (archive-wide master over 21 archives (1902 routes re-flown through SCvx, 2480 columns, LP-bounded), 22 workers) | full catalogue | **22** | **187** | **13189.60 kg** (599.53 kg average) | 12203.96 kg (LP bound 12207.39, gap 3.4 kg) | — | 2411 s re-certification + 1135 s master | — | 3588 s | CPU (H100 host) |

Runs are single-process CPU (16-core WSL2, load shared with an unrelated G4 GPU campaign; the
RTX 5090 was at 100 % throughout and was not used). "Search v2" is the position-space,
reserve-pruned beam search of §6.2; `search2`/`fleet3_full_catalogue` were produced at an
intermediate commit of it (scalar Δe/Δi pre-ranking, no failed-leg skipping) and are kept as
verified artifacts; `reduced_v1_search3` and `fleet3_full_catalogue_v2` are reproducible from HEAD
(`--beam-width 48 --max-deploys 8 --neighbours 64 --refine-top 3` and `--full-catalogue --ships 3
--beam-width 32 --max-deploys 11 --neighbours 64 --refine-top 3 --stop-at-first-certified
--search-budget-seconds 1800`). Ship 1 of the final fleet reproduces the `search2` route exactly.
Best score by depth (proxy kg) for the final ship 1: 1→124.5, 2→221, 3→285, 4→338, 5→404,
6→445, 7→531, 8→548; depths 9–11 produced no completable chain. The second-campaign fleets
(`fleet6_*`, `fleet10_master_v1`) reproduce from HEAD with `--full-catalogue --ships N
--budget-seconds 7200 --retime-budget-seconds 900 --retime-attempts 4` (defaults: beam 24, ≤ 10
deploys, 48 neighbours, refine top 3, cooperative pool on, orphan credit 0); `fleet6_retime_v1`
predates the pool (equivalent to `--no-cooperative`) and used the earlier one-halving price
loop, `fleet6_coop_v1` orphan credit 0.5. `fleet10_master_v1` was the best verified fleet of the second campaign: official `GTOC12_Verify` "Check
successfully!" on the 10-ship file (62 asteroids, 4398.686 kg; fleet rule 10 ≤ 11.6), independent
verifier agreeing per asteroid to 1e-10 kg.

Where the runs stop: 140 of 155 failed chains in the widest run (`beam 48, 96 neighbours`) died
with `camp_negative` — the backward-scheduled collection tour ran past the deploy phase — at
depths 6–10, while every certified route still had 230–430 kg of propellant unspent (final dry +
propellant mass 693–1241 kg vs the 500 kg floor). Time, not mass, is the binding constraint:
our deploy hops take 240–300 days where the references take 140–240, because the candidate
clusters are thinner (per-ship a-spread 0.06–0.10 AU vs 0.035). Pricing time in the beam
heuristic (0.02–0.05 kg/day) or lowering the hop duty pushed the beam into 120–180-day hops at
the authority limit that SCvx could not fly (`full_catalogue_search4/5`: 447 kg), so those knobs
default off; the fix is in candidate generation (tighter co-located clusters), see §8.

Per-leg detail of the final fleet's ship 1 (TOF, certified propellant): E→8846 500 d 515 kg;
deploy hops 300/180/150/300/300/300/240 d = 72/114/100/87/92/88/75 kg; collect hops
300/300/240/300/240/180/240 d = 152/139/50/106/77/66/108 kg; return 400 d 146 kg — i.e. the
reference hop economy (median 78 kg) is reproduced; the gap to a 740 kg reference ship is one to
two more asteroids and ~600 days of collection-phase time.

Variants tried and rejected (all officially verified where they certified): Earth-leg inflation
1.3 (`*_search3`: 457 kg / no certified reduced route); time weight 0.05 + duty 0.8
(`search4`: no certified route, 3 marginal legs infeasible); time weight 0.03 + duty 0.7
(`search5a`: 244 kg, Earth legs pruned); time weight 0.02 + hop duty 0.75 (`search5`: 447 kg).
Second campaign (this section's fleets), rejected on the full-catalogue depth probe (24 launch
targets, proxy kg of the best chain): hop authority ratio 0.5 in the beam (376 kg, depth 6 —
collect hops of the heavy ship no longer fit); Earth-leg model 0.95×/ratio 0.85 from the
reference legs (549 kg proxy but E→6014/15614/26515 at ratio 0.71 failed SCvx, `fleet6_coop_v1`
first launch: no certified ship); cluster prior density ≥ 8 + 150 kg with the certified Earth
envelope (375–446 kg); co-moving-first expansion (464 kg vs 544 kg without); orphan credit 0.5
(fleet 2641.8 vs 2744.9 kg, nine uncollected orphans). Retained: the HEAD leg model
(Earth 1.6×/0.5, hops 1.2×/0.667), joint re-timing with hop ratio 0.45 and per-pair
calibration/bans, bonus-weighted beam scoring, the master.

Score versus wall-clock budget (single process, `fleet6_retime_v1` / `fleet10_master_v1`):
5 min → 592.6 kg (1 ship); 10.5 min → 1018.1 kg (2); 15.5 min → 1437.3 kg (3); 20.6 min →
1884.1 kg (4); 26 min → 2349.4 kg (5); **30 min → 2349.4 kg (5 ships)**; 32 min → 2744.9 kg
(6). `fleet10_master_v1` (same machine, the run also served the test suite for its first
10 minutes): 5.1 min → 583.2 kg (1); 9.8 → 1073.5 (2); 14.2 → 1492.7 (3); 19.5 → 1926.4 (4);
24.0 → 2378.6 (5); **30 min → 2378.6 kg (5 ships; the 6th certifies at 30.3 min → 2824.6)**;
35.5 → 3273.4 (7); 41.2 → 3670.1 (8); 46.3 → 4025.0 (9); **51.2 min → 4398.7 kg (10 ships)**;
the 2 h budget was not needed because the fleet rule, not time, stops the fleet: with an
average of 439.9 kg per ship `N ≤ 2 exp(0.004 M̄) = 11.6`, so an 11th ship is the last admissible
one at this per-ship mass (the references' 740 kg average is what allows their 36–39 ships).
Peak RSS stayed at 0.76–0.78 GB throughout (bound: 2 GB). Every ship
costs ~5–6 min: 2–3 min beam search, 0.5–1 min for up to three SCvx refinements, 1–3 min of
re-timing/extension with SCvx re-certification (each attempt re-flies 14–20 legs).

**Third campaign (cooperative cluster pricing, §6.5).** `cluster_fleet_v1` priced 108 of the
132 co-moving families of the a 2.2–3.0 AU / e ≤ 0.15 / i ≤ 8° box in 4 worker processes over
its 4 h budget (`--workers 4 --max-clusters 132 --budget-seconds 14400`, 3 ships per family,
beam 24, ≤ 10 deploys, re-timing 4 attempts / 600 s, orphan credit 1.0, seed 0): 2141 Earth legs
flown by SCvx (1013 certified), 150 certified ships in 70 families (55 multi-ship bundles), 44
cooperative collects, 11 orphans left after repair, 162 repairs and 508 rejected variants in the
log. Score versus wall clock (verified incumbent fleets, all retained under `fleets/`):
10.6 min → 948.3 kg (3 ships); 14.0 → 2092.3 (6); 18.8 → 3389.7 (9); **30 min → 4692.8 kg (11
ships, 77 asteroids)**; 43.8 → 5382.7 (12); **1 h → 5382.7 kg (12)**; 82.4 → 6229.0 (13);
**2 h → 6357.9 kg (13 ships, 93 asteroids)**; 127.4 → 6874.9 (14); **4 h → 6932.5 kg (14 ships,
99 asteroids, average 495.2 kg)**. Memory: main process 0.43 GB peak, worker peaks 0.25–0.60 GB
(sampled concurrent total 1.2–1.5 GB; the conservative sum-of-peaks bound in the report, main +
4 × worst worker, is 2.8 GB).

Two failure modes cost whole families during the run and were fixed afterwards: an
inconsistent bundle (a collector's foreign epoch went stale when its deployer was re-timed, or a
miner nobody deploys any more) discarded *all* ships of families 19, 138 and 371, and a
`build_visits` `ValueError` crashed the worker of family 66 (`make_consistent` now drops single
ships; `retime_order` fails softly). `cluster_fleet_v3_repair` re-priced those and the three
families the deeper re-pricing had lost (7, 3, 25) with 4 ships per family and 5 re-timing
attempts: 23 ships, 4704.15 kg verified in 36 min. `cluster_fleet_v2_deep` re-priced the six
richest families with 5 slots and 6 attempts / 900 s (7 ships, 2878.0 kg in 46 min): the extra
slots rarely close — after three ships a family's cheap Earth legs and co-moving members are
spent ("beam found no closing chain" for slots 4–5 of families 0 and 1).

The in-run master was the other loss: with the suffix-sum bound it exceeded its 200k-node cap
from the 7th family on and returned the greedy fleet, which is not monotone in the column set
(families 25→31: 12 ships / 5383 kg → 11 / 5148; 49: 12 / 5801 after 14 / 6875; 79: 13 / 6389).
`fleet_master_v1` re-flew all 208 archived routes of the four runs through SCvx (405 s in 6
workers, **0 re-certification failures** — the ZOH SCvx replay is deterministic), built 273
columns (190 primary ships, 18 stand-alone variants, 65 bundle columns) and solved the master
with the ship-rule bound to 5 M nodes (67 s; not exhaustive — 30 M nodes moved the fixed-bonus
objective from 7217.5 to 7219.4 kg at lower collected mass, so 5 M is the reported setting):
**15 ships, 109 asteroids visited (103 mined), 7575.58 kg collected (official `GTOC12_Verify`
"Check successfully!"; independent verifier agrees, max propagation error 56 km), fixed-bonus score
7217.51 kg, average 505.0 kg per ship, fleet rule 15 ≤ 15.08**. The rule is exactly binding: a
16th ship needs the average above 520 kg.

Per-ship table of `fleet_master_v1` (final mass = dry + propellant at Earth return; the 500 kg
floor shows how little propellant the good ships have left):

| # | Source (family / run) | Slot | Asteroids | Collected kg | Final mass kg | Arcs | Launch MJD | Foreign collects | Orphans left |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `fleet10_master_v1` | 1 | 8 | 583.2 | 574.4 | 16 | 64403 | – | – |
| 2 | `cluster_fleet_v1/family_0024` | 2 | 8 | 541.3 | 582.3 | 16 | 64358 | – | – |
| 3 | `cluster_fleet_v1/family_0024` | 3 | 7 | 539.6 | 532.1 | 14 | 64358 | – | – |
| 4 | `cluster_fleet_v2_deep/family_0015` | 3 | 8 | 530.6 | 525.8 | 16 | 64403 | – | 8634 |
| 5 | `cluster_fleet_v1/family_0033` | 2 | 8 | 528.5 | 541.7 | 16 | 64403 | – | 46294 |
| 6 | `cluster_fleet_v1/family_0025` | 1 | 9 | 516.6 | 571.8 | 18 | 64373 | – | 13637 |
| 7 | `cluster_fleet_v1/family_0065` | 2 | 7 | 507.8 | 750.0 | 14 | 64328 | – | – |
| 8 | `cluster_fleet_v1/family_0000` | 1 | 9 | 502.3 | 536.1 | 17 | 64388 | – | 27306, 30267 |
| 9 | `cluster_fleet_v1/family_0065` | 1 | 7 | 501.4 | 511.3 | 14 | 64328 | – | – |
| 10 | `fleet10_master_v1` | 2 | 6 | 490.3 | 584.1 | 12 | 64508 | – | – |
| 11 | `cluster_fleet_v1/family_0001` | 1 | 6 | 486.2 | 630.5 | 12 | 64508 | – | – |
| 12 | `cluster_fleet_v1/family_0065` | 3 | 6 | 469.0 | 592.5 | 12 | 64328 | – | – |
| 13 | `cluster_fleet_v1/family_0014` | 1 | 7 | 464.9 | 516.2 | 14 | 64508 | – | 15535 |
| 14 | `cluster_fleet_v1/family_0328` | 3 | 7 | 461.6 | 586.4 | 14 | 64538 | – | – |
| 15 | `fleet10_master_v1` | 5 | 6 | 452.2 | 634.9 | 12 | 64508 | – | – |

Cooperative statistics of the emitted fleet: 0 foreign collects, 6 of the 109 deployed miners
left uncollected (103 asteroids mined; 240 kg of miners wasted), two multi-ship bundle columns (family 65: 3 ships, family
24: 2 ships) that are self-cleaning ships selected together. Across the priced families the
pricing *did* produce cooperation — 54 foreign collects, 32 deployers with collectors, 19 orphans
left in the 81 archived groups — but the exact master never picked a cooperative pair: a
collector that flies to another ship's miner years later collects 280–330 kg (families 293,
252, 22: 288/329/433 kg) against 450–540 kg for a self-cleaning ship in the same family, so the
pair lowers the fleet average and the rule takes a ship away. The 15 ships launch between MJD
64328 and 64538 (the first 210 days of the window), fly 12–18 arcs each (217 in total), spend
371–618 kg on the Earth leg and return at MJD 69518–69803 (the last 289 days of the window).

**Fourth campaign (per-ship mass levers, §6.6).** `cluster_fleet_v4` priced 47 phasing-aware
families (radius 2.0, deploy + collect epochs in the feature vector) in 4 worker processes over
its 4 h budget (`--workers 4 --ships-per-family 4 --budget-seconds 14400 --collector-harvest
--earth-leg-refinements 8`, beam 24, ≤ 10 deploys, joint harvest on, seed 0; families are
priced until the budget is spent and the last four run to completion, 252.5 min total): 1322
Earth legs flown by SCvx on the grid (533 certified), 324 of them re-optimised continuously (477 →
390 kg median), 119 certified ships in 40 multi-ship families, 67 cooperative collects, 37
deployers with collectors, 22 orphans left after 190 repairs, 226 rejected variants in the log,
38 joint harvests attempted / 0 adopted. Score versus wall clock (verified incumbent fleets, all
retained under `fleets/`): 16.5 min → 884.6 kg (2 ships); **30 min → 884.6 kg (2 ships; the 5-ship
fleet certifies at 34.1 min → 2358.5)**; 38.7 → 3737.0 (8); 49.6 → 4697.8 (11); **1 h → 4841.7 kg
(11 ships, 74 asteroids)**; 67.6 → 5523.2 (12); 96.2 → 6085.4 (13); **2 h → 6167.1 kg (13 ships,
96 asteroids)**; 169.8 → 6817.6 (14); 220.2 → 6926.9; **4 h → 6926.9 kg (14 ships, 103 asteroids;
final fleet at 240.2 min: 6975.7 kg, 14 ships, 100 deployed / 96 mined, average 498.3 kg)**. It
is slower to start than `cluster_fleet_v1` (4 ships per family and 8 SCvx Earth-leg refinements
per certified grid leg make a family 9–40 min instead of 3–28) and lands at the same fleet
(6975.7 vs 6932.5 kg): the per-ship average did not move (498 vs 495 kg). Memory: main 0.46 GB
peak, worker peaks up to 0.81 GB (the conservative sum-of-peaks bound in the report, main + 4 ×
worst worker, is 3.7 GB; the workers' peaks are not simultaneous).

`fleet_master_v2` re-flew all 330 archived routes of the six runs (`cluster_fleet_v1/v2_deep/
v3_repair/v4`, `fleet10_master_v1`, `probe_v4_family`) through SCvx (749 s in 6 workers, **0
re-certification failures**), built 436 columns and solved the master with the ship-rule bound
to 5 M nodes (103 s, not exhaustive): **16 ships, 123 asteroids visited (116 mined), 8324.27 kg
collected (official `GTOC12_Verify` "Check successfully!"; independent verifier agrees, max
propagation error 56 km), fixed-bonus score 7905.05 kg, average 520.3 kg per ship, fleet rule
16 ≤ 16.03**. The rule is again exactly binding: a 17th ship needs the average above 535 kg.
Ships from the fourth campaign contribute 5 of the 16 (families 430, 355, 17, 4, 5 — three of
them at 524–564 kg, the campaign's best), `cluster_fleet_v1` 8, `fleet10_master_v1` 2,
`cluster_fleet_v3_repair` 1. **One cooperative pair enters the incumbent** (ships 10 and 15,
family 0 of `cluster_fleet_v1`: ship 15 collects 27306 and 30267 deployed by ship 10 — 2 foreign
collects, 0 multi-ship bundle columns); the v4 collectors that harvest other ships' miners
(e.g. family 275 slot 3 with 4 foreign collects, 386 kg; family 410 slot 2, 3 foreign, 456 kg)
were not selected because the same families' self-cleaning routes collect more per ship. The
launch spread is used: ship 5 launches at MJD 64763 (435 days into the window).

Per-ship table of `fleet_master_v2` (final mass = dry + propellant at Earth return):

| # | Source (family / run) | Slot | Asteroids | Collected kg | Final mass kg | Arcs | Launch MJD | Foreign collects | Orphans left |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `fleet10_master_v1` | 1 | 8 | 583.2 | 574.4 | 16 | 64403 | – | – |
| 2 | `cluster_fleet_v4/family_0430` | 1 | 8 | 564.3 | 522.5 | 16 | 64373 | – | 39189 |
| 3 | `cluster_fleet_v1/family_0024` | 2 | 8 | 541.3 | 582.3 | 16 | 64358 | – | – |
| 4 | `cluster_fleet_v1/family_0024` | 3 | 7 | 539.6 | 532.1 | 14 | 64358 | – | – |
| 5 | `cluster_fleet_v4/family_0355` | 1 | 9 | 537.6 | 595.7 | 18 | 64763 | – | 39692 |
| 6 | `cluster_fleet_v1/family_0033` | 2 | 8 | 528.5 | 541.7 | 16 | 64403 | – | 46294 |
| 7 | `cluster_fleet_v4/family_0017` | 1 | 7 | 524.0 | 644.4 | 14 | 64388 | – | – |
| 8 | `cluster_fleet_v1/family_0025` | 1 | 9 | 516.6 | 571.8 | 18 | 64373 | – | 13637 |
| 9 | `cluster_fleet_v1/family_0065` | 2 | 7 | 507.8 | 750.0 | 14 | 64328 | – | – |
| 10 | `cluster_fleet_v1/family_0000` | 1 | 9 | 502.3 | 536.1 | 17 | 64388 | – | 27306, 30267 (collected by #15) |
| 11 | `cluster_fleet_v1/family_0065` | 1 | 7 | 501.4 | 511.3 | 14 | 64328 | – | – |
| 12 | `cluster_fleet_v4/family_0004` | 3 | 7 | 499.4 | 520.5 | 14 | 64448 | – | – |
| 13 | `cluster_fleet_v3_repair/family_0007` | 2 | 8 | 498.2 | 538.1 | 16 | 64478 | – | 28289 |
| 14 | `cluster_fleet_v4/family_0005` | 2 | 9 | 494.9 | 596.5 | 17 | 64568 | – | 21437, 47969 |
| 15 | `cluster_fleet_v1/family_0000` | 3 | 8 | 494.9 | 575.4 | 14 | 64388 | 27306, 30267 | – |
| 16 | `fleet10_master_v1` | 2 | 6 | 490.3 | 584.1 | 12 | 64508 | – | – |

Cooperative statistics of the emitted fleet: 2 foreign collects (one collector, one deployer), 7
of the 123 deployed miners left uncollected (116 mined), 246 arcs. The master's 420 rejected
columns are all "dominated or incompatible with the incumbent" — among them every multi-ship
bundle column and every collector with more than two foreign collects.

**Fifth campaign (§6.7).** Two 4 h campaigns ran side by side on the 16 cores (7 workers in
total): `cluster_fleet_v5` (the v4 configuration — radius 2.0 phasing-aware families, continuous
SCvx Earth legs, 4 ships per family, joint harvest — plus the exact collect-tour DP in the beam,
4 workers) and `cluster_fleet_v5c` (the same with collect-epoch families, 3 workers). Verified
intermediate fleets (score at the budget marks, all officially verified): `v5` 30 min 2551.9 kg
(5 ships; the first family closed at 17.9 min with 986.0 kg), 1 h 7695.3 (15), 2 h 8365.9 (16),
4 h 9101.9 (17); `v5c` 30 min — (first family at 30.1 min, 1600.4 kg / 3 ships), 1 h 5569.1
(11), 2 h 8358.0 (16), 4 h 9111.3 (17; its final master fleet is 9100.9 kg with the better
fixed-bonus objective). For comparison the fourth campaign stood at 884.6 / 4841.7 / 6167.1 /
6926.9 kg. Each campaign's own master closed with the LP branch and bound (`v5`: 275 columns,
objective 8631.3, LP bound 8659.8, proven in 81 LPs; `v5c`: 210 columns, 8544.1 / 8547.3, 9 LPs)
where the 200 k-node combinatorial search had not been exhaustive. Per-family telemetry: 38 / 27
families, median 49 / 53 members, 1750 / 1824 s per family, 128 / 94 certified ships (median
455 / 477 kg, best 570.8 / 620.1 kg), 207 / 148 rejected variants retained in the bundle reports;
memory main 0.45 / 0.44 GB, worker peak 0.83 / 0.77 GB (sampled concurrent total 3.8 / 2.7 GB
for 4 / 3 workers).

`fleet_master_v3` re-flew all 554 archived routes of the nine runs (`cluster_fleet_v1/v2_deep/
v3_repair/v4/v5/v5c`, `fleet10_master_v1`, `probe_v4_family`, `probe_v5_family247`) through SCvx
(921 s in 8 workers, **0 re-certification failures**), built 726 columns and solved the master:
the combinatorial search stopped at the 5 M node cap (8785.2 kg greedy start), the LP branch and
bound then searched fleet sizes 18 and 17 (565 LPs, 136 s in total) and **proved** the incumbent
optimal over the archive — fixed-bonus objective 9329.82 kg against an LP bound of 9334.32 kg
(gap 4.5 kg, closed by integrality); the per-size LP relaxations are 8962.0 kg at 17 ships,
9334.3 at 18 and *infeasible* at 19 (the mass floor 19 ln 9.5 / 0.004 = 10 693 kg exceeds what
any 19 compatible columns collect), so no 19-ship fleet exists in this archive. Result: **18
ships, 147 asteroids visited (142 mined), 9888.57 kg collected (official `GTOC12_Verify` "Check
successfully!"; independent verifier agrees to 1e-10 kg per asteroid, max propagation error
113 km), fixed-bonus score 9329.82 kg, average 549.4 kg per ship, fleet rule 18 ≤ 18.004**. The
rule is exactly binding for the third time; a 19th ship needs the average above 562.8 kg. Sources:
`cluster_fleet_v5c` 8 ships, `cluster_fleet_v5` 5, `probe_v5_family247` 1 (14 of 18 from this
iteration), `cluster_fleet_v4` 2, `cluster_fleet_v1` 1, `fleet10_master_v1` 1. No cooperative
column is selected (0 foreign collects; the two adopted v5 harvests — families 459 and 280 — are
below the fleet average).

Per-ship table of `fleet_master_v3` (final mass = dry + propellant at Earth return):

| # | Source (family / run) | Slot | Asteroids | Collected kg | Final mass kg | Arcs | Launch MJD | Foreign collects | Orphans left |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `fleet10_master_v1` | 1 | 8 | 583.2 | 574.4 | 16 | 64403 | – | – |
| 2 | `cluster_fleet_v4/family_0430` | 1 | 8 | 564.3 | 522.5 | 16 | 64373 | – | 39189 |
| 3 | `cluster_fleet_v5c/family_0000` | 2 | 9 | 563.0 | 548.9 | 19 | 64493 | – | – |
| 4 | `cluster_fleet_v5c/family_0026` | 3 | 8 | 559.8 | 507.2 | 17 | 64373 | – | – |
| 5 | `cluster_fleet_v5c/family_0001` | 3 | 8 | 559.3 | 591.3 | 16 | 64403 | – | – |
| 6 | `cluster_fleet_v5c/family_0005` | 3 | 9 | 558.9 | 517.5 | 18 | 64388 | – | – |
| 7 | `cluster_fleet_v5/family_0004` | 3 | 8 | 558.1 | 563.9 | 16 | 64448 | – | – |
| 8 | `cluster_fleet_v5/family_0009` | 1 | 8 | 558.1 | 583.9 | 16 | 64433 | – | – |
| 9 | `cluster_fleet_v5/family_0000` | 4 | 7 | 557.3 | 500.4 | 14 | 64493 | – | – |
| 10 | `cluster_fleet_v5/family_0017` | 1 | 8 | 549.1 | 514.1 | 16 | 64388 | – | – |
| 11 | `probe_v5_family247/family_0247` | 3 | 7 | 547.0 | 536.6 | 14 | 64568 | – | – |
| 12 | `cluster_fleet_v1/family_0024` | 2 | 8 | 541.3 | 582.3 | 16 | 64358 | – | – |
| 13 | `cluster_fleet_v4/family_0355` | 1 | 9 | 537.6 | 595.7 | 18 | 64763 | – | 39692 |
| 14 | `cluster_fleet_v5c/family_0031` | 1 | 8 | 535.8 | 523.7 | 17 | 64328 | – | – |
| 15 | `cluster_fleet_v5/family_0009` | 2 | 8 | 534.3 | 503.4 | 17 | 64433 | – | – |
| 16 | `cluster_fleet_v5c/family_0064` | 3 | 9 | 534.3 | 534.4 | 18 | 64358 | – | 2009 |
| 17 | `cluster_fleet_v5c/family_0429` | 2 | 9 | 528.1 | 561.3 | 17 | 64568 | – | 37801, 41409 |
| 18 | `cluster_fleet_v5c/family_0021` | 2 | 8 | 519.1 | 505.4 | 17 | 64568 | – | – |

Leg-cost distribution of the best fleet before and after this iteration
(`results/gtoc12/leg_stats/after_v5.json`; propellant per certified leg, kg):

| Fleet | Collect hops n | mean | p25 | median | p75 | p90 | share ≤ 75 kg | Deploy hop median | Earth-out mean | Earth-return mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fleet_master_v2` (before, 16 ships) | 116 | 95.3 | 71.2 | 90.2 | 122.7 | 152.5 | 0.233 | 109.6 | 467.6 | 196.8 |
| `cluster_fleet_v5` (17) | 137 | 90.6 | 65.8 | 89.3 | 111.0 | 137.5 | 0.276 | 101.5 | 407.8 | 244.8 |
| `cluster_fleet_v5c` (17) | 136 | 89.3 | 67.2 | 91.5 | 111.8 | 136.8 | 0.309 | 95.5 | 420.1 | 247.7 |
| **`fleet_master_v3` (after, 18)** | 142 | 89.7 | 63.9 | 87.1 | 115.3 | 140.0 | 0.292 | 98.9 | 411.8 | 226.5 |
| Antipodes 37 (self-cleaning) | 338 | 65.9 | 48.2 | 66.1 | 84.9 | 102.2 | 0.459 | 97.1 | 473.5 | 208.0 |
| Antipodes 39 | 356 | 66.9 | 51.4 | 66.0 | 82.4 | 103.0 | 0.459 | 96.3 | 472.3 | 214.4 |
| JPL 36 | 320 | 69.2 | 52.5 | 67.1 | 82.7 | 101.1 | 0.443 | 101.2 | 473.2 | 215.9 |

The collect-hop TOF median is 240 d in our fleets against 181–187 d in the references: the
remaining 21 kg per hop is phase, not order — the references chain members that are closer at
the harvest epoch, so their hops are both shorter and cheaper.

**Sixth campaign (§6.8).** `cluster_fleet_v6` ran 4 h on 4 workers (5 ships per family,
radius-1.75 collect-window families with ≥ 20 members, calibrated DP hop costs, two-pass mass
schedule, 15-day DP lattice, 30-day return grid, Earth-leg prescreen at 0.7; GPU locked by the
G4 campaign, CPU only). Verified intermediate fleets at the budget marks (all officially
verified): 30 min — (the first family closed at 33.3 min with 5 ships / 2665.6 kg, families now
take 818–2630 s, median 1480 s, because each prices 5 slots), **1 h 8545.1 kg (16 ships), 2 h
9887.8 (18; the fifth campaign's 4 h + archive-wide master result), 4 h 10 697.1 (19)**; the final
master over the campaign's own 262 columns is **10 698.0 kg / 19 ships / 159 asteroids (155
mined) / 563.05 kg per ship**, exhaustive in 889 nodes with the LP root bound 9796.8 kg over the
9785.0 kg fixed-bonus objective (closed by branching). The 19th ship was admitted at 170 min
(10 693.8 kg). Per-family telemetry: 36 families priced (median 30.5 members), 136 certified
ships (median 478.6 kg; best 603.7 / 603.3 / 598.4 / 589.3 / 582.8 / 578.6 kg — **9 ships above
the 563 kg threshold**, 23 at or above 535 kg, where the whole pre-v6 archive had one ship above
563), 579 Earth-leg SCvx checks for 600 certified legs (the continuous refinement certifies more
than it checks), 328 rejected variants and 24 repairs retained in the bundle reports; the joint
harvest ran 89 rounds / 138 re-timings / 86 certified re-timed tours and again adopted nothing
into the emitted fleet (no foreign collect in any selected column). Memory: main process 0.43 GB
RSS, worker peak 1.11 GB RSS, **process-tree PSS peak 3.04 GB** for 4 workers (the 2 GB target
is missed; the worker baseline after imports, catalogue, families and the pair table is 110 MB,
and a tracemalloc probe of one slot's beam shows < 80 MB of Python-owned arrays with the RSS at
130–340 MB, so the transient lives in the native side — Lambert batches and SCvx/Clarabel — of
the later slots; see §8).

`fleet_master_v4` re-flew all 695 archived routes of the eleven runs (the nine of `v3` plus
`cluster_fleet_v6` and `probe_v6_family`) through SCvx (1020 s in 8 workers, **0
re-certification failures**), built 903 columns and solved the master: the combinatorial search
stopped at its 2 M node cap (9769.3 kg greedy start), the LP branch and bound (635 LPs, 1.0 s)
then **proved** the incumbent optimal over the archive — fixed-bonus objective 10 146.60 kg
against an LP bound of 10 159.66 kg (gap 13.1 kg, 0.13 %, closed by integrality). Result: **19
ships, 158 asteroids visited (154 mined), 10 700.48 kg collected (official `GTOC12_Verify`
"Check successfully!"; independent verifier agrees to 1e-10 kg per asteroid, max propagation
error 121 km), fixed-bonus score 10 146.60 kg, average 563.18 kg per ship, fleet rule 19 ≤
19.027**. The archive-wide master adds only 2.5 kg to the campaign's own fleet: 10 of the 19
ships come from `cluster_fleet_v6`, 1 from `probe_v6_family`, 3 from `cluster_fleet_v5`, 2 from
`v5c`, 2 from `v4`, 1 from `fleet6_coop_v1`. A 20th ship needs the average above 575.6 kg (20 ln
10 / 0.004 = 11 513 kg over 20 compatible columns); the archive holds 9 columns above that.

Per-ship table of `fleet_master_v4` (final mass = dry + propellant at Earth return; arcs are the
SCvx-refined legs):

| # | Source (run / family / slot) | Slot | Asteroids | Collected kg | Final mass kg | Arcs | Launch MJD | Foreign collects | Orphans left |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `cluster_fleet_v6/family_0054` | 1 | 9 | 603.7 | 523.8 | 18 | 64373 | – | – |
| 2 | `cluster_fleet_v6/family_0006` | 1 | 9 | 603.3 | 507.4 | 16 | 64478 | – | – |
| 3 | `cluster_fleet_v6/family_0025` | 5 | 9 | 589.3 | 541.2 | 16 | 64358 | – | – |
| 4 | `fleet10_master_v1` (`fleet6_coop_v1/ship_01`) | 1 | 8 | 583.2 | 574.4 | 16 | 64403 | – | – |
| 5 | `probe_v6_family/family_0001` | 1 | 9 | 582.8 | 551.4 | 16 | 64463 | – | – |
| 6 | `cluster_fleet_v6/family_0007` | 2 | 9 | 573.7 | 527.0 | 17 | 64448 | – | 44233 |
| 7 | `cluster_fleet_v4/family_0430` | 1 | 8 | 564.3 | 522.5 | 18 | 64373 | – | 39189 |
| 8 | `cluster_fleet_v6/family_0065` | 1 | 9 | 563.4 | 563.0 | 18 | 64343 | – | 50390 |
| 9 | `cluster_fleet_v5c/family_0026` | 3 | 8 | 559.8 | 507.2 | 17 | 64373 | – | – |
| 10 | `cluster_fleet_v5c/family_0001` | 3 | 8 | 559.3 | 591.3 | 17 | 64403 | – | – |
| 11 | `cluster_fleet_v6/family_0025` | 1 | 8 | 558.9 | 562.7 | 17 | 64358 | – | – |
| 12 | `cluster_fleet_v6/family_0008` | 2 | 8 | 558.2 | 534.5 | 17 | 64478 | – | – |
| 13 | `cluster_fleet_v5/family_0009` | 1 | 8 | 558.1 | 583.9 | 19 | 64433 | – | – |
| 14 | `cluster_fleet_v6/family_0035` | 1 | 8 | 550.0 | 507.0 | 16 | 64328 | – | – |
| 15 | `cluster_fleet_v5/family_0017` | 1 | 8 | 549.1 | 514.1 | 19 | 64388 | – | – |
| 16 | `probe_v5_family247/family_0247` | 3 | 7 | 547.0 | 536.6 | 18 | 64568 | – | – |
| 17 | `cluster_fleet_v4/family_0355` | 1 | 9 | 537.6 | 595.7 | 16 | 64763 | – | 39692 |
| 18 | `cluster_fleet_v6/family_0009` | 2 | 8 | 530.6 | 561.6 | 14 | 64553 | – | – |
| 19 | `cluster_fleet_v6/family_0018` | 1 | 8 | 528.1 | 516.1 | 19 | 64718 | – | – |

Leg-cost distribution of the best fleet before and after the sixth iteration
(`results/gtoc12/leg_stats/after_v6.json`; propellant per certified leg, kg):

| Fleet | Collect hops n | mean | p25 | median | p75 | p90 | share ≤ 75 kg | collect TOF median (p25–p75) d | Deploy hop median | Earth-out mean | Earth-return mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fleet_master_v3` (before, 18 ships) | 142 | 89.7 | 63.9 | 87.1 | 115.3 | 140.0 | 0.292 | 240 (195–330) | 98.9 | 411.8 | 226.5 |
| **`cluster_fleet_v6` (after, 19)** | 155 | 88.0 | 66.5 | 84.4 | 104.1 | 133.9 | 0.342 | 240 (180–300) | 91.3 | 414.6 | 279.3 |
| Antipodes 37 / 39, JPL 36 | 338 / 356 / 320 | 65.9–69.2 | 48.2–52.5 | 66.0–67.1 | 82.4–84.9 | 101.1–103.0 | 0.443–0.459 | 181–187 (137–141 to 234–257) | 96.3–101.2 | 472.3–473.5 | 208.0–215.9 |

The calibrated, two-pass DP moved the collect hop from 87.1 to 84.4 kg median (p75 115 → 104,
share ≤ 75 kg 0.29 → 0.34) and the deploy hop to 91.3 kg (below the references' 96–101), and the
ships carry 8.3 asteroids on average (8.1); the per-ship hop propellant fell from 1492 to
1431 kg. The target of ≤ 70 kg / ~180 d was not reached — the TOF median is still 240 d and the
p25 has not moved — and the Earth return got dearer again (279 vs 227 kg mean; references
208–216): the return TOF median fell to 420 d (v3 435, references 473–486) while its propellant
rose, i.e. with the 240–720 d grid the DP trades a cheap return for a later last collect, and
the fleet pays ~55 kg per ship for it — most of what the hops saved. That is the first item of
the next list.

**Seventh iteration (§6.9): the return sweep admits the 20th ship.** `return_sweep_v1` swept
and re-timed the 36 best stand-alone archived ships (3 workers, 50 min, worker peak 221 MB RSS):
13 improved, +140.5 kg in total, the improved returns now 172–247 kg (the 36 ships' return
median 269 → 236 kg); the largest gain is `cluster_fleet_v6/family_0001` ship 2, 598.4 → 633.3
kg with its return 410 → 212 kg. Gains are small per ship because these ships are
propellant-bound at the return: saved return propellant only turns into collected mass where the
re-timer can extend a stay or a hop. `fleet_master_v5` over the eleven archives plus these 13
routes (690 routes re-flown through SCvx in 2461 s on 3 workers, **0 failures**, 918 columns, 2 M
DFS nodes then LP branch and bound, 379 LPs in 1.0 s): **20 ships, 168 asteroids (165 mined),
11 521.42 kg, 576.07 kg average** (the rule needs ≥ 575.6 for 20), fixed-bonus objective
10 573.17 kg vs LP bound 10 589.13 (gap 16.0 kg, **proven optimal over the archive**; the LP
at 21 ships is infeasible). GTOC12_Verify "Check successfully!", independent verifier ok (121 km
max propagation error). Eight of the twenty ships are re-timed `return_sweep_v1` routes; the
selected fleet has one two-ship bundle column (the two re-timed ships of v6 family 5 archived
under one group directory; both stand-alone). Per ship (rank, source, slot, asteroids,
collected kg, final mass kg, refined arcs, launch MJD):

| # | Source | Slot | Ast. | Collected | Final mass | Arcs | Launch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `return_sweep_v1` ← `cluster_fleet_v6/family_0001` | 2 | 8 | 633.3 | 508.0 | 17 | 64478 |
| 2 | `cluster_fleet_v6/family_0054` | 1 | 9 | 603.7 | 523.8 | 19 | 64373 |
| 3 | `cluster_fleet_v6/family_0006` | 1 | 9 | 603.3 | 507.4 | 18 | 64478 |
| 4 | `return_sweep_v1` ← `cluster_fleet_v6/family_0005` | 2 | 9 | 590.1 | 518.0 | 19 | 64478 |
| 5 | `cluster_fleet_v6/family_0025` | 5 | 9 | 589.3 | 541.2 | 19 | 64358 |
| 6 | `return_sweep_v1` ← `cluster_fleet_v6/family_0005` | 3 | 9 | 584.4 | 513.8 | 18 | 64478 |
| 7 | `fleet10_master_v1` | 1 | 8 | 583.2 | 574.4 | 16 | 64403 |
| 8 | `probe_v6_family/family_0001` | 1 | 9 | 582.8 | 551.4 | 19 | 64463 |
| 9 | `return_sweep_v1` ← `cluster_fleet_v5/family_0004` | 3 | 8 | 575.8 | 517.7 | 16 | 64448 |
| 10 | `cluster_fleet_v6/family_0007` | 2 | 9 | 573.7 | 527.0 | 18 | 64448 |
| 11 | `return_sweep_v1` ← `cluster_fleet_v5c/family_0001` | 3 | 8 | 568.0 | 584.0 | 16 | 64403 |
| 12 | `return_sweep_v1` ← `cluster_fleet_v6/family_0025` | 1 | 8 | 564.7 | 564.1 | 17 | 64358 |
| 13 | `cluster_fleet_v4/family_0430` | 1 | 8 | 564.3 | 522.5 | 16 | 64373 |
| 14 | `cluster_fleet_v6/family_0065` | 1 | 9 | 563.4 | 563.0 | 18 | 64343 |
| 15 | `return_sweep_v1` ← `cluster_fleet_v5/family_0247` | 3 | 8 | 562.6 | 536.5 | 16 | 64568 |
| 16 | `cluster_fleet_v5c/family_0026` | 3 | 8 | 559.8 | 507.2 | 17 | 64373 |
| 17 | `cluster_fleet_v6/family_0008` | 2 | 8 | 558.2 | 534.5 | 17 | 64478 |
| 18 | `cluster_fleet_v5/family_0009` | 1 | 8 | 558.1 | 583.9 | 16 | 64433 |
| 19 | `return_sweep_v1` ← `cluster_fleet_v5/family_0017` | 1 | 8 | 552.8 | 513.8 | 16 | 64388 |
| 20 | `cluster_fleet_v6/family_0035` | 1 | 8 | 550.0 | 507.0 | 16 | 64328 |

Leg-cost distribution of the best fleet before and after the seventh iteration
(`results/gtoc12/leg_stats/after_v7.json`):

| Fleet | Collect hops n | mean | p25 | median | p75 | p90 | share ≤ 75 kg | collect TOF median (p25–p75) d | Deploy hop mean | Earth-out mean | Earth-return mean / median | return TOF median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fleet_master_v4` (before, 19 ships) | 154 | 88.1 | 63.5 | 83.8 | 108.9 | 139.7 | 0.341 | 240 (180–300) | 757.5 per ship | 407.3 | 245.0 / 259.2 | 420 |
| **`fleet_master_v5` (after, 20)** | 165 | 88.2 | 64.6 | 85.6 | 109.0 | 139.9 | 0.332 | 225 (180–300) | 755.9 per ship | 418.1 | 227.7 / 229.2 | 435 |
| Antipodes 37 / 39, JPL 36 | 338 / 356 / 320 | 65.9–69.2 | 48.2–52.5 | 66.0–67.1 | 82.4–84.9 | 101.1–103.0 | 0.443–0.459 | 181–187 | 837–851 per ship | 460.1–473.5 | 208.0–215.9 / 206.1–214.2 | 473–486 |

The Earth return moved 245 → 228 kg mean (259 → 229 median) against the references' 208–216;
the eight re-timed ships return at 172–247 kg. The collect hop did not move (85.6 kg / 225 d
median vs 66 kg / 181–187 d): the harvest-window deploy ranking is a negative result (§6.9), so
the collect-hop phase remains the next bottleneck and has to be attacked inside the DP's window
(pair choice at the harvest epochs, not at deploy time). Return TOF is still 435 d median versus
the references' 473–486: the DP still prefers a later last collect to a longer return; only the
SCvx sweep moves it.

**Seventh campaign and the fourteen-archive master.** `cluster_fleet_v7` (3 workers, `nice 19`,
4 h, radius-1.6 / ≥ 18-member families — a new partition so it does not replay v6, return TOF
model in the DP, bounded DP caches) priced 25 families / 76 ships in 14 950 s; its own master
fleet is 9920.47 kg / 18 ships / 551.3 kg average (marks 60 min 5355.8 / 10 ships, 120 min
8571.1 / 16, 240 min 9922.5 / 18) — below v6's 10 698 kg at 4 workers because 3 workers price
0.7× the families and this partition's families are larger (up to 35 members; 797–3046 s each
against v6's 818–2630 s) — but it produced the two richest new columns of the archive (family 11
ship 1 611.9 kg, family 7 ship 1 607.8 → 616.4 kg after the return sweep) and its DP now ends
tours earlier: the return TOF median is 480 d (v6 420, references 473–486) at 232.7 kg mean.
**Process-tree PSS peak 1.19 GB** (v6: 3.04 GB at 4 workers), main 0.43 GB, worker `ru_maxrss`
peak 0.68 GB — the per-phase marks show the remaining live growth is per-slot state parked for
the orphan repair (RSS after trim 70 → 184 → 324 → 437 → 535 MB over four slots of a
35-member family: each slot's DP pair table and Lambert memo), released after the run in
`RouteSearch.release_caches` / `Retimer.release_caches`. `return_sweep_v2` re-timed 13 of v7's
21 best ships (+225.5 kg; family 7 ship 3 528.3 → 602.9 kg). `fleet_master_v6` over all
fourteen archives (779 routes re-flown, 0 failures, 1032 columns — the column DFS overflowed the
interpreter's 1000-frame recursion limit at this size and now raises it for the search) is
**proven optimal: fixed-bonus objective 10 739.27 kg vs LP bound 10 744.85 (gap 5.6 kg), 20
ships, 168 asteroids (164 mined), 11 515.67 kg collected, 575.78 kg average** (rule 20 ≤ 20.01),
both verifiers ok. It collects 5.8 kg *less* raw mass than `v5` (11 521.42) while scoring 166 kg
more on the bonus-weighted objective the master optimises — the two fleets share 13 ships; v6
swaps in the v7 columns (611.9, 616.4, 567.0, 564.3, 542.9 kg) and two `v4` ships for five of
v5's. Its Earth return is **216.5 kg mean / 216.7 median (references 208–216; return TOF median
450 d)** — the ≤ 216 kg target of the work list, met to within 0.5 kg — and the collect hop is
unchanged (84.5 kg / 225 d median; 88.4 mean, p75 111, share ≤ 75 kg 0.32). Per ship (rank,
source, slot, asteroids, collected kg, final mass kg, refined arcs, launch MJD):

| # | Source | Slot | Ast. | Collected | Final mass | Arcs | Launch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `return_sweep_v1` ← `cluster_fleet_v6/family_0001` | 2 | 8 | 633.3 | 508.0 | 17 | 64478 |
| 2 | `return_sweep_v2` ← `cluster_fleet_v7/family_0007` | 1 | 9 | 616.4 | 539.9 | 18 | 64463 |
| 3 | `cluster_fleet_v7/family_0011` | 1 | 9 | 611.9 | 538.7 | 18 | 64478 |
| 4 | `cluster_fleet_v6/family_0054` | 1 | 9 | 603.7 | 523.8 | 19 | 64373 |
| 5 | `cluster_fleet_v6/family_0025` | 5 | 9 | 589.3 | 541.2 | 19 | 64358 |
| 6 | `fleet10_master_v1` | 1 | 8 | 583.2 | 574.4 | 16 | 64403 |
| 7 | `probe_v6_family/family_0001` | 1 | 9 | 582.8 | 551.4 | 19 | 64463 |
| 8 | `return_sweep_v1` ← `cluster_fleet_v5/family_0004` | 3 | 8 | 575.8 | 517.7 | 16 | 64448 |
| 9 | `cluster_fleet_v6/family_0007` | 2 | 9 | 573.7 | 527.0 | 18 | 64448 |
| 10 | `return_sweep_v1` ← `cluster_fleet_v5c/family_0001` | 3 | 8 | 568.0 | 584.0 | 16 | 64403 |
| 11 | `cluster_fleet_v7/family_0006` | 1 | 8 | 567.0 | 516.8 | 16 | 64553 |
| 12 | `return_sweep_v1` ← `cluster_fleet_v6/family_0025` | 1 | 8 | 564.7 | 564.1 | 17 | 64358 |
| 13 | `return_sweep_v2` ← `cluster_fleet_v7/family_0030` | 1 | 8 | 564.3 | 509.4 | 17 | 64373 |
| 14 | `cluster_fleet_v4/family_0430` | 1 | 8 | 564.3 | 522.5 | 16 | 64373 |
| 15 | `cluster_fleet_v6/family_0065` | 1 | 9 | 563.4 | 563.0 | 18 | 64343 |
| 16 | `return_sweep_v1` ← `cluster_fleet_v5/family_0247` | 3 | 8 | 562.6 | 536.5 | 16 | 64568 |
| 17 | `cluster_fleet_v5/family_0009` | 1 | 8 | 558.1 | 583.9 | 16 | 64433 |
| 18 | `return_sweep_v1` ← `cluster_fleet_v5/family_0017` | 1 | 8 | 552.8 | 513.8 | 16 | 64388 |
| 19 | `return_sweep_v2` ← `cluster_fleet_v7/family_0009` | 1 | 8 | 542.9 | 604.5 | 17 | 64568 |
| 20 | `cluster_fleet_v4/family_0355` | 1 | 9 | 537.6 | 595.7 | 18 | 64763 |

Leg-cost distribution of `fleet_master_v6` and `cluster_fleet_v7` (`after_v7.json`, same
columns as the table above): v6 collect hops n 164, mean 88.4, p25 63.5, median 84.5, p75
111.4, p90 140.0, share ≤ 75 kg 0.324, TOF median 225 d (165–300), deploy 767.5 kg per ship,
Earth-out 409.6, **Earth-return 216.5 mean / 216.7 median, TOF median 450 d**; v7 collect hops n
146, mean 87.3, median 85.9, TOF median 240 d, Earth-return 232.7 mean, TOF median 480 d.

**Eighth campaign and the fifteen-archive master.** `cluster_fleet_v8` (3 workers, `nice 19`,
4 h, the *same* radius-1.6 / ≥ 18 partition and seeds as v7 so the two campaigns are a paired
A/B; new: harvest substitution after the beam and the SCvx return sweep flown inside the family
pricing and handed to the DP and the re-timer, §6.10) priced 20 families / 59 ships in 14 938 s
on a machine that also carried another agent's three `joint-itinerary` workers, a CUDA sanitizer
build and a 15 GB process for most of the run (load 6–13 on 16 cores; v7 priced 25 families
alone). Its own master fleet is **10 697.60 kg / 19 ships / 563.0 kg average** (v7: 9920.47 / 18
/ 551.3; marks 60 min 5155.8 / 10 ships, 120 min 8453.2 / 16, 240 min 10 707.2 / 19), PSS peak
0.91 GB (v7 1.19). Paired by family, the best ship rose in 15 of 18 families (median +8.2 kg,
mean +9.6; family 8 +52.6, 25 +31.2, 22 +27.5; family 11 −23.0 and 21 −13.6 because the
substituted or re-timed tour changed which plan the beam certified) and the campaign reproduced
`return_sweep_v2`'s best ship inline: family 7 ship 1 616.4 kg straight out of the pricing (v7
607.8, then 616.4 after the separate sweep), ships 2–3 588.9 / 584.4 (v7 579 / 528 → 602.9 after
the sweep). Five ships exceed the 21-ship threshold of 587.8 kg in the whole archive (v7: two).
*Return integration works*: the v8 fleet's Earth return is **205.2 kg mean / 204.8 median at a
495 d median TOF** — below the references' 208–216 kg and beyond their 473–486 d for the first
time (v6 216.5 / 450 d, v7 232.7 / 480 d); 57 of 59 ships were swept (median 19 certified cells
of 45, 35 SCvx solves, 181 s), the cheapest certified cell is 16 kg (median; p90 66 kg) cheaper
than the return the ship finally flew (the re-timer spends the difference on the tour) and lies
at 600 d for 25 of them, 555 d for 11. *Harvest substitution is a small positive*: 59 beams, 325
endpoints attacked, 612 substitute chains re-flown and re-toured (2059 s total, 35 s per beam),
15 swaps accepted in 11 beams for +122.0 kg collected (+2 kg per ship on average; accepted
swaps +2.5 to +26.7 kg, three raised the score at slightly lower mass) — an order of magnitude
short of the ~150 kg per ship the work list targeted. *The collect hop did not move*: 86.8 kg /
210 d median (mean 87.9, p75 107.5, p90 129.1, share ≤ 75 kg 0.29; v6 84.5 / 225 d, references
66 / 181–187). `fleet_master_v7_v8archives` over all fifteen archives (854 routes re-flown, 0 failures,
1109 columns, LP bound 10 744.90, gap 5.6 kg) selects **the same twenty ships as `v6`**:
11 515.67 kg / 575.78 kg average, both verifiers ok (official "Check successfully!", independent
mass error 1e-10 kg). The v8 columns do not enter because the best of them are the archive's own
ships again (family 7's 616.4 kg is asteroid-for-asteroid the selected `return_sweep_v2` ship,
family 6's 567.0 the selected v7 ship) or conflict with a richer selected ship on the same
asteroids (family 11 588.9 vs 611.9; family 7 584.4 vs 582.8 — +1.6 kg raw, not taken by a
master that optimises the fixed-bonus objective and sits 5.6 kg under its LP bound), and the
five conflict-free v8 ships (588.5, 579.9, 573.3, 568.0, 565.1 kg)
cannot be a 21st ship because the rule then needs (11 515.67 + m)/21 ≥ 587.8, i.e. m ≥ 828 kg.
Leg-cost table `results/gtoc12/leg_stats/after_v8.json`:

| Fleet | Collect hops n | mean | p25 | median | p75 | p90 | share ≤ 75 kg | collect TOF median (p25–p75) d | Deploy hop per ship | Earth-out mean | Earth-return mean / median | return TOF median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fleet_master_v6` = `fleet_master_v7_v8archives` (20 ships) | 164 | 88.4 | 63.5 | 84.5 | 111.4 | 140.0 | 0.324 | 225 (165–300) | 767.5 | 409.6 | 216.5 / 216.7 | 450 |
| `cluster_fleet_v8` (19) | 154 | 87.9 | 66.7 | 86.8 | 107.5 | 129.1 | 0.292 | 210 (150–300) | 772.9 | 422.5 | **205.2 / 204.8** | **495** |
| Antipodes 37 / 39, JPL 36 | 338 / 356 / 320 | 65.9–69.2 | 48.2–52.5 | 66.0–67.1 | 82.4–84.9 | 101.1–103.0 | 0.443–0.459 | 181–187 | 837–851 | 460.1–473.5 | 208.0–215.9 / 206.1–214.2 | 473–486 |

**Eighth iteration (whole-itinerary joint re-optimisation, §6.11).** `joint_itinerary_v2`
(3 workers, `nice 19`, 435 s) re-optimised the 20 `fleet_master_v6` ships plus the 12 best
remaining stand-alone archives: **32 of 32 ships improved, +280.4 kg; the 20 fleet ships gained
+208.4 kg (575.78 → 586.20 kg average, +1.1 to +23.3 kg per ship)** with 101 SCvx
certifications, 95 accepted, 13–70 s per ship at 0.12 GB. Zero asteroids were inserted (the
negative result of §6.11: every co-moving candidate hop fails the authority ratio). The
propellant is *redistributed*, not saved: per fleet ship the deploy hops spend **767.5 → 826.8
kg** (+59; the deploy phase shortens from 1870 to 1804 hop-days, i.e. earlier deploys buy
mining time), the Earth return **216.5 → 204.1 kg** (−12), the collect hops 724.6 → 720.6
(−4), the Earth-out leg is unchanged at 409.6 kg (its TOF is a protected floor and no launch
moved), and the spare final-mass margin is spent to ≈ 0 (final masses 500.6–506.4 kg). Return
arrivals move by at most ±20 d. Per ship (before → after, collected kg; certifications;
propellant by role before → after):

| # | Ship (archive / slot) | Ast. | Before | After | Gain | Cert. | Deploy hops | Collect hops | Return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `return_sweep_v1` ← `cluster_fleet_v6/family_0001` s2 | 8 | 633.3 | 634.4 | +1.1 | 1 | 725 → 738 | 818 → 810 | 212 → 211 |
| 2 | `return_sweep_v2` ← `cluster_fleet_v7/family_0007` s1 | 9 | 616.4 | 622.6 | +6.2 | 3 | 824 → 854 | 691 → 702 | 192 → 186 |
| 3 | `cluster_fleet_v7/family_0011` s1 | 9 | 611.9 | 628.5 | +16.6 | 4 | 608 → 691 | 854 → 814 | 266 → 259 |
| 4 | `cluster_fleet_v6/family_0054` s1 | 9 | 603.7 | 614.6 | +11.0 | 4 | 655 → 682 | 818 → 814 | 228 → 225 |
| 5 | `cluster_fleet_v6/family_0025` s5 | 9 | 589.3 | 612.3 | +22.9 | 4 | 742 → 815 | 711 → 682 | 249 → 244 |
| 6 | `fleet10_master_v1` s1 | 8 | 583.2 | 594.8 | +11.6 | 3 | 729 → 864 | 872 → 817 | 134 → 126 |
| 7 | `cluster_fleet_v6/family_0001` s1 | 9 | 582.8 | 601.8 | +19.0 | 4 | 706 → 786 | 684 → 669 | 298 → 282 |
| 8 | `return_sweep_v1` ← `cluster_fleet_v5/family_0004` s3 | 8 | 575.8 | 578.9 | +3.2 | 2 | 764 → 776 | 750 → 758 | 191 → 188 |
| 9 | `cluster_fleet_v6/family_0007` s2 | 9 | 573.7 | 582.3 | +8.6 | 3 | 674 → 693 | 742 → 755 | 273 → 264 |
| 10 | `return_sweep_v1` ← `cluster_fleet_v5c/family_0001` s3 | 8 | 568.0 | 583.0 | +15.0 | 5 | 707 → 821 | 759 → 746 | 204 → 184 |
| 11 | `cluster_fleet_v7/family_0006` s1 | 8 | 567.0 | 570.0 | +3.0 | 4 | 716 → 733 | 794 → 794 | 241 → 237 |
| 12 | `return_sweep_v1` ← `cluster_fleet_v6/family_0025` s1 | 8 | 564.7 | 573.2 | +8.5 | 4 | 843 → 913 | 622 → 630 | 222 → 206 |
| 13 | `return_sweep_v2` ← `cluster_fleet_v7/family_0030` s1 | 8 | 564.3 | 567.6 | +3.3 | 3 | 671 → 713 | 841 → 816 | 221 → 210 |
| 14 | `cluster_fleet_v4/family_0430` s1 | 8 | 564.3 | 566.7 | +2.5 | 1 | 1095 → 1112 | 569 → 571 | 155 → 152 |
| 15 | `cluster_fleet_v6/family_0065` s1 | 9 | 563.4 | 581.9 | +18.5 | 3 | 762 → 848 | 588 → 583 | 259 → 241 |
| 16 | `return_sweep_v1` ← `cluster_fleet_v5/family_0247` s3 | 8 | 562.6 | 568.7 | +6.0 | 3 | 872 → 948 | 682 → 647 | 189 → 181 |
| 17 | `cluster_fleet_v5/family_0009` s1 | 8 | 558.1 | 581.4 | +23.3 | 4 | 638 → 749 | 723 → 781 | 285 → 197 |
| 18 | `return_sweep_v1` ← `cluster_fleet_v5/family_0017` s1 | 8 | 552.8 | 557.3 | +4.5 | 3 | 829 → 858 | 651 → 655 | 201 → 175 |
| 19 | `return_sweep_v2` ← `cluster_fleet_v7/family_0009` s1 | 8 | 542.9 | 555.6 | +12.7 | 4 | 716 → 848 | 753 → 743 | 187 → 169 |
| 20 | `cluster_fleet_v4/family_0355` s1 | 9 | 537.6 | 548.4 | +10.8 | 5 | 1073 → 1093 | 570 → 623 | 122 → 145 |
| — | *12 non-fleet archives* | 7–9 | 6997.9 | 7069.9 | +72.0 (+1.2 to +14.1) | 1–5 | | | |

At 586.20 kg the re-optimised 20-ship fleet is still 1.6 kg per ship short of the 21-ship
threshold (587.8 kg). `fleet_master_v7` over all sixteen archives (837 routes re-flown through
SCvx, 0 failures, 1078 columns) closes that gap by *composition*: it is **proven optimal at 21
ships — fixed-bonus objective 11 391.12 kg vs LP bound 11 396.76 (gap 5.6 kg), 177 asteroids
(173 mined, 4 miners left), 12 346.48 kg collected, 587.93 kg average (rule 21 ≤ 21.007)**,
both verifiers ok (`Check successfully!`; independent max position error 113 km, mass error
1e-10 kg, no violations). That is **+830.8 kg over `fleet_master_v6`** (+7.2 %): +208.4 kg from
the joint re-optimisation of the ships in place and the remainder from the 21st ship the rule
now admits. All 21 selected columns are `joint_itinerary_v2` routes; 19 of them are the v6
ships' asteroid sets, the master drops the 567.6 kg `cluster_fleet_v7/family_0030` ship and adds
the re-optimised `return_sweep_v1 ← cluster_fleet_v6/family_0005` slot 2 (594.8) and
`return_sweep_v2 ← cluster_fleet_v7/family_0014` slot 2 (595.2). The LP relaxation at 20 ships
is 10 987.4 kg, so the 21st ship is worth ≈ 404 kg on the master's objective. Per ship (rank,
source archive of the re-optimised column, slot, asteroids, collected kg, final mass kg,
refined arcs, launch MJD):

| # | Source (all via `joint_itinerary_v2`) | Slot | Ast. | Collected | Final mass | Arcs | Launch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `return_sweep_v1` ← `cluster_fleet_v6/family_0001` | 2 | 8 | 634.4 | 503.4 | 17 | 64478 |
| 2 | `cluster_fleet_v7/family_0011` | 1 | 9 | 628.5 | 501.8 | 18 | 64478 |
| 3 | `return_sweep_v2` ← `cluster_fleet_v7/family_0007` | 1 | 9 | 622.6 | 504.9 | 18 | 64463 |
| 4 | `cluster_fleet_v6/family_0054` | 1 | 9 | 614.6 | 503.5 | 19 | 64373 |
| 5 | `cluster_fleet_v6/family_0025` | 5 | 9 | 612.3 | 502.2 | 19 | 64358 |
| 6 | `cluster_fleet_v6/family_0001` | 1 | 9 | 601.8 | 503.0 | 19 | 64463 |
| 7 | `return_sweep_v2` ← `cluster_fleet_v7/family_0014` | 2 | 8 | 595.2 | 503.3 | 17 | 64508 |
| 8 | `return_sweep_v1` ← `cluster_fleet_v6/family_0005` | 2 | 9 | 594.8 | 502.0 | 19 | 64478 |
| 9 | `fleet10_master_v1` | 1 | 8 | 594.8 | 501.8 | 16 | 64403 |
| 10 | `return_sweep_v1` ← `cluster_fleet_v5c/family_0001` | 3 | 8 | 583.0 | 501.8 | 16 | 64403 |
| 11 | `cluster_fleet_v6/family_0007` | 2 | 9 | 582.3 | 503.9 | 18 | 64448 |
| 12 | `cluster_fleet_v6/family_0065` | 1 | 9 | 581.9 | 500.6 | 18 | 64343 |
| 13 | `cluster_fleet_v5/family_0009` | 1 | 8 | 581.4 | 503.0 | 16 | 64433 |
| 14 | `return_sweep_v1` ← `cluster_fleet_v5/family_0004` | 3 | 8 | 578.9 | 501.2 | 16 | 64448 |
| 15 | `return_sweep_v1` ← `cluster_fleet_v6/family_0025` | 1 | 8 | 573.2 | 501.9 | 17 | 64358 |
| 16 | `cluster_fleet_v7/family_0006` | 1 | 8 | 570.0 | 504.4 | 16 | 64553 |
| 17 | `return_sweep_v1` ← `cluster_fleet_v5/family_0247` | 3 | 8 | 568.7 | 502.7 | 16 | 64568 |
| 18 | `cluster_fleet_v4/family_0430` | 1 | 8 | 566.7 | 506.4 | 16 | 64373 |
| 19 | `return_sweep_v1` ← `cluster_fleet_v5/family_0017` | 1 | 8 | 557.3 | 505.1 | 16 | 64388 |
| 20 | `return_sweep_v2` ← `cluster_fleet_v7/family_0009` | 1 | 8 | 555.6 | 501.6 | 17 | 64568 |
| 21 | `cluster_fleet_v4/family_0355` | 1 | 9 | 548.4 | 501.1 | 18 | 64763 |

**Ninth iteration (chain-level objective, prior, duals, joint itinerary in the pricing; §6.12).**
`cluster_fleet_v9` (3 workers, `nice 19`, the v7/v8 partition and seeds, launched 10:01 AEST on a
machine that also carried another agent's 7 GB test run for the first hour) priced 20 families /
60 ships in 246 logged minutes; the 260-min wrapper timeout — sized on v8's 249 min — killed the
process in the last family's tail, before the final master and `fleet/` were written, so the
campaign's fleet is its last verified incumbent: **9960.33 kg / 18 ships / 155 asteroids /
553.4 kg average** (v8: 10 697.60 / 19 / 563.0; marks 60 min 5197.9 / 10, 120 min 8343.8 / 16,
240 min 9960.3 / 18), PSS peak 0.92 GB. *Paired by family (19 common), the chain-level objective
is neutral*: the best ship rose in 7 families and fell in 6 (median Δ 0.0 kg, mean −4.6; family 17
+40.7, 25 +19.3, 29 +9.0 and 1 +5.1 via the inline joint step; family 8 −53.8, 6 −52.0, 3 −30.0,
7 −10.3), 6 families reproduced v8's ship exactly. The beam's telemetry says why: over 63 beams the
tour scorer re-scored 9 562 partials (4 935 s, 78 s per beam), 845 of them did not close on the
exact pass and 811 had no tour, and the re-ranking moved 582 partials into the beams (9 per beam,
of 24 × ~7 levels) — but the chain the heuristic ranks first is also the chain the DP ranks first
on most families (family 7 diagnostic: identical `best_by_depth` from depth 5 on, at 48 and at 144
candidates), because at the beam's exchange rate (0.15 kg of score per kg of propellant) both
scores are dominated by the collected mass and the DP-collected spread among a level's survivors
(76 kg at depth 7) tracks the heuristic's deploy-epoch term. Where it did change the winner the
outcome is a coin toss, hence the variance. The *direction* is the reference one, though: over
all 60 campaign ships the collect hop fell from 97.5 to 90.6 kg median (mean 105.1 → 100.5, the
collect tour per ship 696 → 658 kg) and the deploy hops rose 716 → 729 kg per ship — but the
propellant moved was not converted into ore (collected mean 465 → 456 kg per ship, chains
≥ 587.8 kg 3 → 3, ≥ 600 kg 1 → 1, best 616.4 → 606.2). *The inline joint itinerary* ran on 11
self-cleaning slots (8 improved, +146.9 kg, 35 certifications, 389 s) and was skipped on every
deployer that leaves orphans and every collector, which is most slots of a cooperative family.
*The LP duals had nothing to bite on*: the archive LP at N = 21 (LP(22) infeasible) carries
μ = 1316 kg on the ship count and ν = 1.40 on the mass floor and only 22–24 asteroids with a
positive row dual (max 131 kg, median 31; the rent of a selected ship sits on its `x ≤ 1` bound in
a near-integral LP), of which one lay in a priced family (3.2 kg on one member of family 29) — so
63 slots were priced and none was steered. `--dual-bound-share` (added afterwards, tested) moves
the bound duals onto the columns' asteroids — another optimal dual solution — and prices 61
asteroids (median 3.8 kg, sum 1047 kg against 943): still small, because the fleet's rent is
almost entirely the uniform ship-count price μ. `joint_itinerary_v3` (the 40 best stand-alone v8
and v9 ships, 7.8 min) improved 35 of 40 for **+411.5 kg** — v8's ships had never had the joint
pass (22 of 24, +241.0; family 0 slot 3 565.1 → 600.6, family 7 slot 1 616.4 → 622.6) and v9's 13
of 16 (+170.5; family 7 606.2 → 619.2, family 29 588.9 → 601.2) — moving, per ship, 68 kg from
the collect hops (683 → 669) and the return (198 → 190) into the deploy hops (801 → 869), the
reference split. `fleet_master_v8` over all nineteen archives (1006 routes re-flown, 0 failures,
1296 columns, LP bound 11 448.02, gap 6.4 kg, proven optimal, LP(22) infeasible) is **21 ships /
177 asteroids / 12 356.30 kg / 588.40 kg average** (rule 21 ≤ 21.046), both verifiers ok
(official "Check successfully!" 12 356.3 kg; independent 12 356.304 kg, mass error 1e-10 kg, max
position error 113 km): **+9.8 kg over `fleet_master_v7`**, three ships swapped (in: `v3`
re-optimised `cluster_fleet_v9/family_0029` slot 1 601.2, `cluster_fleet_v8/family_0000` slot 3
600.6, `cluster_fleet_v8/family_0030` slot 1 577.0; out: the `v2` 594.8, 595.2 and 578.9 ships),
all 21 columns joint-itinerary routes (16 `v2`, 5 `v3`; one from the v9 campaign). Ship 22 is not
reached: the rule needs 599.5 kg average, i.e. a 22nd ship of 832.7 kg *or* +11.1 kg on every
ship. Leg-cost table `results/gtoc12/leg_stats/after_v9.json` (per-ship split before → after):

| Fleet | Collect hops n | mean | p25 | median | p75 | p90 | share ≤ 75 kg | collect TOF median (p25–p75) d | Deploy hop per ship | Earth-out mean | Earth-return mean / median | return TOF median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cluster_fleet_v8` (19) | 154 | 87.9 | 66.7 | 86.8 | 107.5 | 129.1 | 0.292 | 210 (150–300) | 772.9 | 422.5 | 205.2 / 204.8 | 495 |
| `cluster_fleet_v9` (18) | 144 | 86.8 | 64.9 | 84.6 | 101.2 | 128.6 | 0.292 | 210 (150–300) | 796.6 | 417.6 | 207.7 / 206.1 | 480 |
| `fleet_master_v7` (21) | 173 | 87.3 | 62.7 | 85.7 | 110.7 | 129.6 | 0.301 | 210 (165–300) | 822.9 | 412.7 | 205.6 / 204.0 | 450 |
| **`fleet_master_v8`** (21) | 172 | 86.7 | 64.1 | 84.3 | 110.5 | 129.7 | 0.293 | 210 (161–300) | **838.5** | 407.0 | 204.3 / 198.8 | 450 |
| Antipodes 37 / 39, JPL 36 | 338 / 356 / 320 | 65.9–69.2 | 48.2–52.5 | 66.0–67.1 | 82.4–84.9 | 101.1–103.0 | 0.443–0.459 | 181–187 | 837–851 | 460.1–473.5 | 208.0–215.9 / 206.1–214.2 | 473–486 |

The fleet's deploy spend now *equals* the references' (838.5 vs 837–851 kg per ship) — the joint
re-optimiser bought it — while the collect hop is unchanged at 84 kg / 210 d against 66 / 181–187,
the Earth-out leg is 55–65 kg cheaper (407 vs 460–474: slower, later arrival) and the collected
mass 588 vs 730 kg per ship. Paying the reference deploy price did not buy the reference harvest.

Per-leg detail of `reduced-v1-run2` (propellant, SCvx iterations, solve time, certified endpoint
error): E→1265 600 d 364.7 kg 9 it 1.3 s 0.32 km; 1265→21191 360 d 128.7 kg 7 it 0.5 s 0.04 km;
21191→27292 360 d 153.5 kg 5 it 0.3 s 0.04 km; 27292→40808 120 d 97.4 kg 24 it 0.7 s 0.01 km;
40808→27292 420 d 229.8 kg 7 it 0.6 s 0.02 km; 27292→21191 420 d 90.5 kg 5 it 0.4 s 0.05 km;
21191→1265 420 d 266.3 kg 6 it 0.5 s 0.09 km; 1265→Earth 550 d 235.8 kg 9 it 1.1 s 0.46 km. Final
mass 1273.3 kg ≥ 500 kg + 253.7 kg carried. Run 2's other two refined candidates also verified
officially (252.923 kg and 241.697 kg).

Full catalogue, run 1 (before search v2): screening the 60,000 asteroids over 25 launch epochs ×
13 Earth-leg TOFs (39.1 M zero-revolution Lambert solves) took 956 s and 11.2 GB. Search v2
screens the 10,612-asteroid reference box in 1,500-asteroid blocks (0.65 GB peak) and reaches
depth 8 in ~260 s; failures are retained with reasons in `ship_NN/search.json`.

Independent verifier on the scored files: per-asteroid masses agree with the official verifier to
1e-10 kg; max propagation error 0.52 km on single ships and 14.4 km on the 3-ship fleet file (well
inside the official tolerance; the official binary reports "Check successfully!" on every file).
Fleet rule: 3 ships ≤ 2 exp(0.004 × 464.7) = 12.8.

Artifacts (ignored `results/gtoc12/runs/<run>/`, compact files force-added): `run_report.json`,
`ship_NN/search.json`, `ship_NN/refinements.json`, `ship_NN/candidate_NN/Result.txt`,
`ship_NN/candidate_NN/route_summary.json`, `fleet/Result.txt`, and
`fleet/viewer/{trajectories.json,manifest.json}` for the best fleet. The viewer export follows the
`web/trajectory-viewer` record schema (family `GTOC12`, heliocentric frame, replay decimated to
≤ 512 exact propagated samples with events preserved, event-state transcription, asteroid/Earth
context orbits). Third-campaign runs (`cluster_fleet_v1`, `cluster_fleet_v2_deep`,
`cluster_fleet_v3_repair`, `fleet_master_v1`) commit `run_report.json`, every
`clusters/family_NNNN/bundle.json` and `ship_NN/route_summary.json` (the per-ship JSON, which
embeds the plan and is what `fleet-master` reloads), the intermediate fleets' `fleets/*/fleet.json`
and the best fleet's `fleet/Result.txt` + `fleet.json` + `viewer/manifest.json`
(`fleet_master_v1`, 6.5 MB). Not committed (regenerable): per-ship `Result.txt` files
(re-emitted by `fleet-master` from the JSON), the intermediate fleets' `Result.txt`, the
`columns/` re-certification copies and `fleet/viewer/trajectories.json` (6.0 MB) — regenerate the
viewer data with `PYTHONPATH=src python -m spacepdhcg gtoc12 export-viewer
results/gtoc12/runs/fleet_master_v1/fleet/Result.txt --output
results/gtoc12/runs/fleet_master_v1/fleet/viewer --run-id fleet_master_v1_fleet`, and the whole
fleet with `python -m spacepdhcg gtoc12 fleet-master --run-id fleet_master_v1 --output <dir>
--source results/gtoc12/runs/cluster_fleet_v1 --source results/gtoc12/runs/cluster_fleet_v2_deep
--source results/gtoc12/runs/cluster_fleet_v3_repair --source
results/gtoc12/runs/fleet10_master_v1 --workers 6 --node-cap 5000000` (8 min on 16 cores).
Fourth-campaign runs follow the same rule: `cluster_fleet_v4` commits `run_report.json`, the 47
`bundle.json` and 119 `route_summary.json` files and the 15 intermediate `fleets/*/fleet.json`
(its 137 per-ship/intermediate `Result.txt` files are not committed); `fleet_master_v2` commits
`run_report.json`, `fleet/Result.txt` (6.9 MB, the best verified fleet), `fleet/fleet.json` and
`fleet/viewer/manifest.json`; `probe_v4_family` (the family-0 harvest probe whose 3 ships are
`fleet-master` sources) commits its `run_report.json`, `bundle.json` and `route_summary.json`.
Regenerate the best fleet with `PYTHONPATH=src python -m spacepdhcg gtoc12 fleet-master --run-id
fleet_master_v2 --output <dir> --source results/gtoc12/runs/cluster_fleet_v1 --source
results/gtoc12/runs/cluster_fleet_v2_deep --source results/gtoc12/runs/cluster_fleet_v3_repair
--source results/gtoc12/runs/cluster_fleet_v4 --source results/gtoc12/runs/fleet10_master_v1
--source results/gtoc12/runs/probe_v4_family --workers 6 --node-cap 5000000` (14.5 min on 16
cores; the SCvx replay is deterministic, 330/330 routes re-certify) and its viewer data with
`PYTHONPATH=src python -m spacepdhcg gtoc12 export-viewer
results/gtoc12/runs/fleet_master_v2/fleet/Result.txt --output
results/gtoc12/runs/fleet_master_v2/fleet/viewer --run-id fleet_master_v2_fleet` (6.8 MB
`trajectories.json`, not committed). The leg-cost tables of §6.6 are
`results/gtoc12/leg_stats/{before_v4,after_v4}.json` from `gtoc12 leg-stats --solution
name=path …` over the fleet files and the three reference files.
Fifth-campaign runs: `cluster_fleet_v5` and `cluster_fleet_v5c` commit `run_report.json`, every
`bundle.json` and `route_summary.json` (38 + 27 families, 128 + 94 ships, rejected variants
inside the bundle reports) and the intermediate fleets' `fleets/*/fleet.json` (16 + 10 verified
fleets); `probe_v5_family247` its `run_report.json`, `bundle.json` and `route_summary.json`;
`fleet_master_v3` commits `run_report.json`, `fleet/Result.txt` (7.8 MB, the best verified
fleet), `fleet/fleet.json` and `fleet/viewer/manifest.json`; the leg-cost table above is
`results/gtoc12/leg_stats/after_v5.json`. Regenerate the best fleet with `PYTHONPATH=src python
-m spacepdhcg gtoc12 fleet-master --run-id fleet_master_v3 --output <dir>` with the nine
`--source results/gtoc12/runs/{cluster_fleet_v1,cluster_fleet_v2_deep,cluster_fleet_v3_repair,
cluster_fleet_v4,fleet10_master_v1,probe_v4_family,probe_v5_family247,cluster_fleet_v5,
cluster_fleet_v5c}` `--workers 8 --node-cap 5000000 --lp-node-limit 50000` (18 min on 16 cores;
554/554 routes re-certify) and its viewer data with `PYTHONPATH=src python -m spacepdhcg gtoc12
export-viewer results/gtoc12/runs/fleet_master_v3/fleet/Result.txt --output
results/gtoc12/runs/fleet_master_v3/fleet/viewer --run-id fleet_master_v3_fleet` (8.0 MB
`trajectories.json`, not committed; the dataset title is now derived from the instance —
"GTOC12 full-catalogue OrbitWeaver solution (18 ships, 9888.6 kg)"). The viewer candidate's
importer (`web/trajectory-viewer`, `node scripts/import-gtoc12.mjs --export <viewer dir>
--catalogue <GTOC12_Asteroids_Data.txt> --solution <Result.txt> --fleet <fleet.json> --output
data/gtoc12`, the target of `npm run import-gtoc12`) was run on this export: 18 ships, 147
asteroids, 9140 exact replay samples, all hashes verified, Kepler cross-check 3.6e-6 km over
55 263 context points; its output stays ignored.
Sixth-campaign runs: `cluster_fleet_v6` commits `run_report.json` (with `budget_marks`,
`memory_samples` and `memory_total_pss_peak_mb`), every `bundle.json` and `route_summary.json`
(36 families, 136 ships) and the 13 intermediate fleets' `fleets/*/fleet.json`;
`probe_v6_family` its `run_report.json`, `bundle.json` and `route_summary.json`;
`fleet_master_v4` commits `run_report.json`, `fleet/Result.txt` (7.9 MB, the best verified
fleet), `fleet/fleet.json` and `fleet/viewer/manifest.json`; the calibration fit is
`results/gtoc12/hop_inflation_fit.json`; the leg-cost table above is
`results/gtoc12/leg_stats/after_v6.json`. Regenerate the best fleet with `PYTHONPATH=src python
-m spacepdhcg gtoc12 fleet-master --run-id fleet_master_v4 --output <dir>` with the eleven
`--source results/gtoc12/runs/{cluster_fleet_v1,cluster_fleet_v2_deep,cluster_fleet_v3_repair,
cluster_fleet_v4,fleet10_master_v1,probe_v4_family,cluster_fleet_v5,cluster_fleet_v5c,
probe_v5_family247,cluster_fleet_v6,probe_v6_family}` `--workers 8` (18.5 min on 16 cores;
695/695 routes re-certify), the campaign with `PYTHONPATH=src python -m spacepdhcg gtoc12
cluster-fleet --run-id cluster_fleet_v6 --output <dir> --workers 4 --ships-per-cluster 5
--cluster-radius 1.75 --min-members 20 --collect-epoch-families --collector-harvest
--collect-dp-inflation-fit results/gtoc12/hop_inflation_fit.json --collect-dp-step-days 15
--earth-prescreen-ratio 0.7 --cluster-budget-seconds 2400 --retime-budget-seconds 900
--budget-seconds 14400`, and its viewer data with `PYTHONPATH=src
python -m spacepdhcg gtoc12 export-viewer results/gtoc12/runs/fleet_master_v4/fleet/Result.txt
--output results/gtoc12/runs/fleet_master_v4/fleet/viewer --run-id fleet_master_v4_fleet`
(8.5 MB `trajectories.json`, not committed; title "GTOC12 full-catalogue OrbitWeaver solution
(19 ships, 10700.5 kg)"). The viewer importer was run on this export: 19 ships, 158 asteroids,
9643 exact replay samples, all hashes verified, Kepler cross-check 3.57e-6 km (asteroids) /
3.1e-7 km (Earth) over 59 356 context points; its output stays ignored.
Seventh-iteration runs: `return_sweep_v1` commits `run_report.json`, `ships.jsonl` (one record
per re-timed ship: sweep table, attempts, before/after) and the 13 improved
`ships/<group>/ship_NN/route_summary.json`; `fleet_master_v5` commits `run_report.json`,
`fleet/Result.txt` (the best verified fleet), `fleet/fleet.json` and `fleet/viewer/manifest.json`;
the leg-cost table is `results/gtoc12/leg_stats/after_v7.json`. Regenerate with `PYTHONPATH=src
python -m spacepdhcg gtoc12 retime-returns --run-id return_sweep_v1 --output <dir>` with the
eleven `--source` directories above `--workers 3 --top 36 --budget-seconds 5400
--per-ship-seconds 1200` (50 min), then `fleet-master --run-id fleet_master_v5 --output <dir>`
with the eleven sources plus `--source results/gtoc12/runs/return_sweep_v1` `--workers 3`
(43 min on 3 workers). The viewer export is written by `fleet-master` itself
(`fleet/viewer`, 9.1 MB `trajectories.json`, not committed); the importer was run on it: 20
ships, 168 asteroids, 10 151 exact replay samples, all hashes verified, Kepler cross-check
3.57e-6 km (asteroids) / 3.07e-7 km (Earth) over 63 088 context points; output ignored.
Seventh campaign: `cluster_fleet_v7` commits `run_report.json` (`budget_marks`,
`memory_samples`, `memory_total_pss_peak_mb`), every `bundle.json` (with `memory_phases`) and
`route_summary.json` (25 families, 76 ships) and the intermediate fleets' `fleets/*/fleet.json`;
`return_sweep_v2` its `run_report.json`, `ships.jsonl` and the 13 improved
`route_summary.json`; `fleet_master_v6` its `run_report.json`, `fleet/Result.txt` (the best
verified fleet by the master's objective), `fleet/fleet.json` and `fleet/viewer/manifest.json`.
Commands: `cluster-fleet --run-id cluster_fleet_v7 --output <dir> --workers 3
--ships-per-cluster 5 --cluster-radius 1.6 --min-members 18 --collect-epoch-families
--collector-harvest --collect-dp-inflation-fit results/gtoc12/hop_inflation_fit.json
--collect-dp-step-days 15 --earth-prescreen-ratio 0.7 --cluster-budget-seconds 2400
--retime-budget-seconds 900 --budget-seconds 14400` under `nice -n 19` (249 min);
`retime-returns --run-id return_sweep_v2 --output <dir> --source results/gtoc12/runs/cluster_fleet_v7
--workers 3 --top 21 --budget-seconds 2700` (36 min); `fleet-master --run-id fleet_master_v6
--output <dir>` with the fourteen `--source` directories `--workers 3` (46 min). The v2 viewer
importer was run on `fleet_master_v6/fleet/viewer`: 20 ships, 168 asteroids, 10 150 exact replay
samples, all hashes verified, Kepler cross-check 3.57e-6 km / 3.07e-7 km over 63 088 context
points; output ignored.
Eighth campaign: `cluster_fleet_v8` commits `run_report.json` (`budget_marks`, `memory_samples`,
`memory_total_pss_peak_mb`), every `bundle.json` (with `ships[].search[].substitution` and
`ships[].return_sweep`) and `route_summary.json` (20 families, 59 ships), `fleet/fleet.json`
and the intermediate fleets' `fleets/*/fleet.json` (its `Result.txt` is regenerable and ignored); `fleet_master_v7_v8archives` its
`run_report.json`, `fleet/Result.txt` (the best verified fleet — identical to `v6`'s),
`fleet/fleet.json` and `fleet/viewer/manifest.json`; the leg-cost table is
`results/gtoc12/leg_stats/after_v8.json`. Commands: the v7 `cluster-fleet` line with
`--run-id cluster_fleet_v8 --substitution-budget-seconds 150 --return-sweep-budget-seconds 180`
under `nice -n 19` (249 min); `fleet-master --run-id fleet_master_v7_v8archives --output <dir>` with the
fifteen `--source` directories (`cluster_fleet_v8` added) `--workers 3` (48 min); an external
archive directory (the Lambda box's `cluster_fleet_h100_v1`) is one more `--source`. The v2
viewer importer was run on `fleet_master_v7_v8archives/fleet/viewer`: 20 ships, 168 asteroids, 10 150
exact replay samples, Kepler cross-check 3.57e-6 km / 3.07e-7 km over 63 088 context points;
output ignored.

Eighth iteration: `joint_itinerary_v1` (27 ships, the pre-orphan task selection that matched 15
of the 20 fleet ships) and `joint_itinerary_v2` (32 ships, final code) commit their
`run_report.json`, `ships.jsonl` (per ship: before/after mass, legs by role before/after,
certifications, wall) and every improved ship's `route_summary.json`; `fleet_master_v7` its
`run_report.json`, `fleet/Result.txt` (the best verified fleet), `fleet/fleet.json` and
`fleet/viewer/manifest.json`. Commands: `joint-itinerary --run-id joint_itinerary_v2 --output
<dir> --fleet-report results/gtoc12/runs/fleet_master_v6/run_report.json --top 12 --workers 3
--per-ship-seconds 600 --budget-seconds 5400 --insert-trials 4` with the fourteen `--source` directories under
`nice -n 19` (7 min); `fleet-master --run-id fleet_master_v7 --output <dir>` with the sixteen
`--source` directories `--workers 3` (57 min). The v2 viewer importer was run on
`fleet_master_v7/fleet/viewer`: 21 ships, 177 asteroids, 10 664 exact replay samples, all hashes
verified, Kepler cross-check 3.57e-6 km / 3.07e-7 km over 66 459 context points; output
ignored. The viewer's `check.mjs` asserts a 20-colour ship palette (`fleet.ships.length <= 20`)
and now fails on that assertion alone; the viewer itself wraps colours modulo the palette, so
ship 21 shares ship 1's colour until the palette is extended.
Ninth iteration: `cluster_fleet_v9` commits `run_report.json` (`dual_prices`, `dual_archive`,
`memory_total_pss_peak_mb`), every `bundle.json` (with `ships[].search[].chain_tour`,
`ships[].joint_itinerary`, `asteroid_prices`) and `route_summary.json` (20 families, 60 ships) and
the intermediate fleets' `fleets/*/fleet.json` (no final `fleet/`: the wrapper timeout, see §7);
`joint_itinerary_v3` its `run_report.json`, `ships.jsonl` and every improved ship's
`route_summary.json`; `fleet_master_v8` its `run_report.json`, `fleet/Result.txt` (the best
verified fleet), `fleet/fleet.json` and `fleet/viewer/manifest.json`; the prior is
`benchmarks/gtoc12/chain_prior_v1.json`, the leg-cost table `results/gtoc12/leg_stats/after_v9.json`
and the campaign report `results/gtoc12/leg_stats/v9_report.json`. Commands: `gtoc12 chain-prior
--output benchmarks/gtoc12/chain_prior_v1.json`; the v8 `cluster-fleet` line with `--run-id
cluster_fleet_v9 --chain-tour-scoring --chain-tour-candidates 48 --chain-prior
benchmarks/gtoc12/chain_prior_v1.json --chain-prior-weight 0.5 --dual-archive <each of the
seventeen earlier runs> --dual-target-size 22 --joint-itinerary --joint-budget-seconds 150` under
`nice -n 19` (killed by `timeout 15600` at 260 min); `joint-itinerary --run-id joint_itinerary_v3
--source results/gtoc12/runs/cluster_fleet_v8 --source results/gtoc12/runs/cluster_fleet_v9 --top 40
--workers 3 --per-ship-seconds 600 --budget-seconds 5400 --no-insert` (7.8 min); `fleet-master
--run-id fleet_master_v8 --output <dir>` with the nineteen `--source` directories `--workers 3`
(58 min). Not ingested: the methods worktree's `fleet_master_v7/columns` (833 re-certified copies
of these archives — 0 new asteroid sets, none heavier) and the Windows-side
`results/lambda-h100/gtoc12` copies (fleet `Result.txt` files and run reports only, no
`route_summary.json`; a `Result.txt` → column ingester would be needed).


**Ninth iteration (Lambda H100 host, commit `810f041`): breadth on 26 CPU cores.** The GPU stayed reserved for the G4 campaign; every GTOC12 process ran with `CUDA_VISIBLE_DEVICES=""`.
`cluster_fleet_h100_v2` priced 111 families from the union of 4 partitions (collect_r1.75 47 families, phasing_r1.75 56 families, collect_r1.6 29 families, phasing_r1.6 35 families; `family_partitions`, labels offset per partition) in 9.2 h: 30 min —; 60 min —; 120 min 9104.6 / 17 ships; 240 min 11522.0 / 20 ships; 480 min 12348.7 / 21 ships; final 12348.9 kg, 21 ships, 182 asteroids, 588.0 kg average. Family stops: {"": 9, "time budget before ship slot 3": 1, "family exhausted before ship slot 4": 36, "time budget before ship slot 5": 19, "crashed: ValueError('cannot convert float NaN to integer')": 1, "family exhausted before ship slot 5": 27, "time budget before ship slot 4": 18}.
Chain-mass distribution over the 21 master sources (unique asteroid sets): 1207 chains, 21 ≥ 600 kg, 1 ≥ 650 kg, 0 ≥ 700 kg, best 652.6 kg; per source: `cluster_fleet_v1` 150 chains / 0 ≥ 600; `cluster_fleet_v2_deep` 7 chains / 0 ≥ 600; `cluster_fleet_v3_repair` 23 chains / 0 ≥ 600; `cluster_fleet_v4` 119 chains / 0 ≥ 600; `fleet10_master_v1` 9 chains / 0 ≥ 600; `probe_v4_family` 3 chains / 0 ≥ 600; `cluster_fleet_v5` 128 chains / 0 ≥ 600; `cluster_fleet_v5c` 94 chains / 1 ≥ 600; `probe_v5_family247` 2 chains / 0 ≥ 600; `cluster_fleet_v6` 136 chains / 2 ≥ 600; `probe_v6_family` 5 chains / 0 ≥ 600; `return_sweep_v1` 13 chains / 1 ≥ 600; `cluster_fleet_v7` 76 chains / 2 ≥ 600; `return_sweep_v2` 13 chains / 2 ≥ 600; `joint_itinerary_v1` 27 chains / 8 ≥ 600; `joint_itinerary_v2` 32 chains / 8 ≥ 600; `cluster_fleet_h100_v1` 116 chains / 3 ≥ 600; `joint_itinerary_h100_v1` 294 chains / 3 ≥ 600; `cluster_fleet_h100_v2` 386 chains / 7 ≥ 600; `joint_itinerary_h100_v2` 219 chains / 13 ≥ 600; `cluster_fleet_v8` 59 chains / 1 ≥ 600.
`joint_itinerary_h100_v1`: 294 of 339 ships improved, +4665.0 kg, 0 insertions; chains ≥ 600 kg 9 → 10, 184 min.
`joint_itinerary_h100_v2`: 219 of 231 ships improved, +3211.6 kg, 0 insertions; chains ≥ 600 kg 6 → 13, 20 min.
`fleet_master_h100_v2` over 21 archives (2480 columns): **13189.60 kg, 22 ships, 187 asteroids, 599.53 kg average**; master objective 12203.96 kg, LP bound 12207.39 kg, gap 3.4 kg (not proven; LP relaxation infeasible or below the incumbent beyond 22 ships: {"22": 12207.4}). The 22-ship threshold is 599.5 kg average, 23 ships ~611 kg. Official `GTOC12_Verify`: 2492/2826 emitted `Result.txt` files pass (per-ship diagnostic files of cooperative members fail Error803 by construction); the independent verifier agrees on the fleet (`independent_verify.txt`).
Commands: `~/s/gtoc12_v2_campaign.sh` on the host (cluster-fleet `--workers 22 --ships-per-cluster 5 --cluster-radius 1.75,1.6 --all-family-bands --collect-epoch-families --min-members 20 --beam-width 32 --refine-top 3 --cluster-budget-seconds 6600 --retime-budget-seconds 900 --budget-seconds 28800 --max-clusters 400` with the v6 DP/harvest flags; `joint-itinerary --top 100000 --min-collected-kg 450 --per-ship-seconds 600 --insert-trials 4`; `fleet-master --workers 22`). Artefacts committed as for the earlier campaigns (run reports, `bundle.json`, `route_summary.json`, `ships.jsonl`, verified `fleet.json`s, the master's `fleet/Result.txt`, `official_verification.json`, `chain_stats.json`, `results/gtoc12/leg_stats/after_h100_v2.json`).

## 8. Limitations

- Cooperative collection is modelled end to end (plans, re-timer, refinement, emission, pool,
  master; §6.4) but the emitted fleets are self-cleaning: orphans left by one ship were never
  reachable for the next ship's tour, so the credit defaults to 0. No gravity assists;
  zero-revolution Lambert screening only (justified by the references, but long collect hops of
  600–720 days are then screened as single-arc transfers).
- Fleets are still built greedily (ship k prices against ships 1..k−1's asteroids); the master
  chooses among every certified column afterwards but, without cooperative columns, it can only
  confirm the greedy incumbent (`fleet6_coop_v1`: 20 columns, 9,177 nodes, exhaustive, greedy
  = optimum). Later ships land on thinner clusters (355–465 kg vs 593 kg for ship 1).
- Re-timing is exact on a 15-day lattice with a fixed visit order and per-leg mass profile;
  order changes are limited to single insertions, and the DP's leg model is the same
  inflated-Lambert proxy (per-pair calibrated after each SCvx pass), so 1 in 3–4 re-timed plans
  still fails certification and costs a 1–3 min attempt.
- Impulsive proxies with fixed inflation factors decide the beam; the tails (1.30× hops, 1.74×
  Earth-out, and up to 2.2× on Earth legs the fleet runs certified) mean a few plans per run
  fail SCvx certification and are skipped, not repaired.
- The SCvx leg solver is a Python/Clarabel CPU reference (2-day ZOH nodes); the fixed-pattern
  PDHCG CQP contract is not used yet, so no GPU timing claim exists.
- Best single ship is 592.6 kg vs ≈ 740 kg per archived reference ship (80 %); the 10-ship
  average is 440 kg (59 %) because ships 2–10 get the thinner clusters (355–490 kg). The reduced
  instance is intrinsically sparse (≈1–2 co-located candidates per hop vs ≈15 in the full
  catalogue) and plateaus at 5 asteroids / 314 kg with this search.

- Third campaign (§6.5): cooperative pricing works mechanically (54 foreign collects across
  the archived families, orphan repair, pool-consistent bundles) but never enters the emitted
  fleet: a collector flying to another ship's miners years later collects 280–330 kg where a
  self-cleaning ship in the same family collects 450–540 kg, so under the fleet rule the master
  always prefers the self-cleaning column. The references make cooperation pay because their
  collectors *also* deploy and their deployers *also* collect (every ship does both across a
  cluster several ships share); our deployer/collector split is still sequential (ship k prices
  after ship k−1 inside the family), which is the greedy loop moved one level down.
- Per-ship mass plateaued at 460–583 kg (average 505 kg) with 78 kg of propellant left on
  average: the good ships are propellant-bound, not time-bound — their Earth legs cost 371–618 kg
  (references 428–503 kg) and their hops 75–150 kg (references median 78 kg). Slots 4–5 of a
  family rarely close because the family's cheap Earth legs and co-moving members are used up
  by ships 1–3.
- The combinatorial master is exact only up to its node cap (273 columns need > 30 M nodes;
  `fleet_master_v2`, 436 columns, was not exhaustive at 5 M). Since the fifth iteration the LP
  relaxation and LP branch and bound close it: `fleet_master_v3` (726 columns) is proven optimal
  over its archive with a 4.5 kg LP gap. The proof is over the *archived* columns only — it says
  nothing about routes the pricing never generated.
- Fourth campaign (§6.6): of the three levers, the Earth leg was won (404 vs 460–474 kg per ship,
  70 kg below the references) and the two hop levers were not. The phasing-aware families make
  member pairs cheap at the collect epoch on paper (98 % under 75 kg in the probe) but the beam
  still builds deploy tours whose *re-flown* pairs cost 100+ kg three years later (collect hops
  102 kg median in the v4 fleet, 90 kg in the best fleet, references 66 kg; hops ≤ 75 kg 0.23 vs
  0.44–0.46), and the joint harvest — 38 attempts, none adopted — cannot buy it back inside
  13–40-member families. The Earth-leg saving was absorbed by more expensive deploy hops
  (129 vs 111 kg mean), leaving the v4 average at 498 kg.
- Fifth campaign (§6.7): the exact collect DP raised the fleet average from 520 to 549 kg and
  the admitted fleet from 16 to 18 ships, but it did so by cutting the collect-hop *tail* (p90
  152 → 140 kg, mean 95 → 90) and packing one more asteroid per ship — the median collect hop is
  87 kg against 66, and our collect hops are 240 d long against 181–187 d. The DP prices the
  order and epochs exactly for the members the beam deployed; it cannot make members that are
  4–6 km/s apart at the harvest epoch cheap. The collect-epoch families produced ships of the
  same quality as the deploy-epoch ones (8 vs 5 in the best fleet at 3 vs 4 workers) — the
  radius-2.0 families are still 20–99 members wide (median 49–53) and both weightings pick from
  the same neighbourhoods. The Earth return got dearer (227 vs 197 kg; references 208–216) because the DP
  ends the tour as late as the lattice allows and the return TOF grid (300–900 d in 60 d steps)
  is coarse. The joint harvest still adopts almost nothing (2 of 65 attempts); no cooperative
  column is in the emitted fleet.
- Sixth campaign (§6.8): the calibrated two-pass DP and the tighter, 5-slot families raised the
  fleet from 18 to 19 ships (549 → 563 kg average, 9 archived ships above 563 kg where there was
  one) but the collect hop moved only from 87 to 84 kg median and its TOF not at all (240 d): the
  DP prices the members the beam deployed more truthfully, it still does not *choose* members
  for their harvest-epoch phase, and the 15-day lattice only matters once such members exist.
  The "phase-optimal window" search of the work list is therefore only partly done — the DP
  searches every lattice epoch and every order, but the deploy beam that decides which asteroids
  are in the tour ranks them on the deploy-epoch Lambert cost plus the DP's completion, never on
  a relative-phase zero-crossing at the collect epoch. The Earth return got dearer for the second
  time (279 kg mean; references 208–216) because the DP objective `Σ collected − w × propellant`
  at `w = 1` happily spends 60 kg of return propellant for a later last collect worth 60 kg of
  mass — neutral for the ship, but the fleet rule wants mass, not neutrality, and the joint
  re-timer with SCvx in the loop then certifies that expensive return as is. Memory: the measured
  process-tree PSS peak was 3.04 GB for 4 workers against the 2 GB target (worker RSS 0.9–1.1 GB
  in the collector slots, 110 MB baseline; Python-owned arrays stay under 80 MB in the beam, so
  the transient is native — Lambert batches of the harvest seeding and Clarabel workspaces — and
  has not been localised yet); 3 workers would meet the budget at 0.75× the throughput.
- Seventh iteration (§6.9): the transient was *not* native — it was the collect DP's per-mass
  propellant-fraction cache plus glibc heap retention, both fixed (single slot 695 → 329 MB,
  bit-identical routes). The SCvx return sweep re-timed 13 of 36 archived ships (+140.5 kg) and
  the master admitted the 20th ship at 576.07 kg average — 0.5 kg above the rule's 575.6, so the
  fleet is again binding on the rule (`fleet_master_v6`, over the seventh campaign as well:
  575.78 kg, 20 ≤ 20.01). The return of the best fleet is 216.5 kg mean (references 208–216) at
  450 d median TOF (references 473–486): the sweep only re-times ships whose tour is fixed; the
  DP that builds new tours got the TOF model and now returns at 480 d median (v7) but still pays
  233 kg for it. The harvest-window deploy ranking lost mass at every weight — the collect hop
  (84.5 kg / 225 d vs 66 kg / 181 d) is unchanged and is now the whole gap. Three workers price
  0.7× the families of four in the same 4 h (v7's own fleet 9920 kg vs v6's 10 698); the archive
  master is what turns that into score. The archive-wide master is proven optimal over 1032
  columns; the LP at 21 ships is infeasible.
- Eighth iteration (§6.11): jointly re-optimising every epoch of a certified ship is worth
  +10.4 kg per ship on average (+1.1 to +23.3; 32 of 32 archived ships improved, 95 % of the
  SCvx certifications accepted, 7 min for 32 ships on 3 workers) — enough, with the master's
  re-composition, to admit the **21st ship at 587.93 kg average (`fleet_master_v7`, 12 346.5 kg,
  +830.8 kg, proven optimal)**, and the fleet is binding on the rule again (21 ≤ 21.007). The
  gain is a redistribution — deploy hops +59 kg per ship buy mining time, return −12, collect −4,
  margin spent to zero — not a cheaper flight, and the search saturates once the margin is gone:
  the ≤ 8 d meshes propose nothing because a moved leg is priced at its calibrated residual plus
  a 3 % safety margin. Inserting a 9th/10th asteroid into a converged ship failed on every one of
  the 32 ships: the co-moving neighbourhood (radius 2.5) offers 16–47 members but each hop to
  them costs 6–15 km/s Lambert (authority ratio 2–6 vs 0.55), i.e. the beam already took every
  reachable member of these families. The next ship needs a *new* set, not a re-timed one.

Next bottleneck: the collect hop, inside the DP's window. The fleet rule `N ≤ 2 exp(0.004 M̄)`
is binding at 21 ≤ 21.007 (average 587.93 kg); a 22nd ship needs 599.5 kg average (6 ships of the
fleet are above it), 23 ships 610.6 kg, the references' 740 kg would admit 38. Per ship the gap
to a reference ship is now: collect hops 8 × (85 − 66) ≈ 150 kg, Earth return ≈ 0, deploy hops
and Earth-out *better*. The joint re-optimiser should be run after every future campaign
(`joint-itinerary`, 7 min for 32 ships) before the master, and its evaluator is the natural
place to price member substitution once the DP proposes it. (i) **Collect-hop phase in the
DP**: the deploy beam cannot fix it
(ranking on the harvest-window cost loses the deploy chain); the DP should be allowed to *drop*
a deployed miner from the tour when its harvest hop is dear and re-deploy the slot's time at a
neighbour whose phase crosses zero at the harvest epoch — i.e. member substitution inside
`plan_collect_tour`, priced from the same pair table, then the two-pass mass schedule; target
70 kg / 180 d, ~150 kg per ship. (ii) **Return TOF in the DP**: the sweep shows 470–540 d
returns at 172–215 kg exist for most ships; give the DP the sweep's cells for the camp asteroid
(cheap: one sweep per family, reused by every slot) instead of the model, so new tours end
earlier and return long. (iii) Run the return campaign after every cluster campaign
(`retime-returns` is 50 min for 36 ships on 3 workers) before the master.

- Eighth iteration (§6.10): (ii) is done and closes the return gap (v8 fleet 205 kg / 495 d vs
  references 208–216 / 473–486; the sweep inside the pricing reproduces the post-hoc
  `return_sweep_v2` gains inline, so (iii) is no longer needed for new campaigns). (i) as
  specified — a substitute miner priced from the pair table, exact re-fly, DP re-solve — is
  implemented and is a *small* positive: 15 accepted swaps in 59 beams, +122 kg in total, +2 kg
  per ship against the ~150 kg target. The reason is structural: the beam deploys where the
  deploy hop is cheap, so the dear collect hop is the price of a cheap deploy hop; every
  substitute that would harvest cheaper costs +150–250 kg on the deploy side (family 7 probe),
  and the ones that pay are the few whose predecessor happens to have two cheap neighbours.
  The collect hop is unchanged (86.8 kg / 210 d vs 66 / 181–187) and the archive-wide master
  is unchanged (the v8 columns are duplicates or conflicts of richer archived ships; the rule
  needs an 828 kg 21st ship). The campaign's own fleet did rise (9920 / 18 → 10 698 / 19 on
  the same partition, best ship per family +8 kg median in 15 of 18 families), but the machine
  was shared with two other agents' runs (20 families priced against v7's 25).

Next bottleneck (after the eighth iteration): the collect hop is a *deploy-chain* property, not
a tour property — the references' 66 kg / 181 d hops come from chains whose consecutive miners
are also cheap harvest pairs (their deploy hops cost 837–851 kg per ship against our 768–773:
they pay ~70 kg more on deploy to save ~150 kg on collect). Neither the deploy-time harvest
ranking (§6.9, lost the chain) nor the post-beam substitution (§6.10, pays it back on deploy)
moves the chain, because both price one pair at a time against a chain the beam has already
closed. (i) **Chain-level objective in the beam**: score a partial chain by *deploy propellant +
the DP's actual collect-tour cost of the chain so far* (a Held-Karp on ≤ 9 nodes is 5 ms with the
pair table; the beam expands ~2000 partials per level, so ~10 s per level), instead of deploy
propellant + a per-pair harvest surcharge; the beam then trades the deploy kg for the collect kg
at the exchange rate the DP measures rather than a fixed weight. (ii) **Reference-chain
statistics as a prior**: the archived references give 1000+ (deploy pair, harvest pair) tuples;
fit the joint cost of a chain step from them and use it as the beam's expansion order. (iii) The
fleet rule makes the 21st ship a *whole-fleet* problem (every ship +12 kg): run the master's
LP duals back into the pricing (a per-asteroid price the family search subtracts, as a column
generation step would) so each family prices the ships the master would actually take, and
price on the Lambda box's 26 cores where a 4 h budget covers all 35 families of the partition
instead of 20–25.

- Ninth iteration (§6.12): the chain-level objective is implemented as specified (exact DP tour
  of every shortlisted partial, closure judged by the exact forward pass, deterministic, cached)
  and is *neutral* on the paired campaign (7 families up, 6 down, median 0.0 kg): the heuristic
  and the DP rank the same chain first on most families because both are dominated by the
  collected mass at the beam's 0.15 exchange rate, and the alternative chains the references fly
  (dearer deploy, cheaper harvest) are either not among a level's ~8 500 children or not
  better by the DP's own measure. The collect hop moved 97.5 → 90.6 kg median across the
  campaign's ships and the deploy hops +13 kg per ship — the reference direction — without more
  ore. The reference-chain prior is data (112 ships, 1 014 collect hops) and its penalty is
  exact and monotone, but at weight 0.5 it did not change the winners either. The LP duals are
  correct column-generation prices and cannot steer a campaign whose families are disjoint from
  the archive's contested asteroids: 22–24 priced asteroids at N = 21, one inside a priced
  family; the bound-share variant prices 61 and is still dominated by the ship-count dual. The
  joint itinerary inside the pricing applies to self-cleaning slots only (11 of 60); as a
  post-pass over the best 40 stand-alone v8/v9 ships it is the lever that moved the master:
  +411.5 kg, three swapped fleet ships, `fleet_master_v8` 12 356.30 kg (+9.8 over `v7`). The
  wrapper timeout (`timeout 15600`, sized on v8's 249 min) killed `cluster_fleet_v9` at 260 min
  before its final master; the last verified incumbent (9960.3 kg / 18 ships) stands and every
  emitted ship is archived. One family (10) crashed on a NaN burn schedule reaching
  `plan_collect_tour` from the substitution pass (guarded since: a non-finite burn falls back to
  the two-pass schedule; the NaN's origin in the pass-1 tour is not localised).

Next bottleneck (after the ninth iteration): the fleet's deploy spend now equals the
references' (838.5 vs 837–851 kg per ship) and the return is closed (204 vs 208–216), yet the
collect hop is still 84 kg / 210 d against 66 / 181–187 and the ore 588 vs 730 kg per ship. The
remaining differences are (i) the **Earth-out leg**: ours 407 kg over 580–610 d against 460–474 kg
over 490–565 d — the references spend ~60 kg to arrive ~100 d earlier, which shifts the whole
deploy chain and buys ~2.7 kg per asteroid-year × 9 asteroids; our certified Earth legs are
optimised for propellant with a time term that stops at ~590 d, and every chain of a family
inherits them; (ii) the **collect TOF**: 210 vs 181–187 d median with the same lattice — the DP
picks long hops because at the calibrated inflation a 240-day hop is cheaper than a 150-day one
for the phase our chains present at harvest; the references' consecutive miners sit within
|Δλ| 2.7° (p75 4.8°) at the collect departure (`chain_prior_v1.json`), a geometric target the beam
does not see (the prior prices propellant, not phase). The chain-level objective should therefore
score the *phase alignment of consecutive deploys at the projected harvest epoch* (Δλ from
`CollectPairTable.pair_geometry`, available for free) and the Earth-leg optimiser should trade
propellant for arrival time at the fleet's exchange rate (≈ 0.03 kg of ore per asteroid-day).
(iii) Ship 22 needs +11.1 kg on every ship at once; the joint itinerary is the only lever that has
moved the whole fleet (+208 kg in `v2`, +411 kg over v8/v9 in `v3`), so it should run over every
archived stand-alone ship (992 columns, ~3 h on 3 workers) before the next master.

## 9. How this feeds Paper 2 / OrbitWeaver

The track provides (i) an external, officially scored objective for the integrated route +
trajectory oracle, (ii) an exact independent scorer that reproduces the official one, (iii) a
frozen reduced instance for preregistered comparisons, and (iv) a leg-level SCvx transcription
whose convex subproblem (states, ZOH controls, virtual control, SOC thrust cone, box trust regions)
is the natural fixed-pattern CQP for the persistent PDHCG backend. The reference registry entry is
[`benchmarks/gtoc12/reference_reproductions.json`](../benchmarks/gtoc12/reference_reproductions.json)
in the `literature_baselines.json` profile format.
