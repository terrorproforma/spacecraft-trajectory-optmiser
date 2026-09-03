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
`python scripts/gtoc12/fetch_gtoc12_data.py` (or `spacepdhcg gtoc12 fetch`) into the ignored
directory `benchmarks/gtoc12/data/`. No multi-megabyte dataset is committed.

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
| **`fleet_master_v2`** (master over every archived certified route of all six runs: 330 routes re-flown through SCvx, 436 columns, 5 M nodes) | full catalogue | **16** | **123** (116 mined, 7 miners left) | **8324.27 kg** (per-ship table below) | **7905.05 kg** | 246 | — | 749 s re-certification (6 workers) + 103 s master | 872 s, 0.29 GB main | CPU |

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
- The master is exact only up to its node cap: 273 columns need > 30 M nodes for a proof; with
  5 M nodes the gap to the proven optimum is unknown (30 M moved the objective by 0.03 %).
  `fleet_master_v2` (436 columns) is likewise not exhaustive at 5 M nodes.
- Fourth campaign (§6.6): of the three levers, the Earth leg was won (404 vs 460–474 kg per ship,
  70 kg below the references) and the two hop levers were not. The phasing-aware families make
  member pairs cheap at the collect epoch on paper (98 % under 75 kg in the probe) but the beam
  still builds deploy tours whose *re-flown* pairs cost 100+ kg three years later (collect hops
  102 kg median in the v4 fleet, 90 kg in the best fleet, references 66 kg; hops ≤ 75 kg 0.23 vs
  0.44–0.46), and the joint harvest — 38 attempts, none adopted — cannot buy it back inside
  13–40-member families. The Earth-leg saving was absorbed by more expensive deploy hops
  (129 vs 111 kg mean), leaving the v4 average at 498 kg.

Next bottleneck: the collect hop. Per-ship mass is still the only lever — the fleet rule
`N ≤ 2 exp(0.004 M̄)` is exactly binding at 16 ≤ 16.03 (average 520 kg); every 25 kg of average
buys one more ship (535 kg → 17, 600 kg → 22, the references' 740 kg → 38), and the master will
take any ship above the current average and drop anything below it. Earth legs (404 kg) and
Earth returns (197 kg) are now at or below the reference cost, deploy hops are within 10–20 kg of
it, and the *whole* remaining gap is the collect phase: 90–102 kg per collect hop against 66 kg,
i.e. 170–250 kg per ship over 7 hops, exactly the distance from 500 to 740 kg. What the
references do that the pipeline does not: (i) deploy tours are chosen so that the *collect* tour
is cheap — their collect hops (181–187 d, 66 kg) are as cheap as their deploy hops, which means
the deploy order and epochs are optimised for the return sweep, not the outbound one; the
collect look-ahead tried here (Lambert at +3 years) does not predict the DP's cheap tours, so the
next step is to price the collect tour *exactly* in the beam (run the DP re-timer on the
deploy+collect chain of each surviving partial, or a table of certified collect-pair costs at the
collect epoch per family) rather than by proxy; (ii) clusters tight enough that the pool of
collectable miners at the collect epoch is dense (Δa spread 0.02–0.03 AU across the ships that
share a cluster) — the phasing-aware radius 2.0 families are 13–40 members over 0.06–0.10 AU,
so re-clustering at radius ≤ 1.0 with the collect-epoch phase weighted higher than the deploy-
epoch phase is the family-generation change to make; (iii) the master's node cap (5 M) on 436
columns — a column-generation dual/LP bound, or pruning dominated columns before branching,
would make it exhaustive and let the incumbent be certified optimal over the archive.

## 9. How this feeds Paper 2 / OrbitWeaver

The track provides (i) an external, officially scored objective for the integrated route +
trajectory oracle, (ii) an exact independent scorer that reproduces the official one, (iii) a
frozen reduced instance for preregistered comparisons, and (iv) a leg-level SCvx transcription
whose convex subproblem (states, ZOH controls, virtual control, SOC thrust cone, box trust regions)
is the natural fixed-pattern CQP for the persistent PDHCG backend. The reference registry entry is
[`benchmarks/gtoc12/reference_reproductions.json`](../benchmarks/gtoc12/reference_reproductions.json)
in the `literature_baselines.json` profile format.
