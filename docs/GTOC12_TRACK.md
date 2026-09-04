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
--search-only --full-catalogue --pool-a-min 2.2 --pool-a-max 3.0 --pool-e-max 0.15 --pool-i-max 8]`.

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

## 7. Results (all CPU, 16-core WSL2, single process)

| Run | Instance | Ships | Asteroids | Collected mass (official) | Fixed-bonus score | Refined arcs | Search wall | Refine wall | Total wall | Hardware |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `reduced-v1-run1` (beam 24, ≤4 deploys, before phasing fixes) | gtoc12-reduced-v1 | 1 | 2 (36777, 37351) | **195.044 kg** | 195.044 kg (both B = 1) | 4 | 18.5 s | 7.0 s | 37 s | CPU |
| `reduced-v1-run2` (beam 24, ≤4 deploys, phasing-aware) | gtoc12-reduced-v1 | 1 | 4 (1265, 21191, 27292, 40808) | **253.744 kg** | 249.059 kg | 8 | 24.5 s | 6.0 s | 47 s | CPU |
| `full-catalogue-run1` (beam 8, ≤3 deploys, 39.1 M Lambert screens) | full catalogue (60,000) | 1 | 3 (20194, 23644, 15033) | **249.035 kg** | 202.995 kg | 6 | 956 s | 7.1 s | 963 s (11.2 GB peak RSS) | CPU |
| `reduced_v1_search3` (search v2: beam 48, ≤8 deploys, 64 neighbours) | gtoc12-reduced-v1 | 1 | 5 (37351, 36777, 44316, 6249, 1128) | **314.442 kg** | 314.442 kg | 10 | 194 s | 34 s (3 candidates) | 228 s, 0.55 GB | CPU |
| `full_catalogue_search2` (search v2, beam 32, ≤10 deploys, 64 neighbours, pool 10,612) | full catalogue | 1 | 8 (8846, 27861, 37385, 49900, 8123, 1122, 12992, 57949) | **548.282 kg** | 548.282 kg | 16 | 261 s | 42 s (2 candidates) | 303 s, 0.66 GB | CPU |
| `fleet3_full_catalogue` (search v2 pre-final, 3 ships greedy, beam 24, ≤10, 48 nb) | full catalogue | 2 (ship 3 uncertified) | 13 | **965.804 kg** | 893.263 kg | 26 | 719 s (3 searches) | 32 s | 751 s, 0.75 GB | CPU |
| `fleet3_full_catalogue_v2` (final code, 3 ships greedy, beam 32, ≤11, 64 nb, 1800 s budget/ship) | full catalogue | 3 | 20 | **1394.11 kg** (548.28 + 442.22 + 403.61) | 1318.117 kg | 40 | 814 s (3 searches) | 53 s (5 candidates) | 867 s, 0.77 GB | CPU |

Runs are single-process CPU (16-core WSL2, load shared with an unrelated G4 GPU campaign; the
RTX 5090 was at 100 % throughout and was not used). "Search v2" is the position-space,
reserve-pruned beam search of §6.2; `search2`/`fleet3_full_catalogue` were produced at an
intermediate commit of it (scalar Δe/Δi pre-ranking, no failed-leg skipping) and are kept as
verified artifacts; `reduced_v1_search3` and `fleet3_full_catalogue_v2` are reproducible from HEAD
(`--beam-width 48 --max-deploys 8 --neighbours 64 --refine-top 3` and `--full-catalogue --ships 3
--beam-width 32 --max-deploys 11 --neighbours 64 --refine-top 3 --stop-at-first-certified
--search-budget-seconds 1800`). Ship 1 of the final fleet reproduces the `search2` route exactly.
Best score by depth (proxy kg) for the final ship 1: 1→124.5, 2→221, 3→285, 4→338, 5→404,
6→445, 7→531, 8→548; depths 9–11 produced no completable chain.

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
context orbits).

## 8. Limitations

- Self-cleaning ships only (the JPL file shows 279/320 cooperative collections; cross-ship
  deploy/collect is not modelled); no gravity assists; zero-revolution Lambert screening only
  (justified by the references, but long collect hops of 600–720 days are then screened as
  single-arc transfers).
- Greedy fleets: ship k never revisits ship 1..k−1's choices, so the second and third ships land
  on thinner clusters (442, 404 kg vs 548 kg). A joint fleet assignment (column generation over
  ship routes) is the natural next step for the G7 master.
- Time re-optimisation is at the grid level only (15-day collection scheduling, 30-day launch and
  deploy-wait grids); legs are not re-timed jointly after refinement, and the 230–430 kg of
  propellant left in every certified route is not reclaimed.
- Impulsive proxies with fixed inflation factors decide the beam; the tails (1.30× hops, 1.74×
  Earth-out) mean a few plans per run fail SCvx certification and are skipped, not repaired.
- The SCvx leg solver is a Python/Clarabel CPU reference (2-day ZOH nodes); the fixed-pattern
  PDHCG CQP contract is not used yet, so no GPU timing claim exists.
- Best single ship is 548 kg vs ≈ 740 kg per archived reference ship (74 %); the reduced instance
  is intrinsically sparse (≈1–2 co-located candidates per hop vs ≈15 in the full catalogue) and
  plateaus at 5 asteroids / 314 kg with this search.

Next bottleneck: candidate generation still yields 240–300-day deploy hops because the pool is
ranked per hop rather than per *cluster*; building tight co-located clusters (Δa ≤ 0.04 AU,
phase ±5° over the whole deploy window) up front and searching orders within them, plus a joint
re-timing pass that spends the unused propellant on shorter hops, is what stands between 8 and
10 asteroids per ship.

## 9. How this feeds Paper 2 / OrbitWeaver

The track provides (i) an external, officially scored objective for the integrated route +
trajectory oracle, (ii) an exact independent scorer that reproduces the official one, (iii) a
frozen reduced instance for preregistered comparisons, and (iv) a leg-level SCvx transcription
whose convex subproblem (states, ZOH controls, virtual control, SOC thrust cone, box trust regions)
is the natural fixed-pattern CQP for the persistent PDHCG backend. The reference registry entry is
[`benchmarks/gtoc12/reference_reproductions.json`](../benchmarks/gtoc12/reference_reproductions.json)
in the `literature_baselines.json` profile format.
