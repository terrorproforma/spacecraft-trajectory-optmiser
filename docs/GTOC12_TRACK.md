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

`search.py`: deterministic beam search over self-cleaning routes
`Earth → A₁ → … → A_k (deploy) → camp → A_k → … → A₁ (collect) → Earth`. Deploy hops expand
forwards from a launch grid (Earth legs 300–900 d, hops 60–420 d, optional waits for phasing)
over element-space neighbours with a phase-drift penalty; the collection tour is scheduled
backwards from the window end with per-hop wait windows; Earth returns may arrive in the last 600
days. Costs are Lambert rendezvous ΔV (6 km/s Earth allowance credited) inflated ×1.6 (Earth legs)
and ×1.25 (hops) against a 0.85-duty thrust authority. Beams cap variants per deployed set and
prune first asteroids without a feasible return. Ties break on asteroid ID; no randomness.

`pipeline.py`: each planned leg becomes an `ArcRequest`; `G3TrajectoryOracleAdapter` owns one
`Gtoc12ScvxDriver` per topology group; `BoundedScheduler` orders the work; the certified legs form
a `RouteDefinition` column and pass `solve_certified_route_master`. Collected masses start at the
rule maximum and are scaled down if the final-mass rule (`m_f ≥ 500 kg + carried`) would fail.
The route is emitted as an official file, scored by both verifiers, and exported for the viewer.

CLI: `spacepdhcg gtoc12 run --run-id <id> --output <dir> [--beam-width 24 --max-deploys 4
--refine-top 3 --search-only --full-catalogue]`.

## 7. Results (all CPU, 16-core WSL2, single process)

| Run | Instance | Ships | Asteroids | Collected mass (official) | Fixed-bonus score | Refined arcs | Search wall | Refine wall | Total wall | Hardware |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `reduced-v1-run1` (beam 24, ≤4 deploys, before phasing fixes) | gtoc12-reduced-v1 | 1 | 2 (36777, 37351) | **195.044 kg** | 195.044 kg (both B = 1) | 4 | 18.5 s | 7.0 s | 37 s | CPU |
| `reduced-v1-run2` (beam 24, ≤4 deploys, phasing-aware) | gtoc12-reduced-v1 | 1 | 4 (1265, 21191, 27292, 40808) | **253.744 kg** | 249.059 kg | 8 | 24.5 s | 6.0 s | 47 s | CPU |
| `full-catalogue-run1` (beam 8, ≤3 deploys, 39.1 M Lambert screens) | full catalogue (60,000) | 1 | 3 (20194, 23644, 15033) | **249.035 kg** | 202.995 kg | 6 | 956 s | 7.1 s | 963 s (11.2 GB peak RSS) | CPU |

Per-leg detail of `reduced-v1-run2` (propellant, SCvx iterations, solve time, certified endpoint
error): E→1265 600 d 364.7 kg 9 it 1.3 s 0.32 km; 1265→21191 360 d 128.7 kg 7 it 0.5 s 0.04 km;
21191→27292 360 d 153.5 kg 5 it 0.3 s 0.04 km; 27292→40808 120 d 97.4 kg 24 it 0.7 s 0.01 km;
40808→27292 420 d 229.8 kg 7 it 0.6 s 0.02 km; 27292→21191 420 d 90.5 kg 5 it 0.4 s 0.05 km;
21191→1265 420 d 266.3 kg 6 it 0.5 s 0.09 km; 1265→Earth 550 d 235.8 kg 9 it 1.1 s 0.46 km. Final
mass 1273.3 kg ≥ 500 kg + 253.7 kg carried. Run 2's other two refined candidates also verified
officially (252.923 kg and 241.697 kg).

Full catalogue: screening the 60,000 asteroids over 25 launch epochs × 13 Earth-leg TOFs
(39.1 M zero-revolution Lambert solves, both branches) took 956 s and 11.2 GB; only 16 beam
expansions and 13 completed candidates fit the 40-minute cap, and 11 of the 16 multi-asteroid
partials still failed the collection tour. The single refined route (6 arcs, 7.1 s) verified
officially at 249.035 kg but, unlike the reduced-instance asteroids, its three targets were mined
during the competition, so the fixed-bonus score is 202.995 kg. Where time ended: the search, not
the refinement; catalogue-scale neighbour selection is still an O(N) element-space proxy per node
and the Earth-return grid is recomputed per first asteroid. Failures are retained in
`results/gtoc12/*/search.json`.

Independent verifier on the scored files: max propagation error ≤ 0.46 km, ≤ 8e-5 m/s, ≤ 3e-11 kg;
official and independent per-asteroid masses agree to 1e-10 kg. Fleet rule: 1 ship ≤ 2 exp(0.004 M̄)
(≈ 4.4 for 195 kg).

Artifacts (ignored `results/gtoc12/<run>/`): `search.json`, `refinements.json`, `run_report.json`,
`candidate_NN/Result.txt`, `candidate_NN/route_summary.json`,
`candidate_NN/viewer/{trajectories.json,manifest.json}`. The viewer export follows the
`web/trajectory-viewer` record schema (family `GTOC12`, heliocentric frame, replay decimated to
≤ 512 exact propagated samples with events preserved, event-state transcription, asteroid/Earth
context orbits).

## 8. Limitations

- Single self-cleaning ship; no cross-ship deploy/collect, no gravity assists, zero-revolution
  Lambert screening only (multi-revolution legs and 2-year+ phasing are unexplored).
- Impulsive proxies with fixed inflation factors decide the beam; the refined masses matched the
  proxies within ~1 kg on the scored routes, but proxies still reject many chains.
- The SCvx leg solver is a Python/Clarabel CPU reference (2-day ZOH nodes); the fixed-pattern
  PDHCG CQP contract is not used yet, so no GPU timing claim exists.
- Scores are far below the archived references (≈ 700 kg/ship): this is a first working
  end-to-end pipeline on a 1000-asteroid preregistered subset, not a competitive entry.

## 9. How this feeds Paper 2 / OrbitWeaver

The track provides (i) an external, officially scored objective for the integrated route +
trajectory oracle, (ii) an exact independent scorer that reproduces the official one, (iii) a
frozen reduced instance for preregistered comparisons, and (iv) a leg-level SCvx transcription
whose convex subproblem (states, ZOH controls, virtual control, SOC thrust cone, box trust regions)
is the natural fixed-pattern CQP for the persistent PDHCG backend. The reference registry entry is
[`benchmarks/gtoc12/reference_reproductions.json`](../benchmarks/gtoc12/reference_reproductions.json)
in the `literature_baselines.json` profile format.
