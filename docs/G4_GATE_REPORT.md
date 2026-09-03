# Gate G4 report — adaptive and hybrid matched-quality study

Status date: **2026-09-03**
Decision: **FAIL** (historical evidence unchanged; claim core in progress)
H5: **unresolved**
H6: **unresolved**
G5 authorised: **no**

## H5/H6 claim-core campaign (2026-09-03): launched, not yet resolved

The preregistered 360-group, 3,240-invocation claim core (`benchmarks/g4_h5_h6_claim_core.json`,
SHA-256 `40dc217467ffe32e919d4f901943e0200f69e302cf57cd15ccdfa88bfa0c8d0b`) was launched on
`integration/single-gpu-v1` after the current-head G0–G3 seal (`results/gpu/current-head-b0cd570/`,
evidence index SHA-256 `83b643bf81773b59941f7d7226a71f9283e535d1332b64b435f1acd6f2ba9e53`, source
`b6afb49`). It resolves H5 and H6 only and may not populate any F01–F12 or T01–T08 regime product.
Nothing in this section changes the historical FAIL evidence below, and the full 2,764,800-group
grouped ledger has not been started.

Executable provenance:

- The G3-sealed release executable (`4273cd8a…`) was verified against the report-only head
  `9678134`: `cmake --build` reported `ninja: no work to do` and the code paths were identical to
  `b6afb49`.
- The pre-campaign pilot found a genuine executor defect: a P1-E `N=100` fixed-tight group ran the
  full 91-minute nine-attempt safety boundary without emitting one attempt record because
  `spacepdhcg_cuda_workspace_wait` held the workspace mutex across `cudaEventSynchronize`, which
  blocked the deadline watchdog's `spacepdhcg_cuda_workspace_cancel` until the persistent kernel
  exhausted its 1,000,000-iteration budget. Commit `9a4cbea` releases the mutex while waiting,
  reports a cancelled inner solve as `SPACEPDHCG_CUDA_SCVX_CANCELLED` (an honest launched
  `timeout` with the spent time and iterations), forces a cold boundary after a cancelled attempt,
  adds a native cross-thread cancel-during-wait check to `recovery_test`, and pins the dynamically
  linked `libspacepdhcg_cuda.so` in the capability. The release tree passed 62/62 tests after the
  fix; a 20 s-deadline `--g4-session` on claim-core group 0 emitted nine strictly schema-valid
  `timeout` attempts at 20.0 s each. The G3 seal predates this library change and was not
  re-run; the sanitizer evidence therefore covers `b6afb49`, not `9a4cbea`.
- Scheduler commits `2e34d30`/`a68890b` add GPU contamination control: nvidia-smi samples at every
  group boundary, `/dev/dxg` holders inside WSL2 (excluding the worker's own descendants and
  holders whose `CUDA_VISIBLE_DEVICES` hides every device), and host `nvidia-smi.exe pmon`
  compute contexts sampled every second during the group. A group observed alongside foreign
  compute activity is quarantined as `contaminated` with all raw evidence retained and re-run
  after the foreign activity ends. Commit `44e6939` adds `scripts/gpu/decide_g4_claim_core.py`
  (group re-validation, publication aggregates, and the frozen H5/H6 decision functions).
- The first official group under `9a4cbea` (checkpoint `g4-claim-core-9a4cbea`, retained as
  evidence) exposed the residual gap: its attempts each reached the 600 s deadline with
  1,000,000 PDHCG iterations and canonical residual ≈37 (far from the 1e-6 tier), but one
  attempt ran more than 20 minutes and the group overran the safety boundary, discarding all nine
  attempts. A cancel that fired while the workspace was between inner solves could not reach the
  device, and the following inner solve or identical-CQP re-solve spent its full budget first.
  Commit `26def2b` checks the driver's cancellation flag before every inner solve and re-solve,
  rolls back and reports a cancelled re-solve as `CANCELLED`, makes the watchdog re-assert the
  cancellation every second until the attempt returns, and moves the scheduler's outer boundary
  300 s beyond the executor's own group deadline. Release ctest passed 62/62; 20 s-deadline
  `--g4-session` runs on claim-core groups 0 (fixed-tight), 1 (adaptive) and 3 (hybrid) each
  emitted nine strictly schema-valid launched `timeout` attempts at 20.0 s. The generation-0
  silence of that first group (no attempt record in 91 minutes while foreign host compute ran at
  up to 99 % SM) is recorded as an unexplained observation; the same group's generation 1
  produced seven 600.1 s `timeout` attempts before the overrun.
- Official capability: `/home/angus/g4-executor-capability-26def2b.json`, capability SHA-256
  `93b6dac4c5035e9510db9d2c91b9e53ba6d8943e4c3be9947dd3cbaa5e868903`, source commit `26def2b`,
  executable SHA-256 `16c1883f16f78bdfa4bbd00d341e1b0c90882ce2319742e358535e2a24f4923e`,
  `libspacepdhcg_cuda.so` SHA-256
  `bf31e1249af45e66d23b31ed559201402652048aaf6e16720060edbfa4a4823b`, real CUDA session probe
  passed. The superseded `9a4cbea` capability (`e546583b…`) executed no completed group. Policy
  SHA-256 `9ab3b444…`, matrix SHA-256 `50afe8ff…`, tolerances, seeds, repeats and order are
  unchanged.

Campaign state at the time of this report:

- Checkpoint `build-integration-report/g4-claim-core-26def2b` (ignored, local-only), initialised
  with `run_g4_campaign.py init --claim-core`; 0 of 360 groups completed. The checkpoint pins
  `source_commit=26def2b`; a detached worktree at that commit
  (`/home/angus/worktrees/spacepdhcg-g4-claim-core-26def2b`) hosts the restart, status,
  observer and completion scripts (`build-integration-report/g4-claim-core-26def2b-*.sh`) so this
  branch can advance without invalidating the pin.
- The RTX 5090 is shared with other agents' GPU jobs (WSL and Windows-side); the worker waits for
  foreign compute activity before claiming a group and re-runs contaminated groups. Fixed-tight
  P1-E `N=100` attempts progressed at 0.5–1 ms per PDHCG iteration depending on contention, so
  groups whose attempts reach the frozen 600 s deadline take about 91 minutes each; dispositions
  are recorded only from launched attempts and are not predicted here.
- On completion, `build-integration-report/g4-claim-core-26def2b-finish.sh` re-validates every
  raw attempt and publication aggregate, applies the preregistered H5/H6 decision functions
  (paired bootstrap, seed 20260901 plus scale, 10,000 samples, thresholds unchanged), and seals a
  reproducible archive with an evidence index under `results/gpu/g4/claim-core-26def2b/`
  (local-only; no immutable URI exists).

Amendment `single-gpu-v1.1` and relaunch (2026-09-03): the `26def2b` worker was paused with 0 of
360 groups completed (its one quarantined `contaminated` fixed-tight group and the wait /
contamination logs are retained under `g4-claim-core-26def2b`). Before any group result was
inspected, the preregistered amendment `benchmarks/g4_claim_core_amendment_v1_1.json` (SHA-256
`c691467e77367c63d2ba4b0adc1b290d3e4d731f360cbccae45a7d3cf5b8a1f5`, document
`docs/G4_CLAIM_CORE_AMENDMENT_V1_1.md`) was frozen and applied to the claim core only:

- Contamination policy run-and-flag (Decision A): the worker never waits for GPU idle; foreign
  SM utilisation is sampled before, during and after every attempt, the utilisation deltas and the
  shared lock file `/home/angus/.spacepdhcg-gpu.lock` are recorded, and each measured attempt
  executed alongside foreign compute is flagged `contaminated` (disposition and quality retained,
  timing and energy excluded from every H5/H6 statistic; each coordinate row reports the pair count
  n actually used and its censoring; no re-run).
- Deterministic-replay timeouts: when both warm-ups and `measured/0` reach the attempt deadline
  with bit-identical FNV-1a trace hashes, `measured/1..6` are recorded as
  `timeout_deterministic_replay` (censored, not executed, referencing `measured/0`); any trace
  difference executes all seven.
- Attempt deadline 600 s → 120 s and inner iteration cap 1,000,000 → 200,000 for the claim core; a
  hash-selected 10 % stratified subset (36 `censoring_sensitivity` twins, family × scale × policy,
  committed in the amendment) additionally runs at 600 s / 1M immediately after its core group. The
  preregistered acceptance rule (a twin qualifying where its 120 s core attempt is censored
  invalidates the amendment and reverts the whole core to 600 s) is enforced by
  `decide_g4_claim_core.py` (exit 2, decision withheld).
- Execution order: pure-gpu-ipm → adaptive → hybrid-pdhcg-ipm → fixed-tight, definition order
  within a policy. The claim core never bound execution order to the `solver_order` rotation (it
  is recorded per group as an identity axis only), so the reordering is permitted. Tolerances,
  seeds, repeats, group identities and quality gates are unchanged; `policy_amendment:
  single-gpu-v1.1` is echoed in the checkpoint metadata and every raw and group record.

Two defects were found and fixed before relaunch: the executor bakes `SPACEPDHCG_SOURCE_COMMIT`
at CMake configure time, so a rebuilt-only tree emitted `identity.repository_commit=b6afb49` under
a campaign pinned at the live head (commit `2ef27e1`: the executor reports
`compiled_source_commit`, and the capability generator and scheduler refuse a mismatch); and
`decide_h6` recorded missing residuals as ±inf, which the `allow_nan=False` decision writer would
have rejected on the first failed coordinate (commit `a08f5e2`: null with explicit gates).
New official capability `/home/angus/g4-executor-capability-a08f5e2.json`, SHA-256
`cf057e02944c09573348025ff457544984ce75651220fe5777c1fe64eefdaaef`, source commit `a08f5e2`,
executable SHA-256 `0a7c41c453bfabc6c1b9014d53c2b606f6b0723ef16a36c05a9c60cfbd070132`. Checkpoint
`build-integration-report/g4-claim-core-a08f5e2` (396 groups, schedule SHA-256 `1123b8de…`,
`policy_amendment=single-gpu-v1.1` in metadata) is driven from the detached worktree
`/home/angus/worktrees/spacepdhcg-g4-claim-core-a08f5e2` by
`build-integration-report/g4-claim-core-a08f5e2-{worker,status,observer,finish}.sh`. First ten
groups (all P1-E `N=100` pure-gpu-ipm): every attempt `numerical` at outer iteration 0 with zero
QOCO workspace creations (a pre-existing executor defect candidate, reproduced without the
amendment environment), 90 of 90 measured attempts `contaminated` by a foreign Windows compute
process at 80–98 % SM, 31–136 s per group. No H5/H6 statistic has been formed.

Pure-gpu-ipm defect triage (2026-09-03, checkpoint `g4-claim-core-a08f5e2` paused at a group
boundary after 26 completed groups, all P1-E pure-gpu-ipm, every attempt `numerical`):

- Environment and wiring were not the cause. `/proc/<worker>/environ` and the executor server
  carried `SPACEPDHCG_QOCO_LIBRARY=…/build-current-head-qoco/libqoco.so` (QOCO v0.3.2,
  cuda/cuDSS algebra) and the cuDSS `LD_LIBRARY_PATH`; a foreground `--g4-session` replay of group
  0 with the exact worker environment and with the planner's known-good library
  (`build/qoco-g4/libqoco.so`) reproduced the same nine `numerical` attempts. A stderr diagnostic
  added to the executor showed `api_status=7` (`NUMERICAL_FAILURE`), SCvx `INNER_FAILURE`,
  `qoco_failure=4` (`NUMERICAL`), QOCO setup 0.08–0.19 s and a 5 s first solve: the adapter was
  constructed and QOCO ran. The reported "zero workspace creations" was a reporting gap — the
  driver's failure branch never copied `workspace_creations` into the result.
- Root cause 1 (executor defect, fake `numerical`): with `warm_mode=primal` the reset boundary of
  every attempt after a successful one calls `spacepdhcg_cuda_workspace_warm_start_async(…,
  FULL_RETAINED)` on the PDHCG workspace. Pure IPM never runs the PDHCG kernel, the workspace holds
  no retained solver state, the call returns `INVALID_STATE`, and the executor recorded the attempt
  as a launched `numerical` failure in 0.00 s. On an unconditioned replay this produced the
  alternating pattern `qualified, numerical, qualified, numerical, …` (warm-ups and measured/1,3,5
  lost). Fix: for `SPACEPDHCG_CUDA_SCVX_PURE_QOCO` the warm boundary is the retained QOCO primal plus
  the dual clear and the PDHCG warm start is skipped; the failure branch now reports QOCO
  workspace creations/updates; the adapter records the raw QOCO status.
- Root cause 2 (genuine solver outcome, not an executor defect): at the claim core's fixed axis
  `conditioning=4.0` QOCO reports `numerical error` (P1-E `N=100`, 53 iterations, step lengths
  ~0, `Pcost` 1e17–1e18, primal residual 1e6–1e8; constraint coefficient range 1e2 versus 1e0
  unconditioned) and `maximum iterations reached` (`N=20`). Ruiz scaling (5 or 10 iterations),
  static regularisation 1e-8/1e-10 and one IR step were tried for diagnosis only and diverged to
  NaN at iteration 1; no adapter setting was changed. The same group with `conditioning=0.0`
  qualified every executed attempt (100 outer iterations, canonical residual 2–8e-10, 65–96 s
  per attempt, ≥1 QOCO workspace). Fixed-tight PDHCG on the same `N=100` conditioning-4.0
  coordinate reached its 600 s deadline at residual ≈37 in `g4-claim-core-9a4cbea`. Genuine
  `numerical` attempts at conditioning 4.0 therefore remain valid H6 evidence for the pure-IPM
  baseline; they are not invalidated by this triage. Two caveats are recorded for the H6
  interpretation rather than fixed here (either would be a solver-setting change requiring a
  preregistered amendment): the pure-IPM baseline runs QOCO without equilibration
  (`ruiz_iters=0`) while PDHCG uses the workspace's `refresh_if_needed` scaling, and a single
  QOCO solve is not interruptible, so an IPM attempt whose one solve overruns the 120 s attempt
  deadline is recorded by QOCO's own outcome (`numerical` after 101 iterations / 109–134 s under
  the foreign load) rather than as `timeout`.
- Root cause 3 (executor defect, found on the first re-run group under `857f99a`): QOCO keeps
  state across `qoco_solve` calls on one persistent solver. Its stall handler multiplies
  `solver->settings->kkt_dynamic_reg` by 10 in place and never restores it, and the best-iterate
  tracker (`best_valid`/`best_metric`) is reset only at setup. After attempt 0 ended in
  `numerical error` (101 iterations, 109 s), the numeric-update path re-solved the identical data
  in 62 iterations, then 1 iteration (1.7 s) for every remaining attempt, each restoring "best
  iterate (39)" from the first solve — same disposition, but not independent attempts. Fix
  (adapter only; the vendored QOCO is untouched): restore the configured settings before every
  numeric update and, after any failed solve, tear the solver down and set it up again on the
  current formulation before the next solve, counted in `qoco_workspace_creations` (so the
  invariant is ≥ 1 per group, not exactly 1). The stale-best hazard for a *successful* solve
  followed by a failed one on different data (hybrid handoff) is bounded: the executor's quality
  gate recomputes canonical residuals on the current data, so a stale iterate can be
  `unqualified`, never `qualified`.
- Fail-closed contract: new terminal disposition `executor_defect` (reset-boundary failure, QOCO
  ABI fault, or any driver API status that is not a solver outcome) in the executor, raw-attempt
  and Paper 1 schemas and Python contracts; a group containing one is quarantined, and the
  decision refuses a completed group carrying one. The capability probe now runs a real
  pure-gpu-ipm session and refuses the executor unless all nine attempts launch with ≥1 QOCO
  workspace creation and solver dispositions; it pins the QOCO library hash, and the worker
  refuses to start unless its `SPACEPDHCG_QOCO_LIBRARY` hashes to the pinned library.
  `tests/test_g4_ipm_session_gpu.py` drives the real executor through that probe (9/9 attempts
  ≥1 QOCO workspace, dispositions `unqualified` at one outer iteration, warm boundaries
  `primal`); it passed on the fixed build with a foreign compute job at 99 % SM (12–20 s per
  attempt).
- Campaign hygiene: the 26 completed pure-gpu-ipm groups of `g4-claim-core-a08f5e2` (all P1-E,
  22 × `N=100`, 4 × `N=500`; 182 measured attempts, 84 contaminated) were invalidated with the
  new `invalidate` ledger action (`invalid_executor_defect`, records and result files retained,
  `invalidation.json` sidecars, journal events, `fix_commit=857f99a`); the interrupted group 26
  stays `running`/`interrupted` in that checkpoint, which is never resumed (`claim()` skips every
  existing row and the checkpoint pins `source_commit=a08f5e2`). Option (b) was taken: fix commit
  `857f99a`, CUDA CTest 62/62, capability `1d8c7527…25041` (probe: 9/9 launched, 1 QOCO
  workspace, numeric updates 0..8, all `unqualified`), new checkpoint `g4-claim-core-857f99a`,
  `migrate` from `a08f5e2` imported 0 (no untouched completed group of another policy existed).
  Its first nine groups exposed root cause 3, so that worker was paused at a group boundary after
  9 completed P1-E `N=100` groups (63 measured `numerical`, all contaminated) and those groups
  were invalidated in turn (`fix_commit=ccd5596`, superseded by `g4-claim-core-ccd5596`).
  Current checkpoint: `g4-claim-core-ccd5596` (fix commit `ccd5596`, qoco/scvx CTest 7/7,
  GPU regression test 2:57, capability `d7d27454…b5319`, `migrate` imported 0). Order is
  unchanged (pure-gpu-ipm first).
- First valid IPM group (`ccd5596`, ordinal 0, P1-E `N=100`, conditioning 4.0, foreign job at
  99 % SM): eight launched attempts, each on a freshly built QOCO solver (`workspace_creations`
  1..8), 27–200 IPM iterations and 28–213 s per attempt, all genuine `numerical` (QOCO status 3;
  one `maximum iterations` at 200), all `contaminated`; the ninth attempt is `unrun` because the
  executor's 1140 s group deadline expired at 1216 s. Pure IPM therefore does not qualify on the
  conditioning-4.0 P1-E core; each failing `N=100` IPM group costs ≈20 min. Risk to watch: a
  single QOCO solve cannot be interrupted, so if an `N=2000` IPM solve exceeds the scheduler's
  hard bound (1140 s + 300 s grace) the session is killed, restarted once, and the group ends as
  an error record — nothing is recorded as solver evidence, but ≈48 min per such group is spent.
- Amendment `single-gpu-v1.2` (`docs/G4_CLAIM_CORE_AMENDMENT_V1_2.md`, frozen
  2026-09-03T06:45:00Z) resolves the two caveats above by preregistered rule rather than
  interpretation. Rule A: the IPM baseline runs QOCO's native default equilibration and records
  it; at the pinned commit that default is `ruiz_iters = 0`, and Ruiz-on was probed on the failing
  P1-E `N=100` conditioning-4.0 coordinate and on conditioning 0.0 — with the pinned library it
  NaNs at iteration 8 (two QOCO CUDA-backend defects: `safe_div(1,0)=DBL_MAX` on empty rows and a
  `scale_arrayf` without host fallback), and with those defects patched in a scratch build it still
  ends `numerical` at conditioning 4.0 (54 iterations, 183 s) and turns the qualified
  conditioning-0.0 solves (3/3 qualified, 11–13 QOCO iterations) into `numerical` (6/6). The
  conditioning-4.0 pure-IPM result is a genuine IPM negative. Rule B: a launched attempt whose
  measured wall exceeds the 120 s deadline is `timeout`, never `numerical`, for every backend.
  Rule C: the `N=2000` hard bound is unchanged. **Diagnostic stratum
  `ipm_no_equilibration_v1_1`**: the `pure-gpu-ipm` groups completed under v1.1 in
  `g4-claim-core-ccd5596` (all P1-E `N=100`, every launched attempt `numerical` and
  `contaminated`, 27–200 IPM iterations, 28–304 s per attempt) are retained verbatim in the
  `diagnostic` ledger state, excluded from H6, and cited by the successor checkpoint's
  `diagnostic_strata` metadata. They agree with the amended runs in direction (IPM fails at
  conditioning 4.0) but are not comparable records because they predate rules A and B.

Implementation update (2026-09-02): the authoritative `g4-persistent-group-v1` native executor,
direct per-attempt NVML boundaries, hash-pinned capability probe, and separate 360-group claim-core
checkpoint are implementation-ready on `integration/single-gpu-v1`. No claim-core or full grouped
campaign has been run. H5/H6 and the scientific G4 decision therefore remain unresolved; this
implementation status does not promote or erase the historical FAIL evidence below.

Scope reconciliation (2026-09-02): this report is retained historical evidence and its original
FAIL/unresolved classifications are unchanged. Under `single-gpu-v1`, current-head G4 remains the
active Paper 1 evidence gate, but physical G5 is a separate deferred campaign and does not block a
future scoped freeze after the complete in-scope G4 ledger is qualified and portable. Nothing in
this scope change promotes the failures below.

## Scope and frozen policy

The G4 policy was frozen before evaluation in commit `8f318b9`. The machine-readable contract is
`benchmarks/g4_policy.json`; it fixes all six required modes, forcing and re-solve rules, trust
policy, warm-start and scaling modes, recovery inclusion, quality tiers, tuning/evaluation split,
the full Paper 1 family matrix, H5/H6 thresholds, bootstrap seed/method, solver-order seed, timeout
handling, and 50 ms GPU power sampling.

Implementation and qualification telemetry were committed in `509d9dd`; the G3 compatibility
tolerance correction is `b70bf15`. Evaluation was run from the clean full commit
`b70bf15f3fc84c6ed9138df4cf466a3907c51e06`.

## GPU interior-point baseline

The pinned baseline is QOCO-GPU commit `09f049597deef2a7ead15b3da19a9456ff7d4e53`
(reported version 0.3.2), tree `c85fe82f71a67921868fc761c242de11ac46f4a2`, with
`nvidia-cudss-cu12==0.7.1.6`, CUDA 12.8.93, `sm_120`, float64, and sparse direct cuDSS
factorisation. Its native form is

\[
 \min_x \tfrac12 x^T P x+c^Tx,\qquad Ax=b,\qquad h-Gx\in
 \mathbb R_+^l\times\prod_i\mathcal Q_i.
\]

This supports the required quadratic objective, equalities, nonnegative rows, and SOCs. Exact box
splitting and an orthonormal rotated-SOC-to-SOC conversion are declared in the lock. QOCO exposes
`qoco_set_x0` for an unequilibrated primal start, but exposes no dual warm-start API.

The pinned source built and its CUDA/cuDSS demo executed on the RTX 5090 in six IPM iterations with
objective 4.042, native primal residual `1.413e-8`, and native dual residual `8.258e-9`. This is
availability evidence only. No production trajectory CQP was executed through QOCO, so neither a
pure-IPM trajectory comparison nor a hybrid timing claim is made. The missing production adapter
and unavailable dual handoff make H6 unresolved.

## Qualification result

The first nontrivial tight-quality coordinate in each required nonlinear family failed the
matched-quality prerequisite. The campaign therefore stopped before timing the full matrix:
unqualified trajectories may not enter H5/H6 speed comparisons.

| Family | Coordinate | Canonical residual | Terminal residual | SCvx time (s) | GPU energy (J) |
|---|---:|---:|---:|---:|---:|
| P1-C 3-DoF | N=20, dispersion=0.01 | `5.674e-4` | `1.000` | `35.771` | `4002.01` |
| P1-D 6-DoF | N=20, dispersion=0.05 | `1.480e-1` | `4.990` | `70.501` | `7981.64` |
| P1-E low-thrust | N=100, dispersion=0.01 | `4.731e-1` | `70.800` | `164.240` | `18665.82` |

All three samples:

- returned no accepted step and ended at the outer-iteration limit;
- violated the requested forcing threshold after the identical-CQP re-solve;
- retained matching 64-bit numeric CQP fingerprints across the re-solve;
- reported zero post-create topology allocations and zero hidden CPU fallback;
- were excluded from all performance claims.

The initial diagnosis identified a real architectural omission but overstated its causal role in
this particular archive. The one-outer-iteration qualification materialised its first CQP from the
displaced CPU reference, so it never reached a second reference update. The missing production
updates were nevertheless in scope and are now implemented for the reference-tracking objective,
trust-cone centres and radii, exact-penalty epigraph cost, low-thrust radial halfspaces, 6-DoF
quaternion linearisation, and nonlinear dynamics.

The corrective rerun found two additional production defects: nonlinear merit used unscaled
magnitudes inconsistent with the CPU SCvx policy, and an identical-CQP refined re-solve updated
telemetry but acceptance still evaluated the stale pre-refinement primal. Both are corrected, and
all four fixtures compare every CPU/device numerical vector with maximum absolute difference
`1.8118839761882555e-13` (contract `5e-12`) while changing trust radii and exact-penalty weights
in place. Even after those fixes, displaced P1-C remains
unqualified: a 15-outer fixed-loose diagnostic accepted one step after eight trust reductions but
ended at the minimum radius with scaled dynamics `1.705e-3`, terminal `8.092e-4`, and virtual
control `2.000e-3`. This is retained as new negative evidence, not as a replacement for the frozen
archive. G4 therefore remains FAIL pending an accurate production IPM/hybrid path or a further
solver correction under unchanged policy.

The compact coverage ledger retains 1,080 family/interval/policy/warm-start/quality coordinates.
Unexecuted first-order coordinates are explicitly censored after qualification failure. Pure-IPM
and hybrid coordinates are explicitly unsupported because no production adapter execution exists;
the QOCO smoke run is not relabelled as a trajectory result.

## H5 and H6 decisions

### H5 — unresolved

There are zero matched-quality nontrivial family samples and therefore no valid five-repeat paired
bootstrap, no two-family support region, and no sustained three-coordinate boundary. Adaptive
forcing cannot be supported or rejected from unqualified trajectories.

### H6 — unresolved

There is no archived production QOCO trajectory solve or qualified PDHCG-to-QOCO handoff. The
pinned baseline accepts primal starts but not dual starts. No Pareto-frontier or time-advantage
claim is permitted.

## Timing and energy limitations

Power was sampled with `nvidia-smi` over each isolated benchmark process boundary. The traces are
GPU-only and no idle subtraction was applied. WSL delivered maximum sampling gaps of 1.938 s,
1.930 s, and 1.862 s despite the requested 50 ms cadence, so the integrated energies above are
diagnostic only. Display/shared-machine isolation was not established.

An earlier qualification directory is preserved as negative evidence because other GPU tests ran
concurrently with its power traces. It is excluded from energy conclusions.

## Build, test, and sanitizer status

- Ruff: passed.
- Python: 98/98 tests passed.
- CUDA/native Release: 52/52 tests passed.
- CUDA/native Debug: 52/52 tests passed.
- G3 production regression: passed at the original `1e-6` contract.
- Compute Sanitizer on the modified telemetry/outer lifecycle:
  - memcheck: 0 errors;
  - racecheck: 0 hazards, 0 errors, 0 warnings;
  - synccheck: 0 errors;
  - initcheck: 0 errors.

The sanitizer runs validate telemetry allocation, hashing, stream use, and lifetime behavior. They
do not qualify the failed nonlinear trajectories.

## Criterion-by-criterion decision

1. Policies frozen before evaluation: **PASS**.
2. Mathematically valid GPU IPM pinned and executable: **PASS** for baseline availability.
3. Production pure-IPM trajectory execution: **PASS** for exact HCW trajectory QP/SOCP adapter
   correctness on RTX 5090; **FAIL** for the required nonlinear P1-C/P1-D/P1-E executions.
4. Audited hybrid conversion: **PASS** for qualified primal handoff with explicit dual discard;
   **FAIL** for a matched-quality nonlinear G4 handback.
5. Per-outer policy telemetry and identical-CQP re-solve audit: **PASS**.
6. Matched final nonlinear quality: **FAIL**.
7. H5 resolved under preregistered rules: **FAIL / unresolved**.
8. H6 resolved under preregistered rules: **FAIL / unresolved**.
9. Full primary matrix and repeat/instance count: **FAIL**; censored after qualification failure.
10. Affected tests and four sanitizer tools: **PASS**.
11. Negative, unsupported, and censored records retained: **PASS**.

Gate G4 therefore **does not pass**, and Gate G5 is **not authorised**.

## Integrated harness and adapter follow-up

The audited harness commits and pinned-QOCO adapter were integrated after the corrective campaign.
The CUDA outer executable now rejects a mismatched generated policy SHA, consumes quality-tier
fixed-tight tolerances, scaling and warm modes, and the frozen adaptive phase, re-solve, polish,
and trust parameters instead of duplicating those constants. Requested and actual runtime values
are emitted separately.

Serialized RTX 5090 adapter validation passed 18/18 tests, including exact trajectory QP and SOCP
agreement with Clarabel, persistent same-pattern update, accepted primal warm start, independent
canonical residuals, and explicit discard of the unsupported dual start. This closes the adapter
ABI/runtime gap but does not convert the failed nonlinear qualification into a pass. An integrated
one-outer P1-C diagnostic under policy SHA
`9ab3b444e3dd21fdd2a75c3cebfe8fd8374f9e5ff672cd757afaeb6036530024`
reproduced canonical residual `5.653990342580073e-4`, terminal residual `1e-3`, identical-CQP
re-solve fingerprint, no accepted step, and forcing failure. A 15-outer diagnostic exceeded
30 minutes before completion and was terminated and retained as timeout evidence; it is not
reported as an executed matrix point. H5 and H6 therefore remain unresolved.

## Evidence

Integrated clean-commit negative evidence:

- `results/gpu/g4/qualification/g4-20260901-f48adf4-integrated.tar.gz`
- execution commit: `f48adf4`
- archive SHA-256: `880653ec4cf85aacda5702012e2271e80568a4cf7b8c0c3bf0668e2c8cac7e40`
- decision SHA-256: `98f8863d46f03c58aba79a34fdd0785226df5d0e45ebfde823f405334751e7d4`
- P1-C runtime SHA-256: `ed153267f75fa1a79b18b6f891f91fe6dfbc92e4048d06c1e36ee91a43801087`
- RTX QOCO test SHA-256: `2e5fbb3500e435fe276df5629ee52624867e00e39eea517396e0dfcb832c8386`

The archive is explicitly local-only and non-portable because no immutable artifact URI exists.
That publication limitation is separate from the authoritative G4 FAIL decision.

Clean corrective qualification after the fixed-pattern update implementation:

- `results/gpu/g4/qualification/g4-20260901-2cbebb3-clean-corrective/`
- execution commit recorded by the manifest:
  `2cbebb3b1b66666e6dc95e879260174b32abf22e`
- manifest SHA-256: `fe218070ec0332a3747e9a1f3f6c5705f79c313810d3bb070d3314d48adcd49c`
- decision SHA-256: `d7143c2cc338423a21c0cf606aa71b84812bc784f26efd7e0c30fd8821a09744`
- coverage SHA-256: `ed06f436c410660eb9dc0abc76847c5ab7dae8a6a4603697bad5dbe9608815bc`
- corrected canonical residuals: P1-C `5.654e-4`, P1-D `1.283e-2`,
  P1-E `1.369e-2`;
- corrected scaled terminal residuals: P1-C `1.000e-3`, P1-D `5.243e-2`,
  P1-E `7.080e-2`; qualified samples: 0/3; forcing failures: 3/3.

This corrective directory supplements rather than replaces the frozen original failure archive.
The earlier `g4-20260901-f21f93d-corrective2` directory was produced while the final
initial-boundary fix was still uncommitted and is retained only as contaminated negative evidence,
not clean qualification evidence. That fix was committed as `2cbebb3` before this rerun. The
aborted first corrective directory is likewise retained as additional negative evidence.

Primary isolated qualification archive:

- `results/gpu/g4/qualification/g4-20260901T072000Z-b70bf15-isolated.tar.gz`
- SHA-256: `2bda322a9f85d59c870aeb244dd5b2c64a23ef5f38a706eb2b9a4d051baae5c8`
- evidence-index SHA-256:
  `84db411403dbde4871011e8d3fbf4d85e1534d44e2f53698ab724e007678f7d7`

Preserved contaminated preliminary archive:

- `results/gpu/g4/qualification/g4-20260901T070000Z-509d9dd-contaminated.tar.gz`
- SHA-256: `cac80c4fa5bb6797a36f94cfeac91d5b8d79469e2c739f911f8503a98d9e876d`

The isolated archive contains the exact execution commit, policy and solver-lock hashes, commands,
environment, raw outer-iteration diagnostics, power traces, compact 1,080-coordinate coverage
ledger, QOCO build/smoke logs, test logs, all sanitizer logs, decision record, and artifact index.

## Unified-roadmap native-QOCO validation

The final native QOCO owner was integrated into the unified G4-G7 tree and validated serially on
the RTX 5090. P1-C reproduced the expected two accepted steps in individual correctness runs
(`0.999931728/0.999998351`, terminal `1.230e-13`, canonical `1.303e-11`). Compute Sanitizer
memcheck, racecheck, initcheck, and synccheck each produced a zero-error run. Initcheck with
third-party unused-memory tracking additionally reported only cuDSS/cuBLAS internal reserved
buffers; that diagnostic is retained and is not promoted to a SpacePDHCG uninitialized-access
claim.

The repeatability/globalization correction establishes the pure-QOCO correctness prerequisite but
does not by itself pass G4:

- Pinned QOCO's relative stopping scale admitted absolute KKT residuals above the frozen forcing
  gate when virtual penalties or dual magnitudes were large. A declared patch now requires primal
  and dual residuals to meet `abstol` while retaining QOCO's relative gap test.
- P1-C passed 56/56 independent frozen-coordinate repeats. Every run accepted two steps; canonical
  residual ranged `1.638e-13..5.078e-13` and terminal residual
  `8.319e-14..1.788e-13`.
- P1-D passed 7/7 100-attempt repeats with complete quaternion/path inventory and transactional
  rollback. Runs accepted 24-25 steps, rejected 4-9 candidates, and ended with canonical residual
  `5.015e-12..1.074e-11` and terminal residual `2.772e-12..1.580e-11`.
- P1-E `radius_raise` at frozen trust coordinate `0.25` was independently proven reachable:
  Clarabel solved the identical dumped CQP to primal residual `2.907e-15`, while pre-fix CPU and
  GPU QOCO matched near `3.024e-7`. Penalty ownership and low-thrust KKT refinement were corrected.
  P1-E then passed 7/7, with two accepted nonzero steps per run, canonical residual at most
  `5.178e-9`, and terminal residual `8.260e-14..3.253e-12`.
- Post-fix representative checks retained the persistent negatives: fixed-tight, fixed-loose,
  adaptive, adaptive+polish, and hybrid each hit an explicit 120-second process timeout for
  P1-C, P1-D, and P1-E. Earlier unqualified and 600-second P1-E timeout records remain preserved;
  no timed-out adaptive execution is relabelled as hybrid.
- Earlier 3/7 timing/energy remains non-primary evidence and is retained rather than overwritten.
  No corrected measured campaign is inferred from correctness-only stress runs. `nvidia-smi`
  polling also exceeded the requested 50-ms interval under load.
- Hybrid remains ineligible where persistent PDHCG misses the `1e-6` handoff qualification.

The expanded frozen Cartesian ledger contains 24,883,200 rows. The required seven measured
repeats, two warm-ups, 20 evaluation instances, all coordinates, and portable immutable artifacts
are incomplete. H5 and H6 remain **unresolved**, Gate G4 remains **FAIL**, and these results do not
authorise physical G5 acceptance or Paper 1 freeze.
