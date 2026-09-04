# G4 claim-core amendment `single-gpu-v1.1`

Versioned, preregistered amendment to the H5/H6 claim core
(`benchmarks/g4_h5_h6_claim_core.json`, SHA-256 `40dc2174…0c8d0b`) under campaign scope
`single-gpu-v1`. The machine-readable contract is
`benchmarks/g4_claim_core_amendment_v1_1.json`, locked by
`benchmarks/g4_claim_core_amendment_v1_1.sha256`, validated by
`experiments/schema/g4_claim_core_amendment.schema.json` and
`spacepdhcg.experiments.g4_execution_contract.load_claim_core_amendment`. The scope registry
(`benchmarks/campaign_scopes/single-gpu-v1.json`, `spacepdhcg.campaign_scope`) lists it under
`amendments`.

The amendment was frozen while the claim-core checkpoint held zero completed groups. Its one
quarantined (contaminated) group result was not opened before the amendment was committed. The
original `single-gpu-v1` rules remain readable: the claim-core JSON, the policy JSON and their
locks are unchanged, and the original censoring values are carried verbatim in
`censoring.original`.

Every record produced under the amendment carries `policy_amendment: "single-gpu-v1.1"`: the
checkpoint metadata (`policy_amendment`, `policy_amendment_sha256`), every raw attempt emitted by
the executor, every `execution_group_result`, and the decision and publication-aggregate records
(in `notes`, because the Paper 1 identity block is closed).

## What does not change

- quality tiers, tolerances and matched-quality gates (`benchmarks/g4_policy.json` unchanged);
- the twenty evaluation seeds and the physical instance identities;
- two same-session warm-ups followed by seven measured attempts per group;
- group identities (`g4-group-v1` content addresses) and the frozen `solver_order` rotation;
- one process, one CUDA context and one persistent workspace per group;
- bootstrap seed, sample count, confidence and decision thresholds;
- the core resolves H5 and H6 only and populates no F01–F12/T01–T08 product.

## Decision A — contamination: run-and-flag

The scheduler no longer waits for the GPU to be idle and no longer quarantines and re-runs a
group. Detection is unchanged and honest: `nvidia-smi` utilization/memory/power and a process
probe are sampled at both group boundaries and roughly once per second during the group. WSL2
cannot enumerate foreign PIDs, so foreign host compute is read from Windows `nvidia-smi.exe pmon`
(SM%/MEM% per PID); a WSL `/dev/dxg` holder outside the worker's process tree with CUDA devices
visible is also foreign. The worker holds the shared advisory lock
`/home/angus/.spacepdhcg-gpu.lock` with a JSON payload for the whole group and records any foreign
payload it finds there.

Each raw attempt is attributed a wall-clock window (from the previous attempt's record, or the
session-ready record, to its own record). An attempt whose window overlaps a foreign sample is
flagged `contaminated: true` with a `contamination` summary (samples, max foreign SM%, foreign
processes). Its disposition and quality metrics are retained; its timing and energy are invalid.
The group is `completed_group`; no re-run occurs.

Contaminated measured attempts never enter a timing or energy statistic or a paired bootstrap.
They remain in disposition and quality counts. Every H5/H6 coordinate row reports the pair count
`n` actually used, the contaminated count per policy and the censoring counts.

## Decision B.1 — deterministic-replay timeouts

Every raw attempt carries a `trace` (`inner_iterations`, `outer_iterations`, final canonical,
dynamics, path, terminal and virtual-control residuals, and one
`[phase, requested_tolerance, achieved_residual, accepted, re_solved]` checkpoint per outer
iteration) and its `trace_hash`: 64-bit FNV-1a over the canonical string
`disposition|inner|outer|res…|phase:req:ach:acc:res;…` with doubles printed as `%.17g`. The
executor and `deterministic_trace_hash` in Python are bit-identical
(`--g4-amendment-selftest` proves it).

When warm-up/0, warm-up/1 and measured/0 of a group all reach the attempt deadline (disposition
`timeout`, actually launched) with identical trace hashes, measured/1..6 are recorded as
`timeout_deterministic_replay` (failure class `timeout`, `launched: false`,
`replay_source_attempt_id = <group>/measured-0`, the source trace, zero wall time) and are not
executed. If any of the three traces differ, all seven measured attempts execute. Replayed
attempts count as timeout censoring and never enter a timing statistic.

Because the deadline is wall-clock based, three deadline-cancelled attempts normally differ in
their cancelled iteration count; the replay therefore fires only when the trace is genuinely
deterministic (for example when the inner iteration cap, not the clock, ends every attempt).

## Decision B.2 — 120 s / 200k censoring with a censoring-sensitivity stratum

For the claim core the attempt deadline becomes 120 s (group deadline 9×120+60 = 1140 s) and
every inner PDHCG iteration limit of the policy becomes `min(limit, 200 000)` (fixed-tight limit,
adaptive polish limit and final polish limit; smaller limits are unchanged). The full G4 ledger
keeps 600 s / 1 000 000.

A deterministic 10 % stratified subset (`censoring_sensitivity`) additionally runs at the original
600 s / 1M settings: in each of the 18 family × scale × policy strata the twenty groups are ranked
by `sha256("single-gpu-v1.1-censoring-sensitivity|" + group_id)` and the lowest two are selected
(36 groups). Each twin shares the coordinate and physical instance of its claim-core group and adds
`censoring_stratum: censoring_sensitivity` to its identity (distinct `g4-group-v1` id). The 36
group ids are committed in the amendment JSON and re-derived by the validator.

Acceptance rule (preregistered): every measured attempt (seed, repeat) of a twin is compared with
the same attempt of its claim-core group. If any attempt is qualified under 600 s / 1M while its
120 s / 200k counterpart is `timeout` or `timeout_deterministic_replay` (or the counterpart is
missing while the twin qualified), the amendment is invalid: the decision tool refuses to issue an
H5/H6 decision from the 120 s core and the full core reverts to 600 s / 1M.

## Schedule

Policies run in the order pure-GPU-IPM, adaptive, hybrid, fixed-tight; within a policy the
claim core's definition order is kept, and each sensitivity twin runs immediately after its
claim-core group (396 groups). The claim core never bound execution order to the `solver_order`
rotation (its scheduler ran definition order); `solver_order` stays recorded per group as an
execution-only identity axis, so this reordering is permitted. The schedule hash
(`schedule.schedule_sha256`) is the SHA-256 of the canonical ordered group-id list and is stored
as the checkpoint `schedule_sha256`.
